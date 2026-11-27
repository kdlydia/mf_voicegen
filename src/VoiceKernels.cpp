#include "VoiceKernels.hpp"

namespace MayaFlux::Yantra::TensorKernels::cpu {

namespace {

    float apply(UnaryOp op, float x, float a, float alpha, float beta)
    {
        switch (op) {
        case UnaryOp::Affine:
            return alpha * x + beta;
        case UnaryOp::Exp:
            return std::min(std::exp(x), alpha);
        case UnaryOp::Log:
            return std::log(std::max(x, alpha));
        case UnaryOp::Sin:
            return std::sin(x);
        case UnaryOp::Cos:
            return std::cos(x);
        case UnaryOp::Tanh:
            return std::tanh(std::clamp(x, -15.0F, 15.0F));
        case UnaryOp::LeakyRelu:
            return x >= 0.0F ? x : alpha * x;
        case UnaryOp::Silu:
            return x / (1.0F + std::exp(-x));
        case UnaryOp::Snake: {
            const float s = std::sin(a * x);
            return x + s * s / a;
        }
        case UnaryOp::Clamp:
            return std::min(std::max(x, alpha), beta);
        case UnaryOp::GeluTanh: {
            const float inner = 0.7978845608028654F * (x + 0.044715F * x * x * x);
            return 0.5F * x * (1.0F + std::tanh(std::clamp(inner, -15.0F, 15.0F)));
        }
        case UnaryOp::Fract:
            return x - std::floor(x);
        case UnaryOp::Step:
            return x > alpha ? 1.0F : 0.0F;
        }
        return x;
    }

    float combine(BinaryOp op, float a, float b)
    {
        switch (op) {
        case BinaryOp::Multiply:
            return a * b;
        case BinaryOp::Add:
            return a + b;
        case BinaryOp::Subtract:
            return a - b;
        case BinaryOp::Divide:
            return a / b;
        case BinaryOp::Maximum:
            return std::max(a, b);
        case BinaryOp::Minimum:
            return std::min(a, b);
        case BinaryOp::Atan2:
            return std::atan2(a, b);
        }
        return a;
    }

    float gelu(float x)
    {
        return 0.5F * x * (1.0F + std::erf(x / std::numbers::sqrt2_v<float>));
    }

    float activate(Activation activation, float x)
    {
        switch (activation) {
        case Activation::Relu:
            return std::max(x, 0.0F);
        case Activation::Sigmoid:
            return 1.0F / (1.0F + std::exp(-x));
        case Activation::Gelu:
            return gelu(x);
        case Activation::None:
            break;
        }
        return x;
    }

}

std::vector<float> unary(UnaryOp op, std::span<const float> x, float alpha, float beta)
{
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = apply(op, x[i], alpha, alpha, beta);
    }
    return out;
}

std::vector<float> snake(std::span<const float> x, std::span<const float> a, uint32_t channels, uint32_t inner)
{
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = apply(UnaryOp::Snake, x[i], a[(i / inner) % channels], 1.0F, 0.0F);
    }
    return out;
}

std::vector<float> binary(BinaryOp op, std::span<const float> a, std::span<const float> b, uint32_t inner)
{
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = combine(op, a[i], b[(i / inner) % b.size()]);
    }
    return out;
}

std::vector<float> multiply(std::span<const float> a, std::span<const float> b, uint32_t inner)
{
    return binary(BinaryOp::Multiply, a, b, inner);
}

std::vector<float> conv_transpose1d(const ConvTranspose1d& layer, std::span<const float> x,
    std::span<const float> weights, std::span<const float> bias, uint32_t batch, uint32_t t_in)
{
    const uint32_t t_out = conv_transpose1d_length(layer, t_in);
    const uint32_t groups = std::max(layer.groups, 1U);
    const uint32_t in_per_group = layer.c_in / groups;
    const uint32_t out_per_group = layer.c_out / groups;
    std::vector<float> out(static_cast<size_t>(batch) * layer.c_out * t_out, 0.0F);
    for (uint32_t n = 0; n < batch; ++n) {
        for (uint32_t ci = 0; ci < layer.c_in; ++ci) {
            const uint32_t group = ci / in_per_group;
            for (uint32_t ti = 0; ti < t_in; ++ti) {
                float value = x[(static_cast<size_t>(n) * layer.c_in + ci) * t_in + ti];
                if (layer.relu_input) {
                    value = std::max(value, 0.0F);
                }
                for (uint32_t local = 0; local < out_per_group; ++local) {
                    const uint32_t co = group * out_per_group + local;
                    for (uint32_t k = 0; k < layer.kernel; ++k) {
                        const int64_t to = static_cast<int64_t>(ti) * layer.stride
                            + static_cast<int64_t>(k) * layer.dilation - layer.pad;
                        if (to < 0 || to >= t_out) {
                            continue;
                        }
                        out[(static_cast<size_t>(n) * layer.c_out + co) * t_out + static_cast<size_t>(to)]
                            += weights[(static_cast<size_t>(ci) * out_per_group + local) * layer.kernel + k] * value;
                    }
                }
            }
        }
    }
    for (uint32_t n = 0; n < batch; ++n) {
        for (uint32_t co = 0; co < layer.c_out; ++co) {
            for (uint32_t to = 0; to < t_out; ++to) {
                float& value = out[(static_cast<size_t>(n) * layer.c_out + co) * t_out + to];
                value = activate(layer.activation, value + (bias.empty() ? 0.0F : bias[co]));
            }
        }
    }
    return out;
}

std::vector<float> overlap_add(std::span<const float> frames, std::span<const float> window_sq,
    uint32_t frame_length, uint32_t count, uint32_t hop, uint32_t trim, uint32_t samples)
{
    std::vector<float> out(samples, 0.0F);
    for (uint32_t t = 0; t < samples; ++t) {
        const uint64_t p = static_cast<uint64_t>(t) + trim;
        float sum = 0.0F;
        float envelope = 0.0F;
        for (uint32_t f = 0; f < count; ++f) {
            const uint64_t start = static_cast<uint64_t>(f) * hop;
            if (p < start || p - start >= frame_length) {
                continue;
            }
            const uint64_t r = p - start;
            sum += frames[r * count + f];
            if (!window_sq.empty()) {
                envelope += window_sq[r];
            }
        }
        out[t] = window_sq.empty() ? sum : (envelope > 1.0e-11F ? sum / envelope : 0.0F);
    }
    return out;
}

std::vector<float> gather(std::span<const float> src, std::span<const int32_t> index, uint32_t channels,
    uint32_t channel_stride, uint32_t element_stride)
{
    const size_t count = index.size();
    std::vector<float> out(channels * count);
    for (uint32_t c = 0; c < channels; ++c) {
        for (size_t t = 0; t < count; ++t) {
            const int32_t at = index[t];
            out[c * count + t] = at < 0 ? 0.0F
                                        : src[static_cast<size_t>(c) * channel_stride + static_cast<size_t>(at) * element_stride];
        }
    }
    return out;
}

void lstm_recurrence(std::span<const float> gates_in, std::span<const float> w_hh, std::span<float> state,
    std::span<float> hs, uint32_t steps, uint32_t hidden, bool reverse, uint32_t row_offset)
{
    const auto sigmoid = [](float v) { return 1.0F / (1.0F + std::exp(-v)); };
    const uint32_t rows = 4 * hidden;
    std::vector<float> h(state.begin(), state.begin() + hidden);
    std::vector<float> c(state.begin() + hidden, state.begin() + 2 * hidden);
    std::vector<float> gates(rows);
    for (uint32_t step = 0; step < steps; ++step) {
        const uint32_t t = reverse ? steps - 1 - step : step;
        for (uint32_t row = 0; row < rows; ++row) {
            float acc = gates_in[static_cast<size_t>(row) * steps + t];
            for (uint32_t j = 0; j < hidden; ++j) {
                acc += w_hh[static_cast<size_t>(row) * hidden + j] * h[j];
            }
            gates[row] = acc;
        }
        for (uint32_t r = 0; r < hidden; ++r) {
            const float i_gate = sigmoid(gates[r]);
            const float f_gate = sigmoid(gates[hidden + r]);
            const float g_gate = std::tanh(std::clamp(gates[2 * hidden + r], -15.0F, 15.0F));
            const float o_gate = sigmoid(gates[3 * hidden + r]);
            c[r] = f_gate * c[r] + i_gate * g_gate;
            h[r] = o_gate * std::tanh(std::clamp(c[r], -15.0F, 15.0F));
            hs[static_cast<size_t>(row_offset + r) * steps + t] = h[r];
        }
    }
    std::ranges::copy(h, state.begin());
    std::ranges::copy(c, state.begin() + hidden);
}

}
