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
        }
        return x;
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

std::vector<float> multiply(std::span<const float> a, std::span<const float> b, uint32_t inner)
{
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] * b[(i / inner) % b.size()];
    }
    return out;
}

std::vector<float> conv_transpose1d(const ConvTranspose1d& layer, std::span<const float> x,
    std::span<const float> weights, std::span<const float> bias, uint32_t batch, uint32_t t_in)
{
    const uint32_t t_out = conv_transpose1d_length(layer, t_in);
    std::vector<float> out(static_cast<size_t>(batch) * layer.c_out * t_out, 0.0F);
    for (uint32_t n = 0; n < batch; ++n) {
        for (uint32_t ci = 0; ci < layer.c_in; ++ci) {
            for (uint32_t ti = 0; ti < t_in; ++ti) {
                float value = x[(static_cast<size_t>(n) * layer.c_in + ci) * t_in + ti];
                if (layer.relu_input) {
                    value = std::max(value, 0.0F);
                }
                for (uint32_t co = 0; co < layer.c_out; ++co) {
                    for (uint32_t k = 0; k < layer.kernel; ++k) {
                        const int64_t to = static_cast<int64_t>(ti) * layer.stride
                            + static_cast<int64_t>(k) * layer.dilation - layer.pad;
                        if (to < 0 || to >= t_out) {
                            continue;
                        }
                        out[(static_cast<size_t>(n) * layer.c_out + co) * t_out + static_cast<size_t>(to)]
                            += weights[(static_cast<size_t>(ci) * layer.c_out + co) * layer.kernel + k] * value;
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

}
