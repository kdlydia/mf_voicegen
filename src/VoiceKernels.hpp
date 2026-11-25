#pragma once

#include "MayaFlux/Yantra/Executors/TensorKernels/Stages.hpp"

/**
 * @file VoiceKernels.hpp
 * @brief Tensor kernels that vocoders need beyond MayaFlux's own, in the same style.
 *
 * Four kernels: a unary function, a broadcasting multiply, a transposed
 * convolution and an overlap-add. Each has a push constant layout, a function
 * that returns its DependencyStage, and a CPU version that matches it to float
 * precision. dft_basis and inverse_dft_basis give the weights that turn a
 * pointwise or strided convolution into a short-time Fourier transform and
 * back. The kernels are GLSL files in the module's shaders folder, compiled by
 * ShaderFoundry when first used; mf_voicegen.cmake defines MF_VOICEGEN_SHADER_DIR.
 */

#ifndef MF_VOICEGEN_SHADER_DIR
#error "MF_VOICEGEN_SHADER_DIR must name mf_voicegen's shaders folder (mf_voicegen.cmake defines it)"
#endif

namespace MayaFlux::Yantra::TensorKernels {

inline constexpr const char* unary_shader = "nn_unary.comp";
inline constexpr const char* multiply_shader = "nn_multiply.comp";
inline constexpr const char* conv_transpose1d_shader = "nn_conv_transpose1d.comp";
inline constexpr const char* overlap_add_shader = "nn_overlap_add.comp";

/** @brief Full path of one of the module's kernels. */
[[nodiscard]] inline std::string voice_shader(std::string_view file)
{
    return std::string(MF_VOICEGEN_SHADER_DIR) + "/" + std::string(file);
}

/**
 * @enum UnaryOp
 * @brief The function nn_unary applies. alpha and beta are its two settings.
 */
enum class UnaryOp : uint32_t {
    Affine, ///< alpha * x + beta
    Exp, ///< min(exp(x), alpha)
    Log, ///< log(max(x, alpha))
    Sin,
    Cos,
    Tanh,
    LeakyRelu, ///< x, or alpha * x below zero
    Silu, ///< x * sigmoid(x)
    Snake, ///< x + sin(a x)^2 / a, a per channel or alpha
    Clamp ///< min(max(x, alpha), beta)
};

/** @brief Push constants of nn_unary. */
struct UnaryPC {
    uint64_t src;
    uint64_t dst;
    uint64_t params;
    uint32_t count;
    uint32_t op;
    uint32_t inner;
    uint32_t channels;
    float alpha;
    float beta;
};

/** @brief Push constants of nn_multiply. */
struct MultiplyPC {
    uint64_t a;
    uint64_t b;
    uint64_t dst;
    uint32_t count;
    uint32_t inner;
    uint32_t b_count;
    uint32_t reserved;
};

/** @brief Push constants of nn_conv_transpose1d, laid out as nn_conv1d's. */
struct ConvTranspose1dPC {
    uint64_t src;
    uint64_t weights;
    uint64_t bias;
    uint64_t dst;
    uint32_t batch;
    uint32_t c_in;
    uint32_t t_in;
    uint32_t c_out;
    uint32_t t_out;
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t pad;
    uint32_t flags;
    uint32_t dilation;
};

/** @brief Push constants of nn_overlap_add. */
struct OverlapAddPC {
    uint64_t frames;
    uint64_t window_sq;
    uint64_t dst;
    uint32_t frame_length;
    uint32_t count;
    uint32_t hop;
    uint32_t trim;
    uint32_t samples;
    uint32_t normalise;
};

static_assert(sizeof(UnaryPC) == 48);
static_assert(sizeof(MultiplyPC) == 40);
static_assert(sizeof(ConvTranspose1dPC) == 72);
static_assert(sizeof(OverlapAddPC) == 48);

/**
 * @struct ConvTranspose1d
 * @brief One transposed convolution over [batch, c_in, t_in], with weights [c_in, c_out, kernel] as PyTorch stores them.
 *
 * Input step ti tap k lands on output ti * stride + k * dilation - pad.
 * output_pad adds outputs at the end, as PyTorch's output_padding does.
 */
struct ConvTranspose1d {
    uint32_t c_in {};
    uint32_t c_out {};
    uint32_t kernel {};
    uint32_t stride { 1 };
    uint32_t pad { 0 };
    uint32_t output_pad { 0 };
    uint32_t dilation { 1 };
    bool relu_input { false };
    Activation activation { Activation::None };
};

/**
 * @brief Output length of a transposed convolution, 0 when nothing is left after the padding.
 * @throws std::invalid_argument When kernel, stride or dilation is 0.
 */
[[nodiscard]] inline uint32_t conv_transpose1d_length(const ConvTranspose1d& layer, uint32_t t_in)
{
    if (layer.kernel == 0 || layer.stride == 0 || layer.dilation == 0) {
        detail::refuse("ConvTranspose1d needs a kernel, stride and dilation of at least 1");
    }
    if (t_in == 0) {
        return 0;
    }
    const int64_t full = static_cast<int64_t>(t_in - 1) * layer.stride
        + static_cast<int64_t>(layer.dilation) * (layer.kernel - 1) + layer.output_pad + 1;
    const int64_t length = full - 2 * static_cast<int64_t>(layer.pad);
    return length > 0 ? static_cast<uint32_t>(length) : 0U;
}

/**
 * @brief One nn_unary dispatch over count floats. dst may be src.
 * @param op The function. Snake here uses alpha for every element; snake_stage takes one per channel.
 * @throws std::invalid_argument When count is 0.
 */
[[nodiscard]] inline DependencyStage unary_stage(UnaryOp op, uint64_t src, uint64_t dst, uint32_t count,
    float alpha = 0.0F, float beta = 0.0F)
{
    if (count == 0) {
        detail::refuse("Unary needs at least one element");
    }
    const UnaryPC pc { .src = src, .dst = dst, .params = 0, .count = count, .op = static_cast<uint32_t>(op),
        .inner = 1, .channels = 0, .alpha = alpha, .beta = beta };
    return detail::make_grid_stage(voice_shader(unary_shader).c_str(), pc, count);
}

/**
 * @brief One nn_unary Snake dispatch over [outer, channels, inner] with one a per channel. dst may be src.
 * @param params Device address of the channels values of a.
 * @throws std::invalid_argument When an extent is 0.
 */
[[nodiscard]] inline DependencyStage snake_stage(uint64_t src, uint64_t params, uint64_t dst,
    uint32_t outer, uint32_t channels, uint32_t inner)
{
    if (outer == 0 || channels == 0 || inner == 0) {
        detail::refuse("Snake needs an outer extent, channels and an inner extent");
    }
    const uint32_t count = outer * channels * inner;
    const UnaryPC pc { .src = src, .dst = dst, .params = params, .count = count, .op = static_cast<uint32_t>(UnaryOp::Snake),
        .inner = inner, .channels = channels, .alpha = 1.0F, .beta = 0.0F };
    return detail::make_grid_stage(voice_shader(unary_shader).c_str(), pc, count);
}

/**
 * @brief One nn_multiply dispatch: dst[i] = a[i] * b[(i / inner) % b_count]. dst may be a.
 * @param b_count Values of b, 0 for count.
 * @throws std::invalid_argument When count or inner is 0.
 */
[[nodiscard]] inline DependencyStage multiply_stage(uint64_t a, uint64_t b, uint64_t dst, uint32_t count,
    uint32_t inner = 1, uint32_t b_count = 0)
{
    if (count == 0 || inner == 0) {
        detail::refuse("Multiply needs at least one element and an inner extent");
    }
    const MultiplyPC pc { .a = a, .b = b, .dst = dst, .count = count, .inner = inner,
        .b_count = b_count == 0 ? count : b_count, .reserved = 0 };
    return detail::make_grid_stage(voice_shader(multiply_shader).c_str(), pc, count);
}

/**
 * @brief One nn_conv_transpose1d dispatch: [batch, c_in, t_in] to [batch, c_out, conv_transpose1d_length].
 * @param bias Device address of the bias, [c_out], or 0 for none.
 * @throws std::invalid_argument When the layer is empty or the output would be empty.
 */
[[nodiscard]] inline DependencyStage conv_transpose1d_stage(const ConvTranspose1d& layer, uint64_t src,
    uint64_t weights, uint64_t bias, uint64_t dst, uint32_t batch, uint32_t t_in)
{
    const uint32_t t_out = conv_transpose1d_length(layer, t_in);
    if (layer.c_in == 0 || layer.c_out == 0 || batch == 0 || t_out == 0) {
        detail::refuse("ConvTranspose1d has no input channels, output channels, batch or output length");
    }
    const uint32_t flags = (bias != 0 ? conv1d_flag_bias : 0U)
        | (layer.relu_input ? conv1d_flag_relu_input : 0U)
        | (static_cast<uint32_t>(layer.activation) << conv1d_activation_shift);
    const ConvTranspose1dPC pc {
        .src = src,
        .weights = weights,
        .bias = bias,
        .dst = dst,
        .batch = batch,
        .c_in = layer.c_in,
        .t_in = t_in,
        .c_out = layer.c_out,
        .t_out = t_out,
        .kernel_size = layer.kernel,
        .stride = layer.stride,
        .pad = layer.pad,
        .flags = flags,
        .dilation = layer.dilation,
    };
    return detail::make_grid_stage(voice_shader(conv_transpose1d_shader).c_str(), pc,
        static_cast<size_t>(batch) * layer.c_out * t_out);
}

/**
 * @brief One nn_overlap_add dispatch: count frames of frame_length, hop apart, summed into samples values.
 *
 * frames is [frame_length, count]. Output t is position t + trim of the sum.
 * With a window_sq address, each output is divided by the summed squared
 * window at its position, as an inverse STFT does.
 *
 * @param window_sq Device address of frame_length squared window values, or 0 to only sum.
 * @throws std::invalid_argument When frame_length, count, hop or samples is 0.
 */
[[nodiscard]] inline DependencyStage overlap_add_stage(uint64_t frames, uint64_t window_sq, uint64_t dst,
    uint32_t frame_length, uint32_t count, uint32_t hop, uint32_t trim, uint32_t samples)
{
    if (frame_length == 0 || count == 0 || hop == 0 || samples == 0) {
        detail::refuse("Overlap-add needs a frame length, frames, a hop and output samples");
    }
    const OverlapAddPC pc { .frames = frames, .window_sq = window_sq, .dst = dst, .frame_length = frame_length,
        .count = count, .hop = hop, .trim = trim, .samples = samples, .normalise = window_sq != 0 ? 1U : 0U };
    return detail::make_grid_stage(voice_shader(overlap_add_shader).c_str(), pc, samples);
}

/**
 * @brief Weights of a strided convolution that computes a windowed DFT: [2 * (n / 2 + 1), 1, n].
 *
 * Over a signal [1, 1, length] with stride hop, the output rows are the real
 * parts of bins 0 to n / 2, then the imaginary parts, one column per frame,
 * ready for magnitude_stage.
 *
 * @param window The analysis window, n values.
 */
[[nodiscard]] inline std::vector<float> dft_basis(std::span<const float> window)
{
    const size_t n = window.size();
    const size_t bins = n / 2 + 1;
    std::vector<float> basis(2 * bins * n);
    for (size_t k = 0; k < bins; ++k) {
        for (size_t t = 0; t < n; ++t) {
            const double angle = 2.0 * std::numbers::pi * static_cast<double>((k * t) % n) / static_cast<double>(n);
            basis[k * n + t] = static_cast<float>(window[t] * std::cos(angle));
            basis[(bins + k) * n + t] = static_cast<float>(-window[t] * std::sin(angle));
        }
    }
    return basis;
}

/**
 * @brief Weights of a pointwise convolution that computes windowed inverse real DFT frames: [n, 2 * (n / 2 + 1), 1].
 *
 * The input rows are the real parts of bins 0 to n / 2, then the imaginary
 * parts. The output row t is sample t of each frame times window[t], scaled by
 * 1 / n as an inverse FFT with backward normalisation. The imaginary parts of
 * bin 0 and bin n / 2 are ignored, as irfft ignores them. Overlap-add with the
 * squared window then completes an inverse STFT.
 *
 * @param window The synthesis window, n values. n must be even.
 */
[[nodiscard]] inline std::vector<float> inverse_dft_basis(std::span<const float> window)
{
    const size_t n = window.size();
    const size_t bins = n / 2 + 1;
    std::vector<float> basis(n * 2 * bins);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < bins; ++k) {
            const double angle = 2.0 * std::numbers::pi * static_cast<double>((k * t) % n) / static_cast<double>(n);
            const double weight = (k == 0 || k == n / 2 ? 1.0 : 2.0) * window[t] / static_cast<double>(n);
            const bool edge = k == 0 || k == n / 2;
            basis[t * 2 * bins + k] = static_cast<float>(weight * std::cos(angle));
            basis[t * 2 * bins + bins + k] = edge ? 0.0F : static_cast<float>(-weight * std::sin(angle));
        }
    }
    return basis;
}

/**
 * @namespace MayaFlux::Yantra::TensorKernels::cpu
 * @brief CPU versions of the kernels above, for tests and machines without a device. Each matches its kernel to float precision.
 */
namespace cpu {

    [[nodiscard]] std::vector<float> unary(UnaryOp op, std::span<const float> x, float alpha = 0.0F, float beta = 0.0F);

    /** @brief Snake over [outer, channels, inner] with one a per channel. */
    [[nodiscard]] std::vector<float> snake(std::span<const float> x, std::span<const float> a, uint32_t channels, uint32_t inner);

    [[nodiscard]] std::vector<float> multiply(std::span<const float> a, std::span<const float> b, uint32_t inner = 1);

    [[nodiscard]] std::vector<float> conv_transpose1d(const ConvTranspose1d& layer, std::span<const float> x,
        std::span<const float> weights, std::span<const float> bias, uint32_t batch, uint32_t t_in);

    /** @param window_sq Squared window, or empty to only sum. */
    [[nodiscard]] std::vector<float> overlap_add(std::span<const float> frames, std::span<const float> window_sq,
        uint32_t frame_length, uint32_t count, uint32_t hop, uint32_t trim, uint32_t samples);

}

} // namespace MayaFlux::Yantra::TensorKernels
