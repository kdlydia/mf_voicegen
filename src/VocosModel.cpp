#include "VocosModel.hpp"

#include "VoiceKernels.hpp"

#include "MayaFlux/Journal/Archivist.hpp"
#include "MayaFlux/Yantra/Executors/GpuDispatchCore.hpp"
#include "MayaFlux/Yantra/Executors/ShaderExecutionContext.hpp"
#include "MayaFlux/Yantra/Executors/TensorKernels/Stages.hpp"
#include "MayaFlux/Yantra/Executors/TensorKernels/Weights.hpp"

namespace MayaFlux::Yantra {

namespace {

    using TensorKernels::Activation;
    using TensorKernels::NormAffine;
    using TensorKernels::TensorRequirement;
    using TensorKernels::UnaryOp;

    constexpr uint32_t k_fft = VocosModel::fft_size;
    constexpr uint32_t k_bins = VocosModel::bins;
    constexpr uint32_t k_mels = VocosModel::mels;
    constexpr uint32_t k_width = VocosModel::width;
    constexpr uint32_t k_expanded = 1536;
    constexpr uint32_t k_blocks = 8;
    constexpr uint32_t k_kernel = 7;
    constexpr uint32_t k_head = 2 * k_bins;
    constexpr float k_norm_epsilon = 1.0e-6F;
    constexpr float k_log_floor = 1.0e-7F;
    constexpr float k_magnitude_cap = 100.0F;

    enum Front : size_t {
        AnalysisWindow,
        MelFilters,
        EmbedWeight,
        EmbedBias,
        NormWeight,
        NormBias,
        FinalNormWeight,
        FinalNormBias,
        HeadWeight,
        HeadBias,
        SynthesisWindow,
        FrontCount
    };

    enum Block : size_t {
        DepthWeight,
        DepthBias,
        BlockNormWeight,
        BlockNormBias,
        ExpandWeight,
        ExpandBias,
        ReduceWeight,
        ReduceBias,
        Gamma,
        BlockCount
    };

    enum Derived : size_t {
        AnalysisBasis,
        MelWeights,
        SynthesisBasis,
        WindowSquared,
        DerivedCount
    };

    enum Buffer : uint32_t {
        Audio,
        Spectrum,
        Magnitude,
        Mel,
        Hidden,
        Work,
        Expanded,
        Stats,
        Complex,
        Frames,
        Output,
        WeightsFront,
        WeightsDerived,
        WeightsBlock0
    };

    std::vector<TensorRequirement> front_specs()
    {
        return {
            { "feature_extractor.mel_spec.spectrogram.window", { k_fft } },
            { "feature_extractor.mel_spec.mel_scale.fb", { k_bins, k_mels } },
            { "backbone.embed.weight", { k_width, k_mels, k_kernel } },
            { "backbone.embed.bias", { k_width } },
            { "backbone.norm.weight", { k_width } },
            { "backbone.norm.bias", { k_width } },
            { "backbone.final_layer_norm.weight", { k_width } },
            { "backbone.final_layer_norm.bias", { k_width } },
            { "head.out.weight", { k_head, k_width } },
            { "head.out.bias", { k_head } },
            { "head.istft.window", { k_fft } },
        };
    }

    std::vector<TensorRequirement> block_specs(uint32_t block)
    {
        const std::string prefix = std::format("backbone.convnext.{}.", block);
        return {
            { prefix + "dwconv.weight", { k_width, 1, k_kernel } },
            { prefix + "dwconv.bias", { k_width } },
            { prefix + "norm.weight", { k_width } },
            { prefix + "norm.bias", { k_width } },
            { prefix + "pwconv1.weight", { k_expanded, k_width } },
            { prefix + "pwconv1.bias", { k_expanded } },
            { prefix + "pwconv2.weight", { k_width, k_expanded } },
            { prefix + "pwconv2.bias", { k_width } },
            { prefix + "gamma", { k_width } },
        };
    }

    TensorKernels::PackedWeights pack(const Kakshya::TensorSet& weights, const std::vector<TensorRequirement>& specs)
    {
        auto packed = TensorKernels::pack_weights(weights, specs);
        if (!packed) {
            error<std::invalid_argument>(Journal::Component::Yantra, Journal::Context::ComputeMatrix,
                std::source_location::current(), "VocosModel: {}", packed.error());
        }
        return std::move(*packed);
    }

    std::span<const float> slot(const TensorKernels::PackedWeights& packed, size_t index, size_t count)
    {
        return { packed.values.data() + packed.offsets[index], count };
    }

    /** The layer scale multiplies the second pointwise layer's output, so it folds into that layer's rows and bias. */
    void fold_gamma(TensorKernels::PackedWeights& block)
    {
        const float* gamma = block.values.data() + block.offsets[Gamma];
        float* weight = block.values.data() + block.offsets[ReduceWeight];
        float* bias = block.values.data() + block.offsets[ReduceBias];
        for (uint32_t row = 0; row < k_width; ++row) {
            for (uint32_t column = 0; column < k_expanded; ++column) {
                weight[static_cast<size_t>(row) * k_expanded + column] *= gamma[row];
            }
            bias[row] *= gamma[row];
        }
    }

    /** The STFT, mel and inverse STFT as convolution weights, from the checkpoint's windows and filter bank. */
    TensorKernels::PackedWeights derive(const TensorKernels::PackedWeights& front)
    {
        const auto analysis = TensorKernels::dft_basis(slot(front, AnalysisWindow, k_fft));

        const auto filters = slot(front, MelFilters, static_cast<size_t>(k_bins) * k_mels);
        std::vector<float> mel_weights(static_cast<size_t>(k_mels) * k_bins);
        for (uint32_t bin = 0; bin < k_bins; ++bin) {
            for (uint32_t band = 0; band < k_mels; ++band) {
                mel_weights[static_cast<size_t>(band) * k_bins + bin] = filters[static_cast<size_t>(bin) * k_mels + band];
            }
        }

        const auto window = slot(front, SynthesisWindow, k_fft);
        const auto synthesis = TensorKernels::inverse_dft_basis(window);
        std::vector<float> squared(k_fft);
        for (uint32_t i = 0; i < k_fft; ++i) {
            squared[i] = window[i] * window[i];
        }

        TensorKernels::PackedWeights derived;
        for (const std::vector<float>* table : std::array<const std::vector<float>*, DerivedCount> { &analysis, &mel_weights, &synthesis, &squared }) {
            derived.offsets.push_back(derived.values.size());
            derived.values.insert(derived.values.end(), table->begin(), table->end());
        }
        return derived;
    }

    /** Reflect padding by half a window on both sides, as a centred STFT pads. */
    std::vector<float> centred(std::span<const float> samples)
    {
        constexpr size_t pad = k_fft / 2;
        std::vector<float> padded(samples.size() + 2 * pad);
        for (size_t i = 0; i < pad; ++i) {
            padded[i] = samples[pad - i];
            padded[pad + samples.size() + i] = samples[samples.size() - 2 - i];
        }
        std::ranges::copy(samples, padded.begin() + pad);
        return padded;
    }

}

struct VocosModel::Impl {
    std::shared_ptr<GpuDispatchRecorder> recording;
    std::unique_ptr<ShaderExecutionContext<>> context;
    TensorKernels::PackedWeights front;
    TensorKernels::PackedWeights derived;
    std::array<TensorKernels::PackedWeights, k_blocks> blocks;

    Impl(const Kakshya::TensorSet& weights, std::shared_ptr<GpuDispatchRecorder> shared)
        : recording(shared ? std::move(shared) : std::make_shared<GpuDispatchRecorder>(true))
        , context(std::make_unique<ShaderExecutionContext<>>(GpuComputeConfig {
              .shader_path = TensorKernels::conv1d_shader,
              .workgroup_size = { TensorKernels::element_workgroup, 1, 1 },
              .push_constant_size = sizeof(TensorKernels::Conv1dPC) }))
    {
        context->set_recording(recording);
        if (!context->ensure_gpu_ready()) {
            error<std::runtime_error>(Journal::Component::Yantra, Journal::Context::ComputeMatrix,
                std::source_location::current(), "VocosModel needs a graphics device and none is available");
        }

        front = pack(weights, front_specs());
        derived = derive(front);
        TensorKernels::upload_packed_weights(*context, 0, WeightsFront, front);
        TensorKernels::upload_packed_weights(*context, 0, WeightsDerived, derived);
        front.values = {};
        derived.values = {};

        for (uint32_t block = 0; block < k_blocks; ++block) {
            blocks[block] = pack(weights, block_specs(block));
            fold_gamma(blocks[block]);
            TensorKernels::upload_packed_weights(*context, 0, WeightsBlock0 + block, blocks[block]);
            blocks[block].values = {};
        }
    }

    [[nodiscard]] uint64_t address(uint32_t buffer) const
    {
        return context->shared_buffer_address(0, buffer);
    }

    void ensure_capacity(size_t padded, uint32_t frames)
    {
        using Portal::Graphics::BufferUsageHint;
        constexpr auto type = GpuBufferBinding::ElementType::FLOAT32;
        const size_t t = frames;
        context->ensure_shared_buffer(0, Audio, std::max<size_t>(padded, 1), type, BufferUsageHint::COMPUTE);
        context->ensure_shared_buffer(0, Spectrum, k_head * t, type, BufferUsageHint::COMPUTE);
        context->ensure_shared_buffer(0, Magnitude, k_bins * t, type, BufferUsageHint::COMPUTE);
        context->ensure_shared_buffer(0, Mel, k_mels * t, type);
        context->ensure_shared_buffer(0, Hidden, k_width * t, type, BufferUsageHint::COMPUTE);
        context->ensure_shared_buffer(0, Work, k_width * t, type, BufferUsageHint::COMPUTE);
        context->ensure_shared_buffer(0, Expanded, k_expanded * t, type, BufferUsageHint::COMPUTE);
        context->ensure_shared_buffer(0, Stats, 2 * t, type, BufferUsageHint::COMPUTE);
        context->ensure_shared_buffer(0, Complex, k_head * t, type, BufferUsageHint::COMPUTE);
        context->ensure_shared_buffer(0, Frames, k_fft * t, type, BufferUsageHint::COMPUTE);
        context->ensure_shared_buffer(0, Output, std::max<size_t>(sample_count(frames), 1), type);
    }

    void record_mel(std::vector<DependencyStage>& stages, std::span<const float> samples, uint32_t frames)
    {
        const auto padded = centred(samples);
        context->upload_shared_raw(0, Audio, reinterpret_cast<const uint8_t*>(padded.data()), padded.size() * sizeof(float));

        const auto derived_at = [&](size_t index) { return derived.address(address(WeightsDerived), index); };
        stages.push_back(TensorKernels::conv1d_stage({ .c_in = 1, .c_out = k_head, .kernel = k_fft, .stride = VocosModel::hop },
            address(Audio), derived_at(AnalysisBasis), 0, address(Spectrum), 1, static_cast<uint32_t>(padded.size())));
        stages.push_back(TensorKernels::magnitude_stage(address(Spectrum), address(Magnitude), 1, k_bins, frames));
        stages.push_back(TensorKernels::conv1d_stage({ .c_in = k_bins, .c_out = k_mels, .kernel = 1 },
            address(Magnitude), derived_at(MelWeights), 0, address(Mel), 1, frames));
        stages.push_back(TensorKernels::unary_stage(UnaryOp::Log, address(Mel), address(Mel), k_mels * frames, k_log_floor));
    }

    void record_synthesis(std::vector<DependencyStage>& stages, uint32_t frames)
    {
        const auto front_at = [&](size_t index) { return front.address(address(WeightsFront), index); };
        const auto derived_at = [&](size_t index) { return derived.address(address(WeightsDerived), index); };
        const auto block_at = [&](uint32_t block, size_t index) { return blocks[block].address(address(WeightsBlock0 + block), index); };

        const auto dense = [&](uint64_t src, uint32_t in, uint64_t weight, uint64_t bias, uint64_t dst, uint32_t out,
                               Activation activation = Activation::None) {
            stages.push_back(TensorKernels::conv1d_stage({ .c_in = in, .c_out = out, .kernel = 1, .activation = activation },
                src, weight, bias, dst, 1, frames));
        };
        const auto layer_norm = [&](uint64_t data, uint64_t weight, uint64_t bias) {
            stages.push_back(TensorKernels::normalize_stats_stage(data, address(Stats), 1, k_width, frames, k_norm_epsilon));
            stages.push_back(TensorKernels::normalize_apply_stage(data, address(Stats), weight, bias, data, 1, k_width, frames, NormAffine::Along));
        };

        stages.push_back(TensorKernels::conv1d_stage({ .c_in = k_mels, .c_out = k_width, .kernel = k_kernel, .pad = k_kernel / 2 },
            address(Mel), front_at(EmbedWeight), front_at(EmbedBias), address(Hidden), 1, frames));
        layer_norm(address(Hidden), front_at(NormWeight), front_at(NormBias));

        for (uint32_t block = 0; block < k_blocks; ++block) {
            stages.push_back(TensorKernels::grouped_conv1d_stage(
                { .c_in = k_width, .c_out = k_width, .kernel = k_kernel, .groups = k_width, .pad = k_kernel / 2 },
                address(Hidden), block_at(block, DepthWeight), block_at(block, DepthBias), address(Work), 1, frames));
            layer_norm(address(Work), block_at(block, BlockNormWeight), block_at(block, BlockNormBias));
            dense(address(Work), k_width, block_at(block, ExpandWeight), block_at(block, ExpandBias), address(Expanded), k_expanded, Activation::Gelu);
            dense(address(Expanded), k_expanded, block_at(block, ReduceWeight), block_at(block, ReduceBias), address(Work), k_width);
            stages.push_back(TensorKernels::add_stage(address(Hidden), address(Work), address(Hidden), k_width * frames));
        }
        layer_norm(address(Hidden), front_at(FinalNormWeight), front_at(FinalNormBias));

        const uint32_t half = k_bins * frames;
        const uint64_t magnitude = address(Spectrum);
        const uint64_t phase = magnitude + static_cast<uint64_t>(half) * sizeof(float);
        const uint64_t real = address(Complex);
        const uint64_t imaginary = real + static_cast<uint64_t>(half) * sizeof(float);

        dense(address(Hidden), k_width, front_at(HeadWeight), front_at(HeadBias), address(Spectrum), k_head);
        stages.push_back(TensorKernels::unary_stage(UnaryOp::Exp, magnitude, magnitude, half, k_magnitude_cap));
        stages.push_back(TensorKernels::unary_stage(UnaryOp::Cos, phase, real, half));
        stages.push_back(TensorKernels::unary_stage(UnaryOp::Sin, phase, imaginary, half));
        stages.push_back(TensorKernels::multiply_stage(real, magnitude, real, half));
        stages.push_back(TensorKernels::multiply_stage(imaginary, magnitude, imaginary, half));
        dense(real, k_head, derived_at(SynthesisBasis), 0, address(Frames), k_fft);
        stages.push_back(TensorKernels::overlap_add_stage(address(Frames), derived_at(WindowSquared), address(Output),
            k_fft, frames, VocosModel::hop, k_fft / 2, static_cast<uint32_t>(sample_count(frames))));
    }

    std::vector<float> download(uint32_t buffer, size_t count)
    {
        std::vector<float> out(count);
        context->download_shared(0, buffer, out.data(), out.size() * sizeof(float));
        return out;
    }
};

std::string VocosModel::diagnose(const Kakshya::TensorSet& weights)
{
    if (std::string reason = TensorKernels::check_tensors(weights, front_specs()); !reason.empty()) {
        return reason;
    }
    for (uint32_t block = 0; block < k_blocks; ++block) {
        if (std::string reason = TensorKernels::check_tensors(weights, block_specs(block)); !reason.empty()) {
            return reason;
        }
    }
    return {};
}

uint32_t VocosModel::frame_count(size_t samples) noexcept
{
    if (samples <= fft_size / 2 || samples / hop >= std::numeric_limits<uint32_t>::max()) {
        return 0;
    }
    return static_cast<uint32_t>(1 + samples / hop);
}

size_t VocosModel::sample_count(uint32_t frames) noexcept
{
    return frames < 2 ? 0 : static_cast<size_t>(hop) * (frames - 1);
}

VocosModel::VocosModel(const Kakshya::TensorSet& weights)
    : VocosModel(weights, nullptr)
{
}

VocosModel::VocosModel(const Kakshya::TensorSet& weights, std::shared_ptr<GpuDispatchRecorder> recording)
{
    if (const std::string reason = diagnose(weights); !reason.empty()) {
        error<std::invalid_argument>(Journal::Component::Yantra, Journal::Context::ComputeMatrix,
            std::source_location::current(), "VocosModel: {}", reason);
    }
    m_impl = std::make_unique<Impl>(weights, std::move(recording));
}

VocosModel::~VocosModel() = default;

std::vector<float> VocosModel::mel(std::span<const float> samples)
{
    const uint32_t frames = frame_count(samples.size());
    if (frames == 0) {
        return {};
    }
    m_impl->ensure_capacity(samples.size() + fft_size, frames);
    std::vector<DependencyStage> stages;
    m_impl->record_mel(stages, samples, frames);
    m_impl->context->record_stages(stages);
    return m_impl->download(Mel, static_cast<size_t>(mels) * frames);
}

std::vector<float> VocosModel::synthesize(std::span<const float> mel, uint32_t frames)
{
    if (frames < 2 || mel.size() != static_cast<size_t>(mels) * frames) {
        error<std::invalid_argument>(Journal::Component::Yantra, Journal::Context::ComputeMatrix,
            std::source_location::current(), "VocosModel: synthesize needs at least 2 frames of {} bands; it was given {} values for {} frames",
            mels, mel.size(), frames);
    }
    m_impl->ensure_capacity(0, frames);
    m_impl->context->upload_shared_raw(0, Mel, reinterpret_cast<const uint8_t*>(mel.data()), mel.size() * sizeof(float));
    std::vector<DependencyStage> stages;
    m_impl->record_synthesis(stages, frames);
    m_impl->context->record_stages(stages);
    return m_impl->download(Output, sample_count(frames));
}

std::vector<float> VocosModel::resynthesize(std::span<const float> samples)
{
    const uint32_t frames = frame_count(samples.size());
    if (frames < 2) {
        return {};
    }
    m_impl->ensure_capacity(samples.size() + fft_size, frames);
    std::vector<DependencyStage> stages;
    m_impl->record_mel(stages, samples, frames);
    m_impl->record_synthesis(stages, frames);
    m_impl->context->record_stages(stages);
    return m_impl->download(Output, sample_count(frames));
}

} // namespace MayaFlux::Yantra
