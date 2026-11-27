#include "MayaFlux/Core/Backends/Graphics/Vulkan/VulkanBackend.hpp"
#include "MayaFlux/Core/GlobalGraphicsInfo.hpp"
#include "MayaFlux/Journal/Archivist.hpp"
#include "MayaFlux/Portal/Graphics/ComputePress.hpp"
#include "MayaFlux/Portal/Graphics/ShaderFoundry.hpp"
#include "MayaFlux/Yantra/Executors/GpuResourceManager.hpp"
#include "MayaFlux/Yantra/Executors/ShaderExecutionContext.hpp"

#include "TorchFormat.hpp"
#include "VocosModel.hpp"
#include "VoiceKernels.hpp"

#include <chrono>
#include <fstream>
#include <random>

#include <gtest/gtest.h>

// Brings up a headless Vulkan device, as MayaFlux's own GPU tests do, runs each
// kernel against its CPU version, then runs Vocos on a LibriSpeech excerpt.
// When generate_reference.py has written the Vocos fixtures, the spectrogram
// and audio are compared with the original PyTorch code; otherwise only
// self-consistency is checked and the comparison is skipped.

namespace {

using namespace MayaFlux;
namespace fs = std::filesystem;
namespace TK = Yantra::TensorKernels;
using Clock = std::chrono::steady_clock;

const fs::path k_root = MF_VOICEGEN_TEST_DIR;
const fs::path k_vocos = k_root / "data/tensors/vocos_mel_24khz/pytorch_model.bin";

/** One headless Vulkan device per test process, brought up as MayaFlux's own GPU tests do. */
class Device : public ::testing::Environment {
public:
    enum class State : uint8_t {
        Absent,
        Broken,
        Ready
    };

    void SetUp() override
    {
        Core::GlobalGraphicsConfig config;
        config.windowing_backend = Core::GlobalGraphicsConfig::WindowingBackend::NONE;
        config.backend_info.require_presentation = false;
        m_backend = std::make_shared<Core::VulkanBackend>();
        if (!m_backend->initialize(config)) {
            m_backend.reset();
            return;
        }
        s_state = Portal::Graphics::get_shader_foundry().initialize(m_backend) && Portal::Graphics::get_compute_press().initialize()
            ? State::Ready
            : State::Broken;
    }

    void TearDown() override
    {
        if (s_state == State::Ready) {
            Portal::Graphics::get_compute_press().shutdown();
            Portal::Graphics::get_shader_foundry().shutdown();
        }
        if (m_backend) {
            m_backend->cleanup();
        }
        s_state = State::Absent;
    }

    static State state() { return s_state; }

private:
    std::shared_ptr<Core::VulkanBackend> m_backend;
    static inline State s_state { State::Absent };
};

/** Skips a test without a device, and fails it when the device is there but the shader foundry did not start. */
class Gpu : public ::testing::Test {
protected:
    void SetUp() override
    {
        if (Device::state() == Device::State::Absent) {
            GTEST_SKIP() << "no usable Vulkan device";
        }
        ASSERT_EQ(Device::state(), Device::State::Ready) << "the shader foundry or compute press did not start";
    }
};

double worst(std::span<const float> a, std::span<const float> b)
{
    if (a.size() != b.size() || a.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    double error = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        error = std::max(error, std::abs(static_cast<double>(a[i]) - b[i]) / (1.0 + std::abs(static_cast<double>(b[i]))));
    }
    return error;
}

std::vector<float> random_values(size_t count, float low, float high, uint32_t seed)
{
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> distribution(low, high);
    std::vector<float> out(count);
    for (auto& value : out) {
        value = distribution(engine);
    }
    return out;
}

std::vector<float> read_f32(const fs::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }
    std::vector<float> out(static_cast<size_t>(file.tellg()) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
    return out;
}

void write_wav(const fs::path& path, std::span<const float> samples, uint32_t rate)
{
    std::ofstream file(path, std::ios::binary);
    const auto put = [&](uint32_t value, int bytes) {
        for (int i = 0; i < bytes; ++i) {
            file.put(static_cast<char>((value >> (8 * i)) & 0xFF));
        }
    };
    const auto data_bytes = static_cast<uint32_t>(samples.size() * 2);
    file.write("RIFF", 4);
    put(36 + data_bytes, 4);
    file.write("WAVEfmt ", 8);
    put(16, 4);
    put(1, 2);
    put(1, 2);
    put(rate, 4);
    put(rate * 2, 4);
    put(2, 2);
    put(16, 2);
    file.write("data", 4);
    put(data_bytes, 4);
    for (const float sample : samples) {
        put(static_cast<uint16_t>(static_cast<int16_t>(std::lround(std::clamp(sample, -1.0F, 1.0F) * 32767.0F))), 2);
    }
}

/** A context with a shared recording, as the models use, and buffers by binding. */
struct Bench {
    std::shared_ptr<Yantra::GpuDispatchRecorder> recording = std::make_shared<Yantra::GpuDispatchRecorder>(true);
    std::unique_ptr<Yantra::ShaderExecutionContext<>> context;
    bool ready {};

    Bench()
        : context(std::make_unique<Yantra::ShaderExecutionContext<>>(Yantra::GpuComputeConfig {
              .shader_path = TK::conv1d_shader,
              .workgroup_size = { TK::element_workgroup, 1, 1 },
              .push_constant_size = sizeof(TK::Conv1dPC) }))
    {
        context->set_recording(recording);
        ready = context->ensure_gpu_ready();
    }

    uint64_t put(uint32_t binding, const std::vector<float>& values)
    {
        context->ensure_shared_buffer(0, binding, std::max<size_t>(values.size(), 1),
            Yantra::GpuBufferBinding::ElementType::FLOAT32, Portal::Graphics::BufferUsageHint::COMPUTE);
        context->upload_shared_raw(0, binding, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(float));
        return context->shared_buffer_address(0, binding);
    }

    uint64_t space(uint32_t binding, size_t count)
    {
        context->ensure_shared_buffer(0, binding, count, Yantra::GpuBufferBinding::ElementType::FLOAT32);
        return context->shared_buffer_address(0, binding);
    }

    std::vector<float> get(uint32_t binding, size_t count)
    {
        std::vector<float> out(count);
        context->download_shared(0, binding, out.data(), count * sizeof(float));
        return out;
    }

    void run(std::vector<Yantra::DependencyStage> stages) { context->record_stages(stages); }
};

TEST_F(Gpu, UnaryMatchesTheCpu)
{
    struct Case {
        TK::UnaryOp op;
        float alpha;
        float beta;
        float low;
        float high;
        std::string_view name;
    };
    const std::array<Case, 12> cases { {
        { TK::UnaryOp::Sin, 0.0F, 0.0F, -1000.0F, 1000.0F, "Sin far from zero" },
        { TK::UnaryOp::Cos, 0.0F, 0.0F, -1000.0F, 1000.0F, "Cos far from zero" },
        { TK::UnaryOp::Affine, 1.5F, -0.25F, -4.0F, 4.0F, "Affine" },
        { TK::UnaryOp::Exp, 20.0F, 0.0F, -4.0F, 4.0F, "Exp" },
        { TK::UnaryOp::Log, 1.0e-7F, 0.0F, 0.0F, 3.0F, "Log" },
        { TK::UnaryOp::Sin, 0.0F, 0.0F, -4.0F, 4.0F, "Sin" },
        { TK::UnaryOp::Cos, 0.0F, 0.0F, -4.0F, 4.0F, "Cos" },
        { TK::UnaryOp::Tanh, 0.0F, 0.0F, -20.0F, 20.0F, "Tanh" },
        { TK::UnaryOp::LeakyRelu, 0.1F, 0.0F, -4.0F, 4.0F, "LeakyRelu" },
        { TK::UnaryOp::Silu, 0.0F, 0.0F, -8.0F, 8.0F, "Silu" },
        { TK::UnaryOp::Snake, 0.7F, 0.0F, -4.0F, 4.0F, "Snake with one a" },
        { TK::UnaryOp::Clamp, -1.0F, 0.5F, -4.0F, 4.0F, "Clamp" },
    } };
    for (const auto& c : cases) {
        Bench bench;
        const auto x = random_values(1000, c.low, c.high, 11);
        const uint64_t src = bench.put(0, x);
        const uint64_t dst = bench.space(1, x.size());
        bench.run({ TK::unary_stage(c.op, src, dst, static_cast<uint32_t>(x.size()), c.alpha, c.beta) });
        const double error = worst(bench.get(1, x.size()), TK::cpu::unary(c.op, x, c.alpha, c.beta));
        EXPECT_TRUE(error < 1.0e-5) << std::format("nn_unary {} matches the CPU (worst {:.2e})", c.name, error);
    }

    Bench bench;
    const auto x = random_values(2 * 3 * 50, -4.0F, 4.0F, 12);
    const auto a = random_values(3, 0.5F, 1.5F, 13);
    const uint64_t src = bench.put(0, x);
    const uint64_t params = bench.put(1, a);
    bench.space(2, x.size());
    bench.run({ TK::snake_stage(src, params, bench.space(2, x.size()), 2, 3, 50) });
    const double error = worst(bench.get(2, x.size()), TK::cpu::snake(x, a, 3, 50));
    EXPECT_TRUE(error < 1.0e-5) << std::format("nn_unary Snake with one a per channel matches the CPU (worst {:.2e})", error);
}

TEST_F(Gpu, LargeDispatchReachesEveryElement)
{
    Bench bench;
    constexpr size_t count = 3'000'000;
    std::vector<float> x(count);
    std::iota(x.begin(), x.end(), 0.0F);
    const uint64_t src = bench.put(0, x);
    bench.run({ TK::unary_stage(TK::UnaryOp::Affine, src, bench.space(1, count), count, 2.0F, 1.0F) });
    const auto y = bench.get(1, count);
    EXPECT_TRUE(y[0] == 1.0F && y[count / 2] == 2.0F * static_cast<float>(count / 2) + 1.0F && y[count - 1] == 2.0F * static_cast<float>(count - 1) + 1.0F) << "a dispatch past one axis of workgroups reaches every element";
}

TEST_F(Gpu, MultiplyMatchesTheCpu)
{
    Bench bench;
    const auto a = random_values(4 * 30, -2.0F, 2.0F, 21);
    const auto b = random_values(4, -2.0F, 2.0F, 22);
    const uint64_t pa = bench.put(0, a);
    const uint64_t pb = bench.put(1, b);
    bench.run({ TK::multiply_stage(pa, pb, bench.space(2, a.size()), static_cast<uint32_t>(a.size()), 30, 4) });
    EXPECT_TRUE(bench.get(2, a.size()) == TK::cpu::multiply(a, b, 30)) << "nn_multiply scales each channel of [4, 30] exactly as the CPU";
}

TEST_F(Gpu, ConvTransposeMatchesTheCpu)
{
    Bench bench;
    const TK::ConvTranspose1d layer { .c_in = 5, .c_out = 6, .kernel = 7, .stride = 4, .pad = 2, .output_pad = 1,
        .dilation = 1, .relu_input = true, .activation = TK::Activation::Gelu };
    constexpr uint32_t batch = 2;
    constexpr uint32_t t_in = 40;
    const auto x = random_values(static_cast<size_t>(batch) * layer.c_in * t_in, -2.0F, 2.0F, 31);
    const auto w = random_values(static_cast<size_t>(layer.c_in) * layer.c_out * layer.kernel, -0.5F, 0.5F, 32);
    const auto b = random_values(layer.c_out, -0.5F, 0.5F, 33);
    const uint32_t t_out = TK::conv_transpose1d_length(layer, t_in);
    const size_t out_count = static_cast<size_t>(batch) * layer.c_out * t_out;
    const uint64_t px = bench.put(0, x);
    const uint64_t pw = bench.put(1, w);
    const uint64_t pb = bench.put(2, b);
    bench.run({ TK::conv_transpose1d_stage(layer, px, pw, pb, bench.space(3, out_count), batch, t_in) });
    const double error = worst(bench.get(3, out_count), TK::cpu::conv_transpose1d(layer, x, w, b, batch, t_in));
    EXPECT_TRUE(error < 1.0e-5) << std::format("nn_conv_transpose1d with bias, ReLU input and GELU matches the CPU (worst {:.2e})", error);
}

TEST_F(Gpu, OverlapAddMatchesTheCpu)
{
    constexpr uint32_t length = 64;
    constexpr uint32_t frames = 20;
    constexpr uint32_t hop = 16;
    constexpr uint32_t trim = 32;
    constexpr uint32_t samples = hop * (frames - 1);
    std::vector<float> squared(length);
    for (uint32_t i = 0; i < length; ++i) {
        const double w = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * i / length);
        squared[i] = static_cast<float>(w * w);
    }
    const auto values = random_values(static_cast<size_t>(length) * frames, -1.0F, 1.0F, 41);

    Bench bench;
    const uint64_t pf = bench.put(0, values);
    const uint64_t pw = bench.put(1, squared);
    bench.run({ TK::overlap_add_stage(pf, pw, bench.space(2, samples), length, frames, hop, trim, samples),
        TK::overlap_add_stage(pf, 0, bench.space(3, samples), length, frames, hop, 0, samples) });
    const double normalised = worst(bench.get(2, samples), TK::cpu::overlap_add(values, squared, length, frames, hop, trim, samples));
    const double summed = worst(bench.get(3, samples), TK::cpu::overlap_add(values, {}, length, frames, hop, 0, samples));
    EXPECT_TRUE(normalised < 1.0e-5) << std::format("nn_overlap_add divided by the window envelope matches the CPU (worst {:.2e})", normalised);
    EXPECT_TRUE(summed < 1.0e-6) << std::format("nn_overlap_add as a plain sum matches the CPU (worst {:.2e})", summed);
}

double snr_db(std::span<const float> signal, std::span<const float> reference)
{
    double power = 0.0;
    double error = 0.0;
    for (size_t i = 0; i < std::min(signal.size(), reference.size()); ++i) {
        power += static_cast<double>(reference[i]) * reference[i];
        error += std::pow(static_cast<double>(signal[i]) - reference[i], 2);
    }
    return 10.0 * std::log10(power / std::max(error, 1e-30));
}

/** The Vocos weights and the speech fixture, or why a test cannot run. */
struct VocosInputs {
    std::optional<Kakshya::TensorSet> weights;
    std::vector<float> audio;
    std::string missing;
};

VocosInputs vocos_inputs()
{
    VocosInputs inputs;
    if (!fs::exists(k_vocos)) {
        inputs.missing = "Vocos checkpoint not fetched: " + k_vocos.string();
        return inputs;
    }
    inputs.audio = read_f32(k_root / "fixtures/speech_24k.f32");
    if (inputs.audio.empty()) {
        inputs.missing = "no speech fixture (run generate_reference.py --speech)";
        return inputs;
    }
    IO::TorchCheckpointReader reader;
    inputs.weights = reader.open(k_vocos.string()) ? reader.read_tensors() : std::nullopt;
    return inputs;
}

TEST_F(Gpu, VocosResynthesisKeepsTheSpectrogram)
{
    const auto inputs = vocos_inputs();
    if (!inputs.missing.empty()) {
        GTEST_SKIP() << inputs.missing;
    }
    ASSERT_TRUE(inputs.weights.has_value()) << "the Vocos checkpoint reads";

    const auto built = Clock::now();
    Yantra::VocosModel model(*inputs.weights);
    MF_PRINT("      model ready in {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - built).count());

    const auto& audio = inputs.audio;
    const uint32_t frames = Yantra::VocosModel::frame_count(audio.size());

    const auto mel = model.mel(audio);
    EXPECT_TRUE(mel.size() == static_cast<size_t>(Yantra::VocosModel::mels) * frames
            && std::ranges::all_of(mel, [](float v) { return std::isfinite(v) && v >= std::log(1.0e-7F) - 1e-4F && v < 10.0F; })) << std::format("mel gives {} finite log values for {} frames", mel.size(), frames);

    const auto started = Clock::now();
    const auto output = model.resynthesize(audio);
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    EXPECT_TRUE(output.size() == Yantra::VocosModel::sample_count(frames) && std::ranges::all_of(output, [](float v) { return std::isfinite(v); })) << std::format("resynthesize gives {} finite samples in {:.0f} ms for {:.2f} s of audio", output.size(), ms,
            static_cast<double>(audio.size()) / Yantra::VocosModel::sample_rate);

    const auto split = model.synthesize(mel, frames);
    EXPECT_TRUE(worst(split, output) < 1.0e-5) << "synthesize(mel(x)) equals resynthesize(x), recorded as one submission";

    const auto again = model.mel(output);
    double deviation = 0.0;
    size_t counted = 0;
    for (size_t i = 0; i < std::min(again.size(), mel.size()); ++i) {
        if (mel[i] > -9.0F) {
            deviation += std::abs(static_cast<double>(again[i]) - mel[i]);
            ++counted;
        }
    }
    deviation /= static_cast<double>(std::max<size_t>(counted, 1));
    EXPECT_TRUE(counted > 0 && deviation < 0.3) << std::format("the resynthesis has the input's spectrogram: mean log mel difference {:.3f} over {} cells", deviation, counted);

    for (const auto& [name, values] : { std::pair { "vocos_input.f32", &audio }, std::pair { "vocos_mel.f32", &mel },
             std::pair { "vocos_resynthesis.f32", &output } }) {
        std::ofstream(k_root / "build" / name, std::ios::binary)
            .write(reinterpret_cast<const char*>(values->data()), static_cast<std::streamsize>(values->size() * sizeof(float)));
    }
    const fs::path wav = k_root / "build/vocos_resynthesis.wav";
    write_wav(wav, output, Yantra::VocosModel::sample_rate);
    write_wav(k_root / "build/vocos_input.wav", audio, Yantra::VocosModel::sample_rate);
    MF_PRINT("      wrote {} and vocos_input.wav", wav.string());
}

TEST_F(Gpu, VocosMatchesPyTorch)
{
    const auto inputs = vocos_inputs();
    if (!inputs.missing.empty()) {
        GTEST_SKIP() << inputs.missing;
    }
    if (!fs::exists(k_root / "fixtures/vocos_output.f32")) {
        GTEST_SKIP() << "no PyTorch reference in fixtures (run generate_reference.py --vocos-src)";
    }
    ASSERT_TRUE(inputs.weights.has_value()) << "the Vocos checkpoint reads";

    Yantra::VocosModel model(*inputs.weights);
    const auto& audio = inputs.audio;
    const uint32_t frames = Yantra::VocosModel::frame_count(audio.size());
    const auto mel = model.mel(audio);
    const auto output = model.resynthesize(audio);

    const auto mel_reference = read_f32(k_root / "fixtures/vocos_mel.f32");
    const auto output_reference = read_f32(k_root / "fixtures/vocos_output.f32");

    double mel_error = 0.0;
    for (size_t i = 0; i < std::min(mel.size(), mel_reference.size()); ++i) {
        if (mel_reference[i] > -9.0F) {
            mel_error = std::max(mel_error, std::abs(static_cast<double>(mel[i]) - mel_reference[i]));
        }
    }
    EXPECT_TRUE(mel.size() == mel_reference.size() && mel_error < 1.0e-2) << std::format("mel matches Vocos's feature extractor (worst {:.2e} where the band is above e^-9)", mel_error);

    const auto from_reference = model.synthesize(mel_reference, frames);
    const double decoded = snr_db(from_reference, output_reference);
    EXPECT_TRUE(from_reference.size() == output_reference.size() && decoded > 50.0) << std::format("synthesize matches Vocos's decode to {:.1f} dB", decoded);
    const double round_trip = snr_db(output, output_reference);
    EXPECT_TRUE(round_trip > 40.0) << std::format("resynthesize matches Vocos's forward to {:.1f} dB", round_trip);
    MF_PRINT("      against the original Vocos: mel worst {:.2e}, decode {:.1f} dB, mel and decode {:.1f} dB", mel_error, decoded, round_trip);
}

}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new Device);
    return RUN_ALL_TESTS();
}
