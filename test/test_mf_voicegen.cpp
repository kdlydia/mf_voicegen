#include "MayaFlux/IO/Tensor/Detail/TensorCodec.hpp"
#include "MayaFlux/IO/Tensor/SafetensorsFormat.hpp"
#include "MayaFlux/IO/Tensor/TensorFormat.hpp"
#include "MayaFlux/Journal/Archivist.hpp"
#include "MayaFlux/Kakshya/NDData/TensorConvert.hpp"

#include "TorchFormat.hpp"
#include "VocosModel.hpp"
#include "VoiceKernels.hpp"

#include <fstream>

#include <gtest/gtest.h>

// Runs with no device. The fixtures come from generate_reference.py: small
// torch.save files for the reader and PyTorch outputs for the CPU kernels. The
// Vocos checkpoint is checked when fetch_assets.py has put it in test/data.

namespace {

using namespace MayaFlux;
namespace fs = std::filesystem;
namespace TK = Yantra::TensorKernels;

const fs::path k_root = MF_VOICEGEN_TEST_DIR;
const fs::path k_vocos = k_root / "data/tensors/vocos_mel_24khz/pytorch_model.bin";

std::string fixture(std::string_view name)
{
    return (k_root / "fixtures" / name).string();
}

template <typename T>
std::vector<T> raw_values(const std::optional<Kakshya::TensorData>& tensor)
{
    if (!tensor) {
        return {};
    }
    const auto bytes = tensor->storage();
    std::vector<T> out(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), out.size() * sizeof(T));
    return out;
}

std::vector<float> as_floats(const std::optional<Kakshya::TensorData>& tensor)
{
    if (!tensor) {
        return {};
    }
    const auto view = tensor->access();
    if (!view) {
        return {};
    }
    auto converted = Kakshya::convert_tensor(*view, Kakshya::TensorDType::of<float>());
    return converted ? converted->to_vector<float>().value_or(std::vector<float> {}) : std::vector<float> {};
}

/** Largest |a - b| / (1 + |b|), or infinity when the sizes differ. */
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

std::vector<float> floats(const Kakshya::TensorSet& set, std::string_view name)
{
    const auto* tensor = set.find(name);
    return tensor ? tensor->to_vector<float>().value_or(std::vector<float> {}) : std::vector<float> {};
}

TEST(TorchCheckpointReader, StateDict)
{
    IO::TorchCheckpointReader reader;
    EXPECT_TRUE(reader.open(fixture("state_dict.pt"))) << "state_dict.pt opens";

    std::vector<std::string> names;
    for (const auto& spec : reader.describe()) {
        names.push_back(spec.name);
    }
    EXPECT_TRUE((names == std::vector<std::string> { "a", "t", "row", "half", "brain", "long", "mask", "scalar", "empty" })) << "describe lists the state_dict's tensors in its order";

    const auto a = reader.read("a");
    EXPECT_TRUE((a && a->spec().shape == std::vector<uint64_t> { 2, 3 }
            && raw_values<float>(a) == std::vector<float> { 0, 1, 2, 3, 4, 5 })) << "a float32 tensor reads with its shape and values";

    const auto t = reader.read("t");
    EXPECT_TRUE((t && t->spec().shape == std::vector<uint64_t> { 3, 2 } && t->spec().is_contiguous()
            && raw_values<float>(t) == std::vector<float> { 0, 3, 1, 4, 2, 5 })) << "a transposed view of the same storage is gathered into C order";

    const auto row = reader.read("row");
    EXPECT_TRUE((row && row->spec().shape == std::vector<uint64_t> { 3 } && raw_values<float>(row) == std::vector<float> { 3, 4, 5 })) << "a view at a storage offset reads from that offset";

    const auto half = reader.read("half");
    EXPECT_TRUE((half && half->spec().dtype == Kakshya::TensorDType::float16()
            && as_floats(half) == std::vector<float> { 0.5F, -1.0F, 2.0F })) << "float16 keeps its element type and converts to float";

    const auto brain = reader.read("brain");
    EXPECT_TRUE((brain && brain->spec().dtype == Kakshya::TensorDType::bfloat16()
            && as_floats(brain) == std::vector<float> { 1.0F, -2.0F, 0.25F, 3.0F })) << "bfloat16 keeps its element type and converts to float";

    const auto wide = reader.read("long");
    EXPECT_TRUE((wide && wide->spec().dtype == Kakshya::TensorDType::of<int64_t>()
            && raw_values<int64_t>(wide) == std::vector<int64_t> { 1, -2, 3, int64_t { 1 } << 40 })) << "int64 reads exactly, beyond 32 bits";

    const auto mask = reader.read("mask");
    EXPECT_TRUE((mask && mask->spec().dtype == Kakshya::TensorDType::of<bool>()
            && raw_values<uint8_t>(mask) == std::vector<uint8_t> { 1, 0, 1 })) << "bool reads one byte per element";

    const auto scalar = reader.read("scalar");
    EXPECT_TRUE(scalar && scalar->spec().rank() == 0 && raw_values<double>(scalar) == std::vector<double> { 2.5 }) << "a zero dimensional float64 reads as one value";

    const auto empty = reader.read("empty");
    EXPECT_TRUE((empty && empty->spec().shape == std::vector<uint64_t> { 0, 4 } && empty->byte_size() == 0)) << "an empty tensor reads with its shape and no bytes";

    EXPECT_TRUE(!reader.read("missing") && reader.get_last_error().contains("missing")) << "an unknown name fails with the name";
}

TEST(TorchCheckpointReader, NestedCheckpoint)
{
    IO::TorchCheckpointReader reader;
    EXPECT_TRUE(reader.open(fixture("nested.pt"))) << "nested.pt opens";

    std::vector<std::string> names;
    for (const auto& spec : reader.describe()) {
        names.push_back(spec.name);
    }
    EXPECT_TRUE((names == std::vector<std::string> { "model.layer.weight", "list.0", "list.1", "params.w" })) << "nested dictionaries and lists name tensors by their path";
    EXPECT_TRUE((raw_values<float>(reader.read("params.w")) == std::vector<float> { 4.0F, 5.0F })) << "an nn.Parameter reads as its tensor";

    const auto metadata = reader.tensor_metadata();
    EXPECT_TRUE(metadata.contains("epoch") && metadata.at("epoch") == "7" && metadata.at("name") == "x" && metadata.at("lr") == "0.5") << "number and string leaves become metadata";

    const auto set = reader.read_tensors();
    EXPECT_TRUE(set && set->tensors.size() == 4 && set->is_consistent()) << "read_tensors gives a consistent set";
}

TEST(TorchCheckpointReader, Refusals)
{
    IO::TorchCheckpointReader reader;
    EXPECT_TRUE(!reader.open(fixture("legacy.pt")) && reader.get_last_error().contains("1.6")) << "the format before PyTorch 1.6 is refused with the reason";
    EXPECT_TRUE(!reader.open(fixture("kernels.safetensors"))) << "a safetensors file is not taken for a checkpoint";

    const fs::path truncated = fs::temp_directory_path() / "mf_voicegen_truncated.pt";
    {
        std::ifstream in(fixture("state_dict.pt"), std::ios::binary);
        std::vector<char> head(600);
        in.read(head.data(), static_cast<std::streamsize>(head.size()));
        std::ofstream(truncated, std::ios::binary).write(head.data(), in.gcount());
    }
    EXPECT_TRUE(!reader.open(truncated.string()) && !reader.get_last_error().empty()) << "a truncated checkpoint is refused, not read";
    fs::remove(truncated);

    const auto head = IO::Detail::read_file_head(fixture("state_dict.pt"), 4096);
    EXPECT_TRUE(head && IO::TorchCheckpointReader::probe(head->bytes, head->file_size)) << "probe recognises a checkpoint";
    const auto other = IO::Detail::read_file_head(fixture("kernels.safetensors"), 4096);
    EXPECT_TRUE(other && !IO::TorchCheckpointReader::probe(other->bytes, other->file_size)) << "probe passes over a safetensors file";
}

TEST(TorchCheckpointReader, Registry)
{
    IO::SafetensorsReader::register_with_registry();
    IO::TorchCheckpointReader::register_with_registry();
    auto& registry = IO::TensorFormatRegistry::instance();

    const auto* format = registry.detect(fixture("state_dict.pt"));
    EXPECT_TRUE(format && format->name == "torch") << "the registry picks the torch format for a .pt file";
    const auto* safetensors = registry.detect(fixture("kernels.safetensors"));
    EXPECT_TRUE(safetensors && safetensors->name == "safetensors") << "the registry still picks safetensors for its files";
    EXPECT_TRUE(!registry.create_writer("out.pt")) << "the torch format is read only";
}

TEST(VocosModel, Checkpoint)
{
    if (!fs::exists(k_vocos)) {
        GTEST_SKIP() << "Vocos checkpoint not fetched: " << k_vocos.string();
    }
    const auto* format = IO::TensorFormatRegistry::instance().detect(k_vocos.string());
    EXPECT_TRUE(format && format->name == "torch") << "pytorch_model.bin is recognised by its content, not its extension";

    auto reader = IO::TensorFormatRegistry::instance().create_reader(k_vocos.string());
    EXPECT_TRUE(reader && reader->open(k_vocos.string())) << "the Vocos checkpoint opens";
    if (!reader || !reader->is_open()) {
        return;
    }
    EXPECT_TRUE(reader->describe().size() == 83) << "the Vocos checkpoint lists 83 tensors";

    auto weights = reader->read_tensors();
    EXPECT_TRUE(weights && Yantra::VocosModel::diagnose(*weights).empty()) << "VocosModel accepts the checkpoint's tensors";
    if (!weights) {
        return;
    }
    const auto fb = floats(*weights, "feature_extractor.mel_spec.mel_scale.fb");
    EXPECT_TRUE(fb.size() == 513 * 100 && std::ranges::all_of(fb, [](float v) { return v >= 0.0F; })) << "the mel filter bank reads as 513 by 100 non-negative weights";

    auto damaged = *weights;
    std::erase_if(damaged.tensors, [](const Kakshya::TensorData& t) { return t.spec().name == "backbone.convnext.3.gamma"; });
    EXPECT_TRUE(Yantra::VocosModel::diagnose(damaged).contains("backbone.convnext.3.gamma")) << "diagnose names a missing tensor";
}

TEST(VoiceKernels, CpuMatchesPyTorch)
{
    IO::SafetensorsReader reader;
    const bool opened = reader.open(fixture("kernels.safetensors"));
    EXPECT_TRUE(opened) << "kernels.safetensors opens";
    if (!opened) {
        return;
    }
    const auto set = reader.read_tensors();
    if (!set) {
        ADD_FAILURE() << "kernels.safetensors reads";
        return;
    }

    const auto x = floats(*set, "unary.x");
    const auto positive = floats(*set, "unary.positive");
    constexpr double tolerance = 2.0e-6;
    EXPECT_TRUE(worst(TK::cpu::unary(TK::UnaryOp::Exp, x, 20.0F), floats(*set, "unary.exp")) < tolerance) << "cpu Exp with a cap matches clamp(exp(x))";
    EXPECT_TRUE(worst(TK::cpu::unary(TK::UnaryOp::Log, positive, 1.0e-7F), floats(*set, "unary.log")) < tolerance) << "cpu Log with a floor matches log(clamp(x))";
    EXPECT_TRUE(worst(TK::cpu::unary(TK::UnaryOp::Sin, x), floats(*set, "unary.sin")) < tolerance) << "cpu Sin matches";
    EXPECT_TRUE(worst(TK::cpu::unary(TK::UnaryOp::Cos, x), floats(*set, "unary.cos")) < tolerance) << "cpu Cos matches";
    EXPECT_TRUE(worst(TK::cpu::unary(TK::UnaryOp::Tanh, x), floats(*set, "unary.tanh")) < tolerance) << "cpu Tanh matches";
    EXPECT_TRUE(worst(TK::cpu::unary(TK::UnaryOp::LeakyRelu, x, 0.1F), floats(*set, "unary.leaky_relu")) < tolerance) << "cpu LeakyRelu matches";
    EXPECT_TRUE(worst(TK::cpu::unary(TK::UnaryOp::Silu, x), floats(*set, "unary.silu")) < tolerance) << "cpu Silu matches";
    EXPECT_TRUE(worst(TK::cpu::snake(floats(*set, "snake.x"), floats(*set, "snake.a"), 3, 5), floats(*set, "snake.y")) < tolerance) << "cpu Snake with one a per channel matches";

    const TK::ConvTranspose1d layer { .c_in = 3, .c_out = 4, .kernel = 4, .stride = 3, .pad = 1, .output_pad = 1, .dilation = 2 };
    const auto* expected = set->find("conv_transpose.y");
    EXPECT_TRUE(expected && expected->spec().shape.back() == TK::conv_transpose1d_length(layer, 5)) << "conv_transpose1d_length matches PyTorch's output length";
    EXPECT_TRUE(worst(TK::cpu::conv_transpose1d(layer, floats(*set, "conv_transpose.x"), floats(*set, "conv_transpose.w"),
                    floats(*set, "conv_transpose.b"), 2, 5),
              floats(*set, "conv_transpose.y"))
            < 1.0e-5) << "cpu transposed convolution with stride, padding, output padding and dilation matches PyTorch";

    constexpr uint32_t n = 1024;
    constexpr uint32_t hop = 256;
    constexpr uint32_t frames = 12;
    constexpr uint32_t bins = n / 2 + 1;
    const auto window = floats(*set, "istft.window");
    const auto real = floats(*set, "istft.real");
    const auto imag = floats(*set, "istft.imag");
    const auto basis = TK::inverse_dft_basis(window);
    std::vector<float> spectrum(real);
    spectrum.insert(spectrum.end(), imag.begin(), imag.end());
    std::vector<float> frame_values(static_cast<size_t>(n) * frames, 0.0F);
    for (uint32_t t = 0; t < n; ++t) {
        for (uint32_t c = 0; c < 2 * bins; ++c) {
            const float w = basis[static_cast<size_t>(t) * 2 * bins + c];
            for (uint32_t f = 0; f < frames; ++f) {
                frame_values[static_cast<size_t>(t) * frames + f] += w * spectrum[static_cast<size_t>(c) * frames + f];
            }
        }
    }
    std::vector<float> squared(n);
    std::ranges::transform(window, squared.begin(), [](float w) { return w * w; });
    const auto signal = TK::cpu::overlap_add(frame_values, squared, n, frames, hop, n / 2, hop * (frames - 1));
    EXPECT_TRUE(worst(signal, floats(*set, "istft.y")) < 1.0e-5) << "inverse_dft_basis then overlap_add with the squared window equals torch.istft(center=True)";
}

/** Registers the safetensors and torch formats, once per test process. */
class Formats : public ::testing::Environment {
public:
    void SetUp() override
    {
        IO::SafetensorsReader::register_with_registry();
        IO::TorchCheckpointReader::register_with_registry();
    }
};

}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new Formats);
    return RUN_ALL_TESTS();
}
