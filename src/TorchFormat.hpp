#pragma once

#include "MayaFlux/IO/Tensor/TensorReader.hpp"

#include <fstream>

/**
 * @file TorchFormat.hpp
 * @brief Reader for PyTorch checkpoints as torch.save writes them, without Python and without running pickle code.
 */

namespace MayaFlux::IO {

/**
 * @class TorchCheckpointReader
 * @brief Reads the tensors of a PyTorch checkpoint (.pt, .pth, .bin, .ckpt).
 *
 * torch.save writes a zip archive: a pickle, data.pkl, that describes the
 * saved object, and one stored entry per tensor storage. open() reads the zip
 * directory and interprets the pickle with a small machine that only builds
 * values: numbers, strings, lists, tuples, dictionaries and tensor
 * descriptions. It never imports or calls anything the pickle names, so a
 * checkpoint cannot run code. Objects it does not know become opaque and are
 * skipped, with the tensors under them still found.
 *
 * Every tensor reachable from the saved object is listed, named by its path of
 * dictionary keys and list indices joined with dots: a state_dict gives its own
 * keys, a checkpoint holding {"model": state_dict} gives "model.<key>". Number
 * and string leaves become tensor_metadata() under the same names. read() loads
 * one tensor at its stored element type, gathered into dense C order when it
 * was saved as a view, so tensors that share a storage are read correctly.
 *
 * The legacy format written before PyTorch 1.6 (a bare pickle stream) and
 * archives with compressed entries are refused with the reason.
 *
 * Usage:
 * @code
 * IO::TorchCheckpointReader::register_with_registry();
 * auto weights = get_io_manager()->load_tensors("data/tensors/vocos_mel_24khz/pytorch_model.bin");
 * @endcode
 */
class TorchCheckpointReader : public TensorReader {
public:
    TorchCheckpointReader() = default;
    ~TorchCheckpointReader() override { close(); }

    /** @brief Register the format in the TensorFormatRegistry, read only, for .pt, .pth, .ckpt and recognised .bin files. */
    static void register_with_registry();

    /** @brief Whether the leading bytes are a zip archive whose first entry is a checkpoint's data.pkl. */
    [[nodiscard]] static bool probe(std::span<const std::byte> head, uint64_t file_size);

    [[nodiscard]] bool can_read(const std::string& filepath) const override;
    bool open(const std::string& filepath, FileReadOptions options = FileReadOptions::ALL) override;
    void close() override;
    [[nodiscard]] bool is_open() const override { return m_open; }
    [[nodiscard]] std::vector<std::string> get_supported_extensions() const override { return { "pt", "pth", "ckpt", "bin" }; }

    [[nodiscard]] std::string format_name() const override { return "torch"; }
    [[nodiscard]] std::vector<Kakshya::TensorSpec> describe() const override;
    [[nodiscard]] std::map<std::string, std::string> tensor_metadata() const override { return m_metadata; }
    [[nodiscard]] std::optional<Kakshya::TensorData> read(std::string_view name) override;

private:
    struct Entry {
        Kakshya::TensorSpec spec;
        std::vector<int64_t> strides;
        uint64_t storage_offset {};
        uint64_t data_offset {};
        uint64_t data_size {};
    };

    bool reject(std::string reason);

    std::ifstream m_file;
    std::vector<Entry> m_entries;
    std::map<std::string, std::string> m_metadata;
    bool m_big_endian {};
    bool m_open {};
};

}
