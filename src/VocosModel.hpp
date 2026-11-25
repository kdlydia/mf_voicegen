#pragma once

#include "MayaFlux/Kakshya/NDData/Tensor.hpp"

/**
 * @file VocosModel.hpp
 * @brief The Vocos vocoder on the GPU: a log mel spectrogram at 24 kHz in, audio out.
 */

namespace MayaFlux::Yantra {

class GpuDispatchRecorder;

/**
 * @class VocosModel
 * @brief Vocos for 100 band mel spectrograms at 24 kHz, and the mel spectrograms it expects.
 *
 * Built from the 83 tensors of charactr/vocos-mel-24khz. A mel spectrogram
 * has one frame every 256 samples, each the log magnitude of a 1024 point
 * Hann windowed STFT, centred, folded into 100 mel bands. synthesize() turns
 * frames back into audio: a convolution to 512 channels, eight ConvNeXt
 * blocks, and a layer that predicts each bin's log magnitude and phase,
 * followed by an inverse STFT. mel() computes the spectrogram of audio the way
 * Vocos's feature extractor does, so resynthesize() rebuilds a recording
 * through the network, and anything done to the spectrogram in between is
 * heard as the network renders it.
 *
 * Spectrograms are mels rows of frames values: band after band, each row
 * holding every frame. Every frame depends only on the frames within a few
 * hops, so there is no limit on length beyond device memory, about 2.7 MB per
 * second. The network runs in native float and agrees with PyTorch to float
 * precision. It must not run on the audio thread.
 */
class VocosModel {
public:
    /** Sample rate in Hz the network was trained and runs at. */
    static constexpr uint32_t sample_rate = 24000;
    /** Samples between successive frames. */
    static constexpr uint32_t hop = 256;
    /** Length of the transform and of its Hann window. */
    static constexpr uint32_t fft_size = 1024;
    /** Frequency bins of the transform. */
    static constexpr uint32_t bins = fft_size / 2 + 1;
    /** Mel bands per frame. */
    static constexpr uint32_t mels = 100;
    /** Channels inside the network. */
    static constexpr uint32_t width = 512;

    /**
     * @brief Check that a tensor set holds the weights the network needs.
     * @return Empty when usable, otherwise the first missing tensor or shape mismatch.
     */
    [[nodiscard]] static std::string diagnose(const Kakshya::TensorSet& weights);

    /** @brief Frames the spectrogram of this many samples has: 1 + samples / hop, or 0 at fft_size / 2 samples or fewer. */
    [[nodiscard]] static uint32_t frame_count(size_t samples) noexcept;

    /** @brief Samples synthesize() returns for this many frames: hop * (frames - 1), 0 below two frames. */
    [[nodiscard]] static size_t sample_count(uint32_t frames) noexcept;

    /**
     * @brief Upload the weights and prepare the GPU resources.
     * @throws std::invalid_argument When diagnose rejects the set, before any device is touched.
     * @throws std::runtime_error When no device is available.
     */
    explicit VocosModel(const Kakshya::TensorSet& weights);

    /**
     * @brief As above, appending to a recording the caller shares with other work.
     * @param recording Recording the model records into, compute isolated. A null recording gives the model its own.
     */
    VocosModel(const Kakshya::TensorSet& weights, std::shared_ptr<GpuDispatchRecorder> recording);
    ~VocosModel();

    VocosModel(const VocosModel&) = delete;
    VocosModel& operator=(const VocosModel&) = delete;
    VocosModel(VocosModel&&) = delete;
    VocosModel& operator=(VocosModel&&) = delete;

    /**
     * @brief The log mel spectrogram of mono samples at 24 kHz.
     * @return mels * frame_count(samples.size()) values, band after band. Empty when there are too few samples.
     */
    [[nodiscard]] std::vector<float> mel(std::span<const float> samples);

    /**
     * @brief Audio from a log mel spectrogram.
     * @param mel mels * frames values, band after band.
     * @param frames Frames in mel, at least 2.
     * @return sample_count(frames) samples at 24 kHz.
     * @throws std::invalid_argument When mel does not hold mels * frames values or frames is below 2.
     */
    [[nodiscard]] std::vector<float> synthesize(std::span<const float> mel, uint32_t frames);

    /**
     * @brief mel() then synthesize(), in one submission.
     * @return sample_count(frame_count(samples.size())) samples, empty when there are too few samples.
     */
    [[nodiscard]] std::vector<float> resynthesize(std::span<const float> samples);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace MayaFlux::Yantra
