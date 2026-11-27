# mf_voicegen

Speech synthesis on MayaFlux's tensor kernels. It reads PyTorch checkpoints without Python, adds the kernels vocoders and speech models need, and runs the Vocos vocoder on the GPU. Kokoro-82M is next: `docs/kokoro-operations.md` maps what it runs to kernels.

Needs the MayaFlux nightly from 24 Nov 2026 or later, the first with the public `TensorKernels` headers.

## Pieces

| File | Namespace | Does |
| ---- | --------- | ---- |
| `TorchFormat` | `IO` | `TorchCheckpointReader`: the tensors of a `torch.save` checkpoint (`.pt`, `.pth`, `.ckpt`, `.bin`). Reads the pickle without running it. |
| `VoiceKernels` | `Yantra::TensorKernels` | Six kernels with their stages and CPU versions: `nn_unary` (affine, exp, log, sin, cos, tanh, leaky ReLU, SiLU, Snake, clamp, tanh GELU, fract, step), `nn_binary` (multiply, add, subtract, divide, maximum, minimum, atan2), `nn_conv_transpose1d` (with groups), `nn_overlap_add`, `nn_gather` (embeddings, repeats, nearest upsampling, pads) and `nn_lstm_recurrence` (an LSTM of any input width, either direction, after a `conv1d` input projection). `dft_basis` and `inverse_dft_basis` turn convolutions into an STFT and back; `fold_weight_norm` turns `weight_g` and `weight_v` into weights. |
| `VocosModel` | `Yantra` | Vocos (`charactr/vocos-mel-24khz`): a log mel spectrogram in, 24 kHz audio out, and the mel spectrogram of audio. |

## Use

```cpp
#include "TorchFormat.hpp"
#include "VocosModel.hpp"

void compose()
{
    IO::TorchCheckpointReader::register_with_registry();
    auto weights = get_io_manager()->load_tensors("data/tensors/vocos_mel_24khz/pytorch_model.bin");
    static Yantra::VocosModel vocos(*weights);

    // later, off the audio thread, with samples at 24 kHz:
    auto mel = vocos.mel(samples);
    // ... change the spectrogram ...
    auto audio = vocos.synthesize(mel, Yantra::VocosModel::frame_count(samples.size()));
}
```

- **Weights:** `community.json` lists them, pinned by commit and SHA-256. Weave fetches them when it adds the module to a project. Until the module is in the registry, use MayaFlux's script: `python3 <MayaFlux>/scripts/fetch_assets.py --list community/mf_voicegen/community.json --data data`.
- **Spectrograms** are band after band: 100 rows, each holding every frame. A frame is 256 samples.
- **Timing:** 3 s of speech is resynthesised in about 0.7 s on an Iris Xe, including compiling the kernels on the first call. The call blocks its thread, so on the graphics thread it drops frames. Use a worker for long takes, and never the audio thread.
- **Kernels** are GLSL files in `shaders/`. ShaderFoundry compiles them on first use, and `mf_voicegen.cmake` points `MF_VOICEGEN_SHADER_DIR` at them. `nn_unary` computes sin and cos itself, because the built-in functions are only accurate within ±π and Vocos's phases reach ±150.
- **What Vocos does with odd input:** it rebuilds speech closely, with a mean log mel difference of 0.12 on the test excerpt. A synthetic pulse train comes back several times quieter. It was trained on speech, so the spectrogram should stay speech-like.

## Tests

```sh
python3 <MayaFlux>/scripts/fetch_assets.py --list community.json --data test/data
cmake --preset release
cmake --build --preset release
ctest --preset release
```

The tests use GoogleTest. CMake takes an installed GTest package, or downloads the pinned 1.18.0 release when there is none, so building them without a network needs GTest installed. `ctest` runs each test on its own. The GPU tests skip without a Vulkan device. `Gpu.VocosMatchesPyTorch` compares against the original Vocos code: the mel spectrogram agrees to 2.5e-3 in log magnitude, decoding the same mel to 102.8 dB SNR, and the whole round trip to 84.2 dB.

- `test_mf_voicegen`, 8 tests:
  - the reader on `torch.save` files: every element type, transposed and offset views, nested dictionaries, parameters, the old format, damaged files;
  - the format registry;
  - the Vocos checkpoint, and the Kokoro checkpoint when `kokoro-v1_0.pth` is in `test/data/tensors/kokoro_82m` (it is not in `community.json` yet);
  - the CPU kernels against PyTorch, including `inverse_dft_basis` with `overlap_add` against `torch.istft`;
  - the operations Kokoro adds, against PyTorch: tanh GELU, remainder, angle, grouped and depthwise transposed convolution, embedding, `repeat_interleave`, nearest upsampling, reflection padding, a bidirectional `nn.LSTM`, weight norm.
- `test_mf_voicegen_gpu`, 9 tests:
  - each kernel against its CPU version, the LSTM recurrence after MayaFlux's own `conv1d` as its input projection;
  - Vocos on a LibriSpeech excerpt, writing `test/build/vocos_resynthesis.wav` to listen to.
- `generate_reference.py` writes the fixtures; `test/README.md` says where they come from. `trace_kokoro.py` runs the original Kokoro and records what it executes.
