# mf_voicegen

Speech synthesis on MayaFlux's tensor kernels. It reads PyTorch checkpoints without Python, adds the kernels vocoders need, and runs the Vocos vocoder on the GPU.

Needs the MayaFlux nightly from 24 Nov 2026 or later, the first with the public `TensorKernels` headers.

## Pieces

| File | Namespace | Does |
| ---- | --------- | ---- |
| `TorchFormat` | `IO` | `TorchCheckpointReader`: the tensors of a `torch.save` checkpoint (`.pt`, `.pth`, `.ckpt`, `.bin`). Reads the pickle without running it. |
| `VoiceKernels` | `Yantra::TensorKernels` | Four kernels with their stages and CPU versions: `nn_unary` (affine, exp, log, sin, cos, tanh, leaky ReLU, SiLU, Snake, clamp), `nn_multiply`, `nn_conv_transpose1d`, `nn_overlap_add`. `dft_basis` and `inverse_dft_basis` turn convolutions into an STFT and back. |
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
cmake -G Ninja -B test/build -S test -DCMAKE_BUILD_TYPE=Release
cmake --build test/build
python3 <MayaFlux>/scripts/fetch_assets.py --list community.json --data test/data
./test/build/test_mf_voicegen       # no device needed
./test/build/test_mf_voicegen_gpu   # needs a Vulkan device
```

- `test_mf_voicegen`, 43 checks:
  - the reader on `torch.save` files: every element type, transposed and offset views, nested dictionaries, parameters, the old format, damaged files;
  - the format registry;
  - the Vocos checkpoint;
  - the CPU kernels against PyTorch, including `inverse_dft_basis` with `overlap_add` against `torch.istft`.
- `test_mf_voicegen_gpu`, 23 checks:
  - each kernel against its CPU version;
  - Vocos on a LibriSpeech excerpt, writing `test/build/vocos_resynthesis.wav` to listen to.
- `generate_reference.py` writes the fixtures; `test/README.md` says where they come from.
