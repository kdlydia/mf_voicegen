# Four tensor kernels for vocoders

Draft for a MayaFlux discussion. Not posted.

With the public `TensorKernels` headers, a model can now be written outside MayaFlux the way `HubertPhonemeModel` is. [mf_voicegen](https://github.com/kdlydia/mf_voicegen) runs the Vocos vocoder that way, and it needed four kernels that MayaFlux doesn't have. I'd like to know whether you want them upstream, and to agree their push constant layouts before modules depend on them.

| Kernel | Push constants | Does |
| ------ | -------------- | ---- |
| `nn_unary` | 48 bytes | One function per dispatch, chosen by an op code: affine, exp with a cap, log with a floor, sin, cos, tanh, leaky ReLU, SiLU, Snake (one `a` per channel or one for all), clamp. |
| `nn_multiply` | 40 bytes | `a[i] * b[(i / inner) % b_count]`: element-wise, or one scale per channel. |
| `nn_conv_transpose1d` | 72 bytes, as `nn_conv1d` | PyTorch's `conv_transpose1d` with stride, padding, output padding and dilation. Each invocation gathers its own output, with no atomics. |
| `nn_overlap_add` | 48 bytes | Overlap-add of `[frame_length, frames]`, optionally divided by the summed squared window. After a pointwise convolution with an inverse DFT basis, it is `torch.istft`. |

Each has a stage function in the style of `Stages.hpp` and a CPU version. They are tested against PyTorch and against each other on an Iris Xe. Vocos built from them matches Vocos's forward pass, written with PyTorch's own functions, at 103 dB SNR on speech. The comparison with the original Vocos code is still to run.

Two things came up on the way:

- The built-in `sin` and `cos` on the Iris Xe were off by 1.5e-5 even within ±4, and Vulkan only bounds them within ±π. Vocos's phases reach ±150. `nn_unary` reduces the range itself, Cephes style, and stays within 4e-8 out to ±1000.
- `TensorFormatRegistry` was enough to add a reader for PyTorch checkpoints from outside MayaFlux. It reads the pickle without running it: tensors, containers and numbers only.

Questions:

1. One `nn_unary` with an op code, or one kernel per function?
2. Should `dft_basis`, `inverse_dft_basis` and the CPU versions sit in `TensorKernels`, or in Kinesis?
3. Would you take the checkpoint reader into `IO/Tensor`? Upstream it would use miniz instead of its own reading of stored zip entries.
4. `scripts/fetch_assets.py --list community.json` fetches a module's assets as they are. Is that a supported way to use it before a module is in the registry?
