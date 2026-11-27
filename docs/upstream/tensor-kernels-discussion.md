# Six tensor kernels for speech models

Draft for a MayaFlux discussion. Not posted.

With the public `TensorKernels` headers, a model can now be written outside MayaFlux the way `HubertPhonemeModel` is. [mf_voicegen](https://github.com/kdlydia/mf_voicegen) runs the Vocos vocoder that way, and is mapping Kokoro-82M (StyleTTS 2 with an iSTFTNet decoder) onto the same kernels. Between them they need six kernels that MayaFlux doesn't have. I'd like to know whether you want them upstream, and to agree their push constant layouts before modules depend on them.

| Kernel | Push constants | Does |
| ------ | -------------- | ---- |
| `nn_unary` | 48 bytes | One function per dispatch, chosen by an op code: affine, exp with a cap, log with a floor, sin, cos, tanh, leaky ReLU, SiLU, Snake (one `a` per channel or one for all), clamp, tanh GELU (`gelu_new`), fract, step. |
| `nn_binary` | 40 bytes | `op(a[i], b[(i / inner) % b_count])`, element-wise or one `b` per channel: multiply, add, subtract, divide, maximum, minimum, atan2. |
| `nn_conv_transpose1d` | 80 bytes: `nn_conv1d`'s, then groups | PyTorch's `conv_transpose1d` with stride, padding, output padding, dilation and groups. Each invocation gathers its own output, with no atomics. |
| `nn_overlap_add` | 48 bytes | Overlap-add of `[frame_length, frames]`, optionally divided by the summed squared window. After a pointwise convolution with an inverse DFT basis, it is `torch.istft`. |
| `nn_gather` | 40 bytes | `dst[c, t] = src[c * channel_stride + index[t] * element_stride]`, with int32 indices; a negative index writes zero. Embedding lookups, repeating tokens by duration, nearest upsampling and reflection padding are all this. |
| `nn_lstm_recurrence` | 48 bytes | An LSTM over a whole sequence in one workgroup, from input gates that `conv1d` has already computed, so the input can be any width. A flag runs it backwards, and a row offset lets both directions write one `[2 * hidden, steps]` output. |

Each has a stage function in the style of `Stages.hpp` and a CPU version. They are tested against PyTorch and against each other on an Iris Xe. Vocos built from them matches the original Vocos code on speech: decoding the same mel spectrogram agrees to 102.8 dB SNR, and the whole round trip, with the mel computed on the GPU, to 84.2 dB. For Kokoro, every operation the original code runs now has a kernel or a short CPU step; [docs/kokoro-operations.md](https://github.com/kdlydia/mf_voicegen/blob/main/docs/kokoro-operations.md) has the mapping.

Things that came up on the way:

- The built-in `sin` and `cos` on the Iris Xe were off by 1.5e-5 even within ±4, and Vulkan only bounds them within ±π. Vocos's phases reach ±150. `nn_unary` reduces the range itself, Cephes style, and stays within 4e-8 out to ±1000. `nn_binary` does the same for atan2.
- Kokoro's harmonic source reaches phases of 7e4 rad, past what that reduction covers, so its phase is reduced modulo 2π on the CPU first.
- `TensorFormatRegistry` was enough to add a reader for PyTorch checkpoints from outside MayaFlux. It reads the pickle without running it: tensors, containers and numbers only. It reads Kokoro's 548 tensors as well as Vocos's.
- Concatenation along channels needs no kernel: with channel-major buffers, each part is written at its row offset.

Questions:

1. One `nn_unary` and one `nn_binary` with op codes, or one kernel per function?
2. `lstm_sequence` takes input width equal to hidden, in one direction. Would you rather have `nn_lstm_recurrence` beside it, or `lstm_sequence` gain the precomputed-gates form, the reverse flag and the row offset?
3. Should `dft_basis`, `inverse_dft_basis`, `fold_weight_norm` and the CPU versions sit in `TensorKernels`, or in Kinesis?
4. Would you take the checkpoint reader into `IO/Tensor`? Upstream it would use miniz instead of its own reading of stored zip entries.
