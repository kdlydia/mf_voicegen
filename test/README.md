# Fixtures

Written by `generate_reference.py`. The `.f32` files are raw little endian float32.

| File | Content |
| ---- | ------- |
| `state_dict.pt` | A state_dict from `torch.save`: float32, float16, bfloat16, int64, bool and float64 tensors, a transposed view and a row view of the same storage, a zero dimensional and an empty tensor. |
| `nested.pt` | Nested dictionaries and a list of tensors, an `nn.Parameter`, and number and string leaves. |
| `legacy.pt` | The same kind of file in the format written before PyTorch 1.6, which the reader refuses. |
| `kernels.safetensors` | Inputs and PyTorch outputs for the unary functions, Snake, `conv_transpose1d` and `torch.istft`, and for the operations Kokoro adds: tanh GELU, remainder, the binary operations and `torch.angle`, grouped `conv_transpose1d`, `F.embedding`, `repeat_interleave`, nearest upsampling, reflection padding, a bidirectional `nn.LSTM` with its weights, and `torch._weight_norm`. Indices are stored as floats. |
| `speech_24k.f32` | 3 s of speech at 24 kHz. |
| `vocos_mel.f32`, `vocos_output.f32` | The mel spectrogram (100 bands, band after band) and the decoded audio from the original Vocos code (github.com/gemelo-ai/vocos at `eb39abf`) for `speech_24k.f32`. |

`speech_24k.f32` is `hubert_speech_audio.f32` from MayaFlux's own tests (`mayaflux-tests/models/hubert_phoneme`), resampled from 16 to 24 kHz with torchaudio. That file is the first 3 s of `https://cdn-media.huggingface.co/speech_samples/sample1.flac`, "LibriSpeech sample 1". LibriSpeech (Panayotov, Chen, Povey and Khudanpur, 2015, https://www.openslr.org/12) is released under CC BY 4.0.
