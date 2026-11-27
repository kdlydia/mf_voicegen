# Kokoro's operations

What Kokoro-82M runs, and which kernel does each part in mf_voicegen. Kokoro is StyleTTS 2 with an iSTFTNet decoder. It was picked because the things to modulate, F0, energy, durations and the style vector, are plain tensors between its stages.

## How this was checked

`test/trace_kokoro.py` ran the original code (github.com/hexgrad/kokoro at `dfb907a`, weights `hexgrad/Kokoro-82M` at `f3ff357`) once, recording every aten call with its shapes and the tensors between stages. Input: 48 phonemes, "həlˈO, ðɪs ɪz kˈOkəɹO, ə tˈɛkst tə spˈiʧ mˈɑdᵊl.", with the `af_heart` voice. That gives 50 tokens with the two boundary tokens, 139 frames, and 83 400 samples, 3.48 s at 24 kHz. The run makes about 8 160 aten calls of 88 kinds.

All 548 tensors of `kokoro-v1_0.pth` are used. Their names carry a `module.` prefix. The checkpoint has no affine weights for the InstanceNorms inside AdaIN: they were added to the model only for ONNX export and stay at 1 and 0, so AdaIN is a plain instance norm followed by `(1 + γ)` and `β`.

## Stages

A voice is a `[510, 1, 256]` tensor, with one row for each phoneme count. `ref_s = voice[len(phonemes) - 1]`; the first 128 values style the decoder, the last 128 the prosody.

| Stage | What it is | Out (trace) |
| ----- | ---------- | ----------- |
| ALBERT | Embedding 128, a Linear to 768, 12 layers sharing one set of weights: 12-head attention, a 2048-wide feed-forward with `gelu_new`, residuals and LayerNorms | `[50, 768]` |
| `bert_encoder` | Linear 768 → 512 | `[512, 50]` |
| Duration encoder | 3 × (biLSTM 640 → 2 × 256, then AdaLayerNorm by the prosody style). The style is appended as 128 more channels before every LSTM | `[640, 50]` |
| Durations | biLSTM, Linear 512 → 50, sigmoid, sum of the 50, divided by speed, rounded, at least 1 | 50 counts, 139 in all |
| Alignment | Each token's column repeated by its count, as a matrix product with a 0/1 alignment | `en [640, 139]` |
| F0 and energy | A shared biLSTM, then per branch 3 AdainResBlk1d (the second doubles the length), then a 1×1 conv to one channel | `F0 [278]` Hz, `N [278]` |
| Text encoder | Embedding 512, 3 × (weight-normed conv k5, LayerNorm, leaky ReLU 0.2), biLSTM, then the same alignment | `asr [512, 139]` |
| Decoder | F0 and N each through a stride-2 conv to 139 frames, joined with `asr`; AdainResBlk1d blocks, the last doubling the length | `[512, 278]` |
| Harmonic source | F0 upsampled ×300 nearest; 9 harmonics; phase; sin × 0.1; voiced where F0 > 10; noise; Linear 9 → 1 and tanh | `[83 400]` |
| Source STFT | `torch.stft`, n_fft 20, hop 5, Hann, centred with reflection; magnitude and angle | `[22, 16 681]` |
| Generator | 2 × (leaky ReLU 0.1, ConvTranspose 512→256 k20 s10 p5 then 256→128 k12 s6 p3, plus the source through a conv and an AdaINResBlock1, then 3 Snake resblocks averaged), with a reflection pad (1, 0) in the second; leaky ReLU 0.01; conv k7 to 22 | `[22, 16 681]` |
| iSTFT | exp of the first 11 rows, sin of the last 11 as phase, `torch.istft` n_fft 20 hop 5 | 83 400 samples |

AdainResBlk1d: AdaIN, leaky ReLU 0.2, an optional depthwise ConvTranspose (k3, s2, p1, output padding 1, groups = channels), conv k3, AdaIN, leaky ReLU, conv k3. The shortcut is nearest ×2 and a 1×1 conv. The sum is divided by √2. AdaINResBlock1 is HiFi-GAN's resblock with AdaIN and Snake.

## Operations to kernels

"Upstream" is MayaFlux's `TensorKernels`; the rest are mf_voicegen's.

| Operation | Kernel | Notes |
| --------- | ------ | ----- |
| Linear, Conv1d, LSTM input projection | upstream `conv1d` | A Linear is a k1 conv over channel-major `[C, T]`. |
| Weight norm (89 pairs) | `fold_weight_norm`, CPU, at load | `g · v / ‖v‖` per output row. |
| LayerNorm over channels, InstanceNorm | upstream `normalize` | Per-channel affine from `NormAffine`. |
| AdaIN, AdaLayerNorm | upstream `normalize` | `(1 + γ)` and `β` come from a Linear of the style, once per utterance, and become the per-channel weight and bias. |
| Attention, softmax | upstream attention kernels | As in HuBERT. |
| `gelu_new` | `nn_unary` GeluTanh | New op. |
| Leaky ReLU, tanh, sigmoid, exp, sin, Snake, ÷√2 | `nn_unary`, or a conv's activation | |
| Residual sums, joining 3 resblocks | upstream `add`, `nn_binary` | |
| LSTM recurrence | `nn_lstm_recurrence` | New. Upstream `lstm_sequence` needs input width = hidden ≤ 256 and one direction. Kokoro's LSTMs are 512 or 640 wide and bidirectional, so the input projection is a `conv1d` and this kernel runs the recurrence, `reverse` for the second direction, `row_offset` to write both into one `[512, T]`. |
| Style channels appended before an LSTM | folded into the bias | The style is the same at every step, so `W_s · s` joins `b_ih + b_hh` once per utterance and the conv1d reads only the 512 real channels. |
| Concatenation along channels | none | Channel-major buffers: each part is written at its row offset of one buffer. |
| Embedding | `nn_gather` | Table `[rows, C]`, `channel_stride 1`, `element_stride C`. |
| Alignment (repeat by duration) | `nn_gather` | Index list from the durations. |
| Nearest ×2 | `nn_gather` | Index `t / 2`. |
| Reflection pads: (1, 0), and the STFT's centring | `nn_gather` | Index list. |
| Depthwise ConvTranspose (pool) | `nn_conv_transpose1d`, groups | New field; the push constants grew to 80 bytes. |
| Generator upsampling | `nn_conv_transpose1d` | |
| `torch.stft` | `dft_basis` with a strided `conv1d`, upstream `magnitude`, `nn_binary` Atan2 | Angle is new. |
| `torch.istft` | `inverse_dft_basis`, `nn_overlap_add` | As in Vocos. |
| Voiced flag, F0 > 10 | `nn_unary` Step | New op. |
| `x % 1` of the harmonic increments | `nn_unary` Fract | New op. |

## What stays on the CPU

- **Weight norm, and the AdaIN and AdaLN Linears of the style:** once per load or per utterance.
- **Durations:** the sum, round and clamp of 50 values. Their total sets the length of everything after, so the GPU work splits in two submissions: text to durations, then durations to audio.
- **The harmonic phase:** a cumulative sum over 278 frames, then linear interpolation to 83 400 samples. Two things found in the trace:
  - The random initial phase (`rand_ini`) never reaches the output. The ×300 linear downsample reads sample 149 of each frame and the random value is added at sample 0 only. Changing the seed leaves the sines identical, so the harmonic source is deterministic.
  - The phase reaches 4.6e4 rad for the fundamental and 6.9e4 for the 9th harmonic. F0 is slightly negative in unvoiced frames, and `% 1` turns that into almost a whole cycle per sample. Those frames are muted, but the cumulative phase carries on into voiced ones. `nn_unary`'s sin is exact to about 8192, so the CPU reduces the phase modulo 2π, in double, from the float32 values PyTorch would have, before the sin.
- **Noise:** `randn` for the unvoiced source and the jitter on voiced parts. To compare with the original, the same noise has to be given to both.

## Control tensors

These are where the voice can be changed without retraining:

| Tensor | Shape | Changes |
| ------ | ----- | ------- |
| `ref_s[:128]` | 128 | Timbre, through every AdaIN in the decoder. Interpolating or extrapolating between voices is the obvious move. |
| `ref_s[128:]` | 128 | Prosody: durations, F0 and energy curves. |
| Durations | per token | Timing, before alignment. `speed` divides them all. |
| F0 | 278 frames, Hz | Pitch curve. Also sets the harmonic source directly. |
| N | 278 frames | Energy curve. |
| Harmonic weights | 9 → 1 Linear | The mix of harmonics in the source. Away from the trained mix it sounds less human. |
| Noise amplitude | voiced 0.003, unvoiced 0.033 | Breathiness and the unvoiced texture. |
| Voiced threshold | 10 Hz | Where the source switches from harmonics to noise. |

## Next

`KokoroModel`, in the shape of `VocosModel`: load the 548 tensors, fold the weight norms, then the two submissions. Check it stage by stage against tensors written by the original code, the way `Gpu.VocosMatchesPyTorch` checks Vocos.
