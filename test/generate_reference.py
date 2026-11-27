#!/usr/bin/env python3
"""Writes the test fixtures in test/fixtures from PyTorch and the original Vocos code.

Usage:
    python3 test/generate_reference.py --speech <hubert_speech_audio.f32> --vocos-src <clone of github.com/gemelo-ai/vocos>

The speech is the LibriSpeech excerpt MayaFlux's HuBERT test uses, from
mayaflux-tests/models/hubert_phoneme (CC BY 4.0, see test/README.md).

Needs torch, torchaudio, safetensors, numpy, and for Vocos the packages its code
imports (encodec, einops, scipy, pyyaml, huggingface_hub). The Vocos weights are
read from test/data/tensors/vocos_mel_24khz, where fetch_assets.py puts them.
Without --speech and --vocos-src only the reader and kernel fixtures are written.
"""

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / 'fixtures'
WEIGHTS = HERE / 'data' / 'tensors' / 'vocos_mel_24khz'


def reader_fixtures():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    state = collections.OrderedDict()
    state['a'] = a
    state['t'] = a.t()
    state['row'] = a[1]
    state['half'] = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float16)
    state['brain'] = torch.tensor([1.0, -2.0, 0.25, 3.0], dtype=torch.bfloat16)
    state['long'] = torch.tensor([[1, -2], [3, 1 << 40]], dtype=torch.int64)
    state['mask'] = torch.tensor([True, False, True])
    state['scalar'] = torch.tensor(2.5, dtype=torch.float64)
    state['empty'] = torch.zeros(0, 4)
    torch.save(state, FIXTURES / 'state_dict.pt')

    nested = {
        'model': collections.OrderedDict([('layer.weight', torch.full((2, 2), 7.0))]),
        'list': [torch.tensor([1.0]), torch.tensor([2.0, 3.0])],
        'params': {'w': torch.nn.Parameter(torch.tensor([4.0, 5.0]))},
        'epoch': 7,
        'name': 'x',
        'lr': 0.5,
    }
    torch.save(nested, FIXTURES / 'nested.pt')

    torch.save({'a': a}, FIXTURES / 'legacy.pt', _use_new_zipfile_serialization=False)


def kernel_fixtures():
    generator = torch.Generator().manual_seed(1234)
    out = {}

    x = torch.linspace(-4.0, 4.0, 81)
    positive = torch.linspace(0.0, 3.0, 31)
    out['unary.x'] = x
    out['unary.positive'] = positive
    out['unary.exp'] = torch.clamp(torch.exp(x), max=20.0)
    out['unary.log'] = torch.log(torch.clamp(positive, min=1e-7))
    out['unary.sin'] = torch.sin(x)
    out['unary.cos'] = torch.cos(x)
    out['unary.tanh'] = torch.tanh(x)
    out['unary.leaky_relu'] = F.leaky_relu(x, 0.1)
    out['unary.silu'] = F.silu(x)

    snake_x = torch.randn(2, 3, 5, generator=generator)
    snake_a = torch.rand(3, generator=generator) + 0.5
    a = snake_a.view(1, 3, 1)
    out['snake.x'] = snake_x
    out['snake.a'] = snake_a
    out['snake.y'] = snake_x + torch.sin(a * snake_x) ** 2 / a

    ct_x = torch.randn(2, 3, 5, generator=generator)
    ct_w = torch.randn(3, 4, 4, generator=generator)
    ct_b = torch.randn(4, generator=generator)
    out['conv_transpose.x'] = ct_x
    out['conv_transpose.w'] = ct_w
    out['conv_transpose.b'] = ct_b
    out['conv_transpose.y'] = F.conv_transpose1d(ct_x, ct_w, ct_b, stride=3, padding=1, output_padding=1, dilation=2)

    n_fft, hop, frames = 1024, 256, 12
    window = torch.hann_window(n_fft)
    spec = torch.randn(n_fft // 2 + 1, frames, dtype=torch.complex64, generator=generator)
    out['istft.window'] = window
    out['istft.real'] = spec.real.contiguous()
    out['istft.imag'] = spec.imag.contiguous()
    out['istft.y'] = torch.istft(spec.unsqueeze(0), n_fft, hop, n_fft, window, center=True).squeeze(0)

    kokoro_fixtures(out, x)

    save_file({key: value.contiguous() for key, value in out.items()}, FIXTURES / 'kernels.safetensors')
    (FIXTURES / 'kernels.safetensors').chmod(0o644)


def kokoro_fixtures(out, x):
    """The operations Kokoro adds to Vocos's: see docs/kokoro-operations.md."""
    generator = torch.Generator().manual_seed(82)

    out['unary.gelu_tanh'] = F.gelu(x, approximate='tanh')
    out['unary.fract_x'] = x * 1.3
    out['unary.fract'] = torch.remainder(out['unary.fract_x'], 1.0)
    out['unary.step'] = (x > 0.5).float()

    a = torch.randn(4, 30, generator=generator)
    b = torch.randn(4, 30, generator=generator)
    b = b + 0.5 * torch.sign(b)
    channel = torch.randn(4, generator=generator)
    out['binary.a'] = a
    out['binary.b'] = b
    out['binary.channel'] = channel
    out['binary.multiply'] = a * b
    out['binary.add'] = a + b
    out['binary.subtract'] = a - b
    out['binary.divide'] = a / b
    out['binary.maximum'] = torch.maximum(a, b)
    out['binary.minimum'] = torch.minimum(a, b)
    out['binary.angle'] = torch.angle(torch.complex(b, a))
    out['binary.add_channel'] = a + channel.view(4, 1)

    # Kokoro's AdainResBlk1d pool: a depthwise ConvTranspose1d, k3 s2 p1 op1.
    dw_x = torch.randn(1, 6, 7, generator=generator)
    dw_w = torch.randn(6, 1, 3, generator=generator)
    dw_b = torch.randn(6, generator=generator)
    out['conv_transpose_depthwise.x'] = dw_x
    out['conv_transpose_depthwise.w'] = dw_w
    out['conv_transpose_depthwise.b'] = dw_b
    out['conv_transpose_depthwise.y'] = F.conv_transpose1d(dw_x, dw_w, dw_b, stride=2, padding=1, output_padding=1, groups=6)
    g_x = torch.randn(2, 4, 5, generator=generator)
    g_w = torch.randn(4, 3, 4, generator=generator)
    out['conv_transpose_grouped.x'] = g_x
    out['conv_transpose_grouped.w'] = g_w
    out['conv_transpose_grouped.y'] = F.conv_transpose1d(g_x, g_w, None, stride=2, padding=1, groups=2)

    # Indices are stored as floats; the tests turn them into int32.
    table = torch.randn(10, 4, generator=generator)
    ids = torch.tensor([3, 0, 9, 9, 1, 5])
    columns = torch.randn(3, 5, generator=generator)
    durations = torch.tensor([2, 0, 3, 1, 2])
    out['gather.table'] = table
    out['gather.ids'] = ids.float()
    out['gather.embedding'] = F.embedding(ids, table).t()
    out['gather.columns'] = columns
    out['gather.durations'] = durations.float()
    out['gather.repeat'] = torch.repeat_interleave(columns, durations, dim=1)
    out['gather.nearest'] = F.interpolate(columns[None], scale_factor=2, mode='nearest')[0]
    out['gather.reflect'] = F.pad(columns[None], (1, 0), mode='reflect')[0]

    lstm = torch.nn.LSTM(12, 6, batch_first=True, bidirectional=True)
    with torch.no_grad():
        for parameter in lstm.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.4)
        sequence = torch.randn(1, 9, 12, generator=generator)
        hs, _ = lstm(sequence)
    out['lstm.x'] = sequence[0].t()
    for name, parameter in lstm.named_parameters():
        out['lstm.' + name] = parameter.detach()
    out['lstm.y'] = hs[0].t()

    g = torch.rand(5, 1, 1, generator=generator) + 0.5
    v = torch.randn(5, 3, 4, generator=generator)
    out['weight_norm.g'] = g
    out['weight_norm.v'] = v
    out['weight_norm.w'] = torch._weight_norm(v, g, 0)


def speech_fixture(path):
    """MayaFlux's LibriSpeech excerpt (16 kHz float32) resampled to 24 kHz for Vocos."""
    import torchaudio

    speech = torch.from_numpy(np.fromfile(path, dtype=np.float32)).unsqueeze(0)
    torchaudio.functional.resample(speech, 16000, 24000).squeeze(0).numpy().astype(np.float32).tofile(FIXTURES / 'speech_24k.f32')


def vocos_fixtures(source):
    sys.path.insert(0, str(Path(source).resolve()))
    from vocos.pretrained import Vocos

    model = Vocos.from_hparams(str(WEIGHTS / 'config.yaml'))
    model.load_state_dict(torch.load(WEIGHTS / 'pytorch_model.bin', map_location='cpu', weights_only=True))
    model.eval()

    audio = torch.from_numpy(np.fromfile(FIXTURES / 'speech_24k.f32', dtype=np.float32)).unsqueeze(0)
    with torch.inference_mode():
        mel = model.feature_extractor(audio)
        output = model.decode(mel)

    mel.squeeze(0).contiguous().numpy().tofile(FIXTURES / 'vocos_mel.f32')
    output.squeeze(0).contiguous().numpy().tofile(FIXTURES / 'vocos_output.f32')
    print(f'vocos: {audio.shape[1]} samples, mel {tuple(mel.shape[1:])}, output {output.shape[1]} samples')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--speech', help='hubert_speech_audio.f32 from mayaflux-tests/models/hubert_phoneme')
    parser.add_argument('--vocos-src', help='clone of github.com/gemelo-ai/vocos')
    options = parser.parse_args()

    FIXTURES.mkdir(exist_ok=True)
    reader_fixtures()
    kernel_fixtures()
    if options.speech:
        speech_fixture(options.speech)
    if options.vocos_src:
        vocos_fixtures(options.vocos_src)


if __name__ == '__main__':
    main()
