"""Runs the original Kokoro once and records what it executes, for docs/kokoro-operations.md.

Usage: python3 test/trace_kokoro.py <clone of github.com/hexgrad/kokoro> <weights dir> <output dir>

The weights dir holds kokoro-v1_0.pth, config.json and voices/af_heart.pt from
hexgrad/Kokoro-82M. Writes trace.json (aten operations with shapes, leaf
modules, prosody tensors) and kokoro_original.wav, then checks two things about
the harmonic source: that its random initial phase never reaches the output,
and how large its phase gets. The phonemes are given directly, so misaki is not
needed; the kokoro package is loaded without its __init__ for the same reason.
"""
import collections
import json
import sys
import types
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

source, weights, out = (Path(a) for a in sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)

package = types.ModuleType('kokoro')
package.__path__ = [str(source / 'kokoro')]
sys.modules['kokoro'] = package
from kokoro.model import KModel  # noqa: E402

torch.manual_seed(0)
model = KModel(repo_id='hexgrad/Kokoro-82M', config=str(weights / 'config.json'),
               model=str(weights / 'kokoro-v1_0.pth')).eval()

phonemes = 'həlˈO, ðɪs ɪz kˈOkəɹO, ə tˈɛkst tə spˈiʧ mˈɑdᵊl.'
voice = torch.load(weights / 'voices' / 'af_heart.pt', map_location='cpu', weights_only=True)
ref_s = voice[len(phonemes) - 1]

prosody = {}


def record(name, tensor):
    prosody[name] = {'shape': list(tensor.shape), 'min': float(tensor.min()), 'max': float(tensor.max())}


original_f0n = model.predictor.F0Ntrain


def f0n(x, s):
    F0, N = original_f0n(x, s)
    record('aligned_prosody_features en', x)
    record('style_prosody s', s)
    record('F0_pred', F0)
    record('N_pred', N)
    return F0, N


model.predictor.F0Ntrain = f0n
captured = {}
generator = model.decoder.generator
generator.m_source.register_forward_pre_hook(lambda m, a: captured.update(f0=a[0].clone()))
model.decoder.register_forward_pre_hook(lambda m, a: (record('asr', a[0]), record('style_decoder', a[3]))[0] and None)

leaf_counts = collections.Counter()
leaf_shapes = collections.defaultdict(set)


def leaf_hook(module, inputs, output):
    kind = type(module).__name__
    leaf_counts[kind] += 1
    first = next((t for t in inputs if isinstance(t, torch.Tensor)), None)
    if first is not None:
        leaf_shapes[kind].add(str(tuple(first.shape)))


for module in model.modules():
    if not list(module.children()):
        module.register_forward_hook(leaf_hook)

ops = collections.Counter()
op_shapes = collections.defaultdict(set)


class Ops(TorchDispatchMode):
    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        name = str(func.overloadpacket).replace('aten.', '')
        ops[name] += 1
        shapes = tuple(tuple(a.shape) for a in args if isinstance(a, torch.Tensor))
        if len(op_shapes[name]) < 6:
            op_shapes[name].add(str(shapes))
        return func(*args, **(kwargs or {}))


with torch.no_grad(), Ops():
    output = model(phonemes, ref_s, return_output=True)

audio = output.audio.numpy()
with wave.open(str(out / 'kokoro_original.wav'), 'wb') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(24000)
    w.writeframes((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())

trace = {
    'phonemes': phonemes,
    'input_ids': len(list(filter(None, map(model.vocab.get, phonemes)))) + 2,
    'durations_frames': output.pred_dur.tolist(),
    'audio_samples': int(audio.shape[0]),
    'albert_hidden_act': model.bert.config.hidden_act,
    'albert': {k: getattr(model.bert.config, k) for k in ('embedding_size', 'hidden_size', 'num_hidden_layers', 'num_hidden_groups', 'num_attention_heads', 'intermediate_size')},
    'prosody': prosody,
    'leaf_modules': {k: {'calls': v, 'input_shapes': sorted(leaf_shapes[k])[:4]} for k, v in leaf_counts.most_common()},
    'aten_ops': {k: {'calls': v, 'shapes': sorted(op_shapes[k])[:4]} for k, v in ops.most_common()},
}
(out / 'trace.json').write_text(json.dumps(trace, indent=1, ensure_ascii=False))
print(f"audio {audio.shape[0]} samples ({audio.shape[0] / 24000:.2f} s), {sum(ops.values())} aten calls of {len(ops)} kinds, {sum(leaf_counts.values())} leaf module calls of {len(leaf_counts)} kinds")
print('albert', trace['albert'], trace['albert_hidden_act'])
print('durations', trace['durations_frames'])
for k, v in prosody.items():
    print(k, v)

# The harmonic source, SineGen: the frame phase increments are one sample of
# each 300 sample frame, so rand_ini, added at sample 0 only, is never read.
sine = generator.m_source.l_sin_gen
harmonics = captured['f0'] * torch.arange(1, sine.dim + 1).view(1, 1, -1)
with torch.no_grad():
    sines = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        sines.append(sine._f02sine(harmonics))
    rad = (harmonics / sine.sampling_rate) % 1
    frames = F.interpolate(rad.transpose(1, 2), scale_factor=1 / sine.upsample_scale, mode='linear').transpose(1, 2)
    phase = F.interpolate((torch.cumsum(frames, 1) * 2 * torch.pi).transpose(1, 2) * sine.upsample_scale,
                          scale_factor=sine.upsample_scale, mode='linear')
print('sines differ across rand_ini seeds by', float((sines[0] - sines[1]).abs().max()))
print('frame increments are sample 149 of each frame:', bool(torch.equal(frames[0], rad[0, 149::sine.upsample_scale])))
print('largest |phase| per harmonic', [round(float(v)) for v in phase.abs().amax(dim=(0, 2))])
