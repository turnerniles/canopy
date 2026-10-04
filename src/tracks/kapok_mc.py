"""Overlay a Master of Ceremonies on The Procession, Tubular Bells–style.

    python3 -m tracks.kapok_mc ../out/kapok DIR_WITH_mc_01.wav..mc_13.wav

The clips are spoken introductions, one per entry in kapok_cues.json 'intros'
(in order). They are not part of the code-only score: supply your own
recordings or TTS. Each clip is trimmed, given a little hall, and placed so the
speech ends just as its instrument enters; the music ducks 4 dB under it.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from orchestra import fx  # noqa: E402


def main(out_dir, mc_dir):
    x, sr = sf.read(os.path.join(out_dir, 'kapok.wav'), dtype='float32')
    x = x.T.copy()
    cues = json.load(open(os.path.join(out_dir, 'kapok_cues.json')))
    voice = np.zeros_like(x)
    duck = np.ones(x.shape[1], np.float32)
    for i, (t_cue, text) in enumerate(cues['intros']):
        path = os.path.join(mc_dir, f'mc_{i + 1:02d}.wav')
        if not os.path.exists(path):
            print('missing', path)
            continue
        y, r = sf.read(path, dtype='float32')
        if y.ndim > 1:
            y = y.mean(1)
        if r != sr:
            y = signal.resample_poly(y, sr, r).astype(np.float32)
        e = np.convolve(np.abs(y), np.ones(480) / 480, 'same')
        on = np.nonzero(e > 0.02 * e.max())[0]
        y = y[max(0, on[0] - 480):on[-1] + 2400]
        y = fx.shelf(fx.shelf(y, 180, -3, 'low', sr), 5000, 2, 'high', sr)
        entry = t_cue + 0.9                         # cue is 1.5 beats (0.9 s at 100 BPM) before the entry
        start = int((entry - len(y) / sr + 0.25) * sr)
        k = min(len(y), x.shape[1] - start)
        y = y / (np.sqrt(np.mean(y ** 2)) + 1e-9) * 0.075
        voice[0, start:start + k] += y[:k] * 0.97
        voice[1, start:start + k] += y[:k] * 0.97
        a, b = max(0, start - int(0.15 * sr)), min(x.shape[1], start + k + int(0.3 * sr))
        duck[a:b] = np.minimum(duck[a:b], 10 ** (-4 / 20))
        print(f'{entry:7.2f}s  {text}')
    duck = signal.lfilter([0.002], [1, -0.998], duck - 1).astype(np.float32) + 1
    wet = fx.reverb(voice * 0.25, 'hall', sr)[:, :x.shape[1]]
    mix = x * duck + voice + wet
    mix = fx.limit(mix, -1.0, sr=sr)
    sf.write(os.path.join(out_dir, 'kapok_with_mc.wav'), mix.T, sr, subtype='PCM_24')
    print('wrote', os.path.join(out_dir, 'kapok_with_mc.wav'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
