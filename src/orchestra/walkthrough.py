"""Offline conductor: audition built worlds as a player would hear them.

    python -m orchestra.walkthrough out/worlds/jungle_temple:A,E out/worlds/orbital_station:A,E -o walk.wav

Each area plays three passes: tier 1, tiers 1–2, then everything. On leaving,
non-bed layers cut on the loop line and ring out through their _tail16 while
beds crossfade, and the next area (or world) starts on its own sample 0 —
exactly the rules in docs/ENGINE.md. Worlds at different tempi simply hand
over at a loop boundary (their shared cadence grammar makes that land).
"""
import json
import os
import sys

import numpy as np
import soundfile as sf


def load(d, name):
    for ext in ('.ogg', '.wav'):
        p = os.path.join(d, name + ext)
        if os.path.exists(p):
            x, sr = sf.read(p, dtype='float32')
            return x.T if x.ndim > 1 else np.stack([x, x])
    raise FileNotFoundError(os.path.join(d, name))


def walk(plan, passes=(1, 2, 3), xfade_s=3.0):
    out = []
    pos = 0
    tails = []      # (start, array)
    prev_beds = None
    sr = None
    for d, areas in plan:
        meta = json.load(open(os.path.join(d, 'meta.json')))
        sr = meta['sr']
        N = meta['loop']
        for a in areas:
            tiers = meta['cues']['tiers'][a]
            stems = {s: load(d, f'{a}_{s}') for s in meta['stems'][a]}
            beds = set(meta.get('beds', []))
            for k, tier in enumerate(passes):
                seg = np.zeros((2, N), np.float32)
                for s, x in stems.items():
                    if tiers[s] <= tier:
                        g = np.ones(N, np.float32)
                        if k == 0 and s in beds:
                            g[:int(xfade_s * sr)] = np.linspace(0, 1, int(xfade_s * sr))
                        seg += x * g
                out.append((pos, seg))
                if prev_beds is not None and k == 0:
                    out.append((pos, prev_beds))
                    prev_beds = None
                pos += N
            # leave: tails for non-beds, faded loop start for beds
            for s, x in stems.items():
                if s in beds:
                    fo = x[:, :int(xfade_s * sr)] * np.linspace(1, 0, int(xfade_s * sr))[None]
                    prev_beds = fo if prev_beds is None else prev_beds + fo
                else:
                    tails.append((pos, load(d, f'{a}_{s}_tail16')))
    if prev_beds is not None:
        out.append((pos, prev_beds))
    end = max(p + x.shape[1] for p, x in out + tails)
    mix = np.zeros((2, end), np.float32)
    for p, x in out + tails:
        mix[:, p:p + x.shape[1]] += x
    return mix / max(1.0, np.max(np.abs(mix)) / 0.95), sr


if __name__ == '__main__':
    args = sys.argv[1:]
    o = 'walkthrough.wav'
    if '-o' in args:
        o = args[args.index('-o') + 1]
        args = args[:args.index('-o')] + args[args.index('-o') + 2:]
    plan = []
    for a in [x for i, x in enumerate(args) if x != '--passes' and (i == 0 or args[i - 1] != '--passes')]:
        d, areas = a.split(':') if ':' in a else (a, None)
        meta = json.load(open(os.path.join(d, 'meta.json')))
        plan.append((d, areas.split(',') if areas else list(meta['stems'])))
    passes = (1, 2, 3)
    if '--passes' in sys.argv:
        passes = tuple(int(c) for c in sys.argv[sys.argv.index('--passes') + 1].split(','))
        plan = [p for p in plan if not p[0].startswith('--') and not p[0][0].isdigit()]
    mix, sr = walk(plan, passes)
    sf.write(o, mix.T, sr, subtype='PCM_16')
    print(f'wrote {o} ({mix.shape[1] / sr:.0f} s)')
