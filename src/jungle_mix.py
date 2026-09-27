"""
Canopy — jungle-focus mixes.

  canopy_jungle_bed.wav   jungle in front, a soft instrumental bed underneath, no melody
  canopy_pure_jungle.wav  jungle only (ambience, water, wildlife, tuned birds)
"""
import json, os, sys, glob
import numpy as np
import pyloudnorm as pyln
from synth import SR
import conductor as C

meter = pyln.Meter(SR)

# loudness targets (LUFS, gated) — the jungle leads, the instruments sit underneath
JT = dict(amb=-23.5, water=-25.0, wildlife=-27.0, birds=-26.5, woods=-30.0,
          shaker=-34.5, drums=-31.5, bass=-29.5, keys=-31.0, pad=-29.5, marimba=-33.0, kalimba=-32.5)
JOVR = {
    'B': dict(water=-22.5, wildlife=-30.0, pad=-29.0),
    'C': dict(amb=-23.0, water=-27.0, pad=-28.5, birds=-26.0),
    'D': dict(amb=-24.5, water=-25.5, wildlife=-30.0, pad=-30.0, drums=-31.0),
    'E': dict(amb=-24.5, wildlife=-26.0, drums=-30.0, marimba=-32.0, shaker=-33.0, bass=-29.0),
}
JOFF = dict(A=0.0, B=-0.5, C=0.0, D=-2.0, E=0.0)
BASE_OF = lambda stem: 'amb' if stem.startswith('amb') else stem
STEMS_J = ['amb', 'amb2', 'amb3', 'amb4', 'water', 'wildlife', 'birds', 'woods', 'shaker', 'drums',
           'bass', 'keys', 'pad', 'marimba', 'kalimba']
JSTING = ['gust', 'flock', 'shower', 'woodpecker_roll', 'far_cry', 'fill_hands', 'fill_logs', 'rise']


def jungle_gains():
    g = {}
    for sec in 'ABCDE':
        for stem in STEMS_J:
            k = f'{sec}_{stem}'
            y = np.load(f'out/raw/{k}.npy').astype(np.float64)
            base = BASE_OF(stem)
            tgt = JOVR.get(sec, {}).get(base, JT[base]) + JOFF[sec]
            g[k] = 10 ** ((tgt - meter.integrated_loudness(y.T)) / 20)
    for n in JSTING:
        y = np.load(f'out/raw/st_{n}.npy').astype(np.float64)
        g['st_' + n] = 10 ** ((-26.0 - meter.integrated_loudness(y.T)) / 20)
    return g


def P(**kw):
    return dict(kw)


def add(d, **kw):
    e = dict(d)
    e.update(kw)
    return e


J0 = P(amb=1, water=.8, wildlife=.8, birds=1)
A_BED = add(J0, woods=.7, pad=.8, bass=.8, keys=.7)
A_FULL = add(A_BED, water=.7, wildlife=.9, woods=.6, shaker=.7, drums=.7, marimba=.6)
BLOCKS = [
    ('A', [P(amb=1, water=.8, wildlife=.7), add(J0, woods=.6), add(J0, woods=.7, pad=.6, bass=.6),
           add(J0, woods=.7, pad=.7, bass=.7, keys=.6)], {'amb': 'amb'}),
    ('A', [add(A_BED, wildlife=.9), add(A_BED, wildlife=.9, shaker=.6),
           add(A_BED, wildlife=.9, shaker=.7, drums=.6, marimba=.5), add(A_BED, wildlife=.9, shaker=.7, drums=.6, marimba=.5)],
     {'amb': 'amb2'}),
    ('A', [A_FULL, A_FULL, add(J0, amb=1.1, water=.9, wildlife=1, pad=.5), add(J0, amb=1.1, water=.9, wildlife=1, pad=.5, woods=.6)],
     {'amb': 'amb3'}),
    ('B', [P(amb=1, water=1, wildlife=.9, birds=1), P(amb=1, water=1, wildlife=.9, birds=1, pad=.7, kalimba=.6),
           P(amb=1, water=1, wildlife=.9, birds=1, pad=.7, kalimba=.6, bass=.6, keys=.5),
           P(amb=1, water=1, wildlife=.9, birds=1, pad=.7, kalimba=.6, bass=.6, keys=.5, shaker=.4, marimba=.4)], {}),
    ('C', [add(J0, pad=.8), add(J0, pad=.8, keys=.6, bass=.7), add(J0, pad=.8, keys=.6, bass=.7, kalimba=.5, woods=.6),
           add(J0, pad=.8, keys=.6, bass=.7, kalimba=.5, woods=.6, marimba=.45, drums=.45)], {}),
    ('D', [P(amb=1, water=1, wildlife=.9, birds=.8), P(amb=1, water=1, wildlife=.9, birds=.8, woods=.8, pad=.6),
           P(amb=1, water=1, wildlife=.9, birds=.8, woods=.8, pad=.6, bass=.6, keys=.45),
           P(amb=1, water=1, wildlife=.9, birds=.8, woods=.8, pad=.6, bass=.6, keys=.45, drums=.6, shaker=.6)], {}),
    ('E', [P(amb=1, wildlife=1, birds=1, woods=.8, shaker=.6), P(amb=1, wildlife=1, birds=1, woods=.8, shaker=.6, drums=.7, bass=.7),
           P(amb=1, wildlife=1, birds=1, woods=.8, shaker=.6, drums=.7, bass=.7, marimba=.55, keys=.5),
           P(amb=1, wildlife=1, birds=1, woods=.8, shaker=.6, drums=.7, bass=.7, marimba=.55, keys=.5, kalimba=.5, pad=.5, water=.5)], {}),
    ('A', [add(A_BED, drums=.5, shaker=.5), add(A_BED, drums=.5, shaker=.5), add(J0, water=.8, wildlife=.9, pad=.5, keys=.4),
           add(J0, water=.8, wildlife=.9)], {'amb': 'amb4'}),
    ('A', [J0, J0, J0, J0], {'amb': 'amb2'}),
]
STINGS = [
    ('far_cry', 16 * 2 + 9, 1.0),
    ('shower', 16 * 3 + 1, 0.7),
    ('gust', 16 * 4 + 1, 1.0), ('flock', 16 * 4 + 1, 0.9),
    ('far_cry', 16 * 5 + 1, 0.9),
    ('woodpecker_roll', 16 * 6 + 1, 1.0),
    ('flock', 16 * 7 + 1, 1.0),
]
JUNGLE_ONLY = {'amb', 'water', 'wildlife', 'birds'}


def pure(blocks):
    out = []
    for sec, phs, var in blocks:
        out.append((sec, [{k: v for k, v in ph.items() if k in JUNGLE_ONLY} or {'amb': 1} for ph in phs], var))
    return out


if __name__ == '__main__':
    g = jungle_gains()
    json.dump(g, open('out/gains_jungle.json', 'w'), indent=1)
    os.makedirs('out/mix', exist_ok=True)
    y = C.perform(BLOCKS, 'out/mix/canopy_jungle_bed.wav', STINGS, outro_fade_bars=12, gains=g,
                  beds=('amb', 'water', 'wildlife'))
    print('jungle+bed', y.shape[1] / SR, 's')
    ps = [s for s in STINGS if s[0] in ('gust', 'flock', 'shower', 'far_cry')]
    y = C.perform(pure(BLOCKS), 'out/mix/canopy_pure_jungle.wav', ps, outro_fade_bars=12, gains=g,
                  beds=('amb', 'water', 'wildlife'), bed_xfade=5.0)
    print('pure', y.shape[1] / SR, 's')
