"""loudness-balance stems, build previews, analyse."""
import sys, json, os
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from synth import SR, N, BAR, SPB
from render import STEMS

# target integrated loudness (LUFS, gated) of each stem *when it plays*
BASE = dict(melody=-20.5, bass=-22.0, keys=-25.0, pad=-27.0, marimba=-26.5, kalimba=-25.0,
            drums=-24.0, shaker=-31.0, woods=-28.0, amb=-31.0, water=-32.0, ornament=-27.5)
OVR = {
    'B': dict(water=-26.5, pad=-25.0, melody=-21.5, kalimba=-25.5, marimba=-28.5, drums=-28.0, amb=-32.0, shaker=-32.0),
    'C': dict(pad=-24.5, melody=-19.5, drums=-27.0, bass=-22.5, amb=-30.0, ornament=-26.0, water=-31.0),
    'D': dict(amb=-28.5, pad=-24.0, water=-29.5, melody=-21.5, drums=-26.0, keys=-26.5, bass=-23.0, woods=-27.0,
              shaker=-30.0, ornament=-26.0),
    'E': dict(drums=-22.0, marimba=-25.0, shaker=-29.0, woods=-27.0, bass=-21.5, melody=-20.0),
}
MASTER_PEAK_DB = -1.5

meter = pyln.Meter(SR)


SECTION_OFFSET = dict(A=0.0, B=-1.5, C=0.0, D=-3.5, E=0.0)


def target(sec, stem):
    return OVR.get(sec, {}).get(stem, BASE[stem]) + SECTION_OFFSET[sec]


def load(sec, stem):
    return np.load(f'out/raw/{sec}_{stem}.npy').astype(np.float64)


def stem_gains(secs='ABCDE'):
    g = {}
    for sec in secs:
        for stem in STEMS:
            y = load(sec, stem)
            L = meter.integrated_loudness(y.T)
            g[f'{sec}_{stem}'] = 10 ** ((target(sec, stem) - L) / 20)
    import glob
    for p in sorted(glob.glob('out/raw/st_*.npy')):
        k = os.path.basename(p)[:-4]
        y = np.load(p).astype(np.float64)
        L = meter.integrated_loudness(y.T)
        g[k] = 10 ** ((-25.0 - L) / 20)
    return g


if __name__ == '__main__':
    secs = sys.argv[1] if len(sys.argv) > 1 else 'ABCDE'
    g = stem_gains(secs)
    json.dump(g, open('out/gains_pre.json', 'w'), indent=1)
    for sec in secs:
        full = sum(load(sec, s) * g[f'{sec}_{s}'] for s in STEMS)
        pk = 20 * np.log10(np.max(np.abs(full)))
        print(sec, 'full mix LUFS %.1f  peak %.1f dB' % (meter.integrated_loudness(full.T), pk))
