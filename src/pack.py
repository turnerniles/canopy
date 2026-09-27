"""
Build the engine pack for the jungle-focus score:
  stems/<area>/<A>_<layer>.ogg (+ _tail8 / _tail16 for non-bed layers)
  transitions/*.ogg
  oneshots/<group>/*.wav   (dry creature and instrument one-shots for random emitters)
  source/*.py              (everything needed to regenerate the audio)
  README.md
"""
import tempfile
TMP = tempfile.gettempdir()
import os, json, shutil, subprocess, glob, zipfile
import numpy as np
import soundfile as sf
from synth import *
from encode_web_jungle import stems_for, MASTER, BEDS, STINGS
import wild

OUT = 'out/pack/Canopy'
g = json.load(open('out/gains_jungle.json'))
shutil.rmtree('out/pack', ignore_errors=True)
os.makedirs(OUT, exist_ok=True)


def ogg(y, dst, q='4'):
    tmp = TMP + '/pk.wav'
    sf.write(tmp, y.T, SR, subtype='PCM_24')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:a', 'libvorbis', '-q:a', q, dst], check=True)


names = {'A': 'A_exploration', 'B': 'B_river_waterfall', 'C': 'C_canopy_vista', 'D': 'D_moss_ruins', 'E': 'E_vine_run'}
for sec in 'ABCDE':
    d = f'{OUT}/stems/{names[sec]}'
    os.makedirs(d, exist_ok=True)
    for stem in stems_for(sec):
        k = f'{sec}_{stem}'
        gain = g[k] * MASTER
        ogg(np.load(f'out/raw/{k}.npy') * gain, f'{d}/{k}.ogg')
        if stem not in BEDS:
            ogg(np.load(f'out/raw/{k}_t8.npy') * gain, f'{d}/{k}_tail8.ogg')
            ogg(np.load(f'out/raw/{k}_t16.npy') * gain, f'{d}/{k}_tail16.ogg')
    print(sec, flush=True)
os.makedirs(f'{OUT}/transitions', exist_ok=True)
for n in STINGS:
    ogg(np.load(f'out/raw/st_{n}.npy') * g['st_' + n] * MASTER, f'{OUT}/transitions/{n}.ogg')


# ------------------------------------------------------------- one-shots ----
class _Tr:
    def __init__(self, n):
        self.y = np.zeros(n)

    def add(self, sig, start, pan=0.0, gain=1.0, sends=None, **kw):
        sig = sig if sig.ndim == 1 else sig.mean(0)
        s = int(start)
        e = min(len(self.y), s + len(sig))
        if e > s:
            self.y[s:e] += sig[:e - s] * gain


class _Ctx:
    def __init__(self, n, seed):
        self.rng = np.random.default_rng(seed)
        self.tr = _Tr(n)
        self.room = 'canopy'


def one(name, y, norm=-3.0):
    y = np.asarray(y, float)
    y = y / (np.max(np.abs(y)) + 1e-12) * 10 ** (norm / 20)
    a = np.abs(y)
    idx = np.nonzero(a > 10 ** (-66 / 20))[0]
    if len(idx):
        y = y[:idx[-1] + int(0.02 * SR)]
    y = fade_out(fade_in(y, 0.001), 0.01)
    p = f'{OUT}/oneshots/{name}.wav'
    os.makedirs(os.path.dirname(p), exist_ok=True)
    sf.write(p, y, SR, subtype='PCM_16')


rng = np.random.default_rng(99)
for i in range(4):
    c = _Ctx(int(16 * SR), 10 + i)
    wild.cicada(c, int(0.2 * SR), rng.uniform(8, 12), center=[5200, 5600, 6100, 4800][i], pulse=[190, 210, 230, 175][i])
    one(f'creatures/cicada_swell_{i + 1}', c.tr.y)
for i in range(6):
    c = _Ctx(int(3 * SR), 20 + i)
    wild.warbler(c, int(0.05 * SR), near=0.8)
    one(f'creatures/songbird_{i + 1}', c.tr.y)
for i in range(3):
    c = _Ctx(int(3 * SR), 30 + i)
    wild.woodpecker(c, int(0.05 * SR))
    one(f'creatures/woodpecker_{i + 1}', c.tr.y)
for i in range(3):
    c = _Ctx(int(9 * SR), 40 + i)
    tt = tvec(int(0.045 * SR))
    f0 = [2300, 2800, 3200][i]
    for j in range(12):
        f = f0 * (1 + 0.12 * tt / 0.045)
        c.tr.add(np.sin(TWOPI * np.cumsum(f) / SR) * np.hanning(len(tt)), int((0.1 + j * [0.42, 0.33, 0.5][i]) * SR))
    one(f'creatures/treefrog_{i + 1}', c.tr.y)
for i in range(3):
    n = int(0.9 * SR)
    t = tvec(int(0.22 * SR))
    f0 = [280, 340, 400][i]
    y = np.zeros(n)
    for r in range(3):
        cr = np.sin(TWOPI * f0 * t + 2 * np.sin(TWOPI * f0 * 2 * t)) * (0.5 + 0.5 * np.sin(TWOPI * 32 * t)) * np.hanning(len(t))
        y[int(r * 0.3 * SR):int(r * 0.3 * SR) + len(t)] += lp(cr, 2500)
    one(f'creatures/frog_croak_{i + 1}', y)
for i, pts in enumerate([
        [(0, 2600, 0), (0.012, 3100, 1), (0.05, 3600, .6), (0.07, 3300, 0)],
        [(0, 2000, 0), (0.03, 2100, 1), (0.22, 2100, .8), (0.28, 2500, .9), (0.5, 2450, 0)],
        [(0, 2800, 0), (0.05, 2940, 1), (0.9, 2100, 0.7), (1.2, 1900, 0)],
        [(0, 560, 0), (0.06, 600, 1), (0.25, 580, .8), (0.4, 540, 0)]]):
    one(f'creatures/bird_call_{i + 1}', bird_call(pts, rng, trill_hz=[0, 0, 11, 0][i], trill_depth=0.01, harm=[.12, .12, .25, .3][i]))
one('creatures/bird_trill', bird_call([(0, 4200, 0), (0.02, 4200, 1), (0.6, 3900, .7), (0.63, 4000, 0)], rng, trill_hz=26, trill_depth=0.06))
for n in ['D6', 'E6', 'F#6', 'A6', 'B6']:
    one(f'water/tuned_drip_{n.replace("#", "s")}', bubble(float(mtof(nm(n))), 0.07, rng, rise=0.035))
for i in range(5):
    one(f'water/bubble_{i + 1}', bubble(rng.uniform(500, 2400), rng.uniform(0.006, 0.02), rng, rise=0.2))
for i in range(4):
    n = int(0.06 * SR)
    t = tvec(n)
    f = rng.uniform(250, 700)
    tap = np.sin(TWOPI * f * t * (1 + 0.3 * np.exp(-t / 0.005))) * np.exp(-t / 0.012)
    hit = bp(rng.standard_normal(n), 1800, 6000) * np.exp(-t / 0.002)
    one(f'water/leaf_drip_{i + 1}', 0.6 * tap + 0.5 * hit)
for f, n in [(147, 'D3'), (220, 'A3'), (293.7, 'D4'), (329.6, 'E4'), (370, 'Fs4'), (440, 'A4')]:
    one(f'wood/logdrum_{n}', log_drum(f, 0.7, rng))
one('wood/woodblock', wood_block(1250, 0.7, rng))
for k in ['bass', 'low', 'tone', 'slap', 'ghost']:
    one(f'percussion/handdrum_{k}', hand_drum(k, 0.7, rng))
for f in (78, 96, 128):
    one(f'percussion/felt_tom_{f}hz', tom(f, 0.7, rng))
one('percussion/felt_kick', felt_kick(0.8, rng))
one('percussion/frame_drum', frame_drum(0.7, rng))
for i in range(3):
    one(f'percussion/shaker_{i + 1}', shaker(0.6, rng, length=[0.07, 0.11, 0.2][i]))
one('percussion/seed_rattle_rise', seed_rattle(1.3, 0.6, rng))
for n in ['D4', 'F#4', 'A4', 'B4', 'D5', 'E5', 'F#5', 'A5']:
    one(f'mallets/kalimba_{n.replace("#", "s")}', kalimba(float(mtof(nm(n))), 0.7, rng))
    one(f'mallets/marimba_{n.replace("#", "s")}', marimba(float(mtof(nm(n))), 0.7, rng, hard=0.3))
for n in ['A5', 'D6', 'E6']:
    one(f'mallets/bell_{n}', bell(float(mtof(nm(n))), 0.6, rng))

# ------------------------------------------------------------- source -------
os.makedirs(f'{OUT}/source', exist_ok=True)
for f in ['synth.py', 'score.py', 'render.py', 'wild.py', 'stingers.py', 'conductor.py', 'mix.py', 'jungle_mix.py',
          'encode.py', 'encode_web_jungle.py', 'pack.py']:
    shutil.copy(f, f'{OUT}/source/{f}')
shutil.copy('../docs/ENGINE.md', f'{OUT}/README.md')

os.makedirs('deliver', exist_ok=True)
zp = 'deliver/Canopy_jungle_stems.zip'
if os.path.exists(zp):
    os.remove(zp)
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_STORED) as z:
    for p in sorted(glob.glob(f'{OUT}/**/*', recursive=True)):
        if os.path.isfile(p):
            z.write(p, os.path.relpath(p, 'out/pack'))
print('zip MB %.1f' % (os.path.getsize(zp) / 1e6), 'files', len(glob.glob(f'{OUT}/**/*.*', recursive=True)))
