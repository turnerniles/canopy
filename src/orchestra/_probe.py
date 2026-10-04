"""dev probe (not part of the library API): numeric inspection of presets.
python3 -m orchestra._probe name [pitch] [dur]"""
import sys
import numpy as np
from . import REGISTRY, play, SR, mtof, nm


def mono(y):
    return y if y.ndim == 1 else y.mean(0)


def harmonics(y, f, a=0.25, b=0.75, K=16):
    m = mono(y)
    seg = m[int(a * SR):int(b * SR)]
    seg = seg * np.hanning(len(seg))
    S = np.abs(np.fft.rfft(seg, 1 << 17))
    fr = np.fft.rfftfreq(1 << 17, 1 / SR)
    out = []
    for k in range(1, K + 1):
        band = (fr > f * k * 0.97) & (fr < f * k * 1.03)
        out.append(S[band].max() if band.any() else 0)
    out = np.array(out)
    tot = np.sum(S ** 2)
    harm = np.sum(out ** 2) * 6
    cen = np.sum(fr * S ** 2) / tot
    return 20 * np.log10(out / (out.max() + 1e-12) + 1e-9), cen


def inst_pitch(y, f, win=0.05):
    """frame-wise f0 near f via autocorr-free spectral peak (cents)."""
    m = mono(y)
    hop = int(0.01 * SR)
    L = int(win * SR)
    res = []
    for s in range(0, len(m) - L, hop):
        seg = m[s:s + L] * np.hanning(L)
        S = np.abs(np.fft.rfft(seg, 1 << 15))
        fr = np.fft.rfftfreq(1 << 15, 1 / SR)
        band = (fr > f * 0.75) & (fr < f * 1.33)
        k = np.argmax(S * band)
        res.append(1200 * np.log2(fr[k] / f) if S[k] > 1e-6 else np.nan)
    return np.array(res)


def rms_env(y, win=0.02):
    m = mono(y)
    k = int(win * SR)
    nb = len(m) // k
    return np.sqrt(np.mean(m[:nb * k].reshape(nb, k) ** 2, 1))


if __name__ == '__main__':
    name = sys.argv[1]
    inst = REGISTRY[name]
    p = sys.argv[2] if len(sys.argv) > 2 else (inst.lo + inst.hi) // 2
    p = nm(p) if isinstance(p, str) and not p.isdigit() else int(p)
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    f = float(mtof(p))
    for vel in (0.3, 0.7, 1.0):
        y = play(name, p, dur=dur, vel=vel, seed=3)
        h, cen = harmonics(y, f)
        e = rms_env(y)
        pk = np.argmax(e)
        print(f'vel {vel}: peak {np.abs(y).max():.2f} rmsmax {e.max():.3f} at {pk*0.02:.2f}s  centroid {cen:6.0f} Hz  len {y.shape[-1]/SR:.2f}s')
        print('   harm dB', ' '.join(f'{v:5.0f}' for v in h))
    ip = inst_pitch(y, f)
    print('pitch cents per 50ms:', ' '.join(f'{v:4.0f}' for v in ip[::5][:30]))
    print('env (x0.1 s):', ' '.join(f'{v:.2f}' for v in (e / e.max())[::5][:30]))
