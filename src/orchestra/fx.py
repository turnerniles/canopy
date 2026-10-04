"""Effects: algorithmic reverb rooms, tape echo, bus compressor, limiter, EQ."""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d

from .core import SR, lp, hp, onepole_lp

# ------------------------------------------------------------------ rooms
ROOMS = {
    # name: rt60, predelay, low_mul, high_mul, early reflections, darkness
    'booth':    dict(rt60=0.35, predelay=0.003, er=8, er_spread=0.015, dark=12000),
    'room':     dict(rt60=0.8, predelay=0.006, er=14, er_spread=0.03, dark=10000),
    'studio':   dict(rt60=1.3, predelay=0.012, er=12, er_spread=0.04, dark=9500),   # the Manor
    'canopy':   dict(rt60=1.9, predelay=0.018, er=12, er_spread=0.05, dark=8500),
    'hall':     dict(rt60=2.8, predelay=0.024, er=16, er_spread=0.06, dark=9000, low_mul=1.2),
    'church':   dict(rt60=4.5, predelay=0.035, er=20, er_spread=0.08, dark=7000, low_mul=1.3),
    'ruins':    dict(rt60=3.4, predelay=0.012, er=22, er_spread=0.09, er_gain=1.4, dark=5200, low_mul=1.5),
    'valley':   dict(rt60=6.0, predelay=0.06, er=6, er_spread=0.2, dark=7500, build=0.12),
    'cosmos':   dict(rt60=9.0, predelay=0.08, er=4, er_spread=0.25, dark=11000, build=0.3, high_mul=0.8),
}


def make_ir(rt60=2.0, dur=None, predelay=0.015, low_mul=1.25, high_mul=0.5, er=10, er_spread=0.06,
            er_gain=0.5, dark=9000, seed=1, build=0.02, sr=SR):
    """stereo algorithmic room: multiband exponentially decaying decorrelated
    noise + sparse early reflections (energy-normalised)."""
    dur = dur or min(rt60 * 1.4, 10.0)
    n = int(dur * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    chans = []
    for _ in range(2):
        w = rng.standard_normal(n)
        lo = lp(w, 350, sr=sr)
        hi = hp(w, 3500, sr=sr)
        mid = w - lo - hi
        y = (lo * 10 ** (-3 * t / (rt60 * low_mul)) + mid * 10 ** (-3 * t / rt60) +
             hi * 10 ** (-3 * t / (rt60 * high_mul)))
        y *= 1 - np.exp(-t / build)
        erb = np.zeros(n)
        for _ in range(er):
            p = rng.uniform(0.003, er_spread)
            erb[int(p * sr)] += er_gain * rng.uniform(0.3, 1.0) * rng.choice([-1, 1]) * (1 - p / er_spread * 0.6)
        erb = lp(erb, 6000, sr=sr)
        y = y / (np.sqrt(np.sum(y ** 2)) + 1e-12) + erb * 0.08
        y = lp(y, dark, sr=sr)
        pd = int(predelay * sr)
        chans.append(np.concatenate([np.zeros(pd), y])[:n])
    ir = np.stack(chans)
    ir /= np.sqrt(np.sum(ir ** 2) / 2)
    return ir.astype(np.float32)


_IR = {}


def room_ir(name, sr=SR):
    key = (name, sr)
    if key not in _IR:
        spec = dict(ROOMS[name])
        _IR[key] = make_ir(seed=sum(map(ord, name)), sr=sr, **spec)
    return _IR[key]


def reverb(x, name, sr=SR):
    """x mono (n,) or stereo (2,n) send -> stereo wet (2, n + len(ir) - 1)."""
    ir = room_ir(name, sr)
    if x.ndim == 1:
        return np.stack([signal.oaconvolve(x, ir[c]) for c in range(2)]).astype(np.float32)
    return np.stack([signal.oaconvolve(x[c], ir[c]) for c in range(2)]).astype(np.float32)


# ------------------------------------------------------------------ echo
def tape_echo(x, delay_s, feedback=0.4, damp=3500.0, taps=8, pingpong=True, wow=0.0, sr=SR):
    """Repeat echo with progressively darker, decaying repeats (Oldfield's
    Echoplex/Binson-ish). x stereo (2,n) or mono. Returns stereo, longer."""
    if x.ndim == 1:
        x = np.stack([x, x]) * 0.7071
    d = int(delay_s * sr)
    n = x.shape[1]
    out = np.zeros((2, n + d * taps), np.float32)
    cur = x.astype(np.float64)
    for k in range(1, taps + 1):
        cur = np.stack([onepole_lp(c, damp, sr=sr) for c in cur]) * feedback
        if pingpong:
            cur = cur[::-1]
        s = k * d
        if wow:
            s += int(wow * sr * np.sin(k * 1.7))
        out[:, s:s + n] += cur
        if np.max(np.abs(cur)) < 1e-4:
            break
    return out


# ------------------------------------------------------------------ dynamics
def _env_follow(level, attack, release, sr=SR):
    """peak-ish envelope with separate attack/release (vectorised via two passes)."""
    a_at = np.exp(-1 / (attack * sr))
    a_re = np.exp(-1 / (release * sr))
    # fast attack follower then slow release follower, take the max: cheap approximation
    level = level.astype(np.float32, copy=False)
    up = signal.lfilter(np.float32([1 - a_at]), np.float32([1, -a_at]), level)
    dn = signal.lfilter(np.float32([1 - a_re]), np.float32([1, -a_re]), level)
    np.maximum(up, dn, out=up)
    return up


def compress(x, thr_db=-18.0, ratio=2.5, attack=0.02, release=0.25, makeup_db=0.0, knee_db=6.0, sr=SR):
    """stereo-linked RMS compressor. x (2,n)."""
    lvl = np.sqrt(np.maximum(_env_follow(np.max(x * x, axis=0), attack, release, sr), 1e-12))
    db = (20 * np.log10(lvl + 1e-9)).astype(np.float32)
    over = db - thr_db
    gr = np.where(over <= -knee_db / 2, 0.0,
                  np.where(over >= knee_db / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee_db / 2) ** 2 / (2 * knee_db)))
    g = (10 ** ((makeup_db - gr) / 20)).astype(np.float32)
    return x * g


def limit(x, ceiling_db=-1.0, lookahead=0.004, release=0.08, sr=SR):
    """lookahead brickwall-ish limiter. x (2,n)."""
    ceil = 10 ** (ceiling_db / 20)
    peak = np.max(np.abs(x), axis=0)
    need = np.minimum(1.0, ceil / np.maximum(peak, 1e-9))
    la = max(1, int(lookahead * sr))
    need = minimum_filter1d(need, size=2 * la + 1, mode='nearest')
    a = np.exp(-1 / (release * sr))
    # smooth recovery (one-pole on 1-g, so attack stays instant via the min-filter)
    g = 1 - signal.lfilter([1 - a], [1, -a], 1 - need)
    g = np.minimum(g, need)
    y = x * g
    return np.clip(y, -ceil, ceil).astype(np.float32)


def tape(x, drive=1.2):
    return (np.tanh(drive * x) / np.tanh(drive)).astype(np.float32)


# ------------------------------------------------------------------ EQ
def shelf(x, f0, gain_db, kind='high', sr=SR):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / 2 * np.sqrt(2)
    cw = np.cos(w0)
    s = 1 if kind == 'high' else -1
    b0 = A * ((A + 1) + s * (A - 1) * cw + 2 * np.sqrt(A) * alpha)
    b1 = -2 * s * A * ((A - 1) + s * (A + 1) * cw)
    b2 = A * ((A + 1) + s * (A - 1) * cw - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - s * (A - 1) * cw + 2 * np.sqrt(A) * alpha
    a1 = 2 * s * ((A - 1) - s * (A + 1) * cw)
    a2 = (A + 1) - s * (A - 1) * cw - 2 * np.sqrt(A) * alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])
    return signal.lfilter(b, a, x, axis=-1).astype(np.float32)


def loudness_normalize(x, target_lufs=-14.0, sr=SR):
    try:
        import pyloudnorm as pyln
        y, r = x, sr
        if x.shape[1] > 180 * sr:     # long pieces: measure a half-rate copy (saves ~1 GB; <0.1 LU difference)
            y, r = signal.resample_poly(x, 1, 2, axis=1).astype(np.float32), sr // 2
        meter = pyln.Meter(r)
        L = meter.integrated_loudness(y.T.astype(np.float64))
        del y
        return (x * 10 ** ((target_lufs - L) / 20)).astype(np.float32), L
    except ImportError:
        rms = np.sqrt(np.mean(x ** 2)) + 1e-9
        return (x * (10 ** ((target_lufs + 3) / 20) / rms)).astype(np.float32), None
