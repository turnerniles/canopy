"""
Canopy Orchestra — shared instrument contract.

Every instrument in the library is a *model function* plus a *preset* (a named
set of parameters). Models are pure NumPy/SciPy, render one note at a time, and
never use recordings or samples.

Model contract
--------------
    fn(f, dur, vel, rng, sr, **params) -> np.ndarray

    f      fundamental in Hz (for unpitched instruments: a tuning hint; the
           registry passes mtof(pitch) so "pitch" can retune a drum)
    dur    gate length in seconds (how long the key is held / bow is drawn).
           Percussive models may ignore it; sustaining models must release
           after it.
    vel    0..1 dynamic. Must change timbre as well as level (brighter, more
           noise, harder attack when louder).
    rng    np.random.Generator — the ONLY source of randomness (determinism).
    sr     sample rate.
    returns mono (n,) or stereo (2, n) float array starting at the onset and
           containing the natural release / ring-out. Length should be about
           dur + release, capped sensibly (never more than ~12 s).

The registry wraps every call: it converts to float32, removes DC, applies a
5 ms end fade, and applies a per-preset *calibration gain* so that every preset
plays at a comparable loudness. Composers then set levels in dB per part.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import signal

SR = 48000
TWOPI = 2 * np.pi

# --------------------------------------------------------------------------- pitch
_NAMES = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def nm(s: str) -> int:
    """'F#5' -> 78, 'Bb1' -> 34"""
    s = s.strip()
    pc = _NAMES[s[0].upper()]
    i = 1
    while i < len(s) and s[i] in '#b':
        pc += 1 if s[i] == '#' else -1
        i += 1
    return 12 * (int(s[i:]) + 1) + pc


def mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0)


def ftom(f):
    return 69 + 12 * np.log2(np.asarray(f, dtype=float) / 440.0)


# --------------------------------------------------------------------------- envelopes
def tvec(n, sr=SR):
    return np.arange(n) / sr


def env_perc(n, attack, tau, sr=SR):
    """raised-cosine attack then exponential decay (tau seconds)."""
    t = tvec(n, sr)
    e = np.exp(-np.maximum(t - attack, 0.0) / max(tau, 1e-5))
    na = int(min(n, max(1, attack * sr)))
    e[:na] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    return e


def env_adsr(n, dur, a=0.01, d=0.1, s=0.7, r=0.2, sr=SR, curve=1.0):
    """gate-based ADSR; release starts at dur. Exponential-ish segments."""
    t = tvec(n, sr)
    e = np.empty(n)
    att = np.clip(t / max(a, 1e-5), 0, 1) ** curve
    dec = s + (1 - s) * np.exp(-np.maximum(t - a, 0) / max(d, 1e-5))
    e[:] = np.where(t < a, att, dec)
    k = int(dur * sr)
    if k < n:
        lvl = e[k - 1] if k > 0 else 0.0
        tt = np.arange(n - k) / sr
        e[k:] = lvl * np.exp(-tt / max(r, 1e-5) * 3.0)
    return e


def fade(x, fin=0.0, fout=0.0, sr=SR):
    x = np.array(x, dtype=float, copy=True)
    n = x.shape[-1]
    if fin > 0:
        k = min(n, max(1, int(fin * sr)))
        x[..., :k] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)
    if fout > 0:
        k = min(n, max(1, int(fout * sr)))
        x[..., -k:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(k) / k)
    return x


# --------------------------------------------------------------------------- noise & filters
def noise(n, rng):
    return rng.standard_normal(n)


def pink(n, rng):
    w = rng.standard_normal(n)
    W = np.fft.rfft(w)
    f = np.arange(len(W), dtype=float)
    f[0] = 1
    W /= np.sqrt(f)
    y = np.fft.irfft(W, n=n)
    return y / (np.std(y) + 1e-12)


def _nyq(fc, sr):
    return float(np.clip(fc, 5.0, sr * 0.49))


def lp(x, fc, order=2, sr=SR):
    return signal.sosfilt(signal.butter(order, _nyq(fc, sr), 'low', fs=sr, output='sos'), x)


def hp(x, fc, order=2, sr=SR):
    return signal.sosfilt(signal.butter(order, _nyq(fc, sr), 'high', fs=sr, output='sos'), x)


def bp(x, lo, hi, order=2, sr=SR):
    lo, hi = _nyq(lo, sr), _nyq(hi, sr)
    if hi <= lo * 1.01:
        hi = lo * 1.02
    return signal.sosfilt(signal.butter(order, [lo, hi], 'band', fs=sr, output='sos'), x)


def reson(x, f, q, sr=SR):
    """constant-peak resonator (iirpeak)."""
    f = _nyq(f, sr)
    b, a = signal.iirpeak(f, max(q, 0.5), fs=sr)
    return signal.lfilter(b, a, x)


def formant_bank(x, formants, sr=SR):
    """formants: [(freq, bandwidth_hz, gain_linear), ...] -> parallel resonators."""
    y = np.zeros_like(x, dtype=float)
    for f, bw, g in formants:
        y += g * reson(x, f, max(f / max(bw, 1.0), 0.7), sr)
    return y


def tv_filter(x, fc, kind='low', q=0.707, sr=SR, block=64):
    """time-varying 2-pole SVF-ish filter. fc: scalar or array (len(x)).
    Processed in blocks with carried state (cheap, smooth enough for sweeps)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    fc = np.broadcast_to(np.asarray(fc, dtype=float), (n,))
    y = np.empty(n)
    zi = np.zeros((1, 2))
    for s in range(0, n, block):
        e = min(n, s + block)
        f = _nyq(float(fc[s:e].mean()), sr)
        if kind == 'low':
            b, a = signal.iirfilter(2, f, btype='low', ftype='butter', fs=sr) if q == 0.707 else _rbj_lp(f, q, sr)
        elif kind == 'high':
            b, a = signal.iirfilter(2, f, btype='high', ftype='butter', fs=sr)
        else:  # band
            b, a = signal.iirpeak(f, max(q, 0.5), fs=sr)
        sos = np.hstack([b, a])[None, :]
        y[s:e], zi = signal.sosfilt(sos, x[s:e], zi=zi)
    return y


def _rbj_lp(f, q, sr):
    w0 = TWOPI * f / sr
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


def softclip(x, drive=1.0):
    return np.tanh(drive * x) / np.tanh(drive)


def onepole_lp(x, fc, sr=SR):
    a = np.exp(-TWOPI * _nyq(fc, sr) / sr)
    return signal.lfilter([1 - a], [1, -a], x)


# --------------------------------------------------------------------------- oscillators
def phase_from_freq(freq, sr=SR, phase0=0.0):
    """freq: scalar or array -> running phase (radians)."""
    freq = np.asarray(freq, dtype=float)
    return phase0 + TWOPI * np.cumsum(freq) / sr


def additive(f, partials, n, sr=SR, rng=None, attack=0.002, freq_mul=None):
    """partials: [(ratio, amp, decay_tau_s), ...]. decay None -> sustained.
    freq_mul: optional array (n,) for vibrato / glide multiplying all partials."""
    t = tvec(n, sr)
    base = f * (freq_mul if freq_mul is not None else 1.0)
    ph = phase_from_freq(np.broadcast_to(base, (n,)), sr)
    y = np.zeros(n)
    for r, a, d in partials:
        if f * r >= sr * 0.45:
            continue
        p0 = rng.uniform(0, TWOPI) if rng is not None else 0.0
        e = np.exp(-t / d) if d else 1.0
        y += a * np.sin(r * ph + p0) * e
    if attack:
        y *= np.clip(t / attack, 0, 1)
    return y


def blit_saw(f, n, sr=SR, freq_mul=None, max_h=None, rolloff=None):
    """band-limited saw by additive synthesis (alias-free). f scalar."""
    nh = int(min(max_h or 200, (sr * 0.45) // max(f, 1.0)))
    ph = phase_from_freq(np.broadcast_to(f * (freq_mul if freq_mul is not None else 1.0), (n,)), sr)
    y = np.zeros(n)
    for k in range(1, nh + 1):
        a = 1.0 / k
        if rolloff:
            a /= np.sqrt(1 + (k * f / rolloff) ** 4)
        y += a * np.sin(k * ph)
    return y * (2 / np.pi)


def blit_square(f, n, sr=SR, freq_mul=None, max_h=None, rolloff=None):
    nh = int(min(max_h or 200, (sr * 0.45) // max(f, 1.0)))
    ph = phase_from_freq(np.broadcast_to(f * (freq_mul if freq_mul is not None else 1.0), (n,)), sr)
    y = np.zeros(n)
    for k in range(1, nh + 1, 2):
        a = 1.0 / k
        if rolloff:
            a /= np.sqrt(1 + (k * f / rolloff) ** 4)
        y += a * np.sin(k * ph)
    return y * (4 / np.pi)


def vibrato(n, rate=5.0, depth_cents=12.0, delay=0.25, sr=SR, rng=None):
    """returns a frequency multiplier array with delayed-onset vibrato."""
    t = tvec(n, sr)
    p0 = rng.uniform(0, TWOPI) if rng is not None else 0.0
    ramp = np.clip((t - delay) / 0.35, 0, 1)
    return 2 ** (depth_cents * ramp * np.sin(TWOPI * rate * t + p0) / 1200.0)


def ks_string(f, n, rng, decay_s=2.0, bright=0.5, pick_pos=0.2, sr=SR, excitation=None):
    """Karplus–Strong / simple digital waveguide string.
    decay_s: approximate T60 of the fundamental. bright 0..1 excitation colour.
    pick_pos: comb (pluck position) 0..0.5."""
    L = sr / f
    Li = int(L)
    frac = L - Li
    if excitation is None:
        exc = rng.standard_normal(Li + 2)
        exc = lp(exc, 400 + 9000 * bright, sr=sr)
    else:
        exc = excitation
    # pluck-position comb
    d = max(1, int(pick_pos * Li))
    exc = exc.copy()
    exc[d:] -= exc[:-d] * 0.9
    x = np.zeros(n)
    x[:min(n, len(exc))] = exc[:n]
    # per-period loss g so that amplitude falls 60 dB in decay_s
    g = 10 ** (-3.0 / (decay_s * f))
    g = min(g, 0.99995)
    a = np.zeros(Li + 3)
    a[0] = 1
    # two-tap average (loss filter) with fractional delay split
    a[Li] -= g * 0.5 * (1 - frac)
    a[Li + 1] -= g * 0.5
    a[Li + 2] -= g * 0.5 * frac
    y = signal.lfilter([1.0], a, x)
    return y


def modal(f, modes, n, rng, sr=SR, strike=None):
    """bank of decaying sinusoids: modes [(ratio, amp, tau_s), ...] + optional
    strike noise burst (array)."""
    y = additive(f, modes, n, sr=sr, rng=rng, attack=0.0008)
    if strike is not None:
        k = min(n, len(strike))
        y[:k] += strike[:k]
    return y


def stereo(mono, pan=0.0, width=0.0, rng=None, sr=SR):
    """equal-power pan; width>0 adds a short decorrelating delay on one side."""
    a = (pan + 1) * np.pi / 4
    L = mono * np.cos(a)
    R = mono * np.sin(a)
    if width > 0:
        d = int(0.0007 * width * sr) + 1
        R = np.concatenate([np.zeros(d), R[:-d]]) * (1 - 0.15 * width) + R * 0.15 * width
    return np.stack([L, R])


# --------------------------------------------------------------------------- registry
@dataclass
class Instrument:
    name: str
    fn: Callable
    family: str
    lo: int
    hi: int
    pitched: bool = True
    tags: tuple = ()
    desc: str = ''
    params: dict = field(default_factory=dict)
    sustain: bool = False         # True if dur (gate) matters
    ref_dur: float = 1.0          # gate used for calibration


REGISTRY: dict[str, Instrument] = {}
_CAL_PATH = os.path.join(os.path.dirname(__file__), 'calibration.json')
_CAL: dict[str, float] = {}
if os.path.exists(_CAL_PATH):
    try:
        _CAL = json.load(open(_CAL_PATH))
    except Exception:
        _CAL = {}

TARGET_LOUD = 0.12   # RMS of the loudest 250 ms window of a reference note


def register(name, fn, family, lo='C2', hi='C6', pitched=True, tags=(), desc='',
             sustain=False, ref_dur=1.0, **params):
    """Register a preset. name: 'family.preset' (lowercase, underscores)."""
    lo = nm(lo) if isinstance(lo, str) else int(lo)
    hi = nm(hi) if isinstance(hi, str) else int(hi)
    if name in REGISTRY:
        raise ValueError(f'duplicate instrument {name}')
    REGISTRY[name] = Instrument(name, fn, family, lo, hi, pitched, tuple(tags), desc, params, sustain, ref_dur)
    return REGISTRY[name]


def _raw(inst, pitch, dur, vel, seed, sr):
    rng = np.random.default_rng(seed)
    f = float(mtof(pitch))
    y = inst.fn(f, float(dur), float(vel), rng, sr, **inst.params)
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(y)):
        raise FloatingPointError(f'{inst.name}: non-finite output')
    return y


def loudness(y, sr=SR, win=0.25):
    m = y if y.ndim == 1 else 0.5 * (y[0] + y[1])
    k = max(1, int(win * sr))
    if len(m) <= k:
        return float(np.sqrt(np.mean(m ** 2)) + 1e-12)
    c = np.cumsum(np.concatenate([[0.0], m ** 2]))
    w = (c[k:] - c[:-k]) / k
    return float(np.sqrt(w.max()) + 1e-12)


def calibrate(name, force=False, sr=SR):
    if name in _CAL and not force:
        return _CAL[name]
    inst = REGISTRY[name]
    mid = (inst.lo + inst.hi) / 2 if inst.pitched else 60
    y = _raw(inst, round(mid), inst.ref_dur, 0.8, 12345, sr)
    g = TARGET_LOUD / loudness(y, sr)
    _CAL[name] = float(g)
    return _CAL[name]


def save_calibration():
    json.dump({k: round(v, 6) for k, v in sorted(_CAL.items())}, open(_CAL_PATH, 'w'), indent=0)


def play(name, pitch=60, dur=0.5, vel=0.7, seed=0, sr=SR):
    """Render one note of a preset. pitch: MIDI number (float ok) or note name.
    Returns float32 mono (n,) or stereo (2,n), calibrated."""
    inst = REGISTRY[name]
    if isinstance(pitch, str):
        pitch = nm(pitch)
    y = _raw(inst, pitch, dur, vel, seed, sr)
    y = y - (y.mean(axis=-1, keepdims=True) if y.shape[-1] > sr // 10 else 0.0)
    y = fade(y, fin=0.0, fout=0.005, sr=sr)
    return (y * calibrate(name, sr=sr)).astype(np.float32)


def names(tag=None, family=None):
    return sorted(k for k, v in REGISTRY.items()
                  if (tag is None or tag in v.tags) and (family is None or v.family == family))


def resolve(name, family=None):
    """exact preset name, else the first preset whose name contains `name`
    (so a world spec can say "kalimba" or "frog"). Raises KeyError."""
    if name in REGISTRY:
        return name
    hits = sorted(k for k in REGISTRY if name in k and (family is None or REGISTRY[k].family == family))
    if not hits:
        raise KeyError(f'no instrument matches {name!r}')
    return hits[0]
