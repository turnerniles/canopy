"""
Shared DSP helpers for the wind / voice / creature / synth families
(inst_winds, inst_voices, inst_creatures, inst_synths).

Everything is vectorised: recursive filters go through scipy.signal, and
time-varying filters are processed in blocks with carried state.

Key pieces
    dsf()          Moorer discrete-summation oscillator: band-limited harmonic
                   series with amplitude a**k in O(1) per sample (a may vary per
                   sample -> brightness that follows blowing pressure).
    contour()      pitch track: scoop, delayed/developing vibrato, drift, jitter,
                   end fall.
    gate()         sustain envelope that releases after the gate and reaches
                   silence.
    breath()       band-limited breath noise that is partly pitch-synchronous.
    cascade()      Klatt-style cascade formant filter (unity DC gain -> keeps the
                   fundamental), static or block-time-varying.
    tv_sos()       generic block time-varying SOS filter.
    VOWELS         formant tables (Csound / Peterson-Barney style).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .core import SR, TWOPI, tvec, phase_from_freq, lp, hp, bp, reson, onepole_lp

# --------------------------------------------------------------------------- oscillators


def dsf(freq, a, n=None, odd=False, sr=SR, phase0=0.0, fmax=None, ph=None, normalise=True):
    """Band-limited harmonic series  sum_k a**k sin((k+1) phi)  (or odd harmonics
    only: sum_k a**k sin((2k+1) phi)).
    freq: scalar or (n,) instantaneous frequency; a: scalar or (n,) in [0, 0.985].
    The number of harmonics is fixed from max(freq) so the result stays below
    fmax (default 0.45 sr)."""
    if ph is None:
        freq = np.broadcast_to(np.asarray(freq, dtype=float), (n,)) if np.ndim(freq) == 0 else np.asarray(freq, float)
        ph = phase_from_freq(freq, sr, phase0)
        fm = float(np.max(freq))
    else:
        fm = float(np.max(freq))
    lim = min(fmax or sr * 0.45, sr * 0.45)
    step = 2 if odd else 1
    N = max(1, int((lim / max(fm, 1.0) - 1) // step + 1))
    a = np.clip(np.asarray(a, dtype=float), 0.0, 0.985)
    be = step * ph
    aN = a ** N
    s1 = np.sin(ph)
    # sin(ph - be) is 0 (all harmonics) or -sin(ph) (odd harmonics)
    num = s1 * (1.0 + a) if odd else s1.copy()
    if float(np.max(aN)) > 1e-4:          # truncation terms only matter if audible
        num -= aN * (np.sin(ph + N * be) - a * np.sin(ph + (N - 1) * be))
    den = 1.0 + a * a - 2.0 * a * np.cos(be)
    y = num / den
    if normalise:
        y *= np.sqrt((1 - a * a) / np.maximum(1 - a ** (2 * N), 1e-9))
    return y


def roll_a(f, fr):
    """per-harmonic ratio so that amplitude falls by 1/e every fr Hz."""
    return np.clip(np.exp(-np.asarray(f, float) / np.maximum(fr, 1.0)), 0.0, 0.985)


def saw_from_dsf(freq, n, fr=8000.0, sr=SR, phase0=0.0):
    """saw-like (≈1/k) spectrum: flat DSF through a one-pole tilt."""
    f0 = float(np.mean(freq))
    y = dsf(freq, roll_a(f0, fr), n=n, sr=sr, phase0=phase0)
    return onepole_lp(y, max(f0 * 0.9, 20.0), sr) * 3.0


def polyblep_saw(freq, n, sr=SR, phase0=0.0):
    """cheap anti-aliased saw for synth leads (vectorised PolyBLEP)."""
    freq = np.broadcast_to(np.asarray(freq, float), (n,))
    dt = freq / sr
    p = (phase0 + np.cumsum(dt)) % 1.0
    y = 2 * p - 1
    m1 = p < dt
    t1 = p[m1] / dt[m1]
    y[m1] -= t1 + t1 - t1 * t1 - 1
    m2 = p > 1 - dt
    t2 = (p[m2] - 1) / dt[m2]
    y[m2] -= t2 * t2 + t2 + t2 + 1
    return y


def polyblep_pulse(freq, n, width=0.5, sr=SR, phase0=0.0):
    freq = np.broadcast_to(np.asarray(freq, float), (n,))
    w = np.broadcast_to(np.asarray(width, float), (n,))
    dt = freq / sr
    p = (phase0 + np.cumsum(dt)) % 1.0
    y = np.where(p < w, 1.0, -1.0)

    def blep(t, d):
        out = np.zeros_like(t)
        m1 = t < d
        x = t[m1] / d[m1]
        out[m1] = x + x - x * x - 1
        m2 = t > 1 - d
        x = (t[m2] - 1) / d[m2]
        out[m2] = x * x + x + x + 1
        return out
    y += blep(p, dt)
    y -= blep((p - w) % 1.0, dt)
    return y


# --------------------------------------------------------------------------- control signals


def smooth_noise(n, rate, rng, sr=SR):
    """band-limited random wander, unit-ish std, roughly `rate` Hz bandwidth."""
    k = int(n / sr * rate) + 4
    pts = rng.standard_normal(k)
    x = np.interp(np.arange(n) * (rate / sr), np.arange(k), pts)
    return onepole_lp(x, rate * 1.2, sr) * 1.3


def contour(n, f, dur, rng, sr=SR, scoop=0.0, scoop_t=0.04, vib_rate=5.0, vib_cents=0.0,
            vib_delay=0.3, vib_grow=0.5, vib_wobble=0.06, drift=3.0, drift_rate=1.5,
            jitter=0.0, fall=0.0, fall_t=0.12, fall_at=None, bend=None, cents_out=False):
    """Frequency array (n,) in Hz (computed at a 3 kHz control rate, then
    interpolated).
    scoop:  cents offset at onset decaying with scoop_t (negative = from below)
    vib_*:  vibrato that starts after vib_delay and grows over vib_grow seconds;
            rate wanders by vib_wobble
    drift:  slow random pitch wander (cents)
    jitter: fast random (cents) — voices / rough reeds
    fall:   cents reached after the gate ends (negative = drop off)
    bend:   optional (n,) extra cents (full rate)"""
    D = 16
    crs = sr / D
    m = n // D + 2
    t = np.arange(m) / crs
    c = np.zeros(m)
    if scoop:
        c += scoop * np.exp(-t / max(scoop_t, 1e-4))
    if vib_cents:
        ramp = np.clip((t - vib_delay) / max(vib_grow, 1e-3), 0, 1)
        ramp = ramp * ramp * (3 - 2 * ramp)
        rate = vib_rate * (1 + vib_wobble * smooth_noise(m, 0.8, rng, crs))
        vp = TWOPI * np.cumsum(rate) / crs + rng.uniform(0, TWOPI)
        c += vib_cents * ramp * np.sin(vp)
    if drift:
        c += drift * smooth_noise(m, drift_rate, rng, crs)
    if jitter:
        c += jitter * smooth_noise(m, 35.0, rng, crs)
    if fall:
        t0 = dur if fall_at is None else fall_at
        tt = np.maximum(t - t0, 0)
        c += fall * (1 - np.exp(-tt / max(fall_t, 1e-4)))
    cf = np.interp(np.arange(n) / D, np.arange(m), c)
    if bend is not None:
        cf += bend
    if cents_out:
        return cf
    return f * np.exp2(cf / 1200.0)


def gate(n, dur, att=0.02, rel=0.15, sr=SR, dec=0.0, sus=1.0, shape='cos', swell=0.0):
    """Sustain envelope. Rises over `att` (raised-cosine or exp), optionally
    decays towards `sus` with time-constant `dec`, holds until `dur`, then
    releases exponentially with time-constant `rel` (reaches -60 dB in ~7 rel).
    swell>0 adds a slow crescendo over the held part."""
    t = tvec(n, sr)
    if shape == 'cos':
        a = np.clip(t / max(att, 1e-4), 0, 1)
        e = 0.5 - 0.5 * np.cos(np.pi * a)
    else:
        e = 1 - np.exp(-t / max(att / 3, 1e-4))
    if dec > 0:
        e = e * (sus + (1 - sus) * np.exp(-np.maximum(t - att, 0) / dec))
    if swell:
        e = e * (1 + swell * np.clip((t - att) / max(dur, 1e-3), 0, 1))
    k = int(min(n, max(1, dur * sr)))
    if k < n:
        e[k:] = e[k - 1] * np.exp(-(t[k:] - t[k - 1]) / max(rel, 1e-4))
    return e


def note_len(dur, rel, sr=SR, extra=0.0, cap=12.0):
    """samples needed for gate + release tail to reach about -60 dB."""
    return int(min(cap, dur + 7.0 * rel + extra) * sr) + 1


def env_points(n, pts, sr=SR):
    """piecewise-linear envelope from [(t, value), ...]."""
    ts = np.array([p[0] for p in pts], float)
    vs = np.array([p[1] for p in pts], float)
    return np.interp(tvec(n, sr), ts, vs)


# --------------------------------------------------------------------------- noise


def breath(n, rng, f, lo, hi, sr=SR, ph=None, sync=0.5, tonal=0.0, q=8.0):
    """breath noise: band-limited (lo..hi), amplitude-modulated by the pitch
    phase (sync, pitch-synchronous turbulence) plus `tonal` resonant noise at
    f and 2f (airy, pitched hiss of an edge-tone). Unit-ish RMS."""
    w = rng.standard_normal(n)
    x = bp(w, lo, hi, sr=sr)
    x /= np.std(x) + 1e-12
    if ph is not None and sync:
        x *= (1 - sync) + sync * (0.5 + 0.5 * np.cos(ph)) * 2
    if tonal:
        r = reson(w, f, q, sr) + 0.5 * reson(w, min(2 * f, sr * 0.45), q, sr)
        x += tonal * r / (np.std(r) + 1e-12)
    return x


def burst(n, rng, lo, hi, att=0.001, tau=0.01, sr=SR):
    from .core import env_perc
    x = bp(rng.standard_normal(n), lo, hi, sr=sr)
    return x / (np.std(x) + 1e-12) * env_perc(n, att, tau, sr)


# --------------------------------------------------------------------------- filters


def _res_coefs(F, BW, sr):
    F = np.clip(F, 20.0, sr * 0.47)
    r = np.exp(-np.pi * np.maximum(BW, 5.0) / sr)
    a1 = -2 * r * np.cos(TWOPI * F / sr)
    a2 = r * r
    return a1, a2


def cascade(x, formants, sr=SR):
    """static cascade of unity-DC-gain resonators: formants [(F, BW), ...]"""
    sos = []
    for F, BW in formants:
        if F >= sr * 0.46:
            continue
        a1, a2 = _res_coefs(F, BW, sr)
        sos.append([1 + a1 + a2, 0, 0, 1, a1, a2])
    if not sos:
        return x
    return signal.sosfilt(np.array(sos), x)


def tv_sos(x, sos_blocks, block):
    """x filtered by per-block SOS matrices sos_blocks (nb, S, 6) with state carry."""
    n = len(x)
    nb, S, _ = sos_blocks.shape
    y = np.empty(n)
    zi = np.zeros((S, 2))
    for i in range(nb):
        s = i * block
        if s >= n:
            break
        e = min(n, s + block)
        y[s:e], zi = signal.sosfilt(sos_blocks[i], x[s:e], zi=zi)
    return y


def tv_cascade(x, F, BW, sr=SR, block=256):
    """time-varying cascade formants. F, BW: arrays (k, n) or (k,) per formant."""
    n = len(x)
    F = np.atleast_2d(np.asarray(F, float))
    BW = np.atleast_2d(np.asarray(BW, float))
    k = F.shape[0]
    nb = (n + block - 1) // block
    idx = np.minimum(np.arange(nb) * block + block // 2, n - 1)
    Fb = F[:, idx] if F.shape[1] == n else np.repeat(F[:, :1], nb, 1)
    Bb = BW[:, idx] if BW.shape[1] == n else np.repeat(BW[:, :1], nb, 1)
    a1, a2 = _res_coefs(Fb, Bb, sr)
    sos = np.zeros((nb, k, 6))
    sos[:, :, 0] = (1 + a1 + a2).T
    sos[:, :, 3] = 1
    sos[:, :, 4] = a1.T
    sos[:, :, 5] = a2.T
    return tv_sos(x, sos, block)


def tv_lowpass(x, fc, q=0.707, sr=SR, block=128):
    """resonant 2-pole low-pass (RBJ) with block-varying cutoff fc (scalar/array)."""
    n = len(x)
    fc = np.broadcast_to(np.asarray(fc, float), (n,))
    nb = (n + block - 1) // block
    idx = np.minimum(np.arange(nb) * block + block // 2, n - 1)
    f = np.clip(fc[idx], 20.0, sr * 0.45)
    q = np.broadcast_to(np.asarray(q, float), (n,))[idx]
    w0 = TWOPI * f / sr
    al = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    a0 = 1 + al
    sos = np.zeros((nb, 1, 6))
    sos[:, 0, 0] = (1 - cw) / 2 / a0
    sos[:, 0, 1] = (1 - cw) / a0
    sos[:, 0, 2] = (1 - cw) / 2 / a0
    sos[:, 0, 3] = 1
    sos[:, 0, 4] = -2 * cw / a0
    sos[:, 0, 5] = (1 - al) / a0
    return tv_sos(x, sos, block)


def tv_bandpass(x, fc, q, sr=SR, block=128):
    """constant-peak band-pass with block-varying centre."""
    n = len(x)
    fc = np.broadcast_to(np.asarray(fc, float), (n,))
    nb = (n + block - 1) // block
    idx = np.minimum(np.arange(nb) * block + block // 2, n - 1)
    f = np.clip(fc[idx], 20.0, sr * 0.45)
    q = np.broadcast_to(np.asarray(q, float), (n,))[idx]
    w0 = TWOPI * f / sr
    al = np.sin(w0) / (2 * q)
    a0 = 1 + al
    sos = np.zeros((nb, 1, 6))
    sos[:, 0, 0] = al / a0
    sos[:, 0, 2] = -al / a0
    sos[:, 0, 3] = 1
    sos[:, 0, 4] = -2 * np.cos(w0) / a0
    sos[:, 0, 5] = (1 - al) / a0
    return tv_sos(x, sos, block)


def ladder(x, fc, res=0.3, sr=SR, block=128, drive=1.0):
    """Moog-ish 4-pole low-pass: two cascaded resonant 2-poles with tanh drive."""
    y = tv_lowpass(np.tanh(drive * x), fc, q=0.55 + 2.5 * res, sr=sr, block=block)
    return tv_lowpass(y, fc, q=0.6, sr=sr, block=block)


def peaks(x, formants, sr=SR):
    """parallel constant-peak resonators [(F, Q, gain), ...] (body colour)."""
    y = np.zeros_like(x)
    for F, Q, g in formants:
        if F < sr * 0.45:
            y += g * reson(x, F, Q, sr)
    return y


def var_delay(x, delay_s, sr=SR):
    """fractional time-varying delay (vectorised linear interpolation)."""
    n = len(x)
    idx = np.arange(n) - np.broadcast_to(np.asarray(delay_s, float), (n,)) * sr
    return np.interp(idx, np.arange(n), x, left=0.0, right=0.0)


def comb_tint(x, f, fb=0.8, sr=SR):
    """feedback comb at period 1/f (pitched noise / resonant tube)."""
    d = max(2, int(round(sr / f)))
    a = np.zeros(d + 1)
    a[0] = 1
    a[d] = -fb
    return signal.lfilter([1 - fb], a, x)


# --------------------------------------------------------------------------- vowels

# (F, BW, gain dB) x 5 per vowel per voice type
VOWELS = {
    'soprano': {
        'a': [(800, 80, 0), (1150, 90, -6), (2900, 120, -32), (3900, 130, -20), (4950, 140, -50)],
        'e': [(350, 60, 0), (2000, 100, -20), (2800, 120, -15), (3600, 150, -40), (4950, 200, -56)],
        'i': [(270, 60, 0), (2140, 90, -12), (2950, 100, -26), (3900, 120, -26), (4950, 120, -44)],
        'o': [(450, 70, 0), (800, 80, -11), (2830, 100, -22), (3800, 130, -22), (4950, 135, -50)],
        'u': [(325, 50, 0), (700, 60, -16), (2700, 170, -35), (3800, 180, -40), (4950, 200, -60)],
    },
    'alto': {
        'a': [(800, 80, 0), (1150, 90, -4), (2800, 120, -20), (3500, 130, -36), (4950, 140, -60)],
        'e': [(400, 60, 0), (1600, 80, -24), (2700, 120, -30), (3300, 150, -35), (4950, 200, -60)],
        'i': [(350, 50, 0), (1700, 100, -20), (2700, 120, -30), (3700, 150, -36), (4950, 200, -60)],
        'o': [(450, 70, 0), (800, 80, -9), (2830, 100, -16), (3500, 130, -28), (4950, 135, -55)],
        'u': [(325, 50, 0), (700, 60, -12), (2530, 170, -30), (3500, 180, -40), (4950, 200, -64)],
    },
    'tenor': {
        'a': [(650, 80, 0), (1080, 90, -6), (2650, 120, -7), (2900, 130, -8), (3250, 140, -22)],
        'e': [(400, 70, 0), (1700, 80, -14), (2600, 100, -12), (3200, 120, -14), (3580, 120, -20)],
        'i': [(290, 40, 0), (1870, 90, -15), (2800, 100, -18), (3250, 120, -20), (3540, 120, -30)],
        'o': [(400, 40, 0), (800, 80, -10), (2600, 100, -12), (2800, 120, -12), (3000, 120, -26)],
        'u': [(350, 40, 0), (600, 60, -20), (2700, 100, -17), (2900, 120, -14), (3300, 120, -26)],
    },
    'bass': {
        'a': [(600, 60, 0), (1040, 70, -7), (2250, 110, -9), (2450, 120, -9), (2750, 130, -20)],
        'e': [(400, 40, 0), (1620, 80, -12), (2400, 100, -9), (2800, 120, -12), (3100, 120, -18)],
        'i': [(250, 60, 0), (1750, 90, -30), (2600, 100, -16), (3050, 120, -22), (3340, 120, -28)],
        'o': [(400, 40, 0), (750, 80, -11), (2400, 100, -21), (2600, 120, -20), (2900, 120, -40)],
        'u': [(350, 40, 0), (600, 80, -20), (2400, 100, -32), (2675, 120, -28), (2950, 120, -36)],
    },
}


def voice_type(f):
    if f < 180:
        return 'bass'
    if f < 300:
        return 'tenor'
    if f < 480:
        return 'alto'
    return 'soprano'


def vowel(name, f=None, vtype=None, scale=1.0, bw_mul=1.0):
    """[(F, BW), ...] for a vowel, chosen by voice type (from f) and optionally
    scaled (children / small creatures: scale>1)."""
    vt = vtype or voice_type(f or 220)
    tab = VOWELS[vt][name]
    out = []
    for F, BW, _g in tab:
        F = F * scale
        # a formant below the fundamental is pulled up (singers 'tune' F1 to f0)
        if f is not None and F < f * 1.05:
            F = f * 1.1
        if out and F < out[-1][0] * 1.22:      # keep formants apart (no stacked peaks)
            F = out[-1][0] * 1.22
        out.append((F, BW * bw_mul))
    return out


def vowel_tracks(seq, n, f=None, vtype=None, scale=1.0, bw_mul=1.0, sr=SR):
    """seq: [(t, vowel_name), ...] -> F (5, n), BW (5, n) piecewise-linear morph."""
    ts = [p[0] for p in seq]
    Fs = np.array([[F for F, _ in vowel(v, f, vtype, scale, bw_mul)] for _, v in seq])
    Bs = np.array([[B for _, B in vowel(v, f, vtype, scale, bw_mul)] for _, v in seq])
    t = tvec(n, sr)
    F = np.stack([np.interp(t, ts, Fs[:, i]) for i in range(5)])
    B = np.stack([np.interp(t, ts, Bs[:, i]) for i in range(5)])
    return F, B


def pan_stereo(mono, pan=0.0):
    a = (pan + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)])


def mix_stereo(parts):
    """parts: list of (mono, pan) of possibly different lengths."""
    n = max(len(p[0]) for p in parts)
    out = np.zeros((2, n))
    for m, pan in parts:
        s = pan_stereo(m, pan)
        out[:, :len(m)] += s
    return out


def shift(x, d, n=None):
    """delay x by d samples (zero padded) into length n."""
    n = n or len(x) + d
    y = np.zeros(n)
    k = min(len(x), n - d)
    if k > 0:
        y[d:d + k] = x[:k]
    return y


def tame(y, drive=1.2, pct=99.7):
    """soft-limit the crest factor (tanh around the 99.7th-percentile level);
    keeps calibrated presets from clipping on rare peaks."""
    pk = np.percentile(np.abs(y), pct) + 1e-12
    return np.tanh(drive * y / pk) * pk / drive
