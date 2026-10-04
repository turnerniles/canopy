"""
DSP helpers for the string and keyboard families (inst_strings, inst_keys).

    waveguide()   tuned digital-waveguide string: one-pole frequency-dependent
                  loss (T60 at f0 and at a high frequency), exact Thiran
                  fractional delay, optional allpass dispersion (stiffness).
                  Short loops run as one lfilter; long loops are run period by
                  period with carried filter state (vectorised per period).
    pluck_exc()   plucking / striking excitation (pulse width, nail noise,
                  pluck-position comb).
    body_ir()     cached synthetic body impulse responses (named low modes +
                  a dense statistical high-mode field with spectral envelope).
    wt_osc()      band-limited wavetable oscillator with arbitrary frequency
                  track (vibrato, glide) via np.interp on an FFT-built table.
    warp()        apply a frequency-multiplier track to an existing signal by
                  time warping (vibrato / scoop / glide on any rendered sound).
    human_pitch() delayed vibrato with drift, rate wander and jitter.
    overdrive(), cab(), peq(), compress(), leslie(), tape()
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import signal

from .core import SR, TWOPI, lp, hp, bp, onepole_lp, tvec

# ---------------------------------------------------------------- small utils


def _upsample(c, n, hop):
    """linear upsample of a control-rate curve (point j at sample j*hop)."""
    j = np.arange(n) / hop
    i = j.astype(np.int64)
    i = np.minimum(i, len(c) - 2)
    fr = j - i
    return c[i] + fr * (c[i + 1] - c[i])


def smooth_noise(n, rate, rng, sr=SR):
    """band-limited random curve (std ~1), changes at about `rate` Hz.
    Built at a control rate and linearly upsampled (cheap)."""
    hop = int(np.clip(sr / (rate * 12), 1, 256))
    m = n // hop + 3
    k = max(4, int(m * hop * rate / sr) + 4)
    pts = rng.standard_normal(k)
    c = np.interp(np.linspace(0, k - 3, m), np.arange(k), pts)
    c = onepole_lp(c, max(rate, 0.05), sr / hop)
    c /= np.std(c) + 1e-9
    return _upsample(c, n, hop) if hop > 1 else c[:n]


def cents(c):
    return 2.0 ** (np.asarray(c, dtype=float) / 1200.0)


def _cr_noise(m, rate, rng, cr):
    k = max(4, int(m * rate / cr) + 4)
    pts = rng.standard_normal(k)
    c = np.interp(np.linspace(0, k - 3, m), np.arange(k), pts)
    c = onepole_lp(c, max(rate, 0.05), cr)
    return c / (np.std(c) + 1e-9)


def human_pitch(n, rng, sr=SR, rate=5.5, depth=15.0, delay=0.3, rise=0.4, drift=3.0,
                jitter=0.0, scoop=0.0, scoop_t=0.05, wander=0.08, att_jit=0.0, att_tau=0.07,
                hop=16, return_cents=False):
    """frequency multiplier: scoop (cents, starting offset decaying with scoop_t),
    delayed vibrato whose rate/depth wander, slow drift, fast jitter and extra
    attack jitter. Built at control rate sr/hop and upsampled once."""
    cr = sr / hop
    m = n // hop + 3
    t = np.arange(m) / cr
    c = np.zeros(m)
    if depth:
        r = rate * (1 + 0.04 * rng.standard_normal()) * (1 + wander * _cr_noise(m, 0.7, rng, cr))
        ph = TWOPI * np.cumsum(r) / cr + rng.uniform(0, TWOPI)
        ramp = np.clip((t - delay) / max(rise, 1e-3), 0, 1)
        ramp = ramp * ramp * (3 - 2 * ramp)
        dep = depth * (1 + 0.15 * _cr_noise(m, 0.9, rng, cr))
        c += dep * ramp * np.sin(ph)
    if drift:
        c += drift * _cr_noise(m, 0.6, rng, cr)
    if jitter or att_jit:
        jn = _cr_noise(m, 30.0, rng, cr)
        c += (jitter + att_jit * np.exp(-t / att_tau)) * jn
    if scoop:
        c += scoop * np.exp(-t / max(scoop_t, 1e-4))
    c = _upsample(c, n, hop)
    return c if return_cents else cents(c)


def warp(x, fm, n=None):
    """time-warp x so its pitch is multiplied by fm (array). Output length n."""
    n = len(fm) if n is None else n
    fm = np.broadcast_to(fm, (n,)) if np.ndim(fm) == 0 else fm[:n]
    tau = np.concatenate([[0.0], np.cumsum(fm[:-1])])
    return np.interp(tau, np.arange(len(x)), x, right=0.0)


def shifted(x, d, n=None):
    """delay x by d samples (int) into a length-n array."""
    n = len(x) if n is None else n
    y = np.zeros(n)
    d = int(max(0, d))
    if d < n:
        k = min(n - d, len(x))
        y[d:d + k] = x[:k]
    return y


def gate_env(n, dur, att=0.01, rel=0.1, sr=SR, shape=1.0):
    """attack (raised cosine) / hold / exponential release starting at dur."""
    t = tvec(n, sr)
    a = np.clip(t / max(att, 1e-4), 0, 1)
    e = (0.5 - 0.5 * np.cos(np.pi * a)) ** shape
    k = int(dur * sr)
    if k < n:
        lvl = e[max(0, k - 1)]
        tt = t[k:] - t[k]
        e[k:] = lvl * np.exp(-tt / max(rel, 1e-4))
    return e


def tail_fade(y, frac=0.12, sr=SR, min_s=0.03):
    n = y.shape[-1]
    k = int(max(min_s * sr, frac * n))
    k = min(k, n)
    w = 0.5 + 0.5 * np.cos(np.pi * np.arange(k) / k)
    y = np.array(y, dtype=float, copy=True)
    y[..., -k:] *= w
    return y


def note_n(seconds, sr=SR, cap=12.0):
    return int(min(cap, max(0.05, seconds)) * sr)


# ---------------------------------------------------------------- filters

def peq(f0, gain_db, q=1.0, sr=SR):
    """RBJ peaking biquad as an sos row."""
    f0 = float(np.clip(f0, 10, sr * 0.45))
    A = 10 ** (gain_db / 40)
    w = TWOPI * f0 / sr
    al = np.sin(w) / (2 * q)
    b = np.array([1 + al * A, -2 * np.cos(w), 1 - al * A])
    a = np.array([1 + al / A, -2 * np.cos(w), 1 - al / A])
    return np.hstack([b / a[0], a / a[0]])


def shelf(f0, gain_db, high=True, sr=SR):
    A = 10 ** (gain_db / 40)
    w = TWOPI * float(np.clip(f0, 10, sr * 0.45)) / sr
    cw, sw = np.cos(w), np.sin(w)
    al = sw / 2 * np.sqrt(2)
    sA = 2 * np.sqrt(A) * al
    if high:
        b = [A * ((A + 1) + (A - 1) * cw + sA), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sA)]
        a = [(A + 1) - (A - 1) * cw + sA, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sA]
    else:
        b = [A * ((A + 1) - (A - 1) * cw + sA), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sA)]
        a = [(A + 1) + (A - 1) * cw + sA, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sA]
    b, a = np.array(b), np.array(a)
    return np.hstack([b / a[0], a / a[0]])


def eq(x, rows):
    if not rows:
        return x
    return signal.sosfilt(np.vstack(rows), x)


def lp_sos(fc, order=2, sr=SR):
    return signal.butter(order, float(np.clip(fc, 10, sr * 0.45)), 'low', fs=sr, output='sos')


def hp_sos(fc, order=2, sr=SR):
    return signal.butter(order, float(np.clip(fc, 5, sr * 0.45)), 'high', fs=sr, output='sos')


def res_lp(x, fc, q, sr=SR):
    """static resonant 2-pole lowpass (pickup / cabinet resonance)."""
    w0 = TWOPI * float(np.clip(fc, 20, sr * 0.45)) / sr
    al = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
    a = np.array([1 + al, -2 * cw, 1 - al])
    return signal.lfilter(b / a[0], a / a[0], x)


def _tv_biquad(x, fc, q, sr, block, kind):
    x = np.asarray(x, dtype=float)
    n = len(x)
    fc = np.broadcast_to(np.asarray(fc, dtype=float), (n,))
    nb = (n + block - 1) // block
    # block-mean cutoff, coefficients computed vectorised for all blocks
    pad = np.concatenate([fc, np.full(nb * block - n, fc[-1])]) if nb * block > n else fc
    f = np.clip(pad.reshape(nb, block).mean(axis=1), 20, sr * 0.45)
    w0 = TWOPI * f / sr
    al = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    a0 = 1 + al
    if kind == 'band':
        B = np.stack([al, np.zeros(nb), -al], 1) / a0[:, None]
    else:
        B = np.stack([(1 - cw) / 2, 1 - cw, (1 - cw) / 2], 1) / a0[:, None]
    A = np.stack([np.ones(nb), -2 * cw / a0, (1 - al) / a0], 1)
    y = np.empty(n)
    zi = np.zeros(2)
    lf = signal.lfilter
    for j in range(nb):
        s = j * block
        e = min(n, s + block)
        y[s:e], zi = lf(B[j], A[j], x[s:e], zi=zi)
    return y


def tv_bandpass(x, fc, q, sr=SR, block=160):
    """block time-varying constant-peak bandpass (RBJ), carried state."""
    return _tv_biquad(x, fc, q, sr, block, 'band')


def tv_lowpass(x, fc, q=0.707, sr=SR, block=160):
    return _tv_biquad(x, fc, q, sr, block, 'low')


# ---------------------------------------------------------------- waveguide string

def _ap_delay(c, w):
    return 1.0 - (2.0 / w) * np.arctan2(c * np.sin(w), 1.0 + c * np.cos(w))


def _lp1_delay(p, w):
    return np.arctan2(p * np.sin(w), 1.0 - p * np.cos(w)) / w


def _lp1_mag(p, w):
    return (1 - p) / np.sqrt(1 - 2 * p * np.cos(w) + p * p)


def loss_pole(f, t60, t60_hi, f_hi, sr=SR):
    w0 = TWOPI * f / sr
    wh = TWOPI * min(f_hi, 0.45 * sr) / sr
    if t60_hi >= t60 or wh <= w0:
        return 0.0
    G0 = 10 ** (-3.0 / (f * t60))
    Gh = 10 ** (-3.0 / (f * t60_hi))
    R2 = (Gh / G0) ** 2
    c0, c = np.cos(w0), np.cos(wh)
    A = 1 - R2
    B = -2 * (c0 - R2 * c)
    disc = max(B * B - 4 * A * A, 0.0)
    p = (-B - np.sqrt(disc)) / (2 * A)
    return float(np.clip(p, 0.0, 0.97))


def waveguide(f, n, exc, t60=2.0, t60_hi=0.4, f_hi=4000.0, disp=0.0, nap=0, sr=SR):
    """Digital-waveguide string driven by `exc` (any length). Returns (n,).
    t60: decay of the fundamental; t60_hi: decay at f_hi.
    disp (0..0.9) with nap allpass stages -> stretched (stiff) partials."""
    f = float(f)
    P = sr / f
    w0 = TWOPI * f / sr
    p = loss_pole(f, t60, t60_hi, f_hi, sr)
    G0 = 10 ** (-3.0 / (f * t60))
    g = min(G0 / _lp1_mag(p, w0), 0.99999)
    c = -float(disp)
    D = P - _lp1_delay(p, w0) - (nap * _ap_delay(c, w0) if nap else 0.0)
    Li = int(np.floor(D - 0.5))
    Li = max(Li, 2)
    d = D - Li
    th = w0 * (1 - d) / 2
    eta = np.sin(th) / np.sin(w0 - th)
    B = np.array([eta, 1.0]) * g * (1 - p)
    A = np.convolve([1.0, -p], [1.0, eta])
    for _ in range(nap):
        B = np.convolve(B, [c, 1.0])
        A = np.convolve(A, [1.0, c])
    x = np.zeros(n)
    k = min(n, len(exc))
    x[:k] = exc[:k]
    if Li < 160:
        a = np.zeros(Li + len(B))
        a[:len(A)] += A
        a[Li:Li + len(B)] -= B
        return signal.lfilter(A, a, x)
    # long loop: process one period at a time (the delayed signal is already known)
    y = np.zeros(n)
    zi = np.zeros(max(len(A), len(B)) - 1)
    y[:Li] = x[:Li]
    for s in range(Li, n, Li):
        e = min(n, s + Li)
        w, zi = signal.lfilter(B, A, y[s - Li:e - Li], zi=zi)
        y[s:e] = x[s:e] + w
    return y


def pluck_exc(f, rng, vel, width_ms=1.0, noise=0.25, noise_lp=6000.0, pos=0.2, sr=SR,
              shape='cos'):
    """excitation for a plucked/struck string. width_ms: contact width (finger
    ~1.5-3 ms, nail/pick ~0.2-0.6 ms, hammer ~0.5-2). The pulse has constant
    area (so low harmonics don't depend on width); a narrower pulse only adds
    highs. noise: nail / pick scrape, high-passed so it colours the top without
    randomising the low harmonics. pos: pluck position (0..0.5) comb."""
    P = sr / f
    w = max(2, int(width_ms * 1e-3 * sr))
    m = int(P * 1.2) + w + 8
    e = np.zeros(m)
    if shape == 'cos':
        e[:w] = 0.5 - 0.5 * np.cos(TWOPI * np.arange(w) / w)
    else:  # half sine (hammer)
        e[:w] = np.sin(np.pi * np.arange(w) / w)
    e[:w] *= 1.0 / (np.sum(e[:w]) + 1e-12)
    if noise > 0:
        k = int(min(m, max(w * 2, P)))
        nz = rng.standard_normal(k) * np.exp(-np.arange(k) / (0.3 * k + 1))
        nz = lp(hp(nz, max(1500.0, 4 * f), 2, sr), noise_lp, sr=sr)
        nz *= noise / (np.sqrt(np.sum(nz * nz)) + 1e-12)
        e[:k] += nz
    # contact softness: the whole excitation is low-passed (cutoff rises with force)
    e = signal.sosfilt(lp_sos(noise_lp, 2, sr), e)
    if pos > 0:
        dd = pos * P
        di = int(dd)
        fr = dd - di
        ee = e.copy()
        if di + 1 < m:
            e[di:] -= ee[:m - di] * (1 - fr) * 0.97
            e[di + 1:] -= ee[:m - di - 1] * fr * 0.97
    return e


# ---------------------------------------------------------------- bodies

BODIES = {
    # name: (named modes [(f, tau, gain)], dense (count, lo, hi, q_lo, q_hi, tilt_db_per_oct, hill(f, db, oct_width)), direct)
    'classical': ([(98, .055, 1.0), (196, .05, .9), (240, .035, .45), (395, .03, .6), (560, .02, .35), (780, .018, .25)],
                  (50, 500, 6000, 12, 35, -4.5, (2500, 2, 1.5)), 0.25),
    'steel': ([(102, .06, .9), (175, .04, .55), (215, .045, .8), (380, .03, .55), (520, .02, .45), (900, .015, .35)],
              (60, 500, 8000, 14, 40, -3.0, (3200, 4, 1.4)), 0.35),
    'small': ([(255, .035, 1.0), (420, .03, .7), (640, .02, .5), (980, .015, .35)],
              (40, 700, 8000, 12, 30, -3.0, (3000, 3, 1.5)), 0.3),
    'mandolin': ([(190, .035, .6), (390, .03, .9), (620, .025, .6), (1050, .015, .45), (1500, .012, .3)],
                 (50, 800, 9000, 14, 35, -2.0, (3500, 5, 1.2)), 0.35),
    'banjo': ([(340, .03, 1.0), (542, .025, .8), (726, .02, .7), (781, .02, .5), (902, .018, .5), (992, .015, .4),
               (1073, .015, .35), (1190, .012, .3)],
              (60, 1200, 10000, 10, 25, -1.0, (4000, 6, 1.4)), 0.4),
    'kora': ([(115, .05, .9), (250, .04, .8), (395, .03, .6), (560, .025, .5), (800, .02, .4)],
             (45, 700, 8000, 12, 30, -2.5, (3500, 4, 1.3)), 0.3),
    'harp': ([(150, .06, .8), (290, .05, .9), (450, .04, .6), (680, .03, .45), (930, .02, .35)],
             (55, 600, 7000, 15, 40, -4.0, (2200, 2, 1.5)), 0.3),
    'koto': ([(140, .05, .7), (280, .045, .8), (415, .035, .7), (640, .025, .5), (910, .02, .45), (1350, .015, .35)],
             (55, 700, 8000, 15, 40, -2.5, (2800, 5, 1.2)), 0.35),
    'sitar': ([(120, .05, .8), (260, .04, .8), (520, .03, .6), (830, .02, .45)],
              (50, 700, 9000, 15, 40, -2.0, (3000, 4, 1.4)), 0.35),
    'oud': ([(92, .07, 1.0), (185, .05, .85), (290, .04, .7), (420, .03, .5), (610, .02, .35)],
            (55, 500, 5000, 12, 30, -6.0, (1600, 3, 1.2)), 0.25),
    'dulcimer': ([(230, .06, .8), (460, .05, .7), (700, .04, .6), (950, .03, .5), (1300, .02, .4)],
                 (60, 800, 9000, 18, 45, -2.0, (3500, 3, 1.4)), 0.35),
    'bouzouki': ([(118, .05, .7), (240, .045, .85), (420, .035, .6), (650, .025, .5), (1000, .018, .4)],
                 (55, 700, 9000, 14, 38, -2.0, (3200, 4, 1.3)), 0.35),
    'violin': ([(275, .02, 1.0), (460, .012, .6), (530, .018, .9), (610, .015, .7), (790, .01, .5)],
               (70, 700, 7500, 18, 45, -2.0, (2600, 7, 1.0)), 0.15),
    'viola': ([(225, .022, 1.0), (380, .014, .6), (440, .02, .9), (510, .016, .7), (660, .012, .5)],
              (70, 550, 6500, 18, 45, -3.0, (2100, 6, 1.0)), 0.15),
    'cello': ([(105, .035, 1.0), (175, .025, .7), (205, .03, .9), (300, .02, .7), (420, .015, .5)],
              (70, 400, 5500, 18, 45, -3.5, (1600, 5, 1.1)), 0.15),
    'contrabass': ([(60, .05, 1.0), (98, .035, .8), (125, .04, .9), (175, .03, .6), (260, .02, .45)],
                   (60, 300, 4500, 16, 40, -4.0, (1100, 4, 1.2)), 0.2),
    'erhu': ([(440, .02, .8), (700, .02, .7), (1050, .02, 1.0), (1500, .015, .7)],
             (60, 600, 7000, 14, 35, -1.5, (1300, 8, 1.0)), 0.2),
    'double_speed': ([(200, .03, .9), (360, .025, .55), (430, .03, .8), (760, .02, .55), (1040, .015, .45), (1800, .012, .35)],
                     (60, 1000, 10000, 14, 40, -2.0, (5000, 4, 1.2)), 0.35),
    'piano': ([(85, .04, .6), (160, .04, .7), (250, .03, .6), (370, .03, .5), (520, .02, .45)],
              (90, 400, 9000, 10, 30, -2.5, (2500, 2, 1.5)), 0.6),
    'upright': ([(120, .04, .7), (210, .04, .8), (330, .03, .7), (480, .025, .6), (620, .02, .5)],
                (80, 450, 7000, 12, 30, -4.0, (1200, 4, 1.0)), 0.5),
    'harpsichord': ([(110, .04, .6), (220, .03, .6), (330, .03, .5), (520, .02, .5)],
                    (90, 500, 10000, 15, 40, -1.0, (3000, 4, 1.5)), 0.4),
    'reed_box': ([(170, .03, .6), (340, .025, .6), (900, .02, .5)],
                 (40, 500, 6000, 8, 20, -2.0, (1500, 4, 1.2)), 0.6),
}


def _seed_of(name):
    return sum((i + 1) * ord(ch) for i, ch in enumerate(name)) % (2 ** 31)


@lru_cache(maxsize=64)
def body_ir(name, scale=1.0, sr=SR, length=0.3):
    """synthetic body response (without the direct path). Every mode is a
    cosine-phase decay normalised so its peak gain equals its table gain
    relative to a unit direct path; at resonance it is in phase with the
    direct sound, so peaks reinforce and the direct path floors the valleys."""
    modes, dense, direct = BODIES[name]
    rng = np.random.default_rng(_seed_of(name))
    n = int(length * sr)
    t = tvec(n, sr)
    ir = np.zeros(n)
    for f, tau, g in modes:
        f = f * scale
        m = int(min(n, 8 * tau * sr))
        ir[:m] += 2.5 * g / (tau * sr / 2) * np.cos(TWOPI * f * t[:m] + rng.uniform(-0.15, 0.15)) * np.exp(-t[:m] / tau)
    cnt, lo, hi, qlo, qhi, tilt, hill = dense
    lo *= scale
    hi = min(hi * scale, 0.45 * sr)
    fs = np.exp(rng.uniform(np.log(lo), np.log(hi), cnt))
    for f in fs:
        q = rng.uniform(qlo, qhi)
        tau = q / (np.pi * f)
        gdb = tilt * np.log2(f / lo)
        if hill:
            hf, hdb, hw = hill
            gdb += hdb * np.exp(-0.5 * (np.log2(f / (hf * scale)) / (hw / 2)) ** 2)
        g = 10 ** (gdb / 20) * rng.uniform(0.3, 1.0) * 1.05
        m = int(min(n, 8 * tau * sr))
        ir[:m] += g / (tau * sr / 2) * np.cos(TWOPI * f * t[:m] + rng.uniform(0, TWOPI)) * np.exp(-t[:m] / tau)
    ir *= 0.5 + 0.5 * np.cos(np.pi * np.clip((t - 0.7 * length) / (0.3 * length), 0, 1))
    return ir


def apply_body(y, name, mix=0.8, direct=None, scale=1.0, sr=SR):
    ir = body_ir(name, float(scale), sr)
    d = (BODIES[name][2] + 0.45) if direct is None else direct
    wet = signal.oaconvolve(y, ir)[:len(y)]
    return d * y + mix * wet


# ---------------------------------------------------------------- wavetable oscillator

TABN = 4096


def wt_table(amps, phases=None, N=TABN):
    amps = np.asarray(amps, dtype=float)
    K = min(len(amps), N // 2 - 1)
    spec = np.zeros(N // 2 + 1, complex)
    ph = np.zeros(K) if phases is None else np.asarray(phases)[:K]
    spec[1:K + 1] = amps[:K] * np.exp(1j * (ph - np.pi / 2))
    tab = np.fft.irfft(spec, N) * N / 2
    return np.append(tab, tab[0])


def nharm(f, fmul_max=1.0, sr=SR, fmax=None):
    top = min(0.45 * sr, fmax or 0.45 * sr)
    return max(1, int(top / (f * fmul_max)))


def wt_osc(tab, cyc, *more):
    """tab from wt_table (N+1 points), cyc: running phase in cycles. Extra
    tables share the index computation and are returned as a tuple."""
    N = len(tab) - 1
    x = (cyc % 1.0) * N
    i = x.astype(np.int64)
    np.minimum(i, N - 1, out=i)
    fr = x - i
    outs = []
    for tb in (tab,) + more:
        a = tb[i]
        outs.append(a + fr * (tb[i + 1] - a))
    return outs[0] if not more else tuple(outs)


def cycles(f, fm, n, sr=SR, phase0=0.0):
    fr = f * (np.broadcast_to(fm, (n,)) if np.ndim(fm) == 0 else fm[:n])
    return phase0 + np.cumsum(fr) / sr


# ---------------------------------------------------------------- dynamics / colour

def compress(x, amount=0.6, att=0.004, rel=0.12, sr=SR, ref=None):
    """smooth envelope compressor; amount 0..1 (1 = flatten fully)."""
    a = np.abs(x)
    e1 = onepole_lp(a, 1 / (TWOPI * att), sr)
    env = np.maximum(e1, onepole_lp(a, 1 / (TWOPI * rel), sr))
    ref = ref if ref is not None else np.max(env) + 1e-9
    g = (np.maximum(env, 1e-4 * ref) / ref) ** (-amount)
    return x * g


def overdrive(x, drive=4.0, asym=0.15, sr=SR, os=2):
    """oversampled asymmetric tanh saturation."""
    u = signal.resample_poly(x, os, 1)
    y = np.tanh(drive * u + asym) - np.tanh(asym)
    y = signal.resample_poly(y, 1, os)[:len(x)]
    return y / np.tanh(drive)


def cab(x, kind='4x12', sr=SR):
    if kind == '4x12':
        rows = [hp_sos(75, 2, sr)[0], peq(110, 4, 1.2, sr), peq(500, -3, 0.9, sr), peq(2300, 5, 1.3, sr),
                peq(3600, 3, 2.0, sr)] + list(lp_sos(4800, 4, sr)) + list(lp_sos(7000, 2, sr))
    elif kind == '2x12':
        rows = [hp_sos(90, 2, sr)[0], peq(130, 3, 1.2, sr), peq(1800, 4, 1.0, sr), peq(3000, 2, 2.0, sr)] \
            + list(lp_sos(5200, 4, sr))
    else:  # small combo
        rows = [hp_sos(65, 2, sr)[0], peq(120, 2, 1.0, sr), peq(900, 2, 0.8, sr), peq(2600, 3, 1.5, sr)] + list(lp_sos(6000, 4, sr))
    return signal.sosfilt(np.vstack(rows), x)


def leslie(x, rng, speed=0.0, sr=SR, depth=1.0):
    """rotating speaker: crossover 800 Hz, horn + drum with AM and doppler,
    two mics -> stereo. speed 0 = chorale (slow), 1 = tremolo (fast)."""
    n = len(x)
    t = tvec(n, sr)
    lo = signal.sosfilt(lp_sos(800, 4, sr), x)
    hi = x - lo
    hr = 0.8 + speed * (6.7 - 0.8)
    dr = 0.67 + speed * (5.8 - 0.67)
    hr *= 1 + 0.03 * rng.standard_normal()
    dr *= 1 + 0.03 * rng.standard_normal()
    ph_h = TWOPI * hr * t + rng.uniform(0, TWOPI)
    ph_d = TWOPI * dr * t + rng.uniform(0, TWOPI)
    idx = np.arange(n, dtype=float)
    out = []
    for mic in (0.0, np.pi * 0.62):
        dh = (0.00038 * sr) * depth * (1 + np.sin(ph_h + mic))
        dd = (0.0002 * sr) * depth * (1 + np.sin(ph_d + mic))
        h = np.interp(idx - dh, idx, hi, left=0) * (1 - 0.42 * depth * (0.5 + 0.5 * np.cos(ph_h + mic)))
        # horn directivity: brighter when facing the mic
        d = np.interp(idx - dd, idx, lo, left=0) * (1 - 0.18 * depth * (0.5 + 0.5 * np.cos(ph_d + mic)))
        out.append(h + d)
    return np.stack(out)


def tape(x, rng, sr=SR, wow=6.0, flutter=2.5, hiss=0.002, bw=9000.0, sat=1.4):
    """tape replay colour: wow/flutter pitch, bandwidth, saturation, hiss."""
    n = len(x)
    t = tvec(n, sr)
    c = wow * np.sin(TWOPI * rng.uniform(0.4, 0.8) * t + rng.uniform(0, TWOPI)) \
        + 0.5 * wow * smooth_noise(n, 1.2, rng, sr) \
        + flutter * np.sin(TWOPI * rng.uniform(6, 10) * t) + 0.4 * flutter * smooth_noise(n, 18, rng, sr)
    y = warp(x, cents(c), n)
    y = np.tanh(sat * y) / sat
    y = signal.sosfilt(np.vstack([hp_sos(70, 2, sr)[0], *lp_sos(bw, 4, sr), shelf(3000, -3, True, sr)]), y)
    if hiss:
        y = y + hiss * np.max(np.abs(y)) * bp(rng.standard_normal(n), 1500, 12000, sr=sr)
    return y


def pan2(y, pan=0.0):
    a = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    return np.stack([y * np.cos(a), y * np.sin(a)])


# ---------------------------------------------------------------- level & peak control

def loud_rms(y, sr=SR, win=0.25):
    m = y if y.ndim == 1 else 0.5 * (y[0] + y[1])
    k = max(1, int(win * sr))
    if len(m) <= k:
        return float(np.sqrt(np.mean(m ** 2)) + 1e-12)
    c = np.cumsum(np.concatenate([[0.0], m * m]))
    return float(np.sqrt(((c[k:] - c[:-k]) / k).max()) + 1e-12)


def peak_limit(y, ceil, sr=SR, look=0.002):
    """transparent look-ahead peak limiter (vectorised). A min-filter followed by
    a box average of the same width guarantees gain <= ceil/|y| everywhere."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    a = np.abs(y) if y.ndim == 1 else np.max(np.abs(y), axis=0)
    if a.max() <= ceil:
        return y
    g = np.minimum(1.0, ceil / (a + 1e-12))
    w = max(3, int(2 * look * sr) | 1)
    g = uniform_filter1d(minimum_filter1d(g, w), w)
    w2 = 4 * w + 1                                    # gentler second stage (less flutter)
    g = np.minimum(g, uniform_filter1d(minimum_filter1d(g, w2), w2) * 0.5 + g * 0.5)
    return y * g


def finish(y, vel, sr=SR, crest=5.0, curve=1.5, floor=0.1):
    """normalise a note to a register-independent loudness that follows vel,
    then limit peaks to `crest` x loudness (keeps transients but no overs)."""
    r = loud_rms(y, sr)
    lvl = floor + (1 - floor) * float(np.clip(vel, 0, 1)) ** curve
    y = y * (lvl / r)
    return peak_limit(y, crest * lvl, sr)
