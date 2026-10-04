"""
Canopy Orchestra — membranes, shakers & small percussion: families 'drum', 'perc'.

Membranes use the true circular-membrane mode set (Bessel zeros j_mn) with
strike-position weights |J_m(j_mn r)|, mallet/hand contact-time weighting,
per-mode damping, a tension pitch-drop proportional to strike strength,
optional double-head coupling (beating mode pairs), body air resonances
(Helmholtz for djembe/udu/cajon, pipe modes for congas), snare/rattle buzz
gated by head motion, gate-controlled pitch bends (talking drum, bayan) and
rolls (timpani, bass drum, snare, felt toms) by convolving a stroke train with
the single-stroke response (phase-coherent, like a real resonating head).

Shakers are particle models (PhISEM-like): system energy -> random bead
collisions -> shell resonances. Cymbals are modal clouds + time-frequency
shaped noise with bloom.

Unpitched presets: the pitch argument retunes (f / C4 multiplies the native
tuning); play(name, 60) is the natural sound.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.special import jn_zeros, jv

from .core import *  # noqa: F401,F403
from .core import TWOPI, SR, bp, lp, hp, reson, env_perc, softclip, register
from .inst_mallets import (_bank, _hw, _tc, _amp, _burst, _finish, _tame, _envf, _detune,
                           _slow_noise, _tfnoise, m_bell, P, C4)

# --------------------------------------------------------------------------- membrane modes
_MODES = []
for _m in range(0, 9):
    for _n, _z in enumerate(jn_zeros(_m, 4), 1):
        _MODES.append((_z, _m, _n))
_MODES.sort()
_J01 = _MODES[0][0]
_MR = np.array([z / _J01 for z, _, _ in _MODES])
_MM = np.array([m for _, m, _ in _MODES])
_MZ = np.array([z for z, _, _ in _MODES])
# normalisation: max |J_m| over the drum
_MNORM = np.array([np.abs(jv(m, np.linspace(0, z, 400))).max() for z, m, _ in _MODES])


def _mem(pos, maxr=4.0):
    """ratios & strike weights of ideal membrane modes for a strike at radius
    pos (0 centre .. 1 rim)."""
    k = _MR <= maxr
    w = np.abs(jv(_MM[k], _MZ[k] * pos)) / _MNORM[k]
    return _MR[k], w, _MM[k]


def _roll_train(n, sr, rng, dur, rate, vel):
    """alternating-hand roll stroke train (impulses) for the gate length."""
    times = np.arange(0.0, dur, 1.0 / rate)
    times = times + rng.normal(0, 0.004, len(times))
    times[0] = 0.0
    times = np.clip(times, 0, None)
    amp = rng.uniform(0.8, 1.05, len(times)) * np.where(np.arange(len(times)) % 2, 0.85, 1.0)
    tr = np.zeros(n)
    np.add.at(tr, np.clip((times * sr).astype(int), 0, n - 1), amp)
    return tr


def m_drum(f, dur, vel, rng, sr, f0=200.0, tuned=False, pos=0.6, maxr=4.0, ratios=None,
           amps=None, tau=0.3, tau_ref=None, slope=0.3, qexp=0.8, tau01=1.0, tc=1.5e-3,
           glide=0.06, glide_tau=0.03, body=(), shell=(), noise=0.2, nband=(400.0, 4000.0),
           ntau=0.006, crack=0.0, crack_band=(1200.0, 6000.0), crack_tau=0.012,
           buzz=0.0, buzz_th=0.1, buzz_band=(2500.0, 9000.0), buzz_tau=0.12,
           couple=0.0, couple_depth=0.3, bend=0.0, bend_time=0.5, bend_back=True,
           wobble=None, slosh=0.0, sat=0.0, roll_at=None, roll_rate=15.0, cap=3.0,
           detune=8.0, crest=4.6, norm=0.0):
    """struck membrane (hands, sticks, mallets, beaters)."""
    base = f if tuned else f0 * f / C4
    base = _detune(base, rng, detune)
    rt = base / f0 if not tuned else 1.0
    A = _amp(vel)
    tce = _tc(tc, vel)
    T = tau * ((tau_ref / base) ** slope if tau_ref else 1.0) * rng.uniform(0.9, 1.1)
    if ratios is None:
        rat, w, mm = _mem(float(np.clip(pos + rng.uniform(-0.02, 0.02), 0.0, 0.97)), maxr)
        w = w / (np.sqrt(np.sum(w ** 2)) + 1e-12)   # position changes timbre, not level
    else:
        rat = np.asarray(ratios, float)
        w = np.asarray(amps, float)
        mm = np.ones(len(rat))
    rat = rat * (1 + np.r_[0, rng.uniform(-0.006, 0.006, len(rat) - 1)])
    fr = base * rat
    am = w * _hw(fr, tce) * rng.uniform(0.85, 1.15, len(fr))
    ta = T * (1.0 / rat) ** qexp
    if norm:
        am = am / (np.sqrt(np.sum(am ** 2)) + 1e-9) ** norm   # level-compensate mallet darkening
    if ratios is None:
        ta = np.where((mm == 0) & (rat < 1.01), ta * tau01, ta)
    else:
        ta[0] *= tau01
    beat = np.full(len(fr), couple) * rng.uniform(0.7, 1.3, len(fr)) if couple else None
    # extra resonances: body (air, scales weakly with tuning) and shell (fixed-ish)
    xf, xa, xt, xr = [], [], [], []
    for (hz, a, tb, ri) in body:
        xf.append(hz * rt ** 0.5)
        xa.append(a)
        xt.append(tb)
        xr.append(ri)
    for (hz, a, tb) in shell:
        xf.append(hz * rt ** 0.25 * rng.uniform(0.97, 1.03))
        xa.append(a * _hw(hz, tce))
        xt.append(tb)
        xr.append(0.0)
    one_len = max(8 * max(ta.max(), max(xt) if xt else 0), 0.15)
    rolling = roll_at is not None and dur >= roll_at
    L = min(cap + (dur if rolling else 0.0), one_len + (dur if rolling else 0.0) + max(bend_time, 0) * (bend != 0))
    n = int(L * sr)
    t = np.arange(n) / sr
    # frequency trajectory: tension pitch drop (+ gate bend, + water wobble)
    fm = 1.0 + glide * (0.25 + 0.75 * vel) ** 1.5 * np.exp(-t / glide_tau)
    if bend:
        t1 = max(0.03, min(dur, bend_time))
        u = np.clip((t - 0.02) / t1, 0, 1)
        sh = 0.5 - 0.5 * np.cos(np.pi * u)
        if bend_back and dur < L:
            sh = sh * np.where(t > dur, np.exp(-(t - dur) / 0.06), 1.0)
        fm = fm * 2 ** (bend * sh / 12.0)
    if wobble is not None:
        wd, wr, wt = wobble
        fm = fm * (1 + wd * vel * np.sin(TWOPI * wr * t) * np.exp(-t / wt))
    ymem = _bank(n, sr, fr, am, ta, rng, beat=beat, bdepth=couple_depth if couple else None, fmul=fm)
    y = ymem
    if xf:
        y = y + _bank(n, sr, xf, xa, xt, rng, rise=xr, fmul=fm ** 0.3 if (bend or glide > 0.2) else None)
    y = A * y
    if noise:
        y += noise * A * (0.3 + 0.7 * vel) * _burst(n, sr, rng, nband[0], nband[1] * (0.6 + 0.6 * vel), ntau)
    if crack:
        y += crack * A * (0.4 + 0.6 * vel) * _burst(n, sr, rng, crack_band[0], crack_band[1], crack_tau, att=0.0004)
    if buzz:
        e = _envf(ymem, sr, 60.0)
        d = A * e / (e.max() + 1e-12)
        g = np.maximum(d - buzz_th, 0.0)
        g = signal.lfilter([1 - np.exp(-1 / (0.003 * sr))], [1, -np.exp(-1 / (0.003 * sr))], g)
        g *= np.exp(-t / buzz_tau) ** 0.5
        y += buzz * A * bp(rng.standard_normal(n), buzz_band[0], buzz_band[1], 2, sr) * g * 1.5 / max(A, 0.05) ** 0.5
    if slosh:
        sl = lp(rng.standard_normal(n), 500.0, 2, sr) * env_perc(n, 0.02, 0.25, sr)
        sl *= (1 + np.sin(TWOPI * 5.5 * t)) * 0.5
        y += slosh * A * vel * sl / (np.abs(sl).max() + 1e-12) * 0.3
    if sat:
        pk = np.abs(y).max() / max(A, 1e-3) + 1e-12
        y = np.tanh(2 * sat * y / pk) * pk / (2 * sat)
    y = y - lp(y, 22.0, 1, sr) if base < 120 else y
    if rolling:
        # roll: convolve the single-stroke response with an alternating stroke train
        tr = _roll_train(n, sr, rng, dur, roll_rate * rng.uniform(0.95, 1.05), vel)
        ir = y[:int(min(n, one_len * sr))]
        y = signal.fftconvolve(tr, ir)[:n] * 0.45
    if crest:
        y = _tame(y, sr, crest)
    return _finish(y, sr, cap + (dur if rolling else 0.0))


def m_udu(f, dur, vel, rng, sr, f0=110.0, tone=False, tau=0.22, sweep=0.55, sweep_t=0.06,
          clay=0.3, cap=2.0):
    """udu clay pot. bloop: palm slaps the side hole - the Helmholtz air
    resonance glides up as the hand leaves. tone: finger on the clay body."""
    rt = f / C4
    A = _amp(vel)
    fh = _detune(f0 * rt, rng, 10.0)
    n = int(min(cap, 8 * tau + 0.1) * sr)
    t = np.arange(n) / sr
    if not tone:
        st = sweep * (0.6 + 0.4 * vel) * rng.uniform(0.85, 1.15)
        fm = (1 - st) + st * -np.expm1(-t / sweep_t)
        y = _bank(n, sr, [fh, fh * 2.02, fh * 3.1], [1.0, 0.12, 0.04], [tau, tau * 0.4, tau * 0.2], rng, fmul=fm)
        y += 0.25 * _burst(n, sr, rng, 60.0, 500.0, 0.012, att=0.002)
        cl = clay * 0.4
    else:
        y = 0.35 * _bank(n, sr, [fh], [1.0], [tau * 0.7], rng, rise=0.004)
        cl = clay * 1.6
    cf = 380.0 * rt ** 0.5 * rng.uniform(0.95, 1.05) * np.array([1.0, 1.48, 2.1, 2.72, 3.6])
    tce = _tc(0.6e-3 if tone else 2.5e-3, vel)
    y += cl * _bank(n, sr, cf, np.array([1.0, 0.6, 0.4, 0.25, 0.12]) * _hw(cf, tce),
                    np.array([0.12, 0.08, 0.06, 0.04, 0.03]), rng)
    y += (0.15 if tone else 0.05) * vel * _burst(n, sr, rng, 800.0, 5000.0, 0.003)
    return _finish(_tame(A * y, sr, 6.0), sr, cap)


def m_particles(f, dur, vel, rng, sr, res=((5000.0, 3.0, 1.0),), beads=40, shake=0.1,
                att=0.01, held=False, shake_rate=0.0, release=0.15, hp_f=1500.0,
                tick=0.3, cap=10.0, stereo_w=0.0, bottom=None, body=None, pulse=False,
                grit=1.0, crest=4.6, swish=0.0):
    """particle rattle (egg shaker, caxixi, ganza, seed pods, rainstick, cabasa,
    shekere, tambourine jingles)."""
    rt = f / C4
    A = _amp(vel)
    if held:
        g = max(dur, 0.06)
        L = g + 6 * release + 0.05
    else:
        g = att + shake
        L = att + 7 * shake + 0.08
    L = min(L, cap)
    n = int(L * sr)
    t = np.arange(n) / sr
    # system energy envelope
    if held:
        E = np.clip(t / att, 0, 1) ** 1.5
        if shake_rate:
            ph = TWOPI * shake_rate * rng.uniform(0.9, 1.1) * t
            E = E * (0.25 + 0.75 * np.abs(np.sin(0.5 * ph)) ** 2)
        E = E * (1 + 0.25 * _slow_noise(n, sr, 3.0, rng, 64))
        E = np.where(t < g, E, E[min(int(g * sr), n - 1)] * np.exp(-(t - g) / release))
    else:
        E = np.where(t < att, (t / att) ** 2, np.exp(-(t - att) / shake))
    E = np.clip(E, 0, None) * (0.6 + 0.4 * vel)
    nch = 2 if stereo_w else 1
    outs = []
    for ch in range(nch):
        rate = beads * 60.0 * E * (0.7 + 0.3 * vel)
        ev = rng.random(n) < rate / sr
        amp = rng.uniform(0.0, 1.0, n) ** grit * np.sqrt(E) * ev
        x = amp * rng.choice([-1.0, 1.0], n)
        y = tick * hp(x, hp_f * (0.8 + 0.4 * vel), 2, sr)
        for fr_, q, gg in res:
            y += gg * np.sqrt(q) * reson(x, fr_ * rt * rng.uniform(0.97, 1.03), q, sr)
        if swish:
            y += swish * hp(rng.standard_normal(n), hp_f, 2, sr) * E * 0.05
        outs.append(y)
    y = np.stack([outs[0] + (1 - stereo_w) * outs[1], outs[1] + (1 - stereo_w) * outs[0]]) if nch == 2 else outs[0]
    if bottom is not None:
        # caxixi: seeds hitting the hard gourd bottom at the end of the stroke
        tb, fb, ga = bottom
        k = int(tb * (1.1 - 0.2 * vel) * sr)
        if k < n:
            hit = np.zeros(n)
            m = n - k
            cl = rng.random(m) < np.exp(-np.arange(m) / (0.006 * sr)) * 0.15
            hit[k:] = cl * rng.uniform(0.3, 1.0, m)
            hb = ga * (reson(hit, fb * rt, 6.0, sr) + 0.6 * hp(hit, 3000.0, 2, sr))
            y = y + (hb if y.ndim == 1 else hb[None, :])
    if body is not None:
        fb, ga, tb = body
        bb = ga * vel * _bank(n, sr, [fb * rt], [1.0], [tb], rng, rise=0.002)
        y = y + (bb if y.ndim == 1 else bb[None, :])
    # harder shaking = beads hit harder = brighter
    y = _vbright(y, vel, sr)
    y = A * y
    if crest:
        y = _tame(y, sr, crest)
    return _finish(y, sr, cap)


def _vbright(y, vel, sr, lo=2500.0, hi=16000.0):
    fc = lo + (hi - lo) * float(np.clip(vel, 0, 1)) ** 1.5
    return signal.sosfilt(signal.butter(1, min(fc, 0.45 * sr), fs=sr, output='sos'), y, axis=-1) * \
        (1.0 + 0.5 * (1 - vel))


def m_scrape(f, dur, vel, rng, sr, rate=55.0, accel=0.3, res=((1200.0, 4.0, 1.0),
             (2500.0, 5.0, 0.7), (4200.0, 6.0, 0.4)), dur_min=0.08, tick=0.12, hp_f=2000.0,
             cap=6.0, crest=4.6):
    """guiro: a stick dragged over ridges -> ridge clicks through the gourd."""
    rt = f / C4
    A = _amp(vel)
    g = max(dur, dur_min)
    L = min(cap, g + 0.15)
    n = int(L * sr)
    r0 = rate * (0.75 + 0.5 * vel) * rng.uniform(0.9, 1.1)
    times = [0.0]
    while times[-1] < g:
        u = times[-1] / g
        times.append(times[-1] + rng.uniform(0.9, 1.1) / (r0 * (1 + accel * u)))
    times = np.array(times[:-1])
    u = times / g
    amp = (0.6 + 0.4 * np.sin(np.pi * np.clip(u, 0, 1)) ** 0.5) * rng.uniform(0.6, 1.0, len(times))
    x = np.zeros(n)
    np.add.at(x, np.clip((times * sr).astype(int), 0, n - 1), amp)
    k = np.exp(-np.arange(int(0.002 * sr)) / (0.0004 * sr)) * rng.standard_normal(int(0.002 * sr))
    x = signal.fftconvolve(x, k)[:n]
    y = tick * hp(x, hp_f, 2, sr)
    for fr_, q, gg in res:
        y += gg * np.sqrt(q) * reson(x, fr_ * rt, q, sr)
    y = A * _vbright(y, vel, sr, 1800.0, 12000.0) * (0.8 + 0.2 * vel)
    if crest:
        y = _tame(y, sr, crest)
    return _finish(y, sr, cap)


def m_brush(f, dur, vel, rng, sr, f0=190.0, sweep=1.6, cap=6.0):
    """brushed snare: a tap of wire bristles, then the circular swish for the gate."""
    rt = f / C4
    A = _amp(vel)
    g = max(dur, 0.08)
    L = min(cap, g + 0.35)
    n = int(L * sr)
    t = np.arange(n) / sr
    # tap: bristles hit over ~10 ms, lightly exciting head and snares
    tap = np.zeros(n)
    m = int(0.012 * sr)
    tap[:m] = (rng.random(m) < 0.25) * rng.uniform(0.2, 1.0, m) * np.exp(-np.arange(m) / (0.004 * sr))
    tap *= 3.0 / (tap.sum() + 1e-9)
    head = _bank(n, sr, f0 * rt * np.array([1.0, 1.59, 2.14, 2.3]), [1.0, 0.6, 0.4, 0.3],
                 [0.12, 0.08, 0.06, 0.05], rng)
    y = 0.12 * signal.fftconvolve(tap, head[:int(0.5 * sr)])[:n] * (0.5 + 0.5 * vel)
    y += 2.4 * bp(tap, 2000.0, 9000.0, 2, sr) * (0.6 + 0.4 * vel)
    # swish: circular sweep modulates pressure/speed
    sw = sweep * rng.uniform(0.85, 1.15)
    mod = 0.55 + 0.45 * np.sin(TWOPI * sw * t + rng.uniform(0, 6)) ** 2
    env = np.clip(t / 0.06, 0, 1) * np.where(t < g, 1.0, np.exp(-(t - g) / 0.05)) * (t > 0.02)
    nz = bp(rng.standard_normal(n), 1500.0 * (0.8 + 0.4 * vel), 9000.0, 2, sr)
    nz += 0.3 * bp(rng.standard_normal(n), 400.0, 1500.0, 2, sr)
    y += 0.22 * _vbright(nz * mod * env, vel, sr, 3000.0, 14000.0)
    return _finish(_tame(A * y, sr, 4.6), sr, cap)


def m_cymbal(f, dur, vel, rng, sr, f_lo=320.0, f_hi=11000.0, nmodes=26, tau=2.5,
             tslope=0.45, bloom=0.03, ping=0.0, ping_f=620.0, wash=1.0, tc=0.3e-3,
             roll=False, roll_rate=14.0, cap=8.0, hiss=1.0, tilt=0.0, cascade=0.5):
    """cymbal: inharmonic modal cloud + time-frequency shaped noise wash.
    ping: stick 'bell' partials (ride). roll: soft-mallet swell for the gate."""
    rt = f / C4
    A = _amp(vel)
    tce = _tc(tc, vel)
    g = max(dur, 0.1) if roll else 0.0
    L = min(cap + g, g + 8 * tau * (f_lo / 400.0) ** -tslope * 0.6 + 0.2)
    if roll:
        L = g + min(cap * 0.5, 4.0)
    n = int(L * sr)
    m1 = int(min(n, (8 * tau + 0.2) * sr)) if not roll else int(min(n, min(cap * 0.5, 4.0) * sr))
    u = np.sort(rng.uniform(0, 1, nmodes))
    fr = f_lo * rt * (f_hi / f_lo) ** u
    hwm = _hw(fr, tce)
    am = rng.uniform(0.3, 1.0, nmodes) * np.maximum(hwm, cascade * (0.3 + 0.7 * vel) * 0.6) * \
        (fr / (1000 * rt)) ** tilt
    ta = tau * (1000.0 * rt / fr) ** tslope * rng.uniform(0.7, 1.3, nmodes)
    ri = bloom * (fr / (1000 * rt)) ** 0.5 * (1.3 - 0.6 * vel) + 0.1 * (1 - hwm) * (cascade > 0)
    ev = np.arange(nmodes) % 2 == 0
    ya = _bank(m1, sr, fr[ev], am[ev], ta[ev], rng, rise=ri[ev], floor=6.5)
    yb = _bank(m1, sr, fr[~ev], am[~ev], ta[~ev], rng, rise=ri[~ev], floor=6.5)
    ir = np.stack([ya + 0.6 * yb, 0.6 * ya + yb]) * 0.5
    if ping:
        pf = ping_f * rt * np.array([1.0, 1.52, 2.3, 2.95, 3.9])
        pp = ping * _bank(m1, sr, pf, np.array([1.0, 0.7, 0.5, 0.35, 0.2]) * _hw(pf, tce),
                          [1.2, 0.9, 0.7, 0.5, 0.4], rng)
        ir += pp[None, :]

    def mag(tt, ff):
        fc = np.maximum(ff, 60.0) / (1000.0 * rt)
        d_ = tau * 0.7 * fc ** -tslope
        r_ = bloom * fc ** 0.5 * (1.3 - 0.6 * vel)
        band = (ff > f_lo * rt * 0.8) & (ff < 16000)
        hwf = _hw(ff, tce * 0.7)
        # nonlinear energy cascade: the spectrum brightens after the strike
        casc = hwf + cascade * (0.3 + 0.7 * vel) * (1 - hwf) * -np.expm1(-tt / 0.12)
        gg = wash * casc * fc ** (tilt - 0.15) * band
        return gg * -np.expm1(-(tt + 1e-4) / r_) * np.exp(-tt / d_)
    ir += hiss * 0.35 * _tfnoise(m1, sr, rng, mag, 2)
    if roll:
        tr = _roll_train(n, sr, rng, g, roll_rate * rng.uniform(0.95, 1.05), vel)
        tt = np.arange(n) / sr
        cres = np.clip(tt / g, 0, 1) ** 1.6 * 0.9 + 0.1
        tr *= cres
        import scipy.fft as sfft
        nf = sfft.next_fast_len(n + ir.shape[1])
        T = sfft.rfft(tr, nf)
        y = sfft.irfft(T[None, :] * sfft.rfft(ir, nf, axis=-1), nf, axis=-1)[:, :n] * 0.35
    else:
        y = np.zeros((2, n))
        y[:, :m1] = ir
    return _finish(_tame(A * y, sr, 6.0), sr, cap + g)


def m_claps(f, dur, vel, rng, sr, people=4, spread=0.022, cupped=1100.0, room=0.15, cap=1.0):
    """hand claps: a few people clapping together (smeared), cupped-hand
    resonance, short room tail."""
    rt = f / C4
    A = _amp(vel)
    n = int(cap * sr)
    y = np.zeros(n)
    k = max(1, int(round(people * (0.6 + 0.6 * vel))))
    for i in range(k):
        t0 = 0.0 if i == 0 else abs(rng.normal(0, spread))
        s0 = int(t0 * sr)
        fc = cupped * rt * rng.uniform(0.7, 1.5)
        m = int(0.06 * sr)
        b = _burst(m, sr, rng, fc * 0.5, fc * 3.5 * (0.7 + 0.5 * vel), rng.uniform(0.005, 0.009), att=0.0005)
        b += 0.7 * reson(b, fc, 3.0, sr)
        g = rng.uniform(0.6, 1.0)
        e = min(n, s0 + m)
        y[s0:e] += g * b[:e - s0]
    tail = bp(rng.standard_normal(n), 500.0, 6000.0, 2, sr) * env_perc(n, 0.01, 0.06, sr)
    y += room * tail * np.abs(y).max()
    y = _vbright(y, vel, sr, 2500.0, 14000.0)
    return _finish(_tame(A * y, sr, 4.6), sr, cap)


def m_unp(f, dur, vel, rng, sr, model=None, f0=C4, **kw):
    """retune wrapper for unpitched presets built on a pitched model."""
    return model(f0 * f / C4, dur, vel, rng, sr, **kw)


# =========================================================================== presets
J, AF, LA, AS, CE, OR, OL, SP, WO = ('jungle', 'african', 'latin', 'asian', 'celtic',
                                      'orchestral', 'oldfield', 'space', 'world')
IN = 'inharmonic'


def _d(name, desc, tags, **kw):
    register(name, m_drum, 'drum', pitched=False, tags=tags + (IN,), desc=desc, **kw)


# ---------------------------------------------------------------- hand drums
_d('drum.djembe_bass', 'djembe bass: flat palm in the centre, goblet body booms', (AF, J, WO),
   f0=260.0, pos=0.12, tc=4.5e-3, tau=0.12, tau01=0.6, body=((75.0, 1.4, 0.16, 0.004),),
   glide=0.04, noise=0.25, nband=(80.0, 900.0), ntau=0.01, cap=1.5)
_d('drum.djembe_tone', 'djembe tone: fingers at the edge, ringing round goatskin note', (AF, J, WO),
   f0=260.0, pos=0.78, tc=1.6e-3, tau=0.2, tau01=0.5, body=((75.0, 0.12, 0.1, 0.004),),
   glide=0.05, noise=0.2, nband=(300.0, 2500.0), ntau=0.004, crack=0.15,
   crack_band=(800.0, 3500.0), cap=1.5)
_d('drum.djembe_slap', 'djembe slap: whip-crack of loose fingers on tight skin', (AF, J, WO),
   f0=260.0, pos=0.86, maxr=5.5, tc=0.35e-3, tau=0.1, qexp=0.5, tau01=0.5,
   body=((75.0, 0.05, 0.08, 0.004),), glide=0.08, noise=0.2, nband=(600.0, 5000.0),
   crack=1.6, crack_band=(1200.0, 7000.0), crack_tau=0.014, cap=1.2)
_d('drum.conga_open', 'conga open tone: singing rawhide note over a tall staved shell', (LA, J, WO),
   f0=170.0, pos=0.72, tc=1.4e-3, tau=0.38, tau01=0.5,
   body=((115.0, 0.25, 0.25, 0.006), (345.0, 0.1, 0.12, 0.004)), glide=0.03, glide_tau=0.04,
   noise=0.15, nband=(300.0, 2000.0), crack=0.05, cap=2.0)
_d('drum.conga_muffled', 'conga muffled tone: palm stays on the head, dark short thump', (LA, J, WO),
   f0=165.0, pos=0.55, tc=2.5e-3, tau=0.08, qexp=1.3, body=((115.0, 0.3, 0.08, 0.004),),
   glide=0.02, noise=0.3, nband=(150.0, 1500.0), ntau=0.008, cap=1.0)
_d('drum.conga_slap', 'conga slap: closed, popping crack with a choked ring', (LA, J, WO),
   f0=175.0, pos=0.86, maxr=5.5, tc=0.35e-3, tau=0.06, qexp=0.5, glide=0.05, noise=0.2,
   nband=(500.0, 4000.0), crack=1.8, crack_band=(1000.0, 6500.0), crack_tau=0.01, cap=1.0)
_d('drum.bongo', 'bongo: small tight heads, crisp high ring under the fingertips', (LA, J, WO),
   f0=400.0, pos=0.8, maxr=5.5, tc=0.7e-3, tau=0.22, tau01=0.5, shell=((330.0, 0.15, 0.05),),
   glide=0.06, glide_tau=0.02, crack=0.2, crack_band=(1500.0, 8000.0), noise=0.12,
   nband=(800.0, 5000.0), cap=1.2)
register('drum.talking_drum', m_drum, 'drum', lo='C3', hi='C5', tags=(AF, J, WO, IN),
         desc='talking drum: hooked-stick strike, the arm squeezes the cords - pitch rises with the gate',
         sustain=True, tuned=True, pos=0.28, qexp=1.1, tc=1.0e-3, tau=0.4, tau01=1.0, couple=6.0,
         couple_depth=0.25, glide=0.04, bend=5.0, bend_time=0.5, noise=0.2,
         nband=(300.0, 3000.0), ntau=0.004, crack=0.1, cap=2.0)
register('drum.talking_drum_down', m_drum, 'drum', lo='C3', hi='C5', tags=(AF, J, WO, IN),
         desc='talking drum, struck squeezed: the note sighs downward as the cords relax',
         sustain=True, tuned=True, pos=0.28, qexp=1.1, tc=1.0e-3, tau=0.4, tau01=1.0, couple=6.0,
         couple_depth=0.25, glide=0.04, bend=-5.0, bend_time=0.45, bend_back=False, noise=0.2,
         nband=(300.0, 3000.0), ntau=0.004, crack=0.1, cap=2.0)
_d('drum.dundunba', 'dundunba: big double-headed cowhide bass drum, stick-struck boom', (AF, J, WO),
   f0=70.0, pos=0.45, tc=0.9e-3, tau=0.35, tau01=0.8, couple=3.0, couple_depth=0.3,
   glide=0.05, noise=0.35, nband=(200.0, 2500.0), ntau=0.005, crack=0.3,
   crack_band=(600.0, 3000.0), crack_tau=0.006, cap=2.0)
_d('drum.sangban', 'sangban: middle dunun, round woody bounce', (AF, J, WO),
   f0=108.0, pos=0.45, tc=0.9e-3, tau=0.3, tau01=0.8, couple=4.5, couple_depth=0.3,
   glide=0.05, noise=0.35, nband=(250.0, 3000.0), ntau=0.005, crack=0.35,
   crack_band=(700.0, 3500.0), crack_tau=0.006, cap=2.0)
_d('drum.kenkeni', 'kenkeni: small dunun, tight hard-stick knock that drives the ensemble', (AF, J, WO),
   f0=165.0, pos=0.45, tc=0.8e-3, tau=0.22, tau01=0.8, couple=6.0, couple_depth=0.3,
   glide=0.06, noise=0.35, nband=(300.0, 3500.0), ntau=0.004, crack=0.45,
   crack_band=(900.0, 4500.0), crack_tau=0.006, cap=1.5)
_d('drum.surdo', 'surdo: deep samba bass, felt mallet on plastic, long open hum', (LA, WO),
   f0=52.0, pos=0.3, tc=3.5e-3, tau=0.7, tau01=1.0, couple=1.5, couple_depth=0.35,
   glide=0.04, glide_tau=0.05, noise=0.1, nband=(60.0, 600.0), ntau=0.01, cap=3.0)
_d('drum.bodhran', 'bodhran: goatskin frame drum and wooden tipper, the Ommadawn heartbeat', (CE, OL, WO),
   f0=95.0, pos=0.55, maxr=5.5, tc=1.6e-3, tau=0.2, tau01=0.7, qexp=1.0, glide=0.07, glide_tau=0.02,
   noise=0.3, nband=(200.0, 2500.0), ntau=0.004, crack=0.35, crack_band=(700.0, 4000.0),
   crack_tau=0.006, cap=1.5)
_d('drum.bodhran_muted', 'bodhran with hand pressed behind the skin: higher, tight, clipped', (CE, OL, WO),
   f0=135.0, pos=0.7, maxr=5.5, tc=1.2e-3, tau=0.07, qexp=1.0, glide=0.05, noise=0.35,
   nband=(400.0, 4000.0), crack=0.4, crack_band=(1000.0, 5000.0), crack_tau=0.006, cap=1.0)
_d('drum.frame_drum', 'large frame drum (tar): soft fingers, deep centre dum, long low ring', (WO, AF, CE),
   f0=66.0, pos=0.2, tc=3.5e-3, tau=0.45, tau01=0.8, glide=0.1, glide_tau=0.04, noise=0.15,
   nband=(80.0, 800.0), ntau=0.015, cap=2.5)
register('drum.tabla_dayan', m_drum, 'drum', lo='B3', hi='F5', tags=(AS, WO),
         desc='tabla dayan: the black syahi makes the head ring with true harmonics (tin stroke)',
         tuned=True, ratios=(1.0, 2.0, 2.99, 4.0, 5.01, 2.3, 3.6, 5.6),
         amps=(1.0, 0.7, 0.45, 0.3, 0.15, 0.1, 0.07, 0.05), tau=0.8, tau_ref=262.0, slope=0.3,
         qexp=0.55, tc=0.45e-3, glide=0.006, noise=0.15, nband=(1000.0, 6000.0), ntau=0.003,
         crack=0.1, crack_band=(2000.0, 8000.0), detune=3.0, cap=2.5)
_d('drum.tabla_bayan', 'tabla bayan: deep ghe stroke, the heel slides and the pitch swoops up',
   (AS, WO), sustain=True, f0=95.0, ratios=(1.0, 2.0, 3.02, 1.59, 2.14),
   amps=(1.0, 0.35, 0.15, 0.15, 0.1), tau=0.6, qexp=0.5, tc=2.5e-3, glide=0.04, bend=5.0,
   bend_time=0.45, bend_back=False, noise=0.2, nband=(80.0, 800.0), ntau=0.008, cap=2.0)
register('drum.udu', m_udu, 'drum', pitched=False, tags=(AF, J, WO, IN),
         desc='udu bloop: palm on the clay pot side-hole, a liquid rising boom')
register('drum.udu_tone', m_udu, 'drum', pitched=False, tags=(AF, J, WO, IN),
         desc='udu struck on the belly with fingers: clay ring over a soft air hum',
         tone=True, tau=0.18)
_CAJ = (1.0, 1.9, 2.6, 3.3, 4.1, 5.0, 6.2)
_d('drum.cajon_bass', 'cajon bass: palm on the plywood face, box boom and faint snare', (LA, WO),
   f0=120.0, ratios=_CAJ, amps=(1.0, 0.6, 0.4, 0.3, 0.2, 0.12, 0.08), tc=4e-3, tau=0.08,
   qexp=0.6, body=((85.0, 1.2, 0.1, 0.003),), glide=0.0, buzz=0.2, buzz_th=0.3,
   noise=0.25, nband=(60.0, 600.0), ntau=0.008, cap=1.0)
_d('drum.cajon_slap', 'cajon slap: fingers at the top corner, cracking plate and sizzling wires', (LA, WO),
   f0=120.0, ratios=_CAJ, amps=(0.3, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7), tc=0.5e-3, tau=0.05,
   qexp=0.4, body=((85.0, 0.2, 0.06, 0.003),), glide=0.0, crack=0.9,
   crack_band=(1500.0, 7000.0), buzz=0.8, buzz_th=0.06, buzz_band=(2500.0, 9000.0),
   buzz_tau=0.08, noise=0.2, nband=(800.0, 5000.0), cap=1.0)

# ---------------------------------------------------------------- orchestral & kit
_TIMP = (0.85, 1.0, 1.5, 1.65, 1.98, 2.2, 2.44, 2.94, 3.42)
register('drum.timpani', m_drum, 'drum', lo='C2', hi='C4', tags=(OR, OL, IN),
         desc='pedal timpani: felt mallet, tuned principal tone over the copper kettle; rolls when held',
         sustain=True, ref_dur=0.5, tuned=True, ratios=_TIMP,
         amps=(0.6, 1.0, 0.55, 0.1, 0.35, 0.08, 0.2, 0.1, 0.06), tau=0.9, tau_ref=110.0,
         slope=0.3, qexp=0.4, tau01=0.08, tc=2.4e-3, glide=0.006, glide_tau=0.1, noise=0.12,
         nband=(60.0, 500.0), ntau=0.02, roll_at=1.2, roll_rate=16.0, detune=2.0, cap=4.0)
_d('drum.bass_drum', 'orchestral gran cassa: huge soft beater, deep rolling boom; thunder roll when held',
   (OR, OL), sustain=True, ref_dur=0.5, f0=48.0, pos=0.35, tc=7e-3, tau=0.7, tau01=0.9,
   couple=1.2, couple_depth=0.3, glide=0.03, noise=0.08, nband=(30.0, 300.0), ntau=0.03,
   roll_at=1.2, roll_rate=13.0, cap=4.0)
_d('drum.snare', 'snare drum: stick on coated head, crisp wire sizzle; buzz roll when held', (OR, WO),
   sustain=True, ref_dur=0.5, f0=190.0, pos=0.35, tc=0.5e-3, tau=0.18, tau01=0.6, glide=0.03,
   buzz=1.0, buzz_th=0.02, buzz_band=(2000.0, 11000.0), buzz_tau=0.12, noise=0.25,
   nband=(500.0, 5000.0), ntau=0.003, crack=0.3, shell=((450.0, 0.1, 0.05),), roll_at=1.2,
   roll_rate=24.0, cap=1.5)
register('drum.snare_brush', m_brush, 'drum', pitched=False, tags=(OR, WO, IN),
         desc='brushed snare: wire-brush tap and circular swish for the gate length', sustain=True)
register('drum.tom', m_drum, 'drum', lo='E2', hi='C4', tags=(OR, WO, IN),
         desc='kit tom: stick on a double-headed tom, punchy pitch drop and shell ring',
         tuned=True, pos=0.35, tc=0.7e-3, tau=0.4, tau_ref=110.0, slope=0.3, tau01=0.8,
         couple=3.0, couple_depth=0.25, glide=0.06, glide_tau=0.03, noise=0.25,
         nband=(300.0, 3000.0), ntau=0.003, cap=2.5)
register('drum.tom_felt', m_drum, 'drum', lo='C2', hi='C4', tags=(OR, OL, IN),
         desc='concert toms with felt mallets: round, tuned, ritual; rolls when held',
         sustain=True, ref_dur=0.5, tuned=True, pos=0.4, tc=3.5e-3, tau=0.45, tau_ref=110.0,
         slope=0.3, tau01=0.8, couple=2.0, couple_depth=0.2, glide=0.03, noise=0.06,
         nband=(80.0, 600.0), ntau=0.008, roll_at=1.2, roll_rate=14.0, cap=2.5, norm=0.7)
_d('drum.kick_felt', 'acoustic kick with felt beater and blanket: round woof', (WO,),
   f0=52.0, pos=0.1, tc=4e-3, tau=0.18, glide=0.5, glide_tau=0.03, noise=0.15,
   nband=(60.0, 900.0), ntau=0.006, sat=0.3, cap=1.2)
_d('drum.kick_modern', 'modern produced kick: pitch-swept thump, beater click, saturated', (SP, WO),
   f0=47.0, pos=0.05, maxr=2.5, tc=1e-3, tau=0.3, glide=2.2, glide_tau=0.028, noise=0.5,
   nband=(2000.0, 9000.0), ntau=0.0015, sat=1.5, cap=1.5)
_d('drum.water_drum', 'water drum: half calabash floating in a basin, wobbling liquid dum', (AF, J, WO),
   f0=95.0, ratios=(1.0, 1.6, 2.3), amps=(1.0, 0.2, 0.1), tau=0.25, qexp=1.0, tc=2e-3,
   glide=0.03, wobble=(0.05, 6.5, 0.25), shell=((520.0, 0.25, 0.03), (900.0, 0.15, 0.02),
                                                 (1450.0, 0.08, 0.015)),
   noise=0.15, nband=(100.0, 700.0), slosh=0.6, cap=1.5)


# ---------------------------------------------------------------- shakers & rattles
def _p(name, desc, tags, fn=m_particles, **kw):
    register(name, fn, 'perc', pitched=False, tags=tags + (IN,), desc=desc, **kw)


_p('perc.shaker_egg', 'egg shaker: plastic shell, fine beads, one crisp shake', (WO, LA),
   res=((4500.0, 3.0, 1.0), (7000.0, 5.0, 0.6)), beads=35, shake=0.06, att=0.012, hp_f=2000.0,
   tick=0.4)
_p('perc.caxixi', 'caxixi: woven basket of seeds, soft wicker swish then a hard gourd chick', (LA, AF, WO),
   res=((2200.0, 2.0, 1.0), (3500.0, 3.0, 0.6)), beads=25, shake=0.07, att=0.02, hp_f=1500.0,
   tick=0.25, bottom=(0.06, 1200.0, 1.4))
_p('perc.ganza', 'ganza: metal tube shaker, bright metallic samba hiss', (LA, WO),
   res=((6500.0, 12.0, 1.0), (9000.0, 8.0, 0.7), (3800.0, 6.0, 0.4)), beads=60, shake=0.09,
   att=0.015, hp_f=3000.0, tick=0.5)
_p('perc.seed_rattle', 'seed-pod rattle shaken for the gate length: clacky woody seeds', (J, AF, WO),
   res=((1500.0, 3.0, 1.0), (2800.0, 4.0, 0.6)), beads=12, att=0.05, held=True, sustain=True,
   shake_rate=6.0, release=0.08, hp_f=1000.0, tick=0.1, grit=0.6, stereo_w=0.4)
_p('perc.shekere', 'shekere: bead net slapped against a big gourd, rattling thump', (AF, J, WO),
   res=((3000.0, 2.0, 1.0), (5500.0, 3.0, 0.5)), beads=45, shake=0.11, att=0.01, hp_f=2000.0,
   tick=0.3, body=(180.0, 0.7, 0.07))
_p('perc.cabasa', 'cabasa: steel bead chains twisted on a ridged cylinder, tight chk', (LA, WO),
   res=((5000.0, 2.0, 1.0), (9000.0, 3.0, 0.7)), beads=90, shake=0.05, att=0.02, hp_f=3500.0,
   tick=0.6, swish=1.0)
_p('perc.tambourine', 'tambourine: jingles clash on a small goatskin frame', (WO, CE, OR),
   res=((4200.0, 30.0, 1.0), (6100.0, 25.0, 0.8), (7900.0, 30.0, 0.6), (10500.0, 20.0, 0.4)),
   beads=30, shake=0.08, att=0.004, hp_f=5000.0, tick=0.2, body=(240.0, 0.25, 0.05), grit=0.7)
_p('perc.rainstick', 'rainstick: pebbles trickle through cactus spines for the gate length', (J, LA, WO, SP),
   res=((2200.0, 15.0, 1.0), (3400.0, 18.0, 0.8), (5100.0, 20.0, 0.6), (7600.0, 25.0, 0.4)),
   beads=50, att=0.15, held=True, sustain=True, release=0.35, hp_f=2500.0, tick=0.06, grit=1.5,
   stereo_w=0.6, ref_dur=2.0)
_p('perc.guiro', 'guiro: stick scraped along a ridged gourd for the gate length', (LA, WO),
   fn=m_scrape, sustain=True, ref_dur=0.4)

# ---------------------------------------------------------------- wood & metal hits
_p('perc.claves', 'claves: two rosewood sticks, a pure ringing click over cupped hands', (LA, WO),
   fn=m_unp, model=m_bell, f0=2500.0,
   partials=(P(1.0, 1.0, 1.0), P(2.71, 0.12, 0.4), P(0.36, 0.08, 0.6)),
   tau=0.06, slope=0.0, tau_lim=(0.03, 0.2), tc=0.15e-3, noise=0.1, nband=(2000.0, 8000.0),
   ntau=0.001, cap=1.0, crest=6.0)
_p('perc.woodblock', 'woodblock: hollow slotted block, dry hollow tok', (OR, WO, AS),
   fn=m_unp, model=m_bell, f0=900.0,
   partials=(P(1.0, 1.0, 1.0), P(1.71, 0.45, 0.5), P(2.6, 0.2, 0.35), P(4.1, 0.08, 0.2)),
   tau=0.05, slope=0.0, tau_lim=(0.02, 0.2), tc=0.4e-3, noise=0.3, nband=(1500.0, 7000.0),
   ntau=0.0015, cap=1.0, crest=6.0)
register('perc.temple_block', m_bell, 'perc', lo='C4', hi='C6', tags=(AS, WO),
         desc='temple blocks (mokugyo): round carved hollow skulls, warm pitched knock',
         partials=(P(1.0, 1.0, 1.0), P(1.53, 0.25, 0.4), P(2.32, 0.18, 0.3), P(3.6, 0.06, 0.2)),
         tau=0.12, tau_ref=523.0, slope=0.3, tau_lim=(0.05, 0.3), tc=1.0e-3, noise=0.2,
         nband=(800.0, 5000.0), ntau=0.002, cap=1.5, crest=6.0)
_AGO = (P(1.0, 1.0, 1.0), P(1.43, 0.3, 0.6), P(2.48, 0.35, 0.45), P(3.08, 0.2, 0.35),
        P(4.4, 0.1, 0.25), P(5.9, 0.05, 0.2))
_p('perc.agogo_low', 'agogo low bell: hand-forged iron cone, stick-struck clank', (LA, AF, WO),
   fn=m_unp, model=m_bell, f0=620.0, partials=_AGO, tau=0.35, slope=0.0, tau_lim=(0.1, 1.0),
   tc=0.35e-3, noise=0.2, nband=(2000.0, 9000.0), ntau=0.0015, cap=2.0)
_p('perc.agogo_high', 'agogo high bell: the smaller cone, a fourth above', (LA, AF, WO),
   fn=m_unp, model=m_bell, f0=830.0, partials=_AGO, tau=0.3, slope=0.0, tau_lim=(0.1, 1.0),
   tc=0.35e-3, noise=0.2, nband=(2500.0, 10000.0), ntau=0.0015, cap=2.0)
_p('perc.cowbell', 'cowbell: flat sheet-iron bell, two clanging modes, hand-muted', (LA, WO),
   fn=m_unp, model=m_bell, f0=562.0,
   partials=(P(1.0, 1.0, 1.0), P(1.504, 0.8, 0.8), P(2.62, 0.25, 0.4), P(3.6, 0.12, 0.3),
             P(4.9, 0.05, 0.2)),
   tau=0.15, slope=0.0, tau_lim=(0.05, 0.5), tc=0.4e-3, noise=0.25, nband=(1500.0, 8000.0),
   ntau=0.0015, cap=1.5)
_p('perc.triangle', 'triangle: bent steel rod and beater, a long silvery shimmer', (OR, CE, WO),
   fn=m_unp, model=m_bell, f0=440.0,
   partials=tuple(P(r * (1 + d), a, tm) for r, a, tm in
                  [(1.0, 0.25, 1.0), (2.756, 0.5, 0.9), (5.404, 0.8, 0.8), (8.933, 1.0, 0.7),
                   (13.344, 0.9, 0.6), (18.638, 0.7, 0.5), (24.81, 0.45, 0.4)]
                  for d in (0.0, 0.013)),
   tau=1.8, slope=0.0, tau_lim=(0.5, 4.0), tc=0.05e-3, noise=0.08, nband=(5000.0, 15000.0),
   ntau=0.002, cap=7.0, mics=0.2)
_p('perc.finger_cymbals', 'finger cymbals (zills): two little brass cups kissed rim to rim', (WO, AS, SP),
   fn=m_unp, model=m_bell, f0=2650.0,
   partials=(P(1.0, 1.0, 1.0, 21.0, 0.5), P(2.16, 0.5, 0.7, 34.0, 0.4), P(3.9, 0.25, 0.5),
             P(5.7, 0.1, 0.35)),
   tau=1.7, slope=0.0, tau_lim=(0.6, 3.0), tc=0.08e-3, noise=0.1, nband=(4000.0, 14000.0),
   ntau=0.002, cap=6.0, mics=0.2)
_p('perc.cymbal_ride', 'ride cymbal: stick ping on the bow, a long glassy wash', (OR, WO),
   fn=m_cymbal, ping=0.6, ping_f=620.0, wash=0.6, tau=2.6, tslope=0.4, bloom=0.02,
   tc=0.15e-3, tilt=0.1, cascade=0.3)
_p('perc.cymbal_crash', 'crash cymbal: big bloom of noise through a dense metal cloud', (OR, WO, SP),
   fn=m_cymbal, wash=1.3, tau=1.6, tslope=0.4, bloom=0.035, tc=0.12e-3, nmodes=30, tilt=0.3,
   cascade=0.8)
_p('perc.cymbal_swell', 'suspended cymbal swell: soft-mallet roll growing for the gate', (OR, OL, SP),
   fn=m_cymbal, sustain=True, ref_dur=2.0, roll=True, roll_rate=15.0, wash=1.0, tau=2.0,
   tslope=0.45, bloom=0.03, tc=1.0e-3, nmodes=24, cascade=1.2, tilt=0.3)
_p('perc.hand_claps', 'hand claps: a few people clapping together in a small room', (WO, LA, CE),
   fn=m_claps)
