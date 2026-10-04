"""
Canopy Orchestra — tuned idiophones: families 'mallet', 'bell', 'metal'.

Everything is modal synthesis with (approximately) measured partial ratios:
struck bars (marimba 1:4:10, xylophone 1:3:6, vibraphone 1:4:10, free-free
steel bars 1:2.76:5.40), cantilever tines (1:6.27:17.55), free-free tubes
(chimes: strike tone = half of mode 4), church-bell partial series, boss gongs
with ombak beating and bloom, rubbed bowls/glasses with rotating beats.

Physical ideas used throughout
  * mallet contact time tc -> low-pass weighting of modal amplitudes
    (half-sine force pulse spectrum). Velocity shortens tc (Hertzian contact),
    so louder = brighter, plus more contact noise.
  * per-mode decay: tau_k = T1 * (f1/fk)^qexp (material losses grow with freq).
  * resonator tubes / gourds as an extra slowly-rising mode near the fundamental.
  * mirliton / rattle buzz as a threshold nonlinearity on the resonator signal
    (so soft notes are clean and loud notes buzz, as on a real balafon).
  * gamelan ombak = a pengumbang/pengisep pair detuned by a constant beat rate.

Shared engine helpers (_bank, _dsin, _hw, ...) are also imported by inst_drums.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
import scipy.fft as _sfft

from .core import *  # noqa: F401,F403
from .core import TWOPI, SR, bp, lp, hp, reson, env_perc, softclip, register, mtof

C4 = 261.6256

# =========================================================================== engine
_BLK = 512


def _dsin(m, f, tau, sr, ph=0.0):
    """exp(-t/tau) * sin(2 pi f t + ph), m samples, via a block outer product of
    complex exponentials (imag part only, float32 - several times faster than
    sin()*exp())."""
    a = -1.0 / (tau * sr) if tau else 0.0
    s = complex(a, TWOPI * f / sr)
    nb = -(-m // _BLK)
    blk = np.exp(s * np.arange(_BLK) + 1j * ph)
    big = np.exp(s * _BLK * np.arange(nb))
    out = np.multiply.outer(big.real.astype(np.float32), blk.imag.astype(np.float32))
    out += np.multiply.outer(big.imag.astype(np.float32), blk.real.astype(np.float32))
    return out.ravel()[:m]


def _arr(v, k, d):
    if v is None:
        return np.full(k, float(d))
    return np.array(np.broadcast_to(np.asarray(v, dtype=float), (k,)))


def _bank(n, sr, fr, am, ta, rng=None, ph=None, rise=None, beat=None, bdepth=None,
          fmul=None, floor=8.0):
    """Sum of decaying modes. fr/ta/rise/beat/bdepth: (k,). am: (k,) -> mono
    output or (2, k) -> stereo output (per-mode mic gains). Each mode is only
    computed for floor*tau seconds (-70 dB at floor=8). fmul: optional (n,)
    frequency multiplier (glides / bends) -> slower sin() path."""
    fr = np.atleast_1d(np.asarray(fr, dtype=float))
    k = len(fr)
    am = np.asarray(am, dtype=float)
    st = am.ndim == 2
    if not st:
        am = np.broadcast_to(am, (k,))
    ta = _arr(ta, k, 1.0)
    rise = _arr(rise, k, 0.0)
    beat = _arr(beat, k, 0.0)
    bdepth = _arr(bdepth, k, 0.0)
    if ph is None:
        ph = np.zeros(k)
    elif isinstance(ph, str):
        ph = rng.uniform(0, TWOPI, k)
    else:
        ph = _arr(ph, k, 0.0)
    y = np.zeros((2, n)) if st else np.zeros(n)
    if fmul is not None:
        fm = np.broadcast_to(np.asarray(fmul, dtype=float), (n,))
        G = TWOPI * np.cumsum(fm) / sr
        gmax = float(fm.max())
    else:
        G = None
        gmax = 1.0
    for i in range(k):
        f = fr[i]
        a = am[:, i] if st else am[i]
        if f <= 0 or f * gmax >= 0.46 * sr or not np.any(a):
            continue
        tau = max(ta[i], 1e-4)
        m = min(n, int((floor * tau + 3.0 * rise[i]) * sr) + 16)
        if m <= 0:
            continue
        bt, bd = beat[i], bdepth[i]
        if G is None:
            comps = [(f, 1.0, ph[i])]
            if bt > 0 and bd > 0:
                # AM beat == three exact partials (carrier + two sidebands)
                p0 = rng.uniform(0, TWOPI) if rng is not None else 0.0
                comps = [(f, 1.0 - 0.5 * bd, ph[i]), (f + bt, 0.25 * bd, ph[i] + p0),
                         (f - bt, 0.25 * bd, ph[i] - p0)]
            seg = None
            kr = int(9 * rise[i] * sr) + 1 if rise[i] > 0 else 0
            for (fc, ac, pc) in comps:
                c = _dsin(m, fc, tau, sr, pc)
                if kr > 0.4 * m:
                    # (1 - e^{-t/r}) e^{-t/tau} = e^{-t/tau} - e^{-t (1/tau + 1/r)}
                    c -= _dsin(m, fc, 1.0 / (1.0 / tau + 1.0 / rise[i]), sr, pc)
                seg = ac * c if seg is None else seg + ac * c
            if 0 < kr <= 0.4 * m:
                seg[:kr] *= -np.expm1(-np.arange(kr) / (sr * rise[i]))
        else:
            tt = np.arange(m) / sr
            seg = np.sin(f * G[:m] + ph[i]) * np.exp(-tt / tau)
            if bt > 0 and bd > 0:
                p0 = rng.uniform(0, TWOPI) if rng is not None else 0.0
                seg *= 1.0 - bd * (0.5 - 0.5 * np.cos(TWOPI * bt * tt + p0))
            if rise[i] > 0:
                kr = min(m, int(9 * rise[i] * sr) + 1)
                seg[:kr] *= -np.expm1(-np.arange(kr) / (sr * rise[i]))
        if st:
            y[0, :m] += a[0] * seg
            y[1, :m] += a[1] * seg
        else:
            y[:m] += a * seg
    return y


def _hw(fr, tc):
    """mallet weighting: approx. spectrum of a half-sine force pulse of length tc."""
    x = np.asarray(fr, dtype=float) * tc
    return 1.0 / np.sqrt(1.0 + (x / 0.7) ** 4)


def _tc(tc, vel):
    """contact time shortens with velocity -> brighter when louder."""
    return tc * (1.8 - 1.2 * float(np.clip(vel, 0, 1)))


def _amp(vel):
    return float(np.clip(vel, 0.0, 1.0)) ** 0.9


def _burst(n, sr, rng, lo, hi, tau, att=0.0003, order=2):
    """normalised band-limited noise burst (contact / clack / scrape)."""
    m = int(min(n, (att + 8 * tau) * sr + 8))
    out = np.zeros(n)
    if m < 8:
        return out
    x = bp(rng.standard_normal(m + 256), lo, hi, order, sr)[256:]
    x /= x.std() + 1e-12
    out[:m] = x * env_perc(m, att, tau, sr)
    return out


def _buzz(drive, th, sr, rng, band, fmem, rasp=0.6):
    """mirliton / rattle: a membrane that slaps when the drive exceeds a
    threshold (absolute -> velocity dependent). Half-wave excess gives a
    period-synchronous rasp, noise modulation makes it papery."""
    r = np.maximum(drive - th, 0.0)
    if not np.any(r):
        return np.zeros_like(drive)
    r = r - lp(r, 30.0, 1, sr)
    nz = lp(rng.standard_normal(len(r)), 3500.0, 1, sr)
    nz /= nz.std() + 1e-12
    x = r * (1.0 + rasp * nz)
    return bp(x, band[0], band[1], 2, sr) + 0.8 * reson(x, fmem, 3.0, sr)


def _slow_noise(n, sr, fc, rng, dec=128):
    """unit-variance smooth random control signal (band-limited to ~fc Hz),
    generated at a decimated rate and linearly interpolated (cheap)."""
    m = n // dec + 8
    srd = sr / dec
    z = rng.standard_normal(m + 64)
    b, a = signal.butter(2, min(fc, 0.45 * srd), fs=srd)
    z = signal.lfilter(b, a, z)[64:]
    z /= z.std() + 1e-12
    return np.interp(np.arange(n) / dec, np.arange(m), z)


_TFN = 1024
_TFWIN = np.sin(np.pi * (np.arange(_TFN) + 0.5) / _TFN)


def _tfnoise(n, sr, rng, magfn, nch=1):
    """noise shaped by an arbitrary time-frequency magnitude magfn(t, f) where
    t: (frames, 1) seconds, f: (1, bins) Hz -> (frames, bins). Random-phase
    STFT synthesis with a power-complementary sine window (hop N/2): cheap
    'evolving spectrum' noise for cymbals, gongs, shakers and rain. Output
    has ~unit RMS where magfn == 1."""
    N, hop = _TFN, _TFN // 2
    nf = n // hop + 2
    tf = (np.arange(nf) * hop / sr)[:, None]
    fb = (np.arange(N // 2 + 1) * sr / N)[None, :]
    M = np.asarray(magfn(tf, fb), dtype=float)
    M = np.broadcast_to(M, (nf, N // 2 + 1))
    sh = (nch, nf, N // 2 + 1)
    Z = np.empty(sh, np.complex64)
    Z.real = rng.standard_normal(sh, dtype=np.float32)
    Z.imag = rng.standard_normal(sh, dtype=np.float32)
    Z *= M[None].astype(np.float32)
    fr = _sfft.irfft(Z, n=N, axis=-1)
    fr *= (_TFWIN * np.sqrt(N / 2.0)).astype(np.float32)
    out = np.zeros((nch, nf + 1, hop), np.float32)
    out[:, :nf] += fr[..., :hop]
    out[:, 1:] += fr[..., hop:]
    out = out.reshape(nch, -1)[:, hop // 2: hop // 2 + n]
    return out[0] if nch == 1 else out


def _envf(x, sr, fc=40.0):
    return lp(np.abs(x), fc, 1, sr)


def _finish(y, sr, cap=None, db=-66.0):
    """cap length, trim the inaudible tail, and fade the end so every note
    decays to true silence."""
    y = np.asarray(y, dtype=float)
    if cap is not None:
        y = y[..., :int(cap * sr)]
    m = np.abs(y) if y.ndim == 1 else np.abs(y).max(axis=0)
    pk = float(m.max()) if m.size else 0.0
    if pk <= 0:
        return y[..., :max(1, int(0.05 * sr))]
    idx = np.flatnonzero(m > pk * 10 ** (db / 20))
    end = min(m.size, int(idx[-1]) + int(0.03 * sr))
    y = y[..., :end].copy()
    # if the note is still audible at the end (capped), fade it over longer
    w = int(0.1 * sr)
    lvl = float(m[max(0, end - w):end].max()) / pk
    if lvl > 10 ** (-40 / 20):
        k = int(min(end * 0.35, 2.5 * sr))
    else:
        k = int(min(end * 0.2, 0.03 * sr))
    if k > 1:
        y[..., -k:] *= (0.5 + 0.5 * np.cos(np.pi * np.arange(k) / k)) ** 1.5
    return y


def _tame(y, sr, crest=5.0):
    """gentle transient limiter (soft knee) so very short percussive sounds
    keep a sensible peak-to-loudness ratio after calibration (iterated, since
    limiting the peaks also lowers the loudness)."""
    k = int(0.25 * sr)
    for _ in range(4):
        m = y if y.ndim == 1 else 0.5 * (y[0] + y[1])
        if len(m) > k:
            c = np.cumsum(np.concatenate([[0.0], m ** 2]))
            loud = np.sqrt(max(((c[k:] - c[:-k]) / k).max(), 0.0))
        else:
            loud = np.sqrt(np.mean(m ** 2))
        pk = np.abs(y).max()
        if loud <= 0 or pk <= crest * loud * 1.02:
            return y
        thr = crest * loud * 0.7
        knee = thr * 0.4
        a = np.abs(y)
        y = np.where(a > thr, np.sign(y) * (thr + knee * np.tanh((a - thr) / knee)), y)
    return y


def _pan2(g, pan):
    a = (pan + 1) * np.pi / 4
    return np.array([g * np.cos(a), g * np.sin(a)])


def _detune(f, rng, cents):
    return f * 2 ** (rng.normal(0.0, cents) / 1200.0)


# =========================================================================== models
def m_bar(f, dur, vel, rng, sr, ratios=(1.0, 3.98, 9.88), amps=(1.0, 0.5, 0.2), jit=0.0,
          tau=1.4, tau_ref=220.0, slope=0.65, tau_lim=(0.1, 4.0), qexp=1.1, tc=1.2e-3,
          tube=0.0, tube_tau=1.3, tube_rise=0.012, tube3=0.0, extra=(),
          noise=0.1, nband=(300.0, 3000.0), ntau=0.003,
          buzz=0.0, buzz_th=0.2, buzz_band=(1200.0, 5000.0), buzz_f=2200.0, rasp=0.6,
          ombak=0.0, motor=0.0, motor_depth=0.0, damper=None, cap=8.0, detune=3.0):
    """struck bar (+ optional resonator tube/gourd, mirliton buzz, ombak pair,
    vibraphone motor, damper pedal)."""
    f = _detune(f, rng, detune)
    A = _amp(vel)
    tce = _tc(tc, vel)
    T1 = float(np.clip(tau * (tau_ref / f) ** slope, *tau_lim)) * rng.uniform(0.93, 1.07)
    L = max(8.0 * T1 * max(tube_tau, 1.0), 0.25)
    if damper is not None:
        L = min(L, dur + 8.0 * damper)
    L = min(L, cap)
    n = int(L * sr)
    t = np.arange(n) / sr
    base_rat = np.array(ratios, dtype=float)

    def render(fx):
        rat = base_rat.copy()
        if jit:
            rat[1:] *= 1.0 + rng.uniform(-1, 1, len(rat) - 1) * np.broadcast_to(jit, (len(rat),))[1:]
        fr = fx * rat
        am = np.asarray(amps, dtype=float) * _hw(fr, tce) * rng.uniform(0.9, 1.1, len(rat))
        ta = T1 * (fx / fr) ** qexp
        ffr, fam, fta, frise = [fr[0]], [am[0]], [ta[0]], [0.0]
        if tube:
            ffr.append(fx * 1.0012)
            fam.append(tube * _hw(fx, tce))
            fta.append(T1 * tube_tau)
            frise.append(tube_rise)
        yf = _bank(n, sr, ffr, fam, fta, rng, rise=frise)
        rr, ra, rt, rs = list(fr[1:]), list(am[1:]), list(ta[1:]), [0.0] * (len(fr) - 1)
        for (er, ea, em) in extra:
            rr.append(fx * er)
            ra.append(ea * _hw(fx * er, tce))
            rt.append(T1 * em * (1.0 / er) ** qexp)
            rs.append(0.0)
        if tube3:
            rr.append(fx * 3.0)
            ra.append(tube3 * _hw(fx * 3, tce))
            rt.append(T1 * 0.45)
            rs.append(0.006)
        yr = _bank(n, sr, rr, ra, rt, rng, rise=rs) if rr else np.zeros(n)
        return yf, yr

    def finish_one(yf, yr):
        if motor > 0 and motor_depth > 0:
            p0 = rng.uniform(0, TWOPI)
            lfo = 0.5 - 0.5 * np.cos(TWOPI * motor * t + p0)
            yf = yf * (1.0 - motor_depth * lfo)
            yr = yr * (1.0 - 0.3 * motor_depth * lfo)
        y = A * (yf + yr)
        if buzz > 0:
            d = A * yf / (np.abs(yf).max() + 1e-12)
            y = y + buzz * A * _buzz(d, buzz_th, sr, rng, buzz_band, buzz_f, rasp)
        if noise > 0:
            y = y + noise * A * (0.35 + 0.65 * vel) * _burst(n, sr, rng, nband[0],
                                                               nband[1] * (0.6 + 0.6 * vel), ntau)
        return y

    if ombak > 0:
        outs = []
        for sgn in (-0.5, 0.5):
            yf, yr = render(f + sgn * ombak)
            outs.append(finish_one(yf, yr))
        y = np.stack([0.8 * outs[0] + 0.45 * outs[1], 0.45 * outs[0] + 0.8 * outs[1]])
    else:
        y = finish_one(*render(f))
    if damper is not None and dur < L:
        k = int(dur * sr)
        e = np.ones(n)
        e[k:] = np.exp(-(t[k:] - dur) / damper)
        y = y * e
    return _finish(y, sr, cap)


def m_tine(f, dur, vel, rng, sr, ratios=(1.0, 6.267, 17.55), amps=(1.0, 0.35, 0.08),
           tau=1.3, tau_ref=440.0, slope=0.5, tau_lim=(0.3, 3.0), qexp=1.0, tc=0.6e-3,
           oct2=0.12, box=((260.0, 2.5, 0.35), (820.0, 5.0, 0.12)), noise=0.12,
           nband=(1500.0, 7000.0), rattle=0.0, rattle_th=0.12,
           rattle_res=(3400.0, 5200.0, 7300.0), deze=0.0, cap=6.0, detune=4.0, extra=()):
    """plucked cantilever tine on a sounding board (kalimba, mbira, music box)."""
    f = _detune(f, rng, detune)
    A = _amp(vel)
    tce = _tc(tc, vel)
    T1 = float(np.clip(tau * (tau_ref / f) ** slope, *tau_lim)) * rng.uniform(0.92, 1.08)
    n = int(min(cap, 8 * T1 + 0.05) * sr)
    rat = np.array(ratios, dtype=float) * (1 + np.r_[0, rng.uniform(-0.01, 0.01, len(ratios) - 1)])
    fr = f * rat
    am = np.asarray(amps, dtype=float) * _hw(fr, tce) * rng.uniform(0.85, 1.15, len(fr))
    ta = T1 * (f / fr) ** qexp
    y1 = _bank(n, sr, fr[:1], am[:1], ta[:1], rng)
    yr = _bank(n, sr, fr[1:], am[1:], ta[1:], rng)
    if extra:
        yr += _bank(n, sr, [f * e[0] for e in extra], [e[1] * _hw(f * e[0], tce) for e in extra],
                    [T1 * e[2] for e in extra], rng)
    y = y1 + yr
    if oct2:
        q = y1 * y1
        y = y + oct2 * (0.6 + 0.6 * vel) * hp(q, 40.0, 2, sr)
    y = A * y
    if noise:
        y = y + noise * A * (0.3 + 0.7 * vel) * _burst(n, sr, rng, nband[0], nband[1], 0.0015)
    if box:
        y = y + sum(g * reson(y, fb, q, sr) for fb, q, g in box)
    if rattle > 0:
        e = _envf(y1, sr, 40.0)
        d = A * e / (e.max() + 1e-12)
        g = np.maximum(d - rattle_th, 0.0)
        nz = rng.standard_normal(n)
        siz = hp(nz, 2200.0, 2, sr) * 0.35 + sum(reson(nz, fr_ * (1 + rng.uniform(-.05, .05)), 7.0, sr)
                                                 for fr_ in rattle_res)
        sync = nz * np.maximum(A * y1 / (np.abs(y1).max() + 1e-12) - 0.25, 0.0)
        y = y + rattle * (g * siz * 0.6 + 0.8 * bp(sync, 1200.0, 6000.0, 2, sr))
    if deze:
        y = y + deze * reson(y, 190.0, 1.6, sr) + 0.3 * deze * reson(y, 420.0, 3.0, sr)
    return _finish(y, sr, cap)


def m_log(f, dur, vel, rng, sr, ratios=(1.0, 2.43, 3.9, 5.6), amps=(1.0, 0.3, 0.12, 0.05),
          jit=0.0, tau=0.25, tau_ref=220.0, slope=0.4, tau_lim=(0.04, 1.5), qexp=1.0, tc=2e-3,
          glide=0.02, glide_tau=0.01, cav=0.0, cav_hz=None, cav_ratio=0.5, cav_tau=0.3,
          cav_rise=0.005, wall=(), noise=0.2, nband=(800.0, 4000.0), ntau=0.003, thud=0.0,
          cap=4.0, detune=5.0):
    """wooden tongue / slit / log drum, bamboo tube: wood modes + cavity air
    resonance + a small strike pitch blip."""
    f = _detune(f, rng, detune)
    A = _amp(vel)
    tce = _tc(tc, vel)
    T1 = float(np.clip(tau * (tau_ref / f) ** slope, *tau_lim)) * rng.uniform(0.9, 1.1)
    rat = np.array(ratios, dtype=float)
    if jit:
        rat[1:] *= 1 + rng.uniform(-jit, jit, len(rat) - 1)
    fr = list(f * rat)
    am = list(np.asarray(amps) * _hw(f * rat, tce) * rng.uniform(0.85, 1.15, len(rat)))
    ta = list(T1 * (1.0 / rat) ** qexp)
    rs = [0.0] * len(fr)
    if cav:
        fc = (cav_hz * (f / 220.0) ** 0.3) if cav_hz else f * cav_ratio
        fr.append(fc)
        am.append(cav)
        ta.append(cav_tau)
        rs.append(cav_rise)
    for hz, a, tw in wall:
        fw = hz * (f / 220.0) ** 0.25 * rng.uniform(0.97, 1.03)
        fr.append(fw)
        am.append(a * _hw(fw, tce))
        ta.append(tw)
        rs.append(0.0)
    L = min(cap, max(8 * max(ta), 0.2))
    n = int(L * sr)
    t = np.arange(n) / sr
    fmul = 1.0 + glide * (0.3 + 0.7 * vel) * np.exp(-t / glide_tau) if glide else None
    y = A * _bank(n, sr, fr, am, ta, rng, rise=rs, fmul=fmul)
    if noise:
        y += noise * A * (0.3 + 0.7 * vel) * _burst(n, sr, rng, nband[0], nband[1] * (0.6 + 0.6 * vel), ntau)
    if thud:
        y += thud * A * _burst(n, sr, rng, 40.0, 260.0, 0.018, att=0.001)
    return _finish(y, sr, cap)


def m_angklung(f, dur, vel, rng, sr, rate=10.5, tubes=((1.0, 1.0), (2.0, 0.55)), tau=0.3,
               tau_ref=523.0, slope=0.35, upper=0.18, knock=0.25, cap=None):
    """shaken bamboo tubes: every shake knocks each tube against the frame;
    the gate length sets how long the tremolo lasts."""
    f = _detune(f, rng, 4.0)
    A = _amp(vel)
    T = float(np.clip(tau * (tau_ref / f) ** slope, 0.08, 0.8))
    g = max(dur, 0.12)
    r = rate * rng.uniform(0.92, 1.08) * (0.9 + 0.2 * vel)
    times = np.arange(0.0, g, 1.0 / r)
    times = times + rng.normal(0, 0.005, len(times))
    times[0] = 0.0
    times = np.clip(times, 0, None)
    hit = rng.uniform(0.75, 1.05, len(times)) * np.where(np.arange(len(times)) % 2, 0.72, 1.0)
    hit[0] *= 1.15
    L = times[-1] + 8 * T + 0.05
    if cap:
        L = min(L, cap)
    n = int(L * sr)
    y = np.zeros(n)
    kn = np.zeros(n)
    for j, (ratio, ga) in enumerate(tubes):
        ft = f * ratio
        if ft > 0.45 * sr:
            continue
        mi = int(8 * T * sr)
        ir = _bank(mi, sr, [ft, ft * 3.02, ft * 5.1],
                   [1.0, upper * (0.5 + vel), 0.4 * upper * vel],
                   [T, T * 0.3, T * 0.12], rng)
        tr = np.zeros(n)
        off = 0.0 if j == 0 else rng.uniform(0.004, 0.012)
        idx = np.clip(((times + off) * sr).astype(int), 0, n - 1)
        np.add.at(tr, idx, ga * hit * rng.uniform(0.85, 1.1, len(idx)))
        y += signal.fftconvolve(tr, ir)[:n]
        kn += tr
    if knock:
        ke = signal.fftconvolve(kn, np.exp(-np.arange(int(0.012 * sr)) / (0.0025 * sr)))[:n]
        y += knock * bp(rng.standard_normal(n), 900.0, 4500.0, 2, sr) * ke * 0.5
    return _finish(A * y, sr, cap)


def P(r, a, tm=1.0, beat=0.0, bd=0.0, rise=0.0, c=0):
    """bell partial: ratio, amp, tau multiplier, beat Hz, beat depth, rise s,
    coupled (c=1: excited via nonlinear coupling -> not mallet weighted)."""
    return (r, a, tm, beat, bd, rise, c)


def m_bell(f, dur, vel, rng, sr, partials=(), tau=3.0, tau_ref=262.0, slope=0.5,
           tau_lim=(0.3, 6.0), tc=0.6e-3, jit=0.0, noise=0.15, nband=(2000.0, 10000.0),
           ntau=0.008, cap=9.0, mics=0.0, sat=0.0, detune=2.0, glide=None, cvel=0.0,
           thump=0.0, thump_f=150.0, damper=None, crest=None):
    """generic modal bell/gong/plate: explicit partial table (see P())."""
    f = _detune(f, rng, detune)
    A = _amp(vel)
    tce = _tc(tc, vel)
    T = float(np.clip(tau * (tau_ref / f) ** slope, *tau_lim)) * rng.uniform(0.93, 1.07)
    pt = np.array(partials, dtype=float)
    rat = pt[:, 0] * (1 + rng.uniform(-jit, jit, len(pt)) * (pt[:, 0] != 1.0)) if jit else pt[:, 0]
    fr = f * rat
    w = np.where(pt[:, 6] > 0, (0.25 + vel) ** cvel if cvel else 1.0, _hw(fr, tce))
    am = pt[:, 1] * w * rng.uniform(0.9, 1.1, len(pt))
    ta = T * pt[:, 2] * rng.uniform(0.92, 1.08, len(pt))
    L = min(cap, max(8 * ta.max() + pt[:, 5].max() * 3, 0.3))
    if damper is not None:
        L = min(L, dur + 8 * damper)
    n = int(L * sr)
    t = np.arange(n) / sr
    fmul = None
    if glide is not None:
        st, gt, kind = glide
        sh = -np.expm1(-t / gt) if kind == 'to' else np.exp(-t / gt)
        fmul = 2 ** (st * (0.5 + 0.5 * vel) * sh / 12.0)
    if mics > 0:
        u = rng.uniform(-1, 1, len(pt))
        am = np.stack([am * (1 + mics * u), am * (1 - mics * u)])
    if fmul is None:
        y = _bank(n, sr, fr, am, ta, rng, rise=pt[:, 5], beat=pt[:, 3], bdepth=pt[:, 4])
    else:
        # a glide common to all partials == a time warp of the unglided render
        idx = np.cumsum(fmul) - fmul[0]
        nw = int(idx[-1]) + 2
        y0 = _bank(nw, sr, fr, am, ta, rng, rise=pt[:, 5], beat=pt[:, 3], bdepth=pt[:, 4])
        xp = np.arange(nw)
        y = np.stack([np.interp(idx, xp, c) for c in y0]) if y0.ndim == 2 else np.interp(idx, xp, y0)
    y = A * y
    if noise:
        nz = noise * A * (0.3 + 0.7 * vel) * _burst(n, sr, rng, nband[0], nband[1] * (0.6 + 0.6 * vel), ntau)
        y = y + (np.stack([nz, np.roll(nz, 7)]) if y.ndim == 2 else nz)
    if thump:
        th = thump * A * _bank(n, sr, [thump_f], [1.0], [0.06], rng)
        y = y + (th[None, :] if y.ndim == 2 else th)
    if sat:
        # bronze/tape-like saturation: unity small-signal gain, loud peaks squashed
        pk1 = np.abs(y).max() / max(A, 1e-3) + 1e-12
        gk = 2.0 * sat
        y = np.tanh(gk * y / pk1) * pk1 / gk
    if damper is not None and dur < L:
        k = int(dur * sr)
        e = np.ones(n)
        e[k:] = np.exp(-(t[k:] - dur) / damper)
        y = y * e
    if crest:
        y = _tame(y, sr, crest)
    return _finish(y, sr, cap)


def m_rubbed(f, dur, vel, rng, sr, ratios=(1.0, 2.83, 5.42, 8.77), strike=0.35,
             rub=(1.0, 0.12, 0.02, 0.0), att=0.8, tau=4.0, tau_ref=262.0, slope=0.5,
             tau_lim=(0.5, 8.0), beat=2.0, bdepth=0.5, chatter=0.0, squeak=0.0,
             wobble=3.0, tc=2e-3, cap=10.0, jit=0.01):
    """rubbed rim (singing bowl, glass harp): a struck onset plus friction-driven
    sustain of the lowest modes while the gate is held; degenerate mode pairs
    give the rotating 'wah' beat; then free ring-out."""
    f = _detune(f, rng, 3.0)
    A = _amp(vel)
    T = float(np.clip(tau * (tau_ref / f) ** slope, *tau_lim)) * rng.uniform(0.92, 1.08)
    rat = np.array(ratios, dtype=float) * (1 + np.r_[0, rng.uniform(-jit, jit, len(ratios) - 1)])
    fr = f * rat
    ta = T * (f / fr) ** 0.7
    g = max(dur, 0.05)
    L = min(cap, g + 7 * T)
    n = int(L * sr)
    t = np.arange(n) / sr
    # struck onset
    tce = _tc(tc, vel)
    sa = strike * np.array([1.0, 0.5, 0.25, 0.12][:len(fr)]) * _hw(fr, tce)
    y = _bank(n, sr, fr, sa, ta, rng, beat=beat * (fr / f) ** 0.5, bdepth=0.6 * bdepth)
    # friction sustain - all slow controls on a decimated grid (dec samples)
    dec = 32
    kd = int(g * sr)
    td = np.arange(n // dec + 2) * dec / sr
    build = -np.expm1(-td / att)
    rubenv = np.where(td < g, build, -np.expm1(-g / att))
    fl = _slow_noise(len(td), sr / dec, 6.0, rng, 4)
    xi = np.arange(n) / dec
    for i, ra in enumerate(rub[:len(fr)]):
        if ra <= 0 or fr[i] > 0.45 * sr:
            continue
        rel = np.exp(-np.maximum(td - g, 0.0) / ta[i])
        cents = np.clip(wobble * _slow_noise(len(td), sr / dec, 2.0, rng, 8), -3 * wobble, 3 * wobble)
        phd = TWOPI * np.cumsum(fr[i] * 2 ** (cents / 1200.0)) * dec / sr + rng.uniform(0, TWOPI)
        bt = 1.0 - bdepth * (0.5 - 0.5 * np.cos(TWOPI * beat * (fr[i] / f) ** 0.5 * td + rng.uniform(0, 6)))
        ed = ra * (0.5 + 0.5 * vel) ** (i + 0.3) * rubenv * rel * bt * (1 + 0.08 * fl)
        y += np.sin(np.interp(xi, np.arange(len(td)), phd)) * np.interp(xi, np.arange(len(td)), ed)
    kg = min(n, kd + int(0.02 * sr))
    if chatter > 0 and vel > 0.5:
        c = (vel - 0.5) * 2 * chatter
        tg = np.arange(kg) / sr
        nz = bp(rng.standard_normal(kg), fr[1] * 0.9, min(fr[2] * 1.2, 0.45 * sr), 2, sr)
        bt = np.maximum(np.sin(TWOPI * beat * 2 * tg), 0) ** 4
        y[:kg] += c * 0.3 * nz * bt * -np.expm1(-tg / att) * np.exp(-np.maximum(tg - g, 0) / 0.005)
    if squeak > 0:
        tg = np.arange(kg) / sr
        nz = bp(rng.standard_normal(kg), 1800.0, 5500.0, 2, sr)
        fl2 = np.interp(np.arange(kg) / dec, np.arange(len(td)), fl)
        y[:kg] += squeak * vel * 0.05 * nz * -np.expm1(-tg / att) * (1 + fl2 * 0.5) * \
            np.exp(-np.maximum(tg - g, 0) / 0.005)
    y = A * y
    return _finish(y, sr, cap)


def m_waterphone(f, dur, vel, rng, sr, spread=(1.07, 1.21, 1.5, 1.83), bend=110.0,
                 att=0.35, tau=2.0, cap=8.0, bowl=0.25):
    """bowed steel rod on a water-filled resonator: stick-slip tone, sympathetic
    rods, eerie water-tilt pitch bends, metallic bowl modes; stereo."""
    f = _detune(f, rng, 5.0)
    A = _amp(vel)
    g = max(dur, 0.1)
    L = min(cap, g + 7 * tau)
    n = int(L * sr)
    t = np.arange(n) / sr
    # all slow controls on a decimated grid, then interpolated
    dec = 32
    td = np.arange(n // dec + 2) * dec / sr
    xi = np.arange(n) / dec
    xd = np.arange(len(td))
    # water tilt: slow random bend growing in after the onset
    rates = rng.uniform(0.12, 0.6, 3)
    cur = sum(np.sin(TWOPI * r_ * td + rng.uniform(0, 6)) for r_ in rates) / 1.7
    cur = cur * np.clip(td / 0.6, 0, 1) * bend * rng.uniform(0.7, 1.2)
    cur = cur + 6 * np.sin(TWOPI * rng.uniform(4.5, 6.5) * td)
    fmd = 2 ** (cur / 1200.0)
    bowd = -np.expm1(-np.minimum(td, g) / att)
    reld = np.exp(-np.maximum(td - g, 0.0) / (tau * 0.5))
    fl = _slow_noise(len(td), sr / dec, 9.0, rng, 2)
    phd = TWOPI * np.cumsum(f * fmd) * dec / sr
    ph = np.interp(xi, xd, phd)
    fm = np.interp(xi, xd, fmd)
    br = np.interp(xi, xd, bowd * reld)
    e = np.interp(xi, xd, bowd * reld * (1 + 0.15 * fl))
    s1, c1 = np.sin(ph), np.cos(ph)
    tone = (s1 + 0.14 * (2 * s1 * c1) + 0.05 * (3 * s1 - 4 * s1 ** 3)) * e
    # squeal: inharmonic 2nd cantilever mode jumps in when bowing hard
    if vel > 0.4:
        sw = np.interp(xi, xd, np.clip(np.sin(TWOPI * rng.uniform(0.2, 0.5) * td), 0, 1) ** 2)
        tone = tone + 0.12 * (vel - 0.4) * np.sin(6.27 * ph + 1.3) * e * sw
    bown = bp(rng.standard_normal(n), f * 0.8, f * 1.6, 2, sr) * br * 0.05 * (0.4 + vel)
    # sympathetic rods + bowl, all bent by the water
    sym_f = [f * s for s in spread]
    sym_a = rng.uniform(0.08, 0.2, len(spread))
    am2 = np.stack([sym_a * rng.uniform(0.3, 1.0, len(spread)), sym_a * rng.uniform(0.3, 1.0, len(spread))])
    sym = _bank(n, sr, sym_f, am2, rng.uniform(0.6, 1.0, len(spread)) * tau, rng,
                rise=rng.uniform(0.2, 0.6, len(spread)), ph='r', floor=6.5)
    bf = rng.uniform(260, 340) * np.array([1.0, 2.31, 3.92])
    bam = bowl * np.array([[1.0, 0.5, 0.25], [0.6, 0.7, 0.3]])
    fw = fm ** 0.3
    idx = np.cumsum(fw) - fw[0]
    nw = int(idx[-1]) + 2
    bw0 = _bank(nw, sr, bf, bam, [tau * 1.2, tau * 0.8, tau * 0.5], rng, rise=0.3, ph='r', floor=6.5)
    bw = np.stack([np.interp(idx, np.arange(nw), c) for c in bw0])   # water bend as a time warp
    y = np.stack([tone * 0.9 + bown, tone * 0.75 + np.roll(bown, 31)]) + sym + bw * 0.5
    return _finish(A * y, sr, cap)


def m_cascade(f, dur, vel, rng, sr, nb=18, f_hi=6500.0, f_lo=1800.0, span=0.7,
              ratios=(1.0, 2.4, 3.97, 5.6), amps=(1.0, 0.5, 0.3, 0.15), tau=1.3,
              tc=0.25e-3, descend=True, rebounce=0.0, accel=0.3, cap=6.0):
    """bell tree / mark tree: a quick glissando of small bells or rods, panned
    across the stereo field. f retunes the whole cascade (unpitched)."""
    rt = f / C4
    A = _amp(vel)
    tce = _tc(tc, vel)
    freqs = np.geomspace(f_hi, f_lo, nb) * rt * rng.uniform(0.98, 1.02, nb)
    if not descend:
        freqs = freqs[::-1]
    sp = span * (1.25 - 0.5 * vel)
    u = np.linspace(0, 1, nb)
    times = sp * (u - accel * u * (1 - u)) + rng.normal(0, 0.006, nb)
    times = np.clip(times - times.min(), 0, None)
    taus = tau * (2000.0 * rt / freqs) ** 0.4
    L = min(cap, times.max() + 8 * taus.max())
    n = int(L * sr)
    y = np.zeros((2, n))
    rat = np.asarray(ratios)
    hits = [(times[i], freqs[i], 1.0, i) for i in range(nb)]
    if rebounce:
        for i in range(nb):
            if rng.random() < rebounce:
                hits.append((times[i] + rng.uniform(0.03, 0.12), freqs[i], rng.uniform(0.2, 0.5), i))
    for (t0, fi, g, i) in hits:
        s0 = int(t0 * sr)
        if s0 >= n:
            continue
        m = n - s0
        fr = fi * rat * (1 + np.r_[0, rng.uniform(-0.01, 0.01, len(rat) - 1)])
        a = np.asarray(amps) * _hw(fr, tce) * g * rng.uniform(0.6, 1.0)
        pan = -0.7 + 1.4 * i / max(nb - 1, 1)
        a2 = np.stack([a * np.cos((pan + 1) * np.pi / 4), a * np.sin((pan + 1) * np.pi / 4)])
        y[:, s0:] += _bank(m, sr, fr, a2, taus[i] * (1.0 / rat) ** 0.5, rng, floor=6.5)
    return _finish(A * y, sr, cap)


def m_tamtam(f, dur, vel, rng, sr, f0=60.0, nmodes=24, fmax=7000.0, bloom=0.35, tau=3.5,
             cap=8.0, noise=1.6, tc=5e-3, low=((1.0, 0.5), (1.52, 0.3), (2.1, 0.25), (2.6, 0.18))):
    """orchestral tam-tam: soft low hum at the strike, then the nonlinear
    energy cascade 'blooms' into a roaring shimmer; louder = more bloom."""
    rt = f / C4
    A = _amp(vel)
    tce = _tc(tc, vel)
    L = cap
    n = int(L * sr)
    t = np.arange(n) / sr
    lf = np.array([r for r, _ in low]) * f0 * rt
    la = np.array([a for _, a in low]) * _hw(lf, tce)
    lt = tau * 1.2 * (f0 / (lf / rt)) ** 0.3
    u = rng.uniform(0, 1, nmodes)
    hf = 150.0 * rt * (fmax / 150.0) ** np.sort(u)
    ha = (hf / (150 * rt)) ** -0.15 * rng.uniform(0.4, 1.0, nmodes) * (0.15 + vel) ** 1.5 * 0.6
    hr = bloom * (hf / (1000.0 * rt)) ** 0.5 * (1.3 - 0.6 * vel) * rng.uniform(0.7, 1.3, nmodes)
    ht = tau * (500.0 * rt / hf) ** 0.45 * rng.uniform(0.8, 1.2, nmodes)
    fr = np.r_[lf, hf]
    am = np.r_[la, ha]
    ta = np.r_[lt, ht]
    ri = np.r_[np.zeros(len(lf)) + 0.004, hr]
    bt = np.r_[[0.7, 1.3, 0, 0][:len(lf)], np.zeros(nmodes)]
    bd = np.r_[[0.3, 0.3, 0, 0][:len(lf)], np.zeros(nmodes)]
    # two 'microphones': odd/even modes favour opposite sides (cheap stereo)
    ev = np.arange(len(fr)) % 2 == 0
    ya = _bank(n, sr, fr[ev], am[ev], ta[ev], rng, rise=ri[ev], beat=bt[ev], bdepth=bd[ev], floor=6.5)
    yb = _bank(n, sr, fr[~ev], am[~ev], ta[~ev], rng, rise=ri[~ev], beat=bt[~ev], bdepth=bd[~ev], floor=6.5)
    y = np.stack([ya + 0.55 * yb, 0.55 * ya + yb])
    if noise:
        def mag(tt, ff):
            fc = np.maximum(ff, 50.0) / (1000.0 * rt)
            r_ = bloom * 1.5 * fc ** 0.6 * (1.3 - 0.6 * vel)
            d_ = tau * 0.8 * fc ** -0.5
            g = noise * (0.1 + vel) ** 2 * 0.25 * fc ** -0.2 * ((ff > 400 * rt) & (ff < 13000))
            return g * -np.expm1(-tt / r_) * np.exp(-tt / d_)
        y += _tfnoise(n, sr, rng, mag, 2) * 1.4
    return _finish(A * y, sr, cap)


# =========================================================================== presets
J, AF, LA, AS, CE, OR, OL, SP, WO = ('jungle', 'african', 'latin', 'asian', 'celtic',
                                      'orchestral', 'oldfield', 'space', 'world')
IN = 'inharmonic'

# ---------------------------------------------------------------- mallet: wood bars
_MAR = dict(ratios=(1.0, 3.98, 9.88), extra=((2.71, 0.03, 0.4), (6.3, 0.02, 0.3)), qexp=1.25)
register('mallet.marimba_rosewood', m_bar, 'mallet', lo='C2', hi='C7', tags=(OR, WO, J),
         desc='concert rosewood marimba, cord mallets: warm tube bloom, woody tock',
         amps=(1.0, 0.6, 0.25), tau=1.5, tau_ref=220.0, slope=0.7, tau_lim=(0.15, 2.2),
         tc=0.9e-3, tube=0.55, tube_tau=1.25, tube_rise=0.012, noise=0.12, nband=(250.0, 2500.0),
         cap=7.0, **_MAR)
register('mallet.marimba_bass', m_bar, 'mallet', lo='C2', hi='C5', tags=(OR, WO, J),
         desc='bass marimba with big soft mallets: deep breathing resonator hum',
         amps=(1.0, 0.4, 0.12), tau=1.8, tau_ref=220.0, slope=0.6, tau_lim=(0.4, 2.6),
         tc=3.2e-3, tube=0.85, tube_tau=1.35, tube_rise=0.022, noise=0.06, nband=(80.0, 700.0),
         ntau=0.006, cap=8.0, **_MAR)
register('mallet.marimba_yarn', m_bar, 'mallet', lo='C2', hi='C7', tags=(OR, WO, J, SP),
         desc='marimba with soft yarn mallets: round, almost flute-like, no click',
         amps=(1.0, 0.45, 0.12), tau=1.5, tau_ref=220.0, slope=0.7, tau_lim=(0.18, 2.2),
         tc=2.6e-3, tube=0.65, tube_tau=1.3, tube_rise=0.016, noise=0.035, nband=(150.0, 1200.0),
         ntau=0.005, cap=7.0, **_MAR)
register('mallet.marimba_hard', m_bar, 'mallet', lo='C2', hi='C7', tags=(OR, WO, J),
         desc='marimba with hard plastic mallets: bright knocking attack, ringing 4th partial',
         amps=(1.0, 0.75, 0.42), tau=1.4, tau_ref=220.0, slope=0.7, tau_lim=(0.12, 2.0),
         tc=0.32e-3, tube=0.45, tube_tau=1.2, tube_rise=0.01, noise=0.3, nband=(1200.0, 8000.0),
         ntau=0.002, cap=7.0, **_MAR)
register('mallet.balafon', m_bar, 'mallet', lo='G2', hi='C6', tags=(AF, J, WO),
         desc='West African balafon: rough-tuned bars over gourds with buzzing spider-silk mirlitons',
         ratios=(1.0, 3.25, 6.4), jit=0.06, amps=(1.0, 0.5, 0.2), tau=0.8, tau_ref=220.0,
         slope=0.55, tau_lim=(0.12, 1.6), qexp=1.2, tc=0.9e-3, tube=0.7, tube_tau=0.9,
         tube_rise=0.008, noise=0.2, nband=(400.0, 3500.0), buzz=1.4, buzz_th=0.09,
         buzz_band=(700.0, 4200.0), buzz_f=1800.0, rasp=0.8, cap=4.0, detune=6.0)
register('mallet.gyil', m_bar, 'mallet', lo='C3', hi='E5', tags=(AF, J, WO),
         desc='Ghanaian gyil: dry hardwood keys, heavy nasal gourd buzz on every note',
         ratios=(1.0, 2.62, 5.6), jit=0.05, amps=(1.0, 0.55, 0.25), tau=0.55, tau_ref=220.0,
         slope=0.5, tau_lim=(0.1, 1.2), qexp=1.3, tc=0.7e-3, tube=0.6, tube_tau=0.7,
         tube_rise=0.006, noise=0.25, nband=(500.0, 4000.0), buzz=1.4, buzz_th=0.05,
         buzz_band=(500.0, 3200.0), buzz_f=1300.0, rasp=1.0, cap=3.0, detune=7.0)
register('mallet.bamboo_marimba', m_bar, 'mallet', lo='C3', hi='C6', tags=(LA, J, WO),
         desc='marimba de chonta: palm-wood bars over guadua bamboo tubes, hollow warbling bonk',
         ratios=(1.0, 2.92, 5.9), jit=0.03, amps=(1.0, 0.45, 0.18), tau=0.55, tau_ref=220.0,
         slope=0.45, tau_lim=(0.1, 1.2), qexp=1.1, tc=1.1e-3, tube=0.75, tube_tau=1.1,
         tube_rise=0.009, tube3=0.25, noise=0.18, nband=(300.0, 2200.0), cap=4.0, detune=6.0)
register('mallet.xylophone', m_bar, 'mallet', lo='F4', hi='C8', tags=(OR, WO),
         desc='orchestral xylophone: hard mallets, quint-tuned bars, dry bright crack',
         ratios=(1.0, 3.0, 6.03, 9.6), amps=(1.0, 0.7, 0.3, 0.12), tau=0.7, tau_ref=523.0,
         slope=0.6, tau_lim=(0.08, 1.4), qexp=1.0, tc=0.3e-3, tube=0.35, tube_tau=1.0,
         tube_rise=0.005, noise=0.25, nband=(1800.0, 10000.0), ntau=0.0015, cap=4.0)

# ---------------------------------------------------------------- mallet: tines
register('mallet.kalimba', m_tine, 'mallet', lo='C3', hi='E6', tags=(AF, J, WO, SP),
         desc='box kalimba: thumb-plucked steel tines, bell-like 6.3x overtone, wooden box glow')
register('mallet.mbira_dzavadzimu', m_tine, 'mallet', lo='G2', hi='D5', tags=(AF, J, WO),
         desc='mbira dzavadzimu in a deze gourd: wide keys, sizzling shell-and-bottle-cap rattles',
         ratios=(1.0, 5.9, 16.0), amps=(1.0, 0.3, 0.08), tau=1.6, tau_ref=220.0, slope=0.45,
         tau_lim=(0.4, 3.0), tc=0.75e-3, oct2=0.18, box=((220.0, 2.0, 0.4), (650.0, 4.0, 0.1)),
         rattle=0.9, rattle_th=0.08, deze=0.5, detune=6.0)
register('bell.music_box', m_tine, 'bell', lo='C4', hi='C7', tags=(OR, SP, WO),
         desc='music-box comb: pin-plucked steel teeth, tiny tick, glassy shimmer in a wood case',
         ratios=(1.0, 5.6, 15.5), amps=(1.0, 0.35, 0.12), tau=1.4, tau_ref=1047.0, slope=0.45,
         tau_lim=(0.3, 2.5), tc=0.35e-3, oct2=0.06,
         box=((520.0, 3.0, 0.3), (1300.0, 5.0, 0.18), (2600.0, 6.0, 0.08)), noise=0.2,
         nband=(3000.0, 11000.0), detune=3.0)

# ---------------------------------------------------------------- mallet: wood & bamboo idiophones
register('mallet.slit_drum_small', m_log, 'mallet', lo='C4', hi='C6', tags=(J, AF, WO),
         desc='small hardwood tongue drum: sweet hollow tok with a box breath',
         ratios=(1.0, 2.3, 3.6, 5.1), amps=(1.0, 0.25, 0.1, 0.05), tau=0.35, tau_ref=440.0,
         slope=0.4, tau_lim=(0.08, 1.0), tc=1.6e-3, glide=0.008, cav=0.25, cav_hz=180.0,
         cav_tau=0.12, noise=0.12, nband=(1000.0, 5000.0))
register('mallet.slit_drum', m_log, 'mallet', lo='C3', hi='C5', tags=(J, AF, WO),
         desc='tuned slit drum: knocking wood tongues over a resonant box',
         ratios=(1.0, 2.43, 3.9, 5.6), amps=(1.0, 0.3, 0.12, 0.05), tau=0.3, tau_ref=220.0,
         slope=0.4, tau_lim=(0.07, 1.0), tc=2.2e-3, glide=0.015, cav=0.35, cav_hz=110.0,
         cav_tau=0.18, wall=((900.0, 0.15, 0.02),), noise=0.15, nband=(700.0, 4000.0))
register('mallet.log_drum', m_log, 'mallet', lo='C2', hi='C4', tags=(J, AF, WO, IN),
         desc='giant hollowed-log slit drum (garamut): booming cavity, thudding lip strike',
         ratios=(1.0, 1.83, 2.95, 4.4), jit=0.03, amps=(1.0, 0.45, 0.25, 0.1), tau=0.5,
         tau_ref=110.0, slope=0.4, tau_lim=(0.1, 1.4), tc=1.5e-3, glide=0.03, glide_tau=0.012,
         cav=0.35, cav_hz=70.0, cav_tau=0.35, cav_rise=0.01,
         wall=((450.0, 0.2, 0.05), (1200.0, 0.1, 0.02)), noise=0.25, nband=(300.0, 2500.0),
         thud=0.4)
register('mallet.wooden_agung', m_log, 'mallet', lo='C2', hi='C4', tags=(AS, J, WO),
         desc='wooden agung: hollow log played like a gong, padded beater, humming bloom',
         ratios=(1.0, 1.62, 2.31, 3.1), amps=(1.0, 0.3, 0.15, 0.06), tau=0.9, tau_ref=110.0,
         slope=0.4, tau_lim=(0.2, 1.6), tc=4e-3, glide=0.01, cav=0.8, cav_ratio=1.003,
         cav_tau=1.2, cav_rise=0.025, noise=0.08, nband=(150.0, 1000.0), ntau=0.008, thud=0.3,
         cap=6.0)
register('mallet.bamboo_stamp', m_log, 'mallet', lo='C2', hi='C5', tags=(J, AS, WO),
         desc='bamboo stamping tube: hollow air-column pop with a ground thud and bamboo clack',
         ratios=(1.0, 3.03, 5.08, 7.1), amps=(1.0, 0.4, 0.2, 0.08), tau=0.14, tau_ref=220.0,
         slope=0.25, tau_lim=(0.05, 0.4), qexp=0.6, tc=0.5e-3, glide=0.0,
         wall=((1100.0, 0.25, 0.015), (2600.0, 0.12, 0.008)), thud=0.8, noise=0.2,
         nband=(200.0, 2000.0), cap=2.0)
register('mallet.angklung', m_angklung, 'mallet', lo='C4', hi='C7', tags=(AS, J, WO),
         desc='Sundanese angklung: octave bamboo tubes rattled in their frame for the gate length',
         sustain=True)

# ---------------------------------------------------------------- bell: struck bars & bells
_beam = np.array([1.0, 2.756, 5.404, 8.933, 13.344, 18.638, 24.81, 31.86, 39.9, 48.8])
_chime = _beam * (2.0 / 8.933)   # strike tone = half of mode 4 (modes 4,5,6 ~ 2:3:4)
_TUB = (P(_chime[0], 0.04, 2.0), P(_chime[1], 0.12, 1.8, 0.4, 0.2),
        P(_chime[2], 0.35, 1.4, 0.5, 0.3), P(_chime[3], 1.0, 1.0, 0.35, 0.25),
        P(_chime[4], 0.9, 0.75, 0.8, 0.2), P(_chime[5], 0.7, 0.55), P(_chime[6], 0.45, 0.42, 1.3, 0.2),
        P(_chime[7], 0.3, 0.32), P(_chime[8], 0.15, 0.22), P(_chime[9], 0.08, 0.16))
register('bell.tubular', m_bell, 'bell', lo='C4', hi='F5', tags=(OR, OL, IN),
         desc='orchestral tubular bells: rawhide strike on the cap, clangorous chime, long hum',
         partials=_TUB, tau=2.4, tau_ref=262.0, slope=0.5, tau_lim=(1.2, 3.5), tc=0.38e-3,
         noise=0.25, nband=(2500.0, 12000.0), ntau=0.006, cap=9.0, jit=0.003)
_TUBH = tuple((r, a * (1.0 if r < 3 else 1.35), tm, b, bd, ri, c) for (r, a, tm, b, bd, ri, c) in _TUB)
register('bell.tubular_hammer', m_bell, 'bell', lo='C4', hi='F5', tags=(OR, OL, IN),
         desc='"plus... tubular bells": the chime hit with a hammer, harsh steel clang, wide stereo',
         partials=_TUBH, tau=2.6, tau_ref=262.0, slope=0.5, tau_lim=(1.3, 3.7), tc=0.2e-3,
         noise=0.6, nband=(3000.0, 14000.0), ntau=0.008, cap=9.0, mics=0.35, sat=0.25, jit=0.004)
_CH = (P(0.5, 0.6, 1.0, 0.6, 0.35), P(1.0, 0.5, 0.75, 1.1, 0.3), P(1.183, 0.55, 0.7, 1.7, 0.25),
       P(1.506, 0.25, 0.45), P(2.0, 0.9, 0.6, 0.9, 0.2), P(2.514, 0.3, 0.35), P(2.662, 0.2, 0.3),
       P(3.011, 0.35, 0.3), P(4.166, 0.25, 0.22), P(5.433, 0.12, 0.15), P(6.8, 0.06, 0.1))
register('bell.church', m_bell, 'bell', lo='G2', hi='C5', tags=(OR, OL, CE, IN),
         desc='bronze church bell: hum, prime, minor-third tierce and nominal, iron clapper clang',
         partials=_CH, tau=1.5, tau_ref=196.0, slope=0.5, tau_lim=(0.6, 2.2), tc=0.8e-3,
         noise=0.3, nband=(1500.0, 8000.0), ntau=0.01, mics=0.2, cap=10.0, jit=0.002)
register('bell.glockenspiel', m_bar, 'bell', lo='G5', hi='C8', tags=(OR, SP),
         desc='orchestral glockenspiel: brass mallets on steel bars, piercing sparkling ring',
         ratios=(1.0, 2.71, 5.33, 8.56), amps=(1.0, 0.3, 0.12, 0.05), tau=1.6, tau_ref=1047.0,
         slope=0.35, tau_lim=(0.6, 3.0), qexp=0.6, tc=0.2e-3, noise=0.12,
         nband=(4000.0, 15000.0), ntau=0.0015, cap=7.0)
register('bell.celesta', m_bar, 'bell', lo='C4', hi='C8', tags=(OR, SP),
         desc='celesta: felt hammers on steel plates over wooden resonators, sugar-plum glow',
         ratios=(1.0, 2.76, 5.4), amps=(1.0, 0.25, 0.06), tau=1.1, tau_ref=1047.0, slope=0.4,
         tau_lim=(0.4, 2.5), qexp=0.7, tc=0.8e-3, tube=0.65, tube_tau=0.8, tube_rise=0.006,
         noise=0.05, nband=(200.0, 1500.0), ntau=0.006, damper=0.12, cap=6.0, sustain=True)
register('bell.crotales', m_bar, 'bell', lo='C6', hi='C8', tags=(OR, SP),
         desc='antique cymbals: thick tuned brass discs, pure endless shimmer',
         ratios=(1.0, 2.09, 3.36, 4.83, 6.4), amps=(1.0, 0.22, 0.12, 0.06, 0.03), tau=2.4,
         tau_ref=1047.0, slope=0.3, tau_lim=(1.2, 4.0), qexp=0.5, tc=0.25e-3, noise=0.1,
         nband=(5000.0, 15000.0), ntau=0.0015, cap=9.0)
register('bell.glass_struck', m_bell, 'bell', lo='C4', hi='C7', tags=(SP, WO),
         desc='water-tuned wine glasses tapped with a spoon: pure ping with a wet wobble',
         partials=(P(1.0, 1.0, 1.0, 3.0, 0.15), P(2.83, 0.3, 0.45), P(5.42, 0.1, 0.25),
                   P(8.77, 0.04, 0.15)),
         tau=0.9, tau_ref=523.0, slope=0.3, tau_lim=(0.4, 1.8), tc=0.35e-3, noise=0.15,
         nband=(3000.0, 12000.0), ntau=0.002, cap=7.0, jit=0.01)
register('bell.bell_tree', m_cascade, 'bell', pitched=False, tags=(OR, WO, SP, IN),
         desc='bell tree: a beater slides down nested brass bells, descending sparkle')
register('bell.mark_tree', m_cascade, 'bell', pitched=False, tags=(OR, SP, IN),
         desc='mark-tree chimes: a finger sweep through rods, upward glittering cascade',
         nb=24, f_hi=9000.0, f_lo=2200.0, span=1.0, ratios=(1.0, 2.756, 5.404),
         amps=(1.0, 0.3, 0.1), tau=1.3, tc=0.15e-3, descend=False, rebounce=0.35, accel=0.0,
         cap=7.0)
register('bell.singing_bowl', m_rubbed, 'bell', lo='C3', hi='C5', tags=(AS, SP, WO),
         desc='Tibetan singing bowl: struck, then rubbed with the stick - swelling wah-wah drone',
         sustain=True, ref_dur=2.0, chatter=0.6)
register('bell.glass_harp', m_rubbed, 'bell', lo='C4', hi='C7', tags=(SP, OR, WO),
         desc='glass harp: wet fingers circling water-tuned glasses, crystalline sustained voice',
         ratios=(1.0, 2.32, 4.0), strike=0.03, rub=(1.0, 0.06, 0.01), att=0.22, tau=1.5,
         tau_ref=523.0, slope=0.4, tau_lim=(0.5, 2.5), beat=0.7, bdepth=0.15, squeak=0.8,
         wobble=2.0, tc=3e-3, cap=8.0, sustain=True)

# ---------------------------------------------------------------- metal
register('metal.vibraphone', m_bar, 'metal', lo='F3', hi='F6', tags=(OR, SP),
         desc='vibraphone, motor on: aluminium bars, rotating resonator discs, damper pedal',
         ratios=(1.0, 4.0, 10.0), amps=(1.0, 0.4, 0.1), tau=3.5, tau_ref=220.0, slope=0.6,
         tau_lim=(0.8, 6.0), qexp=0.7, tc=1.1e-3, tube=0.7, tube_tau=1.0, tube_rise=0.01,
         motor=5.5, motor_depth=0.6, noise=0.04, nband=(500.0, 3000.0), damper=0.09, cap=9.0,
         sustain=True)
register('metal.vibraphone_still', m_bar, 'metal', lo='F3', hi='F6', tags=(OR, SP),
         desc='vibraphone, motor off, cord mallets: cool glassy bell-tone with a soft ping',
         ratios=(1.0, 4.0, 10.0), amps=(1.0, 0.55, 0.18), tau=3.5, tau_ref=220.0, slope=0.6,
         tau_lim=(0.8, 6.0), qexp=0.7, tc=0.6e-3, tube=0.6, tube_tau=1.0, tube_rise=0.008,
         noise=0.08, nband=(1500.0, 7000.0), ntau=0.0015, damper=0.09, cap=9.0, sustain=True)
register('metal.saron', m_bar, 'metal', lo='G3', hi='C6', tags=(AS, WO),
         desc='gamelan saron pair: thick bronze keys, horn mallet clang, shimmering ombak beats',
         ratios=(1.0, 2.65, 4.95, 7.3), amps=(1.0, 0.5, 0.25, 0.1), tau=1.4, tau_ref=520.0,
         slope=0.4, tau_lim=(0.6, 3.0), qexp=0.7, tc=0.4e-3, tube=0.2, tube_tau=0.9, noise=0.2,
         nband=(2000.0, 9000.0), ntau=0.002, ombak=6.5, cap=7.0)
register('metal.gender', m_bar, 'metal', lo='C3', hi='C6', tags=(AS, WO, SP),
         desc='gamelan gender pair: thin bronze keys over bamboo tubes, padded mallets, slow ombak',
         ratios=(1.0, 2.72, 5.1), amps=(1.0, 0.25, 0.08), tau=3.0, tau_ref=260.0, slope=0.5,
         tau_lim=(1.0, 5.0), qexp=0.7, tc=1.6e-3, tube=0.45, tube_tau=1.0, tube_rise=0.015,
         noise=0.04, nband=(300.0, 2000.0), ombak=4.5, damper=0.25, cap=9.0, sustain=True)
register('metal.bonang', m_bell, 'metal', lo='C4', hi='C6', tags=(AS, WO, IN),
         desc='bonang: bossed bronze kettle-gong pots on cords, round singing ping',
         partials=(P(1.0, 1.0, 1.0), P(1.52, 0.3, 0.6), P(3.46, 0.18, 0.35), P(3.92, 0.14, 0.3)),
         tau=1.1, tau_ref=523.0, slope=0.5, tau_lim=(0.4, 2.2), tc=1.3e-3,
         glide=(0.25, 0.04, 'from'), noise=0.1, nband=(600.0, 3000.0), ntau=0.004, cap=6.0)
register('metal.kempul', m_bell, 'metal', lo='C3', hi='C5', tags=(AS, WO, IN),
         desc='kempul: hanging bossed gong, padded beater, deep beating ombak swell',
         partials=(P(1.0, 1.0, 1.0, 2.2, 0.45), P(2.03, 0.35, 0.6, 0.0, 0.0, 0.01),
                   P(2.94, 0.25, 0.45), P(3.85, 0.12, 0.35), P(4.6, 0.06, 0.3)),
         tau=2.0, tau_ref=130.0, slope=0.4, tau_lim=(1.0, 3.5), tc=3.5e-3, noise=0.05,
         nband=(100.0, 800.0), ntau=0.008, thump=0.3, thump_f=80.0, cap=9.0)
register('metal.gong_ageng', m_bell, 'metal', lo='C1', hi='C3', tags=(AS, WO, IN, SP),
         desc='gong ageng: huge bossed gong, slow breathing ombak, shimmer blooms and sags in pitch',
         partials=(P(1.0, 1.0, 1.0, 0.9, 0.55), P(1.47, 0.2, 0.6, 0, 0, 0.08, 1),
                   P(2.02, 0.45, 0.7, 1.3, 0.3, 0.05, 1), P(2.33, 0.2, 0.5, 0, 0, 0.12, 1),
                   P(2.76, 0.2, 0.45, 0, 0, 0.15, 1), P(3.07, 0.15, 0.4, 0, 0, 0.2, 1),
                   P(3.62, 0.1, 0.35, 0, 0, 0.25, 1), P(4.5, 0.08, 0.3, 0, 0, 0.3, 1),
                   P(5.32, 0.05, 0.25, 0, 0, 0.35, 1), P(6.1, 0.04, 0.2, 0, 0, 0.4, 1)),
         tau=2.2, tau_ref=49.0, slope=0.3, tau_lim=(1.5, 3.2), tc=6e-3, cvel=1.5,
         glide=(-0.35, 0.8, 'to'), noise=0.03, nband=(80.0, 600.0), ntau=0.01, thump=0.5,
         thump_f=45.0, cap=10.0)
register('metal.opera_gong', m_bell, 'metal', pitched=False, tags=(AS, WO, IN),
         desc='Chinese opera gong (xiaoluo): flat bronze, the pitch whoops upward after the hit',
         partials=(P(1.0, 1.0, 1.0), P(1.38, 0.4, 0.6), P(1.82, 0.3, 0.5), P(2.4, 0.25, 0.4),
                   P(3.1, 0.15, 0.3), P(3.9, 0.1, 0.25), P(4.7, 0.07, 0.2)),
         tau=0.9, tau_ref=330.0, slope=0.0, tau_lim=(0.3, 2.0), tc=0.5e-3,
         glide=(3.0, 0.12, 'to'), noise=0.25, nband=(1000.0, 6000.0), ntau=0.004, cap=4.0)
register('metal.opera_gong_low', m_bell, 'metal', pitched=False, tags=(AS, WO, IN),
         desc='large opera gong (daluo): crashing bronze that sags downward in pitch',
         partials=(P(1.0, 1.0, 1.0), P(1.29, 0.5, 0.7), P(1.71, 0.45, 0.6), P(2.18, 0.35, 0.5),
                   P(2.6, 0.3, 0.45), P(3.3, 0.2, 0.35), P(4.1, 0.15, 0.3), P(5.2, 0.1, 0.25),
                   P(6.3, 0.06, 0.2)),
         tau=1.5, tau_ref=180.0, slope=0.0, tau_lim=(0.5, 3.0), tc=0.8e-3,
         glide=(-1.6, 0.35, 'to'), noise=0.35, nband=(800.0, 7000.0), ntau=0.01, mics=0.3, cap=6.0)
register('metal.tam_tam', m_tamtam, 'metal', pitched=False, tags=(OR, SP, IN, OL),
         desc='orchestral tam-tam: dark hum that blooms into a roaring metallic shimmer')
register('metal.handpan', m_bell, 'metal', lo='D3', hi='A5', tags=(WO, SP),
         desc='handpan: finger-struck steel dome, tuned octave and fifth blooming out of the note',
         partials=(P(1.0, 1.0, 1.0, 0.7, 0.2), P(2.0012, 0.5, 0.75, 0, 0, 0.03, 1),
                   P(2.9955, 0.3, 0.55, 0, 0, 0.045, 1), P(4.02, 0.06, 0.35, 0, 0, 0.05, 1),
                   P(1.335, 0.035, 1.0, 0, 0, 0.12, 1), P(1.68, 0.025, 1.0, 0, 0, 0.15, 1)),
         tau=1.8, tau_ref=294.0, slope=0.4, tau_lim=(0.8, 3.0), tc=1.4e-3, cvel=1.0,
         noise=0.06, nband=(300.0, 3000.0), ntau=0.005, thump=0.15, thump_f=110.0, cap=8.0)
register('metal.steel_pan', m_bell, 'metal', lo='D4', hi='F#6', tags=(LA, WO),
         desc='Trinidad tenor (lead) steel pan: bright singing note, octave swirling in and out',
         partials=(P(1.0, 1.0, 1.0), P(2.0, 0.75, 0.7, 3.5, 0.25, 0.012, 1),
                   P(3.0, 0.35, 0.5, 0, 0, 0.02, 1), P(4.0, 0.15, 0.4, 0, 0, 0.02, 1),
                   P(2.47, 0.04, 0.2)),
         tau=0.7, tau_ref=587.0, slope=0.35, tau_lim=(0.3, 1.4), tc=0.7e-3, cvel=1.0,
         noise=0.1, nband=(1000.0, 6000.0), ntau=0.003, sat=0.15, cap=5.0)
register('metal.steel_pan_double', m_bell, 'metal', lo='F#3', hi='C#6', tags=(LA, WO),
         desc='double-second steel pan: warmer, rounder alto pan with a sympathetic skirt ring',
         partials=(P(1.0, 1.0, 1.0, 1.2, 0.1), P(2.0, 0.55, 0.8, 2.5, 0.2, 0.015, 1),
                   P(3.0, 0.2, 0.6, 0, 0, 0.02, 1), P(4.0, 0.06, 0.4, 0, 0, 0.02, 1),
                   P(1.5, 0.04, 0.9, 0, 0, 0.06, 1)),
         tau=1.0, tau_ref=330.0, slope=0.35, tau_lim=(0.4, 1.8), tc=1.1e-3, cvel=1.0,
         noise=0.07, nband=(600.0, 4000.0), ntau=0.004, sat=0.1, cap=6.0)
register('metal.waterphone', m_waterphone, 'metal', lo='G3', hi='G5', tags=(SP, IN, OR),
         desc='bowed waterphone: steel rods over a sloshing resonator, eerie bending howls',
         sustain=True, ref_dur=2.0)
