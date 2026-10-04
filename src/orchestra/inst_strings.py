"""
Canopy Orchestra — plucked, bowed, guitar and bass families.

Physical-ish models, fully vectorised:
  * plucked strings are tuned digital waveguides (stringlab.waveguide) excited by
    shaped pulses (finger / nail / pick / hammer), pluck-position comb, courses of
    detuned strings, convolved with synthetic body impulse responses;
  * electric guitars add pickup comb + resonance, compression, oversampled
    overdrive and a cabinet;
  * bowed strings are a Helmholtz-sawtooth wavetable (bow position notches,
    pressure-dependent brightness), human vibrato, bow-slip noise and attack
    scratch through dense violin-family body responses (so vibrato makes the
    harmonics flicker through the body resonances, as on the real thing).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .core import *  # noqa: F401,F403
from .core import register, lp, hp, bp, tvec, onepole_lp, SR, TWOPI
from .stringlab import (waveguide, pluck_exc, apply_body, body_ir, warp, human_pitch, gate_env,
                        tail_fade, note_n, compress, overdrive, cab, res_lp, peq, eq, hp_sos, lp_sos,
                        shelf, wt_table, wt_osc, cycles, nharm, smooth_noise, cents, tv_bandpass,
                        shifted, pan2, finish)


# =============================================================================== plucked core

def _course_list(f, courses, oct_below=None):
    """courses: tuple of (ratio, detune_sd_cents, gain, delay_ms, pan)."""
    out = []
    for c in courses:
        r, dsd, g, dms, pan = (tuple(c) + (0.0,) * 5)[:5]
        split = oct_below if oct_below is not None else 1e9
        if r in ('oct', 'octlo') and f >= split:
            continue            # octave courses only below the split
        if r == 'uni' and f < split:
            continue            # unison partner only at/above the split
        r = {'oct': 2.0, 'octlo': 0.5, 'uni': 1.0}.get(r, r)
        out.append((float(r), dsd, g, dms, pan))
    return out


def plucked(f, dur, vel, rng, sr, t60=2.5, f_ref=196.0, t60_k=0.45, t60_hi=0.35, f_hi=4000.0,
            disp=0.0, nap=0, width=1.2, width_vel=0.7, nail=0.2, nail_vel=0.6, exc_lp=3500.0,
            exc_lp_vel=7000.0, pos=0.18, pos_jit=0.02, courses=((1.0, 0.0, 1.0, 0.0, 0.0),),
            oct_split=None, body='classical', body_mix=0.85, direct=None, body_scale=1.0,
            pickup=None, tension=0.0, tension_tau=0.05, cap=6.0, damp=None, hp_hz=45.0,
            eq_rows=(), thump=0.0, thump_hz=(60, 400), knock=0.0, scoop=0.0, scoop_t=0.05,
            vib=None, tremolo=None, buzz=0.0, buzz_th=0.35, stereo_w=0.0, shape='cos',
            out_sat=0.0, crest=5.0, sat_vel=False, clank=0.0):
    vel = float(np.clip(vel, 0.02, 1.0))
    t60e = float(np.clip(t60 * (f_ref / f) ** t60_k, 0.25, 12.0))
    trem = tremolo is not None and dur > 0.3
    if trem:
        t60e = min(t60e, tremolo.get('t60', 1.0))
    if damp is not None:
        n = note_n(min(cap, dur + 7 * damp, t60e + 0.3), sr)
    elif trem:
        n = note_n(dur + tremolo.get('ring', 0.6), sr)
    else:
        n = note_n(min(cap, 0.9 * t60e + 0.25), sr)
    t = tvec(n, sr)
    # pitch track (applied by warping each string's output)
    fm = np.ones(n)
    if tension:
        fm *= cents(tension * vel * vel * np.exp(-t / tension_tau))
    if scoop:
        fm *= cents(scoop * (0.5 + 0.5 * vel) * np.exp(-t / scoop_t))
    if vib:
        fm *= human_pitch(n, rng, sr, **vib)
    warped = bool(tension or scoop or vib)
    ns = int(n * 1.02) + 64 if warped else n

    wid = width * (1.35 - width_vel * vel) * float(np.clip((196.0 / f) ** 0.5, 0.3, 1.4))
    nz = nail * (0.25 + nail_vel * vel)
    elp = exc_lp * (0.3 + 0.7 * vel) + exc_lp_vel * vel ** 1.5
    cl = _course_list(f, courses, oct_split)
    L = np.zeros(n)
    R = np.zeros(n)
    for (r, dsd, g, dms, pan) in cl:
        fc = f * r * cents(rng.normal(0, dsd) if dsd else 0.0)
        pp = float(np.clip(pos + rng.normal(0, pos_jit), 0.04, 0.5))
        if trem:
            x, picks = _tremolo_exc(fc, vel, rng, sr, dur, wid, nz, elp, pp, tremolo, dms, ns, shape)
        else:
            e = pluck_exc(fc, rng, vel, wid, nz, elp, pp, sr, shape)
            dl = abs(rng.normal(dms, 0.25 * dms + 0.1)) * 1e-3 * sr
            if r == 1.0 and dms < 3.0:
                dl = min(dl, 0.22 * sr / fc)        # unison partner: never near antiphase
            x = shifted(e, int(dl), len(e) + int(dl) + 8)
        t60c = float(np.clip(t60e * (1.0 / r) ** (t60_k * 0.7), 0.2, 12.0))
        y = waveguide(fc, ns, x, t60c, min(t60_hi, t60c), f_hi, disp, nap, sr)
        if trem:
            # the plectrum touching the string damps it just before each new pick
            imp = np.zeros(ns)
            idx = np.array(picks[1:], dtype=int) - int(0.003 * sr)
            imp[idx[(idx > 0) & (idx < ns)]] = 1.0
            win = signal.windows.gaussian(int(0.016 * sr) | 1, 0.0035 * sr)
            y *= 1.0 - tremolo.get('duck', 0.7) * np.minimum(1.0, np.convolve(imp, win, 'same'))
        if pickup is not None:
            ppos, pfc, pq = pickup
            d = ppos * sr / fc
            di = int(d)
            yy = y.copy()
            y[di:] -= 0.9 * yy[:len(y) - di]
            y = res_lp(y, pfc, pq, sr)
        if warped:
            y = warp(y, fm, n)
        else:
            y = y[:n]
        a = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        L += g * np.cos(a) * y * np.sqrt(2)
        R += g * np.sin(a) * y * np.sqrt(2)
    mono = 0.5 * (L + R)
    side = 0.5 * (L - R)
    # bridge buzz (kora nyenyemo / sitar-ish rattle): pitch-synchronous gated noise
    if clank and vel > 0.35:
        # hard plucks slap the string against the frets: bright pitch-synchronous rattle
        env = onepole_lp(np.abs(mono), 40.0, sr)
        ex = np.maximum(np.abs(mono) - 0.9 * env * 1.57, 0.0) * np.sign(mono)
        ck = hp(ex, 1200, 2, sr) * np.exp(-t / 0.25)
        mono = mono + clank * ((vel - 0.35) / 0.65) ** 1.5 * ck * 6.0
    if buzz:
        env = onepole_lp(np.abs(mono), 30.0, sr)
        gate = np.maximum(np.abs(mono) - buzz_th * env * 1.6, 0.0)
        bz = hp(gate * (1 + 0.6 * rng.standard_normal(n)), 2500, 2, sr)
        mono = mono + buzz * bz * (0.6 + 0.6 * vel) * 3.0
    if thump:
        k = int(0.06 * sr)
        th = bp(rng.standard_normal(k), thump_hz[0], thump_hz[1], sr=sr) * np.exp(-np.arange(k) / (0.012 * sr))
        mono[:k] += thump * (0.3 + 0.7 * vel) * th * np.max(np.abs(mono)) / (np.max(np.abs(th)) + 1e-9)
    if knock:
        k = int(0.02 * sr)
        kn = hp(rng.standard_normal(k), 1500, 2, sr) * np.exp(-np.arange(k) / (0.002 * sr))
        mono[:k] += knock * vel ** 2 * kn * np.max(np.abs(mono)) / (np.max(np.abs(kn)) + 1e-9)
    if body is not None:
        mono = apply_body(mono, body, body_mix, direct, body_scale, sr)
        if stereo_w or np.any(side):
            side = apply_body(side, body, body_mix, direct, body_scale, sr)
    rows = [hp_sos(hp_hz, 2, sr)[0]] + list(eq_rows)
    mono = eq(mono, rows)
    if out_sat:
        pk = np.max(np.abs(mono)) + 1e-9
        sd = out_sat * (0.4 + 0.9 * vel) if sat_vel else out_sat
        mono = np.tanh(sd * mono / pk) * pk / np.tanh(sd)
    if damp is not None:
        k = int(dur * sr)
        if k < n:
            de = np.ones(n)
            de[k:] = np.exp(-(t[k:] - t[k]) / damp)
            mono *= de
            side *= de
    if stereo_w and not np.any(side):
        d = int(0.0006 * sr)
        side = stereo_w * 0.3 * (shifted(mono, d, n) - mono)
    if np.any(side):
        y = np.stack([mono + side, mono - side])
    else:
        y = mono
    return tail_fade(finish(y, vel, sr, crest), 0.1, sr)


def _tremolo_exc(fc, vel, rng, sr, dur, wid, nz, elp, pp, tr, dms, ns, shape):
    """mandolin-style tremolo: alternate down/up strokes for the whole gate."""
    rate = tr.get('rate', 13.0) + tr.get('rate_vel', 4.0) * vel
    x = np.zeros(ns)
    tk = 0.0
    k = 0
    e_dn = pluck_exc(fc, rng, vel, wid, nz, elp, pp, sr, shape)
    e_up = pluck_exc(fc, rng, vel * 0.8, wid * 1.25, nz * 0.6, elp * 0.7, pp, sr, shape)
    stop = dur - 0.03
    sw = rng.uniform(0, TWOPI)
    picks = []
    while tk < stop:
        e = e_dn if k % 2 == 0 else -e_up
        g = (1.0 if k % 2 == 0 else 0.92) * (1 + 0.1 * rng.standard_normal())
        g *= 1 + 0.12 * np.sin(TWOPI * 0.6 * tk + sw)                # slow swell of the hand
        if k == 0:
            g *= 1.25
        s = int((tk + (dms if k % 2 == 0 else 1.6 - dms * 0.5) * 1e-3) * sr)
        if s + len(e) < ns:
            x[s:s + len(e)] += g * e
            picks.append(s)
        k += 1
        tk += (1.0 / rate) * (1 + 0.07 * rng.standard_normal())
    return x, picks


# =============================================================================== special plucked

def sitar(f, dur, vel, rng, sr, symp=0.22, jawari=0.9):
    """sitar-ish: bright main string, jawari buzz with a falling formant sweep,
    sympathetic (taraf) strings ringing in octaves/fifths, gourd body."""
    vel = float(np.clip(vel, 0.02, 1))
    t60e = float(np.clip(5.0 * (196 / f) ** 0.35, 1.5, 6.0))
    n = note_n(min(4.5, 0.8 * t60e + 0.3), sr)
    t = tvec(n, sr)
    e = pluck_exc(f, rng, vel, 0.35 * (1.3 - 0.6 * vel), 0.4 * (0.3 + 0.7 * vel), 5000 + 7000 * vel, 0.07, sr)
    s = waveguide(f, n, e, t60e, 1.1, 5000, 0.3, 2, sr)
    # small meend-like settle: plucked string starts a touch sharp
    s = warp(np.concatenate([s, np.zeros(400)]), cents(9 * vel * np.exp(-t / 0.08)), n)
    env = onepole_lp(np.abs(s), 25.0, sr) + 1e-9
    # jawari: grazing bridge -> clipped excess -> bright spectrum that slowly darkens
    th = 0.55 * env * 1.57
    ex = np.sign(s) * np.maximum(np.abs(s) - th, 0.0)
    sweep = 900 + 3600 * np.exp(-t / (0.35 + 0.25 * vel))
    bz = tv_bandpass(hp(ex, 600, 2, sr), sweep, 1.6, sr, block=384) + 0.35 * hp(ex, 3000, 2, sr)
    bz *= np.max(np.abs(s)) / (np.max(np.abs(bz)) + 1e-9)
    y = s + jawari * (0.6 + 0.5 * vel) * bz
    # sympathetic strings: resonant combs at octave / fifth / octave+fifth, driven by the note
    sy = np.zeros(n)
    for r, g in ((2.0, 1.0), (1.5, 0.7), (3.0, 0.6), (4.0, 0.5)):
        fs = f * r * cents(rng.normal(0, 3))
        if fs > 3000:
            continue
        drive = hp(y, 300, 2, sr) * 0.02
        sy += g * waveguide(fs, n, drive, 3.5, 1.0, 4000, 0, 0, sr)
    sy *= np.max(np.abs(y)) / (np.max(np.abs(sy)) + 1e-9)
    y = y + symp * sy * np.clip(t / 0.25, 0, 1)
    y = apply_body(y, 'sitar', 0.8, None, 1.0, sr)
    y = eq(y, [hp_sos(70, 2, sr)[0], peq(2500, 3, 1.0, sr)])
    st = np.stack([y + 0.25 * symp * sy, y - 0.25 * symp * sy])
    return tail_fade(finish(st, vel, sr), 0.1, sr)


def berimbau(f, dur, vel, rng, sr, wah_rate=2.2, buzz=0.35):
    """musical bow: stick-struck steel wire, coin buzz, gourd wah via the
    player opening/closing the gourd against the belly (cycles over the gate)."""
    vel = float(np.clip(vel, 0.02, 1))
    n = note_n(min(3.6, max(dur, 0.4) + 1.2), sr)
    t = tvec(n, sr)
    e = pluck_exc(f, rng, vel, 0.3 * (1.5 - vel), 0.8 * (0.3 + 0.7 * vel), 1800 + 9000 * vel ** 1.5, 0.06, sr, 'sine')
    s = waveguide(f, n, e, 2.2 * (110 / f) ** 0.3, 0.6, 4000, 0.82, 6, sr)
    # stick buzz: stick rattles the wire at the strike, coin chatter afterwards
    k = int(0.05 * sr)
    rat = bp(rng.standard_normal(k), 1800, 7000, sr=sr) * np.exp(-np.arange(k) / (0.006 * sr))
    s[:k] += 0.5 * vel * rat * np.max(np.abs(s)) / (np.max(np.abs(rat)) + 1e-9)
    envs = onepole_lp(np.abs(s), 30, sr) + 1e-9
    chat = np.maximum(np.abs(s) - 1.3 * envs, 0) * np.sign(s)
    s = s + buzz * hp(chat, 1500, 2, sr) * 4
    # gourd: closed (muffled) -> open -> closed... while the gate lasts, then closes
    ph = TWOPI * wah_rate * (1 + 0.08 * rng.standard_normal()) * t
    open_ = 0.5 - 0.5 * np.cos(ph)
    gt = np.clip((dur + 0.15 - t) / 0.15, 0, 1)
    open_ = open_ * gt + 0.15
    fc = 330 + 900 * open_
    res = tv_bandpass(s, fc, 3.2, sr)
    y = 0.35 * lp(s, 2500, 2, sr) * (0.4 + 0.6 * open_) + 1.6 * res + 0.9 * lp(s, 1.6 * f, 2, sr)
    y = eq(y, [hp_sos(f * 0.7, 2, sr)[0]])
    return tail_fade(finish(y, vel, sr), 0.1, sr)


# =============================================================================== electric guitars

def electric(f, dur, vel, rng, sr, t60=5.0, t60_hi=0.8, pick_w=0.35, pick_noise=0.35, pos=0.22,
             pickup=(0.22, 3500.0, 1.4), drive=8.0, drive_vel=1.0, asym=0.2, comp=0.7,
             vib=None, scoop=0.0, scoop_t=0.07, bloom=0.0, cabinet='4x12', pre=(), post=(),
             rel=0.12, mix_clean=0.0, cap=8.0, double=0.0, chorus=0.0):
    vel = float(np.clip(vel, 0.02, 1))
    n = note_n(min(cap, dur + 6 * rel), sr)
    ns = int(n * 1.03) + 64
    t = tvec(n, sr)
    t60e = float(np.clip(t60 * (196 / f) ** 0.3, 1.0, 12.0))
    e = pluck_exc(f, rng, vel, pick_w * (1.3 - 0.6 * vel), pick_noise * (0.3 + 0.7 * vel), 5000 + 6000 * vel, pos, sr)
    y = waveguide(f, ns, e, t60e, t60_hi, 3500, 0.2, 2, sr)
    ppos, pfc, pq = pickup
    di = int(ppos * sr / f)
    y[di:] = y[di:] - 0.85 * y[:-di].copy()
    y = res_lp(y, pfc, pq, sr)
    fm = np.ones(n)
    if scoop:
        fm *= cents(scoop * (0.5 + 0.6 * vel) * np.exp(-t / scoop_t))
    if vib:
        fm *= human_pitch(n, rng, sr, **vib)
    y = warp(y, fm, n)
    if bloom:
        # amp feedback: an upper harmonic swells in after ~0.6 s of sustain
        cyc = cycles(f * 2, fm, n, sr)
        be = bloom * np.clip((t - 0.6) / 1.6, 0, 1) ** 1.5 * (1 + 0.2 * smooth_noise(n, 2, rng, sr))
        y = y + be * np.std(y[:int(0.2 * sr)]) * 1.4 * np.sin(TWOPI * cyc)
    clean = y
    if comp:
        y = compress(y, comp, 0.004, 0.15, sr)
    y = eq(y, [hp_sos(110, 2, sr)[0]] + list(pre))
    y = y / (np.max(np.abs(y)) + 1e-9)
    y = overdrive(y, drive * (0.25 + drive_vel * vel), asym, sr, os=4 if drive > 4 else 2)
    y = cab(y, cabinet, sr)
    y = eq(y, list(post))
    if mix_clean:
        c = cab(clean / (np.max(np.abs(clean)) + 1e-9), 'combo', sr)
        y = y + mix_clean * c * np.max(np.abs(y)) / (np.max(np.abs(c)) + 1e-9)
    k = int(dur * sr)
    if k < n:
        y[k:] *= np.exp(-(t[k:] - t[k]) / rel)
    amp = 1.0
    if double or chorus:
        # double tracking: a second, independently humanised take panned apart
        d = int((0.012 + 0.006 * rng.random()) * sr)
        y2 = warp(np.concatenate([y, np.zeros(64)]), human_pitch(n, rng, sr, rate=0.3, depth=0, drift=4, jitter=1.5), n)
        y2 = shifted(y2, d, n)
        return tail_fade(finish(np.stack([y + 0.6 * y2, 0.6 * y + y2]), vel, sr, curve=1.0), 0.08, sr)
    return tail_fade(finish(y, vel, sr, curve=1.0), 0.08, sr)


# =============================================================================== bowed

_BODY_BY_SIZE = ((130.0, 'cello'), (240.0, 'viola'), (1e9, 'violin'))


def _section_body(f):
    for lim, name in _BODY_BY_SIZE:
        if f < lim:
            return name
    return 'violin'


def _bow_voice(f, n, dur, vel, rng, sr, att, rel, bright, bright_vel, vib, noise, scratch,
               jit_att, tremolo, beta):
    t = tvec(n, sr)
    # extra pitch instability while the Helmholtz motion establishes itself
    fm = human_pitch(n, rng, sr, att_jit=jit_att * (1.2 - vel), **vib)
    K = nharm(f, float(fm.max()), sr, 16000 if sr > 40000 else 10500)
    k = np.arange(1, K + 1)
    notch = np.abs(np.sin(np.pi * k * beta)) ** 0.6 + 0.08
    basic = notch / k * 10 ** (rng.normal(0, 1.2, K) / 20)
    fb = bright + bright_vel * vel
    dark = basic / (1 + (k * f / (fb * 0.33)) ** 2)
    brt = basic / (1 + (k * f / fb) ** 2)
    cyc = cycles(f, fm, n, sr, rng.random())
    yd, yb = wt_osc(wt_table(dark), cyc, wt_table(brt))
    a_eff = att * (1.5 - 0.9 * vel)
    env = gate_env(n, dur, a_eff, rel, sr, shape=1.3)
    env *= 1 + 0.035 * smooth_noise(n, 3.0, rng, sr)          # bow pressure/speed wander
    if tremolo:
        rate = tremolo * (1 + 0.08 * rng.standard_normal()) * (1 + 0.05 * smooth_noise(n, 1.0, rng, sr))
        phs = np.cumsum(rate) / sr + rng.random()
        strokes = (0.5 - 0.5 * np.cos(TWOPI * phs)) ** 0.6
        env = env * (0.3 + 0.7 * strokes)
    m = np.clip(env * (0.35 + 0.65 * vel), 0, 1) ** 1.2
    y = (yd + m * (yb - yd)) * env
    # bow noise: slip-synchronous, plus initial scratch
    nz = bp(rng.standard_normal(n), 1800, 9000, sr=sr)
    slip = np.exp(-(cyc % 1.0) / 0.12)
    namp = noise * (0.4 + 0.8 * vel) * env * (0.35 + 0.65 * slip) + scratch * vel * np.exp(-t / 0.045) * np.clip(t / 0.004, 0, 1)
    if tremolo:
        namp = namp * (0.6 + 0.8 * strokes)
    y = y + namp * nz * 0.25
    return y


def bowed(f, dur, vel, rng, sr, body='violin', att=0.08, rel=0.16, bright=2200.0, bright_vel=3600.0,
          vib=None, noise=0.06, scratch=0.25, jit_att=8.0, voices=1, detune=0.0, onset=0.0,
          spread=0.0, tremolo=None, body_mix=1.0, eqr=(), beta=(0.08, 0.14), auto_body=False):
    vel = float(np.clip(vel, 0.02, 1))
    n_out = note_n(min(12.0, dur + 5 * rel + 0.1), sr)
    osr = sr
    if voices > 1:          # ensembles are rendered at half rate, upsampled once
        sr = sr // 2
    n = n_out * sr // osr + 1
    vib = dict(vib or dict(rate=5.6, depth=18, delay=0.22, rise=0.35, drift=3, jitter=2))
    L = np.zeros(n)
    R = np.zeros(n)
    for v in range(voices):
        vv = dict(vib)
        if voices > 1:
            vv['rate'] = vib.get('rate', 5.5) * rng.uniform(0.85, 1.15)
            vv['delay'] = vib.get('delay', 0.2) * rng.uniform(0.6, 1.5)
        fv = f * cents(rng.normal(0, detune * float(np.clip(f / 220.0, 0.45, 1.0)))) if detune else f
        d = int(abs(rng.normal(0, onset)) * sr) if onset else 0
        y = _bow_voice(fv, n - d, dur - d / sr, vel * rng.uniform(0.85, 1.0) if voices > 1 else vel,
                       rng, sr, att * (rng.uniform(0.8, 1.3) if voices > 1 else 1), rel, bright, bright_vel, vv,
                       noise, scratch, jit_att, tremolo, rng.uniform(*beta))
        y = shifted(y, d, n)
        pan = (v / max(1, voices - 1) * 2 - 1) * spread if voices > 1 else 0.0
        if voices > 1:
            pan += rng.normal(0, 0.1)
        a = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        L += np.cos(a) * y
        R += np.sin(a) * y
    bname = _section_body(f) if auto_body else body
    if voices > 1:
        L = signal.resample_poly(L, osr // sr, 1)[:n_out]
        R = signal.resample_poly(R, osr // sr, 1)[:n_out]
        sr = osr
        L = apply_body(L, bname, body_mix, None, 1.0, sr)
        R = apply_body(R, bname, body_mix, None, 1.02, sr)
        out = np.stack([eq(L, [hp_sos(f * 0.5, 2, sr)[0]] + list(eqr)), eq(R, [hp_sos(f * 0.5, 2, sr)[0]] + list(eqr))])
    else:
        y = apply_body(L + R, bname, body_mix, None, 1.0, sr)
        out = eq(y, [hp_sos(min(f * 0.6, 200), 2, sr)[0]] + list(eqr))
    return tail_fade(finish(out, vel, sr, crest=4.0, curve=1.3), 0.06, sr)


def pizz_section(f, dur, vel, rng, sr, players=5):
    body = _section_body(f)
    cs = tuple((1.0, 6.0, rng.uniform(0.7, 1.0), rng.uniform(0, 22), np.clip(-0.7 + 1.4 * i / (players - 1), -1, 1))
               for i in range(players))
    return plucked(f, dur, vel, rng, sr, t60=0.9, f_ref=196, t60_k=0.4, t60_hi=0.12, f_hi=3000,
                   width=2.6, width_vel=0.5, nail=0.05, exc_lp=2500, exc_lp_vel=3500, pos=0.28,
                   courses=cs, body=body, body_mix=1.0, direct=0.1, thump=0.25, thump_hz=(80, 500), cap=3.0)


# =============================================================================== registration

def _r(name, fn, fam, lo, hi, tags, desc, sustain=False, **p):
    register(name, fn, fam, lo=lo, hi=hi, tags=tags, desc=desc, sustain=sustain, **p)


C1 = (1.0, 0.0, 1.0, 0.0, 0.0)

# ---------------------------------------------------------------- pluck family
_r('pluck.mandolin', plucked, 'pluck', 'G3', 'A6', ('oldfield', 'celtic', 'world'),
   'Bright paired-course mandolin; holds of more than 0.3 s become a shimmering down-up tremolo, the classic Oldfield sound',
   sustain=True, t60=1.6, t60_k=0.4, t60_hi=0.3, f_hi=5000, width=0.35, nail=0.45, exc_lp=4000, exc_lp_vel=8000,
   pos=0.12, courses=((1.0, 0.7, 1.0, 0.0, -0.25), (1.0, 1.2, 0.9, 0.7, 0.25)), body='mandolin', body_mix=0.9,
   tremolo=dict(rate=12.5, rate_vel=4.0, t60=0.55, ring=0.7), knock=0.25)
_r('pluck.charango', plucked, 'pluck', 'E4', 'A6', ('latin', 'world'),
   'Andean charango: sparkly nylon courses, the middle course doubled an octave down, tiny bright body',
   t60=1.4, t60_k=0.3, t60_hi=0.25, width=0.6, nail=0.5, exc_lp=4500, pos=0.15,
   courses=((1.0, 0.8, 1.0, 0.0, -0.2), (1.0, 1.3, 0.85, 0.6, 0.2), ('octlo', 1.5, 0.55, 1.2, 0.0)), oct_split=600.0,
   body='small', body_scale=1.15, body_mix=0.9)
_r('pluck.banjo', plucked, 'pluck', 'D3', 'D6', ('world', 'celtic'),
   'Five-string banjo: metal-fingerpick snap and a ringing drum-head body that twangs then dies fast',
   t60=1.3, t60_k=0.3, t60_hi=0.35, f_hi=6000, width=0.22, nail=0.6, exc_lp=7000, exc_lp_vel=8000, pos=0.1,
   body='banjo', body_mix=1.2, direct=0.25, tension=10, tension_tau=0.03, knock=0.3, eq_rows=(peq(2800, 3, 1.0),))
_r('pluck.ukulele', plucked, 'pluck', 'C4', 'C6', ('world', 'latin'),
   'Soprano ukulele: soft nylon thumb-pluck, short plinky decay, small sweet box',
   t60=1.0, t60_k=0.25, t60_hi=0.18, width=1.6, nail=0.12, exc_lp=2800, exc_lp_vel=4000, pos=0.3,
   body='small', body_mix=1.0, direct=0.2)
_r('pluck.cavaquinho', plucked, 'pluck', 'D4', 'A6', ('latin', 'world', 'jungle'),
   'Brazilian cavaquinho: steel-strung chorinho brightness, crisp pick, quick shimmering decay',
   t60=1.3, t60_k=0.3, t60_hi=0.35, f_hi=5000, width=0.3, nail=0.5, exc_lp=5500, exc_lp_vel=8000, pos=0.13,
   body='small', body_scale=1.05, body_mix=0.9, eq_rows=(peq(3500, 3, 1.2),))
_r('pluck.kora', plucked, 'pluck', 'F2', 'A5', ('african', 'jungle', 'world'),
   'West-African kora: bright thumb-plucked harp-lute with calabash boom and sizzling buzzer bridge',
   t60=3.0, t60_k=0.45, t60_hi=0.5, width=0.9, nail=0.3, exc_lp=4000, exc_lp_vel=6000, pos=0.16,
   body='kora', body_mix=1.0, buzz=0.35, buzz_th=0.5, stereo_w=0.5)
_r('pluck.harp', plucked, 'pluck', 'C2', 'G7', ('orchestral', 'celtic'),
   'Concert pedal harp: round finger pluck near mid-string (hollow, odd-harmonic), long glowing soundboard ring',
   t60=4.5, t60_k=0.55, t60_hi=0.6, width=2.2, width_vel=0.5, nail=0.05, exc_lp=2200, exc_lp_vel=4500, pos=0.45,
   pos_jit=0.03, body='harp', body_mix=0.9, stereo_w=0.6)
_r('pluck.harp_wire', plucked, 'pluck', 'C2', 'E6', ('celtic', 'world'),
   'Wire-strung clarsach: fingernail-plucked brass strings, bell-like and endlessly ringing',
   t60=7.0, t60_k=0.45, t60_hi=2.0, f_hi=5000, width=0.4, nail=0.35, exc_lp=6000, pos=0.3, disp=0.3, nap=2,
   courses=((1.0, 0.0, 1.0, 0.0, 0.0), (1.0, 1.2, 0.25, 0.0, 0.0)), body='harp', body_mix=0.7, direct=0.5, stereo_w=0.7)
_r('pluck.koto', plucked, 'pluck', 'D3', 'A6', ('asian', 'world'),
   'Japanese koto: ivory tsume twang near the bridge, nasal paulownia body, plucked string settling in pitch',
   t60=2.2, t60_k=0.35, t60_hi=0.5, f_hi=5000, width=0.3, nail=0.55, exc_lp=5000, exc_lp_vel=7000, pos=0.1,
   body='koto', body_mix=1.0, tension=18, tension_tau=0.06, eq_rows=(peq(1500, 3, 1.2), shelf(250, 5, False)))
_r('pluck.sitar', sitar, 'pluck', 'C3', 'C6', ('asian', 'world'),
   'Sitar-ish: jawari buzz with a sweeping formant and a halo of sympathetic taraf strings')
_r('pluck.berimbau', berimbau, 'pluck', 'A2', 'D4', ('latin', 'jungle', 'world', 'african'),
   'Berimbau musical bow: stick-struck wire, coin buzz and gourd wah-wah that pumps for as long as the gate',
   sustain=True)
_r('pluck.oud', plucked, 'pluck', 'D2', 'F5', ('world',),
   'Arabic oud: fretless paired gut courses, feather-risha attack, deep dark bowl-back thump',
   t60=1.6, t60_k=0.35, t60_hi=0.22, f_hi=3500, width=0.7, nail=0.3, exc_lp=2800, exc_lp_vel=4500, pos=0.2,
   courses=((1.0, 1.0, 1.0, 0.0, -0.2), (1.0, 1.4, 0.85, 0.8, 0.2)), body='oud', body_mix=1.1, direct=0.2,
   scoop=-10, scoop_t=0.04, thump=0.35, thump_hz=(70, 300))
_r('pluck.dulcimer_hammered', plucked, 'pluck', 'D3', 'E6', ('celtic', 'world', 'oldfield'),
   'Hammered dulcimer: wooden hammers on courses of three wires, shimmering unison beating, long sustain',
   t60=4.5, t60_k=0.45, t60_hi=0.9, f_hi=5000, width=0.6, width_vel=0.8, nail=0.25, exc_lp=4000, exc_lp_vel=8000,
   pos=0.11, disp=0.25, nap=2, shape='sine',
   courses=((1.0, 0.8, 1.0, 0.0, -0.35), (1.0, 1.4, 0.9, 0.3, 0.0), (1.0, 1.4, 0.8, 0.6, 0.35)),
   body='dulcimer', body_mix=0.9, direct=0.4)
_r('pluck.bouzouki', plucked, 'pluck', 'G2', 'A5', ('celtic', 'world'),
   'Irish bouzouki: long-scale steel courses, low ones in octaves, jangly plectrum drone-strummer',
   t60=3.0, t60_k=0.4, t60_hi=0.55, f_hi=5000, width=0.35, nail=0.45, exc_lp=4500, exc_lp_vel=7000, pos=0.14,
   courses=((1.0, 0.8, 1.0, 0.0, -0.25), ('oct', 1.5, 0.55, 1.5, 0.25), ('uni', 1.3, 0.9, 0.5, 0.25)),
   oct_split=220.0, body='bouzouki', body_mix=0.9)

# ---------------------------------------------------------------- guitar family
_r('guitar.nylon_spanish', plucked, 'guitar', 'E2', 'B5', ('latin', 'oldfield', 'world'),
   'Spanish classical guitar: warm flesh-and-nail pluck, 100/200/400 Hz cedar-top body bloom',
   t60=3.0, t60_k=0.45, t60_hi=0.3, f_hi=3500, width=1.3, width_vel=0.75, nail=0.3, nail_vel=0.8, exc_lp=2400,
   exc_lp_vel=5500, pos=0.17, body='classical', body_mix=1.0, knock=0.12, stereo_w=0.4)
_r('guitar.steel_acoustic', plucked, 'guitar', 'E2', 'C6', ('oldfield', 'rock', 'world'),
   'Steel-string dreadnought: crisp pick, booming low end, long silvery sustain',
   t60=4.2, t60_k=0.4, t60_hi=0.7, f_hi=5000, width=0.4, nail=0.4, exc_lp=4000, exc_lp_vel=8000, pos=0.15,
   disp=0.15, nap=1, body='steel', body_mix=1.0, stereo_w=0.4)
_r('guitar.twelve_string', plucked, 'guitar', 'E2', 'C6', ('oldfield', 'rock'),
   '12-string acoustic: octave courses struck first, chorusing jangle spread across the stereo field',
   t60=3.4, t60_k=0.4, t60_hi=0.75, f_hi=5000, width=0.3, nail=0.6, exc_lp=4500, exc_lp_vel=9000, pos=0.13,
   courses=((1.0, 0.8, 1.0, 4.0, -0.35), ('oct', 2.2, 0.75, 0.0, 0.4), ('uni', 2.6, 0.9, 0.6, 0.4)), oct_split=250.0,
   body='steel', body_mix=1.0, eq_rows=(shelf(4500, 3, True),))
_r('guitar.electric_clean', electric, 'guitar', 'E2', 'E6', ('rock', 'oldfield', 'space'),
   'Clean single-coil electric: glassy pickup resonance, a breath of valve warmth, muted at gate end',
   sustain=True, t60=6.0, t60_hi=1.0, pick_w=0.35, pick_noise=0.3, pos=0.2, pickup=(0.19, 4800.0, 2.4),
   drive=1.2, drive_vel=0.5, asym=0.05, comp=0.25, cabinet='combo', rel=0.22,
   pre=(peq(3000, 2, 1.0),), post=(shelf(5000, 2, True),))
_r('guitar.oldfield_lead', electric, 'guitar', 'E3', 'D6', ('oldfield', 'rock'),
   'Oldfield sustain lead: singing fuzz with infinite-feeling sustain, slight scoop, slow wide delayed vibrato and blooming feedback',
   sustain=True, t60=9.0, t60_hi=1.5, pick_w=0.4, pick_noise=0.25, pos=0.25, pickup=(0.25, 3000.0, 1.3),
   drive=9.0, drive_vel=1.0, asym=0.25, comp=0.85, scoop=-45, scoop_t=0.06, bloom=0.35,
   vib=dict(rate=5.3, depth=32, delay=0.32, rise=0.45, drift=2.5, jitter=1.0, wander=0.12),
   cabinet='4x12', pre=(peq(750, 7, 0.8), lp_sos(5500, 2)[0]), post=(peq(1600, 2, 0.8),), rel=0.13, cap=12.0)
_r('guitar.crunch_rhythm', electric, 'guitar', 'E2', 'E5', ('rock',),
   'Crunchy humbucker rhythm guitar: bridge-pickup bite into a cranked 2x12, tight palm-muted end',
   sustain=True, t60=3.0, t60_hi=0.5, pick_w=0.3, pick_noise=0.45, pos=0.12, pickup=(0.1, 3800.0, 1.6),
   drive=4.0, drive_vel=1.0, asym=0.12, comp=0.45, cabinet='2x12', pre=(peq(1200, 4, 0.9),), rel=0.05)
_r('guitar.double_speed', plucked, 'guitar', 'E3', 'C6', ('oldfield', 'rock'),
   'Oldfield "double-speed" guitar: half-speed-recorded steel string sped up, chipmunk-bright, compressed and glittering',
   t60=1.5, t60_k=0.3, t60_hi=0.4, f_hi=6000, width=0.18, width_vel=0.6, nail=0.6, exc_lp=7000, exc_lp_vel=8000,
   pos=0.11, body='double_speed', body_mix=1.0, eq_rows=(peq(3800, 3, 1.0), shelf(8000, 2, True)),
   out_sat=2.2, sat_vel=True, courses=((1.0, 0.0, 1.0, 0.0, -0.5), (1.0, 5.0, 0.55, 11.0, 0.6)))

# ---------------------------------------------------------------- bass family
_r('bass.fingered', plucked, 'bass', 'E1', 'G4', ('rock', 'oldfield', 'latin'),
   'Fingered electric bass: round two-finger thump, neck-pickup warmth, muted when the gate ends',
   sustain=True, t60=6.0, f_ref=55, t60_k=0.3, t60_hi=0.4, f_hi=2000, width=2.6, width_vel=0.6, nail=0.05,
   exc_lp=1500, exc_lp_vel=3000, pos=0.2, disp=0.3, nap=2, body=None, pickup=(0.17, 2600.0, 1.3),
   thump=0.35, thump_hz=(50, 250), damp=0.07, clank=0.5, eq_rows=(peq(90, 2, 0.9), peq(700, 1.5, 1.0)), out_sat=1.3, cap=8.0)
_r('bass.picked', plucked, 'bass', 'E1', 'G4', ('rock',),
   'Pick-played bass: clicky plectrum attack, bridge-pickup growl, punchy mids',
   sustain=True, t60=5.0, f_ref=55, t60_k=0.3, t60_hi=0.7, f_hi=3000, width=0.35, width_vel=0.6, nail=0.6,
   exc_lp=4000, exc_lp_vel=5000, pos=0.12, disp=0.35, nap=2, body=None, pickup=(0.2, 3600.0, 1.8),
   damp=0.05, clank=0.4, eq_rows=(shelf(160, 6, False), peq(900, 4, 1.0), peq(2200, 3, 1.4)), out_sat=2.0, knock=0.2, cap=8.0)
_r('bass.fretless', plucked, 'bass', 'E1', 'C5', ('rock', 'space', 'world'),
   'Fretless bass: singing mwah, slides up into each note, slow delayed vibrato',
   sustain=True, t60=8.0, f_ref=55, t60_k=0.3, t60_hi=0.9, f_hi=2000, width=2.0, width_vel=0.6, nail=0.05,
   exc_lp=1800, exc_lp_vel=3000, pos=0.15, body=None, pickup=(0.12, 2200.0, 2.2), scoop=-70, scoop_t=0.07,
   vib=dict(rate=4.8, depth=9, delay=0.45, rise=0.5, drift=2.0, jitter=0.5),
   damp=0.12, eq_rows=(peq(1100, 5, 1.6), peq(80, 2, 1.0)), out_sat=1.2, cap=10.0)
_r('bass.upright_pizz', plucked, 'bass', 'E1', 'D4', ('orchestral', 'latin', 'world'),
   'Double-bass pizzicato: big woody plunk, tension pitch-drop at the attack, fast-fading gut tone',
   sustain=True, t60=2.2, f_ref=55, t60_k=0.35, t60_hi=0.12, f_hi=1500, width=3.6, width_vel=0.6, nail=0.04,
   exc_lp=900, exc_lp_vel=1800, pos=0.3, body='contrabass', body_mix=1.1, direct=0.35, tension=25,
   tension_tau=0.035, thump=0.5, thump_hz=(50, 220), damp=0.15, cap=5.0)


# ---------------------------------------------------------------- bowed family
_r('bowed.violin', bowed, 'bowed', 'G3', 'E7', ('orchestral', 'oldfield'),
   'Solo violin: bright bowed Helmholtz tone through a resonant maple body, singing delayed vibrato',
   sustain=True, body='violin', att=0.07, rel=0.18, bright=2400, bright_vel=4200,
   vib=dict(rate=5.8, depth=20, delay=0.2, rise=0.35, drift=2.5, jitter=1.5), noise=0.07, scratch=0.3)
_r('bowed.fiddle', bowed, 'bowed', 'G3', 'B6', ('celtic', 'world'),
   'Irish fiddle: rosiny short-bow attack, bright and dry, only a hint of late vibrato',
   sustain=True, body='violin', att=0.03, rel=0.1, bright=3000, bright_vel=4500, beta=(0.1, 0.18),
   vib=dict(rate=6.2, depth=6, delay=0.45, rise=0.3, drift=3, jitter=2.5), noise=0.12, scratch=0.55,
   jit_att=12, eqr=(peq(3200, 3, 1.0),))
_r('bowed.viola', bowed, 'bowed', 'C3', 'E6', ('orchestral',),
   'Viola: husky alto bowing, darker body, veiled and warm',
   sustain=True, body='viola', att=0.09, rel=0.2, bright=1800, bright_vel=3200,
   vib=dict(rate=5.5, depth=18, delay=0.22, rise=0.35, drift=2.5, jitter=1.5), noise=0.07, scratch=0.3)
_r('bowed.cello', bowed, 'bowed', 'C2', 'A5', ('orchestral', 'oldfield'),
   'Cello: rich woody bowing with a resonant low body, broad slow vibrato',
   sustain=True, body='cello', att=0.11, rel=0.24, bright=1500, bright_vel=2800,
   vib=dict(rate=5.2, depth=17, delay=0.25, rise=0.4, drift=2.0, jitter=1.2), noise=0.06, scratch=0.3)
_r('bowed.contrabass_arco', bowed, 'bowed', 'E1', 'G4', ('orchestral',),
   'Double bass arco: growling rosin-heavy bow, deep boxy body, slow bloom',
   sustain=True, body='contrabass', att=0.16, rel=0.3, bright=900, bright_vel=1800,
   vib=dict(rate=4.8, depth=10, delay=0.35, rise=0.4, drift=2.0, jitter=1.5), noise=0.09, scratch=0.45)
_r('bowed.erhu', bowed, 'bowed', 'D4', 'A6', ('asian', 'world'),
   'Erhu: nasal python-skin fiddle, slides into each note, wide expressive finger vibrato',
   sustain=True, body='erhu', att=0.06, rel=0.16, bright=2600, bright_vel=3000, beta=(0.12, 0.2),
   vib=dict(rate=6.0, depth=30, delay=0.15, rise=0.3, drift=4, jitter=2, scoop=-90, scoop_t=0.07),
   noise=0.1, scratch=0.25, eqr=(peq(1300, 5, 1.0), hp_sos(350, 2)[0]))
_r('strings.section_warm', bowed, 'bowed', 'C2', 'C7', ('orchestral', 'oldfield', 'space'),
   'Warm string section: eight detuned players, slow swelling attack, independent vibratos, wide stereo',
   sustain=True, auto_body=True, att=0.32, rel=0.38, bright=1400, bright_vel=2400, voices=8, detune=7.0,
   onset=0.02, spread=0.85, noise=0.04, scratch=0.08, jit_att=4,
   vib=dict(rate=5.4, depth=14, delay=0.25, rise=0.5, drift=3, jitter=1.5), eqr=(shelf(4000, -3, True),))
_r('strings.tremolo', bowed, 'bowed', 'C2', 'C7', ('orchestral', 'jungle'),
   'Tremolo strings: a section bowing rapid unsynchronised strokes, nervous rustling shimmer',
   sustain=True, auto_body=True, att=0.08, rel=0.3, bright=1900, bright_vel=3200, voices=6, detune=6.0,
   onset=0.01, spread=0.8, noise=0.09, scratch=0.15, tremolo=13.0,
   vib=dict(rate=5.4, depth=6, delay=0.3, rise=0.5, drift=3, jitter=2))
_r('strings.pizzicato', pizz_section, 'bowed', 'C2', 'C7', ('orchestral', 'jungle'),
   'Pizzicato section: five players plucking slightly apart, soft woody pops')
