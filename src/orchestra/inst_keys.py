"""
Canopy Orchestra — keyboards ('keys') and organs / free reeds ('organ').

  piano       additive stiff-string model: partials f_k = k f sqrt(1 + B k^2) with
              register-dependent B, 1-3 unison strings per note (detuned ->
              beating and two-stage prompt/after-sound decay), felt hammer whose
              cutoff rises with velocity, strike-point comb, soundboard IR,
              hammer thump, dampers (none in the top octave and a half).
  epiano      tine (Rhodes-like) with nonlinear pickup -> bark at high velocity.
  wurli       reed e-piano: cantilever reed + electrostatic pickup clipping.
  clavinet / harpsichord   waveguide strings (inst_strings.plucked).
  celesta, music box       modal bars / comb teeth.
  mellotron   bowed trio / flute rendered then "replayed from tape".
  tonewheel   drawbars, contact click, percussion, tube drive, Leslie.
  combo       Farfisa-style divider pulse organ with vibrato.
  pipe        ranks (principal / flute / stopped / reed / mixture), speech
              transient (harmonics speak before the fundamental), chiff, wind.
  free_reed   harmonium / pump organ / accordion / concertina / bandoneon /
              melodica: pressure-dependent reed spectra, detuned reed sets,
              bellows wobble, reed-chamber body.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .core import *  # noqa: F401,F403
from .core import register, lp, hp, bp, tvec, onepole_lp, ftom, SR, TWOPI
from .stringlab import (apply_body, warp, human_pitch, gate_env, tail_fade, note_n, overdrive,
                        res_lp, peq, eq, hp_sos, lp_sos, shelf, wt_table, wt_osc, cycles, nharm,
                        smooth_noise, cents, shifted, finish, leslie, tape, waveguide, pluck_exc)
from .inst_strings import plucked, bowed


def _sin32(ph):
    """fast sine: reduce phase in float64, evaluate in float32."""
    return np.sin(np.mod(ph, TWOPI).astype(np.float32))


# =============================================================================== piano

def piano(f, dur, vel, rng, sr, b_scale=1.0, unison=0.7, tau_scale=1.0, hammer=1.0, strike=0.12,
          body='piano', body_mix=0.55, thump=0.5, after=0.3, spread=0.55, kmax=80, damper=0.09,
          cap=10.0, bright_tilt=0.35, honky=0.0, tack=0.0):
    vel = float(np.clip(vel, 0.02, 1.0))
    m = float(ftom(f))
    B = (2.5e-4 * 2 ** ((m - 60) / 12 * 0.95)) if m >= 40 else 8.3e-5 * (1 + (40 - m) / 12)
    B *= b_scale
    nstr = 1 if m < 31 else (2 if m < 43 else 3)
    tau2 = float(np.clip(7.0 * (262 / f) ** 0.6, 0.3, 20.0)) * tau_scale
    tau1 = tau2 * 0.15
    no_damp = m >= 90
    damp_t = damper * float(np.clip((262 / f) ** 0.3, 0.6, 1.8))
    if no_damp:
        n = note_n(min(cap, 4.0 * tau2 + 0.1), sr)
    else:
        n = note_n(min(cap, min(dur, 4.0 * tau2) + 7 * damp_t + 0.05), sr)
    t = tvec(n, sr)
    fc_h = (300 + 4500 * vel ** 1.8 * hammer) * (f / 262) ** 0.45 + 2.5 * f
    beta = strike * (1 + 0.06 * rng.standard_normal())
    # unison detuning (cents) of the strings of this note
    if nstr == 1:
        dets = [0.0]
    else:
        u = unison * rng.uniform(0.5, 1.4) + honky
        dets = list(np.linspace(-u, u, nstr) + rng.normal(0, 0.15 * unison + 0.2 * honky, nstr))
    pans = np.linspace(-0.25, 0.25, nstr) if nstr > 1 else [0.0]
    base_pan = float(np.clip((m - 64) / 44, -1, 1)) * spread
    gl = [np.cos((np.clip(base_pan + p, -1, 1) + 1) * np.pi / 4) for p in pans]
    gr = [np.sin((np.clip(base_pan + p, -1, 1) + 1) * np.pi / 4) for p in pans]
    # partials are rendered in bands at decimated rates, then upsampled once
    bands = {d: (np.zeros(n // d + 2), np.zeros(n // d + 2)) for d in (16, 8, 4, 2, 1)}
    a1 = None
    kmx = int(kmax * float(np.clip((110 / f) ** 0.3, 1.0, 1.5)))
    for k in range(1, kmx + 1):
        fk = k * f * np.sqrt(1 + B * k * k)
        if fk > min(0.45 * sr, 15000):
            break
        a = (abs(np.sin(np.pi * k * beta)) + 0.04) / np.sqrt(1 + (fk / fc_h) ** 4) * k ** (-bright_tilt)
        if a1 is None:
            a1 = a
        if a < 1e-3 * a1:
            continue
        d = 16
        while d > 1 and fk * 1.01 > 0.36 * sr / d:
            d //= 2
        bL, bR = bands[d]
        tk2 = tau2 / (1 + (fk / 1600) ** 1.6 + 0.05 * (k - 1))
        tk1 = max(tk2 * 0.15, 0.01)
        Lk = min(len(bL), int(4.5 * tk2 * sr / d) + 8)
        tt = np.arange(Lk) * (d / sr)
        env = after * np.exp((-tt / tk2).astype(np.float32))
        L1 = min(Lk, int(5 * tk1 * sr / d) + 4)
        env[:L1] += (1 - after) * np.exp((-tt[:L1] / tk1).astype(np.float32))
        sL = np.zeros(Lk, np.float32)
        sR = np.zeros(Lk, np.float32)
        for s_ in range(nstr if k <= 14 else min(nstr, 2)):
            fs = fk * 2 ** (dets[s_] / 1200)
            ph0 = rng.uniform(-0.25, 0.25)
            v = _sin32(TWOPI * fs * tt + ph0)
            g = (1.0 + 0.08 * rng.standard_normal()) / nstr
            sL += (g * gl[s_]) * v
            sR += (g * gr[s_]) * v
        bL[:Lk] += a * env * sL
        bR[:Lk] += a * env * sR
    L = np.zeros(n)
    R = np.zeros(n)
    for d, (bL, bR) in bands.items():
        if not np.any(bL):
            continue
        if d == 1:
            L += bL[:n]
            R += bR[:n]
        else:
            up = signal.resample_poly(np.stack([bL, bR]), d, 1, axis=-1)[:, :n]
            L[:up.shape[1]] += up[0]
            R[:up.shape[1]] += up[1]
    # hammer thump (felt + key bottoming) into the soundboard, and a hard 'tack'
    kth = int(0.09 * sr)
    th = lp(rng.standard_normal(kth), 250 + 300 * vel, 2, sr) * np.exp(-np.arange(kth) / (0.016 * sr))
    th = th / (np.max(np.abs(th)) + 1e-9) * thump * vel ** 1.5 * 0.25 * (np.max(np.abs(L)) + 1e-9)
    L[:kth] += th
    R[:kth] += th
    if tack:
        kt = int(0.012 * sr)
        tk = hp(rng.standard_normal(kt), 1800, 2, sr) * np.exp(-np.arange(kt) / (0.0015 * sr))
        tk = tk / (np.max(np.abs(tk)) + 1e-9) * tack * vel ** 2 * 0.2 * np.max(np.abs(L))
        L[:kt] += tk
        R[:kt] += tk
    L = apply_body(L, body, body_mix, None, 1.0, sr)
    R = apply_body(R, body, body_mix, None, 1.03, sr)
    if not no_damp:
        kd = int(dur * sr)
        if kd < n:
            de = np.ones(n)
            de[kd:] = np.exp(-(t[kd:] - t[kd]) / damp_t)
            # damper felt contact: a soft low thud
            kk = min(n - kd, int(0.05 * sr))
            dn = lp(rng.standard_normal(kk), 400, 2, sr) * np.exp(-np.arange(kk) / (0.01 * sr)) * 0.01 * np.max(np.abs(L))
            L *= de
            R *= de
            L[kd:kd + kk] += dn
            R[kd:kd + kk] += dn
    y = np.stack([L, R])
    y = signal.sosfilt(hp_sos(min(28, f * 0.7), 2, sr), y, axis=-1)
    return tail_fade(finish(y, vel, sr, crest=5.0, curve=1.7, floor=0.06), 0.06, sr)


# =============================================================================== electric pianos

def epiano(f, dur, vel, rng, sr, bark=1.0, bell=1.0, trem=0.0, trem_rate=4.8, q_off=0.45):
    """tine e-piano: tine + tonebar motion through a nonlinear magnetic pickup."""
    vel = float(np.clip(vel, 0.02, 1))
    dec = float(np.clip(3.2 * (262 / f) ** 0.55, 0.7, 7.0))
    rel = 0.12
    n = note_n(min(dur, 3 * dec) + 7 * rel, sr)
    t = tvec(n, sr)
    ph = TWOPI * f * t + rng.uniform(0, 1)
    # tine displacement: fundamental, two-stage decay, slight tonebar 2nd
    e = 0.7 * np.exp(-t / dec) + 0.3 * np.exp(-t / (dec * 0.25))
    e *= np.clip(t / 0.0008, 0, 1)
    d = np.sin(ph) * e + 0.04 * np.sin(2 * ph + 0.3) * np.exp(-t / (dec * 0.5))
    A = (0.35 + 1.4 * vel ** 1.3) * bark ** 0.5
    x = A * d + q_off
    y = x / np.sqrt(x * x + 1.0) - q_off / np.sqrt(q_off ** 2 + 1)   # pickup nonlinearity
    y = y / A
    # attack: metallic tine ping (inharmonic) and hammer thunk
    ping = bell * (0.08 + 0.25 * vel ** 1.5) * np.sin(TWOPI * f * 7.1 * t + 1.0) * np.exp(-t / 0.018) \
        + bell * 0.05 * vel * np.sin(TWOPI * f * 19.3 * t) * np.exp(-t / 0.006)
    if f * 7.1 < 0.45 * sr:
        y = y + ping
    k = int(0.03 * sr)
    y[:k] += 0.15 * vel * lp(rng.standard_normal(k), 600, 2, sr) * np.exp(-np.arange(k) / (0.006 * sr))
    kd = int(dur * sr)
    if kd < n:
        y[kd:] *= np.exp(-(t[kd:] - t[kd]) / rel)
    y = eq(y, [hp_sos(40, 2, sr)[0], peq(180, 2, 0.8, sr), peq(1200, -2, 0.7, sr), *lp_sos(9000, 2, sr)])
    if trem:
        lfo = np.sin(TWOPI * trem_rate * t + rng.uniform(0, TWOPI))
        g = np.sqrt(np.clip(0.5 + 0.5 * trem * lfo, 0, 1)), np.sqrt(np.clip(0.5 - 0.5 * trem * lfo, 0, 1))
        y = np.stack([y * g[0], y * g[1]]) * np.sqrt(2)
    return tail_fade(finish(y, vel, sr, crest=5.0, curve=1.5), 0.05, sr)


def wurli(f, dur, vel, rng, sr, trem=0.0, trem_rate=5.5):
    """reed e-piano: struck steel reed (cantilever modes) + electrostatic pickup
    that clips asymmetrically when played hard (the bark)."""
    vel = float(np.clip(vel, 0.02, 1))
    dec = float(np.clip(1.8 * (262 / f) ** 0.5, 0.4, 4.0))
    rel = 0.07
    n = note_n(min(dur, 3 * dec) + 7 * rel, sr)
    t = tvec(n, sr)
    ph = TWOPI * f * t + rng.uniform(0, 1)
    e = (0.75 * np.exp(-t / dec) + 0.25 * np.exp(-t / (0.2 * dec))) * np.clip(t / 0.0012, 0, 1)
    d = np.sin(ph) * e
    if f * 6.27 < 0.45 * sr:
        d += 0.12 * (0.4 + vel) * np.sin(TWOPI * f * 6.27 * t) * np.exp(-t / 0.03)
    drive = 0.9 + 3.2 * vel ** 1.6
    y = np.tanh(drive * d + 0.35) - np.tanh(0.35)
    y = y / np.tanh(drive)
    kd = int(dur * sr)
    if kd < n:
        y[kd:] *= np.exp(-(t[kd:] - t[kd]) / rel)
    y = eq(y, [hp_sos(90, 2, sr)[0], peq(1000, 4, 0.9, sr), peq(2600, 2, 1.5, sr), *lp_sos(5500, 4, sr)])
    if trem:
        y = y * (1 - 0.5 * trem * (0.5 + 0.5 * np.sin(TWOPI * trem_rate * t)))
    return tail_fade(finish(y, vel, sr, crest=4.5, curve=1.4), 0.05, sr)


# =============================================================================== clavinet / harpsichord

def clavinet(f, dur, vel, rng, sr):
    vel = float(np.clip(vel, 0.02, 1))
    rel = 0.025
    t60 = float(np.clip(2.0 * (196 / f) ** 0.4, 0.6, 3.0))
    n = note_n(min(dur, t60) + 9 * rel + 0.02, sr)
    t = tvec(n, sr)
    e = pluck_exc(f, rng, vel, 0.25 * (1.4 - 0.9 * vel), 0.3 * vel, 2500 + 9000 * vel ** 1.5, 0.12, sr, 'sine')
    ns = int(n * 1.01) + 32
    s = waveguide(f, ns, e, t60, 0.35, 5000, 0.2, 1, sr)
    s = warp(s, cents(6 * vel * np.exp(-t / 0.04)), n)          # tangent pressure
    # two pickups (neck + bridge), in phase
    out = np.zeros(n)
    for pos, g in ((0.1, 0.6), (0.27, 1.0)):
        di = int(pos * sr / f)
        out += g * (s - 0.92 * shifted(s, di, n))
    out = res_lp(out, 1100 + 4200 * vel ** 1.2, 2.6, sr)
    kd = int(dur * sr)
    if kd < n:
        out[kd:] *= np.exp(-(t[kd:] - t[kd]) / rel)
        kk = min(n - kd, int(0.01 * sr))      # yarn damper thunk
        out[kd:kd + kk] += 0.03 * np.max(np.abs(out)) * hp(rng.standard_normal(kk), 300, 2, sr) * np.exp(-np.arange(kk) / (0.002 * sr))
    out = eq(out, [hp_sos(60, 2, sr)[0], shelf(250, 6, False, sr), peq(1500, 3, 1.0, sr)])
    out = np.tanh((1.0 + 1.5 * vel) * out / (np.max(np.abs(out)) + 1e-9))
    return tail_fade(finish(out, vel, sr, crest=4.5, curve=1.3), 0.05, sr)


def harpsichord(f, dur, vel, rng, sr, four=0.45, lute=False):
    """8' (+4') quill-plucked strings, dampers at gate end, jack-return click."""
    vel = float(np.clip(vel, 0.02, 1))
    vv = 0.35 + 0.65 * vel           # quill voicing: little touch dynamics, but some
    courses = [(1.0, 0.6, 1.0, 0.0, -0.15)]
    if four and f * 2 < 4500:
        courses.append((2.0, 1.0, four, 2.0, 0.2))
    y = plucked(f, dur, vv, rng, sr, t60=4.5, f_ref=196, t60_k=0.5, t60_hi=1.0 if not lute else 0.15,
                f_hi=5000, width=0.12, width_vel=0.4, nail=0.5, nail_vel=0.5, exc_lp=7000, exc_lp_vel=6000,
                pos=0.09, courses=tuple(courses), body='harpsichord', body_mix=0.9, damp=0.06, cap=6.0,
                hp_hz=55.0, knock=0.15, eq_rows=(shelf(300, 5, False), peq(2500, 2, 1.0)))
    # jack falling back: the quill brushes the string -> soft click at release
    kd = int(dur * sr)
    if y.ndim == 1:
        y = np.stack([y, y])
    if kd + int(0.02 * sr) < y.shape[1]:
        kk = int(0.015 * sr)
        cl = bp(rng.standard_normal(kk), 1500, 6000, sr=sr) * np.exp(-np.arange(kk) / (0.002 * sr))
        y[:, kd:kd + kk] += 0.04 * np.max(np.abs(y)) * cl
    return y * (vel / vv) ** 0.6


# =============================================================================== bars & combs

def celesta(f, dur, vel, rng, sr):
    vel = float(np.clip(vel, 0.02, 1))
    tau = float(np.clip(1.4 * (523 / f) ** 0.5, 0.3, 2.5))
    rel = 0.12
    n = note_n(min(dur, 3.5 * tau) + 7 * rel, sr)
    t = tvec(n, sr)
    y = np.zeros(n)
    modes = [(1.0, 1.0, tau), (2.0, 0.05 + 0.04 * vel, tau * 0.5), (2.756, 0.06 + 0.12 * vel ** 1.5, tau * 0.18),
             (5.404, 0.02 + 0.06 * vel ** 2, tau * 0.06), (8.93, 0.03 * vel ** 2, 0.02)]
    for r, a, tk in modes:
        if f * r < 0.45 * sr:
            y += a * _sin32(TWOPI * f * r * (1 + 0.0007 * rng.standard_normal()) * t) * np.exp(-t / tk)
    att = 0.0025 - 0.0015 * vel
    y *= np.clip(t / att, 0, 1)
    k = int(0.01 * sr)
    y[:k] += 0.05 * vel * hp(rng.standard_normal(k), 3000, 2, sr) * np.exp(-np.arange(k) / (0.0015 * sr))
    # resonator box: a little low-mid warmth that follows the bar
    y = apply_body(y, 'reed_box', 0.25, 1.0, 1.6, sr)
    kd = int(dur * sr)
    if kd < n:
        y[kd:] *= np.exp(-(t[kd:] - t[kd]) / rel)
    return tail_fade(finish(y, vel, sr, crest=5.0, curve=1.5), 0.08, sr)


def music_box(f, dur, vel, rng, sr):
    vel = float(np.clip(vel, 0.02, 1))
    tau = float(np.clip(1.6 * (880 / f) ** 0.45, 0.4, 3.0))
    n = note_n(4.5 * tau, sr)
    t = tvec(n, sr)
    y = np.zeros(n)
    beat = rng.uniform(0.6, 2.0)          # two teeth for the note, slightly apart
    for r, a, tk in ((1.0, 1.0, tau), (1.0 + beat / f, 0.45, tau * 0.9), (6.267, 0.1 + 0.25 * vel, tau * 0.07),
                     (17.55, 0.05 * vel, 0.012), (3.0, 0.02, tau * 0.2)):
        if f * r < 0.45 * sr:
            y += a * _sin32(TWOPI * f * r * t + rng.uniform(0, 1)) * np.exp(-t / tk)
    k = int(0.006 * sr)
    y[:k] += (0.1 + 0.2 * vel) * hp(rng.standard_normal(k), 4000, 2, sr) * np.exp(-np.arange(k) / (0.0007 * sr))
    y = apply_body(y, 'reed_box', 0.35, 1.0, 2.2, sr)
    y = eq(y, [hp_sos(250, 2, sr)[0]])
    return tail_fade(finish(y, vel, sr, crest=5.0, curve=1.4), 0.1, sr)


# =============================================================================== mellotron

def mellotron(f, dur, vel, rng, sr, tape_kind='strings'):
    """a tape-replay keyboard: recorded attack, wobbly tape, 8 s tape limit."""
    vel = float(np.clip(vel, 0.02, 1))
    d = min(dur, 7.8)
    if tape_kind == 'strings':
        y = bowed(f, d, 0.55 + 0.25 * vel, rng, sr, auto_body=True, att=0.12, rel=0.25, bright=1600,
                  bright_vel=1600, voices=3, detune=3.5, onset=0.008, spread=0.3, noise=0.05, scratch=0.15,
                  vib=dict(rate=5.8, depth=16, delay=0.12, rise=0.25, drift=2.5, jitter=1.5))
        y = y.mean(0)
    else:  # flute
        n = note_n(d + 0.5, sr)
        t = tvec(n, sr)
        fm = human_pitch(n, rng, sr, rate=5.0, depth=11, delay=0.2, rise=0.3, drift=3, jitter=1.0,
                         scoop=-15, scoop_t=0.04)
        cyc = cycles(f, fm, n, sr)
        K = nharm(f, 1.05, sr, 9000)
        amps = np.array([1.0, 0.22, 0.09, 0.04, 0.02, 0.01][:K])
        tone = wt_osc(wt_table(amps), cyc)
        env = gate_env(n, d, 0.07, 0.09, sr, 1.5) * (1 + 0.04 * smooth_noise(n, 4, rng, sr))
        br = bp(rng.standard_normal(n), f * 1.5, min(f * 6, 12000), sr=sr)
        chiff = np.exp(-t / 0.05) * np.clip(t / 0.01, 0, 1)
        y = tone * env + (0.06 + 0.25 * chiff) * br * env * (0.5 + 0.5 * vel)
    yy = tape(y / (np.max(np.abs(y)) + 1e-9), rng, sr, wow=4.5, flutter=2.5, hiss=0.003, bw=5000 + 6000 * vel,
              sat=1.2 + 0.8 * vel)
    # each key has its own tape: slight fixed tuning offset
    off = rng.normal(0, 2.5)
    yy = warp(np.concatenate([yy, np.zeros(256)]), np.full(len(yy), 2 ** (off / 1200)), len(yy))
    n = len(yy)
    t = tvec(n, sr)
    if dur > 7.8:                      # the tape ran out
        yy *= np.clip((8.0 - t) / 0.2, 0, 1)
    return tail_fade(finish(pan_wide(yy, rng, sr), vel, sr, crest=4.0, curve=1.0, floor=0.25), 0.05, sr)


def pan_wide(y, rng, sr, w=0.5):
    d = int((0.004 + 0.003 * rng.random()) * sr)
    s = shifted(y, d, len(y))
    return np.stack([y + w * 0.3 * s, y - w * 0.3 * s])


# =============================================================================== organs

_DRAW = (0.5, 1.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)


def tonewheel(f, dur, vel, rng, sr, bars='888000000', perc=None, perc_decay=0.3, click=0.6,
              speed=0.0, drive=0.6, vib=0.0, scoop=0.0, leslie_on=True, wave='sine', rel=0.012):
    vel = float(np.clip(vel, 0.02, 1))
    n = note_n(dur + 0.12 + 7 * rel, sr)
    t = tvec(n, sr)
    fm = np.ones(n)
    if vib:
        fm = human_pitch(n, rng, sr, rate=6.6, depth=vib, delay=0.0, rise=0.01, drift=0.5, jitter=0)
    if scoop:
        fm = fm * cents(scoop * np.exp(-t / 0.05))
    cyc = cycles(f, fm, n, sr)
    y = np.zeros(n)
    levels = [int(c) for c in bars]
    for r, lvl in zip(_DRAW, levels):
        if lvl <= 0:
            continue
        rr = r
        while f * rr > 5900:          # tonewheel foldback at the top
            rr /= 2
        while f * rr < 32:
            rr *= 2
        a = 10 ** ((lvl - 8) * 3 / 20)
        start = int(rng.uniform(0, 0.004) * sr)  # busbar contacts close at slightly different times
        ph = TWOPI * rr * cyc + rng.uniform(0, TWOPI)
        if wave == 'sine':
            v = np.sin(ph) + 0.015 * np.sin(2 * ph)
        else:  # 'divider' (mellow filtered square, Lowrey/combo-ish flute tabs)
            v = np.sin(ph) + 0.18 * np.sin(3 * ph) + 0.05 * np.sin(5 * ph) + 0.03 * np.sin(2 * ph)
        v[:start] = 0.0
        y += a * v
    if perc:
        r = {'2nd': 2.0, '3rd': 3.0}[perc]
        y += (0.5 + 0.5 * vel) * 0.9 * np.sin(TWOPI * r * cyc) * np.exp(-t / perc_decay)
    env = gate_env(n, dur, 0.004, rel, sr)
    y *= env
    # key click (also on release)
    kc = int(0.006 * sr)
    clk = bp(rng.standard_normal(kc), 900, 7000, sr=sr) * np.exp(-np.arange(kc) / (0.0012 * sr))
    pk = np.max(np.abs(y)) + 1e-9
    y[:kc] += click * (0.4 + 0.8 * vel) * 0.3 * pk * clk
    kd = int(dur * sr)
    if kd + kc < n:
        y[kd:kd + kc] += click * 0.12 * pk * clk
    # leakage hum of neighbouring wheels (very quiet)
    y += 0.0025 * pk * np.sin(TWOPI * f * 1.4983 * t + 1) * env
    y = y / pk
    dr = drive * (0.6 + 0.8 * vel)
    if dr > 0:
        y = np.tanh((1 + 2.5 * dr) * y + 0.1 * dr) - np.tanh(0.1 * dr)
    y = eq(y, [hp_sos(45, 2, sr)[0], *lp_sos(9000, 2, sr)])
    if leslie_on:
        y = leslie(y, rng, speed, sr)
    return tail_fade(finish(y, vel, sr, crest=3.5, curve=0.8, floor=0.4), 0.05, sr)


def combo(f, dur, vel, rng, sr, duty=0.3, four=0.45, sixteen=0.25, vib=14.0, bright=3500.0):
    """Farfisa-ish combo organ: divider pulse waves, voice-tab filtering, vibrato."""
    vel = float(np.clip(vel, 0.02, 1))
    rel = 0.015
    n = note_n(dur + 7 * rel + 0.02, sr)
    fm = human_pitch(n, rng, sr, rate=6.8, depth=vib, delay=0.0, rise=0.01, drift=0.5, jitter=0, wander=0.02)
    fb = bright + 5000 * vel
    y = np.zeros(n)
    for r, g, D in ((1.0, 1.0, duty), (2.0, four, 0.5), (0.5, sixteen, 0.5)):
        fr = f * r
        if fr > 7000 or fr < 25:
            continue
        K = nharm(fr, 1.02, sr, 15000)
        k = np.arange(1, K + 1)
        amps = 2 / (np.pi * k) * np.sin(np.pi * k * D) / np.sqrt(1 + (k * fr / fb) ** 2)
        y += g * wt_osc(wt_table(amps), cycles(fr, fm, n, sr, rng.random()))
    y *= gate_env(n, dur, 0.003, rel, sr)
    y = eq(y, [hp_sos(80, 2, sr)[0], peq(1400, 4, 1.0, sr), peq(3200, 3, 1.4, sr), *lp_sos(9000, 2, sr)])
    y = np.tanh(1.4 * y / (np.max(np.abs(y)) + 1e-9))
    return tail_fade(finish(y, vel, sr, crest=3.5, curve=0.8, floor=0.4), 0.05, sr)


# ----------------------------------------------------------------- pipes

_PIPE = {
    'principal': lambda k: (k ** -1.25) * (1.0 if k % 2 else 0.85),
    'octave': lambda k: (k ** -1.35),
    'flute': lambda k: [1.0, 0.18, 0.07, 0.03, 0.015, 0.008][k - 1] if k <= 6 else 0.0,
    'stopped': lambda k: ([1.0, 0.03, 0.28, 0.02, 0.09, 0.01, 0.03][k - 1] if k <= 7 else 0.0),
    'reed': lambda k: (k ** -0.65) * (1 + 1.5 * np.exp(-0.5 * ((k - 8) / 3.0) ** 2)),
    'string': lambda k: (k ** -0.8),
}


def pipe(f, dur, vel, rng, sr, ranks=(('principal', 1.0, 1.0), ('flute', 1.0, 0.6), ('octave', 2.0, 0.45)),
         chiff=0.5, wind=0.02, trem=0.0, att=0.06):
    vel = float(np.clip(vel, 0.02, 1))
    rel = 0.06
    n = note_n(dur + 7 * rel + 0.05, sr)
    t = tvec(n, sr)
    y = np.zeros(n)
    wob = human_pitch(n, rng, sr, rate=0.3, depth=0, drift=1.0, jitter=0.4)
    if trem:
        wob = wob * cents(trem * 9 * np.sin(TWOPI * 6.0 * t))
    for kind, ratio, g in ranks:
        fr = f * ratio
        if kind == 'mixture':          # breaks back an octave when it gets too high
            while fr > 3500:
                fr /= 2
            kind = 'octave'
        if fr > 9000 or fr < 15:
            continue
        fr *= cents(rng.normal(0, 1.5))
        K = nharm(fr, 1.01, sr, 13000)
        k = np.arange(1, K + 1)
        fn = _PIPE[kind]
        amps = np.array([fn(i) for i in k], dtype=float)
        amps *= 1 / np.sqrt(1 + (k * fr / (2500 + 3500 * vel)) ** 2)
        a_f = amps.copy()
        a_f[1:] = 0
        a_h = amps.copy()
        a_h[0] = 0
        cyc = cycles(fr, wob, n, sr, rng.random())
        yf, yh = wt_osc(wt_table(a_f), cyc, wt_table(a_h))
        # speech: bigger (lower) pipes speak slower; harmonics speak first
        sp = att * (1.25 - 0.4 * vel) * float(np.clip((262 / fr) ** 0.4, 0.4, 2.5))
        ef = gate_env(n, dur, sp, rel, sr, 1.0)
        eh = gate_env(n, dur, sp * 0.4, rel * 0.8, sr, 1.0)
        over = 1 + 0.6 * np.exp(-t / (sp * 0.8)) * np.clip(t / (sp * 0.3), 0, 1)   # overblown onset
        y += g * (yf * ef + yh * eh * over)
        if chiff and kind in ('flute', 'stopped', 'principal', 'octave'):
            cn = bp(rng.standard_normal(n), fr * 1.8, min(fr * 7, 15000), sr=sr)
            ce = chiff * (0.4 + 0.6 * vel) * np.exp(-t / (0.025 + sp * 0.3)) * np.clip(t / 0.004, 0, 1)
            y += g * 0.18 * cn * ce
    pk = np.max(np.abs(y)) + 1e-9
    y += wind * pk * bp(rng.standard_normal(n), 900, 5000, sr=sr) * gate_env(n, dur, 0.05, 0.1, sr)
    y = eq(y, [hp_sos(min(30, f * 0.4), 2, sr)[0]])
    return tail_fade(finish(pan_wide(y, rng, sr, 0.6), vel, sr, crest=3.5, curve=0.7, floor=0.45), 0.05, sr)


# ----------------------------------------------------------------- free reeds

def free_reed(f, dur, vel, rng, sr, reeds=((1.0, 0.0, 1.0),), bright=2200.0, bright_vel=3500.0,
              width=0.18, att=0.04, rel=0.07, bellows=(1.2, 0.04), noise=0.02, formants=((1600, 4, 1.2),),
              body_mix=0.35, flat=8.0, jitter=0.0, trem=None, tilt=1.0, hp_hz=60):
    vel = float(np.clip(vel, 0.02, 1))
    n = note_n(dur + 7 * rel + 0.05, sr)
    t = tvec(n, sr)
    a_eff = att * (1.4 - 0.7 * vel)
    env = gate_env(n, dur, a_eff, rel, sr, 1.2)
    br, bd = bellows
    press = env * (1 + bd * smooth_noise(n, br, rng, sr))
    if trem:
        press = press * (1 + trem[1] * np.sin(TWOPI * trem[0] * t))
    y = np.zeros(n)
    for ratio, dc, g in reeds:
        fr = f * ratio * cents(dc + rng.normal(0, 1.0))
        if fr > 8000 or fr < 20:
            continue
        # reeds go slightly flat at low pressure (onset), tiny jitter
        fm = cents(-flat * (1 - np.clip(press, 0, 1)) * np.exp(-t / 0.15)) * human_pitch(
            n, rng, sr, rate=0.2, depth=0, drift=1.2, jitter=jitter)
        K = nharm(fr, 1.01, sr, 14000)
        k = np.arange(1, K + 1)
        base = np.abs(np.sinc(k * width)) ** 0.8 + 0.02
        base *= k ** (-tilt * 0.5) * 10 ** (rng.normal(0, 0.8, K) / 20)
        soft = base / (1 + (k * fr / (bright * 0.45)) ** 2)
        loud = base / (1 + (k * fr / (bright + bright_vel * vel)) ** 2)
        cyc = cycles(fr, fm, n, sr, rng.random())
        ys, yl = wt_osc(wt_table(soft), cyc, wt_table(loud))
        m = np.clip(press * (0.3 + 0.7 * vel), 0, 1)
        y += g * (ys + m * (yl - ys))
    y = y * press
    pk = np.max(np.abs(y)) + 1e-9
    if noise:
        y += noise * pk * bp(rng.standard_normal(n), 1200, 7000, sr=sr) * press * (0.5 + 0.8 * vel)
    rows = [hp_sos(hp_hz, 2, sr)[0]] + [peq(fq, gdb, q, sr) for fq, gdb, q in formants]
    y = eq(y, rows)
    y = apply_body(y, 'reed_box', body_mix, 1.0, 1.0, sr)
    return tail_fade(finish(pan_wide(y, rng, sr, 0.4), vel, sr, crest=4.0, curve=1.2, floor=0.15), 0.05, sr)


# =============================================================================== registration

def _r(name, fn, fam, lo, hi, tags, desc, sustain=True, **p):
    register(name, fn, fam, lo=lo, hi=hi, tags=tags, desc=desc, sustain=sustain, **p)


# ---- keys
_r('keys.grand_piano', piano, 'keys', 'A0', 'C8', ('oldfield', 'orchestral'),
   'Concert grand: stiff inharmonic strings, felt hammers that brighten with force, unison beating, soundboard thump')
_r('keys.upright_piano', piano, 'keys', 'A0', 'C8', ('oldfield', 'world'),
   'Upright parlour piano: shorter, more inharmonic strings, boxy cabinet, a woody knock and quicker fade',
   b_scale=1.8, unison=1.1, tau_scale=0.6, hammer=1.1, strike=0.13, body='upright', body_mix=0.75,
   thump=0.8, after=0.22, spread=0.35, tack=0.3)
_r('keys.honky_tonk', piano, 'keys', 'C1', 'C8', ('rock', 'world'),
   'Honky-tonk saloon piano: badly detuned unisons warbling against each other, hard worn hammers',
   b_scale=1.6, unison=1.0, honky=9.0, tau_scale=0.55, hammer=1.5, strike=0.11, body='upright', body_mix=0.7,
   thump=0.6, after=0.3, spread=0.3, tack=0.6, bright_tilt=0.2)
_r('keys.epiano_tine', epiano, 'keys', 'E1', 'G7', ('space', 'rock'),
   'Tine electric piano: bell-like ping, round body, pickup bark when struck hard')
_r('keys.epiano_suitcase', epiano, 'keys', 'E1', 'G7', ('space',),
   'Suitcase tine piano: mellow bell tone swimming in stereo auto-pan tremolo',
   bark=0.6, bell=0.7, trem=0.8, trem_rate=4.6, q_off=0.3)
_r('keys.epiano_reed', wurli, 'keys', 'A1', 'C7', ('rock', 'space'),
   'Reed electric piano: nasal steel-reed tone that barks and growls under a heavy hand')
_r('keys.clavinet', clavinet, 'keys', 'F1', 'E6', ('rock',),
   'Clavinet: rubber-tip struck string, funky two-pickup quack, yarn-damped staccato')
_r('keys.harpsichord', harpsichord, 'keys', 'F1', 'F6', ('orchestral', 'oldfield'),
   'Harpsichord: 8+4 quill-plucked choirs, brilliant and metallic, jack-return click on release')
_r('keys.celesta', celesta, 'keys', 'C4', 'C8', ('orchestral', 'space'),
   'Celesta: felt hammers on steel bars over wooden resonators, sugar-plum soft shimmer')
_r('keys.music_box', music_box, 'keys', 'C5', 'C8', ('space', 'inharmonic'),
   'Music box: pinned steel comb teeth, tinkling inharmonic ping with paired-tooth beating', sustain=False)
_r('keys.mellotron_strings', mellotron, 'keys', 'G2', 'F5', ('oldfield', 'space'),
   'Mellotron tape strings: wobbly three-violin tape replay, dusty bandwidth, runs out after 8 s',
   tape_kind='strings')
_r('keys.mellotron_flute', mellotron, 'keys', 'G3', 'F6', ('oldfield', 'space'),
   'Mellotron tape flute: breathy, wavering, nostalgic tape flute with recorded vibrato', tape_kind='flute')

# ---- organ
_r('organ.drawbar', tonewheel, 'organ', 'C2', 'C7', ('rock', 'oldfield'),
   'Tonewheel organ 888000000: key click, 3rd-harmonic percussion, warm tube drive, slow Leslie chorale',
   bars='888000000', perc='3rd', perc_decay=0.25, click=0.7, speed=0.0, drive=0.5)
_r('organ.drawbar_fast', tonewheel, 'organ', 'C2', 'C7', ('rock',),
   'Full-drawbar screaming organ through an overdriven Leslie on fast',
   bars='888888888', perc=None, click=0.5, speed=1.0, drive=1.3)
_r('organ.drawbar_soft', tonewheel, 'organ', 'C2', 'C7', ('space', 'oldfield'),
   'Soft gospel/ballad registration 008800000 with slow Leslie: hollow and breathy',
   bars='008800000', perc=None, click=0.35, speed=0.0, drive=0.2)
_r('organ.lowrey', tonewheel, 'organ', 'C2', 'C7', ('oldfield', 'rock'),
   'Lowrey-style home organ: mellow divider flute tabs 16+8+4, lush vibrato, gentle glide into notes, slow rotor',
   bars='606800300', wave='divider', vib=16.0, scoop=-25, click=0.15, speed=0.0, drive=0.35)
_r('organ.farfisa', combo, 'organ', 'C2', 'C7', ('oldfield', 'rock'),
   'Farfisa combo organ: thin reedy pulse waves, bright and buzzy with fast vibrato')
_r('organ.pipe', pipe, 'organ', 'C1', 'C7', ('oldfield', 'orchestral'),
   'Church pipe organ: principal 8, flute 8 and octave 4, chiffing speech and wind, rank beating')
_r('organ.pipe_plenum', pipe, 'organ', 'C1', 'C7', ('orchestral', 'oldfield'),
   'Full organ plenum: 16 bourdon to mixture and trumpet, majestic cathedral grandeur',
   ranks=(('stopped', 0.5, 0.7), ('principal', 1.0, 1.0), ('octave', 2.0, 0.6), ('octave', 4.0, 0.35),
          ('mixture', 6.0, 0.25), ('mixture', 8.0, 0.2), ('reed', 1.0, 0.45)), chiff=0.3, wind=0.015, att=0.05)
_r('organ.pipe_flute', pipe, 'organ', 'C2', 'C7', ('oldfield', 'celtic', 'space'),
   'Chiffy stopped flute 8 + open flute 4 with tremulant: woody, breathy chamber organ',
   ranks=(('stopped', 1.0, 1.0), ('flute', 2.0, 0.5)), chiff=1.0, wind=0.04, trem=1.0, att=0.07)
_r('organ.harmonium', free_reed, 'organ', 'C2', 'C7', ('asian', 'world', 'oldfield'),
   'Hand-pumped harmonium: two reeds a hair apart, bright nasal buzz, bellows breathing',
   reeds=((1.0, -2.5, 1.0), (1.0, 2.5, 0.85)), bright=2600, bright_vel=3000, width=0.14, att=0.05,
   bellows=(0.8, 0.06), noise=0.025, formants=((1300, 5, 1.0), (2800, 3, 1.5)), flat=10.0)
_r('organ.reed_pump', free_reed, 'organ', 'C2', 'C7', ('oldfield', 'celtic'),
   'Victorian pump reed organ: soft 8+4 reeds, slow wheezy speech, vox-humana fan tremolo',
   reeds=((1.0, 0.0, 1.0), (2.0, 1.5, 0.35)), bright=1300, bright_vel=1500, width=0.22, att=0.09, rel=0.1,
   bellows=(0.5, 0.05), noise=0.04, formants=((800, 3, 0.8),), flat=12.0, trem=(5.2, 0.12), tilt=1.4)
_r('organ.accordion_musette', free_reed, 'organ', 'F2', 'A6', ('world', 'celtic'),
   'Musette accordion: three reeds detuned wet, shimmering Parisian beat, bellows swells',
   reeds=((1.0, -15.0, 0.9), (1.0, 0.0, 1.0), (1.0, 16.0, 0.9)), bright=2400, bright_vel=3500, width=0.16,
   att=0.03, bellows=(1.5, 0.05), noise=0.015, formants=((1800, 4, 1.0), (3500, 2, 1.5)))
_r('organ.concertina', free_reed, 'organ', 'G3', 'C7', ('celtic', 'world'),
   'English concertina: single sweet reed per button, quick crisp speech, small hexagonal box',
   reeds=((1.0, 0.0, 1.0),), bright=2800, bright_vel=3000, width=0.2, att=0.018, rel=0.05,
   bellows=(2.0, 0.035), noise=0.02, formants=((2200, 4, 1.2),), body_mix=0.25)
_r('organ.bandoneon', free_reed, 'organ', 'C2', 'A6', ('latin', 'world'),
   'Bandoneon: octave-paired reeds, dark woody tango melancholy, expressive slow swell',
   reeds=((1.0, 0.0, 1.0), (2.0, 2.0, 0.55)), bright=1400, bright_vel=2400, width=0.2, att=0.08, rel=0.09,
   bellows=(0.9, 0.05), noise=0.02, formants=((700, 4, 1.0), (1700, 2, 1.2)), tilt=1.3)
_r('organ.melodica', free_reed, 'organ', 'F3', 'F6', ('world', 'space', 'jungle'),
   'Melodica: mouth-blown single reed, hollow nasal honk with breath and a wavering tongue attack',
   reeds=((1.0, 0.0, 1.0),), bright=2000, bright_vel=3500, width=0.3, att=0.025, rel=0.05,
   bellows=(3.0, 0.05), noise=0.06, formants=((1100, 5, 1.2), (2500, 3, 1.5)), jitter=2.5, flat=15.0,
   body_mix=0.15, tilt=0.6)
