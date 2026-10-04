"""
Canopy Orchestra — winds: families 'flute', 'reed', 'brass'.

One parametric blown-instrument model (`blown`) covers most presets:

    source   Moorer DSF harmonic series (all or odd harmonics) whose rolloff
             follows *blowing pressure* = gate envelope x velocity, so notes
             start darker, bloom, and loud notes are brighter
    pitch    scoop into the note, slow drift, delayed vibrato that develops,
             optional grace note (pipes / whistle cuts) and end fall
    air      breath noise band-limited per instrument, partly pitch-synchronous
             and partly resonant at f / 2f (edge-tone hiss), chiff at onset
    body     parallel resonant peaks (bore / bell / membrane colour) and
             optional cascade formants, high/low-pass, hand 'wah'

Specials: bagpipe and uilleann drones, uilleann regulators, didgeridoo.
"""
from __future__ import annotations

import numpy as np

from .core import *
from .luthier import (dsf, roll_a, contour, gate, note_len, breath, burst, cascade,
                      tv_cascade, tv_lowpass, peaks, smooth_noise, onepole_lp, mix_stereo,
                      env_points, tame)


def blown(f, dur, vel, rng, sr, *,
          odd=False, even=1.0, fr=2000.0, fr_vel=2000.0, fr_env=0.5, fr_att=0.0, tilt=0.0,
          att=0.04, rel=0.08, dec=0.0, sus=1.0, swell=0.0, overshoot=0.0, ov_t=0.05,
          scoop=-20.0, scoop_t=0.03, vib_rate=5.0, vib_cents=0.0, vib_delay=0.3, vib_grow=0.5,
          trem=0.0, drift=3.0, jitter=0.0, fall=0.0, fall_t=0.1,
          noise=0.1, noise_lo=1500.0, noise_hi=7000.0, noise_vel=0.0, sync=0.4, tonal=0.0, nq=8.0,
          chiff=0.0, chiff_lo=1500.0, chiff_hi=7000.0, chiff_tau=0.012, pop=0.0, pop_tau=0.02,
          body=(), body_mix=1.0, formants=(), hp_hz=0.0, lp_hz=0.0, wah=None,
          grace=0.0, grace_t=0.03, shimmer=0.0, flutter=0.0, sub=0.0, vpow=1.0, ppow=1.0, sat=0.0, level=1.0):
    n = note_len(dur, rel, sr, extra=0.06)
    t = tvec(n, sr)
    # ---- pitch: grace note (pipes' gracenote, whistle 'cut'), scoop, vibrato, drift
    bend = None
    if grace:
        g = (t < grace_t).astype(float)
        bend = onepole_lp(g, 250.0, sr) * grace * 100.0
    fa = contour(n, f, dur, rng, sr, scoop=scoop, scoop_t=scoop_t, vib_rate=vib_rate,
                 vib_cents=vib_cents * (0.6 + 0.6 * vel), vib_delay=vib_delay, vib_grow=vib_grow,
                 drift=drift, jitter=jitter, fall=fall, fall_t=fall_t, bend=bend)
    ph = phase_from_freq(fa, sr, rng.uniform(0, TWOPI))
    # ---- pressure envelope
    e = gate(n, dur, att, rel, sr, dec=dec, sus=sus, swell=swell)
    if overshoot:
        e = e * (1 + overshoot * vel * np.exp(-np.maximum(t - att * 0.6, 0) / ov_t) * np.clip(t / max(att, 1e-3), 0, 1))
    if trem:
        tp = TWOPI * vib_rate * t + 1.3
        ramp = np.clip((t - vib_delay) / max(vib_grow, 1e-3), 0, 1)
        e = e * (1 + trem * ramp * np.sin(tp))
    if shimmer:
        e = e * (1 + shimmer * smooth_noise(n, 30.0, rng, sr))
    pres = np.clip(e / (1 + overshoot * vel + 1e-9), 0, 1.2)
    # ---- brightness follows pressure and velocity
    frx = (fr + fr_vel * vel ** vpow) * (1 - fr_env + fr_env * pres ** ppow)
    if fr_att:
        frx = frx * (1 + fr_att * vel * np.exp(-t / 0.04))
    a = roll_a(f, frx)
    if odd:
        y = dsf(fa, a, odd=True, ph=ph)
        if even:
            y = y + even * dsf(fa, a, ph=2 * ph) * a      # even partials: 2,4,6..
    else:
        y = dsf(fa, a, ph=ph)
    if sub:
        y = y + sub * np.sin(ph)
    if tilt:
        y = onepole_lp(y, f * tilt, sr) * (1 + 1.0 / tilt)
    if pop:
        y = y + pop * vel * np.sin(2 * ph) * np.exp(-t / pop_tau)
    if flutter:
        y = y * (1 + flutter * smooth_noise(n, 18.0, rng, sr))
    y = y * e
    # ---- air
    if noise:
        lo = max(noise_lo, f * 0.9)
        hi = max(noise_hi, lo * 1.5)
        nb = breath(n, rng, f, lo, hi, sr, ph=ph, sync=sync, tonal=tonal, q=nq)
        y = y + noise * (1 + noise_vel * (vel - 0.5)) * nb * np.sqrt(np.clip(e, 0, None))
    if chiff:
        k = min(n, int(0.08 * sr))
        y[:k] += chiff * (0.4 + 0.8 * vel) * burst(k, rng, chiff_lo, chiff_hi, 0.0015, chiff_tau, sr)
    # ---- body
    if body:
        y = (1 - min(body_mix, 1.0)) * y + body_mix * (y + peaks(y, body, sr))
    if formants:
        y = cascade(y, formants, sr)
        y /= max(1.0, float(np.mean([F / B for F, B in formants])) ** 0.5)
    if wah is not None:
        f0w, f1w, tw = wah
        fc = f0w + (f1w - f0w) * (1 - np.exp(-t / tw))
        y = tv_lowpass(y, fc * (0.7 + 0.5 * vel), q=1.6, sr=sr)
    if hp_hz:
        y = hp(y, hp_hz, sr=sr)
    if lp_hz:
        y = lp(y, lp_hz * (0.7 + 0.5 * vel), sr=sr)
    if sat:
        pk = np.percentile(np.abs(y), 99.5) + 1e-9
        y = np.tanh(sat * y / pk) * pk / sat
    return y * level * (0.35 + 0.65 * vel)


# --------------------------------------------------------------------------- pipes & drones


def drones(f, dur, vel, rng, sr, *, ratios=(1.0, 1.0, 0.5), detune=(0.0, 2.5, -1.0), fr=4500.0,
           odd=False, att=0.35, rel=0.25, noise=0.04, body=(), strike=0.6, pans=(-0.4, 0.35, 0.0),
           gains=(0.8, 0.7, 1.0), lp_hz=0.0, level=1.0):
    """Pipe drones (sustained). Each drone is a reed into a cylindrical bore; they
    'strike in' with a gurgle (pressure wobble) and beat slowly against each other."""
    n = note_len(dur, rel, sr, extra=0.05)
    t = tvec(n, sr)
    parts = []
    for i, (r, dc) in enumerate(zip(ratios, detune)):
        fi = f * r
        wob = strike * 40 * np.exp(-t / 0.12) * np.sin(TWOPI * rng.uniform(9, 14) * t)
        fa = contour(n, fi, dur, rng, sr, scoop=-60 * strike, scoop_t=0.08, drift=1.0, bend=wob + dc)
        ph = phase_from_freq(fa, sr, rng.uniform(0, TWOPI))
        e = gate(n, dur, att * rng.uniform(0.8, 1.2), rel, sr)
        a = roll_a(fi, (fr * (0.6 + 0.5 * vel)) * (0.5 + 0.5 * e))
        y = (dsf(fa, a, ph=ph, odd=odd) + (0.25 * dsf(fa, a, ph=2 * ph) * a if odd else 0)) * e
        y += noise * breath(n, rng, fi, 1500, 6000, sr, ph=ph, sync=0.7) * e
        if body:
            y = y + peaks(y, body, sr)
        if lp_hz:
            y = lp(y, lp_hz, sr=sr)
        pk = np.percentile(np.abs(y), 99.5) + 1e-9
        y = np.tanh(1.5 * y / pk) * pk
        parts.append((gains[i] * y, pans[i]))
    return tame(mix_stereo(parts), 1.4) * level * (0.5 + 0.5 * vel)


def regulators(f, dur, vel, rng, sr, *, intervals=(0, 7, 12), fr=2600.0, rel=0.035, level=1.0):
    """Uilleann regulators: keyed chord stabs from closed reed pipes, played with
    the wrist — crisp, sweet, slightly uneven onsets."""
    parts = []
    for i, iv in enumerate(intervals):
        fi = f * 2 ** (iv / 12)
        d = int(rng.uniform(0, 0.012) * sr)
        y = blown(fi, max(dur - d / sr, 0.03), vel, rng, sr, odd=True, even=0.35, fr=fr, fr_vel=1500,
                  att=0.008, rel=rel, scoop=-8, drift=1.5, noise=0.04, chiff=0.15,
                  body=((1800, 3, 0.4),), level=[1.0, 0.7, 0.55][i % 3])
        parts.append((np.concatenate([np.zeros(d), y]), [-0.2, 0.25, 0.05][i % 3]))
    return tame(mix_stereo(parts), 1.3) * level


def didgeridoo(f, dur, vel, rng, sr, *, wow_rate=2.2, toot=0.0, level=1.0):
    """Didgeridoo drone: buzzing lips into a long eucalyptus bore. The player's
    vocal tract formants sweep ('wow-wow'), the tongue pushes an overtone band
    up and down, breath pulses accent the rhythm."""
    rel = 0.12
    n = note_len(dur, rel, sr, extra=0.05)
    t = tvec(n, sr)
    fa = contour(n, f, dur, rng, sr, scoop=-35, scoop_t=0.05, drift=4, jitter=4)
    ph = phase_from_freq(fa, sr)
    e = gate(n, dur, 0.06, rel, sr)
    # rhythmic breath/tongue pulses
    pr = wow_rate * 2 * (1 + 0.1 * smooth_noise(n, 0.5, rng, sr))
    pul = 0.5 + 0.5 * np.cos(TWOPI * np.cumsum(pr) / sr + rng.uniform(0, 6))
    e = e * (0.72 + 0.28 * pul ** 2)
    a = roll_a(f, 1800 + 2500 * vel) * np.ones(n)
    src = dsf(fa, a, ph=ph) * e
    src += 0.08 * breath(n, rng, f, 200, 3000, sr, ph=ph, sync=0.9) * e
    # the bore: odd-ish resonances (flared cone)
    src = src + peaks(src, ((f * 2.6, 6, 0.8), (f * 4.3, 7, 0.5), (f * 6.1, 8, 0.3)), sr)
    # vocal tract 'wow': F1 and F2 sweep together; F3 = tongue overtone band
    wp = TWOPI * np.cumsum(wow_rate * (1 + 0.15 * smooth_noise(n, 0.4, rng, sr))) / sr
    wow = 0.5 - 0.5 * np.cos(wp)
    wow = wow ** 1.5
    F1 = 280 + 520 * wow
    F2 = 900 + 1300 * wow
    F3 = 1700 + 900 * (0.5 + 0.5 * np.sin(0.37 * wp + 1.0)) + toot * 600
    y = tv_cascade(src, np.stack([F1, F2, F3]), np.stack([np.full(n, 90.0), np.full(n, 140.0), np.full(n, 160.0)]), sr)
    y = y * (np.std(src) / (np.std(y) + 1e-12)) + 0.3 * lp(src, 250, sr=sr)
    y = tame(y, 1.3)
    return y * level * (0.4 + 0.6 * vel)


# --------------------------------------------------------------------------- registry

# ---- flutes ---------------------------------------------------------------
register('flute.concert', blown, 'flute', lo='C4', hi='C7', tags=('orchestral', 'oldfield'), sustain=True,
         desc='silver concert flute: clean breathy edge tone, flute vibrato that blooms in amplitude and pitch',
         fr=1300, fr_vel=2400, fr_env=0.6, att=0.05, rel=0.07, scoop=-15, vib_rate=5.3, vib_cents=14,
         vib_delay=0.18, vib_grow=0.4, trem=0.12, noise=0.14, noise_lo=1800, noise_hi=9000, tonal=0.6,
         sync=0.5, chiff=0.08, tilt=3.0, body=((2400, 2.5, 0.25),))
register('flute.bamboo', blown, 'flute', lo='G4', hi='D7', tags=('asian', 'world', 'jungle'), sustain=True,
         desc='bamboo dizi: bright, reedy buzz from the dimo membrane, quick ornamental scoops',
         fr=1700, fr_vel=3000, fr_env=0.5, att=0.03, rel=0.06, scoop=-45, scoop_t=0.035, vib_rate=5.8,
         vib_cents=18, vib_delay=0.25, noise=0.16, noise_lo=2000, noise_hi=9000, tonal=0.5, chiff=0.12,
         body=((2900, 14, 0.5), (4300, 16, 0.45), (6100, 18, 0.35)), flutter=0.12)
register('flute.pan_flute', blown, 'flute', lo='C4', hi='G6', tags=('latin', 'world', 'oldfield'), sustain=True,
         desc='stopped-cane pan pipes: hollow odd-harmonic tone, big breathy "puh" chiff, late slow vibrato',
         odd=True, even=0.18, fr=900, fr_vel=1600, fr_env=0.7, att=0.045, rel=0.09, scoop=-25, vib_rate=4.6,
         vib_cents=10, vib_delay=0.45, vib_grow=0.6, noise=0.32, noise_lo=900, noise_hi=6000, tonal=0.9, nq=5,
         sync=0.6, chiff=0.6, chiff_lo=600, chiff_hi=5000, chiff_tau=0.03, pop=0.3)
register('flute.quena', blown, 'flute', lo='G4', hi='E7', tags=('latin', 'world'), sustain=True,
         desc='Andean quena: notched-edge attack with a sforzando puff, wide fast vibrato, wild breath',
         fr=1500, fr_vel=3200, fr_env=0.7, att=0.025, rel=0.07, dec=0.25, sus=0.75, overshoot=0.35,
         scoop=-30, vib_rate=6.2, vib_cents=26, vib_delay=0.2, vib_grow=0.35, trem=0.1, noise=0.26,
         noise_lo=1500, noise_hi=8000, noise_vel=0.6, tonal=0.7, chiff=0.35, chiff_tau=0.02)
register('flute.shakuhachi', blown, 'flute', lo='D4', hi='D6', tags=('asian', 'world'), sustain=True,
         desc='shakuhachi: muraiki breath explosion, meri scoop from below, deep yuri vibrato, atari fall-off',
         fr=900, fr_vel=2200, fr_env=0.8, fr_att=1.2, att=0.06, rel=0.1, dec=0.3, sus=0.8, scoop=-90,
         scoop_t=0.045, vib_rate=4.4, vib_cents=30, vib_delay=0.35, vib_grow=0.6, trem=0.15, fall=-70,
         fall_t=0.08, noise=0.38, noise_lo=700, noise_hi=7000, noise_vel=0.8, sync=0.6, tonal=0.9, nq=4,
         chiff=0.9, chiff_lo=400, chiff_hi=6000, chiff_tau=0.05, flutter=0.1, drift=5)
register('flute.ocarina', blown, 'flute', lo='C4', hi='F6', tags=('latin', 'world', 'chip'), sustain=True,
         desc='clay vessel ocarina: almost pure, round and hooty, gentle chiff, soft vibrato',
         fr=380, fr_vel=500, fr_env=0.4, att=0.035, rel=0.06, scoop=-12, vib_rate=5.0, vib_cents=9,
         vib_delay=0.3, noise=0.07, noise_lo=700, noise_hi=3000, tonal=0.8, chiff=0.08, chiff_lo=800,
         chiff_hi=3500, lp_hz=4000)
register('flute.tin_whistle', blown, 'flute', lo='D5', hi='D7', tags=('celtic', 'oldfield'), sustain=True,
         desc='Irish tin whistle: bright, pure-and-piercing fipple tone, instant attack for cuts and rolls',
         fr=1900, fr_vel=3500, fr_env=0.4, att=0.007, rel=0.035, scoop=-10, scoop_t=0.01, vib_rate=5.6,
         vib_cents=8, vib_delay=0.4, vib_grow=0.5, noise=0.13, noise_lo=2500, noise_hi=11000, tonal=0.5,
         chiff=0.22, chiff_lo=3000, chiff_hi=10000, chiff_tau=0.006, drift=2, body=((3800, 4, 0.2),))
register('flute.tin_whistle_cut', blown, 'flute', lo='D5', hi='D7', tags=('celtic', 'oldfield'), sustain=True,
         desc='tin whistle with a built-in "cut" grace note flicked above each attack (Irish ornament)',
         fr=1900, fr_vel=3500, fr_env=0.4, att=0.006, rel=0.035, scoop=0, grace=3.0, grace_t=0.028,
         vib_rate=5.6, vib_cents=8, vib_delay=0.4, noise=0.13, noise_lo=2500, noise_hi=11000, tonal=0.5,
         chiff=0.3, chiff_lo=3000, chiff_hi=10000, chiff_tau=0.005, drift=2, body=((3800, 4, 0.2),))
register('flute.recorder_alto', blown, 'flute', lo='F4', hi='G6', tags=('orchestral', 'oldfield', 'celtic'),
         sustain=True, desc='alto recorder: woody, plain, nearly vibrato-free; soft "du" tongued fipple chiff',
         fr=950, fr_vel=1400, fr_env=0.3, att=0.022, rel=0.045, scoop=-8, vib_cents=4, vib_delay=0.6,
         noise=0.06, noise_lo=1500, noise_hi=6000, tonal=0.4, chiff=0.18, chiff_lo=1200, chiff_hi=5000,
         chiff_tau=0.01, pop=0.25, pop_tau=0.015, body=((1300, 3, 0.35), (2600, 4, 0.2)))
register('flute.recorder_sopranino', blown, 'flute', lo='F5', hi='G7', tags=('orchestral', 'oldfield'),
         sustain=True, desc='sopranino recorder: tiny, piping, squeaky-bright tongued attack',
         fr=1500, fr_vel=2500, fr_env=0.3, att=0.012, rel=0.035, scoop=-6, vib_cents=5, vib_delay=0.5,
         noise=0.08, noise_lo=3000, noise_hi=12000, tonal=0.4, chiff=0.25, chiff_lo=3000,
         chiff_hi=11000, chiff_tau=0.006, pop=0.35, pop_tau=0.01)
register('flute.flageolet', blown, 'flute', lo='G4', hi='A6', tags=('celtic', 'oldfield', 'orchestral'),
         sustain=True, desc='flageolet: sweet sponge-chamber whistle, gentle, slightly veiled, light vibrato',
         fr=800, fr_vel=1200, fr_env=0.5, att=0.03, rel=0.06, scoop=-12, vib_rate=5.2, vib_cents=10,
         vib_delay=0.3, trem=0.06, noise=0.09, noise_lo=1800, noise_hi=7000, tonal=0.5, chiff=0.1,
         chiff_tau=0.012, lp_hz=6000, body=((1700, 3, 0.25),))
register('flute.nose_flute', blown, 'flute', lo='C4', hi='C6', tags=('asian', 'world', 'jungle'), sustain=True,
         desc='nose flute: soft hollow nasal hoot, wavering unsteady pitch, low airy hiss',
         fr=320, fr_vel=400, fr_env=0.5, att=0.09, rel=0.1, scoop=-40, scoop_t=0.07, vib_rate=3.6,
         vib_cents=16, vib_delay=0.25, drift=9, trem=0.18, noise=0.22, noise_lo=400, noise_hi=2500,
         tonal=0.9, nq=4, sync=0.2, lp_hz=2500)
register('flute.bansuri', blown, 'flute', lo='E3', hi='E6', tags=('asian', 'world'), sustain=True,
         desc='bansuri: deep, breathy bamboo; slow meend glide up into the note, late singing vibrato',
         fr=700, fr_vel=1800, fr_env=0.7, att=0.08, rel=0.1, scoop=-110, scoop_t=0.06, vib_rate=4.8,
         vib_cents=22, vib_delay=0.45, vib_grow=0.7, trem=0.1, noise=0.3, noise_lo=500, noise_hi=6000,
         tonal=1.0, nq=5, sync=0.5, chiff=0.12, chiff_lo=500, chiff_hi=3000, chiff_tau=0.03, drift=4)

# ---- reeds ----------------------------------------------------------------
register('reed.clarinet', blown, 'reed', lo='D3', hi='C6', tags=('orchestral', 'oldfield'), sustain=True,
         desc='clarinet: hollow odd-harmonic chalumeau, warming to brighter clarion when pushed; straight tone',
         odd=True, even=0.12, fr=900, fr_vel=2600, fr_env=0.7, att=0.03, rel=0.05, scoop=-10, vib_cents=0,
         drift=2, noise=0.04, noise_lo=1500, noise_hi=6000, sync=0.8, chiff=0.06, chiff_lo=800,
         chiff_hi=3000, body=((1500, 3, 0.3), (3000, 4, 0.15)), lp_hz=7000)
register('reed.oboe', blown, 'reed', lo='Bb3', hi='F6', tags=('orchestral', 'oldfield'), sustain=True,
         desc='oboe: nasal, plaintive double reed with strong bright formants and a quick, tight vibrato',
         fr=2600, fr_vel=3000, fr_env=0.6, att=0.025, rel=0.05, scoop=-15, vib_rate=5.6, vib_cents=12,
         vib_delay=0.2, vib_grow=0.3, trem=0.05, noise=0.03, noise_lo=2000, noise_hi=7000, sync=0.9,
         shimmer=0.02, body=((1150, 6, 1.6), (2900, 7, 0.9), (4200, 8, 0.3)), hp_hz=250, tilt=0)
register('reed.bassoon', blown, 'reed', lo='Bb1', hi='D5', tags=('orchestral',), sustain=True,
         desc='bassoon: dark buzzing double reed, woody 500 Hz formant, comic in staccato, noble when held',
         fr=900, fr_vel=1500, fr_env=0.6, att=0.035, rel=0.06, scoop=-12, vib_rate=5.0, vib_cents=8,
         vib_delay=0.35, noise=0.03, noise_lo=800, noise_hi=4000, sync=0.9, shimmer=0.02,
         body=((480, 4, 1.4), (1150, 5, 0.8), (2500, 6, 0.25)), lp_hz=5000)
register('reed.soprano_sax', blown, 'reed', lo='Ab3', hi='E6', tags=('oldfield', 'world'), sustain=True,
         desc='soprano sax: bright conical reed, airy subtone at soft, singing developing vibrato',
         fr=1700, fr_vel=3600, fr_env=0.8, att=0.03, rel=0.06, scoop=-35, scoop_t=0.04, vib_rate=5.4,
         vib_cents=20, vib_delay=0.3, vib_grow=0.6, noise=0.1, noise_lo=2500, noise_hi=9000, noise_vel=-0.6,
         sync=0.6, shimmer=0.03, jitter=2, body=((850, 3, 0.6), (2300, 4, 0.5)), lp_hz=9000)
register('reed.highland_chanter', blown, 'reed', lo='G4', hi='A5', tags=('celtic', 'oldfield'), sustain=True,
         desc='Highland bagpipe chanter: blaring, piercing, constant-pressure reed with a gracenote snap',
         fr=5000, fr_vel=1500, fr_env=0.2, att=0.01, rel=0.03, scoop=0, grace=7.0, grace_t=0.025,
         vib_cents=0, drift=1.5, noise=0.04, noise_lo=2500, noise_hi=8000, sync=0.9,
         body=((1400, 4, 0.7), (3100, 5, 0.8), (4800, 6, 0.3)), hp_hz=200)
register('reed.highland_drones', drones, 'reed', lo='A2', hi='A3', tags=('celtic', 'oldfield'), sustain=True,
         desc='Highland drones: two beating tenor drones + bass drone an octave down, striking in with a gurgle',
         ratios=(1.0, 1.0, 0.5), detune=(0.0, 2.5, -1.0), fr=3800, att=0.35, rel=0.25, strike=0.8,
         body=((1200, 4, 0.5), (2600, 5, 0.4)), gains=(0.7, 0.6, 1.0))
register('reed.uilleann_pipes', blown, 'reed', lo='D4', hi='D6', tags=('oldfield', 'celtic'), sustain=True,
         desc='uilleann pipes chanter: sweet, reedy, intimate; crisp tight staccato and a subtle cran flick',
         odd=True, even=0.45, fr=1500, fr_vel=1700, fr_env=0.3, att=0.009, rel=0.03, scoop=0, grace=2.0,
         grace_t=0.016, vib_rate=5.2, vib_cents=6, vib_delay=0.5, drift=2, noise=0.05, noise_lo=1800,
         noise_hi=6500, sync=0.8, tonal=0.2, chiff=0.1, body=((1100, 3, 0.45), (2400, 4, 0.35)), lp_hz=7500)
register('reed.uilleann_drones', drones, 'reed', lo='D2', hi='D3', tags=('oldfield', 'celtic'), sustain=True,
         desc='uilleann drones: sweet humming bass, baritone and tenor drones in octaves (pitch = bass drone)',
         ratios=(1.0, 2.0, 4.0), detune=(0.0, 1.5, -1.2), fr=1500, odd=True, att=0.4, rel=0.3, strike=0.4,
         noise=0.025, gains=(1.0, 0.55, 0.3), pans=(0.0, -0.35, 0.4), lp_hz=3500)
register('reed.uilleann_regulators', regulators, 'reed', lo='G3', hi='G5', tags=('oldfield', 'celtic'),
         sustain=True, desc='uilleann regulators: wrist-struck root-fifth-octave reed chords, crisp and sweet')
register('reed.duduk', blown, 'reed', lo='E3', hi='G5', tags=('world',), sustain=True,
         desc='Armenian duduk: dark, velvety, breath-heavy wide reed; scoops from below, sobbing slow vibrato',
         fr=600, fr_vel=1100, fr_env=0.8, att=0.09, rel=0.1, scoop=-60, scoop_t=0.06, vib_rate=4.6,
         vib_cents=20, vib_delay=0.4, vib_grow=0.7, trem=0.08, noise=0.2, noise_lo=500, noise_hi=4000,
         sync=0.7, tonal=0.5, nq=5, shimmer=0.03, body=((520, 3, 0.8), (1250, 4, 0.5)), lp_hz=4500)
register('reed.harmonica', blown, 'reed', lo='C4', hi='C6', tags=('world', 'oldfield'), sustain=True,
         desc='harmonica: buzzy free reed, cupped-hand wah opening, bluesy draw-bend from below, hand tremolo',
         fr=2200, fr_vel=3000, fr_env=0.6, att=0.025, rel=0.05, scoop=-70, scoop_t=0.05, vib_rate=5.5,
         vib_cents=6, vib_delay=0.35, trem=0.22, noise=0.06, noise_lo=1500, noise_hi=6000, sync=0.9,
         shimmer=0.03, body=((1700, 5, 0.6), (3500, 6, 0.4)), wah=(600, 4200, 0.12), sat=1.0)

# ---- brass ----------------------------------------------------------------
register('brass.french_horn', blown, 'brass', lo='F2', hi='F5', tags=('orchestral', 'oldfield'), sustain=True, vpow=1.6, ppow=1.8, sat=1.2,
         desc='French horn: warm, round, hand-in-bell darkness; noble swell and bloom at forte',
         fr=450, fr_vel=2300, fr_env=0.85, fr_att=0.4, att=0.06, rel=0.1, overshoot=0.15, ov_t=0.08,
         scoop=-30, scoop_t=0.025, vib_cents=4, vib_delay=0.5, drift=2, noise=0.03, noise_lo=400,
         noise_hi=2500, sync=0.9, body=((400, 2, 0.4), (1000, 3, 0.3)), lp_hz=2600)
register('brass.trumpet', blown, 'brass', lo='F#3', hi='D6', tags=('orchestral', 'oldfield'), sustain=True, vpow=1.6, ppow=1.8, sat=1.2,
         desc='open trumpet: brilliant lip-buzz with a fast brassy blat at forte and a slight late vibrato',
         fr=800, fr_vel=6000, fr_env=0.9, fr_att=0.8, att=0.025, rel=0.06, overshoot=0.35, ov_t=0.05,
         scoop=-40, scoop_t=0.018, vib_rate=5.5, vib_cents=7, vib_delay=0.45, drift=2, noise=0.03,
         noise_lo=1000, noise_hi=6000, sync=0.9, body=((1300, 3, 0.7), (2600, 4, 0.5)), lp_hz=11000)
register('brass.trumpet_muted', blown, 'brass', lo='F#3', hi='D6', tags=('orchestral', 'world'), sustain=True,
         desc='harmon-muted trumpet: thin, buzzy, intimate nasal whine with a sultry late vibrato',
         fr=1800, fr_vel=3500, fr_env=0.8, att=0.03, rel=0.06, scoop=-45, scoop_t=0.03, vib_rate=5.0,
         vib_cents=14, vib_delay=0.35, drift=2, noise=0.05, noise_lo=2000, noise_hi=7000, sync=0.9,
         body=((1750, 9, 1.6), (3800, 10, 0.8)), hp_hz=380, lp_hz=7000, sat=1.0)
register('brass.trombone', blown, 'brass', lo='E2', hi='D5', tags=('orchestral',), sustain=True, vpow=1.6, ppow=1.8, sat=1.2,
         desc='tenor trombone: slide scoop into the note, broad buzzing brass that brightens hard with force',
         fr=550, fr_vel=4000, fr_env=0.9, fr_att=0.5, att=0.04, rel=0.08, overshoot=0.2, ov_t=0.06,
         scoop=-70, scoop_t=0.04, vib_rate=5.0, vib_cents=6, vib_delay=0.5, drift=2, noise=0.03,
         noise_lo=600, noise_hi=4000, sync=0.9, body=((600, 3, 0.5), (1300, 3, 0.4)), lp_hz=7000)
register('brass.tuba', blown, 'brass', lo='E1', hi='F4', tags=('orchestral',), sustain=True, vpow=1.6, ppow=1.8, sat=1.2,
         desc='tuba: huge, soft-edged, dark bass brass with a puffed lip attack',
         fr=260, fr_vel=1400, fr_env=0.8, fr_att=0.5, att=0.05, rel=0.09, overshoot=0.1, scoop=-30,
         scoop_t=0.035, vib_cents=3, vib_delay=0.6, drift=2, noise=0.04, noise_lo=200, noise_hi=1500,
         sync=0.9, body=((240, 2, 0.3), (700, 3, 0.25)), lp_hz=1800, sub=0.3)
register('brass.conch', blown, 'brass', lo='F3', hi='D5', tags=('world', 'jungle', 'latin'), sustain=True,
         desc='conch shell horn: hollow ritual blare, breathy and wavering, sagging off the end of the note',
         fr=450, fr_vel=1300, fr_env=0.9, att=0.09, rel=0.12, swell=0.25, scoop=-60, scoop_t=0.06,
         vib_rate=3.5, vib_cents=12, vib_delay=0.4, drift=8, jitter=3, fall=-140, fall_t=0.12, noise=0.22,
         noise_lo=500, noise_hi=4000, sync=0.6, tonal=0.6, nq=5, shimmer=0.05,
         body=((900, 3, 0.8), (1900, 4, 0.5)), lp_hz=4500)
register('brass.alphorn', blown, 'brass', lo='F2', hi='F4', tags=('world', 'orchestral'), sustain=True,
         desc='alphorn: long wooden natural horn, soft pure overtones blooming slowly, mountain-wide',
         fr=420, fr_vel=900, fr_env=0.9, att=0.12, rel=0.15, swell=0.15, scoop=-20, scoop_t=0.06,
         vib_cents=0, drift=3, noise=0.05, noise_lo=300, noise_hi=2000, sync=0.8,
         body=((500, 3, 0.4),), lp_hz=2400)
register('brass.didgeridoo', didgeridoo, 'brass', lo='B1', hi='A2', tags=('world', 'jungle'), sustain=True,
         desc='didgeridoo: buzzing drone with rhythmic "wow" formant sweeps, tongue-pushed overtone band')
