"""
Canopy Orchestra — family 'synth': the space half of the game, Oldfield's
80s/90s synth palette (string machine, analog brass, FM bells, vocoder) and a
2D-platformer chip nod. All oscillators are band-limited (PolyBLEP / DSF /
additive) except the deliberately stepped chip waveforms.
"""
from __future__ import annotations

import numpy as np

from .core import *
from .luthier import (dsf, roll_a, contour, gate, note_len, smooth_noise, polyblep_saw, polyblep_pulse,
                      tv_lowpass, tv_bandpass, ladder, var_delay, mix_stereo, shift, tame, vowel,
                      cascade, env_points, peaks)


def _sweep_env(n, dur, att, dec, sus, rel, sr):
    """filter-envelope (ADSR on a 0..1 scale)."""
    return gate(n, dur, att, rel, sr, dec=dec, sus=sus)


def chorus(x, rng, sr=SR, voices=3, depth=0.0025, base=0.012, rates=(0.6, 0.83, 1.1), fast=6.0, fast_depth=0.00015):
    """ensemble chorus (Solina-style): several modulated delays -> stereo."""
    n = len(x)
    t = tvec(n, sr)
    low = lp(x, 250, order=4, sr=sr)          # keep the bass out of the comb (no fundamental cancellation)
    x = hp(x, 250, order=4, sr=sr)
    L = np.zeros(n)
    R = np.zeros(n)
    for i in range(voices):
        r = rates[i % len(rates)]
        p = rng.uniform(0, TWOPI)
        d = base + depth * (0.5 + 0.5 * np.sin(TWOPI * r * t + p)) + fast_depth * np.sin(TWOPI * fast * t + 2 * p)
        y = var_delay(x, d, sr)
        if i % 2:
            L += y
        else:
            R += y
        if i == voices - 1 and voices % 2:
            L += 0.5 * y
    g = 1 + voices / 2
    return np.stack([0.5 * x + L, 0.5 * x + R]) / g + low * (1.5 / g)


# --------------------------------------------------------------------------- leads


def theremin(f, dur, vel, rng, sr, *, level=1.0):
    """theremin: near-sine with a hint of heterodyne warmth, slow-handed
    portamento into the note, wide singing vibrato, volume-hand swells."""
    rel = 0.12
    n = note_len(dur, rel, sr, extra=0.05)
    t = tvec(n, sr)
    fa = contour(n, f, dur, rng, sr, scoop=-160, scoop_t=0.055, vib_rate=6.3, vib_cents=32 + 12 * vel,
                 vib_delay=0.12, vib_grow=0.35, drift=6, fall=-120, fall_t=0.1)
    ph = phase_from_freq(fa, sr)
    s = np.sin(ph)
    y = s + (0.05 + 0.12 * vel) * s * s + 0.04 * vel * np.sin(3 * ph)   # gentle asymmetric bend
    e = gate(n, dur, 0.1, rel, sr) * (1 + 0.12 * smooth_noise(n, 2.0, rng, sr))
    return hp(y, 30, sr=sr) * e * level * (0.4 + 0.6 * vel)


def moog_lead(f, dur, vel, rng, sr, *, level=1.0):
    """Moog-style lead: two saws + square sub into a driven 4-pole ladder whose
    cutoff sweeps down from a velocity-dependent peak; late vibrato."""
    rel = 0.08
    n = note_len(dur, rel, sr, extra=0.03)
    fa = contour(n, f, dur, rng, sr, scoop=-30, scoop_t=0.02, vib_rate=5.5, vib_cents=12, vib_delay=0.4,
                 drift=2)
    y = polyblep_saw(fa, n, sr=sr) + 0.8 * polyblep_saw(fa * 2 ** (7 / 1200), n, sr=sr, phase0=0.3)
    y += 0.5 * polyblep_pulse(fa * 0.5, n, 0.5, sr=sr)
    fe = _sweep_env(n, dur, 0.005, 0.25, 0.25, rel, sr)
    fc = f * 1.2 + (1500 + 7000 * vel ** 1.5) * fe
    y = ladder(y * 0.5, fc, res=0.45 + 0.2 * vel, sr=sr, drive=1.5)
    return y * gate(n, dur, 0.004, rel, sr) * level * (0.4 + 0.6 * vel)


def analog_brass(f, dur, vel, rng, sr, *, level=1.0):
    """analog synth brass (Jupiter / OB style): detuned saws, filter that blares
    open over ~80 ms then settles, slight pitch blip, slow swell."""
    rel = 0.12
    n = note_len(dur, rel, sr, extra=0.03)
    t = tvec(n, sr)
    fa = contour(n, f, dur, rng, sr, scoop=-25, scoop_t=0.03, vib_rate=5.0, vib_cents=5, vib_delay=0.5,
                 drift=3)
    y = sum(polyblep_saw(fa * 2 ** (d / 1200), n, sr=sr, phase0=rng.uniform(0, 0.08)) for d in (-9, 0, 8))
    fe = np.clip(t / 0.08, 0, 1) ** 1.5 * (0.6 + 0.4 * np.exp(-np.maximum(t - 0.08, 0) / 0.3))
    k = int(dur * sr)
    if k < n:
        fe[k:] = fe[k - 1] * np.exp(-(t[k:] - t[k - 1]) / rel)
    fc = f * 1.0 + (900 + 5000 * vel) * fe
    y = tv_lowpass(y / 3, fc, q=0.9, sr=sr)
    y = chorus(y, rng, sr, voices=2, depth=0.002, base=0.006)
    return y * gate(n, dur, 0.04, rel, sr) * level * (0.4 + 0.6 * vel)


def chip_lead(f, dur, vel, rng, sr, *, level=1.0):
    """8-bit pulse lead: duty switching 12.5% -> 25% -> 50%, stepped 4-bit volume
    envelope, stepped vibrato — the platformer hero theme."""
    rel = 0.05
    n = note_len(dur, rel, sr, extra=0.02)
    t = tvec(n, sr)
    frame = np.floor(t * 60) / 60                          # 60 Hz 'frame' updates
    ramp = np.clip((frame - 0.2) / 0.2, 0, 1)
    vib = 25 * ramp * np.sign(np.sin(TWOPI * 6 * frame))
    fa = f * 2 ** (vib / 1200)
    duty = np.where(frame < 0.03, 0.125, np.where(frame < 0.08, 0.25, 0.5 - 0.25 * (1 - vel)))
    y = polyblep_pulse(fa, n, duty, sr=sr)
    e = gate(n, dur, 0.004, rel, sr, dec=0.3, sus=0.7)
    e = np.floor(e * 15 + 0.5) / 15                         # 4-bit volume
    y = hp(y, 40, sr=sr)
    return y * e * level * (0.4 + 0.6 * vel)


def chip_triangle(f, dur, vel, rng, sr, *, level=1.0):
    """8-bit triangle bass: the NES 32-step staircase triangle; velocity adds a
    little 'pluck' pitch drop and a brighter edge."""
    rel = 0.03
    n = note_len(dur, rel, sr, extra=0.02)
    t = tvec(n, sr)
    fa = f * 2 ** ((150 * vel * np.exp(-t / 0.012)) / 1200)
    p = (np.cumsum(fa) / sr) % 1.0
    tri = 1 - 4 * np.abs(p - 0.5)
    y = np.round(tri * 7.5 + 7.5) / 7.5 - 1                # 16 levels each way
    y = lp(y, 2500 + 9000 * vel, sr=sr)
    e = gate(n, dur, 0.002, rel, sr)
    return hp(y, 25, sr=sr) * e * level * (0.5 + 0.5 * vel)


# --------------------------------------------------------------------------- pads


def supersaw_pad(f, dur, vel, rng, sr, *, voices=7, spread=24.0, level=1.0):
    """supersaw pad: seven detuned saws spread across the stereo field, slow
    filter bloom, long release."""
    rel = 0.3
    n = note_len(dur, rel, sr, extra=0.05)
    t = tvec(n, sr)
    L = np.zeros(n)
    R = np.zeros(n)
    for i in range(voices):
        d = spread * (2 * i / (voices - 1) - 1)
        fa = f * 2 ** ((d + 2 * smooth_noise(n, 0.3, rng, sr)) / 1200)
        y = polyblep_saw(fa, n, sr=sr, phase0=rng.uniform(0, 0.15))
        if i == voices // 2:
            y = y * 1.8                                  # centre oscillator anchors the pitch
        pan = (2 * i / (voices - 1) - 1) * 0.85
        L += y * np.cos((pan + 1) * np.pi / 4)
        R += y * np.sin((pan + 1) * np.pi / 4)
    fe = np.clip(t / 0.6, 0, 1)
    fc = f * 2 + (1500 + 5000 * vel) * fe
    out = np.stack([tv_lowpass(L, fc, q=0.8, sr=sr, block=256), tv_lowpass(R, fc, q=0.8, sr=sr, block=256)])
    return out / voices * 2.5 * gate(n, dur, 0.35, rel, sr) * level * (0.4 + 0.6 * vel)


def warm_pad(f, dur, vel, rng, sr, *, level=1.0):
    """warm analog pad: two slowly pulse-width-modulated oscillators an octave
    apart, soft low-pass, gentle chorus."""
    rel = 0.35
    n = note_len(dur, rel, sr, extra=0.05)
    t = tvec(n, sr)
    pw1 = 0.5 + 0.38 * np.sin(TWOPI * 0.37 * t + rng.uniform(0, 6))
    pw2 = 0.5 + 0.35 * np.sin(TWOPI * 0.53 * t + rng.uniform(0, 6))
    fa = contour(n, f, dur, rng, sr, drift=3, drift_rate=0.5)
    y = polyblep_pulse(fa, n, pw1, sr=sr) + 0.6 * polyblep_pulse(fa * 2 * 2 ** (4 / 1200), n, pw2, sr=sr)
    y = lp(hp(y, 40, sr=sr), 900 + 2200 * vel, order=2, sr=sr)
    out = chorus(y, rng, sr, voices=2, depth=0.003, base=0.008)
    return out * gate(n, dur, 0.45, rel, sr) * level * (0.4 + 0.6 * vel)


def string_machine(f, dur, vel, rng, sr, *, level=1.0):
    """string machine (Solina / ARP-style): divide-down saws (8' + 4'), thin
    high-passed tone, the famous swirling triple-delay ensemble."""
    rel = 0.35
    n = note_len(dur, rel, sr, extra=0.05)
    y = polyblep_saw(f, n, sr=sr) + 0.45 * polyblep_saw(2 * f, n, sr=sr, phase0=0.25)
    y = hp(y, 220, sr=sr)
    y = lp(y, 3500 + 4000 * vel, sr=sr)
    y = y + peaks(y, ((1100, 1.5, 0.4),), sr)
    out = chorus(y, rng, sr, voices=3, depth=0.003, base=0.007, rates=(0.63, 0.81, 0.95), fast=6.2,
                 fast_depth=0.00018)
    return out * gate(n, dur, 0.25, rel, sr) * level * (0.4 + 0.6 * vel)



def nebula_pad(f, dur, vel, rng, sr, *, level=1.0):
    """wavetable 'nebula' sweep pad: the spectrum morphs (odd -> full, dark ->
    bright) while a resonant band scans slowly through the harmonics."""
    rel = 0.4
    n = note_len(dur, rel, sr, extra=0.05)
    t = tvec(n, sr)
    scan = 0.5 - 0.5 * np.cos(TWOPI * 0.18 * t + rng.uniform(0, 1))
    outs = []
    for side, dc in ((0, -6), (1, 6)):
        fa = f * 2 ** ((dc + 3 * smooth_noise(n, 0.4, rng, sr)) / 1200)
        ph = phase_from_freq(fa, sr, rng.uniform(0, 6))
        a = roll_a(f, (400 + 2500 * vel) * (0.4 + 0.6 * scan))
        y = (1 - scan) * dsf(fa, a, ph=ph, odd=True) + scan * dsf(fa, a, ph=ph)
        band = f * (2 + 10 * (0.5 - 0.5 * np.cos(TWOPI * 0.11 * t + side)))
        y = y + 1.5 * tv_bandpass(y, np.minimum(band, 9000), q=5.0, sr=sr, block=256)
        outs.append(y)
    out = tame(np.stack(outs) * gate(n, dur, 0.6, rel, sr), 1.3)
    return out * level * (0.4 + 0.6 * vel)


def vocoder_choir(f, dur, vel, rng, sr, *, bands=14, level=1.0):
    """vocoder choir pad: a bright detuned saw carrier through a bank of
    constant-Q bands whose levels follow a slow wordless vowel (ah -> oh)."""
    rel = 0.3
    n = note_len(dur, rel, sr, extra=0.05)
    t = tvec(n, sr)
    c = sum(polyblep_saw(f * 2 ** (d / 1200), n, sr=sr, phase0=rng.uniform()) for d in (-10, 0, 9))
    c += 0.15 * rng.standard_normal(n)
    centres = np.geomspace(180, 6000, bands)
    va = vowel('a', f, 'tenor')
    vo = vowel('o', f, 'tenor')
    def env_at(fc, vw):
        g = 0.02
        for i, (F, B) in enumerate(vw):
            g += (0.9 ** i) / (1 + ((fc - F) / (1.5 * B + 60)) ** 2)
        return g
    morph = np.clip(t / max(dur, 0.3), 0, 1)
    L = np.zeros(n)
    R = np.zeros(n)
    for i, fc in enumerate(centres):
        y = bp(c, fc / 1.12, fc * 1.12, order=2, sr=sr)
        g = (1 - morph) * env_at(fc, va) + morph * env_at(fc, vo)
        y = y * g
        if i % 2:
            L += y
        else:
            R += y
    bass = lp(c, 200, order=4, sr=sr) * 0.5            # unvoiced low band keeps the fundamental
    L += bass
    R += bass
    out = np.stack([L + 0.4 * R, R + 0.4 * L])
    out = tame(out * (1 + 0.5 * vel) * gate(n, dur, 0.3, rel, sr), 1.3)
    return out * level * (0.4 + 0.6 * vel)


def granular_shimmer(f, dur, vel, rng, sr, *, density=70.0, level=1.0):
    """granular shimmer cloud: a sustained swarm of short pitched grains on the
    note, its octave and twelfth, sparkling across the stereo field."""
    rel = 0.4
    n = note_len(dur, rel, sr, extra=0.15)
    L = np.zeros(n)
    R = np.zeros(n)
    k = int(density * (dur + 0.6))
    ratios = np.array([1, 1, 1, 2, 2, 3, 4])
    for _ in range(k):
        s = rng.uniform(0, dur + 0.4)
        g = max(rng.uniform(0.03, 0.09), 14.0 / f)
        m = int(g * sr)
        r = ratios[int(rng.integers(0, len(ratios)))] if rng.random() < 0.4 + 0.5 * vel else 1
        fr_ = f * r * 2 ** (rng.uniform(-6, 6) / 1200)
        if fr_ > sr * 0.4:
            fr_ = f
        tt = tvec(m, sr)
        # grain phase locked to absolute time so overlapping grains of one partial reinforce
        x = np.sin(TWOPI * fr_ * (tt + int(s * sr) / sr)) * np.hanning(m) / np.sqrt(r)
        pan = rng.uniform(-1, 1)
        a = (pan + 1) * np.pi / 4
        i0 = int(s * sr)
        if i0 + m <= n:
            L[i0:i0 + m] += x * np.cos(a)
            R[i0:i0 + m] += x * np.sin(a)
    e = gate(n, dur, 0.25, rel, sr)
    out = np.stack([L, R]) * e
    return out * level * (0.4 + 0.6 * vel) / np.sqrt(density / 20)


# --------------------------------------------------------------------------- percussive / FX


def fm_bell(f, dur, vel, rng, sr, *, level=1.0):
    """DX7-style electric bell: a 1:14 tine 'ding' operator over a 1:1 body,
    plus an inharmonic 1:3.5 bell pair; index decays so the tone mellows."""
    T = 4.5
    n = int(T * sr)
    t = tvec(n, sr)
    p = TWOPI * f * t
    i1 = (1.5 + 3 * vel) * np.exp(-t / 0.012)
    tine = np.sin(p + i1 * np.sin(14 * p)) * np.exp(-t / 0.5)
    i2 = (0.6 + 1.0 * vel) * np.exp(-t / 0.9)
    body = np.sin(p + i2 * np.sin(p)) * np.exp(-t / 1.5)
    i3 = (1.2 + 1.5 * vel) * np.exp(-t / 0.6)
    bell = np.sin(p + i3 * np.sin(3.5 * p + 0.3)) * np.exp(-t / 1.6)
    y = 0.5 * tine + 1.0 * body + 0.45 * bell
    y *= np.clip(t / 0.001, 0, 1)
    y = fade(y, 0, 1.2, sr)
    return y * level * (0.4 + 0.6 * vel)


def fm_glass(f, dur, vel, rng, sr, *, level=1.0):
    """FM glass marimba: a short, crystalline mallet tone; a 1:4 modulator gives
    the glassy attack, a quiet 1:1 pair the woody body; a fast clean decay."""
    T = 1.6
    n = int(T * sr)
    t = tvec(n, sr)
    p = TWOPI * f * t
    i1 = (2.0 + 4.0 * vel) * np.exp(-t / 0.02)
    glass = np.sin(p + i1 * np.sin(4 * p)) * np.exp(-t / 0.35)
    body = np.sin(p + (0.5 + 0.8 * vel) * np.exp(-t / 0.05) * np.sin(p)) * np.exp(-t / 0.55)
    ping = np.sin(2 * TWOPI * f * 3.98 * t) * np.exp(-t / 0.08) * 0.2 * vel if f * 4 < sr * 0.4 else 0
    y = 0.7 * glass + 0.6 * body + ping
    y *= np.clip(t / 0.0008, 0, 1)
    y = fade(y, 0, 0.4, sr)
    return y * level * (0.4 + 0.6 * vel)


def satellite_ping(f, dur, vel, rng, sr, *, ratio=1.4142, level=1.0):
    """ring-modulated 'satellite' ping: a tone at f ring/amplitude-modulated by
    an irrational partner (metallic sidebands), repeating in a ping-pong echo."""
    T = 2.2
    n = int(T * sr)
    t = tvec(n, sr)
    m = int(0.45 * sr)
    tt = tvec(m, sr)
    car = np.sin(TWOPI * f * tt)
    mod = np.sin(TWOPI * f * ratio * tt + 0.5)
    rm = car * (0.55 + (0.3 + 0.4 * vel) * mod)
    blip = rm * np.exp(-tt / 0.09) * np.clip(tt / 0.001, 0, 1)
    blip += 0.15 * np.sin(TWOPI * f * 2 * tt) * np.exp(-tt / 0.02) * vel
    L = np.zeros(n)
    R = np.zeros(n)
    for i in range(6):
        s = int(i * 0.19 * sr)
        g = 0.55 ** i
        x = blip if i == 0 else lp(blip, 6000 / (1 + i), sr=sr)
        tgt = L if i % 2 == 0 else R
        k = min(m, n - s)
        tgt[s:s + k] += g * x[:k]
        if i == 0:
            R[:k] += 0.6 * x[:k]
    return tame(np.stack([L, R]), 1.3) * level * (0.4 + 0.6 * vel)


def pulsar_blip(f, dur, vel, rng, sr, *, level=1.0):
    """pulsar / beacon blip: a pitch-dropping sine 'pew' into f with a short
    ring, repeated three times like a distant beacon."""
    T = 0.9
    n = int(T * sr)
    m = int(0.16 * sr)
    tt = tvec(m, sr)
    fa = f * 2 ** ((1200 * np.exp(-tt / 0.006)) / 1200)
    ph = phase_from_freq(fa, sr)
    b = (np.sin(ph) + 0.25 * vel * np.sin(3 * ph)) * np.exp(-tt / 0.045) * np.clip(tt / 0.0005, 0, 1)
    y = np.zeros(n)
    for i, g in enumerate((1.0, 0.45, 0.2)):
        s = int(i * 0.24 * sr)
        y[s:s + m] += g * b[:max(0, min(m, n - s))]
    return y * level * (0.4 + 0.6 * vel)


def sub_bass(f, dur, vel, rng, sr, *, level=1.0):
    """sub bass: deep sine with a velocity-driven harmonic growl and a tiny
    pitch-drop thump at the start."""
    rel = 0.06
    n = note_len(dur, rel, sr, extra=0.02)
    t = tvec(n, sr)
    fa = f * 2 ** ((300 * np.exp(-t / 0.008)) / 1200)
    ph = phase_from_freq(fa, sr)
    y = np.sin(ph)
    y = np.tanh((1 + 2.5 * vel) * y) / np.tanh(1 + 2.5 * vel)
    y = lp(y, 220 + 800 * vel, sr=sr)
    return y * gate(n, dur, 0.004, rel, sr) * level * (0.4 + 0.6 * vel)


def acid_bass(f, dur, vel, rng, sr, *, square=False, level=1.0):
    """acid 303-style bass: saw into a squelchy resonant low-pass with a fast
    decaying envelope; accent (velocity) raises resonance and sweep."""
    rel = 0.03
    n = note_len(dur, rel, sr, extra=0.02)
    t = tvec(n, sr)
    y = polyblep_pulse(f, n, 0.5, sr=sr) if square else polyblep_saw(f, n, sr=sr)
    acc = vel ** 1.5
    fe = np.exp(-t / (0.12 + 0.12 * (1 - acc)))
    fc = f * 1.5 + (500 + 4500 * acc) * fe
    y = tv_lowpass(y, fc, q=3.0 + 6.0 * acc, sr=sr, block=64)
    y = tv_lowpass(y, fc * 1.5, q=0.7, sr=sr, block=64)
    y = np.tanh(1.5 * y)
    return hp(y, 30, sr=sr) * gate(n, dur, 0.003, rel, sr) * level * (0.4 + 0.6 * vel)


def noise_riser(f, dur, vel, rng, sr, *, octaves=3.0, level=1.0):
    """noise riser for transitions: resonant noise band and a ghost tone sweeping
    up `octaves` from f over the gate, swelling, then cut."""
    rel = 0.06
    n = note_len(dur, rel, sr, extra=0.02)
    t = tvec(n, sr)
    x = np.clip(t / max(dur, 0.05), 0, 1)
    fc = f * 2 ** (octaves * x ** 1.6)
    w = rng.standard_normal(n)
    y = tv_bandpass(w, np.minimum(fc * 4, 16000), q=1.5 + 3 * vel, sr=sr, block=128)
    y += 0.3 * hp(w, 3000, sr=sr) * x
    ph = phase_from_freq(np.minimum(fc, 12000), sr)
    y += 0.35 * np.sin(ph) * (0.3 + 0.7 * x)
    e = (0.05 + 0.95 * x ** 2) * gate(n, dur, 0.01, rel, sr)
    out = tame(np.stack([y, var_delay(y, 0.0007 + 0.0006 * x, sr)]) * e, 1.5)
    return out * level * (0.4 + 0.6 * vel)


# --------------------------------------------------------------------------- registry

register('synth.theremin', theremin, 'synth', lo='C4', hi='C7', tags=('space', 'oldfield'), sustain=True,
         desc='theremin: eerie near-sine with slow-hand portamento and a wide singing vibrato')
register('synth.moog_lead', moog_lead, 'synth', lo='C3', hi='C6', tags=('space', 'oldfield'), sustain=True,
         desc='Moog-style lead: fat detuned saws + sub through a driven ladder filter that sweeps down')
register('synth.analog_brass', analog_brass, 'synth', lo='C2', hi='C6', tags=('space', 'oldfield'), sustain=True,
         desc='analog synth brass: detuned saw stack whose filter blares open, 80s fanfare')
register('synth.supersaw_pad', supersaw_pad, 'synth', lo='C2', hi='C6', tags=('space',), sustain=True,
         desc='supersaw pad: seven detuned saws spread wide, slow filter bloom, long release')
register('synth.warm_pad', warm_pad, 'synth', lo='C2', hi='C6', tags=('space', 'oldfield'), sustain=True,
         desc='warm analog pad: slow PWM oscillators an octave apart, soft and chorused')
register('synth.string_machine', string_machine, 'synth', lo='C2', hi='C6', tags=('space', 'oldfield'),
         sustain=True, desc='string machine: Solina-style thin saws in a swirling triple-delay ensemble')
register('synth.nebula_pad', nebula_pad, 'synth', lo='C2', hi='C6', tags=('space',), sustain=True,
         desc='wavetable nebula pad: spectrum morphing odd->full while a resonant band scans the harmonics')
register('synth.vocoder_choir', vocoder_choir, 'synth', lo='C2', hi='C5', tags=('space', 'oldfield'),
         sustain=True, desc='vocoder choir pad: detuned saw carrier through vowel-shaped bands, "ah" to "oh"')
register('synth.granular_shimmer', granular_shimmer, 'synth', lo='C3', hi='C6', tags=('space',), sustain=True,
         desc='granular shimmer: a sparkling cloud of pitched grains on the note, octave and twelfth')
register('synth.fm_bell', fm_bell, 'synth', lo='C3', hi='C7', tags=('space', 'oldfield'),
         desc='DX7-style electric bell: glassy tine ding over a mellowing FM body')
register('synth.fm_glass_marimba', fm_glass, 'synth', lo='C3', hi='C7', tags=('space', 'oldfield'),
         desc='FM glass marimba: short crystalline mallet tone with a glassy 1:4 attack')
register('synth.satellite_ping', satellite_ping, 'synth', lo='C4', hi='C7', tags=('space',),
         desc='satellite ping: ring-modulated metallic blip bouncing in a ping-pong echo')
register('synth.pulsar_blip', pulsar_blip, 'synth', lo='C4', hi='C7', tags=('space', 'chip'),
         desc='pulsar beacon: a pitch-dropping "pew" blip repeated three times into the distance')
register('synth.sub_bass', sub_bass, 'synth', lo='C1', hi='C3', tags=('space',), sustain=True,
         desc='sub bass: deep sine with velocity growl and a tiny thump')
register('synth.acid_bass', acid_bass, 'synth', lo='C1', hi='C4', tags=('space',), sustain=True,
         desc='acid 303 bass: squelchy resonant saw; accent opens the sweep and the resonance')
register('synth.chip_lead', chip_lead, 'synth', lo='C3', hi='C7', tags=('chip', 'space'), sustain=True,
         desc='8-bit pulse lead: duty switching, 4-bit volume steps, stepped vibrato')
register('synth.chip_triangle_bass', chip_triangle, 'synth', lo='C1', hi='C4', tags=('chip',), sustain=True,
         desc='8-bit triangle bass: NES staircase triangle with a little pluck')
register('synth.noise_riser', noise_riser, 'synth', lo='C2', hi='C5', tags=('space', 'inharmonic'), sustain=True,
         desc='noise riser: resonant noise band and ghost tone sweeping up three octaves over the gate')
