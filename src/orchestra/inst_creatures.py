"""
Canopy Orchestra — family 'creature': jungle (and space) creatures *as instruments*.

Every call is tuned: the `f` argument is where the call settles (the target
pitch), so in-key creature melodies are possible. Calls with an intentionally
moving pitch approach the target quickly and hold it; truly noisy sounds
(cicada band, beetle clicks, parrot squawk) are tagged 'inharmonic' or
pitched=False, with `f` still tuning their resonance.

Building blocks: tone() (whistle / syrinx tone along a pitch track with a few
harmonics), syllable placement (python loops over *events*, never samples),
and the formant / DSF tools in luthier.py.
"""
from __future__ import annotations

import numpy as np

from .core import *
from .luthier import (dsf, roll_a, contour, gate, note_len, smooth_noise, cascade, tv_cascade,
                      tv_bandpass, peaks, mix_stereo, shift, env_points, burst, breath, tame)


def tone(fa, harm=(1.0, 0.08, 0.03), sr=SR, ph0=0.0):
    """syrinx / whistle tone along frequency track fa with a few harmonics."""
    ph = phase_from_freq(fa, sr, ph0)
    y = np.zeros(len(fa))
    for k, a in enumerate(harm, start=1):
        if a and k * np.max(fa) < sr * 0.45:
            y += a * np.sin(k * ph)
    return y


def place(n, events):
    """sum [(start_sample, signal), ...] into a buffer of length n."""
    y = np.zeros(n)
    for s, x in events:
        s = int(s)
        if s >= n:
            continue
        k = min(len(x), n - s)
        y[s:s + k] += x[:k]
    return y


def hann_env(m, att_frac=0.2, rel_frac=0.4):
    e = np.ones(m)
    a = max(1, int(m * att_frac))
    r = max(1, int(m * rel_frac))
    e[:a] = np.sin(0.5 * np.pi * np.arange(a) / a) ** 2
    e[-r:] *= np.cos(0.5 * np.pi * np.arange(r) / r) ** 2
    return e


def _glide_cents(n, sr, start_cents, tau):
    return start_cents * np.exp(-tvec(n, sr) / tau)


# --------------------------------------------------------------------------- frogs


def treefrog_choir(f, dur, vel, rng, sr, *, frogs=5, rate=(3.0, 6.5), croak=0.05, pulse=(90, 160),
                   detune=12.0, level=1.0):
    """several tree frogs on (nearly) the same note, each with its own rhythm:
    every croak is a short rattle of pulses on a pitched carrier that bends up
    into the note."""
    n = note_len(dur, 0.02, sr, extra=0.15)
    parts = []
    for i in range(frogs):
        fi = f * 2 ** (rng.uniform(-1, 1) * detune / 1200) if i else f
        iv = 1.0 / rng.uniform(*rate)
        t0 = rng.uniform(0, iv * 0.8) if i else 0.0
        pr = rng.uniform(*pulse)
        cl = max(croak * rng.uniform(0.8, 1.3), 40.0 / fi)      # at least ~40 cycles
        m = int(cl * sr)
        tt = tvec(m, sr)
        ev = []
        t = t0
        while t < max(dur, cl):
            fa = fi * 2 ** ((-100 * np.exp(-tt / 0.005) + rng.uniform(-5, 5)) / 1200)
            am = 0.5 + 0.5 * np.cos(TWOPI * pr * tt) ** 2
            # carrier phase locked to absolute time: one frog's croaks stay coherent
            x = tone(fa, (1.0, 0.25 + 0.3 * vel, 0.08 * vel), sr, TWOPI * fi * t) * am * hann_env(m, 0.15, 0.5)
            ev.append((t * sr, x * rng.uniform(0.6, 1.0)))
            t += iv * rng.uniform(0.85, 1.15)
        parts.append((place(n, ev) * (1.0 if i == 0 else 0.55), rng.uniform(-0.8, 0.8) if i else 0.0))
    y = mix_stereo(parts)
    return y * level * (0.4 + 0.6 * vel)


def bullfrog(f, dur, vel, rng, sr, *, rum=16.0, level=1.0):
    """bullfrog 'jug-o-rum': rough low pulse source, 'rrr' pulsing, a vocal sac
    resonance that blooms, a sagging pitch."""
    rel = 0.05
    n = note_len(dur, rel, sr, extra=0.03)
    t = tvec(n, sr)
    fa = contour(n, f, dur, rng, sr, scoop=-80, scoop_t=0.03, jitter=18, drift=3,
                 bend=-25 * np.clip(t / max(dur, 0.05), 0, 1) ** 2)
    ph = phase_from_freq(fa, sr)
    src = dsf(fa, roll_a(f, 900 + 1500 * vel), ph=ph)
    src += 0.15 * hp(rng.standard_normal(n), 200, sr=sr)
    rr = rum * (1 + 0.1 * smooth_noise(n, 2.0, rng, sr))
    am = 0.35 + 0.65 * (0.5 + 0.5 * np.cos(TWOPI * np.cumsum(rr) / sr)) ** 1.5
    e = gate(n, dur, 0.04, rel, sr) * am
    y = cascade(src * e, [(f * 1.2 + 180, 60), (900, 120), (2200, 200)], sr)
    y = y / 6 + 0.6 * lp(src * e, f * 1.5, sr=sr)
    return tame(y, 1.2) * level * (0.4 + 0.6 * vel)


def coqui(f, dur, vel, rng, sr, *, low=-7, level=1.0):
    """Puerto-Rican coquí: 'co' a fifth below (default) then a rising 'QUI' that
    settles on f (the pitch argument is the top note)."""
    co_d, gap, qui_d = 0.09, 0.03, 0.2 + 0.1 * min(dur, 1.0)
    n = int((co_d + gap + qui_d + 0.05) * sr)
    m1 = int(co_d * sr)
    f1 = f * 2 ** (low / 12)
    fa1 = f1 * 2 ** (_glide_cents(m1, sr, -40, 0.02) / 1200)
    co = tone(fa1, (1.0, 0.15, 0.05), sr) * hann_env(m1, 0.1, 0.4)
    m2 = int(qui_d * sr)
    fa2 = f * 2 ** ((-260 * np.exp(-tvec(m2, sr) / 0.025) + 15 * np.exp(-tvec(m2, sr) / 0.2) - 10) / 1200)
    qui = tone(fa2, (1.0, 0.2 * vel, 0.05), sr) * hann_env(m2, 0.08, 0.35)
    y = place(n, [(0, co * 0.7), ((co_d + gap) * sr, qui)])
    return y * level * (0.4 + 0.6 * vel)


# --------------------------------------------------------------------------- insects


def cicada(f, dur, vel, rng, sr, *, pulse=210.0, phrase=0.0, level=1.0):
    """cicada drone: tymbal click train exciting a body resonance centred on f;
    swells in, holds, then dies away with the band sagging (pitch-able buzz band)."""
    rel = 0.18
    n = note_len(dur, rel, sr, extra=0.05)
    t = tvec(n, sr)
    pr = pulse * (1 + 0.04 * smooth_noise(n, 3.0, rng, sr))
    p = np.cumsum(pr) / sr
    click = np.maximum(np.cos(TWOPI * p), 0) ** 12        # sharp tymbal buckling clicks
    x = click * (1 + 0.4 * rng.standard_normal(n)) + 0.15 * rng.standard_normal(n)
    sag = 2 ** ((-150 * np.clip((t - dur) / 0.5, 0, 1)) / 1200)
    y = tv_bandpass(x, f * sag, q=7 + 6 * vel, sr=sr) + 0.4 * tv_bandpass(x, 2 * f * sag, q=9, sr=sr)
    e = gate(n, dur, min(0.35, 0.3 * dur + 0.03), rel, sr, swell=0.3)
    if phrase:
        e *= 0.55 + 0.45 * np.cos(TWOPI * phrase * t) ** 2
    return tame(y * e, 1.5) * level * (0.4 + 0.6 * vel)


def cricket(f, dur, vel, rng, sr, *, pulses=4, prate=30.0, chirp_every=0.33, level=1.0):
    """field cricket: chirps of 3-5 pure carrier pulses at f, repeating while held."""
    n = note_len(dur, 0.01, sr, extra=0.2)
    pd = 0.6 / prate
    m = int(pd * sr)
    ev = []
    t = 0.0
    k = 0
    while t < max(dur, 0.01):
        for j in range(pulses + (1 if rng.random() < 0.3 else 0)):
            fa = f * 2 ** ((rng.uniform(-8, 8) - 30 * tvec(m, sr) / pd) / 1200) * 1.004
            x = tone(fa, (1.0, 0.06 * vel, 0.02), sr, TWOPI * f * (t + j / prate)) * hann_env(m, 0.15, 0.6)
            ev.append(((t + j / prate) * sr, x * (0.75 + 0.25 * rng.random()) * (1 - 0.1 * j)))
        t += chirp_every * rng.uniform(0.92, 1.08)
        k += 1
    return place(n, ev) * level * (0.4 + 0.6 * vel)


def beetle(f, dur, vel, rng, sr, *, clicks=(5, 11), rate=(28, 45), level=1.0):
    """stridulating beetle: rhythmic bursts of hard little resonant ticks."""
    n = note_len(dur, 0.01, sr, extra=0.3)
    m = int(0.012 * sr)
    tt = tvec(m, sr)
    ev = []
    t = 0.0
    fc = f * 8
    while t < max(dur, 0.05):
        k = int(rng.integers(*clicks))
        r = rng.uniform(*rate)
        for j in range(k):
            fr_ = fc * rng.uniform(0.9, 1.12)
            x = np.sin(TWOPI * fr_ * tt) * np.exp(-tt / 0.003) + 0.5 * np.sin(TWOPI * fr_ * 2.7 * tt) * np.exp(-tt / 0.001)
            x += 0.4 * rng.standard_normal(m) * np.exp(-tt / 0.0006)
            acc = 1.0 if j == 0 else 0.6 + 0.3 * rng.random()
            ev.append(((t + j / r) * sr, x * acc))
        t += k / r + rng.uniform(0.08, 0.2)
    y = place(n, ev)
    return tame(hp(y, 600, sr=sr), 2.5, 99.0) * level * (0.4 + 0.6 * vel)


def hummingbird(f, dur, vel, rng, sr, *, wing=52.0, level=1.0):
    """hummingbird wing hum: a buzzing harmonic hum at f flickered by the wing
    beat, wing-air noise, and a hovering wobble as it darts about."""
    rel = 0.08
    n = note_len(dur, rel, sr, extra=0.05)
    t = tvec(n, sr)
    dart = 12 * smooth_noise(n, 1.8, rng, sr)
    fa = contour(n, f, dur, rng, sr, drift=0, jitter=4, bend=dart)
    ph = phase_from_freq(fa, sr)
    y = dsf(fa, roll_a(f, 1200 + 1500 * vel), ph=ph)
    wb = wing * (1 + 0.05 * smooth_noise(n, 2.0, rng, sr))
    flap = 0.55 + 0.45 * np.cos(TWOPI * np.cumsum(wb) / sr) ** 2
    air = bp(rng.standard_normal(n), 300, 3000, sr=sr) * flap * 0.35
    prox = 1 + 0.35 * smooth_noise(n, 1.0, rng, sr)       # darting nearer / farther
    e = gate(n, dur, 0.06, rel, sr) * np.clip(prox, 0.55, 1.35)
    out = tame((y * flap + air) * e, 1.8)
    return out * level * (0.4 + 0.6 * vel)


# --------------------------------------------------------------------------- birds


def songbird(f, dur, vel, rng, sr, *, step=2, speed=15.0, level=1.0):
    """songbird trill: an intro flick, then a sparkling trill between f and the
    note `step` semitones above (in-key), each note a tiny chirped whistle, ending
    with a drop onto f."""
    n = note_len(dur, 0.02, sr, extra=0.2)
    ev = []
    # intro: quick upward slide
    m = int(0.045 * sr)
    fa = f * 2 ** ((np.linspace(-500, 0, m) + 0) / 1200)
    ev.append((0, tone(fa, (1, 0.05), sr) * hann_env(m, 0.2, 0.3) * 0.8))
    t = 0.06
    i = 0
    up = f * 2 ** (step / 12)
    while t < max(dur, 0.25):
        hi = i % 2 == 1
        d = (0.4 if hi else 0.6) * 2 / speed
        m = int(d * sr)
        fc = up if hi else f
        sh = (40 if hi else -25) * np.linspace(-1, 1, m)
        fa = fc * 2 ** ((sh + rng.uniform(-5, 5)) / 1200)
        x = tone(fa, (1.0, 0.07 + 0.08 * vel, 0.02), sr, rng.uniform(0, 6)) * hann_env(m, 0.2, 0.35)
        ev.append((t * sr, x * (0.7 + 0.3 * rng.random()) * (0.85 if hi else 1.0)))
        t += d
        i += 1
    m = int(0.12 * sr)
    fa = f * 2 ** ((150 * np.exp(-tvec(m, sr) / 0.015)) / 1200)
    ev.append((t * sr, tone(fa, (1, 0.06), sr) * hann_env(m, 0.1, 0.6)))
    n = int(max(s + len(x) for s, x in ev)) + int(0.06 * sr)
    return place(n, ev) * level * (0.4 + 0.6 * vel)


def bird_whistle(f, dur, vel, rng, sr, *, glide=-500, level=1.0):
    """tuned bird whistle: a clean glide up into the note, a held whistle with a
    shy vibrato, then a downward flick as it lets go."""
    rel = 0.04
    n = note_len(dur, rel, sr, extra=0.08)
    t = tvec(n, sr)
    bend = glide * np.exp(-t / 0.03) - 260 * np.clip((t - dur) / 0.07, 0, 1)
    fa = contour(n, f, dur, rng, sr, vib_rate=7.5, vib_cents=14, vib_delay=0.25, vib_grow=0.3,
                 drift=3, bend=bend)
    y = tone(fa, (1.0, 0.05 + 0.08 * vel, 0.015), sr)
    y += 0.02 * bp(rng.standard_normal(n), f, min(3 * f, 18000), sr=sr)
    return y * gate(n, dur, 0.012, rel, sr) * level * (0.4 + 0.6 * vel)


def gibbon(f, dur, vel, rng, sr, *, rise=-900, level=1.0):
    """gibbon great-call whoop: a pure hooting voice that swoops up a huge
    interval into the note, holds with a wild excited vibrato, then drops off."""
    rel = 0.06
    n = note_len(dur, rel, sr, extra=0.15)
    t = tvec(n, sr)
    bend = rise * np.exp(-t / 0.035) - 700 * np.clip((t - dur) / 0.12, 0, 1) ** 1.5
    fa = contour(n, f, dur, rng, sr, vib_rate=6.5, vib_cents=20, vib_delay=0.18, vib_grow=0.3,
                 drift=5, jitter=4, bend=bend)
    ph = phase_from_freq(fa, sr)
    a = roll_a(f, 600 + 900 * vel)
    y = dsf(fa, a, ph=ph)
    y = cascade(y, [(f * 1.3, 120), (f * 3.1, 300)], sr) / 3
    y += 0.03 * breath(n, rng, f, 500, 5000, sr, ph=ph, sync=0.6)
    e = gate(n, dur + 0.08, 0.03, rel, sr)
    return y * e * level * (0.4 + 0.6 * vel)


def kookaburra(f, dur, vel, rng, sr, *, rate=8.5, level=1.0):
    """kookaburra laugh: a cackling run of harsh 'ha' syllables that builds,
    peaks and tails off, each arching up to f."""
    k = int(np.clip(dur * rate, 4, 28))
    n = int((k / rate + 0.35) * sr)
    ev = []
    t = 0.0
    for i in range(k):
        x_ = i / max(k - 1, 1)
        inten = np.sin(np.pi * min(1.0, 0.15 + x_)) ** 0.7
        d = 1 / rate * rng.uniform(0.6, 0.85)
        m = int(d * sr)
        tt = np.linspace(0, 1, m)
        arch = -200 * (1 - np.sin(np.pi * np.clip(tt * 1.25, 0, 1))) ** 2 - (180 if (i % 4 == 3) else 0)
        fa = f * 2 ** ((arch + rng.uniform(-15, 15)) / 1200)
        ph = phase_from_freq(fa, sr, rng.uniform(0, 6))
        x = dsf(fa, roll_a(f, 2500 + 2500 * vel * inten), ph=ph)
        x += 0.35 * bp(rng.standard_normal(m), 1500, 6000, sr=sr) * (0.5 + 0.5 * np.cos(ph))
        x = x * hann_env(m, 0.12, 0.5) * (0.4 + 0.6 * inten)
        ev.append((t * sr, x))
        t += 1 / rate * rng.uniform(0.9, 1.1)
    y = place(n, ev)
    y = y + peaks(y, ((f * 2.1, 3, 0.6), (3200, 3, 0.4)), sr)
    return y * level * (0.4 + 0.6 * vel)


def parrot(f, dur, vel, rng, sr, *, level=1.0):
    """parrot squawk: chaotic syrinx (jitter, subharmonics) through a beak
    formant pair, harsh and loud, arching in pitch."""
    d = float(np.clip(0.18 + 0.25 * dur, 0.2, 0.6))
    rel = 0.03
    n = note_len(d, rel, sr, extra=0.03)
    t = tvec(n, sr)
    bend = -200 * np.exp(-t / 0.03) - 250 * np.clip((t - d * 0.6) / (d * 0.4), 0, 1) ** 2
    fa = contour(n, f, d, rng, sr, jitter=45, drift=20, bend=bend)
    ph = phase_from_freq(fa, sr)
    y = dsf(fa, roll_a(f, 3500 + 3000 * vel), ph=ph)
    y += 0.5 * np.sin(0.5 * ph) * (0.5 + 0.5 * np.sign(np.sin(TWOPI * 13 * t)))   # period-doubling bursts
    y += 0.6 * hp(rng.standard_normal(n), 1200, sr=sr)
    F1 = 1400 + 500 * np.sin(np.pi * np.clip(t / d, 0, 1))
    y = tv_cascade(y, np.stack([F1, F1 * 2.1, np.full(n, 4300.0)]),
                   np.stack([np.full(n, 250.0), np.full(n, 350.0), np.full(n, 500.0)]), sr)
    y = tame(y / 20, 1.5)
    return y * gate(n, d, 0.012, rel, sr) * level * (0.4 + 0.6 * vel)


def owl(f, dur, vel, rng, sr, *, level=1.0):
    """owl hoot 'hoo... hoooo': soft breathy low hoots, the long one holds."""
    rel = 0.09
    d1 = 0.16
    gap = 0.12
    n1 = note_len(d1, 0.04, sr)
    t1 = tvec(n1, sr)
    fa1 = f * 2 ** ((-20 * (1 - np.sin(np.pi * np.clip(t1 / d1, 0, 1)))) / 1200)
    h1 = tone(fa1, (1.0, 0.1, 0.03), sr) * gate(n1, d1, 0.04, 0.04, sr) * 0.6
    d2 = max(dur, 0.25)
    n2 = note_len(d2, rel, sr, extra=0.05)
    t2 = tvec(n2, sr)
    bend = -80 * np.exp(-t2 / 0.04) - 140 * np.clip((t2 - d2) / 0.25, 0, 1)
    fa2 = contour(n2, f, d2, rng, sr, vib_rate=4.0, vib_cents=6, vib_delay=0.3, drift=4, bend=bend)
    ph = phase_from_freq(fa2, sr)
    h2 = (np.sin(ph) + (0.08 + 0.1 * vel) * np.sin(2 * ph) + 0.02 * np.sin(3 * ph))
    h2 += 0.08 * breath(n2, rng, f, 300, 2500, sr, ph=ph, sync=0.7, tonal=0.5, q=6)
    h2 *= gate(n2, d2, 0.07, rel, sr)
    s2 = int((d1 + gap) * sr)
    y = place(s2 + n2, [(0, h1), (s2, h2)])
    return y * level * (0.4 + 0.6 * vel)


# --------------------------------------------------------------------------- mammals


def howler(f, dur, vel, rng, sr, *, level=1.0):
    """howler-monkey roar: a growling rough source (period doubling, jitter,
    40 Hz grind) through the hyoid resonator; swells open from 'oo' to 'aa'."""
    rel = 0.15
    n = note_len(dur, rel, sr, extra=0.05)
    t = tvec(n, sr)
    sw = np.clip(t / max(0.4 * dur, 0.15), 0, 1)
    fa = contour(n, f, dur, rng, sr, scoop=-120, scoop_t=0.05, jitter=18, drift=10,
                 bend=30 * np.sin(np.pi * np.clip(t / max(dur, 0.1), 0, 1)) - 200 * np.clip((t - dur) / 0.3, 0, 1))
    ph = phase_from_freq(fa, sr)
    src = dsf(fa, roll_a(f, 1500 + 2500 * vel * (0.5 + 0.5 * sw)), ph=ph)
    src += (0.3 + 0.3 * vel) * np.sin(0.5 * ph) * (0.5 + 0.5 * np.sin(TWOPI * 7 * t))   # subharmonic
    grind = 0.6 + 0.4 * np.abs(np.sin(TWOPI * rng.uniform(35, 45) * t + smooth_noise(n, 8, rng, sr)))
    src = src * grind + 0.25 * hp(rng.standard_normal(n), 300, sr=sr)
    F1 = 380 + 380 * sw
    F2 = 800 + 500 * sw
    y = tv_cascade(src, np.stack([F1, F2, np.full(n, 2400.0)]),
                   np.stack([np.full(n, 110.0), np.full(n, 150.0), np.full(n, 300.0)]), sr)
    y = y * (np.std(src) / (np.std(y) + 1e-9)) + 0.5 * lp(src, f * 1.4, sr=sr)
    e = gate(n, dur, max(0.12, 0.35 * dur), rel, sr)
    return tame(y * e, 1.3) * level * (0.4 + 0.6 * vel)


def monkey_chatter(f, dur, vel, rng, sr, *, rate=12.0, level=1.0):
    """monkey chatter: excited squeaky 'chk-chk-eek' syllables, mostly on f
    with in-key leaps (fourth below, whole tone / major third above)."""
    k = int(np.clip(dur * rate, 3, 30))
    n = int((k / rate + 0.25) * sr)
    ev = []
    t = 0.0
    leaps = [0, 0, 0, 0, 2, -5, 4, 0]
    for i in range(k):
        st = leaps[int(rng.integers(0, len(leaps)))] if i else 0
        d = 1 / rate * rng.uniform(0.45, 0.75) * (1.8 if (st == 0 and rng.random() < 0.25) else 1.0)
        m = int(d * sr)
        tt = np.linspace(0, 1, m)
        shape = -140 * (1 - np.sin(np.pi * np.clip(tt * 1.2, 0, 1))) ** 1.5
        fa = f * 2 ** ((st * 100 + shape + rng.uniform(-10, 10)) / 1200)
        ph = phase_from_freq(fa, sr, rng.uniform(0, 6))
        x = dsf(fa, roll_a(f, 1500 + 2500 * vel), ph=ph)
        x += 0.25 * hp(rng.standard_normal(m), 2000, sr=sr) * np.exp(-tt * 8)
        x = cascade(x, [(f * 1.25, 200), (f * 2.6, 400)], sr) / 3
        ev.append((t * sr, x * hann_env(m, 0.08, 0.5) * rng.uniform(0.6, 1.0)))
        t += 1 / rate * rng.uniform(0.8, 1.25)
    return tame(place(n, ev), 1.4) * level * (0.4 + 0.6 * vel)


def bat(f, dur, vel, rng, sr, *, level=1.0):
    """bat sonar (transposed down to hearing): constant-frequency calls at f
    ending in a steep FM sweep down; the pulse rate speeds up into a feeding buzz."""
    n = note_len(dur, 0.01, sr, extra=0.12)
    ev = []
    t = 0.0
    ipi = 0.11
    while t < max(dur, 0.05):
        cf = rng.uniform(0.018, 0.026)
        fm = 0.006
        m = int((cf + fm) * sr)
        tt = tvec(m, sr)
        c = np.where(tt < cf, 0.0, -1200 * ((tt - cf) / fm))
        fa = f * 2 ** ((c + 4) / 1200)
        x = tone(fa, (1.0, 0.18 * vel, 0.05), sr) * hann_env(m, 0.15, 0.35)
        ev.append((t * sr, x * rng.uniform(0.8, 1.0)))
        t += ipi
        ipi = max(0.03, ipi * 0.9)
    return place(n, ev) * level * (0.4 + 0.6 * vel)


def whale(f, dur, vel, rng, sr, *, level=1.0):
    """whale song moan: rises from below, holds with a slow lazy waver, the
    'throat' formant slowly opening, and a long sinking glide off the end;
    distant underwater reflections."""
    rel = 0.25
    n = note_len(dur, rel, sr, extra=0.6)
    t = tvec(n, sr)
    bend = -260 * np.exp(-t / 0.06) - 500 * np.clip((t - dur) / 0.6, 0, 1)
    fa = contour(n, f, dur, rng, sr, vib_rate=1.6, vib_cents=14, vib_delay=0.2, vib_grow=0.6,
                 drift=6, drift_rate=0.8, bend=bend)
    ph = phase_from_freq(fa, sr)
    src = dsf(fa, roll_a(f, 900 + 1500 * vel), ph=ph)
    src += 0.04 * lp(rng.standard_normal(n), 1500, sr=sr)
    op = np.clip(t / max(dur, 0.2), 0, 1)
    F1 = f * 1.4 + 350 * op
    y = tv_cascade(src, np.stack([F1, F1 * 2.4]), np.stack([np.full(n, 120.0), np.full(n, 220.0)]), sr)
    y = y * (np.std(src) / (np.std(y) + 1e-9)) * 0.7 + 0.5 * lp(src, f * 1.3, sr=sr)
    y = lp(y, 3500, sr=sr) * gate(n, dur, 0.15, rel, sr)
    out = y.copy()
    for dl, g in ((0.11, 0.35), (0.23, 0.22), (0.37, 0.12)):
        out += g * shift(lp(y, 1500, sr=sr), int(dl * sr), n)
    return out * level * (0.4 + 0.6 * vel)


def clack(f, dur, vel, rng, sr, *, double=True, level=1.0):
    """toucan / hornbill bill clack: hollow keratin bill snapping shut — a
    knocking double impact with a tuned wooden resonance (f) and its buzz."""
    n = int(0.35 * sr)
    t = tvec(n, sr)
    modes = [(1.0, 1.0, 0.07), (2.32, 0.45, 0.025), (3.95, 0.3, 0.012), (5.6, 0.2, 0.008)]
    def hit(g):
        y = sum(a * np.sin(TWOPI * f * r * t + rng.uniform(0, 6)) * np.exp(-t / tau) for r, a, tau in modes)
        y += (0.5 + 0.6 * vel) * bp(rng.standard_normal(n), 1500, 7000, sr=sr) * np.exp(-t / 0.0015)
        return y * np.clip(t / 0.0005, 0, 1) * g
    y = hit(1.0)
    if double:
        d = int(rng.uniform(0.028, 0.04) * sr)
        y += shift(hit(0.55), d, n)
    return tame(y, 1.6) * level * (0.4 + 0.6 * vel)


# --------------------------------------------------------------------------- registry

J = 'jungle'
register('creature.treefrog_choir', treefrog_choir, 'creature', lo='C5', hi='C7', tags=(J, 'latin'),
         sustain=True, desc='tree-frog choir on one note: five frogs, rattling pitched croaks in loose overlapping rhythms')
register('creature.bullfrog', bullfrog, 'creature', lo='C2', hi='C4', tags=(J,), sustain=True,
         desc='bullfrog "jug-o-rum": rough low pulsing croak with a booming vocal-sac resonance')
register('creature.coqui', coqui, 'creature', lo='A5', hi='F7', tags=(J, 'latin'),
         desc='coquí two-tone "co-QUÍ": a fifth below, then a rising chirp onto the top note (pitch)')
register('creature.cicada_drone', cicada, 'creature', lo='C6', hi='A8', tags=(J, 'inharmonic'), sustain=True,
         desc='cicada drone: tymbal click buzz in a resonant band centred on the pitch; swells and sags away')
register('creature.cicada_pulse', cicada, 'creature', lo='C6', hi='A8', tags=(J, 'inharmonic'), sustain=True,
         desc='pulsing cicada: the same tymbal buzz phrased in 4 Hz throbs', phrase=2.0, pulse=160.0)
register('creature.cricket', cricket, 'creature', lo='C6', hi='C8', tags=(J,), sustain=True,
         desc='field cricket: pure pitched chirps of four pulses, repeating while the note is held')
register('creature.beetle_clicks', beetle, 'creature', pitched=False, tags=(J, 'inharmonic'), sustain=True,
         desc='stridulating beetle: rhythmic bursts of hard little resonant ticks (pitch tunes the body)')
register('creature.hummingbird', hummingbird, 'creature', lo='C3', hi='C5', tags=(J, 'latin'), sustain=True,
         desc='hummingbird hover: buzzing wing hum on the note, flickering wingbeat air, darting wobble')
register('creature.songbird_trill', songbird, 'creature', lo='C6', hi='C8', tags=(J,), sustain=True,
         desc='songbird trill: flick up, then a sparkling in-key trill on the note and the tone above')
register('creature.bird_whistle', bird_whistle, 'creature', lo='C5', hi='C8', tags=(J, 'oldfield'), sustain=True,
         desc='tuned bird whistle: clean glide up into the note, shy vibrato, a flick down as it lets go')
register('creature.gibbon_whoop', gibbon, 'creature', lo='A4', hi='A6', tags=(J, 'asian'), sustain=True,
         desc='gibbon whoop: pure hooting voice swooping up a huge interval into the note, wild vibrato')
register('creature.kookaburra_laugh', kookaburra, 'creature', lo='G4', hi='C6', tags=(J, 'world'), sustain=True,
         desc='kookaburra laugh: a cackling run of harsh "ha" syllables that builds and tails off')
register('creature.parrot_squawk', parrot, 'creature', lo='C4', hi='C6', tags=(J, 'latin', 'inharmonic'),
         desc='parrot squawk: chaotic, harsh beaky screech arching in pitch (pitch is a tint)')
register('creature.owl_hoot', owl, 'creature', lo='C3', hi='C5', tags=(J, 'oldfield'), sustain=True,
         desc='owl "hoo... hoooo": soft breathy hoots, the long one held on the note and sinking away')
register('creature.howler_roar', howler, 'creature', lo='C2', hi='C4', tags=(J, 'latin'), sustain=True,
         desc='howler-monkey roar: growling swell from "oo" to "aa" through a huge hyoid resonance')
register('creature.monkey_chatter', monkey_chatter, 'creature', lo='C5', hi='C7', tags=(J,), sustain=True,
         desc='monkey chatter: squeaky excited "chk-eek" syllables on the note with in-key leaps')
register('creature.bat_chirps', bat, 'creature', lo='C6', hi='C8', tags=(J, 'space'), sustain=True,
         desc='bat sonar: constant-pitch calls with FM drop-offs, accelerating into a feeding buzz')
register('creature.whale_call', whale, 'creature', lo='C2', hi='C5', tags=('space', 'world'), sustain=True,
         desc='whale moan: rising into the note, lazy waver, slowly opening throat, sinking glide, echoes')
register('creature.toucan_clack', clack, 'creature', lo='C4', hi='C6', tags=(J, 'latin'),
         desc='toucan / hornbill clack: hollow double bill-snap with a tuned wooden knock')
