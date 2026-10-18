"""Whimsical-platformer extras: kazoo, human whistling, creature gibberish,
jaw harp, slide whistle and collectible chimes. Original synthesis, no samples."""
from __future__ import annotations

import numpy as np

from .core import register, tvec, bp, hp, lp, softclip, phase_from_freq, TWOPI, env_perc
from .luthier import contour, gate, note_len, cascade, tv_cascade, vowel, vowel_tracks, saw_from_dsf, tame, \
    burst, dsf, roll_a, smooth_noise

J = 'jungle'


def _crest(y, sr, k=4.0):
    """soft-limit peaks to k x the loudest 250 ms RMS (the level calibration uses)."""
    w = int(0.25 * sr)
    c = np.cumsum(np.concatenate([[0.0], y * y]))
    r = np.sqrt(((c[w:] - c[:-w]) / w).max()) if len(y) > w else np.sqrt(np.mean(y * y))
    lim = k * r + 1e-12
    return lim * np.tanh(y / lim) / (r + 1e-12) * 0.25     # also level-match across the range


def _end_fade(y, sr, sec=0.4):
    k = min(len(y), int(sec * sr))
    y = y.copy()
    y[-k:] *= np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    return y


# ---------------------------------------------------------------- kazoo
def m_kazoo(f, dur, vel, rng, sr, drive=3.0, nasal=1.0):
    """hummed voice -> buzzing paper membrane -> small tube.
    The membrane is a hard nonlinearity: the hum's waveform is clipped and
    re-excites a bright rattle band each period, which is what makes a kazoo."""
    n = note_len(dur, 0.06, sr, extra=0.05)
    fr = contour(n, f, dur, rng, sr, scoop=-60, scoop_t=0.03, vib_rate=5.6, vib_cents=18 * vel,
                 vib_delay=0.18, vib_grow=0.25, drift=4, jitter=3)
    hum = saw_from_dsf(fr, n, fr=2500 + 2500 * vel, sr=sr)
    d = drive * (0.6 + 0.8 * vel)
    buzz = hp(softclip(hum * d, 2.5), 50, sr=sr)
    rattle = hp(np.sign(buzz) * np.abs(buzz) ** 0.3, 1800, sr=sr)
    rattle = bp(rattle * (0.6 + 0.4 * rng.standard_normal(n) * 0.3), 2000, 7500, sr=sr)
    tube = cascade(buzz, [(650, 160), (1300 * nasal, 220), (2700, 380), (3800, 500)], sr=sr)
    y = hp(0.8 * tube + (0.35 + 0.35 * vel) * rattle, 60, sr=sr)
    y *= gate(n, dur, att=0.018, rel=0.06, sr=sr)
    return _end_fade(_crest(y, sr, 5.0), sr, 0.3)


register('reed.kazoo', m_kazoo, 'reed', lo='C4', hi='C6', tags=('world', 'playful'), sustain=True,
         desc='kazoo: hummed tune buzzing a paper membrane — nasal, rattly, cheerfully silly')
register('reed.kazoo_bass', m_kazoo, 'reed', lo='C3', hi='C5', tags=('world', 'playful'), sustain=True,
         drive=4.0, nasal=0.8, desc='low kazoo: a fruitier, growlier buzz an octave down')


# ---------------------------------------------------------------- human whistle
def m_whistle(f, dur, vel, rng, sr):
    n = note_len(dur, 0.05, sr, extra=0.04)
    fr = contour(n, f, dur, rng, sr, scoop=-120, scoop_t=0.035, vib_rate=5.8, vib_cents=22,
                 vib_delay=0.22, vib_grow=0.3, drift=5, jitter=2, fall=-60 * (1 - vel), fall_t=0.05)
    ph = phase_from_freq(fr, sr)
    y = np.sin(ph) + 0.04 * np.sin(2 * ph)
    air = bp(rng.standard_normal(n), f * 0.8, f * 2.2, sr=sr) * (0.05 + 0.04 * (1 - vel))
    e = gate(n, dur, att=0.03, rel=0.05, sr=sr)
    return (y + air) * e


register('voice.whistle', m_whistle, 'voice', lo='C5', hi='C7', tags=('world', 'playful'), sustain=True,
         desc='human whistling: pure lip tone that scoops into each note, airy, a cheeky vibrato')


# ---------------------------------------------------------------- creature gibberish
SYL = ['ba', 'li', 'mo', 'pa', 'de', 'wu', 'ki', 'na', 'yo', 'bu', 'la', 'mi']


def m_babble(f, dur, vel, rng, sr, scale=1.35, rate=9.0, excited=0.0):
    """tiny creatures chattering: wordless syllables (consonant burst + vowel)
    on a bouncy pitch contour around f. Never forms words."""
    n = note_len(dur, 0.05, sr, extra=0.05)
    nsyl = max(1, int(round(dur * rate)))
    step = dur / nsyl
    seq, bend = [], np.zeros(n)
    t = tvec(n, sr)
    amp = np.zeros(n)
    for k in range(nsyl):
        s = SYL[rng.integers(len(SYL))]
        t0 = k * step
        seq += [(t0, s[1] if s[1] in 'aeiou' else 'a'), (t0 + step * 0.9, s[1] if s[1] in 'aeiou' else 'a')]
        a, b = int(t0 * sr), int(min(n, (t0 + step * 0.92) * sr))
        w = np.hanning(max(2, b - a)) ** 0.5
        amp[a:b] = np.maximum(amp[a:b], w * (0.8 + 0.3 * rng.random()))
        up = rng.uniform(-200, 350) + excited * 300 * (k == nsyl - 1)
        bend[a:b] += up * np.sin(np.linspace(0, np.pi, b - a))
    fr = contour(n, f, dur, rng, sr, jitter=12, drift=10, bend=bend)
    src = saw_from_dsf(fr, n, fr=3000, sr=sr)
    F, B = vowel_tracks(seq, n, f=f, vtype='soprano', scale=scale)
    y = tv_cascade(src, F, B, sr=sr) * amp
    for k in range(nsyl):                                   # consonant clicks
        a = int(k * step * sr)
        bl = burst(int(0.012 * sr), rng, 1500, 6000, sr=sr)
        y[a:a + len(bl)] += 0.3 * bl[:max(0, min(len(bl), n - a))]
    return tame(y * gate(n, dur, att=0.01, rel=0.04, sr=sr))


register('voice.teensy_babble', m_babble, 'voice', lo='G4', hi='G6', tags=('world', 'playful', 'inharmonic'), sustain=True,
         desc='tiny-creature gibberish: bouncy wordless chatter, "ba-li-mo-ki", helium-small mouths')
register('voice.teensy_cheer', m_babble, 'voice', lo='G4', hi='G6', tags=('world', 'playful', 'inharmonic'), sustain=True,
         rate=6.0, excited=1.0, desc='tiny-creature cheer: the same voices, slower and swooping up at the end — "yaaay!"')


# ---------------------------------------------------------------- jaw harp
def m_jaw(f, dur, vel, rng, sr, wah=1.0):
    n = note_len(dur, 0.25, sr, extra=0.1)
    t = tvec(n, sr)
    # the reed: a bright, slightly asymmetric pulse train
    ph = phase_from_freq(np.full(n, f), sr)
    y = dsf(np.full(n, f), 0.93, n=n, sr=sr, ph=ph)
    # the mouth: a single resonance sweeping between harmonics 3..12
    sweep = f * (3 + 9 * (0.5 + 0.5 * np.sin(TWOPI * (1.1 + 0.6 * vel) * wah * t + rng.uniform(0, 6))))
    F = np.stack([np.clip(sweep, 200, 6000), np.full(n, 2600.0), np.full(n, 3600.0), np.full(n, 4500.0),
                  np.full(n, 5200.0)])
    Bw = np.stack([np.full(n, 90.0), np.full(n, 300.0), np.full(n, 400.0), np.full(n, 500.0), np.full(n, 600.0)])
    y = tv_cascade(y * 0.3, F, Bw, sr=sr)
    twang = env_perc(n, 0.001, 0.15 + 0.3 * vel, sr)
    y *= np.maximum(twang, gate(n, dur, att=0.002, rel=0.12, sr=sr) * 0.45)
    return _end_fade(_crest(y, sr, 3.5), sr)


register('pluck.jaw_harp', m_jaw, 'pluck', lo='C2', hi='C4', tags=('world', 'playful'), sustain=True,
         desc='jaw harp: boing-ing metal tongue, the mouth sweeping a "wah-wah" through its overtones')


# ---------------------------------------------------------------- slide whistle (jumps, falls)
def m_slide(f, dur, vel, rng, sr, direction=1.0):
    n = note_len(dur, 0.04, sr, extra=0.03)
    t = tvec(n, sr)
    span = 12.0 * (0.6 + 0.6 * vel)                     # semitones travelled
    x = np.clip(t / max(dur, 0.05), 0, 1)
    shape = 1 - (1 - x) ** 2                             # fast start, settles on the pitch
    semis = -direction * span * (1 - shape)
    fr = f * 2 ** (semis / 12) * (1 + 0.006 * np.sin(TWOPI * 6 * t))
    ph = phase_from_freq(fr, sr)
    y = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.05 * np.sin(3 * ph)
    y += bp(rng.standard_normal(n), 1500, 5000, sr=sr) * 0.05
    return y * gate(n, dur, att=0.01, rel=0.04, sr=sr)


register('synth.slide_up', m_slide, 'synth', lo='C5', hi='C7', tags=('playful', 'inharmonic'), sustain=True,
         desc='slide whistle swooping up into the note — a jump')
register('synth.slide_down', m_slide, 'synth', lo='C4', hi='C6', tags=('playful', 'inharmonic'), sustain=True,
         direction=-1.0, desc='slide whistle drooping down onto the note — a fall, a bonk')


# ---------------------------------------------------------------- collectible chimes
def m_glim(f, dur, vel, rng, sr, king=False):
    """a collectible: glass bell + sparkle + a tiny 'ya' when it is the king version."""
    n = int((2.2 if king else 1.6) * sr)
    t = tvec(n, sr)
    parts = [(1.0, 1.0, 0.7), (2.0, 0.35, 0.35), (3.01, 0.22, 0.2), (5.43, 0.08, 0.08), (8.9, 0.04, 0.04)]
    y = np.zeros(n)
    for r, a, d in parts:
        if f * r < sr * 0.45:
            y += a * np.sin(TWOPI * f * r * t + rng.uniform(0, 6)) * np.exp(-t / d)
    sp = bp(rng.standard_normal(n), 6000, 14000, sr=sr) * np.exp(-t / 0.03) * 0.15
    y += sp
    if king:
        for r, a in ((2.0, 0.5), (1.5, 0.35), (1.0 * 2 ** (4 / 12), 0.3)):
            y += a * np.sin(TWOPI * f * r * t * (1 + 0.003 * rng.standard_normal())) * np.exp(-t / 0.6)
        fy = f
        while fy > 1100:
            fy /= 2
        ya = m_babble(fy, 0.12, 0.8, rng, sr, scale=1.5, rate=8)[:n] * 0.5
        y[:len(ya)] += ya
    y *= np.clip(t / 0.0015, 0, 1)
    return _end_fade(_crest(y, sr, 4.0), sr)


register('bell.glim', m_glim, 'bell', lo='C5', hi='C8', tags=('playful',),
         desc='glim: the collectible — a glass ting with a sparkle on top')
register('bell.glim_king', m_glim, 'bell', lo='C5', hi='C8', tags=('playful',), king=True,
         desc='king glim: the doubled, golden version — octave and fifth shimmer plus a tiny "ya!"')
