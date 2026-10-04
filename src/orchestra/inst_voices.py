"""
Canopy Orchestra — family 'voice'. Wordless voices only (no intelligible words).

Source-filter model:
    glottis   band-limited pulse train (Moorer DSF) with a velocity-dependent
              spectral tilt (soft = breathy/dark, loud = pressed/bright),
              jitter and shimmer, pitch-synchronous aspiration noise
    pitch     scoop into the note, vibrato that develops (each singer different),
              drift, optional fall-off
    tract     Klatt-style cascade formants from vowel tables (bass / tenor / alto /
              soprano chosen by pitch, F1 tuned above f0 like real singers),
              optional vowel morphs and mouth closure (hum / 'om')
    ensemble  several singers with their own detune, vibrato and onset, split
              into two formant-varied groups panned left / right

Specials: overtone throat singing, whispered choir, robot formant voice.
"""
from __future__ import annotations

import numpy as np

from .core import *
from .luthier import (dsf, roll_a, contour, gate, note_len, smooth_noise, cascade, tv_cascade,
                      tv_lowpass, vowel, vowel_tracks, voice_type, peaks, mix_stereo, shift,
                      comb_tint, tame, polyblep_pulse, burst)


def _glottis(n, f, dur, vel, rng, sr, *, scoop, scoop_t, vib_rate, vib_cents, vib_delay, vib_grow,
             jitter, drift, fall, fall_t, sag, fr, tilt, breath, shimmer, asp=None):
    """one singer's glottal source (+ pitch-synchronous aspiration from `asp`)."""
    t = tvec(n, sr)
    bend = sag * np.clip(t / max(dur, 1e-3), 0, 1.0) if sag else None
    fa = contour(n, f, dur, rng, sr, scoop=scoop, scoop_t=scoop_t, vib_rate=vib_rate,
                 vib_cents=vib_cents, vib_delay=vib_delay, vib_grow=vib_grow, drift=drift,
                 jitter=jitter, fall=fall, fall_t=fall_t, bend=bend)
    ph = phase_from_freq(fa, sr, rng.uniform(0, TWOPI))
    g = dsf(fa, roll_a(f, fr), ph=ph)
    # glottal tilt (source -12 dB/oct + lip radiation +6): its corner moves a long way
    # with effort, so soft voices are dark and loud ones pressed and bright
    g = onepole_lp(g, tilt, sr) * np.sqrt(1 + (f / tilt) ** 2)
    if shimmer:
        kk = n // 64 + 2
        g *= 1 + shimmer * np.interp(np.arange(n), np.arange(kk) * 64.0, rng.standard_normal(kk))
    if breath and asp is not None:
        c = np.cos(ph)
        g = g + breath * asp[:n] * (0.35 + 0.65 * (0.5 + 0.5 * c) * (0.5 + 0.5 * c))
    return g


def sing(f, dur, vel, rng, sr, *, voices=1, spread=7.0, onset=0.03, vow='a', vseq=None, vtype=None,
         fscale=1.0, bw_mul=1.0, att=0.12, rel=0.16, swell=0.0, scoop=-30.0, scoop_t=0.05,
         vib_rate=5.5, vib_cents=25.0, vib_delay=0.3, vib_grow=0.6, vib_spread=0.15, jitter=5.0,
         drift=4.0, fall=0.0, fall_t=0.1, sag=0.0, fr=2500.0, fr_vel=3000.0, tilt=250.0,
         tilt_vel=900.0, breath=0.05, shimmer=0.04, ring=0.0, ring_f=3000.0, octave=0.0,
         close=None, groups=2, width=0.6, chest=0.0, level=1.0):
    chest = chest if chest else (0.6 if f < 200 else 0.0)
    n = note_len(dur, rel, sr, extra=onset + 0.06)
    vt = vtype or voice_type(f)
    srcs = [np.zeros(n) for _ in range(groups)]
    asps = [hp(rng.standard_normal(n), 400, sr=sr) for _ in range(groups)] if breath else [None] * groups
    for v in range(voices):
        dc = 0.0 if voices == 1 else spread * rng.uniform(-1, 1)
        ff = f * 2 ** (dc / 1200)
        oct_ = (octave > 0 and voices > 2 and v % 3 == 2)
        if oct_:
            ff *= 2
        d = int(rng.uniform(0, onset) * sr) if voices > 1 else 0
        m = n - d
        g = _glottis(m, ff, dur, vel, rng, sr, scoop=scoop * rng.uniform(0.7, 1.3), scoop_t=scoop_t,
                     vib_rate=vib_rate * (1 + vib_spread * rng.uniform(-1, 1)),
                     vib_cents=vib_cents * rng.uniform(0.7, 1.2), vib_delay=vib_delay * rng.uniform(0.8, 1.3),
                     vib_grow=vib_grow, jitter=jitter, drift=drift, fall=fall, fall_t=fall_t, sag=sag,
                     fr=fr + fr_vel * vel, tilt=tilt + 1.4 * tilt_vel * vel ** 2, breath=breath * (1.3 - 0.6 * vel),
                     shimmer=shimmer, asp=asps[v % groups])
        e = gate(m, dur, att * rng.uniform(0.8, 1.25), rel, sr, swell=swell)
        amp = (octave if oct_ else 1.0) * rng.uniform(0.75, 1.0)
        srcs[v % groups] += shift(g * e * amp, d, n)
    outs = []
    for gi, src in enumerate(srcs):
        sc = fscale * (1 + (0.035 * (gi - (groups - 1) / 2) if groups > 1 else 0.0))
        if vseq is None:
            y = cascade(src, vowel(vow, f, vt, sc, bw_mul), sr)
        else:
            seq = [(tt * dur if tt <= 1.5 else tt, vv) for tt, vv in vseq]
            F, B = vowel_tracks(seq, n, f, vt, sc, bw_mul, sr)
            y = tv_cascade(src, F, B, sr, block=384)
        # loudness independent of where the formants fall relative to f0
        k = max(1, int(min(n, dur * sr)))
        y *= np.std(src[:k]) / (np.std(y[:k]) + 1e-12)
        # parallel high branch (Klatt): air and edge above the cascade's last pole
        y = y + 0.12 * hp(src, 4200, order=2, sr=sr)
        if chest:
            y = y + chest * lp(src, f * 1.4, sr=sr)
        if ring:
            y = y + peaks(y, ((ring_f * sc, 5.0, ring),), sr)
        if close is not None and close < 0:
            y = lp(y, 650, sr=sr) * 1.8 + 0.08 * bp(y, 2000, 2600, sr=sr)   # closed lips, nasal ring
        elif close is not None:
            # mouth closes into a hum: low-pass slides down from open to 'mmm'
            t = tvec(n, sr)
            tc = close * dur
            k = np.clip((t - tc) / 0.12, 0, 1)
            y = tv_lowpass(y, 7000 * (1 - k) + 650 * k, sr=sr, block=512) * (1 + 0.8 * k)
        outs.append(y)
    if groups == 1:
        return outs[0] * level * (0.4 + 0.6 * vel)
    pans = np.linspace(-width, width, groups)
    return mix_stereo([(o, p) for o, p in zip(outs, pans)]) * level * (0.4 + 0.6 * vel)


def throat(f, dur, vel, rng, sr, *, picks=(8, 9, 10, 12, 10, 9, 8, 6), step=0.32, q_bw=55.0,
           whistle=4.0, level=1.0):
    """Overtone (khoomei / sygyt-style) singing: a pressed low drone; a very narrow
    resonance formed by the tongue steps from harmonic to harmonic, gliding,
    so the drone's own partials sing a whistled melody above it."""
    rel = 0.2
    n = note_len(dur, rel, sr, extra=0.05)
    t = tvec(n, sr)
    fa = contour(n, f, dur, rng, sr, scoop=-25, scoop_t=0.06, vib_cents=4, vib_rate=4.5, vib_delay=0.6,
                 drift=2, jitter=3)
    ph = phase_from_freq(fa, sr, rng.uniform(0, TWOPI))
    e = gate(n, dur, 0.12, rel, sr)
    g = dsf(fa, roll_a(f, 2500 + 2500 * vel), ph=ph)
    g = onepole_lp(g, 500, sr) * np.sqrt(1 + (f / 500) ** 2)
    g += 0.03 * hp(rng.standard_normal(n), 500, sr=sr)
    drone = cascade(g, [(f * 1.1 + 250, 70), (800, 90), (2300, 140), (3000, 160), (3600, 200)], sr)
    drone /= np.std(drone[:int(min(n, dur * sr))]) + 1e-9
    # whistle centre: piecewise holds on harmonics, smooth glides
    k_pick = np.array(picks, float)
    hold = max(0.12, min(step, dur / max(len(picks), 1)))
    idx = np.minimum((t / hold).astype(int), len(k_pick) - 1)
    kc = onepole_lp(k_pick[idx], 14.0, sr)
    kc[:int(0.05 * sr)] = k_pick[0]
    fc = kc * f
    w = np.zeros(n)
    for k in range(4, 20):
        if k * f > 5500:
            break
        d = (k * fa - fc) / q_bw
        amp = 1.0 / (1 + d * d)
        w += amp * np.sin(k * ph)
    y = drone * 0.35 + whistle * (0.5 + 0.5 * vel) * w * 0.3
    return y * e * level * (0.4 + 0.6 * vel)


def whisper(f, dur, vel, rng, sr, *, groups=3, tint=0.35, level=1.0):
    """Whispered breath choir: noise through moving vowel formants (a -> o -> u),
    three groups of whisperers with their own swells, lightly tinted with the
    pitch by a resonant comb."""
    rel = 0.18
    n = note_len(dur, rel, sr, extra=0.1)
    parts = []
    for i in range(groups):
        d = int(rng.uniform(0, 0.08) * sr)
        m = n - d
        x = rng.standard_normal(m)
        x = x + 0.7 * onepole_lp(x, 900, sr) * 3
        vs = [('a', 'o', 'u'), ('o', 'a', 'e'), ('e', 'a', 'o')][i % 3]
        F, B = vowel_tracks([(0, vs[0]), (dur * 0.5, vs[1]), (dur + 0.1, vs[2])], m, None, 'alto',
                            1 + 0.07 * (i - (groups - 1) / 2), 1.8, sr)
        y = tv_cascade(x, F, B, sr, block=512)
        y /= np.std(y) + 1e-12
        c = comb_tint(y, f * (1 + 0.004 * rng.uniform(-1, 1)), fb=0.92, sr=sr)
        y = (1 - tint) * y + tint * c / (np.std(c) + 1e-12)
        y = hp(y, 350, sr=sr)
        e = gate(m, dur, 0.22 * rng.uniform(0.7, 1.3), rel, sr)
        e *= 1 + 0.35 * smooth_noise(m, 3.0, rng, sr)
        parts.append((shift(y * e, d, n), (i - (groups - 1) / 2) * 0.6))
    out = mix_stereo(parts)
    return out * level * (0.4 + 0.6 * vel) * (1 + 0.5 * vel)


def robot(f, dur, vel, rng, sr, *, step=0.11, crush=7000.0, ring_hz=0.0, level=1.0):
    """Robot formant voice: rigid buzzing pulse, no vibrato, vowels that jump in
    sample-and-hold steps, a low-rate speech-chip resample (aliasing sparkle)."""
    rel = 0.06
    n = note_len(dur, rel, sr, extra=0.03)
    t = tvec(n, sr)
    fa = np.full(n, f) * (1 + 0.004 * np.sign(np.sin(TWOPI * 3.0 * t)))   # stepped 'pitch quantiser'
    y = hp(polyblep_pulse(fa, n, width=0.18 + 0.1 * vel, sr=sr), 40, sr=sr)
    y += 0.1 * rng.standard_normal(n)
    vows = ['a', 'e', 'o', 'i', 'u', 'a', 'o']
    order = rng.permutation(len(vows))
    seq = []
    for i in range(int(dur / step) + 3):
        v = vows[order[i % len(vows)]]
        seq += [(i * step, v), (i * step + step * 0.92, v)]
    F, B = vowel_tracks(seq, n, f, 'tenor', 1.1, 1.0, sr)
    y = tv_cascade(y, F, B, sr) / 15.0
    if ring_hz:
        y = y * (0.6 + 0.4 * np.sin(TWOPI * ring_hz * t))
    # speech-chip resample: zero-order hold at `crush` Hz
    hold = max(1, int(sr / crush))
    y = np.repeat(y[::hold], hold)[:n]
    y = lp(y, 9000, sr=sr)
    e = gate(n, dur, 0.015, rel, sr)
    return y * e * level * (0.4 + 0.6 * vel)


# --------------------------------------------------------------------------- registry

register('voice.choir_aah', sing, 'voice', lo='F2', hi='C6', tags=('oldfield', 'orchestral'), sustain=True,
         desc='mixed choir on "aah": eight singers, each with own vibrato and pitch, warm and wide',
         voices=7, spread=9, onset=0.06, vow='a', att=0.22, rel=0.15, scoop=-25, vib_rate=5.3, vib_cents=22,
         vib_delay=0.35, jitter=6, drift=5, fr=2200, fr_vel=3500, tilt=200, tilt_vel=2800, breath=0.06)
register('voice.choir_ooh', sing, 'voice', lo='F2', hi='C6', tags=('oldfield', 'orchestral'), sustain=True,
         desc='choir on "ooh": soft, dark, hollow; slow swell, gentle vibrato',
         voices=7, spread=8, onset=0.08, vow='u', att=0.3, rel=0.16, swell=0.15, scoop=-20,
         vib_rate=5.0, vib_cents=16, vib_delay=0.4, jitter=5, drift=5, fr=1400, fr_vel=1600, tilt=150,
         tilt_vel=900, breath=0.05)
register('voice.children_choir', sing, 'voice', lo='C4', hi='A5', tags=('oldfield', 'orchestral'), sustain=True,
         desc="children's choir: pure straight-toned 'ah-oh', light and airy, small bright formants",
         voices=6, spread=10, onset=0.05, vseq=((0, 'a'), (1.0, 'o')), vtype='soprano', fscale=1.17,
         bw_mul=1.2, att=0.16, rel=0.16, scoop=-15, vib_rate=5.8, vib_cents=6, vib_delay=0.5, jitter=7,
         drift=6, fr=2000, fr_vel=2200, tilt=350, tilt_vel=1200, breath=0.09)
register('voice.soprano_vocalise', sing, 'voice', lo='C4', hi='C6', tags=('oldfield', 'orchestral'),
         sustain=True, desc='solo soprano vocalise: ringing "ah" turning to "oh", operatic vibrato that blooms',
         voices=1, vseq=((0, 'a'), (0.6, 'a'), (1.2, 'o')), vtype='soprano', groups=1, att=0.1, rel=0.14,
         scoop=-45, scoop_t=0.05, vib_rate=5.7, vib_cents=55, vib_delay=0.22, vib_grow=0.5, jitter=4,
         drift=3, fr=2600, fr_vel=4000, tilt=300, tilt_vel=3500, breath=0.035, ring=1.2, ring_f=3100)
register('voice.male_chant_om', sing, 'voice', lo='C2', hi='C4', tags=('asian', 'world', 'oldfield'),
         sustain=True, desc='low male "om" chant: three monks, "oh" closing to "oo" and a buzzing hum',
         voices=3, spread=5, onset=0.04, vseq=((0, 'o'), (0.45, 'o'), (0.75, 'u')), vtype='bass',
         close=0.85, att=0.18, rel=0.2, scoop=-30, vib_cents=5, vib_delay=0.6, jitter=6, drift=3,
         fr=3000, fr_vel=2000, tilt=150, tilt_vel=900, breath=0.04, groups=2, width=0.4)
register('voice.hum_choir', sing, 'voice', lo='C3', hi='A5', tags=('oldfield', 'world'), sustain=True,
         desc='humming choir "mmm": closed-mouth, warm, intimate, soft buzzing nasal glow',
         voices=6, spread=7, onset=0.06, vow='u', close=-1.0, att=0.2, rel=0.18, scoop=-15, vib_cents=10,
         vib_delay=0.4, jitter=5, drift=4, fr=1800, fr_vel=1200, tilt=250, tilt_vel=400, breath=0.02)
register('voice.chant_call', sing, 'voice', lo='A2', hi='E5', tags=('african', 'world', 'oldfield'),
         sustain=True, desc='African call-and-response leader: strong open chest voice, scooped attack, falls away',
         voices=1, groups=1, vseq=((0, 'a'), (0.7, 'a'), (1.0, 'e')), att=0.035, rel=0.1, scoop=-90,
         scoop_t=0.03, vib_rate=5.8, vib_cents=10, vib_delay=0.45, jitter=8, drift=4, fall=-200, fall_t=0.08,
         sag=-25, fr=3500, fr_vel=4000, tilt=450, tilt_vel=3500, breath=0.04, shimmer=0.06, ring=0.5, ring_f=2600)
register('voice.chant_response', sing, 'voice', lo='A2', hi='E5', tags=('african', 'world', 'oldfield'),
         sustain=True, desc='African chorus response: many open-throated voices with octave doubling, punchy',
         voices=9, spread=14, onset=0.07, vow='a', octave=0.55, att=0.04, rel=0.1, scoop=-70, scoop_t=0.035,
         vib_cents=8, vib_delay=0.5, jitter=9, drift=6, fall=-150, fall_t=0.09, sag=-20, fr=3000, fr_vel=3500,
         tilt=400, tilt_vel=3000, breath=0.05, shimmer=0.07, width=0.8)
register('voice.throat_singing', throat, 'voice', lo='G1', hi='D3', tags=('asian', 'world', 'space'),
         sustain=True, desc='overtone throat singing: pressed drone whose harmonics whistle a stepping melody')
register('voice.whisper_choir', whisper, 'voice', lo='C3', hi='C6', tags=('oldfield', 'space', 'inharmonic'),
         sustain=True, desc='whispered breath choir: ghostly vowel-shaped air, faintly tinted with the note')
register('voice.robot', robot, 'voice', lo='C2', hi='C5', tags=('space', 'chip'), sustain=True,
         desc='robot formant voice: rigid buzz, vowels jumping in steps, speech-chip aliasing sparkle')
