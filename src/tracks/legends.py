"""
GLIMMER LEGENDS — a Rayman Legends–inspired soundpack, played on Canopy Orchestra.

What makes that kind of score work, rebuilt as original music:

  * Five themed worlds, each a looping level theme in three layers that stack
    with the action (explore -> action -> chase). Each has its own band:
      Briar Keep         fairy-tale castle   6/8, G major   harp, pizzicato, bassoon, whistling
      Toadstool Bayou    swamp jug band      swing, F       banjo, jaw harp, tuba, washboard, KAZOO
      Fiesta Calavera    day-of-the-dead     2-beat, Dm     guitar, accordion, trumpets, maracas
      Lagoon Agents      undersea spy surf   Em             vibes, walking bass, surf guitar, brass
      Thunder Olympus    mythic epic         Dm             choir, harp, timpani, horns, trumpets
  * A MUSICAL LEVEL ("Kazoo Rumble", 48 bars, 132 BPM): a gypsy-polka romp whose
    gameplay is choreographed to the music. An action lane (jump / punch /
    slide / glim) is part of the score: every jump lands on a brass "ba!" and a
    cymbal, every punch on a snare-and-clap hit, every slide on a swell. The
    lane is exported as a beatmap, so a level designer places obstacles on beats.
  * Glims: collectible chimes at every semitone (C5–C7) plus golden "king"
    versions, so a game can play each pickup as the next note of the current
    chord — collecting a row of them plays a melody.
  * Per-world stingers: checkpoint, freed (the little creatures cheer),
    king (glims turn gold), death, and level complete.

No samples, no quoted melodies.

    cd src && python3 -m tracks.legends            # engine pack -> ../out/legends
    python3 -m tracks.legends --web                # + player/legends/ (sprites + legends.json)
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import orchestra  # noqa: E402,F401
from orchestra import nm  # noqa: E402
from orchestra.mix import Session  # noqa: E402
from orchestra import theory as th  # noqa: E402
from tracks.microgame import seq, hits, comp  # noqa: E402

SR = 48000
HERE = os.path.dirname(os.path.abspath(__file__))


def P(S, name, inst, **kw):
    """part helper: unique name, sensible defaults."""
    kw.setdefault('sends', {'room': 0.15})
    return S.part(f'{name}#{len(S.parts)}', inst, **kw)


def arp(p, chords, pattern, step, bb, lo, hi, vel=0.5, gate=1.6, t0=0.0):
    """chord-tone arpeggio: pattern = indexes into the chord tones in [lo, hi]."""
    for b, c in enumerate(chords):
        tones = [m for m in range(nm(lo), nm(hi) + 1) if m % 12 in th.pcs(c)]
        for k, ix in enumerate(pattern):
            if ix is None:
                continue
            p.note(t0 + b * bb + k * step, tones[ix % len(tones)], step * gate, vel * (1.1 if k == 0 else 1.0))


def pad(p, chords, bb, lo='G3', n=4, vel=0.5, hi=None):
    prev = th.voice(chords[0], nm(lo), n)
    for b, c in enumerate(chords):
        v = th.lead(prev, c, nm(lo) - 4, nm(hi) if hi else nm(lo) + 16, n)
        prev = v
        for m in v:
            p.note(b * bb, m, bb * 0.98, vel)


def roots(p, chords, pattern, step, bb, lo='E1', vel=0.8, gate=0.8):
    """bass: pattern chars per step: x root, 5 fifth, 8 octave, 3 third, . rest."""
    for b, c in enumerate(chords):
        r = th.bass_note(c.split()[0], nm(lo), nm(lo) + 11)
        ch = th.chord(c.split()[0])
        third = 3 if 3 in ch['ints'] else 4
        for k, x in enumerate(pattern):
            if x == '.':
                continue
            m = {'x': r, '5': r + 7, '8': r + 12, '3': r + third}[x]
            p.note(b * bb + k * step, m, step * gate, vel * (1.1 if k == 0 else 1.0))


# =========================================================== the five worlds
# each world function adds the parts of ONE tier (1 explore, 2 action, 3 chase)
def briar_keep(S, tier):
    bb, st = 3, 0.5                      # 6/8: a bar is 3 quarter beats, six eighths
    prog = ['G', 'Em', 'C', 'D', 'G', 'Em', 'Am', 'D']
    if tier == 1:
        h = P(S, 'harp', 'pluck.harp', pan=-0.3, gain_db=-4, sends={'hall': 0.3})
        arp(h, prog, [0, 2, 4, 5, 4, 2], st, bb, 'G3', 'G5', vel=0.5)
        s = P(S, 'strings', 'strings.section_warm', gain_db=-13, sends={'hall': 0.35})
        pad(s, prog, bb, 'D4', 3)
        c = P(S, 'celesta', 'keys.celesta', pan=0.4, gain_db=-12, sends={'hall': 0.35})
        for b, ch in enumerate(prog):
            if b % 2 == 0:
                c.note(b * bb, th.lead([nm('B5')], ch, nm('G5'), nm('D6'), 1)[0], 1.5, 0.5)
    elif tier == 2:
        pz = P(S, 'pizz', 'strings.pizzicato', gain_db=-3, sends={'hall': 0.2})
        roots(pz, prog, 'x..5..', st, bb, lo='G2', vel=0.8)
        bsn = P(S, 'bassoon', 'reed.bassoon', pan=0.25, gain_db=-6, sends={'hall': 0.25})
        seq(bsn, 'G3 . B3 D4 . B3 | E3 . G3 B3 . G3 | C4 . E4 G4 . E4 | D4 . F#4 A4 . F#4 | '
                 'G3 . B3 D4 . G4 | E4 . D4 B3 . G3 | A3 . C4 E4 . C4 | D4 . C4 A3 . F#3', st, vel=0.6, gate=0.55)
        fd = P(S, 'frame', 'drum.frame_drum', gain_db=-8, sends={'hall': 0.15})
        tb = P(S, 'tamb', 'perc.tambourine', pan=0.45, gain_db=-15, sends={'hall': 0.15})
        for b in range(8):
            hits(fd, 'x..x.x', st, t0=b * bb, vel=0.7)
            hits(tb, '...x..', st, t0=b * bb, vel=0.6)
    else:
        w = P(S, 'whistle', 'voice.whistle', pan=0.05, gain_db=-2, sends={'hall': 0.3, 'echo': 0.08})
        seq(w, 'D5 - B4 G4 A4 B4 | E5 - - D5 - B4 | C5 D5 E5 G5 - E5 | F#5 - - - D5 - | '
               'D5 - B4 G4 A4 B4 | E5 - G5 F#5 - E5 | C5 E5 A5 F#5 - D5 | G5 - - - - .', st, vel=0.7)
        rec = P(S, 'recorder', 'flute.recorder_alto', pan=-0.25, gain_db=-8, sends={'hall': 0.3})
        seq(rec, 'B4 - G4 E4 F#4 G4 | C5 - E5 D5 - C5 | A4 C5 F#5 D5 - A4 | B4 - - - - .', st, t0=4 * bb, vel=0.55)
        gl = P(S, 'glock', 'bell.glockenspiel', pan=0.4, gain_db=-11, sends={'hall': 0.3})
        for b, x in enumerate(['D6', 'E6', 'G6', 'F#6', 'D6', 'G6', 'A6', 'G6']):
            gl.note(b * bb, x, 1.0, 0.55)
        tc = P(S, 'teensies', 'voice.teensy_babble', pan=0.5, gain_db=-10, sends={'hall': 0.3})
        tc.note(7 * bb + 1.5, 'D6', 0.6, 0.6)
    return 8, bb


def toadstool_bayou(S, tier):
    bb, st = 4, 0.5
    prog = ['F', 'F', 'Bb', 'F', 'C7', 'Bb', 'C7', 'F']
    sw = 0.16                           # shuffle
    if tier == 1:
        bj = P(S, 'banjo', 'pluck.banjo', pan=-0.3, gain_db=-6, swing=sw, sends={'room': 0.15})
        arp(bj, prog, [0, 2, 4, 2, 5, 2, 4, 2], st, bb, 'C4', 'C6', vel=0.5, gate=1.2)
        jh = P(S, 'jaw', 'pluck.jaw_harp', pan=0.3, gain_db=-8, sends={'room': 0.15})
        for b, c in enumerate(prog):
            r = th.bass_note(c, nm('C2'), nm('B2'))
            jh.note(b * bb, r + 12, 1.6, 0.7)
            jh.note(b * bb + 2, r + 12, 1.6, 0.6)
        fr = P(S, 'frogs', 'creature.treefrog_choir', pan=0.6, gain_db=-14, sends={'canopy': 0.3})
        for b, c in enumerate(prog):
            for s in (1.5, 3.5):
                fr.note(b * bb + s, th.lead([nm('C6')], c, nm('A5'), nm('F6'), 1)[0], 0.3, 0.5)
    elif tier == 2:
        tu = P(S, 'tuba', 'brass.tuba', gain_db=-3, sends={'room': 0.12})
        roots(tu, prog, 'x...5...', st, bb, lo='C2', vel=0.85, gate=0.75)
        wb = P(S, 'washboard', 'perc.guiro', pan=0.35, gain_db=-13, swing=sw, sends={'room': 0.12})
        ac = P(S, 'accordion', 'organ.accordion_musette', pan=-0.2, gain_db=-12, swing=sw, sends={'room': 0.15})
        for b in range(8):
            hits(wb, 'x.xxx.xx', st, t0=b * bb, vel=0.6)
        comp(ac, prog, '..x...x.', lo='A3', n=3, gate=0.45, vel=0.55)
        sn = P(S, 'snare', 'drum.snare_brush', gain_db=-12, swing=sw, sends={'room': 0.15})
        for b in range(8):
            hits(sn, '..x...x.', st, t0=b * bb, vel=0.6)
    else:
        kz = P(S, 'kazoo', 'reed.kazoo', pan=0.05, gain_db=-1, swing=sw, sends={'room': 0.2, 'echo': 0.06})
        seq(kz, 'A4 C5 - A4 F4 - G4 A4 | C5 - - . A4 C5 D5 C5 | D5 - F5 - D5 C5 Bb4 - | A4 - - - . C5 A4 F4 | '
                'G4 Bb4 C5 - E5 - C5 - | D5 C5 Bb4 - G4 - Bb4 D5 | C5 - A4 F4 G4 - E4 G4 | F4 - - - . . . .', st, vel=0.75)
        kb = P(S, 'kazoo_lo', 'reed.kazoo_bass', pan=-0.3, gain_db=-9, swing=sw, sends={'room': 0.2})
        seq(kb, 'F4 . . . C4 . . . | F4 . . . A4 . . . | Bb3 . . . D4 . . . | F4 . . . C4 . . . | '
                'E4 . . . G4 . . . | D4 . . . F4 . . . | E4 . . . Bb3 . . . | F3 . . . . . . .', st, vel=0.6)
        su = P(S, 'slide', 'synth.slide_up', pan=0.4, gain_db=-12, sends={'room': 0.2})
        su.note(3 * bb + 3, 'C6', 0.5, 0.7)
        sd = P(S, 'slide_dn', 'synth.slide_down', pan=-0.4, gain_db=-12, sends={'room': 0.2})
        sd.note(7 * bb + 2, 'F4', 0.6, 0.7)
        bf = P(S, 'bullfrog', 'creature.bullfrog', gain_db=-6, sends={'canopy': 0.2})
        bf.note(7 * bb + 2.5, 'F2', 0.4, 0.8)
        bf.note(7 * bb + 3.5, 'C2', 0.4, 0.8)
    return 8, bb


def fiesta_calavera(S, tier):
    bb, st = 4, 0.5
    prog = ['Dm', 'Dm', 'A7', 'A7', 'Gm', 'Dm', 'A7', 'Dm']
    if tier == 1:
        g = P(S, 'guitar', 'guitar.nylon_spanish', pan=-0.3, gain_db=-7, sends={'room': 0.15})
        comp(g, prog, '..x...x...x...x.', lo='D3', n=4, strum=0.012, gate=0.5, vel=0.55)
        gt = P(S, 'guitarron', 'bass.upright_pizz', gain_db=-3, sends={'room': 0.1})
        roots(gt, prog, 'x...5...', st, bb, lo='D2', vel=0.85, gate=0.6)
        mr = P(S, 'marimba', 'mallet.marimba_rosewood', pan=0.35, gain_db=-11, sends={'room': 0.2})
        arp(mr, prog, [0, None, 2, None, 4, None, 2, None], st, bb, 'D4', 'D6', vel=0.5)
    elif tier == 2:
        ac = P(S, 'accordion', 'organ.accordion_musette', pan=0.25, gain_db=-10, sends={'room': 0.15})
        comp(ac, prog, '.x.x.x.x', lo='F4', n=3, gate=0.4, vel=0.5)
        sh = P(S, 'maracas', 'perc.shaker_egg', pan=0.45, gain_db=-14, sends={'room': 0.1})
        cl = P(S, 'claves', 'perc.claves', pan=-0.4, gain_db=-12, sends={'room': 0.1})
        cg = P(S, 'conga', 'drum.conga_open', pan=-0.2, gain_db=-9, sends={'room': 0.15})
        for b in range(8):
            hits(sh, 'xxxxxxxxxxxxxxxx', 0.25, t0=b * bb, vel=0.5)
            hits(cl, 'x..x..x...x.x...', 0.25, t0=b * bb, vel=0.6)
            hits(cg, '......x.x.....x.', 0.25, t0=b * bb, vel=0.7)
    else:
        t1 = P(S, 'trumpet', 'brass.trumpet', pan=-0.15, gain_db=-1, sends={'hall': 0.22})
        t2 = P(S, 'trumpet2', 'brass.trumpet', pan=0.2, gain_db=-4, sends={'hall': 0.22})
        mel = ('D5 - F5 - A5 - G5 F5 | E5 F5 G5 - F5 - D5 - | C#5 - E5 - G5 - F5 E5 | D5 E5 F5 - E5 - . . | '
               'D5 - G5 - Bb5 - A5 G5 | F5 G5 A5 - F5 - D5 - | E5 F5 G5 - C#5 - E5 - | D5 - - - . . A4 .')
        seq(t1, mel, st, vel=0.75, gate=0.85)
        scale = th.scale_notes('D', 'harmonic_minor', nm('A3'), nm('D6'))
        toks = mel.replace('|', ' ').split()
        for k, x in enumerate(toks):           # second trumpet: a third below, in D harmonic minor
            if x in '.-':
                continue
            ln = 1
            while k + ln < len(toks) and toks[k + ln] == '-':
                ln += 1
            t2.note(k * st, th.diatonic_shift([nm(x)], -2, scale)[0], ln * st * 0.85, 0.62)
        vn = P(S, 'violin', 'bowed.violin', pan=0.4, gain_db=-11, sends={'hall': 0.3})
        seq(vn, 'D6 - - - A5 - - - | C#6 - - - E6 - - - | D6 - - - G6 - - - | F6 - - - - - - -', 1.0, t0=4 * bb, vel=0.55)
        tc = P(S, 'teensies', 'voice.teensy_cheer', pan=-0.5, gain_db=-9, sends={'hall': 0.3})
        tc.note(7 * bb + 1, 'A5', 0.7, 0.75)
    return 8, bb


def lagoon_agents(S, tier):
    bb, st = 4, 0.5
    prog = ['Em', 'Em', 'C', 'B7', 'Em', 'Em', 'Am', 'B7']
    if tier == 1:
        vb = P(S, 'vibes', 'metal.vibraphone', pan=0.3, gain_db=-8, sends={'hall': 0.3})
        for b, c in enumerate(prog):
            sym = {'Em': 'Em9', 'C': 'Cmaj7', 'B7': 'B7', 'Am': 'Am7'}[c]
            vb.chord(b * bb, th.voice(sym, nm('E4'), 4), 2.5, 0.5, strum=0.01)
            vb.chord(b * bb + 2.5, th.voice(sym, nm('E4'), 3), 1.2, 0.35)
        wb = P(S, 'walking', 'bass.upright_pizz', gain_db=-3, sends={'room': 0.1})
        walk = 'E2 G2 A2 B2 | E2 B1 D2 D#2 | C2 E2 G2 A2 | B1 D#2 F#2 A2 | E2 G2 A2 B2 | E2 D2 C#2 C2 | A1 C2 E2 G2 | B1 F#2 A2 D#2'
        seq(wb, walk, 1.0, vel=0.8, gate=0.85)
        rd = P(S, 'ride', 'perc.cymbal_ride', pan=0.4, gain_db=-17, swing=0.12, sends={'room': 0.1})
        br = P(S, 'brush', 'drum.snare_brush', gain_db=-12, sends={'room': 0.15})
        for b in range(8):
            hits(rd, 'x.xxx.xx', st, t0=b * bb, vel=0.5)
            hits(br, '..x...x.', st, t0=b * bb, vel=0.55)
    elif tier == 2:
        gt = P(S, 'surf', 'guitar.electric_clean', pan=-0.25, gain_db=-5, sends={'canopy': 0.3, 'echo': 0.15})
        riff = 'E3 . . G3 . . A3 A#3 | B3 . . . . . . . | C4 . . B3 . . A3 G3 | F#3 . . . D#3 . . . | ' \
               'E3 . . G3 . . A3 A#3 | B3 . . . E4 . D4 . | A3 . . C4 . . E4 . | D#4 . . . B3 . . .'
        seq(gt, riff, st, vel=0.75, gate=0.7)
        bg = P(S, 'bongos', 'drum.bongo', pan=0.35, gain_db=-11, sends={'room': 0.12})
        org = P(S, 'farfisa', 'organ.farfisa', pan=0.4, gain_db=-16, sends={'room': 0.15})
        for b in range(8):
            hits(bg, 'x.x..x.xx.x..x..', 0.25, t0=b * bb, vel=0.6)
        comp(org, prog, '....x.......x...', lo='G4', n=3, gate=0.6, vel=0.5)
    else:
        fl = P(S, 'flute', 'flute.concert', pan=0.1, gain_db=-3, sends={'hall': 0.3, 'echo': 0.1})
        seq(fl, 'B4 - E5 - G5 - F#5 E5 | D#5 - E5 - - - . . | C5 - E5 - G5 - B5 A5 | F#5 - D#5 - B4 - . . | '
                'B4 - E5 - G5 - B5 - | A5 G5 F#5 - E5 - . . | C5 E5 A5 - G5 F#5 E5 - | D#5 - F#5 - B5 - . .', st, vel=0.7)
        tp = P(S, 'trumpet', 'brass.trumpet_muted', pan=-0.3, gain_db=-6, sends={'hall': 0.2})
        tb = P(S, 'trombone', 'brass.trombone', pan=0.3, gain_db=-7, sends={'hall': 0.2})
        for b, c in enumerate(prog):
            v = th.voice({'Em': 'Em7', 'C': 'Cmaj7', 'B7': 'B7', 'Am': 'Am7'}[c], nm('E4'), 3)
            for s in (3.5,):
                tp.chord(b * bb + s, v, 0.4, 0.7)
                tb.note(b * bb + s, v[0] - 12, 0.4, 0.7)
        sw = P(S, 'slide', 'synth.slide_up', pan=0.5, gain_db=-13, sends={'hall': 0.25})
        sw.note(3 * bb + 3, 'B5', 0.5, 0.6)
    return 8, bb


def thunder_olympus(S, tier):
    bb, st = 4, 0.5
    prog = ['Dm', 'Bb', 'F', 'C', 'Dm', 'Bb', 'Gm', 'A']
    if tier == 1:
        h = P(S, 'harp', 'pluck.harp', pan=-0.35, gain_db=-6, sends={'church': 0.3})
        arp(h, prog, [0, 2, 4, 6, 4, 2, 4, 2], st, bb, 'D3', 'D5', vel=0.5)
        ch = P(S, 'choir_ooh', 'voice.choir_ooh', gain_db=-12, sends={'church': 0.4})
        pad(ch, prog, bb, 'A3', 3)
        lo = P(S, 'low_strings', 'strings.section_warm', gain_db=-12, sends={'church': 0.3})
        for b, c in enumerate(prog):
            lo.note(b * bb, th.bass_note(c, nm('D2'), nm('C#3')), bb * 0.98, 0.55)
    elif tier == 2:
        tm = P(S, 'timpani', 'drum.timpani', gain_db=-5, sends={'church': 0.25})
        bd = P(S, 'bass_drum', 'drum.bass_drum', gain_db=-6, sends={'church': 0.25})
        for b, c in enumerate(prog):
            r = th.bass_note(c, nm('D2'), nm('C#3'))
            tm.note(b * bb, r, 0.9, 0.85)
            tm.note(b * bb + 2.5, r, 0.4, 0.6)
            tm.note(b * bb + 3, r, 0.4, 0.7)
            hits(bd, 'x.......x.x.....', 0.25, t0=b * bb, vel=0.8)
        hn = P(S, 'horns', 'brass.french_horn', pan=0.2, gain_db=-6, sends={'church': 0.3})
        arp(hn, prog, [0, None, 1, 2, None, 1, 0, None], st, bb, 'D3', 'A4', vel=0.6, gate=0.8)
        pz = P(S, 'pizz', 'strings.pizzicato', pan=-0.2, gain_db=-9, sends={'church': 0.2})
        arp(pz, prog, [0, 1, 2, 1, 0, 1, 2, 1], st, bb, 'D3', 'D4', vel=0.55, gate=0.6)
    else:
        tp = P(S, 'trumpets', 'brass.trumpet', gain_db=-1, sends={'church': 0.3})
        seq(tp, 'D5 - A4 - D5 - F5 - | F5 - - - D5 - Bb4 - | C5 - F5 - A5 - G5 F5 | E5 - - - C5 - - - | '
                'D5 - A4 - D5 - F5 - | F5 - G5 - A5 - Bb5 - | A5 - G5 - F5 - E5 - | C#5 - - - E5 - - -', st, vel=0.78)
        ca = P(S, 'choir_aah', 'voice.choir_aah', gain_db=-9, sends={'church': 0.4})
        pad(ca, prog, bb, 'D4', 4, vel=0.6)
        cr = P(S, 'crash', 'perc.cymbal_crash', gain_db=-13, sends={'church': 0.3})
        cr.note(0, 60, 2, 0.8)
        cr.note(4 * bb, 60, 2, 0.8)
        gl = P(S, 'glock', 'bell.glockenspiel', pan=0.4, gain_db=-12, sends={'church': 0.3})
        for b, x in enumerate(['D6', 'F6', 'A6', 'E6', 'D6', 'Bb6', 'A6', 'C#7']):
            gl.note(b * bb, x, 1.0, 0.55)
    return 8, bb


WORLDS = [
    dict(id='briar', chords=['G', 'Em', 'C', 'D', 'G', 'Em', 'Am', 'D'], name='Briar Keep', mood='fairy-tale castle, sunny ramparts', fn=briar_keep, bpm=120, key='G',
         mode='major', lead='voice.whistle', meter='6/8'),
    dict(id='bayou', chords=['F', 'F', 'Bb', 'F', 'C7', 'Bb', 'C7', 'F'], name='Toadstool Bayou', mood='swampy jug band with a kazoo', fn=toadstool_bayou, bpm=108, key='F',
         mode='major', lead='reed.kazoo', meter='4/4 swing'),
    dict(id='fiesta', chords=['Dm', 'Dm', 'A7', 'A7', 'Gm', 'Dm', 'A7', 'Dm'], name='Fiesta Calavera', mood='day-of-the-dead mariachi party', fn=fiesta_calavera, bpm=132,
         key='D', mode='harmonic_minor', lead='brass.trumpet', meter='4/4'),
    dict(id='lagoon', chords=['Em', 'Em', 'C', 'B7', 'Em', 'Em', 'Am', 'B7'], name='Lagoon Agents', mood='undersea spy surf jazz', fn=lagoon_agents, bpm=120, key='E',
         mode='minor', lead='flute.concert', meter='4/4'),
    dict(id='olympus', chords=['Dm', 'Bb', 'F', 'C', 'Dm', 'Bb', 'Gm', 'A'], name='Thunder Olympus', mood='mythic temples in the clouds', fn=thunder_olympus, bpm=100,
         key='D', mode='minor', lead='brass.trumpet', meter='4/4'),
]


# =========================================================== stingers (per world, in its key)
def stinger(S, kind, W):
    tonic = nm(W['key'] + '5')
    minor = 'minor' in W['mode']
    third = 3 if minor else 4
    lead = W['lead']
    lo_lead = lead if lead != 'voice.whistle' else 'flute.recorder_alto'
    if kind == 'checkpoint':
        p = P(S, 'cp', lo_lead, sends={'hall': 0.3})
        for i, iv in enumerate([0, third, 7, 12]):
            p.note(i * 0.25, tonic - 12 + iv if lead.startswith('brass') else tonic + iv - 12, 0.4 if i < 3 else 1.2, 0.7)
        g = P(S, 'cpg', 'bell.glim', pan=0.3, gain_db=-6, sends={'hall': 0.3})
        g.note(0.75, tonic + 12, 1, 0.7)
        return 2.0
    if kind == 'freed':
        tc = P(S, 'cheer', 'voice.teensy_cheer', pan=-0.2, sends={'hall': 0.3})
        tc2 = P(S, 'cheer2', 'voice.teensy_cheer', pan=0.3, gain_db=-3, sends={'hall': 0.3})
        tc.note(0.0, tonic + 7, 1.1, 0.8)
        tc2.note(0.15, tonic + 12, 1.0, 0.75)
        hp_ = P(S, 'gliss', 'pluck.harp', gain_db=-4, sends={'hall': 0.35})
        sc = th.scale_notes(W['key'], W['mode'], tonic - 12, tonic + 12)
        for i, m in enumerate(sc):
            hp_.note(i * 0.06, m, 1.0, 0.5 + 0.02 * i)
        return 2.5
    if kind == 'king':
        g = P(S, 'king', 'bell.glim_king', gain_db=-2, sends={'hall': 0.35})
        for i, iv in enumerate([0, third, 7, 12, 7 + 12]):
            g.note(i * 0.12, tonic + iv, 1.0, 0.75)
        ch = P(S, 'choir', 'voice.choir_aah', gain_db=-8, sends={'hall': 0.4})
        for iv in (0, third, 7):
            ch.note(0.0, tonic - 12 + iv, 1.6, 0.6)
        return 2.5
    if kind == 'death':
        sd = P(S, 'fall', 'synth.slide_down', sends={'room': 0.2})
        sd.note(0, tonic - 12, 0.5, 0.8)
        tm = P(S, 'bonk', 'drum.tom', gain_db=-3, sends={'room': 0.2})
        tm.note(0.5, 60, 0.4, 0.9)
        kz = P(S, 'wah', 'reed.kazoo_bass', gain_db=-3, sends={'room': 0.25})
        kz.note(1.0, tonic - 12 + 7, 0.45, 0.7)
        kz.note(1.5, tonic - 12 + 6, 0.45, 0.65)
        kz.note(2.0, tonic - 12 + 5, 1.0, 0.6)
        return 3.2
    if kind == 'complete':
        p = P(S, 'fanfare', lead if lead != 'voice.whistle' else 'brass.trumpet', sends={'hall': 0.3})
        tune = [(0, 0, 0.4), (0.5, third, 0.4), (1.0, 7, 0.4), (1.5, 12, 1.0), (2.75, 7, 0.25), (3.0, 12, 2.0)]
        base = tonic - 12 if p.inst.startswith('brass') or p.inst.startswith('reed') else tonic
        for t, iv, d in tune:
            p.note(t, base + iv, d, 0.8)
        cr = P(S, 'crash', 'perc.cymbal_crash', gain_db=-9, sends={'hall': 0.3})
        cr.note(3.0, 60, 2.0, 0.85)
        tc = P(S, 'cheer', 'voice.teensy_cheer', pan=0.35, gain_db=-4, sends={'hall': 0.3})
        tc.note(3.0, tonic + 7, 1.2, 0.8)
        gl = P(S, 'glims', 'bell.glim', pan=-0.3, gain_db=-8, sends={'hall': 0.3})
        for i, iv in enumerate([0, third, 7, 12, third + 12, 19, 24]):
            gl.note(3.0 + i * 0.07, tonic + iv, 0.8, 0.6)
        return 5.0
    raise KeyError(kind)


STINGERS = ['checkpoint', 'freed', 'king', 'death', 'complete']


# =========================================================== the musical level
LEVEL_BPM = 132
LEVEL_SECTIONS = [(0, 'Intro'), (4, 'Verse'), (12, 'Verse again'), (20, 'Break'), (24, 'Chorus'), (32, 'Chorus again'),
                  (40, 'Bridge'), (44, 'Finale')]
# 8 eighth-note slots per bar. J jump · P punch · S slide · G glim
ACTIONS = [
    '........', '........', '....G.G.', 'G.G.G.J.',
    'J...P...', '..G.G.G.', 'J...J...', 'P.P.....', 'J...P...', 'GG..GG..', 'J..J..J.', 'S.......',
    'J...P...', '..G.G.GJ', '..P...P.', 'J...S...', 'J.J.J.J.', 'GGGGGGGG', 'P...P...', 'J.....S.',
    '........', 'G...G...', 'G...G...', 'G.G.GGJ.',
    'J...J...', 'P.P.P...', 'J..J..J.', 'GG.GG.GG', 'J...P...', 'S...J...', 'P..P..P.', 'J.......',
    'J.G.J.G.', 'P...P...', 'J..J..J.', 'S.......', 'JJ..JJ..', 'GGGG.P..', 'P.P.P.P.', 'J.....J.',
    'S.......', 'J.J.J.J.', 'P.P.P.P.', 'JJJJJJJ.',
    'J...P...', 'GGGGGGGG', 'J...J...', 'P.......',
]
LEVEL_PROG = (['Am'] * 4 +
              ['Am', 'Am', 'Dm', 'Am', 'E7', 'E7', 'Am', 'Am'] * 2 +
              ['Am', 'Am', 'E7', 'E7'] +
              ['F', 'G', 'Am', 'Am', 'F', 'G', 'E7', 'E7'] * 2 +
              ['Dm', 'Dm', 'E7', 'E7'] +
              ['F', 'G', 'E7', 'Am'])
VERSE = ('A4 - C5 E5 A5 - G#5 A5 | B5 A5 G#5 A5 E5 - - . | F5 - E5 D5 A5 - F5 - | E5 - C5 - A4 - - . | '
         'B4 C5 D5 E5 G#5 - F5 E5 | D5 - B4 - G#4 - E4 - | A4 B4 C5 D5 E5 - A5 - | E5 - - . A4 . . .')
CHORUS = ('C6 - A5 - F5 - A5 C6 | B5 - G5 - D5 - G5 B5 | C6 - B5 A5 E5 - A5 - | B5 C6 B5 A5 G#5 - E5 - | '
          'A5 - C6 - F6 - E6 D6 | D6 - B5 - G5 - B5 D6 | E6 - D6 C6 B5 - G#5 - | E5 - - - . . . .')


def musical_level(S):
    bb, st = 4, 0.5
    S.tempo_at(0, LEVEL_BPM)
    kick = P(S, 'kick', 'drum.kick_modern', gain_db=-4, sends={'room': 0.08})
    snare = P(S, 'snare', 'drum.snare', gain_db=-8, sends={'room': 0.18})
    hat = P(S, 'hat', 'perc.shaker_egg', pan=0.3, gain_db=-15, sends={'room': 0.1})
    tamb = P(S, 'tamb', 'perc.tambourine', pan=-0.4, gain_db=-15, sends={'room': 0.12})
    clap = P(S, 'clap', 'perc.hand_claps', gain_db=-10, sends={'room': 0.2})
    tom = P(S, 'tom', 'drum.tom', gain_db=-8, sends={'room': 0.2})
    tuba = P(S, 'tuba', 'brass.tuba', gain_db=-4, sends={'room': 0.12})
    bass = P(S, 'bass', 'bass.upright_pizz', gain_db=-6, sends={'room': 0.1})
    acc = P(S, 'accordion', 'organ.accordion_musette', pan=-0.25, gain_db=-12, sends={'room': 0.15})
    banjo = P(S, 'banjo', 'pluck.banjo', pan=0.35, gain_db=-12, sends={'room': 0.15})
    kz = P(S, 'kazoo', 'reed.kazoo', pan=0.05, gain_db=-2, sends={'room': 0.2, 'echo': 0.05})
    kz2 = P(S, 'kazoo2', 'reed.kazoo', pan=-0.3, gain_db=-8, sends={'room': 0.2})
    tpt = P(S, 'trumpet', 'brass.trumpet', pan=0.15, gain_db=-2, sends={'hall': 0.22})
    vln = P(S, 'violin', 'bowed.fiddle', pan=0.4, gain_db=-9, sends={'hall': 0.25})
    # the action instruments: these make the gameplay audible
    stab_t = P(S, 'stab', 'brass.trumpet', pan=-0.15, gain_db=-5, sends={'hall': 0.25})
    stab_b = P(S, 'stab_lo', 'brass.trombone', pan=0.2, gain_db=-6, sends={'hall': 0.25})
    crash = P(S, 'crash', 'perc.cymbal_crash', gain_db=-16, sends={'hall': 0.2})
    hitsn = P(S, 'punch', 'drum.snare', gain_db=-5, sends={'room': 0.2})
    hitcl = P(S, 'punch_clap', 'perc.hand_claps', gain_db=-8, sends={'room': 0.2})
    swell = P(S, 'swell', 'perc.cymbal_swell', gain_db=-9, sends={'hall': 0.25})
    slide = P(S, 'slide', 'synth.slide_up', pan=0.4, gain_db=-11, sends={'room': 0.2})
    slide_d = P(S, 'slide_down', 'synth.slide_down', pan=-0.4, gain_db=-11, sends={'room': 0.2})
    cheer = P(S, 'teensies', 'voice.teensy_cheer', pan=0.5, gain_db=-7, sends={'hall': 0.3})

    def sec(bar):
        return max(b for b, _ in LEVEL_SECTIONS if b <= bar)

    for bar, ch in enumerate(LEVEL_PROG):
        t0 = bar * bb
        s = sec(bar)
        root = th.bass_note(ch, nm('E1'), nm('D#2'))
        # ---- drums
        if s == 0:
            if bar >= 1:
                hits(kick, 'x...x...x...x...', 0.25, t0, vel=0.6 + 0.1 * bar)
            if bar == 3:
                hits(snare, '........x.x.xxxx', 0.25, t0, vel=0.6)
        elif s == 20:
            hits(kick, 'x.......x.......', 0.25, t0)
            hits(clap, '....x.......x...', 0.25, t0)
        elif s == 40:
            hits(kick, 'x...x...x...x...', 0.25, t0)
            hits(tom, '..x...x...x.xxxx', 0.25, t0, vel=0.6 + 0.1 * (bar - 40))
        else:
            chorus = s in (24, 32, 44)
            hits(kick, 'x...x...x...x...' if chorus else 'x.......x.......', 0.25, t0)
            hits(snare, '....x.......x...' if bar % 8 != 7 else '....x.......x.xx', 0.25, t0)
            hits(hat, 'x.x.x.x.x.x.x.x.', 0.25, t0, vel=0.55)
            if chorus:
                hits(tamb, '..x...x...x...x.', 0.25, t0, vel=0.6)
        # ---- bass and chords
        if bar >= 2 and s != 40:
            roots(tuba, [ch], 'x...5...', st, bb, lo='E1', vel=0.85, gate=0.7)
            tuba.notes[-2].beat += t0
            tuba.notes[-1].beat += t0
        if s in (24, 32, 44, 40):
            roots(bass, [ch], 'x.5.8.5.', st, bb, lo='E2', vel=0.75, gate=0.6)
            for nt in bass.notes[-4:]:
                nt.beat += t0
        if bar >= 4 and s != 20:
            v = th.voice(ch, nm('A3'), 3)
            for k in (1, 3, 5, 7):
                acc.chord(t0 + k * st, v, st * 0.5, 0.55)
        if s in (24, 32, 40, 44):
            v = th.voice(ch, nm('E4'), 4)
            for k in range(8):
                banjo.chord(t0 + k * st, v, st * 0.6, 0.5 if k % 2 else 0.6, strum=0.008)
    # ---- melodies
    for start in (4, 12):
        seq(kz, VERSE, st, t0=start * bb, vel=0.75)
    seq(kz, 'A4 - - - E5 - - - | A5 - - - E5 - - - | B4 - - - E5 - - - | G#5 - - - B5 - - -', st, t0=20 * bb, vel=0.6)
    for start in (24, 32):
        seq(tpt, CHORUS, st, t0=start * bb, vel=0.8)
        scale = th.scale_notes('A', 'harmonic_minor', nm('C4'), nm('A6'))
        toks = CHORUS.replace('|', ' ').split()
        for k, x in enumerate(toks):
            if x in '.-':
                continue
            ln = 1
            while k + ln < len(toks) and toks[k + ln] == '-':
                ln += 1
            kz2.note(start * bb + k * st, th.diatonic_shift([nm(x) - 12], -2, scale)[0], ln * st * 0.9, 0.62)
    seq(vln, 'D5 F5 A5 D6 F6 - D6 A5 | D5 F5 A5 D6 F6 - D6 A5 | E5 G#5 B5 E6 G#6 - E6 B5 | E5 G#5 B5 D6 E6 - - -', st,
        t0=40 * bb, vel=0.65)
    seq(kz, 'C6 - A5 - F5 - A5 C6 | B5 - G5 - D5 - G5 B5 | B5 C6 B5 A5 G#5 - E5 - | A5 - - - - - - -', st,
        t0=44 * bb, transpose=-12, vel=0.8)
    seq(tpt, 'C6 - A5 - F5 - A5 C6 | B5 - G5 - D5 - G5 B5 | B5 C6 B5 A5 G#5 - E5 - | A5 - - - - - - -', st,
        t0=44 * bb, vel=0.8)
    for bar in (20, 21, 22, 23):
        for b in (1, 3):
            cheer.note(bar * bb + b, 'E6' if b == 1 else 'A5', 0.35, 0.65)
    slide.note(3 * bb + 2.5, 'A5', 0.5, 0.7)
    # ---- the action lane, made audible (and recorded as the beatmap)
    beatmap = []
    for bar, row in enumerate(ACTIONS):
        ch = LEVEL_PROG[bar]
        tones = [m for m in range(nm('E5'), nm('C7') + 1) if m % 12 in th.pcs(ch)]   # inside the glim range
        gi = 0
        for k, a in enumerate(row):
            if a == '.':
                continue
            t = bar * bb + k * st
            ev = dict(t=round(S.clock.sec(t), 4), bar=bar, slot=k, beat=round(t, 3), action=a, chord=ch)
            if a == 'J':
                v = th.voice(ch, nm('E5'), 3)
                stab_t.chord(t, v, 0.22, 0.8)
                stab_b.note(t, th.bass_note(ch, nm('E3'), nm('D#4')), 0.22, 0.75)
                crash.note(t, 60, 1.0, 0.55)
            elif a == 'P':
                hitsn.note(t, 60, 0.3, 0.95)
                hitcl.note(t, 60, 0.3, 0.8)
                stab_b.note(t, th.bass_note(ch, nm('A2'), nm('G#3')), 0.18, 0.85)
            elif a == 'S':
                swell.note(t - 1.0, 60, 1.0, 0.7)
                slide_d.note(t, th.bass_note(ch, nm('A4'), nm('G#5')), 0.4, 0.7)
            elif a == 'G':
                ev['note'] = tones[gi % len(tones)]
                gi += 1
            beatmap.append(ev)
    end = len(LEVEL_PROG) * bb
    for p, ms in ((stab_t, th.voice('Am', nm('E5'), 3)), (tuba, [nm('A1')]), (acc, th.voice('Am', nm('A3'), 3))):
        for m in ms:
            p.note(end, m, 1.5, 0.9)
    crash.note(end, 60, 2.0, 0.85)
    kick.note(end, 60, 0.5, 1.0)
    cheer.note(end + 0.5, 'A5', 1.2, 0.8)
    return len(LEVEL_PROG), bb, beatmap


# =========================================================== rendering
def _session(bpm):
    s = Session(bpm=bpm)
    s.echo.update(beats=0.75, feedback=0.3, damp=3500.0)
    return s


def render_loop(fn, tier, bpm, tail_s=4.0):
    s = _session(bpm)
    bars, bb = fn(s, tier)
    N = int(round(bars * bb * SR * 60 / bpm))
    T = int(tail_s * SR)
    Pp = int(0.2 * SR)
    b = s.render_parts(t_end=(N + T) / SR + 1.0, t_offset=-Pp / SR)
    full = s.wet(b, b['n'])
    body = full[:, Pp:]
    loop = body[:, :N].copy()
    k = N
    while k < body.shape[1]:
        seg = body[:, k:k + N]
        loop[:, :seg.shape[1]] += seg
        k += N
    loop[:, N - Pp:] += full[:, :Pp]
    return bars, bb, loop, body[:, N:N + T]


def render_shot(build, bpm, tail_s=3.0):
    s = _session(bpm)
    length = build(s)
    b = s.render_parts(t_end=s.length_s(tail=tail_s))
    return s.wet(b, b['n']), length


def lufs(x):
    import pyloudnorm as pyln
    return pyln.Meter(SR).integrated_loudness(x.T.astype(np.float64))


def gain_to(x, target):
    return 10 ** ((target - lufs(x)) / 20)


def write(path, x):
    # block-wise: libsndfile's Vorbis encoder can overflow its stack on one huge write
    y = np.ascontiguousarray(np.clip(x, -1, 1).T, dtype=np.float32)
    with sf.SoundFile(path, 'w', SR, 2, format='OGG', subtype='VORBIS') as f:
        for i in range(0, len(y), SR):
            f.write(y[i:i + SR])


def build(out_dir, web=False):
    os.makedirs(out_dir, exist_ok=True)
    pack = dict(name='Glimmer Legends', sr=SR, worlds=[], glims={}, level={})
    web_sprites = {}
    # ---------------- worlds
    for W in WORLDS:
        stems = {}
        for tier in (1, 2, 3):
            bars, bb, loop, tail = render_loop(W['fn'], tier, W['bpm'])
            stems[tier] = (loop, tail)
        full = sum(stems[t][0] for t in (1, 2, 3))
        g = gain_to(np.concatenate([full, full], axis=1), -16.0)
        pk = np.max(np.abs(full)) * g
        if pk > 0.95:
            g *= 0.95 / pk
        d = os.path.join(out_dir, W['id'])
        os.makedirs(d, exist_ok=True)
        entry = dict(id=W['id'], name=W['name'], mood=W['mood'], bpm=W['bpm'], key=f"{W['key']} {W['mode'].replace('_', ' ')}",
                     meter=W['meter'], bars=bars, beats_per_bar=bb, loop_samples=int(stems[1][0].shape[1]),
                     scale=[m for m in th.scale_notes(W['key'], W['mode'], nm('C5'), nm('C7'))],
                     chords=W['chords'], chord_tones=[[m for m in range(nm('C5'), nm('C7') + 1) if m % 12 in th.pcs(c)]
                                                     for c in W['chords']],
                     stems={}, stingers={})
        sprite = []
        for tier in (1, 2, 3):
            loop, tail = stems[tier]
            write(os.path.join(d, f'tier{tier}.ogg'), loop * g)
            write(os.path.join(d, f'tier{tier}_tail.ogg'), tail * g)
            entry['stems'][f'tier{tier}'] = dict(file=f"{W['id']}/tier{tier}.ogg", tail=f"{W['id']}/tier{tier}_tail.ogg")
            sprite.append((f'tier{tier}', loop * g, tail * g))
        for kind in STINGERS:
            x, length = render_shot(lambda S, k=kind: stinger(S, k, W), W['bpm'])
            x = x * gain_to(x, -15.0)
            pkk = np.max(np.abs(x))
            if pkk > 0.95:
                x *= 0.95 / pkk
            write(os.path.join(d, f'{kind}.ogg'), x)
            entry['stingers'][kind] = dict(file=f"{W['id']}/{kind}.ogg", seconds=round(x.shape[1] / SR, 3))
            sprite.append((kind, x, None))
        pack['worlds'].append(entry)
        web_sprites[W['id']] = sprite
        print(f"{W['name']}: 3 stems x {bars} bars ({entry['loop_samples'] / SR:.1f} s) + {len(STINGERS)} stingers",
              flush=True)
    # ---------------- glims (every semitone C5..C7, normal and king)
    gd = os.path.join(out_dir, 'glims')
    os.makedirs(gd, exist_ok=True)
    gsprite = []
    ref = None
    for kind, inst in (('glim', 'bell.glim'), ('king', 'bell.glim_king')):
        for m in range(nm('C5'), nm('C7') + 1):
            def b_(S, m=m, inst=inst):
                p = P(S, 'g', inst, sends={'hall': 0.25})
                p.note(0, m, 0.5, 0.8)
                return 1
            x, _ = render_shot(b_, 120, tail_s=1.5)
            x = x * gain_to(np.concatenate([x, x], axis=1), -16.0 if kind == 'glim' else -15.0)
            write(os.path.join(gd, f'{kind}_{m}.ogg'), x)
            pack['glims'][f'{kind}_{m}'] = f'glims/{kind}_{m}.ogg'
            gsprite.append((f'{kind}_{m}', x, None))
    web_sprites['glims'] = gsprite
    print(f'glims: {len(gsprite)} chimes', flush=True)
    # ---------------- the musical level
    beatmap = []

    def lvl(S):
        nb, bb, bm = musical_level(S)
        beatmap.extend(bm)
        return nb
    x, nbars = render_shot(lvl, LEVEL_BPM, tail_s=4.0)
    x = x * gain_to(x, -14.0)
    pk = np.max(np.abs(x))
    if pk > 0.97:
        from orchestra import fx
        x = fx.limit(x, -1.0)
    write(os.path.join(out_dir, 'kazoo_rumble.ogg'), x)
    pack['level'] = dict(name='Kazoo Rumble', file='kazoo_rumble.ogg', bpm=LEVEL_BPM, beats_per_bar=4, bars=nbars,
                         seconds=round(x.shape[1] / SR, 3), music_end=round(nbars * 4 * 60 / LEVEL_BPM, 3),
                         sections=[dict(bar=b, name=n, t=round(b * 4 * 60 / LEVEL_BPM, 3)) for b, n in LEVEL_SECTIONS],
                         actions={'J': 'jump', 'P': 'punch', 'S': 'slide', 'G': 'glim'}, beatmap=beatmap)
    web_sprites['level'] = [('level', x, None)]
    print(f"Kazoo Rumble: {nbars} bars, {x.shape[1] / SR:.1f} s, {len(beatmap)} beatmap events", flush=True)
    json.dump(pack, open(os.path.join(out_dir, 'legends.json'), 'w'), indent=1)
    if web:
        export_web(pack, web_sprites)
    return pack


def export_web(pack, sprites):
    pdir = os.path.abspath(os.path.join(HERE, '..', '..', 'player', 'legends'))
    os.makedirs(pdir, exist_ok=True)
    gap = int(0.2 * SR)
    web = dict(name=pack['name'], sr=SR, worlds=[], glims={}, level=dict(pack['level']), sprites={})
    offsets = {}
    for sid, items in sprites.items():
        head = np.zeros((2, 8192), np.float32)
        head[:, 1000] = 0.9
        chunks, pos, offs = [head], 8192, {}
        for name, x, tail in items:
            offs[name] = dict(start=pos, length=int(x.shape[1]))
            chunks.append(x.astype(np.float32))
            pos += x.shape[1]
            if tail is not None:
                chunks.append(np.zeros((2, gap), np.float32))
                pos += gap
                offs[name]['tail'] = pos
                offs[name]['tail_length'] = int(tail.shape[1])
                chunks.append(tail.astype(np.float32))
                pos += tail.shape[1]
            chunks.append(np.zeros((2, gap), np.float32))
            pos += gap
        sp = np.concatenate(chunks, axis=1)
        wav = os.path.join(pdir, f'_{sid}.wav')
        sf.write(wav, np.clip(sp, -1, 1).T, SR, subtype='PCM_16')
        m4a = os.path.join(pdir, f'{sid}.m4a')
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', wav, '-c:a', 'aac', '-b:a', '112k', m4a], check=True)
        os.remove(wav)
        open(os.path.join(pdir, f'{sid}.b64.txt'), 'w').write(base64.b64encode(open(m4a, 'rb').read()).decode('ascii'))
        web['sprites'][sid] = dict(file=f'legends/{sid}.m4a', b64=f'legends/{sid}.b64.txt', sync=1000)
        offsets[sid] = offs
        print(f'web sprite {sid}: {sp.shape[1] / SR:.1f} s, {os.path.getsize(m4a) // 1024} KB')
    for W in pack['worlds']:
        w = {k: v for k, v in W.items() if k not in ('stems', 'stingers')}
        w['stems'] = {k: offsets[W['id']][k] for k in ('tier1', 'tier2', 'tier3')}
        w['stingers'] = {k: offsets[W['id']][k] for k in STINGERS}
        web['worlds'].append(w)
    web['glims'] = offsets['glims']
    web['level']['audio'] = offsets['level']['level']
    json.dump(web, open(os.path.join(pdir, 'legends.json'), 'w'), separators=(',', ':'))


if __name__ == '__main__':
    out = os.path.abspath(os.path.join(HERE, '..', '..', 'out', 'legends'))
    build(out, web='--web' in sys.argv)
