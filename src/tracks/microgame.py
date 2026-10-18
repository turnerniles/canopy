"""
MICROGAME RUSH — a WarioWare-style cue system, played on Canopy Orchestra.

Not one long loop but a *run* made of short, bar-exact cues that a game
sequences on the beat:

    interlude (2 bars, stage groove, shows lives/score, next command on bar 2)
      -> microgame (2 bars, one of 8 styles, loops if a game needs longer)
      -> win | lose (1 bar)
      -> every 4 games: speedup (2 bars) and the next tempo tier
      -> game 12: boss_intro (2 bars) -> boss (4-bar loop) -> boss_win | boss_lose
      -> clear (2 bars) and round again, faster;  lives gone: gameover (2 bars)

Everything is rendered at four tempo tiers (120, 144, 160, 180 BPM) so speed-ups
keep the timbres intact. All cues are in C major / A minor: every loop starts on
C or Am and the interlude ends on G7, so any cue can follow any other.

Loops are rendered circularly (seamless, like Canopy stems) with a ring-out
tail; one-shots carry their own tail. Original melodies, no samples.

    cd src && python3 -m tracks.microgame               # cue pack -> ../out/microgame
    python3 -m tracks.microgame --web                   # + player/microgame/ sprites
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
from orchestra import theory as th, fx  # noqa: E402

SR = 48000
TIERS = [120, 144, 160, 180]
STYLES = ['chip', 'funk', 'mariachi', 'surf', 'jungle', 'space', 'toybox', 'polka']
STYLE_NAMES = dict(chip='Chiptune', funk='Funk', mariachi='Mariachi', surf='Surf', jungle='Jungle marimba',
                   space='Space disco', toybox='Toy box', polka='Polka', boss='Boss')


# ------------------------------------------------------------------ pattern helpers
def seq(p, tokens, step, t0=0.0, vel=0.75, gate=0.92, transpose=0, accent=None):
    """tokens: 'C5 . E5 - G5' — one token per step, '.' rest, '-' hold."""
    toks = tokens.replace('|', ' ').split()
    for k, x in enumerate(toks):
        if x in '.-':
            continue
        ln = 1
        while k + ln < len(toks) and toks[k + ln] == '-':
            ln += 1
        v = vel * (1.12 if accent and k % accent == 0 else 1.0)
        p.note(t0 + k * step, nm(x) + transpose, ln * step * gate, v)


def hits(p, pattern, step, t0=0.0, vel=0.75, pitch=60):
    """'x' hit, 'X' accent, 'g' ghost, '.' rest."""
    for k, c in enumerate(pattern.replace('|', '').replace(' ', '')):
        if c == '.':
            continue
        v = vel * (1.25 if c == 'X' else 0.45 if c == 'g' else 1.0)
        p.note(t0 + k * step, pitch, step, min(1.0, v))


def comp(p, chords, rhythm, t0=0.0, lo='C4', n=3, vel=0.55, gate=0.7, strum=0.0, bb=4):
    """chords: one symbol per bar; rhythm: per-bar 16th pattern ('x' hit)."""
    prev = th.voice(chords[0], nm(lo), n)
    steps = rhythm.replace(' ', '')
    st = bb / len(steps)
    for b, c in enumerate(chords):
        v = th.lead(prev, c, nm(lo) - 3, nm(lo) + 14, n)
        prev = v
        for k, ch in enumerate(steps):
            if ch in 'xX':
                p.chord(t0 + b * bb + k * st, v, st * gate * (2 if ch == 'X' else 1), vel * (1.15 if ch == 'X' else 1), strum=strum)


def kit(S, t0, kick, snare, hat, step=0.25, lvl=0, room='room', hat_inst='perc.shaker_egg'):
    k = S.part(f'kick@{t0}', 'drum.kick_modern', gain_db=-4 + lvl, sends={room: 0.08})
    s = S.part(f'snare@{t0}', 'drum.snare', gain_db=-8 + lvl, sends={room: 0.18})
    h = S.part(f'hat@{t0}', hat_inst, pan=0.3, gain_db=-14 + lvl, sends={room: 0.1})
    hits(k, kick, step, t0)
    hits(s, snare, step, t0)
    hits(h, hat, step, t0, vel=0.55)


# ------------------------------------------------------------------ the cues (t0 = 0, 4/4)
def interlude(S):
    """the stage groove between microgames: C - Am | F - G7 (the G7 points at any next cue)."""
    bass = S.part('bass', 'bass.fingered', gain_db=-4, sends={'room': 0.08})
    ep = S.part('ep', 'keys.epiano_tine', pan=-0.25, gain_db=-9, sends={'studio': 0.25})
    clap = S.part('clap', 'perc.hand_claps', pan=0.15, gain_db=-11, sends={'studio': 0.25})
    seq(bass, 'C2 . C3 . . C2 A1 . | A1 . A2 . E2 . G2 . | F1 . F2 . . F1 C2 . | G1 . G2 . B1 . D2 F2', 0.25, vel=0.8)
    comp(ep, ['C', 'Am', 'F', 'G7'], '..x...x.', lo='E4', n=4, bb=2)
    kit(S, 0, 'X...x...X..x..x.|X...x...X.x.x..x', '....x.......x...|....x.......x.xx', 'x.x.x.x.x.x.x.x.|x.x.x.x.x.x.x.x.')
    hits(clap, '....x.......x...|....x.......x...', 0.25)
    return 2


def mg_chip(S):
    lead = S.part('lead', 'synth.chip_lead', gain_db=-6, sends={'room': 0.1, 'echo': 0.1})
    arp = S.part('arp', 'synth.chip_lead', pan=0.35, gain_db=-17, sends={'room': 0.1})
    bass = S.part('bass', 'synth.chip_triangle_bass', gain_db=-4)
    seq(lead, 'E5 G5 C6 - B5 G5 E5 G5 | F5 A5 D6 - C6 B5 G5 B5', 0.5)
    seq(arp, 'C5 E5 G5 C6 C5 E5 G5 C6 C5 E5 G5 C6 C5 E5 G5 C6 | D5 F5 A5 D6 D5 F5 A5 D6 B4 D5 G5 B5 B4 D5 G5 B5', 0.25, vel=0.5)
    seq(bass, 'C2 C3 C2 C3 C2 C3 C2 C3 | D2 D3 D2 D3 G2 G3 G2 B2', 0.5, vel=0.85, gate=0.6)
    kit(S, 0, 'X.......x.x.....|X.......x.x...x.', '....x.......x...|....x.......x.xx', 'x.x.x.x.x.x.x.x.|x.x.x.x.x.x.x.x.', lvl=-2)
    return 2


def mg_funk(S):
    bass = S.part('bass', 'bass.picked', gain_db=-7, sends={'room': 0.06})
    clav = S.part('clav', 'keys.clavinet', pan=-0.3, gain_db=-14, sends={'room': 0.1})
    tpt = S.part('tpt', 'brass.trumpet', pan=0.1, gain_db=4, sends={'studio': 0.2})
    tbn = S.part('tbn', 'brass.trombone', pan=-0.2, gain_db=-3, sends={'studio': 0.2})
    tamb = S.part('tamb', 'perc.tambourine', pan=0.45, gain_db=-14, sends={'room': 0.1})
    seq(bass, 'C2 . . C3 . . Bb1 . C2 . . G2 . Bb2 C3 . | F2 . . F3 . . Eb2 . F2 . . C3 . Eb3 F3 .', 0.25, vel=0.85, gate=0.7)
    comp(clav, ['C7', 'F7'], '.x.x.xx..x.x.x.x', lo='E4', n=3, gate=0.4, vel=0.5)
    seq(tpt, 'C5 . . Eb5 . F5 . G5 | . Bb5 . G5 F5 . Eb5 C5', 0.5, gate=0.6)
    seq(tbn, 'G4 . . Bb4 . C5 . Eb5 | . F5 . Eb5 C5 . Bb4 G4', 0.5, gate=0.6, vel=0.65)
    hits(tamb, '..x...x...x...x.|..x...x...x...x.', 0.25)
    kit(S, 0, 'X..x..x...X..x..|X..x..x...X...x.', '....X..g.g..X..g|....X..g.g..X.gX', 'x.x.x.x.x.x.x.x.|x.x.x.x.x.x.x.x.', hat_inst='perc.cabasa')
    return 2


def mg_mariachi(S):
    t1 = S.part('t1', 'brass.trumpet', pan=-0.2, gain_db=-4, sends={'hall': 0.2})
    t2 = S.part('t2', 'brass.trumpet', pan=0.2, gain_db=-6, sends={'hall': 0.2})
    vln = S.part('vln', 'bowed.violin', pan=0.4, gain_db=-9, sends={'hall': 0.25})
    gtr = S.part('gtr', 'guitar.nylon_spanish', pan=-0.4, gain_db=-7, sends={'room': 0.15})
    vih = S.part('vih', 'pluck.charango', pan=0.5, gain_db=-11, sends={'room': 0.15})
    gtn = S.part('gtn', 'bass.upright_pizz', gain_db=-3, sends={'room': 0.1})
    seq(t1, 'G5 - E5 F5 G5 - C6 - | B5 - G5 A5 B5 - D6 -', 0.5, gate=0.9)
    seq(t2, 'E5 - C5 D5 E5 - E5 - | G5 - E5 F5 G5 - B5 -', 0.5, gate=0.9, vel=0.65)
    seq(vln, 'C6 . . . C6 . . . | D6 . . . D6 . G6 .', 0.5, vel=0.55)
    seq(gtn, 'C2 . G2 . C2 . G2 . | G1 . D2 . G1 . B1 D2', 0.5, vel=0.85, gate=0.6)
    comp(gtr, ['C', 'G7'], '..x...x...x...x.', lo='E3', n=4, strum=0.012, gate=0.5)
    comp(vih, ['C', 'G7'], '..x...x...x...x.', lo='C5', n=3, strum=0.008, gate=0.4, vel=0.45)
    return 2


def mg_surf(S):
    g = S.part('gtr', 'guitar.electric_clean', pan=-0.15, gain_db=-1, sends={'canopy': 0.3, 'echo': 0.15})
    org = S.part('org', 'organ.farfisa', pan=0.35, gain_db=-14, sends={'room': 0.15})
    bass = S.part('bass', 'bass.picked', gain_db=-7)
    tom = S.part('tom', 'drum.tom', gain_db=-9, sends={'room': 0.2})
    mel = 'A4 - - - C5 - - - E5 - - - D5 - C5 - | B4 - - - D5 - - - G5 - - - E5 - D5 -'
    toks = mel.replace('|', ' ').split()
    cur = None
    for k, x in enumerate(toks):          # surf tremolo picking: every 16th re-picked
        cur = x if x != '-' else cur
        g.note(k * 0.25, nm(cur), 0.24, 0.8 if x != '-' else 0.55)
    seq(bass, 'A1 A1 A2 A1 E2 E2 A2 E2 | G1 G1 G2 G1 D2 D2 G2 D2', 0.5, vel=0.8, gate=0.6)
    comp(org, ['Am', 'G'], 'x.......x.......', lo='A4', n=3, gate=1.8, vel=0.45)
    hits(tom, '..x...x...x..xx.|..x...x...x.xxxx', 0.25, pitch=60)
    kit(S, 0, 'X.x...x.X.x...x.|X.x...x.X.x.....', '....x.......x...|....x.......x...', 'x.x.x.x.x.x.x.x.|x.x.x.x.x.x.x.x.', hat_inst='perc.cymbal_ride')
    return 2


def mg_jungle(S):
    mar = S.part('mar', 'mallet.marimba_rosewood', pan=-0.2, gain_db=-3, sends={'canopy': 0.25})
    bal = S.part('bal', 'mallet.balafon', pan=0.3, gain_db=-6, sends={'canopy': 0.2})
    kal = S.part('kal', 'mallet.kalimba', pan=0.45, gain_db=-12, sends={'canopy': 0.3, 'echo': 0.15})
    co = S.part('co', 'drum.conga_open', pan=-0.4, gain_db=-7, sends={'canopy': 0.15})
    cs = S.part('cs', 'drum.conga_slap', pan=-0.4, gain_db=-9, sends={'canopy': 0.15})
    cx = S.part('cx', 'perc.caxixi', pan=0.5, gain_db=-13, sends={'canopy': 0.1})
    tc = S.part('tc', 'creature.toucan_clack', pan=0.6, gain_db=-8, sends={'canopy': 0.2})
    seq(mar, 'C5 E5 G5 E5 A5 G5 E5 D5 | F5 A5 C6 A5 G5 F5 D5 B4', 0.5)
    seq(bal, 'C3 . G3 . A3 . E3 . | F3 . C4 . G3 . D3 .', 0.5, vel=0.75)
    seq(kal, 'G6 . . . E6 . . . | A6 . . . B6 . . .', 0.5, vel=0.5)
    hits(co, 'x..x..x...x..x..|x..x..x...x.x...', 0.25)
    hits(cs, '....x.......x...|....x.......x..x', 0.25)
    hits(cx, 'x.xxx.xxx.xxx.xx|x.xxx.xxx.xxx.xx', 0.25, vel=0.5)
    hits(tc, '......x.......x.|......x...x...x.', 0.25)
    return 2


def mg_space(S):
    lead = S.part('lead', 'synth.moog_lead', gain_db=-5, sends={'hall': 0.25, 'echo': 0.2})
    pad = S.part('pad', 'synth.supersaw_pad', gain_db=-16, sends={'hall': 0.3})
    blip = S.part('blip', 'synth.pulsar_blip', pan=0.35, gain_db=-15, sends={'hall': 0.2})
    sub = S.part('sub', 'synth.sub_bass', gain_db=-3)
    clap = S.part('clap', 'perc.hand_claps', gain_db=-10, sends={'hall': 0.25})
    seq(lead, 'A4 - E5 - A5 G5 E5 - | F5 - C5 - G5 F5 E5 -', 0.5)
    comp(pad, ['Am', 'F'], 'x...............', lo='A3', n=4, gate=3.8, vel=0.45)
    seq(blip, 'A5 C6 E6 A6 A5 C6 E6 A6 A5 C6 E6 A6 A5 C6 E6 A6 | F5 A5 C6 F6 F5 A5 C6 F6 G5 B5 D6 G6 G5 B5 D6 G6', 0.25, vel=0.45)
    seq(sub, 'A1 . A2 . A1 . A2 . | F1 . F2 . G1 . G2 .', 0.5, vel=0.85, gate=0.6)
    hits(clap, '....x.......x...|....x.......x...', 0.25)
    kit(S, 0, 'X...x...X...x...|X...x...X...x.x.', '................|..............xx', '..x...x...x...x.|..x...x...x...x.', room='hall')
    return 2


def mg_toybox(S):
    mb = S.part('mb', 'keys.music_box', gain_db=-4, sends={'room': 0.25})
    gl = S.part('gl', 'bell.glockenspiel', pan=0.35, gain_db=-11, sends={'room': 0.25})
    xy = S.part('xy', 'mallet.xylophone', pan=-0.3, gain_db=-7, sends={'room': 0.15})
    cel = S.part('cel', 'keys.celesta', pan=-0.4, gain_db=-12, sends={'room': 0.2})
    wb = S.part('wb', 'perc.woodblock', pan=0.4, gain_db=-12, sends={'room': 0.1})
    tri = S.part('tri', 'perc.triangle', pan=0.5, gain_db=-16, sends={'room': 0.2})
    seq(mb, 'C6 E6 G6 E6 C6 - G5 - | D6 F6 A6 F6 B5 - G5 -', 0.5)
    seq(gl, 'C7 . . . G6 . . . | F6 . . . G6 . . .', 0.5, vel=0.5)
    seq(xy, 'C5 . G4 . C5 . G4 . | D5 . A4 . G4 . B4 .', 0.5, vel=0.7, gate=0.4)
    comp(cel, ['C', 'G7'], '..x...x...x...x.', lo='E5', n=3, gate=0.4, vel=0.4)
    hits(wb, 'x.x.x.x.x.x.x.x.|x.x.x.x.x.x.x.x.', 0.25, vel=0.55)
    hits(tri, 'x...............|x...............', 0.25)
    return 2


def mg_polka(S):
    cl = S.part('cl', 'reed.clarinet', pan=0.15, gain_db=-4, sends={'room': 0.2})
    acc = S.part('acc', 'organ.accordion_musette', pan=-0.3, gain_db=-9, sends={'room': 0.15})
    tuba = S.part('tuba', 'brass.tuba', gain_db=-3, sends={'room': 0.1})
    seq(cl, 'E5 F5 G5 E5 C5 E5 G5 C6 | B5 A5 G5 F5 E5 D5 C5 -', 0.5)
    seq(tuba, 'C2 . G1 . C2 . G1 . | G1 . D2 . C2 . G1 .', 0.5, vel=0.85, gate=0.55)
    comp(acc, ['C', 'G7'], '..x...x...x...x.', lo='E4', n=3, gate=0.45, vel=0.55)
    kit(S, 0, 'x.......x.......|x.......x.......', '..x...x...x...x.|..x...x...x.x.xx', '................|................', lvl=-2)
    return 2


def boss(S):
    org = S.part('org', 'organ.drawbar_fast', pan=-0.2, gain_db=-7, sends={'church': 0.25})
    gtr = S.part('gtr', 'guitar.crunch_rhythm', pan=0.3, gain_db=-10, sends={'room': 0.1})
    horn = S.part('horn', 'brass.french_horn', gain_db=-3, sends={'church': 0.3})
    ch = S.part('ch', 'voice.choir_aah', gain_db=-10, sends={'church': 0.35})
    tim = S.part('tim', 'drum.timpani', gain_db=-5, sends={'church': 0.2})
    bass = S.part('bass', 'bass.picked', gain_db=-5)
    prog = ['Am', 'F', 'G', 'E']
    for b, c in enumerate(prog):
        r = th.bass_note(c, nm('E2'), nm('D#3'))
        seq(org, ' '.join(['A3', 'A3', 'C4', 'A3', 'D4', 'A3', 'E4', 'D4']), 0.5, t0=b * 4, transpose=r - nm('A2'), vel=0.65, gate=0.6)
        for k in range(8):
            gtr.chord(b * 4 + k * 0.5, [r, r + 7, r + 12], 0.4, 0.6 if k % 2 else 0.75)
        bass.note(b * 4, r - 12, 1.8, 0.85)
        bass.note(b * 4 + 2, r - 12, 1.8, 0.75)
        for x in th.voice(c, nm('A3'), 3):
            ch.note(b * 4, x, 3.9, 0.6)
        tim.note(b * 4, r - 12 if r - 12 >= nm('D2') else r, 0.9, 0.85)
        tim.note(b * 4 + 2.5, r - 12 if r - 12 >= nm('D2') else r, 0.5, 0.6)
    seq(horn, 'A4 - - - E5 - - - | F5 - - - C5 - D5 - | B4 - - - D5 - - - | E5 - - - G#4 - B4 -', 0.5)
    kit(S, 0, 'X...x.x.X...x.x.|X...x.x.X...x.x.|X...x.x.X...x.x.|X...x.x.X.x.x.xx',
        '....x.......x...|....x.......x...|....x.......x...|....x...x.x.xxxx', 'x.x.x.x.x.x.x.x.|' * 3 + 'x.x.x.x.x.x.x.x.', room='hall')
    return 4


# ---- one-shots ------------------------------------------------------------
def win(S):
    t = S.part('t', 'brass.trumpet', gain_db=-2, sends={'hall': 0.25})
    t2 = S.part('t2', 'brass.trumpet', pan=0.25, gain_db=-5, sends={'hall': 0.25})
    gl = S.part('gl', 'bell.glockenspiel', pan=-0.3, gain_db=-8, sends={'hall': 0.3})
    seq(t, 'C5 E5 G5 C6 - - . .', 0.5, gate=0.95)
    seq(t2, 'G4 C5 E5 G5 - - . .', 0.5, gate=0.95, vel=0.65)
    seq(gl, 'C6 E6 G6 C7 . . . .', 0.5, vel=0.6)
    sn = S.part('sn', 'drum.snare', gain_db=-9, sends={'hall': 0.2})
    hits(sn, 'x.x.x.xxX.......', 0.25)
    cr = S.part('cr', 'perc.cymbal_crash', gain_db=-12, sends={'hall': 0.2})
    cr.note(1.5, 60, 2, 0.8)
    return 1


def lose(S):
    t = S.part('t', 'brass.trumpet_muted', gain_db=-2, sends={'room': 0.2})
    tb = S.part('tb', 'brass.tuba', gain_db=-5, sends={'room': 0.15})
    seq(t, 'G4 - F#4 - F4 - E4 - - -', 0.25 * 1.6, gate=0.95)
    tb.note(3, nm('C2'), 0.9, 0.8)
    tom = S.part('tom', 'drum.tom', gain_db=-8, sends={'room': 0.2})
    tom.note(3, 60, 0.6, 0.9)
    return 1


def speedup(S):
    """two bars: a snare roll that crescendos, a chromatic brass climb, and a hit."""
    sn = S.part('sn', 'drum.snare', gain_db=-8, sends={'hall': 0.2})
    for k in range(28):
        sn.note(k * 0.25, 60, 0.25, 0.35 + 0.022 * k)
    br = S.part('br', 'brass.trumpet', gain_db=-3, sends={'hall': 0.25})
    tb = S.part('tb', 'brass.trombone', gain_db=-5, sends={'hall': 0.25})
    for k, x in enumerate(['G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5']):
        br.note(4 + k * 0.4375, nm(x) + 12 if k > 3 else nm(x), 0.35, 0.6 + 0.04 * k)
        tb.note(4 + k * 0.4375, nm(x) - 5, 0.35, 0.55 + 0.04 * k)
    rs = S.part('rs', 'synth.noise_riser', gain_db=-12, sends={'hall': 0.3})
    rs.note(0, 60, 7.6, 0.7)
    hit = S.part('hit', 'perc.cymbal_crash', gain_db=-9, sends={'hall': 0.3})
    hit.note(7.75, 60, 1.5, 0.9)
    for x in ('G4', 'B4', 'D5'):
        br.note(7.75, x, 0.25, 0.9)
    return 2


def boss_intro(S):
    gong = S.part('gong', 'metal.gong_ageng', gain_db=-4, sends={'church': 0.3})
    gong.note(0, nm('A2'), 4, 0.9)
    tim = S.part('tim', 'drum.timpani', gain_db=-5, sends={'church': 0.25})
    tim.note(4, nm('E2'), 3.8, 0.75)          # long gate -> roll
    org = S.part('org', 'organ.pipe_plenum', gain_db=-9, sends={'church': 0.35})
    for x in th.voice('Am', nm('A3'), 4):
        org.note(0.5, x, 7, 0.55)
    org.level(0.5, -12).level(7.5, 0)
    ch = S.part('ch', 'voice.choir_aah', gain_db=-10, sends={'church': 0.4})
    for x in ('E4', 'G#4', 'B4'):
        ch.note(4, x, 3.8, 0.6)
    return 2


def boss_win(S):
    t = S.part('t', 'brass.trumpet', gain_db=-2, sends={'hall': 0.3})
    hn = S.part('hn', 'brass.french_horn', gain_db=-5, sends={'hall': 0.3})
    seq(t, 'G4 C5 E5 G5 - E5 G5 - | C6 - - - - - - -', 0.5)
    for x in ('C4', 'E4', 'G4'):
        hn.note(4, x, 3.5, 0.75)
    tim = S.part('tim', 'drum.timpani', gain_db=-5, sends={'hall': 0.25})
    for t0 in (0, 1.5, 3, 3.5, 4):
        tim.note(t0, nm('C3') if t0 < 4 else nm('C2'), 0.5, 0.8)
    cr = S.part('cr', 'perc.cymbal_crash', gain_db=-9, sends={'hall': 0.2})
    cr.note(4, 60, 3, 0.9)
    gl = S.part('gl', 'bell.glockenspiel', gain_db=-9, sends={'hall': 0.3})
    seq(gl, 'C6 E6 G6 C7 . . . . | E7 . . . . . . .', 0.5, vel=0.6)
    return 2


def gameover(S):
    mb = S.part('mb', 'keys.music_box', gain_db=-3, sends={'hall': 0.35})
    S.tempo_at(4, S.clock.bpm(0) * 0.8)       # winds down like a music box
    seq(mb, 'C6 Bb5 Ab5 G5 | F5 Eb5 D5 C5', 1.0, gate=0.9)
    tb = S.part('tb', 'brass.trombone', gain_db=-8, sends={'hall': 0.3})
    tb.note(4, nm('C3'), 3.5, 0.6)
    return 2


def clear(S):
    win(S)
    t = S.part('t3', 'brass.trumpet', gain_db=-3, sends={'hall': 0.3})
    seq(t, '. . . . . . . . | G5 A5 B5 C6 D6 E6 - -', 0.5, t0=0)
    pi = S.part('pi', 'keys.epiano_tine', gain_db=-10, sends={'hall': 0.3})
    for x in th.voice('Cmaj9', nm('C4'), 5):
        pi.note(4, x, 3.5, 0.55)
    return 2


LOOPS = dict(interlude=interlude, boss=boss, **{f'mg_{s}': globals()[f'mg_{s}'] for s in STYLES})
ONESHOTS = dict(win=win, lose=lose, speedup=speedup, boss_intro=boss_intro, boss_win=boss_win, boss_lose=lose,
                gameover=gameover, clear=clear)


# ------------------------------------------------------------------ rendering
def _session(bpm):
    s = Session(bpm=bpm)
    s.echo.update(beats=0.75, feedback=0.3, damp=3500.0)
    return s


def render_loop(fn, bpm, tail_s=4.0):
    s = _session(bpm)
    bars = fn(s)
    N = int(round(bars * 4 * SR * 60 / bpm))
    T = int(tail_s * SR)
    P = int(0.2 * SR)
    b = s.render_parts(t_end=(N + T) / SR + 1.0, t_offset=-P / SR)
    full = s.wet(b, b['n'])
    body = full[:, P:]
    loop = body[:, :N].copy()
    k = N
    while k < body.shape[1]:
        seg = body[:, k:k + N]
        loop[:, :seg.shape[1]] += seg
        k += N
    loop[:, N - P:] += full[:, :P]
    tail = body[:, N:N + T]
    return bars, loop, tail


def render_oneshot(fn, bpm, tail_s=3.0):
    s = _session(bpm)
    bars = fn(s)
    N = int(round(s.clock.sec(bars * 4) * SR))
    b = s.render_parts(t_end=N / SR + tail_s)
    x = s.wet(b, b['n'])
    return bars, x, N


def loudness(x):
    import pyloudnorm as pyln
    return pyln.Meter(SR).integrated_loudness(x.T.astype(np.float64))


def build(out_dir, web=False):
    os.makedirs(out_dir, exist_ok=True)
    pack = dict(name='Microgame Rush', sr=SR, tiers=TIERS, styles=STYLES, style_names=STYLE_NAMES, key='C major / A minor',
                grammar=dict(round=['interlude', 'microgame', 'win|lose'], speedup_every=4, boss_at=12,
                             boss=['boss_intro', 'boss', 'boss_win|boss_lose', 'clear'], lives=4),
                cues={})
    sprites = []
    for ti, bpm in enumerate(TIERS):
        cues = {}
        rendered = []
        for name, fn in LOOPS.items():
            bars, loop, tail = render_loop(fn, bpm)
            rendered.append((name, 'loop', bars, loop, tail))
        for name, fn in ONESHOTS.items():
            bars, x, N = render_oneshot(fn, bpm)
            rendered.append((name, 'oneshot', bars, x, N))
        # loudness: loops sit at -18 LUFS, jingles a little hotter so they cut through
        for name, kind, bars, a, b in rendered:
            target = -18.0 if kind == 'loop' else -16.0
            ref = np.concatenate([a, a], axis=1) if kind == 'loop' else a
            g = 10 ** ((target - loudness(ref)) / 20)
            a = a * g
            b = b * g if kind == 'loop' else b
            pk = max(np.max(np.abs(a)), np.max(np.abs(b)) if kind == 'loop' else 0)
            if pk > 0.95:
                a, b = (a * 0.95 / pk, b * 0.95 / pk) if kind == 'loop' else (a * 0.95 / pk, b)
            d = os.path.join(out_dir, f'tier{ti}')
            os.makedirs(d, exist_ok=True)
            sf.write(os.path.join(d, f'{name}.ogg'), np.clip(a, -1, 1).T, SR, format='OGG', subtype='VORBIS')
            if kind == 'loop':
                sf.write(os.path.join(d, f'{name}_tail.ogg'), np.clip(b, -1, 1).T, SR, format='OGG', subtype='VORBIS')
            body = a.shape[1] if kind == 'loop' else b
            cues[name] = dict(kind=kind, bars=bars, samples=int(body), seconds=round(body / SR, 4))
            sprites.append((ti, name, kind, a, b if kind == 'loop' else None, body))
        pack['cues'][f'tier{ti}'] = dict(bpm=bpm, cues=cues)
        print(f'tier {ti} ({bpm} BPM): {len(cues)} cues', flush=True)
    json.dump(pack, open(os.path.join(out_dir, 'cues.json'), 'w'), indent=1)
    if web:
        export_web(pack, sprites)
    return pack


def export_web(pack, sprites):
    """one sprite per tier: [sync impulse header][cue][gap][tail][gap]... with offsets in JSON."""
    here = os.path.dirname(os.path.abspath(__file__))
    pdir = os.path.abspath(os.path.join(here, '..', '..', 'player', 'microgame'))
    os.makedirs(pdir, exist_ok=True)
    gap = int(0.25 * SR)
    web = dict(pack, tiers=[])
    for ti, bpm in enumerate(TIERS):
        head = np.zeros((2, 8192), np.float32)
        head[:, 1000] = 0.9                      # sync impulse: found after decode, so codec padding can't shift cues
        chunks, pos, offs = [head], 8192, {}
        for (t, name, kind, a, tail, body) in sprites:
            if t != ti:
                continue
            offs[name] = dict(kind=kind, start=pos, length=int(body), total=int(a.shape[1]))
            chunks.append(a.astype(np.float32))
            pos += a.shape[1]
            if tail is not None:
                chunks.append(np.zeros((2, gap), np.float32))
                pos += gap
                offs[name]['tail'] = pos
                offs[name]['tail_len'] = int(tail.shape[1])
                chunks.append(tail.astype(np.float32))
                pos += tail.shape[1]
            chunks.append(np.zeros((2, gap), np.float32))
            pos += gap
        sp = np.concatenate(chunks, axis=1)
        wav = os.path.join(pdir, f'_tier{ti}.wav')
        sf.write(wav, np.clip(sp, -1, 1).T, SR, subtype='PCM_16')
        m4a = os.path.join(pdir, f'tier{ti}.m4a')
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', wav, '-c:a', 'aac', '-b:a', '128k', m4a], check=True)
        os.remove(wav)
        b64 = base64.b64encode(open(m4a, 'rb').read()).decode('ascii')
        open(os.path.join(pdir, f'tier{ti}.b64.txt'), 'w').write(b64)
        web['tiers'].append(dict(bpm=bpm, file=f'microgame/tier{ti}.m4a', b64=f'microgame/tier{ti}.b64.txt',
                                 sync=1000, cues=offs, beat=60 / bpm))
        print(f'web tier {ti}: {sp.shape[1] / SR:.1f} s sprite, {len(b64) // 1024} KB base64')
    json.dump(web, open(os.path.join(pdir, 'microgame.json'), 'w'), separators=(',', ':'))


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'out', 'microgame')
    build(os.path.abspath(out), web='--web' in sys.argv)
