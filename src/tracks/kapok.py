"""
KAPOK — an original multi-instrumental suite in the manner of Mike Oldfield
(Tubular Bells, Ommadawn, Amarok), played entirely on Canopy Orchestra's
synthesized instruments. No samples, no recordings, no quoted melodies.

The kapok is the tallest tree in the rainforest. The piece climbs it:

  I    Forest Floor        15/8 ostinato (3+2+2 | 3+3+2) that builds layer by layer
  II   Creature Chorus     4/4 hocket: the tune hops between dozens of instruments
                           while frogs, toucans, crickets and gibbons become the band
  III  Rain on the Leaves  3/4 pastoral: guitars, mandolin tremolo, whistles, strings
  IV   Drums of the Kapok  12/8 Ommadawn-style drum ensemble, chant, uilleann pipes
  V    The Procession      instruments introduced one by one ... plus tubular bells;
                           then the canopy opens into the stars (the space half)
  ---  The Kapok Census   Amarok-style roll call: every instrument not featured
                           elsewhere gets a one-beat cameo, so the whole library plays
  VI   Hornbill Hornpipe   an accelerating reel to finish, like Oldfield's codas

    cd src && python3 -m tracks.kapok [--out ../out/kapok] [--sections I,II] [--mc]

Writes kapok.wav (mastered), kapok_credits.md (every instrument, in order of
appearance) and, with --mc, an 'introductions' cue list for an announcer.
"""
from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import orchestra  # noqa: E402
from orchestra import REGISTRY, nm  # noqa: E402
from orchestra.mix import Session, Meter  # noqa: E402
from orchestra import theory as th  # noqa: E402

SUBS = []


def R(*cands, family=None, need=None):
    """resolve a role to a registered preset: exact names first, then substring
    matches (in the order given). Records substitutions for the log."""
    for c in cands:
        if c in REGISTRY:
            return c
    for c in cands:
        hits = sorted(k for k in REGISTRY if c in k and (family is None or REGISTRY[k].family == family))
        if hits:
            return hits[0]
    if need:
        hits = sorted(k for k in REGISTRY if REGISTRY[k].family in need)
        if hits:
            SUBS.append((cands[0], hits[0]))
            return hits[0]
    raise KeyError(f'no instrument for role {cands}')


def has(*cands):
    try:
        R(*cands)
        return True
    except KeyError:
        return False


class Suite:
    """wraps a Session with a beat cursor, section bookkeeping and credits."""

    def __init__(self):
        self.s = Session(bpm=112, master=dict(comp=dict(thr_db=-22, ratio=2.2, attack=0.03, release=0.35),
                                              lufs=-14.0, ceiling=-1.0, tape=1.15,
                                              eq=[(4500, 3.0, 'high'), (250, -1.5, 'low')]))
        self.s.echo.update(beats=0.75, feedback=0.38, damp=3000.0)
        self.cur = 0.0
        self.order = OrderedDict()      # instrument -> first appearance (seconds)
        self.markers = []               # (seconds, label)
        self.intros = []                # (seconds, text) announcer cues
        self._n = 0

    def part(self, inst, pan=0.0, gain_db=0.0, sends=None, **kw):
        self._n += 1
        return self.s.part(f'{inst}#{self._n}', inst, pan=pan, gain_db=gain_db, sends=sends or {'studio': 0.18}, **kw)

    def section(self, label, bpm):
        self.s.tempo_at(self.cur, bpm)
        self.markers.append((self.s.clock.sec(self.cur), label))

    def credits(self):
        first = {}
        for p in self.s.parts.values():
            for n in p.notes:
                t = self.s.clock.sec(n.beat)
                if n.inst not in first or t < first[n.inst]:
                    first[n.inst] = t
        return sorted(first.items(), key=lambda kv: kv[1])


# ============================================================ I. FOREST FLOOR
def forest_floor(S: Suite, bars=24):
    S.section('I. Forest Floor', 112)
    m = Meter([15] * bars, unit=8, start=S.cur)
    prog = ['Dm9', 'Dm9', 'C6/9', 'C6/9', 'Bbmaj9', 'Bbmaj9', 'A7sus4', 'A']
    # the cell: Canopy's motif (A-D-E-A) folded into 3+2+2 | 3+3+2
    cell = ['A4', 'D5', 'E5', 'A5', 'E5', 'D5', 'E5', 'F5', 'E5', 'D5', 'C5', 'D5', 'A4', 'G4', 'A4']
    accents = {0, 3, 5, 7, 10, 13}
    kal = S.part(R('mallet.kalimba', 'kalimba'), pan=-0.25, sends={'canopy': 0.22, 'echo': 0.1}, eq=[(2500, 6.0, 'high')])
    bal = S.part(R('mallet.balafon', 'balafon', 'gyil', 'marimba'), pan=0.3, gain_db=-2, sends={'canopy': 0.2}, eq=[(2000, 4.0, 'high')])
    bmar = S.part(R('mallet.marimba_bass', 'bass_marimba', 'marimba'), pan=0.0, gain_db=-1, sends={'canopy': 0.15})
    glock = S.part(R('mallet.glockenspiel', 'glock'), pan=0.45, gain_db=-5, sends={'hall': 0.3})
    hpan = S.part(R('mallet.handpan', 'handpan', 'hang'), pan=-0.5, gain_db=-5, sends={'canopy': 0.35})
    frogs = S.part(R('creature.treefrog_choir', 'treefrog'), pan=0.6, gain_db=-10, sends={'canopy': 0.3})
    cic = S.part(R('creature.cicada', 'cicada'), pan=-0.7, gain_db=-20, sends={'valley': 0.4})
    shak = S.part(R('perc.caxixi', 'caxixi', 'shaker'), pan=0.35, gain_db=-5, sends={'canopy': 0.15})
    flute = S.part(R('flute.bamboo', 'bamboo', 'flute'), pan=-0.1, gain_db=2, sends={'canopy': 0.35, 'echo': 0.15})
    pflute = S.part(R('flute.pan_flute', 'pan_flute', 'pan'), pan=0.35, gain_db=-4, sends={'canopy': 0.4, 'echo': 0.2})
    strings = S.part(R('bowed.section_warm', 'section', 'strings'), pan=0.0, gain_db=-9, sends={'hall': 0.35})
    mbira = S.part(R('mallet.mbira', 'mbira', 'kalimba'), pan=0.6, gain_db=-5, sends={'canopy': 0.2})
    angk = S.part(R('mallet.angklung', 'angklung'), pan=-0.6, gain_db=-4, sends={'canopy': 0.25})
    rain = S.part(R('perc.rainstick', 'rainstick'), pan=0.0, gain_db=-8, sends={'valley': 0.3})

    for b in range(bars):
        ch = prog[b % 8]
        root = th.bass_note(ch, nm('A1'), nm('G#2'))
        c = list(cell)
        if ch == 'A':
            c = [x.replace('F5', 'E5').replace('C5', 'C#5') for x in c]
        dyn = min(1.0, 0.55 + 0.02 * b)
        # kalimba plays the cell throughout (dropping out in the last bar for the rain)
        if b < bars - 1:
            for k, x in enumerate(c):
                kal.note(m.bar(b, k), x, 0.5, dyn * (0.85 if k in accents else 0.62))
        if 4 <= b < bars - 1:   # balafon: left hand on the group accents, chord tones
            tones = th.lead([nm('D3'), nm('A3'), nm('D4')], ch, nm('F2'), nm('E4'), 3)
            lh = [tones[0], tones[2], tones[1], tones[0], tones[2], tones[1]]
            for k, e in enumerate(sorted(accents)):
                bal.note(m.bar(b, e), lh[k], 1.0, 0.7 if e in (0, 7) else 0.55)
        if 8 <= b:
            bmar.note(m.bar(b, 0), root, 3.3, 0.8)
            bmar.note(m.bar(b, 7), root + 7 if ch not in ('A7sus4', 'A') else root + 12, 3.8, 0.62)
            for e in sorted(accents):
                frogs.note(m.bar(b, e), th.snap(root + 36, th.scale_notes('D', 'dorian', 60, 96)), 0.4,
                           0.45 + 0.2 * (e == 0))
            for k in range(15):
                shak.note(m.bar(b, k), 60, 0.25, 0.5 if k in accents else 0.28)
        if b % 4 == 0 and b >= 4:
            cic.note(m.bar(b), th.bass_note(ch, nm('A5'), nm('G#6')), 4 * 7.5 * 0.95, 0.5)
        if 12 <= b < bars - 1:
            for k in sorted(accents):
                if k in (0, 7, 10):
                    glock.note(m.bar(b, k), nm(c[k]) + 12, 0.75, 0.55)
            v = th.lead([nm('F4'), nm('A4')], ch, nm('C4'), nm('C5'), 2)
            hpan.note(m.bar(b, 0), v[-1], 3.5, 0.6)
            hpan.note(m.bar(b, 7), v[0], 3.5, 0.5)
        if 16 <= b and b % 2 == 0:
            for mm in th.lead([nm('D4'), nm('F4'), nm('A4'), nm('C5')], ch, nm('A3'), nm('E5'), 4):
                strings.note(m.bar(b), mm, 15.0 * 0.5 * 2 * 0.98, 0.45 + 0.01 * b)
        if 20 <= b < bars - 1:
            for k in range(0, 15, 2):
                mbira.note(m.bar(b, k + 1), nm(c[k]) + 12 if k % 4 == 0 else nm(c[k]), 0.4, 0.45)
        if 16 <= b < bars - 1 and b % 2 == 1:
            angk.note(m.bar(b, 0), th.bass_note(ch, nm('D5'), nm('C#6')), 3.5, 0.5)
    # bamboo flute: an augmented, long-breathed statement of the motif (bars 16-23)
    mel = [(16, 0, 7, 'A4'), (16, 7, 8, 'D5'), (17, 0, 10, 'E5'), (17, 10, 5, 'D5'), (18, 0, 7, 'C5'), (18, 7, 8, 'A4'),
           (19, 0, 15, 'G4'), (20, 0, 7, 'F5'), (20, 7, 3, 'E5'), (20, 10, 5, 'D5'), (21, 0, 10, 'C5'), (21, 10, 5, 'A4'),
           (22, 0, 7, 'D5'), (22, 7, 8, 'E5'), (23, 0, 13, 'C#5')]
    for bar, e, d, x in mel:
        if bar < bars:
            flute.note(m.bar(bar, e), x, d * 0.5 * 0.97, 0.65)
            if bar >= 20:   # pan flute answers an octave up, a beat behind
                pflute.note(m.bar(bar, e) + 1.5, nm(x) + 12, d * 0.5 * 0.8, 0.42)
    rain.note(m.bar(bars - 1), 60, 7.0, 0.7)
    kal.level(m.bar(15), 0).level(m.bar(16), -3)
    S.cur = m.end


# ============================================================ II. CREATURE CHORUS
HOCKET_FAMILIES = ('mallet', 'bell', 'pluck', 'guitar', 'keys', 'organ', 'flute', 'reed', 'brass', 'bowed',
                   'voice', 'synth', 'metal')


def creature_chorus(S: Suite, cycles=3, used=None):
    S.section('II. Creature Chorus', 112)
    bb = 4
    prog = ['Dmaj7', 'E/D', 'Bm7', 'Gmaj7#11', 'D/F#', 'Em9', 'Gmaj9', 'A7sus4']
    # (bar, eighth, length in eighths, note)
    tune = [(0, 0, 1, 'F#5'), (0, 1, 2, 'A5'), (0, 3, 1, 'E5'), (0, 4, 2, 'F#5'), (0, 6, 2, 'D5'),
            (1, 0, 2, 'G#5'), (1, 2, 2, 'B5'), (1, 4, 1, 'A5'), (1, 5, 1, 'G#5'), (1, 6, 2, 'E5'),
            (2, 0, 2, 'F#5'), (2, 2, 1, 'D5'), (2, 3, 1, 'F#5'), (2, 4, 2, 'A5'), (2, 6, 2, 'B5'),
            (3, 0, 3, 'C#6'), (3, 3, 1, 'B5'), (3, 4, 2, 'A5'), (3, 6, 2, 'F#5'),
            (4, 0, 2, 'A5'), (4, 2, 2, 'F#5'), (4, 4, 1, 'E5'), (4, 5, 2, 'D5'), (4, 7, 1, 'A4'),
            (5, 0, 1, 'B4'), (5, 1, 1, 'D5'), (5, 2, 2, 'F#5'), (5, 4, 2, 'E5'), (5, 6, 2, 'D5'),
            (6, 0, 2, 'B4'), (6, 2, 2, 'D5'), (6, 4, 1, 'F#5'), (6, 5, 2, 'A5'), (6, 7, 1, 'B5'),
            (7, 0, 4, 'C#6'), (7, 4, 4, 'E5')]
    # hocket pool: every pitched instrument that can reach the tune, not featured elsewhere first
    used = used or set()
    pool = [k for k in sorted(REGISTRY, key=lambda k: (HOCKET_FAMILIES.index(REGISTRY[k].family)
                                                        if REGISTRY[k].family in HOCKET_FAMILIES else 99, k))
            if REGISTRY[k].pitched and REGISTRY[k].family in HOCKET_FAMILIES
            and REGISTRY[k].lo <= nm('A4') and REGISTRY[k].hi >= nm('C#6')]
    fresh = [k for k in pool if k not in used] + [k for k in pool if k in used]
    hparts = {}
    pans = np.linspace(-0.75, 0.75, 9)

    def hpart(inst):
        if inst not in hparts:
            fam = REGISTRY[inst].family
            g = -9 if fam in ('organ', 'reed', 'brass', 'bowed', 'voice', 'synth') or REGISTRY[inst].sustain else -3
            hparts[inst] = S.part(inst, pan=float(pans[len(hparts) % 9]), gain_db=g, sends={'canopy': 0.22, 'echo': 0.08})
        return hparts[inst]

    bull = S.part(R('creature.bullfrog', 'bullfrog', 'frog'), gain_db=-2, sends={'canopy': 0.2})
    tfrog = S.part(R('creature.treefrog_choir', 'treefrog'), pan=-0.4, gain_db=-9, sends={'canopy': 0.3})
    touc = S.part(R('creature.toucan_clack', 'toucan'), pan=0.5, gain_db=-6, sends={'canopy': 0.2})
    crick = S.part(R('creature.cricket', 'cricket'), pan=-0.65, gain_db=-16, sends={'canopy': 0.2})
    gib = S.part(R('creature.gibbon', 'gibbon'), pan=0.7, gain_db=-6, sends={'valley': 0.35})
    howl = S.part(R('creature.howler', 'howler', 'monkey'), pan=-0.6, gain_db=-9, sends={'valley': 0.3})
    song = S.part(R('creature.songbird', 'songbird', 'bird'), pan=0.3, gain_db=-8, sends={'canopy': 0.3})
    blocks = S.part(R('perc.temple_blocks', 'temple', 'woodblock'), pan=0.2, gain_db=-12, sends={'canopy': 0.2})
    claves = S.part(R('perc.claves', 'claves'), pan=-0.3, gain_db=-12, sends={'canopy': 0.15})
    bass = S.part(R('bass.fretless', 'fretless', 'bass.fingered', 'bass'), gain_db=-4, sends={'studio': 0.1})
    hi = 0
    nbars = 8 * cycles
    for b in range(nbars):
        t0 = S.cur + b * bb
        ch = prog[b % 8]
        root = th.bass_note(ch, nm('D1'), nm('C#2'))
        # the band of creatures
        for e, v in ((0, 0.9), (3, 0.6), (6, 0.7)):
            bull.note(t0 + e * 0.5, root + 12, 0.5, v)
        bass.note(t0, root + 12, 1.4, 0.7)
        bass.note(t0 + 1.5, root + 19, 0.45, 0.5)
        bass.note(t0 + 2.5, root + 12, 1.2, 0.6)
        for e in (1, 3, 5, 7):
            tones = th.lead([nm('D5'), nm('F#5')], ch, nm('C5'), nm('A5'), 2)
            tfrog.note(t0 + e * 0.5, tones[e // 2 % 2], 0.3, 0.5)
        touc.note(t0 + 1, 60, 0.25, 0.8)
        touc.note(t0 + 3, 60, 0.25, 0.8)
        if b % 2:
            touc.note(t0 + 3.75, 60, 0.25, 0.55)
        for k in range(16):
            if k % 4 != 0:
                crick.note(t0 + k * 0.25, th.bass_note(ch, nm('C7'), nm('B7')) if has('cricket') else 96, 0.2, 0.35)
        for e in (0, 3, 6):
            claves.note(t0 + e * 0.5, 60, 0.25, 0.5)
        if b % 4 == 3:
            gib.note(t0 + 2, th.bass_note(ch, nm('A4'), nm('G#5')), 1.8, 0.7)
        if b % 8 == 7:
            howl.note(t0 + 0.5, root + 24, 3.0, 0.6)
        # the hocket: each note of the tune goes to the next instrument in the tour
        for (bar, e, ln, x) in [n for n in tune if n[0] == b % 8]:
            inst = fresh[hi % len(fresh)]
            hi += 1
            vel = 0.75 if e in (0, 4) else 0.62
            hpart(inst).note(t0 + e * 0.5, x, ln * 0.5 * 0.92, vel)
            blocks.note(t0 + e * 0.5, 60 + 7 * (e % 2), 0.25, 0.35)
            if b >= 16:   # third time round, a tuned songbird doubles the tune an octave up, sparsely
                if e in (0, 4):
                    song.note(t0 + e * 0.5, nm(x) + 12, 0.6, 0.45)
    S.cur += nbars * bb
    return set(hparts)


# ============================================================ III. RAIN ON THE LEAVES
def rain_on_leaves(S: Suite, reps=4):
    S.section('III. Rain on the Leaves', 90)
    bb = 3
    prog = ['Gmaj9', 'D/F#', 'Em7', 'A7sus4', 'Bm7', 'Gmaj7', 'Em9', 'A']
    tune = [(0, 0, 2, 'B5'), (0, 2, 1, 'A5'), (1, 0, 3, 'F#5'), (2, 0, 1, 'G5'), (2, 1, 1, 'A5'), (2, 2, 1, 'B5'),
            (3, 0, 3, 'E5'), (4, 0, 2, 'D6'), (4, 2, 1, 'C#6'), (5, 0, 2, 'B5'), (5, 2, 1, 'A5'),
            (6, 0, 1.5, 'F#5'), (6, 1.5, 0.5, 'G5'), (6, 2, 1, 'F#5'), (7, 0, 3, 'E5')]
    gtr = S.part(R('guitar.nylon', 'nylon', 'spanish', 'guitar'), pan=-0.2, gain_db=-5, sends={'room': 0.2, 'hall': 0.12})
    harp = S.part(R('pluck.harp', 'harp'), pan=0.45, gain_db=-3, sends={'hall': 0.3})
    mand = S.part(R('pluck.mandolin', 'mandolin'), pan=0.15, gain_db=-1, sends={'hall': 0.22})
    whis = S.part(R('flute.tin_whistle', 'tin_whistle', 'whistle'), pan=-0.35, gain_db=-4, sends={'hall': 0.25})
    reco = S.part(R('flute.recorder', 'recorder'), pan=0.35, gain_db=-5, sends={'hall': 0.25})
    viol = S.part(R('bowed.violin', 'violin', 'fiddle'), pan=-0.25, gain_db=-3, sends={'hall': 0.3})
    cello = S.part(R('bowed.cello', 'cello'), pan=0.25, gain_db=-3, sends={'hall': 0.3})
    oca = S.part(R('flute.ocarina', 'ocarina'), pan=0.0, gain_db=-3, sends={'hall': 0.35, 'echo': 0.15})
    cel = S.part(R('keys.celesta', 'celesta', 'music_box'), pan=0.55, gain_db=-9, sends={'hall': 0.35})
    strings = S.part(R('bowed.section_warm', 'section', 'strings'), gain_db=-11, sends={'hall': 0.4})
    dbass = S.part(R('bass.upright_pizz', 'pizz', 'bass'), gain_db=-5, sends={'hall': 0.15})
    drops = S.part(R('bell.glass_struck', 'glass', 'bowl'), pan=0.0, gain_db=-14, sends={'hall': 0.4})
    rng = np.random.default_rng(31)
    nb = 8 * reps
    prev = [nm('B3'), nm('D4'), nm('F#4'), nm('A4')]
    for b in range(nb):
        t0 = S.cur + b * bb
        ch = prog[b % 8]
        rep = b // 8
        root = th.bass_note(ch, nm('E2'), nm('D#3'))
        tones = th.lead([nm('B3'), nm('D4'), nm('G4')], ch, nm('A3'), nm('B4'), 3)
        pat = [root, tones[0], tones[1], tones[2], tones[1], tones[0]]
        for k, x in enumerate(pat):
            gtr.note(t0 + k * 0.5, x, 1.2 if k == 0 else 0.7, 0.62 if k == 0 else 0.48)
        dbass.note(t0, root - 12 if root - 12 >= nm('E1') else root, 1.5, 0.6)
        if b % 2 == 0:
            v = th.lead(prev, ch, nm('G3'), nm('D5'), 4)
            prev = v
            for x in v:
                strings.note(t0, x, bb * 2 * 0.98, 0.42)
        for _ in range(rng.poisson(2.5)):
            pcs = th.pcs(ch)
            cand = [x for x in range(nm('D6'), nm('D7')) if x % 12 in pcs]
            drops.note(t0 + rng.integers(12) * 0.25, cand[rng.integers(len(cand))], 0.3, rng.uniform(0.25, 0.55))
        for (bar, s, d, x) in [n for n in tune if n[0] == b % 8]:
            if rep == 0:
                mand.note(t0 + s, x, d * 0.95, 0.6)
            elif rep == 1:
                whis.note(t0 + s, x, d * 0.9, 0.62)
                lower = th.diatonic_shift([nm(x)], -2, th.scale_notes('D', 'major', 60, 96))[0]
                reco.note(t0 + s, lower, d * 0.9, 0.55)
            elif rep == 2:
                viol.note(t0 + s, x, d * 0.98, 0.62)
            else:
                oca.note(t0 + s, nm(x) - 12, d * 0.95, 0.6)
                cel.note(t0 + s, nm(x) + 12, 0.8, 0.4)
        if rep == 2:   # cello countermelody: falling chord thirds, half-bar notes
            ct = th.lead([nm('F#3'), nm('A3')], ch, nm('D3'), nm('D4'), 2)
            cello.note(t0, ct[1], 1.9, 0.55)
            cello.note(t0 + 2, ct[0], 0.95, 0.5)
        if rep >= 1:
            harp.note(t0 + 1.5, th.lead([nm('F#5')], ch, nm('D5'), nm('A5'), 1)[0], 1.0, 0.45)
    S.cur += nb * bb



# ============================================================ interlude: THE KAPOK CENSUS
def census(S: Suite, pool):
    """Amarok-style roll call: every instrument the rest of the suite doesn't
    feature gets a one-beat cameo, in key, over a D pedal. Pitched cameos play
    a fragment of the Canopy motif (A-D-E-A) fitted to the chord; unpitched
    ones play a small rhythm. Families travel together so it reads as a tour."""
    if not pool:
        return
    S.section('Interlude: the Kapok Census', 120)
    bb = 4
    prog = ['Dmaj9', 'Dmaj9', 'C/D', 'C/D', 'G/D', 'G/D', 'Asus2/D', 'A/D']
    fam_order = ['mallet', 'metal', 'bell', 'pluck', 'guitar', 'bass', 'bowed', 'keys', 'organ', 'flute', 'reed', 'brass',
                 'voice', 'creature', 'synth', 'drum', 'perc']
    pool = sorted(pool, key=lambda k: (fam_order.index(REGISTRY[k].family) if REGISTRY[k].family in fam_order else 99, k))
    per_bar = 4
    nbars = int(np.ceil(len(pool) / per_bar)) + 1
    ped = S.part(R('organ.pipe_flute', 'pipe'), gain_db=-14, sends={'hall': 0.3})
    kal = S.part(R('mallet.kalimba', 'kalimba'), pan=-0.3, gain_db=-12, sends={'canopy': 0.2, 'echo': 0.1})
    bass = S.part(R('mallet.marimba_bass', 'marimba'), gain_db=-6, sends={'hall': 0.1})
    pans = [-0.6, 0.6, -0.25, 0.25]
    cams = {}
    motif = [nm('A4'), nm('D5'), nm('E5'), nm('A5')]
    for b in range(nbars):
        t0 = S.cur + b * bb
        ch = prog[b % 8]
        for x in (nm('D3'), nm('A3')):
            ped.note(t0, x, bb * 0.99, 0.4)
        for k in range(8):
            kal.note(t0 + k * 0.5, motif[k % 4] if k % 2 == 0 else motif[(k + 1) % 4] - 12, 0.45, 0.45)
        bass.note(t0, nm('D2'), 1.5, 0.6)
        bass.note(t0 + 2.5, nm('A2') if b % 2 else nm('D2'), 1.0, 0.5)
        for j in range(per_bar):
            i = b * per_bar + j
            if i >= len(pool):
                continue
            inst = pool[i]
            info = REGISTRY[inst]
            if inst not in cams:
                loud_metal = any(w in inst for w in ('cymbal', 'tam_tam', 'triangle', 'mark_tree', 'crotales', 'gong'))
                cams[inst] = S.part(inst, pan=pans[j], gain_db=-11 if loud_metal else (-5 if info.pitched else -8), sends={'hall': 0.25})
            p = cams[inst]
            t = t0 + j
            if info.pitched and (info.sustain or info.family in ('synth', 'voice', 'organ', 'bowed')):
                allowed = [m for m in range(info.lo, info.hi + 1) if m % 12 in th.pcs(ch.split('/')[0])]
                x = motif[j % 4]
                while x > info.hi and x - 12 >= info.lo:
                    x -= 12
                while x < info.lo and x + 12 <= info.hi:
                    x += 12
                p.note(t, th.snap(x, allowed) if allowed else x, 0.95, 0.7)
            elif info.pitched:
                frag = motif[j % 2: j % 2 + 3]
                allowed = [m for m in range(info.lo, info.hi + 1) if m % 12 in th.pcs(ch.split('/')[0])]
                for k, x in enumerate(frag):
                    while x > info.hi and x - 12 >= info.lo:
                        x -= 12
                    while x < info.lo and x + 12 <= info.hi:
                        x += 12
                    x = th.snap(x, allowed) if allowed else x
                    p.note(t + k * (1 / 3), x, 0.3 if k < 2 else 0.6, 0.65)
            else:
                for k, v in enumerate((0.8, 0.45, 0.6, 0.5)):
                    p.note(t + k * 0.25, 60, 0.22, v)
    S.cur += nbars * bb


# ============================================================ IV. DRUMS OF THE KAPOK
def drums_of_kapok(S: Suite, bars=32):
    S.section('IV. Drums of the Kapok', 150)
    m = Meter([12] * bars, unit=8, start=S.cur)
    base = ['Dm', 'Dm', 'Dm', 'Dm', 'C', 'C', 'Dm', 'Dm', 'Bb', 'Bb', 'C', 'C', 'Dm', 'Dm', 'A', 'A']
    prog = (base * 3)[:bars]
    P = lambda *a, **k: S.part(*a, **k)  # noqa: E731
    bell = P(R('perc.agogo', 'agogo', 'cowbell'), pan=0.4, gain_db=-8, sends={'canopy': 0.15})
    shk = P(R('perc.ganza', 'ganza', 'egg', 'shaker'), pan=-0.4, gain_db=-12, sends={'canopy': 0.12})
    djb = P(R('drum.djembe_bass', 'djembe'), pan=-0.2, gain_db=-3, sends={'canopy': 0.2})
    djt = P(R('drum.djembe_tone', 'djembe'), pan=-0.25, gain_db=-4, sends={'canopy': 0.2})
    djb2 = P(R('drum.djembe_slap', 'djembe', 'conga'), pan=0.25, gain_db=-5, sends={'canopy': 0.2})
    dun = P(R('drum.dundunba', 'dunun', 'sangban', 'surdo'), pan=0.0, gain_db=-3, sends={'canopy': 0.15})
    sang = P(R('drum.sangban', 'sangban'), pan=-0.35, gain_db=-5, sends={'canopy': 0.15})
    kenk = P(R('drum.kenkeni', 'kenkeni'), pan=0.35, gain_db=-7, sends={'canopy': 0.15})
    talk = P(R('drum.talking', 'talking'), pan=0.5, gain_db=-4, sends={'canopy': 0.25})
    udu = P(R('drum.udu', 'udu'), pan=-0.55, gain_db=-2, sends={'canopy': 0.2})
    bod = P(R('drum.bodhran', 'bodhran', 'frame'), pan=0.15, gain_db=-6, sends={'studio': 0.2})
    surdo = P(R('drum.surdo', 'surdo', 'bass_drum'), gain_db=-4, sends={'hall': 0.15})
    timp = P(R('drum.timpani', 'timpani'), gain_db=-5, sends={'hall': 0.3})
    didj = P(R('reed.didgeridoo', 'didgeridoo', 'didj'), gain_db=-8, sends={'canopy': 0.25})
    call = P(R('voice.chant_call', 'chant'), pan=-0.3, gain_db=-3, sends={'canopy': 0.3})
    resp = P(R('voice.chant_response', 'choir_aah'), pan=0.3, gain_db=-6, sends={'hall': 0.35})
    pipes = P(R('reed.uilleann_pipes', 'uilleann', 'pipes', 'bagpipe'), pan=0.0, gain_db=-2, sends={'hall': 0.3, 'echo': 0.1})
    kora = P(R('pluck.kora', 'kora', 'harp'), pan=0.55, gain_db=-6, sends={'canopy': 0.25})
    berim = P(R('pluck.berimbau', 'berimbau'), pan=-0.6, gain_db=-4, sends={'canopy': 0.2})
    lead = P(R('guitar.oldfield_lead', 'oldfield', 'lead'), pan=0.1, gain_db=-4, sends={'hall': 0.3, 'echo': 0.2})
    # the West African standard bell (traditional) and supporting parts, 12 eighths
    BELL = 'x.x.xx.x.x.x'
    DJ1 = 'b..t.tb..t.t'      # bass / tone
    DJ2 = '..s..s..s.ss'      # slaps
    DUN = 'x..x..x.x...'
    SURD = 'X.....x.....'
    BOD = 'X.xx.xX.xx.x'
    UDU = '...o.....o..'
    pipes_mel = [  # (bar offset in the 8-bar tune, eighth, length eighths, note) D dorian
        (0, 0, 3, 'D5'), (0, 3, 1, 'E5'), (0, 4, 2, 'F5'), (0, 6, 3, 'A5'), (0, 9, 3, 'G5'),
        (1, 0, 6, 'F5'), (1, 6, 2, 'E5'), (1, 8, 1, 'D5'), (1, 9, 3, 'C5'),
        (2, 0, 3, 'D5'), (2, 3, 3, 'F5'), (2, 6, 3, 'A5'), (2, 9, 3, 'C6'),
        (3, 0, 9, 'D6'), (3, 9, 3, 'C6'),
        (4, 0, 3, 'Bb5'), (4, 3, 3, 'A5'), (4, 6, 3, 'G5'), (4, 9, 3, 'F5'),
        (5, 0, 4, 'G5'), (5, 4, 2, 'A5'), (5, 6, 6, 'E5'),
        (6, 0, 3, 'F5'), (6, 3, 3, 'E5'), (6, 6, 3, 'D5'), (6, 9, 3, 'C#5'),
        (7, 0, 12, 'D5')]
    rng = np.random.default_rng(4)
    for b in range(bars):
        ch = prog[b]
        root = th.bass_note(ch, nm('D2'), nm('C#3'))
        acc = 1.0 if b < bars - 2 else 1.1
        for k, c in enumerate(BELL):
            if c == 'x':
                bell.note(m.bar(b, k), 60, 0.25, 0.6 if k else 0.8)
        for k in range(12):
            shk.note(m.bar(b, k), 60, 0.2, 0.45 if k % 3 == 0 else 0.25)
        if b >= 4:
            for k, c in enumerate(DJ1):
                if c != '.':
                    (djb if c == 'b' else djt).note(m.bar(b, k), 60, 0.4, (0.85 if c == 'b' else 0.6) * acc)
            for k, c in enumerate(DUN):
                if c != '.':
                    dun.note(m.bar(b, k), 60, 0.5, 0.75)
            for k, c in enumerate('..x...x..x..'):
                if c != '.':
                    sang.note(m.bar(b, k), 60, 0.4, 0.6)
            for k, c in enumerate('x.x.x.x.x.x.'):
                if c != '.' and b >= 6:
                    kenk.note(m.bar(b, k), 60, 0.3, 0.45 if k % 6 else 0.6)
        if b >= 6:
            for k, c in enumerate(DJ2):
                if c != '.':
                    djb2.note(m.bar(b, k), 72, 0.3, 0.6 * acc)
        if b >= 8:
            for k, c in enumerate(BOD):
                if c != '.':
                    bod.note(m.bar(b, k), 60, 0.25, (0.75 if c == 'X' else 0.45) * acc)
            for k, c in enumerate(UDU):
                if c != '.':
                    udu.note(m.bar(b, k), 50, 0.5, 0.6)
            if b % 2 == 0:   # talking-drum call: squeezes up over a dotted quarter
                talk.note(m.bar(b, 9), root + 24, 1.5, 0.7)
                talk.note(m.bar(b, 10.5), root + 24 + 3, 0.75, 0.55)
        if b >= 12:
            if b % 4 == 0:
                didj.note(m.bar(b), nm('D2'), 4 * 6 * 0.98, 0.6)
            if b < 24 and b % 2 == 0:   # call and response, wordless
                for k, (e, x, d) in enumerate([(0, 'A4', 3), (3, 'C5', 3), (6, 'D5', 6)]):
                    call.note(m.bar(b, e), x, d * 0.5 * 0.9, 0.7)
                for e, x, d in [(0, 'F4', 3), (3, 'E4', 3), (6, 'D4', 5)]:
                    for iv in (0, 7):
                        resp.note(m.bar(b + 1, e), nm(x) + iv, d * 0.5 * 0.9, 0.55)
        if b >= 16:
            for k in range(0, 12, 2):
                tone = th.lead([nm('A4')], ch, nm('D4'), nm('D5'), 1)[0]
                kora.note(m.bar(b, k), tone + (12 if k in (4, 10) else 0), 0.6, 0.5)
            for k in (0, 5, 8):
                berim.note(m.bar(b, k), root + 12 + (2 if k == 5 else 0), 0.5, 0.55)
            for (o, e, d, x) in [n for n in pipes_mel if n[0] == (b - 16) % 8]:
                pipes.note(m.bar(b, e), x, d * 0.5 * 0.96, 0.7)
        if b >= 24:
            for k, c in enumerate(SURD):
                if c != '.':
                    surdo.note(m.bar(b, k), 60, 0.6, 0.9 if c == 'X' else 0.6)
            if b % 2 == 0:
                timp.note(m.bar(b), root, 2.5, 0.7 + 0.03 * (b - 24))
            for (o, e, d, x) in [n for n in pipes_mel if n[0] == (b - 24) % 8]:
                lead.note(m.bar(b, e), nm(x) - 12, d * 0.5 * 0.98, 0.75)
            for x in th.voice(ch, nm('D4'), 3):
                resp.note(m.bar(b), x, 6 * 0.95, 0.5)
    # final unison hit and a held breath
    end = m.end
    for p, pitch in ((djb, 48), (dun, 60), (surdo, 60), (bod, 60), (timp, nm('D2')), (bell, 60)):
        p.note(end, pitch, 1.5, 1.0)
    S.cur = end + 6.0     # 2.4 s of silence-with-reverb


# ============================================================ V. THE PROCESSION
def procession(S: Suite):
    S.section('V. The Procession', 100)
    bb = 4
    riff = [(0, 1.5, 'D2'), (1.5, 0.5, 'D2'), (2, 1, 'A2'), (3, 1, 'C3'), (4, 1, 'D3'), (5, 1, 'C3'), (6, 1, 'A2'),
            (7, 0.5, 'G2'), (7.5, 0.5, 'A2')]
    prog2 = ['D', 'C/D', 'G/D', 'D', 'D', 'C/D', 'Bb', 'C']     # per bar, 8-bar cycle (mixolydian)
    theme = [(0, 0, 4, 'A4'), (1, 0, 2, 'D5'), (1, 2, 2, 'E5'), (2, 0, 4, 'F#5'), (3, 0, 2, 'E5'), (3, 2, 2, 'D5'),
             (4, 0, 3, 'E5'), (4, 3, 1, 'D5'), (5, 0, 2, 'C5'), (5, 2, 2, 'A4'), (6, 0, 4, 'D5'), (7, 0, 2, 'C5'),
             (7, 2, 2, 'E5')]
    intros = [
        ('Grand piano', 'piano'), ('Reed and pipe organ', 'organ'), ('Glockenspiel', 'glock'),
        ('Bass guitar', 'bassg'), ('Double-speed guitar', 'dsg'), ('Two slightly distorted guitars', 'lead2'),
        ('Mandolin', 'mand'), ('Spanish guitar, and introducing acoustic guitar', 'acou'),
        ('Bamboo flutes', 'flutes'), ('The tree-frog choir', 'frogs'), ('Steel pans', 'pans'),
        ('Howler monkeys', 'howl'), ('Plus ... tubular bells!', 'bells')]
    P = {
        'piano': S.part(R('keys.grand_piano', 'grand_piano', 'piano'), pan=-0.1, gain_db=-2, sends={'hall': 0.25}),
        'organ': S.part(R('organ.pipe', 'pipe'), pan=0.0, gain_db=-9, sends={'church': 0.3}),
        'reed': S.part(R('organ.harmonium', 'harmonium', 'reed_organ', 'pump'), pan=0.2, gain_db=-10, sends={'hall': 0.2}),
        'glock': S.part(R('mallet.glockenspiel', 'glock'), pan=0.5, gain_db=-3, sends={'hall': 0.3}),
        'bassg': S.part(R('bass.fingered', 'bass.picked', 'bass'), gain_db=-2, sends={'studio': 0.1}),
        'dsg': S.part(R('guitar.double_speed', 'double', 'electric', 'guitar'), pan=-0.45, gain_db=-8, sends={'studio': 0.2}),
        'lead2a': S.part(R('guitar.oldfield_lead', 'oldfield', 'lead'), pan=-0.3, gain_db=0, sends={'hall': 0.25, 'echo': 0.15}),
        'lead2b': S.part(R('guitar.oldfield_lead', 'oldfield', 'lead'), pan=0.3, gain_db=-2, sends={'hall': 0.25, 'echo': 0.15}),
        'mand': S.part(R('pluck.mandolin', 'mandolin'), pan=0.4, gain_db=-1, sends={'hall': 0.2}),
        'span': S.part(R('guitar.nylon', 'nylon', 'spanish'), pan=-0.55, gain_db=-6, sends={'hall': 0.2}),
        'acou': S.part(R('guitar.steel', 'steel', 'acoustic', '12_string'), pan=0.55, gain_db=-7, sends={'hall': 0.2}),
        'flute1': S.part(R('flute.bamboo', 'bamboo'), pan=-0.25, gain_db=-1, sends={'hall': 0.3}),
        'flute2': S.part(R('flute.pan_flute', 'pan_flute', 'flute'), pan=0.25, gain_db=-6, sends={'hall': 0.3}),
        'frogs': S.part(R('creature.treefrog_choir', 'treefrog'), pan=0.0, gain_db=-8, sends={'canopy': 0.3}),
        'pans': S.part(R('mallet.steel_pan', 'steel_pan', 'steelpan', 'pan'), pan=0.35, gain_db=-4, sends={'hall': 0.25}),
        'howl': S.part(R('creature.howler', 'howler', 'monkey'), pan=-0.5, gain_db=-8, sends={'valley': 0.4}),
        'bells': S.part(R('bell.tubular', 'tubular', 'chime'), pan=0.15, gain_db=1, sends={'church': 0.45}),
        'choir': S.part(R('voice.choir_aah', 'choir'), gain_db=-6, sends={'church': 0.4}),
        'strings': S.part(R('bowed.section_warm', 'section', 'strings'), gain_db=-6, sends={'hall': 0.35}),
        'timp': S.part(R('drum.timpani', 'timpani'), gain_db=-5, sends={'hall': 0.3}),
        'kit_k': S.part(R('drum.kick_felt', 'kick'), gain_db=-6, sends={'studio': 0.1}),
        'kit_s': S.part(R('drum.snare', 'snare'), gain_db=-4, sends={'hall': 0.15}),
    }
    pre = 2          # bars of riff alone (on the bass marimba) before the first introduction
    intro_bars = 2
    total = pre + len(intros) * intro_bars + 8
    entered = set()
    starter = S.part(R('mallet.marimba_bass', 'bass_marimba', 'marimba'), gain_db=-3, sends={'hall': 0.15})
    for b in range(total):
        t0 = S.cur + b * bb
        ch = prog2[b % 8]
        k_int = (b - pre) // intro_bars
        if b >= pre and (b - pre) % intro_bars == 0 and k_int < len(intros):
            entered.add(intros[k_int][1])
            S.intros.append((S.s.clock.sec(t0 - 1.5), intros[k_int][0]))
        full = b >= pre + len(intros) * intro_bars
        # riff (2 bars long, so it pairs with the chord cycle)
        if b % 2 == 0:
            for (o, d, x) in riff:
                xs = nm(x)
                if ch.startswith('Bb'):
                    xs -= 4 if o == 0 else 0
                if b < pre + 6:
                    starter.note(t0 + o, xs + 12, d * 0.9, 0.7)
                if 'piano' in entered:
                    P['piano'].note(t0 + o, xs + 24, d * 0.9, 0.62)
                if 'bassg' in entered:
                    P['bassg'].note(t0 + o, xs, d * 0.92, 0.75)
        tm = [n for n in theme if n[0] == b % 8]
        if 'piano' in entered:
            for x in th.voice(ch, nm('D4'), 3):
                P['piano'].note(t0 + 2, x, 1.8, 0.45)
        if 'organ' in entered:
            for x in th.voice(ch, nm('A3'), 4):
                P['organ'].note(t0, x, bb * 0.98, 0.5)
                P['reed'].note(t0, x + 12, bb * 0.98, 0.45)
        if 'glock' in entered:
            for (_, s, d, x) in tm:
                P['glock'].note(t0 + s, nm(x) + 12, 0.8, 0.55)
        if 'dsg' in entered:
            tones = th.lead([nm('D4'), nm('A4'), nm('D5')], ch, nm('A3'), nm('F#5'), 3)
            for k in range(16):
                P['dsg'].note(t0 + k * 0.25, tones[[0, 2, 1, 2][k % 4]], 0.22, 0.55 if k % 4 else 0.7)
        if 'lead2' in entered:
            for (_, s, d, x) in tm:
                P['lead2a'].note(t0 + s, x, d * 0.98, 0.75)
                third = th.diatonic_shift([nm(x)], -2, th.scale_notes('D', 'mixolydian', 50, 90))[0]
                P['lead2b'].note(t0 + s, third, d * 0.98, 0.68)
        if 'mand' in entered:
            for (_, s, d, x) in tm:
                P['mand'].note(t0 + s, nm(x) + 12, d * 0.95, 0.5)
        if 'acou' in entered:
            v = th.voice(ch, nm('D3'), 5)
            for s in (0, 1.5, 2, 3, 3.5):
                P['span'].chord(t0 + s, v, 0.45, 0.5, strum=0.02)
                P['acou'].chord(t0 + s + 0.02, [x + 12 for x in v[:4]], 0.4, 0.45, strum=-0.015)
        if 'flutes' in entered:
            for (_, s, d, x) in tm:
                P['flute1'].note(t0 + s, nm(x) + 12, d * 0.95, 0.55)
            if b % 2 == 1:
                P['flute2'].note(t0 + 1, th.lead([nm('A5')], ch, nm('E5'), nm('D6'), 1)[0], 2.8, 0.45)
        if 'frogs' in entered:
            for x in th.voice(ch, nm('A4'), 3):
                for s in (0.5, 1.5, 2.5, 3.5):
                    P['frogs'].note(t0 + s, x, 0.3, 0.4)
        if 'pans' in entered:
            tones = [x for x in range(nm('D5'), nm('D6')) if x % 12 in th.pcs(ch)]
            for k in range(8):
                P['pans'].note(t0 + k * 0.5, tones[(k * 2) % len(tones)], 0.45, 0.5)
        if 'howl' in entered and b % 4 == 1:
            P['howl'].note(t0 + 2, th.bass_note(ch, nm('D3'), nm('C#4')), 2.5, 0.6)
        if 'bells' in entered:
            for (_, s, d, x) in tm:
                P['bells'].note(t0 + s, x, max(d, 2), 0.85)
        if full:
            j = b - (pre + len(intros) * intro_bars)
            for x in th.voice(ch, nm('D4'), 4):
                P['choir'].note(t0, x, bb * 0.98, 0.5 + 0.04 * j)
                P['strings'].note(t0, x - 12, bb * 0.98, 0.5 + 0.04 * j)
            P['timp'].note(t0, th.bass_note(ch, nm('D2'), nm('C#3')), 1.5, 0.6 + 0.04 * j)
            for s in (0, 1.5, 2.5):
                P['kit_k'].note(t0 + s, 60, 0.4, 0.7)
            for s in (1, 3):
                P['kit_s'].note(t0 + s, 60, 0.3, 0.55)
    P['piano'].level(S.cur + (pre + 2) * bb, 0).level(S.cur + (pre + 4) * bb, -5)
    S.cur += total * bb
    # final grand chord, then the canopy opens: space synths take the theme
    end = S.cur
    for p, xs in ((P['organ'], [nm('D3'), nm('A3'), nm('D4'), nm('F#4')]), (P['choir'], [nm('D4'), nm('A4'), nm('F#5')]),
                  (P['strings'], [nm('D3'), nm('A3'), nm('F#4')]), (P['bells'], [nm('D5')]), (P['timp'], [nm('D2')]),
                  (P['bassg'], [nm('D2')]), (P['piano'], [nm('D2'), nm('D3'), nm('A3'), nm('F#4')])):
        for x in xs:
            p.note(end, x, 6.0, 0.85)
    S.cur = end + 4
    cosmos(S)


def cosmos(S: Suite, bars=12):
    """the top of the kapok: the sky opens — the bridge into the space half of the game."""
    S.markers.append((S.s.clock.sec(S.cur), 'V. ... into the stars'))
    bb = 4
    prog = ['Dmaj9', 'Dmaj9', 'Gmaj7#11', 'Gmaj7#11', 'Bm11', 'Bm11', 'Asus2', 'A', 'Dmaj9', 'Bbmaj7#11', 'Asus2', 'Dmaj9']
    ther = S.part(R('synth.theremin', 'theremin'), pan=0.0, gain_db=0, sends={'cosmos': 0.45, 'echo': 0.2})
    sm = S.part(R('synth.string_machine', 'string_machine', 'solina', 'pad'), gain_db=-9, sends={'cosmos': 0.4})
    shim = S.part(R('synth.granular', 'shimmer', 'granular'), gain_db=-12, sends={'cosmos': 0.5})
    fmb = S.part(R('synth.fm_bell', 'fm_bell', 'dx'), pan=0.4, gain_db=-12, sends={'cosmos': 0.4, 'echo': 0.25})
    puls = S.part(R('synth.pulsar', 'pulsar', 'beacon', 'blip'), pan=-0.4, gain_db=-16, sends={'cosmos': 0.4})
    whale = S.part(R('creature.whale', 'whale'), pan=-0.3, gain_db=-12, sends={'cosmos': 0.5})
    sub = S.part(R('synth.sub_bass', 'sub'), gain_db=-8, sends={'cosmos': 0.1})
    mello = S.part(R('keys.mellotron_strings', 'mellotron'), pan=0.3, gain_db=-11, sends={'cosmos': 0.35})
    theme = [(0, 0, 4, 'A4'), (1, 0, 2, 'D5'), (1, 2, 2, 'E5'), (2, 0, 6, 'F#5'), (3, 2, 2, 'E5'), (4, 0, 4, 'D5'),
             (5, 0, 4, 'C#5'), (6, 0, 4, 'B4'), (7, 0, 4, 'C#5'), (8, 0, 8, 'A4'), (10, 0, 8, 'E5')]
    prev = [nm('D4'), nm('F#4'), nm('A4'), nm('C#5')]
    for b in range(bars):
        t0 = S.cur + b * bb
        ch = prog[b]
        v = th.lead(prev, ch, nm('A3'), nm('E5'), 4)
        prev = v
        for x in v:
            sm.note(t0, x, bb * 1.02, 0.5)
            if b % 2 == 0:
                shim.note(t0, x + 12, bb * 2, 0.45)
        sub.note(t0, th.bass_note(ch, nm('D1'), nm('C#2')) + 12, bb * 0.98, 0.55)
        tones = [x for x in range(nm('A5'), nm('A6')) if x % 12 in th.pcs(ch)]
        for k in range(0, 16, 3):
            puls.note(t0 + k * 0.25, tones[k % len(tones)], 0.2, 0.4)
        if b % 2 == 1:
            fmb.note(t0 + 1.5, tones[-1], 2.0, 0.45)
        if b in (3, 9):
            whale.note(t0, th.bass_note(ch, nm('A3'), nm('G#4')), 5.0, 0.5)
        if b >= 6:
            mello.note(t0, v[-1], bb * 0.95, 0.4)
    for (bar, s, d, x) in theme:
        ther.note(S.cur + bar * bb + s, x, d * 0.97, 0.6)
    S.cur += bars * bb + 2


# ============================================================ VI. HORNBILL HORNPIPE
def hornpipe(S: Suite):
    S.section('VI. Hornbill Hornpipe', 112)
    A = ['D5 F#5 A5 F#5 D5 F#5 A5 B5', 'A5 F#5 E5 D5 E5 F#5 E5 D5', 'B4 D5 E5 F#5 G5 F#5 E5 D5',
         'C#5 E5 A5 E5 C#5 B4 A4 C#5', 'D5 F#5 A5 F#5 D5 F#5 A5 D6', 'B5 A5 F#5 D5 E5 F#5 G5 B5',
         'A5 F#5 E5 C#5 D5 E5 F#5 A4', 'D5 . . . A4 . D5 .']
    B = ['F#5 A5 D6 A5 F#5 A5 D6 E6', 'D6 B5 A5 F#5 G5 B5 A5 G5', 'F#5 D5 E5 F#5 G5 A5 B5 G5',
         'A5 E5 C#5 E5 A5 . . .', 'F#5 A5 D6 A5 G5 B5 D6 B5', 'A5 F#5 D5 F#5 E5 G5 B5 G5',
         'F#5 D5 E5 C#5 D5 B4 A4 C#5', 'D5 . A4 . D5 . . .']
    chA = ['D', 'D', 'G', 'A', 'D', 'G', 'A', 'D']
    chB = ['D', 'G', 'D', 'A', 'D', 'Em', 'A', 'D']
    whistle = S.part(R('flute.tin_whistle', 'tin_whistle', 'whistle'), pan=-0.2, gain_db=0, sends={'room': 0.2})
    fiddle = S.part(R('bowed.fiddle', 'fiddle', 'violin'), pan=0.2, gain_db=-4, sends={'room': 0.2})
    banjo = S.part(R('pluck.banjo', 'banjo'), pan=0.5, gain_db=-9, sends={'room': 0.15})
    accord = S.part(R('organ.accordion', 'accordion'), pan=-0.5, gain_db=-11, sends={'room': 0.2})
    bod = S.part(R('drum.bodhran', 'bodhran', 'frame'), gain_db=-5, sends={'room': 0.15})
    dbass = S.part(R('bass.upright_pizz', 'pizz', 'bass'), gain_db=-4, sends={'room': 0.1})
    horn = S.part(R('creature.toucan_clack', 'toucan'), pan=0.65, gain_db=-8, sends={'canopy': 0.2})
    bouz = S.part(R('pluck.bouzouki', 'bouzouki', 'mandolin'), pan=-0.65, gain_db=-8, sends={'room': 0.15})
    tunes = [(A, chA)] * 2 + [(B, chB)] * 2 + [(A, chA), (B, chB)]
    bb = 4
    nb = 0
    tempo = 112.0
    for ti, (tune, chs) in enumerate(tunes):
        for i, row in enumerate(tune):
            t0 = S.cur
            if ti >= 2:   # the accelerando: a little faster every bar from the third time through
                tempo = min(172.0, tempo + 2.4)
                S.s.tempo_at(t0, tempo)
            ch = chs[i]
            for k, x in enumerate(row.split()):
                if x == '.':
                    continue
                whistle.note(t0 + k * 0.5, x, 0.46, 0.7 if k % 2 == 0 else 0.55)
                if ti >= 1:
                    fiddle.note(t0 + k * 0.5, x, 0.48, 0.6)
            v = th.voice(ch, nm('D4'), 3)
            for k in range(8):
                if k % 2 == 1:
                    banjo.chord(t0 + k * 0.5, v, 0.3, 0.5, strum=0.01)
            if ti >= 1:
                for s in (0, 2):
                    accord.chord(t0 + s, th.voice(ch, nm('A3'), 4), 1.8, 0.5)
            root = th.bass_note(ch, nm('D2'), nm('C#3'))
            dbass.note(t0, root, 0.9, 0.7)
            dbass.note(t0 + 2, root + 7, 0.9, 0.6)
            for k in range(8):
                bod.note(t0 + k * 0.5, 60, 0.3, 0.75 if k % 4 == 0 else 0.45)
            if ti >= 2:
                horn.note(t0 + 1.5, 60, 0.25, 0.6)
                horn.note(t0 + 3.5, 60, 0.25, 0.6)
                for k in range(0, 8, 2):
                    bouz.note(t0 + k * 0.5 + 0.5, v[k // 2 % 3] - 12, 0.4, 0.45)
            S.cur += bb
            nb += 1
    # last chord, everybody, then a frog has the last word
    end = S.cur
    for p, xs in ((whistle, ['D6']), (fiddle, ['D5', 'A5']), (accord, ['D4', 'F#4', 'A4']), (banjo, ['D4', 'F#4', 'A4']),
                  (dbass, ['D2']), (bod, [60])):
        for x in xs:
            p.note(end, x, 1.5, 0.9)
    frog = S.part(R('creature.bullfrog', 'bullfrog', 'frog'), gain_db=-2, sends={'canopy': 0.25})
    frog.note(end + 3.5, nm('D2'), 0.6, 0.9)
    if has('kookaburra'):
        kook = S.part(R('creature.kookaburra', 'kookaburra', 'laugh'), pan=0.6, gain_db=-8, sends={'valley': 0.4})
        kook.note(end + 4.4, nm('D5'), 1.5, 0.6)
    S.cur = end + 6


# ============================================================ assemble
SECTIONS = OrderedDict([('I', forest_floor), ('II', creature_chorus), ('III', rain_on_leaves), ('IV', drums_of_kapok),
                        ('V', procession), ('VI', hornpipe)])


def compose(which=None, with_census=True):
    """two passes: the first finds which instruments the movements use, the
    second places every remaining one in the Census interlude (after III)."""
    def run(pool):
        S = Suite()
        for key, fn in SECTIONS.items():
            if which and key not in which:
                continue
            if key == 'II':
                fn(S, used=set(S.s.instruments()))
            else:
                fn(S)
            if key == 'III' and pool is not None and (not which or 'C' in which or 'III' in which):
                census(S, pool)
        return S
    S = run(None)
    if not with_census:
        return S
    unused = sorted(set(REGISTRY) - set(S.s.instruments()) - {k for k in REGISTRY if k.startswith('test.')})
    return run(unused)


def main():
    out = '../out/kapok'
    if '--out' in sys.argv:
        out = sys.argv[sys.argv.index('--out') + 1]
    which = None
    if '--sections' in sys.argv:
        which = sys.argv[sys.argv.index('--sections') + 1].split(',')
    census_on = '--no-census' not in sys.argv
    os.makedirs(out, exist_ok=True)
    t = time.time()
    S = compose(which, census_on)
    insts = S.s.instruments()
    nnotes = sum(len(p.notes) for p in S.s.parts.values())
    print(f'composed: {len(S.s.parts)} parts, {nnotes} notes, {len(insts)} distinct instruments, '
          f'{S.s.length_s():.0f} s ({time.time() - t:.1f}s)')
    for a, b in SUBS:
        print(f'  role {a} -> {b}')
    mix, L = S.s.render_chunked(progress=True)
    import soundfile as sf
    sf.write(os.path.join(out, 'kapok.wav'), mix.T, S.s.sr, subtype='PCM_24')
    cred = S.credits()
    with open(os.path.join(out, 'kapok_credits.md'), 'w') as f:
        f.write(f'# KAPOK — instruments in order of appearance ({len(cred)})\n\n')
        f.write('All synthesized by Canopy Orchestra. No samples or recordings.\n\n')
        mk = sorted(S.markers)
        for name, t0 in cred:
            sec = [lab for (ts, lab) in mk if ts <= t0 + 1e-6]
            f.write(f'- {int(t0 // 60)}:{int(t0 % 60):02d}  `{name}` — {REGISTRY[name].desc}  ({sec[-1] if sec else ""})\n')
        f.write('\n## Sections\n\n')
        for ts, lab in mk:
            f.write(f'- {int(ts // 60)}:{int(ts % 60):02d}  {lab}\n')
        if S.intros:
            f.write('\n## Introductions (announcer cues)\n\n')
            for ts, txt in S.intros:
                f.write(f'- {ts:.2f}s  {txt}\n')
    import json
    json.dump(dict(intros=S.intros, markers=S.markers, length=mix.shape[1] / S.s.sr),
              open(os.path.join(out, 'kapok_cues.json'), 'w'), indent=1)
    print(f'wrote {out}/kapok.wav  ({mix.shape[1] / S.s.sr:.0f} s, input loudness {L}) in {time.time() - t:.0f}s')


if __name__ == '__main__':
    main()
