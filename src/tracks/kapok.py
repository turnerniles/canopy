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
        self.bars = []                  # (seconds, meter, chord) — the bar grid, for the web player
        self._n = 0

    def part(self, inst, pan=0.0, gain_db=0.0, sends=None, **kw):
        self._n += 1
        return self.s.part(f'{inst}#{self._n}', inst, pan=pan, gain_db=gain_db, sends=sends or {'studio': 0.18}, **kw)

    def bar(self, beat, meter, chord=None):
        self.bars.append((round(self.s.clock.sec(beat), 4), meter, chord))

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
        S.bar(m.bar(b), '15/8', ch)
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
        S.bar(t0, '4/4', ch)
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
        S.bar(t0, '3/4', ch)
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
        S.bar(t0, '4/4', ch.split('/')[0])
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
        S.bar(m.bar(b), '12/8', ch)
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
def fit_phrase(notes, inst, prefer=0):
    """transpose a whole phrase by octaves so it sits inside the instrument's range
    (keeps the melody's contour, unlike per-note octave folding). prefer: octave
    shift to try first."""
    info = REGISTRY[inst]
    lo, hi = min(notes), max(notes)
    for k in [prefer] + [prefer + d for d in (1, -1, 2, -2, 3, -3)]:
        if lo + 12 * k >= info.lo and hi + 12 * k <= info.hi:
            return 12 * k
    return 12 * round(((info.lo + info.hi) / 2 - (lo + hi) / 2) / 12)


def procession(S: Suite):
    """Tubular Bells–style finale: a riff, then one instrument at a time is
    introduced and *featured* — it plays the theme for four bars, on its own
    melody line — before it settles into the accompaniment and the next one
    arrives. Then everybody plays the theme together, and the canopy opens."""
    S.section('V. The Procession', 100)
    bb = 4
    riff = [(0, 1.5, 'D2'), (1.5, 0.5, 'D2'), (2, 1, 'A2'), (3, 1, 'C3'), (4, 1, 'D3'), (5, 1, 'C3'), (6, 1, 'A2'),
            (7, 0.5, 'G2'), (7.5, 0.5, 'A2')]
    prog = ['D', 'C/D', 'G/D', 'D', 'D', 'C/D', 'Bb', 'C']     # one chord per bar of the 8-bar theme (mixolydian)
    theme = [(0, 0, 4, 'A4'), (1, 0, 2, 'D5'), (1, 2, 2, 'E5'), (2, 0, 4, 'F#5'), (3, 0, 2, 'E5'), (3, 2, 2, 'D5'),
             (4, 0, 3, 'E5'), (4, 3, 1, 'D5'), (5, 0, 2, 'C5'), (5, 2, 2, 'A4'), (6, 0, 4, 'D5'), (7, 0, 2, 'C5'),
             (7, 2, 2, 'E5')]
    TH = [nm(x) for (_, _, _, x) in theme]
    pre = 2            # bars of riff alone before the first introduction
    feat = 4           # bars each new instrument is featured
    # (announcement, key, lead instrument, preferred octave shift for the theme)
    intros = [('Grand piano', 'piano', 'keys.grand_piano', 0),
              ('Reed and pipe organ', 'organ', 'organ.pipe', 0),
              ('Glockenspiel', 'glock', 'bell.glockenspiel', 1),
              ('Bass guitar', 'bassg', 'bass.fingered', -2),
              ('Double-speed guitar', 'dsg', 'guitar.double_speed', 0),
              ('Two slightly distorted guitars', 'lead2', 'guitar.oldfield_lead', 0),
              ('Mandolin', 'mand', 'pluck.mandolin', 0),
              ('Spanish guitar, and introducing acoustic guitar', 'span', 'guitar.nylon_spanish', 0),
              ('Bamboo flutes', 'flutes', 'flute.bamboo', 0),
              ('The tree-frog choir', 'frogs', 'creature.treefrog_choir', 1),
              ('Steel pans', 'pans', 'metal.steel_pan', 0),
              ('Howler monkeys', 'howl', 'creature.howler_roar', -2),
              ('Plus ... tubular bells!', 'bells', 'bell.tubular', 0)]
    L = {}   # featured (lead) parts — loud, centred-ish
    A = {}   # accompaniment parts — what each instrument does after its feature

    def lead(key, inst, pan=0.0, db=0.0, sends=None):
        L[key] = S.part(R(inst), pan=pan, gain_db=db, sends=sends or {'hall': 0.28, 'echo': 0.08})
        return L[key]

    def acc(key, inst, pan=0.0, db=-10.0, sends=None):
        A[key] = S.part(R(inst), pan=pan, gain_db=db, sends=sends or {'hall': 0.22})
        return A[key]

    lead('piano', 'keys.grand_piano', -0.05, 0)
    lead('organ', 'organ.pipe', 0.0, -3, {'church': 0.35})
    lead('glock', 'bell.glockenspiel', 0.3, 1)
    lead('bassg', 'bass.fingered', 0.0, 2, {'studio': 0.12})
    lead('dsg', 'guitar.double_speed', -0.2, -1)
    lead('lead2', 'guitar.oldfield_lead', -0.3, 0, {'hall': 0.3, 'echo': 0.18})
    lead('lead2b', 'guitar.oldfield_lead', 0.3, -2, {'hall': 0.3, 'echo': 0.18})
    lead('mand', 'pluck.mandolin', 0.15, 1)
    lead('span', 'guitar.nylon_spanish', -0.15, 1)
    lead('flutes', 'flute.bamboo', -0.1, 0, {'hall': 0.35, 'echo': 0.12})
    lead('flutes2', 'flute.pan_flute', 0.25, -4, {'hall': 0.35})
    lead('frogs', 'creature.treefrog_choir', 0.0, 2, {'canopy': 0.3})
    lead('pans', 'metal.steel_pan', 0.1, 4)
    lead('howl', 'creature.howler_roar', 0.0, 3, {'valley': 0.3})
    lead('bells', 'bell.tubular', 0.1, 3, {'church': 0.45})
    starter = acc('marimba', 'mallet.marimba_bass', 0.0, -6, {'hall': 0.12})
    acc('piano', 'keys.grand_piano', -0.35, -11)
    acc('organ', 'organ.pipe', 0.0, -13, {'church': 0.3})
    acc('reed', 'organ.harmonium', 0.25, -14)
    acc('glock', 'bell.glockenspiel', 0.55, -14)
    acc('bassg', 'bass.fingered', 0.0, -5, {'studio': 0.1})
    acc('dsg', 'guitar.double_speed', -0.55, -14)
    acc('mand', 'pluck.mandolin', 0.5, -15)
    acc('span', 'guitar.nylon_spanish', -0.6, -13)
    acc('acou', 'guitar.steel_acoustic', 0.6, -10)
    acc('flutes', 'flute.pan_flute', -0.4, -15, {'hall': 0.35})
    acc('frogs', 'creature.treefrog_choir', 0.45, -14, {'canopy': 0.3})
    acc('pans', 'metal.steel_pan', -0.45, -15)
    acc('howl', 'creature.howler_roar', -0.6, -10, {'valley': 0.35})
    tutti = {k: S.part(R(i), pan=p, gain_db=d, sends={'church': 0.35}) for k, i, p, d in
             (('choir', 'voice.choir_aah', 0.0, -6), ('strings', 'strings.section_warm', 0.0, -6),
              ('timp', 'drum.timpani', 0.0, -5), ('kick', 'drum.kick_felt', 0.0, -6), ('snare', 'drum.snare', 0.0, -6))}

    def play_theme(part, t_bar0, half, inst, prefer, vel=0.78, legato=0.97, fn=None):
        shift = fit_phrase(TH, inst, prefer)
        for (bar, s, d, x) in theme:
            if half is not None and bar // 4 != half:
                continue
            t = t_bar0 + (bar - (4 * half if half is not None else 0)) * bb + s
            if fn:
                fn(part, t, nm(x) + shift, d)
            else:
                part.note(t, nm(x) + shift, d * legato, vel)

    def dsg_fn(part, t, m, d):            # double speed: the theme as fast tremolo-picked 16ths
        for k in range(int(d * 4)):
            part.note(t + k * 0.25, m, 0.23, 0.82 if k == 0 else 0.6)

    def span_fn(part, t, m, d):           # Spanish guitar: melody with a thumb bass under it
        part.note(t, m, d * 0.95, 0.8)
        part.note(t, m - 12 if m - 12 >= REGISTRY['guitar.nylon_spanish'].lo else m - 7, d * 0.9, 0.5)

    entered = []
    nbars = pre + feat * len(intros)
    for b in range(nbars + 8):
        t0 = S.cur + b * bb
        tb = b - pre                      # bar within the theme cycle
        ch = prog[tb % 8]
        S.bar(t0, '4/4', ch.split('/')[0])
        k = tb // feat if tb >= 0 else -1
        start_feature = tb >= 0 and tb % feat == 0 and k < len(intros)
        if start_feature:
            name, key, inst, pref = intros[k]
            S.intros.append((S.s.clock.sec(t0 - 1.5), name))
            half = k % 2
            fn = dsg_fn if key == 'dsg' else span_fn if key == 'span' else None
            vel = 0.85 if key in ('bells', 'howl', 'frogs') else 0.78
            play_theme(L[key], t0, half, inst, pref, vel=vel, fn=fn)
            if key == 'lead2':      # the second guitar a third below
                shift = fit_phrase(TH, inst, pref)
                for (bar, s, d, x) in theme:
                    if bar // 4 == half:
                        third = th.diatonic_shift([nm(x) + shift], -2, th.scale_notes('D', 'mixolydian', 40, 96))[0]
                        L['lead2b'].note(t0 + (bar - 4 * half) * bb + s, third, d * 0.97, 0.72)
            if key == 'flutes':     # pan flute answers each phrase a beat later
                for (bar, s, d, x) in theme:
                    if bar // 4 == half and d >= 2:
                        L['flutes2'].note(t0 + (bar - 4 * half) * bb + s + 1, nm(x) + fit_phrase(TH, 'flute.pan_flute', 0) + 12,
                                          d * 0.6, 0.5)
        if tb >= 0 and tb % feat == 0 and k - 1 >= 0 and k - 1 < len(intros):
            entered.append(intros[k - 1][1])      # the previous feature joins the band
        if start_feature and intros[k][1] == 'span':
            entered.append('acou')                # "...and introducing acoustic guitar": strums enter with it
        full = tb >= feat * len(intros)
        # ---------------- the riff: bass marimba until the bass guitar has had its feature
        if b % 2 == (pre % 2):
            for (o, d, x) in riff:
                xs = nm(x) - (4 if ch.startswith('Bb') and o == 0 else 0)
                if 'bassg' in entered:
                    A['bassg'].note(t0 + o, xs, d * 0.92, 0.78)
                else:
                    starter.note(t0 + o, xs + 12, d * 0.9, 0.72)
        # ---------------- accompaniment of everyone already introduced
        if 'piano' in entered:
            v = th.voice(ch, nm('D4'), 3)
            for s in (0, 2):
                A['piano'].chord(t0 + s, v, 1.8, 0.45)
        if 'organ' in entered:
            for x in th.voice(ch, nm('A3'), 4):
                A['organ'].note(t0, x, bb * 0.98, 0.5)
                A['reed'].note(t0, x + 12, bb * 0.98, 0.42)
        if 'glock' in entered:
            A['glock'].note(t0, th.lead([nm('A6')], ch, nm('D6'), nm('D7'), 1)[0], 1.0, 0.5)
        if 'dsg' in entered:
            tones = th.lead([nm('D4'), nm('A4'), nm('D5')], ch, nm('A3'), nm('F#5'), 3)
            for j in range(16):
                A['dsg'].note(t0 + j * 0.25, tones[[0, 2, 1, 2][j % 4]], 0.22, 0.55)
        if 'mand' in entered:
            A['mand'].note(t0, th.lead([nm('F#5')], ch, nm('D5'), nm('A5'), 1)[0], bb * 0.9, 0.45)
        if 'span' in entered:
            v = th.voice(ch, nm('D3'), 5)
            for s in (0, 1.5, 3):
                A['span'].chord(t0 + s, v, 0.6, 0.45, strum=0.02)
        if 'acou' in entered:
            v = th.voice(ch, nm('D3'), 5)
            for s in (0, 1, 1.5, 2, 3, 3.5):
                A['acou'].chord(t0 + s, [x + 12 for x in v[:4]], 0.4, 0.5 if s in (0, 2) else 0.38, strum=-0.015)
        if 'flutes' in entered and b % 2 == 1:
            A['flutes'].note(t0 + 1, th.lead([nm('A5')], ch, nm('E5'), nm('D6'), 1)[0], 2.8, 0.45)
        if 'frogs' in entered:
            for s in (0.5, 1.5, 2.5, 3.5):
                A['frogs'].note(t0 + s, th.lead([nm('A5')], ch, nm('D5'), nm('D6'), 1)[0], 0.3, 0.45)
        if 'pans' in entered:
            tones = [x for x in range(nm('D5'), nm('D6')) if x % 12 in th.pcs(ch)]
            for j in range(8):
                A['pans'].note(t0 + j * 0.5, tones[(j * 2) % len(tones)], 0.45, 0.45)
        if 'howl' in entered and tb % 4 == 3:
            A['howl'].note(t0 + 1, th.bass_note(ch, nm('D3'), nm('C#4')), 2.5, 0.6)
        # ---------------- everybody: the whole theme, twice as grand
        if full:
            j = tb - feat * len(intros)
            for key, inst, pref in (('lead2', 'guitar.oldfield_lead', 0), ('bells', 'bell.tubular', 0),
                                    ('glock', 'bell.glockenspiel', 1), ('flutes', 'flute.bamboo', 0),
                                    ('mand', 'pluck.mandolin', 0)):
                if j == 0:
                    play_theme(L[key], t0, None, inst, pref, vel=0.8)
            for x in th.voice(ch, nm('D4'), 4):
                tutti['choir'].note(t0, x, bb * 0.98, 0.5 + 0.04 * j)
                tutti['strings'].note(t0, x - 12, bb * 0.98, 0.5 + 0.04 * j)
            tutti['timp'].note(t0, th.bass_note(ch, nm('D2'), nm('C#3')), 1.5, 0.6 + 0.04 * j)
            for s in (0, 1.5, 2.5):
                tutti['kick'].note(t0 + s, 60, 0.4, 0.7)
            for s in (1, 3):
                tutti['snare'].note(t0 + s, 60, 0.3, 0.55)
    S.cur += (nbars + 8) * bb
    # final grand chord, then the canopy opens: space synths take the theme
    end = S.cur
    for p, xs in ((A['organ'], [nm('D3'), nm('A3'), nm('D4'), nm('F#4')]), (tutti['choir'], [nm('D4'), nm('A4'), nm('F#5')]),
                  (tutti['strings'], [nm('D3'), nm('A3'), nm('F#4')]), (L['bells'], [nm('D5')]), (tutti['timp'], [nm('D2')]),
                  (A['bassg'], [nm('D2')]), (A['piano'], [nm('D2'), nm('D3'), nm('A3'), nm('F#4')])):
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
        S.bar(t0, '4/4', ch)
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
            S.bar(t0, '4/4', ch)
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
