"""Chord symbols, scales, voicing and motif transforms.

    chord('Gmaj9')            -> pitch classes relative info + default voicing
    voice('Bbmaj7#11', 48, 5) -> 5 MIDI notes, close voicing from C3 upward
    lead(prev, 'Em9', ...)    -> voice-led voicing nearest to prev
"""
from __future__ import annotations

import re
import numpy as np

from .core import nm

PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

# quality -> intervals above the root (semitones)
QUAL = {
    '': [0, 4, 7], 'maj': [0, 4, 7], 'm': [0, 3, 7], 'min': [0, 3, 7], 'dim': [0, 3, 6], 'aug': [0, 4, 8],
    '5': [0, 7], 'sus2': [0, 2, 7], 'sus4': [0, 5, 7], 'sus': [0, 5, 7],
    '6': [0, 4, 7, 9], 'm6': [0, 3, 7, 9], '6/9': [0, 4, 7, 9, 14], 'm6/9': [0, 3, 7, 9, 14],
    '7': [0, 4, 7, 10], 'maj7': [0, 4, 7, 11], 'm7': [0, 3, 7, 10], 'm7b5': [0, 3, 6, 10], 'dim7': [0, 3, 6, 9],
    'mmaj7': [0, 3, 7, 11], '7sus4': [0, 5, 7, 10], '7sus2': [0, 2, 7, 10],
    '9': [0, 4, 7, 10, 14], 'maj9': [0, 4, 7, 11, 14], 'm9': [0, 3, 7, 10, 14], 'add9': [0, 4, 7, 14],
    'madd9': [0, 3, 7, 14], '9sus4': [0, 5, 7, 10, 14], '11': [0, 7, 10, 14, 17], 'm11': [0, 3, 7, 10, 14, 17],
    'maj11': [0, 4, 7, 11, 14, 17], '13': [0, 4, 7, 10, 14, 21], 'maj13': [0, 4, 7, 11, 14, 21],
    '13sus': [0, 5, 7, 10, 14, 21], 'm13': [0, 3, 7, 10, 14, 21],
}
ALTER = {'#11': 18, 'b9': 13, '#9': 15, 'b13': 20, '#5': 8, 'b5': 6, 'add11': 17, 'add13': 21}

SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11], 'ionian': [0, 2, 4, 5, 7, 9, 11],
    'dorian': [0, 2, 3, 5, 7, 9, 10], 'phrygian': [0, 1, 3, 5, 7, 8, 10], 'lydian': [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10], 'minor': [0, 2, 3, 5, 7, 8, 10], 'aeolian': [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11], 'pent_major': [0, 2, 4, 7, 9], 'pent_minor': [0, 3, 5, 7, 10],
    'hirajoshi': [0, 2, 3, 7, 8], 'pelog': [0, 1, 3, 7, 8], 'slendro': [0, 2, 5, 7, 9], 'whole': [0, 2, 4, 6, 8, 10],
}


def _root(s):
    m = re.match(r'^([A-G])([#b]?)', s)
    if not m:
        raise ValueError(f'bad chord {s!r}')
    pc = (PC[m.group(1)] + (1 if m.group(2) == '#' else -1 if m.group(2) == 'b' else 0)) % 12
    return pc, s[m.end():]


def chord(sym: str):
    """'Bbmaj7#11/D' -> dict(root=pc, ints=[...], bass=pc, sym=sym)"""
    sym = sym.strip()
    body, bass = sym, None
    if '/' in sym and not sym.endswith('6/9'):
        body, b = sym.rsplit('/', 1)
        if b and b[0] in PC:
            bass = _root(b)[0]
        else:
            body = sym
    root, rest = _root(body)
    alters = []
    # peel alterations, e.g. 'maj7#11', '7sus4b9', 'add9'
    while rest not in QUAL:
        for k in sorted(ALTER, key=len, reverse=True):
            if k in rest:
                rest = rest.replace(k, '', 1)
                alters.append(ALTER[k])
                break
        else:
            break
    if rest not in QUAL:
        raise ValueError(f'unknown chord quality {rest!r} in {sym!r}')
    ints = sorted(set(QUAL[rest] + alters))
    if ALTER['b9'] in alters and 14 in ints:
        ints.remove(14)
    if ALTER['#11'] in alters and 17 in ints:
        ints.remove(17)
    return dict(root=root, ints=ints, bass=root if bass is None else bass, sym=sym)


def pcs(sym):
    c = chord(sym)
    return sorted({(c['root'] + i) % 12 for i in c['ints']})


def bass_note(sym, lo=nm('E1'), hi=nm('D#2')):
    c = chord(sym)
    m = lo + ((c['bass'] - lo) % 12)
    return m if m <= hi + 12 else m - 12


def voice(sym, lo=nm('C3'), n=4, spread=False):
    """close (or spread) voicing of n notes, lowest note >= lo, root-ish first."""
    c = chord(sym)
    tones = [(c['root'] + i) for i in c['ints']]
    out, k = [], 0
    base = lo + ((c['root'] - lo) % 12)
    while len(out) < n:
        i = c['ints'][k % len(c['ints'])] + 12 * (k // len(c['ints']))
        out.append(base + i + (12 * (k % 2) if spread and k > 1 else 0))
        k += 1
    return sorted(set(out))[:n]


def lead(prev, sym, lo=nm('A2'), hi=nm('E5'), n=None):
    """voice-led voicing: for each previous note pick the nearest chord tone,
    keep notes distinct and within range. Returns sorted MIDI list."""
    want = pcs(sym)
    n = n or len(prev)
    cands = [m for m in range(lo, hi + 1) if m % 12 in want]
    out = []
    for p in sorted(prev)[:n]:
        best = min((c for c in cands if c not in out), key=lambda c: (abs(c - p), c), default=None)
        if best is not None:
            out.append(best)
    # guarantee root/3rd presence when space allows
    ch = chord(sym)
    third = [(ch['root'] + i) % 12 for i in ch['ints'] if i in (3, 4)]
    tc = [c for c in cands if c % 12 in third]
    if tc and not any(m % 12 in third for m in out) and out:
        mid = sorted(out)[len(out) // 2]
        repl = min((c for c in tc if c not in out), key=lambda c: abs(c - mid), default=None)
        if repl is not None:
            out[out.index(mid)] = repl
    return sorted(set(out))


def scale_notes(tonic, mode='major', lo=nm('C2'), hi=nm('C7')):
    if isinstance(tonic, str):
        t = (nm(tonic) if tonic[-1].isdigit() else _root(tonic)[0]) % 12
    else:
        t = tonic % 12
    iv = SCALES[mode]
    return [m for m in range(lo, hi + 1) if (m - t) % 12 in iv]


def snap(m, allowed):
    """nearest allowed MIDI note (list) to m."""
    return min(allowed, key=lambda a: (abs(a - m), a))


def degree(tonic, mode, d, octave=4):
    """scale degree (1-based, may exceed 7 / be <=0) -> MIDI"""
    iv = SCALES[mode]
    t = nm(tonic + str(octave)) if isinstance(tonic, str) and not tonic[-1].isdigit() else nm(tonic)
    k = d - 1
    return t + iv[k % len(iv)] + 12 * (k // len(iv))


# ----------------------------------------------------------------- motifs
def transpose(m, k):
    return [(x + k if x is not None else None) for x in m]


def invert(m, axis=None):
    axis = axis if axis is not None else next(x for x in m if x is not None)
    return [(2 * axis - x if x is not None else None) for x in m]


def retro(m):
    return list(reversed(m))


def rotate(m, k):
    k %= len(m)
    return m[k:] + m[:k]


def diatonic_shift(m, steps, allowed):
    """move each note by `steps` positions within an allowed note list."""
    al = sorted(allowed)
    out = []
    for x in m:
        if x is None:
            out.append(None)
            continue
        i = min(range(len(al)), key=lambda j: abs(al[j] - x))
        out.append(al[max(0, min(len(al) - 1, i + steps))])
    return out
