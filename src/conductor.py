"""
Canopy conductor: phrase-aware layer changes + a composed demo journey.

Rules (shared with the web player, exported to rules.json):
  * layer changes happen only on 4-bar phrase boundaries (bars 1, 5, 9, 13);
    harmonic layers (keys, pad, bass) may only LEAVE on 8-bar boundaries.
  * melodic layers (melody, kalimba, ornament) know their pickups: an entry at
    bar 9 starts on the bar-8 pickup; an exit stops just before it.
  * leaving at bar 9 / bar 17 hands the layer to its rendered ring-out tail.
  * section changes happen on 8- or 16-bar boundaries; the old section is cut
    on the downbeat and rings out through its tails; the new section's first
    pass is cleaned of its own wrapped tail (phantom cancellation).
  * beds (amb, water) crossfade over ~3 s instead.
"""
import json, os, sys
import numpy as np
import soundfile as sf
from synth import SR, SPB, BAR, N, nm, HALF
from score import MELODY, SECTION_INFO
from render import STEMS

RHYTHMIC = {'woods', 'shaker', 'drums', 'bass', 'marimba'}
HARMONIC = {'keys', 'pad', 'bass'}
MELODIC = {'melody', 'kalimba', 'ornament'}
BEDS = {'amb', 'water'}
TL = 9 * SR


def _events_for(sec, stem):
    """(start_beat_abs, end_beat_abs) for pickup analysis (beats from loop start)."""
    ev = []
    if stem == 'melody':
        for (b, beat, d, n) in MELODY[sec]:
            s = (b - 1) * 4 + beat - 1
            ev.append((s, s + d))
    elif stem == 'kalimba' and sec == 'A':
        from score import KALIMBA_FRAG_BARS
        for (b, beat, d, n) in MELODY[sec]:
            if b in KALIMBA_FRAG_BARS['A']:
                s = (b - 1) * 4 + beat - 1
                ev.append((s, s + d))
    elif stem == 'kalimba' and sec == 'E':
        for b in range(2, 17, 2):
            for j in range(4):
                s = (b - 1) * 4 + 3 + j * 0.25
                ev.append((s, s + 0.25))
    elif stem == 'ornament':
        orn = {'A': [(8, 3.0, 1.75), (16, 2.5, 2.5), (4, 4.0, 0.5), (12, 4.0, 0.4)],
               'B': [(8, 3.5, 1.5), (3, 4.0, 0.3)], 'C': [(4, 3.0, 1.0), (7, 4.0, 0.3)],
               'D': [(7, 3.0, 1.0)], 'E': [(16, 3.5, 1.5), (4, 4.0, .3), (8, 4.0, .3), (12, 4.0, .3)]}[sec]
        for (b, beat, d) in orn:
            s = (b - 1) * 4 + beat - 1
            ev.append((s, s + d))
    return ev


def leadins():
    """beats of pickup before each boundary (bars 1,5,9,13) for melodic stems."""
    out = {}
    for sec in 'ABCDE':
        out[sec] = {}
        for stem in STEMS:
            li = [0.0, 0.0, 0.0, 0.0]
            if stem in MELODIC:
                ev = _events_for(sec, stem)
                for k, bnd in enumerate([64, 16, 32, 48]):     # bar 17(=1), 5, 9, 13 in beats
                    best = 0.0
                    for (s, e) in ev:
                        if bnd - 1.01 <= s < bnd - 0.01 and e >= bnd - 0.02:
                            best = max(best, bnd - s)
                    # extend through contiguous pickup notes
                    changed = True
                    while changed and best:
                        changed = False
                        for (s, e) in ev:
                            if abs(e - (bnd - best)) < 0.02 and bnd - s <= 1.51 and bnd - s > best:
                                best = bnd - s
                                changed = True
                    li[k] = round(best, 3)
            out[sec][stem] = li
    return out


LEADIN = leadins()


def gains_table():
    return json.load(open('out/gains_pre.json'))


class Mixer:
    """offline renderer of an adaptive performance."""

    def __init__(self, total_bars, gains):
        self.out = np.zeros((2, total_bars * BAR + TL + BAR))
        self.g = gains
        self.cache = {}

    def buf(self, key):
        if key not in self.cache:
            self.cache[key] = np.load(f'out/raw/{key}.npy').astype(np.float64)
        return self.cache[key]

    def add(self, y, at, env=None):
        j = min(self.out.shape[1], at + y.shape[1])
        if j <= at:
            return
        seg = y[:, :j - at]
        if env is not None:
            seg = seg * env[:j - at]
        self.out[:, at:j] += seg


def ramp(n, a, b, shape='lin'):
    x = np.linspace(0, 1, max(1, n))
    if shape == 'cos':
        x = 0.5 - 0.5 * np.cos(np.pi * x)
    return a + (b - a) * x


def perform(blocks, path, stingers=(), outro_fade_bars=0, gains=None, bed_xfade=3.0, beds=None):
    """blocks: list of (section, [phrase_gain_dict x4]) or (section, phrases, {stem: variant}).
    Each block is one 16-bar pass of its section. A variant swaps the file used for a
    stem (e.g. amb -> amb3) and is treated like a change of section for that stem."""
    g = gains or gains_table()
    BEDSET = set(beds) if beds is not None else BEDS
    blocks = [b if len(b) == 3 else (b[0], b[1], {}) for b in blocks]
    stems_all = sorted({k for (_, phs, _) in blocks for ph in phs for k in ph})
    nb = 16 * len(blocks)
    mx = Mixer(nb, g)
    total = nb * BAR
    XF = int(bed_xfade * SR)
    for bi, (sec, phrases, var) in enumerate(blocks):
        t0 = bi * 16 * BAR
        prev = blocks[bi - 1] if bi > 0 else None
        nxt = blocks[bi + 1] if bi + 1 < len(blocks) else None
        for stem in stems_all:
            vname = var.get(stem, stem)
            new_section = prev is None or prev[0] != sec or prev[2].get(stem, stem) != vname
            leaving = nxt is None or nxt[0] != sec or nxt[2].get(stem, stem) != vname
            key = f'{sec}_{vname}'
            tg = [ph.get(stem, 0.0) for ph in phrases]
            prev_g = (prev[1][3].get(stem, 0.0) if (prev and not new_section) else 0.0)
            if max(tg) == 0 and prev_g == 0:
                continue
            loop = mx.buf(key) * g[key]
            t16 = mx.buf(key + '_t16') * g[key]
            t8 = mx.buf(key + '_t8') * g[key]
            LI = LEADIN.get(sec, {}).get(stem, [0, 0, 0, 0])
            env = np.zeros(16 * BAR)
            cur = prev_g
            for k in range(4):
                a = k * 4 * BAR
                gk = tg[k]
                li = int(LI[k] * SPB)
                if k == 0 and li and not new_section:
                    cur = gk
                if gk > cur:            # ---- entry
                    if stem in BEDSET or stem == 'pad':
                        n = XF if stem in BEDSET else int(1.2 * SR)
                        env[a:a + n] = ramp(n, cur, gk, 'cos')[:len(env[a:a + n])]
                        env[a + n:] = gk
                    else:
                        if li and k > 0:
                            s0 = a - li - int(0.05 * SPB)
                            env[s0:a] = gk
                        n = int(0.02 * SR)
                        env[a:a + n] = ramp(n, cur, gk)
                        env[a + n:] = gk
                elif gk < cur:          # ---- exit
                    if stem in BEDSET:
                        n = XF
                        env[a:a + n] = ramp(n, cur, gk, 'cos')[:len(env[a:a + n])]
                        env[a + n:] = gk
                    elif li == 0 and k in (0, 2):
                        n = int(0.025 * SR)
                        env[a:a + n] = ramp(n, cur, gk)
                        env[a + n:] = gk
                        mx.add((t16 if k == 0 else t8) * (cur - gk), t0 + a)
                    else:
                        n = int(0.15 * SR)
                        s0 = max(0, a - li - n) if li else a
                        env[s0:s0 + n] = ramp(n, cur, gk)
                        env[s0 + n:] = gk
                else:
                    env[a:] = gk
                cur = gk
            if leaving and cur > 0:
                if stem in BEDSET:
                    ext = loop[:, :XF] * ramp(XF, cur, 0, 'cos')
                    mx.add(ext, t0 + 16 * BAR)
                else:
                    mx.add(t16 * cur, t0 + 16 * BAR)
            body = loop * env
            if new_section and stem not in BEDSET:
                body[:, :TL] -= t16 * env[:TL]
            if new_section and stem in BEDSET and bi > 0:
                body[:, :XF] *= ramp(XF, 0, 1, 'cos')
            if nxt is not None and not leaving:
                ng = nxt[1][0].get(stem, 0.0)
                li = int(LI[0] * SPB)
                if ng > cur and li:
                    s_ = 16 * BAR - li - int(0.05 * SPB)
                    body[:, s_:] = loop[:, s_:] * ng
                if ng < cur and li:
                    n = int(0.15 * SR)
                    s_ = 16 * BAR - li - n
                    body[:, s_:s_ + n] = loop[:, s_:s_ + n] * ramp(n, cur, ng)
                    body[:, s_ + n:] = loop[:, s_ + n:] * ng
            mx.add(body, t0)
    for (name, bar_abs, gain) in stingers:
        y = mx.buf('st_' + name) * gain * g['st_' + name]
        mx.add(y, (bar_abs - 1) * BAR - BAR)      # file anchor is 1 bar in; bar_abs is 1-based
    y = mx.out[:, :total + TL]
    if outro_fade_bars:
        n = outro_fade_bars * BAR
        y[:, total - n:total] *= ramp(n, 1, 0, 'cos')
        y[:, total:] = 0
        y = y[:, :total + SR]
    pk = np.max(np.abs(y))
    y *= 10 ** (-1.0 / 20) / max(pk, 10 ** (-1.0 / 20))
    sf.write(path, y.T, SR, subtype='PCM_24')
    return y


# ----------------------------------------------------------- the journey ---
F = 1.0
ALL = {s: 1.0 for s in STEMS}


def P(**kw):
    return dict(kw)


JOURNEY = [
    ('A', [P(amb=1.2, water=.9, woods=.7),
           P(amb=1.1, water=.8, woods=.9, shaker=.7, bass=.8),
           P(amb=1, water=.7, woods=1, shaker=.8, bass=.9, keys=.9),
           P(amb=1, water=.6, woods=1, shaker=.8, bass=.9, keys=.9, pad=.6, kalimba=1)]),
    ('A', [P(amb=.9, water=.5, woods=1, shaker=.9, bass=1, keys=1, pad=.7, kalimba=1),
           P(amb=.9, water=.5, woods=1, shaker=.9, bass=1, keys=1, pad=.7, kalimba=1, drums=.85),
           P(amb=.8, water=.5, woods=1, shaker=.9, bass=1, keys=1, pad=.8, kalimba=.5, drums=.9, marimba=.8, melody=1),
           P(amb=.8, water=.5, woods=1, shaker=.9, bass=1, keys=1, pad=.8, kalimba=.5, drums=.9, marimba=.8, melody=1, ornament=1)]),
    ('A', [P(amb=.7, water=.45, woods=.45, shaker=.9, bass=1, keys=.95, pad=.85, kalimba=.4, drums=1, marimba=.85, melody=1, ornament=1),
           P(amb=.7, water=.45, woods=.45, shaker=.9, bass=1, keys=.95, pad=.85, kalimba=.4, drums=1, marimba=.85, melody=1, ornament=1),
           P(amb=1.25, water=1.0, pad=.6, keys=.5, ornament=.8, kalimba=.75),
           P(amb=1.15, water=.9, pad=.65, keys=.55, ornament=.8, kalimba=.8, bass=.6, woods=.75, shaker=.45)]),
    ('B', [P(amb=1, water=1, pad=1, kalimba=1, ornament=1),
           P(amb=1, water=1, pad=1, kalimba=1, ornament=1, bass=.85, keys=.85, shaker=.6),
           P(amb=.9, water=1, pad=1, kalimba=.8, ornament=1, bass=.9, keys=.9, shaker=.7, melody=1, marimba=.7, drums=.6),
           P(amb=.9, water=1, pad=1, kalimba=.8, ornament=1, bass=.9, keys=.9, shaker=.7, melody=1, marimba=.8, drums=.7, woods=1)]),
    ('C', [P(amb=1, water=.9, pad=1, bass=1, keys=1, melody=1, ornament=1, woods=1),
           P(amb=1, water=.9, pad=1, bass=1, keys=1, melody=1, ornament=1, woods=1, marimba=.7, kalimba=.8),
           P(amb=.9, water=.8, pad=1, bass=1, keys=1, melody=1, ornament=1, woods=1, marimba=.8, kalimba=.8, drums=.8, shaker=.6),
           P(amb=.9, water=.8, pad=1, bass=1, keys=1, melody=1, ornament=1, woods=1, marimba=.9, kalimba=.8, drums=1, shaker=.8)]),
    ('D', [P(amb=1.2, water=1, pad=.8, woods=.9, melody=.8, kalimba=.8),
           P(amb=1, water=1, pad=1, woods=1, melody=.9, kalimba=1, ornament=1, bass=.85),
           P(amb=1, water=1, pad=1, woods=1, melody=.9, kalimba=1, ornament=1, bass=.9, keys=.9, drums=.8, marimba=.7),
           P(amb=1, water=1, pad=1, woods=1, melody=.9, kalimba=1, ornament=1, bass=.9, keys=.9, drums=.9, marimba=.8, shaker=1)]),
    ('E', [P(amb=.8, drums=1, shaker=1, bass=1, marimba=1, woods=1),
           P(amb=.8, drums=1, shaker=1, bass=1, marimba=1, woods=1, keys=.9, kalimba=1, pad=.7),
           P(amb=.8, drums=1, shaker=1, bass=1, marimba=1, woods=1, keys=.9, kalimba=.8, pad=.7, melody=1, water=.5),
           P(amb=.8, drums=1, shaker=1, bass=1, marimba=1, woods=1, keys=1, kalimba=.8, pad=.8, melody=1, water=.5, ornament=1)]),
    ('A', [P(amb=.7, water=.45, woods=.45, shaker=.9, bass=1, keys=.95, pad=.85, kalimba=.4, drums=1, marimba=.85, melody=1, ornament=1),
           P(amb=.7, water=.45, woods=.45, shaker=.9, bass=1, keys=.95, pad=.85, kalimba=.4, drums=1, marimba=.85, melody=1, ornament=1),
           P(amb=1.1, water=.8, woods=.8, shaker=.5, bass=.75, keys=.7, pad=.7, kalimba=.9),
           P(amb=1.25, water=.9, woods=.6, pad=.4, kalimba=.8)]),
    ('A', [P(amb=1.25, water=.9, woods=.5),
           P(amb=1.25, water=.9, woods=.35),
           P(amb=1.2, water=.9),
           P(amb=1.2, water=.9)]),
]
# (stinger, downbeat bar number from start (1-based), gain)
JOURNEY_STINGERS = [
    ('breath', 16 * 2 + 9, 1.0),     # block 3 drops to breathe at bar 9
    ('swell', 16 * 3 + 1, 1.0),      # -> river
    ('fill_logs', 16 * 4 + 1, 1.0),  # -> vista
    ('arrival', 16 * 4 + 1, 0.8),
    ('swell', 16 * 5 + 1, 0.8),      # -> ruins
    ('rise', 16 * 6 + 1, 1.0),       # -> vine run
    ('arrival', 16 * 7 + 1, 0.9),    # -> home
    ('breath', 16 * 7 + 9, 1.0),
    ('pickup', 16 * 1 + 1, 0.8),
]

if __name__ == '__main__':
    os.makedirs('out/mix', exist_ok=True)
    json.dump(LEADIN, open('out/leadins.json', 'w'))
    for sec in 'ABCDE':
        print(sec, {k: v for k, v in LEADIN[sec].items() if any(v)})
    if 'journey' in sys.argv:
        y = perform(JOURNEY, 'out/mix/canopy_journey.wav', JOURNEY_STINGERS, outro_fade_bars=12)
        print('journey', y.shape[1] / SR, 's')
