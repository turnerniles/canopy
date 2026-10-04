"""World builder: a JSON level description -> engine-ready adaptive music.

    python -m orchestra.world worlds/jungle_temple.json [--areas A,B] [--layers pad,bass] [--preview]

A *world* is one tempo/key grid with up to five areas (A–E, like Canopy). Each
area has a 16-bar progression (any chord symbols — see theory.py) and a set of
*layers*. Each layer is an instrument plus a pattern generator:

    pad       sustained, voice-led chords (merges repeated bars)
    comp      rhythmic chord hits        rhythm "x..x..x." (per bar)
    bass      bass line                  rhythm chars: x root, 5 fifth, 8 octave, a approach, - hold
    arp       chord-tone arpeggio        shape up|down|updown|motif, octaves
    ostinato  Oldfield-style cell        scale degrees over the bar, re-harmonised per chord
    motif     leitmotif                  notes like "A4 D5 E5 A5", re-pitched to fit each chord
    drums     kit                        {"x": inst, ...} + {"x": "x...x..."} per bar (UPPER = accent)
    scatter   creature / texture calls   insts list, density per bar, pitched to chord tones
    drone     root + fifth for the whole loop

Every layer has a tier (1 explore, 2 action, 3 climax), gain_db, pan, sends
(room names from fx.ROOMS, plus 'echo'), and optional "bars" to play only some.

Output (out/worlds/<id>/): <area>_<layer>.ogg loops of exactly bars*bar samples
(rendered circularly, so reverb/echo tails wrap seamlessly), <area>_<layer>_tail8 /
_tail16 ring-outs, stingers/*.ogg (collect notes, checkpoint, victory, death,
area transitions) and meta.json — the same schema as player/meta.json plus a
'cues' block (tiers, per-bar chord tones for in-key collectible chains).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from .core import SR, REGISTRY, nm, mtof, resolve
from . import theory as th
from .mix import Session
from . import fx

AREAS = 'ABCDE'


# ------------------------------------------------------------------ patterns
def _steps(rhythm):
    return [c for c in rhythm.replace(' ', '').replace('|', '')]


def _bars(layer, nbars):
    b = layer.get('bars')
    return [x - 1 for x in b] if b else list(range(nbars))


def gen_pad(p, prog, layer, beat_bar, rng):
    lo = nm(layer.get('low', 'A2'))
    hi = nm(layer.get('high', 'E5'))
    nv = layer.get('voices', 4)
    prev = th.voice(prog[0], lo + 7, nv)
    bars = _bars(layer, len(prog))
    i = 0
    while i < len(prog):
        if i not in bars:
            i += 1
            continue
        j = i + 1
        while j < len(prog) and prog[j] == prog[i] and j in bars and layer.get('merge', True):
            j += 1
        v = th.lead(prev, prog[i], lo, hi, nv)
        for m in v:
            p.note(i * beat_bar, m, (j - i) * beat_bar * layer.get('legato', 1.0), layer.get('vel', 0.55))
        prev = v
        i = j


def gen_comp(p, prog, layer, beat_bar, rng):
    lo = nm(layer.get('low', 'C3'))
    hi = nm(layer.get('high', 'G5'))
    nv = layer.get('voices', 4)
    rh = _steps(layer.get('rhythm', 'x...x...x...x...'))
    step = beat_bar / len(rh)
    prev = th.voice(prog[0], lo + 5, nv)
    for b in _bars(layer, len(prog)):
        v = th.lead(prev, prog[b], lo, hi, nv)
        prev = v
        for k, c in enumerate(rh):
            if c in '.-':
                continue
            ln = 1
            while k + ln < len(rh) and rh[k + ln] == '-':
                ln += 1
            vel = layer.get('vel', 0.6) * (1.2 if c.isupper() else 1.0) * rng.uniform(0.9, 1.05)
            p.chord(b * beat_bar + k * step, v, ln * step * layer.get('gate', 0.8), vel, strum=layer.get('strum', 0.0))


def gen_bass(p, prog, layer, beat_bar, rng):
    lo = nm(layer.get('low', 'E1'))
    rh = _steps(layer.get('rhythm', 'x.......x.......'))
    step = beat_bar / len(rh)
    for b in _bars(layer, len(prog)):
        root = th.bass_note(prog[b], lo, lo + 11)
        nxt = th.bass_note(prog[(b + 1) % len(prog)], lo, lo + 11)
        for k, c in enumerate(rh):
            if c in '.-':
                continue
            ln = 1
            while k + ln < len(rh) and rh[k + ln] == '-':
                ln += 1
            m = {'x': root, '5': root + 7, '8': root + 12, '3': root + (3 if 3 in th.chord(prog[b])['ints'] else 4),
                 'a': nxt + (1 if nxt < root else -1) if nxt != root else root + 7,
                 'b': root - 2 if (root - 2) % 12 in th.pcs(prog[b]) or True else root}.get(c.lower(), root)
            vel = layer.get('vel', 0.7) * (1.15 if c.isupper() else 1.0) * rng.uniform(0.92, 1.04)
            p.note(b * beat_bar + k * step, m, ln * step * layer.get('gate', 0.9), vel)


def _tones(sym, lo, hi):
    pcs = th.pcs(sym)
    return [m for m in range(lo, hi + 1) if m % 12 in pcs]


def gen_arp(p, prog, layer, beat_bar, rng):
    lo = nm(layer.get('low', 'D4'))
    hi = nm(layer.get('high', 'D6'))
    rh = _steps(layer.get('rhythm', 'x.x.x.x.x.x.x.x.'))
    step = beat_bar / len(rh)
    shape = layer.get('shape', 'up')
    idx = 0
    for b in _bars(layer, len(prog)):
        tones = _tones(prog[b], lo, hi)
        seq = tones if shape == 'up' else tones[::-1] if shape == 'down' else tones + tones[-2:0:-1]
        if shape == 'motif':
            seq = [tones[i % len(tones)] for i in layer.get('order', [0, 2, 1, 3, 2, 4, 3, 1])]
        for k, c in enumerate(rh):
            if c in '.-':
                continue
            vel = layer.get('vel', 0.55) * (1.2 if c.isupper() else 1.0) * rng.uniform(0.85, 1.05)
            p.note(b * beat_bar + k * step, seq[idx % len(seq)], step * layer.get('gate', 1.5), vel)
            idx += 1
        if layer.get('reset', True):
            idx = 0


def gen_ostinato(p, prog, layer, beat_bar, rng, key=None):
    """scale-degree cell (Oldfield): 'degrees' over 'rhythm'; each degree is
    snapped to the nearest tone of the current chord+scale so the cell keeps
    its contour while the harmony moves underneath."""
    tonic, mode = key
    base = nm(layer.get('base', tonic + '4'))
    scale = th.scale_notes(tonic, mode, base - 24, base + 24)
    degs = layer['degrees']
    rh = _steps(layer.get('rhythm', 'x' * len(degs)))
    step = beat_bar / len(rh)
    for b in _bars(layer, len(prog)):
        ct = th.pcs(prog[b])
        k2 = 0
        for k, c in enumerate(rh):
            if c in '.-':
                continue
            d = degs[k2 % len(degs)]
            k2 += 1
            m = th.degree(tonic, mode, d, octave=int(layer.get('base', tonic + '4')[-1]))
            if layer.get('fit', True) and m % 12 not in ct and rng.random() < layer.get('fit_prob', 0.6):
                m = th.snap(m, [x for x in scale if x % 12 in ct] or scale)
            vel = layer.get('vel', 0.6) * (1.15 if c.isupper() else 1.0) * rng.uniform(0.9, 1.05)
            p.note(b * beat_bar + k * step, m, step * layer.get('gate', 1.0), vel)


def gen_motif(p, prog, layer, beat_bar, rng):
    notes = [nm(x) if x not in ('.', '-') else x for x in layer['notes'].split()]
    step = layer.get('step', 0.5)
    for b in _bars(layer, len(prog)):
        allowed = _tones(prog[b], 36, 100)
        scale = th.scale_notes(layer.get('tonic', 'D'), layer.get('mode', 'lydian'), 36, 100)
        t = b * beat_bar + layer.get('offset', 0.0)
        for k, x in enumerate(notes):
            if x in ('.', '-'):
                continue
            ln = 1
            while k + ln < len(notes) and notes[k + ln] == '-':
                ln += 1
            m = x if x % 12 in th.pcs(prog[b]) or x in scale else th.snap(x, allowed)
            p.note(t + k * step, m, ln * step * 0.95, layer.get('vel', 0.6))


def gen_drums(p, prog, layer, beat_bar, rng):
    kit = layer['kit']
    for b in _bars(layer, len(prog)):
        for ch, rh in layer['rhythm'].items():
            steps = _steps(rh)
            step = beat_bar / len(steps)
            for k, c in enumerate(steps):
                if c in '.-':
                    continue
                vel = layer.get('vel', 0.7) * (1.25 if c.isupper() else 1.0) * rng.uniform(0.85, 1.05)
                if c in 'gG':
                    vel *= 0.45   # ghost
                pitch = layer.get('tune', {}).get(ch, 60)
                p.note(b * beat_bar + k * step, pitch, step, min(1.0, vel), inst=kit[ch])


def gen_scatter(p, prog, layer, beat_bar, rng):
    insts = layer['insts']
    lo = nm(layer.get('low', 'D5'))
    hi = nm(layer.get('high', 'D7'))
    for b in _bars(layer, len(prog)):
        for _ in range(rng.poisson(layer.get('density', 0.5))):
            inst = resolve(insts[rng.integers(len(insts))])
            info = REGISTRY[inst]
            tones = [m for m in _tones(prog[b], max(lo, info.lo), min(hi, info.hi))] or [info.lo]
            m = tones[rng.integers(len(tones))]
            t = b * beat_bar + rng.uniform(0, beat_bar) if not layer.get('grid') else \
                b * beat_bar + layer['grid'] * rng.integers(int(beat_bar / layer['grid']))
            p.note(t, m, layer.get('beats', 1.0) * rng.uniform(0.6, 1.4), layer.get('vel', 0.5) * rng.uniform(0.6, 1.0),
                   inst=inst, pan=float(rng.uniform(-0.8, 0.8)))


def gen_drone(p, prog, layer, beat_bar, rng):
    root = th.bass_note(prog[0], nm(layer.get('low', 'D2')), nm(layer.get('low', 'D2')) + 11)
    for m in [root, root + 7] + ([root + 12] if layer.get('octave') else []):
        p.note(0, m, beat_bar * len(prog), layer.get('vel', 0.5))


GENERATORS = dict(pad=gen_pad, comp=gen_comp, bass=gen_bass, arp=gen_arp, ostinato=gen_ostinato,
                  motif=gen_motif, drums=gen_drums, scatter=gen_scatter, drone=gen_drone)


# ------------------------------------------------------------------ stingers
def stinger_session(world, kind, chord_sym, tonic):
    """short in-key cues. Downbeat (the moment the cue 'lands') is at 1 bar."""
    st = world.get('stingers', {})
    bpm = world['bpm']
    s = Session(bpm=bpm)
    s.master.update(comp=None, lufs=-16.0, tape=None)
    bb = world.get('beats_per_bar', 4)
    tones = _tones(chord_sym, nm('A4'), nm('A6'))
    cfg = st.get(kind, {})
    inst = cfg.get('inst')
    if kind == 'collect':
        p = s.part('c', inst or 'mallet.kalimba', sends={'hall': 0.2})
        p.note(0, tones[cfg.get('index', 0) % len(tones)], 0.5, 0.8)
    elif kind == 'checkpoint':
        p = s.part('c', inst or 'mallet.kalimba', sends={'hall': 0.3, 'echo': 0.15})
        for i, m in enumerate(tones[:5]):
            p.note(bb - 1 + i * 0.25, m, 0.6, 0.6 + 0.08 * i)
    elif kind == 'victory':
        p = s.part('v', inst or 'brass.french_horn' if 'brass.french_horn' in REGISTRY else list(REGISTRY)[0],
                   sends={'hall': 0.35})
        root = nm(tonic + '4')
        fan = [(0, 0, 0.5), (0.5, 4, 0.5), (1, 7, 0.5), (1.5, 12, 2.5), (4.5, 11, 0.5), (5, 12, 3)]
        for t, iv, d in fan:
            p.note(bb + t - 1, root + iv, d, 0.75)
        if cfg.get('perc'):
            q = s.part('vp', cfg['perc'], sends={'hall': 0.3})
            for t in (bb - 1, bb + 0.5, bb + 1.5, bb + 4.5):
                q.note(t, 60, 0.5, 0.8)
    elif kind == 'death':
        p = s.part('d', inst or list(REGISTRY)[0], sends={'hall': 0.4, 'echo': 0.2})
        root = nm(tonic + '5')
        for i, iv in enumerate([0, -2, -5, -9, -12]):
            p.note(bb + i * 0.5, root + iv, 0.6 if i < 4 else 2.5, 0.6 - 0.06 * i)
    elif kind.startswith('to_'):
        p = s.part('t', cfg.get('inst') or inst or list(REGISTRY)[0], sends={'hall': 0.5})
        for i in range(8):
            p.note(i * bb / 8, tones[i % len(tones)], bb / 8 * 1.5, 0.3 + 0.08 * i)
    return s


# ------------------------------------------------------------------ build
def grammar_problems(prog, tonic):
    """Canopy's shared cadence rule: bar 1 rooted on the tonic, bars 8 and 16
    on the dominant — what lets any area (or world) follow any other."""
    t = th.PC[tonic[0]] + (1 if tonic[1:2] == '#' else -1 if tonic[1:2] == 'b' else 0)
    out = []
    if th.chord(prog[0])['root'] % 12 != t % 12:
        out.append(f'bar 1 ({prog[0]}) is not rooted on {tonic}')
    for b in (7, len(prog) - 1):
        if th.chord(prog[b])['root'] % 12 != (t + 7) % 12:
            out.append(f'bar {b + 1} ({prog[b]}) does not rest on the dominant')
    return out



def build(path, areas=None, layers=None, out_root=None, fmt='ogg', preview=False, quiet=False):
    import soundfile as sf
    W = json.load(open(path))
    wid = W['id']
    sr = W.get('sr', SR)
    bb = W.get('beats_per_bar', 4)
    nbars = W.get('bars', 16)
    bpm = W['bpm']
    spb = sr * 60 / bpm
    if abs(spb - round(spb)) > 1e-9:
        raise ValueError(f'{bpm} BPM at {sr} Hz does not give integer samples per beat; pick another tempo')
    spb = int(round(spb))
    N = spb * bb * nbars
    tail = W.get('tail_s', 9.0)
    T = int(tail * sr)
    out = out_root or os.path.join(os.path.dirname(os.path.abspath(path)), '..', 'out', 'worlds', wid)
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    tonic, mode = W.get('key', 'D'), W.get('mode', 'major')
    meta = dict(id=wid, name=W.get('name', wid), sr=sr, bpm=bpm, beats_per_bar=bb, bars=nbars, loop=N, bar=spb * bb,
                beat=spb, tail=T, key=f'{tonic} {mode}', stems={}, sections={}, beds=[], jungle=[], music=[],
                cues=dict(tiers={}, chord_tones={}, instruments={}))
    beds = set()
    all_insts = set()
    auto = W.get('master_gain_db', 'auto') == 'auto'
    gain = -12.0 if auto else float(W['master_gain_db'])
    pending = []          # (path, array) written after the shared gain is known
    area_peaks = []
    for a in (areas or [x for x in AREAS if x in W['areas']]):
        A = W['areas'][a]
        prog = A['progression']
        if len(prog) != nbars:
            raise ValueError(f'area {a}: progression must have {nbars} chords')
        for msg in grammar_problems(prog, tonic):
            print(f'  ! area {a}: {msg}', file=sys.stderr)
        meta['sections'][a] = dict(name=A.get('name', a), mood=A.get('mood', ''), chords=prog)
        meta['stems'][a] = []
        meta['cues']['chord_tones'][a] = [_tones(c, nm('A4'), nm('A6')) for c in prog]
        meta['cues']['tiers'][a] = {}
        mix_preview = np.zeros((2, N), np.float32)
        for lname, L in A['layers'].items():
            if layers and lname not in layers:
                continue
            rng = np.random.default_rng(abs(hash((wid, a, lname))) % (2 ** 32) if False else
                                        sum(map(ord, wid + a + lname)) * 7919)
            s = Session(bpm=bpm, sr=sr)
            s.echo.update(W.get('echo', {}))
            inst = resolve(L.get('inst') or (list(L.get('kit', {}).values()) or L.get('insts', [None]))[0])
            sends = dict(L.get('sends', {A.get('room', 'canopy'): 0.25}))
            p = s.part(lname, inst, pan=L.get('pan', 0.0), gain_db=L.get('gain_db', 0.0) + gain, sends=sends,
                       hpf=L.get('hpf', 0.0), lpf=L.get('lpf', 0.0), width=L.get('width', 0.0),
                       swing=L.get('swing', 0.0) * 0.5, humanize=L.get('humanize', 0.004))
            g = GENERATORS[L['pattern']]
            if L['pattern'] == 'ostinato':
                g(p, prog, L, bb, rng, key=(tonic, mode))
            else:
                g(p, prog, L, bb, rng)
            all_insts.update(p.instruments())
            # full pass: everything, then fold the over-length render onto the loop
            # render with a short pre-roll: humanised notes on the first downbeat may start
            # a few ms before sample 0, and in a loop that sound belongs to the previous pass
            P = int(0.2 * sr)
            buses = s.render_parts(t_end=(N + T) / sr + 2.0, t_offset=-P / sr)
            full = s.wet(buses, buses['n'])
            body = full[:, P:]
            loop = body[:, :N].copy()
            k = N
            while k < body.shape[1]:
                seg = body[:, k:k + N]
                loop[:, :seg.shape[1]] += seg
                k += N
            loop[:, N - P:] += full[:, :P]
            tail16 = body[:, N:N + T]
            buses8 = s.render_parts(t_end=(N + T) / sr + 2.0, onset_before=(N // 2) / sr, t_offset=-P / sr)
            full8 = s.wet(buses8, buses8['n'])
            tail8 = full8[:, P + N // 2:P + N // 2 + T]
            pk = float(np.max(np.abs(loop)))
            end_lvl = float(np.max(np.abs(tail16[:, -int(0.1 * sr):]))) if tail16.shape[1] >= T else 0.0
            if pk > 0 and end_lvl > pk * 1e-3:
                print(f'  ! {a}_{lname}: tail16 still ringing at {tail_s:.0f} s ({20 * np.log10(end_lvl / pk):.0f} dB) '
                      f'— raise "tail_s"', file=sys.stderr)
            if pk > 0.99:
                print(f'  ! {a}_{lname} peaks at {pk:.2f} — lower gain_db', file=sys.stderr)
            base = f'{a}_{lname}'
            pending += [(os.path.join(out, base), loop), (os.path.join(out, base + '_tail16'), _pad(tail16, T)),
                        (os.path.join(out, base + '_tail8'), _pad(tail8, T))]
            meta['stems'][a].append(lname)
            meta['cues']['tiers'][a][lname] = L.get('tier', 1)
            meta['cues']['instruments'][base] = p.instruments()
            if L.get('bed'):
                beds.add(lname)
            mix_preview += loop
            if not quiet:
                print(f'  {base:24s} tier {L.get("tier", 1)}  peak {pk:.2f}  {", ".join(p.instruments())}', flush=True)
        area_peaks.append(float(np.max(np.abs(mix_preview))))
        if preview:
            pv = np.concatenate([mix_preview, mix_preview], axis=1)
            sf.write(os.path.join(out, f'{a}__preview.wav'), (pv / max(1.0, np.max(np.abs(pv)) / 0.9)).T, sr,
                     subtype='PCM_16')
    # one gain for every stem of every area: the loudest full mix peaks at -3 dBFS
    g = (10 ** (-3 / 20) / max(max(area_peaks), 1e-6)) if auto else 1.0
    for path_, arr in pending:
        _write(sf, path_, arr * g, sr, fmt)
    meta['gain_db'] = round(20 * np.log10(g), 2) + gain
    if not quiet:
        print(f'  stems written with shared gain {20 * np.log10(g):+.1f} dB (area mix peaks {[round(p, 2) for p in area_peaks]})')
    del pending
    # stingers in key
    sdir = os.path.join(out, 'stingers')
    os.makedirs(sdir, exist_ok=True)
    first = W['areas'][(areas or [x for x in AREAS if x in W['areas']])[0]]['progression'][0]
    kinds = ['checkpoint', 'victory', 'death'] + [f'to_{a}' for a in (areas or W['areas'])]
    for kind in kinds:
        if kind.startswith('to_') and kind[3:] not in W['areas']:
            continue
        s = stinger_session(W, kind, first if not kind.startswith('to_') else W['areas'][kind[3:]]['progression'][0], tonic)
        x, _ = s.render(progress=False)
        _write(sf, os.path.join(sdir, kind), x, sr, fmt)
    # one collect note per chord tone of the first chord as a pitched set; engines can retune
    for i, m in enumerate(_tones(first, nm('A4'), nm('A6'))):
        s = stinger_session(dict(W, stingers=dict(W.get('stingers', {}), collect=dict(W.get('stingers', {}).get('collect', {}), index=i))),
                            'collect', first, tonic)
        x, _ = s.render(progress=False)
        _write(sf, os.path.join(sdir, f'collect_{m}'), x, sr, fmt)
    meta['beds'] = sorted(beds)
    meta['stingers_downbeat'] = spb * bb
    meta['instruments'] = sorted(all_insts)
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    if not quiet:
        print(f'{wid}: {sum(len(v) for v in meta["stems"].values())} loops, {len(all_insts)} instruments -> {out}')
    return meta


def _pad(x, T):
    if x.shape[1] < T:
        x = np.concatenate([x, np.zeros((2, T - x.shape[1]), np.float32)], axis=1)
    return x


def _write(sf, base, x, sr, fmt):
    x = np.clip(x, -1, 1).T
    if fmt == 'ogg':
        try:
            sf.write(base + '.ogg', x, sr, format='OGG', subtype='VORBIS')
            return
        except Exception:
            pass
    sf.write(base + '.wav', x, sr, subtype='PCM_24')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opt = {a.split('=')[0]: (a.split('=')[1] if '=' in a else True) for a in sys.argv[1:] if a.startswith('--')}
    build(args[0], areas=opt.get('--areas', '').split(',') if opt.get('--areas') else None,
          layers=opt.get('--layers', '').split(',') if opt.get('--layers') else None,
          fmt='wav' if opt.get('--wav') else 'ogg', preview=bool(opt.get('--preview')))
