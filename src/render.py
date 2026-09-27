"""
Canopy — arranger & stem renderer.

Stems (identical set for every section, all 16 bars, all phase-locked):
  amb      jungle ambience (insects, distant birds, leaves, frogs)
  water    environmental water texture (brook bubbles, waterfall-as-pad, tuned drips)
  woods    sparse tuned log drums / wood knocks ("the forest keeps time")
  shaker   light percussion (shaker, rain-shaker, seed rattles)
  drums    full percussion (soft hand drums, felt kick, frame drum, toms)
  bass     rounded melodic bass
  keys     warm tine electric piano harmony
  pad      soft analog pad / drones
  marimba  rhythmic mallet ostinato (intensity layer)
  kalimba  motif fragments & water arpeggios (discovery layer)
  melody   main theme (wooden flute; ocarina in the ruins)
  ornament bird-like flute calls, tuned bird whistles, bells
"""
import os, sys, json, time
import numpy as np
from synth import *
from synth import HALF
from score import HARMONY, MELODY, BASS, KALIMBA_FRAG_BARS, SECTION_INFO

STEMS = ['amb', 'water', 'woods', 'shaker', 'drums', 'bass', 'keys', 'pad',
         'marimba', 'kalimba', 'melody', 'ornament']

# ---------------------------------------------------------------- rooms ----
IR_SPECS = {
    'canopy': dict(rt60=1.9, predelay=0.018, dark=8500, er=12, er_spread=0.05, seed=11),
    'water':  dict(rt60=3.6, predelay=0.025, high_mul=0.75, dark=12000, seed=12),
    'vista':  dict(rt60=5.0, predelay=0.035, dark=9500, low_mul=1.1, seed=13),
    'ruins':  dict(rt60=3.4, predelay=0.012, er=22, er_spread=0.09, er_gain=1.4, dark=5200, low_mul=1.5, seed=14),
    'air':    dict(rt60=7.5, predelay=0.05, dark=7000, seed=15, build=0.2),   # huge distant space for ambience
    'room':   dict(rt60=0.7, predelay=0.006, dark=10000, er=14, er_spread=0.03, seed=16),
}
_IRH = {}


def irH(name):
    if name not in _IRH:
        ir = make_ir(**IR_SPECS[name])
        _IRH[name] = [sfft.rfft(ir[c], n=N, workers=2) for c in range(2)]
    return _IRH[name]


SEC = {
    'A': dict(room='canopy', swing=0.14, echo=(0.75, 1.0, 0.33, 3200)),
    'B': dict(room='water', swing=0.07, echo=(0.75, 1.5, 0.45, 4200)),
    'C': dict(room='vista', swing=0.05, echo=(1.5, 2.0, 0.40, 3000)),
    'D': dict(room='ruins', swing=0.0, echo=(1.25, 1.75, 0.45, 2200)),
    'E': dict(room='canopy', swing=0.12, echo=(0.75, 0.5, 0.28, 3500)),
}

# ------------------------------------------------------------ timing -------

def T(bar, beat=1.0):
    """bar 1..16, beat 1..4.99  -> sample"""
    return int(round(((bar - 1) * 4 + (beat - 1)) * SPB))


def P(bar, p16, swing=0.0):
    s = ((bar - 1) * 16 + p16) * S16
    if int(p16) % 2 == 1:
        s += swing * S16
    return int(round(s))


class Ctx:
    def __init__(self, sec, stem, seed):
        self.sec = sec
        self.cfg = SEC[sec]
        self.H = HARMONY[sec]
        self.rng = np.random.default_rng(seed)
        self.tr = Track(f'{sec}_{stem}')
        self.room = self.cfg['room']

    def jit(self, ms=4.0):
        return int(self.rng.normal(0, ms * 1e-3 * SR))

    def vj(self, v, amt=0.08):
        return float(np.clip(v * (1 + self.rng.normal(0, amt)), 0.02, 1.0))

    def p(self, bar, p16):
        return P(bar, p16, self.cfg['swing'])


SHELF = dict(melody=(2600, 3.0), keys=(3000, 2.0), kalimba=(3200, 2.0), marimba=(2600, 2.5),
             ornament=(3000, 2.0), pad=(3200, 2.0))


TL = 9 * SR                       # tail length (exit ring-out)
M = sfft.next_fast_len(BUF + 12 * SR)
_IRM = {}


def irM(name):
    if name not in _IRM:
        ir = make_ir(**IR_SPECS[name])
        _IRM[name] = [sfft.rfft(ir[c], n=M, workers=2) for c in range(2)]
    return _IRM[name]


def echo_H(delay_beats, fb, damp):
    k = np.arange(M // 2 + 1)
    w = TWOPI * k / M
    d = delay_beats * SPB
    zd = np.exp(-1j * w * d)
    a = np.exp(-TWOPI * damp / SR)
    lpf = (1 - a) / (1 - a * np.exp(-1j * w))
    return zd * lpf / (1 - fb * lpf * zd)


def _process(ctx, L, R, send, send_st, hpf, width):
    """linear processing of one pass (zero-padded FFTs long enough for all tails)."""
    pad = lambda x: np.concatenate([x, np.zeros(M - len(x))])
    L, R = pad(L), pad(R)
    sends = {k: pad(v) for k, v in send.items()}
    dl, dr, fb, damp = ctx.cfg['echo']
    for k, (bl, br) in send_st.items():
        eL = sfft.irfft(sfft.rfft(pad(bl), workers=2) * echo_H(dl, fb, damp), n=M, workers=2)
        eR = sfft.irfft(sfft.rfft(pad(br), workers=2) * echo_H(dr, fb, damp), n=M, workers=2)
        L += eL
        R += eR
        sends.setdefault(ctx.room, np.zeros(M))
        sends[ctx.room] += 0.35 * (eL + eR)
    for k, x in sends.items():
        X = sfft.rfft(x, workers=2)
        hl, hr = irM(k)
        L += sfft.irfft(X * hl, n=M, workers=2)
        R += sfft.irfft(X * hr, n=M, workers=2)
    if width != 1.0:
        Mi, S = 0.5 * (L + R), 0.5 * (L - R) * width
        L, R = Mi + S, Mi - S
    sos = signal.butter(2, hpf, btype='high', fs=SR, output='sos')
    L, R = signal.sosfilt(sos, L), signal.sosfilt(sos, R)
    stem = ctx.tr.name.split('_', 1)[1]
    if stem in SHELF:
        bb, aa = high_shelf_ba(*SHELF[stem])
        L, R = signal.lfilter(bb, aa, L), signal.lfilter(bb, aa, R)
    return np.stack([L, R])


def _fold_long(x):
    out = np.zeros(N)
    out[N - PRE:] += x[:PRE]
    rest = x[PRE:]
    for k in range(0, len(rest), N):
        seg = rest[k:k + N]
        out[:len(seg)] += seg
    return out


def _tail(x, at):
    t = x[:, PRE + at: PRE + at + TL].copy()
    f = int(1.5 * SR)
    t[:, -f:] *= np.linspace(1, 0, f) ** 2
    return t


def finalize(ctx, hpf=35.0, width=1.0):
    tr = ctx.tr
    full = _process(ctx, tr.L, tr.R, tr.send, tr.send_st, hpf, width)
    half = _process(ctx, tr.hL, tr.hR, tr.hsend, tr.hsend_st, hpf, width)
    loop = np.stack([_fold_long(full[0]), _fold_long(full[1])])
    return dict(loop=loop, tail16=_tail(full, N), tail8=_tail(half, HALF))


# ------------------------------------------------------ shared writers -----

def chord_at(H, bar, p16):
    c = H[bar - 1]
    if c['ep2'] is not None and p16 >= 8:
        return c['ep2']
    return c['ep']


def pool_for(H, bar, p16, lo, hi):
    v = chord_at(H, bar, p16)
    pcs = set(x % 12 for x in v) | {H[bar - 1]['bass'] % 12}
    return [m for m in range(lo, hi + 1) if m % 12 in pcs]


def scale_pcs(H, bar):
    c = H[bar - 1]
    return sorted(set(x % 12 for x in c['ep']) | set(x % 12 for x in c['pad']) | {c['bass'] % 12})


def melody_events(ctx, mel, vel=0.7, transpose=0, bars=None, art=True):
    """-> list of (start, dur, midi, vel) with articulation (slur steps, tongue leaps)."""
    ev = []
    m2 = [x for x in mel if bars is None or x[0] in bars]
    for i, (bar, beat, d, n) in enumerate(m2):
        s = T(bar, beat) + ctx.jit(6)
        if s < 0:
            s = 0
        if HALF - int(0.04 * SR) <= s < HALF:
            s = HALF
        dur = d * SPB
        midi = nm(n) + transpose
        slur = False
        if i + 1 < len(m2):
            nb, nbeat, nd, nn = m2[i + 1]
            if abs(T(nb, nbeat) - T(bar, beat + d)) < 10 and abs(nm(nn) - nm(n)) <= 2:
                slur = True
        if art and not slur:
            dur = max(dur * 0.9, dur - 0.035 * SR)
        v = vel * (0.86 + 0.14 * min(d, 2) / 2) * (1 + (midi - 76) * 0.006)
        ev.append((s, int(dur), midi, ctx.vj(v, 0.05)))
    return ev


def add_ep(ctx, bar, p16, d16, voicing, vel, strum=0.012, sends=None, pan_spread=0.35, bark=1.0, oct=0):
    s0 = ctx.p(bar, p16) + ctx.jit(5)
    nv = len(voicing)
    for i, m in enumerate(voicing):
        f = float(mtof(m + oct))
        y = epiano(f, d16 * S16 / SR, ctx.vj(vel * (0.9 + 0.1 * (i == nv - 1))), ctx.rng, bark=bark)
        pan = (i / max(1, nv - 1) - 0.5) * 2 * pan_spread
        ctx.tr.add(y, s0 + int(i * strum * SR), pan=pan, sends=sends)


def add_pad(ctx, cutoff=1300, attack=1.1, release=1.8, gain=1.0, sends=None, voices=3,
            width=0.8, lfo_cycles=2, lfo_depth=0.3, merge=True, air=0.0, key='pad', extra_top=0):
    H = ctx.H
    ph0 = ctx.rng.uniform(0, 6)
    lfo = lambda t: 1 + lfo_depth * np.sin(TWOPI * lfo_cycles * t * SR / N + ph0)
    b = 1
    while b <= 16:
        e = b
        while merge and e < 16 and H[e]['pad'] == H[b - 1]['pad']:
            e += 1
        notes = list(H[b - 1][key])
        if extra_top:
            notes = notes + [max(notes) + extra_top]
        dur = (e - b + 1) * BAR / SR
        st = pad_chord(notes, dur + 0.25, ctx.rng, attack=attack, release=release, voices=voices,
                       cutoff=cutoff, cutoff_lfo=lfo, t0=T(b) / SR, width=width, air=air)
        ctx.tr.add(st, T(b) - int(0.15 * SR), gain=gain, sends=sends)
        b = e + 1


def add_bass_events(ctx, events, vel=0.7, sends=None, glide=False, mwah=1.0, gain=1.0):
    prev = None
    for (bar, p16, d16, n) in events:
        m = nm(n) if isinstance(n, str) else n
        f = float(mtof(m))
        v = vel * (1.12 if p16 == 0 else 0.92)
        gf = float(mtof(prev)) if (glide and prev is not None and abs(prev - m) <= 5 and ctx.rng.random() < 0.5) else None
        y = bass_note(f, d16 * S16 / SR * 0.94, ctx.vj(v, 0.06), ctx.rng, glide_from=gf, mwah=mwah)
        ctx.tr.add(y, ctx.p(bar, p16) + ctx.jit(4), pan=0.0, gain=gain, sends=sends)
        prev = m


def gen_bass(H, mode):
    """generic bass lines built from the harmony"""
    ev = []
    for b in range(1, 17):
        R = H[b - 1]['bass']
        nxt = H[b % 16]['bass']
        pcs = scale_pcs(H, b)
        fifth = R + 7 if (R + 7) % 12 in pcs else R + 5
        ninth = R + 14 if (R + 2) % 12 in pcs else R + 12
        third = R + 3 if (R + 3) % 12 in pcs else R + 4
        appr = nxt + (2 if nxt < R else -1)
        while appr - R > 9:
            appr -= 12
        while R - appr > 9:
            appr += 12
        same_next = (H[b % 16]['bass'] == R)
        if mode == 'long':           # vista
            ev += [(b, 0, 12, R), (b, 12, 4, fifth if same_next else appr)]
        elif mode == 'water':
            if same_next:
                ev += [(b, 0, 10, R), (b, 10, 6, fifth)]
            else:
                ev += [(b, 0, 8, ninth - 12 if ninth - 12 > R else third + 12), (b, 8, 6, fifth), (b, 14, 2, appr)]
        elif mode == 'drone':        # ruins
            if b % 2 == 1:
                ev += [(b, 0, 28, R)]
            else:
                ev += [(b, 12, 4, fifth - 12 if fifth - 12 >= 28 else fifth)]
        elif mode == 'drive':        # traversal
            o = R + 12
            ev += [(b, 0, 3, R), (b, 3, 3, R), (b, 6, 2, fifth), (b, 8, 2, o), (b, 10, 2, R),
                   (b, 12, 2, fifth), (b, 14, 2, appr)]
    return ev


def put_perc(ctx, sig, bar, p16, pan=0.0, gain=1.0, sends=None, jit=3.0):
    ctx.tr.add(sig, ctx.p(bar, p16) + ctx.jit(jit), pan=pan, gain=gain, sends=sends)



def periodic_drone(notes, rng, cutoff=700, voices=3, detune=0.08, lfo_cycles=2, width=0.6, level=0.22):
    Tl = N / SR
    i = np.arange(N)
    t = i / SR
    fc = cutoff * (1 + 0.35 * np.sin(TWOPI * lfo_cycles * i / N + 0.7))
    out = np.zeros((2, N))
    for m in notes:
        for v in range(voices):
            det = (v - (voices - 1) / 2) * detune
            f = round(float(mtof(m)) * 2 ** (det / 12) * Tl) / Tl
            y = np.zeros(N)
            ph0 = rng.uniform(0, TWOPI)
            for k in range(1, 30):
                if k * f > 4 * cutoff:
                    break
                a = (1.0 / k) / np.sqrt(1 + (k * f / fc) ** 4)
                y += a * np.sin(TWOPI * k * f * t + k * ph0)
            p = (v - (voices - 1) / 2) / max(1, (voices - 1) / 2) * width
            gl, gr = pan_gains(p)
            out[0] += y * gl
            out[1] += y * gr
    return out * level / np.sqrt(len(notes) * voices)

# ------------------------------------------------------------- ambience ----

def distant_birds(ctx, count, kinds=('chirp', 'whistle', 'trill', 'coo'), level=1.0, room='air', lowpass=True):
    rng = ctx.rng
    for _ in range(count):
        kind = rng.choice(kinds)
        s = int(rng.uniform(0, N))
        pan = rng.uniform(-0.9, 0.9)
        if kind == 'chirp':
            f0 = rng.uniform(2600, 4600)
            reps = rng.integers(2, 6)
            for r in range(reps):
                pts = [(0, f0 * 0.8, 0), (0.012, f0, 1), (0.05, f0 * rng.uniform(1.1, 1.35), 0.6), (0.07, f0 * 1.2, 0)]
                y = bird_call(pts, rng, vel=0.25 * level)
                ctx.tr.add(lp(y, 6500) if lowpass else y, s + int(r * rng.uniform(0.09, 0.14) * SR), pan=pan,
                           sends={room: 0.9})
        elif kind == 'whistle':
            f0 = rng.uniform(1700, 2900)
            f1 = f0 * rng.choice([0.84, 1.19, 1.26, 0.79])
            pts = [(0, f0 * 0.95, 0), (0.03, f0, 1), (0.22, f0, 0.8), (0.28, f1, 0.9), (0.5, f1 * 0.98, 0)]
            y = bird_call(pts, rng, vel=0.2 * level)
            ctx.tr.add(lp(y, 5000), s, pan=pan, sends={room: 1.0})
        elif kind == 'trill':
            f0 = rng.uniform(3200, 5200)
            L = rng.uniform(0.3, 0.8)
            pts = [(0, f0, 0), (0.02, f0, 1), (L, f0 * rng.uniform(0.8, 1.05), 0.7), (L + 0.03, f0, 0)]
            y = bird_call(pts, rng, trill_hz=rng.uniform(18, 32), trill_depth=0.06, vel=0.12 * level)
            ctx.tr.add(lp(y, 7000), s, pan=pan, sends={room: 0.9})
        elif kind == 'coo':
            f0 = rng.uniform(480, 640)
            for r in range(rng.integers(2, 4)):
                pts = [(0, f0 * 0.9, 0), (0.06, f0, 1), (0.25, f0 * 0.97, 0.8), (0.4, f0 * 0.9, 0)]
                y = bird_call(pts, rng, harm=0.3, vel=0.18 * level)
                ctx.tr.add(lp(y, 2000), s + int(r * 0.55 * SR), pan=pan, sends={room: 0.8})
        elif kind == 'raptor':   # long descending cry, far away
            f0 = rng.uniform(2200, 2900)
            pts = [(0, f0, 0), (0.05, f0 * 1.05, 1), (0.9, f0 * 0.72, 0.7), (1.2, f0 * 0.65, 0)]
            y = bird_call(pts, rng, trill_hz=11, trill_depth=0.01, harm=0.25, vel=0.15 * level)
            ctx.tr.add(lp(y, 5000), s, pan=pan, sends={room: 1.2})


def frogs(ctx, count, level=1.0):
    rng = ctx.rng
    for _ in range(count):
        s = int(rng.uniform(0, N))
        f0 = rng.uniform(260, 420)
        pan = rng.uniform(-0.8, 0.8)
        for r in range(rng.integers(1, 4)):
            n = int(0.22 * SR)
            t = tvec(n)
            y = np.sin(TWOPI * f0 * t + 2 * np.sin(TWOPI * f0 * 2 * t)) * (0.5 + 0.5 * np.sin(TWOPI * 32 * t)) * np.hanning(n)
            ctx.tr.add(lp(y, 2500) * 0.12 * level, s + int(r * 0.35 * SR), pan=pan, sends={'air': 0.6})


def leaves(ctx, level=1.0, lo=250, hi=1800, cycles=3):
    rng = ctx.rng
    x = pink(N, rng)
    x = circ_filter(x, sos=signal.butter(2, [lo, hi], btype='band', fs=SR, output='sos'))
    i = np.arange(N) / N
    g = 0.4 + 0.6 * (0.5 + 0.5 * np.sin(TWOPI * cycles * i + rng.uniform(0, 6))) ** 2
    g *= 0.8 + 0.2 * np.sin(TWOPI * 11 * i + rng.uniform(0, 6))
    y = x * g * level
    y2 = np.roll(y, int(0.37 * SR))
    ctx.tr.add(np.stack([y, y2]), 0)


def bed(ctx, sig_mono, spread=0.37, gain=1.0, sends=None):
    """add a circular mono bed as decorrelated stereo."""
    y2 = np.roll(sig_mono, int(spread * SR))
    ctx.tr.add(np.stack([sig_mono, y2]), 0, gain=gain, sends=sends)


def brook(ctx, density=120, level=1.0, fmin=450, fmax=2600, room=None):
    rng = ctx.rng
    count = int(density * N / SR)
    for _ in range(count):
        f0 = np.exp(rng.uniform(np.log(fmin), np.log(fmax)))
        tau = rng.uniform(0.004, 0.018) * (900 / f0) ** 0.3
        y = bubble(f0, tau, rng, rise=rng.uniform(0.08, 0.3), vel=rng.uniform(0.2, 1.0) * 0.12 * level)
        ctx.tr.add(y, int(rng.uniform(0, N)), pan=rng.uniform(-0.7, 0.7),
                   sends={room or ctx.room: 0.25})
    # the hiss of moving water
    x = pink(N, rng)
    x = circ_filter(x, sos=signal.butter(2, [500, 4500], btype='band', fs=SR, output='sos'))
    i = np.arange(N) / N
    g = 0.8 + 0.2 * np.sin(TWOPI * 5 * i + 1.3)
    bed(ctx, x * g * 0.035 * level, sends={room or ctx.room: 0.2})


def tuned_drips(ctx, prob=0.06, notes=('D6', 'E6', 'F#6', 'A6', 'B6'), level=1.0, tau=0.06, echo=0.5, grid=16):
    rng = ctx.rng
    for b in range(1, 17):
        pcs = scale_pcs(ctx.H, b)
        cand = [nm(n) for n in notes if nm(n) % 12 in pcs] or [nm(notes[0])]
        for p in range(grid):
            if rng.random() < prob:
                m = rng.choice(cand)
                y = bubble(float(mtof(m)), tau * rng.uniform(0.7, 1.3), rng, rise=0.035, vel=0.16 * level * rng.uniform(0.5, 1))
                y2 = 0.3 * bubble(float(mtof(m)) * 2.01, tau * 0.3, rng, rise=0.03, vel=0.1 * level)
                y[:len(y2)] += y2[:len(y)]
                ctx.tr.add(y, ctx.p(b, p * 16 // grid) + ctx.jit(8), pan=rng.uniform(-0.6, 0.6),
                           sends={ctx.room: 0.5, 'echo': echo})


# ------------------------------------------------------------- ornaments ---

def tuned_bird(ctx, bar, beat, seq, speed=0.07, trill=0.0, vel=0.35, pan=0.4, sends=None):
    """a whistled bird call pitched to the harmony: seq = note names; glides between."""
    rng = ctx.rng
    pts = []
    t = 0.0
    for i, n in enumerate(seq):
        f = float(mtof(nm(n)))
        pts.append((t, f * (0.97 if i == 0 else 1.0), 0.0 if i == 0 else 0.9))
        t += 0.012 if i == 0 else 0.0
        pts.append((t + 0.001, f, 1.0))
        t += speed
    pts.append((t + 0.03, float(mtof(nm(seq[-1]))) * 0.985, 0.0))
    y = bird_call(pts, rng, trill_hz=trill, trill_depth=0.012 if trill else 0, harm=0.06, vel=vel)
    ctx.tr.add(y, T(bar, beat) + ctx.jit(5), pan=pan, sends=sends or {ctx.room: 0.5, 'echo': 0.25})


def flute_turn(ctx, bar, beat, seq, step=0.25, vel=0.4, pan=-0.25, kind='whistle'):
    ev = []
    for i, n in enumerate(seq):
        s = T(bar, beat + i * step)
        ev.append((s, int(step * SPB * (1.0 if i < len(seq) - 1 else 2.5)), nm(n), vel * (1 if i else 1.1)))
    w = wind_line(ev, ctx.rng, kind, vib_depth=0.08)
    wh = w if ev[0][0] < HALF else None
    ctx.tr.add_line(w, wh, pan=pan, sends={ctx.room: 0.45, 'echo': 0.15})


def add_wind(ctx, ev, kind='flute', pan=0.05, rev=0.4, echo=0.12, gain=1.0, vib_depth=0.16, vib_rate=5.1, breath=1.0):
    seed = int(ctx.rng.integers(1 << 30))
    w = wind_line(ev, ctx.rng, kind, vib_depth=vib_depth, vib_rate=vib_rate, breath=breath, seed=seed) * gain
    evh = [e for e in ev if e[0] < HALF]
    wh = wind_line(evh, ctx.rng, kind, vib_depth=vib_depth, vib_rate=vib_rate, breath=breath, seed=seed) * gain if evh else None
    ctx.tr.add_line(w, wh, pan=pan, sends={ctx.room: rev, 'echo': echo})


def harmonize_below(H, bar, beat, midi):
    """diatonic-ish harmony note: nearest chord tone 3-5 semitones below."""
    pcs = set(x % 12 for x in chord_at(H, bar, int((beat - 1) * 4)))
    for d in (3, 4, 5, 2):
        if (midi - d) % 12 in pcs:
            return midi - d
    return midi - 5


# =================================================================== A ======

def sec_A(stem, ctx):
    H, r = ctx.H, ctx.rng
    room = ctx.room
    if stem == 'amb':
        bed(ctx, insects(N, r, level=0.030, center=5400, pulse=41, swell_cycles=3), sends={'air': 0.3})
        bed(ctx, insects(N, r, level=0.016, center=7600, pulse=57, swell_cycles=5), sends={'air': 0.3})
        leaves(ctx, level=0.045)
        distant_birds(ctx, 26, level=1.0)
        frogs(ctx, 3, 0.6)
        return finalize(ctx)
    if stem == 'water':
        brook(ctx, density=70, level=0.7, room=room)
        tuned_drips(ctx, prob=0.035, level=0.8)
        return finalize(ctx)
    if stem == 'woods':
        for b in range(1, 17):
            if b % 2 == 1:
                put_perc(ctx, log_drum(220, ctx.vj(0.55), r), b, 0, pan=-0.3, sends={room: 0.35, 'echo': 0.3})
                put_perc(ctx, log_drum(293.7, ctx.vj(0.42), r), b, 7, pan=0.25, sends={room: 0.35, 'echo': 0.3})
            else:
                put_perc(ctx, log_drum(329.6, ctx.vj(0.32), r), b, 3, pan=0.35, sends={room: 0.35, 'echo': 0.3})
                put_perc(ctx, log_drum(293.7, ctx.vj(0.45), r), b, 10, pan=-0.1, sends={room: 0.35, 'echo': 0.3})
            if b % 4 == 0:
                put_perc(ctx, log_drum(370, ctx.vj(0.28), r, decay=0.15), b, 14, pan=0.4, sends={room: 0.3, 'echo': 0.2})
                put_perc(ctx, log_drum(440, ctx.vj(0.24), r, decay=0.13), b, 15, pan=0.5, sends={room: 0.3, 'echo': 0.2})
            if b in (8, 16):
                put_perc(ctx, wood_block(1180, 0.25, r), b, 12, pan=0.6, sends={room: 0.4, 'echo': 0.3})
        return finalize(ctx)
    if stem == 'shaker':
        pat = [0.2, 0.1, 0.5, 0.14]
        for b in range(1, 17):
            for p in range(16):
                v = pat[p % 4] * (1.15 if p == 10 else 1.0)
                put_perc(ctx, shaker(ctx.vj(v, 0.15), r, length=0.07 if p % 4 != 2 else 0.11), b, p, pan=0.35,
                         sends={room: 0.18})
            for p in (1, 5, 13):   # finger taps on a frame drum
                if r.random() < 0.55:
                    put_perc(ctx, hand_drum('ghost', ctx.vj(0.1), r, pitch=1.1), b, p, pan=-0.35, sends={room: 0.2})
        return finalize(ctx)
    if stem == 'drums':
        for b in range(1, 17):
            put_perc(ctx, felt_kick(ctx.vj(0.55), r), b, 0, sends={room: 0.08})
            put_perc(ctx, felt_kick(ctx.vj(0.32), r), b, 10, sends={room: 0.08})
            put_perc(ctx, hand_drum('low', ctx.vj(0.4), r), b, 6, pan=-0.25, sends={room: 0.2})
            put_perc(ctx, hand_drum('tone', ctx.vj(0.3), r), b, 3, pan=0.25, sends={room: 0.2})
            put_perc(ctx, hand_drum('slap', ctx.vj(0.3), r), b, 8, pan=0.2, sends={room: 0.25})
            if b % 8 == 0:
                for i, (p, k) in enumerate([(12, 'tone'), (13, 'tone'), (14, 'low'), (15, 'low')]):
                    put_perc(ctx, hand_drum(k, ctx.vj(0.28 + 0.06 * i), r), b, p, pan=0.3 - 0.2 * i, sends={room: 0.2})
                if b == 16:
                    put_perc(ctx, tom(105, 0.45, r), b, 14, pan=-0.2, sends={room: 0.3})
            else:
                put_perc(ctx, hand_drum('tone', ctx.vj(0.34), r), b, 12, pan=0.25, sends={room: 0.2})
                put_perc(ctx, hand_drum('low', ctx.vj(0.34), r), b, 14, pan=-0.25, sends={room: 0.2})
            for p in (1, 5, 9, 11, 13):
                if r.random() < 0.4:
                    put_perc(ctx, hand_drum('ghost', ctx.vj(0.1), r), b, p, pan=0.1, sends={room: 0.15})
        return finalize(ctx)
    if stem == 'bass':
        add_bass_events(ctx, BASS['A'], vel=0.72, sends={room: 0.06})
        return finalize(ctx, hpf=28)
    if stem == 'keys':
        for b in range(1, 17):
            c = H[b - 1]
            add_ep(ctx, b, 0, 6, c['ep'], 0.5, sends={room: 0.3})
            if b % 4 == 0:
                add_ep(ctx, b, 12, 4, chord_at(H, b, 12), 0.36, sends={room: 0.3})
            else:
                add_ep(ctx, b, 10, 5, chord_at(H, b, 10), 0.4, sends={room: 0.3})
        return finalize(ctx)
    if stem == 'pad':
        add_pad(ctx, cutoff=1100, attack=0.9, release=1.5, gain=1.0, sends={room: 0.5})
        return finalize(ctx)
    if stem == 'marimba':
        pos = [0, 3, 6, 8, 11, 14]
        contour = [0, 2, 4, 1, 3, 5]
        vv = [0.5, 0.34, 0.4, 0.44, 0.3, 0.36]
        for b in range(1, 17):
            for i, p in enumerate(pos):
                pool = pool_for(H, b, p, nm('F#3'), nm('E5'))
                m = pool[(contour[i] + (b % 2) * 2) % len(pool)]
                put_perc(ctx, marimba(float(mtof(m)), ctx.vj(vv[i]), r, hard=0.3), b, p, pan=-0.35,
                         sends={room: 0.25}, jit=5)
        return finalize(ctx)
    if stem == 'kalimba':
        ev = melody_events(ctx, MELODY['A'], vel=0.55, bars=KALIMBA_FRAG_BARS['A'], art=False)
        for (s, d, m, v) in ev:
            ctx.tr.add(kalimba(float(mtof(m)), v, r), s, pan=0.3, sends={room: 0.35, 'echo': 0.3})
        return finalize(ctx)
    if stem == 'melody':
        add_wind(ctx, melody_events(ctx, MELODY['A'], vel=0.62), 'flute', pan=0.05, rev=0.4, echo=0.1)
        return finalize(ctx)
    if stem == 'ornament':
        tuned_bird(ctx, 2, 3.5, ['F#6', 'A6', 'F#6'], speed=0.06)
        tuned_bird(ctx, 4, 4.0, ['A6', 'B6', 'A6', 'F#6'], speed=0.05, pan=-0.45)
        flute_turn(ctx, 8, 3.0, ['B5', 'C#6', 'B5', 'A5', 'E6'], step=0.25, vel=0.34)
        tuned_bird(ctx, 10, 3.5, ['E6', 'A6'], speed=0.09, trill=24, pan=0.5)
        tuned_bird(ctx, 12, 4.0, ['F#6', 'E6', 'F#6', 'E6', 'D6'], speed=0.045, pan=-0.4)
        flute_turn(ctx, 16, 2.5, ['E6', 'D6', 'B5', 'A5'], step=0.5, vel=0.3)
        for (b, ns, v) in [(1, ['A5'], 0.3), (9, ['D6'], 0.3), (13, ['F5', 'A5'], 0.32)]:
            for i, n in enumerate(ns):
                ctx.tr.add(bell(float(mtof(nm(n))), v, r), T(b) + i * 900, pan=0.45 - 0.3 * i, sends={room: 0.6, 'echo': 0.25})
        return finalize(ctx)


# =================================================================== B ======

def sec_B(stem, ctx):
    H, r = ctx.H, ctx.rng
    room = ctx.room
    if stem == 'amb':
        bed(ctx, insects(N, r, level=0.018, center=5000, pulse=37, swell_cycles=2), sends={'air': 0.4})
        leaves(ctx, level=0.03, lo=200, hi=1200, cycles=2)
        distant_birds(ctx, 14, kinds=('whistle', 'coo', 'trill'), level=0.9)
        frogs(ctx, 8, 0.9)
        return finalize(ctx)
    if stem == 'water':
        brook(ctx, density=170, level=1.0, room=room)
        wf = waterfall_voice([H[b]['pad'] for b in range(16)], r, level=0.05, q=90)
        ctx.tr.add(wf, 0, sends={room: 0.5})
        x = pink(N, r)
        x = circ_filter(x, sos=signal.butter(2, 1400, btype='low', fs=SR, output='sos'))
        bed(ctx, x * 0.03, sends={room: 0.3})
        tuned_drips(ctx, prob=0.08, notes=('A5', 'D6', 'E6', 'F#6', 'A6', 'B6', 'C#7'), level=0.9, tau=0.07, echo=0.6)
        return finalize(ctx)
    if stem == 'woods':
        for (b, p, f) in [(2, 7, 440), (6, 3, 370), (10, 11, 440), (14, 7, 587.3), (4, 14, 293.7), (12, 10, 329.6)]:
            put_perc(ctx, log_drum(f, ctx.vj(0.3), r, decay=0.18, bright=0.3), b, p, pan=r.uniform(-0.6, 0.6),
                     sends={room: 0.5, 'echo': 0.55})
        return finalize(ctx)
    if stem == 'shaker':
        for b in range(1, 17):
            for p in range(0, 16, 2):
                put_perc(ctx, shaker(ctx.vj(0.17 if p % 4 else 0.12, 0.2), r, length=0.17, tone=0.8), b, p, pan=-0.3,
                         sends={room: 0.3})
            for p in (3, 11):
                if r.random() < 0.5:
                    put_perc(ctx, shaker(ctx.vj(0.07), r, length=0.06), b, p, pan=0.4, sends={room: 0.3})
        return finalize(ctx)
    if stem == 'drums':
        for b in range(1, 17):
            put_perc(ctx, frame_drum(ctx.vj(0.45), r), b, 0, sends={room: 0.25})
            put_perc(ctx, frame_drum(ctx.vj(0.22), r, f0=72), b, 3, sends={room: 0.25})
            put_perc(ctx, hand_drum('low', ctx.vj(0.18), r, pitch=0.95), b, 6, pan=-0.3, sends={room: 0.3})
            put_perc(ctx, hand_drum('tone', ctx.vj(0.16), r), b, 14, pan=0.3, sends={room: 0.3})
            if b % 4 == 0:
                put_perc(ctx, hand_drum('tone', ctx.vj(0.2), r), b, 11, pan=0.35, sends={room: 0.3})
        return finalize(ctx)
    if stem == 'bass':
        add_bass_events(ctx, gen_bass(H, 'water'), vel=0.6, sends={room: 0.1}, glide=True, mwah=0.6)
        return finalize(ctx, hpf=28)
    if stem == 'keys':
        order = [0, 2, 4, 5, 3, 1, 2, 4]
        for b in range(1, 17):
            v = H[b - 1]['ep']
            for i, p in enumerate(range(0, 16, 2)):
                m = v[order[i] % len(v)]
                add_ep(ctx, b, p, 9, [m + 12 if m < nm('A3') else m], 0.3 + 0.08 * (i == 0), sends={room: 0.45, 'echo': 0.2},
                       pan_spread=0.0, bark=0.5)
        return finalize(ctx)
    if stem == 'pad':
        add_pad(ctx, cutoff=2100, attack=1.8, release=2.6, gain=0.95, sends={room: 0.6}, voices=3, air=1.0,
                width=1.0, lfo_cycles=4, lfo_depth=0.25, extra_top=12)
        return finalize(ctx)
    if stem == 'marimba':
        for b in range(1, 17):
            pool = pool_for(H, b, 0, nm('D4'), nm('D6'))
            up = pool[:8] if len(pool) >= 8 else (pool * 2)[:8]
            start = (b % 2) * 1
            for i in range(8):
                m = up[(i + start) % len(up)]
                put_perc(ctx, marimba(float(mtof(m)), ctx.vj(0.16 + 0.03 * i), r, hard=0.05), b, i, pan=0.35 - 0.1 * (i % 3),
                         sends={room: 0.35}, jit=4)
            for j, p in enumerate((10, 12, 14)):
                m = up[(7 - 2 * j + start) % len(up)]
                put_perc(ctx, marimba(float(mtof(m)), ctx.vj(0.2 - 0.03 * j), r, hard=0.05), b, p, pan=-0.2,
                         sends={room: 0.35}, jit=4)
        return finalize(ctx)
    if stem == 'kalimba':
        for b in range(1, 17):
            pool = pool_for(H, b, 0, nm('A4'), nm('E6'))
            for j, p in enumerate((0, 6, 11)):
                m = pool[-1 - (j * 2 + b) % min(len(pool), 5)]
                ctx.tr.add(kalimba(float(mtof(m)), ctx.vj(0.35 - 0.05 * j), r), ctx.p(b, p) + ctx.jit(6),
                           pan=r.uniform(-0.5, 0.5), sends={room: 0.4, 'echo': 0.55})
        return finalize(ctx)
    if stem == 'melody':
        add_wind(ctx, melody_events(ctx, MELODY['B'], vel=0.55), 'flute', pan=-0.05, rev=0.6, echo=0.2, vib_depth=0.13)
        return finalize(ctx)
    if stem == 'ornament':
        for (b, ns) in [(1, ['A5', 'E6']), (5, ['C#6', 'F#6']), (9, ['G#5', 'C#6']), (13, ['E6', 'A6'])]:
            for i, n in enumerate(ns):
                ctx.tr.add(bell(float(mtof(nm(n))), 0.26, r), T(b) + i * int(0.75 * SPB), pan=-0.4 + 0.8 * i,
                           sends={room: 0.7, 'echo': 0.2})
        tuned_bird(ctx, 3, 4.0, ['C#7', 'B6', 'A6', 'F#6'], speed=0.04, pan=0.55)
        tuned_bird(ctx, 11, 3.5, ['B6', 'A6', 'F#6', 'D6'], speed=0.04, pan=-0.55)
        flute_turn(ctx, 8, 3.5, ['E6', 'F#6', 'E6', 'C#6'], step=0.25, vel=0.28, pan=0.3)
        return finalize(ctx)


# =================================================================== C ======

def sec_C(stem, ctx):
    H, r = ctx.H, ctx.rng
    room = ctx.room
    if stem == 'amb':
        leaves(ctx, level=0.06, lo=150, hi=1100, cycles=2)
        bed(ctx, insects(N, r, level=0.012, center=5600, pulse=43, swell_cycles=2), sends={'air': 0.5})
        distant_birds(ctx, 10, kinds=('raptor', 'whistle', 'chirp'), level=1.0)
        return finalize(ctx)
    if stem == 'water':
        x = pink(N, r)
        x = circ_filter(x, sos=signal.butter(2, 700, btype='low', fs=SR, output='sos'))
        bed(ctx, x * 0.05, sends={'air': 0.4})
        wf = waterfall_voice([H[b]['pad'] for b in range(16)], r, level=0.03, q=70)
        ctx.tr.add(wf, 0, sends={room: 0.6})
        return finalize(ctx)
    if stem == 'woods':
        for (b, p, f, v) in [(1, 0, 293.7, 0.4), (5, 0, 220, 0.36), (9, 0, 293.7, 0.4), (13, 0, 293.7, 0.34), (8, 10, 440, 0.2)]:
            put_perc(ctx, log_drum(f, ctx.vj(v), r, decay=0.25), b, p, pan=r.uniform(-0.5, 0.5), sends={room: 0.6, 'echo': 0.5})
        return finalize(ctx)
    if stem == 'shaker':
        for b in range(1, 17):
            for p in (4, 12):
                put_perc(ctx, shaker(ctx.vj(0.2), r, length=0.22, tone=0.9), b, p, pan=0.3, sends={room: 0.35})
            for p in (2, 6, 10, 14):
                put_perc(ctx, shaker(ctx.vj(0.07), r, length=0.06), b, p, pan=-0.3, sends={room: 0.35})
        return finalize(ctx)
    if stem == 'drums':
        for b in range(1, 17):
            put_perc(ctx, frame_drum(ctx.vj(0.36), r, f0=64), b, 0, sends={room: 0.35})
            if b % 2 == 1:
                put_perc(ctx, felt_kick(ctx.vj(0.35), r), b, 0, sends={room: 0.1})
            put_perc(ctx, frame_drum(ctx.vj(0.16), r, f0=70), b, 10, sends={room: 0.35})
            if b % 4 == 0:     # felt tom swell into the next phrase
                for i, p in enumerate(range(8, 16)):
                    put_perc(ctx, tom(92 + (i % 2) * 18, 0.08 + 0.055 * i, r), b, p, pan=-0.3 + 0.08 * i, sends={room: 0.45})
        return finalize(ctx)
    if stem == 'bass':
        add_bass_events(ctx, gen_bass(H, 'long'), vel=0.62, sends={room: 0.12}, glide=True, mwah=0.5)
        return finalize(ctx, hpf=28)
    if stem == 'keys':
        for b in range(1, 17):
            add_ep(ctx, b, 0, 15, H[b - 1]['ep'], 0.42, strum=0.03, sends={room: 0.5})
            if b % 2 == 0:
                add_ep(ctx, b, 8, 8, chord_at(H, b, 8)[-2:], 0.28, strum=0.05, sends={room: 0.5, 'echo': 0.2})
            elif H[b - 1]['ep2']:
                add_ep(ctx, b, 8, 8, H[b - 1]['ep2'], 0.3, strum=0.03, sends={room: 0.5})
        return finalize(ctx)
    if stem == 'pad':
        add_pad(ctx, cutoff=1500, attack=1.4, release=2.4, gain=1.05, sends={room: 0.6}, voices=4, width=1.0,
                lfo_cycles=2, lfo_depth=0.3, extra_top=12)
        return finalize(ctx, width=1.25)
    if stem == 'marimba':
        order = [0, 2, 1, 3, 2, 4, 3, 5]
        for b in range(1, 17):
            for i, p in enumerate(range(0, 16, 2)):
                pool = pool_for(H, b, p, nm('A3'), nm('A5'))
                m = pool[order[i] % len(pool)]
                put_perc(ctx, marimba(float(mtof(m)), ctx.vj(0.26 + 0.06 * (i == 0)), r, hard=0.15), b, p, pan=-0.4,
                         sends={room: 0.4}, jit=5)
        return finalize(ctx)
    if stem == 'kalimba':
        for b in range(1, 17):
            for j, p in enumerate((6, 14)):
                pool = pool_for(H, b, p, nm('C#5'), nm('E6'))
                m = pool[-1 - ((b + j) % min(4, len(pool)))]
                ctx.tr.add(kalimba(float(mtof(m)), ctx.vj(0.32), r), ctx.p(b, p) + ctx.jit(6), pan=0.45,
                           sends={room: 0.5, 'echo': 0.45})
        return finalize(ctx)
    if stem == 'melody':
        add_wind(ctx, melody_events(ctx, MELODY['C'], vel=0.68), 'flute', pan=0.0, rev=0.65, echo=0.12, vib_depth=0.18)
        return finalize(ctx)
    if stem == 'ornament':
        # second flute: harmony a 3rd-ish below in the "chorus" bars 9-12 and 13-15
        ev = []
        for (bar, beat, d, n) in MELODY['C']:
            if 9 <= bar <= 15:
                hm = harmonize_below(H, bar, beat, nm(n))
                ev.append((T(bar, beat) + ctx.jit(8), int(d * SPB * 0.92), hm, 0.42))
        add_wind(ctx, ev, 'flute', pan=-0.35, rev=0.7, echo=0.1, vib_depth=0.12)
        tuned_bird(ctx, 4, 3.0, ['A6', 'E6'], speed=0.25, pan=0.6, sends={room: 0.8, 'echo': 0.3})
        tuned_bird(ctx, 7, 4.0, ['D7', 'B6', 'A6'], speed=0.05, trill=22, pan=-0.6)
        ctx.tr.add(bell(float(mtof(nm('A5'))), 0.25, r), T(1), pan=0.4, sends={room: 0.7})
        ctx.tr.add(bell(float(mtof(nm('D6'))), 0.22, r), T(9), pan=-0.4, sends={room: 0.7})
        return finalize(ctx)


# =================================================================== D ======

def sec_D(stem, ctx):
    H, r = ctx.H, ctx.rng
    room = ctx.room
    if stem == 'amb':
        y = crickets(N, r, f=4300, level=0.014)
        ctx.tr.add(y, 0, pan=-0.5, sends={'air': 0.4})
        y = crickets(N, r, f=3900, level=0.01, phase=S16 * 2, skip=0.6)
        ctx.tr.add(y, 0, pan=0.55, sends={'air': 0.4})
        bed(ctx, insects(N, r, level=0.01, center=6200, pulse=29, swell_cycles=2), sends={'air': 0.4})
        frogs(ctx, 12, 1.0)
        distant_birds(ctx, 4, kinds=('coo',), level=0.8)
        # wind through stone: noise resonating on D and A
        x = pink(N, r)
        acc = np.zeros(N)
        for f in (293.66, 440.0, 587.3):
            sos = signal.butter(1, [f / 1.006, f * 1.006], btype='band', fs=SR, output='sos')
            acc += circ_filter(x, sos=sos)
        i = np.arange(N) / N
        acc *= (0.5 + 0.5 * np.sin(TWOPI * 3 * i + 0.4)) ** 2
        bed(ctx, acc / (np.max(np.abs(acc)) + 1e-9) * 0.05, sends={room: 0.6})
        return finalize(ctx)
    if stem == 'water':
        tuned_drips(ctx, prob=0.025, notes=('D5', 'E5', 'F5', 'A5', 'C6', 'D6'), level=1.0, tau=0.09, echo=0.7)
        x = pink(N, r)
        x = circ_filter(x, sos=signal.butter(2, 300, btype='low', fs=SR, output='sos'))
        bed(ctx, x * 0.045, sends={room: 0.4})
        return finalize(ctx)
    if stem == 'woods':
        for (b, p, f, v) in [(2, 5, 147, 0.5), (3, 11, 196, 0.35), (6, 2, 147, 0.45), (7, 9, 220, 0.35),
                             (10, 5, 147, 0.5), (11, 14, 175, 0.3), (14, 3, 147, 0.45), (15, 10, 220, 0.32)]:
            put_perc(ctx, log_drum(f, ctx.vj(v), r, decay=0.3, bright=0.25), b, p, pan=r.uniform(-0.7, 0.7),
                     sends={room: 0.6, 'echo': 0.45})
        return finalize(ctx)
    if stem == 'shaker':
        for b in (4, 8, 12, 16):
            ctx.tr.add(seed_rattle(1.3, 0.35, r), T(b, 3.0) + int(2 * SPB - 1.3 * SR), pan=r.uniform(-0.5, 0.5),
                       sends={room: 0.5})
        for b in range(1, 17, 2):
            put_perc(ctx, shaker(0.06, r, length=0.3, tone=0.6), b, 8, pan=0.5, sends={room: 0.5})
        return finalize(ctx)
    if stem == 'drums':
        for b in range(1, 17):
            if b % 2 == 1:
                put_perc(ctx, tom(78, ctx.vj(0.5), r), b, 0, sends={room: 0.35})
                put_perc(ctx, tom(78, ctx.vj(0.26), r), b, 3, sends={room: 0.35})
            else:
                put_perc(ctx, frame_drum(ctx.vj(0.2), r, f0=60), b, 8, sends={room: 0.4})
            if b in (8, 16):
                put_perc(ctx, tom(98, 0.25, r), b, 12, pan=0.3, sends={room: 0.4})
                put_perc(ctx, tom(88, 0.3, r), b, 14, pan=-0.3, sends={room: 0.4})
        return finalize(ctx)
    if stem == 'bass':
        add_bass_events(ctx, gen_bass(H, 'drone'), vel=0.55, sends={room: 0.12}, glide=True, mwah=0.3)
        return finalize(ctx, hpf=26)
    if stem == 'keys':
        for b in range(1, 17, 2):
            add_ep(ctx, b, 0, 24, H[b - 1]['ep'], 0.36, strum=0.05, sends={room: 0.55, 'echo': 0.2}, bark=0.4)
            top = max(H[b]['ep'])
            add_ep(ctx, b + 1, 10, 6, [top + 12], 0.22, sends={room: 0.6, 'echo': 0.45}, bark=0.2)
        return finalize(ctx)
    if stem == 'pad':
        # the D-A drone: every partial quantised to whole cycles per loop -> perfectly periodic
        ctx.tr.add(periodic_drone([nm('D2'), nm('A2')], r, cutoff=650), 0, gain=1.0, sends={room: 0.4})
        add_pad(ctx, cutoff=900, attack=2.2, release=2.8, gain=0.8, sends={room: 0.6}, voices=3, width=0.9,
                lfo_cycles=3, lfo_depth=0.35)
        return finalize(ctx)
    if stem == 'marimba':
        for b in range(1, 17):
            R = H[b - 1]['bass']
            put_perc(ctx, marimba(float(mtof(R + 24)), ctx.vj(0.34), r, hard=0.05), b, 0, pan=-0.3, sends={room: 0.4})
            put_perc(ctx, marimba(float(mtof(R + 31)), ctx.vj(0.22), r, hard=0.05), b, 10, pan=0.2, sends={room: 0.4, 'echo': 0.3})
        return finalize(ctx)
    if stem == 'kalimba':
        ev = melody_events(ctx, MELODY['D'], vel=0.4, bars=[1, 2], art=False)
        for (s, d, m, v) in ev:   # echo of the motif, an octave down, two bars later
            ctx.tr.add(kalimba(float(mtof(m - 12)), v, r), s + 2 * BAR, pan=-0.4, sends={room: 0.5, 'echo': 0.4})
        eb = [(11, 1.5, .5, 'Bb3'), (11, 2, .5, 'Eb4'), (11, 2.5, .5, 'F4'), (11, 3, 1.5, 'Bb4'), (12, 1, 1.5, 'G4'),
              (12, 2.5, .5, 'F4'), (12, 3, 1, 'D4')]
        for (s, d, m, v) in melody_events(ctx, eb, vel=0.4, art=False):
            ctx.tr.add(kalimba(float(mtof(m)), v, r), s, pan=0.4, sends={room: 0.5, 'echo': 0.4})
        return finalize(ctx)
    if stem == 'melody':
        add_wind(ctx, melody_events(ctx, MELODY['D'], vel=0.6), 'ocarina', pan=0.1, rev=0.7, echo=0.18, vib_depth=0.1,
                 vib_rate=4.4)
        return finalize(ctx)
    if stem == 'ornament':
        for (b, n, dark, v) in [(1, 'D5', True, 0.3), (5, 'G4', True, 0.28), (9, 'F5', False, 0.22),
                                (11, 'Eb5', False, 0.24), (13, 'Bb4', False, 0.24), (15, 'A4', False, 0.2)]:
            ctx.tr.add(bell(float(mtof(nm(n))), v, r, dark=dark), T(b), pan=r.uniform(-0.5, 0.5),
                       sends={room: 0.7, 'echo': 0.3})
        tuned_bird(ctx, 7, 3.0, ['E6', 'Bb5'], speed=0.3, pan=-0.6, vel=0.2, sends={room: 0.9, 'echo': 0.3})
        return finalize(ctx)


# =================================================================== E ======

def sec_E(stem, ctx):
    H, r = ctx.H, ctx.rng
    room = ctx.room
    if stem == 'amb':
        bed(ctx, insects(N, r, level=0.028, center=5300, pulse=45, swell_cycles=4), sends={'air': 0.3})
        leaves(ctx, level=0.05, cycles=4)
        distant_birds(ctx, 30, kinds=('chirp', 'trill', 'whistle'), level=1.0)
        return finalize(ctx)
    if stem == 'water':
        brook(ctx, density=60, level=0.6, room=room)
        tuned_drips(ctx, prob=0.03, level=0.7)
        return finalize(ctx)
    if stem == 'woods':
        pa = [(0, 293.7, .42), (3, 440, .3), (6, 329.6, .34), (10, 293.7, .38), (12, 220, .3)]
        pb = [(0, 220, .4), (3, 293.7, .3), (6, 370, .32), (8, 329.6, .28), (11, 293.7, .34)]
        for b in range(1, 17):
            for (p, f, v) in (pa if b % 2 else pb):
                put_perc(ctx, log_drum(f, ctx.vj(v), r, decay=0.16), b, p, pan=0.3 if f > 300 else -0.3,
                         sends={room: 0.25, 'echo': 0.18})
            put_perc(ctx, wood_block(1320, ctx.vj(0.12), r), b, 14, pan=0.55, sends={room: 0.2})
        return finalize(ctx)
    if stem == 'shaker':
        pat = [0.3, 0.15, 0.55, 0.2]
        for b in range(1, 17):
            for p in range(16):
                put_perc(ctx, shaker(ctx.vj(pat[p % 4], 0.12), r, length=0.07), b, p, pan=0.3, sends={room: 0.12})
        return finalize(ctx)
    if stem == 'drums':
        for b in range(1, 17):
            put_perc(ctx, felt_kick(ctx.vj(0.62), r), b, 0, sends={room: 0.06})
            put_perc(ctx, felt_kick(ctx.vj(0.34), r), b, 7, sends={room: 0.06})
            put_perc(ctx, felt_kick(ctx.vj(0.48), r), b, 10, sends={room: 0.06})
            put_perc(ctx, hand_drum('slap', ctx.vj(0.4), r), b, 4, pan=0.2, sends={room: 0.18})
            fill = b in (8, 16)
            if not fill:
                put_perc(ctx, hand_drum('slap', ctx.vj(0.44), r), b, 12, pan=0.2, sends={room: 0.18})
                for (p, k, v, pn) in [(2, 'tone', .28, .3), (3, 'low', .36, -.3), (6, 'low', .32, -.3), (9, 'tone', .26, .3),
                                      (11, 'tone', .3, .3), (14, 'low', .34, -.3), (15, 'tone', .22, .3)]:
                    put_perc(ctx, hand_drum(k, ctx.vj(v), r), b, p, pan=pn, sends={room: 0.15})
            else:
                for (p, k, v, pn) in [(2, 'tone', .28, .3), (3, 'low', .36, -.3), (6, 'low', .32, -.3)]:
                    put_perc(ctx, hand_drum(k, ctx.vj(v), r), b, p, pan=pn, sends={room: 0.15})
                for i, p in enumerate(range(8, 16)):
                    put_perc(ctx, tom([150, 150, 128, 128, 110, 110, 94, 94][i], 0.25 + 0.035 * i, r), b, p,
                             pan=0.4 - 0.1 * i, sends={room: 0.25})
            for p in (1, 5, 13):
                if r.random() < 0.5:
                    put_perc(ctx, hand_drum('ghost', ctx.vj(0.12), r), b, p, pan=0.1, sends={room: 0.1})
        return finalize(ctx)
    if stem == 'bass':
        add_bass_events(ctx, gen_bass(H, 'drive'), vel=0.66, sends={room: 0.05}, mwah=1.1)
        return finalize(ctx, hpf=28)
    if stem == 'keys':
        for b in range(1, 17):
            add_ep(ctx, b, 0, 3, chord_at(H, b, 0), 0.46, sends={room: 0.25})
            add_ep(ctx, b, 6, 2, chord_at(H, b, 6), 0.34, sends={room: 0.25})
            add_ep(ctx, b, 10, 3, chord_at(H, b, 10), 0.4, sends={room: 0.25, 'echo': 0.1})
        return finalize(ctx)
    if stem == 'pad':
        add_pad(ctx, cutoff=1200, attack=0.7, release=1.2, gain=0.8, sends={room: 0.45})
        return finalize(ctx)
    if stem == 'marimba':
        cell = [0, 2, 4, 1, 3, 5, 2, 4]
        acc = {0, 3, 6, 8, 11, 14}
        for b in range(1, 17):
            for p in range(16):
                pool = pool_for(H, b, p, nm('A3'), nm('A5'))
                m = pool[(cell[p % 8] + (p // 8)) % len(pool)]
                v = 0.4 if p in acc else 0.2
                put_perc(ctx, marimba(float(mtof(m)), ctx.vj(v), r, hard=0.45), b, p, pan=-0.35, sends={room: 0.2}, jit=3)
        return finalize(ctx)
    if stem == 'kalimba':
        for b in range(1, 17):
            if b % 2 == 0:
                pool = pool_for(H, b, 12, nm('D5'), nm('E6'))
                for j, p in enumerate((12, 13, 14, 15)):
                    m = pool[min(len(pool) - 1, j + (b % 4 == 0))]
                    ctx.tr.add(kalimba(float(mtof(m)), ctx.vj(0.34 + 0.04 * j), r), ctx.p(b, p) + ctx.jit(4), pan=0.4,
                               sends={room: 0.3, 'echo': 0.35})
            else:
                pool = pool_for(H, b, 0, nm('F#5'), nm('E6'))
                ctx.tr.add(kalimba(float(mtof(pool[-1])), ctx.vj(0.4), r), ctx.p(b, 0), pan=0.4, sends={room: 0.3, 'echo': 0.35})
        return finalize(ctx)
    if stem == 'melody':
        add_wind(ctx, melody_events(ctx, MELODY['E'], vel=0.64), 'flute', pan=0.05, rev=0.35, echo=0.1)
        return finalize(ctx)
    if stem == 'ornament':
        tuned_bird(ctx, 4, 4.0, ['C#7', 'B6', 'G#6', 'E6'], speed=0.04, pan=0.5)
        tuned_bird(ctx, 8, 4.0, ['A6', 'C#7', 'A6'], speed=0.05, trill=26, pan=-0.5)
        tuned_bird(ctx, 12, 4.0, ['F#6', 'A6', 'C#7'], speed=0.045, pan=0.5)
        flute_turn(ctx, 16, 3.5, ['E6', 'D6', 'B5', 'A5'], step=0.25, vel=0.3)
        ctx.tr.add(bell(float(mtof(nm('D6'))), 0.25, r), T(1), pan=0.4, sends={room: 0.5, 'echo': 0.3})
        ctx.tr.add(bell(float(mtof(nm('B5'))), 0.22, r), T(9), pan=-0.4, sends={room: 0.5, 'echo': 0.3})
        return finalize(ctx)


SEC_FN = {'A': sec_A, 'B': sec_B, 'C': sec_C, 'D': sec_D, 'E': sec_E}

# rough stem trims (dB) so that "all stems at unity" is a balanced full mix
TRIM = {s: 0.0 for s in STEMS}


def render_stem(sec, stem):
    seed = (ord(sec) * 1000 + STEMS.index(stem) * 37) % 2 ** 31
    ctx = Ctx(sec, stem, seed)
    return SEC_FN[sec](stem, ctx)


if __name__ == '__main__':
    import soundfile as sf
    secs = sys.argv[1] if len(sys.argv) > 1 else 'ABCDE'
    only = sys.argv[2].split(',') if len(sys.argv) > 2 else STEMS
    os.makedirs('out/raw', exist_ok=True)
    for sec in secs:
        for stem in only:
            t0 = time.time()
            res = render_stem(sec, stem)
            y = res['loop']
            np.save(f'out/raw/{sec}_{stem}.npy', y.astype(np.float32))
            np.save(f'out/raw/{sec}_{stem}_t16.npy', res['tail16'].astype(np.float32))
            np.save(f'out/raw/{sec}_{stem}_t8.npy', res['tail8'].astype(np.float32))
            print(f'{sec} {stem:9s} {time.time() - t0:5.1f}s  peak {20 * np.log10(np.max(np.abs(y)) + 1e-12):6.1f} dB  '
                  f'rms {20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12):6.1f} dB', flush=True)
