"""
Canopy — jungle-focus layers.

New stems (same 16-bar grid, same circular rendering, same ring-out tails):
  birds     the tuned bird calls that used to live in 'ornament' (no flute, no bells)
  wildlife  extra creatures: cicada swells, songbird warbles, woodpeckers,
            tree-frog choruses, drips falling off leaves
  amb2-4    alternate takes of each area's original ambience recipe (new random
            seeds), so the jungle does not repeat every 43 s

Jungle stingers (replace the melodic ones): gust, flock, shower, woodpecker_roll, far_cry
"""
import os, sys, time
import numpy as np
from synth import *
from synth import HALF
import render as R
from render import Ctx, T, P, tuned_bird, finalize, SEC_FN

# ---------------------------------------------------------------- creatures --

def cicada(ctx, start, dur, center=5200, pulse=190, level=1.0, pan=0.0, room='air'):
    """tymbal buzz with the typical slow swell and fade"""
    rng = ctx.rng
    n = int(dur * SR)
    t = tvec(n)
    x = bp(rng.standard_normal(n), center * 0.82, center * 1.22)
    pr = pulse * (1 + 0.03 * np.sin(TWOPI * rng.uniform(0.2, 0.5) * t))
    am = np.maximum(np.sin(TWOPI * np.cumsum(pr) / SR), 0) ** 3
    a = dur * rng.uniform(0.25, 0.4)
    r = dur * rng.uniform(0.25, 0.4)
    env = np.clip(t / a, 0, 1) ** 2 * np.clip((dur - t) / r, 0, 1) ** 1.5
    y = lp(x * am * env, 7500)
    ctx.tr.add(y * level * 0.45, int(start), pan=pan, sends={room: 0.55})


def warbler(ctx, start, level=1.0, pan=0.0, near=0.5, room='air'):
    """a songbird phrase: a few note shapes, repeated and varied"""
    rng = ctx.rng
    base = rng.uniform(2600, 4300)
    shapes = [(rng.uniform(0.85, 1.3), rng.choice(['up', 'down', 'flat', 'trill']), rng.uniform(0.04, 0.1))
              for _ in range(3)]
    t = 0.0
    for i in range(int(rng.integers(6, 14))):
        if rng.random() < 0.72:
            m = shapes[int(rng.integers(0, 3))]
        else:
            m = (rng.uniform(0.8, 1.4), rng.choice(['up', 'down', 'flat']), rng.uniform(0.03, 0.09))
        f, kind, d = base * m[0], m[1], m[2]
        th = 0.0
        if kind == 'up':
            pts = [(0, f * 0.85, 0), (0.006, f * 0.9, 1), (d, f * 1.18, 0.8), (d + 0.006, f * 1.18, 0)]
        elif kind == 'down':
            pts = [(0, f * 1.18, 0), (0.006, f * 1.12, 1), (d, f * 0.86, 0.8), (d + 0.006, f * 0.86, 0)]
        elif kind == 'flat':
            pts = [(0, f, 0), (0.005, f, 1), (d, f * 0.98, 0.8), (d + 0.006, f * 0.98, 0)]
        else:
            pts = [(0, f, 0), (0.005, f, 1), (d, f, 0.8), (d + 0.006, f, 0)]
            th = 45.0
        y = bird_call(pts, rng, trill_hz=th, trill_depth=0.05 if th else 0.0, harm=0.08, vel=0.2 * level)
        y = lp(y, 4000 + 8000 * near)
        ctx.tr.add(y, int(start + t * SR), pan=pan, sends={room: 0.65 - 0.35 * near})
        t += d + rng.uniform(0.02, 0.075)


def woodpecker(ctx, start, level=1.0, pan=0.0, room='air'):
    rng = ctx.rng
    k = int(rng.integers(12, 24))
    rate = rng.uniform(13, 19)
    f = rng.uniform(900, 1500)
    n = int(0.03 * SR)
    t = tvec(n)
    for i in range(k):
        tt = i / rate * (1 + 0.006 * i)
        click = np.sin(TWOPI * f * t) * np.exp(-t / 0.004) + 0.6 * bp(rng.standard_normal(n), 1500, 5000) * np.exp(-t / 0.0015)
        v = level * 0.22 * (0.7 + 0.3 * rng.random()) * (1 - 0.35 * i / k)
        ctx.tr.add(click * v, int(start + tt * SR), pan=pan, sends={room: 0.7})


def treefrogs(ctx, frogs=6, level=1.0, bouts=(2, 5)):
    rng = ctx.rng
    n = int(0.045 * SR)
    tt = tvec(n)
    win = np.hanning(n)
    for _ in range(frogs):
        f0 = rng.uniform(2100, 3300)
        pan = rng.uniform(-0.85, 0.85)
        for _b in range(int(rng.integers(*bouts))):
            s = rng.uniform(0, N)
            L = rng.uniform(2.5, 7.0)
            iv = rng.uniform(0.28, 0.6)
            t = 0.0
            while t < L:
                f = f0 * (1 + 0.12 * tt / 0.045)
                y = np.sin(TWOPI * np.cumsum(f) / SR) * win
                ctx.tr.add(y * 0.05 * level * rng.uniform(0.6, 1.0), int(s + t * SR), pan=pan, sends={'air': 0.55})
                t += iv * rng.uniform(0.9, 1.1)


def leaf_drips(ctx, rate=0.8, level=1.0, room=None):
    rng = ctx.rng
    n = int(0.06 * SR)
    t = tvec(n)
    for _ in range(int(rate * N / SR)):
        f = rng.uniform(250, 700)
        tap = np.sin(TWOPI * f * t * (1 + 0.3 * np.exp(-t / 0.005))) * np.exp(-t / 0.012)
        hit = bp(rng.standard_normal(n), 1800, 6000) * np.exp(-t / 0.002)
        y = (0.6 * tap + 0.5 * hit) * level * 0.1 * rng.uniform(0.3, 1.0)
        ctx.tr.add(y, int(rng.uniform(0, N)), pan=rng.uniform(-0.85, 0.85), sends={room or ctx.room: 0.4})


def U(ctx):
    return int(ctx.rng.uniform(0, N))


def wildlife(sec, ctx):
    r = ctx.rng
    if sec == 'A':
        cicada(ctx, U(ctx), r.uniform(9, 14), center=5400, level=0.9, pan=-0.5)
        cicada(ctx, U(ctx), r.uniform(8, 12), center=6100, pulse=230, level=0.6, pan=0.6)
        for _ in range(9):
            warbler(ctx, U(ctx), level=0.9, pan=r.uniform(-0.8, 0.8), near=r.uniform(0.2, 0.7))
        for _ in range(2):
            woodpecker(ctx, U(ctx), level=0.8, pan=r.uniform(-0.7, 0.7))
        leaf_drips(ctx, rate=0.35, level=0.8)
    elif sec == 'B':
        treefrogs(ctx, frogs=5, level=0.8)
        for _ in range(4):
            warbler(ctx, U(ctx), level=0.7, pan=r.uniform(-0.8, 0.8), near=r.uniform(0.1, 0.5))
        leaf_drips(ctx, rate=1.4, level=1.0)
    elif sec == 'C':
        cicada(ctx, U(ctx), r.uniform(10, 15), center=5000, level=0.55, pan=r.uniform(-0.6, 0.6))
        for _ in range(4):
            warbler(ctx, U(ctx), level=0.6, pan=r.uniform(-0.9, 0.9), near=0.1)
        woodpecker(ctx, U(ctx), level=0.5, pan=r.uniform(-0.8, 0.8))
    elif sec == 'D':
        treefrogs(ctx, frogs=9, level=0.9, bouts=(3, 6))
        leaf_drips(ctx, rate=0.9, level=0.9)
    elif sec == 'E':
        cicada(ctx, U(ctx), r.uniform(8, 12), center=5600, level=0.8, pan=-0.4)
        cicada(ctx, U(ctx), r.uniform(8, 12), center=6300, pulse=240, level=0.6, pan=0.5)
        for _ in range(13):
            warbler(ctx, U(ctx), level=0.9, pan=r.uniform(-0.8, 0.8), near=r.uniform(0.3, 0.8))
        for _ in range(2):
            woodpecker(ctx, U(ctx), level=0.8, pan=r.uniform(-0.7, 0.7))
        leaf_drips(ctx, rate=0.3, level=0.7)
    return finalize(ctx)


def birds(sec, ctx):
    """tuned bird calls (the musical ornaments), without the flute and bells"""
    room = ctx.room
    if sec == 'A':
        tuned_bird(ctx, 2, 3.5, ['F#6', 'A6', 'F#6'], speed=0.06)
        tuned_bird(ctx, 4, 4.0, ['A6', 'B6', 'A6', 'F#6'], speed=0.05, pan=-0.45)
        tuned_bird(ctx, 8, 3.5, ['B6', 'C#7', 'B6', 'E6'], speed=0.07, pan=0.3)
        tuned_bird(ctx, 10, 3.5, ['E6', 'A6'], speed=0.09, trill=24, pan=0.5)
        tuned_bird(ctx, 12, 4.0, ['F#6', 'E6', 'F#6', 'E6', 'D6'], speed=0.045, pan=-0.4)
        tuned_bird(ctx, 15, 2.0, ['D7', 'B6'], speed=0.12, pan=-0.6, vel=0.25)
    elif sec == 'B':
        tuned_bird(ctx, 3, 4.0, ['C#7', 'B6', 'A6', 'F#6'], speed=0.04, pan=0.55)
        tuned_bird(ctx, 8, 3.5, ['E6', 'F#6', 'E6'], speed=0.08, pan=0.3)
        tuned_bird(ctx, 11, 3.5, ['B6', 'A6', 'F#6', 'D6'], speed=0.04, pan=-0.55)
        tuned_bird(ctx, 14, 2.5, ['E7', 'A6'], speed=0.1, pan=-0.2, vel=0.22)
    elif sec == 'C':
        tuned_bird(ctx, 4, 3.0, ['A6', 'E6'], speed=0.25, pan=0.6, sends={room: 0.8, 'echo': 0.3})
        tuned_bird(ctx, 7, 4.0, ['D7', 'B6', 'A6'], speed=0.05, trill=22, pan=-0.6)
        tuned_bird(ctx, 12, 3.0, ['F#6', 'A6'], speed=0.2, pan=0.5, sends={room: 0.8, 'echo': 0.3})
        tuned_bird(ctx, 15, 3.5, ['E6', 'C6'], speed=0.22, pan=-0.4, vel=0.25, sends={room: 0.8, 'echo': 0.3})
    elif sec == 'D':
        tuned_bird(ctx, 7, 3.0, ['E6', 'Bb5'], speed=0.3, pan=-0.6, vel=0.2, sends={room: 0.9, 'echo': 0.3})
        tuned_bird(ctx, 13, 2.0, ['D6', 'A5'], speed=0.35, pan=0.6, vel=0.18, sends={room: 0.9, 'echo': 0.3})
    elif sec == 'E':
        tuned_bird(ctx, 4, 4.0, ['C#7', 'B6', 'G#6', 'E6'], speed=0.04, pan=0.5)
        tuned_bird(ctx, 8, 4.0, ['A6', 'C#7', 'A6'], speed=0.05, trill=26, pan=-0.5)
        tuned_bird(ctx, 12, 4.0, ['F#6', 'A6', 'C#7'], speed=0.045, pan=0.5)
        tuned_bird(ctx, 16, 3.5, ['E6', 'D6', 'B5'], speed=0.07, pan=-0.3)
    return finalize(ctx)


def render_new(sec, stem):
    seed = (ord(sec) * 7919 + sum(map(ord, stem)) * 131) % 2 ** 31
    ctx = Ctx(sec, stem, seed)
    ctx.tr.name = f'{sec}_{stem}'
    if stem == 'birds':
        return birds(sec, ctx)
    if stem == 'wildlife':
        return wildlife(sec, ctx)
    if stem.startswith('amb'):          # alternate take of the original recipe
        return SEC_FN[sec]('amb', ctx)
    raise ValueError(stem)


# ----------------------------------------------------------- jungle stingers --
from stingers import St, beat_at, LEN, ANCHOR


def gust(rng):
    s = St()
    n = int(BAR * 1.6)
    t = tvec(n)
    x = pink(n, rng)
    x = bp(x, 180, 2600)
    peak = BAR / SR
    env = np.where(t < peak, (t / peak) ** 2.5, np.exp(-(t - peak) / 0.7))
    g = (rng.random(n) < 0.004 + 0.03 * env).astype(float) * rng.uniform(0.2, 1, n)
    rustle = bp(g, 2500, 9000) * 3
    y = fade_out((x * 0.25 + rustle * 0.5) * env, 0.6)
    s.add(fade_in(y, 0.1), 0, pan=-0.2, rev=0.3)
    s.add(fade_in(np.roll(y, 900) * 0.8, 0.1), 0, pan=0.3, rev=0.3)
    return s.out()


def flock(rng):
    s = St()
    for b in range(8):
        d = rng.uniform(0.8, 1.6)
        n = int(d * SR)
        t = tvec(n)
        rate = rng.uniform(8, 13)
        flap = np.maximum(np.sin(TWOPI * rate * t), 0) ** 2
        x = bp(rng.standard_normal(n), 300, 2400) * flap * np.exp(-t / (d * 0.55))
        at = beat_at(5) + int(rng.uniform(0, 0.6) * SR)
        s.add(fade_in(x * 0.35, 0.02), at, pan=rng.uniform(-0.8, 0.8), rev=0.25)
        f0 = rng.uniform(2800, 4200)
        for c in range(int(rng.integers(1, 4))):
            pts = [(0, f0 * 0.85, 0), (0.01, f0, 1), (0.05, f0 * 1.25, 0.6), (0.07, f0 * 1.2, 0)]
            s.add(bird_call(pts, rng, vel=0.18), at + int(c * 0.12 * SR), pan=rng.uniform(-0.8, 0.8), rev=0.3)
    return s.out()


def shower(rng):
    s = St()
    n = int(BAR * 2.2)
    t = tvec(n)
    peak = BAR / SR
    dens = np.where(t < peak, (t / peak) ** 1.8, np.exp(-(t - peak) / 1.2))
    drops = (rng.random(n) < 0.0015 + 0.02 * dens).astype(float) * rng.uniform(0.2, 1, n)
    y = bp(drops, 1500, 7000) * 2.5 + 0.25 * lp(drops, 700) * 4
    y += 0.03 * bp(rng.standard_normal(n), 3000, 9000) * dens
    y = fade_out(y, 1.0)
    s.add(fade_in(y * 0.5, 0.05), 0, pan=-0.3, rev=0.35)
    s.add(fade_in(np.roll(y, 1300) * 0.5, 0.05), 0, pan=0.35, rev=0.35)
    return s.out()


def woodpecker_roll(rng):
    s = St()
    n = int(0.03 * SR)
    t = tvec(n)
    k = 18
    for i in range(k):
        tt = beat_at(4) + int(i / 17.0 * SR)
        f = 1150
        click = np.sin(TWOPI * f * t) * np.exp(-t / 0.004) + 0.6 * bp(rng.standard_normal(n), 1500, 5000) * np.exp(-t / 0.0015)
        s.add(click * 0.3 * (0.6 + 0.4 * i / k), tt, pan=0.4, rev=0.3)
    s.add(log_drum(220, 0.5, rng, decay=0.3), beat_at(5), pan=-0.2, rev=0.4)
    return s.out()


def far_cry(rng):
    s = St()
    f0 = 2500
    pts = [(0, f0, 0), (0.05, f0 * 1.05, 1), (0.9, f0 * 0.72, 0.7), (1.2, f0 * 0.65, 0)]
    y = lp(bird_call(pts, rng, trill_hz=11, trill_depth=0.01, harm=0.25, vel=0.3), 5000)
    s.add(y, beat_at(3), pan=0.5, rev=0.5, big=0.8)
    s.add(y * 0.5, beat_at(3) + int(1.4 * SR), pan=0.6, rev=0.5, big=0.8)
    return s.out()


JSTINGERS = dict(gust=gust, flock=flock, shower=shower, woodpecker_roll=woodpecker_roll, far_cry=far_cry)

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    secs = sys.argv[2] if len(sys.argv) > 2 else 'ABCDE'
    if what in ('all', 'stems'):
        for sec in secs:
            for stem in ['birds', 'wildlife', 'amb2', 'amb3', 'amb4']:
                t0 = time.time()
                res = render_new(sec, stem)
                k = f'{sec}_{stem}'
                np.save(f'out/raw/{k}.npy', res['loop'].astype(np.float32))
                np.save(f'out/raw/{k}_t16.npy', res['tail16'].astype(np.float32))
                np.save(f'out/raw/{k}_t8.npy', res['tail8'].astype(np.float32))
                print(k, '%.1fs' % (time.time() - t0), flush=True)
    if what in ('all', 'stingers'):
        for i, (k, fn) in enumerate(JSTINGERS.items()):
            y = fn(np.random.default_rng(300 + i))
            np.save(f'out/raw/st_{k}.npy', y.astype(np.float32))
            print('st', k, '%.1f dB' % (20 * np.log10(np.max(np.abs(y)) + 1e-12)))
