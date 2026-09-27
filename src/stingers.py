"""
Transition stingers. Each file starts exactly ONE BAR before the downbeat it
leads into (anchor = 128 000 samples = 2.667 s) and is 3 bars long.
All pitched material is chosen to sit on every section's bar 8 / bar 16
(A-rooted: Asus2, A, A7, A9sus4) and to resolve onto any bar 1 (Dmaj9 / Dm9).
"""
import numpy as np
from synth import *
from render import IR_SPECS

LEN = 3 * BAR
ANCHOR = BAR
IR = make_ir(**IR_SPECS['canopy'])
IR_BIG = make_ir(**IR_SPECS['vista'])


def beat_at(beat):   # beat in the pre-bar, 1..4.99 ; 5 = downbeat
    return int(round((beat - 1) * SPB))


class St:
    def __init__(self):
        self.L = np.zeros(LEN + SR * 10)
        self.R = np.zeros(LEN + SR * 10)
        self.S = np.zeros(LEN + SR * 10)
        self.B = np.zeros(LEN + SR * 10)

    def add(self, y, at, pan=0.0, rev=0.3, big=0.0, gain=1.0):
        gl, gr = pan_gains(pan)
        j = min(len(self.L), at + len(y))
        self.L[at:j] += y[:j - at] * gl * gain
        self.R[at:j] += y[:j - at] * gr * gain
        self.S[at:j] += y[:j - at] * rev * gain
        self.B[at:j] += y[:j - at] * big * gain

    def out(self):
        n = len(self.L)
        L, R = self.L.copy(), self.R.copy()
        for send, ir in ((self.S, IR), (self.B, IR_BIG)):
            if np.any(send):
                L += signal.fftconvolve(send, ir[0])[:n]
                R += signal.fftconvolve(send, ir[1])[:n]
        y = np.stack([L, R])[:, :LEN]
        y = np.stack([hp(y[0], 35), hp(y[1], 35)])
        f = int(0.8 * SR)
        y[:, -f:] *= np.linspace(1, 0, f) ** 2
        return y


def fill_hands(rng):
    s = St()
    seq = [('tone', .18), ('ghost', .1), ('tone', .22), ('low', .26), ('tone', .28), ('low', .32),
           ('tone', .36), ('low', .42)]
    for i, (k, v) in enumerate(seq):   # 16ths across beats 3-4
        s.add(hand_drum(k, v * 1.3, rng), beat_at(3 + i * 0.25) + int(rng.normal(0, 60)), pan=0.35 - 0.1 * i, rev=0.25)
    s.add(tom(96, 0.5, rng), beat_at(4.5), pan=-0.2, rev=0.3)
    s.add(tom(82, 0.55, rng), beat_at(4.75), pan=-0.35, rev=0.3)
    return s.out()


def fill_logs(rng):
    s = St()
    for i, f in enumerate([440, 370, 329.6, 293.7]):
        s.add(log_drum(f, 0.3 + 0.06 * i, rng), beat_at(4 + i * 0.25), pan=0.5 - 0.3 * i, rev=0.35)
    s.add(log_drum(220, 0.55, rng, decay=0.3), beat_at(5), pan=-0.2, rev=0.45)
    return s.out()


def swell(rng):
    """reverse bloom: a kalimba/bell chord through a long room, reversed so it
    breathes in and stops on the downbeat; plus a rising seed rattle."""
    n = int(BAR * 1.0)
    y = np.zeros(n + SR * 8)
    for i, m in enumerate(['A4', 'D5', 'E5', 'A5', 'E6']):
        k = kalimba(float(mtof(nm(m))), 0.5, rng)
        y[i * 300:i * 300 + len(k)] += k
        b = bell(float(mtof(nm(m))) , 0.2, rng)
        y[:len(b)] += b[:len(y)]
    wet = [signal.fftconvolve(y, IR_BIG[c])[:len(y)] for c in range(2)]
    rev = np.stack([w[:n][::-1] for w in wet])
    env = np.linspace(0, 1, n) ** 2.2
    rev *= env
    rev /= np.max(np.abs(rev)) + 1e-9
    out = np.zeros((2, LEN))
    out[:, ANCHOR - n:ANCHOR] += rev * 0.35
    r = seed_rattle(n / SR, 0.3, rng, rise=True)
    out[0, ANCHOR - n:ANCHOR] += r * 0.7
    out[1, ANCHOR - n:ANCHOR] += np.roll(r, 300) * 0.7
    fl = int(0.004 * SR)
    out[:, ANCHOR - fl:ANCHOR] *= np.linspace(1, 0, fl)
    return out


def pickup(rng):
    """the Canopy motif head compressed to four 16ths, as a kalimba pickup"""
    s = St()
    for i, m in enumerate(['A4', 'D5', 'E5', 'A5']):
        s.add(kalimba(float(mtof(nm(m))), 0.36 + 0.05 * i, rng), beat_at(4 + i * 0.25), pan=-0.3 + 0.2 * i, rev=0.35)
    return s.out()


def arrival(rng):
    s = St()
    s.add(tom(62, 0.45, rng), beat_at(5), rev=0.3, big=0.3)
    s.add(frame_drum(0.4, rng, f0=58), beat_at(5), rev=0.2, big=0.3)
    s.add(bell(float(mtof(nm('D6'))), 0.3, rng), beat_at(5), pan=0.3, rev=0.4, big=0.5)
    s.add(bell(float(mtof(nm('A5'))), 0.26, rng), beat_at(5) + 700, pan=-0.3, rev=0.4, big=0.5)
    wh = bp(rng.standard_normal(int(1.6 * SR)), 900, 6000) * np.hanning(int(1.6 * SR)) ** 3 * 0.05
    s.add(wh, beat_at(4.2), pan=0, rev=0.2, big=0.4)
    return s.out()


def breath(rng):
    """a falling bird-flute sigh (E6-B5-A5) that hands the scene back to the jungle"""
    ev = [(beat_at(3), int(0.5 * SPB), nm('E6'), 0.4), (beat_at(3.5), int(0.5 * SPB), nm('B5'), 0.36),
          (beat_at(4), int(1.6 * SPB), nm('A5'), 0.32)]
    w = wind_line([(a, d, m, v) for a, d, m, v in ev], rng, 'whistle', vib_depth=0.1)
    w = w[PRE:PRE + LEN]
    s = St()
    s.add(w, 0, pan=-0.15, rev=0.5, big=0.4)
    tb = bird_call([(0, 2800, 0), (0.02, 2960, 1), (0.25, 2640, .6), (0.3, 2600, 0)], rng, vel=0.12)
    s.add(tb, beat_at(5.5), pan=0.6, rev=0.3, big=0.6)
    return s.out()


def rise(rng):
    s = St()
    n = BAR
    r = seed_rattle(n / SR, 0.45, rng, rise=True)
    s.add(r, 0, pan=0.3, rev=0.3)
    for i in range(16):
        v = 0.05 + 0.035 * i
        s.add(tom(88 + (i % 2) * 20, v, rng), beat_at(1 + i * 0.25), pan=-0.3 + 0.04 * i, rev=0.35)
    s.add(felt_kick(0.6, rng), beat_at(5), rev=0.1)
    return s.out()


STINGERS = dict(fill_hands=fill_hands, fill_logs=fill_logs, swell=swell, pickup=pickup,
                arrival=arrival, breath=breath, rise=rise)

if __name__ == '__main__':
    import os
    os.makedirs('out/raw', exist_ok=True)
    for i, (k, fn) in enumerate(STINGERS.items()):
        y = fn(np.random.default_rng(100 + i))
        np.save(f'out/raw/st_{k}.npy', y.astype(np.float32))
        print(k, y.shape, '%.1f dB' % (20 * np.log10(np.max(np.abs(y)))))
