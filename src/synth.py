"""
Canopy — original adaptive jungle soundtrack.
DSP / instrument synthesis. Everything here is generated from scratch (no samples).

Timeline constants: 90 BPM, 4/4, 48 kHz.
  1 beat  = 32 000 samples
  1 16th  =  8 000 samples
  1 bar   = 128 000 samples
  16 bars = 2 048 000 samples (42.667 s)  <- every section loop
Every stem is rendered *circularly*: notes, reverb tails, echoes and noise
beds wrap from the end of the loop back onto its start, so each stem loops
sample-perfectly and all stems stay phase-aligned.
"""
import numpy as np
from scipy import signal
import scipy.fft as sfft

SR = 48000
BPM = 90
SPB = SR * 60 // BPM          # 32000
S16 = SPB // 4                # 8000
BAR = SPB * 4                 # 128000
BARS = 16
N = BAR * BARS                # 2048000
PRE = SR                      # pre-roll allowance (events may start slightly early)
TAIL = SR * 12                # tail allowance before folding
BUF = PRE + N + TAIL

TWOPI = 2 * np.pi


def mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0)


_NAMES = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def nm(s):
    """'F#5' -> 78, 'Bb1' -> 34"""
    s = s.strip()
    pc = _NAMES[s[0]]
    i = 1
    while i < len(s) and s[i] in '#b':
        pc += 1 if s[i] == '#' else -1
        i += 1
    octv = int(s[i:])
    return 12 * (octv + 1) + pc


# ----------------------------------------------------------------------------
# envelopes / helpers
# ----------------------------------------------------------------------------

def tvec(n):
    return np.arange(n) / SR


def fade_in(x, sec):
    n = min(len(x), max(1, int(sec * SR)))
    x[:n] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
    return x


def fade_out(x, sec):
    n = min(len(x), max(1, int(sec * SR)))
    x[-n:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(n) / n)
    return x


def env_perc(n, attack, tau):
    t = tvec(n)
    e = np.exp(-np.maximum(t - attack, 0.0) / tau)
    na = max(1, int(attack * SR))
    na = min(na, n)
    e[:na] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    return e


def apply_release(e, dur, rel):
    k = int(dur * SR)
    if k < len(e):
        tt = np.arange(len(e) - k) / SR
        e[k:] *= np.exp(-tt / max(rel, 1e-4))
    return e


def env_asr(n, attack, dur, release, curve=1.0):
    """smooth attack to 1, hold until dur, raised-cosine-ish release"""
    t = tvec(n)
    e = np.ones(n)
    na = min(n, max(1, int(attack * SR)))
    e[:na] = (0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)) ** curve
    k = int(dur * SR)
    if k < n:
        tt = np.arange(n - k) / SR
        e[k:] *= np.exp(-tt / max(release, 1e-4) * 3.0)
    return e


def noise(n, rng):
    return rng.standard_normal(n)


def pink(n, rng):
    w = rng.standard_normal(n)
    W = sfft.rfft(w)
    f = np.arange(len(W))
    f[0] = 1
    W /= np.sqrt(f)
    y = sfft.irfft(W, n=n)
    return y / (np.std(y) + 1e-12)


def bp(x, lo, hi, order=2):
    sos = signal.butter(order, [lo, hi], btype='band', fs=SR, output='sos')
    return signal.sosfilt(sos, x)


def lp(x, fc, order=2):
    sos = signal.butter(order, fc, btype='low', fs=SR, output='sos')
    return signal.sosfilt(sos, x)


def hp(x, fc, order=2):
    sos = signal.butter(order, fc, btype='high', fs=SR, output='sos')
    return signal.sosfilt(sos, x)


def reson(x, f, q):
    b, a = signal.iirpeak(f, q, fs=SR)
    return signal.lfilter(b, a, x)


def softclip(x, drive=1.0):
    return np.tanh(drive * x) / np.tanh(drive)


# ----------------------------------------------------------------------------
# circular (loop-exact) processing in the frequency domain
# ----------------------------------------------------------------------------

def circ_conv(x, ir):
    X = sfft.rfft(x, workers=2)
    H = sfft.rfft(ir, n=len(x), workers=2)
    return sfft.irfft(X * H, n=len(x), workers=2)


def circ_filter(x, sos=None, ba=None):
    """exact steady-state circular IIR filtering"""
    n = len(x)
    w = np.linspace(0, np.pi, n // 2 + 1)
    if sos is not None:
        _, h = signal.sosfreqz(sos, worN=w)
    else:
        _, h = signal.freqz(ba[0], ba[1], worN=w)
    return sfft.irfft(sfft.rfft(x, workers=2) * h, n=n, workers=2)


def circ_echo(x, delay_s, feedback, damp_hz=4000.0, wet=1.0):
    """infinite-feedback echo computed exactly in the frequency domain:
       H = z^-d / (1 - fb*LP(z) z^-d)   (circular, so the loop stays seamless)"""
    n = len(x)
    d = delay_s * SR
    k = np.arange(n // 2 + 1)
    w = TWOPI * k / n
    zd = np.exp(-1j * w * d)
    a = np.exp(-TWOPI * damp_hz / SR)
    lpf = (1 - a) / (1 - a * np.exp(-1j * w))
    H = wet * zd * lpf / (1 - feedback * lpf * zd)
    return sfft.irfft(sfft.rfft(x, workers=2) * H, n=n, workers=2)


def make_ir(rt60=2.0, dur=None, predelay=0.015, low_mul=1.25, high_mul=0.5,
            er=10, er_spread=0.06, er_gain=0.5, dark=9000, seed=1, build=0.02):
    """stereo algorithmic room: multiband exponentially decaying decorrelated
    noise + sparse early reflections."""
    dur = dur or min(rt60 * 1.5, 9.0)
    n = int(dur * SR)
    t = tvec(n)
    rng = np.random.default_rng(seed)
    chans = []
    for ch in range(2):
        w = rng.standard_normal(n)
        lo = lp(w, 350)
        hi = hp(w, 3500)
        mid = w - lo - hi
        y = (lo * 10 ** (-3 * t / (rt60 * low_mul)) +
             mid * 10 ** (-3 * t / rt60) +
             hi * 10 ** (-3 * t / (rt60 * high_mul)))
        y *= 1 - np.exp(-t / build)
        erb = np.zeros(n)
        for _ in range(er):
            p = rng.uniform(0.003, er_spread)
            erb[int(p * SR)] += er_gain * rng.uniform(0.3, 1.0) * rng.choice([-1, 1]) * (1 - p / er_spread * 0.6)
        erb = lp(erb, 6000)
        y = y / (np.sqrt(np.sum(y ** 2)) + 1e-12) + erb * 0.08
        y = lp(y, dark)
        pd = int(predelay * SR)
        y = np.concatenate([np.zeros(pd), y])[:n]
        chans.append(y)
    ir = np.stack(chans)
    ir /= np.sqrt(np.sum(ir ** 2) / 2)
    return ir


# ----------------------------------------------------------------------------
# Track: accumulates a stem on an over-long buffer, then folds it circularly
# ----------------------------------------------------------------------------

def pan_gains(p):
    a = (p + 1) * np.pi / 4
    return np.cos(a), np.sin(a)


HALF = 8 * BAR
LONG_EVENT = 4 * BAR


class Track:
    """Accumulates a stem on an over-long buffer. Everything that starts before
    bar 9 is mirrored into 'half' buffers, so we can render the exact ring-out
    for an exit at bar 9 (tail8) as well as at the loop end (tail16)."""
    def __init__(self, name):
        self.name = name
        self.L = np.zeros(BUF)
        self.R = np.zeros(BUF)
        self.send = {}      # key -> mono buffer
        self.send_st = {}   # key -> (L,R) buffers (echo)
        self.hL = np.zeros(BUF)
        self.hR = np.zeros(BUF)
        self.hsend = {}
        self.hsend_st = {}

    @staticmethod
    def _put(buf, sig, start):
        i = int(start) + PRE
        if i < 0:
            sig = sig[-i:]
            i = 0
        j = min(BUF, i + len(sig))
        if j > i:
            buf[i:j] += sig[:j - i]

    def _route(self, l, r, mono, start, sends, half):
        L, R, S, SS = (self.hL, self.hR, self.hsend, self.hsend_st) if half else (self.L, self.R, self.send, self.send_st)
        self._put(L, l, start)
        self._put(R, r, start)
        if sends:
            for k, amt in sends.items():
                if amt <= 0:
                    continue
                if k.startswith('echo'):
                    if k not in SS:
                        SS[k] = (np.zeros(BUF), np.zeros(BUF))
                    self._put(SS[k][0], l * amt, start)
                    self._put(SS[k][1], r * amt, start)
                else:
                    if k not in S:
                        S[k] = np.zeros(BUF)
                    self._put(S[k], mono * amt, start)

    def add(self, sig, start, pan=0.0, gain=1.0, sends=None, allow_negative=False):
        """sig mono or (2,n). start in samples on the loop timeline."""
        start = int(start)
        if start < 0 and not allow_negative:
            start = 0
        if HALF - int(0.04 * SR) <= start < HALF:   # never let a bar-9 downbeat leak into bar 8
            start = HALF
        if sig.ndim == 1:
            gl, gr = pan_gains(pan)
            l, r = sig * gl * gain, sig * gr * gain
            mono = sig * gain
        else:
            l, r = sig[0] * gain, sig[1] * gain
            mono = 0.5 * (l + r)
        self._route(l, r, mono, start, sends, False)
        if start < HALF - int(0.04 * SR):   # nominal bar-9 downbeats (humanised early) belong to the second half
            if len(l) >= int(0.9 * N):  # loop-length beds / drones: stop at bar 9 (short fade)
                k = HALF - start
                fl = min(k, int(0.5 * SR))
                win = np.ones(k)
                win[k - fl:] = np.linspace(1, 0, fl)
                l, r, mono = l[:k] * win, r[:k] * win, mono[:k] * win
            self._route(l, r, mono, start, sends, True)

    def add_line(self, w, w_half, pan=0.0, sends=None):
        """pre-rendered full-length buffers (index = start+PRE), e.g. wind lines"""
        gl, gr = pan_gains(pan)
        for buf, half in ((w, False), (w_half, True)):
            if buf is None:
                continue
            self._route(buf * gl, buf * gr, buf, -PRE, sends, half)


def fold(buf):
    """fold an over-long buffer (index 0 = timeline -PRE) onto one loop."""
    out = np.zeros(N)
    out[N - PRE:] += buf[:PRE]
    rest = buf[PRE:]
    for k in range(0, len(rest), N):
        seg = rest[k:k + N]
        out[:len(seg)] += seg
    return out


# ----------------------------------------------------------------------------
# INSTRUMENTS (mono unless noted). All return arrays starting at note onset.
# ----------------------------------------------------------------------------

def kalimba(f, vel=0.7, rng=None, length=None, bright=1.0):
    rng = rng or np.random.default_rng()
    dec = float(np.clip(2.3 * (440.0 / f) ** 0.45, 0.6, 3.5))
    n = int((length or dec * 3.2) * SR)
    t = tvec(n)
    ph = TWOPI * f * t
    y = np.sin(ph) * env_perc(n, 0.0015, dec)
    # tine overtone (inharmonic, ~5.9x) and its buzz
    for ratio, amp, d in [(5.93, 0.22 * bright, 0.09), (2.0, 0.05, dec * 0.35), (13.2, 0.06 * bright, 0.025)]:
        if f * ratio < SR * 0.45:
            y += amp * (0.5 + vel) * np.sin(ratio * ph + rng.uniform(0, 6)) * env_perc(n, 0.0008, d)
    # pluck click
    cl = bp(rng.standard_normal(int(0.006 * SR)), 1800, 6000) * env_perc(int(0.006 * SR), 0.0003, 0.0012)
    y[:len(cl)] += 0.18 * bright * vel * cl
    # wooden box resonance
    y += 0.25 * reson(y, 330, 4) + 0.1 * reson(y, 740, 6)
    return y * vel


def marimba(f, vel=0.6, rng=None, hard=0.5):
    rng = rng or np.random.default_rng()
    dec = float(np.clip(1.25 * (220.0 / f) ** 0.6, 0.18, 2.4))
    n = int(dec * 4.5 * SR)
    t = tvec(n)
    ph = TWOPI * f * t
    y = np.sin(ph) * env_perc(n, 0.001, dec)
    y += (0.18 + 0.35 * hard) * vel * np.sin(3.99 * ph) * env_perc(n, 0.0006, dec * 0.18) if f * 4 < 20000 else 0
    y += (0.04 + 0.12 * hard) * vel * np.sin(9.95 * ph) * env_perc(n, 0.0004, dec * 0.05) if f * 10 < 20000 else 0
    # resonator tube bloom
    y += 0.22 * np.sin(ph + 0.4) * (1 - np.exp(-t / 0.012)) * np.exp(-t / (dec * 1.35))
    ml = int(0.008 * SR)
    m = lp(rng.standard_normal(ml), 1500 + 3500 * hard) * env_perc(ml, 0.0005, 0.0018)
    y[:ml] += 0.12 * vel * m
    return y * vel


def log_drum(f, vel=0.6, rng=None, decay=0.22, bright=0.5):
    """tuned slit/log drum - wood knocking in the canopy"""
    rng = rng or np.random.default_rng()
    n = int(decay * 6 * SR)
    t = tvec(n)
    fi = f * (1 + 0.035 * np.exp(-t / 0.008))
    ph = TWOPI * np.cumsum(fi) / SR
    y = np.sin(ph) * env_perc(n, 0.0008, decay)
    y += 0.35 * bright * np.sin(2.43 * ph + 1) * env_perc(n, 0.0005, decay * 0.25)
    y += 0.12 * bright * np.sin(3.9 * ph + 2) * env_perc(n, 0.0004, decay * 0.1)
    cl = int(0.004 * SR)
    y[:cl] += 0.3 * bright * bp(rng.standard_normal(cl), 1500, 5000) * env_perc(cl, 0.0002, 0.0008)
    y += 0.3 * reson(y, f * 0.5, 3) * 0.3
    return y * vel


def wood_block(f=1250, vel=0.5, rng=None):
    rng = rng or np.random.default_rng()
    n = int(0.25 * SR)
    t = tvec(n)
    y = np.sin(TWOPI * f * t) * env_perc(n, 0.0004, 0.035)
    y += 0.5 * np.sin(TWOPI * f * 1.73 * t) * env_perc(n, 0.0003, 0.015)
    y += 0.2 * bp(rng.standard_normal(n), 2000, 7000) * env_perc(n, 0.0002, 0.003)
    return y * vel


def shaker(vel=0.5, rng=None, length=0.09, tone=1.0):
    rng = rng or np.random.default_rng()
    n = int((length + 0.05) * SR)
    t = tvec(n)
    x = rng.standard_normal(n)
    # beads: grainy modulation
    grains = (rng.random(n) < 0.02 * (1 + tone)).astype(float)
    grains = lp(grains, 900) * 30
    x = x * (0.55 + 0.45 * np.clip(grains, 0, 2))
    x = bp(x, 3200 * tone, min(14000, 11000 * tone), order=2)
    a = rng.uniform(0.006, 0.014)
    e = (t / a) ** 2 * np.exp(2 * (1 - t / a))
    e = np.where(t < a, (t / a) ** 1.5, np.exp(-(t - a) / (length * 0.45)))
    return x * e * vel * 0.35


def hand_drum(kind='tone', vel=0.6, rng=None, pitch=1.0):
    """soft hand drum: 'tone', 'low', 'slap', 'ghost', 'bass' """
    rng = rng or np.random.default_rng()
    if kind == 'bass':
        f0, dec, nz, nzf = 78 * pitch, 0.32, 0.05, 700
    elif kind == 'low':
        f0, dec, nz, nzf = 150 * pitch, 0.26, 0.12, 1500
    elif kind == 'slap':
        f0, dec, nz, nzf = 245 * pitch, 0.07, 0.7, 3200
    elif kind == 'ghost':
        f0, dec, nz, nzf = 210 * pitch, 0.05, 0.25, 2500
    else:
        f0, dec, nz, nzf = 215 * pitch, 0.2, 0.15, 2000
    n = int((dec * 6 + 0.05) * SR)
    t = tvec(n)
    fi = f0 * (1 + 0.22 * np.exp(-t / 0.01))
    ph = TWOPI * np.cumsum(fi) / SR
    y = np.sin(ph) * env_perc(n, 0.001, dec)
    for r, a, d in [(1.59, 0.35, 0.4), (2.14, 0.22, 0.25), (2.65, 0.12, 0.18)]:
        y += a * np.sin(r * ph + rng.uniform(0, 6)) * env_perc(n, 0.0008, dec * d)
    nl = int(0.05 * SR)
    hit = lp(rng.standard_normal(nl), nzf) * env_perc(nl, 0.0005, 0.006 if kind != 'slap' else 0.02)
    if kind == 'slap':
        hit = bp(rng.standard_normal(nl), 900, 4500) * env_perc(nl, 0.0004, 0.018)
    y[:nl] += nz * hit * 2
    return y * vel


def tom(f0=110, vel=0.6, rng=None, felt=True):
    rng = rng or np.random.default_rng()
    dec = 0.55 * (110 / f0) ** 0.3
    n = int(dec * 5 * SR)
    t = tvec(n)
    fi = f0 * (1 + 0.3 * np.exp(-t / 0.035))
    ph = TWOPI * np.cumsum(fi) / SR
    y = np.sin(ph) * env_perc(n, 0.002 if felt else 0.0008, dec)
    y += 0.25 * np.sin(1.5 * ph) * env_perc(n, 0.001, dec * 0.3)
    y += 0.1 * np.sin(2.0 * ph) * env_perc(n, 0.001, dec * 0.2)
    nl = int(0.03 * SR)
    y[:nl] += (0.15 if felt else 0.4) * lp(rng.standard_normal(nl), 700 if felt else 2500) * env_perc(nl, 0.001, 0.006)
    return softclip(y * vel * 1.2, 1.2)


def felt_kick(vel=0.6, rng=None):
    rng = rng or np.random.default_rng()
    n = int(0.6 * SR)
    t = tvec(n)
    fi = 50 + 45 * np.exp(-t / 0.025)
    ph = TWOPI * np.cumsum(fi) / SR
    y = np.sin(ph) * env_perc(n, 0.002, 0.2)
    nl = int(0.02 * SR)
    y[:nl] += 0.12 * lp(rng.standard_normal(nl), 900) * env_perc(nl, 0.0005, 0.004)
    return softclip(y * vel, 1.5)


def frame_drum(vel=0.5, rng=None, f0=68):
    rng = rng or np.random.default_rng()
    n = int(1.4 * SR)
    t = tvec(n)
    fi = f0 * (1 + 0.12 * np.exp(-t / 0.03))
    ph = TWOPI * np.cumsum(fi) / SR
    y = np.sin(ph) * env_perc(n, 0.004, 0.42)
    y += 0.3 * np.sin(1.6 * ph) * env_perc(n, 0.003, 0.12)
    nl = int(0.06 * SR)
    y[:nl] += 0.1 * lp(rng.standard_normal(nl), 500) * env_perc(nl, 0.003, 0.02)
    return y * vel


def seed_rattle(length=1.2, vel=0.5, rng=None, rise=True):
    """rain-stick / seed-pod swell used for transitions"""
    rng = rng or np.random.default_rng()
    n = int(length * SR)
    t = tvec(n)
    dens = (t / length) ** 1.7 if rise else np.exp(-t / (length * 0.3))
    clicks = (rng.random(n) < 0.004 + 0.05 * dens).astype(float) * rng.uniform(0.3, 1, n)
    x = bp(clicks, 2500, 11000) * 6
    x += 0.2 * bp(rng.standard_normal(n), 4000, 9000) * dens
    e = dens if rise else np.ones(n)
    x *= e
    x = fade_out(fade_in(x, 0.05), 0.04)
    return x * vel


def bass_note(f, dur, vel=0.7, rng=None, glide_from=None, mwah=1.0):
    """round melodic bass: additive saw with a time-varying spectral
    lowpass (alias-free) + sine body, gentle saturation."""
    rel = 0.09
    n = int((dur + rel * 5) * SR)
    t = tvec(n)
    if glide_from is not None:
        fr = f * (glide_from / f) ** np.exp(-t / 0.035)
    else:
        fr = f * (1 + 0.004 * np.exp(-t / 0.05))
    ph = TWOPI * np.cumsum(fr) / SR
    fc = f * (1.0 + mwah * (0.7 + 1.9 * vel) * np.exp(-t / 0.11)) + 60
    y = np.zeros(n)
    for k in range(1, 18):
        if k * f > 6000:
            break
        a = (1.0 / k) / np.sqrt(1 + (k * f / fc) ** 5)
        y += a * np.sin(k * ph)
    y = 0.45 * y + 0.85 * np.sin(ph) + 0.08 * np.sin(2 * ph)
    e = 0.62 + 0.38 * np.exp(-t / 0.35)
    e *= 1 - np.exp(-t / 0.004)
    e = apply_release(e, dur, rel)
    y = softclip(y * e * vel * 0.9, 1.6)
    return y


def epiano(f, dur, vel=0.6, rng=None, bark=1.0):
    """warm tine electric piano (FM 1:1 + tine transient + pickup asymmetry)"""
    rng = rng or np.random.default_rng()
    dec = float(np.clip(2.8 * (262.0 / f) ** 0.55, 0.8, 5.0))
    rel = 0.18
    n = int((min(dur, dec * 3) + rel * 6) * SR)
    t = tvec(n)
    ph = TWOPI * f * t
    I = (0.35 + 1.5 * vel) * np.exp(-t / 0.28) + 0.18 + 0.2 * vel
    y = np.sin(ph + I * np.sin(ph))
    y += 0.10 * (0.4 + vel) * np.sin(ph * 8.0 + rng.uniform(0, 6)) * env_perc(n, 0.0004, 0.012)
    y += 0.05 * np.sin(2 * ph) * np.exp(-t / (dec * 0.4))
    e = env_perc(n, 0.0015, dec)
    e = apply_release(e, dur, rel)
    y = y * e
    y = y + 0.22 * bark * vel * y ** 2
    return y * vel


def pluck(f, vel=0.5, rng=None, decay=0.996, bright=0.5, length=2.5):
    """Karplus-Strong nylon-ish pluck (subtle plucked strings)"""
    rng = rng or np.random.default_rng()
    n = int(length * SR)
    L = SR / f
    Li = int(L)
    frac = L - Li
    exc = lp(rng.standard_normal(Li + 2), 800 + 6000 * bright) * 0.6
    x = np.zeros(n)
    x[:len(exc)] = exc
    # y[n] = x[n] + g*((1-frac)*y[n-Li] + frac*y[n-Li-1]) averaged
    g = decay
    a = np.zeros(Li + 3)
    a[0] = 1
    a[Li] -= g * 0.5 * (1 - frac)
    a[Li + 1] -= g * 0.5
    a[Li + 2] -= g * 0.5 * frac
    y = signal.lfilter([1.0], a, x)
    y = hp(y, 60)
    y *= env_perc(n, 0.001, length * 0.35)
    return y * vel * 0.8


def bell(f, vel=0.5, rng=None, dark=False, length=None):
    rng = rng or np.random.default_rng()
    if dark:
        parts = [(0.5, 0.35, 1.0), (1.0, 1.0, 0.8), (1.19, 0.4, 0.55), (1.5, 0.3, 0.4),
                 (2.0, 0.25, 0.3), (2.52, 0.15, 0.2), (3.01, 0.08, 0.12)]
        base = 5.0
    else:
        parts = [(1.0, 1.0, 1.0), (2.0, 0.2, 0.5), (2.76, 0.32, 0.33), (5.4, 0.07, 0.12), (8.93, 0.02, 0.05)]
        base = 3.2
    base *= (660.0 / f) ** 0.3
    n = int((length or base * 3) * SR)
    t = tvec(n)
    y = np.zeros(n)
    for r, a, d in parts:
        fr = f * r
        if fr > SR * 0.45:
            continue
        beat = 1 + 0.0015 * rng.uniform(-1, 1)
        y += a * np.sin(TWOPI * fr * beat * t + rng.uniform(0, 6)) * env_perc(n, 0.001, base * d)
    return y * vel * 0.5


def fm_bell(f, vel=0.5, length=4.0):
    n = int(length * SR)
    t = tvec(n)
    I = 2.2 * np.exp(-t / 0.6) + 0.2
    y = np.sin(TWOPI * f * t + I * np.sin(TWOPI * f * 3.5 * t)) * env_perc(n, 0.001, length * 0.3)
    return y * vel * 0.5


def pad_chord(notes, dur, rng=None, attack=1.2, release=2.0, voices=3, detune=0.09,
              cutoff=1400.0, cutoff_lfo=None, t0=0.0, bright=1.0, width=0.8, air=0.0):
    """soft analog pad: detuned band-limited saws through a smooth spectral
    lowpass. Returns stereo (2,n). cutoff_lfo(t_global)->multiplier."""
    rng = rng or np.random.default_rng()
    n = int((dur + release * 1.2) * SR)
    t = tvec(n)
    env = env_asr(n, attack, dur, release, curve=1.5)
    if cutoff_lfo is not None:
        cm = cutoff_lfo(t0 + t)
    else:
        cm = np.ones(n)
    fc = cutoff * cm * (0.75 + 0.25 * env)
    L = np.zeros(n)
    R = np.zeros(n)
    for m in notes:
        f = float(mtof(m))
        for v in range(voices):
            det = (v - (voices - 1) / 2) * detune / max(1, (voices - 1) / 2)
            fv = f * 2 ** (det / 12)
            drift = 1 + 0.0012 * np.sin(TWOPI * rng.uniform(0.08, 0.2) * t + rng.uniform(0, 6))
            ph = TWOPI * np.cumsum(fv * drift) / SR + rng.uniform(0, TWOPI)
            y = np.zeros(n)
            for k in range(1, 40):
                if k * fv > min(9000, 5.5 * cutoff):
                    break
                a = (1.0 / k) / np.sqrt(1 + (k * fv / fc) ** 4)
                y += a * np.sin(k * ph)
            if air > 0:
                y += air * 0.15 * np.sin(2 * ph + 0.3)
            p = (v - (voices - 1) / 2) / max(1, (voices - 1) / 2) * width
            gl, gr = pan_gains(p)
            L += y * gl
            R += y * gr
    sc = 0.25 / np.sqrt(len(notes) * voices)
    return np.stack([L * env * sc, R * env * sc])


def waterfall_voice(chord_notes_by_bar, rng, level=1.0, q=60.0):
    """noise through resonators tuned to the chord of each bar -> 'waterfall
    that becomes a pad'. Returns (2,N) circular."""
    out = np.zeros((2, N))
    base = [pink(N, rng), pink(N, rng)]
    for ch in range(2):
        acc = np.zeros(N)
        for b, notes in enumerate(chord_notes_by_bar):
            seg_env = np.zeros(N)
            s = b * BAR
            # crossfading bar window (raised cosine, 1.5 bars long, overlapping)
            w = np.hanning(int(BAR * 1.6))
            idx = (np.arange(len(w)) + s - int(BAR * 0.3)) % N
            np.add.at(seg_env, idx, w)
            res = np.zeros(N)
            for m in notes:
                f = float(mtof(m))
                sos = signal.butter(1, [f / (1 + 1 / q), f * (1 + 1 / q)], btype='band', fs=SR, output='sos')
                res += circ_filter(base[ch], sos=sos) * (f / 300.0) ** -0.3
            acc += res * seg_env
        out[ch] = acc
    out /= (np.max(np.abs(out)) + 1e-9)
    return out * level


# ----------------------------------------------------------------------------
# legato wind line: wooden flute / ocarina
# ----------------------------------------------------------------------------

def wind_line(notes, rng, kind='flute', vib_rate=5.1, vib_depth=0.16, breath=1.0, legato_gap=0.06, seed=None):
    """notes: list of (start_sample, dur_samples, midi, vel), time-ordered.
    Returns mono buffer of length BUF aligned with Track buffers (index = start+PRE).
    Consecutive notes with small gaps are slurred with portamento."""
    y = np.zeros(BUF)
    if not notes:
        return y
    # group into phrases
    phrases = []
    cur = [notes[0]]
    for nt in notes[1:]:
        prev = cur[-1]
        gap = nt[0] - (prev[0] + prev[1])
        if gap <= legato_gap * SR:
            cur.append(nt)
        else:
            phrases.append(cur)
            cur = [nt]
    phrases.append(cur)

    if kind == 'flute':
        harm = [1.0, 0.42, 0.2, 0.1, 0.05, 0.025, 0.012]
        nb_lo, nb_hi, nb_gain = 1100, 4500, 0.02
    elif kind == 'ocarina':
        harm = [1.0, 0.07, 0.035, 0.012]
        nb_lo, nb_hi, nb_gain = 700, 2800, 0.016
    else:  # 'whistle' (ornament flute, airier)
        harm = [1.0, 0.18, 0.05, 0.02]
        nb_lo, nb_hi, nb_gain = 1800, 6500, 0.03
    nb_gain *= breath

    base_rng = rng
    for ph in phrases:
        s0 = ph[0][0]
        rng = np.random.default_rng(seed + int(s0)) if seed is not None else base_rng
        e0 = ph[-1][0] + ph[-1][1]
        rel = int(0.14 * SR)
        n = int(e0 - s0) + rel + int(0.05 * SR)
        lf = np.zeros(n)
        amp = np.zeros(n)
        tt = tvec(n)
        prev_m = None
        for i, (s, d, m, v) in enumerate(ph):
            a = int(s - s0)
            b = int(min(n, a + d + (rel if i == len(ph) - 1 else int(0.03 * SR))))
            lf[a:] = m  # hold pitch until next note
            seg = np.arange(b - a) / SR
            if prev_m is None:
                # tongued attack
                att = 1 - np.exp(-seg / 0.035)
            else:
                att = 1 - 0.18 * np.exp(-seg / 0.03)  # slur dip
            sustain = v * (1 - 0.1 * (seg / max(d / SR, 1e-3)))  # slight diminuendo over note
            amp[a:b] = np.maximum(amp[a:b], att * sustain)
            prev_m = m
        # release tail
        endk = int(e0 - s0)
        if endk < n:
            amp[endk:] *= np.exp(-np.arange(n - endk) / SR / 0.06)
        # portamento: smooth the log-pitch curve
        # causal portamento (one-pole glide, ~18 ms) so truncated renders share an identical prefix
        ag = np.exp(-1 / (0.018 * SR))
        lf_s = signal.lfilter([1 - ag], [1, -ag], lf - lf[0]) + lf[0]
        # onset scoop
        scoop = -0.35 * np.exp(-tt / 0.05)
        # vibrato, growing on held notes
        vib_env = np.zeros(n)
        for (s, d, m, v) in ph:
            a = int(s - s0)
            b = int(min(n, a + d))
            seg = np.arange(b - a) / SR
            vib_env[a:b] = np.clip((seg - 0.22) / 0.4, 0, 1) * (1 if d / SR > 0.35 else 0.2)
        vib = vib_depth * vib_env * np.sin(TWOPI * vib_rate * tt + rng.uniform(0, 6))
        vib += 0.03 * np.sin(TWOPI * 0.7 * tt)
        midi = lf_s + scoop + vib
        f = mtof(midi)
        phs = TWOPI * np.cumsum(f) / SR
        amp_s = np.clip(lp(amp, 40), 0, None)
        sig = np.zeros(n)
        for h, ha in enumerate(harm, start=1):
            if np.max(f) * h > SR * 0.45:
                break
            # brighter when louder
            sig += ha * (amp_s ** (0.4 * (h - 1))) * np.sin(h * phs)
        sig *= amp_s
        # chiff at tongued starts (drawn first: independent of phrase length)
        ch_n = int(0.04 * SR)
        chf = bp(rng.standard_normal(ch_n), 1500, 6000) * env_perc(ch_n, 0.002, 0.012) * 0.12 * breath
        br = bp(rng.standard_normal(n), nb_lo, nb_hi) * nb_gain * np.sqrt(np.clip(amp_s, 0, None))
        sig[:ch_n] += chf * ph[0][3]
        sig += br
        y[int(s0) + PRE: int(s0) + PRE + n] += sig[:max(0, min(n, BUF - int(s0) - PRE))]
    return y


def bird_call(points, rng, trill_hz=0.0, trill_depth=0.0, harm=0.12, vel=0.5):
    """points: list of (time_s, freq_hz, amp). Smooth glides between them."""
    T = points[-1][0]
    n = int((T + 0.02) * SR)
    tt = tvec(n)
    ts = np.array([p[0] for p in points])
    fs = np.log(np.array([p[1] for p in points]))
    am = np.array([p[2] for p in points])
    lf = np.interp(tt, ts, fs)
    a = np.interp(tt, ts, am)
    f = np.exp(lf)
    if trill_hz > 0:
        f *= 1 + trill_depth * np.sin(TWOPI * trill_hz * tt)
        a *= 0.65 + 0.35 * np.abs(np.sin(np.pi * trill_hz * tt))
    ph = TWOPI * np.cumsum(f) / SR
    y = (np.sin(ph) + harm * np.sin(2 * ph)) * a
    return fade_out(fade_in(y, 0.004), 0.01) * vel


def bubble(f0, tau, rng, rise=0.12, vel=0.5):
    n = int(tau * 7 * SR)
    t = tvec(n)
    f = f0 * (1 + rise * t / tau)
    ph = TWOPI * np.cumsum(f) / SR
    y = np.sin(ph) * np.exp(-t / tau)
    return fade_in(y, 0.0008) * vel


def insects(n, rng, level=1.0, center=4800, pulse=38.0, swell_cycles=3):
    """cicada/cricket bed, circular (period = loop)."""
    t = tvec(n)
    x = noise(n, rng)
    x = circ_filter(x, sos=signal.butter(2, [center * 0.85, center * 1.15], btype='band', fs=SR, output='sos'))
    # buzz AM with integer cycles over the loop
    cyc = round(pulse * n / SR)
    am = 0.5 + 0.5 * np.sin(TWOPI * cyc * np.arange(n) / n) ** 2
    sw = 0.55 + 0.45 * np.sin(TWOPI * swell_cycles * np.arange(n) / n + rng.uniform(0, 6)) ** 2
    return x * am * sw * level


def crickets(n, rng, f=4400, level=1.0, every=S16 * 4, phase=0, skip=0.45):
    """cricket chirps locked to the 8th-note grid (the jungle keeps time)."""
    y = np.zeros(n)
    tt = tvec(int(0.09 * SR))
    ch = np.zeros(len(tt))
    for k in range(3):
        s = int(k * 0.028 * SR)
        m = min(len(tt) - s, int(0.018 * SR))
        seg = np.sin(TWOPI * f * tt[:m]) * np.hanning(m)
        ch[s:s + m] += seg
    for pos in range(phase, n, every):
        g = rng.uniform(0.3, 1.0) * level
        if rng.random() < skip:
            continue
        fr = f * rng.uniform(0.97, 1.03)
        c2 = np.zeros(len(tt))
        for k in range(3):
            s0 = int(k * 0.028 * SR)
            m = min(len(tt) - s0, int(0.018 * SR))
            c2[s0:s0 + m] += np.sin(TWOPI * fr * tt[:m]) * np.hanning(m)
        idx = (np.arange(len(c2)) + pos) % n
        np.add.at(y, idx, c2 * g)
    return y


def high_shelf_ba(f0, gain_db, S=1.0):
    A = 10 ** (gain_db / 40)
    w0 = TWOPI * f0 / SR
    alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)
    cw = np.cos(w0)
    b0 = A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cw)
    b2 = A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cw)
    a2 = (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha
    return np.array([b0, b1, b2]) / a0, np.array([1, a1 / a0, a2 / a0])
