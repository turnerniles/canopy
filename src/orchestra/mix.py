"""Score → audio. A Session holds Parts; each Part is one player on one
instrument (or a rack of instruments) with pan, level, filters, sends and
level automation. Rendering is offline, deterministic and memory-bounded.

    s = Session(bpm=96)
    p = s.part('kalimba', 'mallet.kalimba', pan=-0.3, sends={'hall': .25})
    p.note(beat=0, pitch='D5', beats=0.5, vel=0.8)
    mix = s.render()                      # (2, n) float32, mastered

Time is in *beats* (quarter notes) on the session clock; tempo can change
(Session.tempo_at). Odd meters are just bar lengths in beats — see Meter.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .core import SR, REGISTRY, play, nm, hp, lp, resolve
from . import fx


# ------------------------------------------------------------------ time
class Clock:
    """piecewise-constant tempo map. changes: [(beat, bpm), ...] sorted."""

    def __init__(self, bpm=90.0):
        self.changes = [(0.0, float(bpm))]

    def tempo_at(self, beat, bpm):
        self.changes = sorted([c for c in self.changes if c[0] != beat] + [(float(beat), float(bpm))])

    def sec(self, beat):
        t, prev_b, prev_bpm = 0.0, 0.0, self.changes[0][1]
        for b, bpm in self.changes:
            if b >= beat:
                break
            t += (b - prev_b) * 60.0 / prev_bpm
            prev_b, prev_bpm = b, bpm
        return t + (beat - prev_b) * 60.0 / prev_bpm

    def bpm(self, beat):
        cur = self.changes[0][1]
        for b, bpm in self.changes:
            if b <= beat:
                cur = bpm
        return cur


class Meter:
    """A run of bars with lengths in eighth-notes (or any unit), e.g.
    Meter([15] * 8, unit=8) is eight bars of 15/8. bar(i) -> start beat."""

    def __init__(self, lengths, unit=4, start=0.0):
        self.unit = unit
        self.len = [l * 4.0 / unit for l in lengths]
        self.starts = np.concatenate([[0.0], np.cumsum(self.len)]) + start
        self.start = start

    def bar(self, i, sub=0.0):
        """start beat of bar i (0-based) plus `sub` units into it."""
        return float(self.starts[i] + sub * 4.0 / self.unit)

    @property
    def end(self):
        return float(self.starts[-1])

    def __len__(self):
        return len(self.len)


# ------------------------------------------------------------------ parts
@dataclass
class Note:
    beat: float
    inst: str
    pitch: float
    beats: float
    vel: float
    pan: float
    gain: float
    seed: int


@dataclass
class Part:
    name: str
    inst: str
    pan: float = 0.0
    gain_db: float = 0.0
    sends: dict = field(default_factory=dict)
    hpf: float = 0.0
    lpf: float = 0.0
    width: float = 0.0
    group: str = 'main'          # stem group (for game stems / multitrack export)
    notes: list = field(default_factory=list)
    auto: list = field(default_factory=list)   # [(beat, dB)] level automation
    swing: float = 0.0
    humanize: float = 0.004      # seconds of timing jitter
    eq: list = field(default_factory=list)     # [(f0, gain_db, 'high'|'low'), ...] shelves
    _seed: int = 0

    def note(self, beat, pitch, beats=0.5, vel=0.7, inst=None, pan=None, gain_db=0.0):
        inst = resolve(inst or self.inst)
        if isinstance(pitch, str):
            pitch = nm(pitch)
        info = REGISTRY[inst]
        if info.pitched:   # fold out-of-range notes into the instrument's range by octaves
            while pitch > info.hi and pitch - 12 >= info.lo:
                pitch -= 12
            while pitch < info.lo and pitch + 12 <= info.hi:
                pitch += 12
        self._seed += 1
        self.notes.append(Note(float(beat), inst, float(pitch), float(beats), float(np.clip(vel, 0.02, 1.0)),
                               self.pan if pan is None else float(pan), 10 ** (gain_db / 20), self._seed))
        return self

    def chord(self, beat, pitches, beats=1.0, vel=0.6, strum=0.0, **kw):
        for i, p in enumerate(pitches):
            self.note(beat + i * strum, p, beats, vel, **kw)
        return self

    def level(self, beat, db):
        """automation breakpoint (linear-in-dB ramps between breakpoints)."""
        self.auto.append((float(beat), float(db)))
        return self

    def instruments(self):
        return sorted({n.inst for n in self.notes})


class Session:
    def __init__(self, bpm=90.0, sr=SR, master=None):
        self.clock = Clock(bpm)
        self.sr = sr
        self.parts: dict[str, Part] = {}
        self.echo = dict(beats=0.75, feedback=0.42, damp=3200.0)
        self.master = dict(comp=dict(thr_db=-20, ratio=2.0, attack=0.03, release=0.3), lufs=-14.0, ceiling=-1.0,
                           tape=1.1)
        if master:
            self.master.update(master)
        self._cache = {}
        self._cache_bytes = 0

    def part(self, name, inst, **kw):
        if name in self.parts:
            raise ValueError(f'duplicate part {name}')
        self.parts[name] = Part(name, inst, **kw)
        return self.parts[name]

    def tempo_at(self, beat, bpm):
        self.clock.tempo_at(beat, bpm)

    # -------------------------------------------------------------- rendering
    def _note_audio(self, n: Note, dur_s):
        # the audio must be a pure function of the cache key, or renders become order-dependent
        key = (n.inst, round(n.pitch, 2), round(dur_s, 2), round(n.vel, 2), n.seed % 3)
        y = self._cache.get(key)
        if y is None:
            y = play(n.inst, key[1], dur=key[2], vel=key[3], seed=key[4] + int(key[1] * 7), sr=self.sr)
            if self._cache_bytes > 400e6:
                self._cache.clear()
                self._cache_bytes = 0
            self._cache[key] = y
            self._cache_bytes += y.nbytes
        return y

    def instruments(self):
        return sorted({i for p in self.parts.values() for i in p.instruments()})

    def length_s(self, tail=8.0):
        last = max((self.clock.sec(n.beat + n.beats) for p in self.parts.values() for n in p.notes), default=0)
        return last + tail

    def render_parts(self, parts=None, t_end=None, onset_before=None, rng_seed=7, progress=False,
                     onset_after=None, t_offset=0.0):
        """Render parts into dry stereo + send buses. Returns dict with 'dry'
        (2,n) and sends {name: (2,n)}. onset_before: only notes whose onset
        (seconds) is < this (used for loop tail8 rendering). onset_after /
        t_offset render a time window (chunked rendering): only notes with
        onset >= onset_after, placed relative to t_offset."""
        sr = self.sr
        parts = list(self.parts.values()) if parts is None else parts
        n_total = int(((t_end or self.length_s()) - t_offset) * sr)
        off = int(round(t_offset * sr))
        dry = np.zeros((2, n_total), np.float32)
        sends = {}
        rng = np.random.default_rng(rng_seed)
        t0 = time.time()
        for pi, part in enumerate(parts):
            notes = [n for n in part.notes if (onset_before is None or self.clock.sec(n.beat) < onset_before)
                     and (onset_after is None or self.clock.sec(n.beat) >= onset_after)]
            if not notes:
                continue
            on = [self.clock.sec(n.beat + (part.swing if abs((n.beat * 2) % 2 - 1) < 1e-6 else 0.0)) for n in notes]
            s0 = int(min(on) * sr) - int(0.05 * sr) - off      # may be < 0: trimmed when mixed
            rendered = []
            end = s0
            for n, t in zip(notes, on):
                dur_s = self.clock.sec(n.beat + n.beats) - self.clock.sec(n.beat)
                y = self._note_audio(n, max(0.01, dur_s))
                hj = np.random.default_rng(n.seed * 7919 + int(n.pitch)).normal(0, part.humanize) if part.humanize else 0.0
                st = max(int((t + hj) * sr) - off, s0)
                rendered.append((st, y, n))
                end = max(end, st + y.shape[-1])
            end = min(end, n_total)
            if end <= s0:
                continue
            buf = np.zeros((2, end - s0), np.float32)
            for st, y, n in rendered:
                if st >= end:
                    continue
                k = min(y.shape[-1], end - st)
                if y.ndim == 1:
                    a = (n.pan + 1) * np.pi / 4
                    buf[0, st - s0:st - s0 + k] += y[:k] * (np.cos(a) * n.gain)
                    buf[1, st - s0:st - s0 + k] += y[:k] * (np.sin(a) * n.gain)
                else:
                    # stereo instrument: balance-pan
                    gl = min(1.0, 1 - n.pan) * n.gain
                    gr = min(1.0, 1 + n.pan) * n.gain
                    buf[0, st - s0:st - s0 + k] += y[0, :k] * gl
                    buf[1, st - s0:st - s0 + k] += y[1, :k] * gr
            if part.hpf:
                buf = np.stack([hp(c, part.hpf, sr=sr) for c in buf]).astype(np.float32)
            if part.lpf:
                buf = np.stack([lp(c, part.lpf, sr=sr) for c in buf]).astype(np.float32)
            for f0, gdb, kind in part.eq:
                buf = fx.shelf(buf, f0, gdb, kind, sr=sr)
            if part.width:
                d = int(0.012 * part.width * sr) + 1
                side = 0.5 * (buf[0] - buf[1])
                mid = 0.5 * (buf[0] + buf[1])
                side2 = side.copy()
                side2[d:] += 0.5 * part.width * mid[:-d]
                buf = np.stack([mid + side2, mid - side2]).astype(np.float32)
            g = np.full(buf.shape[1], 10 ** (part.gain_db / 20), np.float32)
            if part.auto:
                pts = sorted(part.auto)
                ts = np.array([self.clock.sec(b) for b, _ in pts]) * sr - s0 - off
                dbs = np.array([d for _, d in pts])
                g *= (10 ** (np.interp(np.arange(buf.shape[1]), ts, dbs) / 20)).astype(np.float32)
            buf *= g
            if s0 < 0:          # sound that starts before this buffer (early pickups / window pre-roll)
                buf = buf[:, -s0:]
                s0 = 0
            dry[:, s0:end] += buf
            for name, amt in part.sends.items():
                if amt <= 0:
                    continue
                if name not in sends:
                    sends[name] = np.zeros((2, n_total), np.float32)
                sends[name][:, s0:end] += buf * amt
            if progress and (pi % 10 == 0):
                print(f'  rendered {pi + 1}/{len(parts)} parts  ({time.time() - t0:.0f}s)', flush=True)
        return dict(dry=dry, sends=sends, n=n_total)

    def wet(self, buses, n_total):
        out = buses['dry'].copy()
        for name, x in buses['sends'].items():
            if name == 'echo':
                d = self.clock.sec(self.echo['beats']) - self.clock.sec(0)
                w = fx.tape_echo(x, d, self.echo['feedback'], self.echo['damp'], sr=self.sr)
                # echoes also get a little room
                w = w[:, :n_total]
                out += w
                out += fx.reverb(w * 0.3, 'hall', self.sr)[:, :n_total]
            else:
                w = fx.reverb(x, name, self.sr)[:, :n_total]
                out[:, :w.shape[1]] += w
        return out

    def master_bus(self, x):
        m = self.master
        for f0, gdb, kind in m.get('eq', []):
            x = fx.shelf(x, f0, gdb, kind, sr=self.sr)
        if m.get('comp'):
            x = fx.compress(x, sr=self.sr, **m['comp'])
        if m.get('tape'):
            x = fx.tape(x * 0.9, m['tape'])
        x, L = fx.loudness_normalize(x, m.get('lufs', -14.0), self.sr)
        x = fx.limit(x, m.get('ceiling', -1.0), sr=self.sr)
        return x, L

    def render_chunked(self, chunk_s=90.0, tail_s=20.0, progress=True):
        """Memory-bounded full render: notes are rendered window by window and
        each window's wet signal (reverb/echo tails included) is overlap-added
        into the master. Exact, because every effect before the master bus is
        linear."""
        total = self.length_s()
        n_total = int(total * self.sr)
        out = np.zeros((2, n_total), np.float32)
        t0 = 0.0
        while t0 < total:
            t1 = min(total, t0 + chunk_s)
            pre = 0.2 if t0 > 0 else 0.0          # humanised notes may land just before t0
            b = self.render_parts(t_end=min(total, t1 + tail_s), onset_after=t0 if t0 > 0 else None,
                                  onset_before=t1 if t1 < total else None, t_offset=t0 - pre)
            w = self.wet(b, b['n'])
            del b
            s = int(round((t0 - pre) * self.sr))
            k = min(w.shape[1], n_total - s)
            out[:, s:s + k] += w[:, :k]
            del w
            if progress:
                print(f'  chunk {t0:6.0f}-{t1:6.0f} s rendered', flush=True)
            t0 = t1
        return self.master_bus(out)

    def render(self, progress=True, groups=False):
        """Full mastered mix. groups=True also returns per-group wet stems (unmastered)."""
        n_total = int(self.length_s() * self.sr)
        if not groups:
            buses = self.render_parts(t_end=n_total / self.sr, progress=progress)
            mix = self.wet(buses, n_total)
            del buses
            return self.master_bus(mix)
        stems = {}
        for g in sorted({p.group for p in self.parts.values()}):
            ps = [p for p in self.parts.values() if p.group == g]
            buses = self.render_parts(ps, t_end=n_total / self.sr, progress=False)
            stems[g] = self.wet(buses, n_total)
        mix = sum(stems.values())
        return self.master_bus(mix), stems
