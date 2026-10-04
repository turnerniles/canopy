"""Mix diagnostics for a composed Suite: each part's loudness (loudest 2 s
window, after its gain) relative to the loudest part, plus the spectral
centroid. Helps balance a mix you can't listen to on the spot."""
import sys
import numpy as np
from orchestra.mix import Session


def part_levels(S, parts=None, sr=48000):
    rows = []
    for name, p in S.s.parts.items():
        if parts and not any(x in name for x in parts):
            continue
        b = S.s.render_parts([p], t_end=S.s.length_s())
        x = b['dry'].mean(0)
        nz = np.nonzero(np.abs(x) > 1e-6)[0]
        if not len(nz):
            continue
        seg = x[nz[0]:nz[-1] + 1]
        k = int(2 * sr)
        c = np.cumsum(np.concatenate([[0], seg ** 2]))
        w = (c[k:] - c[:-k]) / k if len(seg) > k else np.array([np.mean(seg ** 2)])
        loud = 10 * np.log10(w.max() + 1e-12)
        S_ = np.abs(np.fft.rfft(seg[:min(len(seg), 1 << 20)]))
        f = np.fft.rfftfreq(min(len(seg), 1 << 20), 1 / sr)
        cen = float((S_ * f).sum() / (S_.sum() + 1e-9))
        rows.append((loud, name, cen, len(p.notes)))
    top = max(r[0] for r in rows)
    for loud, name, cen, n in sorted(rows, reverse=True):
        print(f'{loud - top:6.1f} dB  {name:40s} centroid {cen:6.0f} Hz  notes {n}')
