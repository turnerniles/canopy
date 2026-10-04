"""Sanity-check every registered preset (or those matching a prefix).

    python -m orchestra.check_instruments [prefix] [--wav out.wav]

Checks: finite output, no clipping after calibration, sensible length, a tail
that decays to silence, render speed, and (for pitched presets not tagged
'inharmonic') that the perceived fundamental is within 35 cents of the target.
--wav writes a demo: every preset plays a short phrase, in registry order.
"""
import sys, time
import numpy as np
from . import REGISTRY, play, names, save_calibration, calibrate, SR, mtof


def f0_estimate(y, sr, fexp):
    m = y if y.ndim == 1 else y.mean(0)
    a, b = int(0.04 * sr), int(0.34 * sr)
    seg = m[a:b] if len(m) > b else m[:int(0.3 * sr)]
    if len(seg) < 2048:
        return None
    seg = seg * np.hanning(len(seg))
    nfft = 1 << 18
    S = np.abs(np.fft.rfft(seg, nfft))
    fr = np.fft.rfftfreq(nfft, 1 / sr)
    # search for the strongest peak within +-1 semitone of the expected f0, and
    # also accept octave errors as warnings only
    lo, hi = fexp * 2 ** (-1 / 12), fexp * 2 ** (1 / 12)
    band = (fr > lo) & (fr < hi)
    if not band.any():
        return None
    k = np.argmax(S * band)
    # strength relative to the overall peak (is the fundamental present at all?)
    return fr[k], S[k] / (S.max() + 1e-12)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    prefix = args[0] if args else ''
    wav = None
    if '--wav' in sys.argv:
        wav = sys.argv[sys.argv.index('--wav') + 1]
    sel = [n for n in names() if n.startswith(prefix)]
    bad = 0
    demo = []
    for name in sel:
        inst = REGISTRY[name]
        calibrate(name, force=True)   # params may have changed since the cache was written
        probs = []
        pitches = [inst.lo, (inst.lo + inst.hi) // 2, inst.hi] if inst.pitched else [60]
        t0 = time.time()
        for p in pitches:
            for vel in (0.3, 1.0):
                y = play(name, p, dur=1.0, vel=vel, seed=p)
                if not np.all(np.isfinite(y)):
                    probs.append('non-finite')
                pk = float(np.max(np.abs(y)))
                if pk > 0.98:
                    probs.append(f'peak {pk:.2f} at {p}/{vel}')
                if y.shape[-1] > 14 * SR:
                    probs.append(f'too long {y.shape[-1] / SR:.1f}s')
                m = y if y.ndim == 1 else y.mean(0)
                tail = m[-int(0.05 * SR):]
                if np.sqrt(np.mean(tail ** 2)) > 0.02 * (np.max(np.abs(m)) + 1e-9):
                    probs.append(f'tail not decayed at {p}')
                if inst.pitched and 'inharmonic' not in inst.tags and vel == 1.0:
                    est = f0_estimate(y, SR, float(mtof(p)))
                    if est:
                        fh, rel = est
                        cents = 1200 * np.log2(fh / float(mtof(p)))
                        if abs(cents) > 35:
                            probs.append(f'pitch {cents:+.0f}c at {p}')
                        elif rel < 0.04:
                            probs.append(f'weak fundamental ({rel:.3f}) at {p}')
        dt = (time.time() - t0) / (len(pitches) * 2)
        if dt > 0.25:
            probs.append(f'slow {dt * 1000:.0f} ms/note')
        status = 'ok ' if not probs else 'BAD'
        if probs:
            bad += 1
        print(f'{status} {name:38s} {inst.family:10s} {dt * 1000:6.0f} ms  ' + '; '.join(sorted(set(probs))))
        if wav:
            demo.append(name)
    # merge with whatever is on disk (other families may be calibrating in parallel)
    import json, os
    from . import core
    if os.path.exists(core._CAL_PATH):
        try:
            disk = json.load(open(core._CAL_PATH))
            disk.update({k: core._CAL[k] for k in sel})
            core._CAL.update({k: v for k, v in disk.items() if k not in sel})
        except Exception:
            pass
    save_calibration()
    print(f'\n{len(sel)} presets, {bad} with problems')
    if wav:
        import soundfile as sf
        out = []
        for name in demo:
            inst = REGISTRY[name]
            if inst.pitched:
                c = (inst.lo + inst.hi) // 2
                seq = [c, c + 4, c + 7, c + 12] if c + 12 <= inst.hi else [c - 5, c - 1, c + 2, c]
            else:
                seq = [60, 60, 60, 60]
            buf = np.zeros((2, int(3.2 * SR)), np.float32)
            for i, p in enumerate(seq):
                y = play(name, p, dur=0.45 if i < 3 else 1.2, vel=[0.6, 0.75, 0.9, 0.8][i], seed=i)
                if y.ndim == 1:
                    y = np.stack([y, y])
                s = int(i * 0.5 * SR)
                k = min(buf.shape[1] - s, y.shape[1])
                buf[:, s:s + k] += y[:, :k]
            out.append(buf)
        sf.write(wav, np.concatenate(out, axis=1).T * 0.7, SR, subtype='PCM_16')
        print('wrote', wav)
    return bad


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
