"""Fast regression checks for Canopy Orchestra (about a minute).

    cd src && python3 -m orchestra.selftest

Covers the things that broke while building it: chord parsing, render
determinism (cache keys), chunked == full render, loop seams that wrap early
pickups, tails that decay, and the shared cadence grammar of the shipped worlds.
"""
import json
import os
import tempfile

import numpy as np

from . import REGISTRY, play, theory as th
from .mix import Session, Meter
from .world import build, grammar_problems

HERE = os.path.dirname(os.path.abspath(__file__))


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print('ok ', msg)


def main():
    check(len(REGISTRY) >= 200, f'{len(REGISTRY)} presets registered')
    check(th.pcs('Bbmaj7#11') == [2, 4, 5, 9, 10], 'chord parser: Bbmaj7#11')
    check(th.pcs('A7sus4b9') == [2, 4, 7, 9, 10], 'chord parser: A7sus4b9')
    check(th.chord('E/D')['bass'] == 2, 'chord parser: slash bass')
    a = play('mallet.kalimba', 'D5', 0.5, 0.8, seed=3)
    b = play('mallet.kalimba', 'D5', 0.5, 0.8, seed=3)
    check(np.array_equal(a, b), 'presets are deterministic')

    def session():
        s = Session(bpm=120)
        s.master.update(lufs=-16.0)
        p = s.part('k', 'mallet.kalimba', sends={'hall': 0.3, 'echo': 0.2}, humanize=0.006)
        q = s.part('b', 'mallet.balafon', sends={'canopy': 0.2})
        m = Meter([15] * 6, unit=8)
        for bar in range(6):
            for k in range(15):
                p.note(m.bar(bar, k), 69 + (k * 5) % 12, 0.27 + 0.01 * (k % 3), 0.5 + 0.03 * (k % 5))
            q.note(m.bar(bar), 50, 2.0, 0.7)
        return s
    x1, _ = session().render(progress=False)
    x2, _ = session().render_chunked(chunk_s=7.0, progress=False)
    n = min(x1.shape[1], x2.shape[1])
    err = 20 * np.log10(np.sqrt(np.mean((x1[:, :n] - x2[:, :n]) ** 2)) / np.sqrt(np.mean(x1 ** 2)) + 1e-12)
    check(err < -60, f'chunked render matches full render ({err:.0f} dB)')

    spec = dict(id='selftest', bpm=120, bars=16, key='D', mode='lydian', tail_s=9.0, areas=dict(A=dict(
        room='canopy', progression=['Dmaj9'] * 7 + ['Asus2'] + ['Gmaj9'] * 7 + ['A'],
        layers=dict(cell=dict(inst='kalimba', pattern='arp', rhythm='x.x.x.x.', humanize=0.006, tier=1),
                    pad=dict(inst='section_warm', pattern='pad', tier=1)))))
    for f in ('jungle_temple.json', 'orbital_station.json'):
        W = json.load(open(os.path.join(HERE, '..', 'worlds', f)))
        for a, A in W['areas'].items():
            check(not grammar_problems(A['progression'], W['key']), f'{f} area {a} obeys the cadence grammar')
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'w.json')
        json.dump(spec, open(path, 'w'))
        meta = build(path, out_root=os.path.join(d, 'out'), fmt='wav', quiet=True)
        import soundfile as sf
        for st in meta['stems']['A']:
            x, _ = sf.read(os.path.join(d, 'out', f'A_{st}.wav'), dtype='float32')
            check(len(x) == meta['loop'], f'{st}: loop is exactly {meta["loop"]} samples')
            dj = np.abs(np.diff(x, axis=0)).max(1)
            seam = np.abs(x[0] - x[-1]).max() / (np.percentile(dj, 99.9) + 1e-9)
            check(seam < 1.5, f'{st}: loop seam is continuous ({seam:.2f}x typical transient)')
            t, _ = sf.read(os.path.join(d, 'out', f'A_{st}_tail16.wav'), dtype='float32')
            check(np.abs(t[-2000:]).max() < 1e-3, f'{st}: tail decays to silence')
    print('selftest passed')


if __name__ == '__main__':
    main()
