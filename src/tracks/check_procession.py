"""Verify The Procession: in each 4-bar feature window, the announced
instrument must be the loudest part, and not buried under the rest of the band."""
import numpy as np
from tracks import kapok

S = kapok.compose(['V'], False)
s = S.s
sr = s.sr
T = s.length_s()
parts = list(s.parts.values())
# render every part once (dry), then measure per window
dry = {}
for p in parts:
    b = s.render_parts([p], t_end=T)['dry']
    from orchestra import fx
    from orchestra.core import hp
    m = 0.5 * (b[0] + b[1])
    dry[p.name] = fx.shelf(hp(m, 120.0, sr=sr), 1500, 4.0, 'high', sr)   # ~K-weighting: what the ear hears
keys = {'Grand piano': 'keys.grand_piano', 'Reed and pipe organ': 'organ.pipe', 'Glockenspiel': 'bell.glockenspiel',
        'Bass guitar': 'bass.fingered', 'Double-speed guitar': 'guitar.double_speed',
        'Two slightly distorted guitars': 'guitar.oldfield_lead', 'Mandolin': 'pluck.mandolin',
        'Spanish guitar, and introducing acoustic guitar': 'guitar.nylon_spanish', 'Bamboo flutes': 'flute.bamboo',
        'The tree-frog choir': 'creature.treefrog_choir', 'Steel pans': 'metal.steel_pan',
        'Howler monkeys': 'creature.howler_roar', 'Plus ... tubular bells!': 'bell.tubular'}
bad = 0
for t_cue, name in S.intros:
    a, b = int((t_cue + 0.9) * sr), int((t_cue + 0.9 + 9.6) * sr)
    lv = {n: float(np.sqrt(np.mean(x[a:b] ** 2)) + 1e-9) for n, x in dry.items()}
    # the first-defined part using the announced instrument is its lead part
    lead = next(p.name for p in parts if p.inst == keys[name])
    others = np.sqrt(sum(v ** 2 for n, v in lv.items() if n != lead))
    rank = sorted(lv, key=lambda n: -lv[n]).index(lead) + 1
    rel = 20 * np.log10(lv[lead] / others)
    top = sorted(lv, key=lambda n: -lv[n])[0]
    ok = rank == 1 and rel > -4
    bad += not ok
    print(f'{"ok " if ok else "BAD"} {name:48s} rank {rank}  lead vs rest {rel:+5.1f} dB   loudest: {top}')
print('problems:', bad)
