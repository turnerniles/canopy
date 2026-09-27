"""
Encode the jungle-focus set for the web player (AAC in .mp4, codec-proof sync layout).

Per area:  amb, amb2, amb3, amb4, water, wildlife, birds          (jungle)
           woods, shaker, drums, bass, keys, pad, marimba          (music bed)
           kalimba (River, Vista, Vine Run only: textures, not the theme)
Tails (bar-8 and bar-16 ring-outs) are bundled per stem for birds + music bed.
"""
import tempfile
TMP = tempfile.gettempdir()
import os, json, glob, subprocess
import numpy as np
import soundfile as sf
from synth import SR, N, BAR, SPB, BPM
from score import HARMONY, SECTION_INFO
from conductor import LEADIN
from encode import web_layout, enc, pack, IMP_AT, HEAD, PAD, GAP

WEB = 'out/webj/audio'
os.makedirs(WEB, exist_ok=True)
g = json.load(open('out/gains_jungle.json'))
JUNGLE = ['amb', 'amb2', 'amb3', 'amb4', 'water', 'wildlife', 'birds']
MUSIC = ['woods', 'shaker', 'drums', 'bass', 'keys', 'pad', 'marimba', 'kalimba']
BEDS = ['amb', 'amb2', 'amb3', 'amb4', 'water', 'wildlife']
STINGS = ['gust', 'flock', 'shower', 'woodpecker_roll', 'far_cry', 'fill_hands', 'fill_logs', 'rise']


def stems_for(sec):
    return JUNGLE + [m for m in MUSIC if not (m == 'kalimba' and sec in 'AD')]


def ld(k):
    return np.load(f'out/raw/{k}.npy').astype(np.float64)


# master: loudest "everything on" configuration peaks at -3 dBFS
worst = 0
for sec in 'ABCDE':
    tot = sum(ld(f'{sec}_{s}') * g[f'{sec}_{s}'] for s in stems_for(sec) if s not in ('amb2', 'amb3', 'amb4'))
    worst = max(worst, np.max(np.abs(tot)))
MASTER = 10 ** (-3 / 20) / worst
print('master gain %.2f dB' % (20 * np.log10(MASTER)))


def web_file(key, y, loop, br):
    tmp = f'{TMP}/wj_{key}.wav'
    sf.write(tmp, web_layout(y, loop).T, SR, subtype='FLOAT')
    enc(tmp, f'{WEB}/{key}.mp4', 'aac', br)
    os.remove(tmp)


if __name__ == '__main__':
    meta_stems = {}
    for sec in 'ABCDE':
        meta_stems[sec] = stems_for(sec)
        for stem in stems_for(sec):
            k = f'{sec}_{stem}'
            gain = g[k] * MASTER
            web_file(k, ld(k) * gain, True, '80k')
            if stem not in BEDS:
                tails, _ = pack([ld(k + '_t16') * gain, ld(k + '_t8') * gain])
                web_file(k + '_tails', tails, False, '56k')
            print(k, flush=True)
    sts = [ld('st_' + n) * g['st_' + n] * MASTER for n in STINGS]
    bundle, offs = pack(sts)
    web_file('stingers', bundle, False, '112k')
    meta = dict(
        sr=SR, bpm=BPM, bars=16, loop=N, bar=BAR, beat=SPB, tail=9 * SR, gap=GAP,
        layout=dict(imp=IMP_AT, head=HEAD, pad=PAD),
        stems=meta_stems, beds=BEDS, jungle=JUNGLE, music=MUSIC,
        leadin={s: {'kalimba': LEADIN[s]['kalimba']} for s in 'ABCDE'},
        sections={s: dict(name=SECTION_INFO[s]['name'], mood=SECTION_INFO[s]['mood'],
                          chords=[c['label'] for c in HARMONY[s]]) for s in 'ABCDE'},
        stingers=dict(anchor=BAR, length=3 * BAR, names=STINGS, offsets=offs),
    )
    json.dump(meta, open('out/webj/meta.json', 'w'), indent=1)
    files = glob.glob(f'{WEB}/*.mp4')
    print('files %d  MB %.1f' % (len(files), sum(os.path.getsize(p) for p in files) / 1e6))
