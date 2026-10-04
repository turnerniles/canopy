"""Export KAPOK for the web player (player/kapok.html).

    cd src && python3 -m tracks.kapok_web ../out/kapok [MC_DIR]

Writes
  player/kapok/kapok.json      timeline: movements, bar grid with meter and chord,
                               credits (234 instruments with first entrance),
                               who is sounding when (merged intervals per instrument),
                               family activity, announcer schedule
  player/audio/kapok.m4a       the suite, AAC (commit it normally with git), and
  player/audio/kapok_b64/*.txt the same bytes as base64 chunks — the player falls
                               back to these when the binary isn't in the checkout
  player/audio/kapok_mc.m4a    (+ _b64) the 13 spoken introductions, one clip after another
"""
import base64
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from orchestra import REGISTRY  # noqa: E402
from tracks import kapok  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PLAYER = os.path.abspath(os.path.join(HERE, '..', '..', 'player'))
CHUNK = 1_500_000      # base64 characters per chunk file


def b64_chunks(path, outdir):
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        os.remove(os.path.join(outdir, f))
    data = base64.b64encode(open(path, 'rb').read()).decode('ascii')
    names = []
    for i in range(0, len(data), CHUNK):
        n = f'part{i // CHUNK:02d}.txt'
        open(os.path.join(outdir, n), 'w').write(data[i:i + CHUNK])
        names.append(n)
    return names


def aac(src, dst, rate='128k', mono=False):
    cmd = ['ffmpeg', '-loglevel', 'error', '-y', '-i', src, '-c:a', 'aac', '-b:a', rate, '-movflags', '+faststart']
    if mono:
        cmd += ['-ac', '1']
    subprocess.run(cmd + [dst], check=True)


def main(out_dir, mc_dir=None):
    S = kapok.compose()
    s = S.s
    T = json.load(open(os.path.join(out_dir, 'kapok_cues.json')))['length']
    fams = sorted({REGISTRY[i].family for i in s.instruments()})
    # who sounds when: merged intervals per instrument (gaps under 0.6 s are bridged)
    spans = {}
    for p in s.parts.values():
        for n in p.notes:
            a = s.clock.sec(n.beat)
            b = max(s.clock.sec(n.beat + n.beats), a + 0.3)
            spans.setdefault(n.inst, []).append((a, b))
    merged = {}
    for inst, iv in spans.items():
        iv.sort()
        out = [list(iv[0])]
        for a, b in iv[1:]:
            if a - out[-1][1] < 0.6:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        merged[inst] = [[round(a, 2), round(b, 2)] for a, b in out]
    # family activity: number of distinct instruments sounding, per 0.25 s frame (0-9)
    F = int(T / 0.25) + 1
    act = {f: np.zeros(F, int) for f in fams}
    for inst, iv in merged.items():
        f = REGISTRY[inst].family
        on = np.zeros(F, bool)
        for a, b in iv:
            on[int(a / 0.25):int(b / 0.25) + 1] = True
        act[f] += on
    activity = {f: ''.join(str(min(9, v)) for v in a) for f, a in act.items()}
    credits = [dict(t=round(t, 2), id=name, family=REGISTRY[name].family, desc=REGISTRY[name].desc)
               for name, t in S.credits()]
    mc = []
    os.makedirs(os.path.join(PLAYER, 'kapok'), exist_ok=True)
    os.makedirs(os.path.join(PLAYER, 'audio'), exist_ok=True)
    if mc_dir:
        sr = 48000
        clips, pos = [], 0
        for i, (t_cue, text) in enumerate(S.intros):
            y, r = sf.read(os.path.join(mc_dir, f'mc_{i + 1:02d}.wav'), dtype='float32')
            if y.ndim > 1:
                y = y.mean(1)
            if r != sr:
                from scipy import signal
                y = signal.resample_poly(y, sr, r).astype(np.float32)
            e = np.convolve(np.abs(y), np.ones(480) / 480, 'same')
            on = np.nonzero(e > 0.02 * e.max())[0]
            y = y[max(0, on[0] - 480):on[-1] + 2400]
            y = y / (np.sqrt(np.mean(y ** 2)) + 1e-9) * 0.1
            entry = t_cue + 0.9
            mc.append(dict(text=text, at=round(entry - len(y) / sr + 0.25, 3), offset=round(pos / sr, 3),
                           dur=round(len(y) / sr, 3)))
            clips += [y, np.zeros(int(0.5 * sr), np.float32)]
            pos += len(y) + int(0.5 * sr)
        tmp = os.path.join(out_dir, 'kapok_mc_clips.wav')
        sf.write(tmp, np.concatenate(clips), sr, subtype='PCM_16')
        aac(tmp, os.path.join(PLAYER, 'audio', 'kapok_mc.m4a'), '64k', mono=True)
    aac(os.path.join(out_dir, 'kapok.wav'), os.path.join(PLAYER, 'audio', 'kapok.m4a'), '128k')
    parts = b64_chunks(os.path.join(PLAYER, 'audio', 'kapok.m4a'), os.path.join(PLAYER, 'audio', 'kapok_b64'))
    mc_parts = b64_chunks(os.path.join(PLAYER, 'audio', 'kapok_mc.m4a'),
                          os.path.join(PLAYER, 'audio', 'kapok_mc_b64')) if mc_dir else []
    doc = dict(title='KAPOK', length=round(T, 3), families=fams,
               movements=[dict(t=round(t, 3), name=l) for t, l in sorted(S.markers)],
               bars=[dict(t=t, meter=m, chord=c) for t, m, c in S.bars],
               credits=credits, spans=merged, activity=dict(frame=0.25, **activity), mc=mc,
               audio=dict(file='audio/kapok.m4a', b64=['audio/kapok_b64/' + p for p in parts],
                          mc_file='audio/kapok_mc.m4a' if mc_dir else None,
                          mc_b64=['audio/kapok_mc_b64/' + p for p in mc_parts]))
    json.dump(doc, open(os.path.join(PLAYER, 'kapok', 'kapok.json'), 'w'), separators=(',', ':'))
    kb = os.path.getsize(os.path.join(PLAYER, 'kapok', 'kapok.json')) // 1024
    print(f'kapok.json {kb} KB, {len(credits)} credits, {len(S.bars)} bars, audio {len(parts)} chunks, mc {len(mc)} clips')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
