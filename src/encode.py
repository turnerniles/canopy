"""
Encode stems for the web player and for engines.

Web files (Opus/Ogg primary, AAC/M4A fallback) use a codec-proof layout:
   [0 .. 8192)            silence with a single sync impulse at sample 1000
   [8192 .. 8192+PAD)     the loop's last PAD samples   (loop files)  / silence (one-shots)
   [.. +LEN)              the content (exactly LEN samples)
   [.. +PAD)              the loop's first PAD samples  (loop files)  / silence
The player finds the impulse (codec delays vary by codec and browser), then
cuts the content out to the exact sample. The overlap pads mean the lossy codec
sees real neighbours at the seam, so the loop point stays click-free.

Engine files: gapless Ogg Vorbis + 24-bit WAV of exactly LEN samples (no pads).
"""
import tempfile
TMP = tempfile.gettempdir()
import os, json, glob, subprocess, sys
import numpy as np
import soundfile as sf
from synth import SR, N, BAR, SPB, BPM
from render import STEMS
from conductor import LEADIN
from score import HARMONY, SECTION_INFO

IMP_AT, HEAD, PAD = 1000, 8192, 4096
MASTER = 10 ** (-1.5 / 20)
g = json.load(open('out/gains_pre.json'))
WEB = 'out/web/audio'
ENG = 'out/engine'
os.makedirs(WEB, exist_ok=True)


def web_layout(y, loop):
    L = y.shape[1]
    out = np.zeros((2, HEAD + PAD + L + PAD + 2048), dtype=np.float32)
    out[:, IMP_AT] = 0.6
    if loop:
        out[:, HEAD:HEAD + PAD] = y[:, -PAD:]
        out[:, HEAD + PAD + L:HEAD + PAD + L + PAD] = y[:, :PAD]
    out[:, HEAD + PAD:HEAD + PAD + L] = y
    return out


def enc(src_wav, dst, codec, br):
    args = ['ffmpeg', '-y', '-loglevel', 'error', '-i', src_wav]
    if codec == 'opus':
        args += ['-c:a', 'libopus', '-b:a', br, '-vbr', 'on', '-application', 'audio', dst]
    elif codec == 'aac':
        args += ['-c:a', 'aac', '-b:a', br, '-movflags', '+faststart', dst]
    elif codec == 'vorbis':
        args += ['-c:a', 'libvorbis', '-q:a', br, dst]
    subprocess.run(args, check=True)


TL = 9 * SR
GAP = SR // 2


def web_file(key, y, loop, br):
    tmp = f'{TMP}/enc_{key}.wav'
    sf.write(tmp, web_layout(y, loop).T, SR, subtype='FLOAT')
    enc(tmp, f'{WEB}/{key}.m4a', 'aac', br)
    os.remove(tmp)


def engine_file(sec, key, y):
    d = f'{ENG}/{sec}'
    os.makedirs(d, exist_ok=True)
    tmp = f'{TMP}/eng_{key}.wav'
    sf.write(tmp, y.T, SR, subtype='PCM_24')
    enc(tmp, f'{d}/{key}.ogg', 'vorbis', '5')
    if os.environ.get('KEEP_WAV'):
        os.replace(tmp, f'{d}/{key}.wav')
    else:
        os.remove(tmp)


def pack(parts):
    """concatenate one-shots with a gap; returns array and offsets (samples from content start)"""
    offs, out = [], []
    pos = 0
    for p in parts:
        offs.append(pos)
        out.append(p)
        out.append(np.zeros((2, GAP)))
        pos += p.shape[1] + GAP
    return np.concatenate(out, axis=1), offs


if __name__ == '__main__':
    only = sys.argv[1] if len(sys.argv) > 1 else 'ABCDE'
    for sec in only:
        for stem in STEMS:
            k = f'{sec}_{stem}'
            gain = g[k] * MASTER
            loop = np.load(f'out/raw/{k}.npy') * gain
            t16 = np.load(f'out/raw/{k}_t16.npy') * gain
            t8 = np.load(f'out/raw/{k}_t8.npy') * gain
            web_file(k, loop, True, '96k')
            tails, offs = pack([t16, t8])
            web_file(k + '_tails', tails, False, '64k')
            engine_file(sec, k, loop)
            engine_file(sec, k + '_tail16', t16)
            engine_file(sec, k + '_tail8', t8)
            print(k, flush=True)
    names = [os.path.basename(p)[3:-4] for p in sorted(glob.glob('out/raw/st_*.npy'))]
    sts = [np.load(f'out/raw/st_{n}.npy') * g['st_' + n] * MASTER for n in names]
    bundle, st_offs = pack(sts)
    web_file('stingers', bundle, False, '112k')
    for n, y in zip(names, sts):
        engine_file('stingers', 'st_' + n, y)
    meta = dict(
        sr=SR, bpm=BPM, beat=SPB, bar=BAR, bars=16, loop=N, tail=TL, tailGap=GAP,
        layout=dict(imp=IMP_AT, head=HEAD, pad=PAD),
        stems=STEMS, leadin=LEADIN,
        sections={s: dict(name=SECTION_INFO[s]['name'], mood=SECTION_INFO[s]['mood'],
                          chords=[c['label'] for c in HARMONY[s]]) for s in 'ABCDE'},
        stingers=dict(anchor=BAR, length=3 * BAR, names=names, offsets=st_offs),
    )
    json.dump(meta, open('out/web/meta.json', 'w'), indent=1)
    tot = sum(os.path.getsize(p) for p in glob.glob(f'{WEB}/*.m4a'))
    print('web MB %.1f  files %d' % (tot / 1e6, len(glob.glob(f'{WEB}/*.m4a'))))
