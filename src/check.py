import tempfile
TMP = tempfile.gettempdir()
import numpy as np, json, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import librosa
from mix import load, STEMS
from synth import SR, BAR, SPB, N, nm
from score import MELODY, HARMONY
sec=sys.argv[1] if len(sys.argv)>1 else 'A'
g=json.load(open('out/gains_pre.json'))
st={s:load(sec,s)*g[f'{sec}_{s}'] for s in STEMS}
full=sum(st.values())
# 1. tonal balance (1/3-oct) of full and per stem
def bands(y):
    m=y.mean(0); F=np.abs(np.fft.rfft(m))**2; f=np.fft.rfftfreq(len(m),1/SR)
    cs=[31.5*2**(i/3) for i in range(28)]
    return cs,[10*np.log10(F[(f>=c/2**(1/6))&(f<c*2**(1/6))].sum()+1e-20) for c in cs]
fig,ax=plt.subplots(1,1,figsize=(12,6))
cs,fb=bands(full); ax.semilogx(cs,np.array(fb)-max(fb),'k',lw=3,label='FULL')
for s in STEMS:
    cs,b=bands(st[s]); ax.semilogx(cs,np.array(b)-max(fb),label=s)
ref=np.array([-4.5*np.log2(max(c,100)/100) for c in cs]); ax.semilogx(cs,ref-3,'k--',label='-4.5dB/oct ref')
ax.set_ylim(-60,5); ax.legend(ncol=4,fontsize=8); ax.grid(alpha=.3); plt.savefig(f'{TMP}/{sec}_bal.png',dpi=65)
print('full band levels:', ' '.join(f'{int(c)}:{b-max(fb):.0f}' for i,(c,b) in enumerate(zip(cs,fb)) if i%3==0))
# 2. melody pitch accuracy
if False:
    y=st['melody'].mean(0)
    y2=librosa.resample(y.astype(np.float32),orig_sr=SR,target_sr=16000)
    f0,vf,_=librosa.pyin(y2,fmin=200,fmax=1600,sr=16000,frame_length=1024,hop_length=160)
    times=np.arange(len(f0))*160/16000
    errs=[]
    for (bar,beat,d,n) in MELODY[sec]:
        t0=((bar-1)*4+beat-1)*SPB/SR; t1=t0+d*SPB/SR
        sel=(times>t0+0.06)&(times<t1-0.03)&vf
        if sel.sum()<3: errs.append((bar,beat,n,'unvoiced')); continue
        est=np.median(librosa.hz_to_midi(f0[sel]))
        if abs(est-nm(n))>0.35: errs.append((bar,beat,n,round(est,2)))
    print('melody notes',len(MELODY[sec]),'pitch errors',len(errs), errs[:8])
# 3. keys chroma vs chord
y=st['keys'].mean(0).astype(np.float32)
C=librosa.feature.chroma_stft(y=y,sr=SR,n_fft=8192,hop_length=4096)
fr=C.shape[1]/16
bad=0
for b in range(16):
    c=C[:,int(b*fr+1):int(b*fr+fr*0.5)].mean(1)
    top=set(np.argsort(c)[-3:])
    pcs=set(x%12 for x in HARMONY[sec][b]['ep'])
    if not top<=pcs|{HARMONY[sec][b]['bass']%12}: bad+=1; print(' bar',b+1,HARMONY[sec][b]['label'],'top chroma',sorted(top),'chord',sorted(pcs))
print('keys bars off-chord:',bad)
# stereo correlation
L,R=full; print('stereo corr %.2f'%np.corrcoef(L,R)[0,1])
