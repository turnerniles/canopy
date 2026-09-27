import tempfile
TMP = tempfile.gettempdir()
import numpy as np, soundfile as sf, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import librosa, librosa.display, pyloudnorm as pyln
from synth import SR, BAR
path=sys.argv[1]; tag=sys.argv[2]
y,sr=sf.read(path); y=y.T
m=pyln.Meter(SR)
print('integrated LUFS %.1f'%m.integrated_loudness(y.T), 'peak %.1f'%(20*np.log10(np.abs(y).max())))
# short-term loudness (3 s windows, 1 s hop)
st=[]; hop=SR
for i in range(0,y.shape[1]-3*SR,hop):
    st.append(m.integrated_loudness(y[:,i:i+3*SR].T) if np.abs(y[:,i:i+3*SR]).max()>1e-5 else -70)
st=np.array(st)
# click detector: HF energy (>6k) in 5ms frames; flag frames > 12 dB above local median
mono=y.mean(0)
from scipy import signal
hf=signal.sosfilt(signal.butter(4,7000,'high',fs=SR,output='sos'),mono)
fr=int(0.005*SR); e=np.array([np.sum(hf[i:i+fr]**2) for i in range(0,len(hf)-fr,fr)])
ed=10*np.log10(e+1e-12)
med=signal.medfilt(ed,201)
spk=np.nonzero(ed-med>14)[0]
bnd=set(range(0,y.shape[1],4*BAR))
near=[s*fr/SR for s in spk if min(abs(s*fr-b) for b in bnd)<0.05*SR]
print('HF spikes total',len(spk),' within 50ms of a phrase boundary:',len(near), [round(x,2) for x in near[:10]])
fig,ax=plt.subplots(2,1,figsize=(18,8),gridspec_kw={'height_ratios':[3,1.4]})
S=librosa.feature.melspectrogram(y=mono.astype(np.float32),sr=SR,n_fft=4096,hop_length=2048,n_mels=128,fmax=12000)
librosa.display.specshow(librosa.power_to_db(S,ref=np.max),sr=SR,hop_length=2048,x_axis='time',y_axis='mel',fmax=12000,ax=ax[0],vmin=-80,cmap='magma')
for b in range(0,y.shape[1],16*BAR): ax[0].axvline(b/SR,color='c',lw=1); ax[1].axvline(b/SR,color='c',lw=1)
for b in range(0,y.shape[1],4*BAR): ax[1].axvline(b/SR,color='gray',lw=0.3)
ax[1].plot(np.arange(len(st))+1.5,st,'k'); ax[1].set_ylim(-45,-8); ax[1].set_xlim(0,y.shape[1]/SR); ax[1].set_ylabel('short-term LUFS')
plt.tight_layout(); plt.savefig(f'{TMP}/{tag}.png',dpi=60)
