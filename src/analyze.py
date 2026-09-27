import numpy as np, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa, librosa.display
import pyloudnorm as pyln
SR=48000
def spectro(y, png, title='', sr=SR, fmax=12000, seg=None, marks=None):
    if y.ndim==2: y=y.mean(0)
    S=librosa.feature.melspectrogram(y=y.astype(np.float32),sr=sr,n_fft=4096,hop_length=1024,n_mels=160,fmax=fmax)
    D=librosa.power_to_db(S,ref=np.max)
    fig,ax=plt.subplots(2,1,figsize=(16,7),gridspec_kw={'height_ratios':[3,1]})
    librosa.display.specshow(D,sr=sr,hop_length=1024,x_axis='time',y_axis='mel',fmax=fmax,ax=ax[0],vmin=-80,cmap='magma')
    ax[0].set_title(title)
    t=np.arange(len(y))/sr
    step=max(1,len(y)//20000)
    ax[1].plot(t[::step],y[::step],lw=0.3); ax[1].set_xlim(0,t[-1])
    if marks is not None:
        for m in marks: ax[0].axvline(m,color='c',lw=0.5,alpha=.6); ax[1].axvline(m,color='r',lw=0.5,alpha=.5)
    plt.tight_layout(); plt.savefig(png,dpi=70); plt.close()
def lufs(y,sr=SR):
    m=pyln.Meter(sr); 
    return m.integrated_loudness(y.T if y.ndim==2 else y)
def peak_db(y): return 20*np.log10(np.max(np.abs(y))+1e-12)
