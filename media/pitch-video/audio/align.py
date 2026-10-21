import numpy as np, soundfile as sf, librosa, json, sys
sys.path.insert(0,'.'); from phrases_new import P
from kokoro_onnx import Kokoro
k=Kokoro("../tts/models/kokoro-v1.0.onnx","../tts/models/voices-v1.0.bin")
SR=24000
ref=[]; bounds=[]; t=0
for s,ps in P.items():
    for j,txt in enumerate(ps):
        a,_=k.create(txt,voice="am_michael",speed=1.0,lang="en-us")
        idx=np.where(np.abs(a)>0.01)[0]; a=a[idx[0]:idx[-1]]
        gap=np.zeros(int(0.35*SR))
        bounds.append((s,j,t,t+len(a)/SR)); ref.append(a); ref.append(gap); t+=len(a)/SR+0.35
ref=np.concatenate(ref)
y,_=librosa.load('brian.wav',sr=SR)
hop=240
def feat(x):
    m=librosa.feature.mfcc(y=x,sr=SR,n_mfcc=20,hop_length=hop,n_fft=1024)[1:]
    return (m-m.mean(1,keepdims=True))/(m.std(1,keepdims=True)+1e-6)
A=feat(ref); B=feat(y)
D,wp=librosa.sequence.dtw(X=A,Y=B,metric='cosine',subseq=False)
wp=wp[::-1]
fa=wp[:,0]*hop/SR; fb=wp[:,1]*hop/SR
def map_t(tr): i=np.searchsorted(fa,tr); i=min(i,len(fb)-1); return float(fb[i])
out=[]
for s,j,a,b in bounds: out.append({'s':s,'j':j,'a':round(map_t(a),3),'b':round(map_t(b),3)})
json.dump(out,open('align.json','w'),indent=0)
for o in out: print(o['s'],o['j'],o['a'],o['b'],P[o['s']][o['j']][:40])
