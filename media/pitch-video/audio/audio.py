import json, numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt
SR=48000
tl=json.load(open('../video/src/timeline.json')); FPS=30
T=tl['total']/FPS; N=int(T*SR)+SR
sc={s['id']:s for s in tl['scenes']}
def t0(s): return sc[s]['from']/FPS
def ph(s,i): return t0(s)+sc[s]['phrases'][i]['f']/FPS
def phe(s,i): p=sc[s]['phrases'][i]; return t0(s)+(p['f']+p['d'])/FPS
rng=np.random.default_rng(7)
music=np.zeros((N,2)); sfx=np.zeros((N,2))
def add(buf,t,sig,gain=1.0,pan=0.0):
    i=int(t*SR); sig=np.asarray(sig)
    if sig.ndim==1: sig=np.stack([sig*(1-max(0,pan)),sig*(1+min(0,pan))],1)
    j=min(N,i+len(sig)); 
    if i<0 or i>=N: return
    buf[i:j]+=sig[:j-i]*gain
def lp(x,fc,order=2): return sosfilt(butter(order,fc,'low',fs=SR,output='sos'),x,axis=0)
def hp(x,fc,order=2): return sosfilt(butter(order,fc,'high',fs=SR,output='sos'),x,axis=0)
def bp(x,lo,hi): return sosfilt(butter(2,[lo,hi],'band',fs=SR,output='sos'),x,axis=0)
def tt(d): return np.arange(int(d*SR))/SR
def env(d,a=0.005,r=None):
    t=tt(d); e=np.minimum(1,t/a)
    if r: e*=np.exp(-t/r)
    return e
def midi(n): return 440*2**((n-69)/12)
def saw(f,d,det=0.0):
    t=tt(d); out=0
    for k in range(1,14):
        if f*k>SR/2.2: break
        out=out+np.sin(2*np.pi*f*k*t+det*k)/k
    return out
# ---------- music ----------
# intensity curve (0..1) over absolute time
pts=[(0,0),(t0(2),0),(t0(2)+1,0.35),(t0(4),0.5),(t0(5),0.5),(t0(5)+0.4,0.15),(t0(6),0.25),(t0(6)+5,0.55),(ph(6,0)+(74+80)/30,0.0),(ph(6,1),0.15),(t0(7)+0.9,0.15),(t0(7)+0.95,1.0),(t0(10),0.9),(t0(10)+0.7,0.3),(ph(10,3),0.85),(t0(11),0.9),(t0(12)+3,0.9),(T-0.5,0.0),(T,0)]
xs,ys=zip(*pts); tg=np.arange(N)/SR; I=np.interp(tg,xs,ys)
I=lp(I,3)  # smooth
BPM=96; beat=60/BPM; bar=4*beat
prog=[(50,[50,53,57,62]),(46,[46,50,53,58]),(53,[53,57,60,65]),(48,[48,52,55,60])]  # Dm Bb F C
pad=np.zeros(N); arp=np.zeros(N); bass=np.zeros(N); kick=np.zeros(N)
nb=int(T/bar)+2
for b in range(nb):
    root,ch=prog[(b//2)%4]; st=b*bar; i=int(st*SR)
    if i>=N: break
    seg=sum(saw(midi(n),bar,det=rng.random()*6)+saw(midi(n)*1.004,bar,det=rng.random()*6) for n in ch)
    e=np.minimum(1,tt(bar)/0.3)*np.minimum(1,(bar-tt(bar))/0.3)
    j=min(N,i+len(seg)); pad[i:j]+=(seg*e)[:j-i]*0.05
    bs=saw(midi(root-12),bar)*e; bass[i:j]+=bs[:j-i]*0.18
    for k in range(16):
        n=ch[[0,1,2,3,2,1,3,2][k%8]]+12
        ts=st+k*beat/4; ii=int(ts*SR)
        d=0.22; s=np.sin(2*np.pi*midi(n)*tt(d))+0.4*np.sin(2*np.pi*midi(n)*2*tt(d)); s*=env(d,0.002,0.06)
        jj=min(N,ii+len(s)); 
        if ii<N: arp[ii:jj]+=s[:jj-ii]*0.12
    for k in range(4):
        ts=st+k*beat; ii=int(ts*SR); d=0.35; tk=tt(d)
        fr=40+80*np.exp(-tk/0.04); ph_=2*np.pi*np.cumsum(fr)/SR; s=np.sin(ph_)*np.exp(-tk/0.12)
        jj=min(N,ii+len(s))
        if ii<N: kick[ii:jj]+=s[:jj-ii]*0.5
pad=lp(pad,1800); bass=lp(bass,400)
padI=np.clip(I*1.4,0,1); arpI=np.clip((I-0.3)/0.6,0,1); kickI=np.clip((I-0.55)/0.4,0,1)
mono=pad*padI+bass*np.clip(I*1.2,0,1)+arp*arpI+kick*kickI
music[:,0]=mono+np.roll(arp*arpI,int(0.012*SR))*0.3; music[:,1]=mono+np.roll(arp*arpI,int(0.019*SR))*0.3
# ---------- sfx ----------
def beep(f=1000,d=0.12,g=0.25): return np.sin(2*np.pi*f*tt(d))*env(d,0.003)*np.minimum(1,(d-tt(d))/0.02)*g
def hit(d=2.5): # cinematic boom
    tk=tt(d); fr=30+70*np.exp(-tk/0.08); s=np.sin(2*np.pi*np.cumsum(fr)/SR)*np.exp(-tk/0.9)
    n=lp(rng.standard_normal(len(tk)),300)*np.exp(-tk/0.3)*0.5
    return (s+n)*0.9
def whoosh(d=1.0,rev=False):
    n=rng.standard_normal(int(d*SR)); t=tt(d)/d; e=(t**2 if not rev else (1-t)**2)
    out=np.zeros_like(n); 
    for k in range(8):
        a,b=int(k*len(n)/8),int((k+1)*len(n)/8); fc=300+4000*((k+0.5)/8 if not rev else 1-(k+0.5)/8)
        out[a:b]=bp(n,fc*0.6,fc*1.4)[a:b]
    return out*e*0.5
def click(): d=0.03; return hp(rng.standard_normal(int(d*SR)),2000)*env(d,0.001,0.004)*0.25
def clunk(): d=0.4; tk=tt(d); return (np.sin(2*np.pi*90*tk)*np.exp(-tk/0.06)+lp(rng.standard_normal(len(tk)),1500)*np.exp(-tk/0.02))*0.6
def stamp(): d=0.6; tk=tt(d); return (np.sin(2*np.pi*(60+120*np.exp(-tk/0.02))*tk)*np.exp(-tk/0.12)+lp(rng.standard_normal(len(tk)),2500)*np.exp(-tk/0.03)*0.6)*0.8
def chime(base=76): 
    d=1.6; out=0
    for k,n in enumerate([base,base+4,base+7]):
        s=np.zeros(int(d*SR)); o=int(k*0.07*SR); tk=tt(d-k*0.07)
        s[o:o+len(tk)]=np.sin(2*np.pi*midi(n)*tk)*np.exp(-tk/0.5); out=out+s
    return out*0.18
def static(d): n=rng.standard_normal(int(d*SR)); return bp(n,800,5000)*(0.5+0.5*(rng.random(len(n))>0.7))*0.15
def zip_(d=1.2):
    t=tt(d); f=200+3000*(t/d)**1.5; s=np.sign(np.sin(2*np.pi*np.cumsum(f)/SR))*0.08+bp(rng.standard_normal(len(t)),1000,6000)*0.15
    return s*np.sin(np.pi*t/d)
def powerdown(d=1.5): t=tt(d); f=400*np.exp(-t/0.5)+30; return np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t/0.8)*0.3
# S1 ECG beeps + hum + flatline
for fr in [5.2,18,30.7,43.4]: add(sfx,t0(1)+fr/30,beep(1000,0.1,0.22))
hum=tt(1.8); h=(np.sin(2*np.pi*50*hum)+0.5*np.sin(2*np.pi*100*hum)+0.25*np.sin(2*np.pi*150*hum))*0.12*np.minimum(1,hum/0.3)
add(sfx,t0(1)+52/30,h)
add(sfx,t0(1)+96/30,beep(1000,0.5,0.2))
# S2 wind + boom
wd=sc[2]['dur']/30; w=lp(rng.standard_normal(int(wd*SR)),700)*np.sin(np.pi*tt(wd)/wd)**1.5*0.5
add(sfx,t0(2),np.stack([w,np.roll(w,800)],1)); add(sfx,t0(2)+0.1,hit(3.0),0.6)
add(sfx,t0(3)-0.9,whoosh(1.0),0.8)
# S3 island tones + snap
for i,n in enumerate([69,72,74,76]): add(sfx,ph(3,i+1)-0.1,np.sin(2*np.pi*midi(n)*tt(0.9))*env(0.9,0.005,0.35)*0.2,pan=(-0.3+i*0.2))
add(sfx,ph(3,5)+0.6,clunk(),0.6)
# S4 counter ticks
for s_,e_ in [(ph(4,0),ph(4,0)+1.3),(ph(4,1),phe(4,1)-0.3),(ph(4,2),ph(4,2)+1.3)]:
    for tk in np.arange(s_,e_,0.06): add(sfx,tk,click(),0.8)
add(sfx,ph(4,3),stamp(),0.7)
# S5 static + glitch
add(sfx,ph(5,1)-0.05,static(0.9),1.0)
# S6 typing, relays, powerdown, rewind
p0=ph(6,0)
for a in [4,22,40]:
    for k in range(14): add(sfx,p0+(a+k*0.9)/30,click(),0.7)
ex=p0+74/30
add(sfx,ex,stamp(),0.5)
for k in range(6): add(sfx,ex+0.7+k*0.33,clunk(),0.7,pan=(-0.5+k*0.2))
add(sfx,ex+70/30,powerdown(1.6),1.0)
add(sfx,ex+80/30,hit(2.0),0.5)
add(sfx,phe(6,0)+0.05,zip_(1.2),1.0)
# S7 riser + hit
add(sfx,ph(7,0)-1.0,whoosh(1.0),1.0); add(sfx,ph(7,0)+0.05,hit(3.0),1.0)
# S8 whooshes on camera moves
for x in [ph(8,1)-0.45,ph(8,2)-0.4]: add(sfx,x,whoosh(0.7),0.7)
# S9 lock + checks + stamps
add(sfx,t0(9)+2.0,clunk(),0.8); add(sfx,t0(9)+2.2,stamp(),0.4)
for i in range(1,5): add(sfx,ph(9,i),beep(1320 if i!=3 else 330,0.12,0.2))
for i in [6,7,8]: add(sfx,ph(9,i),stamp(),0.9)
add(sfx,ph(9,9)+1.3,whoosh(1.2,True),0.6)
# S10 rewind, stamps, chime, heartbeat
add(sfx,t0(10),zip_(0.7),1.0)
add(sfx,ph(10,2),stamp(),0.9); add(sfx,ph(10,2)+1.0,chime(),1.0)
for k in range(3): add(sfx,ph(10,3)+k*0.8,beep(1000,0.08,0.16))
# S11 card whooshes
for i in range(3): add(sfx,ph(11,i)-0.2,whoosh(0.5),0.5)
# S12 final
add(sfx,t0(12)+0.2,beep(1000,0.08,0.16)); add(sfx,ph(12,0),hit(3.5),0.8); add(sfx,ph(12,2)+0.6,chime(74),0.8)
# ---------- mix ----------
vo,vsr=sf.read('vo_full.wav',dtype='float64')
vo=np.interp(np.arange(int(len(vo)*SR/vsr))/SR,np.arange(len(vo))/vsr,vo)
V=np.zeros(N); V[:len(vo)]=vo[:N]
# sidechain duck music under VO
venv=lp(np.abs(V),4); duck=1-0.65*np.clip(venv/0.04,0,1)
mix=music*duck[:,None]*0.24+sfx*0.6+np.stack([V,V],1)*1.0
mix=mix[:int(T*SR)]
r=lambda x: round(20*np.log10(np.sqrt((x**2).mean())+1e-9),1)
MM=(music*duck[:,None]*0.24)[:int(T*SR)]; SS=(sfx*0.6)[:int(T*SR)]
for a in range(0,118,6): print(a, "vo",r(V[a*SR:(a+6)*SR]),"mus",r(MM[a*SR:(a+6)*SR]),"sfx",r(SS[a*SR:(a+6)*SR]))
mix/=np.max(np.abs(mix))*1.05
sf.write('mix.wav',mix,SR); sf.write('music_only.wav',music[:int(T*SR)]/np.max(np.abs(music))*0.8,SR)
print('ok',T)
