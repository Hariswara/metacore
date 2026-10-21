import json, numpy as np, soundfile as sf, sys
sys.path.insert(0,'.'); from phrases_new import P
TEMPO=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
MAXP=float(sys.argv[2]) if len(sys.argv)>2 else 0.3
GS=float(sys.argv[3]) if len(sys.argv)>3 else 1.0
src='brian_t.wav'
y,sr=sf.read(src)
al=json.load(open('align.json'))
FPS=30
# cut each phrase, snapping to low-energy points, timing scaled by tempo
def cut(a,b):
    a/=TEMPO; b/=TEMPO
    i=max(0,int((a-0.04)*sr)); j=min(len(y),int((b+0.08)*sr))
    seg=y[i:j].copy()
    # shrink internal pauses longer than MAXP to MAXP
    win=int(0.02*sr); n=len(seg)//win
    rms=np.array([np.sqrt((seg[k*win:(k+1)*win]**2).mean()) for k in range(n)])
    quiet=rms<10**(-42/20); keep=np.ones(len(seg),bool); k=0
    while k<n:
        if quiet[k]:
            e=k
            while e<n and quiet[e]: e+=1
            run=e-k
            if run*win>MAXP*sr and k>0 and e<n:
                drop=run*win-int(MAXP*sr); a0=k*win+(run*win-drop)//2; keep[a0:a0+drop]=False
            k=e
        else: k+=1
    seg=seg[keep]; f=int(0.012*sr); seg[:f]*=np.linspace(0,1,f); seg[-f:]*=np.linspace(1,0,f); return seg
clips={(o['s'],o['j']):cut(o['a'],o['b']) for o in al}
L={1:(0,[],3.2),2:(0.6,[0.35],0.9),3:(0.3,[0.35,0.3,0.3,0.3,0.4,0.3],0.4),4:(0.2,[0.35,0.15,0.2],0.6),5:(0.2,[0.4,0.35],0.5),
6:(0.2,[1.3],0.4),7:(0.8,[],1.0),8:(0.2,[0.35,0.25,0.25],0.4),9:(0.2,[0.45,0.25,0.25,0.25,0.4,0.25,0.3,0.35,0.45],0.4),
10:(0.6,[0.3,0.25,0.3],0.8),11:(0.2,[0.25,0.3],0.5),12:(0.4,[0.35,0.3],1.6)}
t=0.0; scenes=[]; mix=[]
for s in range(1,13):
    pre,gaps,tail=L[s]
    if s!=1: pre,gaps,tail=pre*GS,[g*GS if g<1 else g*0.85 for g in gaps],tail*GS
    start=t; tt=t+pre; ph=[]
    for j,txt in enumerate(P.get(s,[])):
        c=clips[(s,j)]; d=len(c)/sr
        ph.append({'t':tt-start,'d':d,'text':txt}); mix.append((tt,c)); tt+=d+(gaps[j] if j<len(gaps) else 0)
    t=tt+tail; scenes.append({'id':s,'start':start,'end':t,'phrases':ph})
buf=np.zeros(int((t+1)*sr))
for st,c in mix: i=int(st*sr); buf[i:i+len(c)]+=c
sf.write('../tts/vo_full.wav',buf[:int(t*sr)],sr)
# integer frames, contiguous
out=[]; fr=0
for sc in scenes:
    dur=round(sc['end']*FPS)-round(sc['start']*FPS)
    out.append({'id':sc['id'],'from':fr,'dur':dur,'phrases':[{'f':round(p['t']*FPS),'d':round(p['d']*FPS),'text':p['text']} for p in sc['phrases']]}); fr+=dur
json.dump({'fps':FPS,'total':fr,'scenes':out},open('../video/src/timeline.json','w'),indent=1)
print('total s',round(t,2),'frames',fr)
