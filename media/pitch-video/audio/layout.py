import json, numpy as np, soundfile as sf
P=json.load(open('phrases.json')); FPS=30
L={ # scene: (pre, gaps, tail)
1:(0,[],4.0),
2:(0.7,[0.35],1.0),
3:(0.3,[0.4,0.35,0.35,0.35,0.5,0.3],0.5),
4:(0.2,[0.4,0.15,0.25],0.7),
5:(0.2,[0.5,0.4],0.6),
6:(0.2,[1.4],0.5),
7:(0.9,[],1.1),
8:(0.2,[0.4,0.3,0.25],0.5),
9:(0.2,[0.5,0.3,0.3,0.3,0.5,0.3,0.4,0.4,0.5],0.5),
10:(0.7,[0.35,0.25,0.25],0.9),
11:(0.2,[0.25,0.35],0.6),
12:(0.4,[0.4,0.3],1.8),
}
sr=24000; t=0.0; scenes=[]; mix=[]
for s in range(1,13):
    pre,gaps,tail=L[s]; start=t; tt=t+pre; ph=[]
    for j,p in enumerate(P.get(str(s),[])):
        ph.append({'t':round(tt-start,3),'d':p['d'],'text':p['t']}); mix.append((tt,p['f'])); tt+=p['d']+(gaps[j] if j<len(gaps) else 0)
    t=tt+tail; scenes.append({'id':s,'from':round(start*FPS),'dur':round((t-start)*FPS),'phrases':[{'f':round(x['t']*FPS),'d':round(x['d']*FPS),'text':x['text']} for x in ph]})
total=t; print('total',round(total,2))
buf=np.zeros(int((total+1)*sr),dtype=np.float32)
for st,f in mix:
    a,_=sf.read(f,dtype='float32'); i=int(st*sr); buf[i:i+len(a)]+=a
sf.write('vo_full.wav',buf[:int(total*sr)],sr)
# fix rounding: make frames contiguous
fr=0
for sc in scenes: sc['from']=fr; fr+=sc['dur']
json.dump({'fps':FPS,'total':fr,'scenes':scenes},open('../video/src/timeline.json','w'),indent=1)
print(fr,[ (s['id'],s['dur']) for s in scenes])
