import soundfile as sf, json, numpy as np
from kokoro_onnx import Kokoro
k=Kokoro("models/kokoro-v1.0.onnx","models/voices-v1.0.bin")
P={
2:["Late November, twenty twenty-five.","Cyclone Ditwah crosses Sri Lanka."],
3:["Off the Jaffna coast, four islands make their own power.","Delft.","Nainativu.","Analaitivu.","Eluvaitivu.","No cable to the mainland.","Only diesel, shipped in by sea."],
4:["Around nine thousand people live here, with a hospital, and a water plant.","In twenty twenty-five, these grids burned nine hundred and seventy-five thousand litres of diesel,","for two point nine five gigawatt hours,","at ninety-four to one hundred and eighteen rupees a unit."],
5:["In a storm like this, operators are evacuated.","Telemetry fails.","The grid has to decide on its own."],
6:["Hand that decision to an ordinary AI, and one confident mistake can cascade into a blackout.","So we asked: what if the AI knew, when it didn't know?"],
7:["This is MetaCore."],
8:["It sees the grid as a living graph of weather, load, and generation.","It measures its own uncertainty.","When the answer is clear, it acts fast.","When it isn't, it slows down, and reasons."],
9:["Then, before any action touches the grid, a deterministic physics firewall runs a full power flow on a digital twin.","Voltage.","Line loading.","Generation.","Convergence.","Every action gets a verdict.","Approve.","Reject.","Or unverified, with its risk attached.","Nothing executes without approval, and every rejection teaches the agent why."],
10:["Same storm. Same proposal.","This time, the doubt is caught,","the unsafe action is blocked,","and the lights stay on."],
11:["For the Ceylon Electricity Board.","For the island communities.","For every remote and coastal microgrid that cannot afford a guess."],
12:["MetaCore.","It knows when it doesn't know.","And it verifies, before it acts."],
}
out={}
for s,ps in P.items():
    out[s]=[]
    for j,t in enumerate(ps):
        a,sr=k.create(t,voice="af_heart",speed=1.08,lang="en-us")
        # trim silence
        idx=np.where(np.abs(a)>0.01)[0]; a=a[max(0,idx[0]-240):idx[-1]+2400]
        fn=f"p_{s}_{j}.wav"; sf.write(fn,a,sr); out[s].append({"f":fn,"t":t,"d":round(len(a)/sr,3)})
json.dump(out,open("phrases.json","w"),indent=1)
for s in out: print(s,[x["d"] for x in out[s]])
