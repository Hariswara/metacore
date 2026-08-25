"""End-to-end Module 2 prototype (the Week-7 demo), on synthetic data only.
Trains EDL -> shows u low on normal / high on cyclone -> shows u also rising as
M1's data quality falls -> reports OOD metrics -> calibrates the two-condition
competence-drop trigger -> emits sample M2->M3 messages.

Competence is lost in two independent ways, and the gate fires on either:
  * value    -- the state is far from anything seen in training (cyclone)
  * sensing  -- a modality is missing, so we cannot see the state (comms blackout)
The trigger tests them separately and reports which one fired.
"""

import json
import random

import numpy as np
import torch
import yaml
from contract import build_output
from edl import EDLNet, edl_mse_loss, kl_to_uniform, uncertainty, uncertainty_quality
from evaluate import (
    aupr,
    auroc,
    ece,
    fpr95,
    reliability_table,
    retained_composition,
    risk_coverage,
)
from state_contract import stack_features
from synthetic_data import (
    D,
    Normalizer,
    sample_states_blackout,
    sample_states_id,
    sample_states_ood,
)
from trigger import CompetenceDropTrigger

with open("config.yaml") as fh:
    cfg = yaml.safe_load(fh)

random.seed(cfg["seed"])
np.random.seed(cfg["seed"])
torch.manual_seed(cfg["seed"])
torch.use_deterministic_algorithms(True)
K = cfg["k_classes"]
tr = cfg["train"]
tg = cfg["trigger"]
rng = np.random.default_rng(cfg["seed"])
torch.manual_seed(cfg["seed"])

# States, not bare arrays: everything below consumes M1's StateRepresentation.
# Training draws the realistic mixed-quality population; the evaluation set is held at
# nominal quality so the value axis is measured without the sensing axis on top of it.
tr_states, ytr = sample_states_id(3000, rng)
te_states, yte = sample_states_id(1000, rng, blackout_rate=0.0)
ood_states, _ = sample_states_ood(800, rng)
bo_states, _ = sample_states_blackout(800, rng)

Xtr = stack_features(tr_states)
Xte = stack_features(te_states)
Xood = stack_features(ood_states)
Xbo = stack_features(bo_states)
nz = Normalizer().fit(Xtr)
Xtr_,Xte_,Xood_,Xbo_ = nz(Xtr), nz(Xte), nz(Xood), nz(Xbo)


def observed(states):
    return np.array([s.quality.observed_fraction for s in states], dtype=np.float32)


of_te, of_ood, of_bo = observed(te_states), observed(ood_states), observed(bo_states)
of_train = observed(tr_states)

m = EDLNet(D,K)
opt = torch.optim.Adam(m.parameters(), tr["lr"], weight_decay=tr["weight_decay"])
Xt = torch.tensor(Xtr_)
yt = torch.tensor(ytr)
N = len(Xt)
for ep in range(tr["epochs"]):
    perm = torch.randperm(N)
    for i in range(0,N,tr["batch_size"]):
        idx = perm[i:i+tr["batch_size"]]
        xb = Xt[idx]
        xo = xb + torch.randn_like(xb)*tr["ood_proxy_sigma"]          # far-OOD proxy
        loss = edl_mse_loss(m(xb),yt[idx],ep,tr["kl_anneal_epochs"]) \
               + tr["ood_reg_weight"]*kl_to_uniform(m(xo)+1.0).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

with torch.no_grad():
    ev_id, ev_ood, ev_bo = m(torch.tensor(Xte_)), m(torch.tensor(Xood_)), m(torch.tensor(Xbo_))
    vu_id,p_id,e_id = uncertainty(ev_id)        # value axis: quality-independent
    vu_ood,p_ood,e_ood = uncertainty(ev_ood)
    vu_bo,p_bo,e_bo = uncertainty(ev_bo)
    u_id = uncertainty_quality(ev_id, torch.tensor(of_te))     # magnitude we emit
    u_ood = uncertainty_quality(ev_ood, torch.tensor(of_ood))
    u_bo = uncertainty_quality(ev_bo, torch.tensor(of_bo))
vu_id,vu_ood,vu_bo = vu_id.numpy(),vu_ood.numpy(),vu_bo.numpy()
u_id,u_ood,u_bo = u_id.numpy(),u_ood.numpy(),u_bo.numpy()
p_id,e_id = p_id.numpy(),e_id.numpy()
acc = (p_id.argmax(1)==yte).mean()

print("="*60)
print("MODULE 2 - AUQ ENGINE - PROTOTYPE RESULTS (synthetic)")
print("="*60)
print(f"ID 3-class accuracy         : {acc:.3f}")
print(f"value-only u   ID / OOD     : {vu_id.mean():.3f} / {vu_ood.mean():.3f}")
print(f"emitted    u   ID / OOD     : {u_id.mean():.3f} / {u_ood.mean():.3f}")
print(f"emitted    u   blackout     : {u_bo.mean():.3f}   (value-only would read "
      f"{vu_bo.mean():.3f})")
print(f"AUROC (u, OOD vs ID)        : {auroc(u_ood,u_id):.3f}   (target >= 0.90)")
print(f"AUPR  (OOD)                 : {aupr(u_ood,u_id):.3f}")
print(f"FPR95                       : {fpr95(u_ood,u_id):.3f}   (lower better)")
print(f"ECE   (ID calibration)      : {ece(p_id,yte):.3f}   (lower better)")
print(f"mixed-quality training set  : {(of_train < tg['observed_fraction_floor']).mean():.3f} "
      "of states arrive with a modality missing")

# --- the two-condition competence-drop trigger ---
trig = CompetenceDropTrigger.calibrate(vu_id, tg["value_false_alarm_rate"],
                                       tg["observed_fraction_floor"], tg["hysteresis"])


def fire(value_u, of):
    """Memoryless per-state evaluation: a fresh trigger with no hysteresis, so this
    reports the condition rate rather than the debounced time series."""
    outcomes = [CompetenceDropTrigger(trig.vthr, trig.of_floor, 1).update(float(v), float(o))
                for v, o in zip(value_u, of, strict=True)]
    return (np.array([f for f, _ in outcomes]),
            [r for _, r in outcomes])


print(f"\nvalue threshold             : {trig.vthr:.3f}   "
      f"(observed_fraction floor {trig.of_floor})")
print(f"{'population':<26}  {'fires':>6}   reason breakdown")
for name, vu, of in (("normal ID (of=0.50)", vu_id, of_te),
                     ("cyclone (value-OOD)", vu_ood, of_ood),
                     ("blackout (of<0.40)", vu_bo, of_bo)):
    fired, reasons = fire(vu, of)
    counts = {r: reasons.count(r)/len(reasons) for r in sorted(set(reasons))}
    breakdown = "  ".join(f"{r} {c:.2f}" for r, c in counts.items())
    print(f"{name:<26}  {fired.mean():>6.3f}   {breakdown}")

# --- selective prediction: does u predict error? ---
correct = (p_id.argmax(1) == yte)
covs, risks, aurc = risk_coverage(vu_id, correct)
print("\nSelective prediction on in-distribution states (reject highest u first):")
print(f"  full-coverage accuracy    : {correct.mean():.3f}")
for target in (0.25, 0.50, 0.75):
    i = int(np.argmin(np.abs(covs - target)))
    print(f"  accuracy @ coverage {covs[i]:.2f}  : {1 - risks[i]:.3f}")
print(f"  AURC                      : {aurc:.4f}   (lower better; "
      f"{risk_coverage(np.random.default_rng(0).permutation(vu_id), correct)[2]:.4f} "
      "for a shuffled ranking)")

# --- what a single u ranking does, and does not, reject ---
mixed_states = te_states + ood_states + bo_states
groups = np.array(["normal"]*len(te_states) + ["cyclone"]*len(ood_states)
                  + ["blackout"]*len(bo_states))
u_mixed = np.concatenate([u_id, u_ood, u_bo])
vu_mixed = np.concatenate([vu_id, vu_ood, vu_bo])
of_mixed = np.concatenate([of_te, of_ood, of_bo])
composition = retained_composition(u_mixed, groups)
half = min(composition, key=lambda r: abs(r["coverage"] - 0.50))
base = {g: float((groups == g).mean()) for g in ("normal", "cyclone", "blackout")}
print("\nMixed stream, ranked by the combined u (keep the most confident half):")
print(f"  full stream    : normal {base['normal']:.3f}  cyclone {base['cyclone']:.3f}  "
      f"blackout {base['blackout']:.3f}")
print(f"  kept half      : normal {half['normal']:.3f}  cyclone {half['cyclone']:.3f}  "
      f"blackout {half['blackout']:.3f}")
print("  -> the magnitude alone clears the value axis, and barely touches the sensing")
print("     axis. That is what the observed_fraction floor is for:")
fired_mixed = np.array(
    [CompetenceDropTrigger(trig.vthr, trig.of_floor, 1).update(float(v), float(o))[0]
     for v, o in zip(vu_mixed, of_mixed, strict=True)])
for g in ("normal", "cyclone", "blackout"):
    print(f"     trigger fires on {g:<9}: {fired_mixed[groups == g].mean():.3f}")

# --- emit the evaluation tables as data; plots.py renders them (viz extra) ---
reliability = reliability_table(p_id, yte)
with open("eval_tables.json", "w") as f:
    json.dump({
        "reliability": reliability,
        "ece": float(ece(p_id, yte)),
        "risk_coverage": {"coverage": covs.tolist(), "risk": risks.tolist(), "aurc": aurc},
        "retained_composition": composition,
    }, f, indent=2)
print(f"\n(eval_tables.json written: {len(reliability)} reliability bins, "
      f"{len(covs)} coverage points -- render with `python plots.py`)")

# --- emit a few M2->M3 messages (the mock stream Saabir consumes now) ---
print("\nSample M2 -> M3 contract messages (normal + cyclone + blackout):")
sample = (list(zip(u_id[:2], vu_id[:2], p_id[:2], of_te[:2], strict=True))
          + list(zip(u_ood[:1], vu_ood[:1], p_ood.numpy()[:1], of_ood[:1], strict=True))
          + list(zip(u_bo[:1], vu_bo[:1], p_bo.numpy()[:1], of_bo[:1], strict=True)))
emitter = CompetenceDropTrigger(trig.vthr, trig.of_floor, 1)
with open("sample_m2_to_m3.jsonl","w") as f:
    for u, vu, p, of in sample:
        fired, reason = emitter.update(float(vu), float(of))
        msg = build_output(u, p, 0.0, fired, of, reason)
        f.write(msg.to_json()+"\n")
        print("  "+msg.to_json())
print("\n(sample_m2_to_m3.jsonl written)")
