"""End-to-end Module 2 prototype (the Week-7 demo), on synthetic data only.
Trains EDL -> shows u low on normal / high on cyclone -> shows u also rising as
M1's data quality falls -> reports OOD metrics -> calibrates the competence-drop
trigger -> emits sample M2->M3 messages.

Uncertainty here has two independent axes, and the gate fires on either:
  * value    -- the state is far from anything seen in training (cyclone)
  * quality  -- the state is mostly interpolated or missing (comms blackout)
"""

import random

import numpy as np
import torch
import yaml
from contract import build_output
from edl import EDLNet, edl_mse_loss, kl_to_uniform, uncertainty, uncertainty_quality
from evaluate import aupr, auroc, ece, fpr95
from state_contract import stack_features
from synthetic_data import D, Normalizer, sample_states_id, sample_states_ood
from trigger import CompetenceDropTrigger

with open("config.yaml") as fh:
    cfg = yaml.safe_load(fh)

random.seed(cfg["seed"])
np.random.seed(cfg["seed"])
torch.manual_seed(cfg["seed"])
torch.use_deterministic_algorithms(True)
K = cfg["k_classes"]
tr = cfg["train"]
rng = np.random.default_rng(cfg["seed"])
torch.manual_seed(cfg["seed"])

# States, not bare arrays: everything below consumes M1's StateRepresentation.
tr_states, ytr = sample_states_id(3000, rng)
te_states, yte = sample_states_id(1000, rng)
ood_states, _ = sample_states_ood(800, rng)

Xtr = stack_features(tr_states)
Xte = stack_features(te_states)
Xood = stack_features(ood_states)
nz = Normalizer().fit(Xtr)
Xtr_,Xte_,Xood_ = nz(Xtr), nz(Xte), nz(Xood)

# Per-row, because M1's real states will not all share one quality mask.
of_te = torch.tensor([s.quality.observed_fraction for s in te_states])
of_ood = torch.tensor([s.quality.observed_fraction for s in ood_states])

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
    ev_id = m(torch.tensor(Xte_))
    ev_ood = m(torch.tensor(Xood_))
    u_id_value,p_id,e_id = uncertainty(ev_id)      # value axis only (paper ablation)
    u_ood_value,p_ood,e_ood = uncertainty(ev_ood)
    u_id = uncertainty_quality(ev_id, of_te)       # what we emit
    u_ood = uncertainty_quality(ev_ood, of_ood)
u_id_value,u_ood_value = u_id_value.numpy(),u_ood_value.numpy()
u_id,u_ood = u_id.numpy(),u_ood.numpy()
p_id,e_id = p_id.numpy(),e_id.numpy()
acc = (p_id.argmax(1)==yte).mean()
of_state = te_states[0].quality.observed_fraction

print("="*56)
print("MODULE 2 - AUQ ENGINE - PROTOTYPE RESULTS (synthetic)")
print("="*56)
print(f"ID 3-class accuracy         : {acc:.3f}")
print(f"value-only u   ID / OOD     : {u_id_value.mean():.3f} / {u_ood_value.mean():.3f}")
print(f"emitted    u   ID / OOD     : {u_id.mean():.3f} / {u_ood.mean():.3f}"
      f"   (at observed_fraction {of_state:.2f})")
print(f"AUROC (u, OOD vs ID)        : {auroc(u_ood,u_id):.3f}   (target >= 0.90)")
print(f"AUPR  (OOD)                 : {aupr(u_ood,u_id):.3f}")
print(f"FPR95                       : {fpr95(u_ood,u_id):.3f}   (lower better)")
print(f"ECE   (ID calibration)      : {ece(p_id,yte):.3f}   (lower better)")

# --- the second axis: same in-distribution states, falling data quality ---
print("\nQuality axis (in-distribution states, evidence discounted by observed_fraction):")
for of in (1.00, 0.75, 0.50, 0.25, 0.10):
    with torch.no_grad():
        u_sweep = uncertainty_quality(ev_id, of).mean().item()
        u_sweep_ood = uncertainty_quality(ev_ood, of).mean().item()
    print(f"  observed_fraction {of:.2f} -> u {u_sweep:.3f}   (cyclone at the same quality: "
          f"{u_sweep_ood:.3f})")

trig = CompetenceDropTrigger.calibrate(u_id, cfg["trigger"]["false_alarm_rate"],
                                       cfg["trigger"]["hysteresis"])
print(f"\ncompetence-drop threshold   : {trig.thr:.3f}")
print(f"trigger fires  ID / OOD     : {(u_id>trig.thr).mean():.3f} / {(u_ood>trig.thr).mean():.3f}")
u_blackout = uncertainty_quality(ev_id, 0.10).numpy()
print(f"trigger fires  ID @ of=0.10 : {(u_blackout>trig.thr).mean():.3f}"
      "   (same states, degraded sensing)")

# --- emit a few M2->M3 messages (the mock stream Saabir consumes now) ---
print("\nSample M2 -> M3 contract messages (mixed normal + cyclone):")
stream = (list(zip(u_id[:2], p_id[:2], of_te[:2].tolist(), strict=True))
          + list(zip(u_ood[:2], p_ood.numpy()[:2], of_ood[:2].tolist(), strict=True)))
t2 = CompetenceDropTrigger(trig.thr, 1)
with open("sample_m2_to_m3.jsonl","w") as f:
    for u,p,of in stream:
        msg = build_output(u,p,0.0,t2.update(u),of)
        f.write(msg.to_json()+"\n")
        print("  "+msg.to_json())
print("\n(sample_m2_to_m3.jsonl written)")
