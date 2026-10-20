"""Retrain the EDL head on M1's real state distribution, and validate it there.

    python train_real.py                 # full run -> real_m1_metrics.json
    python train_real.py --quick         # smaller, for a smoke test

WHY THIS EXISTS
---------------
The head trained on `synthetic_data` reads a real M1 row as fully out of distribution:
u = u' = 1.0 with evidence = K. That was never a bug in the head. The normalisation card
was fitted on synthetic draws where `voltage_pu` ~ 1.0 +/- 0.021, so a real offline row's
0.0 lands at z = -48 sigma on that column alone, ~22 sigma RMS across the five electrical
features, against an evidence regulariser trained to drive evidence to zero at 4 sigma.
u = 1.0 was the correct answer to a row that far outside the fitted distribution.

Refitting the card on the real distribution is the root fix, and it is the first thing
this script does.

WHAT IS MEASURED, AND WHERE
---------------------------
Two evaluations, deliberately kept apart, because they answer different questions and
only one of them has a real label:

* **Classification on real hourly states** (accuracy, ECE, AURC). The label is
  discretised from M1's constructed vulnerability proxy -- see m1_real.PROXY_DISCLOSURE.
  These numbers state internal consistency with that proxy, not the prediction of real
  asset failures.
* **OOD detection** (AUROC, AUPR, FPR95) on the Eluvaitivu per-plant monthly ledger,
  n = 1 episode / 3 decay plant-months. Deliberately NOT on hourly data: the hourly
  aggregate attenuates the event 7.1x and its same-season control AUROC is 0.500. See
  m1_real.OOD_DISCLOSURE.

Both disclosures are written into real_m1_metrics.json alongside the numbers, so a table
lifted out of it carries its own caveat.

The synthetic path is untouched. run_demo.py, edl.onnx and the dashboard fallback all
still describe the mock demo, which stays the guaranteed fallback.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import m1_real
import numpy as np
import real_data as rd
import torch
import yaml
from baselines import (
    DROPOUT_P,
    score_edl,
    score_mc_dropout,
    score_softmax,
    train_softmax,
)
from edl import EDLNet, edl_mse_loss, kl_to_uniform, uncertainty
from evaluate import aupr, auroc, ece, fpr95, risk_coverage
from synthetic_data import Normalizer
from trigger import CompetenceDropTrigger

HERE = Path(__file__).parent
OUT = HERE / "real_m1_metrics.json"
# The trained head and the card fitted on the real distribution. Derived from seeded data
# and reproducible with `python train_real.py`, so the checkpoint is git-ignored; the ONNX
# model card that export_onnx.py --real writes from it is committed.
CACHE = HERE / ".real_cache"
CHECKPOINT = CACHE / "edl_real.pt"

# Every 4th hour, keeping all 10 nodes of the hours it keeps. Hourly grid states are
# strongly autocorrelated, so a stride loses little and keeps the diurnal cycle evenly
# covered -- which a random row sample would not guarantee. 131,280 train rows -> 32,820.
TRAIN_STRIDE = 4
N_NODES = 10

# The 8 features left when a comms blackout takes the temporal block with it: the
# topology constants alone. 8/28 = 0.286, under the 0.35 sensing floor in config.yaml.
BLACKOUT_OBSERVED = 8


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_normalizer(xtr: np.ndarray) -> Normalizer:
    """Refit the card on the real distribution -- the root fix for the saturation.

    One guard beyond a plain fit: `Normalizer` adds 1e-6 to every standard deviation, so
    a column that is constant offline (each of the five electrical features is exactly
    0.0) would get sd = 1e-6. That is harmless while the column stays 0, but the moment a
    real SCADA value arrives it would be scaled to ~1e6 sigma and detonate the input. A
    constant column carries no signal either way, so its scale is pinned to 1.
    """
    nz = Normalizer().fit(xtr)
    constant = xtr.std(0) < 1e-12
    nz.sd[constant] = 1.0
    return nz


def train_edl_real(x, y, epochs, cfg, class_weight=None, seed=0):
    """The EDL head, class-weighted, with the same OOD-aware regulariser as the demo."""
    _seed(seed)
    xt = torch.as_tensor(x)
    yt = torch.as_tensor(y)
    model = EDLNet(xt.shape[1], len(m1_real.CLASS_NAMES))
    opt = torch.optim.Adam(model.parameters(), cfg["lr"], weight_decay=cfg["weight_decay"])
    w = None if class_weight is None else torch.as_tensor(class_weight)
    batch = cfg["batch_size"]
    n = len(xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            xb = xt[idx]
            xo = xb + torch.randn_like(xb)*cfg["ood_proxy_sigma"]
            loss = (edl_mse_loss(model(xb), yt[idx], ep, cfg["kl_anneal_epochs"], w)
                    + cfg["ood_reg_weight"]*kl_to_uniform(model(xo)+1.0).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


# ------------------------------------------------------------------ evaluations


def classification_table(methods, xte, yte):
    """Accuracy / ECE / AURC on the real test split, for each uncertainty mechanism.

    No AUROC here: that needs an OOD label, and the hourly record does not carry one that
    survives the 7.1x aggregation. Selective prediction (AURC) is the question that *can*
    be asked of every method on this split -- does the score rank its own errors?
    """
    rows = []
    for name, score in methods:
        s, p = score(xte)
        correct = (p.argmax(1) == yte)
        _, _, aurc = risk_coverage(s, correct)
        rows.append({
            "method": name,
            "accuracy": round(float(correct.mean()), 4),
            "ece": round(float(ece(p, yte)), 4),
            "aurc": round(float(aurc), 4),
            "macro_recall": round(float(np.mean([
                (p.argmax(1)[yte == c] == c).mean() if (yte == c).any() else 0.0
                for c in range(len(m1_real.CLASS_NAMES))])), 4),
        })
    return rows


def ood_case_study(epochs, seed=0):
    """OOD detection at the one resolution where the real episode is visible.

    Per-plant monthly, n = 1 episode / 3 decay plant-months. Same split and features as
    eluvaitivu_decay.experiment_b, extended to score the two baselines alongside EDL so
    the comparison is the same comparison the synthetic table makes.

    The operating-class label here is constructed (the ledger records no safety class);
    it only gives each head something to be evidential about, and the OOD label never
    enters training.
    """
    from eluvaitivu_decay import _operating_class

    columns = rd.PLANT_FEATURES
    rows = rd.read_plant_months()
    nominal, decay = rd.split_plant_months(rows)
    baselines_ = rd.plant_baselines(nominal)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(nominal))
    cut = int(0.75*len(nominal))
    train_rows = [nominal[i] for i in order[:cut]]
    held_rows = [nominal[i] for i in order[cut:]]

    xtr_raw = rd.plant_month_matrix(train_rows, baselines_, columns)
    nz = fit_normalizer(xtr_raw)
    xtr = nz(xtr_raw)
    xheld = nz(rd.plant_month_matrix(held_rows, baselines_, columns))
    xdecay = nz(rd.plant_month_matrix(decay, baselines_, columns))

    weights = {"energy_rel": -0.5, "sfc_l_per_kwh": 0.3,
               "fuel_cost_rs_per_kwh": 0.2, "diesel_rel": 0.2}
    y = _operating_class(xtr_raw, list(columns), weights)

    cfg = {"lr": 2e-3, "weight_decay": 1e-5, "batch_size": 32, "kl_anneal_epochs": 50,
           "ood_reg_weight": 0.1, "ood_proxy_sigma": 4.0}
    edl_m = train_edl_real(xtr, y, epochs, cfg, seed=seed)
    soft_m = train_softmax(xtr, y, epochs, seed=seed)
    mc_m = train_softmax(xtr, y, epochs, dropout=DROPOUT_P, seed=seed)

    table = []
    for name, score in (("Softmax max-prob", lambda a: score_softmax(soft_m, a)),
                        ("MC-Dropout (T=20)", lambda a: score_mc_dropout(mc_m, a)),
                        ("EDL (ours)", lambda a: score_edl(edl_m, a))):
        s_held, _ = score(xheld)
        s_decay, _ = score(xdecay)
        table.append({
            "method": name,
            "auroc": round(float(auroc(s_decay, s_held)), 4),
            "aupr": round(float(aupr(s_decay, s_held)), 4),
            "fpr95": round(float(fpr95(s_decay, s_held)), 4),
            "mean_score_nominal_heldout": round(float(s_held.mean()), 4),
            "mean_score_decay": round(float(s_decay.mean()), 4),
        })
    return {
        "resolution": "per-plant monthly",
        "n_episodes": 1,
        "plant_months": {"nominal_train": len(train_rows),
                         "nominal_heldout": len(held_rows), "decay": len(decay)},
        "rows": table,
        "disclosure": m1_real.OOD_DISCLOSURE,
    }


def trigger_table(model, nz, states, cfg_trigger, train_u):
    """The two axes, on real states. Value threshold calibrated on real training u.

    The sensing axis is exercised by degrading the MASK, never the values: a blackout
    does not corrupt what was measured, it removes the claim that it was measured. Same
    feature rows, fewer of them declared observed -- so any movement in u' is the sensing
    axis and nothing else.
    """
    vthr = float(np.quantile(train_u, 1.0 - cfg_trigger["value_false_alarm_rate"]))
    floor = cfg_trigger["observed_fraction_floor"]

    def fire(u_values, of):
        outcomes = [CompetenceDropTrigger(vthr, floor, 1).update(float(v), float(of))
                    for v in u_values]
        return outcomes

    with torch.no_grad():
        u, _, _ = uncertainty(model(torch.as_tensor(nz(states.x))))
    u = u.numpy()

    nominal_of = states.observed_fraction
    blackout_of = BLACKOUT_OBSERVED/28

    # Rounded for the report, full precision for the artifact: a threshold rounded to 4dp
    # and one fitted at full precision disagree on the states that sit exactly on it.
    out = {"value_threshold": round(vthr, 4), "value_threshold_exact": vthr,
           "observed_fraction_floor": floor, "rows": []}
    for name, of in (("real states, nominal mask", nominal_of),
                     ("real states, temporal block lost", blackout_of)):
        outcomes = fire(u, of)
        fired = np.array([f for f, _ in outcomes])
        reasons = [r for _, r in outcomes]
        out["rows"].append({
            "population": name,
            "observed_fraction": round(float(of), 4),
            "fires": round(float(fired.mean()), 4),
            "reasons": {r: round(reasons.count(r)/len(reasons), 4) for r in sorted(set(reasons))},
        })
    return out


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrain the EDL head on real M1 states.")
    ap.add_argument("--epochs", type=int, default=None, help="default: config.yaml")
    ap.add_argument("--ood-epochs", type=int, default=400)
    ap.add_argument("--quick", action="store_true", help="smoke-test scale")
    args = ap.parse_args()

    with open(HERE / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    seed = int(cfg["seed"])
    tr_cfg = dict(cfg["train"])
    tr_cfg["batch_size"] = 256            # 10x the rows of the synthetic demo
    epochs = args.epochs if args.epochs is not None else (6 if args.quick else 120)
    ood_epochs = 40 if args.quick else args.ood_epochs

    _seed(seed)
    started = time.perf_counter()

    states = m1_real.load()
    train = states.where("train")
    test = states.where("test")

    # Stride over hours, keeping every node of the hours kept.
    keep = np.zeros(len(train.x), dtype=bool)
    keep[::TRAIN_STRIDE*N_NODES] = True
    for k in range(N_NODES):
        keep[k::TRAIN_STRIDE*N_NODES] = True
    xtr_raw, ytr = train.x[keep], train.y[keep]

    nz = fit_normalizer(xtr_raw)
    xtr, xte = nz(xtr_raw), nz(test.x)
    weights = m1_real.class_weights(ytr)

    print(f"real M1 states : {len(states.x)} node-hours "
          f"(train {len(train.x)} -> {len(xtr)} after stride {TRAIN_STRIDE}, "
          f"test {len(test.x)})")
    print(f"class cuts     : V = {states.cuts[0]:.4f} / {states.cuts[1]:.4f} "
          f"(train q{m1_real.CLASS_CUTS[0]:.2f}/q{m1_real.CLASS_CUTS[1]:.2f})")
    print(f"train balance  : { {k: round(v, 3) for k, v in train.class_balance.items()} }")
    print(f"test  balance  : { {k: round(v, 3) for k, v in test.class_balance.items()} }")
    shown = dict(zip(m1_real.CLASS_NAMES, weights.round(3).tolist(), strict=True))
    print(f"class weights  : {shown}")
    print(f"epochs         : {epochs}")
    print()

    edl_m = train_edl_real(xtr, ytr, epochs, tr_cfg, class_weight=weights, seed=seed)
    soft_m = train_softmax(xtr, ytr, epochs, seed=seed)
    mc_m = train_softmax(xtr, ytr, epochs, dropout=DROPOUT_P, seed=seed)

    methods = [
        ("Softmax max-prob", lambda a: score_softmax(soft_m, a)),
        ("MC-Dropout (T=20)", lambda a: score_mc_dropout(mc_m, a)),
        ("EDL (ours)", lambda a: score_edl(edl_m, a)),
    ]
    classification = classification_table(methods, xte, test.y)

    with torch.no_grad():
        u_train, _, _ = uncertainty(edl_m(torch.as_tensor(xtr)))
    triggers = trigger_table(edl_m, nz, test, cfg["trigger"], u_train.numpy())
    ood = ood_case_study(ood_epochs, seed=seed)

    elapsed = time.perf_counter() - started

    payload = {
        "artifact": "module2 EDL head retrained on M1 real states",
        "weighting": states.weighting,
        "class_cuts_quantile": list(m1_real.CLASS_CUTS),
        "class_cuts_v": list(states.cuts),
        "class_names": list(m1_real.CLASS_NAMES),
        "class_balance": {"train": train.class_balance, "test": test.class_balance},
        "class_weights": dict(zip(m1_real.CLASS_NAMES, weights.tolist(), strict=True)),
        "n_rows": {"all": len(states.x), "train_used": len(xtr), "test": len(test.x)},
        "epochs": epochs,
        "seed": seed,
        "observed_fraction": states.observed_fraction,
        "elapsed_s": round(elapsed, 1),
        "label_disclosure": m1_real.PROXY_DISCLOSURE,
        "classification_on_real_hourly_states": classification,
        "two_axis_trigger": triggers,
        "ood_case_study": ood,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    # The head plus the card fitted on the real distribution, so export_onnx.py --real
    # freezes exactly the model these numbers describe rather than retraining its own.
    CACHE.mkdir(exist_ok=True)
    torch.save({
        "state_dict": edl_m.state_dict(),
        "mu": nz.mu, "sd": nz.sd,
        # Calibrated here, not at serve time. Fitting it on demand would mean rebuilding
        # M1's whole 17,544-hour record and re-scoring 400 calibration rows on the first
        # request -- a cold start on the live toggle. It is a training-set property, so it
        # belongs in the artifact next to mu/sd.
        "value_threshold": triggers["value_threshold_exact"],
        "value_false_alarm_rate": cfg["trigger"]["value_false_alarm_rate"],
        "observed_fraction_floor": cfg["trigger"]["observed_fraction_floor"],
        "n_features": int(xtr.shape[1]),
        "classes": len(m1_real.CLASS_NAMES),
        "class_names": list(m1_real.CLASS_NAMES),
        "epochs": epochs, "seed": seed,
        "weighting": states.weighting,
        "class_cuts_quantile": list(m1_real.CLASS_CUTS),
        "class_cuts_v": list(states.cuts),
        "n_train": int(len(xtr)),
        "label_disclosure": m1_real.PROXY_DISCLOSURE,
    }, CHECKPOINT)

    _print_report(payload)
    print(f"({CHECKPOINT.relative_to(HERE)} written -- "
          "export with `python export_onnx.py --real`)")
    print(f"\n({OUT.name} written, {elapsed:.0f}s)")
    return 0


def _print_report(p: dict) -> None:
    print("CLASSIFICATION on real M1 test states (label = discretised V proxy):")
    head = f"  {'method':<22} {'acc':>7} {'macro-rec':>10} {'ECE':>7} {'AURC':>7}"
    print(head)
    print("  " + "-"*(len(head)-2))
    for r in p["classification_on_real_hourly_states"]:
        print(f"  {r['method']:<22} {r['accuracy']:>7.3f} {r['macro_recall']:>10.3f} "
              f"{r['ece']:>7.3f} {r['aurc']:>7.3f}")

    print("\nOOD CASE STUDY -- Eluvaitivu per-plant monthly, n=1 episode / 3 decay months:")
    head = f"  {'method':<22} {'AUROC':>7} {'AUPR':>7} {'FPR95':>7}"
    print(head)
    print("  " + "-"*(len(head)-2))
    for r in p["ood_case_study"]["rows"]:
        print(f"  {r['method']:<22} {r['auroc']:>7.3f} {r['aupr']:>7.3f} {r['fpr95']:>7.3f}")
    print("  (no OOD rate is claimed on hourly data -- 7.1x attenuated, control AUROC 0.500)")

    t = p["two_axis_trigger"]
    print(f"\nTWO-AXIS TRIGGER (value threshold {t['value_threshold']:.4f}, "
          f"sensing floor {t['observed_fraction_floor']}):")
    for r in t["rows"]:
        reasons = "  ".join(f"{k} {v:.2f}" for k, v in r["reasons"].items())
        print(f"  {r['population']:<32} f={r['observed_fraction']:.3f}  "
              f"fires {r['fires']:.3f}   {reasons}")

    print(f"\nLABEL: {p['label_disclosure']}")


if __name__ == "__main__":
    raise SystemExit(main())
