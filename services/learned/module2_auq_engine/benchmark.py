"""Full-scale comparison of the four methods -> comparison_table.json.

A script, not a test: it trains four models at full scale and is meant to be run by hand
when the numbers need refreshing. The test lane asserts the *ordering* at small scale
instead, so CI never pays for this.

    python benchmark.py
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from baselines import (
    DROPOUT_P,
    MC_PASSES,
    score_edl,
    score_mc_dropout,
    score_softmax,
    train_edl,
    train_softmax,
)
from evaluate import aupr, auroc, ece, fpr95, risk_coverage
from state_contract import stack_features
from synthetic_data import Normalizer, sample_states_id, sample_states_ood

OUT = Path(__file__).with_name("comparison_table.json")


def main():
    with open(Path(__file__).with_name("config.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    seed, epochs = cfg["seed"], cfg["train"]["epochs"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    tr_states, ytr = sample_states_id(3000, rng, blackout_rate=0.0)
    te_states, yte = sample_states_id(1000, rng, blackout_rate=0.0)
    ood_states, _ = sample_states_ood(800, rng)
    nz = Normalizer().fit(stack_features(tr_states))
    xtr, xte, xood = (nz(stack_features(s)) for s in (tr_states, te_states, ood_states))

    softmax_m = train_softmax(xtr, ytr, epochs, seed=seed)
    mc_m = train_softmax(xtr, ytr, epochs, dropout=DROPOUT_P, seed=seed)
    edl_m = train_edl(xtr, ytr, epochs, ood_reg=cfg["train"]["ood_reg_weight"], seed=seed)
    noreg_m = train_edl(xtr, ytr, epochs, ood_reg=0.0, seed=seed)

    methods = [
        ("Softmax max-prob", lambda x: score_softmax(softmax_m, x)),
        (f"MC-Dropout (T={MC_PASSES})", lambda x: score_mc_dropout(mc_m, x)),
        ("EDL (ours)", lambda x: score_edl(edl_m, x)),
        ("EDL, no OOD-reg (ablation)", lambda x: score_edl(noreg_m, x)),
    ]

    rows = []
    for name, score in methods:
        s_id, p_id = score(xte)
        s_ood, _ = score(xood)
        correct = (p_id.argmax(1) == yte)
        _, _, aurc = risk_coverage(s_id, correct)
        rows.append({
            "method": name,
            "auroc": round(float(auroc(s_ood, s_id)), 4),
            "aupr": round(float(aupr(s_ood, s_id)), 4),
            "fpr95": round(float(fpr95(s_ood, s_id)), 4),
            "ece": round(float(ece(p_id, yte)), 4),
            "aurc": round(float(aurc), 4),
            "id_accuracy": round(float(correct.mean()), 4),
            "ms_per_sample": round(_time(score, xte), 6),
        })

    OUT.write_text(json.dumps({
        "config": {"seed": seed, "epochs": epochs, "n_train": len(xtr),
                   "n_id_test": len(xte), "n_ood": len(xood), "mc_passes": MC_PASSES},
        "rows": rows,
    }, indent=2) + "\n")

    header = f"{'method':<28} {'AUROC':>7} {'AUPR':>7} {'FPR95':>7} {'ECE':>7} " \
             f"{'AURC':>7} {'ID acc':>7} {'ms/sample':>10}"
    print(header)
    print("-"*len(header))
    for r in rows:
        print(f"{r['method']:<28} {r['auroc']:>7.3f} {r['aupr']:>7.3f} {r['fpr95']:>7.3f} "
              f"{r['ece']:>7.3f} {r['aurc']:>7.3f} {r['id_accuracy']:>7.3f} "
              f"{r['ms_per_sample']:>10.4f}")
    print(f"\n({OUT.name} written)")
    return 0


def _time(score, x, repeats=5):
    """Time the score call itself; `score` already closes over its model."""
    from baselines import time_scoring
    return time_scoring(lambda a: score(a), x, repeats=repeats)


if __name__ == "__main__":
    raise SystemExit(main())
