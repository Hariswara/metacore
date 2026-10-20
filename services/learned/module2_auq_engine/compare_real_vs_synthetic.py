"""Score one live M1 /forecast payload through both heads, and print them side by side.

    python compare_real_vs_synthetic.py
    python compare_real_vs_synthetic.py --timestamp 2025-11-28T08:00

This is the acceptance check for the retrain. It takes a real payload from M1's trained
ST-GNN -- the same object the gateway returns from /api/module1/forecast -- and runs
`infer.OnnxAUQ.score()` over every node with each artifact in turn:

  edl.onnx       the synthetic demo head. Its normalisation card was fitted on draws
                 where voltage_pu ~ 1.0 +/- 0.021, so a real row's 0.0 on that column is
                 z = -48 sigma and the head correctly reports u = 1.0 on everything.
  edl_real.onnx  the head train_real.py fitted on M1's real state distribution, with the
                 card refitted there. This is the root fix.

Both artifacts are git-ignored and rebuilt from seed:

    python export_onnx.py              # edl.onnx      (synthetic)
    python train_real.py               # the real head
    python export_onnx.py --real       # edl_real.onnx

Nothing here changes the synthetic path. edl.onnx is read, never written.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import m1_real
import numpy as np
from infer import OnnxAUQ

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[2]
M1_SRC = REPO_ROOT / "services" / "learned" / "module1_state_forecasting" / "src"

SYNTHETIC = (HERE / "edl.onnx", HERE / "edl.onnx.json")
REAL = (HERE / "edl_real.onnx", HERE / "edl_real.onnx.json")

DEFAULT_TIMESTAMP = "2025-11-28T08:00"
# Hours from the TRAINING split, used only to calibrate the value threshold -- never the
# hour under test, which comes from the test split.
CALIBRATION_HOURS = 400


def forecast_payload(timestamp: str) -> dict:
    """One real /forecast payload, straight from M1's trained ST-GNN."""
    proc = subprocess.run(
        [sys.executable, "-m", "module1.infer.forecast", "--timestamp", timestamp, "--json"],
        cwd=M1_SRC, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise SystemExit(f"M1 forecast failed:\n{proc.stderr.strip()[-2000:]}")
    return json.loads(proc.stdout)


def score_all(artifact, calibration: np.ndarray, payload: dict):
    """Every node of the payload through one artifact, with the trigger calibrated first."""
    model_path, card_path = artifact
    if not model_path.exists():
        return None
    auq = OnnxAUQ.load(model_path=model_path, card_path=card_path)
    auq.calibrate(calibration)

    rows = []
    for node, features in zip(payload["nodes"], payload["node_features"], strict=True):
        auq.reset()                     # per-node, so hysteresis does not leak across nodes
        out = auq.score(features, payload["observed_fraction"])
        rows.append({
            "node": node["id"],
            "u": out.epistemic_uncertainty,
            "class": out.state_class,
            "trigger": out.competence_drop,
            "reason": out.trigger_reason,
        })
    return {"value_threshold": auq.value_threshold, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timestamp", default=DEFAULT_TIMESTAMP)
    args = ap.parse_args()

    payload = forecast_payload(args.timestamp)
    states = m1_real.load()
    train = states.where("train")
    rng = np.random.default_rng(0)
    calibration = train.x[rng.choice(len(train.x), CALIBRATION_HOURS, replace=False)]

    print(f"M1 /forecast payload : {payload['timestamp_lst']}  split={payload['split']}  "
          f"nodes={len(payload['nodes'])}")
    print(f"observed_fraction    : {payload['observed_fraction']}  "
          f"(mask: 12 observed / 11 interpolated / 5 missing)")
    print()

    results = {}
    for name, artifact in (("synthetic (edl.onnx)", SYNTHETIC), ("real (edl_real.onnx)", REAL)):
        scored = score_all(artifact, calibration, payload)
        if scored is None:
            print(f"!! {name}: artifact missing -- see the module docstring for how to build it")
            continue
        results[name] = scored

    if len(results) < 2:
        return 2

    head = f"  {'node':<24} " + " ".join(f"{n:>26}" for n in results)
    print("u per node, same payload, both heads:")
    print(head)
    print("  " + "-"*(len(head)-2))
    for i, node in enumerate(payload["nodes"]):
        cells = []
        for name in results:
            r = results[name]["rows"][i]
            flag = r["reason"] if r["trigger"] else "-"
            cells.append(f"{r['u']:>14.4f}  {flag:>10}")
        print(f"  {node['id']:<24} " + " ".join(cells))

    print()
    for name, scored in results.items():
        u = np.array([r["u"] for r in scored["rows"]])
        fired = sum(r["trigger"] for r in scored["rows"])
        print(f"  {name:<22} u: min {u.min():.4f}  mean {u.mean():.4f}  max {u.max():.4f}   "
              f"saturated(u>0.99) {int((u > 0.99).sum())}/{len(u)}   fires {fired}/{len(u)}")

    real_u = np.array([r["u"] for r in results["real (edl_real.onnx)"]["rows"]])
    verdict = "SENSIBLE" if (real_u < 0.99).any() else "STILL SATURATED"
    print(f"\n  verdict: real head is {verdict} on a live payload "
          f"({int((real_u > 0.99).sum())}/{len(real_u)} nodes at u > 0.99)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
