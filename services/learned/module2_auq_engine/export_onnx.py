"""Export the trained EDL head to ONNX, and verify the export against torch.

    uv sync --package metacore-module2 --extra onnx    # or: pip install onnx onnxruntime onnxscript
    python export_onnx.py

What goes in the graph is only the network: **evidence**. The uncertainty arithmetic
(u = K/S), the observed_fraction discount and the trigger stay in Python, in infer.py.
That is deliberate -- `observed_fraction` is a runtime input that arrives with each state
from M1, and the trigger's thresholds are calibrated per deployment. Freezing either into
the artifact would mean re-exporting to retune a threshold.

Batch-1, fixed: the gate scores one state per control step. `dynamic_shapes` would give a
batched artifact for offline evaluation; the fixed shape is the one that matters for
Health.last_step_latency_ms.

The model card records the normalisation statistics alongside the graph. Without them the
ONNX file is unusable -- it expects standardised input, and mu/sd are training-set
properties that live nowhere else.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from baselines import train_edl
from edl import EDLNet
from state_contract import stack_features
from synthetic_data import FEATURES, Normalizer, sample_states_id

# torch's ONNX exporter prints status with emoji. On a Windows console defaulting to
# cp1252 that raises UnicodeEncodeError mid-export and leaves no artifact behind, so the
# stream is widened before the exporter is ever called.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
MODEL_PATH = HERE / "edl.onnx"
CARD_PATH = HERE / "edl.onnx.json"
REAL_MODEL_PATH = HERE / "edl_real.onnx"
REAL_CARD_PATH = HERE / "edl_real.onnx.json"
OPSET = 18
ATOL = 1e-5


def build():
    """Train the head the same way run_demo.py does, so the artifact is the real model."""
    with open(HERE / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    seed = cfg["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    states, y = sample_states_id(3000, rng, blackout_rate=0.0)
    x = stack_features(states)
    nz = Normalizer().fit(x)
    model = train_edl(nz(x), y, cfg["train"]["epochs"],
                      ood_reg=cfg["train"]["ood_reg_weight"], seed=seed)
    model.eval()
    return model, nz, cfg


def build_real():
    """Load the head train_real.py fitted on M1's real state distribution.

    Deliberately a load, not a retrain: the artifact must be the same model the numbers
    in real_m1_metrics.json describe. Retraining here would produce a second model that
    merely resembles it.
    """
    from train_real import CHECKPOINT

    if not CHECKPOINT.exists():
        raise SystemExit(
            f"no checkpoint at {CHECKPOINT}. Train it with: python train_real.py"
        )
    ckpt = torch.load(CHECKPOINT, weights_only=False)
    model = EDLNet(ckpt["n_features"], ckpt["classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def export(model, sample, out_path=MODEL_PATH):
    # external_data=False keeps the weights inside the .onnx rather than in a sidecar
    # .onnx.data. The exporter defaults to True, which is right for large models and
    # wrong here: this one is ~16 KB, and a gate artifact that silently depends on a
    # second file is a deployment trap.
    torch.onnx.export(
        model, (sample,), str(out_path),
        input_names=["state"], output_names=["evidence"],
        dynamo=True, opset_version=OPSET, external_data=False,
    )


def verify(model, sample, out_path=MODEL_PATH):
    """The export is only useful if it computes the same thing."""
    import onnxruntime as ort

    with torch.no_grad():
        expected = model(sample).numpy()
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"state": sample.numpy()})[0]
    max_diff = float(np.abs(expected - actual).max())
    if not np.allclose(expected, actual, atol=ATOL):
        raise SystemExit(f"ONNX output differs from torch by {max_diff} (atol {ATOL})")
    return max_diff


def main():
    ap = argparse.ArgumentParser(description="Freeze the EDL head to ONNX.")
    ap.add_argument("--real", action="store_true",
                    help="export the head trained on M1's real states (train_real.py) "
                         "instead of the synthetic demo head")
    args = ap.parse_args()
    if args.real:
        return main_real()

    model, nz, cfg = build()
    d_in, k = len(FEATURES), cfg["k_classes"]
    sample = torch.zeros(1, d_in)

    export(model, sample)
    max_diff = verify(model, torch.randn(1, d_in))

    CARD_PATH.write_text(json.dumps({
        "artifact": MODEL_PATH.name,
        "produces": "evidence",
        "note": "Evidence only. u = K/(sum(evidence)+K), the observed_fraction discount "
                "and the competence-drop trigger are computed by infer.py, outside the "
                "graph, because observed_fraction is a per-state runtime input and the "
                "thresholds are calibrated per deployment.",
        "architecture": {"input_dim": d_in, "hidden": 32, "classes": k,
                         "feature_names": list(FEATURES)},
        "io": {"input": "state", "output": "evidence",
               "batch_size": 1,
               "batch_note": "Fixed at 1 for the gate. Re-export with dynamic_shapes "
                             "for batched offline evaluation."},
        "normalisation": {"note": "Input must be standardised with these training-set "
                                  "statistics before inference.",
                          "mean": [float(v) for v in nz.mu],
                          "std": [float(v) for v in nz.sd]},
        "training": {"seed": cfg["seed"], "epochs": cfg["train"]["epochs"],
                     "ood_reg_weight": cfg["train"]["ood_reg_weight"], "n_train": 3000},
        "export": {"opset": OPSET, "exporter": "torch.onnx.export(dynamo=True)",
                   "torch_version": torch.__version__},
        "verification": {"atol": ATOL, "max_abs_diff_vs_torch": max_diff},
    }, indent=2) + "\n")

    size_kb = MODEL_PATH.stat().st_size/1024
    print(f"{MODEL_PATH.name}: {size_kb:.1f} KB, opset {OPSET}, batch-1, "
          f"{d_in} -> evidence[{k}]")
    print(f"verified against torch: max abs diff {max_diff:.2e} (atol {ATOL})")
    print(f"{CARD_PATH.name} written (includes the normalisation statistics)")
    return 0


def main_real():
    """Freeze the real-distribution head, alongside (not over) the synthetic artifact.

    Written to edl_real.onnx rather than edl.onnx on purpose. The synthetic artifact is
    what run_demo.py, infer_one.py and the dashboard fallback describe, and that mock
    demo is the guaranteed PP1 fallback -- overwriting it in place would silently change
    the numbers on the page that is meant to be the safe path. `infer.OnnxAUQ.load` takes
    explicit model_path / card_path, so serving the real head is a load-time choice.
    """
    from metacore_contracts.state_schema import FEATURE_NAMES

    model, ckpt = build_real()
    d_in, k = ckpt["n_features"], ckpt["classes"]
    sample = torch.zeros(1, d_in)

    export(model, sample, REAL_MODEL_PATH)
    max_diff = verify(model, torch.randn(1, d_in), REAL_MODEL_PATH)

    REAL_CARD_PATH.write_text(json.dumps({
        "artifact": REAL_MODEL_PATH.name,
        "produces": "evidence",
        "trained_on": "real Module 1 states (module1.train.dataset), 28-feature contract",
        "note": "Evidence only. u = K/(sum(evidence)+K), the observed_fraction discount "
                "and the competence-drop trigger are computed by infer.py, outside the "
                "graph, because observed_fraction is a per-state runtime input and the "
                "thresholds are calibrated per deployment.",
        "architecture": {"input_dim": d_in, "hidden": 32, "classes": k,
                         "class_names": ckpt["class_names"],
                         "feature_names": list(FEATURE_NAMES)},
        "io": {"input": "state", "output": "evidence", "batch_size": 1,
               "batch_note": "Fixed at 1 for the gate. Re-export with dynamic_shapes "
                             "for batched offline evaluation."},
        "normalisation": {"note": "Fitted on M1's REAL state distribution, where the five "
                                  "electrical features are QUALITY_MISSING and exactly "
                                  "0.0. This is what removes the -48 sigma saturation the "
                                  "synthetic card produced on real rows. Constant columns "
                                  "are scaled by 1, not by the 1e-6 epsilon.",
                          "mean": [float(v) for v in ckpt["mu"]],
                          "std": [float(v) for v in ckpt["sd"]]},
        "trigger": {"note": "Calibrated on real training u at train time, so serving "
                            "needs no calibration pass. Fitting it on demand would mean "
                            "rebuilding M1's 17,544-hour record and re-scoring 400 rows "
                            "on the first request -- a cold start on the live toggle.",
                    "value_threshold": ckpt["value_threshold"],
                    "value_false_alarm_rate": ckpt["value_false_alarm_rate"],
                    "observed_fraction_floor": ckpt["observed_fraction_floor"]},
        "label": {"source": "discretised M1 vulnerability proxy V",
                  "weighting": ckpt["weighting"],
                  "cuts_quantile": ckpt["class_cuts_quantile"],
                  "cuts_v": ckpt["class_cuts_v"],
                  "disclosure": ckpt["label_disclosure"]},
        "training": {"seed": ckpt["seed"], "epochs": ckpt["epochs"],
                     "n_train": ckpt["n_train"], "class_weighted": True},
        "export": {"opset": OPSET, "exporter": "torch.onnx.export(dynamo=True)",
                   "torch_version": torch.__version__},
        "verification": {"atol": ATOL, "max_abs_diff_vs_torch": max_diff},
    }, indent=2) + "\n")

    size_kb = REAL_MODEL_PATH.stat().st_size/1024
    print(f"{REAL_MODEL_PATH.name}: {size_kb:.1f} KB, opset {OPSET}, batch-1, "
          f"{d_in} -> evidence[{k}]")
    print(f"verified against torch: max abs diff {max_diff:.2e} (atol {ATOL})")
    print(f"{REAL_CARD_PATH.name} written (card fitted on the REAL distribution)")
    print(f"{MODEL_PATH.name} left untouched -- the mock demo stays the fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
