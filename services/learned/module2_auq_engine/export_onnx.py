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

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from baselines import train_edl
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


def export(model, sample):
    # external_data=False keeps the weights inside the .onnx rather than in a sidecar
    # .onnx.data. The exporter defaults to True, which is right for large models and
    # wrong here: this one is ~16 KB, and a gate artifact that silently depends on a
    # second file is a deployment trap.
    torch.onnx.export(
        model, (sample,), str(MODEL_PATH),
        input_names=["state"], output_names=["evidence"],
        dynamo=True, opset_version=OPSET, external_data=False,
    )


def verify(model, sample):
    """The export is only useful if it computes the same thing."""
    import onnxruntime as ort

    with torch.no_grad():
        expected = model(sample).numpy()
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"state": sample.numpy()})[0]
    max_diff = float(np.abs(expected - actual).max())
    if not np.allclose(expected, actual, atol=ATOL):
        raise SystemExit(f"ONNX output differs from torch by {max_diff} (atol {ATOL})")
    return max_diff


def main():
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


if __name__ == "__main__":
    raise SystemExit(main())
