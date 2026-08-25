"""Batch-1 latency for the gate: torch-eager against ONNX Runtime.

    uv sync --package metacore-module2 --extra onnx
    python export_onnx.py
    python bench_latency.py

The measurement this file exists to get right: **a gate scores one state per control
step**, so its cost is the latency of a single call, not the amortised cost of a large
batch. Dividing a batch-1000 forward pass by 1000 hides per-call dispatch and kernel
launch entirely and reports a number the real path never sees. Both are printed, with the
ratio, so the difference is on the record rather than implied.

p99 matters as much as p50 here: M3 budgets deliberation against a step deadline, and a
tail that blows the budget is a missed step, not a slow one.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
MODEL_PATH = HERE / "edl.onnx"
OUT = HERE / "latency_table.json"

WARMUP = 50
REPS = 300
BATCH = 1000


def percentiles(fn, x, reps=REPS, warm=WARMUP):
    """Mean / p50 / p99 milliseconds for one call, after a warm-up."""
    for _ in range(warm):
        fn(x)
    samples = np.empty(reps)
    for i in range(reps):
        start = time.perf_counter()
        fn(x)
        samples[i] = (time.perf_counter() - start)*1000.0
    return {"mean_ms": float(samples.mean()),
            "p50_ms": float(np.percentile(samples, 50)),
            "p99_ms": float(np.percentile(samples, 99))}


def amortised(fn, x, reps=20, warm=5):
    """Milliseconds per sample when a large batch is scored in one call. This is
    throughput, and it is NOT the gate's latency -- printed only for the comparison."""
    for _ in range(warm):
        fn(x)
    best = min(_timed(fn, x) for _ in range(reps))
    return best/len(x)


def _timed(fn, x):
    start = time.perf_counter()
    fn(x)
    return (time.perf_counter() - start)*1000.0


def main():
    if not MODEL_PATH.exists():
        print(f"no {MODEL_PATH.name}; run `python export_onnx.py` first", file=sys.stderr)
        return 2
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; `uv sync --package metacore-module2 "
              "--extra onnx`", file=sys.stderr)
        return 2

    card = json.loads((HERE / "edl.onnx.json").read_text())
    d_in = card["architecture"]["input_dim"]

    from edl import EDLNet
    torch_model = EDLNet(d_in, card["architecture"]["classes"]).eval()
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

    x1_t, x1_n = torch.randn(1, d_in), np.random.randn(1, d_in).astype(np.float32)
    xb_t = torch.randn(BATCH, d_in)
    xb_n = np.random.randn(BATCH, d_in).astype(np.float32)

    def torch_fn(x):
        with torch.no_grad():
            return torch_model(x)

    def onnx_fn(x):
        return session.run(None, {"state": x})[0]

    # The exported artifact is fixed at batch 1, so it cannot be run batched at all --
    # feeding it 1000 rows is an InvalidArgument, not a slow path. The amortised figure is
    # therefore torch-only, which is the comparison that matters anyway: the ms/sample
    # column in comparison_table.json is torch-eager amortised, and this is how much
    # rosier that is than what the gate pays.
    torch_row = {"backend": "torch-eager", **percentiles(torch_fn, x1_t),
                 "amortised_batch1000_ms_per_sample": amortised(torch_fn, xb_t)}
    torch_row["amortised_optimism_ratio"] = (
        torch_row["p50_ms"]/torch_row["amortised_batch1000_ms_per_sample"])
    onnx_row = {"backend": "onnxruntime", **percentiles(onnx_fn, x1_n),
                "amortised_batch1000_ms_per_sample": None,
                "amortised_optimism_ratio": None,
                "note": "artifact is fixed batch-1; batched throughput would need a "
                        "separate export with dynamic_shapes"}
    rows = [torch_row, onnx_row]
    del xb_n

    OUT.write_text(json.dumps({
        "method": {"batch_size": 1, "warmup": WARMUP, "reps": REPS,
                   "amortised_batch": BATCH,
                   "note": "p50/p99 are batch-1 per-call latency, the gate's real cost. "
                           "amortised_* is batch throughput and must not be reported as "
                           "latency."},
        "torch_version": torch.__version__,
        "onnxruntime_version": ort.__version__,
        "rows": rows,
    }, indent=2) + "\n")

    header = f"{'backend':<14} {'mean ms':>9} {'p50 ms':>9} {'p99 ms':>9} " \
             f"{'amortised':>11} {'optimism':>9}"
    print("Batch-1 latency (one state per control step -- the gate's real cost)")
    print(header)
    print("-"*len(header))
    for r in rows:
        amort = r["amortised_batch1000_ms_per_sample"]
        amort_s = f"{amort:>11.4f}" if amort else f"{'n/a':>11}"
        ratio_s = (f"{r['amortised_optimism_ratio']:>8.0f}x"
                   if r["amortised_optimism_ratio"] else f"{'-':>9}")
        print(f"{r['backend']:<14} {r['mean_ms']:>9.4f} {r['p50_ms']:>9.4f} "
              f"{r['p99_ms']:>9.4f} {amort_s} {ratio_s}")
    speedup = rows[0]["p50_ms"]/rows[1]["p50_ms"]
    print(f"\nONNX Runtime is {speedup:.1f}x faster than torch-eager at batch-1 (p50).")
    print("'amortised' is batch-1000 throughput per sample and is NOT latency. The "
          "'optimism' column is\nhow much rosier it looks than the number the gate pays "
          "-- the ms/sample column in\ncomparison_table.json is exactly this optimistic. "
          "ONNX shows n/a because the artifact is\nfixed batch-1 and cannot be run "
          "batched at all.")
    print(f"\n({OUT.name} written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
