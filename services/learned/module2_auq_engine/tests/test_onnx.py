"""ONNX export and batch-1 latency.

Skips cleanly when onnxruntime is absent, the same way M1's calibration tests skip
without their workbook: the `onnx` extra is optional, so the default lane must stay green
without it.

Exports a tiny model of its own rather than leaning on the committed edl.onnx -- the
committed artifact is a full-scale training run, and a test that depended on it could not
run on a clean checkout that had not executed the export script.
"""

import time

import numpy as np
import pytest
import torch
from edl import EDLNet

ort = pytest.importorskip("onnxruntime", reason="onnx extra not installed")

D_IN, K = 8, 3
ATOL = 1e-5


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """A tiny exported model -> (torch_model, onnx_session)."""
    pytest.importorskip("onnxscript", reason="onnx extra not installed")
    torch.manual_seed(0)
    model = EDLNet(D_IN, K).eval()
    path = tmp_path_factory.mktemp("onnx") / "tiny.onnx"
    torch.onnx.export(
        model, (torch.zeros(1, D_IN),), str(path),
        input_names=["state"], output_names=["evidence"],
        dynamo=True, opset_version=18, external_data=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return model, session, path


def test_onnx_matches_torch(exported):
    """The export is only worth having if it computes the same thing."""
    model, session, _ = exported
    rng = np.random.default_rng(0)

    for _ in range(5):
        x = rng.standard_normal((1, D_IN)).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        actual = session.run(None, {"state": x})[0]
        assert np.allclose(expected, actual, atol=ATOL), np.abs(expected-actual).max()


def test_evidence_stays_non_negative(exported):
    """Softplus output. A negative would make alpha < 1 and u meaningless."""
    _, session, _ = exported
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, D_IN)).astype(np.float32)*10

    assert (session.run(None, {"state": x})[0] >= 0).all()


def test_artifact_is_self_contained(exported):
    """external_data=False: no sidecar .onnx.data to lose on the way to deployment."""
    _, _, path = exported

    assert not (path.parent / (path.name + ".data")).exists()
    assert list(path.parent.iterdir()) == [path]


def test_runtime_consumer_reproduces_the_two_axes():
    """infer.OnnxAUQ is the path M3 calls. Skipped on a checkout that has not run the
    export script, since it needs the committed artifact rather than a tiny stand-in."""
    from pathlib import Path

    import infer

    if not infer.MODEL_PATH.exists():
        pytest.skip("edl.onnx absent; run `python export_onnx.py`")
    from synthetic_data import sample_states_blackout, sample_states_id, sample_states_ood

    auq = infer.OnnxAUQ.load()
    rng = np.random.default_rng(0)
    calibration, _ = sample_states_id(300, rng, blackout_rate=0.0)
    auq.calibrate([s.node_features[0] for s in calibration])
    assert auq.value_threshold > 0

    def fired_reasons(states):
        auq.reset()
        out = []
        for s in states[:8]:
            for _ in range(auq.hysteresis):        # let the debounce confirm
                msg = auq.score(s.node_features[0], s.quality.observed_fraction)
            out.append((msg.competence_drop, msg.trigger_reason))
        return out

    rng = np.random.default_rng(7)
    normal = fired_reasons(sample_states_id(20, rng, blackout_rate=0.0)[0])
    cyclone = fired_reasons(sample_states_ood(20, rng)[0])
    blackout = fired_reasons(sample_states_blackout(20, rng)[0])

    assert not any(f for f, _ in normal)
    assert all(f for f, _ in cyclone)
    assert all(r == "value" for _, r in cyclone)
    assert all(f for f, _ in blackout)
    assert all(r in ("sensing", "both") for _, r in blackout)
    assert Path(infer.CARD_PATH).exists()


def test_batch_one_latency_is_within_budget(exported):
    """Generous and non-flaky: the point is that a per-state score is sub-millisecond
    territory, not to pin a number on shared CI hardware. bench_latency.py is where the
    real figures come from."""
    _, session, _ = exported
    x = np.zeros((1, D_IN), dtype=np.float32)

    def call():
        session.run(None, {"state": x})

    for _ in range(50):
        call()
    samples = []
    for _ in range(200):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter()-start)*1000.0)
    p50, p99 = np.percentile(samples, [50, 99])

    assert p50 < 5.0, p50
    assert p50 <= p99
