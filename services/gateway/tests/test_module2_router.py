"""Exercises /api/module2/run end-to-end: real subprocess, real infer_one.py, real EDL
head. This is the wiring test (gateway -> subprocess -> module2); the uncertainty maths
itself is tested in services/learned/module2_auq_engine/tests/.

The first scoring call trains and caches the dashboard fixture, so the module-scoped
`nominal` fixture pays that once and the rest of the file reuses the cache.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from gateway.main import app
from gateway.routers.module2 import MODULE2_DIR, N_FEATURES

client = TestClient(app)

# module2's prototype lives as flat modules at its service root (the same reason its own
# conftest does this). Importing synthetic_data here lets the uncertainty assertions run
# against a row the head actually considers in-distribution — see the saturation test for
# why a real M1 row cannot serve that purpose yet.
if str(MODULE2_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE2_DIR))

# The 12 features M1 marks QUALITY_OBSERVED: the four temporal and eight static-topology
# ones. Nominal observed_fraction is therefore 12/28 — see module2's config.yaml.
OBSERVED = frozenset({
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "nominal_kv_norm", "critical_load", "is_bus", "is_pv",
    "is_wind", "is_bess", "is_diesel", "is_load",
})
NOMINAL_OF = 12 / 28


def _feature_names() -> tuple[str, ...]:
    from metacore_contracts.state_schema import FEATURE_NAMES

    return FEATURE_NAMES


def _mask(observed: frozenset[str]) -> list[str]:
    return [
        "QUALITY_OBSERVED" if n in observed else "QUALITY_INTERPOLATED"
        for n in _feature_names()
    ]


def _state(island: str = "eluvaitivu", scenario: str = "normal") -> dict:
    """One real M1 state, straight from /api/module1/assemble — the composition this
    endpoint exists to support."""
    r = client.post("/api/module1/assemble", json={"island": island, "scenario": scenario})
    assert r.status_code == 200, r.text
    data = r.json()
    return {"features": data["node_features"][0], "quality_mask": data["quality_mask"]}


def _id_features() -> list[float]:
    """An in-distribution feature row from module2's own generator.

    The uncertainty assertions need a state the head has evidence for. A real M1 row does
    not qualify today (test_a_real_m1_state_currently_saturates_the_value_axis), so using
    one here would compare 1.0 against 1.0 and assert nothing.
    """
    from synthetic_data import sample_states_id

    states, _ = sample_states_id(1, np.random.default_rng(7), blackout_rate=0.0)
    return states[0].node_features[0].tolist()


# ---------------------------------------------------------------- request validation


def test_module2_rejects_a_request_with_neither_form() -> None:
    assert client.post("/api/module2/run", json={}).status_code == 422


def test_module2_rejects_both_forms_at_once() -> None:
    r = client.post(
        "/api/module2/run",
        json={
            "features": [0.0] * N_FEATURES,
            "quality_mask": _mask(OBSERVED),
            "novelty": 0.5,
            "observed_fraction": 1.0,
        },
    )
    assert r.status_code == 422


def test_module2_rejects_features_without_a_mask() -> None:
    r = client.post("/api/module2/run", json={"features": [0.0] * N_FEATURES})
    assert r.status_code == 422


def test_module2_rejects_the_wrong_feature_width() -> None:
    """The pinned contract is 28 wide; anything else is a caller bug, not a 500."""
    r = client.post(
        "/api/module2/run",
        json={"features": [0.0] * 8, "quality_mask": _mask(OBSERVED)[:8]},
    )
    assert r.status_code == 422


def test_module2_rejects_a_mask_that_does_not_cover_every_feature() -> None:
    r = client.post(
        "/api/module2/run",
        json={"features": [0.0] * N_FEATURES, "quality_mask": ["QUALITY_OBSERVED"] * 4},
    )
    assert r.status_code == 422


def test_router_width_matches_the_pinned_contract() -> None:
    """The router hardcodes 28 so the gateway needs no ML stack; this is the pin that
    stops the two drifting."""
    assert N_FEATURES == len(_feature_names())


# ---------------------------------------------------------------- the contract path


@pytest.fixture(scope="module")
def nominal() -> dict:
    r = client.post(
        "/api/module2/run",
        json={"features": _state()["features"], "quality_mask": _mask(OBSERVED)},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_module2_scores_a_real_m1_state(nominal: dict) -> None:
    assert 0.0 <= nominal["u"] <= 1.0
    assert 0.0 <= nominal["u_q"] <= 1.0
    assert nominal["reason"] in {"none", "value", "sensing", "both"}
    assert nominal["backend"] in {"onnx", "torch"}
    assert nominal["latency_ms"] > 0.0


def test_module2_derives_the_sensing_fraction_from_the_mask(nominal: dict) -> None:
    """The point of the contract form: f is computed from per_feature, not supplied."""
    assert nominal["observed_fraction"] == pytest.approx(NOMINAL_OF, abs=1e-4)


def test_quality_aware_u_is_at_least_plain_u(nominal: dict) -> None:
    """u' = K/(f·Σe + K) discounts evidence by f, so it can only rise as f falls."""
    assert nominal["u_q"] >= nominal["u"] - 1e-9


def test_a_real_m1_state_currently_saturates_the_value_axis(nominal: dict) -> None:
    """Recorded, not asserted as desirable: the head is trained on synthetic_data, and
    M1's real assembled row — whose electrical block is QUALITY_MISSING and zero-filled —
    reads as fully out-of-distribution. Evidence is K (i.e. none at all), so u = u' = 1
    and the quality axis has no room left to move.

    The wiring is correct and f derives correctly from the mask; what is missing is a head
    trained on M1's real feature distribution. Change this test when that lands.
    """
    assert nominal["u"] == pytest.approx(1.0, abs=1e-3)
    assert nominal["evidence"] == pytest.approx(3.0, abs=1e-3)   # K, no evidence at all
    assert nominal["reason"] == "value"


def test_a_degraded_mask_raises_u_and_fires_the_sensing_axis() -> None:
    """Same feature values, fewer of them observed. The value axis must not move — that
    is what makes the two axes separable — while f falls and the sensing axis fires."""
    features = _id_features()
    full = client.post(
        "/api/module2/run", json={"features": features, "quality_mask": _mask(OBSERVED)}
    ).json()
    # Drop the whole temporal block: 8/28 observed, under the 0.35 floor.
    degraded_mask = _mask(OBSERVED - {"hour_sin", "hour_cos", "doy_sin", "doy_cos"})
    degraded = client.post(
        "/api/module2/run", json={"features": features, "quality_mask": degraded_mask}
    ).json()

    assert degraded["observed_fraction"] == pytest.approx(8 / 28, abs=1e-4)
    assert degraded["u"] == pytest.approx(full["u"], abs=1e-6)
    assert degraded["u_q"] > full["u_q"]
    assert degraded["trigger"] is True
    assert degraded["reason"] in {"sensing", "both"}


def test_a_fully_observed_mask_leaves_u_unchanged() -> None:
    """At f = 1 the quality discount is the identity, so u' collapses back onto u."""
    r = client.post(
        "/api/module2/run",
        json={"features": _id_features(), "quality_mask": ["QUALITY_OBSERVED"] * N_FEATURES},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["observed_fraction"] == pytest.approx(1.0)
    assert data["u_q"] == pytest.approx(data["u"], abs=1e-6)


# ---------------------------------------------------------------- the synthetic path


def test_module2_still_serves_the_dashboard_sliders() -> None:
    r = client.post("/api/module2/run", json={"novelty": 0.0, "observed_fraction": 1.0})
    assert r.status_code == 200, r.text
    assert r.json()["observed_fraction"] == pytest.approx(1.0)


def test_high_novelty_fires_the_value_axis() -> None:
    r = client.post("/api/module2/run", json={"novelty": 1.0, "observed_fraction": 1.0})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["trigger"] is True
    assert data["reason"] == "value"


# ---------------------------------------------------------------- the head selector


def _contract_body(**extra) -> dict:
    return {"features": _id_features(), "quality_mask": _mask(OBSERVED), **extra}


def test_head_defaults_to_synthetic() -> None:
    """The fallback demo path must stay the default. Real is strictly opt-in."""
    r = client.post("/api/module2/run", json=_contract_body())
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["head"] == "synthetic"
    assert data["artifact"] == "edl.onnx"
    assert data["baselines"] is not None
    assert data["label_disclosure"] is None


def test_explicit_synthetic_matches_the_default() -> None:
    """Naming the default must not change the answer."""
    default = client.post("/api/module2/run", json=_contract_body()).json()
    explicit = client.post("/api/module2/run", json=_contract_body(head="synthetic")).json()

    for key in ("head", "artifact", "u", "u_q", "observed_fraction", "reason", "trigger"):
        assert default[key] == explicit[key], key


def test_the_novelty_form_still_defaults_to_synthetic() -> None:
    r = client.post("/api/module2/run", json={"novelty": 0.1, "observed_fraction": 1.0})
    assert r.status_code == 200, r.text
    assert r.json()["head"] == "synthetic"


def test_real_head_is_rejected_on_the_novelty_form() -> None:
    """That form manufactures a state, and a head fitted on M1's real distribution has no
    interpretation for one. A 422 is the honest answer, not a quiet number."""
    r = client.post(
        "/api/module2/run",
        json={"novelty": 0.5, "observed_fraction": 1.0, "head": "real"},
    )
    assert r.status_code == 422


def test_an_unknown_head_is_rejected() -> None:
    assert client.post("/api/module2/run", json=_contract_body(head="wishful")).status_code == 422


def _real_artifact_present() -> bool:
    return all((MODULE2_DIR / name).exists()
               for name in ("edl_real.onnx", "edl_real.onnx.json"))


real_head = pytest.mark.skipif(
    not _real_artifact_present(),
    reason="edl_real.onnx not built; run `python train_real.py && python export_onnx.py --real`",
)


@real_head
def test_the_real_head_serves_its_own_artifact() -> None:
    r = client.post("/api/module2/run", json=_contract_body(head="real"))
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["head"] == "real"
    assert data["artifact"] == "edl_real.onnx"
    assert 0.0 <= data["u"] <= 1.0
    assert data["reason"] in {"none", "value", "sensing", "both"}


@real_head
def test_the_real_head_reports_no_baselines() -> None:
    """Deliberately absent rather than faked: those baselines are fitted on the SYNTHETIC
    distribution, so reporting them beside a real-head u invites the exact comparison the
    head field exists to prevent."""
    data = client.post("/api/module2/run", json=_contract_body(head="real")).json()

    assert data["baselines"] is None


@real_head
def test_the_real_head_carries_its_label_disclosure() -> None:
    """Its label is discretised from M1's constructed vulnerability proxy, not an observed
    safety class. That has to travel with the numbers, not only with the report."""
    data = client.post("/api/module2/run", json=_contract_body(head="real")).json()

    assert data["label_disclosure"]
    assert "constructed proxy" in data["label_disclosure"]


@real_head
def test_the_two_heads_disagree_on_the_same_state() -> None:
    """The whole point of the selector. The synthetic head saturates on a real M1 row;
    the real head does not. If these ever matched, one of them would be mislabelled."""
    state = _state()
    body = {"features": state["features"], "quality_mask": state["quality_mask"]}
    synthetic = client.post("/api/module2/run", json={**body, "head": "synthetic"}).json()
    real = client.post("/api/module2/run", json={**body, "head": "real"}).json()

    assert synthetic["u"] == pytest.approx(1.0, abs=1e-3)
    assert synthetic["artifact"] != real["artifact"]


@real_head
def test_warm_primes_the_real_head() -> None:
    """Cheap by construction: the threshold is calibrated at train time, so this is a
    session create plus one forward pass rather than a fixture build."""
    r = client.post("/api/module2/warm")
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["head"] == "real"
    assert data["ready"] is True
    assert data["warm_ms"] > 0.0
    assert data["value_threshold"] > 0.0


def test_a_missing_real_artifact_is_a_503_not_a_500(monkeypatch) -> None:
    """edl_real.onnx is derived and git-ignored, so its absence on a fresh clone is an
    ordinary state. The route must say how to build it, not fail opaquely."""
    from gateway.routers import module2 as router_module

    monkeypatch.setattr(router_module, "REAL_SCRIPT", "serve_real_does_not_exist.py")
    r = client.post("/api/module2/run", json=_contract_body(head="real"))

    assert r.status_code in {503, 500}
    if r.status_code == 500:
        pytest.fail(f"missing artifact surfaced as 500: {r.text[:300]}")
