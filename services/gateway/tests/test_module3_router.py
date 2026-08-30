"""Exercises /api/module3/run end-to-end: real subprocess, real run_demo.py, real
training. Kept fast with a small episode_len/train_episodes so it still runs in CI —
this is the wiring test (gateway -> subprocess -> module3), not a policy-quality test
(those live in services/learned/module3_metapolicy/tests)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from gateway.main import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_module3_run_rejects_out_of_range_config() -> None:
    r = client.post("/api/module3/run", json={"train_episodes": 1})
    assert r.status_code == 422


def test_module3_run_returns_result_shape() -> None:
    r = client.post(
        "/api/module3/run",
        json={"seed": 1, "episode_len": 20, "train_episodes": 10, "eval_episodes": 1},
    )
    assert r.status_code == 200
    data = r.json()
    assert set(data["reward"]) == {"always_s1", "always_s2", "trained_policy"}
    assert set(data["escalation_by_trigger_reason"]) == {"none", "value", "sensing", "both"}
    assert isinstance(data["monotonic_nondecreasing"], bool)
    assert len(data["decisions"]) > 0
    assert len(data["decision_context"]) * 2 == len(data["decisions"])
    ctx = data["decision_context"][0]
    assert {"severity", "trigger_reason", "epistemic_uncertainty", "observed_fraction", "verdict"} <= set(ctx)

    gen = client.post("/api/module3/generate")
    assert gen.status_code == 200, gen.text
    obs = gen.json()["observation"]
    assert len(obs) == 12
    assert obs[0]["name"] == "epistemic_uncertainty"

    proc = client.post("/api/module3/process")
    assert proc.status_code == 200, proc.text
    body = proc.json()
    assert body["chosen"] in {"SYSTEM1", "SYSTEM2"}
    assert body["plan"]["origin"] == body["chosen"]
    assert "load_shed" in body["plan"]
