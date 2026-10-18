"""Exercises /api/module1/forecast end-to-end: real subprocess, real module1.infer.forecast, the
real trained checkpoint. The checkpoint is git-ignored (module1.train.run makes it), so the
success-path tests skip on a clean clone instead of passing against something invented."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from gateway.main import app

client = TestClient(app)
CKPT = (Path(__file__).resolve().parents[3] / "services" / "learned" / "module1_state_forecasting"
        / "artifacts" / "stgnn_v0_1_equal.pt")
needs_checkpoint = pytest.mark.skipif(
    not CKPT.exists(), reason="run module1.train.run --variant v0.1"
)


def test_forecast_rejects_unknown_weighting() -> None:
    r = client.post("/api/module1/forecast", json={"weighting": "vibes"})
    assert r.status_code == 422


@needs_checkpoint
@pytest.mark.parametrize("stamp", ["", "garbage", "2025-13-40T99:00", "2024-01-01T03:00"])
def test_forecast_rejects_garbage_empty_and_history_less_hours(stamp: str) -> None:
    r = client.post("/api/module1/forecast", json={"timestamp": stamp})
    assert r.status_code == 422, r.text


@needs_checkpoint
def test_an_empty_body_forecasts_the_default_ditwah_hour() -> None:
    r = client.post("/api/module1/forecast", json={})
    assert r.status_code == 200, r.text
    assert r.json()["timestamp_lst"] == "2025-11-28T08:00"


@needs_checkpoint
def test_forecast_rejects_an_hour_outside_the_record() -> None:
    r = client.post("/api/module1/forecast", json={"timestamp": "2030-01-01T00:00"})
    assert r.status_code == 422
    assert "not an hour in the record" in r.json()["detail"]


@needs_checkpoint
def test_forecast_returns_every_node_with_six_horizons_and_an_embedding() -> None:
    r = client.post("/api/module1/forecast", json={"timestamp": "2025-11-28T08:00"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["split"] == "test"
    assert len(data["nodes"]) == 10
    for n in data["nodes"]:
        assert len(n["forecast"]) == len(n["persistence"]) == len(n["target"]) == 6
        assert len(n["history"]) == 24
        assert len(n["embedding"]) == data["embedding_dim"] == 64
        assert all(0.0 <= z <= 1.0 for z in n["forecast"])
    assert set(data["ranking_t1"]) == {n["id"] for n in data["nodes"]}


@needs_checkpoint
def test_a_training_hour_is_labelled_as_training() -> None:
    r = client.post("/api/module1/forecast", json={"timestamp": "2024-06-15T18:00"})
    assert r.status_code == 200, r.text
    assert r.json()["split"] == "train"
