"""/api/module1/topology and /api/module1/snapshot: the M1 -> M4 contract over HTTP."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from gateway.main import app

client = TestClient(app)


def test_topology_is_the_islanded_pin() -> None:
    r = client.get("/api/module1/topology")
    assert r.status_code == 200, r.text
    t = r.json()
    assert t["topology_id"] == "northern-islands-v1"
    island = {b["node_id"]: b["island"] for b in t["buses"]}
    assert all(island[ln["from_node"]] == island[ln["to_node"]] for ln in t["lines"])


def test_snapshot_matches_the_topology_and_marks_missing_fields() -> None:
    r = client.post("/api/module1/snapshot", json={"timestamp": "2025-11-28T19:00"})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["topology_id"] == "northern-islands-v1"
    buses = {b["node_id"] for b in client.get("/api/module1/topology").json()["buses"]}
    assert {n["node_id"] for n in s["node_states"]} <= buses
    assert all(n["quality"]["voltage_pu"] == "QUALITY_MISSING" for n in s["node_states"])


@pytest.mark.parametrize("stamp", ["", "garbage", "2030-01-01T00:00"])
def test_snapshot_rejects_garbage_and_out_of_record_hours(stamp: str) -> None:
    r = client.post("/api/module1/snapshot", json={"timestamp": stamp})
    assert r.status_code == 422, r.text
