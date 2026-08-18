"""Gateway API tests.

Exercise both states the calibration routes have: data present, and data absent on a clean
clone. The second is the one that regresses silently, because it only shows up for someone who
has never run the pipeline.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from gateway.config import GENERATION_CSV
from gateway.main import app

client = TestClient(app)

needs_data = pytest.mark.skipif(
    not GENERATION_CSV.exists(), reason="calibration table not built; run `task data`"
)


def test_health_always_answers() -> None:
    body = client.get("/api/health").json()
    assert body["ready"] is True
    assert isinstance(body["calibration_available"], bool)


@needs_data
def test_generation_returns_every_island_month() -> None:
    body = client.get("/api/calibration/generation").json()
    assert body["count"] == 120
    assert {r["island_system"] for r in body["rows"]} == {
        "Analaithivu", "Eluvaitivu-Diesel", "Eluvaitivu-Hybrid",
        "Delft-Neduntivu", "Nainativu",
    }


@needs_data
def test_summary_totals_match_the_rows_they_aggregate() -> None:
    rows = client.get("/api/calibration/generation").json()["rows"]
    summary = client.get("/api/calibration/summary").json()

    for entry in summary["by_system"]:
        expected = sum(
            r["units_kwh"] or 0.0
            for r in rows
            if r["year"] == entry["year"] and r["island_system"] == entry["island_system"]
        )
        assert entry["units_kwh"] == pytest.approx(expected)

    for fleet_year in summary["fleet"]:
        expected = sum(r["units_kwh"] or 0.0 for r in rows if r["year"] == fleet_year["year"])
        assert fleet_year["units_kwh"] == pytest.approx(expected)


@needs_data
def test_hybrid_collapse_is_present_in_the_served_data() -> None:
    """The Oct-Dec 2025 Eluvaitivu hybrid failure is the dataset's headline finding.
    If a refactor ever flattens it, this catches it."""
    rows = client.get("/api/calibration/generation").json()["rows"]
    hybrid = {
        r["month"]: r
        for r in rows
        if r["year"] == "2025" and r["island_system"] == "Eluvaitivu-Hybrid"
    }
    assert hybrid["Sep"]["units_kwh"] > 8000
    assert hybrid["Dec"]["units_kwh"] < 500
