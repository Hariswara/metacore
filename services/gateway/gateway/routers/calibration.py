"""Module 1 calibration artifacts, read-only.

Serves the CEB Jaffna generation table and the aggregates the dashboard's baseline view needs.
Nothing here computes anything the pipeline could have computed — the aggregation is a projection
of the same rows, so a number on screen can always be traced back to a row in the CSV.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, HTTPException

from ..config import GENERATION_CSV

router = APIRouter(prefix="/calibration", tags=["calibration"])

NUMERIC = (
    "diesel_l", "diesel_cost_rs", "units_kwh", "oil_l", "oil_cost_rs",
    "diesel_barrel", "barrel_amount", "sfc_l_per_kwh", "diesel_rs_per_l",
    "fuel_cost_rs_per_kwh", "total_cost_rs_per_kwh",
)

# Fleet operations & maintenance, from the CEB annual summary. Reported fleet-wide only — CEB
# states per-island O&M is not available — so it is surfaced as a separate figure and never
# silently apportioned across islands.
FLEET_OM_RS = {
    "2024": {"repair": 33_252_244.56, "labour": 33_996_235.20, "overtime": 14_729_306.79},
    "2025": {"repair": 44_268_230.83, "labour": 33_429_699.90, "overtime": 14_361_394.33},
}


def _load() -> list[dict[str, Any]]:
    if not GENERATION_CSV.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Calibration table not built at {GENERATION_CSV}. "
                "Run `task data` to generate it from the CEB ledger."
            ),
        )
    rows: list[dict[str, Any]] = []
    with GENERATION_CSV.open() as fh:
        for raw in csv.DictReader(fh):
            row: dict[str, Any] = dict(raw)
            row["month_num"] = int(raw["month_num"])
            for key in NUMERIC:
                value = raw.get(key, "")
                row[key] = float(value) if value not in ("", "None") else None
            rows.append(row)
    return rows


@router.get("/generation")
def generation() -> dict[str, Any]:
    """Every island-month in the ledger — 2 years x 12 months x 5 generating systems."""
    rows = _load()
    return {"rows": rows, "count": len(rows)}


@router.get("/summary")
def summary() -> dict[str, Any]:
    """Per island-year totals, and the fleet roll-up the stat tiles read."""
    rows = _load()

    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        key = (row["year"], row["island_system"])
        for field in ("units_kwh", "diesel_l", "diesel_cost_rs", "oil_cost_rs", "barrel_amount"):
            totals[key][field] += row[field] or 0.0

    by_system = []
    for (year, system), agg in sorted(totals.items()):
        kwh = agg["units_kwh"]
        fuel = agg["diesel_cost_rs"] + agg["oil_cost_rs"] + agg["barrel_amount"]
        by_system.append(
            {
                "year": year,
                "island_system": system,
                "units_kwh": kwh,
                "diesel_l": agg["diesel_l"],
                "diesel_cost_rs": agg["diesel_cost_rs"],
                "oil_cost_rs": agg["oil_cost_rs"],
                "transport_cost_rs": agg["barrel_amount"],
                "sfc_l_per_kwh": agg["diesel_l"] / kwh if kwh else None,
                "fuel_cost_rs_per_kwh": fuel / kwh if kwh else None,
            }
        )

    fleet = []
    for year, om in FLEET_OM_RS.items():
        rowset = [s for s in by_system if s["year"] == year]
        kwh = sum(s["units_kwh"] for s in rowset)
        fuel = sum(
            s["diesel_cost_rs"] + s["oil_cost_rs"] + s["transport_cost_rs"] for s in rowset
        )
        om_total = sum(om.values())
        fleet.append(
            {
                "year": year,
                "units_kwh": kwh,
                "diesel_l": sum(s["diesel_l"] for s in rowset),
                "fuel_cost_rs": fuel,
                "om_cost_rs": om_total,
                "fuel_rs_per_kwh": fuel / kwh if kwh else None,
                "all_in_rs_per_kwh": (fuel + om_total) / kwh if kwh else None,
            }
        )

    return {"by_system": by_system, "fleet": sorted(fleet, key=lambda f: f["year"])}
