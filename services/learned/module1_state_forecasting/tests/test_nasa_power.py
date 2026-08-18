"""NASA POWER stage tests.

The pure functions run everywhere. The data assertions skip when the pull has not been run, so a
clean clone stays green without a network round trip.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from module1.data import nasa_power as np_stage

REPO_ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = REPO_ROOT / "data" / "raw" / "nasa_power"

needs_pull = pytest.mark.skipif(
    not (RAW_DIR / "manifest.json").exists(),
    reason="NASA POWER not pulled; run `task data:pull`",
)


def test_grid_cell_arithmetic_matches_merra2_steps() -> None:
    site = np_stage.Site("probe", 9.760, 79.770)
    assert site.grid_cell == (10.0, 80.0)
    assert np_stage.Site("probe", 9.520, 79.690).grid_cell == (9.5, 80.0)


def test_islands_do_not_resolve_independently() -> None:
    """Four sites, fewer cells. If this ever stops being true the resolution caveat in the
    docs is stale and should be revisited, so fail loudly rather than drift."""
    cells = np_stage.grid_aliasing()
    assert len(cells) < len(np_stage.ISLAND_SITES)
    assert sorted(cells["9.5,80.0"]) == ["Analaitivu", "Delft-Neduntivu", "Nainativu"]


def test_measure_distinct_series_groups_identical_sites() -> None:
    rows = {
        "a": [{"ghi_wh_m2": "1", "wind_10m_ms": "5"}],
        "b": [{"ghi_wh_m2": "1", "wind_10m_ms": "5"}],
        "c": [{"ghi_wh_m2": "1", "wind_10m_ms": "9"}],
    }
    measured = np_stage.measure_distinct_series(rows)
    assert measured["ghi_wh_m2"] == [["a", "b", "c"]]
    assert measured["wind_10m_ms"] == [["a", "b"], ["c"]]


def test_fill_values_never_become_zero() -> None:
    """-999 must read as unknown. Zero irradiance is a physically meaningful value, so a fill
    value silently coerced to 0.0 would be indistinguishable from a real moonless midnight."""
    payload = {
        "properties": {
            "parameter": {
                name: {"2024010100": np_stage.FILL_VALUE} for name in np_stage.PARAMETERS
            }
        }
    }
    rows = np_stage._to_rows(np_stage.Site("probe", 9.5, 79.7), payload)
    assert rows[0]["ghi_wh_m2"] is None
    assert rows[0]["wind_10m_ms"] is None


@needs_pull
def test_every_site_has_two_full_years_of_hours() -> None:
    for site in np_stage.ISLAND_SITES:
        with (RAW_DIR / f"{site.key}_hourly.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 17_544, f"{site.key}: 2024 is a leap year — 731 days x 24 hours"


@needs_pull
def test_pulled_data_passes_its_own_gate() -> None:
    assert np_stage.check(RAW_DIR) == []


@needs_pull
def test_manifest_records_the_measured_resolution_limit() -> None:
    manifest = json.loads((RAW_DIR / "manifest.json").read_text())
    measured = manifest["grid"]["measured_distinct_series"]
    # Solar is on the coarser CERES grid and does not vary across the study area at all.
    assert measured["ghi_wh_m2"]["count"] == 1
    assert measured["wind_10m_ms"]["count"] == 2
