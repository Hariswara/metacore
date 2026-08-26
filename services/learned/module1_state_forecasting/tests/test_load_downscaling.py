"""Load downscaling stage tests.

Two kinds. The pure-function tests pin the properties the method depends on being true and run
anywhere. The artifact tests re-check the produced table and skip on a clean clone, matching the
NASA POWER tests -- the CEB workbook is git-ignored, so CI without it must stay green.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pytest
from module1.data import load as load_stage

REPO_ROOT = Path(__file__).resolve().parents[4]
PROCESSED = REPO_ROOT / "data" / "processed"
LOAD_CSV = PROCESSED / "island_load_hourly.csv"
TIDY_CSV = PROCESSED / "ceb_generation_tidy.csv"

needs_artifacts = pytest.mark.skipif(
    not (LOAD_CSV.exists() and TIDY_CSV.exists()),
    reason="calibration artifacts absent; run `task data`",
)


# ------------------------------------------------------------- pure functions

def test_diurnal_profile_is_a_shape_not_a_level() -> None:
    """Only ratios are used, so the vector's absolute scale must never matter. Doubling it must
    leave the normalised series identical."""
    stamp = datetime(2025, 7, 15, 20)
    a = load_stage._shape(stamp, 28.0, 27.0, 1.0)
    b = load_stage._shape(stamp, 28.0, 27.0, 1.0)
    assert a == b > 0


def test_evening_peak_dominates_overnight_trough() -> None:
    """The one qualitative claim the profile makes about island load. If DIURNAL is ever edited
    into something flat, the peak calibration would silently absorb it."""
    assert max(load_stage.DIURNAL) == load_stage.DIURNAL[20]
    assert min(load_stage.DIURNAL) == load_stage.DIURNAL[3]
    assert max(load_stage.DIURNAL) / min(load_stage.DIURNAL) > 3.0


def test_peakiness_exponent_is_monotone_in_peak() -> None:
    """Calibration bisects on this. A non-monotone response would make the solve meaningless."""
    stamp_peak = datetime(2025, 7, 15, 20)
    stamp_trough = datetime(2025, 7, 15, 3)
    ratios = []
    for exponent in (0.5, 1.0, 2.0):
        peak = load_stage._shape(stamp_peak, 28.0, 28.0, exponent)
        trough = load_stage._shape(stamp_trough, 28.0, 28.0, exponent)
        ratios.append(peak / trough)
    assert ratios == sorted(ratios)


def test_temperature_term_cannot_go_negative() -> None:
    """A reanalysis outlier must not be able to produce a negative load."""
    stamp = datetime(2025, 7, 15, 20)
    assert load_stage._shape(stamp, -200.0, 28.0, 1.0) > 0


def test_leap_year_february_is_29_days() -> None:
    assert load_stage.days_in_month("2024", 2) == 29
    assert load_stage.days_in_month("2025", 2) == 28


def test_eluvaitivu_plants_map_to_one_island_load() -> None:
    """Two metered plants, one physical load. The Oct-Dec 2025 collapse is only legible as a
    substitution because they are summed."""
    assert load_stage.SYSTEM_TO_ISLAND["Eluvaitivu-Diesel"] == "Eluvaitivu"
    assert load_stage.SYSTEM_TO_ISLAND["Eluvaitivu-Hybrid"] == "Eluvaitivu"


def test_installed_capacity_is_year_specific() -> None:
    """The 2024 fleet is larger. A single-year table would hide a capacity violation."""
    assert load_stage.INSTALLED_KVA[("2024", "Nainativu")] > \
        load_stage.INSTALLED_KVA[("2025", "Nainativu")]


def test_anchor_matches_the_documented_measurement() -> None:
    """460 kVA against 880 kVA installed, from the interview and PDF. The only shape observation
    in the dataset -- if it drifts, the whole calibration is unmoored."""
    assert load_stage.REPORTED_MAX_DEMAND_KVA["Nainativu"] == 460.0
    assert load_stage.INSTALLED_KVA[("2025", "Nainativu")] == 880.0


# ----------------------------------------------------------------- artifacts

@needs_artifacts
def test_gate_passes_on_the_committed_artifacts() -> None:
    assert load_stage.check(LOAD_CSV, TIDY_CSV) == []


@needs_artifacts
def test_every_row_is_marked_interpolated() -> None:
    """ADR 0004: downscaled load is never QUALITY_OBSERVED. M2's contribution depends on it."""
    with LOAD_CSV.open(newline="") as fh:
        qualities = {row["quality"] for row in csv.DictReader(fh)}
    assert qualities == {"QUALITY_INTERPOLATED"}


@needs_artifacts
def test_monthly_energy_is_conserved_exactly() -> None:
    """The measured constraint. Enforced independently of the gate so a bug in `check` cannot
    hide a broken energy balance."""
    ledger = load_stage.read_monthly_island_energy(TIDY_CSV)
    totals: dict[tuple[str, int, str], float] = {}
    with LOAD_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            stamp = datetime.fromisoformat(row["timestamp_lst"])
            key = (str(stamp.year), stamp.month, row["island"])
            totals[key] = totals.get(key, 0.0) + float(row["load_kw"])
    assert set(totals) == set(ledger)
    for key, expected in ledger.items():
        assert abs(totals[key] - expected) < load_stage.TOLERANCE_KWH


@needs_artifacts
def test_manifest_separates_measured_from_assumed() -> None:
    """The artifact's whole claim to honesty is this split being explicit and machine-readable."""
    manifest = json.loads((PROCESSED / "load_parameters.json").read_text())
    assert set(manifest["measured_inputs"]) >= {"monthly_energy", "max_demand", "installed_kva"}
    assert set(manifest["assumptions"]) >= {"diurnal_profile", "temp_beta_per_c", "power_factor"}
    assert manifest["quality"] == "QUALITY_INTERPOLATED"


@needs_artifacts
def test_willans_flags_the_intermittently_dispatched_plant() -> None:
    """Eluvaitivu's diesel set backs up the hybrid rather than running continuously, and the fit
    says so with a negative no-load rate. Flagged, never clamped -- it is a finding."""
    fits = json.loads((PROCESSED / "load_parameters.json").read_text())["willans_fuel_model"]
    assert fits["Eluvaitivu-Diesel"]["continuous_running_consistent"] is False
    assert fits["Nainativu"]["continuous_running_consistent"] is True
    assert fits["Nainativu"]["r2"] > 0.9


@needs_artifacts
def test_manifest_identifies_itself() -> None:
    """M2 reads this sidecar to interpret the CSV; unversioned, it has to guess."""
    manifest = json.loads((PROCESSED / "load_parameters.json").read_text())
    assert manifest["artifact"] == load_stage.ARTIFACT_NAME
    assert manifest["version"] == load_stage.ARTIFACT_VERSION
    assert manifest["produced_by"] == "module1.data.load"


@needs_artifacts
def test_version_gate_rejects_a_stale_manifest(tmp_path: Path) -> None:
    """A manifest from an older shape must fail the gate rather than be read as current."""
    stale = json.loads((PROCESSED / "load_parameters.json").read_text())
    stale["version"] = "0.9.0"

    (tmp_path / "island_load_hourly.csv").write_bytes(LOAD_CSV.read_bytes())
    (tmp_path / "load_parameters.json").write_text(json.dumps(stale))

    failures = load_stage.check(tmp_path / "island_load_hourly.csv", TIDY_CSV)
    assert any("version" in f and "0.9.0" in f for f in failures), failures


@needs_artifacts
def test_missing_manifest_fails_the_gate(tmp_path: Path) -> None:
    """The sidecar carries every assumption; a load table without one is not interpretable."""
    (tmp_path / "island_load_hourly.csv").write_bytes(LOAD_CSV.read_bytes())

    failures = load_stage.check(tmp_path / "island_load_hourly.csv", TIDY_CSV)
    assert any("load_parameters.json" in f and "absent" in f for f in failures), failures
