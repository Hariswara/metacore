"""The scenario library is a label set other people's results depend on.

M2's out-of-distribution evaluation selects episodes on `out_of_distribution`. If the window moves
silently, their AUROC moves with it and nothing says so. These tests pin the rule, not the answer:
each one would fail on a change to how the window is derived, which is the thing that must not
drift unnoticed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from module1.data import scenarios

PROCESSED = Path(__file__).resolve().parents[4] / "data" / "processed"
TIDY_CSV = PROCESSED / "ceb_generation_tidy.csv"
EVENTS_CSV = PROCESSED / "events.csv"

needs_artifacts = pytest.mark.skipif(
    not (TIDY_CSV.exists() and EVENTS_CSV.exists()),
    reason="calibration artifacts absent -- run `task data` or `dvc pull`",
)


# ------------------------------------------------------------ pure rule

def test_median_baseline_is_not_dragged_down_by_the_event() -> None:
    """The reason for a median: the collapse is inside the record being baselined against."""
    healthy = {(2024, m): 8000.0 for m in range(1, 13)}
    collapsed = {(2025, m): 8000.0 for m in range(1, 10)}
    collapsed.update({(2025, 10): 5600.0, (2025, 11): 1100.0, (2025, 12): 260.0})

    baseline, _ = scenarios.ratios({**healthy, **collapsed})
    assert baseline == 8000.0  # a mean would land near 7,300 and shrink the anomaly


def test_a_flat_plant_raises_no_event() -> None:
    """Uniform application means the rule must stay silent on the four healthy plants."""
    flat = {(2024, m): 8000.0 for m in range(1, 13)}
    assert scenarios.find_degradation("Nainativu", flat) is None


def test_onset_walk_back_opens_the_window_before_the_collapse() -> None:
    """A window that opens at the collapse hides the part of the problem worth detecting."""
    months = {(2025, m): 8000.0 for m in range(1, 10)}
    months.update({(2025, 10): 5600.0, (2025, 11): 1100.0, (2025, 12): 260.0})

    event = scenarios.find_degradation("Eluvaitivu-Hybrid", months)
    assert event is not None
    # 2025-10 is 0.70 of baseline: above the 0.50 core threshold, below the 0.90 onset threshold.
    assert event["start_month"] == "2025-10"
    assert event["end_month"] == "2025-12"


def test_onset_walk_back_stops_at_a_healthy_month() -> None:
    """It must not run backwards through a normal month and swallow the whole record."""
    months = {(2025, m): 8000.0 for m in range(1, 11)}
    months.update({(2025, 11): 1100.0, (2025, 12): 260.0})

    event = scenarios.find_degradation("Eluvaitivu-Hybrid", months)
    assert event is not None
    assert event["start_month"] == "2025-11"


def test_nominal_windows_exclude_months_belonging_to_an_event() -> None:
    """"Everything not flagged" would quietly include the months either side of a collapse."""
    plants = {"P": {(2025, m): 8000.0 for m in range(1, 13)}}
    excluded = {(2025, 5), (2025, 6), (2025, 7)}

    windows = scenarios.find_nominal("Testland", plants, excluded)
    covered = {m["month"] for w in windows for m in w["monthly"]}
    assert not covered & {"2025-05", "2025-06", "2025-07"}


def test_every_plant_in_the_ledger_maps_to_an_island() -> None:
    """An unmapped plant would be silently dropped from the library instead of failing."""
    assert set(scenarios.SYSTEM_TO_ISLAND) == {
        "Analaithivu",
        "Eluvaitivu-Diesel",
        "Eluvaitivu-Hybrid",
        "Delft-Neduntivu",
        "Nainativu",
    }


# --------------------------------------------------------- built library

@needs_artifacts
def test_gate_passes_on_the_committed_library() -> None:
    assert scenarios.check(PROCESSED, TIDY_CSV) == []


@needs_artifacts
def test_the_rule_flags_exactly_one_plant_on_the_real_ledger() -> None:
    """Uniform application over all five plants; a second hit would mean the rule is too loose."""
    events, _ = scenarios.build(TIDY_CSV)
    ood = [e for e in events if e["out_of_distribution"]]
    assert [e["plant"] for e in ood] == ["Eluvaitivu-Hybrid"]


@needs_artifacts
def test_the_hybrid_decay_is_labelled_where_m2_looks_for_it() -> None:
    with EVENTS_CSV.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    event = next(r for r in rows if r["plant"] == "Eluvaitivu-Hybrid")
    assert event["out_of_distribution"] == "true"
    assert (event["start_month"], event["end_month"]) == ("2025-10", "2025-12")
    assert event["island"] == "Eluvaitivu"
    assert float(event["worst_ratio"]) < 0.05  # December is 3% of baseline


@needs_artifacts
def test_scenario_ref_fields_are_present_for_every_scenario() -> None:
    """These three map field-for-field onto metacore.common.v1.ScenarioRef."""
    library = json.loads((PROCESSED / "scenario_library.json").read_text())
    assert library["library_version"] == scenarios.LIBRARY_VERSION
    for scenario in library["scenarios"]:
        assert scenario["scenario_id"]
        assert isinstance(scenario["out_of_distribution"], bool)


@needs_artifacts
def test_both_distribution_classes_are_populated() -> None:
    """An OOD-only library cannot score a detector, and an ID-only one cannot test it."""
    library = json.loads((PROCESSED / "scenario_library.json").read_text())
    counts = library["counts"]
    assert counts["out_of_distribution"] >= 1
    assert counts["in_distribution"] >= 1


@needs_artifacts
def test_summing_the_island_attenuates_the_event_sevenfold() -> None:
    """The stage's whole reason for existing, as a number rather than as prose.

    Measured across 2025 Q4 against the preceding quarter: the hybrid plant falls 73.4%, island
    demand falls 10.3%. A detector watching only the island series sees a seasonal-looking dip.
    Asserted as a ratio, so the test states the masking rather than a threshold pulled to fit.
    """
    series = scenarios.read_plant_months(TIDY_CSV)
    window = [(2025, 10), (2025, 11), (2025, 12)]
    before = [(2025, 7), (2025, 8), (2025, 9)]

    def mean(plant_keys: list[tuple[int, int]], *plants: str) -> float:
        return sum(sum(series[p][k] for p in plants) for k in plant_keys) / len(plant_keys)

    island_drop = 1 - mean(window, "Eluvaitivu-Diesel", "Eluvaitivu-Hybrid") / mean(
        before, "Eluvaitivu-Diesel", "Eluvaitivu-Hybrid"
    )
    plant_drop = 1 - mean(window, "Eluvaitivu-Hybrid") / mean(before, "Eluvaitivu-Hybrid")

    assert plant_drop > 0.70
    assert island_drop < 0.15
    assert plant_drop / island_drop > 5.0
