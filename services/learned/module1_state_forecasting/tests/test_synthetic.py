"""The fallback has to satisfy the real gates without impersonating the measured record.

Two failure modes are worth guarding against, and they pull in opposite directions. A fallback
that cannot pass the gates leaves them untested in CI, which is the whole reason it exists. A
fallback that reproduces the measured record too closely invites someone to publish a result from
it. These tests pin both edges.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from module1.data import load as load_stage
from module1.data import nasa_power, scenarios, synthetic, validate

PROCESSED = Path(__file__).resolve().parents[4] / "data" / "processed"
REAL_TIDY = PROCESSED / "ceb_generation_tidy.csv"


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One synthetic input set, reused: generating 70,176 weather rows is not free."""
    out = tmp_path_factory.mktemp("synthetic")
    synthetic.build(out)
    return out


# ------------------------------------------------------- it passes the gates

def test_reconciliation_gate_passes_unmodified(built: Path) -> None:
    """The gate CI would otherwise never exercise, since the workbook is not redistributable."""
    assert validate.check(built / "ceb_generation_tidy.csv") == []


def test_meteorology_gate_passes_unmodified(built: Path) -> None:
    assert nasa_power.check(built / "nasa_power") == []


def test_the_whole_downstream_chain_runs(built: Path) -> None:
    """Downscale and scenario extraction, end to end, with no CEB workbook anywhere."""
    rows, manifest = load_stage.downscale(built / "ceb_generation_tidy.csv", built / "nasa_power")
    load_stage.write_csv(rows, built / "island_load_hourly.csv")
    (built / "load_parameters.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tidy = built / "ceb_generation_tidy.csv"
    assert load_stage.check(built / "island_load_hourly.csv", tidy) == []

    events, library = scenarios.build(built / "ceb_generation_tidy.csv")
    scenarios.write_events_csv(events, built / "events.csv")
    (built / "scenario_library.json").write_text(json.dumps(library, indent=2) + "\n")
    assert scenarios.check(built, built / "ceb_generation_tidy.csv") == []


def test_the_degradation_survives_into_the_fallback(built: Path) -> None:
    """Dropping it would turn the scenario gate green while removing the only thing it checks."""
    events, _ = scenarios.build(built / "ceb_generation_tidy.csv")
    ood = [e for e in events if e["out_of_distribution"]]
    assert [e["plant"] for e in ood] == [synthetic.DECAY_PLANT]


# --------------------------------------------------- it is not the real thing

def test_everything_is_labelled_synthetic(built: Path) -> None:
    provenance = json.loads((built / "PROVENANCE.json").read_text())
    manifest = json.loads((built / "nasa_power" / "manifest.json").read_text())
    assert provenance["synthetic"] is True
    assert manifest["synthetic"] is True
    assert "not NASA POWER" in manifest["source"]
    assert "Do not publish" in provenance["warning"]


def test_generation_is_deterministic() -> None:
    """Two members generating the fallback must get identical files, or CI is not reproducible."""
    first = synthetic.build_ledger(seed=synthetic.SEED)
    assert first == synthetic.build_ledger(seed=synthetic.SEED)


@pytest.mark.skipif(not REAL_TIDY.exists(), reason="measured ledger absent")
def test_monthly_values_do_not_match_the_measured_ledger(built: Path) -> None:
    """Annual totals are the real transcribed PDF figures; nothing below the year may be.

    If the monthly rows ever coincided with the workbook, the fallback would have stopped being a
    stand-in and started being a redistribution of data that is not ours to redistribute.
    """
    import csv

    def monthly(path: Path) -> dict[tuple[str, str, str], float]:
        with path.open() as fh:
            return {
                (r["year"], r["month"], r["island_system"]): float(r["units_kwh"])
                for r in csv.DictReader(fh)
            }

    real, fake = monthly(REAL_TIDY), monthly(built / "ceb_generation_tidy.csv")
    assert set(real) == set(fake), "the fallback must cover the same island-months"

    identical = [k for k in real if abs(real[k] - fake[k]) < 1.0]
    assert len(identical) < 0.05 * len(real), (
        f"{len(identical)}/{len(real)} monthly values coincide with the measured ledger"
    )


@pytest.mark.skipif(not REAL_TIDY.exists(), reason="measured ledger absent")
def test_annual_totals_do_match_because_they_are_the_real_figures(built: Path) -> None:
    """The deliberate exception, and the reason the reconciliation gate can run against this."""
    assert validate.check(built / "ceb_generation_tidy.csv") == []
    assert validate.check(REAL_TIDY) == []


def test_the_resolution_finding_is_not_faked(built: Path) -> None:
    """The real pull yields one irradiance series across four islands. That is a finding about
    NASA POWER, not a property the fallback should manufacture -- and a synthetic set that
    reproduced it would let someone cite the finding from generated data."""
    import csv

    series = {}
    for site in nasa_power.ISLAND_SITES:
        with (built / "nasa_power" / f"{site.key}_hourly.csv").open() as fh:
            series[site.key] = [r["ghi_wh_m2"] for r in csv.DictReader(fh)]

    distinct = {tuple(v) for v in series.values()}
    assert len(distinct) == len(nasa_power.ISLAND_SITES)
