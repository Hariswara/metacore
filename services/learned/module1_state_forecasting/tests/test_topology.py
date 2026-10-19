"""The M1 -> M4 topology pin and snapshot: the rules a reviewer would otherwise check by hand.

Physics (does it converge, are voltages legal) is tested in Module 4, which owns OpenDSS. These
tests hold the data to the field record: islands stay islanded, genset ratings are the CEB fleet,
every M1 forecaster node maps onto the network, and a snapshot never invents a measurement.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from module1.assemble import ISLAND_ASSETS
from module1.topology import __main__ as cli
from module1.topology.snapshot import build_snapshot
from module1.topology.spec import DDLO_AMPS, POWER_FACTOR, build_spec

REPO = Path(__file__).resolve().parents[4]
PROCESSED = REPO / "data" / "processed"
needs_artifacts = pytest.mark.skipif(
    not (PROCESSED / "load_parameters.json").exists(),
    reason="calibration artifacts absent -- run `task data`",
)


@pytest.fixture(scope="module")
def spec() -> dict:
    return build_spec(REPO)


@needs_artifacts
def test_committed_artifacts_are_regenerated_exactly() -> None:
    """The .dss, the pin and the fixtures are generated; a hand edit must fail CI."""
    for path, text in cli.artifacts().items():
        assert path.read_text() == text, f"{path.name} is stale; run `module1.topology build`"


@needs_artifacts
def test_no_line_crosses_between_islands(spec: dict) -> None:
    island = {b["node_id"]: b["island"] for b in spec["buses"]}
    for ln in spec["lines"]:
        assert island[ln["from_node"]] == island[ln["to_node"]], ln["edge_id"]


@needs_artifacts
def test_each_island_has_exactly_one_grid_forming_source(spec: dict) -> None:
    islands = {b["island"] for b in spec["buses"]}
    for isl in islands:
        forming = [a for a in spec["assets"] if a["island"] == isl and a["grid_forming"]]
        assert len(forming) == 1, isl


@needs_artifacts
def test_installed_capacity_is_the_ceb_2025_fleet(spec: dict) -> None:
    params = json.loads((PROCESSED / "load_parameters.json").read_text())
    for isl in {b["island"] for b in spec["buses"]}:
        kva = sum(a["rating_kva"] for a in spec["assets"]
                  if a["island"] == isl and a["asset_type"] == "DIESEL")
        assert kva == params["measured_inputs"]["installed_kva"][f"2025/{isl}"], isl


@needs_artifacts
def test_feeders_carry_the_constructed_peak_with_headroom(spec: dict) -> None:
    params = json.loads((PROCESSED / "load_parameters.json").read_text())
    island = {b["node_id"]: b["island"] for b in spec["buses"]}
    for ln in spec["lines"]:
        if "_PS_" not in ln["edge_id"]:
            continue
        peak = params["islands"][island[ln["to_node"]]]["peak_kw"] * 0.5
        amps = peak / POWER_FACTOR / (math.sqrt(3) * spec["base_kv"])
        assert ln["ampacity_a"] >= amps, ln["edge_id"]
        assert ln["ampacity_a"] == ln["circuits"] * DDLO_AMPS


@needs_artifacts
def test_every_forecaster_node_maps_onto_existing_buses(spec: dict) -> None:
    buses = {b["node_id"] for b in spec["buses"]}
    forecaster = {f"{isl}/{a}" for isl, assets in ISLAND_ASSETS.items() for a in assets}
    assert set(spec["node_map"]) == forecaster
    for target in spec["node_map"].values():
        assert target and set(target) <= buses


@needs_artifacts
def test_every_value_states_its_provenance(spec: dict) -> None:
    for row in spec["lines"] + spec["assets"] + spec["loads"]:
        assert row["provenance"] in {"measured", "assumed", "derived"}


# ------------------------------------------------------------ snapshot

@needs_artifacts
def test_snapshot_demand_reproduces_the_island_load_hour() -> None:
    snap = build_snapshot("2025-07-20T20:00", REPO)
    nai = sum(n["p_kw"] for n in snap["node_states"] if n["node_id"].startswith("NAI_F"))
    assert nai == pytest.approx(391.0, abs=0.01)  # the calibrated Nainativu peak


@needs_artifacts
def test_snapshot_never_invents_voltage_or_state_of_charge() -> None:
    snap = build_snapshot("2025-11-28T19:00", REPO)
    for n in snap["node_states"]:
        assert n["voltage_pu"] is None and n["quality"]["voltage_pu"] == "QUALITY_MISSING"
        assert n["soc_fraction"] is None and n["quality"]["soc_fraction"] == "QUALITY_MISSING"
        if n["p_kw"] is not None:
            assert n["quality"]["p_kw"] == "QUALITY_INTERPOLATED"


@needs_artifacts
@pytest.mark.parametrize("stamp, down", [
    ("2025-08-15T12:00", False), ("2025-10-15T12:00", False),  # Oct is onset (0.70), not core
    ("2025-11-15T12:00", True), ("2025-12-15T12:00", True),
])
def test_hybrid_is_out_of_service_only_in_core_decay_months(stamp: str, down: bool) -> None:
    snap = build_snapshot(stamp, REPO)
    assert snap["hybrid_plant_down"] is down
    assert ("ELU_HYB_BESS" in snap["assets_out_of_service"]) is down
    assert "ELU_HYB_WIND" in snap["assets_out_of_service"]  # out of service all record


@needs_artifacts
def test_snapshot_refuses_an_hour_outside_the_record() -> None:
    with pytest.raises(ValueError, match="not an hour"):
        build_snapshot("2030-01-01T00:00", REPO)
