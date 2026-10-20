"""The northern-islands network model: one parameter table, two published artifacts.

This is the M1 -> M4 canonical topology (module1.proto ``GridTopology``), drafted for joint
sign-off with Module 4. From the table below it emits

* ``northern_islands_v1.dss`` -- the OpenDSS model M4's twin compiles, and
* ``module1_topology_v1.json`` -- the contract pin: buses, lines, assets, the node map from M1's
  forecaster graph, and the snapshot format,

so the network M4 solves and the network M1 describes cannot disagree. Every number carries its
provenance. ``MEASURED`` values come from the CEB Jaffna ledger, tender documents and the
17 Aug 2026 interview (data/external/ceb_jaffna/README.md). ``ASSUMED`` values are engineering
placeholders, chosen conservatively and stated so they can be replaced when the distribution
control centre's single-line diagrams are released.

What the field data fixes, and what the previous ``delft_3island.dss`` got wrong:

* Four islands, each fully islanded: no inter-island cable, no mainland source.
* Each island's diesel station is its own grid-forming source.
* Distribution is 400 V LT from the power station, through ~100 A drop-out fuse cut-outs; there
  is no substation on any island.
* Diesel-only on Nainativu, Analaitivu and Delft; the only hybrid (PV, wind, battery) is
  Eluvaitivu's 60 kW plant, beside its diesel station.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

TOPOLOGY_ID = "northern-islands-v1"
SCHEMA_VERSION = {"major": 1, "minor": 0}
BASE_KV = 0.4                 # MEASURED: 400 V LT feeders from each power station
POWER_FACTOR = 0.85           # ASSUMED: same value the load downscaling uses
DDLO_AMPS = 100.0             # MEASURED (field sketch): drop-out fuse cut-out per LV feeder
FEEDER_HEADROOM = 1.25        # ASSUMED: feeders per group sized for 125% of constructed peak
FEEDER_EQUIV_KM = 0.4         # ASSUMED: lumped length equivalent to load spread along ~0.8 km
HYBRID_LINK_KM = 0.3          # ASSUMED: hybrid plant to its nearest feeder group
HYBRID_LINK_AMPS = 125.0      # ASSUMED: covers the 72 kW inverter cluster at 400 V
# ASSUMED: 3x95 mm2 aluminium aerial bundled cable, typical per-circuit sequence impedances.
ABC95 = {"r1": 0.320, "x1": 0.080, "r0": 1.280, "x0": 0.320}
SOURCE_MVASC = 50.0           # ASSUMED: stiff source -- the lead genset's AVR holds 1.0 p.u.
FEEDER_SHARES = {"F1": 0.5, "F2": 0.5}  # ASSUMED: split of island demand between feeder groups

MEASURED, ASSUMED, DERIVED = "measured", "assumed", "derived"


@dataclass(frozen=True)
class Asset:
    element: str        # OpenDSS element name (Generator.<element>)
    asset_type: str     # module1.proto AssetType suffix
    rating_kva: float
    provenance: str
    in_service: bool = True
    note: str = ""


@dataclass(frozen=True)
class Island:
    code: str
    name: str                       # name used in M1's data files and forecaster graph
    gensets_kva: tuple[float, ...]  # MEASURED, 2025 fleet; the largest is the grid-forming source
    critical_note: str
    hybrid: tuple[Asset, ...] = field(default=())


ISLANDS: tuple[Island, ...] = (
    Island("NAI", "Nainativu", (380.0, 250.0, 250.0),
           "F1 assumed to carry the island's critical consumers (shared feeder; interview)"),
    Island("DEL", "Delft-Neduntivu", (330.0, 250.0, 250.0),
           "F1 assumed to carry the island hospital and water plant (shared feeder; interview)"),
    Island("ANA", "Analaitivu", (250.0, 100.0, 100.0),
           "F1 assumed to carry the island's critical consumers (shared feeder; interview)"),
    Island("ELU", "Eluvaitivu", (100.0,),
           "F1 assumed to carry the island's critical consumers (shared feeder; interview)",
           hybrid=(
               Asset("ELU_HYB_PV", "PV", 46.26, MEASURED,
                     note="PV array; 46.26 kW per field sketch and interview (tender ~50 kWp)"),
               Asset("ELU_HYB_WIND", "WIND", 21.0, MEASURED, in_service=False,
                     note="6 x 3.5 kW Windspot; all out of service per interview"),
               Asset("ELU_HYB_BESS", "BESS", 72.0, MEASURED,
                     note="12 x 6 kW SMA Sunny Island; ~134 kWh usable LiFePO4; grid-forming "
                          "inside the hybrid plant; bank expired Q4 2025 (tender)"),
               Asset("ELU_HYB_DG", "DIESEL", 30.0, MEASURED,
                     note="30 kW backup genset inside the hybrid plant"),
           )),
)


def _load_parameters(repo_root: Path) -> dict:
    return json.loads((repo_root / "data" / "processed" / "load_parameters.json").read_text())


def feeder_circuits(peak_kw: float, share: float) -> int:
    """Parallel 100 A feeders needed for a feeder group at 125% of its constructed peak."""
    amps = (peak_kw * share / POWER_FACTOR) / (math.sqrt(3) * BASE_KV)
    return max(1, math.ceil(FEEDER_HEADROOM * amps / DDLO_AMPS))


def build_spec(repo_root: Path) -> dict:
    """The full topology as plain data -- the pin, before serialisation."""
    params = _load_parameters(repo_root)
    q_ratio = math.tan(math.acos(POWER_FACTOR))
    buses, lines, assets, loads, node_map = [], [], [], [], {}
    for idx, isl in enumerate(ISLANDS):
        stats = params["islands"][isl.name]
        peak, mean = float(stats["peak_kw"]), float(stats["mean_kw"])
        ps = f"{isl.code}_PS"
        buses.append({"node_id": ps, "island": isl.name, "asset_type": "BUS", "nominal_kv": BASE_KV,
                      "critical_load": False, "role": "power_station",
                      "x_offset": 10.0 * idx, "y_offset": 0.0})
        for g, kva in enumerate(isl.gensets_kva):
            assets.append({
                "element": f"{isl.code}_DG{g + 1}", "bus": ps, "island": isl.name,
                "asset_type": "DIESEL", "rating_kva": kva, "in_service": True,
                "grid_forming": g == 0, "provenance": MEASURED,
                "note": "lead genset, modelled as the island's stiff source (Vsource)" if g == 0
                        else "standby genset, dispatchable (Generator, 0 kW in base case)"})
        for k, (grp, share) in enumerate(FEEDER_SHARES.items()):
            bus = f"{isl.code}_{grp}"
            n = feeder_circuits(peak, share)
            buses.append({"node_id": bus, "island": isl.name, "asset_type": "BUS",
                          "nominal_kv": BASE_KV, "critical_load": grp == "F1",
                          "role": "feeder_group", "x_offset": 10.0 * idx + 1.0 + k,
                          "y_offset": 1.0, "note": isl.critical_note if grp == "F1" else ""})
            lines.append({
                "edge_id": f"{isl.code}_PS_{grp}", "from_node": ps, "to_node": bus,
                "length_km": FEEDER_EQUIV_KM, "ampacity_a": n * DDLO_AMPS, "circuits": n,
                "r1_ohm_km": round(ABC95["r1"] / n, 5), "x1_ohm_km": round(ABC95["x1"] / n, 5),
                "r0_ohm_km": round(ABC95["r0"] / n, 5), "x0_ohm_km": round(ABC95["x0"] / n, 5),
                "provenance": ASSUMED,
                "note": f"{n} parallel 100 A LV feeders lumped (DDLO rating measured; count, "
                        "length and impedance assumed)"})
            p = round(mean * share, 3)
            loads.append({"element": f"{isl.code}_{grp}_LOAD", "bus": bus, "island": isl.name,
                          "base_kw": p, "base_kvar": round(p * q_ratio, 3),
                          "provenance": DERIVED,
                          "note": "base case = island mean of the constructed hourly load"})
        node_map[f"{isl.name}/bus"] = [f"{isl.code}_F1", f"{isl.code}_F2"]
        node_map[f"{isl.name}/diesel"] = [ps]
        if isl.hybrid:
            hyb = f"{isl.code}_HYB"
            buses.append({"node_id": hyb, "island": isl.name, "asset_type": "BUS",
                          "nominal_kv": BASE_KV, "critical_load": False, "role": "hybrid_plant",
                          "x_offset": 10.0 * idx + 1.0, "y_offset": 2.0})
            lines.append({
                "edge_id": f"{isl.code}_HYB_F1", "from_node": hyb, "to_node": f"{isl.code}_F1",
                "length_km": HYBRID_LINK_KM, "ampacity_a": HYBRID_LINK_AMPS, "circuits": 1,
                "r1_ohm_km": ABC95["r1"], "x1_ohm_km": ABC95["x1"],
                "r0_ohm_km": ABC95["r0"], "x0_ohm_km": ABC95["x0"], "provenance": ASSUMED,
                "note": "connection point of the hybrid plant is not documented; assumed to F1"})
            for a in isl.hybrid:
                assets.append({"element": a.element, "bus": hyb, "island": isl.name,
                               "asset_type": a.asset_type, "rating_kva": a.rating_kva,
                               "in_service": a.in_service, "grid_forming": False,
                               "provenance": a.provenance, "note": a.note})
            node_map[f"{isl.name}/pv"] = [hyb]
            node_map[f"{isl.name}/bess"] = [hyb]
    return {
        "topology_id": TOPOLOGY_ID,
        "schema_version": SCHEMA_VERSION,
        "opendss_master_file": "topology/northern_islands_v1.dss",
        "base_kv": BASE_KV,
        "buses": buses, "lines": lines, "assets": assets, "loads": loads,
        "node_map": node_map,
        "assumptions": {
            "power_factor": POWER_FACTOR, "feeder_headroom": FEEDER_HEADROOM,
            "feeder_equivalent_km": FEEDER_EQUIV_KM, "feeder_shares": FEEDER_SHARES,
            "cable_per_circuit_ohm_km": ABC95, "source_mvasc": SOURCE_MVASC,
            "hybrid_link_km": HYBRID_LINK_KM, "hybrid_link_amps": HYBRID_LINK_AMPS,
        },
        "governance": "x_offset / y_offset are arbitrary layout offsets, never coordinates.",
        "snapshot_format": SNAPSHOT_FORMAT,
    }


# GridStateSnapshot, as published by module1.topology.snapshot.
SNAPSHOT_FORMAT = {
    "message": "metacore.module1.v1.GridStateSnapshot",
    "update_period_s": 3600,
    "node_state_fields": {
        "voltage_pu": "p.u. of 400 V; QUALITY_MISSING offline -- M4's power flow computes it",
        "p_kw": "kW demand at a feeder-group bus; QUALITY_INTERPOLATED (downscaled CEB ledger)",
        "q_kvar": "kvar demand at power factor 0.85; QUALITY_INTERPOLATED",
        "soc_fraction": "storage only; QUALITY_MISSING (no battery telemetry exists)",
    },
    "quality_order": ["voltage_pu", "p_kw", "q_kvar", "soc_fraction"],
    "assets_out_of_service": "assets the snapshot asks M4 to disable for this hour",
}


def render_dss(spec: dict) -> str:
    """The OpenDSS master file. Every element traces to a row in ``spec``."""
    out = [
        "Clear",
        "",
        f"! {TOPOLOGY_ID} -- GENERATED by module1.topology.spec; do not hand-edit.",
        "! Four fully islanded 400 V microgrids (Nainativu, Delft-Neduntivu, Analaitivu,",
        "! Eluvaitivu). No inter-island cable and no mainland source exist. Each island's lead",
        "! diesel genset is its grid-forming source. Provenance of every value: the JSON pin",
        "! packages/contracts/python/metacore_contracts/schema/module1_topology_v1.json",
        "",
        "Set DefaultBaseFrequency=50",
        "",
    ]
    lead = {a["bus"]: a for a in spec["assets"] if a["grid_forming"]}
    first = True
    for bus in (b for b in spec["buses"] if b["role"] == "power_station"):
        a = lead[bus["node_id"]]
        src = (f"bus1={bus['node_id']} basekv={BASE_KV} pu=1.0 phases=3 "
               f"MVAsc3={SOURCE_MVASC} MVAsc1={SOURCE_MVASC}")
        out.append(f"! {bus['island']}: lead genset {a['element']} ({a['rating_kva']:.0f} kVA)")
        if first:
            out.append(f"New Circuit.NorthernIslands {src}")
            first = False
        else:
            out.append(f"New Vsource.{a['element']}_SRC {src}")
    out.append("")
    for ln in spec["lines"]:
        out.append(
            f"New Line.{ln['edge_id']} Bus1={ln['from_node']} Bus2={ln['to_node']} phases=3 "
            f"R1={ln['r1_ohm_km']} X1={ln['x1_ohm_km']} R0={ln['r0_ohm_km']} "
            f"X0={ln['x0_ohm_km']} Length={ln['length_km']} Units=km NormAmps={ln['ampacity_a']}")
    out.append("")
    for a in spec["assets"]:
        if a["grid_forming"]:
            continue
        enabled = "" if a["in_service"] else " enabled=no"
        out.append(f"New Generator.{a['element']} Bus1={a['bus']} kV={BASE_KV} kW=0 kvar=0 "
                   f"kVA={a['rating_kva']} Model=1 phases=3{enabled}")
    out.append("")
    for ld in spec["loads"]:
        out.append(f"New Load.{ld['element']} Bus1={ld['bus']} kV={BASE_KV} kW={ld['base_kw']} "
                   f"kvar={ld['base_kvar']} Model=1 phases=3 Vminpu=0.85 Vmaxpu=1.15")
    out += ["", f"Set VoltageBases=[{BASE_KV}]", "CalcVoltageBases",
            "Set ControlMode=Static", "Set MaxIterations=100", "Set Tolerance=0.0001", ""]
    return "\n".join(out)
