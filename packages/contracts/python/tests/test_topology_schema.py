"""The draft M1 -> M4 pin must fit the proto it pins.

The JSON pin and snapshot fixtures are what M4 builds against today; the proto is what the
streaming path will carry. These tests convert one into the other field by field, so a field
added to one and not the other fails here instead of in someone else's service.
"""

from __future__ import annotations

import pytest
from metacore_contracts import common_pb2, module1_pb2
from metacore_contracts.topology_schema import (
    DSS_PATH,
    TOPOLOGY_ID,
    bus_ids,
    load,
    node_map,
    sample_snapshot,
)

QUALITY = {q: common_pb2.Quality.Value(q) for q in common_pb2.Quality.keys()}


def to_topology(pin: dict) -> module1_pb2.GridTopology:
    t = module1_pb2.GridTopology(topology_id=pin["topology_id"],
                                 opendss_master_file=pin["opendss_master_file"])
    for b in pin["buses"]:
        t.nodes.add(node_id=b["node_id"],
                    asset_type=module1_pb2.AssetType.Value(f"ASSET_TYPE_{b['asset_type']}"),
                    nominal_kv=b["nominal_kv"], critical_load=b["critical_load"],
                    x_offset=b["x_offset"], y_offset=b["y_offset"])
    for ln in pin["lines"]:
        t.edges.add(edge_id=ln["edge_id"], from_node=ln["from_node"], to_node=ln["to_node"],
                    length_km=ln["length_km"], ampacity_a=ln["ampacity_a"])
    for a in pin["assets"]:
        t.assets.add(element=a["element"], bus=a["bus"],
                     asset_type=module1_pb2.AssetType.Value(f"ASSET_TYPE_{a['asset_type']}"),
                     rating_kva=a["rating_kva"], in_service=a["in_service"],
                     grid_forming=a["grid_forming"])
    return t


def to_snapshot(snap: dict, order: list[str]) -> module1_pb2.GridStateSnapshot:
    s = module1_pb2.GridStateSnapshot(topology_id=snap["topology_id"],
                                      update_period_s=snap["update_period_s"],
                                      assets_out_of_service=snap["assets_out_of_service"])
    for ns in snap["node_states"]:
        n = s.node_states.add(node_id=ns["node_id"])
        for f in order:
            if ns[f] is not None:
                setattr(n, f, ns[f])
        n.quality.per_feature.extend(QUALITY[ns["quality"][f]] for f in order)
    return s


def test_pin_converts_to_grid_topology_without_loss() -> None:
    pin = load()
    t = to_topology(pin)
    again = module1_pb2.GridTopology.FromString(t.SerializeToString())
    assert again.topology_id == TOPOLOGY_ID
    assert [n.node_id for n in again.nodes] == bus_ids()
    assert len(again.edges) == len(pin["lines"]) and len(again.assets) == len(pin["assets"])


@pytest.mark.parametrize("name", ["calm", "nainativu_peak", "ditwah_evening"])
def test_fixture_snapshots_convert_to_grid_state_snapshot(name: str) -> None:
    order = load()["snapshot_format"]["quality_order"]
    snap = sample_snapshot(name)
    s = module1_pb2.GridStateSnapshot.FromString(to_snapshot(snap, order).SerializeToString())
    assert s.topology_id == TOPOLOGY_ID
    assert {n.node_id for n in s.node_states} <= set(bus_ids())
    for n in s.node_states:
        assert len(n.quality.per_feature) == len(order)
        assert n.quality.per_feature[0] == common_pb2.QUALITY_MISSING  # voltage: M4 computes it


def test_node_map_targets_exist_and_dss_model_is_shipped() -> None:
    assert DSS_PATH.exists()
    buses = set(bus_ids())
    assert all(set(v) <= buses for v in node_map().values())
