"""The v1 pin is what lets M2 size its evidential head without a mock.

These are contract tests, not unit tests: each one fails on a change that would silently break a
consumer. A rename that only breaks at runtime in someone else's service is exactly what the pin
exists to prevent.
"""

from __future__ import annotations

import pytest
from metacore_contracts import common_pb2, module1_pb2
from metacore_contracts.state_schema import (
    EMBEDDING_DIM,
    FEATURE_COUNT,
    FEATURE_NAMES,
    SCHEMA_VERSION,
    calibration_quality,
    feature_index,
    load,
)

VALID_QUALITY = {q.name for q in common_pb2.Quality.DESCRIPTOR.values}


def test_pinned_values_are_concrete() -> None:
    assert EMBEDDING_DIM == 64
    assert FEATURE_COUNT == 28
    assert SCHEMA_VERSION == (1, 0)


def test_feature_names_are_unique_and_ordered() -> None:
    """node_features is row-major over this order, so a duplicate silently aliases a column."""
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert FEATURE_NAMES == tuple(load()["feature_names"])


def test_every_feature_is_documented() -> None:
    """A feature with no unit or source is a feature a consumer has to guess at."""
    documented = load()["features"]
    assert set(FEATURE_NAMES) == set(documented)
    for name, meta in documented.items():
        assert meta["unit"], f"{name} has no unit"
        assert meta["source"], f"{name} has no source"
        assert meta["group"], f"{name} has no group"


def test_calibration_quality_values_exist_in_the_proto_enum() -> None:
    """The pin must not invent a quality level the QualityMask cannot carry."""
    for name in FEATURE_NAMES:
        assert calibration_quality(name) in VALID_QUALITY


def test_only_deterministic_features_claim_to_be_observed() -> None:
    """ADR 0004: anything not measured carries a mask that is not QUALITY_OBSERVED.

    Nothing derived from the CEB ledger or the NASA POWER reanalysis may claim observation at node
    level -- the ledger is monthly and the reanalysis does not resolve the islands. Only the clock
    and the single-line diagram do.
    """
    groups = load()["features"]
    observed = {n for n in FEATURE_NAMES if calibration_quality(n) == "QUALITY_OBSERVED"}
    deterministic = {"temporal", "topology"}
    offenders = {n: groups[n]["group"] for n in observed if groups[n]["group"] not in deterministic}
    assert not offenders, f"non-deterministic features claiming QUALITY_OBSERVED: {offenders}"


def test_irradiance_is_not_claimed_as_node_level_truth() -> None:
    """One distinct irradiance series covers all four islands -- docs/data/nasa-power-resolution.md.

    Attributing a ~111 km cell to a specific node is an interpolation, and labelling it otherwise
    would hand M2 a confident-looking feature with no inter-island information in it.
    """
    for name in ("ghi_wh_m2_norm", "clearsky_index"):
        assert calibration_quality(name) == "QUALITY_INTERPOLATED"
        note = load()["features"][name]["note"].lower()
        assert "degenerate" in note or "spatial" in note


def test_feature_index_is_stable_and_reports_typos() -> None:
    assert feature_index(FEATURE_NAMES[0]) == 0
    assert feature_index("load_kw_norm") == FEATURE_NAMES.index("load_kw_norm")
    with pytest.raises(KeyError):
        feature_index("load_kw")  # a plausible near-miss


def test_asset_type_one_hot_covers_every_enum_value_except_unspecified() -> None:
    """An unspecified node is all-zero across the is_* columns, not silently folded into a type."""
    enum_types = {
        v.name.removeprefix("ASSET_TYPE_").lower()
        for v in module1_pb2.AssetType.DESCRIPTOR.values
        if v.name != "ASSET_TYPE_UNSPECIFIED"
    }
    one_hot = {n.removeprefix("is_") for n in FEATURE_NAMES if n.startswith("is_")}
    assert one_hot == enum_types


def test_the_mask_is_parallel_to_the_feature_vector() -> None:
    """QualityMask.per_feature is positional; a length mismatch misattributes every flag."""
    mask = common_pb2.QualityMask(per_feature=[common_pb2.QUALITY_OBSERVED] * FEATURE_COUNT)
    state = module1_pb2.StateRepresentation(
        node_count=3,
        embedding_dim=EMBEDDING_DIM,
        feature_names=list(FEATURE_NAMES),
        node_features=[0.0] * (3 * FEATURE_COUNT),
        node_embedding=[0.0] * (3 * EMBEDDING_DIM),
        quality=mask,
    )
    assert len(state.quality.per_feature) == len(state.feature_names)
    assert len(state.node_features) == state.node_count * len(state.feature_names)
    assert len(state.node_embedding) == state.node_count * state.embedding_dim
