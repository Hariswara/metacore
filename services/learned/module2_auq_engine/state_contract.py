"""Dataclass mirror of the M1 -> M2 runtime contract.

Mirrors `metacore.module1.v1.StateRepresentation` and the `metacore.common.v1`
messages it carries (`Envelope`, `SchemaVersion`, `ScenarioRef`, `QualityMask`,
`Quality`), so Module 2 is built against the real contract shape before M1's
producer exists. Swap `synthetic_data.sample_states_*` for M1's real producer
later and nothing downstream of `edl_matrix` / `stack_features` changes.

Field names, order and nesting follow `packages/contracts/proto/module1.proto`
and `packages/contracts/proto/common.proto` exactly. Two notes on the mapping:

* `StateRepresentation` carries no top-level `schema_version` or `timestamp`.
  Both live on `Envelope` -- as `SchemaVersion{major, minor}` (a message, not a
  string) and `emitted_at` (a `google.protobuf.Timestamp`, represented here as
  POSIX seconds).
* The proto flattens `node_embedding` and `node_features` into `repeated float`
  in row-major order. They are kept 2-D here because that is the shape the EDL
  head consumes; `node_count` and `embedding_dim` are the row/column counts a
  real decoder would reshape by, and `__post_init__` enforces the agreement.

We deliberately do NOT import the generated `metacore_contracts.module1_pb2`
yet: its stubs emit a flat `import common_pb2`, so package-qualified import
raises `ModuleNotFoundError`. Tracked in the M1 review; switch over once fixed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# common.proto: enum Quality. Spelled as strings so the mock does not depend on
# the generated enum ints; the names are the proto's, verbatim.
QUALITY_UNSPECIFIED = "QUALITY_UNSPECIFIED"
QUALITY_OBSERVED = "QUALITY_OBSERVED"
QUALITY_INTERPOLATED = "QUALITY_INTERPOLATED"
QUALITY_MISSING = "QUALITY_MISSING"
QUALITY_STALE = "QUALITY_STALE"

QUALITY_VALUES = frozenset({
    QUALITY_UNSPECIFIED,
    QUALITY_OBSERVED,
    QUALITY_INTERPOLATED,
    QUALITY_MISSING,
    QUALITY_STALE,
})


@dataclass
class SchemaVersion:
    """common.proto: message SchemaVersion."""

    major: int
    minor: int


@dataclass
class ScenarioRef:
    """common.proto: message ScenarioRef.

    `out_of_distribution` is the proto's own words: "label used for M2's OOD
    evaluation". It is the contract-carried OOD label, and the only field this
    module should read when deciding whether an episode counts as OOD.
    """

    scenario_id: str
    library_version: str
    out_of_distribution: bool = False


@dataclass
class Envelope:
    """common.proto: message Envelope.

    `emitted_at` stands in for `google.protobuf.Timestamp` as POSIX seconds.
    `scenario` is set during replay and empty in live operation -- per ADR 0004
    there is no live mode, so in practice it is always set.
    """

    schema_version: SchemaVersion
    emitted_at: float
    producer: str = "module1"
    scenario: ScenarioRef | None = None
    trace_id: str = ""


@dataclass
class QualityMask:
    """common.proto: message QualityMask.

    `per_feature` is one `Quality` per entry of `feature_names`, in the same
    order. `observed_fraction` is documented as a convenience -- the share of
    features that are QUALITY_OBSERVED -- so it must never be set independently
    of `per_feature`. Build with `from_per_feature` rather than by hand.
    """

    per_feature: list[str]
    observed_fraction: float

    @classmethod
    def from_per_feature(cls, per_feature) -> QualityMask:
        flags = list(per_feature)
        if not flags:
            return cls([], 0.0)
        observed = sum(1 for q in flags if q == QUALITY_OBSERVED)
        return cls(flags, observed / len(flags))


@dataclass
class StateRepresentation:
    """module1.proto: message StateRepresentation. Field order matches the proto.

    `degraded` means "one or more modalities are absent (e.g. comms blackout)"
    -- it is a sensing-availability flag, NOT an OOD flag. An extreme-but-fully-
    observed state is `degraded=False` with `scenario.out_of_distribution=True`.
    Conflating the two would leak the evaluation label into the state.
    """

    envelope: Envelope
    node_count: int
    embedding_dim: int
    node_embedding: np.ndarray   # (node_count, embedding_dim)
    graph_embedding: np.ndarray | None  # (embedding_dim,) pooled; optional in the proto
    feature_names: list[str]
    node_features: np.ndarray    # (node_count, len(feature_names))
    quality: QualityMask
    degraded: bool = False

    def __post_init__(self) -> None:
        self.node_features = np.asarray(self.node_features, dtype=np.float32)
        self.node_embedding = np.asarray(self.node_embedding, dtype=np.float32)
        if self.graph_embedding is not None:
            self.graph_embedding = np.asarray(self.graph_embedding, dtype=np.float32)

        expected_features = (self.node_count, len(self.feature_names))
        if self.node_features.shape != expected_features:
            raise ValueError(
                f"node_features {self.node_features.shape} != {expected_features} "
                "(node_count x len(feature_names))"
            )
        expected_embedding = (self.node_count, self.embedding_dim)
        if self.node_embedding.shape != expected_embedding:
            raise ValueError(
                f"node_embedding {self.node_embedding.shape} != {expected_embedding} "
                "(node_count x embedding_dim)"
            )
        if len(self.quality.per_feature) != len(self.feature_names):
            raise ValueError(
                f"quality.per_feature has {len(self.quality.per_feature)} entries for "
                f"{len(self.feature_names)} features -- the mask must cover every feature"
            )
        unknown = set(self.quality.per_feature) - QUALITY_VALUES
        if unknown:
            raise ValueError(f"quality.per_feature has non-Quality values: {sorted(unknown)}")

    @property
    def out_of_distribution(self) -> bool:
        """The contract-carried OOD label, or False when no scenario is attached."""
        return bool(self.envelope.scenario and self.envelope.scenario.out_of_distribution)

    def edl_matrix(self, source: str = "features") -> np.ndarray:
        """The (node_count, d) matrix the evidential head consumes.

        `features` is the engineered vector -- the proto keeps it alongside the
        embedding precisely so downstream ablations can use either. `embedding`
        is only meaningful once a real M1 producer is wired in; the mock fills
        it with noise, so training on it measures nothing.
        """
        if source == "features":
            return self.node_features
        if source == "embedding":
            return self.node_embedding
        raise ValueError(f"source must be 'features' or 'embedding', got {source!r}")

    def summary(self) -> dict:
        d = asdict(self)
        d["node_features"] = np.asarray(self.node_features).round(3).tolist()
        d["node_embedding"] = f"<{self.node_count}x{self.embedding_dim} array>"
        d["graph_embedding"] = (
            None if self.graph_embedding is None else f"<{self.embedding_dim} array>"
        )
        return d


def stack_features(states, source: str = "features") -> np.ndarray:
    """Concatenate a sequence of states into one (sum(node_count), d) design matrix."""
    return np.concatenate([s.edl_matrix(source) for s in states], 0).astype(np.float32)
