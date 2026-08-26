"""The pinned Module 1 state-representation contract.

HAND-WRITTEN. `task proto` does not generate or overwrite this file, nor `schema/*.json`.

`module1.proto` declares that `StateRepresentation` has an `embedding_dim` and a `feature_names`
list; it cannot declare what they are. A consumer sizing a head against the proto alone is still
guessing. This module is where the guess stops: it publishes the concrete values, the units, and
-- the part that matters for M2 -- which features are actually observed and which only look it.

    from metacore_contracts.state_schema import EMBEDDING_DIM, FEATURE_NAMES, SCHEMA_VERSION

Changing `EMBEDDING_DIM` or reordering `FEATURE_NAMES` is a minor bump. Renaming or removing a
feature is a major bump and needs an adapter -- see packages/contracts/README.md.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

_SCHEMA_FILE = Path(__file__).parent / "schema" / "module1_state_v1.json"


class SchemaVersion(NamedTuple):
    """Mirrors metacore.common.v1.SchemaVersion without requiring the generated stub."""

    major: int
    minor: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    """The full pin, including per-feature units, sources and quality semantics."""
    return json.loads(_SCHEMA_FILE.read_text())


_SCHEMA = load()

#: Width of the learned per-node embedding. See the rationale field in the JSON.
EMBEDDING_DIM: int = _SCHEMA["embedding_dim"]

#: Column order of StateRepresentation.node_features, and of QualityMask.per_feature.
FEATURE_NAMES: tuple[str, ...] = tuple(_SCHEMA["feature_names"])

#: Stamped into Envelope.schema_version by every M1 producer.
SCHEMA_VERSION = SchemaVersion(
    major=_SCHEMA["schema_version"]["major"],
    minor=_SCHEMA["schema_version"]["minor"],
)

#: Number of engineered features per node. Distinct from EMBEDDING_DIM, which is the learned width.
FEATURE_COUNT: int = len(FEATURE_NAMES)


def feature_index(name: str) -> int:
    """Column of `name` in node_features. Raises KeyError with the valid set on a typo."""
    try:
        return FEATURE_NAMES.index(name)
    except ValueError:
        raise KeyError(f"{name!r} is not in the v1 state schema; have {FEATURE_NAMES}") from None


def calibration_quality(name: str) -> str:
    """The QualityMask value this feature carries in the offline artifacts (ADR 0004).

    A floor, not a promise: at runtime the simulator fills the electrical group and the mask is
    set per step. Useful for asserting that a training set does not silently treat an
    interpolated feature as measured.
    """
    return _SCHEMA["features"][name]["calibration_quality"]


def stamp(envelope: Any) -> Any:
    """Set `envelope.schema_version` to the pinned version. Returns the envelope for chaining."""
    envelope.schema_version.major = SCHEMA_VERSION.major
    envelope.schema_version.minor = SCHEMA_VERSION.minor
    return envelope
