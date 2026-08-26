"""The generated stubs must be importable as an installed package.

`task proto` rewrites the flat imports protoc emits (see fix_python_imports.py). Nothing in the
repo imported these stubs until M2 built against them, so the breakage sat undetected from the
scaffold commit onward. These tests are the tripwire: they fail if the rewrite is skipped, so a
regeneration cannot quietly ship an unimportable contract again.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

MODULES = [
    "common_pb2",
    "module1_pb2",
    "module2_pb2",
    "module3_pb2",
    "verification_pb2",
]

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "metacore_contracts"
FLAT_IMPORT = re.compile(r"^import \w+_pb2 as ", re.MULTILINE)


@pytest.mark.parametrize("name", MODULES)
def test_stub_imports_as_a_package_member(name: str) -> None:
    """The failure this guards against is ModuleNotFoundError: No module named 'common_pb2'."""
    assert importlib.import_module(f"metacore_contracts.{name}") is not None


@pytest.mark.parametrize("path", sorted(PACKAGE_DIR.glob("*_pb2*.py")), ids=lambda p: p.name)
def test_no_flat_imports_survive(path: Path) -> None:
    """A flat `import x_pb2` means the post-generation rewrite did not run."""
    assert not FLAT_IMPORT.search(path.read_text()), (
        f"{path.name} still has a flat protoc import -- run `task proto`, or "
        f"`python packages/contracts/fix_python_imports.py` directly"
    )


def test_state_representation_carries_the_fields_m2_sizes_against() -> None:
    """M2's evidential head is sized from these. Renaming one is a contract break, not a rename."""
    from metacore_contracts import module1_pb2

    fields = {f.name for f in module1_pb2.StateRepresentation.DESCRIPTOR.fields}
    assert {"embedding_dim", "feature_names", "node_embedding", "quality", "degraded"} <= fields


def test_scenario_ref_carries_the_ood_label() -> None:
    """M2's OOD evaluation selects episodes on this field."""
    from metacore_contracts import common_pb2

    fields = {f.name for f in common_pb2.ScenarioRef.DESCRIPTOR.fields}
    assert {"scenario_id", "library_version", "out_of_distribution"} <= fields
