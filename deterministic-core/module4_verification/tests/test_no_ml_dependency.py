"""The learned/deterministic boundary, asserted as a test as well as in CI.

If this ever fails, the verification argument in the paper no longer holds — see
docs/adr/0003-deterministic-core-isolation.md.
"""
import pathlib
import sys
import tomllib

FORBIDDEN = ("torch", "tensorflow", "jax", "sklearn", "scikit-learn", "transformers", "onnxruntime")


def test_pyproject_declares_no_ml_dependency() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    deps = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
    offenders = [d for d in deps if any(f in d.lower() for f in FORBIDDEN)]
    assert not offenders, f"deterministic-core must declare no ML dependency, found: {offenders}"


def test_no_ml_module_is_importable_at_runtime() -> None:
    assert "torch" not in sys.modules
