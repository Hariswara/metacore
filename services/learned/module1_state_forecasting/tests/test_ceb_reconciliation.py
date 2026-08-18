"""The CEB ledger reconciliation gate, run as a test.

Skips when the state-entity data is absent. That is deliberate and is what keeps the promise in
`data/README.md` — nothing in `data/external/` may become a build requirement — honest in code
rather than only in prose (ADR 0004).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from module1.data import ceb, validate

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKBOOK = REPO_ROOT / "data" / "external" / "ceb_jaffna" / "Generation_2024_2025.xlsx"

pytestmark = pytest.mark.skipif(
    not WORKBOOK.exists(), reason="CEB workbook not present; calibration data is optional"
)


@pytest.fixture(scope="module")
def extract_csv(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("ceb") / "tidy.csv"
    ceb.write_csv(ceb.extract(WORKBOOK), out)
    return out


def test_shape(extract_csv: Path) -> None:
    rows = list(csv.DictReader(extract_csv.open()))
    assert len(rows) == 120, "2 years x 12 months x 5 generating systems"
    assert {r["island_system"] for r in rows} == set(ceb.LAYOUT)
    assert {r["year"] for r in rows} == {"2024", "2025"}


def test_reconciles_against_printed_summary(extract_csv: Path) -> None:
    """Every annual total must match the PDF summary CEB produced independently of the sheet."""
    failures = validate.check(extract_csv)
    assert not failures, "\n".join(failures)


def test_extraction_is_deterministic(extract_csv: Path, tmp_path: Path) -> None:
    again = tmp_path / "again.csv"
    ceb.write_csv(ceb.extract(WORKBOOK), again)
    assert again.read_bytes() == extract_csv.read_bytes()


def test_missing_cells_are_none_not_zero() -> None:
    """A blank cell must not become 0.0 — downstream that is the difference between
    'no lube oil was used' and 'we do not know', which is what QualityMask encodes."""
    from module1.data.xlsx import to_float

    assert to_float("") is None
    assert to_float(None) is None
    assert to_float("0") == 0.0
