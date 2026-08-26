"""Stage: reconciliation gate for the CEB Jaffna ledger.

The workbook arrives with no schema and an undocumented two-level header, so the column layout in
`ceb.py` is an interpretation. This module is what makes it a checked one, using the fact that CEB
supplied the same figures twice by independent routes: a monthly spreadsheet and a printed annual
summary produced separately from it (`Data_CEB_Jaffna.pdf`, pp. 1-2). The PDF figures below are
transcribed from that scan and are the reference; the spreadsheet is the thing under test.

Per ADR 0004 the calibration path cannot be validated against held-out telemetry, because none
exists. Reconciliation is what stands in for it. A re-export with shifted columns fails here
rather than silently changing every downstream parameter.

Usage:
    python -m module1.data.validate <tidy.csv>
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from .ceb import LITRES_PER_BARREL

# Annual totals from Data_CEB_Jaffna.pdf pp. 1-2, "Island Generation Summary for the Year".
# (units_kwh, diesel_l, diesel_cost_rs, barrel_amount, oil_l, oil_cost_rs); None = dash in source.
PDF_ANNUAL: dict[tuple[str, str], tuple] = {
    ("2024", "Analaithivu"):
        (398_143, 164_010, 53_573_660.00, 1_230_075.00, 440, 942_155.52),
    ("2024", "Eluvaitivu-Diesel"):
        (88_667, 35_685, 11_686_501.00, 267_637.50, None, None),
    ("2024", "Eluvaitivu-Hybrid"):
        (96_445, 8_820, 2_829_970.00, 66_150.00, 50, 107_369.00),
    ("2024", "Delft-Neduntivu"):
        (1_064_031, 341_385, 113_032_415.00, 307_246.50, 840, 1_800_636.48),
    ("2024", "Nainativu"):
        (1_180_710, 387_110, 126_806_145.00, 2_903_325.00, 120, 257_685.60),
    ("2025", "Analaithivu"):
        (426_418, 163_915, 46_299_565.00, 1_229_362.50, 210, 443_884.14),
    ("2025", "Eluvaitivu-Diesel"):
        (113_276, 46_050, 12_992_598.00, 345_375.00, None, None),
    ("2025", "Eluvaitivu-Hybrid"):
        (83_331, 9_705, 2_748_075.00, 72_787.50, 40, 84_549.36),
    ("2025", "Delft-Neduntivu"):
        (1_126_880, 366_260, 104_574_021.50, 347_947.00, 575, 1_215_397.05),
    ("2025", "Nainativu"):
        (1_198_460, 389_685, 110_125_560.00, 2_922_637.50, 250, 528_433.50),
}

REFERENCE_FIELDS = (
    "units_kwh", "diesel_l", "diesel_cost_rs", "barrel_amount", "oil_l", "oil_cost_rs",
)

# Barrel transport rate in Rs per 200 L barrel, keyed by (year, system) because the rate is
# re-tendered annually. Delft has a dedicated ferry and is supplied roughly 8x cheaper than the
# three islands on the standard rate — that spread is the marine-logistics cost signal M3 gates
# on, so a rate applied to the wrong year or island is not a rounding error downstream.
# Exact in every month of both years; this reconciliation is what established the 2025 change.
BARREL_RATE_RS = {
    ("2024", "Analaithivu"): 1500.0,
    ("2024", "Eluvaitivu-Diesel"): 1500.0,
    ("2024", "Eluvaitivu-Hybrid"): 1500.0,
    ("2024", "Delft-Neduntivu"): 180.0,
    ("2024", "Nainativu"): 1500.0,
    ("2025", "Analaithivu"): 1500.0,
    ("2025", "Eluvaitivu-Diesel"): 1500.0,
    ("2025", "Eluvaitivu-Hybrid"): 1500.0,
    ("2025", "Delft-Neduntivu"): 190.0,
    ("2025", "Nainativu"): 1500.0,
}

TOLERANCE = 0.01  # absolute, in source units; the ledger is exact, not approximate


class ReconciliationError(AssertionError):
    """The spreadsheet no longer agrees with the printed summary."""


def _rows(csv_path: str | Path) -> list[dict]:
    with Path(csv_path).open() as fh:
        return list(csv.DictReader(fh))


def _num(value: str) -> float:
    return float(value) if value not in ("", "None", None) else 0.0


def check(csv_path: str | Path) -> list[str]:
    """Return a list of failure messages; empty means every invariant held."""
    rows = _rows(csv_path)
    failures: list[str] = []

    expected_rows = len(PDF_ANNUAL) * 12
    if len(rows) != expected_rows:
        failures.append(f"row count: expected {expected_rows}, got {len(rows)}")

    # 1. Barrels are litres/200 exactly, row by row. This is what pins the two fuel columns to
    #    each other; if the layout slipped by one column it breaks here first.
    for row in rows:
        litres, barrels = _num(row["diesel_l"]), _num(row["diesel_barrel"])
        if abs(litres / LITRES_PER_BARREL - barrels) > TOLERANCE:
            failures.append(
                f"{row['year']}-{row['month']} {row['island_system']}: "
                f"{litres} L / {LITRES_PER_BARREL} != {barrels} barrels"
            )

    # 2. Transport cost is barrels x the island's rate.
    for row in rows:
        barrels, amount = _num(row["diesel_barrel"]), _num(row["barrel_amount"])
        rate = BARREL_RATE_RS.get((row["year"], row["island_system"]))
        if rate is None:
            failures.append(
                f"{row['year']} {row['island_system']}: no barrel rate on record for this "
                f"island-year — add it once confirmed against the source, do not infer it"
            )
            continue
        if abs(barrels * rate - amount) > TOLERANCE:
            failures.append(
                f"{row['year']}-{row['month']} {row['island_system']}: "
                f"{barrels} barrels x Rs {rate} != Rs {amount}"
            )

    # 3. Annual totals match the independently-produced PDF summary.
    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        key = (row["year"], row["island_system"])
        for field in REFERENCE_FIELDS:
            totals[key][field] += _num(row[field])

    for key, reference in PDF_ANNUAL.items():
        if key not in totals:
            failures.append(f"{key[0]} {key[1]}: absent from extract")
            continue
        for field, expected in zip(REFERENCE_FIELDS, reference, strict=True):
            if expected is None:  # dash in the printed summary; ledger records zero
                expected = 0.0
            actual = totals[key][field]
            if abs(actual - expected) > TOLERANCE:
                failures.append(
                    f"{key[0]} {key[1]} {field}: PDF {expected:,.2f} != extract {actual:,.2f}"
                )
    return failures


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    failures = check(argv[1])
    if failures:
        print(f"RECONCILIATION FAILED — {len(failures)} problem(s):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    checks = len(_rows(argv[1])) * 2 + len(PDF_ANNUAL) * len(REFERENCE_FIELDS)
    print(f"reconciliation OK — {checks} invariants held across {len(PDF_ANNUAL)} island-years")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
