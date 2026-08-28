"""Stage: CEB Jaffna generation ledger -> long-format calibration table.

The workbook is two wide sheets (one per year) with a two-level header: a merged island band on
row 1 and per-metric columns on row 2. Column order differs between island bands — Eluvaitivu's
hybrid block puts the oil columns last — so the layout is declared explicitly below rather than
inferred. The declaration is validated by `validate.py`, which reconciles every annual total
against the independently-produced PDF summary tables.

Usage:
    python -m module1.data.ceb <workbook.xlsx> <out.csv>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from .xlsx import Sheet, read_workbook, to_float

LITRES_PER_BARREL = 200.0

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

HEADER_ROW = 2          # per-metric header
FIRST_DATA_ROW = 3      # January
YEAR_CELL = (1, 0)      # row 1, column A holds the year

# 0-based column index of each metric, per generating system. Verified by reconciliation.
LAYOUT: dict[str, dict[str, int]] = {
    "Analaithivu": {
        "diesel_l": 1, "diesel_cost_rs": 2, "units_kwh": 3,
        "oil_l": 4, "oil_cost_rs": 5, "diesel_barrel": 6, "barrel_amount": 7,
    },
    "Eluvaitivu-Diesel": {
        "diesel_l": 8, "diesel_cost_rs": 9, "units_kwh": 10,
        "oil_l": 11, "oil_cost_rs": 12, "diesel_barrel": 13, "barrel_amount": 14,
    },
    "Eluvaitivu-Hybrid": {
        "diesel_l": 15, "diesel_cost_rs": 16, "units_kwh": 17,
        "diesel_barrel": 18, "barrel_amount": 19, "oil_l": 20, "oil_cost_rs": 21,
    },
    "Delft-Neduntivu": {
        "diesel_l": 22, "diesel_cost_rs": 23, "units_kwh": 24,
        "oil_l": 25, "oil_cost_rs": 26, "diesel_barrel": 27, "barrel_amount": 28,
    },
    "Nainativu": {
        "diesel_l": 29, "diesel_cost_rs": 30, "units_kwh": 31,
        "oil_l": 32, "oil_cost_rs": 33, "diesel_barrel": 34, "barrel_amount": 35,
    },
}

MEASURED = ("diesel_l", "diesel_cost_rs", "units_kwh", "oil_l", "oil_cost_rs",
            "diesel_barrel", "barrel_amount")
DERIVED = ("sfc_l_per_kwh", "diesel_rs_per_l", "fuel_cost_rs_per_kwh", "total_cost_rs_per_kwh")
FIELDS = ("year", "month", "month_num", "island_system", *MEASURED, *DERIVED)

# Columns past the declared layout are the sheet author's scratch area — an unlabelled trailing
# figure and a reversed-month block below the totals row. Not authoritative; not extracted.


def _derive(rec: dict) -> None:
    diesel_l = rec["diesel_l"]
    kwh = rec["units_kwh"]
    diesel_cost = rec["diesel_cost_rs"]
    oil_cost = rec["oil_cost_rs"] or 0.0

    rec["sfc_l_per_kwh"] = round(diesel_l / kwh, 4) if diesel_l and kwh else None
    rec["diesel_rs_per_l"] = round(diesel_cost / diesel_l, 2) if diesel_cost and diesel_l else None
    rec["fuel_cost_rs_per_kwh"] = round(diesel_cost / kwh, 2) if diesel_cost and kwh else None
    rec["total_cost_rs_per_kwh"] = (
        round((diesel_cost + oil_cost) / kwh, 2) if diesel_cost and kwh else None
    )


def extract_sheet(sheet: Sheet) -> list[dict]:
    """One sheet (one year) -> 60 records: 12 months x 5 generating systems."""
    year = sheet[YEAR_CELL[0]].get(YEAR_CELL[1], "").strip()
    records = []
    for offset, month in enumerate(MONTHS):
        row = sheet.get(FIRST_DATA_ROW + offset)
        if row is None:
            continue
        for system, columns in LAYOUT.items():
            rec = {
                "year": year,
                "month": month,
                "month_num": offset + 1,
                "island_system": system,
            }
            for metric, col in columns.items():
                rec[metric] = to_float(row.get(col))
            _derive(rec)
            records.append(rec)
    return records


def extract(workbook_path: str | Path) -> list[dict]:
    records: list[dict] = []
    for sheet in read_workbook(workbook_path).values():
        records.extend(extract_sheet(sheet))
    return records


def write_csv(records: list[dict], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in FIELDS})


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    records = extract(argv[1])
    write_csv(records, argv[2])
    print(f"{argv[2]}: {len(records)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
