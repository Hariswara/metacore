"""Minimal read-only xlsx reader — an xlsx is a zip of XML, and that is all we need from it.

Avoids openpyxl/pandas so the calibration stages stay importable without the module's ML
dependencies. Handles exactly the features the CEB workbooks use: shared strings, inline
strings, numeric cells, and sparse rows.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS = {"m": _MAIN, "r": _REL}

# A sheet row, as {0-based column index: cell value}. Absent keys are empty cells, not zeros —
# the distinction matters upstream of a QualityMask.
Row = dict[int, str]
Sheet = dict[int, Row]


def _column_index(cell_ref: str) -> int:
    """'BQ12' -> 68. Column letters are base-26 with no zero digit."""
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_workbook(path: str | Path) -> dict[str, Sheet]:
    """Return {sheet name: {1-based row number: Row}}, preserving sheet order."""
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_MAIN}}}t")))

        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        rel_root = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rel_root}

        sheets: dict[str, Sheet] = {}
        for sheet_el in workbook.find("m:sheets", _NS):
            target = targets[sheet_el.get(f"{{{_REL}}}id")]
            if not target.startswith("xl/"):
                target = "xl/" + target.lstrip("/")
            sheets[sheet_el.get("name")] = _read_sheet(ET.fromstring(z.read(target)), shared)
        return sheets


def _read_sheet(root: ET.Element, shared: list[str]) -> Sheet:
    rows: Sheet = {}
    for row_el in root.iter(f"{{{_MAIN}}}row"):
        cells: Row = {}
        for c in row_el.findall("m:c", _NS):
            idx = _column_index(c.get("r"))
            cell_type = c.get("t")
            v = c.find("m:v", _NS)
            inline = c.find("m:is", _NS)
            if cell_type == "s" and v is not None:
                cells[idx] = shared[int(v.text)]
            elif cell_type == "inlineStr" and inline is not None:
                cells[idx] = "".join(t.text or "" for t in inline.iter(f"{{{_MAIN}}}t"))
            elif v is not None:
                cells[idx] = v.text
            else:
                cells[idx] = ""
        rows[int(row_el.get("r"))] = cells
    return rows


def to_float(value: str | None) -> float | None:
    """Empty and non-numeric cells become None, never 0.0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
