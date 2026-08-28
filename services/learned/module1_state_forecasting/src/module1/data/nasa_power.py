"""Stage: NASA POWER hourly meteorology for the four Jaffna islands.

Open-source replacement for the Department of Meteorology feed, which quoted Rs 75,000 for two
parameters. POWER is free, has no licence gate, and is therefore allowed to be a build dependency
in a way `data/external/` is not.

Resolution warning, recorded here because it changes what M1 can claim
--------------------------------------------------------------------
POWER serves two products on different grids, and neither resolves these islands:

* **Meteorology** (wind, temperature, humidity, precipitation, pressure) comes from MERRA-2 at
  0.5 deg latitude by 0.625 deg longitude — about 55 km by 69 km here.
* **Solar** (irradiance, clear-sky irradiance) comes from CERES SYN1deg at 1 deg — about 111 km.

The four islands span roughly 27 km. Measured over the full 2024-2025 hourly pull, that yields
**one** distinct irradiance series across all four sites, and **two** distinct meteorological
series. Which islands share a series is an artifact of where a cell boundary happens to fall, not
of geography: Eluvaitivu and Analaitivu are 5 km apart and differ; Analaitivu and Delft are 25 km
apart and are bit-identical.

The consequence is concrete and constrains the model rather than the pipeline: **no spatial solar
gradient between islands exists in this source at all**, and wind only separates one island from
the other three. An ST-GNN cannot learn an inter-island weather gradient it was never shown, so
spatial variation in the resource must be treated as unobserved — not as something the graph
recovers. `manifest.json` records the measured distinct-series count per parameter so this is
visible in the data rather than discovered later inside a result.

Usage:
    python -m module1.data.nasa_power fetch    <out_dir> [--start YYYYMMDD] [--end YYYYMMDD]
    python -m module1.data.nasa_power validate <out_dir>
"""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

API = "https://power.larc.nasa.gov/api/temporal/hourly/point"
COMMUNITY = "RE"  # renewable-energy community: solar and wind parameters, RE unit conventions
FILL_VALUE = -999.0

# MERRA-2 native grid. Used only to report which sites collapse onto one cell.
GRID_LAT_DEG = 0.5
GRID_LON_DEG = 0.625

# POWER parameter -> column name, unit, and the source product's native grid. Order fixes the
# CSV column order. The grid differs by product, which is why solar aliases harder than wind.
PARAMETERS: dict[str, tuple[str, str, str]] = {
    "ALLSKY_SFC_SW_DWN": ("ghi_wh_m2", "Wh/m^2", "CERES SYN1deg 1.0deg"),
    "CLRSKY_SFC_SW_DWN": ("ghi_clearsky_wh_m2", "Wh/m^2", "CERES SYN1deg 1.0deg"),
    "WS10M": ("wind_10m_ms", "m/s", "MERRA-2 0.5x0.625deg"),
    "WS50M": ("wind_50m_ms", "m/s", "MERRA-2 0.5x0.625deg"),
    "T2M": ("temp_2m_c", "C", "MERRA-2 0.5x0.625deg"),
    "RH2M": ("humidity_2m_pct", "%", "MERRA-2 0.5x0.625deg"),
    "PRECTOTCORR": ("precip_mm_hr", "mm/hour", "MERRA-2 0.5x0.625deg"),
    "PS": ("pressure_kpa", "kPa", "MERRA-2 0.5x0.625deg"),
}

DEFAULT_START = "20240101"
DEFAULT_END = "20251231"


@dataclass(frozen=True)
class Site:
    """An island generating site. Coordinates are island centroids to ~1 km."""

    key: str
    latitude: float
    longitude: float

    @property
    def grid_cell(self) -> tuple[float, float]:
        """The MERRA-2 cell this site actually samples."""
        return (
            round(self.latitude / GRID_LAT_DEG) * GRID_LAT_DEG,
            round(self.longitude / GRID_LON_DEG) * GRID_LON_DEG,
        )


ISLAND_SITES: tuple[Site, ...] = (
    Site("Eluvaitivu", 9.760, 79.770),
    Site("Analaitivu", 9.720, 79.790),
    Site("Nainativu", 9.615, 79.775),
    Site("Delft-Neduntivu", 9.520, 79.690),
)

COLUMNS = ("island", "timestamp_lst", *(name for name, _u, _g in PARAMETERS.values()))


@dataclass
class FetchReport:
    site: str
    rows: int
    missing_cells: int = 0
    years: list[str] = field(default_factory=list)


def grid_aliasing() -> dict[str, list[str]]:
    """{cell: [sites sharing it]} — more than one name in a bucket means unresolved sites."""
    buckets: dict[str, list[str]] = {}
    for site in ISLAND_SITES:
        buckets.setdefault(f"{site.grid_cell[0]},{site.grid_cell[1]}", []).append(site.key)
    return buckets


def measure_distinct_series(rows_by_site: dict[str, list[dict]]) -> dict[str, list[list[str]]]:
    """Group sites by identical series, per parameter, from the data actually fetched.

    Measured rather than predicted from the grid arithmetic: the two POWER products sit on
    different grids, and whether a given parameter is interpolated or nearest-neighbour is not
    documented per-parameter. Anything claimed about resolution should come from the bytes.
    """
    present = {c for rows in rows_by_site.values() for r in rows[:1] for c in r}
    grouped: dict[str, list[list[str]]] = {}
    for column, _unit, _grid in PARAMETERS.values():
        # Skip rather than group-on-absent: a column missing everywhere would otherwise land all
        # sites in one bucket and read as "identical series" when it means "no data at all".
        if column not in present:
            continue
        buckets: dict[tuple, list[str]] = {}
        for site, rows in rows_by_site.items():
            buckets.setdefault(tuple(r[column] for r in rows), []).append(site)
        grouped[column] = sorted(buckets.values())
    return grouped


def _request(url: str, attempts: int = 4) -> dict:
    """POWER rate-limits and occasionally times out; back off rather than dropping a year."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < attempts - 1:
                time.sleep(2 ** attempt * 3)
    raise RuntimeError(f"NASA POWER request failed after {attempts} attempts: {last}") from last


def fetch_site(site: Site, start: str, end: str) -> tuple[list[dict], dict]:
    """Hourly series for one site, chunked by calendar year (POWER caps an hourly request)."""
    rows: list[dict] = []
    meta: dict = {}
    for year in range(int(start[:4]), int(end[:4]) + 1):
        chunk_start = max(start, f"{year}0101")
        chunk_end = min(end, f"{year}1231")
        query = {
            "parameters": ",".join(PARAMETERS),
            "community": COMMUNITY,
            "latitude": site.latitude,
            "longitude": site.longitude,
            "start": chunk_start,
            "end": chunk_end,
            "format": "JSON",
        }
        payload = _request(f"{API}?{urllib.parse.urlencode(query)}")
        if not meta:
            meta = {
                "api": payload.get("header", {}).get("api"),
                "sources": payload.get("header", {}).get("sources"),
                "time_standard": payload.get("header", {}).get("time_standard"),
                "fill_value": payload.get("header", {}).get("fill_value"),
                "returned_coordinates": payload.get("geometry", {}).get("coordinates"),
            }
        rows.extend(_to_rows(site, payload))
    return rows, meta


def _to_rows(site: Site, payload: dict) -> list[dict]:
    parameters = payload["properties"]["parameter"]
    stamps = sorted(parameters[next(iter(PARAMETERS))])
    rows = []
    for stamp in stamps:
        # POWER stamps hours as YYYYMMDDHH in local solar time.
        row = {
            "island": site.key,
            "timestamp_lst": f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[8:10]}:00",
        }
        for api_name, (column, _unit, _grid) in PARAMETERS.items():
            value = parameters.get(api_name, {}).get(stamp)
            # A fill value means "not measured". It must never reach a model as a number, and
            # 0.0 would be a plausible-looking lie for irradiance or precipitation.
            row[column] = None if value is None or value == FILL_VALUE else value
        rows.append(row)
    return rows


def write_site_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS})


def fetch(out_dir: Path, start: str = DEFAULT_START, end: str = DEFAULT_END) -> list[FetchReport]:
    reports = []
    manifest: dict = {
        "source": "NASA POWER",
        "endpoint": API,
        "community": COMMUNITY,
        "temporal": "hourly",
        "requested_start": start,
        "requested_end": end,
        "parameters": {
            api: {"column": c, "unit": u, "source_grid": g}
            for api, (c, u, g) in PARAMETERS.items()
        },
        "grid": {
            "native_resolution_deg": {"latitude": GRID_LAT_DEG, "longitude": GRID_LON_DEG},
            "cells_to_sites": grid_aliasing(),
            "distinct_cells": len(grid_aliasing()),
            "site_count": len(ISLAND_SITES),
            "note": (
                "Sites sharing a cell receive identical series. Inter-island weather variation "
                "is unobserved at this resolution — see the module docstring."
            ),
        },
        "sites": {},
    }

    rows_by_site: dict[str, list[dict]] = {}
    for site in ISLAND_SITES:
        rows, meta = fetch_site(site, start, end)
        rows_by_site[site.key] = rows
        write_site_csv(rows, out_dir / f"{site.key}_hourly.csv")
        missing = sum(
            1 for row in rows for column, _u, _g in PARAMETERS.values() if row[column] is None
        )
        reports.append(FetchReport(site.key, len(rows), missing))
        manifest["sites"][site.key] = {
            "latitude": site.latitude,
            "longitude": site.longitude,
            "grid_cell": list(site.grid_cell),
            "rows": len(rows),
            "missing_values": missing,
            "file": f"{site.key}_hourly.csv",
            **meta,
        }
        print(f"{site.key}: {len(rows)} hours, {missing} missing values")

    measured = measure_distinct_series(rows_by_site)
    manifest["grid"]["measured_distinct_series"] = {
        column: {"count": len(groups), "groups": groups} for column, groups in measured.items()
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for column, groups in measured.items():
        if len(groups) < len(ISLAND_SITES):
            print(f"  {column}: {len(groups)} distinct series for {len(ISLAND_SITES)} sites")
    return reports


def _expected_hours(start: str, end: str) -> int:
    first = datetime.strptime(start, "%Y%m%d").date()
    last = datetime.strptime(end, "%Y%m%d").date()
    return ((last - first) + timedelta(days=1)).days * 24


def check(out_dir: Path) -> list[str]:
    """Invariants a plausible-looking but wrong weather pull would fail."""
    failures: list[str] = []
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return [f"no manifest at {manifest_path}; run the fetch stage first"]
    manifest = json.loads(manifest_path.read_text())
    expected = _expected_hours(manifest["requested_start"], manifest["requested_end"])

    for site in ISLAND_SITES:
        path = out_dir / f"{site.key}_hourly.csv"
        if not path.exists():
            failures.append(f"{site.key}: missing {path.name}")
            continue
        with path.open() as fh:
            rows = list(csv.DictReader(fh))

        if len(rows) != expected:
            failures.append(f"{site.key}: {len(rows)} hours, expected {expected}")

        for row in rows:
            for column, _unit, _grid in PARAMETERS.values():
                raw = row[column]
                if raw == "":
                    continue
                value = float(raw)
                # A fill value that survived into the CSV as a number is the failure mode that
                # would quietly train a model on -999 W/m^2 of sunlight.
                if value == FILL_VALUE:
                    failures.append(f"{site.key} {row['timestamp_lst']}: fill value in {column}")
                    break
                if column in ("ghi_wh_m2", "ghi_clearsky_wh_m2", "precip_mm_hr") and value < 0:
                    failures.append(f"{site.key} {row['timestamp_lst']}: negative {column}")
                    break
                if column in ("wind_10m_ms", "wind_50m_ms") and value < 0:
                    failures.append(f"{site.key} {row['timestamp_lst']}: negative {column}")
                    break

        # Physical sanity: no sun at local midnight, and real sun at local midday.
        midnight = [float(r["ghi_wh_m2"]) for r in rows if r["timestamp_lst"].endswith("T00:00")
                    and r["ghi_wh_m2"] != ""]
        midday = [float(r["ghi_wh_m2"]) for r in rows if r["timestamp_lst"].endswith("T12:00")
                  and r["ghi_wh_m2"] != ""]
        if midnight and max(midnight) > 0:
            failures.append(f"{site.key}: non-zero irradiance at local midnight")
        if midday and max(midday) <= 0:
            failures.append(f"{site.key}: no irradiance at any local midday")

    return failures


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] not in ("fetch", "validate"):
        print(__doc__, file=sys.stderr)
        return 2
    command, out_dir = argv[1], Path(argv[2])

    if command == "fetch":
        start = DEFAULT_START
        end = DEFAULT_END
        for i, arg in enumerate(argv):
            if arg == "--start" and i + 1 < len(argv):
                start = argv[i + 1]
            if arg == "--end" and i + 1 < len(argv):
                end = argv[i + 1]
        aliasing = grid_aliasing()
        shared = {c: s for c, s in aliasing.items() if len(s) > 1}
        if shared:
            print(
                f"note: {len(aliasing)} MERRA-2 cells cover {len(ISLAND_SITES)} sites; "
                f"identical series for {'; '.join(', '.join(v) for v in shared.values())}"
            )
        fetch(out_dir, start, end)
        return 0

    failures = check(out_dir)
    if failures:
        print(f"NASA POWER CHECK FAILED — {len(failures)} problem(s):", file=sys.stderr)
        for failure in failures[:25]:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"nasa_power OK — {len(ISLAND_SITES)} sites, {len(grid_aliasing())} distinct grid cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
