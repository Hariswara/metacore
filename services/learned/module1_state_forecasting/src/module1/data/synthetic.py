"""Stage: the synthetic fallback ADR 0004 promises.

`data/README.md` states that nothing in `data/external/` is allowed to become a build requirement,
and ADR 0004 says the fallback is what makes that true in code rather than in prose. It was prose.
A clean clone has no CEB workbook -- it is state-entity data, shared for calibration and not
redistributable -- and until the DVC remote existed it had no artifacts either, so `task data`
failed at the first stage and CI never exercised the pipeline at all.

This generates a full stand-in input set: a monthly generation ledger and two years of hourly
meteorology for the four sites. Every downstream stage then runs untouched, and every gate runs
for real against it.

WHAT IS SYNTHETIC AND WHAT IS NOT. The annual totals are the genuine transcribed figures from
Data_CEB_Jaffna.pdf, which already live in `validate.py` as the reconciliation reference and are
not sourced from the workbook. The monthly distribution beneath them is invented, as is all of the
weather. So the fallback reconciles exactly -- which is the point, because a fallback that could
not pass the reconciliation gate would leave that gate untested in CI -- while resolving nothing
below the year. It is labelled at every layer: `PROVENANCE.json` beside the outputs, and a
`synthetic: true` flag consumers can branch on.

NEVER PUBLISH A RESULT COMPUTED FROM THIS. It exists so the code can be shown to run, not so the
system can be shown to work. The monthly shape here was chosen by hand; any finding drawn from it
is a finding about this file. `task data` against the real workbook, or `task artifacts:pull`, is
what produces the artifacts a result may cite.

The Eluvaitivu hybrid decay is reproduced deliberately rather than left out. The scenario library
gate asserts that the event is present and labelled out-of-distribution, and a fallback dataset
that quietly dropped the phenomenon under study would turn that gate green while removing the only
thing it checks.

Usage:
    python -m module1.data.synthetic build <out_dir>
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

from .ceb import FIELDS, LITRES_PER_BARREL, MONTHS, _derive
from .nasa_power import COLUMNS as WEATHER_COLUMNS
from .nasa_power import DEFAULT_END, DEFAULT_START, ISLAND_SITES
from .validate import BARREL_RATE_RS, PDF_ANNUAL

# Fixed so two people generating the fallback get byte-identical files and a CI failure is
# reproducible. Not a tuning knob.
SEED = 20260317

PROVENANCE_FILE = "PROVENANCE.json"

# Monthly demand shape for a tropical island: hotter and drier Mar-Aug draws more, the NE monsoon
# months draw less. Amplitude kept modest so in-distribution scenario windows still form.
SEASONAL = (0.95, 0.96, 1.02, 1.06, 1.08, 1.05, 1.04, 1.05, 1.01, 0.96, 0.93, 0.94)

# The 2025 Q4 hybrid degradation, as monthly multipliers on the plant's own baseline. Matches the
# shape of the measured collapse (roughly 0.70, 0.13, 0.03) without copying its exact figures.
DECAY_PLANT = "Eluvaitivu-Hybrid"
DECAY_YEAR = "2025"
DECAY = {10: 0.68, 11: 0.13, 12: 0.03}

# Jaffna sits at ~9.7 N. Enough to put the sun overhead near noon and give a real seasonal swing.
SOLAR_CONSTANT_WH = 1120.0
CLEARSKY_FLOOR = 0.18   # thickest monsoon overcast still passes this fraction of clear-sky
MONSOON_MONTHS = (10, 11, 12, 1)


# ------------------------------------------------------------------ ledger

def _allocate(total: float, weights: list[float], quantum: float) -> list[float]:
    """Split `total` across `weights`, rounded to `quantum`, summing back to `total` exactly.

    The residual lands on the largest element rather than the last, so a rounding crumb cannot
    push a small month negative or visibly distort the tail of a decay.
    """
    scale = total / sum(weights)
    parts = [round(w * scale / quantum) * quantum for w in weights]
    residual = total - sum(parts)
    parts[parts.index(max(parts))] += residual
    return parts


def build_ledger(seed: int = SEED) -> list[dict]:
    """Monthly rows in the shape ceb.py produces, reconciling exactly to the printed summary."""
    rng = random.Random(seed)
    rows: list[dict] = []

    for (year, plant), reference in sorted(PDF_ANNUAL.items()):
        units_total, diesel_total, diesel_cost_total, _amount, oil_total, oil_cost_total = reference

        weights = []
        for month_num in range(1, 13):
            weight = SEASONAL[month_num - 1] * rng.uniform(0.96, 1.04)
            if plant == DECAY_PLANT and year == DECAY_YEAR:
                weight *= DECAY.get(month_num, 1.0)
            weights.append(weight)

        # Diesel tracks energy but not proportionally: part-load running is less efficient, so a
        # low month burns more litres per kWh. This is the Willans line showing up in the fallback.
        units = _allocate(float(units_total), weights, 1.0)
        fuel_weights = [w ** 0.92 for w in weights]
        diesel = _allocate(float(diesel_total), fuel_weights, 5.0)
        diesel_cost = _allocate(float(diesel_cost_total), diesel, 0.01)
        oil = _allocate(float(oil_total or 0.0), weights, 5.0) if oil_total else [0.0] * 12
        oil_cost = (
            _allocate(float(oil_cost_total or 0.0), weights, 0.01) if oil_cost_total
            else [0.0] * 12
        )

        rate = BARREL_RATE_RS[(year, plant)]
        for index, month in enumerate(MONTHS):
            barrels = diesel[index] / LITRES_PER_BARREL
            record = {
                "year": year,
                "month": month,
                "month_num": index + 1,
                "island_system": plant,
                "diesel_l": diesel[index],
                "diesel_cost_rs": round(diesel_cost[index], 2),
                "units_kwh": units[index],
                "oil_l": oil[index],
                "oil_cost_rs": round(oil_cost[index], 2),
                "diesel_barrel": barrels,
                "barrel_amount": round(barrels * rate, 2),
            }
            _derive(record)
            rows.append(record)

    rows.sort(key=lambda r: (r["year"], r["month_num"], r["island_system"]))
    return rows


def write_ledger(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in FIELDS})


# ----------------------------------------------------------------- weather

def _clearsky_wh(stamp: datetime, latitude: float) -> float:
    """Clear-sky GHI from solar geometry. Zero when the sun is below the horizon."""
    day = stamp.timetuple().tm_yday
    declination = math.radians(23.45 * math.sin(math.radians(360 * (284 + day) / 365)))
    hour_angle = math.radians(15.0 * (stamp.hour - 12))
    phi = math.radians(latitude)
    sin_elevation = (
        math.sin(phi) * math.sin(declination)
        + math.cos(phi) * math.cos(declination) * math.cos(hour_angle)
    )
    return max(0.0, SOLAR_CONSTANT_WH * sin_elevation)


def build_weather(
    start: str = DEFAULT_START, end: str = DEFAULT_END, seed: int = SEED
) -> dict[str, list[dict]]:
    """Hourly rows per site, in the column order nasa_power.py writes and validates."""
    first = datetime.strptime(start, "%Y%m%d")
    last = datetime.strptime(end, "%Y%m%d") + timedelta(days=1)

    by_site: dict[str, list[dict]] = {}
    for site_index, site in enumerate(ISLAND_SITES):
        # Per-site seed offset: the real pull aliases several sites onto one grid cell, but a
        # fallback that emitted four bit-identical files would make that finding untestable.
        rng = random.Random(seed + site_index)
        rows: list[dict] = []
        stamp = first
        cloud = 0.8
        while stamp < last:
            monsoon = stamp.month in MONSOON_MONTHS
            # Cloud cover as a random walk, so consecutive hours correlate the way weather does.
            cloud += rng.uniform(-0.12, 0.12) + (0.02 if monsoon else -0.02) * (0.7 - cloud)
            cloud = min(1.0, max(CLEARSKY_FLOOR, cloud))

            clearsky = _clearsky_wh(stamp, site.latitude)
            ghi = clearsky * cloud

            diurnal_temp = 3.2 * math.sin(math.radians(15.0 * (stamp.hour - 9)))
            rows.append({
                "island": site.key,
                "timestamp_lst": stamp.strftime("%Y-%m-%dT%H:%M"),
                "ghi_wh_m2": round(ghi, 2),
                "ghi_clearsky_wh_m2": round(clearsky, 2),
                "wind_10m_ms": round(max(0.0, 5.4 + 2.1 * math.sin(math.radians(30 * stamp.month))
                                         + rng.uniform(-1.6, 1.6)), 2),
                "wind_50m_ms": round(max(0.0, 7.0 + 2.6 * math.sin(math.radians(30 * stamp.month))
                                         + rng.uniform(-1.9, 1.9)), 2),
                "temp_2m_c": round(27.4 + diurnal_temp - 1.4 * cloud
                                   + 1.8 * math.sin(math.radians(30 * (stamp.month - 4)))
                                   + rng.uniform(-0.5, 0.5), 2),
                "humidity_2m_pct": round(min(100.0, max(45.0, 74.0 + 14.0 * cloud
                                                        - 0.9 * diurnal_temp
                                                        + rng.uniform(-4.0, 4.0))), 2),
                "precip_mm_hr": round(
                    max(0.0, rng.expovariate(1 / 0.9) - 0.55) if (monsoon and cloud > 0.75)
                    else 0.0, 2),
                "pressure_kpa": round(101.2 - 0.5 * cloud + rng.uniform(-0.25, 0.25), 2),
            })
            stamp += timedelta(hours=1)
        by_site[site.key] = rows
    return by_site


def write_weather(by_site: dict[str, list[dict]], out_dir: Path, start: str, end: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, rows in by_site.items():
        with (out_dir / f"{key}_hourly.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=WEATHER_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    # nasa_power.check() reads requested_start/end from here to compute the expected hour count.
    manifest = {
        "source": "SYNTHETIC — not NASA POWER",
        "synthetic": True,
        "generator": "module1.data.synthetic",
        "seed": SEED,
        "temporal": "hourly",
        "requested_start": start,
        "requested_end": end,
        "site_count": len(ISLAND_SITES),
        "note": (
            "Generated by solar geometry plus a seeded cloud random walk. Contains no measurement. "
            "The spatial-resolution finding in docs/data/nasa-power-resolution.md is a property of "
            "the real pull and is NOT reproduced here: these four series are independent."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


# ------------------------------------------------------------------- stage

def build(out_dir: str | Path, start: str = DEFAULT_START, end: str = DEFAULT_END) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ledger = build_ledger()
    write_ledger(ledger, out / "ceb_generation_tidy.csv")

    weather = build_weather(start, end)
    write_weather(weather, out / "nasa_power", start, end)

    provenance = {
        "synthetic": True,
        "generator": "module1.data.synthetic",
        "seed": SEED,
        "warning": (
            "SYNTHETIC INPUT SET. Do not publish any result computed from this directory. It "
            "exists so the pipeline can be shown to run without the CEB workbook (ADR 0004), not "
            "so the system can be shown to work."
        ),
        "real_content": (
            "Annual totals per island-year are the transcribed Data_CEB_Jaffna.pdf figures already "
            "held in module1/data/validate.py as the reconciliation reference. Nothing below the "
            "year is real, and no weather value is real."
        ),
        "reproduces_deliberately": (
            f"The {DECAY_PLANT} {DECAY_YEAR} Q4 degradation, so the scenario-library gate still "
            f"has its subject. Monthly multipliers: {DECAY}."
        ),
        "does_not_reproduce": (
            "The NASA POWER spatial-resolution finding. The real pull yields one irradiance series "
            "across all four islands; these four are independent."
        ),
        "ledger_rows": len(ledger),
        "weather_rows": sum(len(v) for v in weather.values()),
    }
    (out / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] != "build":
        print(__doc__, file=sys.stderr)
        return 2

    provenance = build(argv[2])
    print(f"{argv[2]}: SYNTHETIC input set — "
          f"{provenance['ledger_rows']} ledger rows, "
          f"{provenance['weather_rows']:,} weather rows")
    print(f"  {argv[2]}/{PROVENANCE_FILE}: do not publish results computed from this")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
