"""Stage: monthly CEB energy -> hourly island load, with the assumption boundary made explicit.

The resolution gap this closes is the central one in the project. The ledger records one energy
figure per island per month; the contracts in `packages/contracts/proto` carry a state snapshot
with an `update_period_s`, and M3 gates a genset start/stop decision. Nothing between a monthly
total and a dispatch decision exists in the measured record, and per ADR 0004 it never will --
there is no SCADA, no historian, and no half-hourly telemetry to back-fill from.

So this stage does not "recover" the hourly load. It constructs one, and is built so that the
line between what is measured and what is assumed stays legible downstream:

  MEASURED, and enforced exactly
    - Monthly energy per island. The hourly series is normalised per month, so it reproduces the
      ledger total to float tolerance no matter what the shape does. A wrong shape stays a wrong
      shape; it cannot become a wrong energy balance.
    - Installed genset capacity per island-year (`INSTALLED_KVA`), as an absolute ceiling.
    - Nainativu's reported maximum demand, 460 kVA against 880 kVA installed. This is the only
      observation of load *shape* anywhere in the dataset, and it is what `PEAKINESS` is
      calibrated against rather than chosen to look plausible.

  ASSUMED, and labelled
    - The intra-day curve (`DIURNAL`), the weekday/weekend split, and the temperature
      coefficient. Every emitted row carries QUALITY_INTERPOLATED. Per ADR 0004 that is not
      decoration: M2's contribution is detecting when the state it was handed is untrustworthy,
      and a synthetic value labelled as observed defeats the experiment.

Why the fuel column does not pin the shape, since it looks like it should: fitting the Willans
line `fuel = a*hours + b*energy` per island gives an excellent fit (Nainativu R^2 = 0.975), and
specific fuel consumption is strongly anti-correlated with monthly energy -- the part-load
efficiency signature of a real genset. But that model is *linear in energy*, so at monthly
aggregation every hourly profile with the same monthly total predicts the same fuel burn. The SFC
correlation is the Willans line restated, not independent information about the curve. The fit is
computed and published anyway, because `a` and `b` are exactly the no-load and marginal fuel rates
M3 needs for its cost objective, and because a physically plausible positive `a` at R^2 = 0.975 is
evidence that Nainativu runs continuously rather than on an evening schedule.

CIRCULARITY WARNING. This artifact is for simulation, dispatch evaluation and M4 power-flow --
not for training or evaluating a load forecaster. Load here is generated from weather through a
known coefficient; a model trained to predict this load from that weather recovers the
coefficient and reports it as skill. Any M1 forecasting result must state which series it used.

Usage:
    python -m module1.data.load downscale <tidy.csv> <weather_dir> <out.csv>
    python -m module1.data.load validate <out.csv> <tidy.csv>
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

QUALITY_INTERPOLATED = "QUALITY_INTERPOLATED"

# Artifact identity, written into load_parameters.json so a consumer can detect a shape it was not
# built for. Distinct from the StateRepresentation contract version in
# metacore_contracts.state_schema -- this versions a calibration artifact, not a wire message, and
# conflating the two would let a reader check the wrong one. Bump the minor on an added field, the
# major on a removed or re-meaning one. Kept as a literal rather than imported: these stages are
# stdlib-only by design (ADR 0004) and must not acquire protobuf to state their own version.
ARTIFACT_NAME = "island_load_hourly"
ARTIFACT_VERSION = "1.0.0"

# Ledger generating system -> the island whose load it serves. Eluvaitivu's two plants feed one
# load: that is what makes the Oct-Dec 2025 hybrid collapse legible as a clean substitution, with
# island demand flat while the split swings. Load is modelled per island; the supply split stays
# in the ledger.
SYSTEM_TO_ISLAND = {
    "Analaithivu": "Analaitivu",          # ledger spelling differs from the weather site name
    "Eluvaitivu-Diesel": "Eluvaitivu",
    "Eluvaitivu-Hybrid": "Eluvaitivu",
    "Delft-Neduntivu": "Delft-Neduntivu",
    "Nainativu": "Nainativu",
}

# Installed generating capacity in kVA per island-year, summed over the plants on that island.
# From `data/external/ceb_jaffna/README.md`, which records a year-specific fleet: 2024 lists
# Analaithivu with 200 kVA x2 rather than 100 kVA x2, and Delft and Nainativu each with an extra
# 380 kVA unit. Used as a ceiling gate, so getting the year wrong would hide a violation.
INSTALLED_KVA = {
    ("2024", "Analaitivu"): 650.0,        # 250 + 200 x2
    ("2025", "Analaitivu"): 450.0,        # 250 + 100 x2
    ("2024", "Eluvaitivu"): 130.0,        # 100 diesel + 30 hybrid
    ("2025", "Eluvaitivu"): 130.0,
    ("2024", "Delft-Neduntivu"): 1210.0,  # 250 x2 + 330 + 380
    ("2025", "Delft-Neduntivu"): 830.0,   # 250 x2 + 330
    ("2024", "Nainativu"): 1260.0,        # 250 x2 + 380 x2
    ("2025", "Nainativu"): 880.0,         # 250 x2 + 380
}

# The one measured statement about load shape in the whole dataset (interview + PDF): Nainativu
# peaks at 460 kVA apparent against 880 kVA installed. Everything about peakiness is calibrated
# to reproduce it.
REPORTED_MAX_DEMAND_KVA = {"Nainativu": 460.0}
ANCHOR_YEAR = "2025"

# Distribution power factor. Not measured -- the ledger records kWh and the demand figure is kVA,
# so a factor is unavoidable to relate them. 0.85 is the usual assumption for a residential LV
# feeder; the gate checks a band rather than the point value so the result does not hinge on it.
POWER_FACTOR = 0.85
POWER_FACTOR_BAND = (0.80, 0.90)

# Normalised 24-hour residential profile for a tropical island microgrid, index 0 = 00:00 local
# solar time. Shape, not level: it is renormalised per month against the ledger. Deep overnight
# trough, modest morning rise, a warm midday plateau (fans), and a sharp lighting-plus-television
# evening peak. Peak-to-mean of this vector is ~2.0, which is what the Nainativu anchor implies
# (460 kVA at pf 0.85 = 391 kW against a 191 kW peak-month mean = 2.04).
DIURNAL = (
    0.55, 0.50, 0.48, 0.47, 0.48, 0.55,   # 00-05  overnight
    0.70, 0.80, 0.82, 0.80, 0.80, 0.82,   # 06-11  morning
    0.85, 0.85, 0.83, 0.80, 0.80, 0.88,   # 12-17  midday plateau
    1.35, 1.70, 1.75, 1.55, 1.10, 0.75,   # 18-23  evening peak
)

# Weekday index 0 = Monday. Islands with no industrial load vary little across the week; keeping
# the factor near unity says that rather than inventing a commercial weekday pattern.
DOW_FACTOR = (0.99, 0.99, 0.99, 0.99, 1.00, 1.01, 1.03)

# Fractional load change per degree C above the island's median temperature -- fan and
# refrigeration response. Assumed. The weather it keys off is measured, but note that NASA POWER
# does not resolve these islands (one irradiance series and two meteorological series across four
# sites, see docs/data/nasa-power-resolution.md), so this term is near-identical island to island.
TEMP_BETA = 0.015

# Exponent applied to the diurnal vector to tune peak-to-mean, solved against the anchor rather
# than set by hand. Bisection bounds; 1.0 is the DIURNAL vector as written.
PEAKINESS_BOUNDS = (0.20, 4.00)

MONTH_NUM_DAYS = {1: 31, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31,
                  8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

FIELDS = ("island", "timestamp_lst", "load_kw", "quality")

TOLERANCE_KWH = 0.5     # float accumulation over ~17.5k hours, not a modelling allowance
LOAD_FACTOR_BAND = (0.20, 0.75)


# ------------------------------------------------------------------ inputs

def read_monthly_island_energy(tidy_csv: str | Path) -> dict[tuple[str, int, str], float]:
    """Ledger rows -> kWh per (year, month_num, island), summed over the plants on each island."""
    totals: dict[tuple[str, int, str], float] = defaultdict(float)
    with Path(tidy_csv).open(newline="") as fh:
        for row in csv.DictReader(fh):
            if not row["units_kwh"]:
                continue
            island = SYSTEM_TO_ISLAND[row["island_system"]]
            totals[(row["year"], int(row["month_num"]), island)] += float(row["units_kwh"])
    return dict(totals)


def read_weather(weather_dir: str | Path) -> dict[str, dict[str, float]]:
    """Island -> {timestamp_lst: temp_2m_c}. Missing temperatures are dropped, not zero-filled."""
    out: dict[str, dict[str, float]] = {}
    for path in sorted(Path(weather_dir).glob("*_hourly.csv")):
        island = path.name.removesuffix("_hourly.csv")
        series: dict[str, float] = {}
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                value = row.get("temp_2m_c")
                if value not in (None, "", "None"):
                    series[row["timestamp_lst"]] = float(value)
        out[island] = series
    return out


# ------------------------------------------------------- Willans decomposition

def willans_fit(tidy_csv: str | Path) -> dict[str, dict[str, float]]:
    """Least-squares `fuel_l = a*hours + b*energy_kwh` per generating system.

    `a` is the no-load burn in L/h and `b` the marginal rate in L/kWh -- the two coefficients M3
    needs to price a genset start against the fuel it saves. Shape-blind by construction (see the
    module docstring); published for the cost model, not used to build the profile.
    """
    obs: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    with Path(tidy_csv).open(newline="") as fh:
        for row in csv.DictReader(fh):
            if not row["units_kwh"] or not row["diesel_l"]:
                continue
            energy, fuel = float(row["units_kwh"]), float(row["diesel_l"])
            if energy <= 0:
                continue
            hours = 24.0 * days_in_month(row["year"], int(row["month_num"]))
            obs[row["island_system"]].append((hours, energy, fuel))

    fits: dict[str, dict[str, float]] = {}
    for system, points in obs.items():
        if len(points) < 6:
            continue
        s_hh = sum(h * h for h, _, _ in points)
        s_ee = sum(e * e for _, e, _ in points)
        s_he = sum(h * e for h, e, _ in points)
        s_hf = sum(h * f for h, _, f in points)
        s_ef = sum(e * f for _, e, f in points)
        det = s_hh * s_ee - s_he * s_he
        if abs(det) < 1e-9:
            continue
        a = (s_hf * s_ee - s_ef * s_he) / det
        b = (s_hh * s_ef - s_he * s_hf) / det
        mean_f = statistics.fmean(f for _, _, f in points)
        ss_res = sum((f - (a * h + b * e)) ** 2 for h, e, f in points)
        ss_tot = sum((f - mean_f) ** 2 for _, _, f in points)
        fits[system] = {
            "no_load_l_per_h": round(a, 4),
            "marginal_l_per_kwh": round(b, 6),
            "r2": round(1 - ss_res / ss_tot, 4) if ss_tot else None,
            "residual_pct_of_mean": round(100 * (ss_res / len(points)) ** 0.5 / mean_f, 2),
            "months": len(points),
            # A negative no-load rate is not physical. It means the continuous-running assumption
            # is wrong for this plant -- it is dispatched intermittently, so `hours` overstates
            # its running time. Flagged rather than clamped.
            "continuous_running_consistent": a > 0,
        }
    return fits


# ------------------------------------------------------------------ profile

def days_in_month(year: str, month_num: int) -> int:
    if month_num == 2:
        y = int(year)
        return 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28
    return MONTH_NUM_DAYS[month_num]


def _shape(stamp: datetime, temp_c: float | None, median_c: float, peakiness: float) -> float:
    """Unnormalised relative load for one hour. Level is meaningless; only ratios are used."""
    base = DIURNAL[stamp.hour] ** peakiness
    base *= DOW_FACTOR[stamp.weekday()]
    if temp_c is not None:
        # Clamped so a reanalysis outlier cannot drive the factor negative.
        base *= max(0.1, 1.0 + TEMP_BETA * (temp_c - median_c))
    return base


def build_island_series(
    island: str,
    monthly_kwh: dict[tuple[str, int, str], float],
    temps: dict[str, float],
    peakiness: float,
) -> list[dict]:
    """Hourly rows for one island, normalised per month to the measured ledger total."""
    median_c = statistics.median(temps.values()) if temps else 0.0

    by_month: dict[tuple[str, int], list[tuple[str, datetime, float]]] = defaultdict(list)
    for stamp_text in temps:
        stamp = datetime.fromisoformat(stamp_text)
        by_month[(str(stamp.year), stamp.month)].append(
            (stamp_text, stamp, temps[stamp_text])
        )

    rows: list[dict] = []
    for (year, month_num), hours in sorted(by_month.items()):
        energy = monthly_kwh.get((year, month_num, island))
        if energy is None:
            continue
        weights = [_shape(s, t, median_c, peakiness) for _, s, t in hours]
        total = sum(weights)
        if total <= 0:
            continue
        # Energy is conserved here and nowhere else: each hour takes its share of the measured
        # month. kWh over a one-hour step is numerically kW.
        for (stamp_text, _, _), weight in zip(hours, weights, strict=True):
            rows.append({
                "island": island,
                "timestamp_lst": stamp_text,
                "load_kw": round(energy * weight / total, 4),
                "quality": QUALITY_INTERPOLATED,
            })
    rows.sort(key=lambda r: r["timestamp_lst"])
    return rows


def calibrate_peakiness(
    monthly_kwh: dict[tuple[str, int, str], float],
    weather: dict[str, dict[str, float]],
    power_factor: float = POWER_FACTOR,
) -> tuple[float, dict]:
    """Solve the diurnal exponent so the anchor island reproduces its reported maximum demand.

    One island has a measured peak, so one parameter is identifiable. The solved exponent is then
    applied to all four -- an assumption of similar customer mix, stated in the report rather than
    hidden in a constant.
    """
    island = next(iter(REPORTED_MAX_DEMAND_KVA))
    target_kw = REPORTED_MAX_DEMAND_KVA[island] * power_factor

    def peak_for(exponent: float) -> float:
        rows = build_island_series(island, monthly_kwh, weather[island], exponent)
        return max(r["load_kw"] for r in rows if r["timestamp_lst"].startswith(ANCHOR_YEAR))

    low, high = PEAKINESS_BOUNDS
    for _ in range(60):
        mid = (low + high) / 2
        if peak_for(mid) < target_kw:
            low = mid
        else:
            high = mid
    exponent = (low + high) / 2
    achieved = peak_for(exponent)
    return exponent, {
        "anchor_island": island,
        "anchor_year": ANCHOR_YEAR,
        "reported_max_demand_kva": REPORTED_MAX_DEMAND_KVA[island],
        "power_factor_assumed": power_factor,
        "target_peak_kw": round(target_kw, 2),
        "achieved_peak_kw": round(achieved, 2),
        "solved_peakiness_exponent": round(exponent, 5),
    }


# ------------------------------------------------------------------ stage

def downscale(tidy_csv: str | Path, weather_dir: str | Path) -> tuple[list[dict], dict]:
    monthly = read_monthly_island_energy(tidy_csv)
    weather = read_weather(weather_dir)

    missing = {i for _, _, i in monthly} - set(weather)
    if missing:
        raise SystemExit(f"no weather series for: {', '.join(sorted(missing))}")

    peakiness, calibration = calibrate_peakiness(monthly, weather)

    rows: list[dict] = []
    per_island: dict[str, dict] = {}
    for island in sorted(set(i for _, _, i in monthly)):
        series = build_island_series(island, monthly, weather[island], peakiness)
        rows.extend(series)
        loads = [r["load_kw"] for r in series]
        per_island[island] = {
            "hours": len(series),
            "mean_kw": round(statistics.fmean(loads), 2),
            "peak_kw": round(max(loads), 2),
            "min_kw": round(min(loads), 2),
            "load_factor": round(statistics.fmean(loads) / max(loads), 4),
        }

    manifest = {
        "artifact": ARTIFACT_NAME,
        "version": ARTIFACT_VERSION,
        "produced_by": "module1.data.load",
        "stage": "load_downscale",
        "quality": QUALITY_INTERPOLATED,
        "measured_inputs": {
            "monthly_energy": "data/processed/ceb_generation_tidy.csv (CEB ledger, reconciled)",
            "temperature": "data/raw/nasa_power/*_hourly.csv (MERRA-2)",
            "max_demand": REPORTED_MAX_DEMAND_KVA,
            "installed_kva": {f"{y}/{i}": v for (y, i), v in INSTALLED_KVA.items()},
        },
        "assumptions": {
            "diurnal_profile": list(DIURNAL),
            "dow_factor": list(DOW_FACTOR),
            "temp_beta_per_c": TEMP_BETA,
            "power_factor": POWER_FACTOR,
            "note": "Intra-day shape is constructed, not measured. Every row is "
                    f"{QUALITY_INTERPOLATED}. Not valid as training data for a load forecaster "
                    "-- see the circularity warning in module1/data/load.py.",
        },
        "calibration": calibration,
        "willans_fuel_model": willans_fit(tidy_csv),
        "islands": per_island,
    }
    return rows, manifest


def write_csv(rows: list[dict], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------- gate

def check(load_csv: str | Path, tidy_csv: str | Path) -> list[str]:
    """Fail the pipeline on anything the measured record can actually contradict."""
    failures: list[str] = []
    with Path(load_csv).open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ["load table is empty"]

    monthly = read_monthly_island_energy(tidy_csv)
    seen: dict[tuple[str, int, str], float] = defaultdict(float)
    hours: dict[str, int] = defaultdict(int)
    peaks: dict[tuple[str, str], float] = defaultdict(float)
    loads: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        if row["quality"] != QUALITY_INTERPOLATED:
            failures.append(
                f"{row['island']} {row['timestamp_lst']}: quality is {row['quality']!r}, "
                f"expected {QUALITY_INTERPOLATED} -- downscaled load is never observed"
            )
        kw = float(row["load_kw"])
        if kw <= 0:
            failures.append(f"{row['island']} {row['timestamp_lst']}: load {kw} is not positive")
        stamp = datetime.fromisoformat(row["timestamp_lst"])
        year = str(stamp.year)
        seen[(year, stamp.month, row["island"])] += kw
        hours[row["island"]] += 1
        peaks[(year, row["island"])] = max(peaks[(year, row["island"])], kw)
        loads[row["island"]].append(kw)

    # 1. Energy conservation against the ledger. The one hard constraint.
    for key, expected in monthly.items():
        actual = seen.get(key)
        if actual is None:
            failures.append(f"{key[0]}-{key[1]:02d} {key[2]}: absent from load table")
        elif abs(actual - expected) > TOLERANCE_KWH:
            failures.append(
                f"{key[0]}-{key[1]:02d} {key[2]}: ledger {expected:,.1f} kWh != "
                f"downscaled {actual:,.1f} kWh"
            )
    for key in set(seen) - set(monthly):
        failures.append(f"{key[0]}-{key[1]:02d} {key[2]}: in load table but not in the ledger")

    # 2. Complete hourly coverage of 2024-2025 (leap year: 731 days).
    for island, count in sorted(hours.items()):
        if count != 17_544:
            failures.append(f"{island}: {count} hours, expected 17,544")

    # 3. Peak within installed capacity, across the whole power-factor band. Checked at the
    #    optimistic end -- if the peak fits even at pf 0.90 it fits everywhere in the band.
    for (year, island), peak_kw in sorted(peaks.items()):
        installed = INSTALLED_KVA.get((year, island))
        if installed is None:
            failures.append(f"{year} {island}: no installed capacity recorded")
            continue
        ceiling = installed * POWER_FACTOR_BAND[1]
        if peak_kw > ceiling:
            failures.append(
                f"{year} {island}: peak {peak_kw:,.1f} kW exceeds {installed:,.0f} kVA "
                f"installed ({ceiling:,.1f} kW at pf {POWER_FACTOR_BAND[1]})"
            )

    # 4. The anchor is reproduced, within the power-factor band it was solved under.
    for island, kva in REPORTED_MAX_DEMAND_KVA.items():
        peak_kw = peaks.get((ANCHOR_YEAR, island))
        low, high = (kva * pf for pf in POWER_FACTOR_BAND)
        if peak_kw is None:
            failures.append(f"{ANCHOR_YEAR} {island}: no peak to compare with reported demand")
        elif not low <= peak_kw <= high:
            failures.append(
                f"{ANCHOR_YEAR} {island}: peak {peak_kw:,.1f} kW outside the reported "
                f"{kva:,.0f} kVA band [{low:,.1f}, {high:,.1f}] kW"
            )

    # 5. Load factor physically plausible for a residential island microgrid.
    for island, series in sorted(loads.items()):
        lf = statistics.fmean(series) / max(series)
        if not LOAD_FACTOR_BAND[0] <= lf <= LOAD_FACTOR_BAND[1]:
            failures.append(
                f"{island}: load factor {lf:.3f} outside plausible {LOAD_FACTOR_BAND}"
            )

    # 6. The sidecar identifies itself. A consumer that reads load_parameters.json to decide how
    #    to interpret the CSV needs to know which shape it is holding; an unversioned manifest
    #    forces it to guess, and a guess that happens to work today breaks silently on the next
    #    change. Cheap to check here, expensive to discover in someone else's service.
    manifest_path = Path(load_csv).with_name("load_parameters.json")
    if not manifest_path.exists():
        failures.append(f"{manifest_path.name}: absent -- the load table has no parameter sidecar")
    else:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("artifact") != ARTIFACT_NAME:
            failures.append(
                f"{manifest_path.name}: artifact {manifest.get('artifact')!r}, "
                f"expected {ARTIFACT_NAME!r}"
            )
        if manifest.get("version") != ARTIFACT_VERSION:
            failures.append(
                f"{manifest_path.name}: version {manifest.get('version')!r}, "
                f"expected {ARTIFACT_VERSION!r} -- regenerate with `task data`"
            )
    return failures


# -------------------------------------------------------------------- cli

def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in ("downscale", "validate"):
        print(__doc__, file=sys.stderr)
        return 2

    if argv[1] == "downscale":
        if len(argv) != 5:
            print(__doc__, file=sys.stderr)
            return 2
        rows, manifest = downscale(argv[2], argv[3])
        write_csv(rows, argv[4])
        manifest_path = Path(argv[4]).with_name("load_parameters.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        cal = manifest["calibration"]
        print(f"{argv[4]}: {len(rows):,} hourly rows across {len(manifest['islands'])} islands")
        print(f"  peakiness {cal['solved_peakiness_exponent']} solved against "
              f"{cal['anchor_island']} {cal['reported_max_demand_kva']:.0f} kVA "
              f"-> {cal['achieved_peak_kw']:.1f} kW (target {cal['target_peak_kw']:.1f})")
        print(f"  {manifest_path}: assumptions, Willans fuel model, per-island statistics")
        return 0

    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    failures = check(argv[2], argv[3])
    if failures:
        print(f"LOAD GATE FAILED — {len(failures)} problem(s):", file=sys.stderr)
        for f in failures[:40]:
            print(f"  {f}", file=sys.stderr)
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more", file=sys.stderr)
        return 1
    print("load gate OK — energy conserved against the ledger, coverage complete, "
          "peaks within installed capacity, reported max demand reproduced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
