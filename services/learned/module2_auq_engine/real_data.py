"""Real Eluvaitivu states, built from M1's committed calibration artifacts.

This is the item-3 input: no synthetic draws, only what M1 published. Three sources,
all in git after the M1 merge:

    data/processed/island_load_hourly.csv     hourly island demand (QUALITY_INTERPOLATED)
    data/raw/nasa_power/Eluvaitivu_hourly.csv hourly meteorology (MERRA-2 / CERES)
    data/processed/scenario_library.json      the ID/OOD labels, and their provenance

WHAT IS REAL HERE, AND WHAT IS NOT. The contract's 28 features are not all available
offline, and pretending otherwise would be the exact failure this module exists to
catch. Of the 28:

  15  real or derived from real   the four meteorology channels, the four resource
                                  channels (ghi, clear-sky index, wind at 10 m and
                                  50 m), pv_available from ghi, load and its ramp,
                                  and the four temporal encodings from the timestamp
   8  static site constants       the topology block. One aggregate node, so the
                                  asset one-hot is is_bus and the rest are zero
   5  absent, zero-filled         the electrical block. There is no SCADA and no
                                  historian (ADR 0004), which is exactly why the pin
                                  marks these QUALITY_MISSING -- so the mask already
                                  says this and no extra bookkeeping is needed

THE LABEL CANNOT BE READ OFF THESE FEATURES. That is M1's central finding, not a gap
here: `island_load_hourly.csv` sums the Eluvaitivu diesel set and hybrid plant into one
demand, and across the decay window the plant falls 73.4% while island demand falls
10.3% -- a 7.1x attenuation. The degradation is a per-plant property and this vector is
per-island. Any separation found here is therefore separation of *hourly load and
weather*, and `eluvaitivu_decay.py` runs a same-season control to say how much of it is
the season rather than the event.

The monthly label is broadcast to hourly: every hour of a flagged month inherits the
flag. The within-month transition is not resolved -- 2025-10 sits at 0.70 of baseline
and the collapse continues through it -- and nothing in the record resolves it.
"""

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from metacore_contracts.state_schema import (
    EMBEDDING_DIM,
    FEATURE_NAMES,
    SCHEMA_VERSION,
    calibration_quality,
    feature_index,
)
from state_contract import Envelope, QualityMask, ScenarioRef, StateRepresentation

REPO_ROOT = Path(__file__).resolve().parents[3]
LOAD_CSV = REPO_ROOT / "data" / "processed" / "island_load_hourly.csv"
WEATHER_CSV = REPO_ROOT / "data" / "raw" / "nasa_power" / "Eluvaitivu_hourly.csv"
LIBRARY_JSON = REPO_ROOT / "data" / "processed" / "scenario_library.json"
EVENTS_CSV = REPO_ROOT / "data" / "processed" / "events.csv"

ISLAND = "Eluvaitivu"
DECAY_SCENARIO_ID = "eluvaitivu-hybrid-decay-2025q4"

IDX = {name: feature_index(name) for name in FEATURE_NAMES}
FEATURE_QUALITY = {name: calibration_quality(name) for name in FEATURE_NAMES}

# Which of the 28 columns this builder can actually fill from the record. Published in
# the results file so a reader does not have to infer it from the code.
REAL_FEATURES = (
    "ghi_wh_m2_norm", "clearsky_index", "wind_10m_ms_norm", "wind_50m_ms_norm",
    "pv_available_kw_norm", "temp_2m_c_norm", "humidity_2m_pct_norm",
    "precip_mm_hr_norm", "pressure_kpa_norm", "load_kw_norm",
    "load_ramp_kw_per_h_norm", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
)
STATIC_FEATURES = (
    "nominal_kv_norm", "critical_load",
    "is_bus", "is_pv", "is_wind", "is_bess", "is_diesel", "is_load",
)
ABSENT_FEATURES = ("p_kw_norm", "q_kvar_norm", "voltage_pu", "soc_fraction", "asset_online")

# Site constants for the Eluvaitivu LT bus, from data/external/ceb_jaffna/README.md and
# docs/data/ceb-jaffna-baseline.md: 400 V distribution against an 11 kV system base, no
# hospital or other critical load on this feeder.
NOMINAL_KV, SYSTEM_BASE_KV = 0.4, 11.0


def _fnum(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_load():
    """{timestamp_lst: load_kw} for Eluvaitivu."""
    out = {}
    with LOAD_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["island"] == ISLAND:
                out[row["timestamp_lst"]] = float(row["load_kw"])
    return out


def read_weather():
    """{timestamp_lst: {column: float}} for the Eluvaitivu MERRA-2 / CERES cell."""
    out = {}
    with WEATHER_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["timestamp_lst"]] = {k: _fnum(v) for k, v in row.items()
                                         if k not in ("island", "timestamp_lst")}
    return out


def load_library():
    """The scenario library, plus the decay episode pulled out for convenience."""
    library = json.loads(LIBRARY_JSON.read_text())
    decay = next(s for s in library["scenarios"] if s["scenario_id"] == DECAY_SCENARIO_ID)
    return library, decay


def read_events():
    with EVENTS_CSV.open(newline="") as fh:
        return list(csv.DictReader(fh))


def months_in(start_month, end_month):
    """Inclusive YYYY-MM range -> the month prefixes it covers."""
    start = datetime.strptime(start_month, "%Y-%m")
    end = datetime.strptime(end_month, "%Y-%m")
    months, year, month = [], start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append(f"{year}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return months


class SiteScaler:
    """Per-unit and z-score statistics for the site, over the whole record.

    Fitted on all 17,544 hours rather than on a split. These are site properties -- the
    island's peak demand, the cell's maximum wind -- exactly what the pin's units say
    ("p.u. of island peak", "p.u. of site max"), and they carry no label. The EDL's own
    Normalizer is still fitted on the training window only; that is the one that would
    leak if it were not.
    """

    def __init__(self, load, weather):
        self.load_peak = max(load.values())
        cols = ("ghi_wh_m2", "wind_10m_ms", "wind_50m_ms", "precip_mm_hr")
        self.maxima = {c: max(w[c] for w in weather.values()) or 1.0 for c in cols}
        for c in ("temp_2m_c", "pressure_kpa"):
            values = np.array([w[c] for w in weather.values()])
            setattr(self, f"{c}_mu", float(values.mean()))
            setattr(self, f"{c}_sd", float(values.std()) or 1.0)


def build_states(months, scenario, scaler, load, weather, ramp_lookup):
    """Hourly StateRepresentations for the given months, tagged with `scenario`."""
    states, kept = [], []
    for stamp_text in sorted(load):
        if stamp_text[:7] not in months or stamp_text not in weather:
            continue
        kept.append(stamp_text)
        w = weather[stamp_text]
        stamp = datetime.fromisoformat(stamp_text)
        x = np.zeros(len(FEATURE_NAMES), np.float32)

        ghi = w["ghi_wh_m2"] / scaler.maxima["ghi_wh_m2"]
        x[IDX["ghi_wh_m2_norm"]] = ghi
        x[IDX["clearsky_index"]] = (w["ghi_wh_m2"] / w["ghi_clearsky_wh_m2"]
                                    if w["ghi_clearsky_wh_m2"] > 0 else 0.0)
        x[IDX["wind_10m_ms_norm"]] = w["wind_10m_ms"] / scaler.maxima["wind_10m_ms"]
        x[IDX["wind_50m_ms_norm"]] = w["wind_50m_ms"] / scaler.maxima["wind_50m_ms"]
        x[IDX["pv_available_kw_norm"]] = ghi          # p.u. of panel rating, so ghi in p.u.

        x[IDX["temp_2m_c_norm"]] = (w["temp_2m_c"] - scaler.temp_2m_c_mu)/scaler.temp_2m_c_sd
        x[IDX["humidity_2m_pct_norm"]] = w["humidity_2m_pct"] / 100.0
        x[IDX["precip_mm_hr_norm"]] = w["precip_mm_hr"] / scaler.maxima["precip_mm_hr"]
        x[IDX["pressure_kpa_norm"]] = ((w["pressure_kpa"] - scaler.pressure_kpa_mu)
                                       / scaler.pressure_kpa_sd)

        x[IDX["load_kw_norm"]] = load[stamp_text] / scaler.load_peak
        x[IDX["load_ramp_kw_per_h_norm"]] = ramp_lookup.get(stamp_text, 0.0)/scaler.load_peak

        x[IDX["hour_sin"]] = math.sin(2*math.pi*stamp.hour/24)
        x[IDX["hour_cos"]] = math.cos(2*math.pi*stamp.hour/24)
        doy = stamp.timetuple().tm_yday
        x[IDX["doy_sin"]] = math.sin(2*math.pi*doy/365)
        x[IDX["doy_cos"]] = math.cos(2*math.pi*doy/365)

        x[IDX["nominal_kv_norm"]] = NOMINAL_KV/SYSTEM_BASE_KV
        x[IDX["is_bus"]] = 1.0
        # critical_load, the other five is_* flags and the whole electrical block stay 0.

        states.append(_state(x, stamp_text, scenario))
    return states, kept


def _state(x, stamp_text, scenario):
    envelope = Envelope(
        schema_version=SCHEMA_VERSION,
        emitted_at=datetime.fromisoformat(stamp_text).timestamp(),
        producer="module1",
        scenario=scenario,
    )
    quality = QualityMask.from_per_feature([FEATURE_QUALITY[f] for f in FEATURE_NAMES])
    return StateRepresentation(
        envelope=envelope,
        node_count=1,
        embedding_dim=EMBEDDING_DIM,
        # No learned embedding exists offline -- M1's GNN is not trained yet. Zeros rather
        # than noise, so anyone who trains on `embedding` by mistake gets an obvious
        # failure instead of a plausible-looking result.
        node_embedding=np.zeros((1, EMBEDDING_DIM), np.float32),
        graph_embedding=np.zeros(EMBEDDING_DIM, np.float32),
        feature_names=list(FEATURE_NAMES),
        node_features=x.reshape(1, -1),
        quality=quality,
        degraded=False,          # every channel that exists is present; none is lost
    )


def load_ramps(load):
    """First difference of hourly load, keyed by the later timestamp."""
    stamps = sorted(load)
    return {b: load[b] - load[a] for a, b in zip(stamps[:-1], stamps[1:], strict=True)}


def scenario_ref(entry):
    """The real ScenarioRef for a library entry -- the contract-carried label."""
    return ScenarioRef(entry["scenario_id"], "1.0.0", bool(entry["out_of_distribution"]))


# ------------------------------------------------------- per-plant monthly

TIDY_CSV = REPO_ROOT / "data" / "processed" / "ceb_generation_tidy.csv"

DECAY_PLANT = "Eluvaitivu-Hybrid"
DECAY_MONTHS = ("2025-10", "2025-11", "2025-12")

# The per-plant monthly operating signature. Deliberately NOT the detection rule's
# quantity alone: `energy_rel` is the plant-relative output the rule thresholds, and the
# rest describe how the plant burned fuel to produce it. `eluvaitivu_decay.py` reruns the
# whole experiment with `energy_rel` removed, to separate "recovers the rule's window"
# from "reads the rule's own input back".
PLANT_FEATURES = ("energy_rel", "diesel_rel", "sfc_l_per_kwh",
                  "fuel_cost_rs_per_kwh", "month_sin", "month_cos")
PLANT_FEATURES_NO_ENERGY = tuple(f for f in PLANT_FEATURES if f != "energy_rel")


def read_plant_months():
    """[{plant, month, units_kwh, diesel_l, sfc, fuel_cost_per_kwh}] from the ledger.

    These rows are QUALITY_OBSERVED -- meter readings, not downscaled estimates. Almost
    nothing else Module 1 publishes can say that, and it is the reason the degradation is
    visible here and nowhere else.
    """
    month_num = {name: i + 1 for i, name in enumerate(
        ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))}
    rows = []
    with TIDY_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "plant": row["island_system"],
                "month": f"{row['year']}-{month_num[row['month']]:02d}",
                "month_num": month_num[row["month"]],
                "units_kwh": _fnum(row["units_kwh"]),
                "diesel_l": _fnum(row["diesel_l"]),
                "sfc_l_per_kwh": _fnum(row["sfc_l_per_kwh"]),
                "fuel_cost_rs_per_kwh": _fnum(row["fuel_cost_rs_per_kwh"]),
            })
    return rows


def plant_month_matrix(rows, baselines, columns=PLANT_FEATURES):
    """Plant-months -> (n, len(columns)) feature matrix.

    `energy_rel` and `diesel_rel` are per-unit of the plant's own baseline, so five plants
    of very different sizes land on a comparable scale. Baselines are computed from the
    NOMINAL months only (see `plant_baselines`), never from the window under test.
    """
    out = np.zeros((len(rows), len(columns)), np.float32)
    for i, row in enumerate(rows):
        base = baselines[row["plant"]]
        values = {
            "energy_rel": row["units_kwh"]/base["units_kwh"],
            "diesel_rel": row["diesel_l"]/base["diesel_l"],
            "sfc_l_per_kwh": row["sfc_l_per_kwh"],
            "fuel_cost_rs_per_kwh": row["fuel_cost_rs_per_kwh"]/100.0,
            "month_sin": math.sin(2*math.pi*row["month_num"]/12),
            "month_cos": math.cos(2*math.pi*row["month_num"]/12),
        }
        for j, name in enumerate(columns):
            out[i, j] = values[name]
    return out


def plant_baselines(rows):
    """Median energy and diesel per plant, over the nominal months handed in.

    Median rather than mean, and over nominal months only: a baseline that included the
    collapse would shrink the very anomaly it is used to measure.
    """
    per_plant = {}
    for row in rows:
        per_plant.setdefault(row["plant"], {"units_kwh": [], "diesel_l": []})
        per_plant[row["plant"]]["units_kwh"].append(row["units_kwh"])
        per_plant[row["plant"]]["diesel_l"].append(row["diesel_l"])
    return {plant: {k: (float(np.median(v)) or 1.0) for k, v in cols.items()}
            for plant, cols in per_plant.items()}


def split_plant_months(rows):
    """(nominal, decay) -- the three flagged Eluvaitivu-Hybrid months against the rest."""
    decay = [r for r in rows if r["plant"] == DECAY_PLANT and r["month"] in DECAY_MONTHS]
    nominal = [r for r in rows if not (r["plant"] == DECAY_PLANT
                                       and r["month"] in DECAY_MONTHS)]
    return nominal, decay
