"""The v0 vulnerability target -- a constructed proxy, never an observation.

No dataset anywhere labels how vulnerable a microgrid asset is, so the target is built here from
three components, each traceable to a measured input:

* **Hazard exposure** ``H(t)`` per island, from NASA POWER 50 m wind and rainfall. Each driver is
  scaled between its *training-period* median and 99.9th percentile, and the two are combined as a
  probabilistic OR, ``1 - (1 - h_wind)(1 - h_rain)``: either hazard alone can take a node down.
  Thresholds come from the training split only, so the test period cannot leak into the label scale.
  The scaling is ratio-based, so the precipitation unit mislabel in ``data.nasa_power`` (stored
  values are a mm/day rate under a mm/hour name) does not affect it.
* **Fragility** ``F(t)`` per node: an asset-type base value, raised for the Eluvaitivu hybrid
  assets by the *previous* month's ledger decay (``1 - energy / median``). Lagged one month because
  an operator only has last month's ledger -- using the current month would leak the label.
* **Electrical stress** ``S(t)`` per island: constructed hourly load over installed genset capacity
  at 0.85 power factor. This inherits the downscaled load's ``QUALITY_INTERPOLATED`` status.

``V = w_H * E_type * H + w_F * F + w_S * S``, where ``E_type`` is how much of the outdoor hazard
reaches an asset (an overhead LV bus takes all of it; a genset in its powerhouse much less).
The type constants are **assumptions**, written down here and in ``ASSUMPTIONS`` so the report can
state them; results are reported under every weighting in ``WEIGHTINGS`` rather than one asserted
choice (proposal SO3).
"""

from __future__ import annotations

import numpy as np

# Fraction of the outdoor hazard that reaches each asset type. The interview grounds the ordering:
# fuses on the overhead LV network drop on wind and rain; gensets and the battery room are indoors.
EXPOSURE: dict[str, float] = {"bus": 1.0, "wind": 1.0, "pv": 0.8, "diesel": 0.4, "bess": 0.3}

# Static fragility by asset type. Gensets are old and deliberately under-protected (250 A MCCB on a
# 250 kVA set); the battery bank is the grid-forming reference whose loss collapses the hybrid.
BASE_FRAGILITY: dict[str, float] = {"bus": 0.5, "wind": 0.9, "pv": 0.4, "diesel": 0.4, "bess": 0.5}

# Assets whose fragility tracks the Eluvaitivu hybrid plant's ledger decay.
HYBRID_ASSETS = frozenset({"pv", "bess", "wind"})

WEIGHTINGS: dict[str, tuple[float, float, float]] = {
    "equal": (1 / 3, 1 / 3, 1 / 3),
    "hazard_heavy": (0.50, 0.25, 0.25),
    "stress_heavy": (0.25, 0.25, 0.50),
}

POWER_FACTOR = 0.85

ASSUMPTIONS = {
    "exposure_by_type": EXPOSURE,
    "base_fragility_by_type": BASE_FRAGILITY,
    "hazard_scaling": "training-split median .. 99.9th percentile, clipped to [0, 1]",
    "hazard_combination": "1 - (1 - h_wind)(1 - h_rain)",
    "fragility_dynamic": "hybrid assets: max(base, 1 - previous-month energy / median)",
    "stress": f"load_kw / (installed_kva x {POWER_FACTOR}), clipped to [0, 1]",
    "weightings": WEIGHTINGS,
    "quality": "QUALITY_INTERPOLATED -- a constructed proxy, not an observation",
}


def scale_between(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear map of [lo, hi] onto [0, 1], clipped. Degenerate ranges map to 0."""
    if hi <= lo:
        return np.zeros_like(x, dtype=float)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def hazard(wind50: np.ndarray, rain: np.ndarray, ref: dict[str, tuple[float, float]]) -> np.ndarray:
    """Island hazard exposure in [0, 1]. ``ref`` holds (lo, hi) per driver, from training rows."""
    hw = scale_between(wind50, *ref["wind_50m_ms"])
    hr = scale_between(rain, *ref["precip_mm_hr"])
    return 1.0 - (1.0 - hw) * (1.0 - hr)


def hazard_reference(wind50: np.ndarray, rain: np.ndarray) -> dict[str, tuple[float, float]]:
    """Scaling bounds computed on training rows only (the caller passes training rows)."""
    return {
        "wind_50m_ms": (float(np.median(wind50)), float(np.percentile(wind50, 99.9))),
        "precip_mm_hr": (float(np.median(rain)), float(np.percentile(rain, 99.9))),
    }


def fragility(asset: str, hybrid_decay_lagged: np.ndarray) -> np.ndarray:
    """Per-hour fragility for one node. ``hybrid_decay_lagged`` is 1 - ratio, already lagged."""
    base = np.full_like(hybrid_decay_lagged, BASE_FRAGILITY[asset], dtype=float)
    if asset in HYBRID_ASSETS:
        return np.maximum(base, np.clip(hybrid_decay_lagged, 0.0, 1.0))
    return base


def stress(load_kw: np.ndarray, installed_kva: np.ndarray) -> np.ndarray:
    return np.clip(load_kw / (installed_kva * POWER_FACTOR), 0.0, 1.0)


def combine(
    h: np.ndarray, f: np.ndarray, s: np.ndarray, asset: str, weighting: str
) -> np.ndarray:
    w_h, w_f, w_s = WEIGHTINGS[weighting]
    return w_h * EXPOSURE[asset] * h + w_f * f + w_s * s
