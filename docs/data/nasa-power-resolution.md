# NASA POWER does not resolve these islands

A measured constraint on Module 1, recorded before it becomes a surprising result.

Data: [`data/raw/nasa_power`](../../data/raw/nasa_power) · stage:
[`module1/data/nasa_power.py`](../../services/learned/module1_state_forecasting/src/module1/data/nasa_power.py)

## What was pulled

Two years of hourly meteorology — 2024-01-01 to 2025-12-31, 17,544 hours per site — for
Eluvaitivu, Analaitivu, Nainativu and Neduntivu (Delft). Irradiance and clear-sky irradiance,
wind at 10 m and 50 m, temperature, humidity, precipitation and surface pressure. Zero missing
values across all four sites.

This replaces the Department of Meteorology feed, which quoted **Rs 75,000 for two parameters**.
POWER is open and licence-free, so it is permitted to be a build dependency in a way that nothing
in `data/external/` is.

## The finding

POWER serves two products on different grids, and neither is fine enough for a 27 km archipelago:

| Product | Parameters | Native grid | Approx. cell |
|---|---|---|---|
| CERES SYN1deg | irradiance, clear-sky irradiance | 1.0° | ~111 km |
| MERRA-2 | wind, temperature, humidity, precipitation, pressure | 0.5° × 0.625° | ~55 × 69 km |

Measured over the full 17,544-hour pull — not inferred from the grid arithmetic:

| Parameter | Distinct series across 4 islands |
|---|---|
| `ghi_wh_m2`, `ghi_clearsky_wh_m2` | **1** |
| `wind_10m_ms`, `wind_50m_ms` | **2** |
| `temp_2m_c`, `humidity_2m_pct` | **2** |
| `precip_mm_hr`, `pressure_kpa` | **2** |

Analaitivu, Nainativu and Delft are **bit-identical** on every parameter for all 17,544 hours.
Only Eluvaitivu differs, and only on the meteorological fields — its irradiance is identical to
the other three as well.

Which islands share a series is an artifact of where a cell boundary falls, not of geography:

- Eluvaitivu ↔ Analaitivu — **5.0 km apart**, land in *different* MERRA-2 cells
- Analaitivu ↔ Delft — **24.8 km apart**, land in the *same* cell

## Why it matters

M1 is a spatiotemporal graph model. Its spatial edges are supposed to carry how conditions differ
between nodes. For the weather channel, **that difference does not exist in this data**: there is
no inter-island solar gradient at all, and wind separates one island from the other three.

The failure mode this note exists to prevent is a model that appears to learn spatial structure
from weather when it is fitting a constant, and an ablation that reports the spatial channel as
uninformative without anyone knowing why. Concretely:

1. **Do not claim spatial weather generalisation** from this source. Inter-island resource
   variation is unobserved, not modelled.
2. **Weather is close to a shared exogenous driver** across the archipelago. The spatial component
   of the graph must earn its place on topology, demand, asset state and hazard exposure — not on
   irradiance.
3. **Ablations must account for this.** Removing spatial weather edges should be expected to change
   little; that is a property of the data, and reporting it as a modelling result would be wrong.

## If island-level resolution is actually needed

In rough order of cost:

- **Downscale physically** — clear-sky index against a terrain and sea-breeze model. Adds
  structure but does not add information; it is an assumption, and must be labelled
  `QUALITY_INTERPOLATED` per ADR 0004.
- **A higher-resolution reanalysis** — ERA5 at 0.25° (~28 km) still would not separate all four,
  but would split them differently. CAMS or SARAH-3 satellite irradiance reaches ~5 km and would.
- **Measure it** — one pyranometer and one anemometer per island. This is the only option that
  produces real inter-island data, and it is the one worth asking CEB about, alongside retrieving
  the SMA inverter logs already on Eluvaitivu.

## Reproducing

```bash
task data:pull   # network fetch, frozen out of the default repro
task data        # runs the validation gate
```

`data/raw/nasa_power/manifest.json` records the request parameters, the API version and sources,
each site's grid cell, and the measured distinct-series count per parameter. The count is computed
from the fetched bytes on every pull, so if POWER ever changes resolution the manifest says so.
