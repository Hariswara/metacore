# Data

All datasets used to build and evaluate MetaCore are **open or simulated**, so no ethics gate and no
external approval blocks development. Blobs are tracked by DVC and ignored by git.

| Path | Source | Use |
|---|---|---|
| `raw/nasa_power` | NASA POWER (MERRA-2 + CERES) | Hourly irradiance, wind, temperature, humidity, rainfall and pressure for the four islands, 2024–2025. **Does not resolve the islands** — one irradiance series and two meteorological series across four sites; see [`docs/data/nasa-power-resolution.md`](../docs/data/nasa-power-resolution.md) |
| `raw/osm_qgis` | OpenStreetMap / QGIS | DEM, topographic wetness index, landslide susceptibility, asset topology |
| `raw/ais` | AIS (anonymised, grid-aggregated) | Vessel transit / port accessibility — fuel-resupply constraint |
| `raw/uci_household_power` | UCI Household Electric Power Consumption | Baseline demand series |
| `processed/` | Derived | Time-aligned, resampled feature tables and graph tensors |
| `processed/island_load_hourly.csv` | Derived | **Constructed, not measured.** Hourly load per island 2024–2025, downscaled from the monthly CEB ledger. Every row `QUALITY_INTERPOLATED`; assumptions in `load_parameters.json`. See [`docs/data/load-downscaling.md`](../docs/data/load-downscaling.md) |
| `external/ceb_jaffna` | CEB / EDL Northern Province | Monthly generation, fuel and cost ledger 2024–2025 for the four islanded microgrids, plus the Eluvaitivu hybrid single-line and battery-replacement tender. **Calibration only** — see below and [`external/ceb_jaffna/README.md`](external/ceb_jaffna/README.md) |
| `external/` | Department of Meteorology | **Calibration only** — see below |

## State-entity data is calibration, never dependency

AWS telemetry, single-line diagrams and SCADA fault logs from CEB and the Department of Meteorology
are used strictly to calibrate the digital twin and the sensing assumptions. The full pipeline runs
end-to-end on open and simulated data if that access is delayed or denied. Nothing in `external/` is
allowed to become a build requirement.

What `external/ceb_jaffna` establishes about the target system, and what each module can draw
from it, is written up in [`docs/data/ceb-jaffna-baseline.md`](../docs/data/ceb-jaffna-baseline.md).

## Governance

Offline and in-memory on the secure SLIIT server. Node coordinates are encoded as arbitrary spatial
offsets in any published output. AIS data is grid-aggregated and anonymised. Anything provided by a
state entity is governed by academic-use-only terms and, where required, a non-disclosure agreement.
