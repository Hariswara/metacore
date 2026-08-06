# Data

All datasets used to build and evaluate MetaCore are **open or simulated**, so no ethics gate and no
external approval blocks development. Blobs are tracked by DVC and ignored by git.

| Path | Source | Use |
|---|---|---|
| `raw/nasa_power` | NASA POWER | Rainfall, wind, solar irradiance, temperature — M1 inputs |
| `raw/osm_qgis` | OpenStreetMap / QGIS | DEM, topographic wetness index, landslide susceptibility, asset topology |
| `raw/ais` | AIS (anonymised, grid-aggregated) | Vessel transit / port accessibility — fuel-resupply constraint |
| `raw/uci_household_power` | UCI Household Electric Power Consumption | Baseline demand series |
| `processed/` | Derived | Time-aligned, resampled feature tables and graph tensors |
| `external/` | CEB / Department of Meteorology | **Calibration only** — see below |

## State-entity data is calibration, never dependency

AWS telemetry, single-line diagrams and SCADA fault logs from CEB and the Department of Meteorology
are used strictly to calibrate the digital twin and the sensing assumptions. The full pipeline runs
end-to-end on open and simulated data if that access is delayed or denied. Nothing in `external/` is
allowed to become a build requirement.

## Governance

Offline and in-memory on the secure SLIIT server. Node coordinates are encoded as arbitrary spatial
offsets in any published output. AIS data is grid-aggregated and anonymised. Anything provided by a
state entity is governed by academic-use-only terms and, where required, a non-disclosure agreement.
