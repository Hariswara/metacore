# Data

All datasets used to build and evaluate MetaCore are **open or simulated**, so no ethics gate and no
external approval blocks development.

## Where the artifacts live

DVC is the pipeline engine — stages, dependency graph, `dvc.lock` provenance and all four gates.
It is **not** the transport. Every output is marked `cache: false` in `dvc.yaml`, so the published
artifacts are tracked in git and a clone has them immediately: no `dvc pull`, no remote, no
account to authorise.

That is a deliberate reversal of the usual DVC arrangement, and it is worth recording why. Two
remotes were tried and both are blocked by policy, not by anything we control:

- **SLIIT OneDrive (`metacore@sliit.lk`).** A Microsoft 365 Group is not a sign-in account, and
  the SLIIT tenant disables user consent for third-party applications — authorising rclone returns
  *"Approval required"*. A private Azure app does not help: reaching a group's document library
  needs `Files.ReadWrite.All` and `Sites.Read.All`, both admin-consent-only.
- **Google Drive.** DVC's built-in OAuth client requests the full `drive` scope, is unverified,
  and Google now refuses it outright — *"This app is blocked"*. The scope is not configurable:
  `dvc_gdrive` never passes `oauth_scope` through to pydrive2.

The whole cache is **9 MB across 10 files**, so git handles it without strain, and the trade is
worth naming: each regeneration of `island_load_hourly.csv` adds ~4 MB to history permanently.
`raw/nasa_power` is `frozen:` in the pipeline and changes only on a deliberate refresh. If the
dataset grows past a few tens of MB, revisit this — a working remote (a SLIIT server over SSH, or
a private Google OAuth client) drops back in as one line in `.dvc/config`, and nothing else moves.

`external/` stays git-ignored in full. The CEB workbook is state-entity data provided for
calibration and is not ours to redistribute, which is exactly what ADR 0004's synthetic fallback
(`task data:synthetic`) exists to survive.

| Path | Source | Use |
|---|---|---|
| `raw/nasa_power` | NASA POWER (MERRA-2 + CERES) | Hourly irradiance, wind, temperature, humidity, rainfall and pressure for the four islands, 2024–2025. **Does not resolve the islands** — one irradiance series and two meteorological series across four sites; see [`docs/data/nasa-power-resolution.md`](../docs/data/nasa-power-resolution.md) |
| `raw/osm_qgis` | OpenStreetMap / QGIS | DEM, topographic wetness index, landslide susceptibility, asset topology |
| `raw/ais` | AIS (anonymised, grid-aggregated) | Vessel transit / port accessibility — fuel-resupply constraint |
| `raw/uci_household_power` | UCI Household Electric Power Consumption | Baseline demand series |
| `processed/` | Derived | Time-aligned, resampled feature tables and graph tensors |
| `processed/island_load_hourly.csv` | Derived | **Constructed, not measured.** Hourly load per island 2024–2025, downscaled from the monthly CEB ledger. Every row `QUALITY_INTERPOLATED`; assumptions in `load_parameters.json`. See [`docs/data/load-downscaling.md`](../docs/data/load-downscaling.md) |
| `processed/events.csv`, `processed/scenario_library.json` | Derived | The shared ID/OOD scenario library (owner: M1, per `common.proto`). One out-of-distribution episode — the Eluvaitivu hybrid decay, 2025-10..2025-12 — and 16 in-distribution control windows, derived from the ledger by a stated rule. `QUALITY_OBSERVED`: these are meter readings, not estimates. See [`docs/data/scenario-library.md`](../docs/data/scenario-library.md) |
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
