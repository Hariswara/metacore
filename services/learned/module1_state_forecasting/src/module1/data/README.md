# Module 1 — calibration path

The offline half of ingestion. See [`docs/adr/0004-two-ingestion-paths.md`](../../../../../../docs/adr/0004-two-ingestion-paths.md).

Reads `data/external/**` and `data/raw/**`, emits a versioned parameter set into `data/processed/`.
Batch only — no sockets, no bus, never on a request path. The streaming half lives in
`services/realtime/ingestion_svc` and shares no code with this; the parameter set is the only
interface between them.

**Standard library only.** `metacore-module1` declares torch and pandas, but nothing here imports
them. These stages are the reconciliation gate for state-entity data and must run in CI in seconds,
without resolving a deep-learning dependency tree first.

| Module | Stage | Role |
|---|---|---|
| `xlsx.py` | — | Read-only xlsx reader (zip + XML). No openpyxl, no pandas. |
| `ceb.py` | `ceb_tidy` | CEB Jaffna generation ledger → long-format calibration table |
| `validate.py` | `ceb_reconcile` | Reconciliation gate — fails the pipeline on any mismatch |

Run from the repo root:

```bash
export PYTHONPATH=services/learned/module1_state_forecasting/src
python -m module1.data.ceb      data/external/ceb_jaffna/Generation_2024_2025.xlsx \
                                data/processed/ceb_generation_tidy.csv
python -m module1.data.validate data/processed/ceb_generation_tidy.csv
```

Or as DVC stages, which is how CI runs them: `dvc repro -f data/dvc.yaml`.

## Why the reconciliation gate exists

ADR 0004 records that this path cannot be validated against held-out telemetry, because none
exists — CEB's islands have no SCADA and no historian. Reconciliation stands in for it.

CEB supplied the same figures twice by independent routes: a monthly spreadsheet and a printed
annual summary produced separately from it. The spreadsheet has no schema and a two-level header
whose column order differs between island bands, so the layout in `ceb.py` is an *interpretation*.
`validate.py` is what makes it a checked one — 300 invariants across 10 island-years, covering
every annual total plus the per-row fuel and transport identities.

This is not ceremony. The gate has already caught one error: the Delft barrel transport rate was
first read as a flat Rs 180 across the dataset, when it is Rs 180 in 2024 and Rs 190 in 2025 —
re-tendered annually. Averaging across years would have quietly biased the marine-logistics cost
that M3 gates on.

## Adding a stage

1. Keep it standard-library-only, or justify the dependency in the PR.
2. Blank source cells become `None`, never `0.0`. Downstream that is the difference between
   "no lube oil was used" and "we do not know", which is what `QualityMask` encodes.
3. Anything synthesized, interpolated or inferred must be marked so — `QUALITY_INTERPOLATED`, not
   `QUALITY_OBSERVED`. M2's entire contribution is detecting when its input is untrustworthy.
4. Give it an invariant in `validate.py`. If you cannot state one, say why in the stage docstring.

## Stage index

| Module | Stage | Role |
|---|---|---|
| `xlsx.py` | — | Read-only xlsx reader (zip + XML) |
| `ceb.py` | `ceb_tidy` | CEB Jaffna generation ledger → long-format calibration table |
| `validate.py` | `ceb_reconcile` | Reconciliation gate against the printed annual summary |
| `nasa_power.py` | `nasa_power_pull` / `nasa_power_check` | Hourly meteorology for the four islands, and its gate |
| `load.py` | `load_downscale` / `load_check` | Monthly island energy → hourly load, and its gate |

`nasa_power_pull` is **frozen** in `data/dvc.yaml`: it is a network fetch of ~70,000 site-hours
from a free public API, so it is deliberately excluded from the default `dvc repro`. Refresh it
with `task data:pull`.

The NASA POWER stage carries a measured constraint worth reading before modelling anything
spatial: the source does not resolve these islands. See
[`docs/data/nasa-power-resolution.md`](../../../../../../docs/data/nasa-power-resolution.md).

`load.py` is the stage where the measured/constructed boundary matters most: it turns 120 monthly
numbers into 70,176 hourly ones. Read the circularity warning in its docstring before using the
output as training data for anything — it is a simulation input, not observed history.
[`docs/data/load-downscaling.md`](../../../../../../docs/data/load-downscaling.md).
