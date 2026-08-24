# CEB Jaffna — Islanded Microgrid Generation Data

Source: Ceylon Electricity Board / Electricity Distribution Lanka (EDL), Northern Province
Division 1 (Jaffna). Obtained 2026-08-19 via Electrical Superintendent Mr. Ramaneetharan.

**Governance:** academic use only. The raw blobs (`*.pdf`, `*.xlsx`) are git-ignored per
[`data/README.md`](../../README.md). Calibration input only — never a build dependency.

## Files

| File | Contents |
|---|---|
| `Data_CEB_Jaffna.pdf` | 17 pp. scan (no text layer). pp.1–2 annual generation summary 2024/2025; p.3 Eluvaitivu 60 kW hybrid schematic; p.4 battery-room SLD (16 Mar 2016); pp.5–15 EDL tender `EDL/NP/ELV/BAT/2026` (battery replacement); pp.16–17 field sketches (hybrid block diagram, old power-station protection chain) |
| `Generation_2024_2025.xlsx` | Monthly fuel/energy ledger. `Sheet1` = 2024, `Sheet2` = 2025 |


## `processed/ceb_generation_tidy.csv` dictionary

| Column | Unit | Notes |
|---|---|---|
| `year`, `month`, `month_num` | — | calendar |
| `island_system` | — | one of the 5 generating systems below |
| `diesel_l` | L | diesel consumed |
| `diesel_cost_rs` | LKR | diesel purchase cost |
| `units_kwh` | kWh | energy generated (sheet column `unit (Q)`) |
| `oil_l`, `oil_cost_rs` | L, LKR | lubricating oil (~Rs 2,100/L) |
| `diesel_barrel` | barrels | **= `diesel_l` / 200** (verified exactly across all 120 rows) |
| `barrel_amount` | LKR | **marine transport cost** = barrels × rate. Rs 1,500/barrel for Analaithivu, Eluvaitivu and Nainativu in both years; **Delft Rs 180/barrel in 2024 and Rs 190 in 2025** (dedicated ferry, re-tendered annually). Exact in every month — see `validate.py` |
| `sfc_l_per_kwh` | L/kWh | derived specific fuel consumption |
| `diesel_rs_per_l` | LKR/L | derived effective diesel price |
| `fuel_cost_rs_per_kwh` | LKR/kWh | derived |
| `total_cost_rs_per_kwh` | LKR/kWh | derived, diesel + lube oil |

Regenerate and re-check from the repo root:

```bash
export PYTHONPATH=services/learned/module1_state_forecasting/src
python3 -m module1.data.ceb   data/external/ceb_jaffna/Generation_2024_2025.xlsx \
                              data/processed/ceb_generation_tidy.csv
python3 -m module1.data.validate data/processed/ceb_generation_tidy.csv
```

The derived table lands in `data/processed/ceb_generation_tidy.csv` — this directory holds only
what CEB supplied. Stage code and the reconciliation rationale:
[`module1/data/README.md`](../../../services/learned/module1_state_forecasting/src/module1/data/README.md).

## The five generating systems

| System | Island | Installed gensets (2025) | 2025 energy |
|---|---|---|---|
| `Analaithivu` | Analaitivu | 250 kVA ×1, 100 kVA ×2 | 426,418 kWh |
| `Eluvaitivu-Diesel` | Eluvaitivu | 100 kVA ×1 | 113,276 kWh |
| `Eluvaitivu-Hybrid` | Eluvaitivu | 30 kVA ×1 + PV/wind/battery | 83,331 kWh |
| `Delft-Neduntivu` | Neduntivu (Delft) | 250 kVA ×2, 330 kVA ×1 | 1,126,880 kWh |
| `Nainativu` | Nainativu | 250 kVA ×2, 380 kVA ×1 | 1,198,460 kWh |

Genset fleet differs between years — 2024 lists Analaithivu with 200 kVA ×2 (not 100 kVA ×2)
and Delft/Nainativu each with an extra 380 kVA unit. Treat capacity as year-specific.

## Validation

Every 2024 and 2025 annual total in the spreadsheet reconciles **exactly** against the
PDF summary tables (energy, diesel litres, diesel cost, transport cost, oil litres, oil cost,
all 5 systems, both years). The column mapping above is confirmed by that reconciliation.

## Known gaps

- Fleet O&M is given **only in aggregate**, not per island (stated explicitly on pp.1–2):
  2024 — repair Rs 33,252,244.56 / labour Rs 33,996,235.20 / OT Rs 14,729,306.79.
  2025 — repair Rs 44,268,230.83 / labour Rs 33,429,699.90 / OT Rs 14,361,394.33.
- No sub-monthly resolution. No load curves, no SCADA logs, no outage register — all
  data entry is manual (per interview), so half-hourly telemetry does not exist.
- Sheet columns beyond the labelled 35 (an unlabelled trailing numeric column and a
  reversed-month scratch block at rows 19–31) are working areas, not authoritative; the
  tidy extract ignores them.
- Solar and wind are **not metered separately** — the hybrid ledger records only diesel
  input and total kWh out. Renewable share must be inferred (see the baseline note).

## Downstream: hourly load

The monthly `units_kwh` column is the measured input to the load downscaling stage, which produces
`data/processed/island_load_hourly.csv` — hourly load per **island** (Eluvaitivu's two plants are
summed, because they serve one load). That series is **constructed, not measured**: monthly energy,
installed capacity and Nainativu's 460 kVA reported maximum demand are enforced; the intra-day
shape is assumed and every row is marked `QUALITY_INTERPOLATED`.

Method, calibration and the circularity warning:
[`docs/data/load-downscaling.md`](../../../docs/data/load-downscaling.md).
