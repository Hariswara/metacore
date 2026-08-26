# Monthly ledger → hourly island load

How the four island load series in `data/processed/island_load_hourly.csv` are produced, what in
them is measured, and what is not. Stage: [`module1/data/load.py`](../../services/learned/module1_state_forecasting/src/module1/data/load.py).
Machine-readable companion: `data/processed/load_parameters.json`.

---

## The gap

The CEB ledger records **one energy figure per island per month** — 120 numbers for two years.
`GridStateSnapshot` in `packages/contracts/proto/module1.proto` carries an `update_period_s`, and
M3 gates a genset start/stop. Between a monthly total and a dispatch decision there is nothing in
the measured record, and per [ADR 0004](../adr/0004-two-ingestion-paths.md) there never will be:
no SCADA, no historian, no half-hourly telemetry to back-fill from.

This stage does not *recover* the hourly load. It **constructs** one, and is built so the line
between measured and constructed stays legible to everything downstream.

## What is measured, and enforced

| Constraint | Source | How it is enforced |
|---|---|---|
| Monthly energy per island | CEB ledger, reconciled against the PDF summary | Series normalised **per month**, so the ledger total is reproduced to float tolerance regardless of shape |
| Installed capacity per island-year | `data/external/ceb_jaffna/README.md` | Peak must fit under installed kVA across the power-factor band |
| Nainativu max demand, **460 kVA** against 880 kVA installed | Interview + PDF | The diurnal exponent is *solved* against it, not chosen |

Energy conservation is the load-bearing one. A wrong shape stays a wrong shape — it cannot become
a wrong energy balance.

## What is assumed, and labelled

The intra-day curve, the weekday/weekend split (0.99–1.03, near-flat because these islands carry
no industrial load), the temperature coefficient (1.5 %/°C), and the 0.85 power factor needed to
relate a kWh ledger to a kVA demand figure.

**Every emitted row carries `QUALITY_INTERPOLATED`.** Per ADR 0004 that is not decoration: M2's
entire contribution is detecting when the state it was handed is untrustworthy, and a synthetic
value labelled as observed defeats the experiment rather than decorating it.

## Method

```
load(h) = E_month × shape(h) / Σ_{h ∈ month} shape(h)

shape(h) = DIURNAL[hour]^γ × DOW[weekday] × (1 + 0.015·(T(h) − T̃))
```

`T(h)` is measured MERRA-2 temperature from `data/raw/nasa_power`. Note that NASA POWER **does not
resolve these islands** ([nasa-power-resolution.md](nasa-power-resolution.md)), so the temperature
term is near-identical across all four — inter-island weather divergence is unobserved, not absent.

`γ` is the single free parameter, solved by bisection so Nainativu's 2025 peak reproduces
460 kVA × 0.85 = 391 kW. One island has a measured peak, so exactly one parameter is identifiable.
The solved γ is then applied to all four — an assumption of similar customer mix.

### The calibration came out at γ = 0.985

The diurnal vector was written from the shape of residential island demand — deep overnight
trough, warm midday plateau, sharp lighting-and-television evening peak — **before** being
compared to the measurement. Solving against the 460 kVA anchor moved it by **1.5 %**. The profile
and the one measured demand figure agree essentially without tuning, which is the closest thing to
external validation this construction can have.

## Result

```
Nainativu, mean kW by hour (local solar time)
  03   72.4  ############
  08  127.6  #####################
  13  134.7  ######################
  17  137.4  ######################
  18  208.2  ##################################
  20  267.5  ############################################   ← peak
  23  115.4  ###################
```

| Island | Mean kW | Peak kW | Min kW | Load factor | Annual peak at |
|---|---:|---:|---:|---:|---|
| Analaitivu | 47.0 | 113.5 | 18.9 | 0.414 | 2025-06-01 20:00 |
| Eluvaitivu | 21.8 | 50.6 | 9.3 | 0.430 | 2025-04-13 20:00 |
| Delft-Neduntivu | 124.9 | 312.3 | 51.2 | 0.400 | 2025-10-12 20:00 |
| Nainativu | 135.6 | **391.0** | 54.4 | 0.347 | 2025-07-20 20:00 |

Load factors of 0.35–0.43 are what residential-dominated island microgrids run at; nothing was
tuned to land there, so it is a weak independent check that the construction is not absurd.

## Why the fuel column does *not* pin the shape

It looks like it should. Specific fuel consumption is strongly anti-correlated with monthly energy
(Nainativu −0.75, Analaithivu −0.94) — the part-load efficiency signature of a real genset. Fitting
the Willans line `fuel = a·hours + b·energy` per system gives:

| System | No-load `a` (L/h) | Marginal `b` (L/kWh) | R² | Residual |
|---|---:|---:|---:|---:|
| Nainativu | 7.64 | 0.2702 | **0.975** | 1.9 % |
| Delft-Neduntivu | 8.27 | 0.2568 | 0.639 | 6.4 % |
| Analaithivu | 15.38 | 0.0705 | 0.540 | 3.2 % |
| Eluvaitivu-Diesel | **−0.48** | 0.4465 | 0.973 | 5.1 % |
| Eluvaitivu-Hybrid | 0.26 | 0.0775 | 0.379 | 30.7 % |

But that model is **linear in energy**, so at monthly aggregation every hourly profile with the
same monthly total predicts the same fuel burn. The SFC correlation is the Willans line restated,
not independent information about the curve. It is computed and published anyway, for two reasons:

1. **`a` and `b` are what M3 needs.** No-load burn against marginal rate *is* the objective
   function for pricing a genset start against the fuel it saves.
2. **The sign of `a` is a finding.** A physically plausible positive no-load rate at R² = 0.975
   says Nainativu runs continuously rather than on an evening schedule. Eluvaitivu-Diesel's
   negative `a` says the opposite — it is dispatched intermittently, consistent with backing up
   the hybrid rather than serving base load. The stage flags this rather than clamping it.

Eluvaitivu-Hybrid's 30.7 % residual is expected: its fuel-to-energy relationship is contaminated
by the renewable share collapsing through Q4 2025.

## Circularity warning

**This artifact is for simulation, dispatch evaluation and M4 power-flow — not for training or
evaluating a load forecaster.**

Load here is generated from weather through a known coefficient. A model trained to predict this
load from that weather will recover the coefficient and report it as forecasting skill. Any M1
forecasting result must state which series it used, and a result on this one measures the
interpolator, not the world.

## Open items

- **γ is calibrated on one island.** Analaitivu, Eluvaitivu and Delft inherit Nainativu's
  peakiness. A reported maximum demand for any other island would make its own γ identifiable —
  worth asking EDL for, since the figure exists on their monthly returns.
- **Power factor is assumed at 0.85.** The gate checks the 0.80–0.90 band, so no conclusion turns
  on the point value, but a measured pf would tighten the anchor.
- **No holiday calendar.** Sri Lankan Poya days and the April new year shift residential load
  measurably, and are not modelled.
- **Local solar time vs Sri Lanka Standard Time.** NASA POWER stamps LST (longitude-based,
  ≈ UTC+5:19 here) while clocks read UTC+5:30 — an 11-minute offset, inside one hourly bin, and
  ignored. It would matter at sub-hourly resolution.

## Reproducing

```bash
task data      # runs load_downscale then load_check with the rest of the pipeline
```

Or directly:

```bash
export PYTHONPATH=services/learned/module1_state_forecasting/src
python3 -m module1.data.load downscale data/processed/ceb_generation_tidy.csv \
        data/raw/nasa_power data/processed/island_load_hourly.csv
python3 -m module1.data.load validate data/processed/island_load_hourly.csv \
        data/processed/ceb_generation_tidy.csv
```
