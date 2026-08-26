# The scenario library

`common.proto` puts the shared ID/OOD scenario library in Module 1's hands: every `ScenarioRef` a
consumer replays — `scenario_id`, `library_version`, `out_of_distribution` — is issued here.

Artifacts: [`data/processed/events.csv`](../../data/processed) and `scenario_library.json` ·
stage: [`module1/data/scenarios.py`](../../services/learned/module1_state_forecasting/src/module1/data/scenarios.py)

## Why it is derived from the monthly ledger and not the hourly series

`island_load_hourly.csv` sums the Eluvaitivu diesel set and the Eluvaitivu hybrid plant into one
island demand. That is the right model: the island has one load, served by both plants together,
and it is what M4's power flow and M3's dispatch operate on.

It also removes the failure this project is written about.

Measured across 2025 Q4 against the preceding quarter:

| | Jul–Sep 2025 | Oct–Dec 2025 | Change |
|---|---|---|---|
| Eluvaitivu hybrid plant | 8,793 kWh/mo | 2,337 kWh/mo | **−73.4%** |
| Eluvaitivu island demand | 17,782 kWh/mo | 15,946 kWh/mo | −10.3% |

Month by month, against the same baseline quarter:

| Month | Hybrid plant | Island demand |
|---|---|---|
| 2025-10 | −35.7% | −6.6% |
| 2025-11 | −87.5% | −11.9% |
| 2025-12 | **−97.0%** | −12.5% |

The diesel set absorbed the difference, month for month — its own output rose from 8,854 kWh in
September to 15,300 kWh in December. A detector watching island demand sees a 10% dip that looks
seasonal. The plant behind it had all but stopped.

**That 7.1× attenuation is the argument for per-asset state representation over aggregate
telemetry**, and it is why the label cannot be read off the hourly artifact. It is derived where
the event is still visible, and where the numbers are still measured: the reconciled monthly
per-plant ledger. These rows are `QUALITY_OBSERVED` — meter readings, not downscaled estimates.
Almost nothing else Module 1 publishes can say that.

## The detection rule

Applied uniformly to all five plants. The window is derived, not hand-placed.

1. **Baseline** — the median of the plant's own monthly energy across the record. Median rather
   than mean because the event is *inside* the record being baselined against; a mean would be
   dragged down by the very months under test and shrink the anomaly it is meant to measure.
2. **Core** — months below **0.50** of baseline. On this dataset: 2025-11 (0.14) and 2025-12
   (0.03) for the hybrid plant, and nothing anywhere else.
3. **Onset** — walk backwards from the core while months stay below **0.90**. This adds 2025-10
   at 0.70. A decay starts before it becomes obvious, and a window that opens at the collapse
   hands a detector the easy half of the problem.

Over the full 24-month, five-plant record the rule fires **once**. The other four plants produce no
core month at all, so the thresholds are not doing hidden work to isolate a pre-chosen answer.

Detecting the onset *without* the label is M2's problem, not this stage's. What is published here
is the window, the rule that produced it, and the monthly ratios — so a disagreement about where
the event starts is a disagreement about a number in a file.

## Contents

| Class | Count | Definition |
|---|---|---|
| Out-of-distribution | 1 | `eluvaitivu-hybrid-decay-2025q4`, 2025-10 … 2025-12 |
| In-distribution | 16 | Non-overlapping 3-month windows where *every* plant on the island sits within 0.85–1.15 of its own median, and no month belongs to an event |

The in-distribution band is deliberately tight, and event months are excluded explicitly rather
than by taking "everything not flagged". A nominal episode that quietly contained the shoulder of
a collapse would inflate a detector's apparent skill.

## Gate

`scenario_check` fails the pipeline when the library stops agreeing with the ledger:

1. `events.csv` and `scenario_library.json` describe the same set — a consumer may read either.
2. Scenario ids are unique; a `ScenarioRef` resolving to two episodes is unreplayable.
3. **The library still reproduces from the ledger.** This is the load-bearing check. It is what
   stops the labels drifting into hand-maintained constants once the numbers look settled.
4. The Eluvaitivu-Hybrid degradation is present and labelled out-of-distribution, by name.
5. Both distribution classes are populated — an OOD-only library cannot score a detector.

## Open items

- **One OOD episode is one episode.** It supports a case study, not a distribution over failure
  modes. Anything reported as an OOD detection rate needs either simulated degradations on top of
  this one or an explicit statement that n = 1.
- **The window is monthly.** M2's evaluation runs on hourly state, so the label has to be
  broadcast across the hours of each month. Every hour in a flagged month inherits the flag; the
  transition inside 2025-10 is not resolved, and nothing in the record resolves it.
- **`ScenarioRef.library_version` is `1.0.0`.** Adding scenarios is a minor bump. Moving an
  existing window is a major bump — it silently changes anyone's published result.
