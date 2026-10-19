# Module 1 forecaster: results

Held-out test quarter: Oct-Dec 2025 (Cyclone Ditwah, Eluvaitivu battery decay). Metrics files:
`forecast_v0_metrics.json`, `forecast_v0_1_metrics.json`. Regenerate with `python -m module1.train.run`.

## The claim, stated carefully

**ST-GNN v0.1 is a residual correction on persistence.** Of its gain over a conventional 1-D
LSTM, the residual head (predict the change from the current score) accounts for about 68% under
the equal weighting and 72% under stress-heavy; under hazard-heavy the split is roughly even (46%
residual head, 54% fused inputs and graph). Against a 1-D LSTM given the same residual head, ST-GNN v0.1 still has the lowest mean
error in all three weightings and catches more high-vulnerability onsets early, but it does not
win at every horizon. The graph helps only under the stress-heavy weighting, and weather inputs
do not help at all, which is consistent with NASA POWER giving three of the four islands
identical weather.

## Mean test RMSE, t+1..t+6 (lower is better)

| Model | Equal | Hazard-heavy | Stress-heavy |
|---|---|---|---|
| Persistence | 0.0377 | 0.0422 | 0.0474 |
| 1-D LSTM | 0.0342 | 0.0421 | 0.0359 |
| 1-D LSTM + residual head | 0.0287 | 0.0399 | 0.0238 |
| ST-GNN v0 | 0.0398 | 0.0481 | 0.0292 |
| **ST-GNN v0.1** | 0.0261 | 0.0373 | 0.0190 |
| Graph-free (v0.1) | 0.0262 | 0.0353 | 0.0230 |
| Time-free (v0.1) | 0.0273 | 0.0376 | 0.0208 |
| No-weather (v0.1) | 0.0259 | 0.0351 | 0.0192 |

## Where the gain comes from (error reduction, positive = helps)

| Weighting | Residual head (LSTM -> LSTM+res) | Fused inputs + graph (LSTM+res -> ST-GNN) | Graph | Temporal encoder | Weather |
|---|---|---|---|---|---|
| Equal | +16.0% | +9.2% | +0.5% | +4.6% | -0.8% |
| Hazard-heavy | +5.1% | +6.5% | -5.7% | +0.8% | -6.4% |
| Stress-heavy | +33.7% | +20.1% | +17.3% | +8.7% | +1.2% |

## Onset recall (share of high-vulnerability onsets forecast ahead of time)

| Weighting | Persistence | 1-D LSTM | 1-D LSTM + residual | ST-GNN v0.1 |
|---|---|---|---|---|
| Equal | 21% | 25% | 45% | 53% |
| Hazard-heavy | 14% | 12% | 15% | 28% |
| Stress-heavy | 14% | 22% | 75% | 85% |

## Caveats

- **The target is constructed** (hazard x fragility x stress). No dataset observes asset
  vulnerability; absolute errors are against this proxy.
- **The stress term uses the downscaled load**, which is itself built from temperature, so part of
  the stress-heavy gain is the model recovering a constructed diurnal shape. Hazard-heavy is the
  conservative weighting.
- **v0.1's two changes were chosen after reading v0's test results.** Both versions are reported;
  future revisions are diagnosed on validation only.
- **The battery decay is an input, not a forecast.** It is held out of training but reaches the
  model through last month's fragility, so in December the model carries a known step forward.
- **Not every horizon:** against the residual LSTM, ST-GNN v0.1 is +0.2% at t+6 (equal) and
  +1.6% at t+1 (hazard-heavy), and slightly worse in hazard-heavy storm hours (0.0626 vs 0.0616).
