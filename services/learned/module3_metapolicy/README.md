# Module 3 — Cost-Aware Metacognitive Gating & Meta-Policy — Starter

A runnable, self-contained **training** starter for **Module 3** (Saabir S., J26-DS-317).
It learns a cost-aware gate over `{M2 uncertainty, M1 hazard context}` that chooses
**System 1** (cheap reactive rules) vs **System 2** (deliberative survival path),
reports reward against always-S1 / always-S2 baselines, and checks that escalation
rate is monotonically non-decreasing across hazard severity — the core thesis of
the module — with **zero dependency on a live M1 or M4 service**. Drop-in location:
`services/learned/module3_metapolicy/`.

> Training only. ONNX export and the Go `gating_decision_svc` hot path are
> deliberately out of scope for this starter (see roadmap).

## Verified results (`python run_demo.py`)
| Metric | Value | Target |
|---|---|---|
| total reward always-S1 | -106.625 | baseline |
| total reward always-S2 | -100.531 | baseline |
| total reward trained policy | **119.476** | beat both baselines |
| avg deliberation cost (S2 / policy) | 0.056 / 0.049 | policy ≤ always-S2 |
| escalation rate normal→…→extreme (excl. sensing) | 0.000 / 0.261 / 1.000 / 1.000 | non-decreasing |
| escalation by `trigger_reason` none/value/sensing/both | 0.000 / 1.000 / **0.000** / 1.000 | sensing → S1 |
| monotonic non-decreasing | True | `True` |

## Run
```bash
pip install -r requirements.txt
python run_demo.py
```

Requires the M2 mock stream. On this branch a vendored copy lives at
`sample_m2_to_m3.jsonl` (`m2-out/0.3`). When Module 2 is merged, `m2_stream.py`
also accepts `../module2_auq_engine/sample_m2_to_m3.jsonl`.

## Cause-aware gating (`m2-out/0.3`)

Duwaragie's contract splits "competence drop" by **cause**. The gate branches on
`trigger_reason`, not on a single scalar:

| `trigger_reason` | Meaning | Gate response |
|---|---|---|
| `value` | unseen conditions (cyclone) | escalate to System 2 |
| `sensing` | missing / blackout data | **conservative System 1** — don't deliberate on absent data |
| `both` | cyclone + comms loss | escalate (worst case) |
| `none` | nominal | stay System 1 unless severity demands otherwise |

`epistemic_uncertainty` is the **quality-adjusted** magnitude; `observed_fraction` is
in the observation vector. Escalating on `sensing` is explicitly penalised in the reward.

## Files
- `synthetic_context.py` — **mock M1 SystemContext** (severity schedule + vulnerability).
- `m2_stream.py` — loads Duwaragie's `sample_m2_to_m3.jsonl`; reason-aware bootstrap replay.
- `system1.py` / `system2.py` — System 1 / System 2 control-path stand-ins (S1 sensing-fallback is more conservative), split so each control path is a single-owner file going forward.
- `priority.py` — shared node priority-tier map (`system1.py` and `system2.py` both need it).
- `verifier.py` — `mock_verify`, the M4 stand-in (neither S1 nor S2).
- `gating_env.py` — Gymnasium env: **12-d** obs (includes reason flags + `observed_fraction`), Discrete(2), budget-forced S1.
- `policy.py` — small MLP + vanilla REINFORCE.
- `evaluate.py` — baselines, escalation-by-severity, escalation-by-`trigger_reason`.
- `run_demo.py` — BC warm-start (cause-aware heuristic) → REINFORCE → `sample_m3_to_m4.jsonl`.
- `M3_TO_M4_CONTRACT.md` — mock stream for Hariswara.
- `config.yaml` — episode length, budget, reward weights (incl. `sensing_escalation_penalty`).

## Two things worth understanding
1. **Budget exhaustion is not an agent action.** When `budget_remaining <= 0` the env
   forces System 1 and sets `budget_exhausted_fallback=true`.
2. **`trigger_reason` is a cleaner reward signal than `competence_drop` alone.** A value
   drop and a sensing drop are different problems; treating them the same wastes
   deliberation budget on blackouts.

## What is mocked vs real
- **Mock now:** M1 `SystemContext`, M4 verdicts, simplified S1/S2 controllers.
- **Real now:** M2's published JSONL mock (`m2-out/0.3`), and `module3.proto` field names (JSON transport).
- **Real later:** live M1 context; live M4 verdicts; ONNX + Go serving.

## Independent-work roadmap
1. Agree live `SystemContext` with Zayan — swap `synthetic_context`.
2. Publish `M3_TO_M4_CONTRACT.md` + `sample_m3_to_m4.jsonl` to Hariswara.
3. Replace `mock_verify` with real M4 verdicts.
4. Ablate cause-aware gate vs a `competence_drop`-only baseline for the paper.
5. ONNX export + Go serving — future work.
