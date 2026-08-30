# Module 3 — Cost-Aware Metacognitive Gating & Meta-Policy — Starter

A runnable, self-contained **training** starter for **Module 3** (Saabir S., J26-DS-317).
It learns a cost-aware gate over `{M2 uncertainty, M1 hazard context}` that chooses
**System 1** (cheap reactive rules) vs **System 2** (deliberative survival path),
reports reward against always-S1 / always-S2 / **u-threshold** baselines, and checks that escalation
rate is monotonically non-decreasing across hazard severity — the core thesis of
the module — with **zero dependency on a live M1 or M4 service**. Drop-in location:
`services/learned/module3_metapolicy/`.

> Training only. ONNX export and the Go `gating_decision_svc` hot path are
> deliberately out of scope for this starter (see roadmap).

## Verified results (`python run_demo.py`)
| Metric | Value | Target |
|---|---|---|
| total reward always-S1 | -107.750 | baseline |
| total reward always-S2 | -99.329 | baseline |
| total reward u>0.40 threshold | (printed by `run_demo.py`) | the "why not just a threshold?" baseline |
| total reward trained policy | **117.976** | beat all three baselines |
| avg deliberation cost (S2 / threshold / policy) | 0.056 / (run) / 0.049 | policy ≤ always-S2 |
| escalation rate normal→…→extreme (excl. sensing) | 0.000 / 0.261 / 1.000 / 1.000 | non-decreasing |
| escalation by `trigger_reason` none/value/sensing/both | 0.000 / 1.000 / **0.000** / 1.000 | sensing → S1 |
| monotonic non-decreasing | True | `True` |

Headline numbers are process-stable: `mock_verify` uses SHA-256 (not salted `hash()`), and the sample `action_id` / `timestamp` are derived from seed + emit index. Re-run `python run_demo.py` to refresh the exact decimals after a config change; the pattern (policy beats all three baselines, escalation non-decreasing, sensing pinned to 0) is what to check.

## Run — from the terminal
```bash
pip install -r requirements.txt
python run_demo.py
```
Trains, evaluates, prints the table above, and overwrites `sample_m3_to_m4.jsonl` (the
committed M4 contract fixture).

Requires the M2 mock stream. On this branch a vendored copy lives at
`sample_m2_to_m3.jsonl` (`m2-out/0.3`) — **keep it synced with
`../module2_auq_engine/sample_m2_to_m3.jsonl`**; `m2_stream.py` prefers the local copy
whenever it exists, so a stale one silently trains against numbers M2 no longer
produces (this bit us once — see `OF_FLOOR` in `m2_stream.py`, which mirrors M2's
`config.yaml: trigger.observed_fraction_floor` and needs updating if M2 recalibrates
again). When Module 2 is merged, `m2_stream.py` also accepts
`../module2_auq_engine/sample_m2_to_m3.jsonl` directly.

## Run — from the dashboard

The gateway exposes an on-demand train+eval run over HTTP; the dashboard's `/gating`
page is a form for it. Two terminals, from the repo root:

```bash
# terminal 1 — gateway (needs this module's deps importable, so run from the shared
# workspace venv, not gateway's own slim one)
uv sync --all-packages
uv run uvicorn gateway.main:app --app-dir services/gateway --reload --port 8000

# terminal 2 — dashboard
cd apps/dashboard && pnpm install && pnpm dev
```
Open the printed dashboard URL (`/gating`), set the run parameters (seed, episode
length, budget, train/eval episodes, reward weights), click **Run**, and wait
~15–30s. Under the hood this is exactly `python run_demo.py <config> <output>` run as
a subprocess by `services/gateway/gateway/routers/module3.py` — same script, same
numbers, just driven by the form instead of a terminal.

Each row in the resulting decision log expands (click it) into the full picture for
that step: the **M2 input** that produced it (`severity`, `trigger_reason`,
`epistemic_uncertainty` — the `u` value — and `observed_fraction`), the **M4 mock
verdict** (`APPROVE`/`REJECT` and any violations), and the **full proposed action
plan** from whichever control path ran — `breakers`, `load_shed` (which nodes, how
much, what priority tier), and `dispatch` (generation setpoints). None of that
context travels over the real `M3_TO_M4_CONTRACT.md` wire format; it's additive,
dashboard-only data from `run_demo.py`'s `context` list (`decision_context` in the
gateway's JSON response) — Hariswara's verifier only ever sees the contract fields.

This path only works when the process running `gateway.main:app` has this module's
deps (`torch`, `gymnasium`, `numpy`, `pyyaml`) importable — true for `uv run` from the
repo root above, **not** for `docker compose` (gateway's own image only installs
`fastapi`/`pydantic`/`grpcio`). Wiring the compose path is future work.

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
- `policy.py` — small MLP + REINFORCE with per-trajectory return standardization (no moving-average baseline).
- `m3_evaluate.py` — always-S1 / always-S2 / **u-threshold** baselines, escalation-by-severity, escalation-by-`trigger_reason`.
- `run_demo.py` — BC warm-start (cause-aware heuristic) → REINFORCE → `sample_m3_to_m4.jsonl`
  (or, given `config_path`/`output_json_path` args, a structured JSON result instead —
  see "Run — from the dashboard").
- `M3_TO_M4_CONTRACT.md` — mock stream for Hariswara.
- `config.yaml` — episode length, budget, reward weights (incl. `sensing_escalation_penalty` and the 0.35 / 0.5 / 0.25 floors).
- `../../../docs/module3-explained.html` — **generated explainer** for the viva; not a source of truth (code + this README are).
- `../../gateway/gateway/routers/module3.py` — `POST /api/module3/run`; drives `run_demo.py`
  as a subprocess for the dashboard.
- `../../../apps/dashboard/src/routes/gating/` — the `/gating` page: run-config form +
  reward/escalation charts + expandable decision log.

## Two things worth understanding
1. **Budget exhaustion is not an agent action.** When `budget_remaining <= 0` the env
   forces System 1 and sets `budget_exhausted_fallback=true`.
2. **`trigger_reason` is a cleaner reward signal than `competence_drop` alone.** A value
   drop and a sensing drop are different problems; treating them the same wastes
   deliberation budget on blackouts. The `u > trigger_threshold` baseline is in the
   results table specifically so that claim can be scored, not asserted.

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
