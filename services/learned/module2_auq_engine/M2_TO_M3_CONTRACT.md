# M2 → M3 Output Contract (draft v0.2) — for Saabir (Module 3)

Per the master plan register and M3 FR1, your gate builds against this **mock stream**
until my real output is live. One message is emitted per state / control step.

| Field | Type | Meaning | Range |
|---|---|---|---|
| `timestamp` | float (epoch s) | when the state was scored | — |
| `epistemic_uncertainty` | float | **quality-aware u** — the gating signal | 0.0 (sure) … 1.0 (no evidence) |
| `aleatoric_proxy` | float | predictive entropy (irreducible noise) | ≥ 0 |
| `competence_drop` | bool | trigger to escalate to System 2 | true / false |
| `state_class` | int | argmax safety class | 0 safe … K-1 critical |
| `class_probabilities` | float[K] | Dirichlet-mean p over safety classes | sums to 1 |
| `observed_fraction` | float | **new in 0.2** — share of M1's features that were `QUALITY_OBSERVED` | 0.0 … 1.0 |
| `schema_version` | string | contract version | `m2-out/0.2` |

## What changed in 0.2

**Additive and backward-compatible.** One new field; nothing renamed, nothing removed. A
consumer written against `0.1` keeps working if it ignores `observed_fraction` — but the
meaning of `epistemic_uncertainty` has widened, so read the next section before you do.

`epistemic_uncertainty` is now **quality-aware**:

```
u = K / (Σ evidence · observed_fraction + K)
```

At `observed_fraction = 1.0` this is exactly the old `u = K/S`, so 0.1 behaviour is a
special case rather than a change.

## Two axes of "the state is untrustworthy"

The gate now fires on either, and they are independent:

| Axis | What it means | What it looks like |
|---|---|---|
| **value** | the state is unlike anything in training | cyclone: `u ≈ 1.0` at *any* `observed_fraction` |
| **quality** | the state is mostly interpolated or missing | comms blackout: `u` climbs as `observed_fraction` falls, on ordinary values |

Measured on the synthetic prototype — in-distribution states, evidence discounted:

| `observed_fraction` | 1.00 | 0.75 | 0.50 | 0.25 | 0.10 |
|---|---|---|---|---|---|
| mean `u` (in-distribution) | 0.105 | 0.130 | 0.173 | 0.271 | 0.440 |
| mean `u` (cyclone) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

`observed_fraction` is what lets you tell the two apart: **high `u` with a high
`observed_fraction` is a genuine novel state; high `u` with a low `observed_fraction` is a
sensing failure.** Those may well deserve different System 2 policies — escalating to
deliberation is the right answer for the first, and possibly "go get better data" for the
second. That is your call, not mine; the field is there so you can make it.

Where the number comes from: M1's `StateRepresentation` carries a `QualityMask`
(`packages/contracts/proto/common.proto`) with a `Quality` per feature, and
`observed_fraction` is the share of them that are `QUALITY_OBSERVED`. Per ADR 0004
anything not measured must not be labelled observed, so expect this to sit **well below
1.0 in practice** — M1's downscaled load series is `QUALITY_INTERPOLATED` on every row.
The mock currently emits `0.5` (four measured weather channels, four constructed
electrical ones).

Example (a cyclone state):
```json
{"timestamp":1787683511.59,"epistemic_uncertainty":1.0,"aleatoric_proxy":0.0,
 "competence_drop":true,"state_class":0,"class_probabilities":[0.333,0.333,0.333],
 "observed_fraction":0.5,"schema_version":"m2-out/0.2"}
```

Gate rule of thumb: **low u → stay System 1; `competence_drop == true` → escalate to System 2.**
Generate a live mock stream with `contract.build_output(...)` (see `run_demo.py`).
Contract to be frozen at PP1 (master plan §5); ping me before you hard-code field names.
