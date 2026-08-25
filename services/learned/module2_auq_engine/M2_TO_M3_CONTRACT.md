# M2 → M3 Output Contract (draft v0.3) — for Saabir (Module 3)

Per the master plan register and M3 FR1, your gate builds against this **mock stream**
until my real output is live. One message is emitted per state / control step.

| Field | Type | Meaning | Range |
|---|---|---|---|
| `timestamp` | float (epoch s) | when the state was scored | — |
| `epistemic_uncertainty` | float | **quality-aware u** — how untrustworthy, as a magnitude | 0.0 (sure) … 1.0 (no evidence) |
| `aleatoric_proxy` | float | predictive entropy (irreducible noise) | ≥ 0 |
| `competence_drop` | bool | trigger to escalate to System 2 | true / false |
| `trigger_reason` | string | **new in 0.3** — which axis fired | `none` / `value` / `sensing` / `both` |
| `state_class` | int | argmax safety class | 0 safe … K-1 critical |
| `class_probabilities` | float[K] | Dirichlet-mean p over safety classes | sums to 1 |
| `observed_fraction` | float | new in 0.2 — share of M1's features that were `QUALITY_OBSERVED` | 0.0 … 1.0 |
| `schema_version` | string | contract version | `m2-out/0.3` |

## What changed

**Both bumps are additive and backward-compatible.** Nothing renamed, nothing removed. A
`0.1` consumer keeps working if it ignores the new fields — but the meaning of
`epistemic_uncertainty` widened in 0.2, so read on before you rely on it.

**0.2 — `epistemic_uncertainty` became quality-aware:**

```
u = K / (Σ evidence · observed_fraction + K)
```

At `observed_fraction = 1.0` this is exactly the old `u = K/S`, so 0.1 behaviour is a
special case rather than a change.

**0.3 — `competence_drop` is now a two-condition OR, and `trigger_reason` says which:**

```
competence_drop = (value_u > value_threshold) OR (observed_fraction < floor)
```

`value_u` is the plain, **quality-independent** `u = K/S`, and the threshold is calibrated
on it — so the value axis means "unusual state" no matter how much of it was observed.

Why not one threshold on the combined `u`? Because it under-fires on sensing loss. The
threshold has to sit above the in-distribution `u` at M1's *nominal* quality, and losing a
modality only pushes the combined `u` part of the way there. Measured: a single-threshold
trigger caught **39%** of blackout states. Testing the two conditions separately catches
**100%**, at the same 5% false-alarm rate on normal operation.

## Two axes of "the state is untrustworthy"

| Axis | What it means | What fires it |
|---|---|---|
| **value** | the state is unlike anything in training | cyclone: `value_u ≈ 1.0` at *any* `observed_fraction` |
| **sensing** | a modality is missing, so we cannot see the state | comms blackout: `observed_fraction` under the floor, on ordinary values |

Measured on the synthetic prototype:

| population | fires | reason breakdown |
|---|---|---|
| normal operation (`observed_fraction` 0.50) | 0.050 | `none` 0.95, `value` 0.05 |
| cyclone (value-OOD) | 1.000 | `value` 1.00 |
| comms blackout (`observed_fraction` < 0.40) | 1.000 | `sensing` 0.96, `both` 0.04 |

And the magnitude, in-distribution states with the evidence discounted:

| `observed_fraction` | 1.00 | 0.75 | 0.50 | 0.25 | 0.10 |
|---|---|---|---|---|---|
| mean `u` (in-distribution) | 0.105 | 0.130 | 0.173 | 0.271 | 0.440 |
| mean `u` (cyclone) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**Why you want `trigger_reason`.** A distribution shift and a sensing failure are different
problems. `value` means the grid is in a state the model has never seen — deliberating is
the right answer. `sensing` means the grid may be perfectly ordinary and we simply cannot
see it — deliberating on absent data buys nothing, and the useful response is closer to
"fall back to a conservative policy" or "go get the data". `both` is a cyclone that also
took the comms out, which is the worst case and probably deserves its own branch. Which
policy attaches to which reason is your call; the field exists so you can make it.

`epistemic_uncertainty` stays the magnitude — how bad — and `trigger_reason` is the kind.

## Where `observed_fraction` comes from

M1's `StateRepresentation` carries a `QualityMask`
(`packages/contracts/proto/common.proto`) with a `Quality` per feature, and
`observed_fraction` is the share of them that are `QUALITY_OBSERVED`. Per ADR 0004
anything not measured must not be labelled observed, so expect this to sit **well below
1.0 in practice** — M1's downscaled load series is `QUALITY_INTERPOLATED` on every row.

Nominal in the mock is `0.5`: four measured weather channels from NASA POWER, four
constructed electrical ones. A blackout drops one to three of the measured channels to
`QUALITY_MISSING`, giving `0.375` / `0.25` / `0.125` — hence a floor of `0.4`, just under
nominal. **The floor is not a tuning knob to taste: it belongs just below whatever M1's
steady-state `observed_fraction` turns out to be**, so ordinary interpolated data does not
fire and a lost modality does. If M1's real mask lands somewhere other than 0.5, this
moves with it.

Examples:
```json
{"timestamp":1787684554.74,"epistemic_uncertainty":1.0,"aleatoric_proxy":0.0,
 "competence_drop":true,"trigger_reason":"value","state_class":0,
 "class_probabilities":[0.333,0.333,0.333],"observed_fraction":0.5,
 "schema_version":"m2-out/0.3"}

{"timestamp":1787684554.74,"epistemic_uncertainty":0.526,"aleatoric_proxy":0.0,
 "competence_drop":true,"trigger_reason":"sensing","state_class":0,
 "class_probabilities":[0.796,0.106,0.098],"observed_fraction":0.375,
 "schema_version":"m2-out/0.3"}
```
Note the second one: `epistemic_uncertainty` is only 0.53 — a single combined threshold
would have let it through. The reason field is what makes it actionable.

Gate rule of thumb: **low u → stay System 1; `competence_drop == true` → escalate to System 2.**
Generate a live mock stream with `contract.build_output(...)` (see `run_demo.py`).
Contract to be frozen at PP1 (master plan §5); ping me before you hard-code field names.
