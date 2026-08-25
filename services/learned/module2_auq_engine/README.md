# Module 2 — Agentic Epistemic Uncertainty Quantification (AUQ) Engine — Starter

A runnable, self-contained starter for **Module 2** (Duwaragie K., J26-DS-317).
It trains an Evidential Deep Learning head on **synthetic** island grid-states and shows
epistemic uncertainty staying low on normal states and rising to 1.0 on cyclone
(out-of-distribution) states — the core thesis of the module. It consumes M1's
`StateRepresentation` contract shape, mocked. Drop-in location in the repo:
`services/learned/module2_auq_engine/`.

**Competence drop triggers on value-OOD *or* sensing loss, and says which.** Uncertainty
has two independent axes: a state can be untrustworthy because it is unlike anything seen
in training (cyclone), or because a modality is missing so we cannot see it (comms
blackout). The emitted `u` is the magnitude, discounting the evidence by M1's
`observed_fraction`:

```
u = K / (Σ evidence · observed_fraction + K)
```

At `observed_fraction = 1.0` this is exactly the plain `u = K/S`, so the value axis is
unchanged and the paper's ablation still has a clean baseline (`edl.uncertainty`).

The **trigger** tests the two axes separately rather than thresholding the combined `u`:

```
competence_drop = (value_u > value_threshold) OR (observed_fraction < floor)
trigger_reason  ∈ {none, value, sensing, both}
```

One combined threshold under-fires on sensing loss — it has to sit above the
in-distribution `u` at M1's nominal quality, and a blackout only pushes `u` part of the way
there. Measured: a single threshold caught **39%** of blackout states; the two-condition
trigger catches **100%**, at the same 5% false-alarm rate on normal operation.

## Verified results (`python run_demo.py`)
| Metric | Value | Target |
|---|---|---|
| ID 3-class accuracy | 0.947 | — |
| value-only u (ID / OOD) | 0.106 / 1.000 | ID low, OOD high |
| emitted u (ID / OOD), at `observed_fraction` 0.50 | 0.172 / 1.000 | ID low, OOD high |
| emitted u (comms blackout) | 0.291 | rises although values are normal |
| AUROC (u, OOD vs ID) | 0.999 | ≥ 0.90 |
| AUPR (OOD) | 0.999 | high |
| FPR95 | 0.004 | low |
| ECE (calibration) | 0.051 | near 0 |

Competence-drop trigger — two conditions, OR'd, with the reason it fired:

| population | fires | reason breakdown |
|---|---|---|
| normal operation (`observed_fraction` 0.50) | 0.050 | `none` 0.95, `value` 0.05 |
| cyclone (value-OOD) | 1.000 | `value` 1.00 |
| comms blackout (`observed_fraction` < 0.40) | 1.000 | `sensing` 0.96, `both` 0.04 |

Magnitude along the quality axis — the same in-distribution states, less of them observed:

| `observed_fraction` | 1.00 | 0.75 | 0.50 | 0.25 | 0.10 |
|---|---|---|---|---|---|
| mean u (in-distribution) | 0.105 | 0.130 | 0.173 | 0.271 | 0.440 |
| mean u (cyclone) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Run
```bash
pip install -r requirements.txt
python run_demo.py
```

## Files
- `state_contract.py` — **dataclass mirror of M1's `StateRepresentation`** + `Envelope` / `QualityMask` / `ScenarioRef`. What the module develops against until M1's producer is live.
- `synthetic_data.py` — **mock M1 state generator** (ID normal + OOD cyclone), emitting `StateRepresentation`. Replace with the real M1→M2 adapter when it lands.
- `edl.py` — EDL head, `u = K/S`, quality-aware `u`, Dirichlet KL, EDL loss, OOD-aware evidence regulariser.
- `trigger.py` — two-condition competence-drop trigger (value threshold OR sensing floor, hysteresis, reason).
- `evaluate.py` — AUROC / AUPR / FPR95 / ECE (NumPy).
- `contract.py` + `M2_TO_M3_CONTRACT.md` — **M2→M3 message + mock stream for Saabir**.
- `run_demo.py` — end-to-end prototype; writes `sample_m2_to_m3.jsonl`.
- `config.yaml` — K, features, training and trigger settings (retune without code changes).

## Two things worth understanding
1. **Why OOD-aware regularisation is in the loss.** Plain EDL extrapolates *confidently* on far-OOD tabular inputs. The `ood_reg_weight` term drives evidence → 0 on far proxy points so cyclone states read u ≈ 1. This is standard practice and is your defensible design choice.
2. **Uncertainty only flags novelty in features the model uses.** The risk label depends on wind/rain too, so the net learns to attend to them; otherwise it would ignore the cyclone dimensions.
3. **The value threshold is calibrated on plain, quality-independent `u`, and the sensing floor sits just below M1's nominal `observed_fraction`.** M1's real state is roughly half interpolated by construction (ADR 0004), so a floor at 1.0 would be a permanent false alarm and a single combined threshold under-fires on real sensing loss. Keeping the axes apart is what makes both fire reliably — and the floor is not a taste knob: it belongs just under whatever M1's steady-state `observed_fraction` turns out to be.
4. **A blackout does not corrupt the feature values, only the mask.** That is deliberate. If missing channels also perturbed the numbers, blackout states would drift value-OOD and the two axes would stop being separable — which is exactly what the trigger is trying to distinguish. The mask carries the provenance; the array carries whatever M1 imputed.

## What is mocked vs real
- **Mock now:** the input states (stand-in for M1's shared ID/OOD scenario library) and the K safety classes' feature ranges.
- **Real later:** swap `synthetic_data.sample_states_*` for M1's real producer — `state_contract.py` already mirrors the message, so nothing downstream of `stack_features` changes. Then feed M4's rejection traces back in (feedback loop) and export to ONNX for the real-time path.
- **Not yet pinned by M1:** `embedding_dim`, a stamped `Envelope.schema_version`, and a label for the Eluvaitivu Oct–Dec 2025 degradation window. The mock embedding is noise — train on `node_features`, not `node_embedding`.

## Independent-work roadmap (while the data model is being set up)
1. **Define the K safety classes** for real (here: safe / stressed / critical) — the load-bearing decision.
2. **Publish `M2_TO_M3_CONTRACT.md` + the mock stream to Saabir** — unblocks his gate immediately.
3. **Agree the mock M1→M2 input shape with Zayan** so your adapter matches when his output lands.
4. **Harden the EDL core:** calibration (ECE, reliability diagram), epistemic/aleatoric split, latency profiling, ONNX export.
5. **Grow the synthetic scenario set** toward Burevi/Ditwah-like profiles for your paper's OOD evaluation.
