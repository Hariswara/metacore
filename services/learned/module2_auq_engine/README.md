# Module 2 — Agentic Epistemic Uncertainty Quantification (AUQ) Engine — Starter

A runnable, self-contained starter for **Module 2** (Duwaragie K., J26-DS-317).
It trains an Evidential Deep Learning head on **synthetic** island grid-states and shows
epistemic uncertainty staying low on normal states and rising to 1.0 on cyclone
(out-of-distribution) states — the core thesis of the module. It consumes M1's
`StateRepresentation` contract shape, mocked. Drop-in location in the repo:
`services/learned/module2_auq_engine/`.

**Competence drop now triggers on value-OOD *or* quality degradation.** Uncertainty has two
independent axes: a state can be untrustworthy because it is unlike anything seen in
training (cyclone), or because M1 could only interpolate it (comms blackout). The emitted
`u` discounts the evidence by M1's `observed_fraction`:

```
u = K / (Σ evidence · observed_fraction + K)
```

At `observed_fraction = 1.0` this is exactly the plain `u = K/S`, so the value axis is
unchanged and the paper's ablation still has a clean baseline (`edl.uncertainty`).

## Verified results (`python run_demo.py`)
| Metric | Value | Target |
|---|---|---|
| ID 3-class accuracy | 0.961 | — |
| value-only u (ID / OOD) | 0.105 / 1.000 | ID low, OOD high |
| emitted u (ID / OOD), at `observed_fraction` 0.50 | 0.173 / 1.000 | ID low, OOD high |
| AUROC (u, OOD vs ID) | 0.999 | ≥ 0.90 |
| AUPR (OOD) | 0.987 | high |
| FPR95 | 0.003 | low |
| ECE (calibration) | 0.047 | near 0 |
| competence-drop trigger (ID / OOD) | 0.05 / 1.00 | catch OOD, few false alarms |

Quality axis — the same in-distribution states, with less of them observed:

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
- `trigger.py` — competence-drop trigger (calibrated threshold + hysteresis).
- `evaluate.py` — AUROC / AUPR / FPR95 / ECE (NumPy).
- `contract.py` + `M2_TO_M3_CONTRACT.md` — **M2→M3 message + mock stream for Saabir**.
- `run_demo.py` — end-to-end prototype; writes `sample_m2_to_m3.jsonl`.
- `config.yaml` — K, features, training and trigger settings (retune without code changes).

## Two things worth understanding
1. **Why OOD-aware regularisation is in the loss.** Plain EDL extrapolates *confidently* on far-OOD tabular inputs. The `ood_reg_weight` term drives evidence → 0 on far proxy points so cyclone states read u ≈ 1. This is standard practice and is your defensible design choice.
2. **Uncertainty only flags novelty in features the model uses.** The risk label depends on wind/rain too, so the net learns to attend to them; otherwise it would ignore the cyclone dimensions.
3. **The trigger threshold is calibrated at the fleet's *nominal* quality, not at perfect quality.** M1's real state is roughly half interpolated by construction (ADR 0004), so escalating on every interpolated row would be a permanent false alarm. The threshold is set on the in-distribution u distribution at the quality M1 actually delivers, and fires when quality drops *below* that baseline.

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
