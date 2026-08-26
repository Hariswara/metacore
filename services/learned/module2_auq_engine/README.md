# Module 2 — Agentic Epistemic Uncertainty Quantification (AUQ) Engine — Starter

A runnable, self-contained starter for **Module 2** (Duwaragie K., J26-DS-317).
It trains an Evidential Deep Learning head on **synthetic** island grid-states and shows
epistemic uncertainty staying low on normal states and rising to 1.0 on cyclone
(out-of-distribution) states — the core thesis of the module. It consumes M1's real
pinned `StateRepresentation` contract — **64-dim embedding, 28 ordered features,
`SchemaVersion(1, 0)`** — read from `metacore_contracts.state_schema` rather than copied,
so the mock cannot drift from the schema. Drop-in location in the repo:
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

**Headline — `u` predicts error.** This is the claim the module rests on, and it is not
the same claim as calibration: ECE asks whether the probabilities are honest, selective
prediction asks whether the uncertainty is *usable* as a decision to abstain.

| Selective prediction (in-distribution) | Value |
|---|---|
| accuracy at full coverage | 0.945 |
| accuracy at 50% coverage (reject highest `u`) | **1.000** |
| accuracy at 75% coverage | 0.997 |
| **AURC** | **0.0057** |
| AURC with the ranking shuffled | 0.0498 |

Rejecting the least-certain half of the states removes every error. The shuffled row is
the control: a random ranking scores ~9× worse, so the result comes from `u` and not from
the data being easy.

| Metric | Value | Target |
|---|---|---|
| ID 3-class accuracy | 0.945 | — |
| value-only u (ID / OOD) | 0.067 / 1.000 | ID low, OOD high |
| emitted u (ID / OOD), at nominal `observed_fraction` 0.4286 | 0.133 / 1.000 | ID low, OOD high |
| emitted u (comms blackout) | 0.209 | rises although values are normal |
| AUROC (u, OOD vs ID) | 0.999 | ≥ 0.90 |
| AUPR (OOD) | 0.999 | high |
| FPR95 | 0.002 | low |
| ECE (calibration) | 0.028 | near 0 |

Competence-drop trigger — two conditions, OR'd, with the reason it fired:

| population | fires | reason breakdown |
|---|---|---|
| normal operation (`observed_fraction` 0.4286) | 0.050 | `none` 0.95, `value` 0.05 |
| cyclone (value-OOD) | 1.000 | `value` 1.00 |
| comms blackout (`observed_fraction` < 0.40) | 1.000 | `sensing` 0.94, `both` 0.06 |

Magnitude along the quality axis — the same in-distribution states, less of them observed:

| `observed_fraction` | 1.00 | 0.75 | 0.50 | **0.4286** | 0.25 | 0.10 |
|---|---|---|---|---|---|---|
| mean u (in-distribution) | 0.067 | 0.085 | 0.118 | **0.133** | 0.199 | 0.356 |
| mean u (cyclone) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### Why the trigger is not just a threshold on `u`

Ranking a mixed stream by the combined `u` and keeping the most-confident half:

| | normal | cyclone | blackout |
|---|---|---|---|
| full stream | 0.385 | 0.308 | 0.308 |
| kept half | 0.653 | **0.000** | **0.347** |

The magnitude clears the value axis completely and does **not** clear the sensing axis —
a blackout at `observed_fraction` ~0.25 discounts a confident state into the same `u`
range as an ordinary one, so blackouts survive the cut (their share even rises, because
the cyclones ahead of them are removed). The explicit `observed_fraction` floor is what
catches them: the two-condition trigger fires on 1.000 of cyclones **and** 1.000 of
blackouts, at 0.050 on normal operation.

## Baselines and the ablation (`python benchmark.py` → `comparison_table.json`)

Same architecture, normalisation and training data throughout, so the comparison is about
the uncertainty mechanism and nothing else.

| method | AUROC | AUPR | FPR95 | ECE | ID acc | ms/sample |
|---|---|---|---|---|---|---|
| Softmax max-prob | 0.019 | 0.384 | 1.000 | 0.054 | 0.937 | 0.0004 |
| MC-Dropout (T=20) | 0.000 | 0.265 | 1.000 | 0.019 | 0.940 | 0.0467 |
| **EDL (ours)** | **0.999** | **0.999** | **0.001** | 0.025 | 0.933 | 0.0008 |
| EDL, no OOD-reg (ablation) | 0.000 | 0.265 | 1.000 | 0.027 | 0.929 | 0.0009 |

Two findings, and the second is the contribution.

**1. The baselines do not merely underperform — they invert.** An AUROC of 0.019 means the
score is *anti*-correlated with being out of distribution. Measured directly: the softmax
network's max probability saturates at **1.0000 on essentially every cyclone state**, above
its mean on normal ones. It is more certain about conditions it has never seen than
about the ones it was trained on. This is the documented behaviour of ReLU networks far
from their training data (Hein et al., *Why ReLU networks yield high-confidence
predictions far away from the training data*, CVPR 2019), not an artefact of this setup —
which is why a better-tuned baseline would not rescue it. AUPR says the same thing: the
positive base rate here is 0.444, and the failing methods sit at 0.265-0.384, below chance.

**2. The OOD-aware regulariser is load-bearing, not the Dirichlet head.** The ablation is
identical in every respect except `ood_reg = 0`, and it collapses to 0.000 — the same
failure as the softmax baseline. Evidential output on its own does not solve far-OOD on
tabular data; driving evidence → 0 on far proxy points is what does.

Worth noting what the table does *not* say: the baselines rank in-distribution errors
just as well as we do — softmax AURC **0.008** against our **0.009**, i.e. marginally
*better*. They fail specifically at recognising a state they have never seen, which is the
one thing this module exists to do. Selective prediction and OOD detection are different
questions, and that is exactly why the module needs both evaluations.

MC-Dropout is both inverted and ~58× slower per sample, because its score costs T=20
forward passes. On a gate that runs per control step, that is the difference between
viable and not.

Every claim above is asserted as an *ordering* in `tests/test_baselines.py` at reduced
scale (~5s) — exact values move with the seed, the ranking does not.

## Serving: ONNX export and batch-1 latency (optional `onnx` extra)

Feasibility evidence for M3's deliberation budget, and for `Health.last_step_latency_ms`
in `packages/contracts/proto/common.proto`.

```bash
uv sync --package metacore-module2 --extra onnx
python export_onnx.py     # writes edl.onnx (+ model card), verified against torch
python bench_latency.py   # writes latency_table.json
```

| backend | mean ms | p50 ms | p99 ms | amortised ms/sample | optimism |
|---|---|---|---|---|---|
| torch-eager | 0.0973 | 0.0751 | 0.3098 | 0.0003 | **276×** |
| ONNX Runtime | 0.0383 | **0.0355** | **0.0870** | n/a | — |

**Batch-1 is the only honest number here.** The gate scores one state per control step, so
its cost is the latency of a single call. Dividing a batch-1000 forward pass by 1000 hides
per-call dispatch and kernel launch completely and reports a figure the real path never
sees — here it is **276× more optimistic** than what the gate actually pays. The
`ms/sample` column in `comparison_table.json` is exactly that optimistic, and should be
read as throughput, not latency.

ONNX Runtime is 2.1× faster than torch-eager at batch-1 (p50 0.0355 ms vs 0.0751 ms) and
cuts the tail by 3.6× (p99 0.0870 ms vs 0.3098 ms). The tail is the number that matters for a
step deadline: a p99 that blows the budget is a missed step, not a slow one. Both backends
are comfortably sub-millisecond, so uncertainty scoring is not what constrains the control
loop — deliberation is, which is the premise M3's cost model rests on.

The amortised column is `n/a` for ONNX because the artifact is **fixed at batch 1** and
cannot be run batched at all — feeding it 1000 rows is an `InvalidArgument`. Batched
offline evaluation needs a separate export with `dynamic_shapes`.

### What is in the graph, and what is not

`edl.onnx` contains the network and produces **evidence** only. `u = K/S`, the
`observed_fraction` discount and the two-condition trigger stay in Python, in `infer.py`.
That split is deliberate: `observed_fraction` is a per-state runtime input arriving with
M1's `QualityMask`, and the thresholds are calibrated per deployment — freezing either
into the artifact would mean re-exporting to retune a threshold.

`edl.onnx.json` is the model card, and it carries the **normalisation statistics**. Without
them the graph is unusable: it expects standardised input, and `mu`/`sd` are training-set
properties that live nowhere else. The artifact is self-contained (`external_data=False`)
— 28 inputs to 3 evidence outputs, opset 18, no sidecar file to lose on the way to deployment.

`edl.onnx` itself is **not committed**: the repo-wide `.gitignore` excludes `*.onnx` under
"Models & artefacts". Regenerate it with `python export_onnx.py` — training is seeded and
the export is verified against torch to 1e-5, so the artifact is reproducible rather than
merely absent. The model card is committed, and records the verification result.

`infer.OnnxAUQ` is the path M3 calls: `load()` → `calibrate(id_features)` → `score(features,
observed_fraction)` → an `M2Output`. It runs the same two axes as the torch path and needs
no torch at serving time.

## Figures (optional `viz` extra)

Metrics are computed in NumPy and emitted as **data** — `run_demo.py` writes
`eval_tables.json` (reliability bins, risk-coverage points, retained composition). Nothing
in the test lane imports a plotting stack.

```bash
uv sync --package metacore-module2 --extra viz   # or: pip install 'matplotlib>=3.7'
python run_demo.py                               # writes eval_tables.json
python plots.py                                  # writes reliability.png, risk_coverage.png
```

## Run
```bash
pip install -r requirements.txt
python run_demo.py
```

## Files
- `state_contract.py` — **dataclass mirror of M1's `StateRepresentation`** + `Envelope` / `QualityMask` / `ScenarioRef`. What the module develops against until M1's producer is live.
- `synthetic_data.py` — **mock M1 state generator** (ID normal + OOD cyclone + comms blackout), emitting `StateRepresentation` over the real 28 pinned features. Replace with the real M1→M2 adapter when it lands.
- `edl.py` — EDL head, `u = K/S`, quality-aware `u`, Dirichlet KL, EDL loss, OOD-aware evidence regulariser.
- `trigger.py` — two-condition competence-drop trigger (value threshold OR sensing floor, hysteresis, reason).
- `evaluate.py` — AUROC / AUPR / FPR95 / ECE plus risk-coverage / AURC, reliability and retained-composition **tables** (NumPy, no plotting).
- `plots.py` — renders those tables (reliability diagram, risk-coverage curve). Needs the `viz` extra; imported by nothing else.
- `baselines.py` — softmax max-prob, MC-Dropout, and the EDL / EDL-without-OOD-reg pair, on one architecture.
- `benchmark.py` + `comparison_table.json` — full-scale comparison **script** (not run in CI) and the table it writes.
- `export_onnx.py` + `edl.onnx` + `edl.onnx.json` — ONNX export **script**, the batch-1 artifact, and its model card (architecture, opset, normalisation statistics).
- `infer.py` — the serving path M3 calls: ONNX evidence + uncertainty + trigger, no torch at inference.
- `bench_latency.py` + `latency_table.json` — batch-1 p50/p99 **script** for torch-eager vs ONNX Runtime.
- `contract.py` + `M2_TO_M3_CONTRACT.md` — **M2→M3 message + mock stream for Saabir**.
- `run_demo.py` — end-to-end prototype; writes `sample_m2_to_m3.jsonl` and `eval_tables.json`.
- `config.yaml` — K, features, training and trigger settings (retune without code changes).

## Four things worth understanding
1. **Why OOD-aware regularisation is in the loss.** Plain EDL extrapolates *confidently* on far-OOD tabular inputs. The `ood_reg_weight` term drives evidence → 0 on far proxy points so cyclone states read u ≈ 1. This is standard practice and is your defensible design choice.
2. **Uncertainty only flags novelty in features the model uses.** The risk label depends on wind/rain too, so the net learns to attend to them; otherwise it would ignore the cyclone dimensions.
3. **The value threshold is calibrated on plain, quality-independent `u`, and the sensing floor sits just below M1's nominal `observed_fraction`.** M1's real state is roughly half interpolated by construction (ADR 0004), so a floor at 1.0 would be a permanent false alarm and a single combined threshold under-fires on real sensing loss. Keeping the axes apart is what makes both fire reliably — and the floor is not a taste knob: it belongs just under whatever M1's steady-state `observed_fraction` turns out to be.
4. **A blackout does not corrupt the feature values, only the mask.** That is deliberate. If missing channels also perturbed the numbers, blackout states would drift value-OOD and the two axes would stop being separable — which is exactly what the trigger is trying to distinguish. The mask carries the provenance; the array carries whatever M1 imputed.

## What is mocked vs real
- **Mock now:** the input *values* only. The schema around them — names, order, embedding width, per-feature quality, version — is the real pinned contract, imported from `metacore_contracts.state_schema`.
- **Real later:** swap `synthetic_data.sample_states_*` for M1's real producer — `state_contract.py` already mirrors the message, so nothing downstream of `stack_features` changes. Then feed M4's rejection traces back in (feedback loop) and export to ONNX for the real-time path.
- **Now pinned by M1:** `embedding_dim = 64`, `SchemaVersion(1, 0)`, the 28 feature names, and the Eluvaitivu Oct–Dec 2025 label (`eluvaitivu-hybrid-decay-2025q4`, in `data/processed/events.csv`). The mock embedding is still noise — train on `node_features`, not `node_embedding`.

## Independent-work roadmap (while the data model is being set up)
1. **Define the K safety classes** for real (here: safe / stressed / critical) — the load-bearing decision.
2. **Publish `M2_TO_M3_CONTRACT.md` + the mock stream to Saabir** — unblocks his gate immediately.
3. **Agree the mock M1→M2 input shape with Zayan** so your adapter matches when his output lands.
4. **Harden the EDL core:** calibration (ECE, reliability diagram), epistemic/aleatoric split, latency profiling, ONNX export.
5. **Grow the synthetic scenario set** toward Burevi/Ditwah-like profiles for your paper's OOD evaluation.
