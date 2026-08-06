# Architecture overview

MetaCore is a dual-process metacognitive agent that converts statistical uncertainty into an
active, physically-verified control signal for an islanded microgrid under multi-hazard stress.

## Three layers

**Layer 1 — perception (`services/learned/module1_state_forecasting`).**
Fuses meteorology, geospatial hazard layers, maritime logistics and grid telemetry into a spatial
graph whose nodes are grid assets, and predicts per-node dynamic physical vulnerability.

**Layer 2 — self-knowledge and decision.**
`module2_auq_engine` places an evidential head over the state representation, isolates epistemic
from aleatoric uncertainty, and emits a competence-drop trigger. `module3_metapolicy` consumes that
trigger and decides whether to stay in the fast rule-based System 1 or pay the cost of deliberating
in System 2. The trained policy is exported to ONNX and served from Go in the hot loop.

**Layer 3 — the firewall (`deterministic-core`).**
Every proposed action is simulated in an OpenDSS digital twin and hard-approved or hard-rejected
against voltage bands, thermal ratings and solver convergence. Rejections become structured
violation evidence and a human-readable causal log.

## Why the gate is synchronous

The orchestrator calls `deterministic-core` with a blocking gRPC call and waits. Fire-and-forget
would make the firewall bypassable under load, which defeats its purpose: the guarantee is not
"unsafe actions are usually caught" but "unsafe actions cannot reach actuation".

## Latency

The reactive path is budgeted end-to-end rather than per module, because the reserve left after
perception, uncertainty, gating and verification *is* the budget Module 3 has available to
deliberate with. The allocation is recorded in the master plan and re-measured from PP1 onward.

## Design philosophy

Deterministic rules are retained wherever they are already the safe, fast, interpretable norm —
reactive protection, economic dispatch, load-shed priority, physics verification. Machine learning
is introduced only for the three sub-problems where no adequate rule exists: forecasting localised
vulnerability, isolating epistemic uncertainty, and learning when to deliberate.
