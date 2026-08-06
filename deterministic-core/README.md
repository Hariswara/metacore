# deterministic-core — Module 4

**AUDIT: zero ML dependencies · sole caller of OpenDSS**

A top-level sibling of `services/`, not a child of it, so the learned/deterministic boundary is
visible in the first line of `ls`. See [ADR 0003](../docs/adr/0003-deterministic-core-isolation.md).

| Package | Role |
|---|---|
| `module4_verification` | Loads the grid state into an OpenDSS twin, applies the proposed action, solves power flow, returns `APPROVE` or `REJECT` with structured violation evidence. Deterministic. |
| `module4_semantic_translation` | Turns that structured evidence into a human-readable causal log. Sits **downstream of the decision** and can never change it. |

## Invariants

1. Neither `pyproject.toml` declares an ML dependency. `infra/ci/check_core_purity.sh` fails the
   build if one appears, and runs as a pre-commit hook.
2. No component outside this directory imports OpenDSS.
3. Failure is safe: bad input or solver non-convergence produces `REJECT`, never `APPROVE`.
4. OpenDSS versions are pinned and physics tests run on Linux in CI, so a verdict is never
   operating-system dependent.

## Scope

Single-step, model-based safety shielding — each proposed action is checked once against the
current state. Multi-step / model-predictive shielding is explicitly future work, not delivered.
