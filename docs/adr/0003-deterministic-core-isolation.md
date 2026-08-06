# ADR 0003 — `deterministic-core` is a top-level package with zero ML dependencies

**Status:** accepted · **Deciders:** whole team (owner: Hariswara, M4)

## Context

The literature motivating this project reports that supplying a model with its own calibration
scores yields no significant improvement in safe action selection, whereas an external
architectural constraint substantially reduces the confident-failure rate. The safety argument
therefore rests on the verifier being external and deterministic — not on the agent's introspection.

That argument is only credible if the isolation is real and auditable, rather than asserted in prose.

## Decision

Module 4 lives in `deterministic-core/`, a **top-level sibling** of `services/`, with:

- its own `pyproject.toml` declaring **no** ML dependency;
- its own Dockerfile and deploy unit;
- exclusive ownership of the OpenDSS bindings — no other component may import them;
- a dedicated CI lane (`.github/workflows/deterministic-core.yml`) that fails the build if an ML dependency
  appears, and the same check as a pre-commit hook.

## Rationale

Putting it inside `services/` would make the boundary a naming convention. As a top-level sibling
it is visible in the first line of `ls`, and the dependency check makes accidental coupling a build
failure rather than a code-review catch.

## Consequences

- Any learned explanation layer must sit downstream of the `APPROVE`/`REJECT` decision, in
  `module4_semantic_translation`, and be described explicitly as such in the paper.
- `deterministic-core` tests run on Linux in CI with pinned OpenDSS versions, so a firewall result
  is never operating-system dependent.
