# Git workflow

Four members, one monorepo, four independently-publishable components. The workflow optimises for
never blocking each other.

## Branches

- `main` — always green. Protected once a remote exists.
- `m<n>/<short-description>` — module work, e.g. `m1/graph-construction`, `m4/powerflow-check`.
- `contracts/<name>` — changes under `packages/contracts`. Always reviewed by every consumer of the
  contract, because a schema change is a change to somebody else's module.
- `docs/<name>`, `infra/<name>` — non-code changes.

## Commits

Conventional-commit prefixes with a module scope:

```
feat(m1): fuse topographic wetness index into node features
fix(m4): reject on solver non-convergence instead of raising
chore(contracts): freeze module1 state representation at v1
docs(adr): record the deterministic-core isolation decision
```

## Pull requests

- One PR per logical change; a PR that touches two modules needs both owners.
- A PR that edits `packages/contracts/proto` must state the version bump and whether it is
  backward-compatible. Breaking a frozen contract requires a major bump and an adapter.
- CI lanes must pass: python, go, frontend, and the `deterministic-core` purity lane.

## The rule that keeps the team unblocked

Publish schemas before implementations. Every member develops against a mock of their upstream
input and publishes their downstream output schema early. A schema is cheap; a rewrite is not.
