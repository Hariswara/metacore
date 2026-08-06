# infra/ci

Shared checks. **Workflows are not here** — GitHub Actions only reads `.github/workflows/`, so
that is where the four lanes live (`python`, `go`, `frontend`, `deterministic-core`). Keeping a
second copy in this folder would guarantee the two drift apart.

| File | Called by |
|---|---|
| `check_core_purity.sh` | `.github/workflows/deterministic-core.yml`, `task lint`, `task audit:core`, the pre-commit hook |

`check_core_purity.sh` is the executable form of
[ADR 0003](../../docs/adr/0003-deterministic-core-isolation.md). It asserts three things:

1. no ML dependency is declared in any `deterministic-core/*/pyproject.toml`;
2. no ML module is imported anywhere inside `deterministic-core/`;
3. OpenDSS is imported only from `deterministic-core/`.

Exit code 1 with the offending file and line on any violation. Add a check here rather than
inlining it in a workflow, so the same rule runs locally before it ever reaches CI.
