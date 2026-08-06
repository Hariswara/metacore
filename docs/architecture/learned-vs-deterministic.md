# The learned / deterministic boundary

This boundary is the project's central claim. It is enforced by the directory layout, by
dependency declarations and by CI — not by convention alone.

## Where the line sits

| | Learned | Deterministic |
|---|---|---|
| Location | `services/learned/*`, `services/realtime/gating_decision_svc` | `deterministic-core/*` |
| Modules | M1, M2, M3 | M4 |
| May be wrong | Yes — that is why M4 exists | No — it enforces physical law |
| Dependencies | PyTorch, PyTorch Geometric, ONNX Runtime | OpenDSS bindings, numpy. **No ML.** |
| Deploy unit | One container per module | Its own container, own lockfile |

## The three rules

1. **`deterministic-core` declares no ML dependency.** Enforced by `infra/ci/check_core_purity.sh`,
   which runs in CI and as a pre-commit hook. If a torch import ever appears, the build fails.
2. **`deterministic-core` is the only caller of OpenDSS.** No learned module may open the digital
   twin, because a learned component that can simulate physics can also learn to approximate it —
   and an approximated firewall is not a firewall.
3. **Nothing crosses the boundary except a published contract.** Learned services and the
   deterministic core exchange protobuf messages defined in `packages/contracts`, never imports.

## Why `deterministic-core` is a top-level sibling

It sits beside `services/`, not inside it, so the split is visible in the first line of `ls`. It has
its own `pyproject.toml`, its own Dockerfile and its own CI lane, which means it can be audited,
tested and cited in isolation — the property the verification argument depends on.

## The one permitted nuance

Semantic translation of a rejection (`module4_semantic_translation`) is downstream of the
`APPROVE`/`REJECT` decision, never part of it. If a learned or template-constrained generator is
ever used there, the decision itself remains deterministic, and the boundary must be described
explicitly in the paper rather than implying the whole module is learning-free.
