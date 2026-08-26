# MetaCore

**A Principled-Hybrid Metacognitive AI Agent for Multi-Hazard Resilience in Islanded Cyber-Physical Microgrids**

Project `J26-DS-317` · SLIIT IT4010 Research Project (2026) · Research Cluster: AIMS · Specialization: Data Science

| Module | Component | Owner | Type |
|---|---|---|---|
| M1 | Multi-Modal Spatiotemporal State Representation | Zayan M.F.M (IT23248212) | Learned — ST-GNN |
| M2 | Agentic Epistemic Uncertainty Quantification | Duwaragie K. (IT23270442) | Learned — Evidential Deep Learning |
| M3 | Cost-Aware Metacognitive Gating & Meta-Policy | Saabir S. (IT23432598) | Learned — Meta-RL |
| M4 | Abductive Physics-Informed Verification | Hariswara S. (IT23291782) | **Deterministic** — OpenDSS |

Supervisor: Mr. Samadhi Chathuranga Rathnayake · Co-Supervisor: Ms. Thamali Dassanayaka

---

## The pipeline

```
                    ┌──────── learned ────────┐   ┌─ deterministic ─┐

  ingestion ──▶  M1 state  ──▶  M2 epistemic  ──▶  M3 gating  ──▶  M4 physics  ──▶ actuation
                 representation  uncertainty       & meta-policy    firewall
                                      ▲                  ▲               │
                                      └──────────────────┴───────────────┘
                                        rejection trace / verdict feedback
```

`M4` is a **hard synchronous gate**. The orchestrator blocks on it; no module reaches actuation
without an `APPROVE`. See [`docs/architecture/learned-vs-deterministic.md`](docs/architecture/learned-vs-deterministic.md).

## Workspace map

| Path | What lives there |
|---|---|
| `apps/dashboard` | React + TypeScript + Vite research dashboard (four module views on one time axis) |
| `services/gateway` | FastAPI — the single external API surface |
| `services/orchestrator` | Pipeline order and the mandatory M4 gate |
| `services/learned/*` | Modules 1–3. **PyTorch lives only under here.** |
| `services/realtime/*` | Go services on the latency-critical path |
| `deterministic-core/*` | **Module 4.** Zero ML dependencies. The only caller of OpenDSS. |
| `packages/contracts` | Protobuf — one source of truth for every cross-module schema |
| `data` | Open + simulated datasets, DVC-tracked (blobs git-ignored) |
| `docs` | Architecture, ADRs, roadmap, git workflow |
| `reports` | SLIIT FoC Data Science deliverables |

## Quickstart

Prerequisites are **pnpm, uv and Go** — nothing else. `task`, `buf`, `dvc`, `pytest` and `ruff` are
declared dev dependencies (package.json and the pyproject `dev` group), so they install with the
repo rather than being expected on the machine.

```bash
pnpm install              # brings in task + buf, pinned
pnpm exec task setup      # uv sync --all-packages, go work sync
```

`task` then lives at `node_modules/.bin/task`; put that directory on your PATH, or keep prefixing
with `pnpm exec`.

```bash
task proto     # regenerate Python / Go / TS stubs from packages/contracts/proto
task data      # rebuild Module 1 calibration artifacts, run the reconciliation gate
task test      # run every language lane
task dev       # docker compose up the default profile
```

Run a subset instead of the whole system:

```bash
docker compose --profile m2 up      # module2 + gateway + dashboard
docker compose --profile core up    # deterministic-core only
```

## Conventions

- **Publish schemas before implementations.** Develop against a mock of your upstream input;
  publish your downstream output schema early. Once frozen, a contract changes only by a
  versioned minor bump with a backward-compatible adapter — never silently.
- **Boundary hygiene.** M1–M3 live under `services/learned` and never import `deterministic-core`.
  CI enforces that `deterministic-core` has no ML dependency.
- **Reproducibility.** Seeded, config-driven, system-as-code. A model card per trained checkpoint,
  a dataset version tag per reported result.
- **Baselines.** No number is reported without the baseline it beats.

Planning documents (individual module plans, master timeline and Gantt charts) live alongside this
repository in the project research folder; `docs/roadmap.md` is the milestone-agnostic in-repo view.
