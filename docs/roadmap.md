# Roadmap

Deliberately **milestone-agnostic** — no calendar dates are baked into the repository. Dated
schedules live in the planning documents (individual module plans, master timeline and Gantt),
so a schedule change is made in one place and never leaves the code out of sync.

Phases are shared by all four modules:

| Phase | Milestone | Definition of done |
|---|---|---|
| A | TAF | Requirements fixed, load-bearing design decisions taken, repo scaffolded, contracts drafted |
| B | Charter + Proposal + Demo | Four working prototypes plus one rehearsed end-to-end walkthrough |
| C | PP1 (~50%) | Four core engines with measured metrics; forward-direction contracts frozen and live |
| D | Papers | Four individual papers drafted with ablations; S1 models and datasets frozen |
| E | Integration | Full chain plus both feedback loops running unattended; unified dashboard |
| F | PP2 (~90%) + Final | Benchmarks on historical cyclone profiles; papers submitted; theses and vivas |

## Contract register

The dependency chain is the critical path, not any single module. Every cross-module schema has an
owner, a consumer and a committed state (`v0` → `frozen` → `live`). The authoritative register is in
the master plan; `packages/contracts/proto` is the executable version of it.

| Contract | Owner | Consumer |
|---|---|---|
| State representation | M1 | M2 |
| Grid topology + state snapshot | M1 | M4 |
| System-context features | M1 | M3 |
| Uncertainty scalar + competence-drop trigger | M2 | M3 |
| Proposed control action | M3 | M4 |
| Verification rejection trace | M4 | M2 |
| Approve/reject decision + reason | M4 | M3 |

## Shared assets

- **ID/OOD scenario library** (owner: M1) — normal operation, Burevi and Ditwah cyclone profiles,
  communication blackout, inter-monsoon dual-drop. Every module evaluates on the same episodes.
- **Scenario replay harness** (owner: M1) — one command, one episode, four modules, reproducible output.
