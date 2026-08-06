# ADR 0001 — One monorepo with plain workspaces, not Nx or Turborepo

**Status:** accepted · **Deciders:** whole team

## Context

Four members produce four independently-publishable components that must nonetheless integrate into
one agent in Semester 2. The stack is polyglot: Python for the learned modules and the deterministic
core, Go for the latency-critical path, TypeScript for the dashboard.

## Decision

A single repository, organised as language-native workspaces unified by one root task runner:

- `uv` workspace for Python services
- `go.work` for the Go modules
- `pnpm` workspace for the dashboard and shared TS types
- `Taskfile.yml` at the root as the only entrypoint a developer needs to learn

## Rationale

Nx and Turborepo are Node-centric. Their caching and task graphs are excellent for JS/TS and do not
natively manage Python or Go; forcing a polyglot repo into them produces configuration you fight
rather than use. Language-native tooling plus one task runner gives the same ergonomics without the
adapter layer.

Critically, this keeps `deterministic-core` genuinely standalone — its own environment, its own
lockfile, no shared JS tooling leaking in. That isolation is what the verification argument claims,
so the build system must not quietly undermine it.

## Consequences

- No cross-language build caching. Acceptable at this repo size.
- Each service owns its own dependency declaration, which is what makes per-service Docker builds
  and the purity check possible.
