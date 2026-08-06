# ADR 0002 — Protobuf contracts as the single source of truth

**Status:** accepted · **Deciders:** whole team

## Context

A polyglot split multiplies the contract surface: the same message must be understood by Python,
Go and TypeScript. Merging the four individual plans also surfaced two date conflicts on
inter-module schemas, which is evidence that the schemas — not the implementations — are the
critical path.

## Decision

Every cross-module message is defined once as protobuf in `packages/contracts/proto`, and Python,
Go and TypeScript stubs are generated from it via `buf`. Runtime transport is gRPC. The streaming
ingestion path may use a message bus, but **the call into `deterministic-core` is always a blocking
synchronous gRPC call**, so the firewall is structurally impossible to bypass.

Every message carries a `schema_version`. Once a contract is frozen it changes only by a versioned
bump with a backward-compatible adapter — never silently.

## Rationale

Publishing a schema is cheap and can happen weeks before the model behind it works. That is exactly
what unblocks a four-person chain where each member's input is somebody else's unfinished output.

## Consequences

- A proto change is a team event: reviewed by every consumer of that contract.
- Generated code is committed so that a fresh clone builds without a proto toolchain.
