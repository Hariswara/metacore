# API documentation

- **Protobuf / gRPC** — generated from `packages/contracts/proto`. Run `task proto` to refresh.
- **OpenAPI** — served by the gateway at `/docs` when `services/gateway` is running.

This folder holds rendered/exported documentation only. The source of truth for every cross-module
message is `packages/contracts/proto`.
