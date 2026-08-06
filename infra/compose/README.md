# Compose profiles

The root `docker-compose.yml` is the entrypoint; profiles keep a partial run cheap.

| Profile | Brings up |
|---|---|
| `default` | The whole agent, end to end |
| `m1` / `m2` / `m3` | One learned module plus gateway and dashboard |
| `core` | `deterministic-core` only — the firewall in isolation |
| `ingest` | Ingestion service and the message bus |
| `train` | Module 3 training container (not part of the runtime path) |
| `web` | Dashboard and gateway only |

Add fragments here if a profile grows beyond what fits comfortably in the root file.
