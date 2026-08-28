"""Module 1 calibration path — the offline half of ingestion.

See `docs/adr/0004-two-ingestion-paths.md`.

Reads `data/external/**` and `data/raw/**`, emits a versioned parameter set into `data/processed/`.
Batch only: nothing here opens a socket, publishes to the bus, or sits on a request path.

Deliberately standard-library only, even though `metacore-module1` declares pandas and torch. These
stages are the reconciliation gate for state-entity data, and they should run in CI in seconds
without resolving a deep-learning dependency tree.
"""
