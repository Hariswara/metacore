# contracts

One source of truth for every message that crosses a module boundary. Python, Go and TypeScript
stubs are generated from `proto/` — nothing is hand-written per language.

```bash
task proto     # buf lint && buf generate && fix_python_imports.py
```

## What is generated and what is not

| Path | Generated? |
|---|---|
| `python/metacore_contracts/*_pb2*.py`, `go/`, `ts/` | Yes — `task proto` overwrites. Do not hand-edit. |
| `fix_python_imports.py` | No. Post-generation step; see below. |
| `python/metacore_contracts/state_schema.py`, `schema/*.json` | No. Hand-written contract pins. `task proto` does not touch them. |

`fix_python_imports.py` runs last. protoc derives its import statement from the *proto* import
path rather than from the Python package the file lands in, so a flat `proto/` directory emits
`import common_pb2` inside `metacore_contracts/module1_pb2.py` — importable only if the package
directory is itself on `sys.path`, which it is not once installed. The script rewrites those to
package-relative imports and is idempotent.

## Contract pins

A `.proto` can declare that a message has an `embedding_dim` and a `repeated string
feature_names`. It cannot declare what they *are*, and a consumer sizing a model against the proto
alone is still guessing. The pins close that gap:

| Pin | Contents |
|---|---|
| [`schema/module1_state_v1.json`](python/metacore_contracts/schema/module1_state_v1.json) | `embedding_dim = 64`, 28 ordered `feature_names`, per-feature unit, source and calibration `QualityMask` value |

```python
from metacore_contracts.state_schema import EMBEDDING_DIM, FEATURE_NAMES, SCHEMA_VERSION
```

## Versioning rule

Every message carries `schema_version`. A contract moves through three states:

| State | Meaning |
|---|---|
| `v0` | Published early so the consumer can build against a real schema instead of a guess |
| `frozen` | Field names, types and units will not change; consumer can build without rework |
| `live` | The producing module actually serves it |

After freezing, a change requires a version bump and a backward-compatible adapter. A breaking
change to a frozen contract needs sign-off from every consumer, because it is a change to somebody
else's module.

## Who owns what

| File | Owner | Consumers |
|---|---|---|
| `common.proto` | Team | all |
| `module1.proto` | Zayan (M1) | M2, M3, M4 |
| `module2.proto` | Duwaragie (M2) | M3 |
| `module3.proto` | Saabir (M3) | M4 |
| `verification.proto` | Hariswara (M4) | M2, M3, dashboard |
