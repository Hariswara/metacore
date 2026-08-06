# contracts

One source of truth for every message that crosses a module boundary. Python, Go and TypeScript
stubs are generated from `proto/` — nothing is hand-written per language.

```bash
task proto     # buf lint && buf generate
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
