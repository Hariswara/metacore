#!/usr/bin/env bash
# Asserts the learned/deterministic boundary: deterministic-core must declare no ML dependency
# and must be the only component that imports OpenDSS.
#
# See docs/adr/0003-deterministic-core-isolation.md. This runs in CI and as a pre-commit hook.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FORBIDDEN='torch|tensorflow|jax|scikit-learn|sklearn|transformers|onnxruntime'
status=0

echo "==> deterministic-core dependency declarations"
while IFS= read -r f; do
  if grep -Eiq "\"($FORBIDDEN)" "$f"; then
    echo "FAIL: ML dependency declared in $f"
    grep -Ein "\"($FORBIDDEN)" "$f"
    status=1
  fi
done < <(find "$ROOT/deterministic-core" -name pyproject.toml)

echo "==> deterministic-core imports"
if grep -REn "^\s*(import|from)\s+($FORBIDDEN)\b" "$ROOT/deterministic-core" --include='*.py' ; then
  echo "FAIL: ML import found inside deterministic-core"
  status=1
fi

echo "==> OpenDSS is imported only by deterministic-core"
if grep -REn "^\s*(import|from)\s+\S*(opendssdirect|dss_python|py_dss_interface)" \
     "$ROOT/services" "$ROOT/apps" "$ROOT/packages" --include='*.py' 2>/dev/null; then
  echo "FAIL: OpenDSS imported outside deterministic-core"
  status=1
fi

[ $status -eq 0 ] && echo "OK: learned/deterministic boundary intact"
exit $status
