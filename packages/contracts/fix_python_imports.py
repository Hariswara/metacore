#!/usr/bin/env python3
"""Rewrite the flat imports protoc emits into package-relative ones.

`protoc --python_out` derives an import statement from the *proto* import path, not from the
Python package the file is written into. Our protos sit flat in `proto/`, so `module1.proto`
importing `common.proto` produces:

    import common_pb2 as common__pb2

inside `metacore_contracts/module1_pb2.py`. That is only importable if the package directory is
itself on sys.path, which it is not when the package is installed normally -- so
`import metacore_contracts.module1_pb2` raises ModuleNotFoundError: No module named 'common_pb2'.
This is protocolbuffers/protobuf#1491, open since 2016 and not going to be fixed upstream.

The real fix is to nest the protos as `proto/metacore/<module>/v1/`, which also clears the
DIRECTORY_SAME_PACKAGE and PACKAGE_DIRECTORY_MATCH lint failures. That relocates the generated Go
and TypeScript trees as well, so it is a contracts-wide change every consumer has to review --
tracked separately. Until then this runs as the last step of `task proto` and is idempotent, so
regenerating never reintroduces the broken form.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# `import <name>_pb2 as <alias>` at column zero. Anchored to _pb2 so `import grpc` and the
# `from google.protobuf import ...` lines are left alone.
FLAT_IMPORT = re.compile(r"^import (\w+_pb2) as (\w+)$", re.MULTILINE)

PACKAGE = Path(__file__).parent / "python" / "metacore_contracts"


def fix(path: Path) -> bool:
    """Rewrite one generated file in place. Returns True if it changed."""
    original = path.read_text()
    patched = FLAT_IMPORT.sub(r"from . import \1 as \2", original)
    if patched == original:
        return False
    path.write_text(patched)
    return True


def main() -> int:
    if not PACKAGE.is_dir():
        print(f"error: {PACKAGE} does not exist -- run `buf generate` first", file=sys.stderr)
        return 1

    changed = [p.name for p in sorted(PACKAGE.glob("*_pb2*.py")) if fix(p)]
    if changed:
        print(f"rewrote flat imports in {len(changed)} file(s): {', '.join(changed)}")
    else:
        print("no flat imports found -- already package-relative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
