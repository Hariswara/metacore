"""Gateway configuration.

The gateway reads the Module 1 *calibration output* (`data/processed/`), never the state-entity
blobs in `data/external/`. That is the boundary ADR 0004 draws: the parameter set is the interface,
and nothing downstream of it needs the raw ledger.
"""

from __future__ import annotations

import os
from pathlib import Path

# services/gateway/app/config.py -> repo root
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]

REPO_ROOT = Path(os.environ.get("METACORE_ROOT", _DEFAULT_ROOT))
PROCESSED_DIR = Path(os.environ.get("METACORE_PROCESSED_DIR", REPO_ROOT / "data" / "processed"))
GENERATION_CSV = PROCESSED_DIR / "ceb_generation_tidy.csv"

# Vite dev server. The compose profile serves the built assets from the same origin, so this is
# only needed when the dashboard runs from `pnpm dev`.
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("METACORE_CORS_ORIGINS", "http://localhost:5173").split(",")
    if o.strip()
]
