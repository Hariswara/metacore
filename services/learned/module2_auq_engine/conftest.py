"""Put this module's root on sys.path for the shared test lane.

The Module 2 prototype lives as flat modules at the service root (edl.py,
synthetic_data.py, ...) and imports them flat, which works when run_demo.py is
run from this directory. The repo-wide lane runs `pytest services` from the
root, where the root pyproject's `pythonpath` only adds each service's `src/`.
This bridges the two without moving files or adding this directory to the
shared `pythonpath`, where generic names like `contract` and `evaluate` would
collide with the other services.
"""

import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))
