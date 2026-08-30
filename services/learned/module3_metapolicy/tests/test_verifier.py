"""mock_verify must not depend on per-process hash randomization."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from verifier import mock_verify

_ACTION = {
    "origin": "SYSTEM1",
    "load_shed": [{"node_id": "N12", "shed_fraction": 0.1, "priority_tier": 3}],
}
_OBS = {"severity": "extreme", "max_node_vulnerability": 0.8}


def test_mock_verify_is_stable_within_process():
    assert mock_verify(_ACTION, _OBS) == mock_verify(_ACTION, _OBS)


def test_mock_verify_survives_hash_randomization():
    root = Path(__file__).resolve().parents[1]
    snippet = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "from verifier import mock_verify\n"
        f"print(json.dumps(mock_verify({_ACTION!r}, {_OBS!r})))\n"
    )
    results = []
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        results.append(json.loads(proc.stdout))
    assert results[0] == results[1]
