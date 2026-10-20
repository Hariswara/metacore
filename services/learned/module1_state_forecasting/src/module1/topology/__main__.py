"""Generate the topology artifacts, or print one snapshot.

    python -m module1.topology build            # writes the .dss model and the JSON pin
    python -m module1.topology build --check    # fails if the committed files are stale
    python -m module1.topology snapshot --timestamp 2025-11-28T12:00
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from module1.topology.snapshot import build_snapshot
from module1.topology.spec import build_spec, render_dss

# topology -> module1 -> src -> service -> learned -> services -> repo root
REPO_ROOT = Path(__file__).resolve().parents[6]
CONTRACTS = REPO_ROOT / "packages" / "contracts" / "python" / "metacore_contracts"
PIN = CONTRACTS / "schema" / "module1_topology_v1.json"
DSS = CONTRACTS / "topology" / "northern_islands_v1.dss"
SAMPLES = CONTRACTS / "schema" / "samples"
# Fixture snapshots M4's tests run against: a quiet hour, the record's Nainativu peak, and the
# Ditwah evening with the hybrid plant down. emitted_at is fixed so `build --check` is stable.
SAMPLE_HOURS = {
    "calm": "2025-10-05T12:00",
    "nainativu_peak": "2025-07-20T20:00",
    "ditwah_evening": "2025-11-28T19:00",
}


def artifacts() -> dict[Path, str]:
    spec = build_spec(REPO_ROOT)
    out = {PIN: json.dumps(spec, indent=1) + "\n", DSS: render_dss(spec)}
    for name, ts in SAMPLE_HOURS.items():
        snap = build_snapshot(ts, REPO_ROOT, emitted_at="replay")
        out[SAMPLES / f"module1_snapshot_v1_{name}.json"] = json.dumps(snap, indent=1) + "\n"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--check", action="store_true")
    s = sub.add_parser("snapshot")
    s.add_argument("--timestamp", required=True)
    s.add_argument("--data-root", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "build":
        stale = []
        for path, text in artifacts().items():
            if args.check:
                if not path.exists() or path.read_text() != text:
                    stale.append(str(path.relative_to(REPO_ROOT)))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                print(f"wrote {path.relative_to(REPO_ROOT)}")
        if stale:
            print("stale, run `python -m module1.topology build`: " + ", ".join(stale),
                  file=sys.stderr)
            return 1
        return 0
    try:
        snap = build_snapshot(args.timestamp, REPO_ROOT, args.data_root)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    json.dump(snap, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
