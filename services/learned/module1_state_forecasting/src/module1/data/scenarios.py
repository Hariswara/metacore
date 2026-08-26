"""Stage: the shared ID/OOD scenario library, derived from the CEB ledger.

`common.proto` puts the scenario library in M1's hands: every `ScenarioRef` a consumer replays --
`scenario_id`, `library_version`, `out_of_distribution` -- is issued here. This stage is what
issues them.

It exists because of a modelling decision made elsewhere and correctly. `island_load_hourly.csv`
sums the two Eluvaitivu plants into one island demand, which is right for dispatch and power flow:
the island has one load, served by a diesel set and a hybrid plant together. But the failure this
whole project is written about is a property of *one of those two plants*, and summing attenuates
it by a factor of seven. Measured across 2025 Q4 against the preceding quarter:

    hybrid plant output   -73.4%          (December alone: -97.0%)
    island demand         -10.3%          (December alone: -12.5%)

The diesel set absorbed the difference, month for month. A detector watching island demand sees a
10% seasonal-looking dip; the plant behind it had all but stopped. That gap is the entire case for
per-asset state representation over aggregate telemetry, and in the hourly artifact it is gone.

So the label cannot be read off the hourly data, and M2's out-of-distribution evaluation has
nothing to select on. This stage restores it, from the monthly per-plant ledger where the event
is still visible and still measured.

  MEASURED
    Monthly energy per plant, from the reconciled CEB ledger. Unlike almost everything else M1
    produces, these rows are QUALITY_OBSERVED -- a meter reading, not a downscaled estimate.

  DERIVED, by a stated rule
    Which months are anomalous. The rule is below and is applied uniformly to all five plants;
    the window is not hand-placed. Running it over the full record flags exactly one plant.

The detection rule, and why each part of it is there:

  1. Baseline. The median of the plant's own monthly energy across the record. Median rather than
     mean because the event itself is in the record -- a mean would be dragged down by the very
     months being tested, shrinking the anomaly it is supposed to measure.
  2. Core. Months below `CORE_RATIO` (0.50) of baseline. On this dataset that is 2025-11 and
     2025-12 for the hybrid plant, and nothing anywhere else.
  3. Onset. Walk backwards from the core while months stay below `ONSET_RATIO` (0.90). A decay
     starts before it becomes obvious, and an evaluation window that opens at the collapse hands
     a detector the easiest part of the problem and hides the part worth measuring. This adds
     2025-10, at 0.70.

Detecting the onset *without* the label is M2's problem, not this stage's. What is published here
is the window, the rule that produced it, and the monthly ratios -- so a disagreement about where
the event starts is a disagreement about a number in a file, not about a recollection.

Usage:
    python -m module1.data.scenarios build <tidy.csv> <out_dir>
    python -m module1.data.scenarios validate <out_dir> <tidy.csv>
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LIBRARY_VERSION = "1.0.0"

QUALITY_OBSERVED = "QUALITY_OBSERVED"

MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# Plant -> island. Two plants serve Eluvaitivu; the island is what M4 models and M3 dispatches,
# the plant is what degrades. Kept identical to the mapping in load.py on purpose.
SYSTEM_TO_ISLAND = {
    "Analaithivu": "Analaitivu",
    "Eluvaitivu-Diesel": "Eluvaitivu",
    "Eluvaitivu-Hybrid": "Eluvaitivu",
    "Delft-Neduntivu": "Delft-Neduntivu",
    "Nainativu": "Nainativu",
}

# See the module docstring. These are the only two free numbers in the stage.
CORE_RATIO = 0.50
ONSET_RATIO = 0.90

# An in-distribution window: every plant on the island within this band of its own baseline, for
# every month of the window. Deliberately tight -- a nominal episode that quietly contains a
# half-anomaly would inflate a detector's apparent skill.
NOMINAL_BAND = (0.85, 1.15)
NOMINAL_WINDOW_MONTHS = 3

EVENT_FIELDS = (
    "scenario_id",
    "island",
    "plant",
    "event_type",
    "start_month",
    "end_month",
    "months",
    "out_of_distribution",
    "baseline_kwh",
    "worst_ratio",
    "detection_rule",
    "note",
)


# ------------------------------------------------------------------ input

def read_plant_months(tidy_csv: str | Path) -> dict[str, dict[tuple[int, int], float]]:
    """Monthly energy per plant, keyed (year, month). The measured half of this stage."""
    series: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
    with Path(tidy_csv).open(newline="") as fh:
        for row in csv.DictReader(fh):
            plant = row["island_system"]
            if plant not in SYSTEM_TO_ISLAND:
                raise SystemExit(f"unknown plant in ledger: {plant!r}")
            key = (int(row["year"]), MONTHS[row["month"]])
            series[plant][key] = float(row["units_kwh"])
    return dict(series)


def _label(key: tuple[int, int]) -> str:
    return f"{key[0]}-{key[1]:02d}"


def _quarter(key: tuple[int, int]) -> str:
    return f"{key[0]}q{(key[1] - 1) // 3 + 1}"


# ------------------------------------------------------------- detection

def ratios(months: dict[tuple[int, int], float]) -> tuple[float, dict[tuple[int, int], float]]:
    """Each month as a fraction of the plant's own median. Returns (baseline, ratios)."""
    baseline = statistics.median(months.values())
    if baseline <= 0:
        raise SystemExit("a plant with a non-positive median cannot be baselined")
    return baseline, {k: v / baseline for k, v in months.items()}


def find_degradation(
    plant: str, months: dict[tuple[int, int], float]
) -> dict | None:
    """Apply the rule to one plant. Returns None when nothing is flagged."""
    keys = sorted(months)
    baseline, ratio = ratios(months)

    core = [k for k in keys if ratio[k] < CORE_RATIO]
    if not core:
        return None

    # Contiguity is not assumed: take the span from first to last flagged month, so a plant that
    # recovered and relapsed produces one window rather than two silently merged ones.
    first, last = keys.index(min(core)), keys.index(max(core))

    # Walk back to the onset.
    while first > 0 and ratio[keys[first - 1]] < ONSET_RATIO:
        first -= 1

    window = keys[first : last + 1]
    island = SYSTEM_TO_ISLAND[plant]
    unit = plant.split("-")[-1].lower()
    return {
        "scenario_id": f"{island.lower()}-{unit}-decay-{_quarter(window[0])}",
        "island": island,
        "plant": plant,
        "event_type": "asset_degradation",
        "start_month": _label(window[0]),
        "end_month": _label(window[-1]),
        "months": len(window),
        "out_of_distribution": True,
        "baseline_kwh": round(baseline, 1),
        "worst_ratio": round(min(ratio[k] for k in window), 4),
        "detection_rule": f"median baseline; core<{CORE_RATIO}; onset walk-back<{ONSET_RATIO}",
        "monthly": [
            {"month": _label(k), "kwh": months[k], "ratio": round(ratio[k], 4)} for k in window
        ],
        "note": (
            "Visible only in the per-plant monthly ledger. island_load_hourly.csv sums both "
            "Eluvaitivu plants: across this window the plant falls 73.4% while island demand "
            "falls 10.3%, a 7.1x attenuation, because the diesel set absorbs the difference."
        ),
    }


def find_nominal(
    island: str, plants: dict[str, dict[tuple[int, int], float]], excluded: set[tuple[int, int]]
) -> list[dict]:
    """Contiguous windows where every plant on the island sits inside NOMINAL_BAND.

    M2 needs in-distribution episodes to score against, and taking "everything not flagged" would
    quietly include the months either side of an event.
    """
    per_plant = {p: ratios(m)[1] for p, m in plants.items()}
    keys = sorted(next(iter(plants.values())))
    low, high = NOMINAL_BAND

    found: list[dict] = []
    i = 0
    while i + NOMINAL_WINDOW_MONTHS <= len(keys):
        window = keys[i : i + NOMINAL_WINDOW_MONTHS]
        ok = all(
            k not in excluded and all(low <= per_plant[p][k] <= high for p in per_plant)
            for k in window
        )
        if ok:
            found.append({
                "scenario_id": f"{island.lower()}-nominal-{_quarter(window[0])}",
                "island": island,
                "plant": "",
                "event_type": "nominal",
                "start_month": _label(window[0]),
                "end_month": _label(window[-1]),
                "months": len(window),
                "out_of_distribution": False,
                "baseline_kwh": round(
                    statistics.fmean(sum(plants[p][k] for p in plants) for k in window), 1
                ),
                "worst_ratio": round(
                    min(per_plant[p][k] for p in per_plant for k in window), 4
                ),
                "detection_rule": f"every plant within {NOMINAL_BAND} of its own median",
                "monthly": [
                    {"month": _label(k), "kwh": sum(plants[p][k] for p in plants), "ratio": None}
                    for k in window
                ],
                "note": "In-distribution control window.",
            })
            i += NOMINAL_WINDOW_MONTHS  # non-overlapping
        else:
            i += 1
    return found


# ------------------------------------------------------------------ stage

def build(tidy_csv: str | Path) -> tuple[list[dict], dict]:
    series = read_plant_months(tidy_csv)

    events: list[dict] = []
    for plant in sorted(series):
        found = find_degradation(plant, series[plant])
        if found:
            events.append(found)

    by_island: dict[str, dict[str, dict]] = defaultdict(dict)
    for plant, months in series.items():
        by_island[SYSTEM_TO_ISLAND[plant]][plant] = months

    excluded = {
        (int(m["month"][:4]), int(m["month"][5:]))
        for e in events
        for m in e["monthly"]
    }
    for island in sorted(by_island):
        events.extend(find_nominal(island, by_island[island], excluded))

    library = {
        "artifact": "scenario_library",
        "library_version": LIBRARY_VERSION,
        "produced_by": "module1.data.scenarios",
        "quality": QUALITY_OBSERVED,
        "source": (
            "data/processed/ceb_generation_tidy.csv (reconciled CEB ledger, monthly, per plant)"
        ),
        "rule": {
            "baseline": "median of the plant's own monthly energy across the record",
            "core_ratio": CORE_RATIO,
            "onset_ratio": ONSET_RATIO,
            "nominal_band": list(NOMINAL_BAND),
            "nominal_window_months": NOMINAL_WINDOW_MONTHS,
        },
        "counts": {
            "out_of_distribution": sum(1 for e in events if e["out_of_distribution"]),
            "in_distribution": sum(1 for e in events if not e["out_of_distribution"]),
        },
        "note": (
            "Consumed as metacore.common.v1.ScenarioRef: scenario_id, library_version and "
            "out_of_distribution map field-for-field. The hourly load artifact carries no label "
            "column by design -- it is one series per island, and the degradation is a per-plant "
            "property that summing removes."
        ),
        "scenarios": events,
    }
    return events, library


def write_events_csv(events: list[dict], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=EVENT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for event in events:
            ood = str(event["out_of_distribution"]).lower()  # csv has no bool; keep it JSON-shaped
            writer.writerow({**event, "out_of_distribution": ood})


# ------------------------------------------------------------------- gate

def check(out_dir: str | Path, tidy_csv: str | Path) -> list[str]:
    """Fail the pipeline when the library stops agreeing with the ledger it claims to describe."""
    failures: list[str] = []
    out = Path(out_dir)
    events_csv, library_json = out / "events.csv", out / "scenario_library.json"

    for path in (events_csv, library_json):
        if not path.exists():
            failures.append(f"{path.name}: absent")
    if failures:
        return failures

    library = json.loads(library_json.read_text())
    with events_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    if library.get("library_version") != LIBRARY_VERSION:
        failures.append(
            f"scenario_library.json: library_version {library.get('library_version')!r}, "
            f"expected {LIBRARY_VERSION!r}"
        )

    # 1. The two files describe the same set. A consumer may reasonably read either.
    if len(rows) != len(library.get("scenarios", [])):
        failures.append(
            f"events.csv has {len(rows)} rows, scenario_library.json has "
            f"{len(library.get('scenarios', []))} scenarios"
        )

    # 2. Scenario ids are unique -- a ScenarioRef that resolves to two episodes is unreplayable.
    ids = [r["scenario_id"] for r in rows]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        failures.append(f"duplicate scenario_id(s): {', '.join(sorted(duplicates))}")

    # 3. The library still reproduces from the ledger. This is the load-bearing check: it is what
    #    stops the labels drifting into hand-maintained constants once the numbers look settled.
    rebuilt, _ = build(tidy_csv)
    if {e["scenario_id"] for e in rebuilt} != set(ids):
        failures.append(
            "the committed library does not reproduce from the ledger -- "
            f"rebuilt {sorted(e['scenario_id'] for e in rebuilt)}, committed {sorted(ids)}"
        )

    # 4. The event this project is about is present and labelled OOD. Named explicitly: M2's
    #    uncertainty evaluation selects on it, and losing it to a refactor should break a build
    #    here rather than quietly halve someone else's result set.
    ood = [r for r in rows if r["out_of_distribution"] == "true"]
    if not any(r["plant"] == "Eluvaitivu-Hybrid" for r in ood):
        failures.append(
            "no out-of-distribution scenario for Eluvaitivu-Hybrid -- the 2025 Q4 hybrid "
            "degradation is the anomaly M2's evaluation is built on"
        )

    # 5. Both classes are non-empty. An OOD-only library cannot score a detector.
    if not ood:
        failures.append("no out-of-distribution scenarios")
    if len(ood) == len(rows):
        failures.append("no in-distribution scenarios to score against")

    return failures


# -------------------------------------------------------------------- cli

def main(argv: list[str]) -> int:
    if len(argv) != 4 or argv[1] not in ("build", "validate"):
        print(__doc__, file=sys.stderr)
        return 2

    if argv[1] == "build":
        events, library = build(argv[2])
        out = Path(argv[3])
        out.mkdir(parents=True, exist_ok=True)
        write_events_csv(events, out / "events.csv")
        (out / "scenario_library.json").write_text(json.dumps(library, indent=2) + "\n")
        counts = library["counts"]
        print(f"{out}/events.csv: {len(events)} scenarios "
              f"({counts['out_of_distribution']} OOD, {counts['in_distribution']} ID)")
        for event in events:
            if event["out_of_distribution"]:
                print(f"  OOD {event['scenario_id']}: {event['plant']} "
                      f"{event['start_month']}..{event['end_month']}, "
                      f"worst {event['worst_ratio']:.0%} of baseline")
        return 0

    failures = check(argv[2], argv[3])
    if failures:
        print(f"SCENARIO GATE FAILED — {len(failures)} problem(s):", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("scenario gate OK — library reproduces from the ledger, ids unique, "
          "both distribution classes present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
