"""Render the evaluation tables as figures. Requires the `viz` extra.

    uv sync --package metacore-module2 --extra viz     # or: pip install 'matplotlib>=3.7'
    python run_demo.py        # writes eval_tables.json
    python plots.py           # writes figures/*.png

Deliberately the only file in the module that imports matplotlib, and deliberately
NOT imported by any test: the metrics are computed and asserted in evaluate.py from
plain NumPy, and this only draws what they produced. CI therefore never installs a
plotting stack.

Four figures, one per claim the module makes:

  uncertainty_histogram  the populations occupy different parts of [0, 1]   (OOD)
  reliability            the probabilities are honest                       (calibration)
  risk_coverage          u ranks errors, so abstention works                (selective)
  u_prime_vs_f           the sensing axis moves u' independently            (two axes)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
TABLES = HERE / "eval_tables.json"
FIGURES = HERE / "figures"

# One colour per population, used identically in every figure that shows them, so the
# reader learns the mapping once.
COLOURS = {"normal": "#1f77b4", "cyclone": "#d62728", "blackout": "#ff7f0e"}


def uncertainty_histogram(hist, out_path):
    """Where each population sits on the u axis.

    The OOD claim in one picture: if the normal and cyclone masses overlap, no threshold
    separates them and AUROC cannot be high. Densities rather than counts because the
    groups are different sizes.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    lo, hi = hist["bin_lo"], hist["bin_hi"]
    centres = [(a + b)/2 for a, b in zip(lo, hi, strict=True)]
    width = (hi[0] - lo[0]) * 0.9

    for name, g in hist["groups"].items():
        ax.bar(centres, g["density"], width=width, alpha=0.55,
               color=COLOURS.get(name), label=f"{name} (n={g['n']}, mean {g['mean']:.3f})")
    for name, g in hist["groups"].items():
        ax.axvline(g["mean"], color=COLOURS.get(name), linestyle=":", linewidth=1)

    ax.set_xlabel("epistemic uncertainty u'")
    ax.set_ylabel("share of population")
    ax.set_title("Where each population sits on the uncertainty axis")
    ax.set_xlim(0, 1.02)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def quality_curve(sweep, out_path, floor=None, nominal=None):
    """u' against the sensing fraction f, on fixed evidence.

    The feature values never change along this axis -- only how much of them is declared
    observed -- so the curves isolate the sensing axis from the value axis. The cyclone
    curve is flat at 1: with no evidence to discount, f has nothing to act on. That is
    why a single threshold on u' cannot stand in for the two-condition trigger, and why
    the floor is drawn here rather than argued about.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    fractions = [r["observed_fraction"] for r in sweep]
    names = [k for k in sweep[0] if k != "observed_fraction"]

    for name in names:
        ax.plot(fractions, [r[name] for r in sweep], "o-", color=COLOURS.get(name),
                label=name, markersize=4)
    if floor is not None:
        ax.axvline(floor, color="0.4", linestyle="--", linewidth=1,
                   label=f"sensing floor {floor:.2f}")
    if nominal is not None:
        ax.axvline(nominal, color="0.6", linestyle=":", linewidth=1,
                   label=f"M1 nominal f {nominal:.3f}")

    ax.set_xlabel("observed fraction f (share of features measured)")
    ax.set_ylabel("quality-aware uncertainty u'")
    ax.set_title("The sensing axis, on fixed evidence")
    # normal and blackout coincide: a blackout does not corrupt the values, only the
    # mask, so the two populations carry the same evidence and differ only in where
    # they sit on this axis. That is what keeps the two axes separable.
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    # Centre-right: the cyclone curve runs flat along the top and the others along the
    # bottom, so the middle of the panel is the only space a legend does not cover data.
    ax.legend(loc="center right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def reliability_diagram(rows, out_path):
    """Confidence vs accuracy per bin, against the y=x line a calibrated model follows."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="0.6", linewidth=1, label="perfectly calibrated")
    ax.plot([r["confidence"] for r in rows], [r["accuracy"] for r in rows],
            "o-", color="#1f77b4", label="observed")
    for r in rows:
        ax.annotate(str(r["count"]), (r["confidence"], r["accuracy"]),
                    textcoords="offset points", xytext=(4, -9), fontsize=7, color="0.4")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability (bin counts annotated)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def risk_coverage_curve(coverages, risks, aurc, out_path, base_risk=None):
    """Risk against coverage. A curve that stays flat and low until high coverage is
    the claim that u ranks errors -- the area under it is AURC."""
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(coverages, risks, "o-", color="#d62728", label=f"selective (AURC {aurc:.4f})")
    if base_risk is not None:
        ax.axhline(base_risk, linestyle="--", color="0.6", linewidth=1,
                   label=f"full-coverage risk {base_risk:.3f}")
    ax.set_xlabel("coverage (fraction of states kept, most confident first)")
    ax.set_ylabel("risk (error rate of the kept set)")
    ax.set_title("Risk-coverage")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    if not TABLES.exists():
        print(f"no {TABLES.name}; run `python run_demo.py` first", file=sys.stderr)
        return 2
    tables = json.loads(TABLES.read_text())
    FIGURES.mkdir(exist_ok=True)
    written = []

    reliability_diagram(tables["reliability"], FIGURES / "reliability.png")
    written.append("reliability.png")

    rc = tables["risk_coverage"]
    risk_coverage_curve(rc["coverage"], rc["risk"], rc["aurc"],
                        FIGURES / "risk_coverage.png", base_risk=rc["risk"][-1])
    written.append("risk_coverage.png")

    # Both are additive to eval_tables.json, so an older file still renders what it has.
    if "uncertainty_histogram" in tables:
        uncertainty_histogram(tables["uncertainty_histogram"],
                              FIGURES / "uncertainty_histogram.png")
        written.append("uncertainty_histogram.png")
    if "quality_sweep" in tables:
        trig = tables.get("trigger", {})
        quality_curve(tables["quality_sweep"], FIGURES / "u_prime_vs_f.png",
                      floor=trig.get("observed_fraction_floor"),
                      nominal=trig.get("nominal_observed_fraction"))
        written.append("u_prime_vs_f.png")

    for name in written:
        print(f"wrote {FIGURES / name}")
    stale = [p for p in (HERE / "reliability.png", HERE / "risk_coverage.png") if p.exists()]
    if stale:
        print(f"note: {', '.join(p.name for p in stale)} at the module root are from "
              "before figures/ existed and can be deleted", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
