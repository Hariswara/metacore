"""Render the evaluation tables as figures. Requires the `viz` extra.

    uv sync --package metacore-module2 --extra viz     # or: pip install 'matplotlib>=3.7'
    python run_demo.py        # writes eval_tables.json
    python plots.py           # writes reliability.png, risk_coverage.png

Deliberately the only file in the module that imports matplotlib, and deliberately
NOT imported by any test: the metrics are computed and asserted in evaluate.py from
plain NumPy, and this only draws what they produced. CI therefore never installs a
plotting stack.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

TABLES = Path(__file__).with_name("eval_tables.json")


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
    here = TABLES.parent

    reliability_diagram(tables["reliability"], here / "reliability.png")
    rc = tables["risk_coverage"]
    risk_coverage_curve(rc["coverage"], rc["risk"], rc["aurc"],
                        here / "risk_coverage.png", base_risk=rc["risk"][-1])
    print(f"wrote {here / 'reliability.png'}")
    print(f"wrote {here / 'risk_coverage.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
