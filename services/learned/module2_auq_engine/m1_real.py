"""Real Module 1 states, with a 3-class label discretised from M1's vulnerability proxy V.

This is the real-data counterpart to `synthetic_data.py`. It exists because the head
trained on synthetic draws reads a real M1 row as fully out of distribution: the
normalisation card's statistics for the five electrical features come from synthetic
spreads where `voltage_pu` ~ 1.0 +/- 0.021, while a real offline row carries 0.0 there.
That is z = -48 sigma on one column and ~22 sigma RMS across the five, against an evidence
regulariser trained to drive evidence to zero at 4 sigma -- so u = 1.0 was the correct
answer to a row that far outside the fitted distribution, not a bug. Refitting the card on
the distribution below is the root fix.

WHAT THE LABEL IS, AND IS NOT
-----------------------------
M1's `module1.train.target` builds a continuous per-node vulnerability

    V = w_H * E_type * H  +  w_F * F  +  w_S * S

from hazard (NASA POWER wind + rain), fragility (asset type, raised by the *previous*
month's ledger decay) and electrical stress (load over installed capacity). M1 is explicit
that this is "a constructed proxy, never an observation", carried as QUALITY_INTERPOLATED.

Discretising it into three classes therefore gives a *traceable* label, not an observed
safety class. No dataset anywhere records how vulnerable a microgrid asset actually was.
`PROXY_DISCLOSURE` below is carried into every metrics artifact this module writes, so the
numbers cannot be read off a results table without it.

LEAKAGE
-------
Four guards, two of them inherited from M1:

* the class cuts are quantiles of the TRAINING split only, applied to every row;
* M1 scales the hazard on training-split quantiles only;
* M1 lags the fragility decay by one month, because an operator only has last month's
  ledger;
* and one that is this module's own: M1's model inputs carry `vulnerability` as a history
  feature, which is the label itself. The 28-feature contract does not include it, so
  mapping through the contract drops it by construction -- see `_contract_rows`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from metacore_contracts.state_schema import FEATURE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[3]
M1_SRC = REPO_ROOT / "services" / "learned" / "module1_state_forecasting" / "src"
if str(M1_SRC) not in sys.path:
    sys.path.insert(0, str(M1_SRC))

# The five the offline path cannot produce (no power-flow simulator, no SCADA -- ADR 0004).
ELECTRICAL = ("p_kw_norm", "q_kvar_norm", "voltage_pu", "soc_fraction", "asset_online")

# Quantiles of training V that cut safe | stressed | critical.
CLASS_CUTS = (0.70, 0.95)
CLASS_NAMES = ("safe", "stressed", "critical")
DEFAULT_WEIGHTING = "equal"

PROXY_DISCLOSURE = (
    "The 3-class label is discretised from M1's constructed vulnerability proxy V "
    "(module1.train.target: hazard x exposure + fragility + stress), which M1 documents as "
    "a constructed proxy, never an observation, and carries as QUALITY_INTERPOLATED. No "
    "dataset records an observed microgrid safety class. Class cuts are quantiles of the "
    "TRAINING split only. Accuracy and ECE are therefore measured against a constructed "
    "target: they state internal consistency with M1's proxy, not the prediction of real "
    "asset failures."
)

OOD_DISCLOSURE = (
    "OOD detection is evaluated ONLY as the Eluvaitivu per-plant monthly case study "
    "(eluvaitivu_decay.py experiment B), n = 1 episode / 3 decay plant-months. No OOD "
    "detection rate is claimed on hourly island-aggregate data: island_load_hourly.csv "
    "sums both Eluvaitivu plants, attenuating the event 7.1x (the plant falls 73.4% while "
    "island demand falls 10.3%), and the same-season control AUROC there is 0.500."
)


@dataclass
class RealStates:
    """Real M1 node-hours on the 28-feature contract, with the discretised V label."""

    x: np.ndarray                 # (n, 28) float32 -- contract order, electrical zero-filled
    y: np.ndarray                 # (n,) int64 in 0..2
    v: np.ndarray                 # (n,) float -- the continuous proxy behind y
    split: np.ndarray             # (n,) str -- train | val | test
    node: np.ndarray              # (n,) str -- Island/asset
    timestamp: np.ndarray         # (n,) str
    quality_mask: list[str]       # 28 flags, identical for every offline row
    observed_fraction: float
    cuts: tuple[float, float]     # the V values the quantiles resolved to
    weighting: str

    def where(self, split: str) -> RealStates:
        m = self.split == split
        return RealStates(self.x[m], self.y[m], self.v[m], self.split[m], self.node[m],
                          self.timestamp[m], self.quality_mask, self.observed_fraction,
                          self.cuts, self.weighting)

    @property
    def class_balance(self) -> dict[str, float]:
        counts = np.bincount(self.y, minlength=len(CLASS_NAMES))
        return {n: float(c / counts.sum()) for n, c in zip(CLASS_NAMES, counts, strict=True)}


def _contract_rows(x_m1: np.ndarray) -> np.ndarray:
    """M1's (T, N, F) model inputs -> (T, N, 28) contract rows.

    M1's first 23 model features are exactly the contract's 23 non-electrical features in
    the same order -- asserted here, not assumed, so a reordering upstream fails loudly
    rather than silently scrambling columns. The remaining two M1 inputs are `fragility`
    and `vulnerability`; `vulnerability` IS the label, so stopping at column 23 is what
    keeps the label out of M2's inputs.
    """
    from module1.train.dataset import FEATURES

    non_electrical = tuple(n for n in FEATURE_NAMES if n not in ELECTRICAL)
    if FEATURES[:23] != non_electrical:
        raise RuntimeError(
            "M1's feature order no longer matches the contract's non-electrical block; "
            "the column mapping in m1_real._contract_rows must be revisited"
        )

    t, n, _ = x_m1.shape
    out = np.zeros((t, n, len(FEATURE_NAMES)), dtype=np.float32)
    for col, name in enumerate(non_electrical):
        out[:, :, FEATURE_NAMES.index(name)] = x_m1[:, :, col]
    return out


def quality_mask() -> list[str]:
    """The offline mask every real M1 row carries: 12 observed, 11 interpolated, 5 missing."""
    from metacore_contracts.state_schema import calibration_quality

    return [calibration_quality(name) for name in FEATURE_NAMES]


def load(weighting: str = DEFAULT_WEIGHTING,
         cuts: tuple[float, float] = CLASS_CUTS) -> RealStates:
    """Every real node-hour M1 can produce, as contract rows with a 3-class V label."""
    from module1.train import dataset as ds

    data = ds.build()
    v = np.asarray(data.v[weighting], dtype=float)          # (T, N)
    x = _contract_rows(np.asarray(data.x))                  # (T, N, 28)
    t, n = v.shape

    node_ids = np.asarray(getattr(data.graph, "node_ids", None)
                          if getattr(data.graph, "node_ids", None) is not None
                          else [f"node{i}" for i in range(n)])

    split = np.empty((t, n), dtype=object)
    for name, (a, b) in data.split_index.items():
        split[a:b, :] = name

    tr0, tr1 = data.split_index["train"]
    edges = tuple(float(e) for e in np.quantile(v[tr0:tr1], cuts))
    y = np.digitize(v, edges).astype(np.int64)

    mask = quality_mask()
    observed = sum(1 for q in mask if q == "QUALITY_OBSERVED") / len(mask)

    stamps = np.repeat(np.asarray(data.timestamps)[:, None], n, axis=1)
    nodes = np.repeat(node_ids[None, :], t, axis=0)

    return RealStates(
        x=x.reshape(-1, len(FEATURE_NAMES)),
        y=y.reshape(-1),
        v=v.reshape(-1),
        split=split.reshape(-1).astype(str),
        node=nodes.reshape(-1).astype(str),
        timestamp=stamps.reshape(-1).astype(str),
        quality_mask=mask,
        observed_fraction=float(observed),
        cuts=edges,
        weighting=weighting,
    )


def class_weights(y: np.ndarray, n_classes: int = len(CLASS_NAMES)) -> np.ndarray:
    """Inverse-frequency weights, normalised to mean 1.

    `critical` is ~5% of rows at the q0.70/q0.95 cuts, so an unweighted Bayes-risk loss
    buys most of its accuracy by never predicting it. Normalising to mean 1 keeps the loss
    on the same scale as the unweighted one, so the existing KL annealing schedule still
    applies unchanged.
    """
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return (w / w.mean()).astype(np.float32)
