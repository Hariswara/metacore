"""Baselines and the OOD-regularisation ablation.

Two claims, and neither is about a specific number:

  1. The standard magnitude baselines fail on far-OOD tabular data -- not "do worse",
     but invert: they are *more* confident on cyclone states than on normal ones.
  2. The OOD-aware evidence regulariser is what fixes it. Plain EDL fails the same way,
     so the contribution is the regulariser and not the Dirichlet head.

Deliberately self-contained and small (800 samples, 40 epochs, four models, ~6s). It does
NOT use the session-scoped fixture, because the point is to train four things the same way
and compare them. Full-scale numbers come from `python benchmark.py`, which is a script.

Assertions are orderings and inequalities only. The numbers move with the seed and the
draw; the ranking is the claim.
"""

import numpy as np
import pytest
import torch
from baselines import (
    DROPOUT_P,
    score_edl,
    score_mc_dropout,
    score_softmax,
    time_scoring,
    train_edl,
    train_softmax,
)
from evaluate import aupr, auroc
from state_contract import stack_features
from synthetic_data import Normalizer, sample_states_id, sample_states_ood

N_TRAIN, N_ID, N_OOD, EPOCHS = 800, 400, 300, 40
INVERTED = 0.5          # AUROC below this means the score is anti-correlated with OOD-ness


@pytest.fixture(scope="module")
def bench():
    """Four models on identical data, plus a scorer for each. ~6s."""
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    tr, ytr = sample_states_id(N_TRAIN, rng, blackout_rate=0.0)
    te, yte = sample_states_id(N_ID, rng, blackout_rate=0.0)
    ood, _ = sample_states_ood(N_OOD, rng)
    nz = Normalizer().fit(stack_features(tr))
    xtr, xte, xood = (nz(stack_features(s)) for s in (tr, te, ood))

    softmax_m = train_softmax(xtr, ytr, EPOCHS, seed=0)
    mc_m = train_softmax(xtr, ytr, EPOCHS, dropout=DROPOUT_P, seed=0)
    ours_m = train_edl(xtr, ytr, EPOCHS, ood_reg=0.1, seed=0)
    noreg_m = train_edl(xtr, ytr, EPOCHS, ood_reg=0.0, seed=0)

    return {
        "xte": xte, "xood": xood, "yte": yte,
        "softmax": lambda x: score_softmax(softmax_m, x),
        "mc_dropout": lambda x: score_mc_dropout(mc_m, x),
        "ours": lambda x: score_edl(ours_m, x),
        "noreg": lambda x: score_edl(noreg_m, x),
    }


def _detection(bench, name):
    s_ood, _ = bench[name](bench["xood"])
    s_id, _ = bench[name](bench["xte"])
    return auroc(s_ood, s_id), aupr(s_ood, s_id)


def _accuracy(bench, name):
    _, p = bench[name](bench["xte"])
    return (p.argmax(1) == bench["yte"]).mean()


def test_every_method_actually_learns_the_task(bench):
    """Guard the whole comparison: a baseline that fails at OOD detection must not be
    failing because it is simply a broken classifier."""
    for name in ("softmax", "mc_dropout", "ours", "noreg"):
        assert _accuracy(bench, name) > 0.80, (name, _accuracy(bench, name))


def test_ours_beats_the_softmax_baseline(bench):
    ours, _ = _detection(bench, "ours")
    softmax, _ = _detection(bench, "softmax")

    assert ours > softmax, (ours, softmax)


def test_ours_beats_mc_dropout(bench):
    ours, _ = _detection(bench, "ours")
    mc, _ = _detection(bench, "mc_dropout")

    assert ours > mc, (ours, mc)


def test_ood_regularisation_is_load_bearing(bench):
    """The ablation. Same head, same data, same everything but ood_reg."""
    ours, _ = _detection(bench, "ours")
    noreg, _ = _detection(bench, "noreg")

    assert ours - noreg >= 0.30, (ours, noreg)


def test_baselines_invert_rather_than_merely_underperform(bench):
    """The confident-extrapolation result, and the reason a stronger baseline would not
    rescue these: a ReLU network is *more* certain far from its training data, so the
    score is anti-correlated with OOD-ness rather than uninformative."""
    for name in ("softmax", "noreg"):
        detection, _ = _detection(bench, name)
        assert detection < INVERTED, (name, detection)


def test_aupr_agrees_with_auroc(bench):
    """A second metric, so one number is not carrying the story. AUPR below the positive
    base rate is the same statement as AUROC below 0.5."""
    base_rate = N_OOD / (N_OOD + N_ID)
    _, ours_aupr = _detection(bench, "ours")
    assert ours_aupr > base_rate

    for name in ("softmax", "noreg"):
        _, detection_aupr = _detection(bench, name)
        assert detection_aupr < base_rate, (name, detection_aupr, base_rate)


def test_mc_dropout_costs_more_per_sample(bench):
    """T stochastic passes against one. The single-pass advantage is the reason EDL is
    viable on the real-time path at all."""
    mc = time_scoring(bench["mc_dropout"], bench["xte"], repeats=3)
    ours = time_scoring(bench["ours"], bench["xte"], repeats=3)

    assert mc > 2*ours, (mc, ours)
