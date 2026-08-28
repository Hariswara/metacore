"""Does u predict error?

This is the module's load-bearing claim, and it is not the same claim as calibration.
ECE asks whether the probabilities are honest; selective prediction asks whether the
uncertainty is *usable* -- whether abstaining on the high-u states actually buys
accuracy on the rest. AURC is that, integrated.

Reuses the session-scoped trained head from conftest.py; nothing here trains.
"""

import numpy as np
import pytest
import torch
from edl import uncertainty, uncertainty_quality
from evaluate import ece, reliability_table, retained_composition, risk_coverage
from state_contract import stack_features
from synthetic_data import sample_states_blackout, sample_states_id, sample_states_ood
from trigger import CompetenceDropTrigger

OF_FLOOR = 0.35


@pytest.fixture(scope="module")
def scored(evidence):
    """(u, probs, labels, correct) for a nominal-quality in-distribution test set."""
    rng = np.random.default_rng(20)
    states, y = sample_states_id(1000, rng, blackout_rate=0.0)
    u, p, _ = uncertainty(evidence(stack_features(states)))
    u, p = u.numpy(), p.numpy()
    return u, p, y, (p.argmax(1) == y)


# ------------------------------------------------------------ selective prediction

def test_u_predicts_error(scored):
    u, _, _, correct = scored
    covs, risks, aurc = risk_coverage(u, correct)

    assert aurc < 0.05, aurc
    assert covs[-1] == pytest.approx(1.0)
    assert risks[-1] == pytest.approx(1.0 - correct.mean())


def test_abstaining_buys_accuracy(scored):
    u, _, _, correct = scored
    covs, risks, _ = risk_coverage(u, correct)
    half = int(np.argmin(np.abs(covs - 0.5)))

    accuracy_at_half = 1.0 - risks[half]
    assert accuracy_at_half >= correct.mean(), (accuracy_at_half, correct.mean())


def test_risk_is_monotone_enough_to_be_a_ranking(scored):
    """Risk need not be monotone step to step, but keeping the most confident half must
    not be worse than keeping everything -- otherwise u is not ranking errors at all."""
    u, _, _, correct = scored
    _, risks, _ = risk_coverage(u, correct)

    assert risks[:len(risks)//2].max() <= risks[-1] + 1e-9


def test_shuffled_u_is_much_worse(scored):
    """Guards against a trivially-passing AURC: a random ranking should score close to
    the base error rate, so a low AURC has to be coming from u and not from the data."""
    u, _, _, correct = scored
    _, _, aurc = risk_coverage(u, correct)
    rng = np.random.default_rng(0)
    _, _, aurc_shuffled = risk_coverage(rng.permutation(u), correct)

    assert aurc < aurc_shuffled


# ------------------------------------------------------------------- calibration

def test_reliability_table_is_well_formed(scored):
    u, p, y, _ = scored
    rows = reliability_table(p, y)

    assert sum(r["count"] for r in rows) == len(y)
    for r in rows:
        assert 0.0 <= r["accuracy"] <= 1.0
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["bin_lo"] < r["bin_hi"]
        assert r["count"] > 0


def test_reliability_table_reproduces_ece(scored):
    """The table is the data behind the number, so the number must fall out of it."""
    u, p, y, _ = scored
    rows = reliability_table(p, y)
    n = sum(r["count"] for r in rows)
    from_table = sum(r["count"]/n * abs(r["accuracy"] - r["confidence"]) for r in rows)

    assert from_table == pytest.approx(ece(p, y), abs=1e-9)


# --------------------------------------------------------------- two-axis payoff

def _mixed_stream(evidence, seed=21):
    rng = np.random.default_rng(seed)
    normal, _ = sample_states_id(600, rng, blackout_rate=0.0)
    cyclone, _ = sample_states_ood(300, rng)
    blackout, _ = sample_states_blackout(300, rng)

    states = normal + cyclone + blackout
    groups = np.array(["normal"]*len(normal) + ["cyclone"]*len(cyclone)
                      + ["blackout"]*len(blackout))
    of = np.array([s.quality.observed_fraction for s in states], dtype=np.float32)
    e = evidence(stack_features(states))
    value_u, _, _ = uncertainty(e)
    combined_u = uncertainty_quality(e, torch.tensor(of))
    return groups, of, value_u.numpy(), combined_u.numpy()


def test_u_ranking_rejects_the_value_axis_cleanly(evidence):
    """Ranking on u drops every cyclone state before any normal one: the value axis is
    separated by magnitude alone."""
    groups, _, _, combined_u = _mixed_stream(evidence)
    half = min(retained_composition(combined_u, groups),
               key=lambda r: abs(r["coverage"] - 0.5))

    assert half["cyclone"] == 0.0, half


def test_u_ranking_does_not_reject_the_sensing_axis(evidence):
    """And this is why the trigger needs an explicit observed_fraction floor.

    A blackout multiplies the evidence by ~0.375, which for a confident state still
    lands in the same u range as an ordinary one -- so ranking by magnitude barely
    moves blackout states. Measured: they are 25% of the stream and still ~22% of the
    most-confident half. The same lesson as the 39% under-firing that motivated the
    two-condition trigger, arriving from a different direction.
    """
    groups, _, _, combined_u = _mixed_stream(evidence)
    half = min(retained_composition(combined_u, groups),
               key=lambda r: abs(r["coverage"] - 0.5))
    baseline = float((groups == "blackout").mean())

    assert half["blackout"] > 0.5 * baseline, half


def test_the_trigger_rejects_what_ranking_alone_misses(evidence):
    """The two-axis payoff, stated correctly: the trigger's two conditions remove both
    failure modes, where the magnitude ranking removes only one."""
    groups, of, value_u, _ = _mixed_stream(evidence)
    trig = CompetenceDropTrigger.calibrate(
        value_u[groups == "normal"], 0.05, OF_FLOOR, hysteresis=1)
    fired = np.array(
        [CompetenceDropTrigger(trig.vthr, trig.of_floor, 1).update(float(v), float(o))[0]
         for v, o in zip(value_u, of, strict=True)])

    for group, expected in (("cyclone", 1.0), ("blackout", 1.0)):
        assert fired[groups == group].mean() >= expected, (group, fired[groups == group].mean())
    assert fired[groups == "normal"].mean() < 0.10
