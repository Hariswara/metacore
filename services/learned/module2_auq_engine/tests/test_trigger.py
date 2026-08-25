"""The competence-drop trigger fires on either axis, and says which one.

A single scalar threshold under-fires on sensing loss: it has to sit above the
in-distribution u at M1's nominal quality, and a blackout only pushes the combined u
part of the way there. Testing the two conditions separately catches both reliably,
which is what these tests pin.
"""

import numpy as np
import pytest
from edl import uncertainty
from state_contract import stack_features
from synthetic_data import sample_states_blackout, sample_states_id, sample_states_ood
from trigger import (
    REASON_BOTH,
    REASON_NONE,
    REASON_SENSING,
    REASON_VALUE,
    CompetenceDropTrigger,
)

OF_FLOOR = 0.4
FALSE_ALARM_RATE = 0.05


def _value_u(evidence, states):
    u, _, _ = uncertainty(evidence(stack_features(states)))
    return u.numpy()


def _observed(states):
    return np.array([s.quality.observed_fraction for s in states], dtype=np.float32)


def _fire(trig, value_u, observed_fraction):
    """Memoryless per-state evaluation -- a fresh trigger with no hysteresis, so this
    measures the condition rate rather than the debounced time series."""
    outcomes = [CompetenceDropTrigger(trig.vthr, trig.of_floor, 1).update(float(v), float(o))
                for v, o in zip(value_u, observed_fraction, strict=True)]
    return np.array([f for f, _ in outcomes]), [r for _, r in outcomes]


@pytest.fixture(scope="module")
def calibrated(evidence):
    """Threshold fitted on plain value-u over nominal-quality in-distribution states."""
    rng = np.random.default_rng(10)
    id_states, _ = sample_states_id(1500, rng, blackout_rate=0.0)
    trig = CompetenceDropTrigger.calibrate(
        _value_u(evidence, id_states), FALSE_ALARM_RATE, OF_FLOOR, hysteresis=2)
    return trig, evidence


def test_normal_operation_false_alarm_rate_is_low(calibrated):
    trig, evidence = calibrated
    rng = np.random.default_rng(11)
    states, _ = sample_states_id(1000, rng, blackout_rate=0.0)
    fired, reasons = _fire(trig, _value_u(evidence, states), _observed(states))

    assert fired.mean() < 0.10, fired.mean()
    assert REASON_SENSING not in reasons  # nominal quality must never look like a blackout


def test_cyclone_fires_on_the_value_axis(calibrated):
    trig, evidence = calibrated
    rng = np.random.default_rng(12)
    states, _ = sample_states_ood(800, rng)
    fired, reasons = _fire(trig, _value_u(evidence, states), _observed(states))

    assert fired.mean() > 0.95, fired.mean()
    assert all(r == REASON_VALUE for r in reasons)


def test_blackout_fires_on_the_sensing_axis(calibrated):
    """The case a single combined threshold missed: ordinary values, lost modality."""
    trig, evidence = calibrated
    rng = np.random.default_rng(13)
    states, _ = sample_states_blackout(800, rng)
    value_u = _value_u(evidence, states)
    fired, reasons = _fire(trig, value_u, _observed(states))

    assert fired.mean() > 0.95, fired.mean()
    # Sensing is the reason for the overwhelming majority; the handful of "both" are
    # blackout states that also happened to clear the value threshold, at its own 5%.
    assert reasons.count(REASON_SENSING) / len(reasons) > 0.90
    assert set(reasons) <= {REASON_SENSING, REASON_BOTH}
    # And the point of the redesign: the value axis alone would have missed them.
    assert (value_u > trig.vthr).mean() < 0.10


def test_compound_failure_reports_both(calibrated):
    """A cyclone that also takes the comms out is not the same event as either alone."""
    trig, evidence = calibrated
    rng = np.random.default_rng(14)
    states, _ = sample_states_ood(300, rng, blackout_rate=1.0)
    fired, reasons = _fire(trig, _value_u(evidence, states), _observed(states))

    assert fired.mean() == 1.0
    assert all(r == REASON_BOTH for r in reasons)


def test_hysteresis_debounces_a_single_step_spike():
    trig = CompetenceDropTrigger(value_threshold=0.5, of_floor=OF_FLOOR, hysteresis=2)

    assert trig.update(0.9, 1.0) == (False, REASON_NONE)   # one step over: not yet
    assert trig.update(0.1, 1.0) == (False, REASON_NONE)   # and it passed

    assert trig.update(0.9, 1.0)[0] is False               # sustained: first step
    assert trig.update(0.9, 1.0) == (True, REASON_VALUE)   # second confirms


def test_sensing_axis_also_debounces():
    trig = CompetenceDropTrigger(value_threshold=0.5, of_floor=OF_FLOOR, hysteresis=2)

    assert trig.update(0.1, 0.2) == (False, REASON_NONE)
    assert trig.update(0.1, 0.2) == (True, REASON_SENSING)


def test_reason_never_contradicts_the_state():
    """While the trigger is up the reason must say why; it must never hand M3 a
    competence drop with reason 'none' on the way back down."""
    trig = CompetenceDropTrigger(value_threshold=0.5, of_floor=OF_FLOOR, hysteresis=2)
    sequence = [(0.9, 1.0), (0.9, 1.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0)]

    seen = []
    for value_u, of in sequence:
        seen.append(trig.update(value_u, of))

    for fired, reason in seen:
        assert (reason == REASON_NONE) == (not fired), seen
    assert seen[2] == (True, REASON_VALUE)   # still up, still explained
    assert seen[3] == (False, REASON_NONE)   # cleared


def test_value_threshold_does_not_move_with_data_quality(evidence):
    """Calibrating on plain value-u is what keeps the value axis quality-independent.
    Fit it on the same states at two different qualities and it must not budge."""
    rng = np.random.default_rng(15)
    nominal, _ = sample_states_id(600, rng, blackout_rate=0.0)
    rng = np.random.default_rng(15)
    degraded, _ = sample_states_blackout(600, rng)

    a = CompetenceDropTrigger.calibrate(_value_u(evidence, nominal), FALSE_ALARM_RATE)
    b = CompetenceDropTrigger.calibrate(_value_u(evidence, degraded), FALSE_ALARM_RATE)

    assert a.vthr == pytest.approx(b.vthr)
