"""Epistemic uncertainty has two independent axes, and the gate fires on either.

  * value   -- the state is unlike anything seen in training (cyclone)
  * quality -- the state is mostly interpolated or missing (comms blackout)

The second axis is what makes M1's QualityMask load-bearing rather than metadata:
observed_fraction discounts the evidence, so a state M1 could only interpolate
cannot be scored as confidently as one it measured.
"""

import json

import numpy as np
import pytest
import torch
from contract import SCHEMA_VERSION, M2Output, build_output
from edl import uncertainty, uncertainty_quality
from synthetic_data import sample_id, sample_ood

QUALITY_SWEEP = (1.0, 0.75, 0.5, 0.25, 0.1)


def test_uncertainty_rises_as_quality_falls(evidence):
    """The quality axis. Same in-distribution states, less of them observed."""
    rng = np.random.default_rng(1)
    e = evidence(sample_id(1000, rng)[0])
    us = [uncertainty_quality(e, of).mean().item() for of in QUALITY_SWEEP]

    assert all(us[i] < us[i+1] for i in range(len(us)-1)), us
    # Qualitative: low when fully observed, materially higher when barely observed. The
    # exact values move with the width of the feature vector.
    assert us[0] < 0.20 and us[-1] > 3*us[0], us


def test_value_ood_stays_high_regardless_of_quality(evidence):
    """The value axis is independent: a cyclone reads u ~ 1 at any data quality."""
    rng = np.random.default_rng(2)
    e = evidence(sample_ood(400, rng))

    assert uncertainty_quality(e, 1.0).mean().item() >= 0.95
    assert uncertainty_quality(e, 0.5).mean().item() >= 0.95


def test_full_observation_reproduces_plain_uncertainty(evidence):
    """observed_fraction=1 must be exactly the old u, or this is a change to the
    value axis rather than a second one, and the paper ablation is meaningless."""
    rng = np.random.default_rng(3)
    e = evidence(sample_id(500, rng)[0])
    plain, _, _ = uncertainty(e)

    assert torch.allclose(uncertainty_quality(e, 1.0), plain)


def test_accepts_a_per_row_observed_fraction(evidence):
    """M1's real states will not share one mask, so the batch form has to work."""
    rng = np.random.default_rng(4)
    e = evidence(sample_id(4, rng)[0])
    per_row = torch.tensor([1.0, 0.5, 0.25, 0.1])

    batched = uncertainty_quality(e, per_row)
    one_at_a_time = torch.stack([
        uncertainty_quality(e[i:i+1], of.item())[0] for i, of in enumerate(per_row)
    ])

    assert batched.shape == (4,)
    assert torch.allclose(batched, one_at_a_time)


def test_uncertainty_is_bounded(evidence):
    """u is a gating signal in [0,1]; M3 thresholds it directly."""
    rng = np.random.default_rng(5)
    e = evidence(sample_id(500, rng)[0])

    for of in QUALITY_SWEEP:
        u = uncertainty_quality(e, of)
        assert float(u.min()) > 0.0
        assert float(u.max()) <= 1.0


def test_output_contract_carries_observed_fraction():
    """m2-out/0.3 is additive: M3 gets the new fields and can ignore them."""
    p = np.array([0.7, 0.2, 0.1])
    msg = build_output(0.42, p, 0.3, True, 0.5, "sensing")

    assert SCHEMA_VERSION == "m2-out/0.3"
    assert msg.schema_version == "m2-out/0.3"
    assert msg.observed_fraction == 0.5
    assert msg.trigger_reason == "sensing"
    assert msg.epistemic_uncertainty == pytest.approx(0.42)
    assert msg.state_class == 0

    round_tripped = M2Output(**json.loads(msg.to_json()))
    assert round_tripped == msg
