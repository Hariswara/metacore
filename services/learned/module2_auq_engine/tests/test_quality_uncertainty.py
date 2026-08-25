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
from edl import EDLNet, edl_mse_loss, kl_to_uniform, uncertainty, uncertainty_quality
from synthetic_data import sample_id, sample_ood

QUALITY_SWEEP = (1.0, 0.75, 0.5, 0.25, 0.1)


@pytest.fixture(scope="module")
def trained():
    """One trained head for the whole module. Training is seeded, so doing it once is
    identical to doing it per test and costs a sixth of the wall clock."""
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    Xtr, ytr = sample_id(3000, rng)
    mu, sd = Xtr.mean(0), Xtr.std(0)+1e-6
    m = EDLNet(Xtr.shape[1], 3)
    opt = torch.optim.Adam(m.parameters(), 2e-3, weight_decay=1e-5)
    Xt = torch.tensor(((Xtr-mu)/sd).astype('float32'))
    yt = torch.tensor(ytr)
    N = len(Xt)
    for ep in range(250):
        perm = torch.randperm(N)
        for i in range(0, N, 128):
            idx = perm[i:i+128]
            xb = Xt[idx]
            xo = xb + torch.randn_like(xb)*4.0
            loss = edl_mse_loss(m(xb), yt[idx], ep, 50) + 0.1*kl_to_uniform(m(xo)+1.0).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m, mu, sd


def _evidence(trained, x):
    m, mu, sd = trained
    return m(torch.tensor(((x-mu)/sd).astype('float32'))).detach()


def test_uncertainty_rises_as_quality_falls(trained):
    """The quality axis. Same in-distribution states, less of them observed."""
    rng = np.random.default_rng(1)
    e = _evidence(trained, sample_id(1000, rng)[0])
    us = [uncertainty_quality(e, of).mean().item() for of in QUALITY_SWEEP]

    assert all(us[i] < us[i+1] for i in range(len(us)-1)), us
    assert us[0] < 0.20 and us[-1] > 0.35, us


def test_value_ood_stays_high_regardless_of_quality(trained):
    """The value axis is independent: a cyclone reads u ~ 1 at any data quality."""
    rng = np.random.default_rng(2)
    e = _evidence(trained, sample_ood(400, rng))

    assert uncertainty_quality(e, 1.0).mean().item() >= 0.95
    assert uncertainty_quality(e, 0.5).mean().item() >= 0.95


def test_full_observation_reproduces_plain_uncertainty(trained):
    """observed_fraction=1 must be exactly the old u, or this is a change to the
    value axis rather than a second one, and the paper ablation is meaningless."""
    rng = np.random.default_rng(3)
    e = _evidence(trained, sample_id(500, rng)[0])
    plain, _, _ = uncertainty(e)

    assert torch.allclose(uncertainty_quality(e, 1.0), plain)


def test_accepts_a_per_row_observed_fraction(trained):
    """M1's real states will not share one mask, so the batch form has to work."""
    rng = np.random.default_rng(4)
    e = _evidence(trained, sample_id(4, rng)[0])
    per_row = torch.tensor([1.0, 0.5, 0.25, 0.1])

    batched = uncertainty_quality(e, per_row)
    one_at_a_time = torch.stack([
        uncertainty_quality(e[i:i+1], of.item())[0] for i, of in enumerate(per_row)
    ])

    assert batched.shape == (4,)
    assert torch.allclose(batched, one_at_a_time)


def test_uncertainty_is_bounded(trained):
    """u is a gating signal in [0,1]; M3 thresholds it directly."""
    rng = np.random.default_rng(5)
    e = _evidence(trained, sample_id(500, rng)[0])

    for of in QUALITY_SWEEP:
        u = uncertainty_quality(e, of)
        assert float(u.min()) > 0.0
        assert float(u.max()) <= 1.0


def test_output_contract_carries_observed_fraction():
    """m2-out/0.2 is additive: M3 gets the new field and can ignore it."""
    p = np.array([0.7, 0.2, 0.1])
    msg = build_output(0.42, p, 0.3, True, 0.5)

    assert SCHEMA_VERSION == "m2-out/0.2"
    assert msg.schema_version == "m2-out/0.2"
    assert msg.observed_fraction == 0.5
    assert msg.epistemic_uncertainty == pytest.approx(0.42)
    assert msg.state_class == 0

    round_tripped = M2Output(**json.loads(msg.to_json()))
    assert round_tripped == msg
