"""Seeded training is bit-reproducible in-process.

This is the cheap half of the reproducibility story and it passes on its own. The
half that needed fixing was the dependency: the reported metrics drifted between
environments because `torch>=2.4` floated, not because the seeding was wrong, so
the torch minor version is now pinned in pyproject.toml and requirements.txt.

Deliberately light -- 400 samples, 20 epochs -- because it trains twice and its job
is to catch a seeding regression, not to produce a good model.
"""

import numpy as np
import torch
from edl import EDLNet, edl_mse_loss, kl_to_uniform, uncertainty
from synthetic_data import sample_id

SEED = 0
N = 400
EPOCHS = 20
BATCH = 128


def _train_and_score():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    x, y = sample_id(N, rng)
    mu, sd = x.mean(0), x.std(0)+1e-6
    xt = torch.tensor(((x-mu)/sd).astype('float32'))
    yt = torch.tensor(y)
    m = EDLNet(x.shape[1], 3)
    opt = torch.optim.Adam(m.parameters(), 2e-3, weight_decay=1e-5)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), BATCH):
            idx = perm[i:i+BATCH]
            xb = xt[idx]
            xo = xb + torch.randn_like(xb)*4.0
            loss = edl_mse_loss(m(xb), yt[idx], ep, 50) + 0.1*kl_to_uniform(m(xo)+1.0).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        u, _, _ = uncertainty(m(xt))
    return u.numpy()


def test_two_seeded_runs_agree():
    a = _train_and_score()
    b = _train_and_score()

    assert np.allclose(a, b), float(np.abs(a-b).max())


def test_two_seeded_runs_are_bit_identical():
    """Stronger than allclose, and it holds today. If this starts failing while
    test_two_seeded_runs_agree still passes, something nondeterministic crept into
    the training path -- thread-count-dependent reductions are the usual cause, and
    torch.set_num_threads(1) next to the seeding is the usual fix."""
    assert np.array_equal(_train_and_score(), _train_and_score())


def test_data_generation_is_seeded():
    """The mock draws must not drift either -- the states are half the experiment."""
    a, ya = sample_id(64, np.random.default_rng(SEED))
    b, yb = sample_id(64, np.random.default_rng(SEED))

    assert np.array_equal(a, b)
    assert np.array_equal(ya, yb)
