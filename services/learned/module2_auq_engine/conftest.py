"""Shared test setup for Module 2.

Two jobs: put this module's root on sys.path for the repo-wide lane, and train the
evidential head once for the whole session rather than once per test file.

The prototype lives as flat modules at the service root (edl.py, synthetic_data.py,
...) and imports them flat, which works when run_demo.py is run from this directory.
The repo-wide lane runs `pytest services` from the root, where the root pyproject's
`pythonpath` only adds each service's `src/`. This bridges the two without moving
files or adding this directory to the shared `pythonpath`, where generic names like
`contract` and `evaluate` would collide with the other services.
"""

import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

TRAIN_EPOCHS = 250
TRAIN_N = 3000


@pytest.fixture(scope="session")
def trained_head():
    """One trained head for the whole session -> (model, mu, sd).

    Training is seeded, so doing it once is identical to doing it per test and saves
    roughly 18 seconds per test file that would otherwise repeat it.
    """
    import numpy as np
    import torch
    from edl import EDLNet, edl_mse_loss, kl_to_uniform
    from synthetic_data import sample_id

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    Xtr, ytr = sample_id(TRAIN_N, rng)
    mu, sd = Xtr.mean(0), Xtr.std(0)+1e-6
    m = EDLNet(Xtr.shape[1], 3)
    opt = torch.optim.Adam(m.parameters(), 2e-3, weight_decay=1e-5)
    Xt = torch.tensor(((Xtr-mu)/sd).astype('float32'))
    yt = torch.tensor(ytr)
    N = len(Xt)
    for ep in range(TRAIN_EPOCHS):
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


@pytest.fixture(scope="session")
def evidence(trained_head):
    """Callable: raw feature array -> the head's evidence tensor, detached."""
    import torch

    m, mu, sd = trained_head

    def _evidence(x):
        return m(torch.tensor(((x-mu)/sd).astype('float32'))).detach()

    return _evidence
