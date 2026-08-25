"""Baselines and the OOD-regularisation ablation, for the paper's comparison table.

Four methods, all on the same architecture, normalisation and training data so the
comparison is about the uncertainty mechanism and nothing else:

  softmax max-prob   the standard cheap baseline; score = 1 - max softmax probability
  MC-Dropout (T=20)  the standard Bayesian-ish baseline; score = predictive entropy
                     over T stochastic passes
  EDL (ours)         Dirichlet evidence with the OOD-aware regulariser
  EDL, no OOD-reg    the ablation: identical, with ood_reg = 0

Every score is oriented so that **higher means more out-of-distribution**, which is what
`evaluate.auroc(pos=ood, neg=id)` expects.

The ablation is the point of this file. Plain EDL is not enough on far-OOD tabular data
-- it extrapolates confidently, exactly as the softmax baseline does -- so the claim that
the regulariser is load-bearing has to be measured, not asserted.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from edl import EDLNet, edl_mse_loss, kl_to_uniform, uncertainty

HIDDEN = 32
LR = 2e-3
WEIGHT_DECAY = 1e-5
BATCH = 128
KL_ANNEAL_EPOCHS = 50
OOD_PROXY_SIGMA = 4.0
DEFAULT_OOD_REG = 0.1
MC_PASSES = 20
DROPOUT_P = 0.2


class MLP(nn.Module):
    """Same shape as EDLNet, but emitting logits rather than evidence.

    Dropout sits after each hidden activation so MC-Dropout has something to sample;
    at p=0 this is architecturally identical to the EDL trunk.
    """

    def __init__(self, d_in, k, hidden=HIDDEN, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, k),
        )

    def forward(self, x):
        return self.net(x)


def _batches(n, generator=None):
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, BATCH):
        yield perm[i:i+BATCH]


def train_softmax(x, y, epochs, dropout=0.0, seed=0):
    """Plain cross-entropy classifier. Also the trunk for MC-Dropout (dropout > 0)."""
    torch.manual_seed(seed)
    xt, yt = torch.as_tensor(x), torch.as_tensor(y)
    m = MLP(xt.shape[1], int(yt.max().item())+1, dropout=dropout)
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WEIGHT_DECAY)
    m.train()
    for _ in range(epochs):
        for idx in _batches(len(xt)):
            loss = F.cross_entropy(m(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return m


def train_edl(x, y, epochs, ood_reg=DEFAULT_OOD_REG, seed=0):
    """The EDL head. `ood_reg=0.0` is the ablation -- everything else is identical."""
    torch.manual_seed(seed)
    xt, yt = torch.as_tensor(x), torch.as_tensor(y)
    m = EDLNet(xt.shape[1], int(yt.max().item())+1, hidden=HIDDEN)
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WEIGHT_DECAY)
    for ep in range(epochs):
        for idx in _batches(len(xt)):
            xb = xt[idx]
            loss = edl_mse_loss(m(xb), yt[idx], ep, KL_ANNEAL_EPOCHS)
            if ood_reg:
                xo = xb + torch.randn_like(xb)*OOD_PROXY_SIGMA
                loss = loss + ood_reg*kl_to_uniform(m(xo)+1.0).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


# ------------------------------------------------------------------ scoring
# Each returns (ood_score, class_probabilities). Higher score = more OOD.

def score_softmax(m, x):
    with torch.no_grad():
        p = F.softmax(m(torch.as_tensor(x)), dim=1)
    return (1.0 - p.max(1).values).numpy(), p.numpy()


def score_mc_dropout(m, x, passes=MC_PASSES, seed=0):
    """Predictive entropy of the mean over T stochastic passes, dropout left on."""
    torch.manual_seed(seed)
    xt = torch.as_tensor(x)
    m.train()                      # keep dropout active at inference; no batchnorm here
    with torch.no_grad():
        p = torch.stack([F.softmax(m(xt), dim=1) for _ in range(passes)]).mean(0)
    m.eval()
    ent = -(p*torch.log(p+1e-9)).sum(1)
    return ent.numpy(), p.numpy()


def score_edl(m, x):
    with torch.no_grad():
        u, p, _ = uncertainty(m(torch.as_tensor(x)))
    return u.numpy(), p.numpy()


# ------------------------------------------------------------------ latency

def time_scoring(score_fn, x, repeats=5):
    """Milliseconds per sample for one scoring pass, best of `repeats` after a warm-up.

    Best-of rather than mean: we are after the method's inherent cost, and the tail here
    is scheduling noise. MC-Dropout pays T forward passes for its score, which is the
    number that matters for a real-time gate.
    """
    score_fn(x)                    # warm up
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        score_fn(x)
        best = min(best, time.perf_counter() - start)
    return best*1000.0/len(x)
