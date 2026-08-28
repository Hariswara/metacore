"""Small MLP policy and vanilla REINFORCE update for the 2-action gating MDP."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    def __init__(self, d_in: int = 12, n_actions: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def act(self, obs, rng_torch=None):
        """Sample an action; returns (action:int, log_prob:Tensor)."""
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    def action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(obs), dim=-1)


def reinforce_update(policy, opt, trajectories, baseline: float, momentum: float = 0.9):
    """Vanilla REINFORCE with a scalar moving-average baseline.

    Per-trajectory returns are standardized before the advantage is taken against
    ``baseline`` so long episodes do not drown the severity-local signal.
    trajectories: list of dicts with keys log_probs (list[Tensor]), rewards (list[float]).
    Returns (loss_value, new_baseline).
    """
    all_returns = []
    losses = []
    for traj in trajectories:
        rewards = traj["rewards"]
        log_probs = traj["log_probs"]
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + G
            returns.append(G)
        returns.reverse()
        all_returns.extend(returns)
        rt = torch.tensor(returns, dtype=torch.float32)
        rt = (rt - rt.mean()) / (rt.std() + 1e-8)
        for lp, R in zip(log_probs, rt.tolist()):
            losses.append(-lp * (R - 0.0))  # standardized return is the advantage

    if not losses:
        return 0.0, baseline

    loss = torch.stack(losses).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()

    mean_R = float(sum(all_returns) / len(all_returns))
    new_baseline = momentum * baseline + (1.0 - momentum) * mean_R
    return float(loss.item()), new_baseline
