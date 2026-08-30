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


def reinforce_update(policy, opt, trajectories):
    """REINFORCE with per-trajectory return standardization.

    Each trajectory's undiscounted returns are z-scored, and that standardized
    return *is* the advantage. There is no moving-average baseline — a scalar
    baseline of raw returns would be a different scale from the z-scores and
    would not enter the gradient.

    trajectories: list of dicts with keys log_probs (list[Tensor]), rewards (list[float]).
    Returns (loss_value, mean_undiscounted_return) — the mean is diagnostic only.
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
        advantage = (rt - rt.mean()) / (rt.std() + 1e-8)
        for lp, adv in zip(log_probs, advantage.tolist(), strict=True):
            losses.append(-lp * adv)

    if not losses:
        return 0.0, 0.0

    loss = torch.stack(losses).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()

    mean_R = float(sum(all_returns) / len(all_returns))
    return float(loss.item()), mean_R
