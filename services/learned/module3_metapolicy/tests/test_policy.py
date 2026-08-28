"""Policy / escalation monotonicity and cause-aware heuristic tests."""
from __future__ import annotations

import numpy as np
import torch
import yaml
from pathlib import Path

from gating_env import GatingEnv, OBS_DIM
from policy import MLPPolicy
from evaluate import escalation_rate_by_severity
from synthetic_context import HAZARD_STAGES


def _cfg():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    cfg["env"]["episode_len"] = 40
    cfg["env"]["budget_per_episode"] = 20
    return cfg


def _cause_aware_rule(obs: np.ndarray) -> int:
    """Mirror run_demo heuristic: sensing-only → S1; value → S2 under severe+."""
    u, rv, rs, sev = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[8])
    if rs >= 0.5 and rv < 0.5:
        return 0
    if rv >= 0.5:
        if sev >= 0.66:
            return 1
        if sev >= 0.33 and u >= 0.85:
            return 1
        return 0
    return 1 if (sev >= 0.99 and u >= 0.5) else 0


def test_escalation_monotonic_in_uncertainty():
    rates_proxy = []
    for u in np.linspace(0.0, 1.0, 11):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = float(u)
        obs[1] = 1.0  # value axis
        obs[8] = 0.67  # severe
        rates_proxy.append(_cause_aware_rule(obs))
    assert all(rates_proxy[i] <= rates_proxy[i + 1] for i in range(len(rates_proxy) - 1))

    env = GatingEnv(_cfg(), rng=np.random.default_rng(0))
    rates = escalation_rate_by_severity(_cause_aware_rule, env, n_episodes=6)
    assert set(rates) == set(HAZARD_STAGES)
    assert rates["extreme"] + 1e-9 >= rates["normal"]


def test_sensing_only_stays_system1():
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0] = 0.53
    obs[1] = 0.0  # not value
    obs[2] = 1.0  # sensing
    obs[8] = 0.67
    assert _cause_aware_rule(obs) == 0


def test_value_axis_escalates():
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0] = 1.0
    obs[1] = 1.0
    obs[2] = 0.0
    obs[8] = 0.67
    assert _cause_aware_rule(obs) == 1


def test_policy_forward_shape():
    p = MLPPolicy(d_in=OBS_DIM)
    x = torch.zeros(1, OBS_DIM)
    logits = p(x)
    assert logits.shape == (1, 2)
    probs = p.action_probs(x)
    assert abs(float(probs.sum().detach()) - 1.0) < 1e-5
