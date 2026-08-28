"""GatingEnv shape, budget invariants, and trigger_reason branching."""
from __future__ import annotations

import numpy as np
import yaml
from pathlib import Path

from gating_env import GatingEnv, OBS_DIM


def _cfg():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    cfg["env"]["episode_len"] = 20
    cfg["env"]["budget_per_episode"] = 3
    return cfg


def test_env_reset_step_shapes():
    env = GatingEnv(_cfg(), rng=np.random.default_rng(0))
    obs, info = env.reset()
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert env.action_space.n == 2
    obs2, reward, terminated, truncated, info = env.step(0)
    assert obs2.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert "budget_exhausted_fallback" in info
    assert "trigger_reason" in info
    assert "observed_fraction" in info
    assert "proposed" in info
    assert "verdict" in info


def test_budget_exhaustion_forces_system1():
    cfg = _cfg()
    cfg["env"]["budget_per_episode"] = 2
    cfg["env"]["episode_len"] = 10
    env = GatingEnv(cfg, rng=np.random.default_rng(1))
    env.reset()
    _, _, _, _, info0 = env.step(1)
    assert info0["effective_action"] == 1
    _, _, _, _, info1 = env.step(1)
    assert info1["effective_action"] == 1
    _, _, _, _, info2 = env.step(1)
    assert info2["requested_action"] == 1
    assert info2["effective_action"] == 0
    assert info2["budget_exhausted_fallback"] is True
    assert info2["proposed"]["origin"] == "SYSTEM1"


def test_sensing_escalation_is_penalised():
    """Deliberating on a sensing-only drop must not be rewarded."""
    cfg = _cfg()
    env = GatingEnv(cfg, rng=np.random.default_rng(2))
    env.reset()
    # Force a sensing-only raw state into the env's last observation path.
    env._last_raw = {
        "epistemic_uncertainty": 0.53,
        "competence_drop": True,
        "trigger_reason": "sensing",
        "observed_fraction": 0.375,
        "state_class": 0,
        "class_probabilities": [0.8, 0.1, 0.1],
        "max_node_vulnerability": 0.3,
        "mean_node_vulnerability": 0.2,
        "top_at_risk_nodes": ["N12", "N11"],
        "time_to_hazard_onset_min": 10.0,
        "severity": "normal",
        "budget_remaining": 3.0,
    }
    _, reward_s2, _, _, info_s2 = env.step(1)
    assert info_s2["trigger_reason"] == "sensing"
    assert reward_s2 < 0  # sensing_escalation_penalty + deliberation_cost
