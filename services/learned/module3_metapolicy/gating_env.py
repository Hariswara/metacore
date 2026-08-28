"""Gymnasium environment for cost-aware System 1 / System 2 gating.

Observation is a 12-d Box; action is Discrete(2). When the deliberation budget
is exhausted the env forces System 1 regardless of the policy's pick.

Branches on M2's trigger_reason (m2-out/0.3):
  value / both → deliberation is the right response (unseen conditions)
  sensing      → conservative System 1 (missing data; thinking harder buys nothing)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from synthetic_context import sample_scenario, severity_index
from m2_stream import load_jsonl, replay_stream
from system1 import system1_action
from system2 import system2_action
from verifier import mock_verify

OBS_DIM = 12


def _reason_flags(reason: str) -> tuple[float, float]:
    """Return (reason_value, reason_sensing) binary flags."""
    r = reason or "none"
    return (
        1.0 if r in ("value", "both") else 0.0,
        1.0 if r in ("sensing", "both") else 0.0,
    )


def _build_obs(m2: dict, ctx: dict, budget_remaining: float, budget_total: float, k: int) -> np.ndarray:
    probs = list(m2["class_probabilities"])
    while len(probs) < k:
        probs.append(0.0)
    probs = probs[:k]
    sev = severity_index(ctx["severity"])
    tth = float(np.clip(ctx["time_to_hazard_onset_min"], -30.0, 30.0)) / 30.0
    rv, rs = _reason_flags(str(m2.get("trigger_reason", "none")))
    return np.array(
        [
            float(m2["epistemic_uncertainty"]),
            rv,
            rs,
            float(m2["state_class"]) / max(k - 1, 1),
            float(probs[0]),
            float(probs[1]),
            float(probs[2]),
            float(ctx["max_node_vulnerability"]),
            float(sev) / 3.0,
            float(tth),
            float(budget_remaining) / max(budget_total, 1e-6),
            float(m2.get("observed_fraction", 1.0)),
        ],
        dtype=np.float32,
    )


class GatingEnv(gym.Env):
    """Fixed-length episode zipping mock M1 context with resampled M2 stream."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, rng: np.random.Generator | None = None):
        super().__init__()
        self.cfg = cfg
        self.k = int(cfg.get("k_classes", 3))
        self.episode_len = int(cfg["env"]["episode_len"])
        self.budget_total = float(cfg["env"]["budget_per_episode"])
        self.reward_cfg = cfg["reward"]
        self.m2_path = cfg.get("m2_stream_path")  # None → default beside M2
        self._rng = rng if rng is not None else np.random.default_rng(cfg.get("seed", 0))

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self._m2_records = load_jsonl(self.m2_path)
        self._contexts: list[dict] = []
        self._m2_steps: list[dict] = []
        self._t = 0
        self._budget = self.budget_total
        self._last_raw: dict = {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._contexts = sample_scenario(self._rng, self.episode_len, self.cfg.get("scenario"))
        severity_schedule = [c["severity"] for c in self._contexts]
        self._m2_steps = list(
            replay_stream(
                self._m2_records,
                self._rng,
                self.episode_len,
                severity_schedule=severity_schedule,
            )
        )
        self._t = 0
        self._budget = self.budget_total
        obs = self._obs_at(0)
        return obs, {}

    def _obs_at(self, t: int) -> np.ndarray:
        m2 = self._m2_steps[t]
        ctx = self._contexts[t]
        self._last_raw = {**m2, **ctx, "budget_remaining": self._budget}
        return _build_obs(m2, ctx, self._budget, self.budget_total, self.k)

    def step(self, action):
        action = int(action)
        raw = self._last_raw
        budget_exhausted = self._budget <= 0
        effective = 0 if budget_exhausted else action

        ctrl_rng = self._rng
        if effective == 1:
            proposed = system2_action(raw, ctrl_rng)
            cost = float(self.reward_cfg["deliberation_cost"])
            self._budget -= 1.0
        else:
            proposed = system1_action(raw)
            cost = 0.0

        severity_norm = severity_index(raw["severity"]) / 3.0
        u = float(raw["epistemic_uncertainty"])
        thr = float(self.reward_cfg["trigger_threshold"])
        reason = str(raw.get("trigger_reason", "none"))
        value_axis = reason in ("value", "both")
        sensing_only = reason == "sensing"

        # Cause-aware reward (Duwaragie m2-out/0.3):
        # value/both → escalate; sensing-only → stay on conservative S1.
        if effective == 1 and sensing_only:
            benefit = float(self.reward_cfg["sensing_escalation_penalty"])
        elif effective == 1 and severity_norm < 0.25 and u < thr and not value_axis:
            benefit = float(self.reward_cfg["needless_escalation_penalty"])
        elif effective == 1 and severity_norm < 0.25 and not value_axis:
            benefit = 0.5 * float(self.reward_cfg["needless_escalation_penalty"])
        elif effective == 1 and value_axis:
            benefit = (
                float(self.reward_cfg["benefit_scale"])
                * max(severity_norm, 0.35)
                * max(u, 0.5)
            )
        elif effective == 1:
            benefit = float(self.reward_cfg["benefit_scale"]) * severity_norm * u
        elif effective == 0 and value_axis and severity_norm > 0.5:
            benefit = -float(self.reward_cfg["missed_escalation_penalty"])
        else:
            benefit = 0.0

        verdict = mock_verify(proposed, raw)
        reject = (
            float(self.reward_cfg["reject_penalty"])
            if verdict["decision"] == "REJECT"
            else 0.0
        )
        reward = float(benefit - cost - reject)

        info = {
            "budget_exhausted_fallback": bool(budget_exhausted and action == 1),
            "effective_action": effective,
            "requested_action": action,
            "proposed": proposed,
            "verdict": verdict,
            "benefit": benefit,
            "cost": cost,
            "reject": reject,
            "severity": raw["severity"],
            "epistemic_uncertainty": u,
            "competence_drop": bool(raw.get("competence_drop", False)),
            "trigger_reason": reason,
            "observed_fraction": float(raw.get("observed_fraction", 1.0)),
        }

        self._t += 1
        terminated = False
        truncated = self._t >= self.episode_len
        if truncated:
            obs = np.zeros(OBS_DIM, dtype=np.float32)
        else:
            obs = self._obs_at(self._t)
        return obs, reward, terminated, truncated, info
