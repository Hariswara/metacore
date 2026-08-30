"""Evaluation helpers: baselines, escalation-by-severity, deliberation cost.

No number is reported without the baseline it beats (repo convention).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from synthetic_context import HAZARD_STAGES


def _obs_u(obs) -> float:
    if hasattr(obs, "detach"):
        return float(obs.detach().reshape(-1)[0])
    return float(np.asarray(obs).reshape(-1)[0])


def threshold_action(obs, threshold: float) -> int:
    """Naive ``u > threshold`` rule — the 'why not just a threshold?' baseline."""
    return int(_obs_u(obs) > threshold)


def _select_action(mode, policy, obs_t, env):
    if mode == "always_s1":
        return 0
    if mode == "always_s2":
        return 1
    if mode == "threshold":
        thr = float(env.reward_cfg["trigger_threshold"])
        return threshold_action(obs_t, thr)
    # policy
    with torch.no_grad():
        logits = policy(obs_t)
        return int(torch.argmax(logits, dim=-1).item())


def run_episode(env, mode: str, policy=None) -> dict:
    obs, _ = env.reset()
    total_reward = 0.0
    total_cost = 0.0
    n_esc = 0
    by_sev = defaultdict(lambda: {"n": 0, "esc": 0})
    steps = 0
    while True:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = _select_action(mode, policy, obs_t, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_cost += float(info.get("cost", 0.0))
        eff = int(info.get("effective_action", action))
        sev = info.get("severity", "normal")
        by_sev[sev]["n"] += 1
        if eff == 1:
            n_esc += 1
            by_sev[sev]["esc"] += 1
        steps += 1
        if terminated or truncated:
            break
    return {
        "total_reward": total_reward,
        "avg_deliberation_cost": total_cost / max(steps, 1),
        "escalation_rate": n_esc / max(steps, 1),
        "by_severity": dict(by_sev),
        "steps": steps,
    }


def run_baseline(env, mode: str, policy=None, n_episodes: int = 5) -> float:
    """Mean total reward over n_episodes.

    mode in {always_s1, always_s2, threshold, policy}.
    ``threshold`` is the naive ``u > trigger_threshold`` rule.
    """
    rewards = [run_episode(env, mode, policy)["total_reward"] for _ in range(n_episodes)]
    return float(np.mean(rewards))


def avg_deliberation_cost(env, mode: str, policy=None, n_episodes: int = 5) -> float:
    costs = [
        run_episode(env, mode, policy)["avg_deliberation_cost"] for _ in range(n_episodes)
    ]
    return float(np.mean(costs))


def escalation_rate_by_severity(
    policy_or_rule, env, n_episodes: int = 10, exclude_sensing_only: bool = True
) -> dict:
    """% escalation per severity bucket.

    When ``exclude_sensing_only`` is True (default), sensing-only blackout steps
    are omitted — they correctly stay on System 1 and would otherwise break
    severity monotonicity under Duwaragie's cause-aware gate.
    """
    if isinstance(policy_or_rule, str):
        mode = policy_or_rule
        policy = None
        rule_fn = None
    elif callable(policy_or_rule) and not hasattr(policy_or_rule, "forward"):
        mode = "rule"
        policy = None
        rule_fn = policy_or_rule
    else:
        mode = "policy"
        policy = policy_or_rule
        rule_fn = None

    agg = {s: {"n": 0, "esc": 0} for s in HAZARD_STAGES}
    for _ in range(n_episodes):
        obs, _ = env.reset()
        while True:
            if rule_fn is not None:
                action = int(rule_fn(obs))
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = _select_action(mode, policy, obs_t, env)
            obs, _, terminated, truncated, info = env.step(action)
            if exclude_sensing_only and info.get("trigger_reason") == "sensing":
                if terminated or truncated:
                    break
                continue
            sev = info.get("severity", "normal")
            if sev in agg:
                agg[sev]["n"] += 1
                if int(info.get("effective_action", action)) == 1:
                    agg[sev]["esc"] += 1
            if terminated or truncated:
                break

    rates = {}
    for s in HAZARD_STAGES:
        n = agg[s]["n"]
        rates[s] = float(agg[s]["esc"] / n) if n else 0.0
    return rates


def escalation_rate_by_trigger_reason(policy_or_rule, env, n_episodes: int = 10) -> dict:
    """% escalation per M2 trigger_reason bucket (none/value/sensing/both).

    Target behaviour for the cause-aware gate:
      value / both → high escalation
      sensing      → low escalation (conservative S1)
      none         → low escalation
    """
    reasons = ("none", "value", "sensing", "both")
    if isinstance(policy_or_rule, str):
        mode = policy_or_rule
        policy = None
        rule_fn = None
    elif callable(policy_or_rule) and not hasattr(policy_or_rule, "forward"):
        mode = "rule"
        policy = None
        rule_fn = policy_or_rule
    else:
        mode = "policy"
        policy = policy_or_rule
        rule_fn = None

    agg = {r: {"n": 0, "esc": 0} for r in reasons}
    for _ in range(n_episodes):
        obs, _ = env.reset()
        while True:
            if rule_fn is not None:
                action = int(rule_fn(obs))
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = _select_action(mode, policy, obs_t, env)
            obs, _, terminated, truncated, info = env.step(action)
            reason = info.get("trigger_reason", "none")
            if reason in agg:
                agg[reason]["n"] += 1
                if int(info.get("effective_action", action)) == 1:
                    agg[reason]["esc"] += 1
            if terminated or truncated:
                break

    rates = {}
    for r in reasons:
        n = agg[r]["n"]
        rates[r] = float(agg[r]["esc"] / n) if n else 0.0
    return rates


def is_nondecreasing(rates: dict) -> bool:
    vals = [rates[s] for s in HAZARD_STAGES]
    return all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))
