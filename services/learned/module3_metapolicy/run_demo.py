"""End-to-end Module 3 prototype on the published M2 mock stream + synthetic M1 context.
Trains a REINFORCE gating policy -> reports reward vs always-S1 / always-S2 /
u-threshold -> checks escalation monotonicity by severity -> emits sample M3->M4 messages.

Usage: python run_demo.py [config_path] [output_json_path]
Both args are optional and backward-compatible: with neither, behaviour is unchanged
(reads ./config.yaml, writes ./sample_m3_to_m4.jsonl). Passing output_json_path is how
the gateway's /module3/run route drives a one-off run with overridden knobs — it writes
the full result (reward, escalation, decisions) as JSON there instead of the committed
sample_m3_to_m4.jsonl, so an on-demand run never clobbers the checked-in M4 contract fixture."""
from __future__ import annotations

import json
import random
import sys

import numpy as np
import torch
import yaml
from gating_env import OBS_DIM, GatingEnv
from m3_evaluate import (
    avg_deliberation_cost,
    escalation_rate_by_severity,
    escalation_rate_by_trigger_reason,
    is_nondecreasing,
    run_baseline,
)
from policy import MLPPolicy, reinforce_update

config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
output_json_path = sys.argv[2] if len(sys.argv) > 2 else None
policy_path = sys.argv[3] if len(sys.argv) > 3 else None

cfg = yaml.safe_load(open(config_path))
random.seed(cfg["seed"])
np.random.seed(cfg["seed"])
torch.manual_seed(cfg["seed"])
torch.use_deterministic_algorithms(True)

rng = np.random.default_rng(cfg["seed"])
env = GatingEnv(cfg, rng=rng)
policy = MLPPolicy(d_in=OBS_DIM, n_actions=2)
# Mild S1 prior so early training does not burn the whole budget on calm steps.
with torch.no_grad():
    policy.net[-1].bias[0] = 0.4
    policy.net[-1].bias[1] = -0.4
opt = torch.optim.Adam(policy.parameters(), lr=cfg["train"]["lr"])
batch_n = int(cfg["train"]["batch_episodes"])
n_train = int(cfg["train"]["episodes"])

print("=" * 56)
print("MODULE 3 - GATING / META-POLICY - TRAINING")
print("=" * 56)

def heuristic_action(obs_np: np.ndarray) -> int:
    """Cause-aware rule (m2-out/0.3): value/both → S2; sensing-only → S1."""
    u = float(obs_np[0])
    reason_value = float(obs_np[1])
    reason_sensing = float(obs_np[2])
    sev = float(obs_np[8])
    # Sensing-only blackout: conservative System 1 — don't deliberate on absent data.
    if reason_sensing >= 0.5 and reason_value < 0.5:
        return 0
    # Value axis: escalate under severe/extreme (save budget; elevated only if very unsure).
    if reason_value >= 0.5:
        if sev >= 0.66:
            return 1
        if sev >= 0.33 and u >= 0.85:
            return 1
        return 0
    if sev >= 0.99 and u >= 0.5:
        return 1
    return 0

# Behaviour-cloning warm-start onto the cost-aware heuristic, then REINFORCE fine-tune.
bc_epochs = 40
bc_loss_fn = torch.nn.CrossEntropyLoss()
print(f"  warm-start BC ({bc_epochs} episodes) ...")
for _ in range(bc_epochs):
    obs, _ = env.reset()
    while True:
        target = heuristic_action(obs)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = policy(obs_t)
        loss = bc_loss_fn(logits, torch.tensor([target]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        obs, _, terminated, truncated, _ = env.step(target)
        if terminated or truncated:
            break

trajectories = []
for ep in range(n_train):
    obs, _ = env.reset()
    log_probs, rewards = [], []
    while True:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        obs, reward, terminated, truncated, info = env.step(int(action.item()))
        rewards.append(float(reward))
        if terminated or truncated:
            break
    trajectories.append({"log_probs": log_probs, "rewards": rewards})
    if len(trajectories) >= batch_n:
        loss, mean_r = reinforce_update(policy, opt, trajectories)
        trajectories = []
        if (ep + 1) % 25 == 0:
            print(f"  episode {ep+1:4d}/{n_train}  loss={loss:.4f}  mean_R={mean_r:.3f}")

if trajectories:
    loss, mean_r = reinforce_update(policy, opt, trajectories)

# --- evaluation ---
n_eval = int(cfg["eval"]["n_episodes"])
print("\n" + "=" * 56)
print("MODULE 3 - RESULTS (vs baselines)")
print("=" * 56)

r_s1 = run_baseline(env, "always_s1", n_episodes=n_eval)
r_s2 = run_baseline(env, "always_s2", n_episodes=n_eval)
r_thr = run_baseline(env, "threshold", n_episodes=n_eval)
r_pol = run_baseline(env, "policy", policy=policy, n_episodes=n_eval)
c_pol = avg_deliberation_cost(env, "policy", policy=policy, n_episodes=n_eval)
c_s2 = avg_deliberation_cost(env, "always_s2", n_episodes=n_eval)
c_thr = avg_deliberation_cost(env, "threshold", n_episodes=n_eval)
thr = float(cfg["reward"]["trigger_threshold"])

print(f"total reward  always-S1           : {r_s1:.3f}")
print(f"total reward  always-S2           : {r_s2:.3f}")
print(f"total reward  u>{thr:.2f} threshold     : {r_thr:.3f}")
print(f"total reward  trained policy      : {r_pol:.3f}   (target: beat all baselines)")
print(f"avg deliber. cost  always-S2      : {c_s2:.3f}")
print(f"avg deliber. cost  u-threshold    : {c_thr:.3f}")
print(f"avg deliber. cost  trained policy : {c_pol:.3f}")

rates = escalation_rate_by_severity(policy, env, n_episodes=n_eval)
print("\nescalation rate by severity (excl. sensing-only; target: non-decreasing):")
for sev, rate in rates.items():
    print(f"  {sev:10s}  {rate:.3f}")
mono = is_nondecreasing(rates)
print(f"monotonic non-decreasing         : {mono}")

by_reason = escalation_rate_by_trigger_reason(policy, env, n_episodes=n_eval)
print("\nescalation rate by trigger_reason (value/both high; sensing -> S1):")
for reason, rate in by_reason.items():
    print(f"  {reason:10s}  {rate:.3f}")

# --- emit sample M3 -> M4: one PCA+GD pair per severity stage ---
print("\nSample M3 -> M4 contract messages (ProposedControlAction / GatingDecision):")
obs, _ = env.reset()
by_sev = {}
records = []
# Context is NOT part of the M3->M4 wire contract (M3_TO_M4_CONTRACT.md) -- Hariswara's
# verifier only ever sees `records`. This is purely for the gateway/dashboard's benefit,
# so a viewer can see the M2 input (u, trigger_reason, severity, observed_fraction) and
# M4-mock verdict that produced each action, alongside the proposed load_shed/dispatch.
context = []
emitted = 0
while True:
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = int(torch.argmax(policy(obs_t), dim=-1).item())
    obs, reward, terminated, truncated, info = env.step(action)
    sev = info["severity"]
    action_id = f"m3-{int(cfg['seed']):04d}-{emitted:02d}-{sev}"
    proposed = info["proposed"]
    eff = int(info["effective_action"])
    origin = "SYSTEM2" if eff == 1 else "SYSTEM1"
    pca = {
        "action_id": action_id,
        "origin": origin,
        "breakers": proposed.get("breakers", []),
        "load_shed": proposed.get("load_shed", []),
        "dispatch": proposed.get("dispatch", []),
        "rationale": proposed.get("rationale", ""),
        "schema_version": "m3-out/0.1",
        "message_type": "ProposedControlAction",
    }
    gd = {
        "action_id": action_id,
        "chosen": origin,
        "epistemic_at_decision": float(info["epistemic_uncertainty"]),
        "expected_survival_benefit": float(info["benefit"]),
        "deliberation_cost": float(info["cost"]),
        "latency_ms": 2.0 if eff == 0 else 18.0,
        "budget_exhausted_fallback": bool(info["budget_exhausted_fallback"]),
        "schema_version": "m3-out/0.1",
        "message_type": "GatingDecision",
        "timestamp": 1_704_067_200.0 + float(emitted),
    }
    if sev not in by_sev:
        by_sev[sev] = (pca, gd)
        records.extend([pca, gd])
        context.append({
            "action_id": action_id,
            "severity": sev,
            "trigger_reason": info["trigger_reason"],
            "epistemic_uncertainty": float(info["epistemic_uncertainty"]),
            "observed_fraction": float(info["observed_fraction"]),
            "competence_drop": bool(info["competence_drop"]),
            "reward": float(reward),
            "verdict": info["verdict"]["decision"],
            "violations": info["verdict"]["violations"],
        })
        emitted += 1
        print(json.dumps(gd))
    if terminated or truncated:
        break

if output_json_path:
    result = {
        "reward": {
            "always_s1": r_s1,
            "always_s2": r_s2,
            "threshold": r_thr,
            "trained_policy": r_pol,
        },
        "avg_deliberation_cost": {
            "always_s2": c_s2,
            "threshold": c_thr,
            "trained_policy": c_pol,
        },
        "escalation_by_severity": rates,
        "monotonic_nondecreasing": mono,
        "escalation_by_trigger_reason": by_reason,
        "decisions": records,
        "decision_context": context,
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    print(f"\nwrote result -> {output_json_path}")
else:
    with open("sample_m3_to_m4.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nwrote {len(records)} records -> sample_m3_to_m4.jsonl")

if policy_path:
    from pathlib import Path as _Path
    dest = _Path(policy_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "d_in": OBS_DIM, "n_actions": 2}, dest)
    print(f"wrote policy -> {dest}")
