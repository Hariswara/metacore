"""Dashboard step-through: generate one 12-d gate input, then run the saved policy.

Train still lives in run_demo.py (saves policy.pt into artifacts/dashboard/).
These commands are the two buttons after that: Generate situation / Process gate.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml

from gating_env import OBS_DIM, GatingEnv, describe_observation
from policy import MLPPolicy
from system1 import system1_action
from system2 import system2_action

STATE_DIR = Path(__file__).resolve().parent / "artifacts" / "dashboard"
POLICY_PATH = STATE_DIR / "policy.pt"
CONFIG_PATH = STATE_DIR / "last_config.yaml"
PENDING_PATH = STATE_DIR / "pending.json"
COUNTER_PATH = STATE_DIR / "generate_count.txt"


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _next_generate_index() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    if COUNTER_PATH.is_file():
        try:
            n = int(COUNTER_PATH.read_text(encoding="utf-8").strip() or "0")
        except ValueError:
            n = 0
    COUNTER_PATH.write_text(str(n + 1), encoding="utf-8")
    return n


def generate(output_path: Path) -> dict:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError("no trained config — click Train first")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    n = _next_generate_index()
    seed = int(cfg.get("seed", 0)) + n
    rng = np.random.default_rng(seed)
    env = GatingEnv(cfg, rng=rng)
    env.reset(seed=seed)
    t = int(rng.integers(0, env.episode_len))
    obs = env._obs_at(t)
    raw = _jsonable(env._last_raw)
    payload = {
        "step_index": t,
        "episode_len": env.episode_len,
        "observation": describe_observation(obs),
        "raw": raw,
        "seed_used": seed,
    }
    PENDING_PATH.write_text(json.dumps({"obs": [float(x) for x in obs], "raw": raw}), encoding="utf-8")
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def process(output_path: Path) -> dict:
    if not POLICY_PATH.is_file():
        raise FileNotFoundError("no saved policy — click Train first")
    if not PENDING_PATH.is_file():
        raise FileNotFoundError("no generated situation — click Generate first")

    pending = json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    obs = torch.tensor(pending["obs"], dtype=torch.float32).unsqueeze(0)
    raw = pending["raw"]

    blob = torch.load(POLICY_PATH, map_location="cpu")
    policy = MLPPolicy(d_in=int(blob.get("d_in", OBS_DIM)), n_actions=int(blob.get("n_actions", 2)))
    policy.load_state_dict(blob["state_dict"])
    policy.eval()
    with torch.no_grad():
        logits = policy(obs)
        action = int(torch.argmax(logits, dim=-1).item())
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()

    budget_left = float(raw.get("budget_remaining", 1.0))
    budget_exhausted = budget_left <= 0
    effective = 0 if budget_exhausted else action
    rng = np.random.default_rng(0)
    cost = 0.0
    if effective == 1:
        proposed = system2_action(raw, rng)
        origin = "SYSTEM2"
        if CONFIG_PATH.is_file():
            cost = float(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["reward"]["deliberation_cost"])
        else:
            cost = 0.10
    else:
        proposed = system1_action(raw)
        origin = "SYSTEM1"

    action_id = str(uuid.uuid4())
    payload = {
        "chosen": origin,
        "requested": "SYSTEM2" if action == 1 else "SYSTEM1",
        "budget_exhausted_fallback": bool(budget_exhausted and action == 1),
        "action_probs": {"SYSTEM1": float(probs[0]), "SYSTEM2": float(probs[1])},
        "epistemic_at_decision": float(raw.get("epistemic_uncertainty", 0.0)),
        "deliberation_cost": cost,
        "plan": {
            "action_id": action_id,
            "origin": origin,
            "breakers": proposed.get("breakers", []),
            "load_shed": proposed.get("load_shed", []),
            "dispatch": proposed.get("dispatch", []),
            "rationale": proposed.get("rationale", ""),
            "schema_version": "m3-out/0.1",
            "message_type": "ProposedControlAction",
        },
        "observation": describe_observation(pending["obs"]),
        "raw": raw,
    }
    output_path.write_text(json.dumps(_jsonable(payload)), encoding="utf-8")
    return payload


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] not in {"generate", "process"}:
        print("usage: python interactive.py generate|process <output.json>", file=sys.stderr)
        return 2
    cmd, out = sys.argv[1], Path(sys.argv[2])
    try:
        generate(out) if cmd == "generate" else process(out)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
