"""Generate / process dashboard step-through, without a full REINFORCE run."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

import interactive
from gating_env import OBS_DIM
from policy import MLPPolicy


def test_generate_then_process(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    cfg["env"]["episode_len"] = 20
    state = tmp_path / "dashboard"
    state.mkdir()
    monkeypatch.setattr(interactive, "STATE_DIR", state)
    monkeypatch.setattr(interactive, "POLICY_PATH", state / "policy.pt")
    monkeypatch.setattr(interactive, "CONFIG_PATH", state / "last_config.yaml")
    monkeypatch.setattr(interactive, "PENDING_PATH", state / "pending.json")
    monkeypatch.setattr(interactive, "COUNTER_PATH", state / "generate_count.txt")

    (state / "last_config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    policy = MLPPolicy(d_in=OBS_DIM, n_actions=2)
    torch.save({"state_dict": policy.state_dict(), "d_in": OBS_DIM, "n_actions": 2}, state / "policy.pt")

    generated = interactive.generate(tmp_path / "gen.json")
    assert len(generated["observation"]) == OBS_DIM
    assert generated["observation"][0]["name"] == "epistemic_uncertainty"
    assert "trigger_reason" in generated["raw"]
    assert (state / "pending.json").is_file()

    processed = interactive.process(tmp_path / "proc.json")
    assert processed["chosen"] in {"SYSTEM1", "SYSTEM2"}
    assert processed["plan"]["origin"] == processed["chosen"]
    assert "load_shed" in processed["plan"]
    assert processed["plan"]["message_type"] == "ProposedControlAction"
    assert json.loads((tmp_path / "proc.json").read_text())["chosen"] == processed["chosen"]
