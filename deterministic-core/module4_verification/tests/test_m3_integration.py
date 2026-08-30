"""Integration Tests for Module 4 with M3 Sample Action Stream."""

from verification.firewall.verifier import PhysicsVerifier
from verification.types import ProposedControlAction

SAMPLE_M3_ACTIONS = [
    {
        "action_id": "e362fe37-72f6-46d9-b61a-76d3ab2f9f06",
        "origin": "SYSTEM1",
        "breakers": [],
        "load_shed": [{"node_id": "N8", "shed_fraction": 0.1117, "priority_tier": 3}],
        "dispatch": [],
        "rationale": "S1 reactive shed vuln=0.14",
    },
    {
        "action_id": "bcca98d3-11e5-43e4-b631-b4c2f262dc0c",
        "origin": "SYSTEM1",
        "breakers": [],
        "load_shed": [{"node_id": "N9", "shed_fraction": 0.0947, "priority_tier": 3}],
        "dispatch": [],
        "rationale": "S1 sensing-fallback shed vuln=0.21",
    },
    {
        "action_id": "66a45b32-ca0c-4682-8251-f515e58cc27e",
        "origin": "SYSTEM2",
        "breakers": [{"edge_id": "E_crit_1", "closed": True}],
        "load_shed": [
            {"node_id": "N11", "shed_fraction": 0.25, "priority_tier": 3},
            {"node_id": "N8", "shed_fraction": 0.0983, "priority_tier": 3},
        ],
        "dispatch": [{"node_id": "N4", "p_kw": 147.37, "q_kvar": 10.0}],
        "rationale": "S2 survival opt vuln=0.63 protect-tier1",
    },
    {
        "action_id": "a80f5b54-1dbb-4d0d-834c-4fdb114d164b",
        "origin": "SYSTEM2",
        "breakers": [{"edge_id": "E_crit_1", "closed": True}],
        "load_shed": [
            {"node_id": "N8", "shed_fraction": 0.25, "priority_tier": 3},
            {"node_id": "N12", "shed_fraction": 0.2325, "priority_tier": 3},
        ],
        "dispatch": [{"node_id": "N4", "p_kw": 184.21, "q_kvar": 10.0}],
        "rationale": "S2 survival opt vuln=0.88 protect-tier1",
    },
]


def test_m3_sample_actions_execution() -> None:
    verifier = PhysicsVerifier()

    for raw_action in SAMPLE_M3_ACTIONS:
        action = ProposedControlAction.model_validate(raw_action)
        verdict = verifier.verify(action)

        assert verdict.action_id == action.action_id
        assert verdict.decision in ("DECISION_APPROVE", "DECISION_REJECT")
        assert verdict.solve_latency_ms >= 0.0
