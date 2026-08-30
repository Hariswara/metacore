"""Tests for Module 4 Semantic Translation and Grounding Invariants."""

from translation.abductive.attribution import AbductiveAttributor
from translation.templates.causal_logger import TemplateCausalLogger
from translation.types import (
    BreakerCommand,
    CausalLog,
    Decision,
    DispatchSetpoint,
    ProposedControlAction,
    VerificationVerdict,
    Violation,
    ViolationType,
)


def test_grounded_causal_log_generation() -> None:
    # 1. Simulate an unsafe action with undervoltage & line thermal overload
    action = ProposedControlAction(
        action_id="act-unsafe-001",
        origin="SYSTEM2",
        breakers=[BreakerCommand(edge_id="Line_2_3", closed=False)],
        dispatch=[DispatchSetpoint(node_id="N8", p_kw=3500.0, q_kvar=500.0)],
        rationale="Overload test",
    )

    raw_violations = [
        Violation(
            type=ViolationType.VIOLATION_TYPE_UNDERVOLTAGE,
            element_id="N8",
            limit=0.95,
            measured=0.91,
            margin_fraction=-0.042,
        ),
        Violation(
            type=ViolationType.VIOLATION_TYPE_THERMAL_OVERLOAD,
            element_id="LINE_SOURCE_N1",
            limit=300.0,
            measured=330.0,
            margin_fraction=0.10,
        ),
    ]

    # 2. Run Abductive Attribution
    attributed_violations = AbductiveAttributor.attribute_violations(action, raw_violations)
    assert len(attributed_violations) == 2
    assert "breaker.Line_2_3" in attributed_violations[0].attributed_component

    # 3. Generate Grounded Causal Log
    verdict = VerificationVerdict(
        action_id=action.action_id,
        decision=Decision.DECISION_REJECT,
        violations=attributed_violations,
        solve_latency_ms=0.45,
    )

    causal_log: CausalLog = TemplateCausalLogger.generate_log(verdict)

    # Invariant: Log text must mention the elements and the attributed cause
    assert "N8" in causal_log.text
    assert "LINE_SOURCE_N1" in causal_log.text
    assert "Undervoltage" in causal_log.text
    assert "Thermal overload" in causal_log.text

    # Strict Grounding Invariant: grounded_entities MUST be a subset of violation element_ids
    violation_elements = {v.element_id for v in verdict.violations}
    for entity in causal_log.grounded_entities:
        assert entity in violation_elements, f"Entity {entity} not grounded in violation elements!"


def test_approved_action_causal_log() -> None:
    verdict = VerificationVerdict(
        action_id="act-safe-001",
        decision=Decision.DECISION_APPROVE,
        violations=[],
        solve_latency_ms=0.32,
    )

    causal_log = TemplateCausalLogger.generate_log(verdict)
    assert "verified safe" in causal_log.text
    assert len(causal_log.grounded_entities) == 0
