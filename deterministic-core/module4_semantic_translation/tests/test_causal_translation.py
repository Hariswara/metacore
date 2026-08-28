"""Tests for Semantic Translation, Abductive Attribution, and Grounding Constraints."""
import pytest
from translation.abductive.attribution import AbductiveAttributor
from translation.templates.causal_logger import TemplateCausalLogger
from translation.types import Generator
from verification.types import (
    BreakerCommand,
    Decision,
    DispatchSetpoint,
    ProposedControlAction,
    VerificationVerdict,
    Violation,
    ViolationType,
)


def test_grounded_causal_log_generation() -> None:
    action = ProposedControlAction(
        action_id="act-test-001",
        origin="SYSTEM2",
        breakers=[BreakerCommand(edge_id="Line_2_3", closed=False)],
        load_shed=[],
        dispatch=[],
        rationale="Deliberate islanding",
    )

    raw_violations = [
        Violation(
            type=ViolationType.VIOLATION_TYPE_UNDERVOLTAGE,
            element_id="N8",
            measured=0.8850,
            limit=0.9500,
            margin_fraction=-0.0684,
            attributed_component="",
        ),
        Violation(
            type=ViolationType.VIOLATION_TYPE_THERMAL_OVERLOAD,
            element_id="LINE_N8_N9",
            measured=165.0,
            limit=150.0,
            margin_fraction=0.1000,
            attributed_component="",
        ),
    ]

    # 1. Attribute violations
    attributed_violations = AbductiveAttributor.attribute_violations(action, raw_violations)
    assert attributed_violations[0].attributed_component == "breaker.Line_2_3"

    # 2. Build verdict
    verdict = VerificationVerdict(
        action_id=action.action_id,
        decision=Decision.DECISION_REJECT,
        violations=attributed_violations,
        solve_latency_ms=12.4,
    )

    # 3. Generate causal log
    log = TemplateCausalLogger.generate_log(verdict)

    assert log.action_id == action.action_id
    assert "REJECTED" in log.text
    assert "N8" in log.text
    assert "LINE_N8_N9" in log.text
    assert log.generator == Generator.GENERATOR_TEMPLATE

    # 4. Strict Grounding Invariant: grounded_entities must be a subset of violation element_ids
    violation_elements = {v.element_id for v in raw_violations}
    for entity in log.grounded_entities:
        assert entity in violation_elements, f"Entity {entity} is ungrounded!"


def test_approved_action_causal_log() -> None:
    verdict = VerificationVerdict(
        action_id="act-safe-001",
        decision=Decision.DECISION_APPROVE,
        violations=[],
        solve_latency_ms=8.5,
    )

    log = TemplateCausalLogger.generate_log(verdict)
    assert log.action_id == "act-safe-001"
    assert "verified safe" in log.text
    assert len(log.grounded_entities) == 0
