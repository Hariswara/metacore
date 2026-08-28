"""Physics Verifier Engine: The hard, deterministic safety firewall.

ZERO ML DEPENDENCIES.
"""
from typing import List, Optional
from ..opendss.circuit import CircuitTwin
from ..powerflow.action_applicator import ActionApplicator
from ..powerflow.solver import PowerFlowSolver
from .limits import PhysicsLimitsChecker, SafetyLimitsConfig
from ..types import (
    Decision,
    ProposedControlAction,
    RejectionTrace,
    VerificationVerdict,
    Violation,
)


class PhysicsVerifier:
    """Hard synchronous gatekeeper for power grid control actions."""

    def __init__(
        self,
        circuit: Optional[CircuitTwin] = None,
        limits_checker: Optional[PhysicsLimitsChecker] = None,
        spec_version: str = "safety-spec-v1.0",
        network_model_version: str = "delft-3island-v1",
    ) -> None:
        self.circuit = circuit or CircuitTwin()
        self.applicator = ActionApplicator(self.circuit)
        self.limits_checker = limits_checker or PhysicsLimitsChecker()
        self.spec_version = spec_version
        self.network_model_version = network_model_version

    def verify(self, action: ProposedControlAction) -> VerificationVerdict:
        """Evaluates a proposed action against the OpenDSS physical twin."""
        try:
            # 1. Reset circuit to pristine baseline before applying action
            self.circuit.reset_to_base()

            # 2. Apply proposed control action
            malformed_violations = self.applicator.apply_action(action)
            if malformed_violations:
                return self._build_verdict(
                    action_id=action.action_id,
                    decision=Decision.DECISION_REJECT,
                    violations=malformed_violations,
                    latency_ms=0.5,
                )

            # 3. Execute AC power flow solve
            converged, latency_ms = PowerFlowSolver.solve_snapshot()

            # 4. Check physics bounds
            violations = self.limits_checker.check_limits(self.circuit, converged)

            # 5. Formulate verdict
            decision = Decision.DECISION_APPROVE if len(violations) == 0 else Decision.DECISION_REJECT

            return self._build_verdict(
                action_id=action.action_id,
                decision=decision,
                violations=violations,
                latency_ms=latency_ms,
            )
        finally:
            # Always leave circuit in baseline state
            self.circuit.reset_to_base()

    def build_rejection_trace(self, action_id: str, violations: List[Violation]) -> RejectionTrace:
        """Calculates normalized severity score and formats feedback for Module 2."""
        if not violations:
            return RejectionTrace(action_id=action_id, violations=[], severity=0.0)

        # Max normalized margin across all violations capped at 1.0
        max_margin = max(abs(v.margin_fraction) for v in violations)
        severity = min(1.0, max(0.0, float(max_margin)))

        return RejectionTrace(
            action_id=action_id,
            violations=violations,
            severity=round(severity, 4),
        )

    def _build_verdict(
        self,
        action_id: str,
        decision: Decision,
        violations: List[Violation],
        latency_ms: float,
    ) -> VerificationVerdict:
        return VerificationVerdict(
            action_id=action_id,
            decision=decision,
            violations=violations,
            solve_latency_ms=round(latency_ms, 3),
            spec_version=self.spec_version,
            network_model_version=self.network_model_version,
        )
