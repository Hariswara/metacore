"""Action Applicator: Translates ProposedControlAction into OpenDSS state changes.

ZERO ML DEPENDENCIES.
"""
from typing import List, Tuple
from ..opendss.circuit import CircuitTwin
from ..types import ProposedControlAction, Violation, ViolationType


class ActionApplicator:
    """Applies breakers, load shedding, and generator dispatch commands to a CircuitTwin."""

    def __init__(self, circuit: CircuitTwin) -> None:
        self.circuit = circuit

    def apply_action(self, action: ProposedControlAction) -> List[Violation]:
        """Applies all commands within the action. Returns violations if any elements are malformed."""
        malformed_violations: List[Violation] = []

        # 1. Apply breaker commands
        for breaker in action.breakers:
            success = self.circuit.set_line_state(breaker.edge_id, breaker.closed)
            if not success:
                malformed_violations.append(
                    Violation(
                        type=ViolationType.VIOLATION_TYPE_MALFORMED_ACTION,
                        element_id=breaker.edge_id,
                        measured=0.0,
                        limit=0.0,
                        margin_fraction=1.0,
                        attributed_component=f"breaker.{breaker.edge_id}",
                    )
                )

        # 2. Apply load shedding commands
        for shed in action.load_shed:
            success = self.circuit.set_load_shed(shed.node_id, shed.shed_fraction)
            if not success:
                malformed_violations.append(
                    Violation(
                        type=ViolationType.VIOLATION_TYPE_MALFORMED_ACTION,
                        element_id=shed.node_id,
                        measured=shed.shed_fraction,
                        limit=1.0,
                        margin_fraction=1.0,
                        attributed_component=f"load_shed.{shed.node_id}",
                    )
                )

        # 3. Apply generator dispatch setpoints
        for disp in action.dispatch:
            success = self.circuit.set_generator_dispatch(disp.node_id, disp.p_kw, disp.q_kvar)
            if not success:
                malformed_violations.append(
                    Violation(
                        type=ViolationType.VIOLATION_TYPE_MALFORMED_ACTION,
                        element_id=disp.node_id,
                        measured=disp.p_kw,
                        limit=0.0,
                        margin_fraction=1.0,
                        attributed_component=f"dispatch.{disp.node_id}",
                    )
                )

        return malformed_violations
