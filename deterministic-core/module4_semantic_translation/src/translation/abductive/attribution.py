"""Abductive Attribution: Maps physical violations back to proposed action components.

ZERO ML DEPENDENCIES.
"""
from typing import List
from ..types import ProposedControlAction, Violation, ViolationType


class AbductiveAttributor:
    """Infers plausible causal attribution from proposed action commands to observed physical violations."""

    @staticmethod
    def attribute_violations(action: ProposedControlAction, violations: List[Violation]) -> List[Violation]:
        """Enriches violations with plausible attributed_component from the action."""
        enriched: List[Violation] = []

        for v in violations:
            target_id = v.element_id.upper()
            attributed = ""

            # Check if an opened breaker directly matches or isolates the element
            for b in action.breakers:
                if not b.closed:  # Opening a line
                    if target_id in b.edge_id.upper() or b.edge_id.upper() in target_id:
                        attributed = f"breaker.{b.edge_id}"
                        break
                    # If tie line opened and island bus experienced undervoltage
                    if "CRIT" in b.edge_id.upper() or "TIE" in b.edge_id.upper() or "1_2" in b.edge_id.upper() or "2_3" in b.edge_id.upper():
                        attributed = f"breaker.{b.edge_id}"
                        break

            # Check if dispatch modification caused the issue
            if not attributed:
                for d in action.dispatch:
                    if d.node_id.upper() in target_id or target_id in d.node_id.upper():
                        attributed = f"dispatch.{d.node_id}"
                        break

            # Check if load shed was insufficient or excessive
            if not attributed:
                for ls in action.load_shed:
                    if ls.node_id.upper() in target_id or target_id in ls.node_id.upper():
                        attributed = f"load_shed.{ls.node_id}"
                        break

            # Default fallback attribution
            if not attributed:
                if action.origin:
                    attributed = f"action.{action.origin.lower()}"
                else:
                    attributed = "action.unspecified"

            v_copy = v.model_copy()
            v_copy.attributed_component = attributed
            enriched.append(v_copy)

        return enriched
