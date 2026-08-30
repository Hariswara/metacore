"""Abductive Attribution: Maps physical violations back to proposed action components.

ZERO ML DEPENDENCIES.
"""

from ..types import ProposedControlAction, Violation


class AbductiveAttributor:
    """Infers causal attribution from action commands to observed violations."""

    @staticmethod
    def attribute_violations(
        action: ProposedControlAction, violations: list[Violation]
    ) -> list[Violation]:
        """Enriches violations with plausible attributed_component from the action."""
        enriched: list[Violation] = []

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
                    tie_keywords = ("CRIT", "TIE", "1_2", "2_3")
                    if any(kw in b.edge_id.upper() for kw in tie_keywords):
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
