"""Template Causal Logger: Synthesizes grounded natural language explanations from violation evidence.

ZERO ML DEPENDENCIES.
"""
from typing import List
from ..types import CausalLog, Decision, Generator, VerificationVerdict, Violation, ViolationType


class TemplateCausalLogger:
    """Generates grounded causal diagnostic text from verification results."""

    @staticmethod
    def generate_log(verdict: VerificationVerdict) -> CausalLog:
        """Constructs a grounded causal log from a VerificationVerdict."""
        if verdict.decision == Decision.DECISION_APPROVE or not verdict.violations:
            return CausalLog(
                action_id=verdict.action_id,
                text="Action verified safe: All bus voltages and line ampacities remain within statutory bounds (power flow solved in {:.1f} ms).".format(
                    verdict.solve_latency_ms
                ),
                grounded_entities=[],
                generator=Generator.GENERATOR_TEMPLATE,
            )

        # Build explanation phrases for each violation
        phrases: List[str] = []
        grounded_entities: List[str] = []

        for v in verdict.violations:
            element = v.element_id
            if element not in grounded_entities and element != "GLOBAL_CIRCUIT":
                grounded_entities.append(element)

            cause_str = f" [Attributed to: {v.attributed_component}]" if v.attributed_component else ""

            if v.type == ViolationType.VIOLATION_TYPE_UNDERVOLTAGE:
                margin_pct = abs(v.margin_fraction) * 100.0
                phrases.append(
                    f"Undervoltage on Bus {element} ({v.measured:.4f} pu < limit {v.limit:.4f} pu, margin: -{margin_pct:.1f}%){cause_str}"
                )
            elif v.type == ViolationType.VIOLATION_TYPE_OVERVOLTAGE:
                margin_pct = abs(v.margin_fraction) * 100.0
                phrases.append(
                    f"Overvoltage on Bus {element} ({v.measured:.4f} pu > limit {v.limit:.4f} pu, margin: +{margin_pct:.1f}%){cause_str}"
                )
            elif v.type == ViolationType.VIOLATION_TYPE_THERMAL_OVERLOAD:
                margin_pct = abs(v.margin_fraction) * 100.0
                phrases.append(
                    f"Thermal overload on Line {element} ({v.measured:.1f} A > ampacity {v.limit:.1f} A, margin: +{margin_pct:.1f}%){cause_str}"
                )
            elif v.type == ViolationType.VIOLATION_TYPE_NON_CONVERGENCE:
                phrases.append("AC power flow solver failed to converge under proposed topological switching.")
            elif v.type == ViolationType.VIOLATION_TYPE_MALFORMED_ACTION:
                phrases.append(f"Malformed action: Element {element} does not exist in target grid topology.")
            else:
                phrases.append(f"Unspecified violation on element {element}.")

        summary_text = "Action REJECTED: " + "; ".join(phrases) + f" (power flow checked in {verdict.solve_latency_ms:.1f} ms)."

        return CausalLog(
            action_id=verdict.action_id,
            text=summary_text,
            grounded_entities=grounded_entities,
            generator=Generator.GENERATOR_TEMPLATE,
        )
