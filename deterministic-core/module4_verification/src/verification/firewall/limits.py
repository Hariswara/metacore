"""Physics Limits: Evaluates simulated voltages, thermal currents, and convergence.

ZERO ML DEPENDENCIES.
"""
from typing import Dict, List, Optional
from ..opendss.circuit import CircuitTwin
from ..types import Violation, ViolationType


class SafetyLimitsConfig:
    """Configurable boundaries for statutory voltage and thermal ratings."""

    def __init__(
        self,
        vmin_pu: float = 0.95,
        vmax_pu: float = 1.05,
        thermal_overload_margin: float = 0.0,
    ) -> None:
        self.vmin_pu = vmin_pu
        self.vmax_pu = vmax_pu
        self.thermal_overload_margin = thermal_overload_margin


class PhysicsLimitsChecker:
    """Checks circuit state against statutory power system safety limits."""

    def __init__(self, config: Optional[SafetyLimitsConfig] = None) -> None:
        self.config = config or SafetyLimitsConfig()

    def check_limits(self, circuit: CircuitTwin, converged: bool) -> List[Violation]:
        """Evaluates all physical boundary limits across the circuit."""
        violations: List[Violation] = []

        # 1. Non-convergence is a hard physical violation
        if not converged:
            violations.append(
                Violation(
                    type=ViolationType.VIOLATION_TYPE_NON_CONVERGENCE,
                    element_id="GLOBAL_CIRCUIT",
                    measured=0.0,
                    limit=1.0,
                    margin_fraction=1.0,
                    attributed_component="power_flow_solver",
                )
            )
            return violations

        # 2. Voltage limit checks
        bus_voltages = circuit.get_bus_voltages_pu()
        for bus_id, v_pu in bus_voltages.items():
            if bus_id.upper() == "SOURCEBUS":
                continue  # Skip ideal slack bus

            if v_pu < self.config.vmin_pu:
                margin = (self.config.vmin_pu - v_pu) / self.config.vmin_pu
                violations.append(
                    Violation(
                        type=ViolationType.VIOLATION_TYPE_UNDERVOLTAGE,
                        element_id=bus_id,
                        measured=round(float(v_pu), 4),
                        limit=self.config.vmin_pu,
                        margin_fraction=round(float(margin), 4),
                        attributed_component="",
                    )
                )
            elif v_pu > self.config.vmax_pu:
                margin = (v_pu - self.config.vmax_pu) / self.config.vmax_pu
                violations.append(
                    Violation(
                        type=ViolationType.VIOLATION_TYPE_OVERVOLTAGE,
                        element_id=bus_id,
                        measured=round(float(v_pu), 4),
                        limit=self.config.vmax_pu,
                        margin_fraction=round(float(margin), 4),
                        attributed_component="",
                    )
                )

        # 3. Thermal line loading checks
        line_loadings = circuit.get_line_loadings()
        for line_id, data in line_loadings.items():
            if data["enabled"] <= 0.5:
                continue  # De-energized / open lines carry no current

            max_amps = data["max_amps"]
            norm_amps = data["norm_amps"]
            margin = data["margin_fraction"]

            if norm_amps > 0.0 and max_amps > norm_amps * (1.0 + self.config.thermal_overload_margin):
                violations.append(
                    Violation(
                        type=ViolationType.VIOLATION_TYPE_THERMAL_OVERLOAD,
                        element_id=line_id,
                        measured=round(float(max_amps), 2),
                        limit=round(float(norm_amps), 2),
                        margin_fraction=round(float(margin), 4),
                        attributed_component="",
                    )
                )

        return violations
