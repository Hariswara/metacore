"""Module 4 — Physical Verification & Causal Translation Gateway Router.

Provides real-time REST API endpoints for OpenDSS AC power-flow verification,
physics limits evaluation (voltage and thermal ampacity), abductive attribution,
and grounded operator explanation logs.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from translation.abductive.attribution import AbductiveAttributor
from translation.templates.causal_logger import TemplateCausalLogger
from translation.types import (
    BreakerCommand as TransBreakerCommand,
)
from translation.types import (
    DispatchSetpoint as TransDispatchSetpoint,
)
from translation.types import (
    LoadShedCommand as TransLoadShedCommand,
)
from translation.types import (
    ProposedControlAction as TransProposedAction,
)
from verification.firewall.verifier import PhysicsVerifier
from verification.opendss.circuit import load_pin
from verification.powerflow.solver import PowerFlowSolver
from verification.types import (
    BreakerCommand,
    Decision,
    DispatchSetpoint,
    LoadShedCommand,
    ProposedControlAction,
    ViolationType,
)

router = APIRouter(prefix="/module4", tags=["module4"])

_verifier: PhysicsVerifier | None = None


def _get_verifier() -> PhysicsVerifier:
    global _verifier
    if _verifier is None:
        _verifier = PhysicsVerifier()
    return _verifier


class VerifyRequest(BaseModel):
    action_id: str = Field(default="act-web-001", description="Action unique identifier")
    origin: str = Field(default="SYSTEM1", description="SYSTEM1 or SYSTEM2")
    rationale: str = Field(default="", description="Operator or AI rationale")
    breakers: list[BreakerCommand] = Field(default_factory=list)
    load_shed: list[LoadShedCommand] = Field(default_factory=list)
    dispatch: list[DispatchSetpoint] = Field(default_factory=list)


class ViolationDTO(BaseModel):
    type: ViolationType
    element_id: str
    limit: float
    measured: float
    margin_fraction: float
    attributed_component: str = ""


class CausalLogDTO(BaseModel):
    action_id: str
    text: str
    grounded_entities: list[str]
    generator: str


class GridBusState(BaseModel):
    bus_name: str
    island: str
    voltage_pu: float
    status: str  # "SAFE", "UNDERVOLTAGE", "OVERVOLTAGE"


class GridLineState(BaseModel):
    line_name: str
    current_amps: float
    norm_amps: float
    margin_fraction: float
    is_closed: bool
    status: str  # "NORMAL", "OVERLOAD", "TRIPPED"


class VerifyResponse(BaseModel):
    action_id: str
    decision: Decision
    solve_latency_ms: float
    violations: list[ViolationDTO]
    rejection_severity: float
    causal_log: CausalLogDTO
    buses: list[GridBusState]
    lines: list[GridLineState]


PRESETS: dict[str, dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # 3 APPROVED (SAFE) ACTIONS -- northern-islands-v1 bus and generator names
    # -------------------------------------------------------------------------
    "nominal_safe": {
        "title": "1. Nominal S1 Feeder Shed (Approved)",
        "description": "Routine minor load shed on Nainativu's non-critical feeder group F2.",
        "payload": {
            "action_id": "act-001-nominal-safe",
            "origin": "SYSTEM1",
            "rationale": "S1 reactive shed vuln=0.14",
            "breakers": [],
            "load_shed": [{"node_id": "NAI_F2", "shed_fraction": 0.1117, "priority_tier": 3}],
            "dispatch": [],
        },
    },
    "cyclone_survival": {
        "title": "2. Cyclone Ditwah Survival Dispatch (Approved)",
        "description": (
            "System 2 starts Eluvaitivu's backup genset and sheds non-critical feeders."
        ),
        "payload": {
            "action_id": "act-002-cyclone-survival",
            "origin": "SYSTEM2",
            "rationale": "S2 survival opt vuln=0.63 protect-tier1",
            "breakers": [],
            "load_shed": [
                {"node_id": "ELU_F2", "shed_fraction": 0.25, "priority_tier": 3},
                {"node_id": "DEL_F2", "shed_fraction": 0.0983, "priority_tier": 3},
            ],
            "dispatch": [{"node_id": "ELU_HYB_DG", "p_kw": 20.0, "q_kvar": 10.0}],
        },
    },
    "peak_power_sharing": {
        "title": "3. Peak Standby Genset Dispatch (Approved)",
        "description": (
            "System 2 brings Nainativu's and Delft's standby gensets on during the evening peak."
        ),
        "payload": {
            "action_id": "act-003-peak-sharing",
            "origin": "SYSTEM2",
            "rationale": "S2 survival opt vuln=0.88 protect-tier1",
            "breakers": [],
            "load_shed": [{"node_id": "DEL_F2", "shed_fraction": 0.2325, "priority_tier": 3}],
            "dispatch": [
                {"node_id": "NAI_DG2", "p_kw": 120.0, "q_kvar": 40.0},
                {"node_id": "DEL_DG2", "p_kw": 100.0, "q_kvar": 30.0},
            ],
        },
    },
    # -------------------------------------------------------------------------
    # 3 REJECTED (UNSAFE) ACTIONS
    # -------------------------------------------------------------------------
    "unsafe_undervolt": {
        "title": "4. Battery Fast-Charge Voltage Collapse (Rejected)",
        "description": (
            "Charging Eluvaitivu's battery at full rating pulls 72 kW down 0.7 km of 400 V "
            "cable, dragging the hybrid bus below 0.95 pu."
        ),
        "payload": {
            "action_id": "act-004-unsafe-undervolt",
            "origin": "SYSTEM2",
            "rationale": "Aggressive battery recharge before the storm",
            "breakers": [],
            "load_shed": [],
            "dispatch": [{"node_id": "ELU_HYB_BESS", "p_kw": -72.0, "q_kvar": 0.0}],
        },
    },
    "unsafe_overvolt": {
        "title": "5. Battery Export Over-Voltage (Rejected)",
        "description": (
            "Exporting the battery at full rating into a lightly loaded feeder lifts the hybrid "
            "bus above 1.05 pu."
        ),
        "payload": {
            "action_id": "act-005-unsafe-overvolt",
            "origin": "SYSTEM2",
            "rationale": "Uncompensated battery export",
            "breakers": [],
            "load_shed": [],
            "dispatch": [{"node_id": "ELU_HYB_BESS", "p_kw": 72.0, "q_kvar": 0.0}],
        },
    },
    "unsafe_overload": {
        "title": "6. Hybrid Link Thermal Overload (Rejected)",
        "description": "Running every hybrid unit at once exceeds the 125 A hybrid link rating.",
        "payload": {
            "action_id": "act-006-unsafe-overload",
            "origin": "SYSTEM2",
            "rationale": "Hybrid plant flat out",
            "breakers": [],
            "load_shed": [],
            "dispatch": [{"node_id": "ELU_HYB", "p_kw": 140.0, "q_kvar": 0.0}],
        },
    },
}

_BUS_ISLAND: dict[str, str] = {b["node_id"].upper(): b["island"] for b in load_pin()["buses"]}


def _classify_bus_island(bus_name: str) -> str:
    """The island a bus belongs to, from the northern-islands-v1 topology pin."""
    return _BUS_ISLAND.get(bus_name.upper(), "Unknown")


@router.get("/presets")
async def get_presets() -> dict[str, Any]:
    """Returns the 6 curated representative control actions (3 safe, 3 unsafe)."""
    return PRESETS


@router.post("/verify", response_model=VerifyResponse)
async def verify_action(req: VerifyRequest) -> VerifyResponse:
    """Evaluates a proposed action against OpenDSS physics and emits grounded causal log."""
    try:
        verifier = _get_verifier()
        action_verif = ProposedControlAction(
            action_id=req.action_id,
            origin=req.origin,
            breakers=req.breakers,
            load_shed=req.load_shed,
            dispatch=req.dispatch,
            rationale=req.rationale,
        )

        action_trans = TransProposedAction(
            action_id=req.action_id,
            origin=req.origin,
            breakers=[TransBreakerCommand(**b.model_dump()) for b in req.breakers],
            load_shed=[TransLoadShedCommand(**ls.model_dump()) for ls in req.load_shed],
            dispatch=[TransDispatchSetpoint(**d.model_dump()) for d in req.dispatch],
            rationale=req.rationale,
        )

        # 1. Physical Simulation & Limit Evaluation
        verifier.circuit.reset_to_base()
        malformed = verifier.applicator.apply_action(action_verif)

        if malformed:
            violations = malformed
            decision = Decision.DECISION_REJECT
            solve_latency_ms = 0.0
            bus_voltages = verifier.circuit.get_bus_voltages_pu()
            line_loadings = verifier.circuit.get_line_loadings()
        else:
            converged, solve_latency_ms = PowerFlowSolver.solve_snapshot()
            violations = verifier.limits_checker.check_limits(verifier.circuit, converged)
            decision = (
                Decision.DECISION_APPROVE
                if len(violations) == 0
                else Decision.DECISION_REJECT
            )
            # Capture snapshot electrical state WHILE solver solution is live in memory
            bus_voltages = verifier.circuit.get_bus_voltages_pu()
            line_loadings = verifier.circuit.get_line_loadings()

        # Reset circuit to clean base state after capturing solved state
        verifier.circuit.reset_to_base()

        # 2. Abductive Attribution
        if violations:
            violations = AbductiveAttributor.attribute_violations(action_trans, violations)

        verdict = verifier._build_verdict(
            action_id=action_verif.action_id,
            decision=decision,
            violations=violations,
            latency_ms=solve_latency_ms,
        )

        # 3. Severity feedback trace
        trace = verifier.build_rejection_trace(verdict.action_id, verdict.violations)

        # 4. Grounded Causal Log
        causal_log = TemplateCausalLogger.generate_log(verdict, include_latency=True)

        # 5. Build DTOs for snapshot grid state
        buses_dto: list[GridBusState] = []
        for bus, v_pu in sorted(bus_voltages.items()):
            if v_pu < 0.95:
                status = "UNDERVOLTAGE"
            elif v_pu > 1.05:
                status = "OVERVOLTAGE"
            else:
                status = "SAFE"
            buses_dto.append(
                GridBusState(
                    bus_name=bus,
                    island=_classify_bus_island(bus),
                    voltage_pu=round(v_pu, 4),
                    status=status,
                )
            )

        lines_dto: list[GridLineState] = []
        for line, data in sorted(line_loadings.items()):
            max_amps = data["max_amps"]
            norm_amps = data["norm_amps"]
            margin = data["margin_fraction"]
            is_closed = data["enabled"] > 0.5
            if not is_closed:
                st = "TRIPPED"
            elif margin > 0.0:
                st = "OVERLOAD"
            else:
                st = "NORMAL"
            lines_dto.append(
                GridLineState(
                    line_name=line,
                    current_amps=round(max_amps, 2),
                    norm_amps=round(norm_amps, 2),
                    margin_fraction=round(margin, 4),
                    is_closed=is_closed,
                    status=st,
                )
            )

        return VerifyResponse(
            action_id=verdict.action_id,
            decision=verdict.decision,
            solve_latency_ms=round(verdict.solve_latency_ms, 3),
            violations=[ViolationDTO(**v.model_dump()) for v in verdict.violations],
            rejection_severity=round(trace.severity, 4),
            causal_log=CausalLogDTO(
                action_id=causal_log.action_id,
                text=causal_log.text,
                grounded_entities=causal_log.grounded_entities,
                generator=causal_log.generator.value,
            ),
            buses=buses_dto,
            lines=lines_dto,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}") from exc
