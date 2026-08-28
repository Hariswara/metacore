"""Canonical data types and schemas for Module 4 — OpenDSS Physics Verification.

ZERO ML DEPENDENCIES.
These types mirror packages/contracts/proto/verification.proto and module3.proto.
"""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Decision(str, Enum):
    """Firewall verdict decision."""
    DECISION_UNSPECIFIED = "DECISION_UNSPECIFIED"
    DECISION_APPROVE = "DECISION_APPROVE"
    DECISION_REJECT = "DECISION_REJECT"


class ViolationType(str, Enum):
    """Specific physical violation category."""
    VIOLATION_TYPE_UNSPECIFIED = "VIOLATION_TYPE_UNSPECIFIED"
    VIOLATION_TYPE_UNDERVOLTAGE = "VIOLATION_TYPE_UNDERVOLTAGE"
    VIOLATION_TYPE_OVERVOLTAGE = "VIOLATION_TYPE_OVERVOLTAGE"
    VIOLATION_TYPE_THERMAL_OVERLOAD = "VIOLATION_TYPE_THERMAL_OVERLOAD"
    VIOLATION_TYPE_NON_CONVERGENCE = "VIOLATION_TYPE_NON_CONVERGENCE"
    VIOLATION_TYPE_MALFORMED_ACTION = "VIOLATION_TYPE_MALFORMED_ACTION"


class Violation(BaseModel):
    """Structured violation evidence."""
    type: ViolationType
    element_id: str = Field(description="Offending bus, line or transformer ID")
    measured: float = Field(description="Simulated physical value (pu voltage or Amperes)")
    limit: float = Field(description="Statutory or thermal boundary limit")
    margin_fraction: float = Field(description="Signed margin past the limit (e.g. +0.15 = 15% exceedance)")
    attributed_component: str = Field(default="", description="Plausible causing component from ProposedControlAction")


class BreakerCommand(BaseModel):
    """Switch or breaker state command."""
    edge_id: str
    closed: bool


class LoadShedCommand(BaseModel):
    """Emergency load shedding command."""
    node_id: str
    shed_fraction: float = Field(ge=0.0, le=1.0, description="0.0 to 1.0 fraction of load to shed")
    priority_tier: int = Field(default=3, description="1 = most critical, 3 = shed first")


class DispatchSetpoint(BaseModel):
    """Active and reactive power dispatch setpoint for generators or BESS."""
    node_id: str
    p_kw: float = Field(description="Active power in kW")
    q_kvar: float = Field(default=0.0, description="Reactive power in kvar")


class ProcessPath(str, Enum):
    """Origin path of the proposed action."""
    PROCESS_PATH_UNSPECIFIED = "PROCESS_PATH_UNSPECIFIED"
    PROCESS_PATH_SYSTEM1 = "SYSTEM1"
    PROCESS_PATH_SYSTEM2 = "SYSTEM2"


class ProposedControlAction(BaseModel):
    """Action proposed by Module 3 (Meta-Policy)."""
    action_id: str
    origin: Optional[str] = "SYSTEM1"
    breakers: List[BreakerCommand] = Field(default_factory=list)
    load_shed: List[LoadShedCommand] = Field(default_factory=list)
    dispatch: List[DispatchSetpoint] = Field(default_factory=list)
    rationale: str = ""
    schema_version: Optional[str] = "m3-out/0.1"


class VerificationVerdict(BaseModel):
    """Hard synchronous verdict emitted by Module 4."""
    action_id: str
    decision: Decision
    violations: List[Violation] = Field(default_factory=list)
    solve_latency_ms: float = Field(ge=0.0, description="AC power flow solve latency in ms")
    spec_version: str = "safety-spec-v1.0"
    network_model_version: str = "delft-3island-v1"


class RejectionTrace(BaseModel):
    """Feedback payload for Module 2 uncertainty calibration."""
    action_id: str
    violations: List[Violation] = Field(default_factory=list)
    severity: float = Field(ge=0.0, le=1.0, description="Normalized 0..1 severity of worst violation")
