"""Mock M4 verifier — stand-in for deterministic-core until Hariswara's real
verification service is live. Not a control path: neither System 1 nor System 2.
"""
from __future__ import annotations

import hashlib


def _stable_unit(seed_key: tuple) -> float:
    """Process-stable unit interval. Built-in ``hash()`` is salted per process."""
    digest = hashlib.sha256(repr(seed_key).encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "big") % 1000) / 1000.0


def mock_verify(action: dict, obs_raw: dict) -> dict:
    """Stand-in M4: APPROVE / REJECT with structured-ish violations.

    REJECT probability rises when:
    - a critical (tier-1) node is shed, or
    - System 1 is used under extreme severity.
    """
    severity = obs_raw.get("severity", "normal")
    origin = action.get("origin", "SYSTEM1")
    shed_critical = any(
        cmd.get("priority_tier") == 1 and float(cmd.get("shed_fraction", 0)) > 0
        for cmd in action.get("load_shed") or []
    )

    p_reject = 0.02
    if shed_critical:
        p_reject += 0.55
    if origin == "SYSTEM1" and severity == "extreme":
        p_reject += 0.35
    elif origin == "SYSTEM1" and severity == "severe":
        p_reject += 0.15
    if origin == "SYSTEM2":
        p_reject *= 0.4  # deliberation reduces rejection risk

    seed_key = (
        origin,
        severity,
        tuple(
            (c.get("node_id"), round(float(c.get("shed_fraction", 0)), 3))
            for c in (action.get("load_shed") or [])
        ),
        round(float(obs_raw.get("max_node_vulnerability", 0)), 3),
    )
    h = _stable_unit(seed_key)
    if h < p_reject:
        violations = []
        if shed_critical:
            violations.append(
                {
                    "type": "MALFORMED_ACTION",
                    "element_id": "tier1",
                    "measured": 1.0,
                    "limit": 0.0,
                    "margin_fraction": -1.0,
                    "attributed_component": "load_shed",
                }
            )
        if origin == "SYSTEM1" and severity in ("severe", "extreme"):
            violations.append(
                {
                    "type": "UNDERVOLTAGE",
                    "element_id": "N1",
                    "measured": 0.88,
                    "limit": 0.95,
                    "margin_fraction": -0.07,
                    "attributed_component": "dispatch",
                }
            )
        if not violations:
            violations.append(
                {
                    "type": "THERMAL_OVERLOAD",
                    "element_id": "E1",
                    "measured": 1.15,
                    "limit": 1.0,
                    "margin_fraction": -0.15,
                    "attributed_component": "load_shed",
                }
            )
        return {"decision": "REJECT", "violations": violations}
    return {"decision": "APPROVE", "violations": []}
