"""Exercises /api/module4 endpoints (presets, verify) directly and end-to-end."""

from __future__ import annotations

import pytest
from gateway.routers.module4 import (
    VerifyRequest,
    get_presets,
    verify_action,
)
from verification.types import Decision


@pytest.mark.anyio
async def test_module4_presets_count() -> None:
    data = await get_presets()
    assert len(data) == 6
    assert "nominal_safe" in data
    assert "cyclone_survival" in data
    assert "peak_power_sharing" in data
    assert "unsafe_undervolt" in data
    assert "unsafe_overvolt" in data
    assert "unsafe_overload" in data


@pytest.mark.anyio
async def test_module4_3_approved_actions() -> None:
    presets = await get_presets()
    for key in ("nominal_safe", "cyclone_survival", "peak_power_sharing"):
        req = VerifyRequest(**presets[key]["payload"])
        res = await verify_action(req)
        assert res.decision == Decision.DECISION_APPROVE
        assert len(res.violations) == 0
        assert "verified safe" in res.causal_log.text


@pytest.mark.anyio
async def test_module4_3_rejected_actions() -> None:
    presets = await get_presets()
    for key in ("unsafe_undervolt", "unsafe_overvolt", "unsafe_overload"):
        req = VerifyRequest(**presets[key]["payload"])
        res = await verify_action(req)
        assert res.decision == Decision.DECISION_REJECT
        assert len(res.violations) > 0
        assert res.rejection_severity > 0.0
        assert len(res.causal_log.grounded_entities) > 0
