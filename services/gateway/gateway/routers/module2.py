"""Module 2 — score one situation against the trained EDL engine.

Shells out to services/learned/module2_auq_engine/infer_one.py, the same script a
developer runs by hand. Mirrors routers/module3.py: a dev-tool endpoint so the dashboard
can show real model output instead of a client-side approximation, not the live gRPC hot
path.

Two request forms, and the first is the real one:

* **contract** — `features` (the pinned 28-feature `node_features` row) plus
  `quality_mask` (the matching per-feature `Quality` flags). The sensing fraction f is
  *derived* from the mask by `QualityMask.from_per_feature`, never taken as an input: the
  contract documents `observed_fraction` as a convenience over `per_feature`, so accepting
  it as a number would let a caller claim a data quality its own mask does not support.
  This is exactly the shape `/api/module1/assemble` emits, so M1 -> M2 composes directly.
* **synthetic** — `novelty` plus `observed_fraction`, the dashboard's two sliders. There is
  no mask behind a manufactured state, so f is taken as given on this path alone.

Exactly one form per request; sending both, or neither, is a 422.

The script prints one JSON object on stdout (module3's scripts write a file instead —
this one is a single small payload, so a pipe is enough).

Requires the process running the gateway to have module2's deps (torch, numpy, pyyaml,
and ideally the `onnx` extra for onnxruntime) importable — true for
`uv run uvicorn gateway.main:app` from the repo root, not for the gateway's slim image.
Without onnxruntime the script falls back to the torch EDL head; the response shape is
identical either way and `backend` says which ran.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

router = APIRouter(prefix="/module2", tags=["module2"])

MODULE2_DIR = Path(__file__).resolve().parents[3] / "learned" / "module2_auq_engine"
# The first call trains the baseline fixture and caches it; later calls are ~1s.
RUN_TIMEOUT_S = 300

# The pinned contract width. Asserted against metacore_contracts.state_schema in the
# router test rather than imported, so the gateway keeps no ML-stack dependency.
N_FEATURES = 28


class Module2RunRequest(BaseModel):
    """Either a real state vector, or the two synthetic knobs."""

    features: list[float] | None = Field(
        None, description=f"the pinned {N_FEATURES}-feature node_features row"
    )
    quality_mask: list[str] | None = Field(
        None, description="one Quality flag per feature, same order"
    )
    novelty: float | None = Field(None, ge=0.0, le=1.0)
    observed_fraction: float | None = Field(None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def exactly_one_form(self) -> Module2RunRequest:
        contract = self.features is not None or self.quality_mask is not None
        synthetic = self.novelty is not None

        if contract and synthetic:
            raise ValueError(
                "send either features+quality_mask or novelty+observed_fraction, not both"
            )
        if not contract and not synthetic:
            raise ValueError("send features+quality_mask, or novelty+observed_fraction")

        if contract:
            if self.features is None or self.quality_mask is None:
                raise ValueError("the contract form needs both features and quality_mask")
            if len(self.features) != N_FEATURES:
                raise ValueError(
                    f"features must have {N_FEATURES} entries, got {len(self.features)}"
                )
            if len(self.quality_mask) != N_FEATURES:
                raise ValueError(
                    f"quality_mask must have {N_FEATURES} entries, got {len(self.quality_mask)}"
                )
        elif self.observed_fraction is None:
            raise ValueError("the synthetic form needs observed_fraction alongside novelty")
        return self


class Module2Baselines(BaseModel):
    softmax: float
    mc_dropout: float
    edl: float


class Module2RunResult(BaseModel):
    u: float
    u_q: float
    evidence: float
    observed_fraction: float
    trigger: bool
    reason: str
    latency_ms: float
    baselines: Module2Baselines
    backend: str
    value_threshold: float


def _argv_and_stdin(req: Module2RunRequest) -> tuple[list[str], str | None]:
    """The infer_one.py invocation for this request form.

    The contract form goes over stdin: a 28-float vector plus 28 flag names is well past
    the comfortable length for a command line, and Windows has a hard limit on it.
    """
    if req.features is not None:
        payload = json.dumps({"features": req.features, "quality_mask": req.quality_mask})
        return ["--state-json", "-"], payload
    return (
        ["--novelty", str(req.novelty), "--observed-fraction", str(req.observed_fraction)],
        None,
    )


@router.post("/run", response_model=Module2RunResult)
def run_module2(req: Module2RunRequest) -> Module2RunResult:
    args, stdin_payload = _argv_and_stdin(req)
    try:
        proc = subprocess.run(
            [sys.executable, "infer_one.py", *args],
            cwd=MODULE2_DIR,
            input=stdin_payload,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Module 2 scoring timed out") from exc

    if proc.returncode != 0:
        detail = proc.stderr.strip()[-4000:] or "Module 2 scoring failed with no stderr output"
        raise HTTPException(status_code=500, detail=detail)

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        detail = proc.stdout.strip()[-4000:] or "Module 2 scoring produced no JSON on stdout"
        raise HTTPException(status_code=500, detail=detail) from exc

    return Module2RunResult(**payload)
