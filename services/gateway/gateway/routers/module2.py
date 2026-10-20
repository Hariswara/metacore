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

Two heads, and `head` selects between them:

* **synthetic** (the default) -- the head trained on synthetic_data, served by
  infer_one.py. This is what run_demo.py and the dashboard have always shown and what
  the PP1 fallback demo rests on, so it stays the default and its behaviour is unchanged.
* **real** -- the head trained on M1's real state distribution, served by serve_real.py.
  Only meaningful on the contract form: scoring a manufactured novelty-slider state with
  a head fitted on real data would produce a number with no interpretation, so that
  combination is a 422 rather than a quiet answer.

The two heads are fitted on different distributions against different targets, so their
numbers are NOT comparable -- a synthetic u of 0.05 and a real u of 0.05 do not mean the
same thing. Every response therefore names the head that produced it, and the real head
carries its label disclosure, so the two cannot be confused on a slide.

edl_real.onnx is derived and git-ignored, so its absence is an ordinary state on a fresh
clone rather than a failure: serve_real.py exits 2 and this endpoint answers 503 with the
command that builds it, the way routers/module1.py does for the ST-GNN checkpoint.

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
from typing import Literal

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
    head: Literal["synthetic", "real"] = Field(
        "synthetic", description="which trained head serves this request"
    )

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

        if self.head == "real" and not contract:
            raise ValueError(
                "head='real' needs a real state (features + quality_mask). The novelty "
                "slider manufactures a state, and a head fitted on M1's real distribution "
                "has no interpretation for one"
            )
        return self


class Module2Baselines(BaseModel):
    softmax: float
    mc_dropout: float
    edl: float


class Module2RunResult(BaseModel):
    # Which head produced these numbers. Not decoration: the two are fitted on different
    # distributions against different targets and their values are not comparable.
    head: Literal["synthetic", "real"] = "synthetic"
    artifact: str = "edl.onnx"
    # Set only by the real head: its label is discretised from M1's constructed
    # vulnerability proxy, not an observed safety class.
    label_disclosure: str | None = None
    u: float
    u_q: float
    evidence: float
    observed_fraction: float
    trigger: bool
    reason: str
    latency_ms: float
    # None on the real head: the baselines are fitted on the SYNTHETIC
    # distribution, so reporting them beside a real-head u would invite exactly
    # the comparison the head field exists to prevent.
    baselines: Module2Baselines | None = None
    backend: str
    value_threshold: float
    # Empty when the input was a clean, full-width, finite vector. Non-empty means the
    # engine repaired something and forced the trigger -- see module2/sanitize.py.
    input_problems: list[str] = []


SYNTHETIC_SCRIPT = "infer_one.py"
REAL_SCRIPT = "serve_real.py"
# serve_real.py exits 2 when edl_real.onnx has not been built. Same convention as
# routers/module1.py uses for the ST-GNN checkpoint.
ARTIFACT_MISSING = 2


def _invocation(req: Module2RunRequest) -> tuple[str, list[str], str | None]:
    """(script, argv, stdin) for this request.

    The contract form goes over stdin either way: a 28-float vector plus 28 flag names is
    well past the comfortable length for a command line, and Windows caps it outright.
    """
    if req.features is not None:
        payload = json.dumps({"features": req.features, "quality_mask": req.quality_mask})
        script = REAL_SCRIPT if req.head == "real" else SYNTHETIC_SCRIPT
        return script, ["--state-json", "-"], payload
    # Novelty form: synthetic only -- the validator rejects head="real" here.
    return (
        SYNTHETIC_SCRIPT,
        ["--novelty", str(req.novelty), "--observed-fraction", str(req.observed_fraction)],
        None,
    )


def _run(script: str, args: list[str], stdin_payload: str | None) -> dict:
    """Run one of the two scoring scripts and return its JSON, or raise an HTTPException."""
    try:
        proc = subprocess.run(
            [sys.executable, script, *args],
            cwd=MODULE2_DIR,
            input=stdin_payload,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Module 2 scoring timed out") from exc

    if proc.returncode == ARTIFACT_MISSING:
        # Derived, git-ignored artifact that has not been built yet. A 503 with the
        # command that fixes it, never a 500: nothing is broken.
        raise HTTPException(status_code=503, detail=proc.stderr.strip()[-4000:])
    if proc.returncode != 0:
        detail = proc.stderr.strip()[-4000:] or "Module 2 scoring failed with no stderr output"
        raise HTTPException(status_code=500, detail=detail)

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        detail = proc.stdout.strip()[-4000:] or "Module 2 scoring produced no JSON on stdout"
        raise HTTPException(status_code=500, detail=detail) from exc


@router.post("/run", response_model=Module2RunResult)
def run_module2(req: Module2RunRequest) -> Module2RunResult:
    script, args, stdin_payload = _invocation(req)
    payload = _run(script, args, stdin_payload)
    # The synthetic script predates the head field and does not set it; defaulting here
    # rather than there keeps infer_one.py untouched.
    payload.setdefault("head", req.head)
    payload.setdefault("artifact", "edl.onnx")
    return Module2RunResult(**payload)


class Module2WarmResult(BaseModel):
    head: str
    artifact: str
    ready: bool
    warm_ms: float
    value_threshold: float
    label_disclosure: str = ""


@router.post("/warm", response_model=Module2WarmResult)
def warm_real_head() -> Module2WarmResult:
    """Prime the real head so the first toggle in a live demo is not the call that pays.

    Only the real head has one, and it is cheap by construction rather than by caching:
    its threshold is calibrated at train time and carried in the model card, so there is
    no fixture to train and no calibration pass to run -- this is a session create plus
    one forward pass, well under a second.

    The synthetic head needs no equivalent endpoint: it warms itself on its first scoring
    call and caches to .dashboard_cache/, which is the behaviour the fallback demo has
    always had and which this change deliberately leaves alone.
    """
    return Module2WarmResult(**_run(REAL_SCRIPT, ["--warm"], None))
