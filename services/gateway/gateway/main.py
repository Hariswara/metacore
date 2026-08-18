"""FastAPI gateway — the single external API surface.

Every route the dashboard calls lands here. Module services are never reached directly from the
browser, so the gateway is the one place authentication, rate limiting and schema versioning get
applied when they arrive.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import CORS_ORIGINS, GENERATION_CSV
from .routers import calibration

app = FastAPI(
    title="MetaCore gateway",
    version="0.0.0",
    summary="External API surface for the MetaCore research dashboard",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(calibration.router, prefix="/api")


@app.get("/api/health")
def health() -> dict[str, object]:
    """Readiness plus what the gateway can actually serve right now.

    `calibration_available` is false on a clean clone with no CEB data — that is a supported
    state, not an error, so the dashboard can say so plainly instead of failing to load.
    """
    return {
        "ready": True,
        "service": "gateway",
        "calibration_available": GENERATION_CSV.exists(),
        "calibration_path": str(GENERATION_CSV),
    }
