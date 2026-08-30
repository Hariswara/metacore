"""Stand-in for Module 1's SystemContext stream so Module 3 can train before
M1 publishes a real (or even mock) output. Shape mirrors module1.proto
HazardSeverity / SystemContext. Replace sample_scenario with the real M1
adapter once Zayan's contract is live."""
from __future__ import annotations

HAZARD_STAGES = ["normal", "elevated", "severe", "extreme"]

# Vulnerability ranges increase with severity (max_node_vulnerability draw).
_VULN_RANGE = {
    "normal": (0.0, 0.20),
    "elevated": (0.20, 0.45),
    "severe": (0.45, 0.75),
    "extreme": (0.75, 1.0),
}

# How many top-at-risk nodes to name at each stage.
_TOP_COUNT = {
    "normal": 1,
    "elevated": 3,
    "severe": 5,
    "extreme": 8,
}

NODE_POOL = [f"N{i}" for i in range(1, 13)]


def sample_scenario(rng, n_steps: int, cfg: dict | None = None) -> list[dict]:
    """Build a normal→elevated→severe→extreme episode of length n_steps.

    time_to_hazard_onset_min counts down through zero (hazard arrives mid-episode)
    and goes negative once the hazard is underway.
    """
    cfg = cfg or {}
    # Minutes of lead time at t=0; hazard onset at roughly mid-episode.
    onset_at = float(cfg.get("hazard_onset_step_frac", 0.55)) * n_steps
    lead_at_start = float(cfg.get("lead_time_at_start_min", 30.0))
    minutes_per_step = lead_at_start / max(onset_at, 1.0)

    # Stage schedule: equal quarters unless overridden.
    boundaries = cfg.get("stage_boundaries")
    if boundaries is None:
        q = n_steps // 4
        boundaries = [q, 2 * q, 3 * q, n_steps]
    else:
        boundaries = list(boundaries)
        boundaries[-1] = n_steps

    steps: list[dict] = []
    stage_i = 0
    for t in range(n_steps):
        while stage_i < len(HAZARD_STAGES) - 1 and t >= boundaries[stage_i]:
            stage_i += 1
        severity = HAZARD_STAGES[stage_i]
        lo, hi = _VULN_RANGE[severity]
        max_v = float(rng.uniform(lo, hi))
        mean_v = float(rng.uniform(lo, max(lo + 1e-6, max_v)))
        n_top = _TOP_COUNT[severity]
        top = [str(n) for n in rng.choice(NODE_POOL, size=n_top, replace=False)]
        tth = lead_at_start - t * minutes_per_step  # crosses 0 at onset_at
        steps.append(
            {
                "max_node_vulnerability": max_v,
                "mean_node_vulnerability": mean_v,
                "top_at_risk_nodes": top,
                "time_to_hazard_onset_min": float(tth),
                "severity": severity,
            }
        )
    return steps


def severity_index(severity: str) -> int:
    """0..3 index into HAZARD_STAGES (UNSPECIFIED omitted)."""
    try:
        return HAZARD_STAGES.index(severity)
    except ValueError:
        return 0
