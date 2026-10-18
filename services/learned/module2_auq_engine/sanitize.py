"""Make any input scoreable, and say what was wrong with it.

A gate on a control loop must not raise. If M1 hands M2 a truncated vector, a NaN from a
divide-by-zero upstream, or nothing at all, the honest answer is "I cannot see this state"
-- which is exactly what maximum uncertainty plus a fired trigger means. Throwing instead
pushes the decision onto a caller that has no uncertainty machinery of its own, and in the
worst case drops the step entirely, which is the one outcome the module exists to prevent.

The degradation is not a special case bolted onto the maths. An unusable feature is an
*unobserved* feature, so sanitising rewrites its quality flag to QUALITY_MISSING and lets
the sensing axis do the work it already does:

    f = observed / 28  ->  u' = K / (f·Σe + K)

All 28 features unusable gives f = 0, hence u' = K/K = 1 and `observed_fraction` below any
sensible floor, so the trigger fires on the sensing axis with no extra logic. Partial
damage degrades proportionally rather than all at once.

One case the arithmetic cannot cover: a vector of the wrong *width* is a contract
violation, not a quality problem. Padding it keeps the module running, but the padded
reading is not evidence about anything, so `Sanitized.contract_violation` is set and the
caller must fire the trigger regardless of where f landed.

`validate_state` is the strict counterpart, for callers that want the error -- the HTTP
layer rejects a malformed request with a 422 rather than scoring it, because a caller bug
should be visible as a caller bug. Graceful degradation is for the runtime path, where
there is no one to show an error to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from metacore_contracts.state_schema import FEATURE_NAMES
from state_contract import (
    QUALITY_MISSING,
    QUALITY_VALUES,
    QualityMask,
)

N_FEATURES = len(FEATURE_NAMES)

# What a feature is replaced with once its flag is rewritten to QUALITY_MISSING. Zero is
# the same filler M1 uses for its absent electrical block, and the value is inert either
# way: a QUALITY_MISSING feature contributes nothing to f, which is what scales the
# evidence. Not NaN -- that would propagate through the network into the evidence itself.
FILLER = 0.0


@dataclass
class Sanitized:
    """A scoreable state, plus everything that had to be fixed to make it one."""

    features: np.ndarray                      # (N_FEATURES,) float32, all finite
    quality_mask: list[str]                   # N_FEATURES flags
    problems: list[str] = field(default_factory=list)
    contract_violation: bool = False

    @property
    def ok(self) -> bool:
        """True only for a clean, full-width, finite vector with a valid mask.

        Callers force the trigger on `not ok` rather than trusting f alone. A handful of
        NaNs leaves f above the sensing floor, but an upstream divide-by-zero is still a
        reason to escalate -- the floor is calibrated for lost modalities, not for
        arithmetic that went wrong.
        """
        return not self.problems

    @property
    def observed_fraction(self) -> float:
        """Derived from the rewritten mask, so unusable features have already lowered it."""
        return QualityMask.from_per_feature(self.quality_mask).observed_fraction

    def summary(self) -> str:
        return "; ".join(self.problems) if self.problems else "clean"


def _coerce(features) -> tuple[np.ndarray, list[str]]:
    """Anything iterable -> a 1-D float array, or an all-missing row and a problem."""
    problems: list[str] = []
    if features is None:
        return np.array([], dtype=np.float64), ["features was None"]
    try:
        arr = np.asarray(features, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        # Strings, dicts, ragged nesting, objects -- nothing numeric to recover.
        return np.array([], dtype=np.float64), [
            f"features was not numeric ({type(features).__name__})"
        ]
    return arr, problems


def _coerce_mask(quality_mask, n: int) -> tuple[list[str], list[str]]:
    """The caller's flags, or an all-observed default, padded/truncated to n."""
    problems: list[str] = []
    if quality_mask is None:
        # No mask at all: assume the values are as good as they look and let the feature
        # checks below demote whatever is unusable. This is the CLI/test convenience path;
        # a real M1 state always carries one.
        return ["QUALITY_OBSERVED"] * n, problems

    flags = [str(q) for q in quality_mask]
    unknown = sorted(set(flags) - QUALITY_VALUES)
    if unknown:
        problems.append(f"quality_mask had non-Quality values {unknown}, treated as missing")
        flags = [q if q in QUALITY_VALUES else QUALITY_MISSING for q in flags]

    if len(flags) != n:
        problems.append(f"quality_mask covered {len(flags)} of {n} features")
        flags = (flags + [QUALITY_MISSING] * n)[:n]
    return flags, problems


def sanitize_state(features, quality_mask=None) -> Sanitized:
    """Coerce any input into a scoreable 28-feature state, demoting what cannot be used.

    Never raises. The returned `problems` list is empty exactly when the input was already
    a clean, full-width, finite vector.
    """
    arr, problems = _coerce(features)
    flags, mask_problems = _coerce_mask(quality_mask, N_FEATURES)
    problems += mask_problems

    unusable = bool(problems)          # _coerce already said why; don't say it twice
    contract_violation = False
    if arr.size != N_FEATURES:
        contract_violation = True
        if arr.size == 0:
            if not unusable:
                problems.append(f"features was empty, expected {N_FEATURES}")
        else:
            problems.append(f"features had {arr.size} entries, expected {N_FEATURES}")
        # Pad short vectors with missing features; truncate long ones. Either way the
        # reading is not evidence about anything -- contract_violation says so.
        padded = np.full(N_FEATURES, FILLER, dtype=np.float64)
        take = min(arr.size, N_FEATURES)
        padded[:take] = arr[:take]
        for i in range(take, N_FEATURES):
            flags[i] = QUALITY_MISSING
        arr = padded

    nonfinite = ~np.isfinite(arr)
    if nonfinite.any():
        where = [FEATURE_NAMES[i] for i in np.flatnonzero(nonfinite)]
        shown = ", ".join(where[:4]) + (f", +{len(where) - 4} more" if len(where) > 4 else "")
        problems.append(f"{nonfinite.sum()} non-finite feature(s) ({shown}), treated as missing")
        arr = np.where(nonfinite, FILLER, arr)
        for i in np.flatnonzero(nonfinite):
            flags[i] = QUALITY_MISSING

    return Sanitized(arr.astype(np.float32), flags, problems, contract_violation)


def validate_state(features, quality_mask) -> tuple[np.ndarray, list[str]]:
    """Strict counterpart to `sanitize_state`: raise ValueError instead of degrading.

    For the HTTP layer and anywhere a caller bug should surface as a caller bug.
    """
    arr = np.asarray(features, dtype=np.float64).reshape(-1)
    if arr.size != N_FEATURES:
        raise ValueError(
            f"expected {N_FEATURES} features (the pinned contract), got {arr.size}"
        )
    if not np.isfinite(arr).all():
        bad = [FEATURE_NAMES[i] for i in np.flatnonzero(~np.isfinite(arr))]
        raise ValueError(f"features had non-finite values at {bad}")

    flags = [str(q) for q in quality_mask]
    if len(flags) != N_FEATURES:
        raise ValueError(
            f"quality_mask must cover all {N_FEATURES} features, got {len(flags)}"
        )
    unknown = sorted(set(flags) - QUALITY_VALUES)
    if unknown:
        raise ValueError(f"quality_mask has non-Quality values: {unknown}")
    return arr.astype(np.float32), flags
