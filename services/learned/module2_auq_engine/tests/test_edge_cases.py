"""Garbage in, maximum uncertainty out -- never an exception.

A gate on a control loop must not raise. If M1 hands M2 a truncated vector, a NaN from a
divide-by-zero upstream, or nothing at all, the honest reading is "I cannot see this
state", which is what u' = 1 plus a fired trigger means. Raising instead pushes the
decision onto a caller with no uncertainty machinery of its own, and at worst drops the
control step -- the one outcome the module exists to prevent.

Two layers, deliberately different:

  sanitize.sanitize_state   repairs and reports            (the runtime path)
  sanitize.validate_state   raises ValueError              (the HTTP/caller-bug path)

The repair is not a special case bolted onto the maths: an unusable feature is an
*unobserved* feature, so its quality flag becomes QUALITY_MISSING, f falls out of the mask
as it always does, and u' = K/(f·Σe + K) does the rest.
"""

import numpy as np
import pytest
from sanitize import FILLER, N_FEATURES, sanitize_state, validate_state
from state_contract import QUALITY_MISSING, QUALITY_OBSERVED
from trigger import REASON_BOTH, REASON_SENSING

NAN = float("nan")
INF = float("inf")

# Anything that is not a 28-wide finite numeric vector.
GARBAGE = [
    pytest.param(None, id="none"),
    pytest.param([], id="empty-list"),
    pytest.param(np.array([]), id="empty-array"),
    pytest.param("not a vector", id="string"),
    pytest.param({"p_kw_norm": 0.1}, id="dict"),
    pytest.param([0.1] * 8, id="too-short-the-old-8-feature-mock"),
    pytest.param([0.1] * 40, id="too-long"),
    pytest.param([NAN] * N_FEATURES, id="all-nan"),
    pytest.param([INF] * N_FEATURES, id="all-inf"),
    pytest.param([-INF] + [0.1] * (N_FEATURES - 1), id="one-negative-inf"),
    pytest.param([NAN] * 3 + [0.1] * (N_FEATURES - 3), id="three-nan"),
]


# ---------------------------------------------------------------- sanitize: never raises


@pytest.mark.parametrize("bad", GARBAGE)
def test_sanitize_never_raises(bad):
    clean = sanitize_state(bad)

    assert clean.features.shape == (N_FEATURES,)
    assert np.isfinite(clean.features).all()
    assert len(clean.quality_mask) == N_FEATURES


@pytest.mark.parametrize("bad", GARBAGE)
def test_sanitize_reports_every_repair(bad):
    """`problems` is empty exactly when nothing had to be fixed, so `ok` is the flag a
    caller forces the trigger on."""
    clean = sanitize_state(bad)

    assert not clean.ok
    assert clean.problems
    assert clean.summary() != "clean"


def test_a_clean_state_is_left_alone():
    features = np.linspace(-1.0, 1.0, N_FEATURES)
    clean = sanitize_state(features, [QUALITY_OBSERVED] * N_FEATURES)

    assert clean.ok
    assert clean.problems == []
    assert clean.observed_fraction == pytest.approx(1.0)
    assert clean.features == pytest.approx(features.astype(np.float32))


def test_unusable_features_become_missing_and_lower_f():
    """The mechanism: a NaN is not a weird value, it is an absent measurement."""
    clean = sanitize_state([NAN] * 7 + [0.1] * (N_FEATURES - 7))

    assert clean.quality_mask[:7] == [QUALITY_MISSING] * 7
    assert clean.quality_mask[7:] == [QUALITY_OBSERVED] * (N_FEATURES - 7)
    assert clean.observed_fraction == pytest.approx((N_FEATURES - 7) / N_FEATURES)
    assert clean.features[:7] == pytest.approx([FILLER] * 7)


def test_an_entirely_unreadable_state_drives_f_to_zero():
    """f = 0 gives u' = K/K = 1 with no special case, and sits under any sensible floor."""
    assert sanitize_state([NAN] * N_FEATURES).observed_fraction == 0.0
    assert sanitize_state([]).observed_fraction == 0.0
    assert sanitize_state(None).observed_fraction == 0.0


def test_a_short_vector_is_padded_and_flagged_as_a_contract_violation():
    """Width is a contract problem, not a quality one. Padding keeps the module running,
    but the padded reading is not evidence, so the caller must escalate regardless of f."""
    clean = sanitize_state([0.1] * 8)

    assert clean.contract_violation
    assert clean.quality_mask[8:] == [QUALITY_MISSING] * (N_FEATURES - 8)
    assert clean.observed_fraction == pytest.approx(8 / N_FEATURES)


def test_a_long_vector_is_truncated_and_still_flagged():
    """f stays at 1 here, which is exactly why `ok` and not f is the escalation gate."""
    clean = sanitize_state([0.1] * 40)

    assert clean.contract_violation
    assert not clean.ok
    assert clean.observed_fraction == pytest.approx(1.0)


def test_a_mask_of_nonsense_flags_is_treated_as_missing():
    clean = sanitize_state([0.1] * N_FEATURES, ["NOT_A_QUALITY"] * N_FEATURES)

    assert not clean.ok
    assert clean.quality_mask == [QUALITY_MISSING] * N_FEATURES
    assert clean.observed_fraction == 0.0


def test_a_short_mask_is_padded_with_missing():
    clean = sanitize_state([0.1] * N_FEATURES, [QUALITY_OBSERVED] * 4)

    assert not clean.ok
    assert clean.observed_fraction == pytest.approx(4 / N_FEATURES)


def test_filler_is_not_nan():
    """The filler is fed to the network. NaN would propagate into the evidence itself and
    make u meaningless rather than maximal."""
    assert np.isfinite(FILLER)


# ---------------------------------------------------------------- validate: does raise


@pytest.mark.parametrize("bad", GARBAGE)
def test_validate_raises_on_everything_sanitize_repairs(bad):
    """The strict counterpart. A caller bug should surface as a caller bug."""
    with pytest.raises((ValueError, TypeError)):
        validate_state(bad, [QUALITY_OBSERVED] * N_FEATURES)


def test_validate_accepts_a_clean_state():
    features, flags = validate_state([0.1] * N_FEATURES, [QUALITY_OBSERVED] * N_FEATURES)

    assert features.shape == (N_FEATURES,)
    assert flags == [QUALITY_OBSERVED] * N_FEATURES


def test_validate_rejects_a_bad_mask():
    with pytest.raises(ValueError, match="non-Quality"):
        validate_state([0.1] * N_FEATURES, ["NOT_A_QUALITY"] * N_FEATURES)

    with pytest.raises(ValueError, match="cover all"):
        validate_state([0.1] * N_FEATURES, [QUALITY_OBSERVED] * 3)


# ---------------------------------------------------------------- end to end


@pytest.mark.parametrize("bad", GARBAGE)
def test_the_engine_degrades_instead_of_raising(bad):
    """The claim, end to end through the real head: bad input scores, and it scores as
    maximum uncertainty with the trigger fired."""
    from infer_one import score_state

    result = score_state(bad)

    assert result["u_q"] == pytest.approx(1.0)
    assert result["trigger"] is True
    assert result["reason"] in {REASON_SENSING, REASON_BOTH}
    assert result["input_problems"]
    assert np.isfinite(result["u"])


def test_a_clean_state_through_the_engine_reports_no_problems():
    """The control: the degradation path must not fire on an ordinary state."""
    from infer_one import score_state
    from synthetic_data import sample_states_id

    states, _ = sample_states_id(1, np.random.default_rng(7), blackout_rate=0.0)
    state = states[0]
    result = score_state(state.node_features[0], list(state.quality.per_feature))

    assert result["input_problems"] == []
    assert result["u_q"] < 1.0
    assert result["observed_fraction"] == pytest.approx(12 / 28, abs=1e-4)
