"""The real-M1 loader: contract shape, label provenance, and the leakage guards.

The label here is discretised from M1's constructed vulnerability proxy, so most of what
is worth pinning is *not* a number -- it is that the cuts come from the training split
only, that the proxy's own history feature never reaches M2's inputs, and that the column
mapping from M1's model features to the 28-feature contract is checked rather than assumed.
"""

import m1_real
import numpy as np
import pytest
from metacore_contracts.state_schema import FEATURE_NAMES

N_FEATURES = len(FEATURE_NAMES)


@pytest.fixture(scope="module")
def states():
    return m1_real.load()


def test_rows_are_contract_shaped(states):
    assert states.x.shape[1] == N_FEATURES
    assert states.x.dtype == np.float32
    assert len(states.x) == len(states.y) == len(states.v) == len(states.split)


def test_every_row_is_finite(states):
    """The head is fed these directly; a NaN here would propagate into the evidence."""
    assert np.isfinite(states.x).all()


def test_the_electrical_block_is_zero_filled(states):
    """The offline path has no power-flow simulator, so these five are QUALITY_MISSING and
    exactly 0.0 -- which is precisely the distribution the card must be refitted on."""
    idx = [FEATURE_NAMES.index(n) for n in m1_real.ELECTRICAL]
    assert (states.x[:, idx] == 0.0).all()
    assert [states.quality_mask[i] for i in idx] == ["QUALITY_MISSING"] * 5


def test_the_offline_mask_matches_m1(states):
    counts = {q: states.quality_mask.count(q) for q in set(states.quality_mask)}
    assert counts["QUALITY_OBSERVED"] == 12
    assert counts["QUALITY_INTERPOLATED"] == 11
    assert counts["QUALITY_MISSING"] == 5
    assert states.observed_fraction == pytest.approx(12 / 28, abs=1e-4)


def test_the_vulnerability_history_feature_never_reaches_the_inputs():
    """M1 feeds its own model a `vulnerability` history column -- the label itself. Routing
    through the 28-feature contract is what drops it; this is the leakage guard."""
    from module1.train.dataset import FEATURES

    assert "vulnerability" in FEATURES
    assert "vulnerability" not in FEATURE_NAMES
    assert "fragility" not in FEATURE_NAMES


def test_the_column_mapping_is_asserted_not_assumed():
    """A reordering upstream must fail loudly rather than scramble columns silently."""
    scrambled = np.zeros((2, 3, 25), dtype=np.float32)
    import module1.train.dataset as ds

    original = ds.FEATURES
    try:
        ds.FEATURES = ("not_a_feature",) + original[1:]
        with pytest.raises(RuntimeError, match="no longer matches the contract"):
            m1_real._contract_rows(scrambled)
    finally:
        ds.FEATURES = original


def test_class_cuts_come_from_the_training_split_only(states):
    """Fitting the cuts on all rows would let the test period set its own label scale."""
    train = states.where("train")
    lo, hi = states.cuts

    assert lo == pytest.approx(np.quantile(train.v, m1_real.CLASS_CUTS[0]), abs=1e-9)
    assert hi == pytest.approx(np.quantile(train.v, m1_real.CLASS_CUTS[1]), abs=1e-9)


def test_training_balance_is_what_the_cuts_ask_for(states):
    """q0.70/q0.95 on train, so train must land on 70/25/5 by construction."""
    balance = states.where("train").class_balance

    assert balance["safe"] == pytest.approx(0.70, abs=0.01)
    assert balance["stressed"] == pytest.approx(0.25, abs=0.01)
    assert balance["critical"] == pytest.approx(0.05, abs=0.01)


def test_the_test_split_is_genuinely_shifted(states):
    """The test period is the Eluvaitivu decay quarter. If `critical` were not over-
    represented there relative to train, the split would not be testing anything."""
    train = states.where("train").class_balance
    test = states.where("test").class_balance

    assert test["critical"] > 2 * train["critical"]


def test_labels_agree_with_the_continuous_proxy(states):
    """y is a discretisation of v and nothing else."""
    lo, hi = states.cuts
    assert (states.y[states.v < lo] == 0).all()
    assert (states.y[(states.v >= lo) & (states.v < hi)] == 1).all()
    assert (states.y[states.v >= hi] == 2).all()


def test_class_weights_lift_the_sparse_class(states):
    w = m1_real.class_weights(states.where("train").y)

    assert w.shape == (3,)
    assert w[2] > w[1] > w[0]              # critical is rarest, so weighted hardest
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-5)


def test_the_disclosures_are_present_and_say_what_they_must():
    """These strings are carried into every metrics artifact; they are the reason a
    reader cannot mistake the accuracy for a claim about real asset failures."""
    assert "constructed proxy" in m1_real.PROXY_DISCLOSURE
    assert "TRAINING split only" in m1_real.PROXY_DISCLOSURE
    assert "n = 1" in m1_real.OOD_DISCLOSURE
    assert "7.1x" in m1_real.OOD_DISCLOSURE


def test_splits_partition_the_record(states):
    sizes = {s: len(states.where(s).x) for s in ("train", "val", "test")}

    assert sum(sizes.values()) == len(states.x)
    assert sizes["train"] == 131280          # 13128 hours x 10 nodes
    assert sizes["val"] == sizes["test"] == 22080
