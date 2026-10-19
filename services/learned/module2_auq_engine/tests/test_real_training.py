"""The retrain on M1's real distribution: the loss weighting, the card, and the fix.

The headline claim this file pins is narrow and checkable: a head whose normalisation
card was fitted on M1's real states does not saturate on a real state, and the synthetic
one does. Everything else here guards a way that could silently stop being true.
"""

import m1_real
import numpy as np
import pytest
import torch
import train_real
from edl import EDLNet, edl_mse_loss, uncertainty
from metacore_contracts.state_schema import FEATURE_NAMES

N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------- the weighted loss


def _evidence(n=32, k=3, seed=0):
    torch.manual_seed(seed)
    return torch.rand(n, k)*5.0


def test_class_weight_defaults_to_the_unweighted_loss():
    """The synthetic demo calls this without weights and must be bit-identical."""
    ev = _evidence()
    y = torch.randint(0, 3, (len(ev),))

    plain = edl_mse_loss(ev, y, 10, 50)
    explicit_none = edl_mse_loss(ev, y, 10, 50, None)

    assert torch.equal(plain, explicit_none)


def test_uniform_class_weights_reproduce_the_unweighted_loss():
    """Weights normalised to mean 1 keep the loss on the same scale, so a uniform vector
    must be a no-op -- that is what lets the KL annealing schedule stay untuned."""
    ev = _evidence()
    y = torch.randint(0, 3, (len(ev),))

    plain = edl_mse_loss(ev, y, 10, 50)
    uniform = edl_mse_loss(ev, y, 10, 50, torch.ones(3))

    assert torch.allclose(plain, uniform, atol=1e-6)


def test_class_weight_actually_reweights():
    """A weight vector that favours one class must move the loss."""
    ev = _evidence()
    y = torch.randint(0, 3, (len(ev),))

    plain = edl_mse_loss(ev, y, 10, 50)
    skewed = edl_mse_loss(ev, y, 10, 50, torch.tensor([0.1, 0.1, 2.8]))

    assert not torch.allclose(plain, skewed, atol=1e-4)


# ---------------------------------------------------------------- the card


def test_constant_columns_are_scaled_by_one_not_by_epsilon():
    """Normalizer adds 1e-6 to every sd. For the five electrical features -- constant 0.0
    offline -- that would scale the first real SCADA value to ~1e6 sigma. A constant
    column carries no signal either way, so its scale is pinned to 1."""
    x = np.zeros((100, N_FEATURES), dtype=np.float32)
    x[:, 5] = np.linspace(0.0, 1.0, 100)          # one column that does vary

    nz = train_real.fit_normalizer(x)
    electrical = [FEATURE_NAMES.index(n) for n in m1_real.ELECTRICAL]

    assert np.allclose(nz.sd[electrical], 1.0)
    assert nz.sd[5] > 1e-3                        # the varying column keeps its own scale


def test_the_card_keeps_a_constant_column_at_zero():
    """And the consequence: an offline row still standardises to 0 on those columns."""
    x = np.zeros((50, N_FEATURES), dtype=np.float32)
    x[:, 5] = np.linspace(0.0, 1.0, 50)

    nz = train_real.fit_normalizer(x)
    out = nz(x[:1])
    electrical = [FEATURE_NAMES.index(n) for n in m1_real.ELECTRICAL]

    assert np.allclose(out[0, electrical], 0.0)
    assert np.isfinite(out).all()


def test_a_real_row_through_the_synthetic_card_is_far_out_of_distribution():
    """The root cause, as a number. voltage_pu is ~1.0 in the synthetic draws and 0.0 on
    every real offline row, so the synthetic card puts a real row tens of sigma out --
    well past the 4-sigma proxy the evidence regulariser is trained against. u = 1.0 was
    the correct answer to that row, which is why the card and not the head was the fix."""
    from baselines import OOD_PROXY_SIGMA
    from state_contract import stack_features
    from synthetic_data import Normalizer, sample_states_id

    states, _ = sample_states_id(500, np.random.default_rng(0), blackout_rate=0.0)
    nz = Normalizer().fit(stack_features(states))

    real_row = np.zeros((1, N_FEATURES), dtype=np.float32)
    z = abs(nz(real_row)[0][FEATURE_NAMES.index("voltage_pu")])

    assert z > 10 * OOD_PROXY_SIGMA


# ---------------------------------------------------------------- the fix


@pytest.fixture(scope="module")
def real_head():
    """The trained head, if train_real.py has been run on this checkout."""
    if not train_real.CHECKPOINT.exists():
        pytest.skip(f"no {train_real.CHECKPOINT.name}; run `python train_real.py`")
    ckpt = torch.load(train_real.CHECKPOINT, weights_only=False)
    model = EDLNet(ckpt["n_features"], ckpt["classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _u(model, ckpt, x):
    xn = ((np.asarray(x, dtype=np.float32) - ckpt["mu"]) / ckpt["sd"]).astype(np.float32)
    with torch.no_grad():
        u, _, _ = uncertainty(model(torch.as_tensor(xn)))
    return u.numpy()


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_the_real_head_does_not_saturate_on_real_states(real_head, split):
    """The claim the retrain exists to make.

    "Does not saturate" is a statement about the bulk, not about every row. The head keeps
    a genuine high-uncertainty tail -- roughly 6-10% of rows sit above u = 0.99 -- and that
    tail is present in the TRAINING split too, so it is the head declining to commit on
    hard states rather than an artifact of the test period's shift. The median is the
    honest summary, and against the synthetic head's uniform 1.0000 the contrast is
    unambiguous either way.
    """
    model, ckpt = real_head
    states = m1_real.load().where(split)
    sample = states.x[:: max(1, len(states.x)//1000)]

    u = _u(model, ckpt, sample)

    assert np.median(u) < 0.25, np.median(u)
    assert u.mean() < 0.5, u.mean()
    assert (u > 0.99).mean() < 0.25, (u > 0.99).mean()
    assert np.isfinite(u).all()


def test_the_real_head_still_separates_within_the_real_distribution(real_head):
    """Not saturating is only half of it -- u must still vary with the state. Rows the
    label calls `critical` should carry more uncertainty on average than `safe` ones."""
    model, ckpt = real_head
    states = m1_real.load().where("test")
    step = max(1, len(states.x)//2000)
    x, y = states.x[::step], states.y[::step]

    u = _u(model, ckpt, x)

    assert u[y == 2].mean() > u[y == 0].mean()


def test_the_checkpoint_carries_its_own_disclosure(real_head):
    """The label provenance travels with the artifact, not only with the report."""
    _, ckpt = real_head

    assert "constructed proxy" in ckpt["label_disclosure"]
    assert ckpt["class_cuts_quantile"] == list(m1_real.CLASS_CUTS)
    assert ckpt["class_names"] == list(m1_real.CLASS_NAMES)


def test_the_synthetic_artifact_is_not_overwritten():
    """edl.onnx is what the mock demo serves, and the mock demo is the fallback."""
    assert train_real.CHECKPOINT.parent.name == ".real_cache"

    import export_onnx

    assert export_onnx.REAL_MODEL_PATH.name == "edl_real.onnx"
    assert export_onnx.MODEL_PATH.name == "edl.onnx"
    assert export_onnx.REAL_MODEL_PATH != export_onnx.MODEL_PATH
