"""Module 2 develops against the real M1 -> M2 contract, not a bare array.

Two kinds of test. The contract tests pin the shape of what M1 will hand us and
run in milliseconds. The end-to-end test trains the evidential head through
`StateRepresentation` and asserts the thing the module exists to do: epistemic
uncertainty rises on out-of-distribution states.
"""

import numpy as np
import pytest
import torch
from edl import EDLNet, edl_mse_loss, kl_to_uniform, uncertainty
from evaluate import auroc
from state_contract import (
    QUALITY_OBSERVED,
    QUALITY_VALUES,
    Envelope,
    QualityMask,
    ScenarioRef,
    SchemaVersion,
    StateRepresentation,
    stack_features,
)
from synthetic_data import EMBEDDING_DIM, sample_states_id, sample_states_ood

# --------------------------------------------------------------- contract


def test_state_fields():
    rng = np.random.default_rng(0)
    states, y = sample_states_id(4, rng)
    s = states[0]

    assert len(states) == 4 and len(y) == 4
    assert s.node_count == 1
    assert s.node_features.shape == (s.node_count, len(s.feature_names))
    assert s.node_embedding.shape == (s.node_count, s.embedding_dim)
    assert s.embedding_dim == EMBEDDING_DIM
    assert s.graph_embedding.shape == (s.embedding_dim,)


def test_envelope_matches_the_proto_shape():
    """Envelope carries the version and the timestamp -- not StateRepresentation."""
    rng = np.random.default_rng(0)
    states, _ = sample_states_id(1, rng)
    env = states[0].envelope

    assert isinstance(env.schema_version, SchemaVersion)
    assert isinstance(env.schema_version.major, int)
    assert isinstance(env.schema_version.minor, int)
    assert env.producer == "module1"
    assert env.emitted_at > 0
    assert not hasattr(states[0], "schema_version")
    assert not hasattr(states[0], "timestamp")


def test_quality_mask_covers_every_feature_and_is_self_consistent():
    rng = np.random.default_rng(0)
    states, _ = sample_states_id(1, rng)
    s = states[0]
    mask = s.quality

    assert len(mask.per_feature) == len(s.feature_names)
    assert set(mask.per_feature) <= QUALITY_VALUES
    observed = sum(1 for q in mask.per_feature if q == QUALITY_OBSERVED)
    assert mask.observed_fraction == pytest.approx(observed / len(mask.per_feature))


def test_nothing_synthetic_is_labelled_observed():
    """ADR 0004 constraint 3. The electrical channels have no measured record."""
    rng = np.random.default_rng(0)
    states, _ = sample_states_id(1, rng)
    s = states[0]
    quality_of = dict(zip(s.feature_names, s.quality.per_feature, strict=True))

    for constructed in ("voltage_pu", "load_factor", "freq_dev_hz", "gen_margin"):
        assert quality_of[constructed] != QUALITY_OBSERVED
    assert s.quality.observed_fraction < 1.0


def test_ood_label_travels_in_scenario_ref_not_degraded():
    """`degraded` is sensing availability; OOD is ScenarioRef. Keeping them apart
    is what stops the evaluation label leaking into the state."""
    rng = np.random.default_rng(0)
    id_states, _ = sample_states_id(2, rng)
    ood_states, y_ood = sample_states_ood(2, rng)

    assert y_ood is None
    assert all(not s.out_of_distribution for s in id_states)
    assert all(s.out_of_distribution for s in ood_states)
    assert all(not s.degraded for s in id_states + ood_states)
    assert ood_states[0].envelope.scenario.library_version


def test_contract_violations_are_rejected():
    """The mock is only useful if it fails the way the real decoder would."""
    rng = np.random.default_rng(0)
    good = sample_states_id(1, rng)[0][0]

    def build(**overrides):
        kwargs = dict(
            envelope=good.envelope,
            node_count=good.node_count,
            embedding_dim=good.embedding_dim,
            node_embedding=good.node_embedding,
            graph_embedding=good.graph_embedding,
            feature_names=good.feature_names,
            node_features=good.node_features,
            quality=good.quality,
        )
        kwargs.update(overrides)
        return StateRepresentation(**kwargs)

    build()  # the unmodified state must construct

    with pytest.raises(ValueError, match="node_features"):
        build(node_features=good.node_features[:, :-1])
    with pytest.raises(ValueError, match="node_embedding"):
        build(embedding_dim=good.embedding_dim + 1)
    with pytest.raises(ValueError, match="per_feature"):
        build(quality=QualityMask.from_per_feature(good.quality.per_feature[:-1]))
    with pytest.raises(ValueError, match="non-Quality"):
        build(quality=QualityMask.from_per_feature(["MEASURED"] * len(good.feature_names)))


def test_stack_features_selects_the_right_matrix():
    rng = np.random.default_rng(0)
    states, _ = sample_states_id(5, rng)

    assert stack_features(states).shape == (5, len(states[0].feature_names))
    assert stack_features(states, "embedding").shape == (5, EMBEDDING_DIM)
    with pytest.raises(ValueError, match="source must be"):
        stack_features(states, "graph")


def test_hand_built_state_needs_no_synthetic_data():
    """A minimal construction, so the contract stays readable on its own."""
    state = StateRepresentation(
        envelope=Envelope(
            schema_version=SchemaVersion(0, 1),
            emitted_at=1.0,
            scenario=ScenarioRef("ditwah-2025-delft", "mock/0.1", out_of_distribution=True),
        ),
        node_count=2,
        embedding_dim=3,
        node_embedding=np.zeros((2, 3)),
        graph_embedding=np.zeros(3),
        feature_names=["a", "b"],
        node_features=np.zeros((2, 2)),
        quality=QualityMask.from_per_feature([QUALITY_OBSERVED, QUALITY_OBSERVED]),
    )

    assert state.out_of_distribution
    assert state.quality.observed_fraction == 1.0
    assert state.summary()["envelope"]["producer"] == "module1"


# --------------------------------------------------------------- end to end


def test_edl_separates_ood_through_contract():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    id_s, y = sample_states_id(3000, rng)
    te_s, _ = sample_states_id(1000, rng)
    ood_s, _ = sample_states_ood(800, rng)

    Xtr, Xte, Xood = stack_features(id_s), stack_features(te_s), stack_features(ood_s)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6

    def n(a):
        return ((a - mu) / sd).astype(np.float32)

    m = EDLNet(Xtr.shape[1], 3)
    opt = torch.optim.Adam(m.parameters(), 2e-3, weight_decay=1e-5)
    Xt = torch.tensor(n(Xtr))
    yt = torch.tensor(y)
    N = len(Xt)
    for ep in range(300):
        perm = torch.randperm(N)
        for i in range(0, N, 128):
            idx = perm[i:i + 128]
            xb = Xt[idx]
            xo = xb + torch.randn_like(xb) * 4.0
            loss = edl_mse_loss(m(xb), yt[idx], ep, 50) + 0.1 * kl_to_uniform(m(xo) + 1.0).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        u_id, _, _ = uncertainty(m(torch.tensor(n(Xte))))
        u_ood, _, _ = uncertainty(m(torch.tensor(n(Xood))))
    u_id, u_ood = u_id.numpy(), u_ood.numpy()

    assert u_ood.mean() > u_id.mean()
    assert auroc(u_ood, u_id) >= 0.95
