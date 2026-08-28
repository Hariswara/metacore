"""Stand-in for Module 1's shared ID/OOD scenario library, so Module 2 can be
built and evaluated BEFORE M1's real producer exists (see plan NFR4).

Pinned to the real contract: the feature names, their order, the embedding width and
the per-feature quality all come from `metacore_contracts.state_schema`, not from
constants here. When M1 bumps the pin, this mock moves with it or the tests fail --
which is the point. Replace sample_states_* with M1's producer and nothing downstream
of `stack_features` changes.

sample_id / sample_ood return bare arrays. The sample_states_* trio wraps the same
draws in the real M1 -> M2 contract message.
"""
import time

import numpy as np
from metacore_contracts.state_schema import (
    EMBEDDING_DIM,
    FEATURE_NAMES,
    SCHEMA_VERSION,
    calibration_quality,
    feature_index,
)
from state_contract import (
    QUALITY_MISSING,
    QUALITY_OBSERVED,
    Envelope,
    QualityMask,
    ScenarioRef,
    StateRepresentation,
)

FEATURES = list(FEATURE_NAMES)
D = len(FEATURES)

# Column indices, resolved through the contract so a reorder upstream cannot silently
# shuffle what this file thinks it is writing.
IDX = {name: feature_index(name) for name in FEATURES}

ASSET_TYPES = ("is_bus", "is_pv", "is_wind", "is_bess", "is_diesel", "is_load")
ASSET_MIX = (0.24, 0.14, 0.10, 0.10, 0.14, 0.28)

# Quality is not hand-written: it is read off the pin. 12 QUALITY_OBSERVED (the temporal
# and static-topology groups), 11 QUALITY_INTERPOLATED (resource, meteorology, demand)
# and 5 QUALITY_MISSING (the electrical group -- no SCADA exists, ADR 0004). Nominal
# observed_fraction is therefore 12/28 = 0.4286, and that is what the sensing floor in
# config.yaml is calibrated against.
FEATURE_QUALITY = {name: calibration_quality(name) for name in FEATURES}
OBSERVED_FEATURES = tuple(f for f in FEATURES if FEATURE_QUALITY[f] == QUALITY_OBSERVED)
NOMINAL_OBSERVED_FRACTION = len(OBSERVED_FEATURES)/D

# The temporal block. A comms blackout that takes the clock source out drops all four at
# once, which is the modality-loss case the sensing axis exists to catch.
TEMPORAL_FEATURES = ("hour_sin", "hour_cos", "doy_sin", "doy_cos")


def _cyclic(value, period):
    angle = 2.0*np.pi*value/period
    return np.sin(angle), np.cos(angle)


def _common(x, n, rng):
    """Temporal and static-topology columns. Identical in distribution for ID and OOD --
    a cyclone does not change what time it is or what an asset is plugged into."""
    hour = rng.integers(0, 24, n)
    doy = rng.integers(1, 366, n)
    x[:, IDX["hour_sin"]], x[:, IDX["hour_cos"]] = _cyclic(hour, 24)
    x[:, IDX["doy_sin"]], x[:, IDX["doy_cos"]] = _cyclic(doy, 365)

    kind = rng.choice(len(ASSET_TYPES), n, p=ASSET_MIX)
    for i, name in enumerate(ASSET_TYPES):
        x[:, IDX[name]] = (kind == i).astype(np.float32)
    x[:, IDX["nominal_kv_norm"]] = rng.choice([0.4/11.0, 1.0], n, p=[0.75, 0.25])
    x[:, IDX["critical_load"]] = (rng.random(n) < 0.12).astype(np.float32)
    return hour, kind


def sample_id(n, rng):
    """Normal-operation island states + a 3-class safety label (safe/stressed/critical)."""
    x = np.zeros((n, D), np.float32)
    hour, kind = _common(x, n, rng)

    daylight = np.clip(np.sin(np.pi*(hour - 6)/12), 0, None)
    clearsky = np.clip(rng.beta(5, 2, n), 0, 1)
    x[:, IDX["ghi_wh_m2_norm"]] = daylight*clearsky
    x[:, IDX["clearsky_index"]] = clearsky
    x[:, IDX["wind_10m_ms_norm"]] = np.clip(rng.beta(2, 5, n), 0, 1)
    x[:, IDX["wind_50m_ms_norm"]] = np.clip(x[:, IDX["wind_10m_ms_norm"]]*1.25, 0, 1)
    x[:, IDX["pv_available_kw_norm"]] = x[:, IDX["ghi_wh_m2_norm"]]*(kind == 1)

    x[:, IDX["temp_2m_c_norm"]] = rng.normal(0, 1, n)
    x[:, IDX["humidity_2m_pct_norm"]] = np.clip(rng.normal(0.8, 0.07, n), 0, 1)
    x[:, IDX["precip_mm_hr_norm"]] = np.clip(rng.exponential(0.04, n), 0, 1)
    x[:, IDX["pressure_kpa_norm"]] = rng.normal(0, 1, n)

    # Demand: the evening peak the load-downscaling stage constructs, plus a warm-day term.
    diurnal = 0.45 + 0.35*np.clip(np.sin(np.pi*(hour - 5)/19), 0, None) \
        + 0.35*np.exp(-0.5*((hour - 20)/2.2)**2)
    load = np.clip(diurnal*(1 + 0.05*x[:, IDX["temp_2m_c_norm"]]), 0.1, 1.0)
    x[:, IDX["load_kw_norm"]] = load
    x[:, IDX["load_ramp_kw_per_h_norm"]] = rng.normal(0, 0.06, n)

    # Electrical: QUALITY_MISSING in the calibration artifacts, filled by the simulator.
    x[:, IDX["p_kw_norm"]] = np.clip(load*rng.normal(1.0, 0.05, n), 0, 1.2)
    x[:, IDX["q_kvar_norm"]] = x[:, IDX["p_kw_norm"]]*rng.normal(0.35, 0.06, n)
    x[:, IDX["voltage_pu"]] = np.clip(rng.normal(1.0, 0.02, n), 0.94, 1.06)
    x[:, IDX["soc_fraction"]] = np.clip(rng.beta(5, 2, n), 0, 1)*(kind == 3)
    x[:, IDX["asset_online"]] = (rng.random(n) > 0.02).astype(np.float32)

    return x, _risk_class(x)


def sample_ood(n, rng):
    """Cyclone / extreme states, far outside the training distribution (unlabelled)."""
    x = np.zeros((n, D), np.float32)
    hour, kind = _common(x, n, rng)

    # Storm: overcast, cyclonic wind, torrential rain, collapsing surface pressure.
    daylight = np.clip(np.sin(np.pi*(hour - 6)/12), 0, None)
    clearsky = np.clip(rng.beta(1.5, 6, n)*0.4, 0, 1)
    x[:, IDX["ghi_wh_m2_norm"]] = daylight*clearsky
    x[:, IDX["clearsky_index"]] = clearsky
    x[:, IDX["wind_10m_ms_norm"]] = np.clip(rng.uniform(1.8, 3.4, n), 0, None)
    x[:, IDX["wind_50m_ms_norm"]] = x[:, IDX["wind_10m_ms_norm"]]*1.3
    x[:, IDX["pv_available_kw_norm"]] = x[:, IDX["ghi_wh_m2_norm"]]*(kind == 1)

    x[:, IDX["temp_2m_c_norm"]] = rng.normal(0.8, 1.6, n)
    x[:, IDX["humidity_2m_pct_norm"]] = np.clip(rng.normal(0.97, 0.02, n), 0, 1)
    x[:, IDX["precip_mm_hr_norm"]] = rng.uniform(2.5, 6.0, n)
    x[:, IDX["pressure_kpa_norm"]] = rng.normal(-4.5, 0.8, n)

    load = np.clip(rng.uniform(0.9, 1.35, n), 0, None)
    x[:, IDX["load_kw_norm"]] = load
    x[:, IDX["load_ramp_kw_per_h_norm"]] = rng.normal(0, 0.45, n)

    x[:, IDX["p_kw_norm"]] = np.clip(load*rng.normal(1.05, 0.2, n), 0, None)
    x[:, IDX["q_kvar_norm"]] = x[:, IDX["p_kw_norm"]]*rng.normal(0.6, 0.2, n)
    x[:, IDX["voltage_pu"]] = np.clip(rng.normal(1.0, 0.09, n), 0.78, 1.22)
    x[:, IDX["soc_fraction"]] = np.clip(rng.beta(1.2, 6, n), 0, 1)*(kind == 3)
    x[:, IDX["asset_online"]] = (rng.random(n) > 0.35).astype(np.float32)

    return x


def _risk_class(x):
    """3-class safety label, driven by the electrical and demand groups -- with wind and
    rain given real weight so the net has to attend to the cyclone dimensions rather than
    learning to ignore them."""
    risk = (0.30*x[:, IDX["load_kw_norm"]]
            + 0.24*np.abs(x[:, IDX["voltage_pu"]] - 1.0)*10
            + 0.14*x[:, IDX["q_kvar_norm"]]
            + 0.10*(1.0 - x[:, IDX["asset_online"]])
            + 0.14*x[:, IDX["wind_10m_ms_norm"]]
            + 0.08*np.clip(x[:, IDX["precip_mm_hr_norm"]]/0.2, 0, 1))
    edges = np.quantile(risk, [0.5, 0.83])
    return np.digitize(risk, edges).astype(np.int64)


class Normalizer:
    def fit(self, x):
        self.mu = x.mean(0)
        self.sd = x.std(0)+1e-6
        return self

    def __call__(self, x):
        return ((x-self.mu)/self.sd).astype(np.float32)


# --- the same draws, wrapped in the M1 -> M2 contract -----------------------

PRODUCER = "module1"

SCENARIO_LIBRARY_VERSION = "mock/0.3"
SCENARIO_ID_ID = "mock-island-normal"
SCENARIO_ID_OOD = "mock-island-cyclone"
SCENARIO_ID_BLACKOUT = "mock-island-blackout"

# Share of states that arrive with a modality missing. M1 has no live feed at all
# (ADR 0004), so this stands in for the replay case where a channel is absent for
# part of an episode.
BLACKOUT_RATE = 0.15


def _blackout_mask(rng):
    """Per-feature Quality with a modality lost.

    The temporal block goes first: it is four of the twelve QUALITY_OBSERVED features and
    losing it is the clean modality-loss case (12/28 -> 8/28 = 0.286, under the floor).
    Some blackouts also take topology channels, which is the shallower case.

    Note what is NOT done here: the feature *values* are left alone. A blackout must move
    the sensing axis and nothing else, or blackout states drift value-OOD too and the two
    axes stop being separable -- which is the whole point of testing them apart.
    """
    lost = set(TEMPORAL_FEATURES)
    extra = tuple(f for f in OBSERVED_FEATURES if f not in lost)
    n_extra = int(rng.integers(0, 3))
    if n_extra:
        lost.update(rng.choice(extra, size=n_extra, replace=False).tolist())
    return QualityMask.from_per_feature(
        [QUALITY_MISSING if f in lost else FEATURE_QUALITY[f] for f in FEATURES]
    )


def _make_state(x_row, rng, scenario_id, out_of_distribution, blackout=False):
    """One island node, one timestep, as M1 will hand it to us."""
    node_features = np.asarray(x_row, np.float32).reshape(1, -1)
    node_embedding = rng.standard_normal((1, EMBEDDING_DIM)).astype(np.float32)
    envelope = Envelope(
        schema_version=SCHEMA_VERSION,
        emitted_at=time.time(),
        producer=PRODUCER,
        scenario=ScenarioRef(
            SCENARIO_ID_BLACKOUT if blackout else scenario_id,
            SCENARIO_LIBRARY_VERSION,
            out_of_distribution,
        ),
    )
    quality = (_blackout_mask(rng) if blackout
               else QualityMask.from_per_feature([FEATURE_QUALITY[f] for f in FEATURES]))
    return StateRepresentation(
        envelope=envelope,
        node_count=1,
        embedding_dim=EMBEDDING_DIM,
        node_embedding=node_embedding,
        graph_embedding=node_embedding.mean(0),
        feature_names=list(FEATURES),
        node_features=node_features,
        quality=quality,
        # This is exactly what `degraded` is for: a modality is absent. It stays False
        # for a cyclone, which is fully observed -- that is distribution shift, and it
        # lives in scenario.out_of_distribution.
        degraded=blackout,
    )


def _blackout_flags(n, rng, blackout_rate):
    if blackout_rate <= 0:
        return np.zeros(n, dtype=bool)
    return rng.random(n) < blackout_rate


def sample_states_id(n, rng, blackout_rate=BLACKOUT_RATE):
    """n in-distribution states + their 3-class safety labels.

    Mixed quality by default: most arrive at M1's nominal observed_fraction of 0.4286,
    a minority with a modality missing. Pass blackout_rate=0.0 for a clean
    nominal-quality population.
    """
    x, y = sample_id(n, rng)
    flags = _blackout_flags(n, rng, blackout_rate)
    states = [_make_state(x[i], rng, SCENARIO_ID_ID, False, blackout=bool(flags[i]))
              for i in range(n)]
    return states, y


def sample_states_ood(n, rng, blackout_rate=0.0):
    """n out-of-distribution (cyclone) states. Unlabelled, so the label slot is None.

    Nominal quality by default, so the value axis is measured without the sensing axis
    on top of it. Raise blackout_rate to build the compound case (a cyclone that also
    takes the comms out), which is the one that should report reason "both".
    """
    x = sample_ood(n, rng)
    flags = _blackout_flags(n, rng, blackout_rate)
    states = [_make_state(x[i], rng, SCENARIO_ID_OOD, True, blackout=bool(flags[i]))
              for i in range(n)]
    return states, None


def sample_states_blackout(n, rng):
    """n comms-blackout states: ordinary in-distribution VALUES, low observed_fraction.

    The sensing axis on its own. These are not out of distribution -- nothing about the
    grid is unusual, we just cannot see it -- so out_of_distribution stays False and a
    value-only trigger is expected to miss them.
    """
    x, y = sample_id(n, rng)
    states = [_make_state(x[i], rng, SCENARIO_ID_ID, False, blackout=True) for i in range(n)]
    return states, y
