"""Stand-in for Module 1's shared ID/OOD scenario library, so Module 2 can be
built and evaluated BEFORE M1's real state representation exists (see plan NFR4).
Replace sample_* with the real M1 adapter once the M1->M2 contract is live.

sample_id / sample_ood return bare arrays (what run_demo.py uses). The
sample_states_* pair below wraps the same draws in the real M1 -> M2 contract
message, so everything downstream is written against StateRepresentation now."""
import time

import numpy as np
from state_contract import (
    QUALITY_INTERPOLATED,
    QUALITY_MISSING,
    QUALITY_OBSERVED,
    Envelope,
    QualityMask,
    ScenarioRef,
    SchemaVersion,
    StateRepresentation,
)

FEATURES = ["voltage_pu","load_factor","freq_dev_hz","wind_ms",
            "rainfall_mm","solar_wm2","gen_margin","temp_c"]
D = len(FEATURES)

def sample_id(n, rng):
    """Normal-operation island states + a 3-class safety label (safe/stressed/critical)."""
    x = np.zeros((n, D), np.float32)
    x[:,0] = np.clip(rng.normal(1.0,0.02,n),0.94,1.06)   # voltage (pu)
    x[:,1] = rng.uniform(0.3,0.9,n)                       # load factor
    x[:,2] = rng.normal(0,0.05,n)                         # frequency deviation (Hz)
    x[:,3] = rng.uniform(0,12,n)                          # wind (m/s)
    x[:,4] = rng.exponential(2,n)                         # rainfall (mm)
    x[:,5] = rng.uniform(0,900,n)                         # solar (W/m2)
    x[:,6] = rng.uniform(0.1,0.6,n)                       # generation margin
    x[:,7] = rng.normal(28,3,n)                           # temperature (C)
    # physically-motivated risk -> safety class (wind & rain DO matter, so the
    # model learns to use them; otherwise it would ignore the cyclone features)
    risk = (0.32*x[:,1] + 0.26*np.abs(x[:,0]-1.0)*10 + 0.12*np.abs(x[:,2])*10
            + 0.18*(x[:,3]/12) + 0.12*np.clip(x[:,4]/10,0,1))
    q = np.quantile(risk,[0.5,0.83])
    y = np.digitize(risk,q).astype(np.int64)              # 0 safe, 1 stressed, 2 critical
    return x, y

def sample_ood(n, rng):
    """Cyclone / extreme states, far outside the training distribution (unlabelled)."""
    x = np.zeros((n, D), np.float32)
    x[:,0] = np.clip(rng.normal(1.0,0.09,n),0.80,1.20)
    x[:,1] = rng.uniform(0.85,1.15,n)
    x[:,2] = rng.normal(0,0.4,n)
    x[:,3] = rng.uniform(25,45,n)                         # cyclonic wind
    x[:,4] = rng.uniform(40,120,n)                        # extreme rainfall
    x[:,5] = rng.uniform(0,300,n)
    x[:,6] = rng.uniform(-0.1,0.1,n)                      # generation deficit
    x[:,7] = rng.normal(30,4,n)
    return x

class Normalizer:
    def fit(self, x):
        self.mu = x.mean(0)
        self.sd = x.std(0)+1e-6
        return self

    def __call__(self, x):
        return ((x-self.mu)/self.sd).astype(np.float32)


# --- the same draws, wrapped in the M1 -> M2 contract -----------------------

# Envelope.schema_version is SchemaVersion{major, minor}, not a string. M1 has
# not pinned a value yet (the .proto defines the message but nothing stamps it),
# so 0.1 is ours until they do -- see the M1 review.
SCHEMA_VERSION = SchemaVersion(major=0, minor=1)
PRODUCER = "module1"

# TODO: take M1's real embedding_dim. It is not pinned anywhere yet, and the
# mock embedding below is noise -- train on node_features, not node_embedding.
EMBEDDING_DIM = 16

SCENARIO_LIBRARY_VERSION = "mock/0.2"
SCENARIO_ID_ID = "mock-island-normal"
SCENARIO_ID_OOD = "mock-island-cyclone"
SCENARIO_ID_BLACKOUT = "mock-island-blackout"

# Share of states that arrive with a modality missing. M1 has no live feed at all
# (ADR 0004), so this stands in for the replay case where a channel is absent for
# part of an episode.
BLACKOUT_RATE = 0.15

# A blackout drops one to three of the four measured channels, leaving
# observed_fraction at 3/8, 2/8 or 1/8 -- under the 0.4 floor, above nothing.
BLACKOUT_DROP_RANGE = (1, 3)

# Per-feature Quality. ADR 0004 constraint 3: anything not measured must not be
# QUALITY_OBSERVED. The weather channels come from NASA POWER reanalysis and are
# measured; the electrical channels do not exist in the measured record at all
# (no SCADA, no historian) and reach us through M1's downscaling stage, which
# labels every row QUALITY_INTERPOLATED. Mocking a fully-observed state would
# let us build against one that can never occur.
FEATURE_QUALITY = {
    "voltage_pu":   QUALITY_INTERPOLATED,
    "load_factor":  QUALITY_INTERPOLATED,
    "freq_dev_hz":  QUALITY_INTERPOLATED,
    "wind_ms":      QUALITY_OBSERVED,
    "rainfall_mm":  QUALITY_OBSERVED,
    "solar_wm2":    QUALITY_OBSERVED,
    "gen_margin":   QUALITY_INTERPOLATED,
    "temp_c":       QUALITY_OBSERVED,
}


def _blackout_mask(rng):
    """Per-feature Quality with one to three measured channels lost.

    Note what is NOT done here: the feature *values* are left alone. A blackout must
    move the sensing axis and nothing else, or blackout states drift value-OOD too and
    the two axes stop being separable -- which is the whole point of testing them apart.
    The mask carries the provenance; the array carries whatever M1 imputed.
    """
    observed = [f for f in FEATURES if FEATURE_QUALITY[f] == QUALITY_OBSERVED]
    low, high = BLACKOUT_DROP_RANGE
    n_dropped = int(rng.integers(low, high + 1))
    dropped = set(rng.choice(observed, size=n_dropped, replace=False).tolist())
    return QualityMask.from_per_feature(
        [QUALITY_MISSING if f in dropped else FEATURE_QUALITY[f] for f in FEATURES]
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

    Mixed quality by default: most arrive at M1's nominal observed_fraction of 0.5,
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
