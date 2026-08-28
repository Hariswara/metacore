"""The real Eluvaitivu degradation, at both resolutions.

These tests pin the honest findings and nothing else. In particular there is deliberately
NO assertion that the island-aggregate decay-vs-nominal AUROC is high: it is ~0.85, and
the same-season control shows that figure is October rather than the collapse. Asserting
it would lock in an artefact.

What is asserted:

  A  the seasonal control lands near chance -- the negative result is real, not a
     training accident, and the naive figure is meaningfully above it
  B  uncertainty rises on the flagged plant-months at the per-plant monthly resolution,
     where the event is actually observable

Reduced scale so the lane stays bounded; the committed numbers come from
`python eluvaitivu_decay.py`, which runs it at full scale.
"""

import numpy as np
import pytest
import real_data as rd
from eluvaitivu_decay import experiment_a, experiment_b

CHANCE_BAND = (0.35, 0.65)


@pytest.fixture(scope="module")
def aggregate():
    """Experiment A at reduced scale, without the full-cycle stage. ~6s."""
    return experiment_a(quick=True, with_full_cycle=False)


@pytest.fixture(scope="module")
def per_plant():
    """Experiment B. The whole ledger is 120 rows, so this is fast at full scale."""
    return experiment_b(epochs=250)


# ------------------------------------- A: the negative result, and its control

def test_seasonal_control_lands_near_chance(aggregate):
    """The load-bearing check. Against the same three calendar months a year earlier --
    same season, no decay -- the separation disappears."""
    seasonal = aggregate["auroc_vs_seasonal_control"]

    assert CHANCE_BAND[0] <= seasonal <= CHANCE_BAND[1], seasonal


def test_the_naive_comparison_is_inflated_by_season(aggregate):
    """And this is why the control is not optional: without it the same model reports a
    number that reads as detection."""
    naive = aggregate["auroc_naive_decay_vs_nominal"]
    seasonal = aggregate["auroc_vs_seasonal_control"]

    assert naive > seasonal + 0.10, (naive, seasonal)
    assert aggregate["verdict"] == "not detectable"


def test_decay_hours_are_not_more_uncertain_than_the_same_season(aggregate):
    """Stated as means rather than as a ranking, because it is the blunter fact: the
    decay quarter is not elevated against the same quarter without a decay."""
    means = aggregate["mean_u"]

    assert means["decay_2025_q4"] < means["seasonal_control_2024_q4"] + 0.10


# ------------------------------- B: recovery where the event is observable

def test_uncertainty_rises_on_the_flagged_plant_months(per_plant):
    """The positive result: unsupervised, the OOD label never entering training."""
    means = per_plant["mean_u"]

    assert means["decay_2025_q4"] > means["nominal_all"], means
    assert per_plant["verdict"] == "recovered"


def test_most_of_the_window_clears_the_nominal_band(per_plant):
    """At least two of the three flagged months clear the 95th percentile of nominal.

    Deliberately not "which two". With one episode, three decay months and 117 nominal
    plant-months, the per-month detail moves with training length (2-3 of 3 across 200-400
    epochs) while the mean separation does not. Asserting a particular month would be
    pinning noise.
    """
    flagged = per_plant["flagged_at_95th_percentile_of_nominal"]

    assert sum(flagged.values()) >= 2, flagged
    assert per_plant["auroc_decay_vs_nominal_heldout"] >= 0.80, per_plant


def test_recovery_survives_dropping_the_rule_s_own_input(per_plant):
    """The circularity check. M1 derived the window by thresholding plant-relative energy,
    so `energy_rel` is the rule's own input. Removing it leaves only the fuel signature --
    how much diesel the plant burned per kWh -- and the separation survives."""
    reduced = experiment_b(epochs=250, columns=rd.PLANT_FEATURES_NO_ENERGY)

    assert "energy_rel" not in reduced["features"]
    assert reduced["mean_u"]["decay_2025_q4"] > reduced["mean_u"]["nominal_all"], reduced
    # Weaker than with the energy feature, and that gap is the honest measure of how much
    # the full result owes to reading the rule's own input back. Still well above chance.
    assert reduced["auroc_decay_vs_nominal_heldout"] >= 0.55, reduced


# --------------------------------------------------- the real data underneath

def test_the_label_is_the_real_scenario_ref():
    """Not a mock id: the contract-carried label M1 published."""
    _, decay = rd.load_library()
    ref = rd.scenario_ref(decay)

    assert ref.scenario_id == "eluvaitivu-hybrid-decay-2025q4"
    assert ref.library_version == "1.0.0"
    assert ref.out_of_distribution is True
    assert (decay["start_month"], decay["end_month"]) == ("2025-10", "2025-12")


def test_feature_accounting_partitions_the_contract():
    """The honest accounting: every one of the 28 pinned features is real, a static site
    constant, or absent -- and the absent ones are the electrical block the pin already
    marks QUALITY_MISSING."""
    buckets = rd.REAL_FEATURES + rd.STATIC_FEATURES + rd.ABSENT_FEATURES

    assert sorted(buckets) == sorted(rd.FEATURE_NAMES)
    assert len(buckets) == 28
    assert all(rd.FEATURE_QUALITY[f] == "QUALITY_MISSING" for f in rd.ABSENT_FEATURES)


def test_monthly_label_broadcasts_to_every_hour():
    """The window is monthly and the evaluation is hourly, so each hour inherits the flag.
    Oct + Nov + Dec 2025 = 744 + 720 + 744."""
    load, weather = rd.read_load(), rd.read_weather()
    scaler, ramps = rd.SiteScaler(load, weather), rd.load_ramps(load)
    _, decay = rd.load_library()

    states, stamps = rd.build_states(list(rd.DECAY_MONTHS), rd.scenario_ref(decay),
                                     scaler, load, weather, ramps)

    assert len(states) == 744 + 720 + 744
    assert all(s.out_of_distribution for s in states)
    assert {s[:7] for s in stamps} == set(rd.DECAY_MONTHS)
    assert np.isclose(states[0].quality.observed_fraction, 12/28)
