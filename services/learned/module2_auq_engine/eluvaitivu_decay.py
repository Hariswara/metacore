"""Real-data evaluation of the Eluvaitivu hybrid-plant degradation, 2025 Q4.

    python eluvaitivu_decay.py      # writes results/eluvaitivu_decay.json

Two experiments on the same real episode, at two resolutions, because the answer is
different at each and the difference is the finding:

  A  island-aggregate hourly   the degradation is NOT detectable. A naive comparison
                               against the nominal window scores ~0.85 AUROC, and a
                               same-season control collapses it to chance. The 0.85 is
                               October, not the collapse.

  B  per-plant monthly         the degradation IS recovered, unsupervised. Uncertainty
                               rises on all three flagged months without the label ever
                               entering training.

Together: **M2's uncertainty recovers the real Eluvaitivu degradation at the per-plant
monthly resolution where it is observable, and does not detect it in island-aggregate
hourly telemetry** -- empirical support for the project's per-asset-over-aggregate
premise, measured rather than argued.

Why A fails is not a defect in A. `island_load_hourly.csv` sums the Eluvaitivu diesel set
and hybrid plant into one island demand -- correct for dispatch and power flow, and it
removes the event. M1's `module1/data/scenarios.py` says so directly: across the window
the plant falls 73.4% while island demand falls 10.3%, a 7.1x attenuation. The raw
monthly island totals show it plainly: 2025 Q4 demand is *higher* than 2024 Q4 while the
plant behind it stopped.

n = 1. This is ONE degradation episode. Hour-level and month-level AUROCs have many
points, but the number of independent events is one, so this is a case study and not a
distribution over failure modes. Both statements belong in anything published from it.
"""

import json
from pathlib import Path

import numpy as np
import real_data as rd
import torch
from edl import EDLNet, edl_mse_loss, kl_to_uniform, uncertainty
from evaluate import aupr, auroc
from state_contract import stack_features
from synthetic_data import Normalizer

RESULTS = Path(__file__).with_name("results") / "eluvaitivu_decay.json"

SEED = 0
LR, WEIGHT_DECAY, BATCH = 2e-3, 1e-5, 128
KL_ANNEAL, OOD_SIGMA, OOD_REG = 50, 4.0, 0.1


def _train(x, y, epochs, seed=SEED, hidden=32):
    """EDL with the OOD-aware regulariser. The label `y` is a constructed operating class
    (see `_operating_class`); the OOD label never enters training in either experiment."""
    torch.manual_seed(seed)
    xt, yt = torch.tensor(x), torch.tensor(y)
    m = EDLNet(xt.shape[1], int(yt.max().item()) + 1, hidden=hidden)
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WEIGHT_DECAY)
    for ep in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), BATCH):
            idx = perm[i:i + BATCH]
            xb = xt[idx]
            xo = xb + torch.randn_like(xb)*OOD_SIGMA
            loss = (edl_mse_loss(m(xb), yt[idx], ep, KL_ANNEAL)
                    + OOD_REG*kl_to_uniform(m(xo) + 1.0).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


def _u(model, x):
    with torch.no_grad():
        u, _, _ = uncertainty(model(torch.tensor(x)))
    return u.numpy()


def _operating_class(x, columns, weights):
    """A 3-class operating label from the training window's own quantiles.

    The safety class does not exist in the record, so the EDL's classification task is
    constructed. That is fine and is not what is being measured: the OOD signal is the
    Dirichlet evidence, and the label only gives the head something to be evidential
    about. Quantiles come from the training rows, so no test row informs it.
    """
    idx = {name: i for i, name in enumerate(columns)}
    risk = sum(w*x[:, idx[name]] for name, w in weights.items() if name in idx)
    return np.digitize(risk, np.quantile(risk, [0.5, 0.83])).astype(np.int64)


# ------------------------------------------------- A: island-aggregate hourly

def experiment_a(epochs=200, quick=False, with_full_cycle=True):
    """The negative result, with the control that makes it a result rather than a miss."""
    load, weather = rd.read_load(), rd.read_weather()
    scaler, ramps = rd.SiteScaler(load, weather), rd.load_ramps(load)
    library, decay_entry = rd.load_library()
    nominal_entry = next(s for s in library["scenarios"]
                         if s["island"] == rd.ISLAND and not s["out_of_distribution"])

    def states(months, entry):
        return rd.build_states(months, rd.scenario_ref(entry), scaler,
                               load, weather, ramps)[0]

    nominal_months = rd.months_in(nominal_entry["start_month"], nominal_entry["end_month"])
    train_states = states(nominal_months[:2], nominal_entry)
    heldout = states(nominal_months[2:], nominal_entry)
    decay = states(list(rd.DECAY_MONTHS), decay_entry)
    control = states(["2024-10", "2024-11", "2024-12"], nominal_entry)

    weights = {"load_kw_norm": 0.6, "wind_10m_ms_norm": 0.25, "precip_mm_hr_norm": 0.15}
    xtr = stack_features(train_states)
    nz = Normalizer().fit(xtr)
    model = _train(nz(xtr), _operating_class(xtr, list(rd.FEATURE_NAMES), weights),
                   epochs=60 if quick else epochs)

    def mean_u(group):
        return float(_u(model, nz(stack_features(group))).mean())

    u_held = _u(model, nz(stack_features(heldout)))
    u_decay = _u(model, nz(stack_features(decay)))
    u_control = _u(model, nz(stack_features(control)))

    naive = float(auroc(u_decay, u_held))
    seasonal = float(auroc(u_decay, u_control))

    result = {
        "resolution": "island-aggregate hourly",
        "verdict": "not detectable",
        "train_window": f"{nominal_months[0]}..{nominal_months[1]}",
        "hours": {"train": len(train_states), "heldout_nominal": len(heldout),
                  "decay": len(decay), "seasonal_control": len(control)},
        "mean_u": {"heldout_nominal_2024_05": float(u_held.mean()),
                   "seasonal_control_2024_q4": float(u_control.mean()),
                   "decay_2025_q4": float(u_decay.mean())},
        "auroc_naive_decay_vs_nominal": naive,
        "auroc_vs_seasonal_control": seasonal,
        "aupr_vs_seasonal_control": float(aupr(u_decay, u_control)),
        "per_month_mean_u": {
            **{m: mean_u(states([m], nominal_entry)) for m in
               ("2024-10", "2024-11", "2024-12")},
            **{m: mean_u(states([m], decay_entry)) for m in rd.DECAY_MONTHS},
        },
        "interpretation": (
            f"The naive comparison scores {naive:.3f}, which reads as detection. Against "
            f"the same three calendar months a year earlier -- same season, no decay -- it "
            f"is {seasonal:.3f}, i.e. chance. The naive figure is the season."
        ),
    }
    if with_full_cycle:
        result.update(_full_cycle(states, nominal_entry, weights, quick, epochs))
    return result


def _full_cycle(states, nominal_entry, weights, quick, epochs):
    """Train on a whole seasonal cycle (2024) and walk u through 2025 month by month.

    The fairest form of A: a model that has seen every season cannot mistake October for
    novelty, so anything left is the event. Nothing is left.
    """
    train = states([f"2024-{m:02d}" for m in range(1, 13)], nominal_entry)
    xtr = stack_features(train)
    nz = Normalizer().fit(xtr)
    model = _train(nz(xtr), _operating_class(xtr, list(rd.FEATURE_NAMES), weights),
                   epochs=50 if quick else epochs)

    monthly = {}
    for month in [f"2025-{m:02d}" for m in range(1, 13)]:
        group = states([month], nominal_entry)
        monthly[month] = float(_u(model, nz(stack_features(group))).mean())

    reference = [u for m, u in monthly.items() if m not in rd.DECAY_MONTHS]
    flagged = [monthly[m] for m in rd.DECAY_MONTHS]
    return {
        "full_cycle": {
            "train": "2024-01..2024-12 (a complete seasonal cycle, no decay)",
            "monthly_mean_u_2025": monthly,
            "mean_u_2025_q1_q3": float(np.mean(reference)),
            "mean_u_2025_q4": float(np.mean(flagged)),
            "note": ("Flat across all twelve months. The decay quarter is not elevated, "
                     "and island demand in 2025 Q4 is higher than in 2024 Q4 while the "
                     "plant behind it had all but stopped."),
        }
    }


# ---------------------------------------------------- B: per-plant monthly

def experiment_b(epochs=400, columns=rd.PLANT_FEATURES, seed=SEED):
    """The positive result: unsupervised recovery of the flagged window.

    RECOVERS, not discovers. M1 derived the window by a stated threshold rule over these
    same monthly numbers, so an unsupervised method agreeing with it is evidence that the
    signal is real and strong -- not an independent discovery. The variant with
    `energy_rel` removed is the check on how much of that agreement is the method reading
    the rule's own input back.
    """
    rows = rd.read_plant_months()
    nominal, decay = rd.split_plant_months(rows)
    baselines = rd.plant_baselines(nominal)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(nominal))
    cut = int(0.75*len(nominal))
    train_rows = [nominal[i] for i in order[:cut]]
    heldout_rows = [nominal[i] for i in order[cut:]]

    xtr = rd.plant_month_matrix(train_rows, baselines, columns)
    nz = Normalizer().fit(xtr)
    weights = {"energy_rel": -0.5, "sfc_l_per_kwh": 0.3, "fuel_cost_rs_per_kwh": 0.2,
               "diesel_rel": 0.2}
    model = _train(nz(xtr), _operating_class(xtr, list(columns), weights),
                   epochs=epochs, seed=seed, hidden=16)

    u_train = _u(model, nz(xtr))
    u_held = _u(model, nz(rd.plant_month_matrix(heldout_rows, baselines, columns)))
    u_decay = _u(model, nz(rd.plant_month_matrix(decay, baselines, columns)))
    u_nominal_all = np.concatenate([u_train, u_held])

    per_month = {row["month"]: float(u) for row, u in zip(decay, u_decay, strict=True)}
    threshold = float(np.quantile(u_nominal_all, 0.95))
    return {
        "resolution": "per-plant monthly",
        "verdict": "recovered",
        "features": list(columns),
        "plant_months": {"nominal_train": len(train_rows),
                         "nominal_heldout": len(heldout_rows), "decay": len(decay)},
        "mean_u": {"nominal_all": float(u_nominal_all.mean()),
                   "nominal_heldout": float(u_held.mean()),
                   "decay_2025_q4": float(u_decay.mean())},
        "per_month_u": per_month,
        "flagged_at_95th_percentile_of_nominal": {
            m: bool(u > threshold) for m, u in per_month.items()},
        "nominal_95th_percentile_u": threshold,
        "auroc_decay_vs_nominal_heldout": float(auroc(u_decay, u_held)),
        "auroc_decay_vs_all_nominal": float(auroc(u_decay, u_nominal_all)),
        "per_month_stability": (
            "Which individual months clear the band moves with training length (2-3 of 3 "
            "across 200-400 epochs); the mean separation and the AUROC do not. With one "
            "episode, three decay months and 117 nominal plant-months, the per-month "
            "detail is noise and the aggregate is the result."
        ),
    }


def main():
    a = experiment_a()
    b_full = experiment_b()
    b_reduced = experiment_b(columns=rd.PLANT_FEATURES_NO_ENERGY)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode": {
            "scenario_id": rd.DECAY_SCENARIO_ID,
            "library_version": "1.0.0",
            "out_of_distribution": True,
            "plant": rd.DECAY_PLANT,
            "months": list(rd.DECAY_MONTHS),
            "monthly_ratio_of_baseline": {"2025-10": 0.7016, "2025-11": 0.1362,
                                          "2025-12": 0.0324},
        },
        "n_episodes": 1,
        "n_statement": (
            "One real degradation episode. The hour-level and month-level AUROCs have "
            "many points, but the number of independent events is ONE: this is a case "
            "study, not a distribution over failure modes. Any reported OOD detection "
            "rate needs either simulated degradations on top of this or an explicit n=1."
        ),
        "label_broadcast": (
            "The library window is monthly; evaluation A is hourly, so every hour of a "
            "flagged month inherits the flag. The within-month transition is unresolved "
            "-- 2025-10 sits at 0.70 of baseline and the collapse continues through it -- "
            "and nothing in the record resolves it."
        ),
        "feature_accounting_experiment_a": {
            "real_or_derived_from_real": list(rd.REAL_FEATURES),
            "static_site_constants": list(rd.STATIC_FEATURES),
            "absent_zero_filled": list(rd.ABSENT_FEATURES),
            "note": ("The five absent features are the electrical block. There is no "
                     "SCADA and no historian (ADR 0004), which is why the pinned schema "
                     "already marks them QUALITY_MISSING."),
        },
        "experiment_a_island_aggregate_hourly": a,
        "experiment_b_per_plant_monthly": b_full,
        "experiment_b_without_energy_feature": b_reduced,
        "headline": (
            "M2's uncertainty recovers the real Eluvaitivu degradation at the per-plant "
            "monthly resolution where it is observable, and does not detect it in "
            "island-aggregate hourly telemetry -- empirical support for the project's "
            "per-asset-over-aggregate-telemetry premise."
        ),
    }
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n")

    print("A  island-aggregate hourly")
    print(f"   naive AUROC (decay vs nominal window)  : {a['auroc_naive_decay_vs_nominal']:.3f}")
    print(f"   AUROC vs SAME-SEASON control (2024 Q4) : {a['auroc_vs_seasonal_control']:.3f}")
    print(f"   full-cycle mean u  2025 Q1-Q3 / Q4     : "
          f"{a['full_cycle']['mean_u_2025_q1_q3']:.3f} / {a['full_cycle']['mean_u_2025_q4']:.3f}")
    print(f"   -> {a['verdict']}")
    print("\nB  per-plant monthly")
    print(f"   mean u nominal / decay                 : "
          f"{b_full['mean_u']['nominal_all']:.3f} / {b_full['mean_u']['decay_2025_q4']:.3f}")
    print("   per-month u                            : "
          + "  ".join(f"{m}={u:.3f}" for m, u in b_full["per_month_u"].items()))
    print(f"   AUROC decay vs held-out nominal        : "
          f"{b_full['auroc_decay_vs_nominal_heldout']:.3f}")
    print(f"   same, without the energy feature       : "
          f"{b_reduced['auroc_decay_vs_nominal_heldout']:.3f}")
    print(f"   -> {b_full['verdict']} (recovers M1's window; n=1 episode)")
    print(f"\n({RESULTS.relative_to(Path(__file__).parent)} written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
