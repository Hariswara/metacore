# CEB Jaffna islands — measured baseline

What the 2024–2025 CEB ledger and the EDL tender documents establish about the four
islanded microgrids MetaCore targets, and what each fact is load-bearing for.

Data: [`data/external/ceb_jaffna/`](../../data/external/ceb_jaffna/README.md).
Corroborating interview: Electrical Superintendent Mr. Ramaneetharan, 2026-08-19.

---

## 1. The system under study

Four islands off Jaffna — Analaitivu, Eluvaitivu, Nainativu, Neduntivu (Delft) — each a
**fully islanded diesel microgrid** with no mainland interconnection. Eluvaitivu additionally
carries a 60 kW hybrid plant, commissioned 2016, running in parallel with its diesel station.

Fleet in 2025: **2.95 GWh** generated from **975,615 L** of diesel.

### Eluvaitivu hybrid — the only non-diesel asset in the fleet

From the EDL tender `EDL/NP/ELV/BAT/2026` and the 2016 drawings:

| Element | Specification |
|---|---|
| Solar PV | ~50 kWp (≈97 panels) on SMA 12 kW + 15 kW inverters. Field sketch and interview both say **46.26 kW** |
| Wind | 6 × Windspot 3.5 kW = **21 kW**, Aurora 4.2 kW inverters |
| Diesel backup | **30 kW** (a 60/100 kVA unit was proposed and never installed) |
| Battery | 4 × 33.715 kWh LiFePO4 ≈ **134 kWh usable**, 48 V |
| Inverters | 12 × SMA Sunny Island SI 8.0H-11, 6 kW each, in 4 three-phase clusters (1 Master + 3 Extension) via SMA Multicluster 36–180 kW |
| Protection | Battery fuse switch 250 A, HRC fuse 250 A, 125 A MRC distribution fuses |

Bus architecture: PV and wind are **AC-coupled** into the Multicluster board; the battery is the
DC link and **the grid-forming reference**. Consequence, stated directly in the interview: when the
battery bank fails, the whole plant collapses — solar and wind cannot run without it. There is no
black-start path that does not go through either the battery or the diesel genset.

## 2. Generation cost is 3–4× the mainland, and the tariff does not cover it

Annual, derived from the ledger:

| System | Year | Energy (kWh) | SFC (L/kWh) | Fuel Rs/kWh | Transport Rs/kWh | Fuel+oil+transport Rs/kWh |
|---|---|---|---|---|---|---|
| Analaithivu | 2024 | 398,143 | 0.412 | 134.6 | 3.09 | **140.0** |
| Analaithivu | 2025 | 426,418 | 0.384 | 108.6 | 2.88 | **112.5** |
| Eluvaitivu-Diesel | 2024 | 88,667 | 0.402 | 131.8 | 3.02 | **134.8** |
| Eluvaitivu-Diesel | 2025 | 113,276 | 0.407 | 114.7 | 3.05 | **117.7** |
| **Eluvaitivu-Hybrid** | 2024 | 96,445 | **0.091** | 29.3 | 0.69 | **31.1** |
| **Eluvaitivu-Hybrid** | 2025 | 83,331 | **0.116** | 33.0 | 0.87 | **34.9** |
| Delft-Neduntivu | 2024 | 1,064,031 | 0.321 | 106.2 | 0.29 | **108.2** |
| Delft-Neduntivu | 2025 | 1,126,880 | 0.325 | 92.8 | 0.31 | **94.2** |
| Nainativu | 2024 | 1,180,710 | 0.328 | 107.4 | 2.46 | **110.1** |
| Nainativu | 2025 | 1,198,460 | 0.325 | 91.9 | 2.44 | **94.8** |

Fleet 2025, all known cost: fuel Rs 283.9 M + aggregate O&M Rs 92.1 M = **Rs 376.0 M for 2.95 GWh
= Rs 127.5/kWh**, before capital and depreciation. Against a domestic block tariff of roughly
Rs 30/unit, that is a **~Rs 98/kWh subsidy and a ~Rs 288 M/yr recovery gap**. The interview frames
the same arithmetic from the other side: 30 units billed at Rs 900 that "should be over Rs 9,000",
i.e. ~Rs 300/unit once capital is loaded in.

Two structural details the cost table exposes:

- **Small plants are the least efficient.** Analaithivu and Eluvaitivu-Diesel burn 0.38–0.41 L/kWh;
  Delft and Nainativu 0.32–0.33. Oversized gensets on light load — Eluvaitivu's demand is ~50 kVA
  behind a 100 kVA set, Analaitivu's ~100 kVA behind 250 kVA. Every marginal kWh on the small
  islands costs ~25% more fuel than on the big ones.
- **Marine logistics is a real, island-specific cost.** Diesel moves in 200 L barrels at a flat
  Rs 1,500/barrel to Analaitivu, Eluvaitivu and Nainativu, but **Rs 180/barrel to Delft in 2024,
  rising to Rs 190 in 2025** — a 7.9–8.3× spread that has nothing to do with distance and
  everything to do with which islands have a ferry. The rate is re-tendered annually and must be
  read per year, not averaged across the dataset. Nainativu alone pays Rs 2.92 M/yr in barrel
  transport.

Diesel price itself fell from ~Rs 346/L (Jan 2024) to ~Rs 283/L (2025), which is why 2025 Rs/kWh
drops without any efficiency gain. **Cost trends in this dataset are dominated by fuel price, not
by operations** — deflate before drawing any operational conclusion.

## 3. The hybrid works, and there is a clean before/after failure event

Using each year's Eluvaitivu diesel-only SFC as the counterfactual for the same island and the
same load:

| Year | Hybrid kWh | Diesel-attributable kWh | **Renewable + storage share** | Diesel avoided | Value avoided |
|---|---|---|---|---|---|
| 2024 | 96,445 | 21,915 | **77.3 %** | 29,995 L | Rs 9.82 M |
| 2025 | 83,331 | 23,873 | **71.4 %** | 24,171 L | Rs 6.82 M |

A 60 kW hybrid displaced ~77% of its own generation on a single island. Extrapolating that share to
the four diesel-only systems would avoid **~747,000 L and ~Rs 216 M per year**.

### The collapse, month by month

Eluvaitivu is the fleet's only island where two independently metered plants serve one load, so the
hybrid's failure is visible as a clean substitution — total island demand barely moves while the
split swings completely:

| 2025 | Hybrid kWh | Hybrid L | Diesel kWh | Diesel L | Island kWh | Hybrid share |
|---|---|---|---|---|---|---|
| Aug | 8,872 | 995 | 9,380 | 3,645 | 18,252 | 48.6 % |
| Sep | 8,363 | 1,175 | 8,854 | 3,525 | 17,217 | 48.6 % |
| Oct | 5,653 | 1,010 | 10,951 | 4,285 | 16,604 | **34.0 %** |
| Nov | 1,097 | 125 | 14,577 | 6,315 | 15,674 | **7.0 %** |
| Dec | 261 | 40 | 15,300 | 6,720 | 15,561 | **1.7 %** |

Island demand holds at ~15,600 kWh/month while hybrid output falls 97% in three months and diesel
absorbs all of it — diesel burn goes from ~3,500 to ~6,700 L/month. This is the battery-bank
expiry the tender was raised to fix ("the existing battery banks have expired and can no longer
support reliable operation"), compounded by the wind turbines being out of service. The interview
confirms both: all six turbines broken, three previously working, and total system collapse
following battery failure.

**This is the most valuable single feature of the dataset for MetaCore.** It is a real,
instrumented, ground-truth degradation-to-failure trajectory on an islanded microgrid — a
labelled anomaly with 22 months of healthy operation before it.

## 4. Demand is seasonal and peaky

2025 monthly energy:

| Island | Mean | Peak | Trough | Peak/trough |
|---|---|---|---|---|
| Analaithivu | 35,535 | 40,460 (Oct) | 25,466 (Feb) | 1.59 |
| Delft-Neduntivu | 93,907 | 112,530 (Oct) | 72,305 (Jan) | 1.56 |
| Nainativu | 99,872 | 142,280 (Jul) | 73,250 (Feb) | 1.94 |

Nainativu at 1.94× peak-to-trough is the hardest dispatch problem in the fleet. Reported maximum
demand is 460 kVA apparent, against 880 kVA installed.

## 5. There is no automation anywhere in the fleet

Every control action described in the interview is manual, and the field sketch of the old power
station confirms the protection chain:

```
generator ──[MCCB]──▶ panel ──[MCCB]──▶ cut-out / fuse disconnector (DDLO) ──▶ 400 V feeder
  250 kVA              250 A            drop-down lift-off, ~100 A
```

- Gensets are **under-protected by design**: a 250 kVA set should carry a 400 A MCCB but is fitted
  with 250 A, deliberately, because the machines are old and the operator wants them to trip early.
- Load shedding is **physical** — someone travels to the pole and pulls a DDLO fuse.
- **Hospitals cannot be isolated.** They sit on a shared feeder; supplying the hospital means
  supplying every customer on that line. There is no switching arrangement to do otherwise.
- Restoration after a storm runs **4–5 days**. Fuses drop on wind and rain; a failed transformer
  requires a phone call to the power station and a manual line stop before anyone can approach it.
- Diesel arrives by boat. If the boat does not run, the island does not generate — and during a
  cyclone the boat does not run.
- Data entry and reporting are manual and printed. There is no historian and no SCADA.

Protection relays (OCEF, OV) exist **only inside the Eluvaitivu hybrid's SMA equipment** — nowhere
else in the fleet.

## 6. What this means for MetaCore

| Module | What the data supplies |
|---|---|
| **M1** — spatiotemporal state | 120 island-months of energy, fuel and cost across 5 systems. Monthly resolution only; sub-monthly state must come from simulation calibrated to these totals |
| **M2** — epistemic uncertainty | The Oct–Dec 2025 hybrid decay is a labelled degradation trajectory: the model should get *more* uncertain as the battery fades, and the ledger says exactly when it did |
| **M3** — cost-aware gating | Real, island-specific marginal costs: Rs 92–115/kWh fuel, Rs 180–190 vs Rs 1,500/barrel transport, Rs 2,100/L lube oil. The genset start/stop decision the interview asks the agent to make now has a real objective function |
| **M4** — physics verification | A complete, dimensioned single-line: 48 V DC bus, 12 × 6 kW inverters, 134 kWh usable, 250 A battery fuses, 408 A discharge / 345 A charge per cluster, 400 V LT distribution, 250 A genset MCCB, ~100 A DDLO cut-outs. Enough to build the OpenDSS model and enforce real ampacity limits |

### The headline argument for the research

The fleet spends **Rs 376 M/yr to deliver 2.95 GWh**, recovers roughly a quarter of it, and has
**zero automated control**. The one hybrid asset displaced 77% of its fuel — then failed, and the
failure was neither predicted nor detected until output had already dropped 97%, because nobody was
watching a number that a model could have watched. That is precisely the gap a metacognitive agent
is supposed to close.

## 7. Open items

- **Renewable generation is not separately metered.** Solar and wind share are inferred from a
  diesel-SFC counterfactual, not measured. State this as a limitation; do not present 77.3% as a
  measured value. Ask EDL whether the SMA Multicluster / Sunny Island logs are retrievable — the
  inverters record it even though the ledger does not.
- **Per-island O&M** is not disaggregated; Rs 92 M/yr is fleet-wide only.
- **Meteorology** is now pulled from NASA POWER instead of paying the Department of Meteorology's
  Rs 75,000 quotation — two years of hourly data for all four islands, in
  `data/raw/nasa_power`. It is free and licence-free, so unlike `external/` it is allowed to be a
  build dependency. **It does not resolve the islands**, which is a constraint on M1 rather than on
  the pipeline: see [`nasa-power-resolution.md`](nasa-power-resolution.md).
- **Node coordinates** are held by the Division 1 distribution control centre, not supplied here.
  Per `data/README.md` governance, publish them only as arbitrary spatial offsets.
- Substations at Killinochchi, Madhavachchi and Mannar are mainland; the islands have no
  substation, only a distribution control centre.
- **Planned hybrid rollout** (interview, not yet documented): Delft 700 kW, Nainativu 700 kW,
  Analaitivu 300 kW, Eluvaitivu 60 kW. Delft is the next site. Get the design documents when issued.
