import { useEffect, useState } from "react";
import { CostBars } from "../../components/CostBars";
import { LedgerTable } from "../../components/LedgerTable";
import { SourceSplit } from "../../components/SourceSplit";
import { StatTile } from "../../components/StatTile";
import type { FleetSummary, GenerationRow, SystemSummary } from "../../lib";
import { api, ApiError, decimal, gwh, litres, rupeesM } from "../../lib";
import "./BaselineView.css";

interface Data {
  rows: GenerationRow[];
  systems: SystemSummary[];
  fleet: FleetSummary[];
}

export function BaselineView() {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.generation(), api.summary()])
      .then(([generation, summary]) => {
        if (cancelled) return;
        setData({ rows: generation.rows, systems: summary.by_system, fleet: summary.fleet });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof ApiError ? err.message : "Something went wrong loading the data.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <div className="state state--error">
        <h2 className="state__title">No calibration data</h2>
        <p className="state__body">{error}</p>
      </div>
    );
  }

  if (!data) {
    return <div className="state">Loading the ledger…</div>;
  }

  const fleet2025 = data.fleet.find((f) => f.year === "2025");
  const hybrid = data.systems.find(
    (s) => s.year === "2025" && s.island_system === "Eluvaitivu-Hybrid",
  );
  const dieselRef = data.systems.find(
    (s) => s.year === "2025" && s.island_system === "Eluvaitivu-Diesel",
  );

  // Renewable share is inferred, not metered: the hybrid ledger records diesel in and total kWh
  // out, with no separate PV or wind meter. The counterfactual is the diesel station's own
  // specific fuel consumption on the same island in the same year.
  const renewableShare =
    hybrid && dieselRef && hybrid.sfc_l_per_kwh && dieselRef.sfc_l_per_kwh
      ? 1 - hybrid.sfc_l_per_kwh / dieselRef.sfc_l_per_kwh
      : null;

  return (
    <div className="baseline">
      <div className="baseline__tiles">
        <StatTile
          label="Generated, 2025"
          value={gwh(fleet2025?.units_kwh ?? 0)}
          unit="GWh"
          note="Four islands, five generating systems"
        />
        <StatTile
          label="Diesel burned"
          value={litres(fleet2025?.diesel_l ?? 0)}
          unit="L"
          note={`Rs ${rupeesM(fleet2025?.fuel_cost_rs ?? 0)} in fuel, oil and transport`}
        />
        <StatTile
          label="All-in cost"
          value={`Rs ${decimal(fleet2025?.all_in_rs_per_kwh ?? 0, 0)}`}
          unit="/kWh"
          note="Fuel plus fleet maintenance and labour, before capital"
          severity="critical"
        />
        <StatTile
          label="Hybrid displacement"
          value={renewableShare === null ? "—" : `${decimal(renewableShare * 100, 0)}%`}
          note="Of the hybrid plant's own output, inferred — not metered"
        />
      </div>

      <SourceSplit rows={data.rows} />
      <CostBars systems={data.systems} year="2025" />
      <LedgerTable systems={data.systems} />

      <p className="baseline__provenance">
        Source: CEB / EDL Northern Province generation ledger, 2024–2025, reconciled against the
        printed annual summary — 300 invariants across 10 island-years. Renewable share is inferred
        from a diesel specific-fuel-consumption counterfactual; solar and wind are not separately
        metered.
      </p>
    </div>
  );
}
