import { useState } from "react";
import type { SystemSummary } from "../lib";
import { decimal, kwh, litres, rupeesM, systemLabel } from "../lib";
import "./LedgerTable.css";

interface Props {
  systems: SystemSummary[];
}

/** The table view every chart on this page is a projection of. Present by default rather than
 *  behind a toggle: this is a research dashboard, and the reader's next question is the number. */
export function LedgerTable({ systems }: Props) {
  const [year, setYear] = useState("2025");
  const years = [...new Set(systems.map((s) => s.year))].sort();
  const rows = systems
    .filter((s) => s.year === year)
    .sort((a, b) => b.units_kwh - a.units_kwh);

  return (
    <section className="ledger">
      <div className="ledger__head">
        <h3 className="ledger__title">Annual totals by generating system</h3>
        <div className="ledger__years" role="group" aria-label="Year">
          {years.map((y) => (
            <button
              key={y}
              type="button"
              className={`ledger__year${y === year ? " is-on" : ""}`}
              aria-pressed={y === year}
              onClick={() => setYear(y)}
            >
              {y}
            </button>
          ))}
        </div>
      </div>

      <div className="ledger__scroll">
        <table className="ledger__table">
          <thead>
            <tr>
              <th scope="col">System</th>
              <th scope="col" className="num">Generated (kWh)</th>
              <th scope="col" className="num">Diesel (L)</th>
              <th scope="col" className="num">L / kWh</th>
              <th scope="col" className="num">Transport (Rs)</th>
              <th scope="col" className="num">Fuel Rs / kWh</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.island_system}>
                <th scope="row">{systemLabel(r.island_system)}</th>
                <td className="num">{kwh(r.units_kwh)}</td>
                <td className="num">{litres(r.diesel_l)}</td>
                <td className="num">
                  {r.sfc_l_per_kwh === null ? "—" : decimal(r.sfc_l_per_kwh, 3)}
                </td>
                <td className="num">{rupeesM(r.transport_cost_rs)}</td>
                <td className="num">
                  {r.fuel_cost_rs_per_kwh === null ? "—" : decimal(r.fuel_cost_rs_per_kwh, 1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
