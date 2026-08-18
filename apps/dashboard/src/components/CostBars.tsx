import { useId, useState } from "react";
import type { SystemSummary } from "../lib";
import { decimal, systemLabel } from "../lib";
import "./CostBars.css";

interface Props {
  systems: SystemSummary[];
  year: string;
}

/** One measure, five entities — a ranked bar. The hybrid sits an order of magnitude below the
 *  diesel plants, which is the whole argument, so it is direct-labelled rather than left to the
 *  axis. Single series, so no legend: the title names the measure. */
export function CostBars({ systems, year }: Props) {
  const titleId = useId();
  const [active, setActive] = useState<string | null>(null);

  const data = systems
    .filter((s) => s.year === year && s.fuel_cost_rs_per_kwh !== null)
    .sort((a, b) => (b.fuel_cost_rs_per_kwh ?? 0) - (a.fuel_cost_rs_per_kwh ?? 0));
  const max = Math.max(...data.map((d) => d.fuel_cost_rs_per_kwh ?? 0));

  return (
    <figure className="cost">
      <figcaption className="cost__head">
        <h3 id={titleId} className="cost__title">
          Cost of fuel per unit generated, {year}
        </h3>
        <p className="cost__sub">
          Diesel, lubricating oil and marine barrel transport, divided by units generated. Excludes
          maintenance and labour, which CEB reports fleet-wide only.
        </p>
      </figcaption>

      <ul className="cost__list" aria-labelledby={titleId}>
        {data.map((d) => {
          const value = d.fuel_cost_rs_per_kwh ?? 0;
          const isHybrid = d.island_system === "Eluvaitivu-Hybrid";
          return (
            <li
              key={d.island_system}
              className={`cost__row${active === d.island_system ? " is-active" : ""}`}
              onMouseEnter={() => setActive(d.island_system)}
              onMouseLeave={() => setActive(null)}
            >
              <span className="cost__name">{systemLabel(d.island_system)}</span>
              <span className="cost__track">
                <span
                  className="cost__bar"
                  style={{
                    width: `${(value / max) * 100}%`,
                    background: isHybrid ? "var(--series-hybrid)" : "var(--series-diesel)",
                  }}
                />
              </span>
              <span className="cost__value">
                Rs {decimal(value, 1)}
                <span className="cost__unit">/kWh</span>
              </span>
            </li>
          );
        })}
      </ul>

      <p className="cost__foot">
        <span className="key">
          <span className="key__swatch" style={{ background: "var(--series-hybrid)" }} />
          The hybrid plant runs at roughly a third of the cost of the diesel station on the same
          island, serving the same load.
        </span>
      </p>
    </figure>
  );
}
