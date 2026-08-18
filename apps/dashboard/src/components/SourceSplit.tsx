import { useId, useState } from "react";
import type { GenerationRow } from "../lib";
import { kwh, percent } from "../lib";
import "./SourceSplit.css";

interface Props {
  rows: GenerationRow[];
}

interface Month {
  key: string;
  year: string;
  month: string;
  hybrid: number;
  diesel: number;
  total: number;
}

const PLOT = { w: 980, h: 300, top: 16, right: 12, bottom: 34, left: 56 };

/** Eluvaitivu is the only island where two metered plants serve one load, so the hybrid's
 *  failure shows as pure substitution: the split swings while the total barely moves. Stacked
 *  bars carry both facts at once — segment heights for the split, column height for the total. */
export function SourceSplit({ rows }: Props) {
  const titleId = useId();
  const [active, setActive] = useState<number | null>(null);

  const months = buildMonths(rows);
  const max = Math.max(...months.map((m) => m.total));
  const inner = {
    w: PLOT.w - PLOT.left - PLOT.right,
    h: PLOT.h - PLOT.top - PLOT.bottom,
  };
  const band = inner.w / months.length;
  const barW = band - 4;
  const y = (v: number) => PLOT.top + inner.h - (v / max) * inner.h;
  const ticks = [0, 5000, 10000, 15000, 20000].filter((t) => t <= max * 1.05);

  const hovered = active === null ? null : months[active];

  return (
    <figure className="split">
      <figcaption className="split__head">
        <h3 id={titleId} className="split__title">
          Eluvaitivu — where the island&rsquo;s power came from, month by month
        </h3>
        <p className="split__sub">
          Total demand holds near 15,600&nbsp;kWh a month throughout. From October 2025 the hybrid
          plant collapses and the diesel station absorbs all of it.
        </p>
        <div className="split__legend">
          <span className="key">
            <span className="key__swatch" style={{ background: "var(--series-hybrid)" }} />
            Hybrid plant
          </span>
          <span className="key">
            <span className="key__swatch" style={{ background: "var(--series-diesel)" }} />
            Diesel station
          </span>
        </div>
      </figcaption>

      <div className="split__plot">
        <svg
          viewBox={`0 0 ${PLOT.w} ${PLOT.h}`}
          role="img"
          aria-labelledby={titleId}
          className="split__svg"
        >
          {ticks.map((t) => (
            <g key={t}>
              <line
                x1={PLOT.left}
                x2={PLOT.w - PLOT.right}
                y1={y(t)}
                y2={y(t)}
                stroke={t === 0 ? "var(--rule-strong)" : "var(--rule)"}
                strokeWidth="1"
              />
              <text x={PLOT.left - 10} y={y(t) + 4} className="split__tick" textAnchor="end">
                {t === 0 ? "0" : `${t / 1000}k`}
              </text>
            </g>
          ))}

          {months.map((m, i) => {
            const x = PLOT.left + i * band + 2;
            const hybridH = (m.hybrid / max) * inner.h;
            const dieselH = (m.diesel / max) * inner.h;
            // 2px surface gap between stacked segments, per the mark spec.
            const gap = m.hybrid > 0 && m.diesel > 0 ? 2 : 0;
            return (
              <g
                key={m.key}
                onMouseEnter={() => setActive(i)}
                onMouseLeave={() => setActive(null)}
                onFocus={() => setActive(i)}
                onBlur={() => setActive(null)}
                tabIndex={0}
                role="button"
                aria-label={`${m.month} ${m.year}: hybrid ${kwh(m.hybrid)} kWh, diesel ${kwh(
                  m.diesel,
                )} kWh`}
                className={`split__col${active === i ? " is-active" : ""}`}
              >
                <rect
                  x={x - 2}
                  y={PLOT.top}
                  width={band}
                  height={inner.h}
                  fill="transparent"
                  className="split__hit"
                />
                <rect
                  x={x}
                  y={y(m.diesel)}
                  width={barW}
                  height={Math.max(dieselH - gap, 0)}
                  rx="2"
                  fill="var(--series-diesel)"
                />
                <rect
                  x={x}
                  y={y(m.total)}
                  width={barW}
                  height={Math.max(hybridH, 0)}
                  rx="2"
                  fill="var(--series-hybrid)"
                />
              </g>
            );
          })}

          {months.map((m, i) =>
            m.month === "Jan" || m.month === "Jul" ? (
              <text
                key={`x-${m.key}`}
                x={PLOT.left + i * band + band / 2}
                y={PLOT.h - 12}
                className="split__tick"
                textAnchor="middle"
              >
                {m.month} {m.year.slice(2)}
              </text>
            ) : null,
          )}
        </svg>

        {hovered ? (
          <div
            className="split__tip"
            style={{
              left: `${((PLOT.left + months.indexOf(hovered) * band + band / 2) / PLOT.w) * 100}%`,
            }}
          >
            <div className="split__tip-title">
              {hovered.month} {hovered.year}
            </div>
            <dl className="split__tip-rows">
              <div>
                <dt>
                  <span className="key__swatch" style={{ background: "var(--series-hybrid)" }} />
                  Hybrid
                </dt>
                <dd>
                  {kwh(hovered.hybrid)} kWh
                  <span className="split__tip-share">
                    {percent(hovered.hybrid / hovered.total, 0)}
                  </span>
                </dd>
              </div>
              <div>
                <dt>
                  <span className="key__swatch" style={{ background: "var(--series-diesel)" }} />
                  Diesel
                </dt>
                <dd>
                  {kwh(hovered.diesel)} kWh
                  <span className="split__tip-share">
                    {percent(hovered.diesel / hovered.total, 0)}
                  </span>
                </dd>
              </div>
              <div className="split__tip-total">
                <dt>Island total</dt>
                <dd>{kwh(hovered.total)} kWh</dd>
              </div>
            </dl>
          </div>
        ) : null}
      </div>
    </figure>
  );
}

function buildMonths(rows: GenerationRow[]): Month[] {
  const byKey = new Map<string, Month>();
  for (const row of rows) {
    if (row.island_system !== "Eluvaitivu-Hybrid" && row.island_system !== "Eluvaitivu-Diesel") {
      continue;
    }
    const key = `${row.year}-${String(row.month_num).padStart(2, "0")}`;
    const entry =
      byKey.get(key) ??
      { key, year: row.year, month: row.month, hybrid: 0, diesel: 0, total: 0 };
    if (row.island_system === "Eluvaitivu-Hybrid") entry.hybrid = row.units_kwh ?? 0;
    else entry.diesel = row.units_kwh ?? 0;
    entry.total = entry.hybrid + entry.diesel;
    byKey.set(key, entry);
  }
  return [...byKey.values()].sort((a, b) => a.key.localeCompare(b.key));
}
