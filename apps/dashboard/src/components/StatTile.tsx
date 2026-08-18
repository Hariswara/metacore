import type { ReactNode } from "react";
import "./StatTile.css";

interface Props {
  label: string;
  value: string;
  unit?: string;
  note?: ReactNode;
  severity?: "neutral" | "critical";
}

/** Summary before detail: the fleet's headline figures, readable without touching a chart. */
export function StatTile({ label, value, unit, note, severity = "neutral" }: Props) {
  return (
    <div className={`tile tile--${severity}`}>
      <div className="tile__label">{label}</div>
      <div className="tile__value">
        {value}
        {unit ? <span className="tile__unit">{unit}</span> : null}
      </div>
      {note ? <div className="tile__note">{note}</div> : null}
    </div>
  );
}
