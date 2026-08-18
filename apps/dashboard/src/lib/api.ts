/**
 * Typed client for the gateway. Every field here mirrors a column the Module 1 calibration
 * pipeline produces, so a number on screen traces back to a row in the ledger.
 */

export type IslandSystem =
  | "Analaithivu"
  | "Eluvaitivu-Diesel"
  | "Eluvaitivu-Hybrid"
  | "Delft-Neduntivu"
  | "Nainativu";

export interface GenerationRow {
  year: string;
  month: string;
  month_num: number;
  island_system: IslandSystem;
  diesel_l: number | null;
  diesel_cost_rs: number | null;
  units_kwh: number | null;
  oil_l: number | null;
  oil_cost_rs: number | null;
  diesel_barrel: number | null;
  barrel_amount: number | null;
  sfc_l_per_kwh: number | null;
  diesel_rs_per_l: number | null;
  fuel_cost_rs_per_kwh: number | null;
  total_cost_rs_per_kwh: number | null;
}

export interface SystemSummary {
  year: string;
  island_system: IslandSystem;
  units_kwh: number;
  diesel_l: number;
  diesel_cost_rs: number;
  oil_cost_rs: number;
  transport_cost_rs: number;
  sfc_l_per_kwh: number | null;
  fuel_cost_rs_per_kwh: number | null;
}

export interface FleetSummary {
  year: string;
  units_kwh: number;
  diesel_l: number;
  fuel_cost_rs: number;
  om_cost_rs: number;
  fuel_rs_per_kwh: number | null;
  all_in_rs_per_kwh: number | null;
}

export interface Health {
  ready: boolean;
  service: string;
  calibration_available: boolean;
  calibration_path: string;
}

/** The gateway answers 503 with a `detail` explaining what to run. Surface that, not a stack. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function get<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`);
  } catch {
    throw new ApiError("Cannot reach the gateway. Is it running on port 8000?", 0);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(body?.detail ?? `Request failed (${response.status})`, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => get<Health>("/health"),
  generation: () => get<{ rows: GenerationRow[]; count: number }>("/calibration/generation"),
  summary: () =>
    get<{ by_system: SystemSummary[]; fleet: FleetSummary[] }>("/calibration/summary"),
};
