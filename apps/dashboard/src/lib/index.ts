// API client and websocket stream against services/gateway.
// Requests go through vite's `/api` proxy (vite.config.ts) -> local uvicorn or compose `gateway`.

export interface Module3RunConfig {
  seed: number;
  episodeLen: number;
  budgetPerEpisode: number;
  trainEpisodes: number;
  evalEpisodes: number;
  deliberationCost: number;
  benefitScale: number;
  sensingEscalationPenalty: number;
}

export const MODULE3_DEFAULT_CONFIG: Module3RunConfig = {
  seed: 0,
  episodeLen: 80,
  budgetPerEpisode: 45,
  trainEpisodes: 150,
  evalEpisodes: 12,
  deliberationCost: 0.1,
  benefitScale: 4.0,
  sensingEscalationPenalty: -1.2,
};

// action_id ties a ProposedControlAction to its GatingDecision — see M3_TO_M4_CONTRACT.md.
export interface ProposedControlAction {
  action_id: string;
  origin: "SYSTEM1" | "SYSTEM2";
  breakers: Array<{ edge_id: string; closed: boolean }>;
  load_shed: Array<{ node_id: string; shed_fraction: number; priority_tier: number }>;
  dispatch: Array<{ node_id: string; p_kw: number; q_kvar: number }>;
  rationale: string;
  schema_version: string;
  message_type: "ProposedControlAction";
}

export interface GatingDecision {
  action_id: string;
  chosen: "SYSTEM1" | "SYSTEM2";
  epistemic_at_decision: number;
  expected_survival_benefit: number;
  deliberation_cost: number;
  latency_ms: number;
  budget_exhausted_fallback: boolean;
  schema_version: string;
  message_type: "GatingDecision";
  timestamp: number;
}

export type Module3Record = ProposedControlAction | GatingDecision;

// Not part of the M3->M4 wire contract (M3_TO_M4_CONTRACT.md) -- this is the M2 input
// state and M4-mock verdict that produced each decision, added purely for the dashboard.
// See run_demo.py's `context` list.
export interface Module3DecisionContext {
  action_id: string;
  severity: string;
  trigger_reason: "none" | "value" | "sensing" | "both";
  epistemic_uncertainty: number;
  observed_fraction: number;
  competence_drop: boolean;
  reward: number;
  verdict: "APPROVE" | "REJECT";
  violations: Array<{
    type: string;
    element_id: string;
    measured: number;
    limit: number;
    margin_fraction: number;
    attributed_component: string;
  }>;
}

export interface Module3RunResult {
  reward: { always_s1: number; always_s2: number; trained_policy: number };
  avg_deliberation_cost: { always_s2: number; trained_policy: number };
  escalation_by_severity: Record<string, number>;
  monotonic_nondecreasing: boolean;
  escalation_by_trigger_reason: Record<string, number>;
  decisions: Module3Record[];
  decision_context: Module3DecisionContext[];
}

export class Module3RunError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "Module3RunError";
  }
}

function toRequestBody(cfg: Module3RunConfig) {
  return {
    seed: cfg.seed,
    episode_len: cfg.episodeLen,
    budget_per_episode: cfg.budgetPerEpisode,
    train_episodes: cfg.trainEpisodes,
    eval_episodes: cfg.evalEpisodes,
    deliberation_cost: cfg.deliberationCost,
    benefit_scale: cfg.benefitScale,
    sensing_escalation_penalty: cfg.sensingEscalationPenalty,
  };
}

export async function runModule3(cfg: Module3RunConfig): Promise<Module3RunResult> {
  return postModule3("/api/module3/run", toRequestBody(cfg));
}

export interface ObservationField {
  index: number;
  name: string;
  source: "M1" | "M2" | "env" | string;
  meaning: string;
  value: number;
}

export interface Module3GenerateResult {
  step_index: number;
  episode_len: number;
  observation: ObservationField[];
  raw: {
    epistemic_uncertainty: number;
    trigger_reason: "none" | "value" | "sensing" | "both";
    observed_fraction: number;
    competence_drop: boolean;
    severity: string;
    max_node_vulnerability: number;
    top_at_risk_nodes?: string[];
    [key: string]: unknown;
  };
  seed_used: number;
}

export interface Module3ProcessResult {
  chosen: "SYSTEM1" | "SYSTEM2";
  requested: "SYSTEM1" | "SYSTEM2";
  budget_exhausted_fallback: boolean;
  action_probs: { SYSTEM1: number; SYSTEM2: number };
  epistemic_at_decision: number;
  deliberation_cost: number;
  plan: ProposedControlAction;
  observation: ObservationField[];
  raw: Module3GenerateResult["raw"];
}

export async function generateModule3Situation(): Promise<Module3GenerateResult> {
  return postModule3("/api/module3/generate");
}

export async function processModule3Gate(): Promise<Module3ProcessResult> {
  return postModule3("/api/module3/process");
}

async function postModule3<T>(url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    const payload = await res.json().catch(() => null);
    const detail =
      typeof payload?.detail === "string"
        ? payload.detail
        : Array.isArray(payload?.detail)
          ? payload.detail.map((d: { msg?: string }) => d.msg).join("; ")
          : `request failed with status ${res.status}`;
    throw new Module3RunError(detail, res.status);
  }
  return (await res.json()) as T;
}

// Pairs each ProposedControlAction with its GatingDecision (and M2 input / M4 verdict
// context, when present) by action_id, in emission order.
export interface Module3DecisionRow {
  actionId: string;
  action: ProposedControlAction;
  decision: GatingDecision;
  context?: Module3DecisionContext;
}

export function pairModule3Decisions(
  records: Module3Record[],
  context: Module3DecisionContext[] = [],
): Module3DecisionRow[] {
  const byId = new Map<string, Partial<Module3DecisionRow>>();
  for (const rec of records) {
    const entry = byId.get(rec.action_id) ?? { actionId: rec.action_id };
    if (rec.message_type === "ProposedControlAction") entry.action = rec;
    else entry.decision = rec;
    byId.set(rec.action_id, entry);
  }
  const contextById = new Map(context.map((c) => [c.action_id, c]));
  const rows: Module3DecisionRow[] = [];
  for (const entry of byId.values()) {
    if (entry.action && entry.decision) {
      rows.push({
        actionId: entry.actionId!,
        action: entry.action,
        decision: entry.decision,
        context: contextById.get(entry.actionId!),
      });
    }
  }
  return rows;
}
