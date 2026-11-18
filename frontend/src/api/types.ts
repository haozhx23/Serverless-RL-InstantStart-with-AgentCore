// Shapes returned by the backend. Event schema v1: docs/architecture.md.

// Unions written as interface keys, one per line, so a release can leave a member out (release-manifest.json features).
interface BackendKeys {
  live: true;
  replay: true;
}
export type Backend = keyof BackendKeys;
export type Engine = "training-session" | "mtrl";
export type Lane = "client" | "aws";
export type RunStatus = "starting" | "running" | "stopping" | "completed" | "failed" | "stopped" | "interrupted";
export type Tokens = { prefill: number; sample: number; train: number };

interface EventBase { v: number; seq: number; ts: number; t: number; run: string }
export type RunEvent = EventBase & (
  | { type: "run_started"; scenario: string; backend: Backend; model: string; region: string; total_steps: number;
      params: Record<string, unknown>; engine?: Engine; recorded_at?: string; recorded_run?: string; speed?: number;
      variants?: Record<string, string | number> }
  | { type: "job"; name: string; arn: string; status: string; secondary: string | null; failure_reason?: string | null;
      mlflow_url?: string; agent_version?: string | null }
  | { type: "billing"; source: string; tokens: Tokens }
  | { type: "policy_check"; rows: PolicyRow[]; checks: PolicyCheck[]; n: number }
  | { type: "dataset"; splits?: Record<string, DatasetSplit>; rollouts?: string; uris?: Record<string, string> }
  | { type: "session"; state: "creating" | "running" | "stopped" | "failed"; name: string | null; ready_sec?: number }
  | { type: "step"; step: number; total: number }
  | { type: "span_start"; lane: Lane; op: string; step: number | null; t0: number; calls?: number; role?: "probe";
      where?: SandboxWhere }
  | { type: "span"; lane: Lane; op: string; step: number | null; t0: number; sec: number; ok: boolean;
      tokens?: Partial<Tokens>; note?: string; calls?: number; role?: "probe";
      /** Your code that ran in the environment sandbox, not in this process; its requests are sandbox_calls. */
      where?: SandboxWhere; sandbox_calls?: number }
  | ({ type: "sandbox" } & SandboxEvent)
  | { type: "probe"; id: string; question: string; answer: string; step: number; weights: string }
  | { type: "reward_check"; rows: { case: string; example: string; reward_mean: number }[]; checks: PolicyCheck[] }
  | { type: "metrics"; step: number; values: Record<string, number> }
  | { type: "rollouts"; step: number; items: Rollout[] }
  | { type: "compare"; label: string; items: CompareItem[] }
  | { type: "eval"; label: string; unit: string; n: number; before: number; after: number }
  | { type: "artifact"; kind: "CHECKPOINT" | "MODEL"; arn: string; description: string; step: number | null }
  | { type: "handled"; code: string; message: string }
  | { type: "log"; level: "info" | "warn" | "error"; msg: string }
  | ({ type: "mlflow" } & MlflowWhere)
  | { type: "run_finished"; status: "completed" | "failed" | "stopped"; error: string | null; source?: string }
);
export type EventOf<T extends RunEvent["type"]> = Extract<RunEvent, { type: T }>;
export type Span = EventOf<"span">;

/** Where sandboxed code ran: an AgentCore Runtime session. */
interface SandboxWhereKeys {
  agentcore: true;
}
export type SandboxWhere = keyof SandboxWhereKeys;
export interface SandboxDist { p50: number; max: number; n: number }
/** The environment sandbox's side of a run (templates/_shared/sandbox_client.py, docs/architecture.md 4). */
export type SandboxEvent =
  | { state: "ready"; where: SandboxWhere; runtime: string | null; platform: string | null; code_sha: string;
      session_prefix: string; rate: number }
  | { state: "batch"; step: number | null; what: string; ok: boolean; sessions_new: number; calls: number; retries: number;
      cold_start: SandboxDist | null; open: number; created: number }
  | { state: "stopped"; step: number | null; sessions: number; open: number; failed: { session: string; error: string }[] }
  | { state: "closed"; where: SandboxWhere; runtime: string | null; platform: string | null; created: number; stopped: number;
      calls: number; retries: number; cold_start: SandboxDist | null; failed: { session: string; error: string }[];
      session_prefix: string };

/** What AgentCore Runtime used for a live run, at list price (GET /api/runs/{id}/agentcore). */
export interface AgentcoreUsage {
  kind: "sandbox" | "agent"; vcpu_hours: number; gb_hours: number; usd: number | null;
  price: { vcpu_h: number; gb_h: number } | null; platform: string | null; price_as_of: string;
  last_record_ts: number | null; source: string; checked_at: number;
  sessions_reported?: number; sessions_created?: number | null;
}

/** Where a live training-session run published itself in MLflow (templates/_shared/srl_mlflow.py). */
export interface MlflowWhere {
  state: "publishing" | "failed" | "ended"; tracking_uri: string; experiment: string;
  experiment_id: string | null; run_id: string | null; status?: string | null; error?: string | null;
  published?: { metric_steps: number; traces: number; model_packages: number } | null;
}
/** Why a run did or did not publish to MLflow (run.json; null for runs created before publishing existed). */
interface MlflowModeKeys {
  publish: true;
  off: true;
  "no-app": true;
  replay: true;
  service: true;
}
export type MlflowMode = keyof MlflowModeKeys;

/** The models each kind of launch can use (GET /api/scenarios). */
export interface Models {
  live: string[] | null;
  mtrl_live: string[] | null;
}

export interface PolicyRow { policy: string; description: string; n: number; reward_mean: number; reward_std: number; exact_match: number; coverage: number }
export interface PolicyCheck { check: string; ok: boolean; why: string }
export interface DatasetSplit { n: number; seed: number; decisions: Record<string, number>; avg_prompt_chars: number }

export interface Rollout { group: number; prompt: string; answer: number | string; completion: string; reward: number; stop_reason: string }
export interface CompareItem { prompt: string; expected: string; before: string; after: string; before_ok: boolean; after_ok: boolean }

export interface MetricSpec { key: string; label: string; format: string; min?: number; max?: number }
export interface ParamSpec {
  key: string; label: string; type: "int" | "float" | "bool" | "choice" | "model" | "mtrl_model"; default: unknown;
  min?: number; max?: number; choices?: (string | number)[]; advanced?: boolean; help?: string; slot?: boolean;
}

export interface Recording {
  name: string; source_run: string; scenario: string; recorded_ts: number; model: string; region: string;
  duration_s: number | null; cost_usd: number | null; tokens: Tokens; eval: RunSummary["eval"];
  /** Comes with the project (templates/<id>/recordings/), the recording account's ID scrubbed. */
  shipped: boolean;
}

/** A completed live run an estimate is based on, as it ran (backend/srl/estimate.py). */
export interface EstimateRun {
  run: string; date: string; steps: number; model: string; usd: number | null; eval: RunSummary["eval"];
  custom?: boolean; differs?: { key: string; measured: unknown; planned: unknown }[];
}
/** What a live run with given settings costs, from live runs with the same settings (docs/architecture.md 3.9). */
export interface CostEstimate {
  status: "measured" | "subset" | "constant" | "partial" | "none";
  usd: { low: number; high: number } | null; tokens: { low: Tokens; high: Tokens } | null;
  steps: number | null; measured_steps: number | null; basis: EstimateRun[];
  /** partial: steps beyond the measured ones, left out of usd; step_usd is one measured step's range. */
  unestimated_steps: number; step_usd: { low: number; high: number } | null;
  reason: { code: "mtrl" | "not_described" | "no_price" | "no_live_runs" | "custom_code" | "model" | "differs";
            model?: string; measured_models?: string[] } | null;
  nearest: EstimateRun[]; price_as_of: string;
  /** A sandbox scenario: usd covers training tokens only; AgentCore Runtime is measured per run, not estimated. */
  agentcore: "not_estimated" | null;
}

// The loop a scenario runs, as drawn on the loop map. Nodes are in code order; the driver
// side holds the for-loop (client: your process; aws: SageMaker AI, as in MTRL).
export interface LoopNode {
  op: string; lane: Lane; title: string; call: string; out?: string; slot?: string; primitive?: string; observed?: boolean;
  where?: "agentcore";  // your code, run in the environment sandbox (AgentCore Runtime) instead of your process
  file?: string;   // show this file for the node instead of loop.py
  function?: string;  // and highlight this top-level function in it
}
export interface LoopSpec {
  driver: Lane; header: string; header_out?: string; before: string[]; after: string[]; nodes: LoopNode[];
}
export interface VariantSpec { id: string | number; label: string; short?: string; file?: string; summary?: string; watch?: string }
export interface SlotSpec {
  key: string; node: string | null; label: string; file?: string; param?: string; contract?: string; function?: string;
  editable?: boolean; default?: string; variants: VariantSpec[];
}

/** Where an official sample came from (scenario.json `source`). */
export interface ScenarioSource {
  repo: string; path: string; commit: string; license: string; url: string; published?: string; published_url?: string;
}

export interface Scenario {
  id: string; order: number; engine: Engine; title: string; technique: string; summary: string;
  source?: ScenarioSource; probes?: boolean; probe_example?: string;
  /** A live run sends code to the environment sandbox, so it also needs the sandbox to be ready (scenario.json). */
  sandbox?: { executes: string };
  status: "derived" | "live-verified" | "planned"; plan?: string; loop?: LoopSpec; slots?: SlotSpec[];
  evidence?: { basis: string; date: string; region: string; model: string; duration_s: number | null; cost_usd: number | null; note: string };
  you_write?: string[]; aws_runs?: string[]; you_will_see?: string[]; files?: string[];
  metrics?: MetricSpec[]; params?: ParamSpec[]; recordings: Recording[];
}

export interface RunSummary {
  id: string; scenario: string; title: string; backend: Backend; engine: Engine; status: RunStatus; alive: boolean;
  job_name: string | null; job_status: string | null; billed_tokens: Tokens | null;
  error: string | null; model: string | null; region: string | null; step: number | null; total_steps: number | null;
  session_name: string | null; session_state: string | null; ready_sec: number | null; tokens: Tokens;
  artifacts: number; last_metrics: Record<string, number>; cost_usd: number | null;
  eval: { label: string; before: number; after: number; n: number; unit: string } | null;
  started_ts: number | null; ended_ts: number | null; created_ts: number; duration_s: number | null;
  recorded_at: string | null; recording: string | null; files: string[]; metrics: MetricSpec[]; exit_code: number | null;
  variants: Record<string, string | number>; custom_files: string[]; loop: LoopSpec | null; slots: SlotSpec[];
  probes: boolean; probe_example: string | null; mlflow: MlflowWhere | null; mlflow_mode: MlflowMode | null;
  sandbox: { where: SandboxWhere | null; runtime: string | null; platform: string | null; created: number; open: number;
             retries: number; failed_stops: number; closed: boolean } | null;
  /** A replay of a recording that came with the project. */
  recording_shipped: boolean;
}

export type CheckStatus = "pass" | "fail" | "warn" | "unverified";
export interface Check {
  id: string; group: "account" | "machine" | "tracking" | "sandbox" | "mtrl"; title: string; status: CheckStatus; observed: string;
  why: string; fix: { id: string; label: string; detail: string } | null; checked_at: number;
}
export interface Readiness {
  region: string; account: string | null; checked_at: number; checks: Check[]; models: string[];
  live_ready: boolean; live_blockers: string[];
  /** Sandbox scenarios: the training-session checks plus the environment sandbox's. */
  sandbox_ready: boolean; sandbox_blockers: string[];
  mtrl_models: string[]; mtrl_ready: boolean; mtrl_blockers: string[];
  /** The MLflow app live training-session runs publish to; null when none is selected. */
  mlflow_app: { arn: string; name: string | null; status: string | null; usable: boolean } | null;
}

export interface MtrlSelection {
  runtime_arn: string | null; mlflow_app_arn: string | null; job_role_name: string; job_role_arn: string | null;
  output_group: string; checkpoint_group: string;
}
export interface MtrlOptions {
  runtimes: { arn: string; name: string; status: string; updated: string }[];
  mlflow_apps: { arn: string; name: string; status: string }[];
  job_roles: string[]; errors: string[]; selection: MtrlSelection;
}
export interface SandboxOptions {
  runtimes: { arn: string; name: string; status: string; updated: string }[]; errors: string[]; default_name: string;
  code_sha: string; selection: { runtime_arn: string | null; source: "selected" | "default name" | null; default_name: string };
}
export interface SandboxActive { active: number | null; at: number | null; limit: number; limit_source: string; error: string | null }

export interface MtrlJob {
  name: string; arn: string; status: string; secondary: string | null; created: string | null; ended: string | null;
  failure_reason: string | null;
}

export interface AwsContext {
  region: string; account: string | null; caller_arn: string | null; identity_error: string | null;
  role_name: string; role_arn: string | null; model_package_group: string; model_package_group_arn: string | null;
  bucket: string | null; s3_output_path: string | null; supported_regions: string[]; session_prefix: string;
}

export interface Quota { limit: number; limit_source: string; active: number | null; active_names: string[]; error: string | null }

export interface SessionRow {
  name: string; status: string; created: string | null; tokens: Tokens | null; cost_usd: number | null;
  hub_model: string | null; ours: boolean; failure_reason?: string | null;
}

export interface PackageRow {
  arn: string; version: number | null; description: string; created: string | null; run_id: string | null;
  kind: "CHECKPOINT" | "MODEL" | null; s3_uri: string | null; status: string | null;
}

export interface Pricing {
  as_of: string; source: string; unit: string; models: Record<string, Tokens>;
  mtrl: { models: Record<string, Tokens>; basis: string };
  agentcore: { as_of: string; unit: string; regions: Record<string, Record<string, { vcpu_h: number; gb_h: number }>> };
}

export interface Billing { name: string; status: string; tokens: Tokens; cost_usd: number | null; checked_at: number }
