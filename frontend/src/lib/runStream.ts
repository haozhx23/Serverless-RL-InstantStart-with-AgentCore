import { useEffect, useReducer } from "react";
import type { Backend, CompareItem, DatasetSplit, Engine, EventOf, MlflowWhere, PolicyCheck, PolicyRow, Rollout, RunEvent, SandboxDist,
  SandboxWhere, Span, Tokens } from "../api/types";

/** The environment sandbox's side of a run, from its `sandbox` events (templates/_shared/sandbox_client.py). */
export interface SandboxView {
  where: SandboxWhere; runtime: string | null; platform: string | null; codeSha: string;
  created: number; stopped: number; open: number; retries: number; calls: number;
  cold: SandboxDist | null;        // first requests of new sessions: the closing summary, else the latest batch
  failedStops: { session: string; error: string }[]; closed: boolean;
}

// Everything the run console draws, reduced from the run's event stream (SSE).
export interface RunView {
  connection: "connecting" | "open" | "ended" | "problem";
  problem: string | null;
  lastSeq: number;
  lastT: number;
  started: EventOf<"run_started"> | null;
  backend: Backend | null;
  engine: Engine;
  job: { name: string; arn: string; status: string; secondary: string | null; failureReason: string | null; mlflowUrl: string | null } | null;
  billed: Tokens | null;                    // MTRL: what the job reports; the only token count there is
  billingPoints: { t: number; tokens: Tokens }[];
  policyCheck: { rows: PolicyRow[]; checks: PolicyCheck[]; n: number } | null;
  dataset: { splits?: Record<string, DatasetSplit>; rollouts?: string; uris?: Record<string, string> };
  session: { state: string | null; name: string | null; readySec: number | null; createdAt: number | null };
  step: number | null;
  totalSteps: number | null;
  stepStarts: { step: number; t: number }[];
  spans: Span[];
  metrics: { step: number; values: Record<string, number> }[];
  rollouts: Record<number, Rollout[]>;
  compare: { label: string; items: CompareItem[] } | null;
  evaluation: EventOf<"eval"> | null;
  artifacts: EventOf<"artifact">[];
  handled: EventOf<"handled">[];
  logs: EventOf<"log">[];
  tokens: Tokens;
  counts: { updates: number; rebinds: number; samples: number; checkpoints: number; models: number };
  finished: EventOf<"run_finished"> | null;
  /** Operations that have started and not ended yet: what the loop map lights up. */
  open: EventOf<"span_start">[];
  probes: EventOf<"probe">[];
  rewardCheck: EventOf<"reward_check"> | null;
  /** Weight versions: the trainer moves with every optim_step, the sampler only when it is rebound. */
  versions: { trainer: number; sampler: number | null };
  apiCalls: number;
  /** Where the run published itself in MLflow: the latest mlflow event, keeping the ids once known. */
  mlflow: MlflowWhere | null;
  sandbox: SandboxView | null;
}

const initial: RunView = {
  connection: "connecting", problem: null, lastSeq: -1, lastT: 0, started: null, backend: null, engine: "training-session",
  job: null, billed: null, billingPoints: [], policyCheck: null, dataset: {},
  session: { state: null, name: null, readySec: null, createdAt: null }, step: null, totalSteps: null,
  stepStarts: [], spans: [], metrics: [], rollouts: {}, compare: null, evaluation: null, artifacts: [],
  handled: [], logs: [], tokens: { prefill: 0, sample: 0, train: 0 },
  counts: { updates: 0, rebinds: 0, samples: 0, checkpoints: 0, models: 0 }, finished: null,
  open: [], probes: [], rewardCheck: null, versions: { trainer: 0, sampler: null }, apiCalls: 0, mlflow: null, sandbox: null,
};

const sameOp = (a: { lane: string; op: string; step: number | null; t0: number }, b: typeof a) =>
  a.lane === b.lane && a.op === b.op && a.step === b.step && Math.abs(a.t0 - b.t0) < 1e-6;

/** API requests a span sent: its own count, or one for an AWS operation that did not say. */
export const callsOf = (s: Span) => s.calls ?? (s.lane === "aws" ? 1 : 0);

type Action =
  | { kind: "events"; events: RunEvent[] }
  | { kind: "open" }
  | { kind: "end" }
  | { kind: "problem"; message: string }
  | { kind: "reset" };

function apply(s: RunView, e: RunEvent): RunView {
  // The stream restarts from the top on reconnect; already-seen events are dropped here.
  if (e.seq <= s.lastSeq) return s;
  const n: RunView = { ...s, lastSeq: e.seq, lastT: Math.max(s.lastT, e.t) };
  switch (e.type) {
    case "run_started":
      return { ...n, started: e, backend: e.backend, totalSteps: e.total_steps, engine: e.engine ?? "training-session" };
    case "job":
      return { ...n, job: { name: e.name, arn: e.arn, status: e.status, secondary: e.secondary,
                            failureReason: e.failure_reason ?? s.job?.failureReason ?? null,
                            mlflowUrl: e.mlflow_url ?? s.job?.mlflowUrl ?? null } };
    case "billing":
      return { ...n, billed: e.tokens, tokens: e.tokens, billingPoints: [...s.billingPoints, { t: e.t, tokens: e.tokens }] };
    case "policy_check":
      return { ...n, policyCheck: { rows: e.rows, checks: e.checks, n: e.n } };
    case "dataset":
      return { ...n, dataset: { ...s.dataset, ...(e.splits ? { splits: e.splits } : {}), ...(e.rollouts ? { rollouts: e.rollouts } : {}),
                                ...(e.uris ? { uris: e.uris } : {}) } };
    case "session":
      return {
        ...n,
        session: {
          state: e.state,
          name: e.name ?? s.session.name,
          readySec: e.ready_sec ?? s.session.readySec,
          createdAt: e.state === "creating" ? e.t : s.session.createdAt,
        },
      };
    case "step":
      return { ...n, step: e.step, totalSteps: e.total, stepStarts: [...s.stepStarts, { step: e.step, t: e.t }] };
    case "span_start":
      return { ...n, open: [...s.open, e] };
    case "span": {
      const tokens = { ...s.tokens };
      if (s.engine !== "mtrl") for (const [k, v] of Object.entries(e.tokens ?? {})) tokens[k as keyof Tokens] += v ?? 0;
      const c = { ...s.counts };
      const training = e.role !== "probe";
      if (e.op === "optim_step" && e.ok) c.updates += 1;
      if (e.op === "save_weights_and_rebind" && e.ok && training) c.rebinds += 1;
      if (e.op === "sample" && e.ok) c.samples += 1;
      const versions = { ...s.versions };
      if (e.op === "optim_step" && e.ok) versions.trainer += 1;
      if (e.op === "save_weights_and_rebind" && e.ok && training) versions.sampler = versions.trainer;
      return { ...n, spans: [...s.spans, e], tokens, counts: c, lastT: Math.max(n.lastT, e.t0 + e.sec),
               open: s.open.filter((o) => !sameOp(o, e)), versions,
               apiCalls: s.apiCalls + (s.engine === "mtrl" ? 0 : callsOf(e)) };
    }
    case "probe":
      return { ...n, probes: [...s.probes, e] };
    case "reward_check":
      return { ...n, rewardCheck: e };
    case "metrics":
      return { ...n, metrics: [...s.metrics, { step: e.step, values: e.values }] };
    case "rollouts":
      return { ...n, rollouts: { ...s.rollouts, [e.step]: e.items } };
    case "compare":
      return { ...n, compare: { label: e.label, items: e.items } };
    case "eval":
      return { ...n, evaluation: e };
    case "artifact": {
      const c = { ...s.counts };
      if (e.kind === "CHECKPOINT") c.checkpoints += 1;
      else c.models += 1;
      return { ...n, artifacts: [...s.artifacts, e], counts: c };
    }
    case "handled":
      return s.handled.some((h) => h.code === e.code)
        ? { ...n, handled: s.handled.map((h) => (h.code === e.code ? e : h)) }
        : { ...n, handled: [...s.handled, e] };
    case "log":
      return { ...n, logs: [...s.logs, e] };
    case "mlflow":
      return { ...n, mlflow: { state: e.state, tracking_uri: e.tracking_uri, experiment: e.experiment,
                               experiment_id: e.experiment_id ?? s.mlflow?.experiment_id ?? null,
                               run_id: e.run_id ?? s.mlflow?.run_id ?? null, status: e.status ?? null, error: e.error ?? null,
                               published: e.published ?? null } };
    case "sandbox": {
      if (e.state === "ready") {
        return { ...n, sandbox: { where: e.where, runtime: e.runtime, platform: e.platform, codeSha: e.code_sha, created: 0,
                                  stopped: 0, open: 0, retries: 0, calls: 0, cold: null, failedStops: [], closed: false } };
      }
      const sb = s.sandbox;
      if (!sb) return n;
      if (e.state === "batch") {
        return { ...n, sandbox: { ...sb, created: e.created, open: e.open, retries: sb.retries + e.retries, calls: sb.calls + e.calls,
                                  cold: e.cold_start ?? sb.cold } };
      }
      if (e.state === "stopped") {
        return { ...n, sandbox: { ...sb, open: e.open, stopped: sb.stopped + e.sessions, failedStops: [...sb.failedStops, ...e.failed] } };
      }
      return { ...n, sandbox: { ...sb, created: e.created, stopped: e.stopped, open: 0, retries: e.retries, calls: e.calls,
                                cold: e.cold_start, failedStops: [...sb.failedStops, ...e.failed], closed: true } };
    }
    case "run_finished":
      return { ...n, finished: e, open: [] };
    default:
      return n;
  }
}

function reducer(s: RunView, a: Action): RunView {
  switch (a.kind) {
    case "events":
      return a.events.reduce(apply, s.connection === "connecting" ? { ...s, connection: "open" } : s);
    case "open":
      return { ...s, connection: "open" };
    case "end":
      return { ...s, connection: "ended" };
    case "problem":
      return { ...s, connection: "problem", problem: a.message };
    case "reset":
      return initial;
  }
}

/** Subscribe to /api/runs/:id/events. Events are batched per animation frame. */
export function useRunStream(runId: string): RunView {
  const [state, dispatch] = useReducer(reducer, initial);

  useEffect(() => {
    dispatch({ kind: "reset" });
    const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`, { withCredentials: true });
    let queue: RunEvent[] = [];
    let frame = 0;
    const flush = () => {
      frame = 0;
      const batch = queue;
      queue = [];
      if (batch.length) dispatch({ kind: "events", events: batch });
    };
    source.onopen = () => dispatch({ kind: "open" });
    source.onmessage = (m) => {
      queue.push(JSON.parse(m.data) as RunEvent);
      if (!frame) frame = requestAnimationFrame(flush);
    };
    source.addEventListener("end", () => {
      flush();
      dispatch({ kind: "end" });
      source.close();
    });
    source.addEventListener("problem", (m) => {
      flush();
      dispatch({ kind: "problem", message: JSON.parse((m as MessageEvent).data).error });
      source.close();
    });
    return () => {
      if (frame) cancelAnimationFrame(frame);
      source.close();
    };
  }, [runId]);

  return state;
}

/** MTRL: cost at each billing report from the job (cumulative counters). */
export function billingTrace(points: { t: number; tokens: Tokens }[], price: Tokens | undefined): { t: number; usd: number }[] {
  if (!price || !points.length) return [];
  return [{ t: 0, usd: 0 }, ...points.map((p) => ({
    t: p.t, usd: (p.tokens.prefill * price.prefill + p.tokens.sample * price.sample + p.tokens.train * price.train) / 1e6,
  }))];
}

/** Cumulative estimated cost after each AWS span, for the meter trace under the tape. */
export function costTrace(spans: Span[], price: Tokens | undefined): { t: number; usd: number }[] {
  if (!price) return [];
  let total = 0;
  const pts: { t: number; usd: number }[] = [{ t: 0, usd: 0 }];
  for (const s of [...spans].sort((a, b) => a.t0 + a.sec - (b.t0 + b.sec))) {
    if (!s.tokens) continue;
    total += ((s.tokens.prefill ?? 0) * price.prefill + (s.tokens.sample ?? 0) * price.sample
      + (s.tokens.train ?? 0) * price.train) / 1e6;
    pts.push({ t: s.t0 + s.sec, usd: total });
  }
  return pts;
}
