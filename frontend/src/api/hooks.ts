import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./client";
import type {
  AgentcoreUsage, AwsContext, Billing, CostEstimate, DeployTarget, Models, MtrlJob, MtrlOptions, PackageRow, Pricing, Quota, Readiness, Recording,
  RunSummary, SandboxActive, SandboxOptions, Scenario, SessionRow,
} from "./types";

const ACTIVE = new Set(["starting", "running", "stopping"]);

export const useAuthStatus = () =>
  useQuery({ queryKey: ["auth"], queryFn: () => api<{ required: boolean; authenticated: boolean }>("/api/auth/status") });

export const useAwsContext = () =>
  useQuery({ queryKey: ["context"], queryFn: () => api<AwsContext>("/api/context"), staleTime: 60_000 });

export const usePricing = () =>
  useQuery({ queryKey: ["pricing"], queryFn: () => api<Pricing>("/api/pricing"), staleTime: Infinity });

export const useReadiness = () =>
  useQuery({ queryKey: ["readiness"], queryFn: () => api<Readiness>("/api/readiness"), staleTime: 5 * 60_000 });

export const useQuota = () =>
  useQuery({ queryKey: ["quota"], queryFn: () => api<Quota>("/api/quota"), refetchInterval: 30_000 });

export const useScenarios = () =>
  useQuery({
    queryKey: ["scenarios"],
    queryFn: () => api<{ scenarios: Scenario[]; models: Models }>("/api/scenarios"),
  });

export const useRuns = () =>
  useQuery({
    queryKey: ["runs"],
    queryFn: () => api<{ runs: RunSummary[] }>("/api/runs"),
    refetchInterval: (q) => (q.state.data?.runs.some((r) => ACTIVE.has(r.status)) ? 3000 : 15_000),
  });

export const useRun = (id: string) =>
  useQuery({
    queryKey: ["run", id],
    queryFn: () => api<RunSummary>(`/api/runs/${id}`),
    refetchInterval: (q) => (q.state.data && !q.state.data.alive ? false : 3000),
  });

/** What a live launch with these settings would cost, from live runs with the same settings. */
export const useEstimate = (body: { scenario: string; params: Record<string, unknown>; variants: Record<string, unknown>;
                                    files: Record<string, string> } | null) =>
  useQuery({
    queryKey: ["estimate", body], queryFn: () => api<CostEstimate>("/api/estimate", { body: body! }),
    enabled: !!body, placeholderData: keepPreviousData,
  });

/** The same for an existing run's settings. */
export const useRunEstimate = (id: string, enabled: boolean) =>
  useQuery({ queryKey: ["estimate-run", id], queryFn: () => api<CostEstimate>(`/api/runs/${id}/estimate`), enabled });

export const useScenarioFile = (id: string, name: string | null) =>
  useQuery({
    queryKey: ["scenario-file", id, name],
    queryFn: () => api<{ name: string; content: string }>(`/api/scenarios/${id}/files/${name}`),
    enabled: !!name, staleTime: Infinity,
  });

export const useRunFile = (id: string, name: string | null) =>
  useQuery({
    queryKey: ["run-file", id, name],
    queryFn: () => api<{ name: string; content: string }>(`/api/runs/${id}/files/${name}`),
    enabled: !!name,
  });

// Billing counters lag the operations by tens of seconds to minutes, so keep polling for a
// few minutes after the run ends, until the numbers settle.
export const useBilling = (run: RunSummary | undefined) =>
  useQuery({
    queryKey: ["billing", run?.id],
    queryFn: () => api<Billing>(`/api/runs/${run!.id}/billing`),
    enabled: !!run && run.backend === "live" && run.engine !== "mtrl" && !!run.session_name,
    refetchInterval: () => {
      if (!run) return false;
      if (run.alive) return 20_000;
      const ended = run.ended_ts ?? 0;
      return Date.now() / 1000 - ended < 300 ? 30_000 : false;
    },
  });

// AgentCore usage logs arrived about 20 minutes after the sessions ended (2026-11-10; the docs say up to 60), so a
// run's AgentCore cost is read again every few minutes for an hour and a half after it ends.
export const useAgentcoreUsage = (run: RunSummary | undefined) =>
  useQuery({
    queryKey: ["agentcore", run?.id],
    queryFn: () => api<AgentcoreUsage>(`/api/runs/${run!.id}/agentcore`),
    enabled: !!run && run.backend === "live" && (run.engine === "mtrl" ? !!run.job_name : run.sandbox?.where === "agentcore"),
    refetchInterval: () => {
      if (!run) return false;
      if (run.alive) return 60_000;
      return Date.now() / 1000 - (run.ended_ts ?? 0) < 5400 ? 180_000 : false;
    },
    retry: false,
  });

export const useSessions = (scope: "ours" | "all") =>
  useQuery({
    queryKey: ["sessions", scope],
    queryFn: () => api<{ sessions: SessionRow[]; active: number; errors: string[]; describe_limit: number }>(
      `/api/sessions?scope=${scope}`),
    refetchInterval: 20_000,
  });

export const useArtifacts = () =>
  useQuery({
    queryKey: ["artifacts"],
    queryFn: () => api<{ group: string; group_exists: boolean; packages: PackageRow[]; errors: string[] }>("/api/artifacts"),
  });

export const useSandboxOptions = (enabled = true) =>
  useQuery({ queryKey: ["sandbox-options"], queryFn: () => api<SandboxOptions>("/api/sandbox/options"), enabled, staleTime: 60_000 });

/** The runtimes the Setup page deploys; polled every 3 s while one of their jobs runs. */
export const useDeployTargets = () =>
  useQuery({
    queryKey: ["deploy"], queryFn: () => api<{ targets: DeployTarget[] }>("/api/deploy"), staleTime: 30_000,
    refetchInterval: (q) => (q.state.data?.targets.some((t) => t.job?.state === "running") ? 3000 : false),
  });

export const useDeployLog = (id: string, open: boolean, running: boolean) =>
  useQuery({
    queryKey: ["deploy-log", id], queryFn: () => api<{ lines: string[] }>(`/api/deploy/${id}/log`), enabled: open,
    refetchInterval: open && running ? 3000 : false,
  });

export const useSandboxActive = () =>
  useQuery({ queryKey: ["sandbox-active"], queryFn: () => api<SandboxActive>("/api/sandbox/active"), refetchInterval: 60_000 });

export const useMtrlOptions = (enabled = true) =>
  useQuery({ queryKey: ["mtrl-options"], queryFn: () => api<MtrlOptions>("/api/mtrl/options"), enabled, staleTime: 60_000 });

export const useMtrlQuota = () =>
  useQuery({ queryKey: ["mtrl-quota"], queryFn: () => api<Quota>("/api/mtrl/quota"), refetchInterval: 60_000 });

export const useMtrlJobs = () =>
  useQuery({ queryKey: ["mtrl-jobs"], queryFn: () => api<{ jobs: MtrlJob[] }>("/api/mtrl/jobs"), refetchInterval: 30_000 });

export const useRecordings = () =>
  useQuery({ queryKey: ["recordings"], queryFn: () => api<{ recordings: Recording[] }>("/api/recordings") });

export function useInvalidate() {
  const qc = useQueryClient();
  return (...keys: string[]) => keys.forEach((k) => qc.invalidateQueries({ queryKey: [k] }));
}

export function useApiMutation<TBody, TResult>(fn: (body: TBody) => Promise<TResult>, invalidate: string[] = []) {
  const inv = useInvalidate();
  return useMutation({ mutationFn: fn, onSuccess: () => inv(...invalidate) });
}
