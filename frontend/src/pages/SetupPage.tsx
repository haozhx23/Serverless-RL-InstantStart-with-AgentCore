import { Alert, App, Button, Select, Tooltip } from "antd";
import { CheckCircleFilled, CloseCircleFilled, ExclamationCircleFilled, ExportOutlined, MinusCircleOutlined, ReloadOutlined } from "@ant-design/icons";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { Link } from "react-router";
import { api } from "../api/client";
import { useAwsContext, useInvalidate, useMtrlOptions, useMtrlQuota, useQuota, useReadiness, useSandboxOptions } from "../api/hooks";
import type { Check, CheckStatus, MtrlSelection, Quota, Readiness } from "../api/types";
import { Linkified, PageHeader, QuotaPips, Section, useLimitSource } from "../components/Bits";
import { timeOfDay } from "../lib/format";

const STATUS: Record<CheckStatus, { icon: React.ReactNode; color: string }> = {
  pass: { icon: <CheckCircleFilled />, color: "var(--live)" },
  fail: { icon: <CloseCircleFilled />, color: "var(--error)" },
  warn: { icon: <ExclamationCircleFilled />, color: "var(--warning)" },
  unverified: { icon: <MinusCircleOutlined />, color: "var(--text-3)" },
};

const GROUPS: Check["group"][] = ["account", "machine", "tracking", "sandbox", "mtrl"];

export default function SetupPage() {
  const { t } = useTranslation();
  const { message } = App.useApp();
  const qc = useQueryClient();
  const invalidate = useInvalidate();
  const readiness = useReadiness();
  const sessionQuota = useQuota();
  const mtrlQuota = useMtrlQuota();
  const ctx = useAwsContext().data;
  const [busy, setBusy] = useState<string | null>(null);

  const setResult = (r: Readiness) => qc.setQueryData(["readiness"], r);
  const rerun = async () => {
    setBusy("rerun");
    try { setResult(await api<Readiness>("/api/readiness/run", { method: "POST" })); invalidate("quota", "mtrl-quota", "scenarios"); }
    catch (e) { message.error((e as Error).message); }
    finally { setBusy(null); }
  };
  const fix = async (c: Check) => {
    setBusy(c.id);
    try {
      const r = await api<{ message: string; readiness: Readiness }>(`/api/readiness/fix/${c.fix!.id}`, { method: "POST" });
      setResult(r.readiness);
      message.success(r.message);
      invalidate("artifacts", "scenarios");
    } catch (e) { message.error((e as Error).message); }
    finally { setBusy(null); }
  };
  const changeRegion = async (region: string) => {
    setBusy("region");
    try {
      await api("/api/context", { method: "PUT", body: { region } });
      await qc.invalidateQueries();
    } catch (e) { message.error((e as Error).message); }
    finally { setBusy(null); }
  };

  const data = readiness.data;
  let n = 0;
  return (
    <>
      <PageHeader crumbs={t("setup.crumbs")} title={t("shell.nav.setup")}
                  extra={<>
                    <span className="secondary" style={{ fontSize: 13 }}>{t("setup.region")}</span>
                    <Select value={ctx?.region} loading={busy === "region"} onChange={changeRegion} style={{ width: 140 }}
                            options={(ctx?.supported_regions ?? []).map((r) => ({ value: r, label: r }))} />
                    <Button icon={<ReloadOutlined />} loading={busy === "rerun"} onClick={rerun}>{t("setup.rerun")}</Button>
                  </>}>
        <p className="secondary" style={{ margin: "8px 0 0", fontSize: 13 }}>
          {data ? t("setup.checkedAt", { time: timeOfDay(data.checked_at), region: data.region })
            : readiness.isLoading ? t("setup.checking") : null}
        </p>
      </PageHeader>
      <div className="page-body" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {readiness.error && <Alert type="error" showIcon title={(readiness.error as Error).message} />}
        {data && (
          <Alert showIcon type={data.live_ready && data.mtrl_ready ? "success" : "warning"}
                 title={<>
                   <div>{data.live_ready ? t("setup.tsReady") : t("setup.tsBlocked")}</div>
                   {data.live_ready && !data.sandbox_ready && <div>{t("setup.sandboxBlocked")}</div>}
                   <div>{data.mtrl_ready ? t("setup.mtrlReady") : t("setup.mtrlBlocked")}
                   </div>
                 </>} />
        )}
        {GROUPS.map((g) => {
          const checks = data?.checks.filter((c) => c.group === g) ?? [];
          if (!checks.length) return null;
          return (
            <Section key={g} title={t(`setup.group.${g}`)} hint={t(`setup.group.${g}Hint`)}>
              {g === "tracking" && <MlflowPicker current={data?.mlflow_app ?? null} onChanged={setResult} />}
              {g === "sandbox" && <SandboxPicker onChanged={setResult} />}
              {g === "mtrl" && <MtrlPickers onChanged={setResult} />}
              {checks.map((c) => {
                n += 1;
                const s = STATUS[c.status];
                return (
                  <div className="check-row" key={c.id}>
                    <span className="idx num">{g === "account" ? n : ""}</span>
                    <span style={{ color: s.color, display: "inline-flex", gap: 6, alignItems: "center", fontWeight: 500 }}>{s.icon}{t(`setup.status.${c.status}`)}</span>
                    <span>
                      <div style={{ fontWeight: 600 }}>{c.title}</div>
                      {c.id === "quota" || c.id === "mtrl-quota"
                        ? <QuotaUsage query={c.id === "quota" ? sessionQuota : mtrlQuota} observed={c.observed} />
                        : <Linkified className="observed mono" text={c.observed} />}
                    </span>
                    <span className="why">{c.why}</span>
                    <span className="action">
                      {c.fix && (
                        <Tooltip title={c.fix.detail}>
                          <Button size="small" type={c.status === "fail" ? "primary" : "default"} loading={busy === c.id}
                                  onClick={() => fix(c)}>{c.fix.label}</Button>
                        </Tooltip>
                      )}
                    </span>
                  </div>
                );
              })}
            </Section>
          );
        })}
        {ctx && (
          <Section title={t("setup.names.title")} hint={t("setup.names.hint")}>
            <dl className="kv">
              <dt>{t("setup.names.role")}</dt><dd className="mono">{ctx.role_arn ?? ctx.role_name}</dd>
              <dt>{t("setup.names.group")}</dt><dd className="mono">{ctx.model_package_group_arn ?? ctx.model_package_group}</dd>
              <dt>{t("setup.names.s3")}</dt><dd className="mono">{ctx.s3_output_path ?? "—"}</dd>
              <dt>{t("setup.names.sessions")}</dt><dd className="mono">{ctx.session_prefix}-&lt;8 hex&gt;</dd>
              <dt>{t("setup.names.sandbox")}</dt><dd className="mono">srl_instantstart_sandbox · SRL-InstantStart-SandboxRuntime · ECR serverless-rl-instantstart-sandbox</dd>
              <dt>{t("setup.names.mtrlCheckpoints")}</dt><dd className="mono">serverless-rl-instantstart-mtrl-checkpoints</dd>
              <dt>{t("setup.names.mtrlData")}</dt><dd className="mono">{ctx.bucket ? `s3://${ctx.bucket}/serverless-rl-instantstart/mtrl/<run>/` : "—"}</dd>
              <dt>{t("setup.names.mlflowExperiments")}</dt><dd className="mono">serverless-rl-instantstart/&lt;scenario&gt; · MTRL: serverless-rl-instantstart</dd>
            </dl>
          </Section>
        )}
      </div>
    </>
  );
}

/** Current occupancy belongs next to the corresponding concurrency check, not in the global header. */
function QuotaUsage({ query, observed }: { query: UseQueryResult<Quota, Error>; observed: string }) {
  const { t } = useTranslation();
  const limitSource = useLimitSource();
  const quota = query.error ? undefined : query.data;
  const detail = quota ? t("setup.quota.source", { source: limitSource(quota.limit_source) })
    : query.error?.message || observed;
  return (
    <>
      <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 8, marginTop: 6 }}>
        {quota && <QuotaPips large limit={quota.limit} active={quota.active} />}
        <span className="num">{quota ? `${quota.active ?? "?"}/${quota.limit}` : query.isPending ? "…" : "—"}</span>
        <Link to="/capacity" style={{ fontSize: 12 }}>{t("setup.quota.manage")}</Link>
      </div>
      <Linkified className="observed mono" text={detail} />
      {quota?.error && <Linkified className="observed mono" text={quota.error} />}
    </>
  );
}

/** Saves picks for this region (stored with the MTRL selections, which have always held the MLflow app). */
function useSelection(onChanged: (r: Readiness) => void) {
  const { message } = App.useApp();
  const invalidate = useInvalidate();
  const opts = useMtrlOptions();
  const [busy, setBusy] = useState(false);
  const save = async (patch: Partial<MtrlSelection>) => {
    setBusy(true);
    try {
      const r = await api<{ readiness: Readiness }>("/api/mtrl/selection", { method: "PUT", body: patch });
      onChanged(r.readiness);
      invalidate("mtrl-options", "scenarios");
    } catch (e) { message.error((e as Error).message); }
    finally { setBusy(false); }
  };
  return { opts, sel: opts.data?.selection, busy, save };
}

/** The MLflow app both engines log to: training-session runs publish there, MTRL jobs require it. */
function MlflowPicker({ current, onChanged }: { current: Readiness["mlflow_app"]; onChanged: (r: Readiness) => void }) {
  const { t } = useTranslation();
  let mlflowNote = t("setup.pickers.mlflowNote");
  const { opts, sel, busy, save } = useSelection(onChanged);
  if (opts.error) return <Alert type="warning" showIcon title={(opts.error as Error).message} style={{ marginBottom: 12 }} />;
  // The options list IAM roles too and can take a while; until then, show the app the checks already named.
  const apps = opts.data?.mlflow_apps ?? (current ? [{ arn: current.arn, name: current.name ?? current.arn, status: current.status ?? "?" }] : []);
  const value = sel?.mlflow_app_arn ?? current?.arn;
  return (
    <div className="pickers">
      <label>
        <span>{t("setup.pickers.mlflow")}</span>
        <Select loading={opts.isLoading || busy} value={value ?? undefined} placeholder={t("setup.pickers.mlflowPlaceholder")}
                onChange={(v) => save({ mlflow_app_arn: v })} style={{ width: "100%" }}
                options={apps.map((a) => ({ value: a.arn, label: `${a.name} · ${a.status}` }))} />
      </label>
      <span className="picker-action">
        {value && <a href="/api/mlflow/open" target="_blank" rel="noreferrer"><ExportOutlined /> {t("setup.pickers.openMlflow")}</a>}
      </span>
      <p className="muted" style={{ fontSize: 12, margin: 0, gridColumn: "1 / -1" }}>{mlflowNote}</p>
    </div>
  );
}

/** The environment sandbox runtime: deployed from the host (scripts/deploy_sandbox.py), picked here. */
function SandboxPicker({ onChanged }: { onChanged: (r: Readiness) => void }) {
  const { t } = useTranslation();
  const { message } = App.useApp();
  const invalidate = useInvalidate();
  const opts = useSandboxOptions();
  const [busy, setBusy] = useState(false);
  if (opts.error) return <Alert type="warning" showIcon title={(opts.error as Error).message} style={{ marginBottom: 12 }} />;
  const sel = opts.data?.selection;
  const save = async (runtime_arn: string | null) => {
    setBusy(true);
    try {
      const r = await api<{ readiness: Readiness }>("/api/sandbox/selection", { method: "PUT", body: { runtime_arn } });
      onChanged(r.readiness);
      invalidate("sandbox-options", "scenarios");
    } catch (e) { message.error((e as Error).message); }
    finally { setBusy(false); }
  };
  return (
    <div className="pickers">
      <label>
        <span>{t("setup.pickers.sandbox")}</span>
        <Select loading={opts.isLoading || busy} value={sel?.runtime_arn ?? undefined} placeholder={t("setup.pickers.sandboxPlaceholder")}
                onChange={(v) => save(v)} style={{ width: "100%" }}
                options={(opts.data?.runtimes ?? []).map((r) => ({ value: r.arn, label: `${r.name} · ${r.status}` }))} />
      </label>
      <span className="picker-action">
        {sel?.source === "selected" && <a onClick={() => save(null)}>{t("setup.pickers.sandboxDefault", { name: sel.default_name })}</a>}
        {sel?.source === "default name" && <span className="muted" style={{ fontSize: 12 }}>{t("setup.pickers.sandboxByName")}</span>}
      </span>
      {opts.data?.errors.map((e) => <Alert key={e} type="warning" showIcon title={e} />)}
      <p className="muted" style={{ fontSize: 12, margin: 0, gridColumn: "1 / -1" }}>
        {t("setup.pickers.sandboxNote", { sha: opts.data?.code_sha ?? "…" })}
      </p>
    </div>
  );
}

/** The MTRL resources that are picked rather than created: they usually exist already. */
function MtrlPickers({ onChanged }: { onChanged: (r: Readiness) => void }) {
  const { t } = useTranslation();
  const { opts, sel, busy, save } = useSelection(onChanged);
  if (opts.error) return <Alert type="warning" showIcon title={(opts.error as Error).message} style={{ marginBottom: 12 }} />;
  const roles = Array.from(new Set([...(opts.data?.job_roles ?? []), ...(sel ? [sel.job_role_name] : [])]));
  return (
    <div className="pickers">
      <label>
        <span>{t("setup.pickers.runtime")}</span>
        <Select loading={opts.isLoading || busy} value={sel?.runtime_arn ?? undefined} placeholder={t("setup.pickers.runtimePlaceholder")}
                onChange={(v) => save({ runtime_arn: v })} style={{ width: "100%" }}
                options={(opts.data?.runtimes ?? []).map((r) => ({ value: r.arn, label: `${r.name} · ${r.status}` }))} />
      </label>
      <label>
        <span>{t("setup.pickers.role")}</span>
        <Select loading={opts.isLoading || busy} value={sel?.job_role_name} onChange={(v) => save({ job_role_name: v })}
                style={{ width: "100%" }} options={roles.map((r) => ({ value: r, label: r }))} />
      </label>
      {opts.data?.errors.map((e) => <Alert key={e} type="warning" showIcon title={e} />)}
      <p className="muted" style={{ fontSize: 12, margin: 0, gridColumn: "1 / -1" }}>
        {t("setup.pickers.note")}
      </p>
    </div>
  );
}
