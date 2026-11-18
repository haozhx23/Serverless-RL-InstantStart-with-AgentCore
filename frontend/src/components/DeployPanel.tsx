import { Alert, App, Button, Modal, Tooltip } from "antd";
import {
  CheckCircleFilled, CloseCircleFilled, ExclamationCircleFilled, LoadingOutlined, MinusCircleOutlined,
} from "@ant-design/icons";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router";
import { api } from "../api/client";
import { useAwsContext, useDeployLog, useDeployTargets, useInvalidate } from "../api/hooks";
import type { DeployJob, DeployStep, DeployTarget } from "../api/types";
import { seconds, timeOfDay } from "../lib/format";

// Deploying a runtime from the Setup page (docs/ui-design.md §6.3): the environment sandbox or an MTRL
// scenario's agent. The two buttons are always there with the same labels, whatever is deployed; what is
// deployed, and the latest job, are written next to them. The launch buttons of the scenarios never turn
// into deploy buttons: a scenario's drawer shows the compact form of this panel instead (§6.1).

const STEP_ICON: Record<DeployStep["state"], React.ReactNode> = {
  pending: <MinusCircleOutlined style={{ color: "var(--text-3)" }} />,
  running: <LoadingOutlined style={{ color: "var(--link)" }} />,
  done: <CheckCircleFilled style={{ color: "var(--live)" }} />,
  failed: <CloseCircleFilled style={{ color: "var(--error)" }} />,
  skipped: <MinusCircleOutlined style={{ color: "var(--text-3)" }} />,
};

/** The deploy target a scenario depends on, if any: the sandbox, or its own agent. */
export function deployTargetOf(s: { id: string; sandbox?: unknown; agent?: unknown }): string | null {
  if (s.sandbox) return "sandbox";
  if (s.agent) return `agent-${s.id}`;
  return null;
}

const took = (job: { started: number | null; ended: number | null }) =>
  job.started && job.ended ? seconds(job.ended - job.started) : null;

/** Whether the runtime matches this checkout, in one line. */
function useCondition() {
  const { t } = useTranslation();
  return (target: DeployTarget) => {
    const d = target.deployed;
    switch (target.condition) {
      case "missing": return { tone: "warn" as const, text: t("setup.deploy.condition.missing", { name: target.runtime_name, region: target.region }) };
      case "busy": return { tone: "busy" as const, text: t("setup.deploy.condition.busy", { status: d!.status }) };
      case "failed": return { tone: "fail" as const, text: t("setup.deploy.condition.failed", { status: d!.status }) };
      case "current": return { tone: "ok" as const, text: t("setup.deploy.condition.current") };
      case "outdated": return { tone: "fail" as const, text: t("setup.deploy.condition.outdated") };
      case "unrecorded": return { tone: "warn" as const, text: t("setup.deploy.condition.unrecorded") };
      default: return { tone: "warn" as const, text: target.error ?? "…" };
    }
  };
}

const TONE_ICON = {
  ok: <CheckCircleFilled style={{ color: "var(--live)" }} />,
  warn: <ExclamationCircleFilled style={{ color: "var(--warning)" }} />,
  fail: <CloseCircleFilled style={{ color: "var(--error)" }} />,
  busy: <LoadingOutlined style={{ color: "var(--link)" }} />,
};

/** Start a deploy or a delete after the user has read what it does. */
function useStart(target: DeployTarget | undefined) {
  const { t } = useTranslation();
  const { modal, message } = App.useApp();
  const invalidate = useInvalidate();
  const account = useAwsContext().data?.account;
  return (action: "deploy" | "delete") => {
    if (!target) return;
    const v = { ...target, account: account ?? "?",
                prefix: `s3://${target.bucket}/serverless-rl-instantstart/image-builds/${target.repository}/` };
    const kind = target.kind === "sandbox" ? "sandbox" : "agent";
    modal.confirm({
      width: 640,
      title: t(action === "deploy" ? "setup.deploy.confirmDeploy" : "setup.deploy.confirmDelete", { name: target.runtime_name }),
      okText: t(action === "deploy" ? "setup.deploy.deploy" : "setup.deploy.delete"),
      okButtonProps: action === "delete" ? { danger: true } : undefined,
      cancelText: t("setup.deploy.cancel"),
      content: action === "deploy" ? (
        <div className="deploy-confirm">
          <p>{t("setup.deploy.creates", v)}</p>
          <ul>
            <li>{t("setup.deploy.res.role", v)} <span className="mono">{target.role_name}</span></li>
            <li>{t("setup.deploy.res.builder")} <span className="mono">{target.build_project}</span> · <span className="mono">{target.build_role}</span></li>
            <li>{t("setup.deploy.res.repository")} <span className="mono">{target.repository}</span></li>
            <li>{t("setup.deploy.res.runtime", v)} <span className="mono">{target.runtime_name}</span></li>
            {target.usage_log_group && <li>{t("setup.deploy.res.usage")} <span className="mono">{target.usage_log_group}</span></li>}
            <li>{t("setup.deploy.res.source")} <span className="mono">{v.prefix}</span></li>
          </ul>
          <p>{t(`setup.deploy.after.${kind}`, v)}</p>
          <p>{t(`setup.deploy.time.${kind}`)} {t(`setup.deploy.cost.${kind}`)}</p>
          <details>
            <summary>{t("setup.deploy.permissions")}</summary>
            <p className="mono" style={{ fontSize: 11.5 }}>
              {Object.entries(target.permissions).map(([svc, actions]) => `${svc}: ${actions.join(", ")}`).join(" · ")}
            </p>
          </details>
        </div>
      ) : (
        <div className="deploy-confirm">
          <p>{t("setup.deploy.deletes", v)}</p>
          <ul>
            <li>{t("setup.deploy.res.runtime", v)} <span className="mono">{target.runtime_name}</span></li>
            <li>{t("setup.deploy.res.repository")} <span className="mono">{target.repository}</span></li>
            <li>{t("setup.deploy.res.role", v)} <span className="mono">{target.role_name}</span></li>
            {target.usage_log_group && <li>{t("setup.deploy.res.usageDelivery")}</li>}
            <li>{t("setup.deploy.res.builderIfLast")} <span className="mono">{target.build_project}</span></li>
            <li>{t("setup.deploy.res.source")} <span className="mono">{v.prefix}</span></li>
          </ul>
          <p>{t(`setup.deploy.deleteAfter.${kind}`)} {t("setup.deploy.deleteTime")}</p>
        </div>
      ),
      onOk: async () => {
        try {
          await api(`/api/deploy/${target.id}`, { method: "POST", body: { action } });
          invalidate("deploy");
        } catch (e) {
          message.error((e as Error).message);
        }
      },
    });
  };
}

function LogModal({ target, onClose }: { target: DeployTarget; onClose: () => void }) {
  const { t } = useTranslation();
  const log = useDeployLog(target.id, true, target.job?.state === "running");
  const pre = useRef<HTMLPreElement>(null);
  useEffect(() => { pre.current?.scrollTo({ top: pre.current.scrollHeight }); }, [log.data]);
  return (
    <Modal open width={960} onCancel={onClose} footer={null} title={t("setup.deploy.logTitle", { name: target.runtime_name })}>
      <pre ref={pre} className="deploy-log mono">{log.data?.lines.join("\n") || (log.isLoading ? "…" : t("setup.deploy.logEmpty"))}</pre>
    </Modal>
  );
}

function JobView({ target, job }: { target: DeployTarget; job: DeployJob }) {
  const { t } = useTranslation();
  const [log, setLog] = useState(false);
  const title = job.state === "running" ? t(`setup.deploy.job.running.${job.action}`)
    : t(`setup.deploy.job.${job.state}.${job.action}`, { time: timeOfDay(job.ended ?? job.started), took: took(job) ?? "—" });
  return (
    <div className="deploy-job">
      <div className="deploy-job-head">
        <span>{title}</span>
        <button type="button" className="linklike" onClick={() => setLog(true)}>{t("setup.deploy.showLog")}</button>
      </div>
      {job.error && <Alert type="error" showIcon title={job.error} style={{ margin: "4px 0 8px" }} />}
      <ol className="deploy-steps">
        {job.steps.map((s) => (
          <li key={s.id} className={s.state}>
            <span className="icon">{STEP_ICON[s.state]}</span>
            <span className="name">{t(`setup.deploy.steps.${s.id}`)}</span>
            <span className="time num">{s.state === "running" ? "" : took(s) ?? ""}</span>
            <span className="detail">{t(`setup.deploy.stepState.${s.state}`)}</span>
          </li>
        ))}
      </ol>
      {log && <LogModal target={target} onClose={() => setLog(false)} />}
    </div>
  );
}

/** Refresh what depends on a runtime once one of its jobs ends. */
function useAfterJob(target: DeployTarget | undefined) {
  const invalidate = useInvalidate();
  const last = useRef<string | undefined>(undefined);
  const state = target?.job ? `${target.job.id}:${target.job.state}` : undefined;
  useEffect(() => {
    if (last.current?.endsWith(":running") && state && !state.endsWith(":running")) {
      invalidate("readiness", "sandbox-options", "mtrl-options", "scenarios");
    }
    last.current = state;
  }, [state]);  // eslint-disable-line react-hooks/exhaustive-deps
}

function Buttons({ target, compact }: { target: DeployTarget | undefined; compact?: boolean }) {
  const { t } = useTranslation();
  const ctx = useAwsContext().data;
  const start = useStart(target);
  const running = target?.job?.state === "running" ? target.job.action : null;
  let why: string | null = null;
  if (!ctx?.account) why = t("setup.deploy.noIdentity");
  else if (target?.blocked_by.length) why = t("setup.deploy.blocked", { runs: target.blocked_by.join(", ") });
  else if (running) why = t("setup.deploy.alreadyRunning");
  const disabled = !target || !!why;
  return (
    <span className="deploy-actions">
      <Tooltip title={why}>
        <Button type="primary" size={compact ? "small" : "middle"} disabled={disabled} loading={running === "deploy"}
                onClick={() => start("deploy")}>{t("setup.deploy.deploy")}</Button>
      </Tooltip>
      {!compact && (
        <Tooltip title={why}>
          <Button danger disabled={disabled} loading={running === "delete"} onClick={() => start("delete")}>{t("setup.deploy.delete")}</Button>
        </Tooltip>
      )}
    </span>
  );
}

/** The full panel: in the Environment sandbox and MTRL sections of the Setup page. */
export default function DeployPanel({ targetId, label }: { targetId: string; label?: string }) {
  const { t } = useTranslation();
  const targets = useDeployTargets();
  const target = targets.data?.targets.find((x) => x.id === targetId);
  const condition = useCondition();
  useAfterJob(target);
  if (targets.error) return <Alert type="warning" showIcon title={(targets.error as Error).message} style={{ marginBottom: 12 }} />;
  const c = target ? condition(target) : null;
  const kind = target?.kind === "agent" ? "agent" : "sandbox";
  return (
    <div className="deploy-panel" id={`deploy-${targetId}`}>
      <div className="deploy-head">
        <div className="deploy-what">
          <div className="deploy-title">
            {label ?? t(`setup.deploy.title.${kind}`)} <span className="mono">{target?.runtime_name ?? "…"}</span>
          </div>
          {c && <div className="deploy-state">{TONE_ICON[c.tone]} <span>{c.text}</span></div>}
          {target?.error && target.condition && <div className="observed mono">{target.error}</div>}
        </div>
        <Buttons target={target} />
      </div>
      {target?.job && <JobView target={target} job={target.job} />}
      <p className="muted deploy-note">{t(`setup.deploy.note.${kind}`)}</p>
    </div>
  );
}

/** One line in a scenario's launch drawer: what the scenario's live runs need deployed, and the same button. */
export function DeployLine({ targetId }: { targetId: string }) {
  const { t } = useTranslation();
  const targets = useDeployTargets();
  const target = targets.data?.targets.find((x) => x.id === targetId);
  const condition = useCondition();
  useAfterJob(target);
  const c = target ? condition(target) : null;
  const job = target?.job;
  const step = job?.state === "running" ? job.steps.find((s) => s.state === "running") : null;
  return (
    <div className="deploy-line">
      <span className="deploy-line-text">
        <strong>{t(`setup.deploy.title.${target?.kind === "agent" ? "agent" : "sandbox"}`)}</strong>{" "}
        {c && <>{TONE_ICON[c.tone]} {c.text.replace(/[。.]$/, "")}</>}
        {step && <> · {t("setup.deploy.nowAt", { step: t(`setup.deploy.steps.${step.id}`) })}</>}
        {job && job.state === "failed" && <> · <span style={{ color: "var(--error)" }}>{t("setup.deploy.lastFailed")}</span></>}
        {" "}<Link to={`/setup#deploy-${targetId}`}>{t("setup.deploy.details")}</Link>
      </span>
      <Buttons target={target} compact />
    </div>
  );
}
