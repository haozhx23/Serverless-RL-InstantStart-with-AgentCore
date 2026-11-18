import { Alert, App, Button, Empty, Progress, Segmented, Select, Table, Tabs, Tag, Tooltip } from "antd";
import { CheckOutlined, CloseOutlined, DownloadOutlined, ExclamationCircleOutlined, ExportOutlined, SaveOutlined, StopOutlined } from "@ant-design/icons";
import { useEffect, useMemo, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Link, useParams } from "react-router";
import { api } from "../api/client";
import { useAgentcoreUsage, useBilling, useInvalidate, usePricing, useReadiness, useRun, useRunFile } from "../api/hooks";
import type { CompareItem, PolicyRow, Rollout, RunSummary } from "../api/types";
import CostMeter from "../components/CostMeter";
import MetricChart from "../components/MetricChart";
import { Ident, LaneKey, Linkified, PageHeader, Section } from "../components/Bits";
import { ProvenanceBadge, RunStatusTag } from "../components/Provenance";
import SplitTape, { stepBreakdown } from "../components/SplitTape";
import LoopPanel from "../components/LoopPanel";
import AskModel from "../components/AskModel";
import { AgentcoreLine, SandboxCard } from "../components/SandboxCard";
import { int, opLabel, seconds, usd } from "../lib/format";
import { billingTrace, useRunStream, type RunView } from "../lib/runStream";
import { useEventText } from "../i18n/content";

function useNow(active: boolean) {
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => {
    if (!active) return;
    const h = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(h);
  }, [active]);
  return now;
}

export default function RunConsolePage() {
  const { t } = useTranslation();
  const te = useEventText();
  const { id = "" } = useParams();
  const { message } = App.useApp();
  const invalidate = useInvalidate();
  const summaryQ = useRun(id);
  const run = summaryQ.data;
  const view = useRunStream(id);
  const pricing = usePricing().data;
  const billing = useBilling(run);
  const agentcore = useAgentcoreUsage(run);
  const alive = !!run?.alive;
  const [tapeMode, setTapeMode] = useState<"follow" | "whole">("whole");
  const [view3, setView3] = useState<"loop" | "time" | "table">("loop");
  useEffect(() => { if (alive) setTapeMode("follow"); }, [alive]);
  useEffect(() => { if (view.finished) { setTapeMode("whole"); invalidate("run", "runs", "quota"); } },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [view.finished]);

  if (summaryQ.error) {
    return <div className="page-body"><Alert type="error" showIcon title={(summaryQ.error as Error).message} /></div>;
  }
  const backend = view.backend ?? run?.backend ?? null;
  const model = view.started?.model ?? run?.model ?? null;
  const mtrl = (view.started ? view.engine : run?.engine) === "mtrl";
  const price = model ? (mtrl ? pricing?.mtrl.models[model] : pricing?.models[model]) : undefined;
  let tapePrice = price;
  const trace = mtrl ? billingTrace(view.billingPoints, price) : undefined;

  const stop = async () => {
    try {
      await api(`/api/runs/${id}/stop`, { method: "POST" });
      message.info(mtrl ? t("console.toast.stoppingJob") : t("console.toast.stoppingSession"));
      invalidate("run");
    } catch (e) { message.error((e as Error).message); }
  };
  const record = async () => {
    try {
      const r = await api<{ name: string }>(`/api/runs/${id}/record`, { method: "POST" });
      message.success(t("console.toast.recorded", { name: r.name }));
      invalidate("scenarios", "recordings");
    } catch (e) { message.error((e as Error).message); }
  };

  return (
    <>
      <PageHeader
        crumbs={<><Link to="/runs">{t("shell.nav.runs")}</Link> / <span className="mono">{id}</span></>}
        title={run?.title ?? "…"}
        extra={<>
          <ProvenanceBadge backend={backend} recordedAt={view.started?.recorded_at ?? run?.recorded_at} running={alive} />
          {run && <RunStatusTag status={run.status} />}
          {alive && <Button danger icon={<StopOutlined />} onClick={stop} disabled={run?.status === "stopping"}>{t("console.action.stop")}</Button>}
          <Tooltip title={t("console.action.exportTip")}>
            <Button icon={<DownloadOutlined />} href={`/api/runs/${id}/export`}>{t("console.action.export")}</Button>
          </Tooltip>
          {backend === "live" && run?.status === "completed" && (
            <Tooltip title={t("console.action.recordTip")}>
              <Button icon={<SaveOutlined />} onClick={record}>{t("console.action.record")}</Button>
            </Tooltip>
          )}
        </>}>
        {mtrl ? <JobStrip view={view} run={run} alive={alive} /> : <RunStrip view={view} run={run} alive={alive} />}
      </PageHeader>

      <div className="page-body">
        <Problems view={view} run={run} />
        <Section title={mtrl ? t("console.loopSection.titleJob") : t("console.loopSection.title")}
                 hint={view3 === "loop" ? (mtrl ? t("console.loopSection.hintLoopJob") : t("console.loopSection.hintLoop"))
                   : view3 === "time" ? (mtrl ? t("console.loopSection.hintTimeJob") : t("console.loopSection.hintTime"))
                   : t("console.loopSection.hintTable")}
                 extra={<div style={{ display: "flex", gap: 8 }}>
                   {view3 === "time" && (
                     <Segmented size="small" value={tapeMode} onChange={(v) => setTapeMode(v as "follow" | "whole")}
                                options={[{ value: "follow", label: t("console.loopSection.last90") }, { value: "whole", label: t("console.loopSection.whole") }]} />
                   )}
                   <Segmented size="small" value={view3} onChange={(v) => setView3(v as "loop" | "time" | "table")}
                              options={[{ value: "loop", label: t("console.loopSection.viewLoop") }, { value: "time", label: t("console.loopSection.viewTime") },
                                        { value: "table", label: t("console.loopSection.viewTable") }]} />
                 </div>}>
          {view3 === "table" ? <StepTable view={view} price={price} />
            : view3 === "time" || !run?.loop ? (
              <SplitTape spans={view.spans} stepStarts={view.stepStarts} price={tapePrice}
                         running={alive} lastT={view.lastT} mode={tapeMode} trace={trace} />
            ) : <LoopPanel run={run} view={view} price={price} alive={alive} />}
        </Section>

        <div className="console-grid">
          <div style={{ display: "flex", flexDirection: "column", gap: 16, minWidth: 0 }}>
            <Section title={t("console.signal.title")} hint={run?.metrics.length ? t("console.signal.hint") : undefined}>
              <div className="metric-grid">
                {(run?.metrics ?? []).map((m) => (
                  <MetricChart key={m.key} spec={m}
                               points={view.metrics.filter((p) => m.key in p.values).map((p) => ({ step: p.step, value: p.values[m.key] }))} />
                ))}
              </div>
            </Section>
            <Section title={t("console.outputs.title")}>
              <Tabs size="small" items={mtrl ? [
                { key: "check", label: t("console.outputs.rewardCheck"), children: <RewardCheck view={view} /> },
                { key: "dataset", label: t("console.outputs.dataset"), children: <Dataset view={view} /> },
                { key: "rollouts", label: t("console.outputs.agent"), children: <AgentRollouts view={view} /> },
                { key: "code", label: t("console.outputs.code"), children: <Code id={id} files={run?.files ?? []} mtrl /> },
                { key: "log", label: t("console.outputs.log"), children: <Log id={id} view={view} alive={alive} /> },
              ] : [
                { key: "compare", label: t("console.outputs.compare"), children: <Compare view={view}
                  /> },
                ...(run?.probes === false ? [] : [{ key: "ask", label: view.probes.length ? t("console.outputs.askCount", { n: view.probes.length }) : t("console.outputs.ask"),
                  // A loop that rebinds its sampler every step answers with it; the others save a snapshot first.
                  children: <AskModel runId={id} view={view} alive={alive} backend={backend} example={run?.probe_example ?? null}
                                      snapshot={!run?.loop?.nodes.some((n) => n.op === "save_weights_and_rebind")} /> }]),
                ...(view.rewardCheck ? [{ key: "check", label: t("console.outputs.rewardCheck"), children: <SessionRewardCheck view={view} /> }] : []),
                ...(Object.keys(view.rollouts).length ? [{ key: "rollouts", label: t("console.outputs.rollouts"), children: <Rollouts view={view} /> }] : []),
                { key: "log", label: t("console.outputs.log"), children: <Log id={id} view={view} alive={alive} /> },
              ]} />
            </Section>
          </div>

          <div className="console-side">
            <Section title={t("console.side.meter")}>
              <CostMeter tokens={view.tokens} price={price} priceAsOf={pricing?.as_of} backend={backend}
                         engine={mtrl ? "mtrl" : "training-session"} priceBasis={pricing?.mtrl.basis}
                         billing={billing.data} billingError={billing.error ? (billing.error as Error).message : undefined}
                         recordedCost={backend === "replay" ? run?.cost_usd : undefined}
              />
              {(mtrl || view.sandbox) && (
                <AgentcoreLine kind={mtrl ? "agent" : "sandbox"} usage={agentcore.data} backend={backend}
                               error={agentcore.error ? (agentcore.error as Error).message : undefined} />
              )}
            </Section>
            {view.sandbox && <Section title={t("console.sandbox.title")}><SandboxCard sandbox={view.sandbox} /></Section>}
            <Section title={t("console.side.managed")}>{mtrl ? <ManagedMtrl view={view} /> : <Managed view={view} />}</Section>
            <Section title={t("console.side.handled")} hint={`${view.handled.length}`}>
              {view.handled.length ? view.handled.map((h) => (
                <div className="handled-item" key={h.code}>
                  <div>{te(h.message)}</div>
                  <div className="code mono">{h.code}</div>
                </div>
              )) : <span className="muted">{t("console.side.nothingYet")}</span>}
            </Section>
            <Section title={t("console.mlflow.card.title")}><MlflowCard view={view} run={run} /></Section>
            <Section title={t("console.side.saved")} hint={t("console.side.savedHint", { n: view.artifacts.length })}>
              {view.artifacts.length ? <SavedList artifacts={view.artifacts} /> : <span className="muted">{t("console.side.savedEmpty")}</span>}
            </Section>
          </div>
        </div>
      </div>
    </>
  );
}

function SavedList({ artifacts }: { artifacts: RunView["artifacts"] }) {
  const { t } = useTranslation();
  const te = useEventText();
  const [all, setAll] = useState(false);
  const shown = all ? artifacts : artifacts.slice(-3);
  const checkpoints = artifacts.filter((a) => a.kind === "CHECKPOINT").length;
  return (
    <>
      {artifacts.length > 3 && (
        <p className="secondary" style={{ fontSize: 12.5, marginTop: 0 }}>
          {t("console.side.savedSummary", { checkpoints, models: artifacts.length - checkpoints })}
        </p>
      )}
      {shown.map((a) => (
                <div key={a.arn} className="handled-item">
                  <Tag color={a.kind === "CHECKPOINT" ? "blue" : "default"}>{a.kind}</Tag>
                  <span className="secondary" style={{ fontSize: 12.5 }}>{te(a.description)}</span>
                  <div><Ident value={a.arn} short /></div>
                </div>
      ))}
      {artifacts.length > 3 && (
        <button type="button" className="linklike" onClick={() => setAll((v) => !v)}>
          {all ? t("console.side.showLast") : t("console.side.showAll", { n: artifacts.length })}
        </button>
      )}
    </>
  );
}

function RunStrip({ view, run, alive }: { view: RunView; run?: RunSummary; alive: boolean }) {
  const { t } = useTranslation();
  const now = useNow(alive);
  const started = view.started?.ts ?? run?.started_ts ?? null;
  const elapsed = started ? (alive ? now - started : run?.duration_s ?? view.lastT) : null;
  const state = view.session.state;
  const stages = [
    { key: "creating", label: t("console.strip.creating") },
    { key: "running", label: view.session.readySec !== null ? t("console.strip.runningReady", { sec: seconds(view.session.readySec) }) : t("console.strip.running") },
    { key: "stopped", label: state === "failed" ? t("console.strip.failed") : t("console.strip.stopped") },
  ];
  const at = state === "failed" ? 2 : stages.findIndex((s) => s.key === state);
  const total = view.totalSteps ?? run?.total_steps ?? 0;
  const done = view.metrics.length;
  return (
    <div className="run-strip">
      <span>{t("console.strip.session")} <Ident value={view.session.name ?? run?.session_name} /></span>
      <span className="lifecycle" aria-label={t("console.strip.sessionAria", { state: state ?? t("console.strip.notStarted") })}>
        {stages.map((s, i) => <span key={s.key} className={`st${i === at ? " on" : i < at ? " done" : ""}`}>{s.label}</span>)}
      </span>
      <MlflowLink view={view} runId={run?.id ?? view.started?.run} />
      <span className="sep" />
      <span className="mono">{view.started?.model ?? run?.model ?? "—"}</span>
      <span>{view.started?.region ?? run?.region}</span>
      <span className="sep" />
      <span style={{ display: "inline-flex", alignItems: "center", gap: 8, minWidth: 220 }}>
        <span className="num">{total ? t("console.strip.steps", { done: Math.min(done, total), total }) : "—"}</span>
        <Progress percent={total ? Math.round((Math.min(done, total) / total) * 100) : 0} size="small" showInfo={false}
                  strokeColor="var(--ink)" style={{ width: 120, margin: 0 }} />
      </span>
      <span className="num">{seconds(elapsed)}</span>
    </div>
  );
}

/** Where this run went in MLflow: a link that signs in through a presigned URL, or why it did not publish. */
function MlflowLink({ view, runId }: { view: RunView; runId?: string }) {
  const { t } = useTranslation();
  const where = view.mlflow;
  if (!where || !runId) return null;
  if (!where.run_id) {
    return (
      <Tooltip title={t("console.mlflow.notPublished", { error: where.error ?? "" })}>
        <span className="mlflow-off"><ExclamationCircleOutlined /> {t("console.mlflow.off")}</span>
      </Tooltip>
    );
  }
  const tip = view.backend === "replay" ? "console.mlflow.replayTip"
    : where.state === "failed" ? "console.mlflow.partialTip" : where.state === "ended" ? "console.mlflow.endedTip" : "console.mlflow.publishingTip";
  return (
    <Tooltip title={t(tip, { experiment: where.experiment, error: where.error ?? "" })}>
      <a href={`/api/runs/${encodeURIComponent(runId)}/mlflow`} target="_blank" rel="noreferrer"
         className={where.state === "failed" ? "mlflow-partial" : undefined}>
        {where.state === "failed" ? <ExclamationCircleOutlined /> : <ExportOutlined />} MLflow
      </a>
    </Tooltip>
  );
}

/** Where this run went in MLflow, what reached it, and links into MLflow's own views (docs/ui-design.md 6.2). */
function MlflowCard({ view, run }: { view: RunView; run?: RunSummary }) {
  const { t } = useTranslation();
  const selected = useReadiness().data?.mlflow_app;
  if (!run) return null;
  const where = view.mlflow ?? run.mlflow;
  const note = (text: React.ReactNode) => <p className="muted" style={{ fontSize: 12.5, margin: 0 }}>{text}</p>;
  if (view.engine === "mtrl") {
    return view.job?.mlflowUrl
      ? <><p style={{ fontSize: 13, marginTop: 0 }}>{t("console.mlflow.card.mtrl")}</p>
          <a href={view.job.mlflowUrl} target="_blank" rel="noreferrer"><ExportOutlined /> {t("console.mlflow.card.openJob")}</a></>
      : note(t("console.mlflow.card.mtrlPending"));
  }
  if (where?.run_id) {
    const app = selected?.arn === where.tracking_uri && selected.name ? selected.name : where.tracking_uri.split("/").pop();
    const link = (v: string | null, label: string) => (
      <a href={`/api/runs/${encodeURIComponent(run.id)}/mlflow${v ? `?view=${v}` : ""}`} target="_blank" rel="noreferrer">
        <ExportOutlined /> {label}
      </a>
    );
    const state = view.backend === "replay" ? t("console.mlflow.card.replay")
      : where.state === "publishing" ? t("console.mlflow.card.publishing")
      : where.state === "ended" ? t("console.mlflow.card.ended", { status: where.status })
      : t("console.mlflow.card.partial");
    return (
      <div className="mlflow-card">
        <p style={{ fontSize: 13, marginTop: 0 }}>
          {where.state === "failed" && <ExclamationCircleOutlined style={{ color: "var(--warning)" }} />} {state}
        </p>
        {where.state === "failed" && where.error && <p className="mono observed" style={{ fontSize: 11.5 }}>{where.error}</p>}
        <dl className="kv">
          <dt>{t("console.mlflow.card.app")}</dt><dd>{app}</dd>
          <dt>{t("console.mlflow.card.experiment")}</dt><dd className="mono">{where.experiment}</dd>
          <dt>{t("console.mlflow.card.run")}</dt><dd><Ident value={where.run_id} /></dd>
        </dl>
        {where.published && (
          <p className="secondary" style={{ fontSize: 12.5 }}>
            {t("console.mlflow.card.published", { steps: where.published.metric_steps, traces: where.published.traces,
                                                  packages: where.published.model_packages })}
          </p>
        )}
        <div className="mlflow-links">
          {link(null, t("console.mlflow.card.openRun"))}
          {link("model-metrics", t("console.mlflow.card.charts"))}
          {link("traces", t("console.mlflow.card.traces"))}
          {link("experiment", t("console.mlflow.card.experimentRuns"))}
        </div>
        {view.backend !== "replay" && note(t("console.mlflow.card.estimate"))}
      </div>
    );
  }
  if (where?.state === "failed") return note(t("console.mlflow.notPublished", { error: where.error ?? "" }));
  switch (run.mlflow_mode) {
    case "publish": return note(run.alive ? t("console.mlflow.card.connecting") : t("console.mlflow.card.neverStarted"));
    case "off": return note(t("console.mlflow.card.off"));
    case "no-app": return note(<Trans i18nKey="console.mlflow.card.noApp" components={{ link: <Link to="/setup" /> }} />);
    case "replay": return note(run.recording_shipped ? t("console.mlflow.card.replayShipped") : t("console.mlflow.card.replayNone"));
    default:
      return note(t("console.mlflow.card.legacy"));
  }
}

function Problems({ view, run }: { view: RunView; run?: RunSummary }) {
  const { t } = useTranslation();
  const te = useEventText();
  return (
    <>
      {view.problem && <Alert className="problem" type="warning" showIcon title={te(view.problem)} />}
      {view.finished?.status === "failed" && /not allowlisted|not authorized/.test(view.finished.error ?? "") && (
        <Alert className="problem" type="error" showIcon title={t("console.problem.refusedTitle")}
               description={<>
                 <p style={{ margin: "0 0 6px" }}>{t("console.problem.refusedBody")}</p>
                 <Linkified className="mono" text={view.finished.error ?? ""} />
               </>} />
      )}
      {view.finished?.status === "failed" && !/not allowlisted|not authorized/.test(view.finished.error ?? "")
        && !view.logs.some((l) => l.level === "error" && view.finished?.error && l.msg.includes(view.finished.error)) && (
        <Alert className="problem" type="error" showIcon title={t("console.problem.failed")}
               description={<span className="mono">{view.finished.error}</span>} />
      )}
      {run?.status === "interrupted" && !view.problem && (
        <Alert className="problem" type="warning" showIcon title={t("console.problem.interrupted")}
               description={<Trans i18nKey="console.problem.interruptedBody" components={{ link: <Link to="/capacity" /> }} />} />
      )}
      {view.logs.filter((l) => l.level !== "info" && !/not allowlisted|not authorized/.test(l.msg)).map((l) => (
        <Alert key={l.seq} className="problem" type={l.level === "error" ? "error" : "warning"} showIcon title={te(l.msg)} />
      ))}
    </>
  );
}

function StepTable({ view, price }: { view: RunView; price?: { prefill: number; sample: number; train: number } }) {
  const { t } = useTranslation();
  const rows = useMemo(() => stepBreakdown(view.spans, price), [view.spans, price]);
  type Row = (typeof rows)[number];
  let billed = [
    { title: "Prefill", align: "right" as const, render: (_: unknown, r: Row) => int(r.tokens.prefill) },
    { title: "Sample", align: "right" as const, render: (_: unknown, r: Row) => int(r.tokens.sample) },
    { title: "Train", align: "right" as const, render: (_: unknown, r: Row) => int(r.tokens.train) },
    { title: t("console.table.estUsd"), dataIndex: "usd", align: "right" as const, render: (v: number | null) => usd(v, 5) },
  ];
  return (
    <>
      <Table size="small" pagination={false} rowKey="key" dataSource={rows} className="num" columns={[
        { title: "", render: (_, r) => r.step !== null ? t("loop.tape.rowStep", { n: r.step }) : r.key === "before" ? t("loop.tape.rowBefore") : t("loop.tape.rowAfter") },
        { title: <LaneKey who="you" text={t("common.lane.you")} />, dataIndex: "clientSec", align: "right", render: (v: number) => seconds(v) },
        { title: <LaneKey who="aws" text={t("common.lane.aws")} />, dataIndex: "awsSec", align: "right", render: (v: number) => seconds(v) },
        ...billed,
      ]} />
    </>
  );
}

function Compare(props: {
  view: RunView;
}) {
  const { view } = props;
  const { t } = useTranslation();
  const te = useEventText();
  let box: string | undefined;
  if (!view.compare) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={t("console.compare.empty")} />;
  }
  const e = view.evaluation;
  const mark = (ok: boolean) => ok
    ? <CheckOutlined className="answer-ok" aria-label={t("console.compare.matches")} />
    : <CloseOutlined className="answer-bad" aria-label={t("console.compare.noMatch")} />;
  return (
    <div className={box} style={{ position: "relative" }}>
      {e && (
        <div className="compare-head">
          <span className="big num">{e.before} → {e.after}</span>
          <span className="secondary">{t(e.unit === "samples" ? "console.compare.ofSamples" : "console.compare.of", { n: e.n, label: te(e.label) })}</span>
        </div>
      )}
      <Table<CompareItem> size="small" pagination={false} rowKey="prompt" dataSource={view.compare.items} columns={[
        { title: t("console.compare.question"), dataIndex: "prompt", width: "34%" },
        { title: t("console.compare.expected"), dataIndex: "expected", render: (v: string) => <span className="mono">{v}</span> },
        // Answers can be whole functions: keep their lines, and scroll a long one inside its cell.
        { title: t("console.compare.base"), render: (_, r) => <span>{mark(r.before_ok)} <span className="mono answer-text">{r.before || "∅"}</span></span> },
        { title: t("console.compare.trained"), render: (_, r) => <span>{mark(r.after_ok)} <span className="mono answer-text">{r.after || "∅"}</span></span> },
      ]} />
    </div>
  );
}

function SessionRewardCheck({ view }: { view: RunView }) {
  const { t } = useTranslation();
  const te = useEventText();
  const rc = view.rewardCheck!;
  const failed = rc.checks.filter((c) => !c.ok);
  return (
    <>
      <div className="compare-head">
        <span className="big num">{rc.checks.length - failed.length} / {rc.checks.length}</span>
        <span className="secondary">{t("console.check.sessionSummary")}</span>
      </div>
      <Table size="small" pagination={false} rowKey="case" dataSource={rc.rows} className="num" columns={[
        { title: t("console.check.answer"), dataIndex: "case", render: (v: string) => te(v) },
        { title: t("console.check.example"), dataIndex: "example", render: (v: string) => <span className="mono">{v}</span> },
        { title: t("console.check.meanReward"), dataIndex: "reward_mean", align: "right", render: (v: number) => v.toFixed(3) },
      ]} />
      <div style={{ marginTop: 12 }}>
        {rc.checks.map((c) => (
          <div key={c.check} className="handled-item" style={{ display: "flex", gap: 8 }}>
            {c.ok ? <CheckOutlined className="answer-ok" aria-label={t("console.check.passed")} />
              : <CloseOutlined style={{ color: "var(--error)" }} aria-label={t("console.check.failed")} />}
            <span><b>{te(c.check)}</b> <span className="secondary">{te(c.why)}</span></span>
          </div>
        ))}
      </div>
    </>
  );
}

function Rollouts({ view }: { view: RunView }) {
  const { t } = useTranslation();
  const steps = Object.keys(view.rollouts).map(Number).sort((a, b) => a - b);
  const [step, setStep] = useState<number | null>(null);
  const shown = step ?? steps[steps.length - 1];
  if (!steps.length) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={t("console.rollouts.empty")} />;
  }
  return (
    <>
      <div style={{ marginBottom: 8, display: "flex", gap: 8, alignItems: "center" }}>
        <span className="secondary">{t("console.rollouts.step")}</span>
        <Select size="small" value={shown} onChange={setStep} style={{ width: 90 }} options={steps.map((s) => ({ value: s, label: s }))} />
        <span className="muted" style={{ fontSize: 12 }}>{t("console.rollouts.note")}</span>
      </div>
      <Table<Rollout & { key: number }> size="small" rowKey="key"
                      dataSource={view.rollouts[shown].map((r, i) => ({ ...r, key: i }))}
                      pagination={{ pageSize: 16, size: "small" }} columns={[
        { title: t("console.rollouts.group"), dataIndex: "group", width: 64 },
        { title: t("console.rollouts.prompt"), dataIndex: "prompt" },
        { title: t("console.rollouts.answer"), dataIndex: "answer", render: (v) => <span className="mono">{String(v)}</span> },
        { title: t("console.rollouts.completion"), dataIndex: "completion", render: (v: string) => <span className="mono answer-text">{v || "∅"}</span> },
        { title: t("console.rollouts.reward"), dataIndex: "reward", align: "right", width: 80 },
        { title: t("console.rollouts.stop"), dataIndex: "stop_reason", width: 90 },
      ]} />
    </>
  );
}

function Code({ id, files, mtrl }: { id: string; files: string[]; mtrl?: boolean }) {
  const { t } = useTranslation();
  const options = [...files, "config.json"];
  const [name, setName] = useState<string>(options[0]);
  const file = useRunFile(id, name);
  return (
    <>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
        <Segmented size="small" value={name} onChange={(v) => setName(v as string)} options={options} />
        <span className="muted" style={{ fontSize: 12 }}>
          {mtrl ? t("console.code.mtrlNote") : <Trans i18nKey="console.code.note" components={{ code: <span className="mono" /> }} />}
        </span>
      </div>
      <pre className="code-view">{file.data?.content ?? (file.error ? String(file.error) : "…")}</pre>
    </>
  );
}

function Log({ id, view, alive }: { id: string; view: RunView; alive: boolean }) {
  const { t } = useTranslation();
  const out = useRunFile(id, "stdout.log");
  useEffect(() => { if (alive) { const h = setInterval(() => out.refetch(), 3000); return () => clearInterval(h); } }, [alive, out]);
  return <pre className="code-view" style={{ maxHeight: 360 }}>{out.data?.content || (view.lastSeq < 0 ? "…" : t("console.code.noOutput"))}</pre>;
}

function Managed({ view }: { view: RunView }) {
  const { t } = useTranslation();
  let showTrainTokens = true;
  const c = view.counts;
  const stopped = view.session.state === "stopped";
  return (
    <>
      <dl className="kv">
        <dt>{t("console.managed.session")}</dt>
        <dd>{view.session.readySec !== null ? t("console.managed.upIn", { sec: seconds(view.session.readySec) }) : "—"}{stopped ? t("console.managed.stopped") : ""}</dd>
        <dt>{t("console.managed.updates")}</dt><dd className="num">{c.updates}</dd>
        <dt>{t("console.managed.rebinds")}</dt><dd className="num">{c.rebinds}</dd>
        <dt>{t("console.managed.samples")}</dt><dd className="num">{c.samples}</dd>
        <dt>{t("console.managed.packages")}</dt><dd className="num">{t("console.managed.packagesValue", { checkpoints: c.checkpoints, models: c.models })}</dd>
        {showTrainTokens && <><dt>{t("console.managed.trainTokens")}</dt><dd className="num">{int(view.tokens.train)}</dd></>}
      </dl>
      <p className="muted" style={{ fontSize: 12.5, margin: "10px 0 0" }}>{t("console.managed.noInfra")}</p>
    </>
  );
}

// The job's SecondaryStatus values at positions 2–5 are matched to place the strip; labels are translated.
const JOB_STAGES = ["Reward check", "Submit", "Starting", "Downloading", "Training", "Uploading", "Done"];
const JOB_STAGE_KEYS = ["stageCheck", "stageSubmit", "stageStarting", "stageDownloading", "stageTraining", "stageUploading", "stageDone"] as const;

function JobStrip({ view, run, alive }: { view: RunView; run?: RunSummary; alive: boolean }) {
  const { t } = useTranslation();
  const now = useNow(alive);
  const started = view.started?.ts ?? run?.started_ts ?? null;
  const elapsed = started ? (alive ? now - started : run?.duration_s ?? view.lastT) : null;
  const job = view.job;
  const terminal = job && ["Completed", "Failed", "Stopped"].includes(job.status);
  let at = !view.policyCheck ? 0 : !job ? 1 : terminal ? 6 : Math.max(2, JOB_STAGES.indexOf(job.secondary ?? "Starting"));
  const failedEarly = view.finished?.status === "failed" && !job;
  if (failedEarly) at = view.policyCheck && view.policyCheck.checks.every((c) => c.ok) ? 1 : 0;
  const total = view.totalSteps ?? run?.total_steps ?? 0;
  const done = view.step ?? 0;
  return (
    <div className="run-strip">
      <span>{t("console.strip.job")} <Ident value={job?.name ?? run?.job_name} /></span>
      <span className="lifecycle" aria-label={t("console.strip.jobAria", { state: job?.secondary ?? job?.status ?? t("console.strip.notSubmitted") })}>
        {JOB_STAGES.map((stage, i) => (
          <span key={stage} className={`st${i === at ? (failedEarly ? " on fail" : " on") : i < at ? " done" : ""}`}>
            {i === 6 && terminal ? job!.status : t(`console.strip.${JOB_STAGE_KEYS[i]}`)}
          </span>
        ))}
      </span>
      {job?.mlflowUrl && <a href={job.mlflowUrl} target="_blank" rel="noreferrer"><ExportOutlined /> MLflow</a>}
      <span className="sep" />
      <span className="mono">{view.started?.model ?? run?.model ?? "—"}</span>
      <span className="sep" />
      <span style={{ display: "inline-flex", alignItems: "center", gap: 8, minWidth: 220 }}>
        <span className="num">{total ? t("console.strip.steps", { done: Math.min(done, total), total }) : "—"}</span>
        <Progress percent={total ? Math.round((Math.min(done, total) / total) * 100) : 0} size="small" showInfo={false}
                  strokeColor="var(--ink)" style={{ width: 120, margin: 0 }} />
      </span>
      <span className="num">{seconds(elapsed)}</span>
    </div>
  );
}


function RewardCheck({ view }: { view: RunView }) {
  const { t } = useTranslation();
  const te = useEventText();
  const pc = view.policyCheck;
  if (!pc) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={t("console.check.mtrlEmpty")} />;
  }
  const failed = pc.checks.filter((c) => !c.ok);
  return (
    <>
      <div className="compare-head">
        <span className="big num">{pc.checks.length - failed.length} / {pc.checks.length}</span>
        <span className="secondary">{t("console.check.mtrlSummary", { n: pc.n })}</span>
      </div>
      <Table<PolicyRow> size="small" pagination={false} rowKey="policy" dataSource={pc.rows} className="num" columns={[
        { title: t("console.check.policy"), dataIndex: "policy", render: (v: string, r) => <span><span className="mono">{v}</span><div className="muted" style={{ fontSize: 12 }}>{te(r.description)}</div></span> },
        { title: t("console.check.reward"), dataIndex: "reward_mean", align: "right", render: (v: number, r) => <span><b>{v.toFixed(4)}</b><span className="muted"> ±{r.reward_std.toFixed(3)}</span></span> },
        { title: t("console.check.exact"), dataIndex: "exact_match", align: "right", render: (v: number) => `${Math.round(v * 100)}%` },
        { title: t("console.check.lookups"), dataIndex: "coverage", align: "right", render: (v: number) => `${Math.round(v * 100)}%` },
      ]} />
      <div style={{ marginTop: 12 }}>
        {pc.checks.map((c) => (
          <div key={c.check} className="handled-item" style={{ display: "flex", gap: 8 }}>
            {c.ok ? <CheckOutlined className="answer-ok" aria-label={t("console.check.passed")} />
              : <CloseOutlined style={{ color: "var(--error)" }} aria-label={t("console.check.failed")} />}
            <span><b>{te(c.check)}</b> <span className="secondary">{te(c.why)}</span></span>
          </div>
        ))}
      </div>
    </>
  );
}

function Dataset({ view }: { view: RunView }) {
  const { t } = useTranslation();
  const te = useEventText();
  const d = view.dataset;
  if (!d.splits) return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={t("console.dataset.empty")} />;
  return (
    <>
      <Table size="small" pagination={false} rowKey="split" className="num"
             dataSource={Object.entries(d.splits).map(([split, v]) => ({ split, ...v }))} columns={[
        { title: t("console.dataset.split"), dataIndex: "split" },
        { title: t("console.dataset.tickets"), dataIndex: "n", align: "right" },
        { title: t("console.dataset.decisions"), dataIndex: "decisions", render: (v: Record<string, number>) => Object.entries(v).map(([k, n]) => `${k} ${n}`).join(" · ") },
        { title: t("console.dataset.avgChars"), dataIndex: "avg_prompt_chars", align: "right", render: (v: number) => int(v) },
        { title: t("console.dataset.seed"), dataIndex: "seed", align: "right" },
      ]} />
      {d.rollouts && <p className="secondary" style={{ fontSize: 12.5 }}>{te(d.rollouts)}.</p>}
      {d.uris && <dl className="kv">{Object.entries(d.uris).map(([k, v]) => <><dt key={`${k}-k`}>{k}</dt><dd key={`${k}-v`} className="mono">{v}</dd></>)}</dl>}
      <p className="muted" style={{ fontSize: 12 }}>{t("console.dataset.note")}</p>
    </>
  );
}

function AgentRollouts({ view }: { view: RunView }) {
  const { t } = useTranslation();
  const steps = Object.keys(view.rollouts).map(Number).sort((a, b) => a - b);
  const all = steps.flatMap((s) => view.rollouts[s].map((r, i) => ({ ...r, key: `${s}-${i}`, step: s })));
  const handled = view.spans.filter((s) => s.op === "agent_rollouts").reduce((a, s) => a + Number((s as unknown as { count?: number }).count ?? 0), 0);
  if (!all.length) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description={t("console.agent.empty")} />;
  }
  let seen = t("console.agent.seen", { n: int(handled), shown: Math.min(all.length, 96) });
  return (
    <>
      <p className="secondary" style={{ fontSize: 12.5, marginTop: 0 }}>
        {seen}
      </p>
      <Table size="small" rowKey="key" dataSource={all.slice(-96).reverse()} pagination={{ pageSize: 16, size: "small" }} columns={[
        { title: t("console.agent.step"), dataIndex: "step", width: 64 },
        { title: t("console.agent.ticket"), dataIndex: "prompt", render: (v: string) => <span className="mono">{v}</span> },
        { title: t("console.agent.decision"), dataIndex: "completion", render: (v: string) => <span className="mono">{v}</span> },
        { title: t("console.agent.expected"), dataIndex: "answer", render: (v: string) => <span className="mono">{v}</span> },
        { title: t("console.agent.reward"), dataIndex: "reward", align: "right", width: 90, render: (v: number) => v.toFixed(3) },
        { title: "", dataIndex: "stop_reason", width: 120 },
      ]} />
    </>
  );
}

function ManagedMtrl({ view }: { view: RunView }) {
  const { t } = useTranslation();
  const phases = view.spans.filter((s) => s.lane === "aws" && s.op.startsWith("job_"));
  const dispatched = view.metrics.reduce((a, m) => a + (m.values["training/num_trajectories"] ?? 0), 0);
  return (
    <>
      <dl className="kv">
        <dt>{t("console.managed.jobPhases")}</dt>
        <dd>{phases.length ? phases.map((p) => `${opLabel(p.op)} ${seconds(p.sec)}`).join(" · ") : "—"}</dd>
        <dt>{t("console.managed.trainingSteps")}</dt>
        <dd className="num">{view.totalSteps ? t("console.managed.stepsOf", { n: view.step ?? 0, total: view.totalSteps }) : view.step ?? 0}</dd>
        <dt>{t("console.managed.dispatched")}</dt><dd className="num">{dispatched ? int(dispatched) : "—"}</dd>
        <dt>{t("console.managed.outputPackage")}</dt><dd>{view.artifacts.length ? t("console.managed.registered") : "—"}</dd>
      </dl>
      <p className="muted" style={{ fontSize: 12.5, margin: "10px 0 0" }}>{t("console.managed.mtrlNote")}</p>
    </>
  );
}
