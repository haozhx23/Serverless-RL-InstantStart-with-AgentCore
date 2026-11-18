import { App, Alert, Button, Collapse, Drawer, Form, InputNumber, Segmented, Select, Switch } from "antd";
import { useEffect, useMemo, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Link, useNavigate } from "react-router";
import { useEstimate, useInvalidate } from "../api/hooks";
import { api } from "../api/client";
import type { Backend, Models, ParamSpec, Readiness, RunSummary, Scenario } from "../api/types";
import { useLang } from "../i18n";
import { joinList, readyFor } from "../lib/engines";
import { int, seconds, usd } from "../lib/format";
import Composer, { LoopPreview } from "./Composer";
import CostEstimateView from "./CostEstimate";
import { DeployLine, deployTargetOf } from "./DeployPanel";

// Starting a run of one scenario: live or a replay (docs/ui-design.md §6.1).
// Training-session scenarios get the loop map: the composer when they have slots, read-only when not.
// MTRL scenarios get the rollout count.

export type Mode = Backend;

export default function LaunchDrawer({ scenario, initialMode, readiness, models, onClose }: {
  scenario: Scenario; initialMode: Mode; readiness?: Readiness; models: Models; onClose: () => void;
}) {
  const { t } = useTranslation();
  const lang = useLang();
  const { message } = App.useApp();
  const navigate = useNavigate();
  const invalidate = useInvalidate();
  const mtrl = scenario.engine === "mtrl";
  const { ready: liveReady, blockers } = readyFor(scenario.engine, readiness, lang, !!scenario.sandbox);
  const deployTarget = deployTargetOf(scenario);  // the runtime its live runs need: always shown, with its own button
  const [mode, setMode] = useState<Mode>(initialMode);
  const [recording, setRecording] = useState(scenario.recordings[scenario.recordings.length - 1]?.name);
  const [speed, setSpeed] = useState(4);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [publish, setPublish] = useState(true);
  const [form] = Form.useForm();
  const params = scenario.params ?? [];
  const initial = useMemo(() => Object.fromEntries(params.filter((p) => !p.slot).map((p) => [p.key, p.default])), [params]);
  const slots = scenario.slots ?? [];
  const [choices, setChoices] = useState<Record<string, string | number>>(() => Object.fromEntries(slots.map((sl) => [
    sl.key, sl.param ? params.find((p) => p.key === sl.param)!.default as string | number : sl.default!])));
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [slotKey, setSlotKey] = useState(slots.find((sl) => sl.node)?.key ?? slots[0]?.key);
  const showLoop = !!scenario.loop && !mtrl && mode !== "replay";
  const composing = showLoop && slots.length > 0;
  const live = mode === "live";
  let modelList = mtrl ? models.mtrl_live ?? readiness?.mtrl_models ?? [] : models.live ?? readiness?.models ?? [];
  const modelOptions = modelList.map((m) => ({ value: m, label: m }));
  const steps = Form.useWatch("max_steps", form);
  const batch = Form.useWatch("global_batch_size", form);
  const group = Form.useWatch("group_size", form);
  const rollouts = mtrl && steps && batch && group ? steps * batch * group : null;
  const values = Form.useWatch([], form) as Record<string, unknown> | undefined;
  // What POST /api/runs gets: the form's values, the param slots' choices, code slots not edited, edited code.
  const launchBody = (formValues: Record<string, unknown>) => {
    const paramSlots = Object.fromEntries(slots.filter((sl) => sl.param).map((sl) => [sl.param!, choices[sl.key]]));
    const variants = Object.fromEntries(slots.filter((sl) => sl.file && edits[sl.file] === undefined).map((sl) => [sl.key, choices[sl.key]]));
    return { scenario: scenario.id, params: { ...formValues, ...paramSlots }, variants, files: edits };
  };
  const estimate = useEstimate(useDebounced(mode === "replay" ? null : launchBody(values ?? initial), 300));

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      let run: RunSummary;
      if (mode === "replay") {
        run = await api<RunSummary>(`/api/recordings/${encodeURIComponent(recording!)}/replay`, { body: { speed } });
      } else {
        run = await api<RunSummary>("/api/runs", { body: { ...launchBody(await form.validateFields()), backend: mode, mlflow: publish } });
      }
      const toast: Record<Mode, string> = {
        live: t(mtrl ? "scenarios.drawer.toastJob" : "scenarios.drawer.toastLive"),
        replay: t("scenarios.drawer.toastReplay"),
      };
      message.success(toast[mode]);
      invalidate("runs", "quota", "mtrl-quota");
      navigate(`/runs/${run.id}`);
    } catch (e) {
      if ((e as { errorFields?: unknown }).errorFields) return; // field errors are shown inline
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const actions: Record<Mode, string> = {
    live: t(mtrl ? "scenarios.drawer.actionJob" : "scenarios.drawer.actionLive"),
    replay: t("scenarios.drawer.actionReplay"),
  };
  const action = actions[mode];
  let editedNote = t("scenarios.drawer.editedBody");
  let estimateLabel = t("scenarios.drawer.estimateLive");
  const modes = [
    { value: "live", label: t("scenarios.drawer.modeLive"), disabled: !liveReady },
    ...(scenario.recordings.length ? [{ value: "replay", label: t("scenarios.drawer.modeReplay") }] : []),
  ];

  return (
    <Drawer open onClose={onClose} size={showLoop ? 900 : 560} title={<span className="display">{scenario.title}</span>}
            footer={
              <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
                <Button onClick={onClose}>{t("scenarios.drawer.cancel")}</Button>
                <Button type="primary" loading={busy} onClick={submit}
                        disabled={(live && !liveReady) || (mode === "replay" && !recording)}>
                  {action}
                </Button>
              </div>
            }>
      <Segmented block value={mode} onChange={(v) => setMode(v as Mode)} options={modes} style={{ marginBottom: 14 }} />
      {deployTarget && <DeployLine targetId={deployTarget} />}
      {live && !liveReady && (
        <Alert type="warning" showIcon style={{ marginBottom: 14 }} title={t(mtrl ? "scenarios.drawer.blockedJobs" : "scenarios.drawer.blockedRuns")}
               description={<Trans i18nKey="scenarios.drawer.missing" values={{ blockers: joinList(t, blockers) }}
                                   components={{ link: <Link to="/setup" /> }} />} />
      )}
      {live && liveReady && !mtrl && (
        <Alert type="info" showIcon style={{ marginBottom: 14 }} title={t("scenarios.drawer.liveTitle", { region: readiness!.region })}
               description={scenario.evidence && t(scenario.status === "live-verified" ? "scenarios.drawer.liveBodyTemplate" : "scenarios.drawer.liveBody", { duration: seconds(scenario.evidence.duration_s),
                 cost: usd(scenario.evidence.cost_usd), model: scenario.evidence.model, date: scenario.evidence.date })} />
      )}
      {live && liveReady && scenario.sandbox && (
        <Alert type="info" showIcon style={{ marginBottom: 14 }} title={t("scenarios.drawer.sandboxLiveTitle")}
               description={t("scenarios.drawer.sandboxLiveBody")} />
      )}
      {live && liveReady && mtrl && (
        <Alert type="info" showIcon style={{ marginBottom: 14 }} title={t("scenarios.drawer.jobTitle", { region: readiness!.region })}
               description={t("scenarios.drawer.jobBody")} />
      )}
      {mode !== "replay" && (
        <div className="drawer-estimate">
          <div className="eyebrow">{estimateLabel}</div>
          {estimate.error ? <span className="observed">{(estimate.error as Error).message}</span>
            : <CostEstimateView estimate={estimate.data} small />}
        </div>
      )}
      {composing && (
        <Composer scenario={scenario} choices={choices} edits={edits} slotKey={slotKey!} onSlot={setSlotKey}
                  onChoice={(k, v) => setChoices((c) => ({ ...c, [k]: v }))}
                  onEdit={(file, code) => setEdits((e) => {
                    const next = { ...e };
                    if (code === undefined) delete next[file]; else next[file] = code;
                    return next;
                  })} />
      )}
      {showLoop && !composing && <LoopPreview scenario={scenario} />}
      {composing && live && Object.keys(edits).length > 0 && (
        <Alert type="warning" showIcon style={{ marginBottom: 14 }} title={t("scenarios.drawer.editedTitle")} description={editedNote} />
      )}
      {mode === "replay" ? (
        <Form layout="vertical">
          <Form.Item label={t("scenarios.drawer.recording")}>
            <Select value={recording} onChange={setRecording}
                    options={scenario.recordings.map((r) => ({ value: r.name, label: `${new Date(r.recorded_ts * 1000).toISOString().slice(0, 16).replace("T", " ")} · ${r.model} · ${usd(r.cost_usd)}${r.shipped ? ` · ${t("scenarios.drawer.shipped")}` : ""}` }))} />
            {scenario.recordings.find((r) => r.name === recording)?.shipped && (
              <p className="muted" style={{ fontSize: 12, margin: "6px 0 0" }}>{t("scenarios.drawer.shippedNote")}</p>
            )}
          </Form.Item>
          <Form.Item label={t("scenarios.drawer.speed")}>
            <Segmented value={speed} onChange={(v) => setSpeed(v as number)} options={[1, 2, 4, 8].map((v) => ({ value: v, label: `${v}×` }))} />
          </Form.Item>
        </Form>
      ) : (
        <Form form={form} layout="vertical" initialValues={initial} requiredMark={false}>
          {showLoop && <div className="eyebrow" style={{ margin: "4px 0 10px" }}>{t("scenarios.drawer.params")}</div>}
          {params.filter((p) => !p.advanced && !(composing && p.slot)).map((p) => <ParamField key={p.key} p={p} modelOptions={modelOptions} />)}
          {rollouts !== null && (
            <p className="secondary" style={{ fontSize: 12.5, marginTop: -6 }}>
              {t("scenarios.drawer.rollouts", { n: int(rollouts), steps, batch, group })}
            </p>
          )}
          <Collapse ghost size="small" items={[{
            key: "adv", label: mtrl ? t("scenarios.drawer.advancedJob") : t("scenarios.drawer.advanced"),
            children: <>
              {!mtrl && params.some((p) => p.key === "adam_beta2") && <p className="muted" style={{ fontSize: 12, marginTop: 0 }}>{t("scenarios.drawer.adamNote")}</p>}
              {params.filter((p) => p.advanced && !(composing && p.slot)).map((p) => <ParamField key={p.key} p={p} modelOptions={modelOptions} />)}
            </>,
          }]} />
        </Form>
      )}
      {live && !mtrl && <MlflowToggle app={readiness?.mlflow_app ?? null} checked={publish} onChange={setPublish} />}
      {error && <Alert type="error" showIcon title={error} style={{ marginTop: 12 }} />}
    </Drawer>
  );
}

/** `value` once it has stopped changing for `ms` (compared by content, so a new object with the same values is no change). */
function useDebounced<T>(value: T, ms: number): T {
  const key = JSON.stringify(value);
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const h = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, ms]);
  return settled;
}

/** Whether a live training-session run publishes itself to the MLflow app picked on Setup. */
function MlflowToggle({ app, checked, onChange }: {
  app: Readiness["mlflow_app"]; checked: boolean; onChange: (v: boolean) => void;
}) {
  const { t } = useTranslation();
  if (!app?.usable) {
    return (
      <p className="muted mlflow-toggle" style={{ fontSize: 12.5 }}>
        <Trans i18nKey={app ? "scenarios.drawer.mlflowUnusable" : "scenarios.drawer.mlflowNone"}
               values={{ name: app?.name ?? app?.arn, status: app?.status ?? "?" }} components={{ link: <Link to="/setup" /> }} />
      </p>
    );
  }
  return (
    <label className="mlflow-toggle">
      <Switch size="small" checked={checked} onChange={onChange} />
      <span>
        <span style={{ fontWeight: 500 }}>{t("scenarios.drawer.mlflow", { name: app.name })}</span>
        <span className="muted" style={{ display: "block", fontSize: 12 }}>{t("scenarios.drawer.mlflowHint")}</span>
      </span>
    </label>
  );
}

function ParamField({ p, modelOptions }: { p: ParamSpec; modelOptions: { value: string; label: string }[] }) {
  const { t } = useTranslation();
  const rules = [{ required: p.type !== "bool", message: t("scenarios.drawer.required", { label: p.label }) }];
  const common = { name: p.key, label: p.label, extra: p.help, rules };
  if (p.type === "bool") return <Form.Item {...common} valuePropName="checked"><Switch /></Form.Item>;
  if (p.type === "model" || p.type === "mtrl_model") return <Form.Item {...common}><Select options={modelOptions} /></Form.Item>;
  if (p.type === "choice") return <Form.Item {...common}><Select options={p.choices!.map((c) => ({ value: c, label: String(c) }))} /></Form.Item>;
  return (
    <Form.Item {...common}>
      <InputNumber min={p.min} max={p.max} precision={p.type === "int" ? 0 : undefined} style={{ width: 200 }} />
    </Form.Item>
  );
}
