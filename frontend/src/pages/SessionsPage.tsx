import { Alert, App, Button, Popconfirm, Segmented, Table, Tag } from "antd";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { api } from "../api/client";
import { useInvalidate, useMtrlJobs, useMtrlQuota, useQuota, useSandboxActive, useSessions } from "../api/hooks";
import type { MtrlJob, SessionRow } from "../api/types";
import { PageHeader, QuotaPips, useLimitSource } from "../components/Bits";
import { dateTime, int, timeOfDay, usd } from "../lib/format";

const ACTIVE = new Set(["Created", "Scheduled", "Blocked", "Running", "Stopping"]);

export default function SessionsPage() {
  const { t } = useTranslation();
  const limitSource = useLimitSource();
  const { message } = App.useApp();
  const invalidate = useInvalidate();
  const [scope, setScope] = useState<"ours" | "all">("ours");
  const sessions = useSessions(scope);
  const quota = useQuota().data;
  const stop = async (name: string) => {
    try { await api(`/api/sessions/${encodeURIComponent(name)}/stop`, { method: "POST" }); message.info(t("capacity.toast.stopping", { name })); invalidate("sessions", "quota"); }
    catch (e) { message.error((e as Error).message); }
  };
  const stopAll = async () => {
    try {
      const r = await api<{ results: { name: string; stopped: boolean; error: string | null }[] }>("/api/sessions/stop-all", { method: "POST" });
      const failed = r.results.filter((x) => !x.stopped);
      if (failed.length) message.error(t("capacity.toast.notStopped", { list: failed.map((f) => `${f.name} (${f.error})`).join(", ") }));
      else message.success(r.results.length ? t("capacity.toast.stoppingMany", { n: r.results.length }) : t("capacity.toast.noneRunning"));
      invalidate("sessions", "quota");
    } catch (e) { message.error((e as Error).message); }
  };
  const activeOurs = sessions.data?.sessions.filter((s) => s.ours && ACTIVE.has(s.status) && s.status !== "Stopping").length ?? 0;
  return (
    <>
      <PageHeader crumbs={t("capacity.crumbs")} title={t("shell.nav.capacity")}
                  extra={<Popconfirm title={t("capacity.stopAll.confirm", { n: activeOurs })} description={t("capacity.stopAll.confirmBody")}
                                     onConfirm={stopAll} okText={t("capacity.stopAll.ok")} okButtonProps={{ danger: true }} disabled={!activeOurs}>
                    <Button danger disabled={!activeOurs}>{t("capacity.stopAll.button")}</Button>
                  </Popconfirm>} />
      <div className="page-body" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <h2 className="display" style={{ fontSize: 17, margin: "4px 0 0" }}>{t("capacity.sessions.title")}</h2>
        <div className="section-card" style={{ padding: "16px 18px", display: "flex", gap: 28, alignItems: "center", flexWrap: "wrap" }}>
          {quota && <QuotaPips large limit={quota.limit} active={quota.active} />}
          <div>
            <div style={{ fontSize: 22, fontWeight: 600 }} className="num">{quota ? t("capacity.sessions.inUse", { active: quota.active ?? "?", limit: quota.limit }) : "…"}</div>
            <div className="secondary" style={{ fontSize: 13 }}>{quota ? t("capacity.sessions.captionSource", { source: limitSource(quota.limit_source) }) : t("capacity.sessions.caption")}</div>
          </div>
          <p className="secondary" style={{ margin: 0, fontSize: 13 }}>
            {t("capacity.sessions.note")}
          </p>
        </div>
        {quota?.error && <Alert type="warning" showIcon title={quota.error} />}
        {sessions.error && <Alert type="error" showIcon title={(sessions.error as Error).message} />}
        {sessions.data?.errors.map((e) => <Alert key={e} type="warning" showIcon title={e} />)}
        <div className="section-card">
          <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--hairline)" }}>
            <Segmented value={scope} onChange={(v) => setScope(v as "ours" | "all")}
                       options={[{ value: "ours", label: t("capacity.sessions.ours") }, { value: "all", label: t("capacity.sessions.all") }]} />
          </div>
          <Table<SessionRow> loading={sessions.isLoading} rowKey="name" dataSource={sessions.data?.sessions} size="middle"
            columns={[
              { title: t("capacity.sessions.session"), dataIndex: "name", render: (v: string, r) => <span className="mono">{v}{!r.ours && <Tag style={{ marginLeft: 8 }}>{t("capacity.sessions.other")}</Tag>}</span> },
              { title: t("capacity.sessions.status"), dataIndex: "status", render: (v: string) => <Tag color={ACTIVE.has(v) ? "processing" : v === "Failed" ? "error" : "default"}>{v}</Tag> },
              { title: t("capacity.sessions.created"), dataIndex: "created", render: (v: string | null) => dateTime(v) },
              { title: t("capacity.sessions.model"), dataIndex: "hub_model", render: (v) => <span className="mono muted">{v ?? "—"}</span> },
              { title: t("capacity.sessions.billed"), align: "right", render: (_, r) => r.tokens ? <span className="num mono">{int(r.tokens.prefill + r.tokens.sample + r.tokens.train)}</span> : "—" },
              { title: t("capacity.sessions.cost"), align: "right", render: (_, r) => <span className="num">{usd(r.cost_usd)}</span> },
              { title: "", render: (_, r) => ACTIVE.has(r.status) && r.status !== "Stopping" ? (
                <Popconfirm title={t("capacity.sessions.confirm", { name: r.name })} onConfirm={() => stop(r.name)} okText={t("capacity.sessions.ok")}>
                  <Button size="small" danger>{t("capacity.sessions.stop")}</Button>
                </Popconfirm>) : null },
            ]} />
        </div>
        <SandboxSessions />
        <MtrlJobs />
      </div>
    </>
  );
}

/** AgentCore Runtime sessions in the account and region: the environment sandbox's, and MTRL agents'. */
function SandboxSessions() {
  const { t } = useTranslation();
  const a = useSandboxActive().data;
  return (
    <>
      <h2 className="display" style={{ fontSize: 17, margin: "12px 0 0" }}>{t("capacity.agentcore.title")}</h2>
      <div className="section-card" style={{ padding: "16px 18px", display: "flex", gap: 28, alignItems: "center", flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: 22, fontWeight: 600 }} className="num">
            {a ? (a.active === null ? "?" : t("capacity.sessions.inUse", { active: int(a.active), limit: int(a.limit) })) : "…"}
          </div>
          <div className="secondary" style={{ fontSize: 13 }}>
            {a ? (a.at ? t("capacity.agentcore.at", { time: timeOfDay(a.at), source: a.limit_source }) : t("capacity.agentcore.noData", { source: a.limit_source }))
              : t("capacity.agentcore.caption")}
          </div>
        </div>
        <p className="secondary" style={{ margin: 0, fontSize: 13 }}>{t("capacity.agentcore.note")}</p>
      </div>
      {a?.error && <Alert type="warning" showIcon title={a.error} />}
    </>
  );
}

const JOB_ACTIVE = new Set(["InProgress", "Stopping"]);

function MtrlJobs() {
  const { t } = useTranslation();
  const limitSource = useLimitSource();
  const { message } = App.useApp();
  const invalidate = useInvalidate();
  const jobs = useMtrlJobs();
  const quota = useMtrlQuota().data;
  const stop = async (name: string) => {
    try { await api(`/api/mtrl/jobs/${encodeURIComponent(name)}/stop`, { method: "POST" }); message.info(t("capacity.toast.stopping", { name })); invalidate("mtrl-jobs", "mtrl-quota"); }
    catch (e) { message.error((e as Error).message); }
  };
  return (
    <>
      <h2 className="display" style={{ fontSize: 17, margin: "12px 0 0" }}>{t("capacity.jobs.title")}</h2>
      <div className="section-card" style={{ padding: "16px 18px", display: "flex", gap: 28, alignItems: "center", flexWrap: "wrap" }}>
        {quota && <QuotaPips large limit={quota.limit} active={quota.active} />}
        <div>
          <div style={{ fontSize: 22, fontWeight: 600 }} className="num">{quota ? t("capacity.sessions.inUse", { active: quota.active ?? "?", limit: quota.limit }) : "…"}</div>
          <div className="secondary" style={{ fontSize: 13 }}>{quota ? t("capacity.jobs.captionSource", { source: limitSource(quota.limit_source) }) : t("capacity.jobs.caption")}</div>
        </div>
        <p className="secondary" style={{ margin: 0, fontSize: 13 }}>
          {t("capacity.jobs.note")}
        </p>
      </div>
      {jobs.error && <Alert type="error" showIcon title={(jobs.error as Error).message} />}
      <div className="section-card">
        <Table<MtrlJob> loading={jobs.isLoading} rowKey="name" dataSource={jobs.data?.jobs} size="middle"
          locale={{ emptyText: t("capacity.jobs.empty") }}
          columns={[
            { title: t("capacity.jobs.job"), dataIndex: "name", render: (v: string) => <span className="mono">{v}</span> },
            { title: t("capacity.jobs.status"), render: (_, j) => <Tag color={JOB_ACTIVE.has(j.status) ? "processing" : j.status === "Failed" ? "error" : "default"}>{j.status}{j.secondary ? ` · ${j.secondary}` : ""}</Tag> },
            { title: t("capacity.jobs.created"), dataIndex: "created", render: (v: string | null) => dateTime(v) },
            { title: t("capacity.jobs.ended"), dataIndex: "ended", render: (v: string | null) => dateTime(v) },
            { title: t("capacity.jobs.failure"), dataIndex: "failure_reason", render: (v: string | null) => <span className="secondary" style={{ fontSize: 12.5 }}>{v ?? ""}</span> },
            { title: "", render: (_, j) => j.status === "InProgress" ? (
              <Popconfirm title={t("capacity.jobs.confirm", { name: j.name })} onConfirm={() => stop(j.name)} okText={t("capacity.jobs.ok")}>
                <Button size="small" danger>{t("capacity.jobs.stop")}</Button>
              </Popconfirm>) : null },
          ]} />
      </div>
    </>
  );
}
