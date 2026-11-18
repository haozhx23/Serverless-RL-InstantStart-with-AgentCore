import { Alert, Button, Empty, Table, Tooltip } from "antd";
import { ExclamationCircleOutlined, ExportOutlined } from "@ant-design/icons";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Link, useNavigate } from "react-router";
import { useRuns } from "../api/hooks";
import type { RunSummary } from "../api/types";
import { PageHeader } from "../components/Bits";
import { ProvenanceBadge, RunStatusTag } from "../components/Provenance";
import { HOME } from "../lib/engines";
import { clock, seconds, usd } from "../lib/format";

const published = (r: RunSummary) => !!r.mlflow?.run_id;

export default function RunsPage() {
  const { t } = useTranslation();
  const { data, error, isLoading } = useRuns();
  const navigate = useNavigate();
  const [picked, setPicked] = useState<string[]>([]);
  const runs = data?.runs ?? [];
  const chosen = runs.filter((r) => picked.includes(r.id) && published(r));
  const apps = new Set(chosen.map((r) => r.mlflow!.tracking_uri));
  const compareBlocked = chosen.length < 2 ? t("runs.compareNeedTwo") : apps.size > 1 ? t("runs.compareOneApp") : null;
  return (
    <>
      <PageHeader crumbs={t("runs.crumbs")} title={t("shell.nav.runs")} />
      <div className="page-body">
        {error && <Alert type="error" showIcon title={(error as Error).message} style={{ marginBottom: 12 }} />}
        <div className="section-card">
          <div className="runs-toolbar">
            <span className="secondary">{t("runs.compareHint")}</span>
            <Tooltip title={compareBlocked}>
              {/* MLflow's compare view: parameters side by side, metric curves overlaid, parallel coordinates */}
              <Button icon={<ExportOutlined />} disabled={!!compareBlocked} target="_blank" rel="noreferrer"
                      href={compareBlocked ? undefined : `/api/mlflow/compare?runs=${chosen.map((r) => encodeURIComponent(r.id)).join(",")}`}>
                {t("runs.compare", { n: chosen.length })}
              </Button>
            </Tooltip>
          </div>
          <Table<RunSummary>
            loading={isLoading} rowKey="id" dataSource={runs} size="middle" pagination={{ pageSize: 20 }}
            rowSelection={{ selectedRowKeys: picked, onChange: (keys) => setPicked(keys as string[]),
                            getCheckboxProps: (r) => ({ disabled: !published(r), title: published(r) ? undefined : t("runs.notInMlflow") }) }}
            onRow={(r) => ({
              onClick: (ev) => {  // the checkbox and the links in a row do their own thing
                if ((ev.target as HTMLElement).closest(".ant-table-selection-column, a")) return;
                navigate(`/runs/${r.id}`);
              },
              style: { cursor: "pointer" },
            })}
            locale={{ emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={
              <>{t("runs.empty")}<br />
                <Button type="primary" style={{ marginTop: 10 }} onClick={() => navigate(HOME)}>{t("runs.open")}</Button></>} /> }}
            columns={[
              { title: t("runs.run"), dataIndex: "id", className: "nowrap", render: (v: string) => <Link to={`/runs/${v}`} className="mono">{v}</Link> },
              { title: t("runs.scenario"), dataIndex: "title", className: "wrap-col" },
              { title: t("runs.source"), className: "nowrap", render: (_, r) => <ProvenanceBadge backend={r.backend} recordedAt={r.recorded_at} compact /> },
              { title: t("runs.status"), className: "nowrap", render: (_, r) => <RunStatusTag status={r.status} /> },
              { title: t("runs.started"), className: "nowrap", render: (_, r) => clock(r.started_ts ?? r.created_ts) },
              { title: t("runs.duration"), align: "right", className: "nowrap", render: (_, r) => <span className="num">{seconds(r.duration_s)}</span> },
              { title: t("runs.steps"), align: "right", className: "nowrap", render: (_, r) => <span className="num">{r.step !== null ? `${r.step + 1}/${r.total_steps}` : "—"}</span> },
              { title: t("runs.result"), align: "right", className: "nowrap", render: (_, r) => r.eval
                  ? <Tooltip title={r.eval.label}><span className="num">{r.eval.before} → {r.eval.after} / {r.eval.n}</span></Tooltip>
                  : <span className="muted">—</span> },
              { title: t("runs.cost"), align: "right", className: "nowrap", render: (_, r) => {
                  let cost = usd(r.cost_usd);
                  return <span className="num">{cost}</span>;
                } },
              { title: "MLflow", align: "center", render: (_, r) => published(r) ? (
                  <Tooltip title={r.mlflow!.state === "failed" ? t("runs.mlflowPartial") : t("runs.mlflowOpen", { experiment: r.mlflow!.experiment })}>
                    <a href={`/api/runs/${encodeURIComponent(r.id)}/mlflow`} target="_blank" rel="noreferrer" aria-label={t("runs.mlflowOpen", { experiment: r.mlflow!.experiment })}>
                      {r.mlflow!.state === "failed" ? <ExclamationCircleOutlined style={{ color: "var(--warning)" }} /> : <ExportOutlined />}
                    </a>
                  </Tooltip>
                ) : <span className="muted">—</span> },
            ]}
          />
        </div>
      </div>
    </>
  );
}
