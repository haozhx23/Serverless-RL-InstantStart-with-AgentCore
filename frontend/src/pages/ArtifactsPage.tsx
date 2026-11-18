import { Alert, App, Button, Empty, Popconfirm, Table, Tag, Typography } from "antd";
import { Trans, useTranslation } from "react-i18next";
import { Link, useNavigate } from "react-router";
import { api } from "../api/client";
import { useArtifacts, useInvalidate } from "../api/hooks";
import type { PackageRow } from "../api/types";
import { PageHeader } from "../components/Bits";
import { dateTime } from "../lib/format";

export default function ArtifactsPage() {
  const { t } = useTranslation();
  const { message } = App.useApp();
  const navigate = useNavigate();
  const invalidate = useInvalidate();
  const { data, error, isLoading } = useArtifacts();
  const remove = async (arn: string) => {
    try {
      const r = await api<{ deleted: boolean; objects_deleted: number; artifacts_retained: boolean; retained_reason: string | null }>(
        "/api/artifacts/delete", { body: { arn } });
      if (r.artifacts_retained) message.warning(t("artifacts.toastKept", { reason: r.retained_reason }));
      else message.success(t("artifacts.toastDeleted", { n: r.objects_deleted }));
      invalidate("artifacts");
    } catch (e) { message.error((e as Error).message); }
  };
  return (
    <>
      <PageHeader crumbs={t("artifacts.crumbs")} title={t("shell.nav.artifacts")} />
      <div className="page-body" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <p className="secondary" style={{ margin: 0, fontSize: 13 }}>
          <Trans i18nKey="artifacts.intro" values={{ group: data?.group ?? "…" }} components={{ code: <span className="mono" />, b: <b /> }} />
        </p>
        {error && <Alert type="error" showIcon title={(error as Error).message} />}
        {data && !data.group_exists && (
          <Alert type="info" showIcon title={t("artifacts.noGroup")}
                 description={<Trans i18nKey="artifacts.noGroupBody" components={{ link: <Link to="/setup" /> }} />} />
        )}
        {data?.errors.map((e) => <Alert key={e} type="warning" showIcon title={e} />)}
        <div className="section-card">
          <Table<PackageRow> className="artifacts-table" tableLayout="auto" scroll={{ x: true }}
            loading={isLoading} rowKey="arn" dataSource={data?.packages} size="middle"
            locale={{ emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={t("artifacts.empty")} /> }}
            columns={[
              { title: t("artifacts.version"), dataIndex: "version", width: "1%", className: "nowrap",
                render: (v) => <span className="num">{v}</span> },
              { title: t("artifacts.kind"), dataIndex: "kind", width: "1%", className: "nowrap", render: (k: PackageRow["kind"]) => k ? (
                <span><Tag color={k === "CHECKPOINT" ? "blue" : "default"}>{k}</Tag>
                  <span className="muted" style={{ fontSize: 12 }}>{k === "CHECKPOINT" ? t("artifacts.resumable") : t("artifacts.forSampling")}</span></span>) : "—" },
              { title: t("artifacts.description"), dataIndex: "description", className: "artifact-description" },
              { title: t("artifacts.run"), dataIndex: "run_id", width: "1%", className: "nowrap",
                render: (v: string | null) => v ? <a className="mono" onClick={() => navigate(`/runs/${v}`)}>{v}</a> : "—" },
              { title: t("artifacts.created"), dataIndex: "created", width: "1%", className: "nowrap",
                render: (v: string | null) => dateTime(v) },
              { title: "S3", dataIndex: "s3_uri", width: "1%", className: "nowrap", render: (v: string | null) => v ? (
                <Typography.Text className="mono artifact-s3" ellipsis={{ tooltip: v }} copyable={{ text: v }}>
                  {v.length > 48 ? `…${v.slice(-44)}` : v}
                </Typography.Text>
              ) : "—" },
              { title: "", width: "1%", className: "nowrap", render: (_, r) => (
                <div className="artifact-actions">
                  <Typography.Text className="mono" title={r.arn} copyable={{ text: r.arn }}>ARN</Typography.Text>
                  <Popconfirm title={t("artifacts.confirm")} description={t("artifacts.confirmBody")}
                              okText={t("artifacts.ok")} okButtonProps={{ danger: true }} onConfirm={() => remove(r.arn)}>
                    <Button size="small" danger>{t("artifacts.delete")}</Button>
                  </Popconfirm>
                </div>) },
            ]} />
        </div>
      </div>
    </>
  );
}
