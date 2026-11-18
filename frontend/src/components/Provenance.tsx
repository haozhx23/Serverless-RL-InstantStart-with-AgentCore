import { Tag, Tooltip } from "antd";
import { CheckCircleFilled, CloseCircleFilled, ExclamationCircleFilled, LoadingOutlined, MinusCircleFilled, PauseCircleFilled } from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import type { Backend, RunStatus } from "../api/types";

// Where a run's numbers come from. Shown on every run so a replay is never read as a live run
// (docs/ui-design.md §3.2).
/** compact: the source's name only (for tables); the full wording and the recording date move to the tooltip. */
export function ProvenanceBadge({ backend, recordedAt, running, compact }: {
  backend: Backend | null; recordedAt?: string | null; running?: boolean; compact?: boolean;
}) {
  const { t } = useTranslation();
  if (!backend) return null;
  const fullText: Record<Backend, string> = {
    live: t("common.prov.live"),
    replay: recordedAt ? t("common.prov.replay", { date: recordedAt }) : t("common.prov.replayNoDate"),
  };
  const shortText: Record<Backend, string> = {
    live: t("common.prov.liveShort"),
    replay: t("common.prov.replayNoDate"),
  };
  const helpText: Record<Backend, string> = {
    live: t("common.prov.liveHelp"),
    replay: t("common.prov.replayHelp"),
  };
  const full = fullText[backend];
  const text = compact ? shortText[backend] : full;
  const explain = helpText[backend];
  const help = compact ? <>{full}<br />{explain}</> : explain;
  return (
    <Tooltip title={help}>
      <span className={`prov ${backend}${running ? " running" : ""}`} tabIndex={0}>
        <span className="dot" aria-hidden />
        {text}
      </span>
    </Tooltip>
  );
}

const STATUS: Record<RunStatus, { color: string; icon: React.ReactNode }> = {
  starting: { color: "processing", icon: <LoadingOutlined /> },
  running: { color: "processing", icon: <LoadingOutlined /> },
  stopping: { color: "warning", icon: <LoadingOutlined /> },
  completed: { color: "success", icon: <CheckCircleFilled /> },
  failed: { color: "error", icon: <CloseCircleFilled /> },
  stopped: { color: "default", icon: <PauseCircleFilled /> },
  interrupted: { color: "warning", icon: <ExclamationCircleFilled /> },
};

export function RunStatusTag({ status }: { status: RunStatus }) {
  const { t } = useTranslation();
  const s = STATUS[status];
  if (!s) return <Tag icon={<MinusCircleFilled />} style={{ marginInlineEnd: 0 }}>{status}</Tag>;
  return <Tag color={s.color} icon={s.icon} style={{ marginInlineEnd: 0 }}>{t(`common.status.${status}`)}</Tag>;
}
