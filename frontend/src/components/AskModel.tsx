import { Alert, App, Button, Empty, Input, Table } from "antd";
import { SendOutlined } from "@ant-design/icons";
import { useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { api } from "../api/client";
import type { Backend } from "../api/types";
import type { RunView } from "../lib/runStream";
import { useEventText } from "../i18n/content";

// Ask the model a question while the loop runs. The question is left in the run's inbox;
// the loop answers it between two steps with whatever weights it has at that point, so the
// answer is a reading of training in progress. The UI never samples the model itself.

export default function AskModel({ runId, view, alive, backend, snapshot, example }: {
  runId: string; view: RunView; alive: boolean; backend: Backend | null; snapshot: boolean;
  /** A question in the training data's language, from the scenario (scenario.json probe_example). */
  example: string | null;
}) {
  const { t } = useTranslation();
  const te = useEventText();
  const { message } = App.useApp();
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [queued, setQueued] = useState<{ id: string; question: string }[]>([]);
  const answered = new Set(view.probes.map((p) => p.id));
  const pending = queued.filter((q) => !answered.has(q.id));
  const canAsk = alive && backend !== "replay" && !view.finished;

  const send = async () => {
    if (!text.trim()) return;
    setBusy(true);
    try {
      const r = await api<{ id: string; question: string }>(`/api/runs/${runId}/probe`, { body: { question: text } });
      setQueued((q) => [...q, r]);
      setText("");
    } catch (e) { message.error((e as Error).message); }
    finally { setBusy(false); }
  };

  return (
    <>
      <p className="secondary" style={{ fontSize: 12.5, marginTop: 0 }}>
        <Trans i18nKey="console.ask.intro" components={{ code: <span className="mono" /> }}
               values={{ target: snapshot ? t("console.ask.targetSnapshot") : t("console.ask.targetSampler") }} />
      </p>
      {canAsk ? (
        <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
          <Input value={text} onChange={(e) => setText(e.target.value)} onPressEnter={send} maxLength={500}
                 placeholder={example ? t("console.ask.placeholderExample", { example }) : t("console.ask.placeholder")} />
          <Button type="primary" icon={<SendOutlined />} loading={busy} onClick={send} disabled={!text.trim()}>{t("console.ask.send")}</Button>
        </div>
      ) : (
        <Alert type="info" showIcon style={{ marginBottom: 12 }}
               title={backend === "replay" ? t("console.ask.replay") : t("console.ask.notRunning")} />
      )}
      {pending.length > 0 && (
        <p className="muted" style={{ fontSize: 12.5 }}>{t("console.ask.waiting", { list: pending.map((q) => `“${q.question}”`).join(", ") })}</p>
      )}
      {view.probes.length ? (
        <Table size="small" pagination={false} rowKey="id" dataSource={[...view.probes].reverse()} columns={[
          { title: t("console.ask.question"), dataIndex: "question" },
          { title: t("console.ask.answer"), dataIndex: "answer", render: (v: string) => <span className="mono">{v || "∅"}</span> },
          { title: t("console.ask.atStep"), dataIndex: "step", width: 80, align: "right" },
          { title: t("console.ask.answeredBy"), dataIndex: "weights", width: 230, render: (v: string) => te(v) },
        ]} />
      ) : !pending.length && <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={t("console.ask.empty")} />}
    </>
  );
}
