import { Button, Tooltip } from "antd";
import { useTranslation } from "react-i18next";
import type { Scenario } from "../api/types";
import { joinList } from "../lib/engines";
import { seconds, usd } from "../lib/format";
import { primitivesOf } from "../lib/primitives";
import type { Mode } from "./LaunchDrawer";
import LoopMap from "./LoopMap";

// A scenario on its engine's page: a card with its loop and launch buttons, or a planned one
// with no buttons (docs/ui-design.md §6.1).

export function ScenarioCard({ s, liveReady, blockers, onLaunch }: {
  s: Scenario; liveReady: boolean; blockers: string[]; onLaunch: (m: Mode) => void;
}) {
  const { t } = useTranslation();
  const rec = s.recordings[s.recordings.length - 1];
  const mtrl = s.engine === "mtrl";
  return (
    <article className="scenario-card">
      <span className="eyebrow">{t(mtrl ? "scenarios.card.eyebrowMtrl" : "scenarios.card.eyebrowTs", { technique: s.technique })}</span>
      <h3 className="display">{s.title}</h3>
      {s.source && <OfficialSource source={s.source} />}
      <p className="summary">{s.summary}</p>
      {s.loop ? <MiniLoop s={s} /> : (
        <div className="labor">
          <span className="who you">{t("scenarios.card.youWrite")}</span>
          <span className="items">{s.you_write?.map((f) => <span className="mono" key={f}>{f}</span>)}</span>
          <span className="who aws">{t("scenarios.card.awsRuns")}</span>
          <span className="items">{s.aws_runs?.map((op) => <span key={op}>{op}</span>)}</span>
        </div>
      )}
      <div className="labor-line">
        <span className="who you">{t("scenarios.card.youWrite")}</span>
        <span>{s.you_write?.map((f) => <span className="mono" key={f}>{f} </span>)}{mtrl && <span className="muted">{t("scenarios.card.deployed")}</span>}</span>
      </div>
      {s.sandbox && (
        <div className="labor-line">
          <span className="who you">{t("scenarios.card.sandbox")}</span>
          <span>{s.sandbox.executes} <span className="muted">· {t("scenarios.card.sandboxBilled")}</span></span>
        </div>
      )}
      {!!s.slots?.length && (
        <div className="labor-line">
          <span className="who">{t("scenarios.card.swap")}</span>
          <span>{s.slots.map((sl) => <span key={sl.key} className="slot-pill">⇄ {sl.label}{sl.editable ? " ✎" : ""}</span>)}</span>
        </div>
      )}
      {s.you_will_see && (
        <ul style={{ margin: 0, paddingInlineStart: 18, fontSize: 13, color: "var(--text-2)", lineHeight: 1.55 }}>
          {s.you_will_see.map((t) => <li key={t}>{t}</li>)}
        </ul>
      )}
      <div className="evidence">
        {/* The scenario's own evidence first: the note under it describes that run (default settings), while the
            latest recording may be a run of another slot variant. */}
        {s.status === "live-verified" && s.evidence ? (
          <>{t("scenarios.card.evidenceLive", { date: s.evidence.date, duration: seconds(s.evidence.duration_s),
                                            cost: usd(s.evidence.cost_usd), model: s.evidence.model })}</>
        ) : rec ? (
          <>{t("scenarios.card.lastLive", { date: new Date(rec.recorded_ts * 1000).toISOString().slice(0, 10),
                                            duration: seconds(rec.duration_s), cost: usd(rec.cost_usd), model: rec.model })}</>
        ) : s.evidence && s.evidence.duration_s !== null ? (
          <>
            <span>{t("scenarios.card.adapted", { date: s.evidence.date, duration: seconds(s.evidence.duration_s),
                                                  cost: usd(s.evidence.cost_usd), model: s.evidence.model })}</span>
            <div className="muted">{t("scenarios.card.notLive")}</div>
          </>
        ) : s.evidence ? <span>{s.evidence.note}</span> : null}
        {s.status === "live-verified" && s.evidence?.note && <div className="muted">{s.evidence.note}</div>}
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <Tooltip title={liveReady ? "" : t("scenarios.card.finishSetup", { blockers: joinList(t, blockers) })}>
          <Button type="primary" disabled={!liveReady} onClick={() => onLaunch("live")}>
            {mtrl ? t("scenarios.card.submitJob") : t("scenarios.card.launchLive")}
          </Button>
        </Tooltip>
        {s.recordings.length > 0 && <Button onClick={() => onLaunch("replay")}>{t("scenarios.card.replay")}</Button>}
      </div>
    </article>
  );
}

/** A scenario that cannot run in this app yet: what it is and when it might. */
export function PlannedCard({ s }: { s: Scenario }) {
  const { t } = useTranslation();
  return (
    <div className="planned">
      <span className="eyebrow">{t("scenarios.planned", { technique: s.technique })}</span>
      <strong>{s.title}</strong>
      {s.source && <OfficialSource source={s.source} />}
      <div>{s.summary}</div>
      {s.loop && <MiniLoop s={s} />}
      <div className="muted" style={{ marginTop: 6 }}>{s.plan}</div>
    </div>
  );
}

/** Where an official sample comes from, and what AWS published for it. */
function OfficialSource({ source }: { source: NonNullable<Scenario["source"]> }) {
  const { t } = useTranslation();
  return (
    <div className="official-source">
      <span className="official-badge">{t("scenarios.card.official")}</span>
      <a href={source.url} target="_blank" rel="noreferrer" className="mono">
        {t("scenarios.card.sourceLine", { repo: source.repo, commit: source.commit || "main", license: source.license })}
      </a>
      {source.published && (
        <div className="published">
          <b>{t("scenarios.card.published")}</b> {source.published}
          {source.published_url && <> (<a href={source.published_url} target="_blank" rel="noreferrer">{t("scenarios.card.readMore")}</a>)</>}
        </div>
      )}
    </div>
  );
}

function MiniLoop({ s }: { s: Scenario }) {
  const { t } = useTranslation();
  const loop = s.loop!;
  const mtrl = loop.driver === "aws";
  const prims = mtrl ? [] : primitivesOf(loop);
  return (
    <div className="mini-loop">
      <LoopMap loop={loop} mode="compact" ariaLabel={t("scenarios.card.miniAria", { title: s.title })}
               slotLabels={Object.fromEntries((s.slots ?? []).map((sl) => [sl.key, sl.label]))}
               zoneTitles={mtrl ? { client: [t("loop.zone.agent"), ""], aws: [t("loop.zone.awsRuns"), ""] }
                 : { client: [t("loop.zone.process"), ""], aws: [t("loop.zone.aws"), ""] }} />
      {prims.length > 0 && <div className="muted mini-prims">{t("scenarios.card.calls")} <span className="mono">{prims.join(" · ")}</span></div>}
    </div>
  );
}
