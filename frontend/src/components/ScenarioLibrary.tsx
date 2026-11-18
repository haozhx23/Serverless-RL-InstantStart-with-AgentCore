import { Alert } from "antd";
import { useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Link } from "react-router";
import { useReadiness, useScenarios } from "../api/hooks";
import type { Engine, Scenario } from "../api/types";
import { useLang } from "../i18n";
import { joinList, readyFor } from "../lib/engines";
import LaunchDrawer, { type Mode } from "./LaunchDrawer";
import { PlannedCard, ScenarioCard } from "./ScenarioCard";

// The part of a scenario page both engines share (docs/ui-design.md §6.1). Each page puts its
// own thesis strip above this.

// The group title says which side of the API the loop is on; the page title names the product.
const GROUP = {
  "training-session": { title: "tsTitle", detail: "tsDetail" },
  mtrl: { title: "mtrlTitle", detail: "mtrlDetail" },
} as const;

/** One engine's scenarios: the runnable cards, the planned ones and the launch drawer. */
export default function ScenarioLibrary({ engine }: { engine: Engine }) {
  const { t } = useTranslation();
  const lang = useLang();
  const { data, error } = useScenarios();
  const readiness = useReadiness();
  const [launch, setLaunch] = useState<{ scenario: Scenario; mode: Mode } | null>(null);
  const all = data?.scenarios.filter((s) => (s.engine ?? "training-session") === engine) ?? [];
  // Keep official examples first, then make the AgentCore sandbox workflow easy to find.
  const runnable = all.filter((s) => s.status !== "planned")
    .sort((a, b) => Number(!!b.source) - Number(!!a.source) || Number(!!b.sandbox) - Number(!!a.sandbox));
  const planned = all.filter((s) => s.status === "planned");

  return (
    <>
      {error && <Alert type="error" title={t("scenarios.loadError", { error: (error as Error).message })} />}
      <section className="engine-group">
        <div className="engine-head">
          <h2 className="display">{t(`scenarios.engine.${GROUP[engine].title}`)}</h2>
          <span className="secondary">{t(`scenarios.engine.${GROUP[engine].detail}`)}</span>
        </div>
        <div className="scenario-grid">
          {runnable.map((s) => {
            // A sandbox scenario also needs the environment sandbox (docs/ui-design.md §6.1).
            const { ready, blockers } = readyFor(engine, readiness.data, lang, !!s.sandbox);
            return <ScenarioCard key={s.id} s={s} liveReady={ready} blockers={blockers}
                                 onLaunch={(mode) => setLaunch({ scenario: s, mode })} />;
          })}
        </div>
        {planned.length > 0 && (
          <div className="planned-row" style={{ marginTop: 12 }}>
            {planned.map((s) => <PlannedCard key={s.id} s={s} />)}
          </div>
        )}
      </section>
      {launch && data && (
        <LaunchDrawer key={`${launch.scenario.id}-${launch.mode}`} scenario={launch.scenario} initialMode={launch.mode}
                      readiness={readiness.data} models={data.models} onClose={() => setLaunch(null)} />
      )}
    </>
  );
}

/** The thesis strip's last row: what every run of this engine shows, and whether it can run live here. */
export function EngineStatus({ engine, note }: { engine: Engine; note: string }) {
  const { t } = useTranslation();
  const lang = useLang();
  const readiness = useReadiness();
  const r = readiness.data;
  const { ready, blockers } = readyFor(engine, r, lang);
  const label = t(engine === "mtrl" ? "scenarios.thesis.mtrl" : "scenarios.thesis.ts");
  return (
    <div className="status">
      <span className="secondary">{note}</span>
      {readiness.isLoading ? <span className="muted">{t("scenarios.thesis.checking")}</span> : !r ? null : (
        <span style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
          {ready
            ? <span><Trans i18nKey="scenarios.thesis.ready" values={{ label, region: r.region }} components={{ b: <b /> }} /></span>
            : <span><Trans i18nKey="scenarios.thesis.blocked" components={{ err: <span style={{ color: "var(--error)" }} /> }}
                           values={{ label, blockers: joinList(t, blockers) || t("scenarios.thesis.setupFallback") }} /></span>}
          <Link to="/setup">{t("shell.nav.setup")}</Link>
          <Link to="/paths">{t("scenarios.thesis.paths")}</Link>
        </span>
      )}
    </div>
  );
}
