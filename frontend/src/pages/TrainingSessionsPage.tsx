import { Tooltip } from "antd";
import { useScenarios } from "../api/hooks";
import { Trans, useTranslation } from "react-i18next";
import { LaneKey, PageHeader } from "../components/Bits";
import ScenarioLibrary, { EngineStatus } from "../components/ScenarioLibrary";
import { primitives } from "../lib/primitives";

/** Training sessions: your process runs the loop and every block on SageMaker AI is one API call (docs/ui-design.md §6.1.1). */
export default function TrainingSessionsPage() {
  const { t } = useTranslation();
  // The third legend entry only when a scenario on this page runs code in the environment sandbox.
  const sandbox = useScenarios().data?.scenarios.some((s) => s.engine === "training-session" && s.sandbox && s.status !== "planned");
  return (
    <>
      <PageHeader crumbs={t("scenarios.crumbs")} title={t("shell.nav.ts")} />
      <div className="page-body" style={{ display: "flex", flexDirection: "column", gap: 18 }}>
        <section className="thesis">
          <div>
            <h2>
              <Trans i18nKey="scenarios.thesis.title" components={{ you: <span className="you" />, aws: <span className="aws" /> }} />
            </h2>
            <p className="thesis-sub">{t("scenarios.thesis.sub")}</p>
            <div className="prim-strip" aria-label={t("scenarios.thesis.apiAria")}>
              <span className="ledger-head">{t("scenarios.thesis.api")}</span>
              {primitives().map((p) => (
                <Tooltip key={p.name} title={p.what}><span className="prim used"><span className="mono">{p.name}</span></span></Tooltip>
              ))}
            </div>
            <div className="legend-row" style={{ marginTop: 12 }}>
              <LaneKey who="you" text={<Trans i18nKey="scenarios.thesis.legendYou" components={{ b: <b /> }} />} />
              <LaneKey who="aws" text={<Trans i18nKey="scenarios.thesis.legendAws" components={{ b: <b /> }} />} />
              {sandbox && <LaneKey who="sandbox" text={<Trans i18nKey="scenarios.thesis.legendSandbox" components={{ b: <b /> }} />} />}
            </div>
          </div>
          <EngineStatus engine="training-session" note={t("scenarios.thesis.status")} />
        </section>
        <ScenarioLibrary engine="training-session" />
      </div>
    </>
  );
}
