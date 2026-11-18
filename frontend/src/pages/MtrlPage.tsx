import { Tooltip } from "antd";
import { Trans, useTranslation } from "react-i18next";
import { LaneKey, PageHeader } from "../components/Bits";
import ScenarioLibrary, { EngineStatus } from "../components/ScenarioLibrary";

// What an MTRL agent's code uses, in the order a rollout meets them: the SDK's agent contract
// (sagemaker.train.rft, sagemaker.core.token_generator in sagemaker-train 1.27.0); what each does is
// interface text (scenarios.mtrlThesis.items).
const CONTRACT = [
  { key: "handler", name: "@sagemaker_rft_handler" },
  { key: "headers", name: "make_inference_headers(metadata)" },
  { key: "token", name: "generate_token(region)" },
  { key: "endpoint", name: "{endpoint}/v1" },
  { key: "reward", name: 'return {"reward": …}' },
] as const;

/** Multi-turn RL: SageMaker AI runs the loop and calls your agent for every rollout (docs/ui-design.md §6.1.2). */
export default function MtrlPage() {
  const { t } = useTranslation();
  return (
    <>
      <PageHeader crumbs={t("scenarios.crumbs")} title={t("scenarios.mtrlPageTitle")} />
      <div className="page-body" style={{ display: "flex", flexDirection: "column", gap: 18 }}>
        <section className="thesis">
          <div>
            <h2>
              <Trans i18nKey="scenarios.mtrlThesis.title" components={{ you: <span className="you" />, aws: <span className="aws" /> }} />
            </h2>
            <p className="thesis-sub">{t("scenarios.mtrlThesis.sub")}</p>
            {/* Code in your agent, so the "you" tint rather than the amber of service calls. */}
            <div className="prim-strip" aria-label={t("scenarios.mtrlThesis.contractAria")}>
              <span className="ledger-head">{t("scenarios.mtrlThesis.contract")}</span>
              {CONTRACT.map((c) => (
                <Tooltip key={c.key} title={t(`scenarios.mtrlThesis.items.${c.key}`)}>
                  <span className="prim you"><span className="mono">{c.name}</span></span>
                </Tooltip>
              ))}
            </div>
            <div className="legend-row" style={{ marginTop: 12 }}>
              <LaneKey who="you" text={<Trans i18nKey="scenarios.mtrlThesis.legendYou" components={{ b: <b /> }} />} />
              <LaneKey who="aws" text={<Trans i18nKey="scenarios.mtrlThesis.legendAws" components={{ b: <b /> }} />} />
            </div>
          </div>
          <EngineStatus engine="mtrl" note={t("scenarios.mtrlThesis.status")} />
        </section>
        <ScenarioLibrary engine="mtrl" />
      </div>
    </>
  );
}
