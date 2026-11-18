import { useTranslation } from "react-i18next";
import type { CostEstimate } from "../api/types";
import { usd, usdRange } from "../lib/format";

// What a live run with given settings costs, measured on live runs with the same settings
// (backend/srl/estimate.py). Shown in the launch drawer. No number is shown that no live run measured.
export default function CostEstimateView({ estimate, small }: { estimate: CostEstimate | undefined; small?: boolean }) {
  const { t } = useTranslation();
  if (!estimate) return <span className="muted">…</span>;
  const e = estimate;
  // A sandbox scenario's AgentCore Runtime cost is measured per run, never estimated (backend/srl/estimate.py).
  const agentcore = e.agentcore === "not_estimated"
    ? <p className="muted estimate-note">{t("common.estimate.agentcoreNot")}</p> : null;
  const dates = [...new Set(e.basis.map((b) => b.date))].join(", ");
  const count = e.basis.length;
  const hero = small ? "estimate-hero small" : "meter-hero";
  if (e.status === "none") {
    const r = e.reason!;
    return (
      <div className="estimate">
        <div className={`${hero} none`}>{t("common.estimate.none")}</div>
        <p className="secondary estimate-text">
          {t(`common.estimate.reason.${r.code}`, { model: r.model, models: (r.measured_models ?? []).join(", ") })}
        </p>
        {e.nearest.length > 0 && (
          <div className="estimate-nearest">
            <div className="muted">{t("common.estimate.nearest")}</div>
            {e.nearest.map((n) => (
              <div key={n.run} className="estimate-row">
                <span className="num">{t("common.estimate.nearestRow", { date: n.date, steps: n.steps, usd: usd(n.usd, 5) })}</span>
                {" "}
                <span className="muted">
                  {n.custom ? t("common.estimate.editedCode") : n.differs?.length
                    ? t("common.estimate.differsRow", { list: n.differs.map((d) => `${d.key} ${String(d.measured)} → ${String(d.planned)}`).join(", ") })
                    : t("common.estimate.sameSettings")}
                </span>
              </div>
            ))}
          </div>
        )}
        {agentcore}
      </div>
    );
  }
  const vars = { count, dates, steps: e.steps, measured: e.measured_steps, rest: e.unestimated_steps,
                 low: usd(e.step_usd?.low, 5), high: usd(e.step_usd?.high, 5) };
  return (
    <div className="estimate">
      <div className={hero}>{usdRange(e.usd, 5)}</div>
      {e.status === "partial" && <div className="estimate-scope">{t("common.estimate.partialHead", vars)}</div>}
      <p className="secondary estimate-text">
        {t(`common.estimate.${e.status}`, vars)}
        {count > 1 && e.usd!.low !== e.usd!.high && <> {t("common.estimate.spread")}</>}
      </p>
      <p className="muted estimate-note">{t("common.estimate.basisNote", { asOf: e.price_as_of })}</p>
      {agentcore}
    </div>
  );
}
