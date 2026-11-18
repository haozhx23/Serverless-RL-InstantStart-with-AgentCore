import { Tooltip } from "antd";
import { Fragment } from "react";
import { useTranslation } from "react-i18next";
import type { Backend, Billing, Engine, Tokens } from "../api/types";
import { costOf, int, timeOfDay, usd } from "../lib/format";

// The page's one hero number: estimated cost = tokens × list price, live. Underneath, the
// three billing dimensions, then what AWS has actually counted so far (it lags).
export default function CostMeter(props: {
  tokens: Tokens; price: Tokens | undefined; priceAsOf?: string; backend: Backend | null;
  billing?: Billing; billingError?: string; recordedCost?: number | null; engine?: Engine; priceBasis?: string;
}) {
  const { tokens, price, priceAsOf, backend, billing, billingError, recordedCost, engine, priceBasis } = props;
  const { t } = useTranslation();
  const mtrl = engine === "mtrl";
  const cost = costOf(tokens, price);
  const rows = (["prefill", "sample", "train"] as const).map((k) => ({
    k, n: tokens[k], usd: price ? (tokens[k] * price[k]) / 1e6 : null, unit: price?.[k],
  }));
  const max = Math.max(1, ...rows.map((r) => r.usd ?? 0));
  const help = { prefill: t("console.meter.helpPrefill"), sample: t("console.meter.helpSample"), train: t("console.meter.helpTrain") };
  const asOf = priceAsOf ? t("console.meter.asOf", { date: priceAsOf }) : "";
  return (
    <div>
      <div className="eyebrow">{backend === "replay" ? t("console.meter.replayEyebrow")
        : mtrl ? t("console.meter.jobEyebrow") : t("console.meter.liveEyebrow")}</div>
      <div className="meter-hero">{cost === null ? t("console.meter.priceUnknown") : usd(cost, 5)}</div>
      <div className="muted" style={{ fontSize: 12 }}>
        {mtrl ? t("console.meter.jobNote", { asOf }) : t("console.meter.liveNote", { asOf })}
      </div>
      <div className="meter-rows">
        {rows.map((r) => (
          <Fragment key={r.k}>
            <Tooltip title={help[r.k]}>
              <span tabIndex={0} style={{ textTransform: "capitalize", borderBottom: "1px dotted var(--text-3)" }}>{r.k}</span>
            </Tooltip>
            <span className="meter-bar" aria-hidden><span style={{ width: `${((r.usd ?? 0) / max) * 100}%` }} /></span>
            <span className="num mono" style={{ textAlign: "right" }}>{int(r.n)}</span>
          </Fragment>
        ))}
      </div>
      {price && (
        <div className="muted" style={{ fontSize: 11.5, marginTop: 6 }}>
          {t("console.meter.unitPrices", { prefill: price.prefill, sample: price.sample, train: price.train })}
        </div>
      )}
      {mtrl && priceBasis && <div className="muted" style={{ fontSize: 11.5, marginTop: 4 }}>{t("console.meter.basis", { basis: priceBasis })}</div>}
      {backend === "live" && !mtrl && (
        <div style={{ marginTop: 14, paddingTop: 10, borderTop: "1px solid var(--hairline)", fontSize: 13 }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
            <span>{t("console.meter.billed")}</span>
            <span className="num" style={{ fontWeight: 600 }}>{billing ? usd(billing.cost_usd, 5) : "—"}</span>
          </div>
          <div className="muted" style={{ fontSize: 12, lineHeight: 1.5, marginTop: 2 }}>
            {billingError ? t("console.meter.billingError", { error: billingError }) : billing
              ? t("console.meter.billingDetail", { time: timeOfDay(billing.checked_at), prefill: int(billing.tokens.prefill),
                                                   sample: int(billing.tokens.sample), train: int(billing.tokens.train) })
              : t("console.meter.billingPending")}
          </div>
        </div>
      )}
      {backend === "replay" && recordedCost !== undefined && (
        <div className="muted" style={{ fontSize: 12, marginTop: 10 }}>{t("console.meter.replayNote")}</div>
      )}
    </div>
  );
}
