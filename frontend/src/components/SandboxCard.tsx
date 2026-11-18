import { useTranslation } from "react-i18next";
import type { AgentcoreUsage, Backend } from "../api/types";
import { int, seconds, timeOfDay, usd } from "../lib/format";
import type { SandboxView } from "../lib/runStream";
import { Ident } from "./Bits";

// The environment sandbox's side of a run (docs/ui-design.md §6.2): where the code ran, how many
// sessions it opened and stopped, their cold starts. What AgentCore billed is in the meter, so
// the page shows one amount for it, not two.

export function SandboxCard({ sandbox }: { sandbox: SandboxView }) {
  const { t } = useTranslation();
  const name = sandbox.runtime?.split("/").pop()?.replace(/-[A-Za-z0-9]{10}$/, "");
  let where: React.ReactNode = <><Ident value={name ?? "?"} /> · {t("console.sandbox.platform", { platform: sandbox.platform ?? "?" })}</>;
  let isolation = t("console.sandbox.isolation");
  return (
    <>
      <dl className="kv">
        <dt>{t("console.sandbox.where")}</dt>
        <dd>{where}</dd>
        <dt>{t("console.sandbox.sessions")}</dt>
        <dd className="num">{t("console.sandbox.sessionsValue", { created: int(sandbox.created), stopped: int(sandbox.stopped) })}
          {sandbox.open > 0 && !sandbox.closed ? t("console.sandbox.openNow", { n: sandbox.open }) : ""}</dd>
        <dt>{t("console.sandbox.cold")}</dt>
        <dd className="num">{sandbox.cold ? t("console.sandbox.coldValue", { p50: seconds(sandbox.cold.p50), max: seconds(sandbox.cold.max) }) : "—"}</dd>
        <dt>{t("console.sandbox.requests")}</dt>
        <dd className="num">{int(sandbox.calls)}{sandbox.retries ? t("console.sandbox.retried", { n: sandbox.retries }) : ""}</dd>
        <dt>{t("console.sandbox.code")}</dt><dd className="mono">sandbox_exec {sandbox.codeSha}</dd>
      </dl>
      {sandbox.failedStops.length > 0 && (
        <p style={{ fontSize: 12.5, color: "var(--error)", margin: "8px 0 0" }}>
          {t("console.sandbox.notStopped", { n: sandbox.failedStops.length, error: sandbox.failedStops[0].error })}
        </p>
      )}
      <p className="muted" style={{ fontSize: 12.5, margin: "10px 0 0" }}>
        {isolation}
      </p>
    </>
  );
}

/** The meter's AgentCore Runtime line for a live run: the sandbox's sessions, or (MTRL) the agent's runtime. */
export function AgentcoreLine({ kind, usage, error, backend }: {
  kind: "sandbox" | "agent"; usage?: AgentcoreUsage; error?: string; backend: Backend | null;
}) {
  const { t } = useTranslation();
  if (backend === "replay") {
    return <div className="muted" style={{ fontSize: 12, marginTop: 10 }}>{t("console.meter.agentcoreReplay")}</div>;
  }
  if (backend !== "live") return null;
  const reported = usage && (usage.kind === "agent" ? usage.vcpu_hours > 0 || usage.gb_hours > 0 : (usage.sessions_reported ?? 0) > 0);
  return (
    <div style={{ marginTop: 14, paddingTop: 10, borderTop: "1px solid var(--hairline)", fontSize: 13 }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
        <span>{t(kind === "agent" ? "console.meter.agentcoreAgent" : "console.meter.agentcoreSandbox")}</span>
        <span className="num" style={{ fontWeight: 600 }}>
          {usage && reported ? (usage.usd === null ? t("console.meter.priceUnknown") : usd(usage.usd, 5)) : "—"}
        </span>
      </div>
      <div className="muted" style={{ fontSize: 12, lineHeight: 1.5, marginTop: 2 }}>
        {error ? t("console.meter.billingError", { error })
          : !usage ? t("console.meter.agentcorePending")
          : !reported ? t("console.meter.agentcoreNotYet", { time: timeOfDay(usage.checked_at) })
          : usage.kind === "sandbox"
            ? t("console.meter.agentcoreSessions", { reported: int(usage.sessions_reported ?? 0), created: int(usage.sessions_created ?? 0),
                                                     vcpu: (usage.vcpu_hours * 3600).toFixed(0), gb: (usage.gb_hours * 3600).toFixed(0),
                                                     time: timeOfDay(usage.checked_at) })
            : t("console.meter.agentcoreRuntime", { vcpu: (usage.vcpu_hours * 3600).toFixed(0), gb: (usage.gb_hours * 3600).toFixed(0),
                                                    time: timeOfDay(usage.checked_at) })}
      </div>
      <div className="muted" style={{ fontSize: 11.5, marginTop: 4 }}>
        {t(kind === "agent" ? "console.meter.agentcoreNoteAgent" : "console.meter.agentcoreNote",
           { asOf: usage?.price_as_of ?? "2026-11-10", platform: usage?.platform ?? "?" })}
      </div>
    </div>
  );
}
