import { Typography } from "antd";
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";

export function PageHeader({ crumbs, title, extra, children }: { crumbs?: ReactNode; title: ReactNode; extra?: ReactNode; children?: ReactNode }) {
  return (
    <header className="page-header">
      {crumbs && <div className="eyebrow" style={{ textTransform: "none", letterSpacing: 0, fontWeight: 400 }}>{crumbs}</div>}
      <div className="row">
        <h1 className="display">{title}</h1>
        {extra && <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>{extra}</div>}
      </div>
      {children}
    </header>
  );
}

export function Section({ title, hint, extra, children, bodyClass }: { title: ReactNode; hint?: ReactNode; extra?: ReactNode; children: ReactNode; bodyClass?: string }) {
  return (
    <section className="section-card">
      <div className="head">
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
          <h2>{title}</h2>
          {hint && <span className="hint">{hint}</span>}
        </div>
        {extra}
      </div>
      <div className={`body ${bodyClass ?? ""}`} style={{ position: "relative" }}>{children}</div>
    </section>
  );
}

/** Monospace identifier with a copy button; ARNs and session names are things people paste. */
export function Ident({ value, short }: { value: string | null | undefined; short?: boolean }) {
  if (!value) return <span className="muted">—</span>;
  const shown = short && value.length > 48 ? `…${value.slice(-44)}` : value;
  return (
    <Typography.Text className="mono" copyable={{ text: value }} title={value}>
      {shown}
    </Typography.Text>
  );
}

export function QuotaPips({ limit, active, large }: { limit: number; active: number | null; large?: boolean }) {
  const { t } = useTranslation();
  const n = Math.min(limit, 16);
  return (
    <span className={`pips${large ? " large" : ""}`} aria-label={t("shell.pips", { active: active ?? "?", limit })}>
      {Array.from({ length: n }, (_, i) => <span key={i} className={`pip${active !== null && i < active ? " on" : ""}`} />)}
    </span>
  );
}

export function LaneKey({ who, text }: { who: "you" | "aws" | "sandbox"; text: ReactNode }) {
  return (
    <span className="lane-key">
      <span className={`swatch${who === "sandbox" ? " sandbox" : ""}`} style={{ background: who === "aws" ? "var(--aws)" : "var(--you)" }} />
      <span>{text}</span>
    </span>
  );
}

/** Text with its URLs made clickable (service messages sometimes carry the link to act on). */
export function Linkified({ text, className }: { text: string; className?: string }) {
  const parts = text.split(/(https?:\/\/[^\s]+[^\s.,;)])/g);
  return (
    <span className={className} style={{ wordBreak: "break-word" }}>
      {parts.map((p, i) => /^https?:\/\//.test(p) ? <a key={i} href={p} target="_blank" rel="noreferrer">{p}</a> : <span key={i}>{p}</span>)}
    </span>
  );
}

/** Where a quota's limit was read from, in the interface language (the backend reports it in English). */
export function useLimitSource() {
  const { t } = useTranslation();
  const known: Record<string, string> = {
    "account quota": t("shell.limitSource.account"),
    "AWS default": t("shell.limitSource.awsDefault"),
    "default (not read)": t("shell.limitSource.notRead"),
  };
  return (source: string) => known[source] ?? source;
}
