import type { Tokens } from "../api/types";
import i18n, { localeTag } from "../i18n";

export function usd(value: number | null | undefined, digits?: number): string {
  if (value === null || value === undefined) return "—";
  if (value === 0) return "$0";
  const d = digits ?? (value < 0.01 ? 4 : value < 1 ? 3 : 2);
  return `$${value.toFixed(d)}`;
}

/** A measured range, or one value when both ends print the same. */
export function usdRange(r: { low: number; high: number } | null | undefined, digits?: number): string {
  if (!r) return "—";
  const [lo, hi] = [usd(r.low, digits), usd(r.high, digits)];
  return lo === hi ? lo : `${lo}–${hi}`;
}

export const int = (n: number | null | undefined) => (n === null || n === undefined ? "—" : n.toLocaleString("en-US"));

export function seconds(s: number | null | undefined): string {
  if (s === null || s === undefined) return "—";
  if (s < 0.1) return `${Math.round(s * 1000)} ms`;
  if (s < 10) return `${s.toFixed(1)} s`;
  if (s < 90) return `${Math.round(s)} s`;
  const whole = Math.round(s);  // round first: 239.6 s is 4 min 0 s, not 3 min 60 s
  const m = Math.floor(whole / 60);
  const r = whole % 60;
  return m < 60 ? `${m} min ${r} s` : `${Math.floor(m / 60)} h ${m % 60} min`;
}

export function clock(ts: number | null | undefined): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString(localeTag(), { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function timeOfDay(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString(localeTag(), { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/** An ISO timestamp from the backend, as date and time in the interface language. */
export function dateTime(iso: string | null | undefined): string {
  return iso ? new Date(iso).toLocaleString(localeTag()) : "—";
}

/** Format a metric with the scenario's format string: "0.000", "0.00", "0%", "0.0000". */
export function metric(value: number | undefined, format: string): string {
  if (value === undefined || Number.isNaN(value)) return "—";
  if (format.endsWith("%")) return `${Math.round(value * 100)}%`;
  if (format === "0" && Math.abs(value) >= 10_000) {
    return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 }).format(value);
  }
  const decimals = (format.split(".")[1] ?? "").length;
  return value.toFixed(decimals);
}

export function costOf(tokens: Partial<Tokens>, price: Tokens | undefined): number | null {
  if (!price) return null;
  return ((tokens.prefill ?? 0) * price.prefill + (tokens.sample ?? 0) * price.sample + (tokens.train ?? 0) * price.train) / 1e6;
}

/** The short name of an operation, as shown on phase chips and timeline blocks. */
export function opLabel(op: string): string {
  const labels = i18n.t("loop.ops", { returnObjects: true }) as Record<string, string>;
  return labels[op] ?? op.replaceAll("_", " ");
}
