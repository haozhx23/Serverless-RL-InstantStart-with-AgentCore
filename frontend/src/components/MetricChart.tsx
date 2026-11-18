import { useState } from "react";
import { useTranslation } from "react-i18next";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { MetricSpec } from "../api/types";
import { metric } from "../lib/format";
import { useTheme } from "../theme/ThemeProvider";

// One metric, one series, so no legend: the title names it. Thin ink line, value labeled at
// the end, crosshair tooltip, and a table view so no value is reachable only by hovering.
export default function MetricChart({ spec, points }: { spec: MetricSpec; points: { step: number; value: number }[] }) {
  const { t } = useTranslation();
  let cell = "metric-cell";
  const { color } = useTheme();
  const [asTable, setAsTable] = useState(false);
  const last = points[points.length - 1];
  const domain: [number | string, number | string] = [spec.min ?? "auto", spec.max ?? "auto"];
  return (
    <div className={cell}>
      <div className="title">
        <span>{spec.label}
        </span>
        <span className="last num">{last ? metric(last.value, spec.format) : "—"}</span>
      </div>
      {asTable ? (
        <table className="num" style={{ width: "100%", fontSize: 12.5, margin: "8px 0", borderCollapse: "collapse" }}>
          <thead><tr><th style={{ textAlign: "left" }}>{t("console.chart.step")}</th><th style={{ textAlign: "right" }}>{spec.label}</th></tr></thead>
          <tbody>{points.map((p) => (
            <tr key={p.step}><td>{p.step}</td><td style={{ textAlign: "right" }}>{metric(p.value, spec.format)}</td></tr>
          ))}</tbody>
        </table>
      ) : (
        <div style={{ height: 132 }}>
          {points.length === 0 ? (
            <div className="muted" style={{ fontSize: 12.5, paddingTop: 48, textAlign: "center" }}>{t("console.chart.noValues")}</div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={points} margin={{ top: 10, right: 12, bottom: 0, left: 0 }}>
                <CartesianGrid vertical={false} stroke={color.grid} />
                <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} allowDecimals={false}
                       tick={{ fontSize: 11, fill: color.text3 }} tickLine={false} axisLine={{ stroke: color.hairline }} />
                <YAxis domain={domain} width={44} allowDecimals={spec.format !== "0"} tick={{ fontSize: 11, fill: color.text3 }} tickLine={false}
                       axisLine={false} tickFormatter={(v: number) => metric(v, spec.format)} />
                <Tooltip cursor={{ stroke: color.ink, strokeWidth: 1 }} isAnimationActive={false}
                         content={({ active, payload }) => active && payload?.length ? (
                           <div className="tape-tip" style={{ position: "static" }}>
                             <div className="v">{metric(payload[0].value as number, spec.format)}</div>
                             <div className="k">{t("loop.tape.step", { n: payload[0].payload.step })}</div>
                           </div>) : null} />
                <Line dataKey="value" type="linear" stroke={color.ink} strokeWidth={2} isAnimationActive={false}
                      dot={(p: { cx?: number; cy?: number; index?: number }) => p.index === points.length - 1 && p.cx !== undefined
                        ? <circle key="end" cx={p.cx} cy={p.cy} r={4} fill={color.ink} stroke={color.surface} strokeWidth={2} />
                        : <g key={p.index} />}
                      activeDot={{ r: 4, stroke: color.surface, strokeWidth: 2, fill: color.ink }} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      )}
      <button type="button" className="linklike" onClick={() => setAsTable((v) => !v)}>
        {asTable ? t("console.chart.asChart") : t("console.chart.asTable")}
      </button>
    </div>
  );
}
