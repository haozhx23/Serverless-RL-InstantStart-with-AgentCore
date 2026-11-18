import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import type { Span, Tokens } from "../api/types";
import { costTrace } from "../lib/runStream";
import { costOf, int, opLabel, seconds, usd } from "../lib/format";
import { useFittedText } from "../lib/svgText";
import { useTheme } from "../theme/ThemeProvider";
import { alpha } from "../theme/tokens";

/** Each track's label, color and wash in the current theme and language. */
function useLanes() {
  const { t } = useTranslation();
  const { color } = useTheme();
  return {
    client: { label: t("common.lane.you"), long: t("common.lane.youLong"), color: color.you, wash: alpha(color.you, 0.1) },
    aws: { label: t("common.lane.aws"), long: t("common.lane.awsLong"), color: color.aws, wash: alpha(color.aws, 0.12) },
  };
}

// The run console's signature: two tracks over one time axis. YOU = this run's own process,
// AWS GPU = training-session primitives. The thin line underneath is the estimated cost; it
// only climbs at the end of AWS blocks, because only those are billed (docs/ui-design.md §4).

interface Props {
  spans: Span[];
  stepStarts: { step: number; t: number }[];
  price: Tokens | undefined;
  running: boolean;
  lastT: number;
  mode: "follow" | "whole";
  /** Precomputed cost points (MTRL: from the job's billing reports) instead of span tokens. */
  trace?: { t: number; usd: number }[];
}

const GUTTER = 78;
const RIGHT = 72;
const LANE_H = 28;
const Y_STEP = 14;
const Y_YOU = 22;
const Y_AWS = Y_YOU + LANE_H + 10;
const Y_AXIS = Y_AWS + LANE_H + 6;
const Y_TRACE_TOP = Y_AXIS + 26;
const TRACE_H = 40;
const HEIGHT = Y_TRACE_TOP + TRACE_H + 8;
const FOLLOW_WINDOW_S = 90;
const NICE_TICKS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600];

function useWidth() {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(900);
  useLayoutEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

export default function SplitTape({ spans, stepStarts, price, running, lastT, mode, trace: given }: Props) {
  const { t } = useTranslation();
  const { color } = useTheme();
  const lane = useLanes();
  const [ref, width] = useWidth();
  const [tip, setTip] = useState<{ x: number; y: number; block: Block } | null>(null);

  const plotW = Math.max(120, width - GUTTER - RIGHT);
  const end = Math.max(lastT, 1);
  const [d0, d1] = mode === "follow" && end > FOLLOW_WINDOW_S ? [end - FOLLOW_WINDOW_S, end] : [0, end * 1.02];
  const x = (t: number) => GUTTER + ((t - d0) / (d1 - d0)) * plotW;

  const tickStep = NICE_TICKS.find((s) => (d1 - d0) / s <= 8) ?? 3600;
  const ticks: number[] = [];
  for (let t = Math.ceil(d0 / tickStep) * tickStep; t <= d1; t += tickStep) ticks.push(t);

  const trace = useMemo(() => given ?? costTrace(spans, price), [given, spans, price]);
  const maxCost = trace.length ? trace[trace.length - 1].usd : 0;
  let maxLabel = usd(maxCost);
  let tapeClass = "tape-wrap tape";
  const ty = (v: number) => Y_TRACE_TOP + TRACE_H - (maxCost > 0 ? (v / maxCost) * TRACE_H : 0);
  const tracePath = useMemo(() => {
    if (trace.length < 2) return "";
    const pts: string[] = [];
    let prev = trace[0];
    pts.push(`M${x(Math.max(d0, prev.t))},${ty(prev.usd)}`);
    for (const p of trace.slice(1)) {
      pts.push(`L${x(Math.max(d0, p.t))},${ty(prev.usd)}`, `L${x(Math.max(d0, p.t))},${ty(p.usd)}`);
      prev = p;
    }
    pts.push(`L${x(end)},${ty(prev.usd)}`);
    return pts.join(" ");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trace, d0, d1, plotW, end, maxCost]);

  let lastLabelEnd = -Infinity;
  const blocks = toBlocks(spans.filter((s) => s.t0 + s.sec >= d0 && s.t0 <= d1), x, GUTTER, GUTTER + plotW);
  const labelOf = (b: Block) => (b.spans.length === 1 ? opLabel(b.spans[0].op) : b.spans.map((s) => opLabel(s.op)).join(" + "));
  const visibleSteps = stepStarts.filter((s) => s.t >= d0 && s.t <= d1);
  const stepText = (n: number) => t("loop.tape.step", { n });
  // A block shows its name only if the name fits inside it; step labels only where they do not collide.
  const blockLabels = [...new Set(blocks.map(labelOf))];
  const stepLabels = [...new Set(visibleSteps.map((s) => stepText(s.step)))];
  const gutter = { client: lane.client.label, aws: lane.aws.label, est: t("loop.tape.estCost") };
  const [probe, measured] = useFittedText({ blockLabels, stepLabels, gutter }, (fit) => ({
    block: Object.fromEntries(blockLabels.map((l) => [l, fit.width(l, "tape-block-label")])),
    step: Object.fromEntries(stepLabels.map((l) => [l, fit.width(l, "tape-step")])),
    client: fit.clip(gutter.client, GUTTER - 15 - 6, "tape-lane"),
    aws: fit.clip(gutter.aws, GUTTER - 15 - 6, "tape-lane"),
    est: fit.clip(gutter.est, GUTTER - 6, "tape-note"),
  }));

  const show = (block: Block, el: SVGGElement) => {
    const box = el.getBoundingClientRect();
    const host = ref.current!.getBoundingClientRect();
    const left = Math.min(box.left - host.left, host.width - 330);
    setTip({ x: Math.max(0, left), y: box.bottom - host.top + 6, block });
  };

  return (
    <div className={tapeClass} ref={ref}>
      <svg width={width} height={HEIGHT} role="group" aria-label={t("loop.tape.aria")}>
        <text ref={probe} visibility="hidden" aria-hidden />
        {(["client", "aws"] as const).map((k) => (
          <g key={k}>
            <rect x={0} y={(k === "client" ? Y_YOU : Y_AWS) + 9} width={10} height={10} rx={2} fill={lane[k].color} />
            <text x={15} y={(k === "client" ? Y_YOU : Y_AWS) + 18} className="tape-lane" fill={color.text}>{measured?.[k] ?? lane[k].label}</text>
          </g>
        ))}
        <text x={0} y={Y_TRACE_TOP + 16} className="tape-note" fill={color.text2}>{measured?.est ?? gutter.est}</text>

        <rect x={GUTTER} y={Y_YOU} width={plotW} height={LANE_H} fill={lane.client.wash} opacity={0.35} rx={3} />
        <rect x={GUTTER} y={Y_AWS} width={plotW} height={LANE_H} fill={lane.aws.wash} opacity={0.35} rx={3} />
        <line x1={GUTTER} x2={GUTTER + plotW} y1={Y_AXIS} y2={Y_AXIS} stroke={color.hairline} />
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={x(tick)} x2={x(tick)} y1={Y_AXIS} y2={Y_AXIS + 4} stroke={color.hairline} />
            <text x={x(tick)} y={Y_AXIS + 16} fontSize={11} textAnchor="middle" fill={color.text3} className="num">
              {tick >= 60 ? `${Math.floor(tick / 60)}:${String(Math.round(tick % 60)).padStart(2, "0")}` : `${tick}s`}
            </text>
          </g>
        ))}

        {visibleSteps.map((s) => {
          const sx = x(s.t);
          const text = stepText(s.step);
          const w = measured?.step[text];
          const label = w !== undefined && sx + 3 > lastLabelEnd + 8;
          if (label) lastLabelEnd = sx + 3 + w;
          return (
            <g key={`step-${s.step}`}>
              <line x1={sx} x2={sx} y1={Y_STEP} y2={Y_AXIS} stroke={color.text3} strokeWidth={1} opacity={0.5} />
              {label && <text x={sx + 3} y={Y_STEP + 4} className="tape-step" fill={color.text2}>{text}</text>}
            </g>
          );
        })}

        {blocks.map((b) => {
          const y = b.lane === "client" ? Y_YOU : Y_AWS;
          const w = b.x1 - b.x0;
          // 2px surface gap between touching blocks, so consecutive operations stay distinct.
          const [rx0, rw] = w > 4 ? [b.x0 + 1, w - 2] : [b.x0, Math.max(2, w)];
          const label = labelOf(b);
          const lw = measured?.block[label];
          const fits = lw !== undefined && rw >= lw + 12;
          const sec = b.spans.reduce((a, s) => a + s.sec, 0);
          return (
            <g key={b.key} className="block" tabIndex={0}
               aria-label={`${label}, ${lane[b.lane].long}, ${seconds(sec)}${b.ok ? "" : `, ${t("loop.tape.failed")}`}`}
               onMouseEnter={(e) => show(b, e.currentTarget)} onMouseLeave={() => setTip(null)}
               onFocus={(e) => show(b, e.currentTarget)} onBlur={() => setTip(null)}>
              <rect x={b.x0 - 3} y={y - 4} width={Math.max(10, w + 6)} height={LANE_H + 8} fill="transparent" />
              <rect x={rx0} y={y} width={rw} height={LANE_H} rx={3} fill={b.ok ? lane[b.lane].color : color.error} />
              {fits && <text x={rx0 + 6} y={y + 18} className="tape-block-label" fill={color.onLane} style={{ pointerEvents: "none" }}>{label}</text>}
            </g>
          );
        })}

        {tracePath && (
          <>
            <path d={`${tracePath} L${x(end)},${Y_TRACE_TOP + TRACE_H} L${x(Math.max(d0, 0))},${Y_TRACE_TOP + TRACE_H} Z`}
                  fill={lane.aws.wash} stroke="none" />
            <path d={tracePath} fill="none" stroke={lane.aws.color} strokeWidth={2} strokeLinejoin="round" />
            <text x={x(end) + 8} y={ty(maxCost) + 4} fontSize={12} fontWeight={600} fill={color.text} className="num">
              {maxLabel}
            </text>
          </>
        )}
        <line x1={GUTTER} x2={GUTTER + plotW} y1={Y_TRACE_TOP + TRACE_H} y2={Y_TRACE_TOP + TRACE_H} stroke={color.hairline} />

        {running && <line x1={x(end)} x2={x(end)} y1={Y_STEP} y2={Y_TRACE_TOP + TRACE_H} stroke={color.ink} opacity={0.35} />}
        {!spans.length && (
          <text x={GUTTER + 8} y={Y_AWS - 2} fontSize={13} fill={color.text3}>{t("loop.tape.waiting")}</text>
        )}
      </svg>
      {tip && <Tip {...tip} price={price} lane={lane} />}
    </div>
  );
}

interface Block { key: string; lane: Span["lane"]; x0: number; x1: number; ok: boolean; spans: Span[] }

/** Spans as drawn blocks. Adjacent spans too narrow to tell apart (a few ms of client work)
 *  merge into one block whose tooltip lists each of them, instead of stacking hit targets. */
function toBlocks(spans: Span[], x: (t: number) => number, lo: number, hi: number): Block[] {
  const out: Block[] = [];
  const last: Partial<Record<Span["lane"], Block>> = {};
  for (const s of [...spans].sort((a, b) => a.t0 - b.t0)) {
    const x0 = Math.max(lo, x(s.t0));
    const x1 = Math.max(x0 + 2, Math.min(hi, x(s.t0 + s.sec)));
    const prev = last[s.lane];
    // Only narrow neighbors merge; a narrow block next to a wide one stays its own block.
    if (prev && x0 - prev.x1 < 4 && x1 - x0 < 8 && prev.x1 - prev.x0 < 8) {
      prev.x1 = Math.max(prev.x1, x1);
      prev.ok = prev.ok && s.ok;
      prev.spans.push(s);
      continue;
    }
    const b: Block = { key: `${s.lane}-${s.seq}`, lane: s.lane, x0, x1, ok: s.ok, spans: [s] };
    out.push(b);
    last[s.lane] = b;
  }
  return out;
}

function Tip({ x, y, block, price, lane }: {
  x: number; y: number; block: Block; price?: Tokens; lane: ReturnType<typeof useLanes>;
}) {
  const { t } = useTranslation();
  const tokens: Tokens = { prefill: 0, sample: 0, train: 0 };
  for (const s of block.spans) for (const k of ["prefill", "sample", "train"] as const) tokens[k] += s.tokens?.[k] ?? 0;
  const sec = block.spans.reduce((a, s) => a + s.sec, 0);
  const steps = [...new Set(block.spans.map((s) => s.step).filter((v) => v !== null))];
  const tokenParts = (["prefill", "sample", "train"] as const).filter((k) => tokens[k]).map((k) => `${k} ${int(tokens[k])}`);
  const cost = block.lane === "aws" ? costOf(tokens, price) : 0;
  const one = block.spans.length === 1 ? block.spans[0] : null;
  let billed = tokenParts.length ? `≈ ${usd(cost)}` : one?.op.startsWith("job_") ? t("loop.tape.perToken") : t("common.notBilled");
  let sandboxNote = t("loop.tape.sandboxNote");
  return (
    <div className="tape-tip" style={{ left: x, top: y }} role="status">
      <div className="v">{seconds(sec)} · {one ? opLabel(one.op) : t("loop.tape.operations", { n: block.spans.length })}{block.ok ? "" : ` · ${t("loop.tape.failed")}`}</div>
      <div className="k">{lane[block.lane].long}{steps.length ? ` · ${t("loop.tape.stepList", { steps: steps.join(", ") })}` : ""}</div>
      {one?.note && <div>{one.note}</div>}
      {!one && <div>{block.spans.map((s) => `${opLabel(s.op)} ${seconds(s.sec)}`).join(" · ")}</div>}
      {tokenParts.length > 0 && <div>{tokenParts.join(" · ")}</div>}
      {block.lane === "aws" && (
        <div>{billed}</div>
      )}
      {block.lane === "client" && (one?.op === "agent_rollouts"
        ? <div className="k">{t("loop.tape.agentNote")}</div>
        : block.spans.some((s) => s.where)
          ? <div className="k">{sandboxNote}</div>
          : <div className="k">{t("loop.tape.localNote")}</div>)}
    </div>
  );
}

export interface StepRow {
  key: string; step: number | null; clientSec: number; awsSec: number; tokens: Tokens; usd: number | null;
}

/** The tape as a table: seconds per track, tokens and estimated cost for every step. */
export function stepBreakdown(spans: Span[], price: Tokens | undefined): StepRow[] {
  const rows = new Map<string, StepRow>();
  let seenStep = false;
  for (const s of spans) {
    const key = s.step !== null ? `s${s.step}` : seenStep ? "after" : "before";
    if (s.step !== null) seenStep = true;
    const r = rows.get(key) ?? { key, step: s.step, clientSec: 0, awsSec: 0, tokens: { prefill: 0, sample: 0, train: 0 }, usd: 0 };
    if (s.lane === "client") r.clientSec += s.sec;
    else r.awsSec += s.sec;
    for (const k of ["prefill", "sample", "train"] as const) r.tokens[k] += s.tokens?.[k] ?? 0;
    rows.set(key, r);
  }
  return [...rows.values()].map((r) => ({ ...r, usd: costOf(r.tokens, price) }));
}
