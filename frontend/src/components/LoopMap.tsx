import type { KeyboardEvent, ReactNode } from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import type { Lane, LoopNode, LoopSpec, Span, Tokens } from "../api/types";
import { costOf, int, seconds, usd } from "../lib/format";
import { useFittedText, type Fit } from "../lib/svgText";

// The run console's signature (docs/ui-design.md §4): the training loop drawn as the code
// runs it. Left: your process. Right: SageMaker AI. The dashed line between them is the API;
// every arrow that crosses it is a call. Nodes are in code order; a call from your side goes
// straight across, the result comes back as an elbow, and the wire along the bottom returns
// control to the for-loop, on whichever side owns it (your process, or the job in MTRL).

export interface NodeStat {
  count: number; sec: number; calls: number; tokens: Tokens; last: Span | null; failed: number;
  sandboxCalls: number;  // requests to the environment sandbox (AgentCore Runtime), not to SageMaker AI
}

type Mode = "full" | "compose" | "compact";

interface Props {
  loop: LoopSpec;
  mode?: Mode;
  params?: Record<string, unknown>;
  stats?: Record<string, NodeStat>;
  active?: string | null;          // op of the node running now; "__header" for the loop itself
  selected?: string | null;
  onSelect?: (op: string) => void;
  /** Ops that can be clicked; default: every node when onSelect is given. */
  clickable?: (node: LoopNode | "header") => boolean;
  slotLabels?: Record<string, string>;   // slot key -> chosen variant label
  zoneTitles?: { client: [string, string]; aws: [string, string] };
  rightIsYours?: boolean;                 // HyperPod comparison: both sides are the customer's
  awsFooter?: ReactNode;
  clientFooter?: ReactNode;
  headerDetail?: string;
  price?: Tokens;
  dimInside?: boolean;                    // MTRL: the job is training, its inside is not observable
  ariaLabel?: string;
}

// Labels are fitted to these boxes by measuring them as drawn (lib/svgText.ts), not by counting characters.
// edge/edgeMin: an edge label's font size, and the smallest it is shrunk to before it is cut (.lm-edge-label is 10.5px).
const DIMS = {
  full: { pad: 30, nw: 236, api: 150, nh: 60, rp: 82, zt: 50, foot: 56, title: 13, call: 10.5, stat: 11, edge: 10.5, edgeMin: 9 },
  compose: { pad: 26, nw: 220, api: 120, nh: 46, rp: 64, zt: 44, foot: 28, title: 12.5, call: 10, stat: 0, edge: 10.5, edgeMin: 9 },
  compact: { pad: 14, nw: 150, api: 56, nh: 26, rp: 36, zt: 22, foot: 10, title: 10.5, call: 0, stat: 0, edge: 0, edgeMin: 0 },
} as const;

// An edge label's lines: a line is taken to be as tall as its font size, ASC of it above the baseline (generous
// enough for CJK glyphs); lineH is the distance between baselines.
const ASC = 0.8;
const lineH = (size: number) => size + 1.5;

interface Placed { key: string; node: LoopNode | null; lane: Lane; row: number; x: number; y: number }
// max: the room for the edge's label, in svg units, from where it is anchored to the nearest node or line.
// band: the label's free vertical room [top, bottom]: no node, other label or line inside it (docs/ui-design.md §8).
// stack "up": the lines end at ly and grow upward (a label sitting on its line); "center": they are centred on ly.
// Either way they are moved only as far as it takes to stay inside the band. lines: the most lines it may take.
// alt: the other side of a vertical line, for a label that does not fit on this one.
interface Side { lx: number; anchor: "start" | "middle" | "end"; max: number }
interface Edge extends Side { key: string; d: string; label: string; ly: number; crossing: boolean; into: string;
                              band: [number, number]; stack: "up" | "center"; lines: number; alt?: Side }

export function fill(text: string | undefined, params?: Record<string, unknown>): string {
  if (!text) return "";
  return text.replace(/\{(\w+)\}/g, (m, k) => (params && params[k] !== undefined ? String(params[k]) : m));
}

export function layout(loop: LoopSpec, mode: Mode) {
  const D = DIMS[mode];
  const clientX = D.pad;
  const awsX = D.pad + D.nw + D.api;
  const apiX = D.pad + D.nw + D.api / 2;
  const width = awsX + D.nw + D.pad;
  const top = D.zt + 10;
  const placed: Placed[] = [{ key: "__header", node: null, lane: loop.driver, row: 0, x: 0, y: 0 }];
  for (const n of loop.nodes) {
    const prev = placed[placed.length - 1];
    const row = prev.lane === "client" && n.lane === "aws" ? prev.row : prev.row + 1;
    placed.push({ key: n.op, node: n, lane: n.lane, row, x: 0, y: 0 });
  }
  for (const p of placed) { p.x = p.lane === "client" ? clientX : awsX; p.y = top + p.row * D.rp; }
  const rows = placed[placed.length - 1].row + 1;
  const bottomWire = top + (rows - 1) * D.rp + D.nh + (mode === "compact" ? 9 : 18);
  const height = bottomWire + D.foot;
  const mid = (p: Placed) => p.y + D.nh / 2;
  const cx = (p: Placed) => p.x + D.nw / 2;
  // Ports on a node's API-facing side, so a call leaving and a result arriving never share a line:
  // results arrive high, calls cross low, results leave lower still.
  const arriveY = (p: Placed) => p.y + D.nh * 0.3;
  const callY = (p: Placed) => p.y + D.nh * 0.62;
  const leaveY = (p: Placed) => p.y + D.nh * 0.86;
  const r = mode === "compact" ? 4 : 8;

  const edges: Edge[] = [];
  for (let i = 1; i < placed.length; i++) {
    const a = placed[i - 1], b = placed[i];
    const label = a.node ? a.node.out ?? "" : loop.header_out ?? "";
    if (a.row === b.row) { // a call: straight across the API
      const y = callY(a);
      // A result coming back into `a` runs along the API line and turns in just above this label: one line only.
      const before = placed[i - 2];
      const resultIn = !!before && before.lane !== a.lane && before.row !== a.row;
      edges.push({ key: `${a.key}-${b.key}`, d: `M${a.x + D.nw},${y} H${b.x - 2}`, label, lx: apiX, ly: y - 7,
                   anchor: "middle", crossing: true, into: b.key, max: D.api - 10,
                   band: [resultIn ? arriveY(a) + 1.5 : a.y + 1, y - 2], stack: "up", lines: resultIn ? 1 : 2 });
    } else if (a.lane === b.lane) {
      const x = cx(a);
      // Up to the API line; on the AWS side up to the svg's edge, or to the wire back to the loop when the job owns it.
      const right = a.lane === "client" ? apiX - 8 : loop.driver === "aws" ? width - Math.max(4, D.pad / 2) - 6 : width - 8;
      // Between rows, the API column next to an AWS-side line is empty: room for the label when the right side is too narrow.
      const alt = a.lane === "aws" ? { lx: x - 8, anchor: "end" as const, max: x - 8 - (apiX + 7) } : undefined;
      edges.push({ key: `${a.key}-${b.key}`, d: `M${x},${a.y + D.nh} V${b.y - 2}`, label, lx: x + 8,
                   ly: (a.y + D.nh + b.y) / 2 + 3, anchor: "start", crossing: false, into: b.key, max: right - (x + 8),
                   band: [a.y + D.nh + 1.5, b.y - 1.5], stack: "center", lines: Infinity, alt });
    } else { // the result comes back across the API to the other side
      const y0 = leaveY(a), y1 = arriveY(b);
      const fromX = a.lane === "aws" ? a.x : a.x + D.nw;
      const toX = b.lane === "client" ? b.x + D.nw + 2 : b.x - 2;
      const dir = a.lane === "aws" ? -1 : 1;
      const d = `M${fromX},${y0} H${apiX - dir * r} Q${apiX},${y0} ${apiX},${y0 + r} V${y1 - r} Q${apiX},${y1} ${apiX + dir * r},${y1} H${toX}`;
      // Below this edge's own top segment; above the label of the call `b` makes next, if it makes one in its row.
      const next = placed[i + 1];
      const bottom = next && next.row === b.row ? callY(b) - 7 - ASC * D.edge - 1.5 : y1 + 6;
      edges.push({ key: `${a.key}-${b.key}`, d, label, lx: apiX + (a.lane === "aws" ? 6 : -6), ly: (y0 + y1) / 2,
                   anchor: a.lane === "aws" ? "start" : "end", crossing: true, into: b.key, max: D.api / 2 - 9,
                   band: [y0 + 1.5, bottom], stack: "center", lines: Infinity });
    }
  }
  // The wire back to the loop header: the end of a step hands control back to whoever owns the loop.
  const last = placed[placed.length - 1], head = placed[0];
  const outer = head.lane === "client" ? Math.max(4, D.pad / 2) : width - Math.max(4, D.pad / 2);
  const headSide = head.lane === "client" ? head.x - 2 : head.x + D.nw + 2;
  const sx = cx(last);
  const toward = outer < sx ? -1 : 1;
  edges.push({
    key: "loop-back",
    d: `M${sx},${last.y + D.nh} V${bottomWire - r} Q${sx},${bottomWire} ${sx + toward * r},${bottomWire} H${outer - toward * r} `
      + `Q${outer},${bottomWire} ${outer},${bottomWire - r} V${mid(head) + r} Q${outer},${mid(head)} ${outer - toward * r},${mid(head)} H${headSide}`,
    label: last.node?.out ?? "", lx: (sx + outer) / 2, ly: bottomWire - 5, anchor: "middle",
    crossing: (last.lane !== head.lane), into: "__header", max: Math.abs(sx - outer) - 16,
    band: [last.y + D.nh + 1.5, bottomWire - 2], stack: "up", lines: 1,
  });
  return { D, placed, edges, width, height, apiX, bottomWire, clientX, awsX };
}

// An edge label as drawn: its lines, their font size, the side of the line they are on, and whether the text was cut.
interface FittedLabel { lines: string[]; size: number; side: Side; cut: boolean }

/** The most lines of `size` that fit in the edge's band, and at least one: a label is never dropped. */
function linesAt(e: Edge, size: number): number {
  return Math.max(1, Math.min(e.lines, Math.floor((e.band[1] - e.band[0] - size) / lineH(size)) + 1));
}

/** The largest size at which the label fits whole, on its side of the line or else the other; failing that, the smallest size, cut. */
function fitLabel(fit: Fit, e: Edge, text: string, sizes: number[], base: number): FittedLabel {
  const whole = (lines: string[]) => lines.join("").replace(/\s+/g, "") === text.replace(/\s+/g, "");
  const wrap = (side: Side, size: number) => fit.wrap(text, side.max, "lm-edge-label", size === base ? undefined : size, linesAt(e, size));
  const own: Side = { lx: e.lx, anchor: e.anchor, max: e.max };
  for (const size of sizes) {
    for (const side of e.alt ? [own, e.alt] : [own]) {
      const lines = wrap(side, size);
      if (whole(lines)) return { lines, size, side, cut: false };
    }
  }
  const size = sizes[sizes.length - 1];
  const side = e.alt && e.alt.max > own.max ? e.alt : own;
  return { lines: wrap(side, size), size, side, cut: true };
}

/** Baselines of a fitted label: stacked up from ly or centred on it, then moved only as far as it takes to stay in the band. */
function baselines(e: Edge, f: FittedLabel): number[] {
  const n = f.lines.length, lh = lineH(f.size);
  const first = e.stack === "up" ? e.ly - (n - 1) * lh : e.ly - ((n - 1) / 2) * lh;
  const top = Math.max(e.band[0], Math.min(first - ASC * f.size, e.band[1] - ((n - 1) * lh + f.size)));
  return f.lines.map((_, i) => top + ASC * f.size + i * lh);
}

export function nodeStat(t: TFunction, stat: NodeStat | undefined, lane: Lane, price: Tokens | undefined,
                         where?: LoopNode["where"],
): string {
  if (!stat || !stat.count) return t("loop.stat.notRun");
  const each = seconds(stat.sec / stat.count);
  const parts = [t("loop.stat.count", { n: stat.count }), t("loop.stat.each", { t: each })];
  if (where === "agentcore") {
    // Your code, but not in your process: AgentCore bills its sessions (docs/ui-design.md §4). The request
    // count is on the run console's sandbox card; here it would push the billing note out of the box.
    let note = t("loop.stat.sandboxBilled");
    parts.push(note);
  } else if (lane === "aws") {
    const tok = stat.tokens.prefill + stat.tokens.sample + stat.tokens.train;
    if (stat.calls > stat.count) parts.splice(1, 0, t("loop.stat.calls", { n: int(Math.round(stat.calls / stat.count)) }));
    let priced = !!tok;
    if (priced) parts.push(usd(costOf(stat.tokens, price)));
    else if (!tok) parts.push(t("common.notBilled"));
  } else parts.push(t("common.notBilled"));
  if (stat.failed) parts.push(t("loop.stat.failed", { n: stat.failed }));
  return parts.join(" · ");
}

// Where each line of a node starts, and the room to its right: the badge and the slot dot take theirs first.
const TEXT_X = { full: 13, compose: 13, compact: 8 } as const;
const BADGE_PAD = 7;   // the badge's text inset, each side
const BADGE_GAP = 6;   // between a node's title and its badge

export default function LoopMap(props: Props) {
  const { t } = useTranslation();
  const { loop, mode = "full", params, stats, active, selected, onSelect, slotLabels, price } = props;
  const { D, placed, edges, width, height, apiX } = layout(loop, mode);
  const zones = props.zoneTitles ?? {
    client: [t("loop.zone.process"), t("loop.zone.processSub")],
    aws: [t("loop.zone.aws"), t("loop.zone.awsSub")],
  };
  const rightClass = props.rightIsYours ? "you" : "aws";
  const canClick = (n: LoopNode | "header") => !!onSelect && (props.clickable ? props.clickable(n) : true);
  const key = (e: KeyboardEvent, op: string) => {
    if ((e.key === "Enter" || e.key === " ") && onSelect) { e.preventDefault(); onSelect(op); }
  };
  const activeNode = placed.find((p) => p.key === active);
  const tx = TEXT_X[mode];

  // Everything that is fitted, as plain data: the measuring below reads only this.
  const texts = placed.map((p) => {
    const isHeader = p.key === "__header";
    const n = p.node;
    const unobserved = n?.observed === false;
    const isActive = active === p.key;
    return {
      key: p.key,
      title: isHeader ? (mode === "compact" ? (loop.driver === "client" ? t("loop.map.yourLoop") : t("loop.map.jobLoop")) : loop.header) : n!.title,
      titleCls: `lm-title${isHeader && mode !== "compact" ? " mono" : ""}`,
      titleSize: isHeader && mode !== "compact" ? D.title - 1.5 : D.title,
      slot: n?.slot ? slotLabels?.[n.slot] ?? null : null,
      // under a node its call; under the loop header (full map only) who drives the loop
      call: isHeader ? (mode === "full" ? (loop.driver === "client" ? t("loop.map.yourLoopDrives") : t("loop.map.jobDrives")) : null)
        : D.call > 0 ? fill(n!.call, params) : null,
      stat: D.stat > 0 ? (isActive && !isHeader ? t("loop.map.runningNow")
        : isHeader ? props.headerDetail ?? "" : unobserved ? t("loop.map.unobserved")
        : nodeStat(t, stats?.[p.key], p.lane, price, n?.where,
        )) : null,
      statCls: `lm-stat${isActive ? " live" : ""}`,
    };
  });
  const labels = mode === "compact" ? [] : edges.map((e) => ({ e, text: fill(e.label, params) }));
  const sizes: number[] = [];
  for (let sz = D.edge; sz >= D.edgeMin; sz -= 0.5) sizes.push(sz);
  const subs = mode === "compact" ? null : { client: zones.client[1], aws: zones.aws[1], clientMax: apiX - 26, awsMax: width - apiX - 28 };

  const [probe, fitted] = useFittedText({ mode, texts, labels, sizes, subs }, (fit) => ({
    nodes: Object.fromEntries(texts.map((x) => {
      let badgeW = 0, badge = "";
      if (x.slot && mode !== "compact") {
        badgeW = Math.min(D.nw * 0.5, fit.width(`⇄ ${x.slot}`, "lm-badge-text") + 2 * BADGE_PAD);
        badge = fit.clip(`⇄ ${x.slot}`, badgeW - 2 * BADGE_PAD, "lm-badge-text");
      }
      const right = x.slot ? (mode === "compact" ? D.nw - 15 : D.nw - BADGE_PAD - badgeW - BADGE_GAP) : D.nw - 8;
      return [x.key, {
        title: fit.clip(x.title, right - tx, x.titleCls, x.titleSize), badge, badgeW,
        call: x.call === null ? null : fit.clip(x.call, D.nw - tx - 9, "lm-call", D.call),
        stat: x.stat === null ? null : fit.clip(x.stat, D.nw - tx - 9, x.statCls),
      }];
    })),
    edges: Object.fromEntries(labels.filter((l) => l.text).map((l) => [l.e.key, fitLabel(fit, l.e, l.text, sizes, D.edge)])),
    subs: subs && { client: fit.clip(subs.client, subs.clientMax, "lm-zone-sub"), aws: fit.clip(subs.aws, subs.awsMax, "lm-zone-sub") },
  }));

  return (
    <svg className={`loopmap ${mode}`} viewBox={`0 0 ${width} ${height}`} width="100%" role="group"
         aria-label={props.ariaLabel ?? t("loop.map.aria")}
         style={{ maxWidth: mode === "compact" ? width * 1.25 : undefined }}>
      <defs>
        <marker id={`lm-arrow-${mode}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth={mode === "compact" ? 5 : 7}
                markerHeight={mode === "compact" ? 5 : 7} orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" className="lm-arrowhead" />
        </marker>
      </defs>
      <text ref={probe} visibility="hidden" aria-hidden />
      <rect x={0} y={0} width={apiX} height={height} className="lm-zone you" />
      <rect x={apiX} y={0} width={width - apiX} height={height} className={`lm-zone ${rightClass}`} />
      <line x1={apiX} x2={apiX} y1={mode === "compact" ? 4 : 10} y2={height - 4} className="lm-api" />
      {mode !== "compact" ? (
        <>
          <text x={14} y={20} className="lm-zone-title you">{zones.client[0]}</text>
          <text x={14} y={35} className="lm-zone-sub">{fitted?.subs?.client ?? zones.client[1]}</text>
          <text x={width - 14} y={20} textAnchor="end" className={`lm-zone-title ${rightClass}`}>{zones.aws[0]}</text>
          <text x={width - 14} y={35} textAnchor="end" className="lm-zone-sub">{fitted?.subs?.aws ?? zones.aws[1]}</text>
          <g transform={`translate(${apiX},${18})`}>
            <rect x={-18} y={-11} width={36} height={18} rx={9} className="lm-api-pill" />
            <text x={0} y={2} textAnchor="middle" className="lm-api-text">API</text>
          </g>
        </>
      ) : (
        <>
          <text x={6} y={14} className="lm-zone-title you small">{zones.client[0]}</text>
          <text x={width - 6} y={14} textAnchor="end" className={`lm-zone-title ${rightClass} small`}>{zones.aws[0]}</text>
        </>
      )}

      {edges.map((e) => {
        const flowing = e.crossing && active === e.into && activeNode?.lane === "aws";
        const f = fitted?.edges[e.key];
        const ys = f ? baselines(e, f) : [];
        return (
          <g key={e.key} className={`lm-edge${e.crossing ? " crossing" : ""}${flowing ? " flowing" : ""}`}>
            {f?.cut && <title>{fill(e.label, params)}</title>}
            <path d={e.d} markerEnd={`url(#lm-arrow-${mode})`} />
            {f?.lines.map((line, i) => (
              <text key={i} x={f.side.lx} y={ys[i]} textAnchor={f.side.anchor} className="lm-edge-label"
                    style={f.size === D.edge ? undefined : { fontSize: f.size }}>{line}</text>
            ))}
          </g>
        );
      })}

      {placed.map((p, idx) => {
        const isHeader = p.key === "__header";
        const n = p.node;
        const text = texts[idx];
        const fit = fitted?.nodes[p.key];
        const clickableNode = canClick(isHeader ? "header" : n!);
        const isActive = active === p.key;
        const isSel = selected === p.key;
        const unobserved = n?.observed === false;
        const laneClass = p.lane === "aws" && props.rightIsYours ? "you" : p.lane === "client" ? "you" : "aws";
        const who = p.lane === "aws" ? "SageMaker AI" : n?.where === "agentcore" ? t("loop.map.yourCodeSandbox") : t("loop.map.yourCode");
        const nodeTip = isHeader ? "" : t("loop.map.tip", { title: n!.title, who, call: fill(n!.call, params) });
        const tip = isHeader ? `${text.title}${props.headerDetail ? `: ${props.headerDetail}` : ""}`
          : text.slot ? t("loop.map.tipSlot", { tip: nodeTip, slot: text.slot }) : nodeTip;
        return (
          <g key={p.key} transform={`translate(${p.x},${p.y})`}
             className={`lm-node ${laneClass}${isHeader ? " header" : ""}${isActive ? " on" : ""}${isSel ? " sel" : ""}`
               + `${unobserved ? " unobserved" : ""}${unobserved && props.dimInside ? " inside" : ""}${clickableNode ? " click" : ""}`
               + `${n?.slot ? " slot" : ""}${n?.where === "agentcore" ? " agentcore" : ""}`}
             tabIndex={clickableNode ? 0 : undefined} role={clickableNode ? "button" : undefined} aria-pressed={clickableNode ? isSel : undefined}
             aria-label={tip} onClick={clickableNode ? () => onSelect!(p.key) : undefined}
             onKeyDown={clickableNode ? (e) => key(e, p.key) : undefined}>
            <title>{tip}</title>
            {isActive && <rect className="lm-halo" x={-4} y={-4} width={D.nw + 8} height={D.nh + 8} rx={10} />}
            <rect className="lm-box" width={D.nw} height={D.nh} rx={mode === "compact" ? 5 : 8} />
            <rect className="lm-accent" width={mode === "compact" ? 3 : 4} height={D.nh} rx={2} />
            <text x={tx} y={mode === "compact" ? D.nh / 2 + 3.5 : 19} className={text.titleCls} style={{ fontSize: text.titleSize }}>
              {fit?.title ?? text.title}
            </text>
            {fit && fit.badgeW > 0 && (
              <g transform={`translate(${D.nw - fit.badgeW - BADGE_PAD},${7})`} className="lm-badge">
                <rect width={fit.badgeW} height={17} rx={8.5} />
                <text x={BADGE_PAD} y={12} className="lm-badge-text">{fit.badge}</text>
              </g>
            )}
            {text.slot && mode === "compact" && <circle cx={D.nw - 8} cy={D.nh / 2} r={3} className="lm-slot-dot" />}
            {text.call !== null && (
              <text x={tx} y={mode === "full" && !isHeader ? 35 : 36} className="lm-call" style={{ fontSize: D.call }}>{fit?.call ?? text.call}</text>
            )}
            {text.stat !== null && <text x={tx} y={51} className={text.statCls}>{fit?.stat ?? text.stat}</text>}
          </g>
        );
      })}

      {mode === "full" && (
        <>
          {props.clientFooter && <text x={14} y={height - 14} className="lm-foot">{props.clientFooter}</text>}
          {props.awsFooter && <text x={width - 14} y={height - 14} textAnchor="end" className="lm-foot">{props.awsFooter}</text>}
        </>
      )}
    </svg>
  );
}

/** In-loop work per node op. Before/after-loop work and work done to answer questions are not counted. */
export function loopStats(spans: Span[], loop: LoopSpec): Record<string, NodeStat> {
  const ops = new Set(loop.nodes.map((n) => n.op));
  const out: Record<string, NodeStat> = {};
  for (const s of spans) {
    if (!ops.has(s.op) || s.role === "probe" || (s.step === null && loop.driver === "client")) continue;
    const st = out[s.op] ?? { count: 0, sec: 0, calls: 0, tokens: { prefill: 0, sample: 0, train: 0 }, last: null, failed: 0,
                              sandboxCalls: 0 };
    st.count += 1;
    st.sec += s.sec;
    st.calls += s.calls ?? (s.lane === "aws" ? 1 : 0);
    st.sandboxCalls += s.sandbox_calls ?? 0;
    for (const k of ["prefill", "sample", "train"] as const) st.tokens[k] += s.tokens?.[k] ?? 0;
    st.last = s;
    if (!s.ok) st.failed += 1;
    out[s.op] = st;
  }
  return out;
}
