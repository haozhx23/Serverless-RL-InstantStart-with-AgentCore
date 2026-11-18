import { Button, Select, Tag, Tooltip } from "antd";
import { AimOutlined } from "@ant-design/icons";
import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useRunFile } from "../api/hooks";
import type { LoopSpec, RunSummary, Span, Tokens } from "../api/types";
import { costOf, int, opLabel, seconds, usd } from "../lib/format";
import { callsOf, type RunView } from "../lib/runStream";
import LoopCode, { findBlock, symbolBlock } from "./LoopCode";
import LoopMap, { fill, loopStats, nodeStat } from "./LoopMap";
import { primitives } from "../lib/primitives";

// The run console's hero (docs/ui-design.md §4): the loop map, what ran before and after the
// loop, the code behind whichever block is selected (or running, when following), and the
// session API's primitives this run composed.

// Ops that run in a helper rather than under an ev.span of their own in loop.py.
const CALL_SITE: Record<string, RegExp> = { create_session: /start_session\(/, stop_session: /finish\(ev/ };

type Sel = { kind: "node"; op: string } | { kind: "phase"; phase: "before" | "after"; index: number };

interface PhaseChip { op: string; span: Span | null; open: boolean }

function phases(view: RunView, loop: LoopSpec): { before: PhaseChip[]; after: PhaseChip[] } {
  const first = view.stepStarts[0]?.t ?? Infinity;
  const last = view.stepStarts[view.stepStarts.length - 1]?.t ?? Infinity;
  const outside = view.spans.filter((s) => s.role !== "probe" && (loop.driver === "aws" || s.step === null));
  const openOutside = view.open.filter((o) => o.role !== "probe" && (loop.driver === "aws" || o.step === null));
  const assign = (ops: string[], inPhase: (t: number) => boolean) => {
    const used = new Set<number>();
    return ops.map((op) => {
      const span = outside.find((s) => s.op === op && inPhase(s.t0) && !used.has(s.seq)) ?? null;
      if (span) used.add(span.seq);
      const open = !span && openOutside.some((o) => o.op === op && inPhase(o.t0));
      return { op, span, open };
    });
  };
  return { before: assign(loop.before, (t) => t < first), after: assign(loop.after, (t) => t > last) };
}

export default function LoopPanel({ run, view, price, alive }: {
  run: RunSummary; view: RunView; price?: Tokens; alive: boolean;
}) {
  const { t } = useTranslation();
  const loop = run.loop!;
  const params = (view.started?.params ?? {}) as Record<string, unknown>;
  const stats = useMemo(() => loopStats(view.spans, loop), [view.spans, loop]);
  const mtrl = loop.driver === "aws";
  const slotByKey = Object.fromEntries(run.slots.map((s) => [s.key, s]));
  const slotLabels: Record<string, string> = {};
  for (const sl of run.slots) {
    const v = run.variants[sl.key];
    if (v === undefined) continue;
    const variant = sl.variants.find((x) => x.id === v);
    slotLabels[sl.key] = v === "custom" ? t("loop.panel.edited") : variant?.short ?? variant?.label ?? String(v);
  }

  // What is running now: the latest operation that started and has not ended.
  const now = [...view.open].filter((o) => o.role !== "probe").sort((a, b) => b.t0 - a.t0)[0];
  const nodeOps = new Set(loop.nodes.map((n) => n.op));
  const activeOp = now && nodeOps.has(now.op) && (mtrl || now.step !== null) ? now.op : null;
  const lastActive = useRef<string | null>(null);
  if (activeOp) lastActive.current = activeOp;

  const [follow, setFollow] = useState(true);
  const firstSlotNode = (mtrl ? loop.nodes.find((n) => n.lane === "client") : loop.nodes.find((n) => n.slot))?.op ?? loop.nodes[0]?.op;
  const [sel, setSel] = useState<Sel>({ kind: "node", op: firstSlotNode });
  const shown: Sel = follow && alive && lastActive.current ? { kind: "node", op: lastActive.current } : sel;
  const pick = (s: Sel) => { setSel(s); setFollow(false); };

  const ph = useMemo(() => phases(view, loop), [view, loop]);
  const node = shown.kind === "node" ? loop.nodes.find((n) => n.op === shown.op) : undefined;
  const chip = shown.kind === "phase" ? ph[shown.phase][shown.index] : undefined;
  const slot = node?.slot ? slotByKey[node.slot] : undefined;

  // Which file and lines to show.
  const files = [...run.files, "config.json"];
  // Runs created before a slot existed do not have its file; their code is all in loop.py.
  const preferred = (slot?.file && run.files.includes(slot.file) ? slot.file : null) ?? node?.file ?? "loop.py";
  const [file, setFile] = useState<string>(preferred);
  useEffect(() => { setFile(preferred); }, [preferred, shown.kind === "node" ? shown.op : `${shown.phase}${shown.index}`]); // eslint-disable-line react-hooks/exhaustive-deps
  const code = useRunFile(run.id, file);
  const content = code.data?.content ?? "";
  const range = useMemo((): [number, number] | null => {
    if (!content) return null;
    if (slot?.file && slot.file === file && slot.function) return symbolBlock(content, slot.function);
    if (node?.file && node.file === file && node.function) return symbolBlock(content, node.function);
    if (file !== "loop.py") return null;
    if (shown.kind === "node") return findBlock(content, shown.op, mtrl ? "any" : "loop");  // also a slot node of an older run
    const op = chip!.op;
    if (CALL_SITE[op]) {
      const i = content.split("\n").findIndex((l) => CALL_SITE[op].test(l));
      return i >= 0 ? [i, i] : null;
    }
    return findBlock(content, op, mtrl ? "any" : "outside");
  }, [content, file, shown, slot, chip, mtrl, node]);

  const lastSpan = node ? stats[node.op]?.last : chip?.span;
  const lane = node?.lane ?? chip?.span?.lane ?? "client";
  const ledger = primitives().map((p) => ({ ...p, calls: view.spans.filter((s) => p.ops.includes(s.op) && s.lane === "aws").reduce((a, s) => a + callsOf(s), 0) }));
  // Time in this process only: code sent to the environment sandbox ran (and billed) in AgentCore Runtime.
  const clientSec = view.spans.filter((s) => s.lane === "client" && !s.where).reduce((a, s) => a + s.sec, 0);
  const behind = view.versions.sampler === null ? null : view.versions.trainer - view.versions.sampler;
  let nodeTotal = node?.lane === "aws" && stats[node.op] ? costOf(stats[node.op].tokens, price) : null;
  let mapBox = "loop-map-box";
  const stepNow = view.step;
  const total = view.totalSteps ?? run.total_steps;

  return (
    <div className="loop-panel">
      <div className="loop-left">
        <PhaseRow label={t("loop.panel.before")} chips={ph.before} phase="before" sel={shown} onPick={pick} price={price} over={!!view.finished} />
        <div className={mapBox}>
          <LoopMap loop={loop} params={params} stats={stats} active={activeOp} price={price}
                   selected={shown.kind === "node" ? shown.op : null} onSelect={(op) => pick({ kind: "node", op })}
                   clickable={(n) => n !== "header" || !mtrl}
                   slotLabels={slotLabels} dimInside={mtrl && view.job?.secondary === "Training"}
                   headerDetail={stepNow === null ? (total ? t("loop.panel.steps", { total }) : "")
                     : t(view.finished ? "loop.panel.stepOfDone" : "loop.panel.stepOf", { step: stepNow, total })}
                   zoneTitles={mtrl ? { client: [t("loop.zone.code"), t("loop.zone.codeSub")],
                                        aws: [t("loop.zone.aws"), view.job ? t("loop.zone.jobSub", { name: view.job.name }) : t("loop.zone.jobSubNoName")] }
                     : { client: [t("loop.zone.process"), t("loop.zone.processSub")],
                         aws: [t("loop.zone.aws"), t("loop.zone.sessionSub", { name: view.session.name ?? run.session_name ?? "" }).trim()] }}
                   clientFooter={mtrl ? t("loop.panel.mtrlFooter") : t("loop.panel.tsFooter", { calls: int(view.apiCalls), sec: seconds(clientSec) })}
                   awsFooter={mtrl ? (view.job ? t("loop.panel.jobFooter", { status: view.job.secondary ?? view.job.status }) : t("loop.panel.noJob"))
                     : t(behind ? "loop.panel.trainerFooterBehind" : "loop.panel.trainerFooter", {
                         updates: view.versions.trainer, behind,
                         sampler: view.versions.sampler === null ? t("loop.panel.samplerBase") : t("loop.panel.samplerAfter", { n: view.versions.sampler }) })} />
        </div>
        <PhaseRow label={t("loop.panel.after")} chips={ph.after} phase="after" sel={shown} onPick={pick} price={price} over={!!view.finished} />
        {!mtrl && (
          <div className="ledger" aria-label={t("loop.panel.ledgerAria")}>
            <span className="ledger-head">{t("loop.panel.ledger")}</span>
            {ledger.map((p) => (
              <Tooltip key={p.name} title={p.what}>
                <span className={`prim${p.calls ? " used" : ""}`}><span className="mono">{p.name}</span>{p.calls ? <b className="num"> {int(p.calls)}</b> : null}</span>
              </Tooltip>
            ))}
          </div>
        )}
      </div>

      <div className="loop-right">
        <div className="loop-code-head">
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            <span className={`lane-dot ${lane === "client" ? "you" : "aws"}`} />
            <strong>{node ? node.title : chip ? opLabel(chip.op) : ""}</strong>
            {slot && <Tag className="slot-tag">⇄ {run.variants[slot.key] === "custom" ? t("loop.panel.editedInBrowser")
              : slot.variants.find((x) => x.id === run.variants[slot.key])?.label ?? slot.label}</Tag>}
            <span style={{ flex: 1 }} />
            {alive && (
              <Button size="small" type={follow ? "primary" : "default"} icon={<AimOutlined />} onClick={() => setFollow((v) => !v)}>
                {follow ? t("loop.panel.following") : t("loop.panel.follow")}
              </Button>
            )}
          </div>
          <div className="secondary" style={{ fontSize: 12.5, marginTop: 4 }}>
            {node ? <>{node.lane === "aws" ? "SageMaker AI" : node.where === "agentcore" ? t("loop.panel.inSandbox") : t("common.lane.youLong")}
              {node.observed === false ? t("loop.panel.insideJob") : ""} ·{" "}
              <span className="mono">{fill(node.call, params)}</span></> : chip ? (chip.span ? `${seconds(chip.span.sec)}${chip.span.note ? ` · ${chip.span.note}` : ""}`
                : chip.open ? t("loop.panel.running") : t("loop.panel.hasNotRun")) : null}
          </div>
          {node && node.observed !== false && (
            <div className="muted" style={{ fontSize: 12, marginTop: 2 }}>
              {nodeStat(t, stats[node.op], node.lane, price, node.where,
              )}
              {lastSpan?.note ? t("loop.panel.last", { note: lastSpan.note }) : ""}
              {nodeTotal ? t("loop.panel.total", { usd: usd(nodeTotal) }) : ""}
            </div>
          )}
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center", margin: "8px 0" }}>
          <Select size="small" value={file} onChange={setFile} style={{ minWidth: 180 }} options={files.map((f) => ({
            value: f, label: <span className="mono">{f}{run.custom_files.includes(f) ? t("loop.panel.fileEdited") : ""}</span> }))} />
          <span className="muted" style={{ fontSize: 12 }}>
            {range ? t("loop.panel.lines", { from: range[0] + 1, to: range[1] + 1 }) : node?.observed === false ? t("loop.panel.noCode") : t("loop.panel.executed")}
          </span>
        </div>
        <LoopCode content={content || (code.error ? String(code.error) : "…")} highlight={range} />
      </div>
    </div>
  );
}

function PhaseRow({ label, chips, phase, sel, onPick, price, over }: {
  label: string; chips: PhaseChip[]; phase: "before" | "after"; sel: Sel; onPick: (s: Sel) => void; price?: Tokens;
  over: boolean;
}) {
  const { t } = useTranslation();
  if (!chips.length) return null;
  return (
    <div className="phase-row">
      <span className="phase-label">{label}</span>
      {chips.map((c, i) => {
        const on = sel.kind === "phase" && sel.phase === phase && sel.index === i;
        const lane = c.span?.lane;
        let cost = c.span && c.span.lane === "aws" && c.span.tokens ? costOf(c.span.tokens, price) : null;
        const tip = c.span ? `${opLabel(c.op)}: ${seconds(c.span.sec)}${c.span.note ? `, ${c.span.note}` : ""}`
          + `${c.span.tokens ? t("loop.panel.chipTokens", { n: int((c.span.tokens.prefill ?? 0) + (c.span.tokens.sample ?? 0) + (c.span.tokens.train ?? 0)) }) : ""}`
          + `${cost ? `, ≈ ${usd(cost)}` : ""}` : c.open ? t("loop.panel.chipRunning", { op: opLabel(c.op) })
          : over ? t("loop.panel.chipUnused", { op: opLabel(c.op) }) : t("loop.panel.chipNotYet", { op: opLabel(c.op) });
        return (
          <button key={`${c.op}-${i}`} type="button" title={tip} aria-pressed={on}
                  className={`phase-chip${lane ? ` ${lane === "client" ? "you" : "aws"}` : ""}${c.span ? " done" : ""}${c.open ? " open" : ""}${c.span && !c.span.ok ? " failed" : ""}${!c.span && !c.open && over ? " skipped" : ""}${on ? " sel" : ""}`}
                  onClick={() => onPick({ kind: "phase", phase, index: i })}>
            {opLabel(c.op)}{c.span ? <span className="num"> {seconds(c.span.sec)}</span> : null}
          </button>
        );
      })}
    </div>
  );
}
