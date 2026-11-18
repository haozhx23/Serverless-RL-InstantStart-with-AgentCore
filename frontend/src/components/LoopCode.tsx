import { useEffect, useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";

// Code with line numbers and one highlighted range: the block of loop.py a node of the loop
// map stands for. Blocks are found from the code itself: every operation is wrapped in
// `with ev.span("<lane>", "<op>")`, so the span call and the lines indented under it are the
// block. The same op can occur inside the for-loop and outside it (sampling the base model
// before training); `where` picks which.

const indent = (line: string) => line.length - line.trimStart().length;

function blockEnd(lines: string[], start: number): number {
  const base = indent(lines[start]);
  let end = start;
  for (let i = start + 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    if (indent(lines[i]) <= base) break;
    end = i;
  }
  return end;
}

export function loopRange(src: string): [number, number] | null {
  const lines = src.split("\n");
  const start = lines.findIndex((l) => /^\s*for step in range\(/.test(l));
  return start < 0 ? null : [start, blockEnd(lines, start)];
}

/** [first, last] line index (0-based) of the block for `op`, or null when the code has none. */
export function findBlock(src: string, op: string, where: "loop" | "outside" | "any" = "any", nth = 0): [number, number] | null {
  const lines = src.split("\n");
  if (op === "__header") {
    const r = loopRange(src);
    if (!r) return null;
    const stepLine = lines.findIndex((l, i) => i > r[0] && /ev\.step\(/.test(l));
    return [r[0], stepLine > r[0] ? stepLine : r[0]];
  }
  const re = new RegExp(`ev\\.(span|record_span)\\(\\s*"(client|aws)"\\s*,\\s*f?"${op.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}"`);
  const loop = loopRange(src);
  const inLoop = (i: number) => !!loop && i > loop[0] && i <= loop[1];
  const hits = lines.map((l, i) => (re.test(l) ? i : -1)).filter((i) => i >= 0)
    .filter((i) => where === "any" || (where === "loop" ? inLoop(i) : !inLoop(i)));
  const at = hits[Math.min(nth, hits.length - 1)];
  return at === undefined ? null : [at, blockEnd(lines, at)];
}

/** The top-level function or assignment `name` (a slot's contract), through its last indented line. */
export function symbolBlock(src: string, name: string): [number, number] | null {
  const lines = src.split("\n");
  const at = lines.findIndex((l) => new RegExp(`^(def ${name}\\b|${name}\\s*[:=])`).test(l));
  if (at < 0) return null;
  let end = at;
  for (let i = at + 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    if (indent(lines[i]) === 0 && !/^[\])}]/.test(lines[i])) break;
    end = i;
  }
  return [at, end];
}

export default function LoopCode({ content, highlight, maxHeight = 460 }: {
  content: string; highlight: [number, number] | null; maxHeight?: number;
}) {
  const { t } = useTranslation();
  const box = useRef<HTMLDivElement>(null);
  const lines = useMemo(() => content.split("\n"), [content]);
  useEffect(() => {
    if (!highlight || !box.current) return;
    const el = box.current.querySelector<HTMLElement>(`[data-line="${highlight[0]}"]`);
    if (el) box.current.scrollTo({ top: Math.max(0, el.offsetTop - 60), behavior: "smooth" });
  }, [highlight, content]);
  const width = String(lines.length).length;
  return (
    <div className="loop-code" ref={box} style={{ maxHeight }} role="region" aria-label={t("loop.panel.codeAria")}>
      {lines.map((line, i) => {
        const on = !!highlight && i >= highlight[0] && i <= highlight[1];
        const comment = /^\s*#/.test(line);
        return (
          <div key={i} data-line={i} className={`ln${on ? " hl" : ""}${on && i === highlight![0] ? " first" : ""}`}>
            <span className="no">{String(i + 1).padStart(width, " ")}</span>
            <span className={`tx${comment ? " cm" : ""}`}>{line || " "}</span>
          </div>
        );
      })}
    </div>
  );
}
