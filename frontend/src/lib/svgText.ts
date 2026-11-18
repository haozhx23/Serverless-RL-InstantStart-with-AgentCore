import { useLayoutEffect, useRef, useState, type RefObject } from "react";

// SVG text neither wraps nor truncates, so the loop map and the timeline fit their labels by
// measuring how this browser lays them out: a hidden <text> inside the same <svg> takes each
// candidate string with the class and font size of the real label, and getComputedTextLength()
// gives its width in the svg's own units. Measuring in place matters. Glyph advances change with
// the size the svg is drawn at: the same mono label measured 0.63 em per character in the run
// console and 0.67 em in the launch drawer, where the map is drawn 1.34x larger, and canvas
// measureText was up to 6% off both (Chrome, 2026-10-29). Fixed per-character widths cannot hold.

/** Width of `text` drawn with class `cls` (and an inline font size, if the label sets one). */
export type WidthOf = (text: string, cls: string, size?: number) => number;

export interface Fit {
  width: WidthOf;
  /** `text`, cut with an ellipsis to fit `max`. */
  clip(text: string, max: number, cls: string, size?: number): string;
  /** Up to `lines` lines of at most `max`; breaks at spaces and between CJK characters, never before closing punctuation. Text that does not fit ends in an ellipsis. */
  wrap(text: string, max: number, cls: string, size?: number, lines?: number): string[];
}

// Hangul Jamo, CJK punctuation through Yi, Hangul syllables, CJK compatibility, full-width forms:
// one break opportunity per character. Everything else breaks at whitespace.
const WIDE_RANGES = "\\u1100-\\u115F\\u2E80-\\uA4CF\\uAC00-\\uD7A3\\uF900-\\uFAFF\\uFE30-\\uFE4F\\uFF00-\\uFF60\\uFFE0-\\uFFE6";
// Closing punctuation never starts a line: it is one token with what comes before it.
const CLOSING = "，。、；：！？）》」』】〉〕”’…";
const TOKENS = new RegExp(`[^\\s${WIDE_RANGES}]+[${CLOSING}]*|[${WIDE_RANGES}][${CLOSING}]*|\\s+`, "g");
const ELLIPSIS = "…";

/** Clipping and wrapping on top of any width function (the DOM's in the app, a fake one in tests). */
export function makeFit(widthOf: WidthOf): Fit {
  const cache = new Map<string, number>();
  const width: WidthOf = (text, cls, size) => {
    const key = `${cls}|${size ?? ""}|${text}`;
    let w = cache.get(key);
    if (w === undefined) { w = widthOf(text, cls, size); cache.set(key, w); }
    return w;
  };
  const clip: Fit["clip"] = (text, max, cls, size) => {
    if (width(text, cls, size) <= max) return text;
    const chars = Array.from(text);
    let lo = 0, hi = chars.length - 1;  // the longest prefix that fits with the ellipsis
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2);
      if (width(chars.slice(0, mid).join("").trimEnd() + ELLIPSIS, cls, size) <= max) lo = mid; else hi = mid - 1;
    }
    return chars.slice(0, lo).join("").trimEnd() + ELLIPSIS;
  };
  const wrap: Fit["wrap"] = (text, max, cls, size, lines = 3) => {
    const out: string[] = [];
    let line = "", lineStart = 0;
    for (const m of text.matchAll(TOKENS)) {
      const tok = m[0];
      if (/^\s+$/.test(tok)) { if (line) line += " "; continue; }
      if (line.trim() && width(line + tok, cls, size) > max) {
        if (out.length === lines - 1) {  // no line left: the rest goes on this one, cut to fit
          out.push(clip(text.slice(lineStart).replace(/\s+/g, " ").trim(), max, cls, size));
          return out;
        }
        out.push(line.trimEnd());
        line = tok;
        lineStart = m.index!;
      } else {
        if (!line) lineStart = m.index!;
        line += tok;
      }
    }
    if (line.trim()) out.push(line.trimEnd());
    return out.map((l) => clip(l, max, cls, size));  // a single word longer than the line
  };
  return { width, clip, wrap };
}

/** Widths measured with `probe`, a <text> with no React children inside the svg being drawn; null when it is not rendered. */
function probeFit(probe: SVGTextElement): Fit | null {
  const widthOf: WidthOf = (text, cls, size) => {
    probe.setAttribute("class", cls);
    probe.style.fontSize = size ? `${size}px` : "";
    probe.textContent = text;
    return probe.getComputedTextLength();
  };
  if (!(widthOf("M", "") > 0)) return null;  // display: none somewhere above; measured again once it has a size
  return makeFit(widthOf);
}

/** Bumps when web fonts finish loading, so labels first measured with a fallback font are measured again. */
function useFontsVersion(): number {
  const [n, setN] = useState(0);
  useLayoutEffect(() => {
    const fonts = document.fonts;
    if (!fonts) return;
    const bump = () => setN((v) => v + 1);
    void fonts.ready.then(bump);
    fonts.addEventListener("loadingdone", bump);
    return () => fonts.removeEventListener("loadingdone", bump);
  }, []);
  return n;
}

/** The svg's drawn width in CSS pixels, rounded: glyph advances depend on it, and it is 0 while hidden. */
function useDrawnWidth(probe: RefObject<SVGTextElement | null>): number {
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const svg = probe.current?.ownerSVGElement;
    if (!svg) return;
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width)));
    ro.observe(svg);
    return () => ro.disconnect();
  }, [probe]);
  return w;
}

/**
 * Fit an svg's labels by measuring them. Render `<text ref={probe} visibility="hidden" aria-hidden />`
 * (no children, no className: the measuring sets them) inside the svg. `inputs` must describe everything `compute` reads (it is compared as JSON); the result is
 * recomputed when it changes, when fonts load and when the svg's drawn size changes, before paint.
 * Returns null until the first measurement (that render is never painted) and while the svg is hidden.
 */
export function useFittedText<T>(inputs: unknown, compute: (fit: Fit) => T): [RefObject<SVGTextElement | null>, T | null] {
  const probe = useRef<SVGTextElement | null>(null);
  const fonts = useFontsVersion();
  const drawn = useDrawnWidth(probe);
  const key = `${JSON.stringify(inputs)}|${fonts}|${drawn}`;
  const [state, setState] = useState<{ key: string; value: T | null }>({ key: "", value: null });
  useLayoutEffect(() => {
    const el = probe.current;
    const fit = el && probeFit(el);
    setState({ key, value: fit ? compute(fit) : null });
    if (el) {  // leave nothing behind for selectors, styles or tests to find
      el.removeAttribute("class");
      el.style.fontSize = "";
      el.textContent = "";
    }
  }, [key]);  // compute reads only what `inputs` describes, so the key covers it
  return [probe, state.key === key ? state.value : null];
}
