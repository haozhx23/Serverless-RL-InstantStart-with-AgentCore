// Design tokens. docs/ui-design.md §3 is the spec; change it there first.
// `you` and `aws` mean exactly one thing each, everywhere: whose code is doing the work.
// Both modes are selected, not flipped. Validated with the dataviz palette script:
//   light on #FFFFFF  - CVD ΔE 32 (protan), contrast ≥ 3:1
//   dark  on #161B22  - CVD ΔE 28.5 (protan), contrast ≥ 3:1; text and status colors ≥ 4.5:1
// ThemeProvider writes every token as a CSS variable on <html> (textMuted → --text-muted,
// ink2 → --ink-2); global.css only reads them, so this file is the one source of color.

export type ThemeMode = "light" | "dark";

const light = {
  ink: "#232F3E",          // primary actions, selection, chart lines: the strongest neutral
  ink2: "#37475A",
  onInk: "#FFFFFF",        // text on an ink fill
  text: "#16191F",
  text2: "#545B64",
  text3: "#7D8998",
  paper: "#F3F5F7",        // app background
  surface: "#FFFFFF",      // cards
  surface2: "#FAFBFC",     // panels inside a card: code, labor rows, mini loops
  elevated: "#FFFFFF",     // popovers, drawers
  track: "#F0F2F5",        // meter bars, lifecycle steps not reached
  chip: "#E6EAEE",
  hairline: "#E3E7EC",
  grid: "#EDF0F3",
  node: "#FFFFFF",         // loop map boxes
  nodeStroke: "#D5DBE1",
  edge: "#8E99A6",
  dashed: "#C8D0D8",
  lineNo: "#A7B0BA",
  link: "#2B4FB8",
  menuSelected: "#EEF0FA",
  tableHead: "#F7F8FA",
  tip: "#232F3E",          // the timeline's tooltip
  onTip: "#FFFFFF",
  tipMuted: "#B9C3CE",
  bar: "#232F3E",          // top bar: Squid Ink in both modes, the family's identity
  bar2: "#37475A",
  brand: "#FF9900",        // the top bar's bottom line, nothing else
  you: "#3B48CC",
  aws: "#D97400",
  onLane: "#FFFFFF",       // labels on a you/aws block
  live: "#1D8102",
  replay: "#0073BB",
  success: "#1D8102",
  warning: "#B7791F",
  error: "#D13212",
};

export type Palette = { [K in keyof typeof light]: string };

const dark: Palette = {
  ink: "#DCE3EA",
  ink2: "#B7C2CE",
  onInk: "#0E1217",
  text: "#E6EAEF",
  text2: "#A9B4C0",
  text3: "#808C99",
  paper: "#0E1217",
  surface: "#161B22",
  surface2: "#1B2129",
  elevated: "#1E242C",
  track: "#232A33",
  chip: "#2A323C",
  hairline: "#2A323C",
  grid: "#222932",
  node: "#1D232B",
  nodeStroke: "#38424D",
  edge: "#6F7A86",
  dashed: "#3E4853",
  lineNo: "#5D6874",
  link: "#7FA7F5",
  menuSelected: "rgba(116, 128, 236, 0.16)",
  tableHead: "#1B2129",
  tip: "#2C343E",
  onTip: "#E6EAEF",
  tipMuted: "#A9B4C0",
  bar: "#232F3E",
  bar2: "#37475A",
  brand: "#FF9900",
  you: "#7480EC",
  aws: "#D47D0B",
  onLane: "#FFFFFF",
  live: "#3FB950",
  replay: "#4AA3E8",
  success: "#3FB950",
  warning: "#D9A23A",
  error: "#F26D5B",
};

export const palettes: Record<ThemeMode, Palette> = { light, dark };

export const font = {
  sans: "'Archivo Variable', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'Noto Sans CJK SC', 'Helvetica Neue', Arial, sans-serif",
  mono: "'Martian Mono Variable', ui-monospace, 'SFMono-Regular', Menlo, 'PingFang SC', 'Microsoft YaHei', 'Noto Sans CJK SC', monospace",
} as const;

/** The CSS variable a token is published as: textMuted → --text-muted, ink2 → --ink-2. */
export const cssVar = (token: string) => `--${token.replace(/([a-z])([A-Z0-9])/g, "$1-$2").toLowerCase()}`;

/** A hex color at some opacity, for washes drawn in SVG (CSS uses color-mix instead). */
export function alpha(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}
