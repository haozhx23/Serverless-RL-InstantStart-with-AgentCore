import { createContext, useContext, useEffect, useLayoutEffect, useMemo, useState, type ReactNode } from "react";
import { cssVar, font, palettes, type Palette, type ThemeMode } from "./tokens";

// Light, dark, or whatever the operating system prefers (docs/ui-design.md §3.5). The choice is
// kept in this browser's localStorage. index.html applies the same choice before any CSS loads,
// so a dark page never flashes white; this module keeps it in sync once the app runs.

export type ThemePref = "system" | "light" | "dark";
const KEY = "srl.theme";
const media = () => window.matchMedia("(prefers-color-scheme: dark)");

export function readThemePref(): ThemePref {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" || v === "system" ? v : "system";
  } catch {
    return "system";  // storage blocked: follow the system, and the choice is not remembered
  }
}

export function resolveTheme(pref: ThemePref): ThemeMode {
  return pref === "system" ? (media().matches ? "dark" : "light") : pref;
}

/** Publish a palette as CSS variables on <html>. Called before the first render, then on every change. */
export function applyTheme(mode: ThemeMode) {
  const root = document.documentElement;
  for (const [k, v] of Object.entries(palettes[mode])) root.style.setProperty(cssVar(k), v);
  root.style.setProperty("--sans", font.sans);
  root.style.setProperty("--mono", font.mono);
  root.dataset.theme = mode;
  root.style.colorScheme = mode;
  root.style.background = palettes[mode].paper;
}

interface ThemeState { pref: ThemePref; mode: ThemeMode; color: Palette; setPref: (p: ThemePref) => void }
const Ctx = createContext<ThemeState | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [pref, setPrefState] = useState<ThemePref>(readThemePref);
  const [systemDark, setSystemDark] = useState(() => media().matches);
  useEffect(() => {
    const m = media();
    const on = (e: MediaQueryListEvent) => setSystemDark(e.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);
  const mode: ThemeMode = pref === "system" ? (systemDark ? "dark" : "light") : pref;
  useLayoutEffect(() => applyTheme(mode), [mode]);
  const value = useMemo<ThemeState>(() => ({
    pref, mode, color: palettes[mode],
    setPref: (p) => {
      try { localStorage.setItem(KEY, p); } catch (e) { console.warn(`Theme choice not saved: ${(e as Error).message}`); }
      setPrefState(p);
    },
  }), [pref, mode]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useTheme(): ThemeState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useTheme() must be called inside <ThemeProvider>");
  return v;
}
