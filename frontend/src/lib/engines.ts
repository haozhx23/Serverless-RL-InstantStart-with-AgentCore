import type { TFunction } from "i18next";
import type { Engine, Readiness } from "../api/types";

// The two ways this app trains, each with its own scenario page (docs/ui-design.md §6.1).
// Only the entry is split: the run console, Runs, Artifacts, Capacity and Setup serve both.
export const ENGINE_PATH: Record<Engine, string> = {
  "training-session": "/scenarios/training-session",
  mtrl: "/scenarios/mtrl",
};

/** Where the app opens, and where /scenarios and unknown paths lead: training sessions are the demo's main line. */
export const HOME = ENGINE_PATH["training-session"];

/** Whether live runs of an engine can start, and the titles of the checks in the way (localized by the backend).
 * A scenario that runs code in the environment sandbox also needs the sandbox's checks (`sandbox`). */
export function readyFor(engine: Engine, r: Readiness | undefined, lang: "en" | "zh", sandbox = false) {
  if (!r) return { ready: false, blockers: [] as string[] };
  const ids = engine === "mtrl" ? r.mtrl_blockers : sandbox ? r.sandbox_blockers : r.live_blockers;
  return { ready: engine === "mtrl" ? r.mtrl_ready : sandbox ? r.sandbox_ready : r.live_ready,
           blockers: ids.map((id) => {
             const title = r.checks.find((c) => c.id === id)?.title ?? id;
             return lang === "en" ? title.charAt(0).toLowerCase() + title.slice(1) : title;
           }) };
}

/** Check titles in a sentence, with the interface language's list separator. */
export const joinList = (t: TFunction, items: string[]) => items.join(t("scenarios.listSep"));
