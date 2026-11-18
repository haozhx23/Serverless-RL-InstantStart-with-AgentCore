import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { useLang } from ".";

/**
 * Fixed sentences that run code writes into events ("Handled for you" notes, reward-check
 * names and reasons, comparison labels), translated when the backend's catalog has them
 * (templates/_events.zh.json and each scenario.zh.json). Anything else stays as the run
 * wrote it: notes with numbers, logs, errors (docs/ui-design.md §7.1).
 */
export function useEventText() {
  const lang = useLang();
  const { data } = useQuery({
    queryKey: ["i18n-events"], staleTime: Infinity,
    queryFn: () => api<{ zh: Record<string, string> }>("/api/i18n/events"),
  });
  return (text: string) => (lang === "zh" && data?.zh[text]) || text;
}
