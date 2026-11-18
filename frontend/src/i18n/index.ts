import i18n from "i18next";
import LanguageDetector from "i18next-browser-languagedetector";
import { initReactI18next, useTranslation } from "react-i18next";
import artifacts from "./msg/artifacts";
import capacity from "./msg/capacity";
import common from "./msg/common";
import consoleNs from "./msg/console";
import loop from "./msg/loop";
import paths from "./msg/paths";
import runs from "./msg/runs";
import scenarios from "./msg/scenarios";
import setup from "./msg/setup";
import shell from "./msg/shell";

// Interface language: English (the source) or Simplified Chinese (docs/ui-design.md §7.1).
// The first visit follows the browser's language; the choice is then kept in localStorage.
// Text that comes from the backend (scenarios, Setup checks) is localized there: api() sends the
// chosen language as Accept-Language, and main.tsx refetches every query when it changes.

const NS = { common, shell, loop, scenarios, console: consoleNs, setup, capacity, artifacts, runs, paths };
const pick = <L extends "en" | "zh">(lang: L) =>
  Object.fromEntries(Object.entries(NS).map(([k, v]) => [k, v[lang]])) as { [K in keyof typeof NS]: (typeof NS)[K][L] };

const en = pick("en");
export type Messages = typeof en;

export type Lang = "en" | "zh";
export const LANGS: { value: Lang; label: string }[] = [{ value: "en", label: "English" }, { value: "zh", label: "简体中文" }];

i18n.use(LanguageDetector).use(initReactI18next).init({
  resources: { en: { translation: en }, zh: { translation: pick("zh") } },
  supportedLngs: ["en", "zh"],
  nonExplicitSupportedLngs: true,  // zh-CN, zh-TW → zh
  load: "languageOnly",
  fallbackLng: "en",
  detection: { order: ["localStorage", "navigator"], lookupLocalStorage: "srl.lang", caches: ["localStorage"] },
  interpolation: { escapeValue: false },  // React escapes
});

export const currentLang = (): Lang => (i18n.resolvedLanguage === "zh" ? "zh" : "en");
/** BCP 47 tag for Intl formatting and <html lang>. */
export const localeTag = (): string => (currentLang() === "zh" ? "zh-CN" : "en-US");

const syncHtmlLang = () => { document.documentElement.lang = currentLang() === "zh" ? "zh-CN" : "en"; };
syncHtmlLang();
i18n.on("languageChanged", syncHtmlLang);

export function useLang(): Lang {
  const { i18n: inst } = useTranslation();
  return inst.resolvedLanguage === "zh" ? "zh" : "en";
}

export default i18n;
