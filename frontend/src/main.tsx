import "@fontsource-variable/archivo/wdth.css";
import "@fontsource-variable/martian-mono/wdth.css";
import "./theme/global.css";
import i18n, { useLang } from "./i18n";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { App as AntApp, ConfigProvider } from "antd";
import enUS from "antd/locale/en_US";
import zhCN from "antd/locale/zh_CN";
import { StrictMode, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";
import AppRoutes from "./App";
import AuthGate from "./components/AuthGate";
import { antdTheme } from "./theme/antdTheme";
import { applyTheme, readThemePref, resolveTheme, ThemeProvider, useTheme } from "./theme/ThemeProvider";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
});
// Scenario text and Setup checks come localized from the backend: refetch them in the new language.
i18n.on("languageChanged", () => { void queryClient.invalidateQueries(); });

// CSS variables before the first render, so nothing paints with an unset color.
applyTheme(resolveTheme(readThemePref()));

function Providers({ children }: { children: ReactNode }) {
  const { mode } = useTheme();
  const lang = useLang();
  return (
    <ConfigProvider theme={antdTheme(mode)} locale={lang === "zh" ? zhCN : enUS}>
      <AntApp style={{ height: "100%" }}>{children}</AntApp>
    </ConfigProvider>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <Providers>
        <QueryClientProvider client={queryClient}>
          <AuthGate>
            <BrowserRouter>
              <AppRoutes />
            </BrowserRouter>
          </AuthGate>
        </QueryClientProvider>
      </Providers>
    </ThemeProvider>
  </StrictMode>,
);
