import { Dropdown, Layout, Menu, Tooltip } from "antd";
import {
  BranchesOutlined, CheckSquareOutlined, CodeOutlined, ContainerOutlined, DesktopOutlined, GlobalOutlined,
  HistoryOutlined, MoonOutlined, PlayCircleOutlined, RobotOutlined, SunOutlined,
} from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import { Link, Outlet, useLocation, useNavigate } from "react-router";
import { useAwsContext, useReadiness } from "../api/hooks";
import { LANGS, useLang } from "../i18n";
import { ENGINE_PATH, HOME } from "../lib/engines";
import { useTheme, type ThemePref } from "../theme/ThemeProvider";

const { Sider, Content } = Layout;

function SetupDot() {
  const { t } = useTranslation();
  const { color } = useTheme();
  const { data } = useReadiness();
  if (!data) return null;
  const failing = data.checks.filter((c) => c.status === "fail").length;
  const warn = data.checks.filter((c) => c.status === "warn").length;
  const c = failing ? color.error : warn ? color.warning : color.success;
  const label = failing ? t("shell.setupDot.failing", { count: failing }) : warn ? t("shell.setupDot.warn", { count: warn })
    : t("shell.setupDot.ok");
  return <span className="nav-dot" style={{ background: c }} aria-label={label} title={label} />;
}

const THEME_ICON: Record<ThemePref, React.ReactNode> = { light: <SunOutlined />, dark: <MoonOutlined />, system: <DesktopOutlined /> };

/** Theme and language: per browser, kept in localStorage (docs/ui-design.md §3.5, §7.1). */
function Preferences() {
  const { t, i18n } = useTranslation();
  const lang = useLang();
  const { pref, setPref } = useTheme();
  const themeName = t(`shell.theme.${pref}`);
  return (
    <>
      <Dropdown trigger={["click"]} menu={{
        selectable: true, selectedKeys: [lang],
        items: LANGS.map((l) => ({ key: l.value, label: <span lang={l.value === "zh" ? "zh-CN" : "en"}>{l.label}</span> })),
        onClick: ({ key }) => { void i18n.changeLanguage(key); },
      }}>
        <button className="chip" aria-label={`${t("shell.language")}: ${LANGS.find((l) => l.value === lang)!.label}`}>
          <GlobalOutlined />
          <span>{lang === "zh" ? "中文" : "EN"}</span>
        </button>
      </Dropdown>
      <Dropdown trigger={["click"]} menu={{
        selectable: true, selectedKeys: [pref],
        items: (["light", "dark", "system"] as const).map((p) => ({ key: p, icon: THEME_ICON[p], label: t(`shell.theme.${p}`) })),
        onClick: ({ key }) => setPref(key as ThemePref),
      }}>
        <button className="chip" aria-label={`${t("shell.theme.label")}: ${themeName}`} title={`${t("shell.theme.label")}: ${themeName}`}>
          {THEME_ICON[pref]}
        </button>
      </Dropdown>
    </>
  );
}

export default function AppShell() {
  const { t } = useTranslation();
  const location = useLocation();
  const navigate = useNavigate();
  const ctx = useAwsContext().data;
  const items = [
    { key: "/setup", icon: <CheckSquareOutlined />, label: <span>{t("shell.nav.setup")}<SetupDot /></span> },
    { key: ENGINE_PATH["training-session"], icon: <CodeOutlined />, label: t("shell.nav.ts") },
    { key: ENGINE_PATH.mtrl, icon: <RobotOutlined />, label: t("shell.nav.mtrl") },
    { key: "/runs", icon: <PlayCircleOutlined />, label: t("shell.nav.runs") },
    { key: "/artifacts", icon: <ContainerOutlined />, label: t("shell.nav.artifacts") },
    { type: "divider" as const },
    { key: "/capacity", icon: <HistoryOutlined />, label: t("shell.nav.capacity") },
    { type: "divider" as const },
    { key: "/paths", icon: <BranchesOutlined />, label: t("shell.nav.paths") },
  ];
  // The item whose page this is, or whose page this is under (/runs/<id> is under /runs).
  const section = items.flatMap((i) => ("key" in i ? [i.key] : []))
    .find((k) => location.pathname === k || location.pathname.startsWith(`${k}/`)) ?? "";

  return (
    <Layout style={{ minHeight: "100%" }}>
      <header className="app-header">
        <Link to={HOME} className="wordmark">
          <span className="name">Serverless RL InstantStart</span>
          <span className="sub">{t("shell.product.sub")}</span>
        </Link>
        <div className="right">
          <Tooltip title={t("shell.regionTip")}>
            <button className="chip" onClick={() => navigate("/setup")}>
              <span>{ctx?.region ?? "…"}</span>
              <span className="mono" style={{ color: "#AEB8C3" }}>
                {ctx?.account ? `${ctx.account.slice(0, 4)}…${ctx.account.slice(-4)}` : ctx?.identity_error ? t("shell.noCredentials") : ""}
              </span>
            </button>
          </Tooltip>
          <Preferences />
        </div>
      </header>
      <Layout hasSider>
        <Sider width={208} theme="light" style={{ borderRight: "1px solid var(--hairline)" }}>
          <Menu mode="inline" selectedKeys={[section]} items={items} onClick={({ key }) => navigate(key)}
                style={{ borderInlineEnd: 0, paddingTop: 8 }} />
        </Sider>
        <Content style={{ minWidth: 0 }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
}
