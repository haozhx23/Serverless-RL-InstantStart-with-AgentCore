import { bilingual } from "../types";

// The app shell: navigation, the top bar, the theme and language switches, the sign-in page.
export default bilingual({
  nav: {
    setup: "Setup",
    // The two scenario pages, one per engine. "MTRL" is the same in both languages; the page
    // itself spells it out (scenarios.mtrlPageTitle).
    ts: "Training sessions",
    mtrl: "MTRL",
    runs: "Runs",
    artifacts: "Artifacts",
    capacity: "Capacity",
    paths: "Paths",
  },
  setupDot: {
    failing_one: "{{count}} check failing",
    failing_other: "{{count}} checks failing",
    warn_one: "{{count}} warning",
    warn_other: "{{count}} warnings",
    ok: "all checks pass",
  },
  product: { sub: "SageMaker AI × AgentCore" },
  regionTip: "Region and account this app works in. Change the region on the Setup page.",
  noCredentials: "no credentials",
  pips: "{{active}} of {{limit}} slots in use",
  limitSource: {
    account: "account quota",
    awsDefault: "AWS default",
    notRead: "default (not read)",
  },
  theme: {
    label: "Theme",
    light: "Light",
    dark: "Dark",
    system: "Match system",
  },
  language: "Language",
  auth: {
    backendDown: "Cannot reach the backend: {{error}}",
    prompt: "Enter the access token printed when the app started.",
    placeholder: "Access token",
    signIn: "Sign in",
  },
}, {
  nav: {
    setup: "就绪检查",
    ts: "Training session",
    mtrl: "MTRL",
    runs: "运行",
    artifacts: "产物",
    capacity: "容量",
    paths: "训练方式",
  },
  setupDot: {
    failing_one: "{{count}} 项检查未通过",
    failing_other: "{{count}} 项检查未通过",
    warn_one: "{{count}} 项警告",
    warn_other: "{{count}} 项警告",
    ok: "全部检查通过",
  },
  product: { sub: "SageMaker AI × AgentCore" },
  regionTip: "本应用使用的 AWS 区域和账号。可在就绪检查页切换区域。",
  noCredentials: "无凭证",
  pips: "并发名额已用 {{active}} / {{limit}}",
  limitSource: {
    account: "账号配额",
    awsDefault: "AWS 默认值",
    notRead: "默认值（未读取配额）",
  },
  theme: {
    label: "主题",
    light: "浅色",
    dark: "深色",
    system: "跟随系统",
  },
  language: "语言",
  auth: {
    backendDown: "无法连接后端：{{error}}",
    prompt: "请输入应用启动时输出的访问令牌。",
    placeholder: "访问令牌",
    signIn: "登录",
  },
});
