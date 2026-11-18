import { theme, type ThemeConfig } from "antd";
import { font, palettes, type ThemeMode } from "./tokens";

// Primary actions are ink, not orange: orange is reserved for the AWS GPU track. In dark mode
// ink turns light, so text on primary fills (buttons, switch handles) takes onInk.
export function antdTheme(mode: ThemeMode): ThemeConfig {
  const c = palettes[mode];
  return {
    algorithm: mode === "dark" ? theme.darkAlgorithm : theme.defaultAlgorithm,
    token: {
      colorPrimary: c.ink,
      colorLink: c.link,
      colorSuccess: c.success,
      colorWarning: c.warning,
      colorError: c.error,
      colorInfo: c.replay,
      colorText: c.text,
      colorTextSecondary: c.text2,
      colorBorderSecondary: c.hairline,
      colorBgLayout: c.paper,
      colorBgContainer: c.surface,
      colorBgElevated: c.elevated,
      fontFamily: font.sans,
      fontFamilyCode: font.mono,
      fontSize: 14,
      borderRadius: 6,
    },
    components: {
      Layout: { headerBg: "transparent", headerHeight: 56, headerPadding: "0 20px", siderBg: c.surface, bodyBg: c.paper },
      Menu: { itemBg: c.surface, itemSelectedBg: c.menuSelected, itemSelectedColor: c.ink, itemHeight: 38 },
      Card: { headerFontSize: 15 },
      // Selected rows take the same tint as the selected menu item; the default derives from ink and turns rows dark.
      Table: { headerBg: c.tableHead, headerColor: c.text2, cellPaddingBlockSM: 6, rowSelectedBg: c.menuSelected,
               rowSelectedHoverBg: c.menuSelected },
      Button: { primaryShadow: "none", fontWeight: 500, primaryColor: c.onInk },
      Switch: { handleBg: c.onInk },
      Tabs: { inkBarColor: c.ink, itemSelectedColor: c.ink, itemHoverColor: c.ink2 },
      Progress: { remainingColor: c.track },
    },
  };
}
