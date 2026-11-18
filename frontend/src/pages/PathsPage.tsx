import { Fragment } from "react";
import type { TFunction } from "i18next";
import { Trans, useTranslation } from "react-i18next";
import type { LoopSpec } from "../api/types";
import { PageHeader } from "../components/Bits";
import LoopMap from "../components/LoopMap";

type Zones = { client: [string, string]; aws: [string, string] };

// Where the loop lives on each path, drawn with the same grammar as the run console.
function loops(t: TFunction): { loop: LoopSpec; zones: Zones; yours?: boolean }[] {
  const n = (op: string, lane: "client" | "aws", title: string) => ({ op, lane, title, call: "" });
  return [
    { loop: { driver: "client", header: "", before: [], after: [], nodes: [
        n("s", "aws", t("paths.node.sample")), n("r", "client", t("paths.node.reward")),
        n("fb", "aws", t("paths.node.forwardBackward")), n("o", "aws", t("paths.node.optimStep"))] },
      zones: { client: [t("paths.zone.process"), ""], aws: [t("paths.zone.aws"), ""] } },
    { loop: { driver: "aws", header: "", before: [], after: [], nodes: [
        n("s", "aws", t("paths.node.sampleTurn")), n("a", "client", t("paths.node.agent")), n("u", "aws", t("paths.node.update"))] },
      zones: { client: [t("paths.zone.agent"), ""], aws: [t("paths.zone.aws"), ""] } },
    { loop: { driver: "client", header: "", before: [], after: [], nodes: [
        n("s", "aws", t("paths.node.workers")), n("r", "client", t("paths.node.reward")), n("t", "aws", t("paths.node.trainer"))] },
      zones: { client: [t("paths.zone.code"), ""], aws: [t("paths.zone.cluster"), ""] }, yours: true },
  ];
}

// Boundaries from the 2026-08/10 research (260818-hypd-rl-boundaries.md, training-session report).
const COLS = ["ts", "mtrl", "hp"] as const;
const ROWS = ["who", "environment", "change", "models", "billing", "when"] as const;

export default function PathsPage() {
  const { t } = useTranslation();
  const drawn = loops(t);
  return (
    <>
      <PageHeader crumbs={t("paths.crumbs")} title={t("shell.nav.paths")} />
      <div className="page-body" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <p className="secondary" style={{ margin: 0, fontSize: 13.5, maxWidth: 860 }}>{t("paths.intro")}</p>
        <div className="paths">
          <div className="corner" />
          {COLS.map((c, i) => (
            <div key={c} className="colhead">
              <div className="display" style={{ fontSize: 16 }}>{t(`paths.${c}.name`)}</div>
              <div className="muted" style={{ fontSize: 12, marginBottom: 8 }}>{t(`paths.${c}.sub`)}</div>
              <div className="mini-loop">
                <LoopMap loop={drawn[i].loop} mode="compact" zoneTitles={drawn[i].zones} rightIsYours={drawn[i].yours}
                         ariaLabel={t("paths.aria", { name: t(`paths.${c}.name`) })} />
              </div>
            </div>
          ))}
          {ROWS.map((r) => (
            <Fragment key={r}>
              <div className="rowhead">{t(`paths.row.${r}`)}</div>
              {COLS.map((c) => <div key={c}>{t(`paths.${c}.${r}`)}</div>)}
            </Fragment>
          ))}
        </div>
        <p style={{ fontSize: 13.5, margin: 0 }}>
          <Trans i18nKey="paths.footer"
                 components={{ link: <a href="https://github.com/haozhx23/HyperPod-InstantStart" target="_blank" rel="noreferrer" /> }} />
        </p>
      </div>
    </>
  );
}
