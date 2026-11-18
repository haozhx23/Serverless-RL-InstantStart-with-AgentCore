import { Button, Input } from "antd";
import { EditOutlined, UndoOutlined } from "@ant-design/icons";
import { Trans, useTranslation } from "react-i18next";
import { useScenarioFile } from "../api/hooks";
import type { Scenario, SlotSpec } from "../api/types";
import LoopCode, { symbolBlock } from "./LoopCode";
import LoopMap from "./LoopMap";

// The launch drawer's slot composer (docs/ui-design.md §6.1.1): the loop map with its swappable
// blocks, the variants of the selected slot, and that slot's code, editable in place.

export default function Composer({ scenario, choices, edits, slotKey, onSlot, onChoice, onEdit }: {
  scenario: Scenario; choices: Record<string, string | number>; edits: Record<string, string>; slotKey: string;
  onSlot: (k: string) => void; onChoice: (k: string, v: string | number) => void; onEdit: (file: string, code: string | undefined) => void;
}) {
  const { t } = useTranslation();
  const slots = scenario.slots!;
  const labels = Object.fromEntries(slots.map((sl) => {
    const v = sl.variants.find((x) => x.id === choices[sl.key]);
    return [sl.key, sl.file && edits[sl.file] !== undefined ? t("loop.panel.edited") : v?.short ?? v?.label ?? String(choices[sl.key])];
  }));
  const slot = slots.find((sl) => sl.key === slotKey)!;
  const params = Object.fromEntries((scenario.params ?? []).map((p) => [p.key, p.default]));
  const offMap = slots.filter((sl) => !sl.node);
  return (
    <section className="composer">
      <div className="composer-head">
        <b>{t("scenarios.composer.title")}</b>
        <span className="muted">{t("scenarios.composer.hint")}</span>
      </div>
      <LoopMap loop={scenario.loop!} mode="compose" params={params} selected={slot.node ?? null} slotLabels={labels}
               onSelect={(op) => { const sl = slots.find((x) => x.node === op); if (sl) onSlot(sl.key); }}
               clickable={(n) => n !== "header" && !!n.slot}
               zoneTitles={{ client: [t("loop.zone.process"), t("loop.zone.composeSub")], aws: [t("loop.zone.aws"), t("loop.zone.composeAwsSub")] }} />
      {offMap.length > 0 && (
        <div className="other-slots">
          <span className="muted">{t("scenarios.composer.alsoYours")}</span>
          {offMap.map((sl) => (
            <button key={sl.key} type="button" className={`slot-pill btn${sl.key === slotKey ? " on" : ""}`} onClick={() => onSlot(sl.key)}>
              ⇄ {sl.label} <span className="mono">{sl.file}</span>{sl.file && edits[sl.file] !== undefined ? t("loop.panel.fileEdited") : ""}
            </button>
          ))}
        </div>
      )}
      <SlotEditor key={slot.key} scenarioId={scenario.id} slot={slot} value={choices[slot.key]}
                  onPick={(v) => { onChoice(slot.key, v); if (slot.file) onEdit(slot.file, undefined); }}
                  edit={slot.file ? edits[slot.file] : undefined} onEdit={(code) => slot.file && onEdit(slot.file, code)} />
    </section>
  );
}

/** A training-session scenario with no slots (docs/ui-design.md §6.1.1): the same loop map, read-only. */
export function LoopPreview({ scenario }: { scenario: Scenario }) {
  const { t } = useTranslation();
  const params = Object.fromEntries((scenario.params ?? []).map((p) => [p.key, p.default]));
  return (
    <section className="composer">
      <div className="composer-head">
        <b>{t("scenarios.composer.title")}</b>
        <span className="muted">{t(scenario.source ? "scenarios.composer.hintOfficial" : "scenarios.composer.hintNoSlots")}</span>
      </div>
      <LoopMap loop={scenario.loop!} mode="compose" params={params}
               zoneTitles={{ client: [t("loop.zone.process"), t("loop.zone.processSub")], aws: [t("loop.zone.aws"), t("loop.zone.composeAwsSub")] }} />
    </section>
  );
}

function SlotEditor({ scenarioId, slot, value, onPick, edit, onEdit }: {
  scenarioId: string; slot: SlotSpec; value: string | number; onPick: (v: string | number) => void;
  edit: string | undefined; onEdit: (code: string | undefined) => void;
}) {
  const { t } = useTranslation();
  const variant = slot.variants.find((v) => v.id === value) ?? slot.variants[0];
  const file = useScenarioFile(scenarioId, slot.file ? variant.file ?? null : "loop.py");
  const edited = edit !== undefined;
  const content = file.data?.content;
  const highlight = !content ? null : slot.file ? (slot.function ? symbolBlock(content, slot.function) : null)
    : findParamUse(content, slot.param!);
  return (
    <div className="slot-editor">
      <div className="slot-head">
        <span className="eyebrow">{slot.label}</span>
        {slot.contract && <span className="mono muted" style={{ fontSize: 11.5 }}>{slot.contract}</span>}
      </div>
      <div className="variant-grid">
        {slot.variants.map((v) => (
          <button key={String(v.id)} type="button" aria-pressed={v.id === value && !edited}
                  className={`variant${v.id === value && !edited ? " on" : ""}`} onClick={() => onPick(v.id)}>
            <b>{v.label}</b>
            {v.summary && <span>{v.summary}</span>}
            {v.watch && <span className="muted">{t("scenarios.composer.watch", { watch: v.watch })}</span>}
          </button>
        ))}
        {edited && (
          <div className="variant on edited">
            <b>{t("scenarios.composer.edited")}</b>
            <span><Trans i18nKey="scenarios.composer.yourVersion" values={{ file: slot.file, variant: variant.label }}
                         components={{ code: <span className="mono" /> }} /></span>
          </div>
        )}
      </div>
      <div className="slot-code-bar">
        <span className="mono">{slot.file ?? "loop.py"}</span>
        {slot.file && variant.file !== slot.file && !edited && <span className="muted">{t("scenarios.composer.copiedFrom", { file: variant.file })}</span>}
        {!slot.file && <span className="muted"><Trans i18nKey="scenarios.composer.usesParam" values={{ param: slot.param, value: JSON.stringify(value) }}
                                                      components={{ code: <span className="mono" /> }} /></span>}
        <span style={{ flex: 1 }} />
        {slot.editable && !edited && (
          <Button size="small" icon={<EditOutlined />} disabled={!content} onClick={() => onEdit(content ?? "")}>{t("scenarios.composer.edit")}</Button>
        )}
        {edited && <Button size="small" icon={<UndoOutlined />} onClick={() => onEdit(undefined)}>{t("scenarios.composer.discard")}</Button>}
      </div>
      {edited ? (
        <>
          <Input.TextArea className="code-edit" value={edit} onChange={(e) => onEdit(e.target.value)} spellCheck={false}
                          autoSize={{ minRows: 10, maxRows: 26 }} aria-label={t("scenarios.composer.editAria", { file: slot.file })} />
          <p className="muted" style={{ fontSize: 12, margin: "6px 0 0" }}>
            <Trans i18nKey="scenarios.composer.checkNote" values={{ contract: slot.contract }} components={{ code: <span className="mono" /> }} />
            {slot.key === "reward" ? ` ${t("scenarios.composer.rewardNote")}` : ""}
          </p>
        </>
      ) : (
        <LoopCode content={content ?? (file.error ? String(file.error) : "…")} highlight={highlight} maxHeight={slot.file ? 300 : 220} />
      )}
    </div>
  );
}

/** The lines of loop.py that read a param slot's value, e.g. p["loss_fn"]. */
function findParamUse(src: string, param: string): [number, number] | null {
  const lines = src.split("\n");
  const at = lines.findIndex((l) => l.includes(`p["${param}"]`) && !/^\s*#/.test(l));
  if (at < 0) return null;
  const startIdx = Math.max(0, at - (/^\s*(with|if)\b/.test(lines[at]) ? 0 : 1));
  return [startIdx, Math.min(lines.length - 1, at + 2)];
}
