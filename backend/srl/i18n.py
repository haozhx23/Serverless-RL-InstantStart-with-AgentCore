"""Interface language for the text the backend serves: scenarios, Setup checks, run text.

English is the source language: scenario.json, the check functions and the run templates are
written in it. The UI asks for Chinese with ``Accept-Language: zh-CN`` (docs/architecture.md 3.7).

Scenario text. ``templates/<id>/scenario.zh.json`` holds::

    {"fields": {"<path>": {"en": "<source>", "zh": "<translation>"}},
     "events": {"<sentence a run writes into an event>": "<translation>"}}

A path names one translatable field (``scenario_fields`` lists them: title, node.sample.title,
slot.reward.partial.watch, ...). A translation is used only while its "en" still equals the
field's current text, so an edited English sentence shows in English until its translation is
updated; tests/test_i18n.py fails until then. A planned card is a directory too and has its own
scenario.zh.json; ``templates/_events.zh.json`` holds the event sentences the shared run code
writes.

Setup checks carry both languages from where they are written: an ``L(en, zh)`` pair for the
title, the reason and the fix. Observed values, errors and run logs are never translated.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path

LANGS = ("en", "zh")


def request_lang(accept_language: str | None) -> str:
    """"zh" when the first language asked for is Chinese (zh, zh-CN, zh-Hans...), else "en"."""
    first = (accept_language or "").split(",")[0].strip().lower()
    return "zh" if first.startswith("zh") else "en"


class L(dict):
    """A sentence in both languages, where it is written: L("Create role", "创建角色")."""

    def __init__(self, en: str, zh: str):
        super().__init__(en=en, zh=zh)


def en(text) -> str:
    return text["en"] if isinstance(text, L) else text


def bilingual(text) -> dict | None:
    """The {"en", "zh"} pair for a text, or None when it has no translation."""
    return dict(text) if isinstance(text, L) else None


# -- scenario fields --------------------------------------------------------------------------

def scenario_fields(meta: dict, optional: bool = True) -> Iterator[tuple[str, dict | list, str | int]]:
    """(path, container, key) of every translatable string in a scenario-shaped dict.

    Works on a scenario from scenario.json and on the snapshot a run keeps in run.json
    (title, loop, slots, metrics), which has the same shape where it has these fields.
    File names, API names and parameter keys are not text and are not listed. The loop's
    header and each node's call are usually code ("for step in range(steps)"), so they are
    optional: a catalog may translate them when a scenario writes prose there (MTRL's
    "inside the job"), and the completeness test does not ask for them.
    """
    for k in ("title", "technique", "summary", "plan"):
        if isinstance(meta.get(k), str) and meta[k]:
            yield k, meta, k
    for i, item in enumerate(meta.get("you_will_see") or []):
        yield f"you_will_see.{i}", meta["you_will_see"], i
    evidence = meta.get("evidence") or {}
    if isinstance(evidence.get("note"), str) and evidence["note"]:
        yield "evidence.note", evidence, "note"
    sandbox = meta.get("sandbox") or {}
    if isinstance(sandbox.get("executes"), str) and sandbox["executes"]:
        yield "sandbox.executes", sandbox, "executes"
    source = meta.get("source") or {}
    if isinstance(source.get("published"), str) and source["published"]:
        yield "source.published", source, "published"
    loop = meta.get("loop") or {}
    if optional and loop.get("header"):
        yield "loop.header", loop, "header"
    if loop.get("header_out"):
        yield "loop.header_out", loop, "header_out"
    for node in loop.get("nodes") or []:
        yield f"node.{node['op']}.title", node, "title"
        if optional and node.get("call"):
            yield f"node.{node['op']}.call", node, "call"
        if node.get("out"):
            yield f"node.{node['op']}.out", node, "out"
    for slot in meta.get("slots") or []:
        yield f"slot.{slot['key']}.label", slot, "label"
        for variant in slot.get("variants") or []:
            for k in ("label", "short", "summary", "watch"):
                if variant.get(k):
                    yield f"slot.{slot['key']}.{variant['id']}.{k}", variant, k
    for param in meta.get("params") or []:
        yield f"param.{param['key']}.label", param, "label"
        if param.get("help"):
            yield f"param.{param['key']}.help", param, "help"
    for metric in meta.get("metrics") or []:
        yield f"metric.{metric['key']}.label", metric, "label"


def scenario_strings(meta: dict, optional: bool = True) -> dict[str, str]:
    """path -> current English text, for every translatable field of a scenario (`optional`: see scenario_fields)."""
    return {path: box[key] for path, box, key in scenario_fields(meta, optional)}


def load_catalog(path: Path, meta: dict | None) -> dict:
    """Read one catalog file; a missing file is an empty catalog. See `parse_catalog`."""
    if not path.exists():
        return {"fields": {}, "events": {}}
    return parse_catalog(json.loads(path.read_text(encoding="utf-8")), meta, str(path))


def parse_catalog(data: dict, meta: dict | None, where: str) -> dict:
    """Check one catalog, {"fields": {...}, "events": {...}}, against the scenario it translates.

    Raises ValueError for a broken catalog (bad shape, a path the scenario does not have), so a
    typo stops the app at start-up instead of silently showing English. Missing or outdated
    translations are not errors here; tests/test_i18n.py reports them.
    """
    unknown = sorted(set(data) - {"fields", "events"})
    if unknown:
        raise ValueError(f"{where}: unknown top-level keys {unknown}; expected 'fields' and 'events'")
    fields = data.get("fields", {})
    events = data.get("events", {})
    if meta is not None:
        known = scenario_strings(meta)
        for p, entry in fields.items():
            if p not in known:
                raise ValueError(f"{where}: {p!r} is not a translatable field of scenario {meta['id']!r}")
            if not (isinstance(entry, dict) and isinstance(entry.get("en"), str) and isinstance(entry.get("zh"), str)):
                raise ValueError(f"{where}: {p!r} must be {{\"en\": ..., \"zh\": ...}} strings")
    elif fields:
        raise ValueError(f"{where}: 'fields' needs a scenario to belong to")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in events.items()):
        raise ValueError(f"{where}: 'events' must map English sentences to translations")
    return {"fields": fields, "events": events}


def merge_events(catalogs: list[tuple[str, dict]]) -> dict[str, str]:
    """One sentence -> translation map from several catalogs. The same sentence translated two ways is an error."""
    out: dict[str, str] = {}
    seen: dict[str, str] = {}
    for where, cat in catalogs:
        for k, v in cat["events"].items():
            if k in out and out[k] != v:
                raise ValueError(f"{where} and {seen[k]} translate {k!r} differently: {v!r} vs {out[k]!r}")
            out[k], seen[k] = v, where
    return out


def localize(meta: dict, fields: dict, lang: str) -> dict:
    """A copy of a scenario (or a run's snapshot of it) with its text in `lang`.

    A field is translated only when the catalog has it and its "en" matches the current text.
    """
    out = copy.deepcopy(meta)
    if lang == "en" or not fields:
        return out
    for path, box, key in scenario_fields(out):
        entry = fields.get(path)
        if entry and entry["en"] == box[key]:
            box[key] = entry[lang]
    return out


# -- Setup checks -----------------------------------------------------------------------------

def check_texts(title, why, fix: dict | None) -> dict:
    """The English fields of a check, plus their Chinese under "i18n" when they have one."""
    zh = {}
    if bilingual(title):
        zh["title"] = title["zh"]
    if bilingual(why):
        zh["why"] = why["zh"]
    out_fix = None
    if fix:
        out_fix = {**fix, "label": en(fix["label"]), "detail": en(fix["detail"])}
        fzh = {k: fix[k]["zh"] for k in ("label", "detail") if bilingual(fix[k])}
        if fzh:
            zh["fix"] = fzh
    return {"title": en(title), "why": en(why), "fix": out_fix, "i18n": {"zh": zh} if zh else {}}


def localize_checks(result: dict, lang: str) -> dict:
    """A copy of a readiness result with each check's title, reason and fix in `lang`."""
    if lang == "en":
        return result
    out = {**result, "checks": []}
    for c in result["checks"]:
        zh = (c.get("i18n") or {}).get(lang) or {}
        c = {**c, "title": zh.get("title", c["title"]), "why": zh.get("why", c["why"])}
        if c.get("fix") and zh.get("fix"):
            c["fix"] = {**c["fix"], **zh["fix"]}
        out["checks"].append(c)
    return out


def pick(text, lang: str) -> str:
    """One language of an L pair (or a plain string, which is English only)."""
    return text[lang] if isinstance(text, L) else text
