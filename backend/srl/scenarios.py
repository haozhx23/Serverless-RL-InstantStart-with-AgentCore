"""Scenario catalog: one directory per scenario card under templates/, runnable or planned.

A card is templates/<id>/ and holds everything that belongs to it: scenario.json, its Chinese
catalog scenario.zh.json, the files its loop needs, its slot variants and its shipped live recordings
(recordings/).
Nothing outside the directory is specific to it, so the release can leave a card out by not copying the directory
(release-manifest.json `scenarios`). templates/_shared/ is copied next to a card's files when a run
is created. A planned card (status "planned") has only scenario.json and scenario.zh.json, and
cannot be launched.

A scenario's ``slots`` are the parts of its loop a customer is expected to change. A file
slot (reward.py, advantage.py, ...) has variants stored under the template directory; the
chosen one is copied into the run under the slot's file name, or replaced by code edited in
the browser. A param slot (loss_fn, rebind_every) is an ordinary parameter whose choices
the UI presents on the loop map.

Scenario text can be served in Chinese: each card's scenario.zh.json and templates/_events*.zh.json
are loaded and checked here (see srl/i18n.py).
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

from .errors import Invalid, NotFound
from .i18n import load_catalog, localize, merge_events

MAX_CUSTOM_BYTES = 64 * 1024
RECORDINGS = "recordings"  # a card's shipped live recordings (scripts/ship_recording.py); never copied into a run
PLANNED_FILES = {"scenario.json", "scenario.zh.json"}


class ParamError(Invalid):
    """A launch's parameters, slot choices or edited code that the scenario does not accept."""


class ScenarioRegistry:
    def __init__(self, templates_dir: Path):
        self.dir = templates_dir
        self._runnable: dict[str, dict] = {}
        self._planned: list[dict] = []
        self._zh: dict[str, dict] = {}   # scenario id -> its Chinese catalog
        for meta_path in sorted(templates_dir.glob("*/scenario.json")):
            if meta_path.parent.name.startswith("_"):
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["id"] != meta_path.parent.name:
                raise ValueError(f"{meta_path}: id {meta['id']!r} must match its directory name")
            if meta.get("status") == "planned":
                extra = sorted({p.name for p in meta_path.parent.iterdir()} - PLANNED_FILES)
                if extra:
                    raise ValueError(f"{meta_path.parent}: a planned card holds only scenario.json and "
                                     f"scenario.zh.json, found {', '.join(extra)}; give it status and files "
                                     "when it becomes runnable")
                self._planned.append(meta)
            else:
                for f in meta["files"]:
                    if not (meta_path.parent / f).is_file():
                        raise ValueError(f"{meta_path}: listed file {f} does not exist")
                _check_slots(meta_path, meta)
                _check_estimate(meta_path, meta)
                _check_sandbox(meta_path, meta)
                _check_probe_example(meta_path, meta)
                _check_recordings(meta_path.parent / RECORDINGS, meta["id"])
                self._runnable[meta["id"]] = meta
            self._zh[meta["id"]] = load_catalog(meta_path.parent / "scenario.zh.json", meta)
        # The shared run code's sentences: _events.zh.json, plus one _events.<part>.zh.json per part of it
        # that a release can leave out (release-manifest.json features).
        shared = [(str(p), load_catalog(p, None)) for p in sorted(templates_dir.glob("_events*.zh.json"))]
        self._events_zh = merge_events(shared + [(f"{sid} scenario.zh.json", c) for sid, c in sorted(self._zh.items())])

    def all(self, lang: str = "en") -> list[dict]:
        metas = sorted([*self._runnable.values(), *self._planned], key=lambda m: m.get("order", 99))
        return [self.localized(m, lang) for m in metas]

    def localized(self, meta: dict, lang: str) -> dict:
        """A copy of a scenario, or of a run's snapshot of one, with its text in `lang`."""
        return localize(meta, self._zh.get(meta.get("scenario", meta.get("id")), {}).get("fields", {}), lang)

    def catalog(self, scenario_id: str) -> dict:
        """The Chinese catalog of one scenario: {"fields": ..., "events": ...} (empty when it has none)."""
        return self._zh.get(scenario_id, {"fields": {}, "events": {}})

    def events(self, lang: str) -> dict[str, str]:
        """Event sentence -> translation, across the shared and every scenario's catalog."""
        return self._events_zh if lang == "zh" else {}

    def get(self, scenario_id: str) -> dict:
        if scenario_id not in self._runnable:
            planned = {m["id"] for m in self._planned}
            if scenario_id in planned:
                raise NotFound(f"scenario {scenario_id!r} is planned and cannot be launched yet")
            raise NotFound(f"unknown scenario {scenario_id!r}")
        return self._runnable[scenario_id]

    def template_dir(self, scenario_id: str) -> Path:
        self.get(scenario_id)
        return self.dir / scenario_id

    def shipped_recording_dirs(self) -> list[Path]:
        """The recordings/ of every runnable card that has one: the live recordings that come with the project."""
        return [self.dir / sid / RECORDINGS for sid in sorted(self._runnable) if (self.dir / sid / RECORDINGS).is_dir()]

    def engine(self, scenario_id: str) -> str:
        return self.get(scenario_id).get("engine", "training-session")

    def needs_sandbox(self, scenario_id: str) -> bool:
        """Whether a live run of this scenario sends code to the environment sandbox (scenario.json `sandbox`)."""
        return "sandbox" in self.get(scenario_id)

    def source(self, scenario_id: str, name: str) -> str:
        """A template file a launch can show before it runs: the scenario's files and every slot variant."""
        meta = self.get(scenario_id)
        allowed = set(meta["files"]) | {v["file"] for sl in meta.get("slots", []) if "file" in sl for v in sl["variants"]}
        if name not in allowed:
            raise NotFound(f"{name} is not a file of scenario {scenario_id}")
        return (self.dir / scenario_id / name).read_text(encoding="utf-8")

    def resolve_slots(self, scenario_id: str, variants: dict, files: dict) -> dict:
        """What each file slot of a new run gets: a variant to copy, or code edited in the browser.

        Returns {"copy": {target: source}, "write": {target: code}, "chosen": {slot: variant id or "custom"}}.
        Unknown slots, unknown variants, files that are not editable slots and code that does
        not parse or lacks the slot's function are errors, never ignored.
        """
        meta = self.get(scenario_id)
        file_slots = {sl["key"]: sl for sl in meta.get("slots", []) if "file" in sl}
        unknown = sorted(set(variants) - set(file_slots))
        if unknown:
            raise ParamError(f"{scenario_id} has no code slot named {', '.join(unknown)}; "
                             f"its code slots are: {', '.join(file_slots) or 'none'}")
        editable = {sl["file"]: sl for sl in file_slots.values() if sl.get("editable")}
        not_editable = sorted(set(files) - set(editable))
        if not_editable:
            raise ParamError(f"{', '.join(not_editable)} cannot be edited for {scenario_id}; "
                             f"editable files: {', '.join(editable) or 'none'}")
        plan: dict = {"copy": {}, "write": {}, "chosen": {}}
        for key, sl in file_slots.items():
            if sl["file"] in files:
                plan["write"][sl["file"]] = _check_code(sl, files[sl["file"]])
                plan["chosen"][key] = "custom"
                continue
            chosen = variants.get(key, sl["default"])
            variant = next((v for v in sl["variants"] if v["id"] == chosen), None)
            if variant is None:
                raise ParamError(f"{key} must be one of {', '.join(str(v['id']) for v in sl['variants'])}, got {chosen!r}")
            plan["copy"][sl["file"]] = variant["file"]
            plan["chosen"][key] = chosen
        return plan

    def validate_params(self, scenario_id: str, given: dict, models: list[str]) -> dict:
        """Defaults filled in, every value type- and range-checked. Unknown keys are an error."""
        spec = {p["key"]: p for p in self.get(scenario_id)["params"]}
        unknown = sorted(set(given) - set(spec))
        if unknown:
            raise ParamError(f"unknown parameter(s) for {scenario_id}: {', '.join(unknown)}")
        out = {}
        for key, p in spec.items():
            value = given.get(key, p["default"])
            out[key] = _check(p, value, models)
        return out


def _check(p: dict, value, models: list[str]):
    key, kind = p["key"], p["type"]
    if kind == "bool":
        if not isinstance(value, bool):
            raise ParamError(f"{key} must be true or false, got {value!r}")
        return value
    if kind in ("model", "mtrl_model"):
        if value not in models:
            raise ParamError(f"{key}: {value!r} is not trainable here; choose one of {', '.join(models)}")
        return value
    if kind == "choice":
        if isinstance(value, bool) or value not in p["choices"]:
            raise ParamError(f"{key} must be one of {', '.join(map(str, p['choices']))}, got {value!r}")
        return value
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            else:
                raise ParamError(f"{key} must be an integer, got {value!r}")
    elif kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ParamError(f"{key} must be a number, got {value!r}")
        value = float(value)
    else:
        raise ValueError(f"parameter {key} has unknown type {kind!r}")
    lo, hi = p.get("min"), p.get("max")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ParamError(f"{key} must be between {lo} and {hi}, got {value}")
    return value


def _check_slots(meta_path: Path, meta: dict) -> None:
    """A scenario's slots must point at things that exist: variant files, params, loop nodes."""
    params = {p["key"]: p for p in meta.get("params", [])}
    nodes = {n["op"] for n in meta.get("loop", {}).get("nodes", [])}
    for sl in meta.get("slots", []):
        where = f"{meta_path}: slot {sl['key']}"
        if sl.get("node") is not None and sl["node"] not in nodes:
            raise ValueError(f"{where} names loop node {sl['node']!r}, which is not in loop.nodes")
        if "file" in sl:
            if sl["file"] not in meta["files"]:
                raise ValueError(f"{where}: {sl['file']} must be listed in files")
            for v in sl["variants"]:
                if not (meta_path.parent / v["file"]).is_file():
                    raise ValueError(f"{where}: variant {v['id']} file {v['file']} does not exist")
            if sl["default"] not in {v["id"] for v in sl["variants"]}:
                raise ValueError(f"{where}: default {sl['default']!r} is not one of its variants")
        elif "param" in sl:
            p = params.get(sl["param"])
            if p is None or p["type"] != "choice":
                raise ValueError(f"{where}: param {sl['param']!r} must be a choice parameter")
            if [v["id"] for v in sl["variants"]] != p["choices"]:
                raise ValueError(f"{where}: variants {[v['id'] for v in sl['variants']]} must match the choices "
                                 f"of {sl['param']} {p['choices']}")
        else:
            raise ValueError(f"{where} needs either file or param")


def _check_estimate(meta_path: Path, meta: dict) -> None:
    """How a training-session scenario's cost is estimated from live runs (srl/estimate.py).

    ``steps`` names the int parameter that counts training steps; ``unbilled_params`` lists
    parameters that only change operations the service does not bill (saves), so a live run
    that set them differently still measures the same tokens. MTRL scenarios have none.
    """
    est = meta.get("estimate")
    if meta.get("engine", "training-session") == "mtrl":
        if est is not None:
            raise ValueError(f"{meta_path}: estimate applies to training-session scenarios only")
        return
    if not isinstance(est, dict) or "steps" not in est:
        raise ValueError(f"{meta_path}: training-session scenarios need estimate.steps (see srl/estimate.py)")
    params = {p["key"]: p for p in meta.get("params", [])}
    if params.get(est["steps"], {}).get("type") != "int":
        raise ValueError(f"{meta_path}: estimate.steps {est['steps']!r} must name an int parameter")
    unknown = sorted(set(est.get("unbilled_params", [])) - set(params))
    if unknown:
        raise ValueError(f"{meta_path}: estimate.unbilled_params names unknown parameters {', '.join(unknown)}")


def _check_sandbox(meta_path: Path, meta: dict) -> None:
    """`sandbox` (what the scenario runs there) and loop nodes with `where: "agentcore"` go together."""
    nodes = meta.get("loop", {}).get("nodes", [])
    placed = {n.get("where") for n in nodes} - {None}
    if placed - {"agentcore"}:
        raise ValueError(f"{meta_path}: loop node 'where' must be 'agentcore', got {sorted(placed - {'agentcore'})}")
    sb = meta.get("sandbox")
    if sb is None:
        if placed:
            raise ValueError(f"{meta_path}: a loop node runs in the sandbox (where: agentcore) but there is no `sandbox`")
        return
    if meta.get("engine", "training-session") != "training-session":
        raise ValueError(f"{meta_path}: `sandbox` applies to training-session scenarios (an MTRL agent is its own runtime)")
    if not isinstance(sb, dict) or not isinstance(sb.get("executes"), str) or set(sb) != {"executes"}:
        raise ValueError(f"{meta_path}: `sandbox` must be {{\"executes\": \"what runs there\"}}")
    if not placed:
        raise ValueError(f"{meta_path}: `sandbox` is set but no loop node has where: agentcore")
    for n in nodes:
        if n.get("where") and n["lane"] != "client":
            raise ValueError(f"{meta_path}: node {n['op']} runs in the sandbox, so its lane is client (your code)")


def _check_probe_example(meta_path: Path, meta: dict) -> None:
    """`probe_example`: a question the Ask box suggests, in the training data's language (model input, not UI text)."""
    if "probe_example" not in meta:
        return
    if not isinstance(meta["probe_example"], str) or not meta["probe_example"].strip():
        raise ValueError(f"{meta_path}: probe_example must be a nonempty string")
    if meta.get("probes", True) is False or meta.get("engine", "training-session") != "training-session":
        raise ValueError(f"{meta_path}: probe_example needs a training-session scenario that answers questions "
                         "(probes is not false)")


def _check_recordings(recordings: Path, scenario_id: str) -> None:
    """A card's shipped recordings are recordings of that card, so they leave the release with it."""
    if not recordings.is_dir():
        return
    for d in sorted(recordings.iterdir()):
        info = d / "recording.json"
        if not info.is_file():
            raise ValueError(f"{d}: not a recording (no recording.json); only scripts/ship_recording.py writes here")
        recorded = json.loads(info.read_text(encoding="utf-8")).get("scenario")
        if recorded != scenario_id:
            raise ValueError(f"{info}: a recording of {recorded!r} is in the card {scenario_id!r}; "
                             f"it belongs in templates/{recorded}/{RECORDINGS}/")


def _check_code(slot: dict, code) -> str:
    """Code edited in the browser: it must parse and define the slot's function (or name)."""
    name = slot["file"]
    if not isinstance(code, str) or not code.strip():
        raise ParamError(f"{name}: the edited code is empty")
    if len(code.encode("utf-8")) > MAX_CUSTOM_BYTES:
        raise ParamError(f"{name}: {len(code.encode('utf-8')):,} bytes is over the {MAX_CUSTOM_BYTES:,}-byte limit")
    try:
        tree = ast.parse(code, filename=name)
    except SyntaxError as e:
        raise ParamError(f"{name} line {e.lineno}: {e.msg}. Nothing was started.") from None
    defined = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            defined |= {t.id for t in targets if isinstance(t, ast.Name)}
    if slot["function"] not in defined:
        raise ParamError(f"{name} must define {slot['contract']}; the edited code defines "
                         f"{', '.join(sorted(defined)) or 'nothing at the top level'}")
    return code
