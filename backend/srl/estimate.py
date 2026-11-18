"""What a training-session run will cost, from what live runs of the same configuration cost.

Parameters alone cannot say what a run will cost: how long the model's completions are, and in
RL how many groups carry a gradient, are only known once the model has run. So the estimate
comes from completed live runs whose scenario, model, parameters and code slots all match,
priced at list price. Several matching runs give a range, not an average.

For N planned training steps and runs that measured n:

    measured  N == n   what those runs cost
    subset    N <  n   their tokens outside the steps (evaluations, reference passes) plus
                       their first N steps
    constant  N >  n   every measured step used exactly the same tokens (SFT and DPO train
                       on the same data each step), so N steps cost N of them
    partial   N >  n   the steps differ (RL: as the model gets more answers right, fewer
                       groups carry a gradient and completions change; in a measured RL run
                       a step cost half as much at the end as at the start). Only the measured n
                       steps are priced; the other N - n are not estimated, and the range
                       of one measured step is given beside it, not added to it. In back-tests,
                       averaging the first steps overestimated the rest by 5-29%
                       (docs/architecture.md 3.9).

Only the number of steps, and parameters the scenario declares as changing unbilled
operations only (``estimate.unbilled_params`` in scenario.json, e.g. save_checkpoint), may
differ. Anything else that differs, code edited in the browser, or no matching run means no
estimate ("none"), with the reason and the closest live runs instead. Questions asked during
a run (role="probe") are left out: they are extra calls, not part of the loop.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from . import pricing
from .events import EventLogError, read_events, summarize

KINDS = ("prefill", "sample", "train")


def _zero() -> dict:
    return {k: 0 for k in KINDS}


def _add(a: dict, b: dict, times: float = 1.0) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) * times for k in KINDS}


class CostBasis:
    """Completed live training-session runs, read from run directories and recordings.

    A recording is a copy of a run directory, so the same run can appear more than once; it
    is counted once. Each directory is parsed again only when its events.jsonl changes.
    """

    def __init__(self, sources: Callable[[], Iterable[Path]]):
        self._sources = sources  # the directories whose subdirectories are runs or recordings
        self._cache: dict[Path, tuple[tuple, dict | None]] = {}

    def runs(self, scenario: str) -> list[dict]:
        seen, out = set(), []
        for root in self._sources():
            if not root.is_dir():
                continue
            for d in sorted(root.iterdir()):
                run = self._load(d)
                if run and run["scenario"] == scenario and run["run"] not in seen:
                    seen.add(run["run"])
                    out.append(run)
        return out

    def _load(self, d: Path) -> dict | None:
        events_path, config_path = d / "events.jsonl", d / "config.json"
        if not events_path.is_file() or not config_path.is_file():
            return None
        st = events_path.stat()
        key = (st.st_mtime_ns, st.st_size)
        cached = self._cache.get(d)
        if cached and cached[0] == key:
            return cached[1]
        run = _parse(config_path, events_path)
        self._cache[d] = (key, run)
        return run


def _parse(config_path: Path, events_path: Path) -> dict | None:
    """One completed live run's tokens, split into outside the steps and per step; None for any other run."""
    try:
        events = read_events(events_path)
    except EventLogError:
        return None
    if not events or events[0]["type"] != "run_started" or events[0].get("backend") != "live":
        return None
    s = summarize(events)
    if s["status"] != "completed" or s["engine"] != "training-session":
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    outside, per_step = _zero(), {}
    for e in events:
        if e["type"] != "span" or not e.get("tokens") or e.get("role") == "probe":
            continue
        if e.get("step") is None:
            outside = _add(outside, e["tokens"])
        else:
            per_step[e["step"]] = _add(per_step.get(e["step"], _zero()), e["tokens"])
    started = events[0]
    return {"run": started["run"], "scenario": started["scenario"], "model": started.get("model"),
            "ts": started["ts"], "params": config.get("params", {}), "variants": config.get("variants", {}),
            "custom": bool(config.get("custom_files")), "outside": outside,
            "per_step": [per_step.get(i, _zero()) for i in range(max(per_step) + 1)] if per_step else [],
            "eval": s["eval"]}


def _tokens_for(run: dict, steps: int) -> dict:
    """Tokens outside the steps plus the first `steps` steps; beyond the measured ones, repeats of a step.

    Only called with more steps than measured when every measured step used the same tokens.
    """
    measured = run["per_step"]
    tokens = run["outside"]
    for t in measured[:steps]:
        tokens = _add(tokens, t)
    if steps > len(measured):
        tokens = _add(tokens, measured[-1], steps - len(measured))
    return {k: round(v) for k, v in tokens.items()}


def _constant(run: dict) -> bool:
    return all(t == run["per_step"][0] for t in run["per_step"])


def _differences(meta: dict, run: dict, params: dict, file_variants: dict) -> list[dict]:
    """What a live run set differently from the planned run, apart from steps and unbilled-only parameters."""
    est = meta["estimate"]
    skip = {est["steps"], *est.get("unbilled_params", [])}
    out = [{"key": k, "measured": run["params"].get(k), "planned": v}
           for k, v in params.items() if k not in skip and run["params"].get(k) != v]
    out += [{"key": k, "measured": run["variants"].get(k), "planned": v}
            for k, v in file_variants.items() if run["variants"].get(k) != v]
    return out


def _as_run(run: dict) -> dict:
    """A live run as it ran: its date, steps, list-price cost and held-out result."""
    return {"run": run["run"], "date": datetime.fromtimestamp(run["ts"], tz=timezone.utc).strftime("%Y-%m-%d"),
            "steps": len(run["per_step"]), "model": run["model"],
            "usd": pricing.estimate(run["model"], _tokens_for(run, len(run["per_step"]))), "eval": run["eval"]}


def estimate(meta: dict, basis: CostBasis, params: dict, file_variants: dict, custom: bool) -> dict:
    """The cost of a run of `meta` with validated `params`, from matching live runs (see the module docstring).

    ``file_variants`` is the chosen variant of each code slot; ``custom`` says whether any
    slot's code was edited. ``status`` is one of the cases in the module docstring; for
    "partial", ``usd`` covers the ``measured_steps`` only and ``unestimated_steps`` are left
    out, with ``step_usd`` the range of one measured step. For "none", ``reason`` says why and
    ``nearest`` lists the closest live runs.
    """
    out = {"status": "none", "usd": None, "tokens": None, "steps": None, "measured_steps": None, "basis": [],
           "unestimated_steps": 0, "step_usd": None, "reason": None, "nearest": [], "price_as_of": pricing.PRICES_AS_OF,
           # A sandbox scenario also pays AgentCore Runtime, which the token count says nothing about. It is
           # measured per run (GET /api/runs/{id}/agentcore), not estimated here: "usd" is training tokens only.
           "agentcore": "not_estimated" if "sandbox" in meta else None}
    if meta.get("engine", "training-session") == "mtrl":
        # Pre-run estimates currently cover training-session loops only; MTRL reports usage during the job.
        return {**out, "reason": {"code": "mtrl"}}
    est = meta.get("estimate")
    if not est:
        return {**out, "reason": {"code": "not_described"}}
    model, steps = params["base_model"], params[est["steps"]]
    out["steps"] = steps
    if pricing.price_for(model) is None:
        return {**out, "reason": {"code": "no_price", "model": model}}
    runs = [r for r in basis.runs(meta["id"]) if r["per_step"]]
    if not runs:
        return {**out, "reason": {"code": "no_live_runs"}}
    if custom:
        return {**out, "reason": {"code": "custom_code"}, "nearest": _nearest(meta, runs, params, file_variants)}
    same_model = [r for r in runs if r["model"] == model and not r["custom"]]
    if not same_model:
        return {**out, "reason": {"code": "model", "measured_models": sorted({r["model"] for r in runs})},
                "nearest": _nearest(meta, runs, params, file_variants)}
    matching = [r for r in same_model if not _differences(meta, r, params, file_variants)]
    if not matching:
        return {**out, "reason": {"code": "differs"}, "nearest": _nearest(meta, same_model, params, file_variants)}
    exact = [r for r in matching if len(r["per_step"]) == steps]
    longer = [r for r in matching if len(r["per_step"]) > steps]
    if exact or longer:
        chosen, status, priced_steps = exact or longer, "measured" if exact else "subset", steps
    else:
        most = max(len(r["per_step"]) for r in matching)
        chosen = [r for r in matching if len(r["per_step"]) == most]
        status = "constant" if all(_constant(r) for r in chosen) else "partial"
        priced_steps = steps if status == "constant" else most
    tokens = [_tokens_for(r, priced_steps) for r in chosen]
    usd = [pricing.estimate(model, t) for t in tokens]
    out.update(status=status, usd={"low": min(usd), "high": max(usd)},
               tokens={"low": {k: min(t[k] for t in tokens) for k in KINDS},
                       "high": {k: max(t[k] for t in tokens) for k in KINDS}},
               measured_steps=len(chosen[0]["per_step"]),
               basis=[_as_run(r) for r in sorted(chosen, key=lambda r: r["ts"])])
    if status == "partial":
        step_usd = [pricing.estimate(model, t) for r in chosen for t in r["per_step"]]
        out.update(unestimated_steps=steps - priced_steps, step_usd={"low": min(step_usd), "high": max(step_usd)})
    return out


def _nearest(meta: dict, runs: list[dict], params: dict, file_variants: dict, limit: int = 3) -> list[dict]:
    """The live runs that differ in the fewest settings, what differs, and what each cost as it ran."""
    ranked = sorted(runs, key=lambda r: (r["custom"], len(_differences(meta, r, params, file_variants)), -r["ts"]))
    return [{**_as_run(r), "custom": r["custom"], "differs": _differences(meta, r, params, file_variants)}
            for r in ranked[:limit]]


def for_launch(registry, basis: CostBasis, scenario_id: str, params: dict, variants: dict, custom: bool,
               models: list[str]) -> dict:
    """The estimate for a run about to be launched, from the launch form's values (defaults filled in)."""
    meta = registry.get(scenario_id)
    if meta.get("engine", "training-session") == "mtrl":
        return estimate(meta, basis, {}, {}, False)
    validated = registry.validate_params(scenario_id, params, models)
    chosen = registry.resolve_slots(scenario_id, variants, {})["chosen"]  # code edits only matter as `custom`
    return estimate(meta, basis, validated, chosen, custom)


def for_run(registry, basis: CostBasis, config: dict) -> dict:
    """The estimate for an existing run's configuration: what a live run with its settings would cost."""
    meta = registry.get(config["scenario"])
    if meta.get("engine", "training-session") == "mtrl":
        return estimate(meta, basis, {}, {}, False)
    file_slots = {sl["key"] for sl in meta.get("slots", []) if "file" in sl}
    variants = {k: v for k, v in config.get("variants", {}).items() if k in file_slots}
    return estimate(meta, basis, config["params"], variants, bool(config.get("custom_files")))
