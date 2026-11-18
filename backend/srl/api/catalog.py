"""What can be run and what it costs: the scenarios and their template files, the translations
of sentences runs write into events, the token prices."""

from __future__ import annotations

from fastapi import APIRouter, Body

from .. import estimate, pricing
from .deps import Ctx, Lang, State

router = APIRouter(tags=["catalog"])


@router.get("/api/scenarios")
def get_scenarios(state: State, lang: Lang, ctx: Ctx):
    recs = state.runs.recordings()
    scenarios = [{**meta, "recordings": [r for r in recs if r["scenario"] == meta["id"]]}
                 for meta in state.registry.all(lang)]
    models = {"live": state.account.models(ctx.region), "mtrl_live": state.account.mtrl_models(ctx.region)}
    return {"scenarios": scenarios, "models": models}


@router.get("/api/scenarios/{scenario_id}/files/{name:path}")
def get_scenario_file(scenario_id: str, name: str, state: State):
    return {"name": name, "content": state.registry.source(scenario_id, name)}


@router.get("/api/i18n/events")
def get_event_text(state: State):
    """Sentences runs write into events, translated: the UI looks each one up as it renders."""
    return {"zh": state.registry.events("zh")}


@router.get("/api/pricing")
def get_pricing(lang: Lang):
    return pricing.table(lang)


@router.post("/api/estimate")
def post_estimate(state: State, scenario: str = Body(...), params: dict = Body(default_factory=dict),
                  variants: dict = Body(default_factory=dict), files: dict = Body(default_factory=dict)):
    """What a launch with these settings would cost live, from matching live runs (srl/estimate.py)."""
    known = set(pricing.HUB_NAME)
    models = sorted(known)
    return estimate.for_launch(state.registry, state.basis, scenario, params, variants, bool(files), models)
