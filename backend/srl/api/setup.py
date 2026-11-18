"""Setup page: the region, the readiness checks and their fixes, the MTRL and environment sandbox selections,
and deploying their runtimes."""

from __future__ import annotations

from fastapi import APIRouter, Body

from ..aws import context as aws_context
from ..aws import deploy as aws_deploy
from ..aws import mtrl as aws_mtrl
from ..aws import readiness as aws_readiness
from ..aws import sandbox as aws_sandbox
from ..i18n import localize_checks, pick
from .deps import Ctx, Identity, Lang, State

router = APIRouter(tags=["setup"])


@router.get("/api/context")
def get_context(ctx: Ctx):
    return ctx.public()


@router.put("/api/context")
def put_context(state: State, region: str = Body(..., embed=True)):
    aws_context.set_region(region)
    state.account.forget_readiness()
    return aws_context.resolve(refresh=True).public()


@router.get("/api/readiness")
def get_readiness(state: State, lang: Lang):
    return localize_checks(state.account.readiness(), lang)


@router.post("/api/readiness/run")
def post_readiness(state: State, lang: Lang):
    return localize_checks(state.account.refresh_readiness(), lang)


@router.post("/api/readiness/fix/{fix_id}")
def post_fix(fix_id: str, state: State, lang: Lang, ctx: Ctx):
    message = aws_readiness.fix(fix_id, ctx)
    return {"message": pick(message, lang), "readiness": localize_checks(state.account.refresh_readiness(), lang)}


@router.get("/api/mtrl/options")
def mtrl_options(ctx: Identity):
    return {**aws_mtrl.options(ctx), "selection": aws_mtrl.selection(ctx)}


@router.put("/api/mtrl/selection")
def mtrl_selection(state: State, lang: Lang, ctx: Ctx, runtime_arn: str | None = Body(None),
                   mlflow_app_arn: str | None = Body(None), job_role_name: str | None = Body(None)):
    selected = aws_mtrl.set_selection(ctx, runtime_arn=runtime_arn, mlflow_app_arn=mlflow_app_arn,
                                      job_role_name=job_role_name)
    return {"selection": selected, "readiness": localize_checks(state.account.refresh_readiness(), lang)}


@router.get("/api/sandbox/options")
def sandbox_options(ctx: Identity):
    return aws_sandbox.options(ctx)


@router.put("/api/sandbox/selection")
def sandbox_selection(state: State, lang: Lang, ctx: Identity, runtime_arn: str | None = Body(None, embed=True)):
    """Select the environment sandbox runtime for this region; null goes back to the one with the default name."""
    selected = aws_sandbox.set_selection(ctx, runtime_arn)
    return {"selection": selected, "readiness": localize_checks(state.account.refresh_readiness(), lang)}


@router.get("/api/deploy")
def deploy_targets(state: State, ctx: Ctx):
    """The runtimes the Setup page deploys (the sandbox, each MTRL scenario's agent): names, versions, latest job."""
    return {"targets": aws_deploy.overview(ctx, state.deploys, state.runs.alive_live_configs())}


@router.post("/api/deploy/{target_id}")
def deploy_start(target_id: str, state: State, ctx: Identity, action: str = Body("deploy", embed=True)):
    """Start a deploy (build the image in CodeBuild, create or update the runtime) or a delete; poll GET /api/deploy."""
    t = aws_deploy.target(target_id)
    found = aws_deploy.find_runtime(ctx, t.runtime_name)
    blocking = aws_deploy.blocking_runs(t, state.runs.alive_live_configs(), found["agentRuntimeArn"] if found else None)
    return state.deploys.start(t, action, ctx, blocking, after=state.account.forget_readiness)


@router.get("/api/deploy/{target_id}/log")
def deploy_log(target_id: str, state: State, tail: int = 400):
    aws_deploy.target(target_id)
    return {"lines": state.deploys.log(target_id, tail)}
