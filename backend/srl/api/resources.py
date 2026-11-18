"""Sessions and Artifacts pages: what runs leave in the account. Training sessions and their
quota, MTRL jobs and their quota, AgentCore Runtime sessions (the environment sandbox), the
model packages runs saved."""

from __future__ import annotations

from fastapi import APIRouter, Body

from ..aws import artifacts as aws_artifacts
from ..aws import mtrl as aws_mtrl
from ..aws import readiness as aws_readiness
from ..aws import sandbox as aws_sandbox
from ..aws import sessions as aws_sessions
from .deps import Ctx, Identity

router = APIRouter(tags=["resources"])


@router.get("/api/quota")
def get_quota(ctx: Ctx):
    return aws_readiness.session_quota(ctx)


@router.get("/api/sessions")
def get_sessions(ctx: Identity, scope: str = "ours"):
    return aws_sessions.list_sessions(ctx, scope)


@router.post("/api/sessions/{name}/stop")
def stop_session(name: str, ctx: Ctx):
    return aws_sessions.stop(ctx, name)


@router.post("/api/sessions/stop-all")
def stop_all(ctx: Ctx):
    return {"results": aws_sessions.stop_all_ours(ctx)}


@router.get("/api/mtrl/quota")
def mtrl_quota(ctx: Ctx):
    return aws_mtrl.job_quota(ctx)


@router.get("/api/mtrl/jobs")
def mtrl_jobs(ctx: Identity):
    return {"jobs": aws_mtrl.list_jobs(ctx)}


@router.post("/api/mtrl/jobs/{name}/stop")
def mtrl_stop(name: str, ctx: Ctx):
    aws_mtrl.stop_job(ctx, name)
    return {"stopping": name}


@router.get("/api/sandbox/active")
def sandbox_active(ctx: Ctx):
    return aws_sandbox.active_sessions(ctx)


@router.get("/api/artifacts")
def get_artifacts(ctx: Identity):
    return aws_artifacts.list_packages(ctx)


@router.post("/api/artifacts/delete")
def delete_artifact(ctx: Ctx, arn: str = Body(..., embed=True)):
    return aws_artifacts.delete(ctx, arn)
