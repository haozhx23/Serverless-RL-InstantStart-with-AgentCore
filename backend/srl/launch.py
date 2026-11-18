"""Launching a run: what a live run must pass before it starts, and the AWS settings it gets.

POST /api/runs is where this tool starts spending money in the account, so every reason to
refuse a launch is checked here, before a run directory exists. The run gets what it needs
in its config.json and never reads the app's settings again.
"""

from __future__ import annotations

from .aws import mlflow as aws_mlflow
from .aws import mtrl as aws_mtrl
from .aws import readiness as aws_readiness
from .aws import sandbox as aws_sandbox
from .aws.context import AwsContext
from .errors import ApiError
from .state import AppState


def launch(state: AppState, ctx: AwsContext, scenario: str, backend: str, params: dict, variants: dict,
           files: dict, mlflow: bool) -> dict:
    """Checks what this backend needs, validates the parameters against the models it can use, creates the run."""
    state.runs.check_backend(backend)  # before anything is asked of AWS
    mtrl = state.registry.engine(scenario) == "mtrl"
    if mtrl:
        return _launch_mtrl(state, ctx, scenario, params, variants, files)
    models = _session_preflight(state, ctx)
    validated = state.registry.validate_params(scenario, params, models)
    aws = ctx.run_aws_config()
    if state.registry.needs_sandbox(scenario):
        aws["sandbox"] = _sandbox_preflight(ctx)
    # Only live runs publish to MLflow (srl_events.open_log checks again).
    tracking = aws_mlflow.run_config(ctx, scenario, validated["base_model"]) if mlflow else None
    mode = "off" if not mlflow else "publish" if tracking else "no-app"
    return state.runs.create(scenario, backend, validated, aws, variants, files, mlflow=tracking, mlflow_mode=mode)


def _launch_mtrl(state: AppState, ctx: AwsContext, scenario: str, params: dict, variants: dict, files: dict) -> dict:
    """The job writes its own metrics to the MLflow app it is given, so the `mlflow` switch does not apply."""
    models = _mtrl_preflight(ctx)
    validated = state.registry.validate_params(scenario, params, models)
    aws = ctx.run_aws_config()
    aws["mtrl"] = aws_mtrl.run_config(ctx)
    # Which agent the job will call, and what its AgentCore usage is priced at.
    rt = aws_sandbox.describe(ctx, aws["mtrl"]["runtime_arn"])
    aws["mtrl"].update(agent_version=rt.get("agentRuntimeVersion"), agent_platform=aws_sandbox.platform_of(rt))
    return state.runs.create(scenario, "live", validated, aws, variants, files, mlflow_mode="service")


def _require_identity(ctx: AwsContext) -> None:
    if not ctx.ready:
        raise ApiError(f"No AWS identity, so a live run cannot start: {ctx.identity_error}")


def _session_preflight(state: AppState, ctx: AwsContext) -> list[str]:
    """A live training-session run needs its role, group and bucket, and a free session slot. Returns its models."""
    _require_identity(ctx)
    blockers = [c for c in (aws_readiness.check_role(ctx), aws_readiness.check_group(ctx),
                            aws_readiness.check_bucket(ctx)) if c["status"] != "pass"]
    if blockers:
        raise ApiError("Setup is incomplete: " + "; ".join(f"{c['title']}: {c['observed']}" for c in blockers)
                       + ". Fix it on the Setup page.")
    q = aws_readiness.session_quota(ctx)
    if q["active"] is None:
        raise ApiError(f"Could not count active sessions: {q['error']}")
    if q["active"] >= q["limit"]:
        raise ApiError(f"All {q['limit']} session slots are in use ({', '.join(q['active_names'])}). "
                       "Stop one on the Sessions page, or request a higher quota.")
    return state.account.live_models(ctx)


def _sandbox_preflight(ctx: AwsContext) -> dict:
    """A live run of a sandbox scenario needs the runtime READY and running this checkout's sandbox_exec.py."""
    checks, rt, usage_group = aws_sandbox.run_checks(ctx)
    blockers = [c for c in checks if c["status"] in ("fail", "unverified")]
    if blockers or rt is None:
        raise ApiError("The environment sandbox is not ready: " + "; ".join(f"{c['title']}: {c['observed']}" for c in blockers)
                       + ". See the Environment sandbox section of the Setup page.")
    return aws_sandbox.run_config(rt, usage_group)


def _mtrl_preflight(ctx: AwsContext) -> list[str]:
    """Fresh MTRL checks before a live job: anything not passing blocks, with its observed state."""
    _require_identity(ctx)
    bucket = aws_readiness.check_bucket(ctx)
    checks, models = aws_mtrl.run_checks(ctx)
    blockers = [c for c in [bucket, *checks] if c["status"] != "pass"]
    if blockers:
        raise ApiError("MTRL is not ready: " + "; ".join(f"{c['title']}: {c['observed']}" for c in blockers)
                       + ". See the MTRL section of the Setup page.")
    return models
