"""Opening MLflow: each route signs the browser in to an app (302 to a presigned URL), see docs/architecture.md 3.8."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import RedirectResponse

from ..aws import mlflow as aws_mlflow
from ..errors import ApiError, Invalid, NotFound
from ..runs import RunManager
from .deps import Identity, State

router = APIRouter(tags=["mlflow"])


def _published(runs: RunManager, run_id: str) -> dict:
    where = runs.summary(run_id).get("mlflow")
    if not where or not where.get("run_id"):
        hint = " (a live run does when an MLflow app is selected on the Setup page)"
        raise NotFound(f"{run_id} did not publish to MLflow" + (f": {where['error']}" if where and where.get("error") else hint))
    return where


@router.get("/api/runs/{run_id}/mlflow")
def open_run_in_mlflow(run_id: str, state: State, view: str | None = None):
    """Shows the run the app published, one of its tabs, or its experiment."""
    where = _published(state.runs, run_id)
    return RedirectResponse(aws_mlflow.open_url(where["tracking_uri"], where["experiment_id"], where["run_id"], view), 302)


@router.get("/api/mlflow/compare")
def compare_in_mlflow(state: State, runs_: str = Query(..., alias="runs")):
    """MLflow's compare view for runs of this app (comma-separated ids) that published to the same MLflow app."""
    ids = [r for r in runs_.split(",") if r]
    if len(ids) < 2:
        raise Invalid(f"comparing needs at least two runs, got {len(ids)}")
    where = {r: _published(state.runs, r) for r in ids}
    apps = {w["tracking_uri"] for w in where.values()}
    if len(apps) > 1:
        raise ApiError("these runs published to different MLflow apps (" + ", ".join(sorted(apps))
                       + "); MLflow compares runs within one app")
    return RedirectResponse(aws_mlflow.compare_url(apps.pop(), [(w["experiment_id"], w["run_id"]) for w in where.values()]), 302)


@router.get("/api/mlflow/open")
def open_mlflow(ctx: Identity):
    """The MLflow app selected for this region."""
    arn = aws_mlflow.app_arn(ctx)
    if not arn:
        raise NotFound(f"No MLflow app is selected for {ctx.region}; pick one on the Setup page")
    return RedirectResponse(aws_mlflow.open_url(arn), 302)
