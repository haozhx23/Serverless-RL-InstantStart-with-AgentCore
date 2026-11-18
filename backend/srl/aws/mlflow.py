"""Experiment tracking: the SageMaker AI MLflow App that live runs publish to.

One app per region, picked on the Setup page (stored with the MTRL selections, which have
always held it). MTRL jobs log to it from the service side (the trainer is given the app);
training-session runs have no MLflow integration in the SDK (sagemaker-train 1.27.0), so the
run process publishes its own events there (templates/_shared/srl_mlflow.py), with the
credentials the run has. This module names the destination, checks that it exists, and opens
it in the browser through a presigned URL, which the run never needs.
"""

from __future__ import annotations

import json
import time
from urllib.parse import quote

import boto3

from .. import pricing, settings
from ..errors import Invalid
from ..i18n import L, check_texts
from .context import AwsContext, client

USABLE = ("Created", "Updated")   # DescribeMlflowApp Status values an app takes runs in
TRACES_PER_STEP = 4                # rollouts per step published as traces, one per group
RUN_TABS = ("overview", "model-metrics", "traces", "artifacts")  # the run page's tabs in MLflow 3.16
VIEWS = (*RUN_TABS, "experiment")  # experiment: the runs list of the run's experiment, to compare it with the others
# Why a run did or did not publish, kept in run.json so the console can say it (docs/architecture.md 3.8).
MODES = ("publish", "off", "no-app", "replay", "service")


def app_arn(ctx: AwsContext) -> str | None:
    from .mtrl import selection
    return selection(ctx)["mlflow_app_arn"]


def experiment_name(scenario: str) -> str:
    return f"{settings.S3_PREFIX}/{scenario}"


def run_config(ctx: AwsContext, scenario: str, model: str) -> dict | None:
    """The mlflow block of a live training-session run's config.json, or None when no app is selected.

    prices (USD per million tokens, list price as of pricing.PRICES_AS_OF) let the run publish an
    estimated cost next to its metrics; they are left out for a model with no known price.
    """
    arn = app_arn(ctx)
    if not arn:
        return None
    out = {"tracking_uri": arn, "experiment": experiment_name(scenario), "traces_per_step": TRACES_PER_STEP}
    price = pricing.price_for(model)
    if price:
        out["prices"] = {**price, "as_of": pricing.PRICES_AS_OF}
    return out


def check(ctx: AwsContext) -> tuple[dict, dict | None]:
    """The Setup check, and the selected app as the launch drawer shows it (None when none is selected).

    Never blocks a run: without an app, a run still logs everything to events.jsonl.
    """
    title = L("MLflow app for experiment tracking", "用于实验追踪的 MLflow app")
    why = L("Live training-session runs publish their parameters, per-step metrics, model packages and sample rollouts "
            "to this app, next to your team's other experiments; MTRL jobs log to the same app. Without one, runs are "
            "only in this app's run list.",
            "training session 的实时运行会将参数、每步指标、模型包和选取的 rollout 发布到此 app，与团队的其他实验集中管理。MTRL 任务也写入同一个 app。未选择 app 时，运行记录仅显示在本应用的运行列表中。")

    def result(status, observed, app=None):
        return ({"id": "mlflow", "group": "tracking", "status": status, "observed": observed,
                 **check_texts(title, why, None), "checked_at": time.time()}, app)

    if not ctx.ready:
        return result("unverified", "needs AWS credentials")
    arn = app_arn(ctx)
    if not arn:
        return result("warn", "no MLflow app selected; live runs are not published to MLflow. Pick one above")
    try:
        app = client("sagemaker", ctx).describe_mlflow_app(Arn=arn)
    except Exception as e:
        return result("warn", f"{arn}: {type(e).__name__}: {e}",
                      {"arn": arn, "name": None, "status": None, "usable": False})
    info = {"arn": arn, "name": app.get("Name"), "status": app.get("Status"), "usable": app.get("Status") in USABLE}
    observed = f"{app.get('Name')} ({app.get('Status')}, MLflow {app.get('MlflowVersion')}), {arn}"
    if not info["usable"]:
        return result("warn", f"{observed}; runs can be published once it is {' or '.join(USABLE)}", info)
    return result("pass", observed, info)


# The app's UI (MLflow 3.16) routes with the URL fragment, which survives the sign-in redirect. Without
# workspace it lands on the workspace picker at #/, and without workflowType it opens in GenAI mode;
# runs logged without a workspace are in "default" (checked in a browser, 2026-10-29).
UI_QUERY = "workspace=default&workflowType=machine_learning"


def _signed(arn: str) -> str:
    region = arn.split(":")[3]
    return boto3.client("sagemaker", region_name=region).create_presigned_mlflow_app_url(Arn=arn)["AuthorizedUrl"]


def open_url(arn: str, experiment_id: str | None = None, run_id: str | None = None, view: str | None = None) -> str:
    """A presigned URL that signs the browser in to the app, deep-linked to an experiment's runs, one run or a run's tab."""
    if view is not None and view not in VIEWS:
        raise Invalid(f"view must be one of {', '.join(VIEWS)}, got {view!r}")
    url = _signed(arn)
    if experiment_id and run_id and view != "experiment":
        return f"{url}#/experiments/{experiment_id}/runs/{run_id}{'/' + view if view else ''}?{UI_QUERY}"
    if experiment_id:
        return f"{url}#/experiments/{experiment_id}/runs?{UI_QUERY}"
    return f"{url}#/experiments?{UI_QUERY}"


def compare_url(arn: str, runs: list[tuple[str, str]]) -> str:
    """MLflow's own compare view for (experiment_id, run_id) pairs: parameters side by side, metrics overlaid.

    The query is the one MLflow's UI builds itself (getCompareRunPageRoute): JSON lists of run and experiment ids.
    """
    run_ids = json.dumps([r for _, r in runs])
    experiment_ids = json.dumps(sorted({e for e, _ in runs}))
    return f"{_signed(arn)}#/compare-runs?runs={quote(run_ids)}&experiments={quote(experiment_ids)}&{UI_QUERY}"
