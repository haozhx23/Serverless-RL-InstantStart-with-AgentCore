"""Training sessions in the region: list with token usage, stop, read billing counters."""

from __future__ import annotations

from botocore.exceptions import ClientError

from .. import pricing, settings
from ..errors import Invalid
from .context import AwsContext, client
from .readiness import ACTIVE_SESSION_STATES

DESCRIBE_LIMIT = 30


def _usage(detail: dict) -> dict:
    u = detail.get("BillableTokenUsage") or {}
    return {"prefill": u.get("PrefillTokenCount", 0), "sample": u.get("SampleTokenCount", 0),
            "train": u.get("TrainingTokenCount", 0)}


def _hub_name(detail: dict) -> str | None:
    arn = (detail.get("ModelConfig") or {}).get("BaseModelArn") or ""
    parts = arn.split("/")
    return parts[-2] if len(parts) >= 2 else None


def describe(ctx: AwsContext, name: str) -> dict:
    d = client("sagemaker", ctx).describe_training_session(TrainingSessionName=name)["TrainingSessionDetail"]
    tokens, hub = _usage(d), _hub_name(d)
    return {"name": name, "status": d.get("TrainingSessionStatus"), "hub_model": hub, "tokens": tokens,
            "cost_usd": pricing.estimate(hub, tokens) if hub else None,
            "created": d.get("CreationTime"), "failure_reason": d.get("FailureReason"),
            "ours": name.startswith(settings.SESSION_PREFIX + "-"),
            "model_package_group_arn": d.get("ModelPackageGroupArn")}


def list_sessions(ctx: AwsContext, scope: str) -> dict:
    if scope not in ("ours", "all"):
        raise Invalid("scope must be 'ours' or 'all'")
    sm = client("sagemaker", ctx)
    summaries, token = [], None
    while True:
        kw = {"MaxResults": 100, **({"NextToken": token} if token else {})}
        if scope == "ours":
            kw["NameContains"] = settings.SESSION_PREFIX
        r = sm.list_training_sessions(**kw)
        summaries += r.get("TrainingSessionSummaries", [])
        token = r.get("NextToken")
        if not token:
            break
    summaries.sort(key=lambda s: s.get("CreationTime") or 0, reverse=True)
    rows, errors = [], []
    for s in summaries:
        name = s["TrainingSessionArn"].rsplit("/", 1)[-1]
        if scope == "ours" and not name.startswith(settings.SESSION_PREFIX + "-"):
            continue
        if len(rows) < DESCRIBE_LIMIT:
            try:
                rows.append(describe(ctx, name))
                continue
            except ClientError as e:
                errors.append(f"{name}: {e.response['Error'].get('Message')}")
        rows.append({"name": name, "status": s.get("TrainingSessionStatus"), "created": s.get("CreationTime"),
                     "tokens": None, "cost_usd": None, "hub_model": None,
                     "ours": name.startswith(settings.SESSION_PREFIX + "-")})
    active = [r for r in rows if r["status"] in ACTIVE_SESSION_STATES]
    return {"sessions": rows, "active": len(active), "errors": errors, "describe_limit": DESCRIBE_LIMIT}


def stop(ctx: AwsContext, name: str) -> dict:
    client("sagemaker", ctx).stop_training_session(TrainingSessionName=name)
    return describe(ctx, name)


def stop_all_ours(ctx: AwsContext) -> list[dict]:
    out = []
    for s in list_sessions(ctx, "ours")["sessions"]:
        if s["status"] in ACTIVE_SESSION_STATES and s["status"] != "Stopping":
            try:
                client("sagemaker", ctx).stop_training_session(TrainingSessionName=s["name"])
                out.append({"name": s["name"], "stopped": True, "error": None})
            except ClientError as e:
                out.append({"name": s["name"], "stopped": False, "error": e.response["Error"].get("Message")})
    return out
