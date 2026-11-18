"""The environment sandbox, as the backend sees it: the AgentCore Runtime that sandbox scenarios run code in.

A training-session scenario whose scenario.json has ``sandbox`` sends model-generated code to this
runtime instead of running it in the loop's process (docs/architecture.md 3.10). The runtime is
deployed from the host by scripts/deploy_sandbox.py, because building its ARM64 image needs docker,
which this app's container does not have. Here it is only selected, checked and read:

- the selection is per region in data/settings.json; with none, the runtime with the default name
  (settings.SANDBOX_RUNTIME) is used when it exists;
- the checks are read-only and never open a session (a session bills);
- a run's AgentCore usage comes from the runtime's USAGE_LOGS (per session, per second), matched
  to the run by the run id inside every session id, priced at list price. Records arrived about
  20 minutes after the sessions ended on 2026-11-10; the AWS docs say up to 60.

MTRL agents run in AgentCore Runtime too; ``runtime_usage`` reads a runtime's usage metrics over a
time window for them (their session ids are the job's, so they cannot be matched to a run).
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

from botocore.exceptions import ClientError

from .. import pricing, settings
from ..errors import ApiError, Invalid
from ..i18n import L, check_texts
from .context import AwsContext, client, load_settings, save_settings

READY = "READY"
LONG_IDLE_S = 300
DEPLOY_CMD = "backend/.venv/bin/python scripts/deploy_sandbox.py"
# Active sessions per account (2026-06 release notes): 5,000 in us-east-1 and us-west-2, 2,500 elsewhere.
ACTIVE_SESSION_LIMIT = {"us-east-1": 5000, "us-west-2": 5000}
ACTIVE_SESSION_LIMIT_OTHER = 2500


def local_code_sha() -> str:
    """The version of templates/_shared/sandbox_exec.py in this checkout, computed as the file computes it."""
    return hashlib.sha256((settings.TEMPLATES_DIR / "_shared" / "sandbox_exec.py").read_bytes()).hexdigest()[:12]


def _err(e: Exception) -> str:
    if isinstance(e, ClientError):
        return f"{e.response['Error'].get('Code')}: {e.response['Error'].get('Message')}"
    return f"{type(e).__name__}: {e}"


def _check(id, title, status, observed, why, fix=None):
    """One sandbox check. title, why and the fix's label and detail may be L(en, zh) pairs; observed is never translated."""
    return {"id": id, "group": "sandbox", "status": status, "observed": observed, **check_texts(title, why, fix),
            "checked_at": time.time()}


# -- selection -----------------------------------------------------------------------------

def _runtimes(ctx: AwsContext) -> list[dict]:
    out, token = [], None
    ctl = client("bedrock-agentcore-control", ctx)
    while True:
        r = ctl.list_agent_runtimes(**({"nextToken": token} if token else {}))
        out += r.get("agentRuntimes", [])
        token = r.get("nextToken")
        if not token:
            return out


def selection(ctx: AwsContext, runtimes: list[dict] | None = None) -> dict:
    """{"runtime_arn", "source"}: the selected runtime, else the one with the default name, else none."""
    chosen = load_settings().get("sandbox", {}).get(ctx.region, {}).get("runtime_arn")
    if chosen:
        return {"runtime_arn": chosen, "source": "selected", "default_name": settings.SANDBOX_RUNTIME}
    if runtimes is None:
        runtimes = _runtimes(ctx)
    named = next((rt for rt in runtimes if rt["agentRuntimeName"] == settings.SANDBOX_RUNTIME), None)
    return {"runtime_arn": named["agentRuntimeArn"] if named else None, "source": "default name" if named else None,
            "default_name": settings.SANDBOX_RUNTIME}


def set_selection(ctx: AwsContext, runtime_arn: str | None) -> dict:
    if runtime_arn is not None and not runtime_arn.startswith("arn:aws:bedrock-agentcore:"):
        raise Invalid(f"runtime_arn must be an AgentCore runtime ARN, got {runtime_arn!r}")
    data = load_settings()
    region = data.setdefault("sandbox", {}).setdefault(ctx.region, {})
    if runtime_arn is None:
        region.pop("runtime_arn", None)  # back to the default name
    else:
        region["runtime_arn"] = runtime_arn
    save_settings(data)
    return selection(ctx)


def options(ctx: AwsContext) -> dict:
    out = {"runtimes": [], "errors": [], "default_name": settings.SANDBOX_RUNTIME, "code_sha": local_code_sha()}
    try:
        runtimes = _runtimes(ctx)
        out["runtimes"] = [{"arn": rt["agentRuntimeArn"], "name": rt.get("agentRuntimeName"), "status": rt.get("status"),
                            "updated": str(rt.get("lastUpdatedAt"))} for rt in runtimes]
        out["selection"] = selection(ctx, runtimes)
    except Exception as e:
        out["errors"].append(f"AgentCore runtimes: {_err(e)}")
        out["selection"] = {"runtime_arn": None, "source": None, "default_name": settings.SANDBOX_RUNTIME}
    return out


# -- checks --------------------------------------------------------------------------------

def _runtime_id(arn: str) -> str:
    return arn.rsplit("/", 1)[-1]


def describe(ctx: AwsContext, arn: str) -> dict:
    return client("bedrock-agentcore-control", ctx).get_agent_runtime(agentRuntimeId=_runtime_id(arn))


def platform_of(rt: dict) -> str:
    """A runtime's platform version. Runtimes created without one have none in get_agent_runtime (an MTRL agent's
    runtime, 2026-11-10), and the docs say such a runtime uses V1 ("If you omit platformVersion when you create a runtime, the
    runtime uses V1", Platform versions, read 2026-11-10)."""
    return rt.get("platformVersion") or "V1"


def check_runtime(ctx: AwsContext) -> tuple[dict, dict | None]:
    """The runtime: selected or found, READY, running this checkout's sandbox_exec.py. Returns (check, runtime)."""
    title = L("Sandbox runtime", "沙箱 runtime")
    why = L("Sandbox scenarios run model-written code in this AgentCore Runtime, one session (a microVM) per "
            "rollout, instead of in the run's process, which holds this machine's AWS credentials. It must run the "
            f"same sandbox_exec.py as this checkout. Deploy or update it on this machine with {DEPLOY_CMD}.",
            "沙箱场景把模型写的代码放到这个 AgentCore Runtime 里执行，每条 rollout 一个 session（一台 microVM），"
            "而不是在持有本机 AWS 凭证的运行进程里执行。它运行的 sandbox_exec.py 必须与本仓库的版本一致。"
            f"在本机用 {DEPLOY_CMD} 部署或更新。")
    try:
        sel = selection(ctx)
    except Exception as e:
        return _check("sandbox-runtime", title, "unverified", f"cannot list AgentCore runtimes: {_err(e)}", why), None
    if not sel["runtime_arn"]:
        return _check("sandbox-runtime", title, "fail", f"none selected, and no runtime named {settings.SANDBOX_RUNTIME} "
                      f"in {ctx.region}; deploy it with {DEPLOY_CMD}", why), None
    try:
        rt = describe(ctx, sel["runtime_arn"])
    except Exception as e:
        return _check("sandbox-runtime", title, "fail", f"{sel['runtime_arn']}: {_err(e)}", why), None
    env = rt.get("environmentVariables") or {}
    life = rt.get("lifecycleConfiguration") or {}
    network = (rt.get("networkConfiguration") or {}).get("networkMode")
    kind = "container" if "containerConfiguration" in (rt.get("agentRuntimeArtifact") or {}) else "code"
    deployed, local = env.get("SRL_SANDBOX_CODE"), local_code_sha()
    observed = (f"{rt.get('agentRuntimeName')} v{rt.get('agentRuntimeVersion')} ({sel['source']}), {rt.get('status')}, "
                f"platform {platform_of(rt)}, {kind}, {network} network, idle timeout "
                f"{life.get('idleRuntimeSessionTimeout')} s, sandbox_exec {deployed or 'unknown'}")
    if rt.get("status") != READY:
        return _check("sandbox-runtime", title, "fail", observed, why), rt
    if deployed != local:
        return _check("sandbox-runtime", title, "fail", observed + f"; this checkout has sandbox_exec {local}: redeploy "
                      f"with {DEPLOY_CMD}", why), rt
    idle = life.get("idleRuntimeSessionTimeout") or 0
    if idle > LONG_IDLE_S:
        return _check("sandbox-runtime", title, "warn", observed + f"; a session a run fails to stop bills its memory "
                      f"for up to {idle} s", why), rt
    return _check("sandbox-runtime", title, "pass", observed, why), rt


def check_role(ctx: AwsContext, rt: dict | None) -> dict:
    title = L("Sandbox execution role", "沙箱执行角色")
    why = L("Code running in the sandbox can obtain this role's credentials (measured on 2026-11-10), so the role "
            "should allow nothing beyond pulling the image and writing logs. Managed policies attached to it are "
            "reported here.",
            "沙箱里运行的代码能拿到这个角色的临时凭证（2026-11-10 实测），因此角色只应允许拉取镜像和写日志。"
            "这里列出挂在它上面的托管策略。")
    if rt is None:
        return _check("sandbox-role", title, "unverified", "no sandbox runtime to read the role of", why)
    role = rt.get("roleArn", "").rsplit("/", 1)[-1]
    iam = client("iam", ctx)
    try:
        managed = [p["PolicyName"] for p in iam.list_attached_role_policies(RoleName=role)["AttachedPolicies"]]
        inline = iam.list_role_policies(RoleName=role)["PolicyNames"]
    except Exception as e:
        return _check("sandbox-role", title, "unverified", f"{role}: {_err(e)}", why)
    observed = f"{role}: inline {', '.join(inline) or 'none'}; managed {', '.join(managed) or 'none'}"
    if managed:
        return _check("sandbox-role", title, "warn", observed + " (model-written code gets these permissions too)", why)
    return _check("sandbox-role", title, "pass", observed, why)


def usage_destination(ctx: AwsContext, runtime_arn: str) -> str | None:
    """The log group the runtime's USAGE_LOGS are delivered to, or None."""
    logs = client("logs", ctx)
    sources = [s["name"] for page in logs.get_paginator("describe_delivery_sources").paginate()
               for s in page.get("deliverySources", [])
               if s.get("logType") == "USAGE_LOGS" and runtime_arn in (s.get("resourceArns") or [])]
    if not sources:
        return None
    for page in logs.get_paginator("describe_deliveries").paginate():
        for d in page.get("deliveries", []):
            if d.get("deliverySourceName") in sources:
                dest = logs.get_delivery_destination(name=d["deliveryDestinationArn"].rsplit(":", 1)[-1].split("/")[-1])
                arn = dest["deliveryDestination"]["deliveryDestinationConfiguration"]["destinationResourceArn"]
                if ":log-group:" in arn:
                    return arn.split(":log-group:", 1)[1].removesuffix(":*")
    return None


def check_usage_logs(ctx: AwsContext, rt: dict | None) -> tuple[dict, str | None]:
    title = L("Sandbox usage logs", "沙箱用量日志")
    why = L("The run console shows what a run's sandbox sessions cost from the runtime's USAGE_LOGS (per session, "
            "per second, arriving about 20 minutes later). Without them a run's AgentCore cost cannot be shown. "
            "The deploy script sets them up.",
            "运行控制台根据 runtime 的 USAGE_LOGS（按 session、按秒，约 20 分钟后到达）显示一次运行的沙箱费用。"
            "没有它就无法显示运行的 AgentCore 费用。部署脚本会配置它。")
    if rt is None:
        return _check("sandbox-usage", title, "unverified", "no sandbox runtime", why), None
    try:
        group = usage_destination(ctx, rt["agentRuntimeArn"])
    except Exception as e:
        return _check("sandbox-usage", title, "unverified", _err(e), why), None
    if not group:
        return _check("sandbox-usage", title, "warn", f"no USAGE_LOGS delivery for {rt.get('agentRuntimeName')}; "
                      f"redeploy with {DEPLOY_CMD}", why), None
    return _check("sandbox-usage", title, "pass", f"delivered to {group}", why), group


def run_checks(ctx: AwsContext) -> tuple[list[dict], dict | None, str | None]:
    """(checks, runtime description, usage log group)."""
    runtime, rt = check_runtime(ctx)
    usage, group = check_usage_logs(ctx, rt)
    return [runtime, check_role(ctx, rt), usage], rt, group


def run_config(rt: dict, usage_group: str | None) -> dict:
    """The aws.sandbox block written into a sandbox scenario's config.json."""
    return {"runtime_arn": rt["agentRuntimeArn"], "runtime_name": rt.get("agentRuntimeName"),
            "version": rt.get("agentRuntimeVersion"), "qualifier": "DEFAULT", "platform": platform_of(rt),
            "code_sha": (rt.get("environmentVariables") or {}).get("SRL_SANDBOX_CODE"), "usage_log_group": usage_group}


# -- usage and cost ------------------------------------------------------------------------

def _price(region: str, platform: str | None) -> dict | None:
    return pricing.agentcore_price(region, platform)


def run_usage(ctx: AwsContext, run_id: str, sandbox: dict, start_ts: float, end_ts: float | None) -> dict:
    """A run's sandbox sessions as their USAGE_LOGS report them so far, and their list-price cost."""
    group = sandbox.get("usage_log_group")
    if not group:
        raise ApiError("This run's sandbox runtime had no USAGE_LOGS delivery when the run started, so its AgentCore "
                       f"usage cannot be read per run. Redeploy the sandbox ({DEPLOY_CMD}) for future runs.")
    logs = client("logs", ctx)
    per: dict[str, dict] = {}
    last = None
    kw = {"logGroupName": group, "filterPattern": f'"{run_id}"', "startTime": int((start_ts - 60) * 1000),
          "endTime": int(((end_ts or time.time()) + 3 * 3600) * 1000)}
    try:
        for page in logs.get_paginator("filter_log_events").paginate(**kw):
            for e in page.get("events", []):
                try:
                    m = json.loads(e["message"])
                except ValueError:
                    continue
                # Observed format (2026-11-10): one record per session per second, session id in attributes["session.id"].
                sid = (m.get("attributes") or {}).get("session.id") or ""
                if not sid.startswith(run_id):
                    continue
                mets = m.get("metrics") or {}
                s = per.setdefault(sid, {"vcpu_h": 0.0, "gb_h": 0.0})
                s["vcpu_h"] += float(mets.get("agent.runtime.vcpu.hours.used") or 0)
                s["gb_h"] += float(mets.get("agent.runtime.memory.gb_hours.used") or 0)
                last = max(last or 0, e["timestamp"] / 1000)
    except ClientError as e:
        raise ApiError(f"Reading {group} failed: {_err(e)}") from None
    vcpu, gb = sum(s["vcpu_h"] for s in per.values()), sum(s["gb_h"] for s in per.values())
    price = _price(ctx.region, sandbox.get("platform"))
    return {"kind": "sandbox", "sessions_reported": len(per), "vcpu_hours": vcpu, "gb_hours": gb,
            "usd": vcpu * price["vcpu_h"] + gb * price["gb_h"] if price else None, "price": price,
            "platform": sandbox.get("platform"), "price_as_of": pricing.AGENTCORE_PRICES_AS_OF,
            "last_record_ts": last, "source": f"USAGE_LOGS in {group}", "checked_at": time.time()}


def runtime_usage(ctx: AwsContext, runtime_arn: str, start_ts: float, end_ts: float | None, platform: str | None) -> dict:
    """A runtime's whole usage between two times, from its CPUUsed-vCPUHours and MemoryUsed-GBHours metrics."""
    cw = client("cloudwatch", ctx)
    start = datetime.fromtimestamp(start_ts - 120, timezone.utc)
    end = datetime.fromtimestamp((end_ts or time.time()) + 600, timezone.utc)
    dims = [{"Name": "Service", "Value": "AgentCore.Runtime"}, {"Name": "Resource", "Value": runtime_arn}]
    queries = [{"Id": f"m{i}", "ReturnData": True, "MetricStat": {"Stat": "Sum", "Period": 60, "Metric": {
        "Namespace": "AWS/Bedrock-AgentCore", "MetricName": name, "Dimensions": dims}}}
        for i, name in enumerate(("CPUUsed-vCPUHours", "MemoryUsed-GBHours"))]
    try:
        r = cw.get_metric_data(MetricDataQueries=queries, StartTime=start, EndTime=end)
    except ClientError as e:
        raise ApiError(f"Reading the AgentCore usage metrics of {runtime_arn} failed: {_err(e)}") from None
    got = {q["Id"]: q for q in r["MetricDataResults"]}
    vcpu, gb = sum(got["m0"]["Values"]), sum(got["m1"]["Values"])
    stamps = got["m0"]["Timestamps"] + got["m1"]["Timestamps"]
    price = _price(ctx.region, platform)
    return {"kind": "agent", "vcpu_hours": vcpu, "gb_hours": gb,
            "usd": vcpu * price["vcpu_h"] + gb * price["gb_h"] if price else None, "price": price, "platform": platform,
            "price_as_of": pricing.AGENTCORE_PRICES_AS_OF,
            "last_record_ts": max(s.timestamp() for s in stamps) if stamps else None,
            "source": f"CPUUsed-vCPUHours and MemoryUsed-GBHours of the whole runtime, {start:%H:%M}-{end:%H:%M} UTC",
            "checked_at": time.time()}


def active_sessions(ctx: AwsContext) -> dict:
    """Active AgentCore Runtime sessions in the account and region (all runtimes), from ActiveSessionCount."""
    limit = ACTIVE_SESSION_LIMIT.get(ctx.region, ACTIVE_SESSION_LIMIT_OTHER)
    out = {"active": None, "at": None, "limit": limit, "limit_source": "documented default (2026-06)", "error": None}
    if not ctx.ready:
        out["error"] = "needs AWS credentials"
        return out
    now = datetime.now(timezone.utc)
    try:
        r = client("cloudwatch", ctx).get_metric_data(
            MetricDataQueries=[{"Id": "a", "ReturnData": True, "MetricStat": {"Stat": "Maximum", "Period": 60, "Metric": {
                "Namespace": "AWS/Bedrock-AgentCore", "MetricName": "ActiveSessionCount",
                "Dimensions": [{"Name": "Service", "Value": "AgentCore.Runtime"}]}}}],
            StartTime=datetime.fromtimestamp(now.timestamp() - 900, timezone.utc), EndTime=now, ScanBy="TimestampDescending")
        res = r["MetricDataResults"][0]
        if res["Values"]:
            out["active"], out["at"] = int(res["Values"][0]), res["Timestamps"][0].timestamp()
        # No datapoint in 15 minutes: active stays None, and the page says there is no recent data, not that it is 0.
    except ClientError as e:
        out["error"] = f"ActiveSessionCount: {_err(e)}"
    return out
