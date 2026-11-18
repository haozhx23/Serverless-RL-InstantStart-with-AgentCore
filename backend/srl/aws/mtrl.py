"""Multi-turn RL (MTRL) prerequisites, access probe and jobs, as the backend sees them.

An MTRL run needs, besides the bucket: a job execution role, an AgentCore Runtime running
the agent, two model package groups (output and intermediate checkpoints, which must
differ), an MLflow app, a free MTRL job slot (default quota 1), and account-level access to
create AgentRFT jobs. The last one is not in the documented prerequisites: on 2026-08-24
and 2026-10-28 CreateJob refused this internal account, the second time with
"This internal AWS account is not allowlisted for this feature".
"""

from __future__ import annotations

import json
import threading
import time

from botocore.exceptions import ClientError

from .. import settings
from ..errors import Invalid, NotFound
from ..i18n import L, check_texts
from .context import AwsContext, client, load_settings, save_settings

JOB_TRUST = "job.sagemaker.amazonaws.com"
JOB_POLICY = "arn:aws:iam::aws:policy/AmazonSageMakerJobFullAccess"
RUNTIME_POLICY = "arn:aws:iam::aws:policy/AmazonSageMakerJobRuntimeAccess"
# The SDK checks the role before submitting; the managed policy lacks these two (2026-08-24).
JOB_EXTRA_ACTIONS = ["cloudwatch:PutMetricData", "ecr:GetAuthorizationToken"]
JOB_EXTRA_POLICY_NAME = "ServerlessRLInstantStartMTRLExtras"
ACTIVE_JOB_STATES = ("InProgress", "Stopping")
PROBE_MODEL = "openai-reasoning-gpt-oss-20b"

_models_cache: dict[str, tuple[float, list[str]]] = {}
_lock = threading.Lock()


def _err(e: Exception) -> str:
    if isinstance(e, ClientError):
        return f"{e.response['Error'].get('Code')}: {e.response['Error'].get('Message')}"
    return f"{type(e).__name__}: {e}"


def _check(id, title, status, observed, why, fix=None):
    """One MTRL check. title, why and the fix's label and detail may be L(en, zh) pairs; observed is never translated."""
    return {"id": id, "group": "mtrl", "status": status, "observed": observed, **check_texts(title, why, fix),
            "checked_at": time.time()}


# -- selections ----------------------------------------------------------------------------

def selection(ctx: AwsContext) -> dict:
    """The runtime, MLflow app and job role chosen on the Setup page for this region."""
    s = load_settings().get("mtrl", {}).get(ctx.region, {})
    role = s.get("job_role_name") or settings.MTRL_JOB_ROLE
    return {"runtime_arn": s.get("runtime_arn"), "mlflow_app_arn": s.get("mlflow_app_arn"), "job_role_name": role,
            "job_role_arn": f"arn:aws:iam::{ctx.account}:role/{role}" if ctx.account else None,
            "output_group": settings.MODEL_PACKAGE_GROUP,
            "checkpoint_group": settings.MTRL_CHECKPOINT_GROUP}


def set_selection(ctx: AwsContext, **fields) -> dict:
    allowed = {"runtime_arn", "mlflow_app_arn", "job_role_name"}
    unknown = set(fields) - allowed
    if unknown:
        raise Invalid(f"unknown MTRL setting(s): {', '.join(sorted(unknown))}")
    data = load_settings()
    region = data.setdefault("mtrl", {}).setdefault(ctx.region, {})
    region.update({k: v for k, v in fields.items() if v is not None})
    save_settings(data)
    return selection(ctx)


def run_config(ctx: AwsContext) -> dict:
    """The aws.mtrl block written into an MTRL run's config.json."""
    s = selection(ctx)
    group = lambda name: f"arn:aws:sagemaker:{ctx.region}:{ctx.account}:model-package-group/{name}"  # noqa: E731
    return {"runtime_arn": s["runtime_arn"], "mlflow_app_arn": s["mlflow_app_arn"], "job_role_arn": s["job_role_arn"],
            "output_group_arn": group(s["output_group"]), "checkpoint_group_arn": group(s["checkpoint_group"]),
            "output_prefix": f"s3://{ctx.bucket}/{settings.S3_PREFIX}/mtrl", "mlflow_experiment": settings.S3_PREFIX}


# -- options for the pickers ---------------------------------------------------------------

def options(ctx: AwsContext) -> dict:
    out = {"runtimes": [], "mlflow_apps": [], "job_roles": [], "errors": []}
    try:
        token = None
        while True:
            r = client("bedrock-agentcore-control", ctx).list_agent_runtimes(**({"nextToken": token} if token else {}))
            out["runtimes"] += [{"arn": rt["agentRuntimeArn"], "name": rt.get("agentRuntimeName"),
                                 "status": rt.get("status"), "updated": str(rt.get("lastUpdatedAt"))}
                                for rt in r.get("agentRuntimes", [])]
            token = r.get("nextToken")
            if not token:
                break
    except Exception as e:
        out["errors"].append(f"AgentCore runtimes: {_err(e)}")
    try:
        r = client("sagemaker", ctx).list_mlflow_apps()
        out["mlflow_apps"] = [{"arn": a["Arn"], "name": a.get("Name"), "status": a.get("Status")}
                              for a in r.get("Summaries", [])]
    except Exception as e:
        out["errors"].append(f"MLflow apps: {_err(e)}")
    try:
        for page in client("iam", ctx).get_paginator("list_roles").paginate():
            for role in page["Roles"]:
                if JOB_TRUST in json.dumps(role.get("AssumeRolePolicyDocument", {})):
                    out["job_roles"].append(role["RoleName"])
    except Exception as e:
        out["errors"].append(f"IAM roles: {_err(e)}")
    return out


# -- checks --------------------------------------------------------------------------------

def supported_models(ctx: AwsContext) -> list[str]:
    with _lock:
        cached = _models_cache.get(ctx.region)
    if cached and time.time() - cached[0] < 600:
        return cached[1]
    import boto3
    from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer
    models = MultiTurnRLTrainer.list_supported_models(session=boto3.Session(region_name=ctx.region))
    with _lock:
        _models_cache[ctx.region] = (time.time(), models)
    return models


def check_region(ctx: AwsContext) -> dict:
    ok = ctx.region in settings.MTRL_REGIONS
    return _check("mtrl-region", L("Region", "区域"), "pass" if ok else "fail",
                  f"{ctx.region}{'' if ok else ' does not offer MTRL'}",
                  L(f"MTRL is offered in {', '.join(settings.MTRL_REGIONS)}; Gemma 4 and Qwen 3.6 only in us-west-2.",
                    f"MTRL 支持的区域为 {'、'.join(settings.MTRL_REGIONS)}；Gemma 4 和 Qwen 3.6 仅在 us-west-2 提供。"))


def check_models(ctx: AwsContext) -> tuple[dict, list[str]]:
    why = L("Model ids come from the service (Hub content names such as openai-reasoning-gpt-oss-20b).",
            "模型 ID 以服务返回值为准，使用 Hub 内容名称，例如 openai-reasoning-gpt-oss-20b。")
    title = L("Supported models", "支持的模型")
    try:
        models = supported_models(ctx)
    except Exception as e:
        return _check("mtrl-models", title, "fail", _err(e), why), []
    return _check("mtrl-models", title, "pass" if models else "fail", ", ".join(models) or "none", why), models


def check_job_role(ctx: AwsContext) -> dict:
    s = selection(ctx)
    title = L("Job execution role", "任务执行角色")
    why = L(f"SageMaker AI assumes it to run the job. Trusted by {JOB_TRUST}, with AmazonSageMakerJobFullAccess plus "
            "cloudwatch:PutMetricData and ecr:GetAuthorizationToken: without those two the SDK refuses to submit "
            "(RoleValidationError), and the managed policy does not include them.",
            f"SageMaker AI 使用此角色运行任务。角色需信任 {JOB_TRUST}，附加 AmazonSageMakerJobFullAccess 策略，并额外授予 "
            "cloudwatch:PutMetricData 和 ecr:GetAuthorizationToken 权限。这两项权限不包含在托管策略中，"
            "缺少时 SDK 会拒绝提交任务并返回 RoleValidationError。")
    fix = {"id": "mtrl-create-job-role", "label": L("Create role", "创建角色"),
           "detail": L(f"Creates {s['job_role_name']} trusted by {JOB_TRUST}, attaches AmazonSageMakerJobFullAccess and "
                       f"an inline policy with {' and '.join(JOB_EXTRA_ACTIONS)}.",
                       f"创建角色 {s['job_role_name']}，配置对 {JOB_TRUST} 的信任，附加 AmazonSageMakerJobFullAccess 策略，"
                       f"并添加包含 {' 和 '.join(JOB_EXTRA_ACTIONS)} 权限的内联策略。")}
    iam = client("iam", ctx)
    try:
        role = iam.get_role(RoleName=s["job_role_name"])["Role"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            return _check("mtrl-job-role", title, "fail", f"{s['job_role_name']} does not exist", why, fix)
        return _check("mtrl-job-role", title, "unverified", _err(e), why)
    problems = []
    if JOB_TRUST not in json.dumps(role.get("AssumeRolePolicyDocument", {})):
        problems.append(f"not trusted by {JOB_TRUST}")
    attached = [p["PolicyArn"] for p in iam.list_attached_role_policies(RoleName=role["RoleName"])["AttachedPolicies"]]
    if JOB_POLICY not in attached:
        problems.append("AmazonSageMakerJobFullAccess is not attached")
    try:
        result = iam.simulate_principal_policy(PolicySourceArn=role["Arn"], ActionNames=JOB_EXTRA_ACTIONS)
        denied = [r["EvalActionName"] for r in result["EvaluationResults"] if r["EvalDecision"] != "allowed"]
        if denied:
            problems.append(f"missing {', '.join(denied)}")
    except ClientError as e:
        problems.append(f"could not simulate its policies ({_err(e)})")
    if problems:
        return _check("mtrl-job-role", title, "fail", f"{role['Arn']}: " + "; ".join(problems), why,
                      {**fix, "id": "mtrl-repair-job-role", "label": L("Repair role", "修复角色"),
                       "detail": L("Adds the missing trust statement, managed policy and inline permissions.",
                                   "补充缺失的信任声明、托管策略和内联权限。")})
    return _check("mtrl-job-role", title, "pass", role["Arn"], why)


def check_runtime(ctx: AwsContext) -> dict:
    s = selection(ctx)
    title = L("Agent in AgentCore Runtime", "AgentCore Runtime 中的 agent")
    why = L("SageMaker AI calls this runtime for every rollout. It must be READY and run under a role with "
            "AmazonSageMakerJobRuntimeAccess (Sample, CompleteRollout, UpdateReward). Deploy the sample's agent with "
            "deploy/setup_aws.sh in the scenario's files; AgentCore Runtime needs an ARM64 image.",
            "SageMaker AI 在每条 rollout 中调用此 runtime。runtime 必须处于 READY 状态，执行角色需附加 "
            "AmazonSageMakerJobRuntimeAccess 策略（Sample、CompleteRollout、UpdateReward）。"
            "可通过场景文件中的 deploy/setup_aws.sh 部署样例 agent；AgentCore Runtime 需要 ARM64 镜像。")
    if not s["runtime_arn"]:
        return _check("mtrl-runtime", title, "fail", "no runtime selected; pick one below", why)
    try:
        rt = client("bedrock-agentcore-control", ctx).get_agent_runtime(agentRuntimeId=s["runtime_arn"].rsplit("/", 1)[-1])
    except Exception as e:
        return _check("mtrl-runtime", title, "fail", f"{s['runtime_arn']}: {_err(e)}", why)
    image = ((rt.get("agentRuntimeArtifact") or {}).get("containerConfiguration") or {}).get("containerUri")
    observed = f"{rt.get('agentRuntimeName')} {rt.get('status')} · image {image or '?'} · updated {rt.get('lastUpdatedAt')}"
    if rt.get("status") != "READY":
        return _check("mtrl-runtime", title, "fail", observed, why)
    role_name = (rt.get("roleArn") or "").rsplit("/", 1)[-1]
    try:
        attached = [p["PolicyArn"] for p in client("iam", ctx).list_attached_role_policies(RoleName=role_name)["AttachedPolicies"]]
        if RUNTIME_POLICY not in attached:
            return _check("mtrl-runtime", title, "fail", observed + f"; its role {role_name} lacks AmazonSageMakerJobRuntimeAccess", why)
    except ClientError as e:
        return _check("mtrl-runtime", title, "unverified", observed + f"; role not checked: {_err(e)}", why)
    return _check("mtrl-runtime", title, "pass", observed, why)


def check_groups(ctx: AwsContext) -> dict:
    s = selection(ctx)
    title = L("Model package groups", "Model package group")
    why = L("MTRL registers the trained adapter in an output group and intermediate checkpoints in a second, "
            "different group. Passing both stops the SDK from creating default groups.",
            "MTRL 将训练后的 adapter 注册到输出 model package group，并将中间 checkpoint 注册到另一个 group。显式传入这两个不同的 group，可以避免 SDK 创建默认 group。")
    missing = []
    for name in (s["output_group"], s["checkpoint_group"]):
        try:
            client("sagemaker", ctx).describe_model_package_group(ModelPackageGroupName=name)
        except ClientError as e:
            if "does not exist" in e.response["Error"].get("Message", "").lower():
                missing.append(name)
            else:
                return _check("mtrl-groups", title, "unverified", _err(e), why)
    if missing:
        return _check("mtrl-groups", title, "fail", f"missing: {', '.join(missing)}", why,
                      {"id": "mtrl-create-groups", "label": L("Create groups", "创建 group"),
                       "detail": L(f"Creates {', '.join(missing)}.", f"创建 {'、'.join(missing)}。")})
    return _check("mtrl-groups", title, "pass", f"{s['output_group']} (output), {s['checkpoint_group']} (checkpoints)", why)


def check_mlflow(ctx: AwsContext) -> dict:
    s = selection(ctx)
    why = L("MTRL writes per-step metrics and rollout traces to an MLflow app; this app reads reward and turns from it. "
            "Selecting one explicitly stops the SDK from creating an app.",
            "MTRL 将每步指标和 rollout 轨迹写入 MLflow app，本应用从中读取 reward 和轮数。显式选择 app，可以避免 SDK 创建默认 app。")
    title = L("MLflow app", "MLflow app")
    if not s["mlflow_app_arn"]:
        return _check("mtrl-mlflow", title, "fail", "no MLflow app selected; pick one under Experiment tracking", why)
    try:
        app = client("sagemaker", ctx).describe_mlflow_app(Arn=s["mlflow_app_arn"])
    except Exception as e:
        return _check("mtrl-mlflow", title, "fail", f"{s['mlflow_app_arn']}: {_err(e)}", why)
    return _check("mtrl-mlflow", title, "pass", f"{app.get('Name')} ({app.get('Status')})", why)


def job_quota(ctx: AwsContext) -> dict:
    out = {"limit": 1, "limit_source": "default (not read)", "active": None, "active_names": [], "error": None}
    try:
        q = client("service-quotas", ctx).get_service_quota(ServiceCode="sagemaker", QuotaCode=settings.MTRL_QUOTA_CODE)["Quota"]
        out["limit"], out["limit_source"] = int(q["Value"]), "account quota"
    except ClientError as e:
        out["error"] = f"quota {settings.MTRL_QUOTA_CODE}: {_err(e)}"
    try:
        jobs = list_jobs(ctx)
        active = [j["name"] for j in jobs if j["status"] in ACTIVE_JOB_STATES]
        out["active"], out["active_names"] = len(active), active
    except ClientError as e:
        out["error"] = f"list_jobs: {_err(e)}"
    return out


def check_quota(ctx: AwsContext) -> dict:
    q = job_quota(ctx)
    why = L("Only one MTRL training job runs at a time by default, and a failed job must finish cleaning up before a retry.",
            "默认最多同时运行一个 MTRL 训练任务。任务失败后，需要等待清理完成才能重试。")
    title = L("Concurrent MTRL jobs", "并发 MTRL 任务")
    if q["active"] is None:
        return _check("mtrl-quota", title, "unverified", q["error"], why)
    observed = f"{q['active']} of {q['limit']} in use ({q['limit_source']})"
    return _check("mtrl-quota", title, "fail" if q["active"] >= q["limit"] else "pass", observed, why)


def check_access(ctx: AwsContext) -> dict:
    title = L("Permission to create MTRL jobs", "创建 MTRL 任务的权限")
    why = L("Creating AgentRFT jobs can need account-level access that IAM does not grant and the prerequisites do "
            "not mention. The probe sends one CreateJob with a role that does not exist: it creates no job, and the "
            "reply shows whether the account-level gate is passed.",
            "创建 AgentRFT 任务可能还需要账号级访问权限，IAM 无法授予该权限，前置条件文档也未说明。权限检查会使用不存在的角色发送一次 CreateJob 请求，不会创建任务；返回结果可用于判断账号是否具备访问权限。")
    fix = {"id": "mtrl-check-access", "label": L("Check access", "检查权限"),
           "detail": L("Calls CreateJob once with a valid document and a non-existent role. No job is created; the SDK "
                       "may write an empty prefix marker under the app's S3 prefix.",
                       "使用有效配置和不存在的角色调用一次 CreateJob，不会创建任务。SDK 可能会在本应用的 S3 前缀下写入一个空的前缀标记。")}
    last = load_settings().get("mtrl_access", {}).get(ctx.region)
    if not last:
        return _check("mtrl-access", title, "unverified", "not probed yet", why, fix)
    result = {**_check("mtrl-access", title, last["status"], last["observed"], why, fix), "checked_at": last["checked_at"]}
    return result


def run_checks(ctx: AwsContext) -> tuple[list[dict], list[str]]:
    region = check_region(ctx)
    if region["status"] == "fail":
        return [region], []
    models_check, models = check_models(ctx)
    checks = [region, check_access(ctx), models_check, check_job_role(ctx), check_runtime(ctx), check_groups(ctx),
              check_mlflow(ctx), check_quota(ctx)]
    return checks, models


# -- access probe --------------------------------------------------------------------------

def probe_access(ctx: AwsContext) -> dict:
    """One CreateJob that cannot create anything; records whether the account-level gate is passed."""
    s, rc = selection(ctx), run_config(ctx)
    missing = [k for k in ("runtime_arn", "mlflow_app_arn") if not s[k]]
    if missing:
        raise RuntimeError(f"Select {' and '.join(m.replace('_arn', '').replace('_', ' ') for m in missing)} first: "
                           "the probe needs a valid job document")
    import boto3
    from sagemaker.core.helper.session_helper import Session
    from sagemaker.train import multi_turn_rl_trainer as m
    from sagemaker.train.agent_rft_job import JOB_CATEGORY

    session = Session(boto_session=boto3.Session(region_name=ctx.region))
    trainer = m.MultiTurnRLTrainer(
        model=PROBE_MODEL, agent_env=s["runtime_arn"], training_dataset=f"{rc['output_prefix']}/access-probe/train.jsonl",
        s3_output_path=f"{rc['output_prefix']}/access-probe/", output_model_package_group=rc["output_group_arn"],
        intermediate_checkpoint_model_package_group=rc["checkpoint_group_arn"], mlflow_app_arn=s["mlflow_app_arn"],
        role=s["job_role_arn"], accept_eula=True, sagemaker_session=session)
    doc = trainer._build_job_config_document()
    name = f"srl-access-probe-{int(time.time())}"
    sm = client("sagemaker", ctx)
    fake_role = f"arn:aws:iam::{ctx.account}:role/srl-probe-role-that-does-not-exist"
    try:
        sm.create_job(JobName=name, JobCategory=JOB_CATEGORY, RoleArn=fake_role,
                      JobConfigSchemaVersion=m.JOB_CONFIG_SCHEMA_VERSION, JobConfigDocument=doc)
        # Not expected: a job exists. Stop it so it cannot run under a role that does not exist.
        try:
            sm.stop_job(JobName=name, JobCategory=JOB_CATEGORY)
        except ClientError:
            pass
        status, observed = "warn", f"CreateJob accepted a non-existent role and created {name}; it was stopped"
    except ClientError as e:
        msg = e.response["Error"].get("Message", "")
        if "not allowlisted" in msg or "not authorized" in msg.lower():
            status, observed = "fail", f"Refused at the account level: {msg}"
        elif "role" in msg.lower():
            status, observed = "pass", f"Passed the account-level gate; refused only for the fake role: {msg}"
        else:
            status, observed = "unverified", f"Unexpected reply, cannot tell: {_err(e)}"
    data = load_settings()
    data.setdefault("mtrl_access", {})[ctx.region] = {"status": status, "observed": observed, "checked_at": time.time()}
    save_settings(data)
    return check_access(ctx)


# -- fixes ---------------------------------------------------------------------------------

def fix(fix_id: str, ctx: AwsContext) -> L:
    """Create or repair one MTRL resource, or probe access. Returns what was done, in both languages."""
    s = selection(ctx)
    if fix_id == "mtrl-check-access":
        r = probe_access(ctx)
        return L(f"Access check: {r['observed']}", f"权限检查：{r['observed']}")
    if fix_id in ("mtrl-create-job-role", "mtrl-repair-job-role"):
        iam = client("iam", ctx)
        trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": JOB_TRUST},
                                                         "Action": ["sts:AssumeRole", "sts:TagSession"]}]}
        if fix_id == "mtrl-create-job-role":
            iam.create_role(RoleName=s["job_role_name"], AssumeRolePolicyDocument=json.dumps(trust),
                            Description="MTRL job execution role for Serverless RL InstantStart")
        else:
            iam.update_assume_role_policy(RoleName=s["job_role_name"], PolicyDocument=json.dumps(trust))
        iam.attach_role_policy(RoleName=s["job_role_name"], PolicyArn=JOB_POLICY)
        iam.put_role_policy(RoleName=s["job_role_name"], PolicyName=JOB_EXTRA_POLICY_NAME, PolicyDocument=json.dumps(
            {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": JOB_EXTRA_ACTIONS, "Resource": "*"}]}))
        return L(f"{s['job_role_name']} is ready. New IAM roles can take about 10 s to be usable.",
                 f"{s['job_role_name']} 已就绪。新建 IAM 角色后，可能需要约 10 s 才能使用。")
    if fix_id == "mtrl-create-groups":
        created = []
        for name in (s["output_group"], s["checkpoint_group"]):
            try:
                client("sagemaker", ctx).describe_model_package_group(ModelPackageGroupName=name)
            except ClientError:
                client("sagemaker", ctx).create_model_package_group(
                    ModelPackageGroupName=name, ModelPackageGroupDescription="Serverless RL InstantStart MTRL")
                created.append(name)
        return L(f"Created {', '.join(created) or 'nothing (both exist)'}.",
                 f"已创建 {'、'.join(created)}。" if created else "两个 group 均已存在，无需新建。")
    raise NotFound(f"unknown fix {fix_id!r}")


# -- jobs ----------------------------------------------------------------------------------

def list_jobs(ctx: AwsContext) -> list[dict]:
    from sagemaker.train.agent_rft_job import JOB_CATEGORY
    sm = client("sagemaker", ctx)
    out, token = [], None
    while True:
        kw = {"JobCategory": JOB_CATEGORY, "MaxResults": 100, **({"NextToken": token} if token else {})}
        r = sm.list_jobs(**kw)
        for j in r.get("JobSummaries", []):
            out.append({"name": j.get("JobName"), "arn": j.get("JobArn"), "status": j.get("JobStatus"),
                        "secondary": j.get("SecondaryStatus"), "created": j.get("CreationTime"),
                        "ended": j.get("EndTime"), "failure_reason": j.get("FailureReason")})
        token = r.get("NextToken")
        if not token:
            break
    return out


def stop_job(ctx: AwsContext, name: str) -> None:
    from sagemaker.train.agent_rft_job import JOB_CATEGORY
    client("sagemaker", ctx).stop_job(JobName=name, JobCategory=JOB_CATEGORY)
