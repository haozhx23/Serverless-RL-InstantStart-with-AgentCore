"""Setup checks: can this machine start a training session in this account and region?

Each check is read-only and reports one of four states:
    pass        verified just now
    fail        verified just now, and a launch would fail; a fix is offered when we can make one
    warn        works, with a cost or delay worth knowing about
    unverified  could not be checked; the reason is in `observed`

Fixes create the named resource exactly as the Training Session prerequisites require.
They never modify a resource this tool did not name, except attaching the managed
policy and trust statement to the tool's own role.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import threading
import time

from botocore.exceptions import ClientError

from .. import settings
from ..errors import ApiError, NotFound
from ..i18n import L, check_texts
from .context import AwsContext, client

TRUST_PRINCIPAL = "training-session.sagemaker.amazonaws.com"
ROLE_POLICY = "arn:aws:iam::aws:policy/AmazonSageMakerTrainingSessionFullAccess"
ACTIVE_SESSION_STATES = ("Created", "Scheduled", "Blocked", "Running", "Stopping")
DEFAULT_TOKENIZER = "openai/gpt-oss-20b"

_tokenizer_warmup: dict = {"state": "idle", "error": None}


def _check(id, group, title, status, observed, why, fix=None):
    """One check. title, why and the fix's label and detail may be L(en, zh) pairs; observed is never translated."""
    return {"id": id, "group": group, "status": status, "observed": observed, **check_texts(title, why, fix),
            "checked_at": time.time()}


def _err(e: Exception) -> str:
    if isinstance(e, ClientError):
        return f"{e.response['Error'].get('Code')}: {e.response['Error'].get('Message')}"
    return f"{type(e).__name__}: {e}"


# -- checks --------------------------------------------------------------------------------

def check_credentials(ctx: AwsContext) -> dict:
    why = L("Runs and these checks use the AWS credentials of the machine running this app.",
            "训练运行和就绪检查均使用本应用所在机器的 AWS 凭证。")
    title = L("AWS credentials", "AWS 凭证")
    if ctx.ready:
        return _check("credentials", "account", title, "pass", ctx.caller_arn, why)
    return _check("credentials", "account", title, "fail", ctx.identity_error or "no identity", why)


def check_models(ctx: AwsContext) -> tuple[dict, list[str]]:
    why = L("Training sessions offer a fixed list of base models per region. The list comes from the "
            "service, so names are used exactly as returned (Gemma is lowercase, unlike the docs).",
            "training session 在每个区域提供固定的基座模型列表。模型名称以服务返回值为准（Gemma 的名称为小写，与文档不同）。")
    title = L("Trainable base models", "可训练的基座模型")
    if not ctx.ready:
        return _check("models", "account", title, "unverified", "needs AWS credentials", why), []
    try:
        from sagemaker.train.training_session import ServiceClient
        models = ServiceClient(region=ctx.region, role_arn=ctx.role_arn).list_base_models()
    except Exception as e:
        return _check("models", "account", title, "fail", _err(e), why), []
    if not models:
        return _check("models", "account", title, "fail", f"none offered in {ctx.region}", why), []
    return _check("models", "account", title, "pass", ", ".join(models), why), models


def check_role(ctx: AwsContext) -> dict:
    title = L("Execution role", "执行角色")
    why = L("The session assumes this role. It must be under /service-role/ with \"SageMaker\" in its name, "
            "or creating a session fails with an AccessDenied on iam:PassRole that does not say why.",
            "session 使用此角色运行。角色必须位于 /service-role/ 路径下，且名称包含 \"SageMaker\"；否则创建 session 时"
            "会返回 iam:PassRole 的 AccessDenied 错误，但不会说明具体原因。")
    fix = {"id": "create-role", "label": L("Create role", "创建角色"),
           "detail": L(f"Creates {ctx.role_name} under /service-role/, trusted by {TRUST_PRINCIPAL}, "
                       f"with AmazonSageMakerTrainingSessionFullAccess attached.",
                       f"在 /service-role/ 路径下创建角色 {ctx.role_name}，配置对 {TRUST_PRINCIPAL} 的信任，"
                       "并附加 AmazonSageMakerTrainingSessionFullAccess 策略。")}
    if not ctx.ready:
        return _check("role", "account", title, "unverified", "needs AWS credentials", why)
    iam = client("iam", ctx)
    try:
        role = iam.get_role(RoleName=ctx.role_name)["Role"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            return _check("role", "account", title, "fail", f"{ctx.role_name} does not exist", why, fix)
        return _check("role", "account", title, "unverified", _err(e), why)
    problems = []
    if role["Path"] != "/service-role/":
        problems.append(f"path is {role['Path']}, not /service-role/ (a role's path cannot be changed)")
    trust = json.dumps(role.get("AssumeRolePolicyDocument", {}))
    if TRUST_PRINCIPAL not in trust:
        problems.append(f"not trusted by {TRUST_PRINCIPAL}")
    try:
        attached = [p["PolicyArn"] for p in iam.list_attached_role_policies(RoleName=ctx.role_name)["AttachedPolicies"]]
        if ROLE_POLICY not in attached:
            problems.append("AmazonSageMakerTrainingSessionFullAccess is not attached")
    except ClientError as e:
        problems.append(f"could not list attached policies ({_err(e)})")
    if not problems:
        return _check("role", "account", title, "pass", role["Arn"], why)
    fixable = role["Path"] == "/service-role/"
    repair = {**fix, "id": "repair-role", "label": L("Repair role", "修复角色"),
              "detail": L("Adds the missing trust statement and managed policy to this role.",
                          "为此角色补充缺失的信任声明和托管策略。")} if fixable else None
    return _check("role", "account", title, "fail", "; ".join(problems), why, repair)


def check_group(ctx: AwsContext) -> dict:
    title = L("Model package group", "Model package group")
    why = L("Every checkpoint and set of weights a run saves is registered as a version in this group, in your account.",
            "运行保存的 checkpoint 和权重，均注册为账号内所选 model package group 的版本。")
    fix = {"id": "create-group", "label": L("Create group", "创建 group"),
           "detail": L(f"Creates model package group {ctx.model_package_group}.",
                       f"创建 model package group {ctx.model_package_group}。")}
    if not ctx.ready:
        return _check("group", "account", title, "unverified", "needs AWS credentials", why)
    try:
        client("sagemaker", ctx).describe_model_package_group(ModelPackageGroupName=ctx.model_package_group)
        return _check("group", "account", title, "pass", ctx.model_package_group_arn, why)
    except ClientError as e:
        # A missing group is reported as a ValidationException whose message says it does not exist.
        if "does not exist" in e.response["Error"].get("Message", "").lower():
            return _check("group", "account", title, "fail", f"{ctx.model_package_group} does not exist", why, fix)
        return _check("group", "account", title, "unverified", _err(e), why)


def check_bucket(ctx: AwsContext) -> dict:
    title = L("Output bucket", "输出 S3 bucket")
    why = L("Weights are written under this prefix. The bucket name starts with sagemaker- because the caller "
            "managed policy only allows deleting objects there, which cleanup needs.",
            "权重文件保存在此前缀下。清理文件需要删除对象，而调用方的托管策略只允许删除名称以 sagemaker- 开头的 bucket 中的对象，因此 bucket 名称采用此前缀。")
    fix = {"id": "create-bucket", "label": L("Create bucket", "创建 bucket"),
           "detail": L(f"Creates bucket {ctx.bucket} in {ctx.region}.", f"在 {ctx.region} 创建 bucket {ctx.bucket}。")}
    if not ctx.ready:
        return _check("bucket", "account", title, "unverified", "needs AWS credentials", why)
    try:
        client("s3", ctx).head_bucket(Bucket=ctx.bucket)
        return _check("bucket", "account", title, "pass", ctx.s3_output_path, why)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket", "NotFound"):
            return _check("bucket", "account", title, "fail", f"{ctx.bucket} does not exist", why, fix)
        return _check("bucket", "account", title, "unverified", _err(e), why)


def session_quota(ctx: AwsContext) -> dict:
    """{limit, limit_source, active, active_names} for concurrent training sessions in the region."""
    out = {"limit": settings.DEFAULT_CONCURRENT_SESSIONS, "limit_source": "default (not read)",
           "active": None, "active_names": [], "error": None}
    if not ctx.ready:
        out["error"] = "needs AWS credentials"
        return out
    sq = client("service-quotas", ctx)
    for fn, label in ((sq.get_service_quota, "account quota"), (sq.get_aws_default_service_quota, "AWS default")):
        try:
            q = fn(ServiceCode="sagemaker", QuotaCode=settings.CONCURRENT_SESSIONS_QUOTA_CODE)["Quota"]
            out["limit"], out["limit_source"] = int(q["Value"]), label
            break
        except ClientError as e:
            out["error"] = f"quota {settings.CONCURRENT_SESSIONS_QUOTA_CODE}: {_err(e)}"
    try:
        sm = client("sagemaker", ctx)
        names, token = [], None
        while True:
            kw = {"MaxResults": 100, **({"NextToken": token} if token else {})}
            r = sm.list_training_sessions(**kw)
            for s in r.get("TrainingSessionSummaries", []):
                if s.get("TrainingSessionStatus") in ACTIVE_SESSION_STATES:
                    names.append(s["TrainingSessionArn"].rsplit("/", 1)[-1])
            token = r.get("NextToken")
            if not token:
                break
        out["active"], out["active_names"] = len(names), names
    except ClientError as e:
        out["error"] = f"list_training_sessions: {_err(e)}"
    return out


def check_quota(ctx: AwsContext) -> dict:
    title = L("Concurrent sessions", "并发 session")
    why = L("Each session holds one slot until it is stopped, even while idle and billing nothing. "
            "The default is 2; request more for a workshop.",
            "每个 session 在停止前都会占用一个并发名额，空闲且不计费时也不例外。默认配额为 2 个；如需开展实操培训，请申请更多名额。")
    q = session_quota(ctx)
    if q["active"] is None:
        return _check("quota", "account", title, "unverified", q["error"] or "unknown", why)
    observed = f"{q['active']} of {q['limit']} in use ({q['limit_source']})"
    if q["error"] and q["limit_source"].startswith("default"):
        observed += f"; limit not read: {q['error']}"
    if q["active"] >= q["limit"]:
        return _check("quota", "account", title, "fail", observed + " — stop a session before launching", why)
    status = "pass" if not q["limit_source"].startswith("default") else "warn"
    return _check("quota", "account", title, status, observed, why)


def check_sdk() -> dict:
    title = L("Training Session SDK", "Training Session SDK")
    why = L("The templates were verified with sagemaker-train 1.27.0 and boto3 1.43.119. The AWS CLI 2.36.15 "
            "has no training-session commands, so everything goes through the Python SDK.",
            "模板已使用 sagemaker-train 1.27.0 和 boto3 1.43.119 验证。AWS CLI 2.36.15 不提供 training-session 命令，因此相关操作均通过 Python SDK 执行。")
    found = {}
    for pkg in ("sagemaker-train", "boto3", "transformers"):
        try:
            found[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            found[pkg] = None
    observed = ", ".join(f"{k} {v or 'missing'}" for k, v in found.items())
    if not found["sagemaker-train"] or not found["boto3"] or not found["transformers"]:
        return _check("sdk", "machine", title, "fail", observed, why)
    if found["sagemaker-train"] != settings.TESTED_SDK["sagemaker-train"]:
        return _check("sdk", "machine", title, "warn", observed + " (templates verified with 1.27.0)", why)
    return _check("sdk", "machine", title, "pass", observed, why)


def check_tokenizer() -> dict:
    title = L("Tokenizer cache", "Tokenizer 缓存")
    why = L("The service takes token ids, so each run tokenizes locally. The first download of the "
            "gpt-oss-20b tokenizer took about 49 s, which looks like a hang during a demo.",
            "服务接收 token ID，因此每次运行都需要在本地进行分词。首次下载 gpt-oss-20b tokenizer 约需 49 s，演示时可能被误认为程序无响应。")
    fix = {"id": "warm-tokenizer", "label": L("Download now", "现在下载"),
           "detail": L(f"Downloads the {DEFAULT_TOKENIZER} tokenizer into the local cache.",
                       f"将 {DEFAULT_TOKENIZER} 的 tokenizer 下载到本地缓存。")}
    if _tokenizer_warmup["state"] == "running":
        return _check("tokenizer", "machine", title, "warn", "downloading…", why)
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(DEFAULT_TOKENIZER, "tokenizer.json")
    except Exception as e:
        return _check("tokenizer", "machine", title, "unverified", _err(e), why)
    if isinstance(cached, str):
        return _check("tokenizer", "machine", title, "pass", f"{DEFAULT_TOKENIZER} cached", why)
    observed = f"{DEFAULT_TOKENIZER} not cached"
    if _tokenizer_warmup["error"]:
        observed += f"; last download failed: {_tokenizer_warmup['error']}"
    return _check("tokenizer", "machine", title, "warn", observed, why, fix)


def run_all(ctx: AwsContext) -> dict:
    from . import mlflow, mtrl, sandbox

    models_check, models = check_models(ctx)
    checks = [check_credentials(ctx), models_check, check_role(ctx), check_group(ctx), check_bucket(ctx),
              check_quota(ctx), check_sdk(), check_tokenizer()]
    live_blockers = [c["id"] for c in checks if c["status"] == "fail"]
    tracking, mlflow_app = mlflow.check(ctx)  # never fails: a run without MLflow still has its event log
    checks.append(tracking)
    # Sandbox scenarios also need the environment sandbox; a warning (no usage logs, a broad role) does not block.
    sandbox_checks = sandbox.run_checks(ctx)[0] if ctx.ready else []
    sandbox_blockers = live_blockers + [c["id"] for c in sandbox_checks if c["status"] in ("fail", "unverified")]
    mtrl_checks, mtrl_models = mtrl.run_checks(ctx) if ctx.ready else ([], [])
    # MTRL also needs the credentials, the bucket and the SDK above.
    shared = [c["id"] for c in checks if c["id"] in ("credentials", "bucket", "sdk") and c["status"] == "fail"]
    mtrl_blockers = shared + [c["id"] for c in mtrl_checks if c["status"] != "pass"]
    if not ctx.ready:
        mtrl_blockers.append("credentials")
    if not ctx.ready:
        sandbox_blockers.append("credentials")
    return {"region": ctx.region, "account": ctx.account, "checked_at": time.time(),
            "checks": checks + sandbox_checks + mtrl_checks,
            "models": models, "live_ready": not live_blockers, "live_blockers": live_blockers,
            "sandbox_ready": not sandbox_blockers, "sandbox_blockers": sorted(set(sandbox_blockers)),
            "mtrl_models": mtrl_models, "mtrl_ready": not mtrl_blockers, "mtrl_blockers": sorted(set(mtrl_blockers)),
            "mlflow_app": mlflow_app}


# -- fixes ---------------------------------------------------------------------------------

def fix(fix_id: str, ctx: AwsContext) -> L:
    """Create or repair one resource. Returns what was done, in both languages."""
    if fix_id.startswith("mtrl-"):
        from . import mtrl
        if not ctx.ready:
            raise ApiError(f"cannot {fix_id}: no AWS identity ({ctx.identity_error})")
        return mtrl.fix(fix_id, ctx)
    if fix_id == "warm-tokenizer":
        return _start_tokenizer_warmup()
    if not ctx.ready:
        raise RuntimeError(f"cannot {fix_id}: no AWS identity ({ctx.identity_error})")
    if fix_id in ("create-role", "repair-role"):
        iam = client("iam", ctx)
        trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": TRUST_PRINCIPAL},
                                                         "Action": ["sts:AssumeRole", "sts:TagSession"]}]}
        if fix_id == "create-role":
            iam.create_role(Path="/service-role/", RoleName=ctx.role_name, AssumeRolePolicyDocument=json.dumps(trust),
                            Description="Execution role for Serverless RL InstantStart training sessions")
        else:
            iam.update_assume_role_policy(RoleName=ctx.role_name, PolicyDocument=json.dumps(trust))
        iam.attach_role_policy(RoleName=ctx.role_name, PolicyArn=ROLE_POLICY)
        return L(f"{ctx.role_name} is ready. New IAM roles can take about 10 s to be usable.",
                 f"{ctx.role_name} 已就绪。新建 IAM 角色后，可能需要约 10 s 才能使用。")
    if fix_id == "create-group":
        client("sagemaker", ctx).create_model_package_group(
            ModelPackageGroupName=ctx.model_package_group,
            ModelPackageGroupDescription="Checkpoints and weights saved by Serverless RL InstantStart runs")
        return L(f"Created model package group {ctx.model_package_group}.",
                 f"已创建 model package group {ctx.model_package_group}。")
    if fix_id == "create-bucket":
        kw = {} if ctx.region == "us-east-1" else {"CreateBucketConfiguration": {"LocationConstraint": ctx.region}}
        client("s3", ctx).create_bucket(Bucket=ctx.bucket, **kw)
        return L(f"Created bucket {ctx.bucket}.", f"已创建 bucket {ctx.bucket}。")
    raise NotFound(f"unknown fix {fix_id!r}")


def _start_tokenizer_warmup() -> L:
    if _tokenizer_warmup["state"] == "running":
        return L("Already downloading.", "下载已在进行中。")

    def run():
        _tokenizer_warmup.update(state="running", error=None)
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
            _tokenizer_warmup["state"] = "done"
        except Exception as e:
            _tokenizer_warmup.update(state="failed", error=_err(e))
    threading.Thread(target=run, daemon=True).start()
    return L(f"Downloading the {DEFAULT_TOKENIZER} tokenizer in the background.",
             f"正在后台下载 {DEFAULT_TOKENIZER} 的 tokenizer。")
