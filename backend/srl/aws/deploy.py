"""Deploy an AgentCore Runtime from this checkout, from the Setup page: the environment sandbox
(docs/architecture.md 3.10) or a scenario's MTRL agent (3.4).

Both are container images built from a Dockerfile under templates/. AgentCore Runtime runs only
linux/arm64 images, and neither this app's container nor, often, the machine it runs on has docker
with arm64 emulation, so AWS CodeBuild builds the image on an ARM64 build host in the account: the
build context goes to the output bucket as a zip, and one CodeBuild project builds it and pushes it to
ECR. scripts/deploy_sandbox.py can build the sandbox with the machine's docker instead (--build local);
everything around the image is this module.

A deploy is a job of fixed steps in a background thread; the Setup page polls it. Every step can be
repeated: an existing role is updated, an image whose tag (a hash of its build files) is already in
ECR is not built again, an existing runtime gets a new version. Settings of an existing runtime that the
target does not set (environment variables, lifecycle, network...) are kept, so a runtime someone set up
by hand keeps them. Delete removes only resources with the target's names.

Neither runs while a live run uses the runtime: a new version would answer the run's next sessions
with other code, and the run's first-answer check would fail it.

The targets: the sandbox is templates/_sandbox/ (shared, so its names live in settings.py); an MTRL
scenario names its agent in scenario.json `agent` (runtime, repository, role, dockerfile, context,
smoke_test), so nothing here knows a scenario card.
"""

from __future__ import annotations

import hashlib
import io
import json
import secrets
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from botocore.exceptions import ClientError

from .. import settings
from ..errors import ApiError, Invalid, NotFound
from .context import AwsContext, client, load_settings, save_settings

# CodeBuild: one project builds every target's image (docs/architecture.md 6).
BUILD_IMAGE = "aws/codebuild/amazonlinux-aarch64-standard:3.0"
BUILD_COMPUTE = "BUILD_GENERAL1_SMALL"  # ARM, 2 vCPU, 4 GiB
BUILD_TIMEOUT_MIN = 60
BUILD_POLL_S = 5
DOCKER_LIBRARY_MIRROR = "public.ecr.aws/docker/library"

SANDBOX_FILES = ("_sandbox/Dockerfile", "_sandbox/Dockerfile.dockerignore", "_sandbox/requirements.txt",
                 "_sandbox/main.py", "_shared/sandbox_exec.py")
SANDBOX_LIFECYCLE = {"idleRuntimeSessionTimeout": 120, "maxLifetime": 3600}
# Platform V2 restores every session from a snapshot; a new version's first sessions were slow until the
# snapshot had served about this many (docs/261110-agentcore-runtime-sandbox.md 8.1). About $0.004.
SANDBOX_WARM = 32

AGENT_RUNTIME_POLICY = "arn:aws:iam::aws:policy/AmazonSageMakerJobRuntimeAccess"
AGENT_CODE_ENV = "SRL_AGENT_CODE"
SANDBOX_CODE_ENV = "SRL_SANDBOX_CODE"

# Settings of an existing runtime kept on update when the target does not set them.
KEEP_ON_UPDATE = ("networkConfiguration", "authorizerConfiguration", "requestHeaderConfiguration",
                  "protocolConfiguration", "lifecycleConfiguration", "metadataConfiguration",
                  "filesystemConfigurations", "capacityProviderConfiguration", "platformVersion")
READY_TIMEOUT_S = 15 * 60

STEPS = {
    ("sandbox", "deploy"): ["role", "builder", "image", "runtime", "usage_logs", "warm", "select"],
    ("agent", "deploy"): ["role", "builder", "image", "runtime", "smoke", "select"],
    ("sandbox", "delete"): ["usage_logs", "runtime", "repository", "role", "builder", "sources", "select"],
    ("agent", "delete"): ["runtime", "repository", "role", "builder", "sources", "select"],
}

# What the identity running this app needs, shown before a deploy (Setup) and in the docs.
PERMISSIONS = {
    "iam": ["GetRole", "CreateRole", "UpdateAssumeRolePolicy", "PutRolePolicy", "AttachRolePolicy", "PassRole",
            "ListAttachedRolePolicies", "ListRolePolicies", "DetachRolePolicy", "DeleteRolePolicy", "DeleteRole"],
    "codebuild": ["BatchGetProjects", "CreateProject", "UpdateProject", "StartBuild", "BatchGetBuilds", "DeleteProject"],
    "ecr": ["CreateRepository", "DescribeImages", "DeleteRepository"],
    "s3": ["PutObject", "ListBucket", "DeleteObject"],
    "logs": ["GetLogEvents", "GetDeliverySource"],
    "bedrock-agentcore": ["ListAgentRuntimes", "GetAgentRuntime", "GetAgentRuntimeEndpoint", "CreateAgentRuntime",
                          "UpdateAgentRuntime", "DeleteAgentRuntime", "InvokeAgentRuntime", "StopRuntimeSession"],
}
SANDBOX_PERMISSIONS = {"logs": ["CreateLogGroup", "PutDeliverySource", "PutDeliveryDestination", "CreateDelivery",
                                "DescribeDeliveries", "DeleteDelivery", "DeleteDeliverySource",
                                "DeleteDeliveryDestination"]}


class DeployError(Exception):
    """A step that could not be done: what, where, and what AWS or the build said."""


def _err(e: Exception) -> str:
    if isinstance(e, ClientError):
        return f"{e.operation_name}: {e.response['Error'].get('Code')}: {e.response['Error'].get('Message')}"
    return f"{type(e).__name__}: {e}"


def _code(e: ClientError) -> str:
    return e.response["Error"].get("Code", "")


# -- targets -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Target:
    id: str                       # "sandbox", or "agent-<scenario id>"
    kind: str                     # "sandbox" | "agent"
    runtime_name: str
    repository: str
    role_name: str
    context: Path                 # the build context
    files: tuple[str, ...]        # relative to context, in hash order; a name ending in / is a directory
    dockerfile: str               # relative to context
    code_env: str                 # the runtime environment variable that records what code it runs
    scenario: str | None = None
    platform: str | None = None   # None: the service's default (V1) for a new runtime, kept on update
    lifecycle: dict | None = None
    smoke_test: str | None = None  # agent: a script in the card, run with --runtime-arn and --region
    warm: int = 0

    def build_files(self) -> list[tuple[str, Path]]:
        """(name in the context, path) for every file of the build context, in hash order."""
        out = []
        for name in self.files:
            path = self.context / name
            if name.endswith("/"):
                if not path.is_dir():
                    raise DeployError(f"{path}: listed as a build directory of {self.id}, but it is not a directory")
                for f in sorted(path.rglob("*")):
                    if f.is_file() and "__pycache__" not in f.parts and f.suffix not in (".pyc", ".pyo"):
                        out.append((f.relative_to(self.context).as_posix(), f))
            elif path.is_file():
                out.append((name, path))
            else:
                raise DeployError(f"{path}: listed as a build file of {self.id}, but it does not exist")
        return out

    def image_tag(self) -> str:
        """A hash of the build files (the same as scripts/deploy_sandbox.py has always tagged the sandbox with)."""
        h = hashlib.sha256()
        for name, path in self.build_files():
            h.update(name.encode() + b"\0" + path.read_bytes())
        return h.hexdigest()[:12]

    def code_version(self) -> str:
        """What the runtime's code_env must say for the runtime to run this checkout's code."""
        if self.kind == "sandbox":
            # The run's first-answer check compares sandbox_exec.py's own hash (sandbox_client), not the image's.
            return hashlib.sha256((self.context / "_shared" / "sandbox_exec.py").read_bytes()).hexdigest()[:12]
        return self.image_tag()

    def env(self) -> dict[str, str]:
        env = {self.code_env: self.code_version()}
        if self.kind == "sandbox" and self.platform == "V2":
            env["SRL_SANDBOX_PREWARM"] = "1"
        return env

    def description(self) -> str:
        what = "environment sandbox" if self.kind == "sandbox" else f"MTRL agent of templates/{self.scenario}/"
        return f"Serverless RL InstantStart {what}, code {self.code_version()}"


def sandbox_target(platform: str = "V2") -> Target:
    return Target(id="sandbox", kind="sandbox", runtime_name=settings.SANDBOX_RUNTIME,
                  repository=settings.SANDBOX_REPOSITORY, role_name=settings.SANDBOX_ROLE,
                  context=settings.TEMPLATES_DIR, files=SANDBOX_FILES, dockerfile="_sandbox/Dockerfile",
                  code_env=SANDBOX_CODE_ENV, platform=platform, lifecycle=SANDBOX_LIFECYCLE,
                  warm=SANDBOX_WARM if platform == "V2" else 0)


def agent_target(scenario_id: str, agent: dict) -> Target:
    return Target(id=f"agent-{scenario_id}", kind="agent", scenario=scenario_id, runtime_name=agent["runtime"],
                  repository=agent["repository"], role_name=agent["role"],
                  context=settings.TEMPLATES_DIR / scenario_id, files=tuple(agent["context"]),
                  dockerfile=agent["dockerfile"], code_env=AGENT_CODE_ENV, smoke_test=agent.get("smoke_test"))


def agent_targets() -> list[Target]:
    """The agents MTRL scenarios declare, read from templates/ (the registry checked them at start-up)."""
    out = []
    for meta_path in sorted(settings.TEMPLATES_DIR.glob("*/scenario.json")):
        if meta_path.parent.name.startswith("_"):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "planned" and meta.get("agent"):
            out.append(agent_target(meta["id"], meta["agent"]))
    return out


def targets() -> list[Target]:
    return [sandbox_target(), *agent_targets()]


def target(target_id: str) -> Target:
    for t in targets():
        if t.id == target_id:
            return t
    raise NotFound(f"unknown deploy target {target_id!r}; known: {', '.join(t.id for t in targets())}")


def agent_target_for_runtime(runtime_name: str | None) -> Target | None:
    return next((t for t in agent_targets() if t.runtime_name == runtime_name), None)


# -- jobs ----------------------------------------------------------------------------------

class Job:
    """One deploy or delete: its steps, their state and timing, and a log. Saved after every change."""

    def __init__(self, t: Target, action: str, region: str, path: Path, echo=None):
        self.target, self.action, self.region = t, action, region
        self.path, self.log_path = path, path.with_suffix(".log")
        self.echo = echo
        self._lock = threading.Lock()
        self.data = {"id": f"{t.id}-{time.strftime('%y%m%d-%H%M%S')}", "target": t.id, "action": action,
                     "region": region, "state": "running", "started": time.time(), "ended": None, "error": None,
                     "result": {}, "steps": [{"id": s, "state": "pending", "started": None, "ended": None,
                                              "detail": None} for s in STEPS[(t.kind, action)]]}
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")
        self._save()

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def log(self, line: str) -> None:
        stamped = f"{time.strftime('%H:%M:%S')} {line}"
        with self._lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(stamped + "\n")
        if self.echo:
            self.echo(stamped)

    def _step(self, sid: str) -> dict:
        return next(s for s in self.data["steps"] if s["id"] == sid)

    def detail(self, sid: str, text: str) -> None:
        with self._lock:
            self._step(sid)["detail"] = text
            self._save()
        self.log(f"[{sid}] {text}")

    @contextmanager
    def step(self, sid: str):
        with self._lock:
            s = self._step(sid)
            s.update(state="running", started=time.time())
            self._save()
        try:
            yield
        except BaseException as e:
            with self._lock:
                s.update(state="failed", ended=time.time())
                if not s["detail"]:
                    s["detail"] = _err(e) if not isinstance(e, DeployError) else str(e)
                self._save()
            raise
        with self._lock:
            s.update(state="skipped" if s["state"] == "skipped" else "done", ended=time.time())
            self._save()

    def skip(self, sid: str, why: str) -> None:
        with self._lock:
            self._step(sid).update(state="skipped", detail=why)
            self._save()
        self.log(f"[{sid}] skipped: {why}")

    def finish(self, error: str | None = None) -> None:
        with self._lock:
            self.data.update(state="failed" if error else "succeeded", ended=time.time(), error=error)
            self._save()
        self.log(f"{self.action} {'failed: ' + error if error else 'succeeded'}")

    def result(self, **kw) -> None:
        with self._lock:
            self.data["result"].update(kw)
            self._save()

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self.data))


class DeployJobs:
    """The latest job per target (one at a time per target), kept in data/deploys/<target>.json and .log."""

    def __init__(self, data_dir: Path):
        self.dir = data_dir / "deploys"
        self._lock = threading.Lock()
        self._running: dict[str, Job] = {}

    def latest(self, target_id: str) -> dict | None:
        with self._lock:
            job = self._running.get(target_id)
        if job:
            return job.snapshot()
        path = self.dir / f"{target_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["state"] == "running":  # its thread is gone: the app restarted during the job
            data["state"] = "interrupted"
            data["error"] = ("the app restarted while this job ran; AWS may have finished the step that was running "
                             "(see the runtime's state). Run it again.")
            for s in data["steps"]:
                if s["state"] == "running":
                    s.update(state="failed", detail=(s["detail"] + "; " if s["detail"] else "") + "interrupted")
        return data

    def log(self, target_id: str, tail: int = 400) -> list[str]:
        path = self.dir / f"{target_id}.log"
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8").splitlines()[-tail:]

    def start(self, t: Target, action: str, ctx: AwsContext, blocking_runs: list[str], after=None) -> dict:
        if action not in ("deploy", "delete"):
            raise Invalid(f"action must be deploy or delete, got {action!r}")
        if not ctx.ready:
            raise ApiError(f"No AWS identity: {ctx.identity_error}")
        if blocking_runs:
            raise ApiError(f"{', '.join(blocking_runs)} {'uses' if len(blocking_runs) == 1 else 'use'} "
                           f"{t.runtime_name}: stop {'it' if len(blocking_runs) == 1 else 'them'} or wait until "
                           f"{'it ends' if len(blocking_runs) == 1 else 'they end'}, then {action} again. A new "
                           "version would answer the run's next sessions with other code.")
        with self._lock:
            if t.id in self._running:
                raise ApiError(f"a {self._running[t.id].action} of {t.id} is already running")
            job = Job(t, action, ctx.region, self.dir / f"{t.id}.json")
            self._running[t.id] = job

        def work():
            try:
                run(job, ctx)
            finally:
                with self._lock:
                    self._running.pop(t.id, None)
                if after:
                    after()
        threading.Thread(target=work, name=f"deploy-{t.id}", daemon=True).start()
        return job.snapshot()


def run(job: Job, ctx: AwsContext, build_image=None) -> None:
    """Do the job's steps in order; the first failure ends it. build_image replaces CodeBuild (the CLI's --build local)."""
    try:
        if job.action == "deploy":
            _deploy(job, ctx, build_image or build_with_codebuild)
        else:
            _delete(job, ctx)
        job.finish()
    except Exception as e:  # recorded in the job, which is what the Setup page shows
        job.finish(str(e) if isinstance(e, DeployError) else _err(e))


# -- deploy --------------------------------------------------------------------------------

def _deploy(job: Job, ctx: AwsContext, build_image) -> None:
    t = job.target
    job.log(f"deploy {t.id} to {ctx.account} {ctx.region} as {ctx.caller_arn}: runtime {t.runtime_name}, "
            f"code {t.code_version()}")
    if t.kind == "sandbox":  # before anything is created: the usage logs' names must not be another runtime's
        _usage_source_owner(client("logs", ctx), t)
    with job.step("role"):
        role_arn = ensure_runtime_role(ctx, t, job)
    with job.step("builder"):
        if build_image is build_with_codebuild:
            ensure_builder(ctx, job)
        else:
            job.skip("builder", "the image is built on this machine")
    with job.step("image"):
        image = build_image(ctx, t, job)
        job.result(image=image)
    with job.step("runtime"):
        rt = deploy_runtime(ctx, t, role_arn, image, job)
        job.result(runtime_arn=rt["agentRuntimeArn"], version=rt.get("agentRuntimeVersion"))
    if t.kind == "sandbox":
        with job.step("usage_logs"):
            ensure_usage_logs(ctx, rt["agentRuntimeArn"], t, job)
        with job.step("warm"):
            if t.warm:
                warm(ctx, rt["agentRuntimeArn"], t, t.warm, job)
            else:
                job.skip("warm", f"platform {t.platform} has no snapshot to warm")
    else:
        with job.step("smoke"):
            if t.smoke_test:
                smoke(ctx, t, rt["agentRuntimeArn"], job)
            else:
                job.skip("smoke", "the scenario names no smoke test")
    with job.step("select"):
        select(ctx, t, rt["agentRuntimeArn"], job)


def ensure_runtime_role(ctx: AwsContext, t: Target, job: Job) -> str:
    """The runtime's execution role, trusted by AgentCore for this account only.

    The sandbox's role may pull its image and write the runtime's logs, nothing else: code in the sandbox can
    obtain its credentials (measured 2026-11-10). An agent's role also calls the Job Runtime Service
    (AmazonSageMakerJobRuntimeAccess: Sample, CompleteRollout, UpdateReward) and sends telemetry.
    """
    iam = client("iam", ctx)
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"}, "Action": "sts:AssumeRole",
        "Condition": {"StringEquals": {"aws:SourceAccount": ctx.account}}}]}
    statements = [
        {"Sid": "EcrAuth", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
        {"Sid": "EcrPull", "Effect": "Allow", "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
         "Resource": f"arn:aws:ecr:{ctx.region}:{ctx.account}:repository/{t.repository}"},
        {"Sid": "Logs", "Effect": "Allow",
         "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
         "Resource": f"arn:aws:logs:{ctx.region}:{ctx.account}:log-group:/aws/bedrock-agentcore/runtimes/*"},
        {"Sid": "LogGroups", "Effect": "Allow", "Action": "logs:DescribeLogGroups", "Resource": "*"}]
    if t.kind == "agent":
        statements.append({"Sid": "Telemetry", "Effect": "Allow", "Resource": "*",
                           "Action": ["xray:PutTraceSegments", "xray:PutTelemetryRecords", "cloudwatch:PutMetricData"]})
    try:
        arn = iam.get_role(RoleName=t.role_name)["Role"]["Arn"]
        iam.update_assume_role_policy(RoleName=t.role_name, PolicyDocument=json.dumps(trust))
        job.detail("role", f"{t.role_name} exists; trust and policy updated")
        created = False
    except ClientError as e:
        if _code(e) != "NoSuchEntity":
            raise
        what = "environment sandbox" if t.kind == "sandbox" else "MTRL agent"
        arn = iam.create_role(RoleName=t.role_name, AssumeRolePolicyDocument=json.dumps(trust),
                              Description=f"Serverless RL InstantStart {what} (AgentCore Runtime execution role)"
                              )["Role"]["Arn"]
        created = True
    policy_name = "SandboxRuntimeMinimal" if t.kind == "sandbox" else "ECRAndLogs"
    iam.put_role_policy(RoleName=t.role_name, PolicyName=policy_name,
                        PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": statements}))
    if t.kind == "agent":
        iam.attach_role_policy(RoleName=t.role_name, PolicyArn=AGENT_RUNTIME_POLICY)
    if created:
        job.detail("role", f"{t.role_name} created; waiting 15 s for IAM to propagate")
        time.sleep(15)
    return arn


def _repositories() -> list[str]:
    return sorted({t.repository for t in targets()})


def ensure_builder(ctx: AwsContext, job: Job) -> None:
    """The CodeBuild service role and project that build every target's image (ARM64, privileged for docker)."""
    iam, cb = client("iam", ctx), client("codebuild", ctx)
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "codebuild.amazonaws.com"}, "Action": "sts:AssumeRole",
        "Condition": {"StringEquals": {"aws:SourceAccount": ctx.account}}}]}
    group = f"/aws/codebuild/{settings.IMAGE_BUILD_PROJECT}"
    policy = {"Version": "2012-10-17", "Statement": [
        {"Sid": "Logs", "Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
         "Resource": [f"arn:aws:logs:{ctx.region}:{ctx.account}:log-group:{group}",
                      f"arn:aws:logs:{ctx.region}:{ctx.account}:log-group:{group}:*"]},
        {"Sid": "Source", "Effect": "Allow", "Action": ["s3:GetObject", "s3:GetObjectVersion"],
         "Resource": f"arn:aws:s3:::{ctx.bucket}/{_build_prefix()}/*"},
        {"Sid": "SourceBucket", "Effect": "Allow", "Action": ["s3:GetBucketAcl", "s3:GetBucketLocation"],
         "Resource": f"arn:aws:s3:::{ctx.bucket}"},
        {"Sid": "EcrAuth", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
        {"Sid": "EcrPush", "Effect": "Allow",
         "Action": ["ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
                    "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
         "Resource": [f"arn:aws:ecr:{ctx.region}:{ctx.account}:repository/{r}" for r in _repositories()]}]}
    try:
        role_arn = iam.get_role(RoleName=settings.IMAGE_BUILD_ROLE)["Role"]["Arn"]
        iam.update_assume_role_policy(RoleName=settings.IMAGE_BUILD_ROLE, PolicyDocument=json.dumps(trust))
        created = False
    except ClientError as e:
        if _code(e) != "NoSuchEntity":
            raise
        role_arn = iam.create_role(RoleName=settings.IMAGE_BUILD_ROLE, AssumeRolePolicyDocument=json.dumps(trust),
                                   Description="Serverless RL InstantStart: builds AgentCore Runtime images in CodeBuild"
                                   )["Role"]["Arn"]
        created = True
    iam.put_role_policy(RoleName=settings.IMAGE_BUILD_ROLE, PolicyName="ImageBuild", PolicyDocument=json.dumps(policy))
    project = dict(
        name=settings.IMAGE_BUILD_PROJECT,
        description="Serverless RL InstantStart: builds the ARM64 images of AgentCore runtimes (Setup page)",
        # Each build brings its source (a zip in S3) and its buildspec; the project only fixes the host.
        source={"type": "NO_SOURCE", "buildspec": "version: 0.2\nphases:\n  build:\n    commands:\n"
                "      - echo 'start builds from the Setup page: they pass their own buildspec' && exit 1\n"},
        artifacts={"type": "NO_ARTIFACTS"},
        environment={"type": "ARM_CONTAINER", "image": BUILD_IMAGE, "computeType": BUILD_COMPUTE,
                     "privilegedMode": True, "imagePullCredentialsType": "CODEBUILD"},
        serviceRole=role_arn, timeoutInMinutes=BUILD_TIMEOUT_MIN, queuedTimeoutInMinutes=BUILD_TIMEOUT_MIN,
        logsConfig={"cloudWatchLogs": {"status": "ENABLED", "groupName": group}})
    exists = bool(cb.batch_get_projects(names=[settings.IMAGE_BUILD_PROJECT]).get("projects"))
    # A role created moments ago is not assumable by CodeBuild yet: it says so, and a retry succeeds.
    for attempt in range(13):
        try:
            (cb.update_project if exists else cb.create_project)(**project)
            break
        except ClientError as e:
            if _code(e) != "InvalidInputException" or "role" not in str(e).lower() or attempt == 12:
                raise
            time.sleep(5)
    job.detail("builder", f"CodeBuild project {settings.IMAGE_BUILD_PROJECT} ({'updated' if exists else 'created'}, "
               f"{BUILD_IMAGE}, {BUILD_COMPUTE}); role {settings.IMAGE_BUILD_ROLE} "
               f"({'created' if created else 'updated'})")


def _build_prefix() -> str:
    return f"{settings.S3_PREFIX}/image-builds"


def _ensure_repository(ecr, t: Target, job: Job) -> None:
    try:
        ecr.create_repository(repositoryName=t.repository)
        job.log(f"ECR repository {t.repository} created")
    except ClientError as e:
        if _code(e) != "RepositoryAlreadyExistsException":
            raise


def image_uri(ctx: AwsContext, t: Target) -> str:
    return f"{ctx.account}.dkr.ecr.{ctx.region}.amazonaws.com/{t.repository}:{t.image_tag()}"


def pushed(ecr, t: Target, tag: str) -> dict | None:
    """The ECR image detail of this tag, or None."""
    try:
        return ecr.describe_images(repositoryName=t.repository, imageIds=[{"imageTag": tag}])["imageDetails"][0]
    except ClientError as e:
        if _code(e) in ("ImageNotFoundException", "RepositoryNotFoundException"):
            return None
        raise


def docker_library_refs(dockerfile: str) -> list[str]:
    """Docker Hub official images a Dockerfile starts FROM (python:3.13-slim), which the build pulls from
    the ECR Public mirror instead: CodeBuild hosts share their addresses, and Docker Hub limits anonymous pulls
    per address. Images with a registry or a namespace, stage names and ARG references are left alone."""
    refs, stages = [], set()
    for line in dockerfile.splitlines():
        words = line.strip().split()
        if len(words) < 2 or words[0].upper() != "FROM":
            continue
        args = [w for w in words[1:] if not w.startswith("--")]
        if not args:
            continue
        ref = args[0]
        if len(args) >= 3 and args[1].upper() == "AS":
            stages.add(args[2].lower())
        if ref.lower() in stages or "$" in ref or "/" in ref or ref == "scratch":
            continue
        refs.append(ref if ":" in ref or "@" in ref else f"{ref}:latest")
    return list(dict.fromkeys(refs))


def buildspec(ctx: AwsContext, image: str, dockerfile: str, mirror: list[str]) -> str:
    """Every command on its own line (CodeBuild fails the phase on the first non-zero exit); each is a JSON string,
    which YAML reads as a double-quoted scalar."""
    registry = image.split("/", 1)[0]
    pre = ["docker version", f"aws ecr get-login-password --region {ctx.region} | "
                             f"docker login --username AWS --password-stdin {registry}"]
    for ref in mirror:
        pre += [f"docker pull --quiet {DOCKER_LIBRARY_MIRROR}/{ref}", f"docker tag {DOCKER_LIBRARY_MIRROR}/{ref} {ref}"]
    build = [f"docker build --platform linux/arm64 -f {dockerfile} -t {image} .", f"docker push {image}"]
    lines = ["version: 0.2", "phases:", "  pre_build:", "    commands:", *[f"      - {json.dumps(c)}" for c in pre],
             "  build:", "    commands:", *[f"      - {json.dumps(c)}" for c in build]]
    return "\n".join(lines) + "\n"


def context_zip(t: Target) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, path in t.build_files():
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
    return buf.getvalue()


def build_with_codebuild(ctx: AwsContext, t: Target, job: Job) -> str:
    """Build and push the target's image in CodeBuild, unless ECR has its tag already. Returns the image URI."""
    ecr = client("ecr", ctx)
    _ensure_repository(ecr, t, job)
    tag, image = t.image_tag(), image_uri(ctx, t)
    if pushed(ecr, t, tag):
        job.detail("image", f"{t.repository}:{tag} is already in ECR (same build files); not built again")
        return image
    s3 = client("s3", ctx)
    try:
        s3.head_bucket(Bucket=ctx.bucket, ExpectedBucketOwner=ctx.account)
    except ClientError as e:
        raise DeployError(f"output bucket {ctx.bucket}: {_err(e)}. The build context is uploaded there; create it "
                          "with the Output bucket check of the Setup page first.") from None
    body = context_zip(t)
    key = f"{_build_prefix()}/{t.repository}/{tag}.zip"
    s3.put_object(Bucket=ctx.bucket, Key=key, Body=body, ExpectedBucketOwner=ctx.account)
    mirror = docker_library_refs((t.context / t.dockerfile).read_text(encoding="utf-8"))
    job.log(f"build context s3://{ctx.bucket}/{key} ({len(body) / 1024:.0f} KiB, {len(t.build_files())} files)"
            + (f"; base images from {DOCKER_LIBRARY_MIRROR}: {', '.join(mirror)}" if mirror else ""))
    cb = client("codebuild", ctx)
    build = cb.start_build(projectName=settings.IMAGE_BUILD_PROJECT, sourceTypeOverride="S3",
                           sourceLocationOverride=f"{ctx.bucket}/{key}",
                           buildspecOverride=buildspec(ctx, image, t.dockerfile, mirror))["build"]
    job.result(build_id=build["id"])
    job.detail("image", f"CodeBuild {build['id']}: {build.get('currentPhase')}")
    logs, token, t0 = client("logs", ctx), None, time.time()
    while True:
        build = cb.batch_get_builds(ids=[build["id"]])["builds"][0]
        token = _tail_build_log(logs, build, token, job)
        if build["buildStatus"] != "IN_PROGRESS":
            break
        job.detail("image", f"CodeBuild {build['id'].split(':')[-1]}: {build.get('currentPhase')}, "
                   f"{time.time() - t0:.0f} s")
        time.sleep(BUILD_POLL_S)
    time.sleep(2)
    _tail_build_log(logs, build, token, job)
    if build["buildStatus"] != "SUCCEEDED":
        failed = [p for p in build.get("phases", []) if p.get("phaseStatus") not in (None, "SUCCEEDED")]
        why = "; ".join(f"{p['phaseType']} {p['phaseStatus']}: " + " ".join(c.get("message", "") for c in
                                                                             p.get("contexts", [])).strip()
                        for p in failed) or build["buildStatus"]
        raise DeployError(f"CodeBuild build {build['id']} {build['buildStatus']}: {why}. The build log is below "
                          f"and in CloudWatch Logs /aws/codebuild/{settings.IMAGE_BUILD_PROJECT}.")
    detail = pushed(ecr, t, tag)
    if not detail:
        raise DeployError(f"CodeBuild build {build['id']} succeeded, but ECR has no {t.repository}:{tag}")
    media = detail.get("imageManifestMediaType", "")
    if "index" in media or "manifest.list" in media:
        raise DeployError(f"{t.repository}:{tag} was pushed as {media} (a multi-platform index); AgentCore Runtime "
                          "needs a single linux/arm64 image")
    job.detail("image", f"{t.repository}:{tag} pushed by CodeBuild in {time.time() - t0:.0f} s, "
               f"{detail.get('imageSizeInBytes', 0) / 2**20:.1f} MiB, {media}")
    return image


def _tail_build_log(logs, build: dict, token: str | None, job: Job) -> str | None:
    info = build.get("logs") or {}
    group, stream = info.get("groupName"), info.get("streamName")
    if not group or not stream:
        return token
    while True:
        kw = {"logGroupName": group, "logStreamName": stream, "startFromHead": True, **({"nextToken": token} if token else {})}
        try:
            r = logs.get_log_events(**kw)
        except ClientError as e:
            if _code(e) == "ResourceNotFoundException":  # the stream appears a little after the build starts
                return token
            raise
        for ev in r.get("events", []):
            job.log("  | " + ev["message"].rstrip("\n"))
        if r.get("nextForwardToken") == token:
            return token
        token = r.get("nextForwardToken")


def find_runtime(ctx: AwsContext, name: str) -> dict | None:
    ctl, token = client("bedrock-agentcore-control", ctx), None
    while True:
        r = ctl.list_agent_runtimes(**({"nextToken": token} if token else {}))
        for rt in r.get("agentRuntimes", []):
            if rt["agentRuntimeName"] == name:
                return rt
        token = r.get("nextToken")
        if not token:
            return None


def runtime_request(t: Target, role_arn: str, image: str, existing: dict | None) -> dict:
    """CreateAgentRuntime / UpdateAgentRuntime arguments: the target's settings over the existing runtime's."""
    kw = {k: existing[k] for k in KEEP_ON_UPDATE if existing and existing.get(k)}
    env = dict((existing or {}).get("environmentVariables") or {})
    env.update(t.env())
    kw.update(agentRuntimeArtifact={"containerConfiguration": {"containerUri": image}}, roleArn=role_arn,
              environmentVariables=env, description=t.description())
    kw.setdefault("networkConfiguration", {"networkMode": "PUBLIC"})
    if t.platform:
        kw["platformVersion"] = t.platform
    if t.lifecycle:
        kw["lifecycleConfiguration"] = t.lifecycle
    return kw


def deploy_runtime(ctx: AwsContext, t: Target, role_arn: str, image: str, job: Job) -> dict:
    ctl = client("bedrock-agentcore-control", ctx)
    found = find_runtime(ctx, t.runtime_name)
    existing = ctl.get_agent_runtime(agentRuntimeId=found["agentRuntimeId"]) if found else None
    if existing and existing.get("status") not in ("READY", "CREATE_FAILED", "UPDATE_FAILED"):
        raise DeployError(f"runtime {t.runtime_name} is {existing.get('status')}; wait until it is READY and deploy again")
    kw = runtime_request(t, role_arn, image, existing)
    if existing:
        before = existing.get("agentRuntimeVersion")
        r = ctl.update_agent_runtime(agentRuntimeId=existing["agentRuntimeId"], **kw)
        job.detail("runtime", f"{t.runtime_name}: update from version {before} requested")
    else:
        before = None
        r = None
        for attempt in range(7):  # a role created moments ago may not be assumable yet
            try:
                r = ctl.create_agent_runtime(agentRuntimeName=t.runtime_name, **kw)
                break
            except ClientError as e:
                if _code(e) != "ValidationException" or "role" not in str(e).lower() or attempt == 6:
                    raise
                time.sleep(10)
        job.detail("runtime", f"{t.runtime_name}: create requested")
    rid, t0 = r["agentRuntimeId"], time.time()
    while True:
        g = ctl.get_agent_runtime(agentRuntimeId=rid)
        if g["status"] == "READY" or g["status"].endswith("FAILED"):
            break
        if time.time() - t0 > READY_TIMEOUT_S:
            raise DeployError(f"runtime {t.runtime_name} still {g['status']} after {READY_TIMEOUT_S} s")
        job.detail("runtime", f"{t.runtime_name}: {g['status']}, {time.time() - t0:.0f} s")
        time.sleep(5)
    if g["status"] != "READY":
        raise DeployError(f"runtime {t.runtime_name} is {g['status']}: {g.get('failureReason')}")
    # Sessions go to the DEFAULT endpoint, which moves to the new version on its own: wait for it.
    version = g.get("agentRuntimeVersion")
    while True:
        e = ctl.get_agent_runtime_endpoint(agentRuntimeId=rid, endpointName="DEFAULT")
        if e.get("status") == "READY" and str(e.get("liveVersion")) == str(version):
            break
        if e.get("status", "").endswith("FAILED") or time.time() - t0 > READY_TIMEOUT_S:
            raise DeployError(f"runtime {t.runtime_name} is READY at version {version}, but its DEFAULT endpoint is "
                              f"{e.get('status')} at version {e.get('liveVersion')}: {e.get('failureReason')}")
        job.detail("runtime", f"{t.runtime_name}: DEFAULT endpoint {e.get('status')}, version {e.get('liveVersion')} "
                   f"-> {version}")
        time.sleep(5)
    job.detail("runtime", f"{t.runtime_name}: READY in {time.time() - t0:.0f} s, version "
               f"{(before + ' -> ') if before else ''}{version}, platform {g.get('platformVersion') or 'V1'}")
    return g


def _usage_source_owner(logs, t: Target) -> list[str] | None:
    """The runtime ARNs the sandbox's delivery source points at (None: there is none); refuses another runtime's."""
    try:
        arns = logs.get_delivery_source(name=settings.SANDBOX_USAGE_SOURCE)["deliverySource"].get("resourceArns") or []
    except ClientError as e:
        if _code(e) == "ResourceNotFoundException":
            return None
        raise
    foreign = [a for a in arns if not a.rsplit("/", 1)[-1].startswith(t.runtime_name + "-")]
    if foreign:
        raise DeployError(f"delivery source {settings.SANDBOX_USAGE_SOURCE} belongs to {', '.join(foreign)}, not to "
                          f"{t.runtime_name}; it is left alone. Give this sandbox its own with SRL_SANDBOX_USAGE_SOURCE "
                          "and SRL_SANDBOX_USAGE_DEST.")
    return arns


def ensure_usage_logs(ctx: AwsContext, runtime_arn: str, t: Target, job: Job) -> None:
    """USAGE_LOGS (per session, per second) to the log group the run console reads a run's AgentCore cost from."""
    logs, group = client("logs", ctx), settings.SANDBOX_USAGE_GROUP
    _usage_source_owner(logs, t)
    try:
        logs.create_log_group(logGroupName=group)
    except ClientError as e:
        if _code(e) != "ResourceAlreadyExistsException":
            raise
    dest = logs.put_delivery_destination(name=settings.SANDBOX_USAGE_DEST, deliveryDestinationConfiguration={
        "destinationResourceArn": f"arn:aws:logs:{ctx.region}:{ctx.account}:log-group:{group}"})
    logs.put_delivery_source(name=settings.SANDBOX_USAGE_SOURCE, resourceArn=runtime_arn, logType="USAGE_LOGS")
    dest_arn = dest["deliveryDestination"]["arn"]
    # CreateDelivery answers an existing source -> destination pair with success, not a conflict (2026-11-18):
    # look first, so the job says which it was.
    existing = [d["id"] for page in logs.get_paginator("describe_deliveries").paginate()
                for d in page.get("deliveries", []) if d.get("deliverySourceName") == settings.SANDBOX_USAGE_SOURCE
                and d.get("deliveryDestinationArn") == dest_arn]
    if existing:
        job.detail("usage_logs", f"USAGE_LOGS delivery {existing[0]} exists -> {group}")
        return
    try:
        logs.create_delivery(deliverySourceName=settings.SANDBOX_USAGE_SOURCE, deliveryDestinationArn=dest_arn)
        job.detail("usage_logs", f"USAGE_LOGS delivery created -> {group}")
    except ClientError as e:
        if _code(e) != "ConflictException":
            raise
        job.detail("usage_logs", f"USAGE_LOGS delivery exists -> {group}")


def _invoke(agentcore, runtime_arn: str, sid: str, payload: dict) -> dict:
    r = agentcore.invoke_agent_runtime(agentRuntimeArn=runtime_arn, qualifier="DEFAULT", runtimeSessionId=sid,
                                       contentType="application/json", accept="application/json",
                                       payload=json.dumps(payload).encode())
    return json.loads(r["response"].read())


def warm(ctx: AwsContext, runtime_arn: str, t: Target, n: int, job: Job) -> None:
    """Open n sessions, run Python once in each, stop them; every answer must come from this checkout's code."""
    import boto3
    from botocore.config import Config
    agentcore = boto3.client("bedrock-agentcore", region_name=ctx.region, config=Config(
        read_timeout=90, connect_timeout=15, max_pool_connections=32, retries={"total_max_attempts": 1}))
    expected, prefix = t.code_version(), f"srl-deploy-warm-{time.strftime('%y%m%d%H%M%S')}"

    def one(i: int):
        sid = f"{prefix}-{i:03d}-{secrets.token_hex(6)}"
        t0 = time.monotonic()
        try:
            for attempt in range(6):
                try:
                    a = _invoke(agentcore, runtime_arn, sid, {"op": "reset", "seed": i, "files": {"w.py": "print('warm')"}})
                    break
                except ClientError as e:  # throttled while ramping up: back off and retry
                    if _code(e) != "ThrottlingException" or attempt == 5:
                        raise
                    time.sleep(0.5 * 2 ** attempt)
            cold = time.monotonic() - t0
            r = _invoke(agentcore, runtime_arn, sid, {"op": "run", "argv": ["{python}", "w.py"], "timeout_s": 30})
            if a.get("code_sha") != expected or not r.get("ok") or r["result"]["exit_code"] != 0:
                raise DeployError(f"warm-up session {sid}: expected sandbox_exec {expected} to run w.py, got "
                                  f"{json.dumps(a)[:300]} / {json.dumps(r)[:300]}")
            return cold, r["result"]["sec"]
        finally:
            try:
                agentcore.stop_runtime_session(agentRuntimeArn=runtime_arn, qualifier="DEFAULT", runtimeSessionId=sid)
            except Exception as e:  # noqa: BLE001 - reported; the idle timeout ends it
                job.log(f"warm-up session {sid} not stopped: {_err(e)}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = []
        for i in range(n):
            futures.append(pool.submit(one, i))
            time.sleep(1 / 20)  # at most 20 new sessions per second, as runs do
        got = [f.result() for f in futures]
    colds, runs = sorted(c for c, _ in got), sorted(r for _, r in got)
    job.detail("warm", f"{n} sessions in {time.time() - t0:.0f} s, each answered by sandbox_exec {expected}: first call "
               f"p50 {colds[n // 2]:.2f} s (max {colds[-1]:.2f}), first program p50 {runs[n // 2]:.2f} s "
               f"(max {runs[-1]:.2f})")


def smoke(ctx: AwsContext, t: Target, runtime_arn: str, job: Job) -> None:
    """The scenario's own smoke test against the new runtime: exit status 0 is a pass."""
    script = t.context / t.smoke_test
    if not script.is_file():
        raise DeployError(f"{script}: the scenario's smoke_test does not exist")
    job.detail("smoke", f"{t.smoke_test} --runtime-arn {runtime_arn}")
    try:
        p = subprocess.run([sys.executable, str(script), "--runtime-arn", runtime_arn, "--region", ctx.region],
                           cwd=t.context, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise DeployError(f"{t.smoke_test} did not finish in 600 s") from None
    for line in (p.stdout + p.stderr).splitlines():
        job.log("  | " + line)
    if p.returncode != 0:
        tail = " / ".join(x for x in (p.stdout + p.stderr).strip().splitlines()[-3:])
        raise DeployError(f"{t.smoke_test} exited {p.returncode}: {tail}")
    job.detail("smoke", f"{t.smoke_test} passed")


def select(ctx: AwsContext, t: Target, runtime_arn: str, job: Job) -> None:
    """Point the Setup selection at the runtime just deployed: that is what the button was pressed for."""
    from . import mtrl, sandbox
    section = "sandbox" if t.kind == "sandbox" else "mtrl"
    previous = load_settings().get(section, {}).get(ctx.region, {}).get("runtime_arn")
    if t.kind == "sandbox":
        sandbox.set_selection(ctx, runtime_arn)
    else:
        mtrl.set_selection(ctx, runtime_arn=runtime_arn)
    job.detail("select", f"selected {runtime_arn}" + (f" (was {previous})" if previous and previous != runtime_arn else ""))


# -- delete --------------------------------------------------------------------------------

def _delete(job: Job, ctx: AwsContext) -> None:
    t = job.target
    job.log(f"delete {t.id} from {ctx.account} {ctx.region} as {ctx.caller_arn}")
    found = find_runtime(ctx, t.runtime_name)
    if t.kind == "sandbox":
        with job.step("usage_logs"):
            delete_usage_logs(ctx, t, job)
    with job.step("runtime"):
        if found:
            delete_runtime(ctx, found, job)
        else:
            job.skip("runtime", f"no runtime named {t.runtime_name}")
    with job.step("repository"):
        try:
            client("ecr", ctx).delete_repository(repositoryName=t.repository, force=True)
            job.detail("repository", f"ECR repository {t.repository} and its images deleted")
        except ClientError as e:
            if _code(e) != "RepositoryNotFoundException":
                raise
            job.skip("repository", f"no ECR repository {t.repository}")
    with job.step("role"):
        if not delete_role(ctx, t.role_name):
            job.skip("role", f"no role {t.role_name}")
        else:
            job.detail("role", f"{t.role_name} deleted")
    with job.step("builder"):
        others = [o.runtime_name for o in targets() if o.id != t.id and find_runtime(ctx, o.runtime_name)]
        if others:
            job.skip("builder", f"kept: it also builds {', '.join(others)}")
        else:
            delete_builder(ctx, job)
    with job.step("sources"):
        n = delete_sources(ctx, t)
        job.detail("sources", f"{n} build context zip{'s' if n != 1 else ''} deleted from "
                   f"s3://{ctx.bucket}/{_build_prefix()}/{t.repository}/")
    with job.step("select"):
        cleared = clear_selection(ctx, t, found["agentRuntimeArn"] if found else None)
        if cleared:
            job.detail("select", f"cleared the Setup selection of {found['agentRuntimeArn']}")
        else:
            job.skip("select", "the Setup selection was another runtime")


def delete_usage_logs(ctx: AwsContext, t: Target, job: Job) -> None:
    """The delivery, its source and destination. The log group stays: past runs' AgentCore costs are read from it."""
    logs = client("logs", ctx)
    if _usage_source_owner(logs, t) is None:
        job.skip("usage_logs", f"no delivery source {settings.SANDBOX_USAGE_SOURCE}")
        return
    for page in logs.get_paginator("describe_deliveries").paginate():
        for d in page.get("deliveries", []):
            if d.get("deliverySourceName") == settings.SANDBOX_USAGE_SOURCE:
                logs.delete_delivery(id=d["id"])
    for name, fn in ((settings.SANDBOX_USAGE_SOURCE, logs.delete_delivery_source),
                     (settings.SANDBOX_USAGE_DEST, logs.delete_delivery_destination)):
        try:
            fn(name=name)
        except ClientError as e:
            if _code(e) != "ResourceNotFoundException":
                raise
    job.detail("usage_logs", f"USAGE_LOGS delivery removed; log group {settings.SANDBOX_USAGE_GROUP} kept "
               "(past runs' costs)")


def delete_runtime(ctx: AwsContext, found: dict, job: Job) -> None:
    ctl = client("bedrock-agentcore-control", ctx)
    ctl.delete_agent_runtime(agentRuntimeId=found["agentRuntimeId"])
    t0 = time.time()
    while True:
        try:
            g = ctl.get_agent_runtime(agentRuntimeId=found["agentRuntimeId"])
        except ClientError as e:
            if _code(e) == "ResourceNotFoundException":
                break
            raise
        if time.time() - t0 > READY_TIMEOUT_S:
            raise DeployError(f"runtime {found['agentRuntimeName']} still {g.get('status')} after {READY_TIMEOUT_S} s")
        job.detail("runtime", f"{found['agentRuntimeName']}: {g.get('status')}, {time.time() - t0:.0f} s")
        time.sleep(5)
    job.detail("runtime", f"{found['agentRuntimeName']} deleted in {time.time() - t0:.0f} s")


def delete_role(ctx: AwsContext, name: str) -> bool:
    iam = client("iam", ctx)
    try:
        for p in iam.list_attached_role_policies(RoleName=name)["AttachedPolicies"]:
            iam.detach_role_policy(RoleName=name, PolicyArn=p["PolicyArn"])
    except ClientError as e:
        if _code(e) == "NoSuchEntity":
            return False
        raise
    for p in iam.list_role_policies(RoleName=name)["PolicyNames"]:
        iam.delete_role_policy(RoleName=name, PolicyName=p)
    iam.delete_role(RoleName=name)
    return True


def delete_builder(ctx: AwsContext, job: Job) -> None:
    cb = client("codebuild", ctx)
    had = bool(cb.batch_get_projects(names=[settings.IMAGE_BUILD_PROJECT]).get("projects"))
    if had:
        cb.delete_project(name=settings.IMAGE_BUILD_PROJECT)
    role = delete_role(ctx, settings.IMAGE_BUILD_ROLE)
    if not had and not role:
        job.skip("builder", "no CodeBuild project or role")
        return
    job.detail("builder", f"CodeBuild project {settings.IMAGE_BUILD_PROJECT} and role {settings.IMAGE_BUILD_ROLE} "
               f"deleted (no other runtime of this app is deployed); its log group "
               f"/aws/codebuild/{settings.IMAGE_BUILD_PROJECT} is kept")


def delete_sources(ctx: AwsContext, t: Target) -> int:
    s3, n = client("s3", ctx), 0
    prefix = f"{_build_prefix()}/{t.repository}/"
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=ctx.bucket, Prefix=prefix,
                                                              ExpectedBucketOwner=ctx.account):
        keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if keys:
            s3.delete_objects(Bucket=ctx.bucket, Delete={"Objects": keys}, ExpectedBucketOwner=ctx.account)
            n += len(keys)
    return n


def clear_selection(ctx: AwsContext, t: Target, runtime_arn: str | None) -> bool:
    data = load_settings()
    region = data.get("sandbox" if t.kind == "sandbox" else "mtrl", {}).get(ctx.region, {})
    if not runtime_arn or region.get("runtime_arn") != runtime_arn:
        return False
    region.pop("runtime_arn")
    save_settings(data)
    return True


# -- what the Setup page shows -------------------------------------------------------------

def blocking_runs(t: Target, live_configs: list[dict], runtime_arn: str | None) -> list[str]:
    """Live runs still going that use the target's runtime."""
    out = []
    for c in live_configs:
        aws = c.get("aws") or {}
        used = (aws.get("sandbox") or {}).get("runtime_arn") if t.kind == "sandbox" else (aws.get("mtrl") or {}).get(
            "runtime_arn")
        if used and (used == runtime_arn or used.rsplit("/", 1)[-1].startswith(t.runtime_name + "-")):
            out.append(c["run_id"])
    return out


def permissions(t: Target) -> dict[str, list[str]]:
    out = {k: list(v) for k, v in PERMISSIONS.items()}
    if t.kind == "sandbox":
        out["logs"] += SANDBOX_PERMISSIONS["logs"]
    return out


def overview(ctx: AwsContext, jobs: DeployJobs, live_configs: list[dict]) -> list[dict]:
    """Every target: its names, the code this checkout would deploy, what is deployed, its latest job."""
    out = []
    for t in targets():
        item = {"id": t.id, "kind": t.kind, "scenario": t.scenario, "runtime_name": t.runtime_name,
                "repository": t.repository, "role_name": t.role_name, "build_project": settings.IMAGE_BUILD_PROJECT,
                "build_role": settings.IMAGE_BUILD_ROLE, "platform": t.platform, "warm": t.warm,
                "smoke_test": t.smoke_test, "permissions": permissions(t), "usage_log_group": settings.SANDBOX_USAGE_GROUP
                if t.kind == "sandbox" else None, "bucket": ctx.bucket, "region": ctx.region,
                "local_version": None, "deployed": None, "condition": None, "error": None,
                "job": jobs.latest(t.id), "blocked_by": []}
        try:
            item["local_version"] = t.code_version()
        except (OSError, DeployError) as e:
            item["error"] = f"build files: {e}"
        if not ctx.ready:
            item["error"] = item["error"] or f"No AWS identity: {ctx.identity_error}"
            out.append(item)
            continue
        try:
            found = find_runtime(ctx, t.runtime_name)
            if found:
                g = client("bedrock-agentcore-control", ctx).get_agent_runtime(agentRuntimeId=found["agentRuntimeId"])
                env = g.get("environmentVariables") or {}
                item["deployed"] = {
                    "arn": g["agentRuntimeArn"], "status": g.get("status"), "version": g.get("agentRuntimeVersion"),
                    "code_version": env.get(t.code_env), "platform": g.get("platformVersion") or "V1",
                    "image": ((g.get("agentRuntimeArtifact") or {}).get("containerConfiguration") or {}).get("containerUri"),
                    "updated": str(g.get("lastUpdatedAt"))}
            item["condition"] = condition(item["deployed"], item["local_version"])
            item["blocked_by"] = blocking_runs(t, live_configs, found["agentRuntimeArn"] if found else None)
        except Exception as e:  # shown next to the buttons; the buttons still work
            item["error"] = _err(e)
        out.append(item)
    return out


def condition(deployed: dict | None, local: str | None) -> str:
    """missing / busy / failed / current / outdated / unrecorded (deployed outside this app, version unknown)."""
    if not deployed:
        return "missing"
    status = deployed["status"] or ""
    if status.endswith("FAILED"):
        return "failed"
    if status != "READY":
        return "busy"
    if not deployed["code_version"]:
        return "unrecorded"
    return "current" if deployed["code_version"] == local else "outdated"
