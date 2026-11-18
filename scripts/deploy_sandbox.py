#!/usr/bin/env python3
"""Deploy (or update) the environment sandbox: one AgentCore Runtime that sandbox scenarios run code in.

Run it on the host, not in the app's container (it needs docker buildx for linux/arm64):

    backend/.venv/bin/python scripts/deploy_sandbox.py                # us-west-2, platform V2
    backend/.venv/bin/python scripts/deploy_sandbox.py --platform V1 --region us-east-1

Then select the runtime on the Setup page (it is found by name when nothing is selected).
Safe to re-run: existing resources are updated, and the runtime gets a new version.

What it creates, all named in docs/architecture.md 6:

- IAM role SRL-InstantStart-SandboxRuntime, trusted by bedrock-agentcore.amazonaws.com for this
  account only. It may pull the image and write the runtime's logs, nothing else: code running
  in the sandbox can obtain this role's credentials (measured 2026-11-10), so it must stay this small.
- ECR repository serverless-rl-instantstart-sandbox, with an ARM64 image of templates/_sandbox/
  and templates/_shared/sandbox_exec.py, tagged with a hash of those files.
- AgentCore Runtime srl_instantstart_sandbox: container, PUBLIC network, idle timeout 120 s, at
  most 1 h per session, platform V2 by default (cold start 1.5-2.0 s at any load, measured).
  Environment SRL_SANDBOX_CODE carries sandbox_exec.py's version, which the Setup page compares
  with the checkout's; V2 also gets SRL_SANDBOX_PREWARM=1.
- USAGE_LOGS delivery to /aws/vendedlogs/bedrock-agentcore/serverless-rl-instantstart/sandbox-usage:
  per-session, per-second usage, from which the run console shows a run's AgentCore cost.

After the runtime is READY it opens --warm sessions (default 32), runs Python once in each and
stops them. On V2 a new version's first sessions were slow (a first Python program took up to
1.4 s, a first pytest up to 13 s) until the snapshot had served about that many sessions. The
warm-up costs about $0.004 at list price.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
sys.path.insert(0, str(TEMPLATES / "_shared"))
import sandbox_exec  # noqa: E402
from sandbox_client import LiveTransport  # noqa: E402

RUNTIME = os.environ.get("SRL_SANDBOX_RUNTIME", "srl_instantstart_sandbox")
ROLE = os.environ.get("SRL_SANDBOX_ROLE", "SRL-InstantStart-SandboxRuntime")
REPO = "serverless-rl-instantstart-sandbox"
USAGE_GROUP = "/aws/vendedlogs/bedrock-agentcore/serverless-rl-instantstart/sandbox-usage"
USAGE_SOURCE = "srl-instantstart-sandbox-usage"
USAGE_DEST = "srl-instantstart-sandbox-usage-cwl"
LIFECYCLE = {"idleRuntimeSessionTimeout": 120, "maxLifetime": 3600}
IMAGE_FILES = ["_sandbox/Dockerfile", "_sandbox/Dockerfile.dockerignore", "_sandbox/requirements.txt",
               "_sandbox/main.py", "_shared/sandbox_exec.py"]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sh(cmd: list[str], **kw) -> str:
    r = subprocess.run(cmd, text=True, capture_output=True, **kw)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    return r.stdout


def image_tag() -> str:
    h = hashlib.sha256()
    for name in IMAGE_FILES:
        h.update(name.encode() + b"\0" + (TEMPLATES / name).read_bytes())
    return h.hexdigest()[:12]


def ensure_role(iam, account: str, region: str) -> str:
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"}, "Action": "sts:AssumeRole",
        "Condition": {"StringEquals": {"aws:SourceAccount": account}}}]}
    policy = {"Version": "2012-10-17", "Statement": [
        {"Sid": "EcrAuth", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
        {"Sid": "EcrPull", "Effect": "Allow", "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
         "Resource": f"arn:aws:ecr:{region}:{account}:repository/{REPO}"},
        {"Sid": "Logs", "Effect": "Allow",
         "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
         "Resource": f"arn:aws:logs:{region}:{account}:log-group:/aws/bedrock-agentcore/runtimes/*"},
        {"Sid": "LogGroups", "Effect": "Allow", "Action": "logs:DescribeLogGroups", "Resource": "*"}]}
    try:
        arn = iam.get_role(RoleName=ROLE)["Role"]["Arn"]
        iam.update_assume_role_policy(RoleName=ROLE, PolicyDocument=json.dumps(trust))
        log(f"role {ROLE} exists")
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust),
                              Description="Serverless RL InstantStart environment sandbox (AgentCore Runtime); "
                                          "code running in the sandbox gets these permissions")["Role"]["Arn"]
        log(f"role {ROLE} created; waiting 15 s for IAM to propagate")
        time.sleep(15)
    iam.put_role_policy(RoleName=ROLE, PolicyName="SandboxRuntimeMinimal", PolicyDocument=json.dumps(policy))
    return arn


def build_image(ecr, account: str, region: str, tag: str) -> str:
    try:
        ecr.create_repository(repositoryName=REPO)
        log(f"ECR repository {REPO} created")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        pass
    registry = f"{account}.dkr.ecr.{region}.amazonaws.com"
    image = f"{registry}/{REPO}:{tag}"
    try:
        ecr.describe_images(repositoryName=REPO, imageIds=[{"imageTag": tag}])
        log(f"image {REPO}:{tag} already pushed")
        return image
    except ecr.exceptions.ImageNotFoundException:
        pass
    password = ecr.get_authorization_token()["authorizationData"][0]["authorizationToken"]
    import base64
    user, secret = base64.b64decode(password).decode().split(":", 1)
    sh(["docker", "login", "--username", user, "--password-stdin", registry], input=secret)
    log(f"building linux/arm64 image {REPO}:{tag}")
    t0 = time.time()
    sh(["docker", "buildx", "build", "--platform", "linux/arm64", "--provenance=false",
        "-f", str(TEMPLATES / "_sandbox" / "Dockerfile"), "-t", image, "--push", str(TEMPLATES)], timeout=3600)
    size = ecr.describe_images(repositoryName=REPO, imageIds=[{"imageTag": tag}])["imageDetails"][0]["imageSizeInBytes"]
    log(f"image pushed in {time.time() - t0:.0f} s, {size / 2**20:.1f} MB")
    return image


def find_runtime(ctl) -> dict | None:
    token = None
    while True:
        r = ctl.list_agent_runtimes(**({"nextToken": token} if token else {}))
        for rt in r.get("agentRuntimes", []):
            if rt["agentRuntimeName"] == RUNTIME:
                return rt
        token = r.get("nextToken")
        if not token:
            return None


def deploy_runtime(ctl, role_arn: str, image: str, platform: str) -> dict:
    env = {"SRL_SANDBOX_CODE": sandbox_exec.CODE_SHA, **({"SRL_SANDBOX_PREWARM": "1"} if platform == "V2" else {})}
    kw = dict(agentRuntimeArtifact={"containerConfiguration": {"containerUri": image}}, roleArn=role_arn,
              networkConfiguration={"networkMode": "PUBLIC"}, lifecycleConfiguration=LIFECYCLE,
              platformVersion=platform, environmentVariables=env,
              description=f"Serverless RL InstantStart environment sandbox, sandbox_exec {sandbox_exec.CODE_SHA}")
    existing = find_runtime(ctl)
    if existing:
        r = ctl.update_agent_runtime(agentRuntimeId=existing["agentRuntimeId"], **kw)
        log(f"runtime {RUNTIME}: update requested")
    else:
        r = ctl.create_agent_runtime(agentRuntimeName=RUNTIME, **kw)
        log(f"runtime {RUNTIME}: create requested")
    t0 = time.time()
    while True:
        g = ctl.get_agent_runtime(agentRuntimeId=r["agentRuntimeId"])
        if g["status"] == "READY" or g["status"].endswith("FAILED"):
            break
        time.sleep(5)
    log(f"runtime {RUNTIME}: {g['status']} after {time.time() - t0:.0f} s, version {g.get('agentRuntimeVersion')}, "
        f"platform {g.get('platformVersion')}")
    if g["status"] != "READY":
        raise SystemExit(f"runtime {RUNTIME} is {g['status']}: {g.get('failureReason')}")
    return g


def ensure_usage_logs(logs, account: str, region: str, runtime_arn: str) -> None:
    try:
        logs.create_log_group(logGroupName=USAGE_GROUP)
    except logs.exceptions.ResourceAlreadyExistsException:
        pass
    dest = logs.put_delivery_destination(name=USAGE_DEST, deliveryDestinationConfiguration={
        "destinationResourceArn": f"arn:aws:logs:{region}:{account}:log-group:{USAGE_GROUP}"})
    logs.put_delivery_source(name=USAGE_SOURCE, resourceArn=runtime_arn, logType="USAGE_LOGS")
    try:
        logs.create_delivery(deliverySourceName=USAGE_SOURCE, deliveryDestinationArn=dest["deliveryDestination"]["arn"])
        log(f"USAGE_LOGS delivery created -> {USAGE_GROUP}")
    except logs.exceptions.ConflictException:
        log(f"USAGE_LOGS delivery exists -> {USAGE_GROUP}")


def warm(runtime_arn: str, region: str, n: int) -> None:
    """Open n sessions, run Python once in each, stop them; check every answer comes from this code."""
    if n <= 0:
        return
    t = LiveTransport(runtime_arn, region)
    prefix = f"srl-deploy-warm-{time.strftime('%y%m%d%H%M%S')}"

    def one(i: int):
        sid = f"{prefix}-{i:03d}-{secrets.token_hex(6)}"
        t0 = time.monotonic()
        try:
            for attempt in range(6):
                try:
                    a = t.invoke(sid, {"op": "reset", "seed": i, "files": {"w.py": "print('warm')"}})
                    break
                except Exception as e:  # throttled while ramping up: back off and retry
                    if "Throttling" not in str(e) or attempt == 5:
                        raise
                    time.sleep(0.5 * 2 ** attempt)
            cold = time.monotonic() - t0
            r = t.invoke(sid, {"op": "run", "argv": ["{python}", "w.py"], "timeout_s": 30})
            if a.get("code_sha") != sandbox_exec.CODE_SHA or not r.get("ok") or r["result"]["exit_code"] != 0:
                raise RuntimeError(f"unexpected answer: {a} / {r}")
            return cold, r["result"]["sec"]
        finally:
            try:
                t.stop(sid)
            except Exception as e:  # noqa: BLE001 - reported, the idle timeout ends it
                log(f"warm-up session {sid} not stopped: {e}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = []
        for i in range(n):
            futures.append(pool.submit(one, i))
            time.sleep(1 / 20)  # at most 20 new sessions per second, as runs do
        got = [f.result() for f in futures]
    colds, runs = sorted(c for c, _ in got), sorted(r for _, r in got)
    log(f"warmed with {n} sessions in {time.time() - t0:.0f} s: first call p50 {colds[n // 2]:.2f} s "
        f"(max {colds[-1]:.2f}), first program p50 {runs[n // 2]:.2f} s (max {runs[-1]:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--region", default=os.environ.get("SRL_REGION", "us-west-2"))
    ap.add_argument("--platform", choices=["V1", "V2"], default="V2")
    ap.add_argument("--warm", type=int, default=32, help="sessions to warm the new version with (0: none)")
    a = ap.parse_args()
    s = boto3.session.Session(region_name=a.region)
    account = s.client("sts").get_caller_identity()["Account"]
    log(f"account {account}, region {a.region}, sandbox_exec {sandbox_exec.CODE_SHA}")
    try:
        role_arn = ensure_role(s.client("iam"), account, a.region)
        image = build_image(s.client("ecr"), account, a.region, image_tag())
        rt = deploy_runtime(s.client("bedrock-agentcore-control"), role_arn, image, a.platform)
        ensure_usage_logs(s.client("logs"), account, a.region, rt["agentRuntimeArn"])
    except ClientError as e:
        raise SystemExit(f"{e.operation_name} failed: {e.response['Error'].get('Code')}: "
                         f"{e.response['Error'].get('Message')}") from None
    warm(rt["agentRuntimeArn"], a.region, a.warm)
    log(f"done: {rt['agentRuntimeArn']}. Select it on the Setup page (Environment sandbox).")


if __name__ == "__main__":
    main()
