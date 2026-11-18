#!/usr/bin/env python3
"""Deploy, update or delete the environment sandbox from the command line: the same steps as the Deploy or update
and Delete buttons of the Setup page (backend/srl/aws/deploy.py), printed as they run.

    backend/.venv/bin/python scripts/deploy_sandbox.py                   # AWS CodeBuild builds the image (us-west-2, V2)
    backend/.venv/bin/python scripts/deploy_sandbox.py --build local     # this machine's docker buildx builds it
    backend/.venv/bin/python scripts/deploy_sandbox.py --platform V1 --region us-east-1
    backend/.venv/bin/python scripts/deploy_sandbox.py --delete

The Setup page needs none of this: its button does the same from the app. The job is recorded where the app
keeps it (data/deploys/), so the Setup page shows its result too; do not run both at once.

--build local needs docker buildx that can build linux/arm64 (AgentCore Runtime runs only ARM64 images). On
an x86 machine that means QEMU emulation registered with
    docker run --privileged --rm tonistiigi/binfmt --install arm64
which this script checks before building. The resources and their names are in docs/architecture.md 6;
SRL_SANDBOX_RUNTIME, SRL_SANDBOX_ROLE, SRL_SANDBOX_REPOSITORY, SRL_IMAGE_BUILD_PROJECT and SRL_IMAGE_BUILD_ROLE
override them.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import os
import platform
import subprocess
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from srl import settings  # noqa: E402
from srl.aws import deploy  # noqa: E402
from srl.aws.context import AwsContext  # noqa: E402

QEMU = "docker run --privileged --rm tonistiigi/binfmt --install arm64"


def sh(cmd: list[str], **kw) -> str:
    r = subprocess.run(cmd, text=True, capture_output=True, **kw)
    if r.returncode != 0:
        raise deploy.DeployError(f"command failed ({r.returncode}): {' '.join(cmd[:4])} ...\n{r.stdout[-2000:]}\n"
                                 f"{r.stderr[-3000:]}")
    return r.stdout


def build_local(ctx: AwsContext, t: deploy.Target, job: deploy.Job) -> str:
    """Build and push the image with this machine's docker buildx (linux/arm64, one manifest, no attestations)."""
    if platform.machine() not in ("aarch64", "arm64"):
        builders = sh(["docker", "buildx", "inspect", "--bootstrap"])
        if "linux/arm64" not in builders:
            raise deploy.DeployError(f"docker buildx on this {platform.machine()} machine cannot build linux/arm64: "
                                     f"register QEMU emulation first ({QEMU}), or build in AWS CodeBuild (--build "
                                     "codebuild, the default)")
    ecr = deploy.client("ecr", ctx)
    deploy._ensure_repository(ecr, t, job)
    tag, image = t.image_tag(), deploy.image_uri(ctx, t)
    if deploy.pushed(ecr, t, tag):
        job.detail("image", f"{t.repository}:{tag} is already in ECR (same build files); not built again")
        return image
    registry = image.split("/", 1)[0]
    user, secret = base64.b64decode(ecr.get_authorization_token()["authorizationData"][0]["authorizationToken"]
                                    ).decode().split(":", 1)
    sh(["docker", "login", "--username", user, "--password-stdin", registry], input=secret)
    job.detail("image", f"building linux/arm64 {t.repository}:{tag} with docker buildx on this machine")
    sh(["docker", "buildx", "build", "--platform", "linux/arm64", "--provenance=false", "--sbom=false",
        "-f", str(t.context / t.dockerfile), "-t", image, "--push", str(t.context)], timeout=3600)
    detail = deploy.pushed(ecr, t, tag)
    if not detail:
        raise deploy.DeployError(f"docker pushed {image}, but ECR has no {t.repository}:{tag}")
    job.detail("image", f"{t.repository}:{tag} pushed, {detail.get('imageSizeInBytes', 0) / 2**20:.1f} MiB")
    return image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--region", default=os.environ.get("SRL_REGION", "us-west-2"))
    ap.add_argument("--platform", choices=["V1", "V2"], default="V2")
    ap.add_argument("--build", choices=["codebuild", "local"], default="codebuild",
                    help="where the ARM64 image is built (default: AWS CodeBuild in your account)")
    ap.add_argument("--warm", type=int, help=f"sessions to warm a V2 version with (default {deploy.SANDBOX_WARM}; 0: none)")
    ap.add_argument("--delete", action="store_true", help="delete the sandbox's resources instead")
    a = ap.parse_args()
    ident = boto3.session.Session(region_name=a.region).client("sts").get_caller_identity()
    ctx = AwsContext(region=a.region, account=ident["Account"], caller_arn=ident["Arn"], identity_error=None)
    t = deploy.sandbox_target(a.platform)
    if a.warm is not None:
        t = dataclasses.replace(t, warm=a.warm)
    job = deploy.Job(t, "delete" if a.delete else "deploy", ctx.region, settings.DATA_DIR / "deploys" / f"{t.id}.json",
                     echo=lambda line: print(line, flush=True))
    deploy.run(job, ctx, build_image=build_local if a.build == "local" else None)
    done = job.snapshot()
    if done["state"] != "succeeded":
        raise SystemExit(f"{done['action']} failed: {done['error']}")
    if not a.delete:
        print(f"done: {done['result'].get('runtime_arn')} is selected on the Setup page (Environment sandbox).")


if __name__ == "__main__":
    main()
