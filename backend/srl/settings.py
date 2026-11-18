"""Process-wide settings, read once from the environment.

Everything a run needs at runtime is written into that run's config.json, so changing
these later never changes what an existing run did.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("SRL_ROOT", Path(__file__).resolve().parents[2]))
TEMPLATES_DIR = Path(os.environ.get("SRL_TEMPLATES", ROOT / "templates"))
DATA_DIR = Path(os.environ.get("SRL_DATA", ROOT / "data"))
STATIC_DIR = Path(os.environ.get("SRL_STATIC", ROOT / "frontend" / "dist"))

HOST = os.environ.get("SRL_HOST", "127.0.0.1")
PORT = int(os.environ.get("SRL_PORT", "3199"))
# Required when HOST is not a loopback address: anyone who reaches the port can start
# sessions with this machine's AWS credentials.
AUTH_TOKEN = os.environ.get("SRL_AUTH_TOKEN") or None

# Regions and quota from the Training Session docs and our 2026-10-26 checks.
SUPPORTED_REGIONS = ["us-west-2", "us-east-1", "ap-south-1"]
DEFAULT_REGION = os.environ.get("SRL_REGION", "us-west-2")
CONCURRENT_SESSIONS_QUOTA_CODE = "L-152DC2E3"
DEFAULT_CONCURRENT_SESSIONS = 2

# Names of the account resources this tool creates or reuses. The role must sit under
# /service-role/ and contain "SageMaker": the caller managed policy only allows
# iam:PassRole on that pattern. The bucket must start with "sagemaker-": the same policy
# only grants s3:DeleteObject there, which cleanup needs.
ROLE_NAME = os.environ.get("SRL_ROLE_NAME", "AmazonSageMaker-ServerlessRL-InstantStart")
MODEL_PACKAGE_GROUP = os.environ.get("SRL_MODEL_PACKAGE_GROUP", "serverless-rl-instantstart")
S3_PREFIX = os.environ.get("SRL_S3_PREFIX", "serverless-rl-instantstart")
SESSION_PREFIX = "instantstart"

# Multi-turn RL (MTRL) jobs. Regions from the MTRL overview page (2026-10-28); quota codes
# read in this account on 2026-08-24. The output model package group is shared with
# training sessions; checkpoints need a second, different group.
MTRL_REGIONS = ["us-west-2", "us-east-1"]
MTRL_QUOTA_CODE = "L-A9CEE0B1"
MTRL_JOB_ROLE = os.environ.get("SRL_MTRL_JOB_ROLE", "AmazonSageMaker-ServerlessRL-MTRLJob")
MTRL_CHECKPOINT_GROUP = os.environ.get("SRL_MTRL_CHECKPOINT_GROUP", "serverless-rl-instantstart-mtrl-checkpoints")

# Versions the templates were verified with (live, 2026-10-27).
TESTED_SDK = {"sagemaker-train": "1.27.0", "boto3": "1.43.119"}

# The environment sandbox (docs/architecture.md 3.10): an AgentCore Runtime deployed from the Setup page
# (srl/aws/deploy.py) or by scripts/deploy_sandbox.py. Without a selection on the Setup page, the runtime with
# this name is used.
SANDBOX_RUNTIME = os.environ.get("SRL_SANDBOX_RUNTIME", "srl_instantstart_sandbox")
SANDBOX_ROLE = os.environ.get("SRL_SANDBOX_ROLE", "SRL-InstantStart-SandboxRuntime")
SANDBOX_REPOSITORY = os.environ.get("SRL_SANDBOX_REPOSITORY", "serverless-rl-instantstart-sandbox")
# Where the sandbox's USAGE_LOGS go (per session, per second; a run's AgentCore cost is read from them).
SANDBOX_USAGE_GROUP = os.environ.get("SRL_SANDBOX_USAGE_GROUP",
                                     "/aws/vendedlogs/bedrock-agentcore/serverless-rl-instantstart/sandbox-usage")
SANDBOX_USAGE_SOURCE = os.environ.get("SRL_SANDBOX_USAGE_SOURCE", "srl-instantstart-sandbox-usage")
SANDBOX_USAGE_DEST = os.environ.get("SRL_SANDBOX_USAGE_DEST", "srl-instantstart-sandbox-usage-cwl")

# AWS CodeBuild builds the ARM64 images of the runtimes the Setup page deploys (srl/aws/deploy.py).
IMAGE_BUILD_PROJECT = os.environ.get("SRL_IMAGE_BUILD_PROJECT", "serverless-rl-instantstart-image-build")
IMAGE_BUILD_ROLE = os.environ.get("SRL_IMAGE_BUILD_ROLE", "SRL-InstantStart-ImageBuild")
