#!/usr/bin/env bash
# 从零把这个样例搭到 AWS 上。每一步都可重复执行（已存在则跳过）。
#
# 这是"最基础的标准方式"：只用 aws cli + docker buildx + boto3，
# 不依赖 agentcore CLI、CDK 或 CloudFormation，便于逐步排查。
#
# 顺序不能变：IAM -> 镜像 -> AgentCore Runtime -> 数据集 -> 起训。
set -euo pipefail

ACCOUNT="${ACCOUNT:-$(aws sts get-caller-identity --query Account --output text)}"
REGION="${REGION:-us-west-2}"
REPO="${REPO:-aftersales-mtrl-agent}"
TAG="${TAG:-v1}"
RUNTIME_NAME="${RUNTIME_NAME:-aftersales_mtrl_agent}"
BUCKET="${BUCKET:-sagemaker-${REGION}-${ACCOUNT}}"
PREFIX="${PREFIX:-aftersales-mtrl}"
JOB_ROLE="${JOB_ROLE:-AftersalesMTRLJobExecRole}"
AGENT_ROLE="${AGENT_ROLE:-AftersalesMTRLAgentRuntimeRole}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AWS_PAGER=""

log() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------- 1. IAM
# 两个角色的职责不同，不能合并：
#   JOB_ROLE   由 job.sagemaker.amazonaws.com 承担，负责读 S3、写 checkpoint、
#              调用 AgentCore Runtime、写 MLflow。托管策略 AmazonSageMakerJobFullAccess。
#   AGENT_ROLE 由 bedrock-agentcore.amazonaws.com 承担，是 Agent 运行时身份，
#              负责调 Sample / CompleteRollout / UpdateReward。
#              托管策略 AmazonSageMakerJobRuntimeAccess，另需 ECR 拉取与日志写入。
log "1. IAM 角色"
cat > /tmp/trust-job.json <<JSON
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Principal":{"Service":"job.sagemaker.amazonaws.com"},"Action":["sts:AssumeRole","sts:TagSession"]},
 {"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":["sts:AssumeRole","sts:TagSession"]}]}
JSON
cat > /tmp/trust-agentcore.json <<JSON
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Principal":{"Service":"bedrock-agentcore.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
cat > /tmp/agent-extra.json <<JSON
{"Version":"2012-10-17","Statement":[
 {"Sid":"ECRPull","Effect":"Allow","Action":["ecr:GetAuthorizationToken","ecr:BatchCheckLayerAvailability","ecr:GetDownloadUrlForLayer","ecr:BatchGetImage"],"Resource":"*"},
 {"Sid":"Logs","Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents","logs:DescribeLogStreams","logs:DescribeLogGroups"],"Resource":"*"},
 {"Sid":"Telemetry","Effect":"Allow","Action":["xray:PutTraceSegments","xray:PutTelemetryRecords","cloudwatch:PutMetricData"],"Resource":"*"}]}
JSON

aws iam create-role --role-name "$JOB_ROLE" \
  --assume-role-policy-document file:///tmp/trust-job.json >/dev/null 2>&1 || echo "  $JOB_ROLE 已存在"
aws iam attach-role-policy --role-name "$JOB_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerJobFullAccess
aws iam create-role --role-name "$AGENT_ROLE" \
  --assume-role-policy-document file:///tmp/trust-agentcore.json >/dev/null 2>&1 || echo "  $AGENT_ROLE 已存在"
aws iam attach-role-policy --role-name "$AGENT_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerJobRuntimeAccess
aws iam put-role-policy --role-name "$AGENT_ROLE" \
  --policy-name ECRAndLogs --policy-document file:///tmp/agent-extra.json
echo "  JOB_ROLE   = arn:aws:iam::${ACCOUNT}:role/${JOB_ROLE}"
echo "  AGENT_ROLE = arn:aws:iam::${ACCOUNT}:role/${AGENT_ROLE}"

# ---------------------------------------------------- 2. ARM64 镜像推到 ECR
# AgentCore Runtime 只支持 arm64。x86 构建机需要先注册 qemu：
log "2. 构建并推送 ARM64 镜像"
aws ecr create-repository --repository-name "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || echo "  ECR repo 已存在"
if ! docker buildx inspect --bootstrap 2>/dev/null | grep -q linux/arm64; then
  echo "  注册 qemu arm64 模拟器"
  docker run --privileged --rm tonistiigi/binfmt --install arm64 >/dev/null
fi
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}"
docker buildx build --platform linux/arm64 -f "${HERE}/deploy/Dockerfile" -t "$IMAGE" --push "$HERE"
echo "  IMAGE = $IMAGE"

# ------------------------------------------------- 3. AgentCore Runtime
log "3. 创建 / 更新 AgentCore Runtime"
ARTIFACT="{\"containerConfiguration\":{\"containerUri\":\"${IMAGE}\"}}"
EXISTING=$(aws bedrock-agentcore-control list-agent-runtimes --region "$REGION" \
  --query "agentRuntimes[?agentRuntimeName=='${RUNTIME_NAME}'].agentRuntimeId | [0]" --output text)
if [ "$EXISTING" = "None" ] || [ -z "$EXISTING" ]; then
  RUNTIME_ARN=$(aws bedrock-agentcore-control create-agent-runtime --region "$REGION" \
    --agent-runtime-name "$RUNTIME_NAME" \
    --description "MTRL rollout agent: e-commerce after-sales adjudication" \
    --agent-runtime-artifact "$ARTIFACT" \
    --network-configuration '{"networkMode":"PUBLIC"}' \
    --role-arn "arn:aws:iam::${ACCOUNT}:role/${AGENT_ROLE}" \
    --query agentRuntimeArn --output text)
else
  RUNTIME_ARN=$(aws bedrock-agentcore-control update-agent-runtime --region "$REGION" \
    --agent-runtime-id "$EXISTING" \
    --agent-runtime-artifact "$ARTIFACT" \
    --network-configuration '{"networkMode":"PUBLIC"}' \
    --role-arn "arn:aws:iam::${ACCOUNT}:role/${AGENT_ROLE}" \
    --query agentRuntimeArn --output text)
fi
echo "  RUNTIME_ARN = $RUNTIME_ARN"

# --------------------------------------------------------- 4. 数据集到 S3
log "4. 生成并上传数据集"
aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1 || aws s3 mb "s3://${BUCKET}" --region "$REGION"
( cd "$HERE" && PYTHONPATH=. python3 -m sample.dataset \
    --out-dir data --train-size 128 --eval-size 32 --format jsonl )
aws s3 cp "${HERE}/data/train.jsonl" "s3://${BUCKET}/${PREFIX}/train.jsonl" --region "$REGION"
aws s3 cp "${HERE}/data/eval.jsonl"  "s3://${BUCKET}/${PREFIX}/eval.jsonl"  --region "$REGION"

# --------------------------------------------------------------- 5. 汇总
log "搭建完成，接下来"
cat <<NEXT
  # 起训之前先做 AgentCore 预检（不花训练费用）：
  (cd "${HERE}" && PYTHONPATH=. python3 deploy/smoke_test_runtime.py --runtime-arn ${RUNTIME_ARN})

  # 预检通过后，在 InstantStart 的 Setup 页选择：
  Agent runtime: ${RUNTIME_ARN}
  Job execution role: ${JOB_ROLE}

  完成 model package group 和 MLflow App 检查。
  在 MTRL 页面选择售后场景，再点击 Submit live job。
  InstantStart 会按所选参数重新生成并上传该次运行的数据集。
NEXT
