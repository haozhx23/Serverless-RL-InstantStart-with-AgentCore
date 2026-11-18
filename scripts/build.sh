#!/usr/bin/env bash
# Build the image. Local only: nothing is pushed.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-serverless-rl-instantstart:latest}
docker build -t "$IMAGE" .
docker images "$IMAGE" --format 'built {{.Repository}}:{{.Tag}} ({{.Size}})'
