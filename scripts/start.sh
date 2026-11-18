#!/usr/bin/env bash
# Run the image on this machine (Linux, e.g. an EC2 instance with an instance role).
#
# Host networking, like HyperPod InstantStart: the app listens on 127.0.0.1:3199 of the
# host, and the container reaches the instance metadata service for role credentials.
# Open it from your laptop with SSM or SSH port forwarding, e.g.
#   aws ssm start-session --target <instance-id> --document-name AWS-StartPortForwardingSession \
#     --parameters portNumber=3199,localPortNumber=3199
#
# To listen on all interfaces instead, set SRL_AUTH_TOKEN (the app refuses otherwise):
#   SRL_AUTH_TOKEN=$(openssl rand -hex 24) SRL_HOST=0.0.0.0 bash scripts/start.sh
#
# templates/ (the code each run executes) is mounted read-only from this checkout, not
# baked into the image, so editing it needs no rebuild. An edited .py file takes effect at
# the next run (a run copies its template when it is created); scenario.json and the
# *.zh.json catalogs are read at startup, so restart after editing them: docker restart <NAME>.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-serverless-rl-instantstart:latest}
NAME=${NAME:-serverless-rl-instantstart}
PORT=${SRL_PORT:-3199}
# A missing bind-mount source would be created as an empty, root-owned directory.
if [ ! -d templates/_shared ]; then
  echo "$PWD/templates/_shared not found: the image has no run templates, they are mounted from this checkout" >&2
  exit 1
fi
mkdir -p data "$HOME/.aws"
docker rm -f "$NAME" >/dev/null 2>&1 || true
if ss -ltnH "sport = :$PORT" | grep -q .; then
  echo "port $PORT is already in use by: $(ss -ltnpH "sport = :$PORT" | grep -o 'users:.*')" >&2
  echo "stop that process or set SRL_PORT" >&2
  exit 1
fi
docker run -d --name "$NAME" --network host --restart unless-stopped \
  -e SRL_HOST="${SRL_HOST:-127.0.0.1}" -e SRL_PORT="$PORT" -e SRL_REGION="${SRL_REGION:-us-west-2}" \
  ${SRL_AUTH_TOKEN:+-e SRL_AUTH_TOKEN="$SRL_AUTH_TOKEN"} \
  ${AWS_PROFILE:+-e AWS_PROFILE="$AWS_PROFILE"} \
  -v "$PWD/data:/app/data" \
  -v "$PWD/templates:/app/templates:ro" \
  -v "$HOME/.aws:/home/app/.aws:ro" \
  -v serverless-rl-instantstart-hf-cache:/home/app/.cache/huggingface \
  "$IMAGE" >/dev/null
# With --restart, a crashing container keeps restarting quietly; wait until it actually serves.
for _ in $(seq 1 30); do
  curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null && break
  sleep 1
done
if ! curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null; then
  echo "$NAME did not become healthy on port $PORT; last log lines:" >&2
  docker logs --tail 20 "$NAME" >&2
  exit 1
fi
echo "started $NAME: http://${SRL_HOST:-127.0.0.1}:$PORT"
[ -n "${SRL_AUTH_TOKEN:-}" ] && echo "access token: $SRL_AUTH_TOKEN"
echo "logs: docker logs -f $NAME"
