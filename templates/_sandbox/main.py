"""AgentCore Runtime entry point of the environment sandbox (docs/architecture.md 3.10).

Deployed by scripts/deploy_sandbox.py, which builds this directory together with
templates/_shared/sandbox_exec.py into one ARM64 image. All the work is in sandbox_exec.py;
this file only connects it to BedrockAgentCoreApp.

Each session id gets its own Executor and work directory. In a fresh microVM there is exactly
one; the answers carry the count, and the caller refuses a session that shares its process.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from bedrock_agentcore.runtime import BedrockAgentCoreApp

import sandbox_exec

WORK = Path(os.environ.get("SRL_SANDBOX_WORKDIR", "/tmp/srl-work"))
EXECUTORS: dict[str, sandbox_exec.Executor] = {}

# Platform V2 snapshots the process after start-up: Python started once here is warm in every session.
if os.environ.get("SRL_SANDBOX_PREWARM") == "1":
    sandbox_exec.prewarm()

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict, context) -> dict:
    sid = getattr(context, "session_id", None) or "no-session-id"
    ex = EXECUTORS.get(sid)
    if ex is None:
        # Every executor reports how many sessions this process has served, so a shared microVM shows up.
        known = list(EXECUTORS)
        ex = EXECUTORS[sid] = sandbox_exec.Executor(WORK / hashlib.sha256(sid.encode()).hexdigest()[:16])
        ex.session_ids.extend(known)
        for other in known:
            EXECUTORS[other].session_ids.append(sid)
    return ex.handle(payload, sid)


if __name__ == "__main__":
    app.run()
