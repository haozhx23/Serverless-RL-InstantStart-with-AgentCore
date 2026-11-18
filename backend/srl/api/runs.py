"""Runs and recordings: launch, read, follow (Server-Sent Events), stop, ask, export, record, replay.

The work is in srl/launch.py (what a launch must pass) and srl/runs.py (run directories and
processes); these routes only translate HTTP to those calls.
"""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, Body, Request
from fastapi.responses import Response, StreamingResponse

from .. import estimate
from ..aws import sandbox as aws_sandbox
from ..aws import sessions as aws_sessions
from ..errors import ApiError, Invalid, NotFound
from ..events import EventLogError, read_from
from ..launch import launch
from ..runs import RunManager
from .deps import Ctx, Identity, Lang, State

router = APIRouter(tags=["runs"])


@router.post("/api/runs")
def post_run(state: State, ctx: Ctx, scenario: str = Body(...), backend: str = Body(...),
             params: dict = Body(default_factory=dict), variants: dict = Body(default_factory=dict),
             files: dict = Body(default_factory=dict), mlflow: bool = Body(True)):
    return launch(state, ctx, scenario, backend, params, variants, files, mlflow)


@router.get("/api/runs")
def get_runs(state: State, lang: Lang):
    return {"runs": [state.registry.localized(r, lang) for r in state.runs.list()]}


@router.get("/api/runs/{run_id}")
def get_run(run_id: str, state: State, lang: Lang):
    return state.registry.localized(state.runs.summary(run_id), lang)


@router.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str, state: State):
    return state.runs.stop(run_id)


@router.post("/api/runs/{run_id}/probe")
def probe_run(run_id: str, state: State, question: str = Body(..., embed=True)):
    return state.runs.probe(run_id, question)


@router.get("/api/runs/{run_id}/files/{name:path}")
def get_file(run_id: str, name: str, state: State):
    return {"name": name, "content": state.runs.read_file(run_id, name)}


@router.get("/api/runs/{run_id}/export")
def export_run(run_id: str, state: State):
    data = state.runs.export_zip(run_id)
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{run_id}.zip"'})


@router.get("/api/runs/{run_id}/billing")
def run_billing(run_id: str, state: State, ctx: Ctx):
    s = state.runs.summary(run_id)
    if s["backend"] != "live":
        raise ApiError(f"{run_id} is a {s['backend']} run; AWS billed nothing for it")
    if s["engine"] == "mtrl":
        raise ApiError(f"{run_id} is an MTRL run; its billed tokens come from the job in its event stream")
    if not s["session_name"]:
        raise ApiError(f"{run_id} has no session yet")
    return {**aws_sessions.describe(ctx, s["session_name"]), "checked_at": time.time()}


@router.get("/api/runs/{run_id}/agentcore")
def run_agentcore(run_id: str, state: State, ctx: Identity):
    """What AgentCore Runtime used for this live run, at list price: its sandbox sessions, or its MTRL agent."""
    s = state.runs.summary(run_id)
    if s["backend"] != "live":
        raise ApiError(f"{run_id} is a {s['backend']} run; AgentCore billed nothing for it")
    aws = state.runs.config(run_id)["aws"]
    if s["engine"] == "mtrl":
        m = aws.get("mtrl") or {}
        if not s["job_name"] or not m.get("runtime_arn"):
            raise ApiError(f"{run_id} has not submitted an MTRL job yet, so no agent has been called")
        platform = m.get("agent_platform") or aws_sandbox.platform_of(aws_sandbox.describe(ctx, m["runtime_arn"]))
        return aws_sandbox.runtime_usage(ctx, m["runtime_arn"], s["started_ts"], s["ended_ts"], platform)
    sandbox = aws.get("sandbox")
    if not sandbox:
        raise NotFound(f"{run_id} did not use an environment sandbox")
    return {**aws_sandbox.run_usage(ctx, run_id, sandbox, s["started_ts"], s["ended_ts"]),
            "sessions_created": (s["sandbox"] or {}).get("created")}


@router.get("/api/runs/{run_id}/estimate")
def run_estimate(run_id: str, state: State):
    """What a live run with this run's settings costs, from matching live runs.

    """
    return estimate.for_run(state.registry, state.basis, state.runs.config(run_id))


@router.post("/api/runs/{run_id}/record")
def record_run(run_id: str, state: State):
    return state.runs.record(run_id)


@router.get("/api/recordings")
def get_recordings(state: State):
    return {"recordings": state.runs.recordings()}


@router.post("/api/recordings/{name}/replay")
def replay(name: str, state: State, speed: float = Body(1.0, embed=True)):
    if not 0.25 <= speed <= 20:
        raise Invalid("speed must be between 0.25 and 20")
    return state.runs.create_replay(name, speed)


@router.get("/api/runs/{run_id}/events")
async def run_events(run_id: str, request: Request, state: State):
    path = state.runs.events_path(run_id)
    return StreamingResponse(_event_stream(state.runs, run_id, path, request), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _event_stream(runs: RunManager, run_id: str, path, request: Request):
    """The run's events.jsonl as SSE, from the start: every complete line once, until run_finished.

    Ends with `event: end`. A process that exits without run_finished, or a log that cannot be
    read, is reported as `event: problem` first.
    """
    offset, finished, quiet = 0, False, 0.0
    while not finished:
        if await request.is_disconnected():
            return
        try:
            events, offset = read_from(path, offset)
        except EventLogError as e:
            yield f"event: problem\ndata: {json.dumps({'error': str(e)})}\n\n"
            return
        for e in events:
            yield f"id: {e['seq']}\ndata: {json.dumps(e)}\n\n"
            finished = finished or e["type"] == "run_finished"
        if finished:
            break
        if not events and not runs.is_alive(run_id):
            more, offset = read_from(path, offset)  # catch lines written just before exit
            for e in more:
                yield f"id: {e['seq']}\ndata: {json.dumps(e)}\n\n"
            if not any(e["type"] == "run_finished" for e in more):
                yield ("event: problem\ndata: " + json.dumps({"error": "The run's process is no longer "
                       "running and never reported an outcome. If it was live, check the Sessions page "
                       "for a session it may have left running."}) + "\n\n")
            break
        quiet = 0.0 if events else quiet + 0.5
        if quiet >= 15:
            yield ": keep-alive\n\n"
            quiet = 0.0
        await asyncio.sleep(0.5)
    yield "event: end\ndata: {}\n\n"
