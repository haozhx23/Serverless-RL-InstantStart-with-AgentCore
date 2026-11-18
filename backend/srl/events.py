"""Reading a run's events.jsonl and reducing it to a summary.

The file is written by the run's own process (see templates/_shared/srl_events.py); the
backend only reads it. Only complete lines are consumed, so a line being written is
picked up on the next read. A complete line that is not valid JSON is an error, never
skipped: the UI would otherwise show a run that silently lost events.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import pricing

FINAL_STATUSES = ("completed", "failed", "stopped")


class EventLogError(Exception):
    pass


def read_events(path: Path, since: int = 0) -> list[dict]:
    if not path.exists():
        return []
    events = []
    with open(path, "rb") as f:
        for i, raw in enumerate(f):
            if not raw.endswith(b"\n"):
                break  # still being written
            if i < since:
                continue
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError as e:
                raise EventLogError(f"{path}: line {i + 1} is not valid JSON ({e.msg})") from e
    return events


def read_from(path: Path, offset: int) -> tuple[list[dict], int]:
    """Complete lines after byte ``offset``, and the offset to continue from (for tailing)."""
    if not path.exists():
        return [], offset
    with open(path, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    end = chunk.rfind(b"\n") + 1  # ignore a trailing partial line
    events = []
    for raw in chunk[:end].splitlines():
        try:
            events.append(json.loads(raw))
        except json.JSONDecodeError as e:
            raise EventLogError(f"{path}: a line after byte {offset} is not valid JSON ({e.msg})") from e
    return events, offset + end


def summarize(events: list[dict]) -> dict:
    """Fields the run list and headers need; the console reduces the full stream itself."""
    s = {"status": None, "error": None, "backend": None, "model": None, "region": None,
         "step": None, "total_steps": None, "session_name": None, "session_state": None,
         "ready_sec": None, "tokens": {"prefill": 0, "sample": 0, "train": 0},
         "artifacts": 0, "last_metrics": {}, "eval": None, "started_ts": None, "ended_ts": None,
         "recorded_at": None, "duration_s": None, "engine": "training-session", "job_name": None,
         "job_status": None, "billed_tokens": None, "mlflow": None, "sandbox": None}
    for e in events:
        t = e["type"]
        if t == "run_started":
            s.update(backend=e.get("backend"), model=e.get("model"), region=e.get("region"),
                     total_steps=e.get("total_steps"), started_ts=e["ts"], recorded_at=e.get("recorded_at"),
                     engine=e.get("engine", "training-session"))
        elif t == "job":
            s.update(job_name=e.get("name"), job_status=e.get("secondary") or e.get("status"))
        elif t == "billing":
            s["billed_tokens"] = e["tokens"]
        elif t == "session":
            s["session_state"] = e["state"]
            if e.get("name"):
                s["session_name"] = e["name"]
            if e.get("ready_sec") is not None:
                s["ready_sec"] = e["ready_sec"]
        elif t == "step":
            s["step"] = e["step"]
        elif t == "span":
            for k, v in (e.get("tokens") or {}).items():
                s["tokens"][k] = s["tokens"].get(k, 0) + v
        elif t == "metrics":
            s["last_metrics"] = e["values"]
        elif t == "artifact":
            s["artifacts"] += 1
        elif t == "eval":
            s["eval"] = {k: e.get(k) for k in ("label", "before", "after", "n", "unit")}
        elif t == "mlflow":  # where a live training-session run published itself (srl_mlflow.py)
            s["mlflow"] = {k: e.get(k) for k in ("state", "tracking_uri", "experiment", "experiment_id", "run_id",
                                                  "status", "error", "published")}
        elif t == "sandbox":  # the environment sandbox this run sent code to (sandbox_client.py)
            sb = s["sandbox"] = s["sandbox"] or {"where": None, "runtime": None, "platform": None, "created": 0,
                                                  "open": 0, "retries": 0, "failed_stops": 0, "closed": False}
            if e["state"] == "ready":
                sb.update(where=e.get("where"), runtime=e.get("runtime"), platform=e.get("platform"))
            elif e["state"] == "batch":
                sb["created"], sb["open"] = e.get("created", sb["created"]), e.get("open", sb["open"])
                sb["retries"] += e.get("retries", 0)
            elif e["state"] == "stopped":
                sb["open"] = e.get("open", sb["open"])
                sb["failed_stops"] += len(e.get("failed") or [])
            elif e["state"] == "closed":
                sb.update(created=e.get("created", sb["created"]), open=0, closed=True,
                          failed_stops=sb["failed_stops"] + len(e.get("failed") or []))
        elif t == "run_finished":
            s.update(status=e["status"], error=e.get("error"), ended_ts=e["ts"])
        s["duration_s"] = e.get("t")
    # MTRL tokens are counted by the service and reported by the job; nothing is counted locally.
    if s["engine"] == "mtrl":
        s["tokens"] = s["billed_tokens"] or {"prefill": 0, "sample": 0, "train": 0}
    s["cost_usd"] = pricing.estimate(s["model"], s["tokens"], s["engine"]) if s["model"] else None
    return s
