"""Append-only event log for one run: the only channel between a training loop and the UI.

The loop script writes one JSON object per line to ``events.jsonl``; the backend tails the
file and streams it to the browser. The UI never calls training operations itself, so what
you see is exactly what this process did. Schema (v1) is documented in
docs/architecture.md; every event carries:

    v     schema version (1)
    seq   0-based line number, strictly increasing
    ts    wall-clock time (unix seconds)
    t     seconds since the run started (monotonic clock)
    type  run_started | session | step | span_start | span | metrics | rollouts | compare |
          eval | artifact | handled | log | probe | reward_check | mlflow | sandbox | run_finished

Standard library only, so an exported run needs nothing extra to keep logging. A run whose
config.json names an MLflow destination also publishes what it logs there (srl_mlflow.py, loaded
only then); the log stays the source of truth either way.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import threading
import time
from typing import Any, Iterator

SCHEMA_VERSION = 1


class EventLog:
    def __init__(self, path: str, run_id: str, echo: bool = True):
        self.path = path
        self.run_id = run_id
        self.echo = echo
        # Continue numbering if the file already has lines (a replay or a re-opened log).
        self.seq = 0
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.seq = sum(1 for _ in f)
        self._f = open(path, "a", encoding="utf-8", buffering=1)  # line-buffered
        self._t0 = time.monotonic()
        self._wall0 = time.time()
        self._lock = threading.Lock()  # a sink may log from its own thread
        self._sinks: list = []

    def attach(self, sink) -> None:
        """Pass every event to ``sink.on_event(event)`` after it is written, in order.

        Before ``run_finished`` is written, ``sink.finish(status)`` runs first, so whatever the
        sink still has to say (its own events) lands before the line that ends the stream.
        """
        self._sinks.append(sink)

    # -- core -------------------------------------------------------------------------

    def now(self) -> float:
        return round(time.monotonic() - self._t0, 3)

    def emit(self, type: str, **fields: Any) -> dict:
        if type == "run_finished":
            for sink in self._sinks:  # outside the lock: a finishing sink may still log
                sink.finish(fields.get("status"))
        with self._lock:
            event = {"v": SCHEMA_VERSION, "seq": self.seq, "ts": round(time.time(), 3), "t": self.now(),
                     "run": self.run_id, "type": type, **fields}
            self._f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            self.seq += 1
            for sink in self._sinks:
                sink.on_event(event)
        return event

    def close(self) -> None:
        self._f.close()

    # -- helpers ----------------------------------------------------------------------

    @contextlib.contextmanager
    def span(self, lane: str, op: str, step: int | None = None, **fields: Any) -> Iterator[dict]:
        """Time one operation and record it on a lane.

        lane is "client" (your process: tokenize, reward, advantage, eval) or "aws"
        (a training-session primitive running on SageMaker AI GPUs). ``calls`` is the
        number of API requests the block sends (eight concurrent ``sample`` calls are one
        block with calls=8); ``role="probe"`` marks work done to answer a question from the
        run console rather than for training. Your code that runs in an AgentCore Runtime
        sandbox instead of this process is lane "client" with ``where="agentcore"``
        and counts its requests as ``sandbox_calls``, so neither
        its time nor its requests are mistaken for this process's or SageMaker AI's. The caller may add a ``tokens`` dict ({"prefill": n,
        "sample": n, "train": n}) or a ``note`` to the yielded dict before the block ends.
        A ``span_start`` is written when the block begins and the ``span`` when it exits,
        including when it raises, so a failed call still shows up on the timeline.
        """
        if lane not in ("client", "aws"):
            raise ValueError(f"lane must be 'client' or 'aws', got {lane!r}")
        extra: dict = dict(fields)
        t0 = self.now()
        # The start marker lets the UI light up the block of the loop that is running now.
        self.emit("span_start", lane=lane, op=op, step=step, t0=t0,
                  **{k: v for k, v in fields.items() if k in ("calls", "role", "where")})
        ok = True
        try:
            yield extra
        except BaseException:
            ok = False
            raise
        finally:
            extra["sec"] = round(self.now() - t0, 3)  # readable by the caller after the block
            self.emit("span", lane=lane, op=op, step=step, t0=t0, ok=ok, **extra)

    def at(self, wall_ts: float) -> float:
        """Run-relative seconds for a wall-clock timestamp reported by a service."""
        return round(wall_ts - self._wall0, 3)

    def record_span(self, lane: str, op: str, t0: float, sec: float, step: int | None = None, **fields: Any) -> None:
        """A span whose start and duration come from elsewhere (e.g. a job's status transitions)."""
        if lane not in ("client", "aws"):
            raise ValueError(f"lane must be 'client' or 'aws', got {lane!r}")
        self.emit("span", lane=lane, op=op, step=step, t0=round(t0, 3), sec=round(max(0.0, sec), 3), ok=True, **fields)

    def step(self, step: int, total: int) -> None:
        self.emit("step", step=step, total=total)

    def metrics(self, step: int, **values: float) -> None:
        self.emit("metrics", step=step, values=values)
        if self.echo:
            shown = ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in values.items())
            print(f"step {step}: {shown}", flush=True)

    def log(self, msg: str, level: str = "info") -> None:
        self.emit("log", level=level, msg=msg)
        if self.echo:
            print(msg, file=sys.stderr if level in ("warn", "error") else sys.stdout, flush=True)

    def handled(self, code: str, message: str) -> None:
        """Record a known pitfall this script works around, so the UI can say what was done and why."""
        self.emit("handled", code=code, message=message)

    def session(self, state: str, name: str | None = None, **fields: Any) -> None:
        self.emit("session", state=state, name=name, **fields)
        if self.echo and state in ("running", "stopped", "failed"):
            print(f"session {name}: {state}", flush=True)

    def artifact(self, kind: str, arn: str, description: str, step: int | None = None) -> None:
        self.emit("artifact", kind=kind, arn=arn, description=description, step=step)
        if self.echo:
            print(f"{kind}: {arn}", flush=True)

    def finished(self, status: str, error: str | None = None) -> None:
        if status not in ("completed", "failed", "stopped"):
            raise ValueError(f"unknown final status {status!r}")
        self.emit("run_finished", status=status, error=error)


def open_log(config: dict) -> EventLog:
    """Open the event log next to the script (or at $SRL_EVENTS), publishing to MLflow when configured.

    Only live runs publish, even when config.json names an MLflow app (as an exported live run's does).
    """
    ev = EventLog(os.environ.get("SRL_EVENTS", "events.jsonl"), run_id=config["run_id"])
    live = (os.environ.get("SRL_BACKEND") or config.get("backend", "live")) == "live"
    if config.get("mlflow") and live:
        from srl_mlflow import MlflowSink  # needs mlflow and sagemaker-mlflow, which sagemaker-train brings
        ev.attach(MlflowSink(ev, config))
    return ev


def install_stop_handlers() -> None:
    """Turn SIGTERM into KeyboardInterrupt so ``finally: tc.stop()`` runs on every stop path.

    SIGINT already raises KeyboardInterrupt. Without this, a plain ``kill`` would end the
    process before the session is stopped, and the idle session would keep holding quota.
    """
    def _raise(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, _raise)


def load_config(path: str = "config.json") -> dict:
    with open(os.environ.get("SRL_CONFIG", path), encoding="utf-8") as f:
        return json.load(f)
