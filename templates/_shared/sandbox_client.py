"""Environment sandboxes for a training-session loop (docs/architecture.md 3.10).

``sandboxes(ev, config)`` returns a SandboxPool. For a live run it is backed by AgentCore Runtime
sessions of the runtime in config.json (``aws.sandbox``).

What the pool does for the loop, each from a measurement (docs/261110-agentcore-runtime-sandbox.md 4, 8):

- A session is opened by its first request: AgentCore creates a session for an id it has not seen.
  First requests are paced to at most SESSIONS_PER_SEC per second. The account quota is 25 new
  sessions per second; 6,000 sessions opened at a steady 25/s were never throttled, while 128-256
  sent at once were.
- Throttling ("New session creation rate exceeded") is retried with backoff, at most MAX_ATTEMPTS
  times. Every other failure, including "Runtime initialization time exceeded", raises SandboxError
  naming the sandbox, the session and AWS's own words. A sandbox failure is never a task result:
  the loop must not score it as a reward of 0, so it is raised, not returned.
- The first answer of every session is checked: it must come from the same sandbox_exec.py as this
  run's copy (CODE_SHA), and from a process no other session has used.
- close() stops every session the pool opened, including ones whose first request failed (they
  bill too: about 30 s each, measured). The loop calls it in ``finally``. The runtime's idle
  timeout is the backstop, not the mechanism.
- Every batch, every stop and the close are written as ``sandbox`` events (docs/architecture.md 4).
"""

from __future__ import annotations

import json
import random
import secrets
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import sandbox_exec

SESSIONS_PER_SEC = 20.0       # below the 25/s account quota, which other work in the account shares
MAX_ATTEMPTS = 6
RETRYABLE = {"ThrottlingException", "ServiceQuotaExceededException", "TooManyRequestsException"}
GONE = {"ResourceNotFoundException"}  # stopping a session that already ended (idle timeout) is not a failure
WORKERS = 32


class SandboxError(RuntimeError):
    """The sandbox itself failed, not the task in it. Never scored: it ends the run, loudly."""


@dataclass
class RunResult:
    exit_code: int | None
    timed_out: bool
    sec: float
    stdout: str
    stderr: str


class _Failure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


# -- transports ----------------------------------------------------------------------------

class LiveTransport:
    """AgentCore Runtime: InvokeAgentRuntime and StopRuntimeSession on one runtime endpoint."""

    where = "agentcore"

    def __init__(self, runtime_arn: str, region: str, qualifier: str = "DEFAULT"):
        import boto3
        from botocore.config import Config
        self.runtime_arn, self.qualifier = runtime_arn, qualifier
        # Retries are the pool's (paced, counted), not botocore's.
        self._client = boto3.client("bedrock-agentcore", region_name=region, config=Config(
            read_timeout=sandbox_exec.MAX_TIMEOUT_S + 60, connect_timeout=15, max_pool_connections=WORKERS * 2,
            retries={"total_max_attempts": 1}))

    def _call(self, fn, **kw):
        from botocore.exceptions import BotoCoreError, ClientError
        try:
            return fn(agentRuntimeArn=self.runtime_arn, qualifier=self.qualifier, **kw)
        except ClientError as e:
            err = e.response.get("Error", {})
            raise _Failure(err.get("Code", "ClientError"), err.get("Message", str(e))) from None
        except BotoCoreError as e:
            raise _Failure(type(e).__name__, str(e)) from None

    def invoke(self, sid: str, payload: dict) -> dict:
        r = self._call(self._client.invoke_agent_runtime, runtimeSessionId=sid, contentType="application/json",
                       accept="application/json", payload=json.dumps(payload).encode())
        body = r["response"].read()
        try:
            return json.loads(body)
        except ValueError:
            raise _Failure("BadResponse", f"not JSON: {body[:300]!r}") from None

    def stop(self, sid: str) -> None:
        self._call(self._client.stop_runtime_session, runtimeSessionId=sid)


# -- sandboxes -----------------------------------------------------------------------------

class Sandbox:
    """One session: an environment that keeps its files and processes between requests until stopped."""

    def __init__(self, pool: "SandboxPool", sid: str, label: str):
        self.pool, self.sid, self.label = pool, sid, label
        self.state = "new"  # new -> opening -> open -> stopped

    def reset(self, files: dict[str, str], seed: int) -> dict:
        """Empty the work directory, write `files`, seed `random` (the loop's seed, never a start-up value)."""
        return self.pool._call(self, {"op": "reset", "seed": seed, "files": files})

    def write(self, files: dict[str, str]) -> dict:
        return self.pool._call(self, {"op": "write", "files": files})

    def read(self, path: str, max_bytes: int = 64 * 1024) -> str:
        return self.pool._call(self, {"op": "read", "path": path, "max_bytes": max_bytes})["text"]

    def run(self, argv: list[str], timeout_s: float) -> RunResult:
        """Run a program in the work directory. Its failure or timeout is a result; the sandbox's is SandboxError."""
        r = self.pool._call(self, {"op": "run", "argv": argv, "timeout_s": timeout_s})
        if r["cleanup_failed"]:
            raise SandboxError(f"sandbox {self.label} (session {self.sid}): a program outlived its {timeout_s:g} s timeout "
                               "outside its process group (it called setsid?) and still holds its output; the session "
                               "cannot be trusted for another rollout")
        return RunResult(r["exit_code"], r["timed_out"], r["sec"], r["stdout"], r["stderr"])


class SandboxPool:
    def __init__(self, ev, transport, run_id: str, platform: str | None = None, rate: float = SESSIONS_PER_SEC):
        self.ev, self.transport, self.run_id, self.platform = ev, transport, run_id, platform
        self.where = transport.where
        self._interval = 1.0 / rate
        self._next_open = 0.0
        self._lock = threading.Lock()
        self._sessions: dict[str, Sandbox] = {}
        self._closing = False
        self._batch: dict | None = None
        self.totals = {"created": 0, "stopped": 0, "calls": 0, "retries": 0, "cold": []}
        ev.emit("sandbox", state="ready", where=self.where, runtime=transport.runtime_arn, platform=platform,
                code_sha=sandbox_exec.CODE_SHA, session_prefix=run_id, rate=rate)
        if self.where == "agentcore":
            ev.handled("sandbox_isolation", "Model-generated code runs in AgentCore Runtime sessions under the sandbox's "
                       "own execution role, not in this process, which holds this machine's AWS credentials and the "
                       "training session.")
        ev.handled("sandbox_paced", "New sandbox sessions are opened at most 20 per second and throttling is retried a "
                   "few times. The account quota is 25 per second; hundreds opened at once were throttled.")
        ev.handled("sandbox_stop_in_finally", "Every sandbox session this run opens is stopped in a finally block, on "
                   "success, error and Stop. A live session bills for its memory until it stops or idles out.")
        ev.handled("sandbox_checked", "Each sandbox's first answer must come from the same sandbox_exec.py as this run's "
                   "copy, and from a microVM no other session has used; otherwise the run stops with an error.")

    # -- handles ------------------------------------------------------------------------

    def new(self, n: int, label: str) -> list[Sandbox]:
        """Handles for `n` sandboxes; each session is opened by its first request."""
        out = []
        for i in range(n):
            # AgentCore session ids are 33-100 characters; the run id lets usage logs be matched to this run.
            sid = f"{self.run_id}-{label}-{i:03d}-{secrets.token_hex(6)}"
            out.append(Sandbox(self, sid, f"{label}-{i}"))
        return out

    # -- requests -----------------------------------------------------------------------

    def _pace(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_open)
            self._next_open = slot + self._interval
        if slot > now:
            time.sleep(slot - now)

    def _call(self, box: Sandbox, payload: dict) -> dict:
        first = box.state == "new"
        if box.state == "stopped":
            raise SandboxError(f"sandbox {box.label} (session {box.sid}) was already stopped")
        if first:
            with self._lock:
                if self._closing:
                    raise SandboxError("the run is ending; no new sandbox sessions are opened")
                box.state = "opening"
                self._sessions[box.sid] = box  # stopped at close even if this first request fails
            self._pace()
        t0, attempt = time.monotonic(), 0
        while True:
            attempt += 1
            try:
                answer = self.transport.invoke(box.sid, payload)
                break
            except _Failure as f:
                if f.code in RETRYABLE and attempt < MAX_ATTEMPTS:
                    self._count("retries")
                    time.sleep(min(8.0, 0.5 * 2 ** (attempt - 1)) * random.uniform(0.5, 1.5))
                    continue
                raise SandboxError(f"sandbox {box.label} (session {box.sid}), {payload['op']}: {f.code}: {f.message}"
                                   + (f" (after {attempt} attempts)" if attempt > 1 else "")) from None
        sec = time.monotonic() - t0
        self._count("calls")
        if not answer.get("ok"):
            raise SandboxError(f"sandbox {box.label} (session {box.sid}) refused {payload['op']}: {answer.get('error')}")
        if first:
            if answer.get("code_sha") != sandbox_exec.CODE_SHA:
                raise SandboxError(f"sandbox {box.label} (session {box.sid}) runs sandbox_exec.py version "
                                   f"{answer.get('code_sha')}, this run has {sandbox_exec.CODE_SHA}. Redeploy the "
                                   "sandbox (python3 scripts/deploy_sandbox.py) so that what runs there is this code.")
            if answer.get("sessions") != 1:
                raise SandboxError(f"sandbox {box.label} (session {box.sid}) answered from a process that has served "
                                   f"{answer.get('sessions')} sessions; sessions must not share a microVM")
            with self._lock:
                box.state = "open"
                self.totals["created"] += 1
                self.totals["cold"].append(sec)
                if self._batch is not None:
                    self._batch["cold"].append(sec)
        return answer["result"]

    def _count(self, key: str) -> None:
        with self._lock:
            self.totals[key] += 1
            if self._batch is not None:
                self._batch[key] += 1

    # -- batches ------------------------------------------------------------------------

    def map(self, fn, items: list, step: int | None = None, what: str = "") -> list:
        """fn(item) for every item, concurrently; one ``sandbox`` event when the batch ends."""
        with self._lock:
            self._batch = {"calls": 0, "retries": 0, "cold": []}
        pool = ThreadPoolExecutor(max_workers=max(1, min(WORKERS, len(items))))
        ok = False
        try:
            futures = [pool.submit(fn, it) for it in items]
            results = [f.result() for f in futures]
            ok = True
            return results
        finally:
            pool.shutdown(wait=ok, cancel_futures=not ok)
            with self._lock:
                b, self._batch = self._batch, None
                open_now = sum(1 for s in self._sessions.values() if s.state in ("opening", "open"))
            self.last_batch = b
            self.ev.emit("sandbox", state="batch", step=step, what=what, ok=ok, sessions_new=len(b["cold"]),
                         calls=b["calls"], retries=b["retries"], cold_start=_dist(b["cold"]), open=open_now,
                         created=self.totals["created"])

    def stop(self, boxes: list[Sandbox], step: int | None = None, check: bool = True) -> None:
        """Stop these sessions now (a rollout is scored). Raises SandboxError if a stop fails and `check`."""
        failed = self._stop([b for b in boxes if b.state in ("opening", "open")])
        with self._lock:
            open_now = sum(1 for s in self._sessions.values() if s.state in ("opening", "open"))
        self.ev.emit("sandbox", state="stopped", step=step, sessions=len(boxes) - len(failed), open=open_now,
                     failed=failed)
        if failed and check:
            raise SandboxError(f"{len(failed)} sandbox session(s) could not be stopped: {failed[0]['error']}")

    def _stop(self, boxes: list[Sandbox]) -> list[dict]:
        def one(box: Sandbox):
            try:
                self.transport.stop(box.sid)
            except _Failure as f:
                if f.code not in GONE:
                    return {"session": box.sid, "error": f"{f.code}: {f.message}"}
            box.state = "stopped"
            with self._lock:
                self.totals["stopped"] += 1
            return None
        if not boxes:
            return []
        with ThreadPoolExecutor(max_workers=max(1, min(WORKERS, len(boxes)))) as pool:
            return [r for r in pool.map(one, boxes) if r]

    def close(self) -> list[dict]:
        """Stop every session still open. For ``finally``: never raises; returns the stops that failed."""
        with self._lock:
            self._closing = True
            left = [s for s in self._sessions.values() if s.state in ("opening", "open")]
        try:
            failed = self._stop(left)
        except BaseException as e:  # still report; the sessions idle out on their own
            failed = [{"session": s.sid, "error": repr(e)} for s in left]
        self.ev.emit("sandbox", state="closed", where=self.where, runtime=self.transport.runtime_arn,
                     platform=self.platform, created=self.totals["created"], stopped=self.totals["stopped"],
                     calls=self.totals["calls"], retries=self.totals["retries"], cold_start=_dist(self.totals["cold"]),
                     failed=failed, session_prefix=self.run_id)
        for f in failed:
            self.ev.log(f"Sandbox session {f['session']} could not be stopped ({f['error']}). It ends after the "
                        "runtime's idle timeout; until then it bills for its memory.", level="error")
        return failed


def _dist(xs: list[float]) -> dict | None:
    if not xs:
        return None
    return {"p50": round(statistics.median(xs), 3), "max": round(max(xs), 3), "n": len(xs)}


def sandboxes(ev, config: dict) -> SandboxPool:
    """The pool for this run, backed by AgentCore Runtime sessions of the selected sandbox runtime.
    """
    sb = (config.get("aws") or {}).get("sandbox") or {}
    if not sb.get("runtime_arn"):
        raise SandboxError("config.json has no aws.sandbox.runtime_arn. A live run of this scenario needs the "
                           "environment sandbox: deploy it (python3 scripts/deploy_sandbox.py) and select it on the "
                           "Setup page.")
    return SandboxPool(ev, LiveTransport(sb["runtime_arn"], config["aws"]["region"], sb.get("qualifier", "DEFAULT")),
                       config["run_id"], platform=sb.get("platform"))
