"""The environment sandbox: what runs inside every AgentCore Runtime session (docs/architecture.md 3.10).

One AgentCore Runtime session is one microVM and one rollout's environment. The training loop
(loop.py, outside the sandbox) sends one request per environment action and keeps sampling to
itself; nothing here calls a model or holds a training session.

Requests, as ``payload["op"]``:

    info    {}                                   who is answering: code version, python, request count
    reset   {"seed": int, "files": {path: text}}  wipe the work directory, write files, reseed random
    write   {"files": {path: text}}               add or replace files in the work directory
    read    {"path": str, "max_bytes": int}       read a file of the work directory
    run     {"argv": [...], "timeout_s": float}   run a program in the work directory, in its own
                                                  process group; "{python}" in argv is this Python

Every answer is {"ok": true, "op", "result", "code_sha", "sessions", "invocations", "handler_sec"},
or {"ok": false, "op", "error"} when the request itself is wrong. A program that fails, prints
garbage or runs out of time is a result ({"exit_code", "timed_out", ...}), not an error: whether
that is a bad answer is for the caller's reward to decide.

Rules this file follows, each measured on 2026-11-10 (docs/261110-agentcore-runtime-sandbox.md 4, 8):

- Model code never runs in this process: ``run`` starts a separate process group and, on timeout,
  kills the whole group (``os.killpg``). Killing only the shell left a busy-looping program running
  (and billing) in Debian images, whose /bin/sh is dash.
- Programs get a minimal environment (no AWS_* variables). The execution role's credentials can
  still be fetched from inside the microVM, so the role must allow next to nothing.
- Random numbers come from the caller's seed (``reset``) or os.urandom. On platform V2 the process
  is restored from a snapshot, and values drawn at start-up repeat across sessions (10 sessions
  showed 2 distinct values).
- ``sessions`` counts the session ids this process has served. It is 1 in a fresh microVM; the
  caller treats anything else as broken isolation (8,250 sessions never shared a microVM).

Standard library only, so the sandbox image needs nothing but bedrock-agentcore around it.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path, PurePosixPath

with open(__file__, "rb") as _f:
    #: The version of this file. The deployed runtime reports it with every answer; a run refuses a
    #: sandbox whose version differs from its own copy (redeploy with scripts/deploy_sandbox.py).
    CODE_SHA = hashlib.sha256(_f.read()).hexdigest()[:12]

MAX_FILE_BYTES = 1 << 20      # per file written by reset / write
MAX_FILES = 64
MAX_OUTPUT = 64 * 1024        # stdout and stderr kept from one program, each
MAX_TIMEOUT_S = 600.0
CLEANUP_WAIT_S = 3.0          # after killing a timed-out group, how long to wait for its pipes to close


def _clip(raw: bytes, limit: int) -> tuple[str, bool]:
    text = raw[:limit].decode("utf-8", errors="replace")
    return text, len(raw) > limit


def _relative(path: str) -> PurePosixPath:
    p = PurePosixPath(path)
    if not path or p.is_absolute() or ".." in p.parts:
        raise ValueError(f"path {path!r} must be relative to the work directory and must not contain '..'")
    return p


class Executor:
    """One session's environment: a work directory and the programs started in it."""

    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)
        self.invocations = 0
        self.session_ids: list[str] = []
        self._groups: list[int] = []   # process groups started by `run`; whatever is left of them dies on reset
        self._lock = threading.Lock()  # one request at a time per session, whatever the platform does
        self.workdir.mkdir(parents=True, exist_ok=True)

    # -- entry point ------------------------------------------------------------------------

    def handle(self, payload: dict, session_id: str | None = None) -> dict:
        t0 = time.monotonic()
        with self._lock:
            if session_id is not None and session_id not in self.session_ids:
                self.session_ids.append(session_id)
            self.invocations += 1
            op = payload.get("op") if isinstance(payload, dict) else None
            ops = {"info": self._info, "reset": self._reset, "write": self._write, "read": self._read, "run": self._run}
            if op not in ops:
                return {"ok": False, "op": op, "error": f"unknown op {op!r}; expected one of {sorted(ops)}",
                        "code_sha": CODE_SHA}
            try:
                result = ops[op](payload)
            except (ValueError, TypeError, KeyError, OSError) as e:
                return {"ok": False, "op": op, "error": f"{type(e).__name__}: {e}", "code_sha": CODE_SHA}
            return {"ok": True, "op": op, "result": result, "code_sha": CODE_SHA, "sessions": len(self.session_ids),
                    "invocations": self.invocations, "handler_sec": round(time.monotonic() - t0, 4)}

    # -- operations -------------------------------------------------------------------------

    def _info(self, p: dict) -> dict:  # noqa: ARG002
        return {"python": sys.version.split()[0], "pid": os.getpid(), "workdir": str(self.workdir)}

    def _reset(self, p: dict) -> dict:
        self._kill_groups()
        if self.workdir.exists():
            shutil.rmtree(self.workdir)
        self.workdir.mkdir(parents=True)
        seed = p.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError(f"seed must be an integer, got {seed!r}")
        random.seed(seed if seed is not None else int.from_bytes(os.urandom(8), "big"))
        written = self._put(p.get("files") or {})
        return {"files": written, "seeded": seed is not None}

    def _write(self, p: dict) -> dict:
        return {"files": self._put(p.get("files") or {})}

    def _read(self, p: dict) -> dict:
        path = self.workdir / _relative(p.get("path", ""))
        limit = int(p.get("max_bytes", MAX_OUTPUT))
        raw = path.read_bytes()
        text, cut = _clip(raw, limit)
        return {"path": str(_relative(p["path"])), "text": text, "bytes": len(raw), "truncated": cut}

    def _run(self, p: dict) -> dict:
        argv = p.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            raise TypeError("argv must be a non-empty list of strings (no shell is used)")
        argv = [sys.executable if a == "{python}" else a for a in argv]
        timeout = float(p.get("timeout_s", 10))
        if not 0 < timeout <= MAX_TIMEOUT_S:
            raise ValueError(f"timeout_s must be in (0, {MAX_TIMEOUT_S:g}], got {timeout}")
        # No AWS_* or other inherited variables reach the program.
        env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": str(self.workdir),
               "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
        t0 = time.monotonic()
        proc = subprocess.Popen(argv, cwd=self.workdir, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True)
        self._groups.append(proc.pid)
        timed_out, cleanup_failed = False, False
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill(proc.pid)
            try:  # the pipes close once every process holding them is gone
                out, err = proc.communicate(timeout=CLEANUP_WAIT_S)
            except subprocess.TimeoutExpired:
                # Something left the group (setsid) and still holds the pipes: the caller should stop this session.
                cleanup_failed = True
                proc.kill()
                out, err = b"", b""
        sec = round(time.monotonic() - t0, 4)
        stdout, out_cut = _clip(out or b"", MAX_OUTPUT)
        stderr, err_cut = _clip(err or b"", MAX_OUTPUT)
        return {"exit_code": None if timed_out else proc.returncode, "timed_out": timed_out,
                "cleanup_failed": cleanup_failed, "sec": sec, "stdout": stdout, "stderr": stderr,
                "truncated": out_cut or err_cut}

    # -- helpers ----------------------------------------------------------------------------

    def _put(self, files: dict) -> int:
        if not isinstance(files, dict):
            raise TypeError("files must map relative paths to text")
        if len(files) > MAX_FILES:
            raise ValueError(f"{len(files)} files; at most {MAX_FILES} per request")
        for path, text in files.items():
            if not isinstance(text, str):
                raise TypeError(f"{path}: content must be text")
            data = text.encode("utf-8")
            if len(data) > MAX_FILE_BYTES:
                raise ValueError(f"{path}: {len(data):,} bytes is over the {MAX_FILE_BYTES:,}-byte limit")
            target = self.workdir / _relative(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return len(files)

    @staticmethod
    def _kill(pgid: int) -> None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def _kill_groups(self) -> None:
        for pgid in self._groups:
            self._kill(pgid)
        self._groups.clear()


def prewarm() -> float:
    """Start Python once, so it is in the V2 snapshot: unwarmed, a session's first program took up to 1.4 s."""
    t0 = time.monotonic()
    subprocess.run([sys.executable, "-c", "import json, importlib.util, math, copy"], check=True, timeout=60,
                   stdin=subprocess.DEVNULL, capture_output=True)
    return round(time.monotonic() - t0, 3)


if __name__ == "__main__":  # a quick local check: python sandbox_exec.py
    ex = Executor(Path(os.environ.get("TMPDIR", "/tmp")) / "srl-sandbox-selftest")
    print(json.dumps(ex.handle({"op": "reset", "seed": 1, "files": {"a.py": "print(41 + 1)"}}), indent=1))
    print(json.dumps(ex.handle({"op": "run", "argv": ["{python}", "a.py"], "timeout_s": 5}), indent=1))
