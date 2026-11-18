"""Runs: one directory and one child process per run.

data/runs/<run_id>/ holds everything a run is: the loop script and its helpers (copied from
templates/), config.json, the events.jsonl the script writes, stdout.log, and run.json
(this manager's bookkeeping: pid, timestamps). The run's state is derived from its events
plus whether its process is still alive, so there is no second source of truth to drift.

Exporting a run zips the directory: what you download is exactly what ran.

Recordings are kept live runs that can be replayed without AWS. This machine's own are in
data/recordings/; the ones that come with the project are in their card's directory,
templates/<id>/recordings/ (read-only, scrubbed of the recording account's ID by
scripts/ship_recording.py), so a card left out of a release takes its recordings with it.
"""

from __future__ import annotations

import io
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

from .errors import ApiError, NotFound
from .events import read_events, summarize
from .scenarios import RECORDINGS, ScenarioRegistry

BACKENDS = ("live",)
# templates/_shared/ files copied into every run, and into every replay: the platform's code is always this
# installation's, not a recording's (scripts/ship_recording.py does not ship it).
SHARED = ("srl_events.py", "srl_mlflow.py", "ts_client.py", "srl_replay.py", "mtrl_client.py",
          "sandbox_client.py", "sandbox_exec.py")
COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "scenario.json", "scenario.zh.json", "data",
                                     "variants", RECORDINGS)
EXPORT_SKIP_DIRS = ("__pycache__", "inbox")
MAX_QUESTION_CHARS = 500
REQUIREMENTS = (
    "# Versions the templates were verified with on 2026-10-27.\n"
    "sagemaker-train[tokenizer]==1.27.0\n"
    "boto3>=1.43.119\n"
)
STOP_GRACE_S = 120   # a live stop() normally returns in seconds; save_state can take ~53 s
TERM_GRACE_S = 15


class RunError(ApiError):
    """A request about a run that cannot be honored; the message says why. 409 unless given a status."""


def _new_run_id() -> str:
    return f"r-{datetime.now().strftime('%y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


class RunManager:
    def __init__(self, data_dir: Path, registry: ScenarioRegistry):
        self.registry = registry
        self.runs_dir = data_dir / "runs"
        self.recordings_dir = data_dir / "recordings"
        self.shipped_dirs = registry.shipped_recording_dirs()  # come with the project; never written here
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    # -- creating runs ------------------------------------------------------------------

    @staticmethod
    def check_backend(backend: str) -> None:
        if backend not in BACKENDS:
            raise RunError(f"backend must be one of {', '.join(BACKENDS)}, got {backend!r}", 400)

    def create(self, scenario_id: str, backend: str, params: dict, aws: dict,
               variants: dict | None = None, files: dict | None = None, mlflow: dict | None = None,
               mlflow_mode: str | None = None) -> dict:
        self.check_backend(backend)
        meta = self.registry.get(scenario_id)
        plan = self.registry.resolve_slots(scenario_id, variants or {}, files or {})
        chosen = {**plan["chosen"], **{sl["key"]: params[sl["param"]] for sl in meta.get("slots", []) if "param" in sl}}
        run_id = _new_run_id()
        d = self.runs_dir / run_id
        d.mkdir()
        shared = self.registry.dir / "_shared"
        for name in SHARED:
            shutil.copy2(shared / name, d / name)
        template = self.registry.template_dir(scenario_id)
        shutil.copytree(template, d, ignore=COPY_IGNORE, dirs_exist_ok=True)
        for target, source in plan["copy"].items():  # the chosen variant takes the slot's file name
            if source != target:
                shutil.copy2(template / source, d / target)
        for target, code in plan["write"].items():
            (d / target).write_text(code, encoding="utf-8")
        config = {"run_id": run_id, "scenario": scenario_id, "backend": backend, "aws": aws, "params": params,
                  "variants": chosen, "custom_files": sorted(plan["write"])}
        if mlflow:
            config["mlflow"] = mlflow  # srl_events.open_log publishes the run there (srl_mlflow.py)
        _write_json(d / "config.json", config)
        (d / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
        (d / "README.md").write_text(_readme(meta, config), encoding="utf-8")
        _write_json(d / "run.json", {"id": run_id, "scenario": scenario_id, "title": meta["title"],
                                     "backend": backend, "engine": meta.get("engine", "training-session"),
                                     "created_ts": time.time(), "files": meta["files"],
                                     "metrics": meta["metrics"], "region": aws["region"],
                                     "variants": chosen, "custom_files": sorted(plan["write"]),
                                     "loop": meta.get("loop"), "slots": _slot_labels(meta),
                                     "probes": meta.get("probes", True), "probe_example": meta.get("probe_example"),
                                     "mlflow_mode": mlflow_mode})
        self._spawn(run_id, ["loop.py"], aws["region"])
        return self.summary(run_id)

    def create_replay(self, recording: str, speed: float) -> dict:
        src = self._recording_dir(recording)
        info = json.loads((src / "recording.json").read_text(encoding="utf-8"))
        shipped = src.parent != self.recordings_dir
        meta = self.registry.get(info["scenario"])
        run_id = _new_run_id()
        d = self.runs_dir / run_id
        d.mkdir()
        shutil.copytree(src, d, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("events.jsonl", "stdout.log", "run.json", "recording.json",
                                                      "config.json", "data", "__pycache__", "inbox"))
        for name in SHARED:  # shipped recordings carry none of these; this machine's own may carry old copies
            shutil.copy2(self.registry.dir / "_shared" / name, d / name)
        config = json.loads((src / "config.json").read_text(encoding="utf-8"))
        config.update(run_id=run_id, backend="replay", recording=recording)
        config.pop("mlflow", None)  # the recorded run already published; its own mlflow events say where
        _write_json(d / "config.json", config)
        _write_json(d / "run.json", {"id": run_id, "scenario": info["scenario"], "title": meta["title"],
                                     "backend": "replay", "recording": recording, "created_ts": time.time(),
                                     "engine": meta.get("engine", "training-session"),
                                     "files": meta["files"], "metrics": meta["metrics"],
                                     "region": config["aws"]["region"], "variants": config.get("variants", {}),
                                     "custom_files": config.get("custom_files", []),
                                     "loop": meta.get("loop"), "slots": _slot_labels(meta),
                                     "probes": meta.get("probes", True), "probe_example": meta.get("probe_example"),
                                     "mlflow_mode": "replay",
                                     "recording_shipped": shipped})
        self._spawn(run_id, ["srl_replay.py", str(src / "events.jsonl"), "--speed", str(speed)],
                    config["aws"]["region"])
        return self.summary(run_id)

    def record(self, run_id: str) -> dict:
        """Keep a finished live run as a recording that can be replayed without AWS."""
        s = self.summary(run_id)
        if s["backend"] != "live":
            raise RunError(f"only live runs can be recorded; {run_id} is a {s['backend']} run")
        if s["status"] != "completed":
            raise RunError(f"only completed runs can be recorded; {run_id} is {s['status']}")
        name = f"{s['scenario']}-{datetime.fromtimestamp(s['started_ts']).strftime('%y%m%d')}-{run_id}"
        dst = self.recordings_dir / name
        if dst.exists() or any((root / name).exists() for root in self.shipped_dirs):
            raise RunError(f"recording {name} already exists")
        shutil.copytree(self._dir(run_id), dst, ignore=shutil.ignore_patterns("run.json", "*.tmp"))
        info = {"name": name, "source_run": run_id, "scenario": s["scenario"], "recorded_ts": s["started_ts"],
                "model": s["model"], "region": s["region"], "duration_s": s["duration_s"],
                "cost_usd": s["cost_usd"], "tokens": s["tokens"], "eval": s["eval"]}
        _write_json(dst / "recording.json", info)
        return info

    def recording_roots(self) -> list[Path]:
        """Where recordings are, this machine's first."""
        return [self.recordings_dir, *self.shipped_dirs]

    def recordings(self) -> list[dict]:
        """Every recording, oldest first. A run kept here and also shipped is listed once, as this machine's."""
        out, seen = [], set()
        for root in self.recording_roots():
            if not root.is_dir():
                continue
            for d in sorted(root.iterdir()):
                if not (d / "recording.json").exists():
                    continue
                info = json.loads((d / "recording.json").read_text(encoding="utf-8"))
                if info["source_run"] in seen:
                    continue
                seen.add(info["source_run"])
                out.append({**info, "shipped": root != self.recordings_dir})
        return sorted(out, key=lambda r: r["recorded_ts"])

    # -- process control ----------------------------------------------------------------

    def _spawn(self, run_id: str, argv: list[str], region: str) -> None:
        d = self._dir(run_id)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "AWS_DEFAULT_REGION": region, "SRL_EVENTS": str(d / "events.jsonl")}
        log = open(d / "stdout.log", "ab")
        proc = subprocess.Popen([sys.executable, *argv], cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)  # own process group: Stop signals reach the whole run
        log.close()
        with self._lock:
            self._procs[run_id] = proc
        self._update_meta(run_id, pid=proc.pid, command=[Path(sys.executable).name, *argv])
        threading.Thread(target=self._reap, args=(run_id, proc), daemon=True).start()

    def _reap(self, run_id: str, proc: subprocess.Popen) -> None:
        code = proc.wait()
        self._update_meta(run_id, exit_code=code, exited_ts=time.time())
        events = read_events(self._events_path(run_id))
        if not any(e["type"] == "run_finished" for e in events):
            stopping = self._meta(run_id).get("stop_requested_ts") is not None
            self._append(run_id, events, "run_finished", status="stopped" if stopping else "failed",
                         error=None if stopping else
                         f"the run's process exited with code {code} before reporting an outcome; see stdout.log",
                         source="backend")

    def _alive(self, run_id: str, meta: dict) -> bool:
        with self._lock:
            proc = self._procs.get(run_id)
        if proc is not None:
            return proc.poll() is None
        pid = meta.get("pid")
        if not pid:
            return False
        try:  # after a backend restart: same pid AND running in this run's directory
            return os.readlink(f"/proc/{pid}/cwd") == str(self._dir(run_id))
        except OSError:
            return False

    def stop(self, run_id: str) -> dict:
        meta = self._meta(run_id)
        if not self._alive(run_id, meta):
            raise RunError(f"{run_id} is not running")
        self._update_meta(run_id, stop_requested_ts=time.time())
        pid = meta["pid"]
        os.killpg(pid, signal.SIGINT)  # the loop's finally stops the session

        def escalate():
            for sig, grace in ((signal.SIGTERM, STOP_GRACE_S), (signal.SIGKILL, TERM_GRACE_S)):
                deadline = time.time() + grace
                while time.time() < deadline:
                    if not self._alive(run_id, meta):
                        return
                    time.sleep(1)
                try:
                    os.killpg(pid, sig)
                except ProcessLookupError:
                    return
        threading.Thread(target=escalate, daemon=True).start()
        return self.summary(run_id)

    def probe(self, run_id: str, question) -> dict:
        """Queue a question for a running loop. The loop answers it between steps; nothing is called here."""
        meta = self._meta(run_id)
        if meta.get("engine", "training-session") != "training-session":
            raise RunError(f"{run_id} is an MTRL run; SageMaker AI runs that loop, so there is no step to answer between")
        if meta["backend"] == "replay":
            raise RunError(f"{run_id} is a replay; it repeats recorded events and cannot answer new questions")
        if not meta.get("probes", True):
            raise RunError(f"{run_id} runs {meta['scenario']}, whose loop does not answer questions between steps "
                           "(scenario.json sets probes: false)")
        status = self.summary(run_id)["status"]
        if status not in ("starting", "running"):
            raise RunError(f"{run_id} is not running ({status}), so nothing will answer")
        if not isinstance(question, str) or not question.strip():
            raise RunError("the question is empty", 400)
        if len(question) > MAX_QUESTION_CHARS:
            raise RunError(f"the question has {len(question)} characters; the limit is {MAX_QUESTION_CHARS}", 400)
        inbox = self._dir(run_id) / "inbox"
        inbox.mkdir(exist_ok=True)
        probe_id = f"q-{time.time_ns()}-{secrets.token_hex(2)}"
        tmp = inbox / f"{probe_id}.tmp"
        tmp.write_text(json.dumps({"id": probe_id, "question": question.strip(), "asked_ts": time.time()}), encoding="utf-8")
        tmp.replace(inbox / f"{probe_id}.json")  # the loop only reads complete .json files
        return {"id": probe_id, "question": question.strip()}

    # -- reading ------------------------------------------------------------------------

    def summary(self, run_id: str) -> dict:
        meta = self._meta(run_id)
        s = summarize(read_events(self._events_path(run_id)))
        alive = self._alive(run_id, meta)
        status = s["status"]
        if status is None:
            if alive:
                # Running once the training session is up, or (MTRL) once the job exists.
                status = "stopping" if meta.get("stop_requested_ts") else (
                    "running" if s["session_state"] == "running" or s["job_name"] else "starting")
            else:
                status = "interrupted"
        backend = s["backend"] or meta["backend"]
        loop, slots = meta.get("loop"), meta.get("slots")
        if loop is None:  # runs created before loops were described: use the scenario's current description
            try:
                current = self.registry.get(meta["scenario"])
                loop, slots = current.get("loop"), _slot_labels(current)
            except NotFound:  # its scenario no longer exists, or is only planned: no loop map for this run
                pass
        return {**s, "id": run_id, "scenario": meta["scenario"], "title": meta["title"], "backend": backend,
                "engine": meta.get("engine", s["engine"]),
                "status": status, "alive": alive, "created_ts": meta["created_ts"], "files": meta["files"],
                "metrics": meta["metrics"], "recording": meta.get("recording"), "region": s["region"] or meta["region"],
                "exit_code": meta.get("exit_code"), "variants": meta.get("variants", {}),
                "custom_files": meta.get("custom_files", []), "loop": loop, "slots": slots or [],
                "probes": meta.get("probes", True), "probe_example": meta.get("probe_example"),
                "mlflow_mode": meta.get("mlflow_mode"),
                "recording_shipped": meta.get("recording_shipped", False)}

    def list(self) -> list[dict]:
        ids = sorted((d.name for d in self.runs_dir.iterdir() if (d / "run.json").exists()), reverse=True)
        return [self.summary(i) for i in ids]

    def config(self, run_id: str) -> dict:
        return json.loads((self._dir(run_id) / "config.json").read_text(encoding="utf-8"))

    def events_path(self, run_id: str) -> Path:
        self._meta(run_id)
        return self._events_path(run_id)

    def is_alive(self, run_id: str) -> bool:
        return self._alive(run_id, self._meta(run_id))

    def read_file(self, run_id: str, name: str) -> str:
        allowed = set(self._meta(run_id)["files"]) | {"config.json", "stdout.log", "README.md", "requirements.txt"}
        if name not in allowed:
            raise RunError(f"{name} is not a file of {run_id}", 404)
        path = (self._dir(run_id) / name).resolve()
        if self._dir(run_id).resolve() not in path.parents:
            raise RunError(f"{name} is not a file of {run_id}", 404)
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    def export_zip(self, run_id: str) -> bytes:
        d = self._dir(run_id)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(d.rglob("*")):
                rel = f.relative_to(d)
                if (f.is_file() and rel.name != "run.json" and f.suffix not in (".tmp", ".pyc")
                        and not set(EXPORT_SKIP_DIRS) & set(rel.parts)):
                    z.write(f, f"{run_id}/{rel.as_posix()}")
        return buf.getvalue()

    # -- internals ----------------------------------------------------------------------

    def _dir(self, run_id: str) -> Path:
        d = self.runs_dir / run_id
        if not run_id.startswith("r-") or d.parent != self.runs_dir or not (d / "run.json").exists():
            raise RunError(f"unknown run {run_id!r}", 404)
        return d

    def _recording_dir(self, name: str) -> Path:
        if "/" not in name and not name.startswith("."):
            for root in self.recording_roots():
                if (root / name / "recording.json").exists():
                    return root / name
        raise RunError(f"unknown recording {name!r}", 404)

    def _events_path(self, run_id: str) -> Path:
        return self.runs_dir / run_id / "events.jsonl"

    def _meta(self, run_id: str) -> dict:
        return json.loads((self._dir(run_id) / "run.json").read_text(encoding="utf-8"))

    def _update_meta(self, run_id: str, **fields) -> None:
        with self._lock:
            path = self._dir(run_id) / "run.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            meta.update(fields)
            _write_json(path, meta)

    def _append(self, run_id: str, events: list[dict], type: str, **fields) -> None:
        """Write an event on the run's behalf; only called once its process has exited."""
        last_t = events[-1]["t"] if events else 0.0
        event = {"v": 1, "seq": len(events), "ts": round(time.time(), 3), "t": last_t, "run": run_id,
                 "type": type, **fields}
        with open(self._events_path(run_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")


def _slot_labels(meta: dict) -> list[dict]:
    """The slot fields a run console needs to label its loop map (no code, no long text)."""
    return [{"key": sl["key"], "label": sl["label"], "node": sl.get("node"), "file": sl.get("file"),
             "param": sl.get("param"), "function": sl.get("function"),
             "variants": [{"id": v["id"], "label": v["label"], "short": v.get("short")} for v in sl["variants"]]}
            for sl in meta.get("slots", [])]


def _readme(meta: dict, config: dict) -> str:
    files = "\n".join(f"- `{f}`" for f in meta["files"])
    labels = {sl["key"]: sl for sl in meta.get("slots", [])}
    chosen = config.get("variants") or {}
    slot_lines = []
    for key, value in chosen.items():
        sl = labels[key]
        name = "edited in the browser" if value == "custom" else next(
            (v["label"] for v in sl["variants"] if v["id"] == value), str(value))
        where = f"`{sl['file']}`" if sl.get("file") else f"`{sl['param']}` in config.json"
        slot_lines.append(f"- {sl['label']} ({where}): {name}")
    slots = ("\n## Slots this run used\n\n" + "\n".join(slot_lines) + "\n") if slot_lines else ""
    helpers = ["- `srl_events.py`, `ts_client.py`, `mtrl_client.py`: event log, training-session client, MTRL jobs"]
    helpers.append("- `srl_mlflow.py`: publishes a live run to the MLflow app in `config.json` (`mlflow`), if it names one")
    helpers.append("- `sandbox_client.py`, `sandbox_exec.py`: the environment sandbox, for scenarios that run code there "
                   "(`aws.sandbox` in `config.json`)")
    commands = [f"python loop.py                      # backend from config.json (\"{config['backend']}\")",
                "SRL_BACKEND=live python loop.py     # on SageMaker AI, with your AWS credentials"]
    never = ""
    helpers_text, commands_text = "\n".join(helpers), "\n".join(commands)
    return f"""# {meta['title']}

Run `{config['run_id']}` exported from Serverless RL InstantStart.

{meta['summary']}
{slots}
## Files

{files}
- `config.json`: the parameters and AWS resources this run used
{helpers_text}
- `events.jsonl`: everything this run recorded, one event per line (if it ran)

## Run it again

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
{commands_text}
```

A live run needs the AWS resources named in `config.json` to exist in your account (the
app's Setup page checks and creates them). Training-session runs stop their session, and
every sandbox session they open, in every exit path. MTRL runs submit a job that runs on
SageMaker AI until it finishes or is stopped. Model packages stay in their group until you
delete them. A live run whose `config.json` has an `mlflow` entry publishes a new MLflow run
there; delete the entry to run without MLflow.{never}
"""
