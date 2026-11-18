"""Publish a run's events to MLflow: a SageMaker AI MLflow App (by ARN) or any MLflow tracking URI.

The event log stays the source of truth; this sink mirrors it into MLflow, so a run sits next to
the team's other experiments. It follows the AWS cookbook's MLflow publishing
(training_sessions_cookbook/service/mlflow_log.py, commit 7f2d3e0): one MLflow run per training
run, params at the start, per-step metrics, model package ARNs as tags, a few rollouts per step
as traces, the run's code as artifacts, and FINISHED / FAILED / KILLED at the end.

What goes where (docs/architecture.md 3.8):
    run_started   experiment serverless-rl-instantstart/<scenario> (tagged as model training, or
                  the MLflow 3 UI lists the runs as evaluations), a run named after the run id,
                  params = the run's parameters (values cut to MLflow's 500 characters), tags srl.*
    session       tag srl.session; metric session/ready_sec
    span          token counts, published with the next metrics as tokens/prefill|sample|train,
                  and once more at the end, so the last value is the run's total
    metrics       every numeric value at its step, plus time/elapsed_sec (seconds since the run
                  started) and, when config.json carries list prices, cost/est_usd: the tokens so
                  far at those prices, an estimate (BillableTokenUsage is the bill)
    rollouts      up to `traces_per_step` rollouts of a step, one per group, as traces named
                  rollout_step<N>_<i> with tags step and reward
    artifact      tag model_package.<nn> = "<KIND> <arn> <description>"
    eval          metrics eval/before, eval/after, eval/n
    compare, reward_check, policy_check   JSON artifacts; handled notes as handled.json at the end
    sandbox       ready: tag srl.sandbox (where it ran, and the runtime); batch: metrics sandbox/sessions
                  (opened so far) and sandbox/cold_start_p50_sec at the batch's step; closed: sandbox.json.
                  Not the AgentCore cost: its usage logs arrive about 20 minutes after the run
    run_finished  the run's *.py files and config.json as artifacts under code/; the MLflow run
                  ends FINISHED (completed), FAILED (failed) or KILLED (stopped)

Calls run on one background thread, in event order, so a slow tracking server (starting a run in a
SageMaker MLflow App took 22.4 s once and 1.5 s once, 2026-10-29) never lengthens a training step. A
failure is written to the run's own log as a warning, once per kind of call; after five failures
in a row publishing stops, with a warning saying so. Training goes on either way: losing the
mirror must not lose a billed run, and nothing is lost from the log. The sink writes `mlflow`
events (state publishing, failed or ended) so the console can show where the run went.
"""

from __future__ import annotations

import numbers
import os
import queue
import threading

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # mlflow 3.17 prints a hint for coding agents on import

MAX_PARAM_CHARS = 500     # MLflow's limit for a param value
MAX_TAG_CHARS = 5000      # and for a tag value
MAX_FAILURES = 5          # in a row, before publishing stops
FINISH_TIMEOUT_S = 60     # how long the end of a run waits for the queue to drain; Stop allows 120 s in all
END_STATUS = {"completed": "FINISHED", "failed": "FAILED", "stopped": "KILLED"}


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


class MlflowSink:
    def __init__(self, ev, config: dict):
        m = config["mlflow"]
        self.ev = ev
        self.uri = m["tracking_uri"]
        self.experiment = m["experiment"]
        self.traces_per_step = int(m.get("traces_per_step", 4))
        self.extra_tags = dict(m.get("tags") or {})
        self.prices = m.get("prices")  # USD per million tokens, by kind; absent for a model with no known price
        self.run_name = config["run_id"]
        self._mlflow = None
        self._run = None
        self._stopped = False
        self._failures = 0
        self._reported: set[str] = set()
        self._tokens = {"prefill": 0, "sample": 0, "train": 0}
        self._last_step = 0
        self._last_t = 0.0
        self._artifacts = 0
        self._handled: list[dict] = []
        self._published = {"metric_steps": 0, "traces": 0, "model_packages": 0}
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._work, name="srl-mlflow", daemon=True)
        self._thread.start()

    # -- called by EventLog ------------------------------------------------------------

    def on_event(self, event: dict) -> None:
        if event["type"] in ("mlflow", "log", "span_start", "step", "probe"):
            return  # its own events, and what MLflow has no place for
        self._q.put(event)

    def finish(self, status: str | None) -> None:
        self._q.put(("finish", status))
        try:
            self._thread.join(FINISH_TIMEOUT_S)
        except KeyboardInterrupt:  # a second Stop while waiting: the process is ending anyway, so let run_finished be written
            self.ev.log("MLflow: stopped while publishing the end of the run; the MLflow run may stay RUNNING. "
                        "events.jsonl has everything.", level="warn")
            return
        if self._thread.is_alive():
            self.ev.log(f"MLflow: publishing had not finished after {FINISH_TIMEOUT_S} s, so the run ends without it. "
                        "Whatever did not reach MLflow is in events.jsonl.", level="warn")

    # -- the publishing thread ---------------------------------------------------------

    def _work(self) -> None:
        while True:
            item = self._q.get()
            if isinstance(item, tuple):
                self._end(item[1])
                return
            if self._stopped:
                continue
            try:
                self._handle(item)
            except Exception as e:  # reported in the run's log; the mirror must not stop training
                self._failed(item["type"], e)
            else:
                self._failures = 0

    def _where(self) -> dict:
        info = getattr(self._run, "info", None)
        return {"tracking_uri": self.uri, "experiment": self.experiment,
                "experiment_id": getattr(info, "experiment_id", None), "run_id": getattr(info, "run_id", None)}

    def _failed(self, what: str, e: Exception) -> None:
        self._failures += 1
        if what not in self._reported:
            self._reported.add(what)
            self.ev.log(f"MLflow: publishing {what} failed ({type(e).__name__}: {e}). Training continues; "
                        "events.jsonl has everything.", level="warn")
        if self._failures >= MAX_FAILURES and not self._stopped:
            self._stopped = True
            self.ev.log(f"MLflow: stopped publishing after {MAX_FAILURES} failures in a row. The run continues; "
                        "events.jsonl has everything.", level="warn")
            self.ev.emit("mlflow", state="failed", error=f"{type(e).__name__}: {e}", **self._where())

    def _handle(self, e: dict) -> None:
        t = e["type"]
        self._last_t = float(e.get("t") or self._last_t)  # the end of the run publishes the time of its last event
        if t == "run_started":
            self._start(e)
            return
        if self._run is None:
            return
        m = self._mlflow
        if t == "session":
            if e.get("name"):
                m.set_tag("srl.session", e["name"])
            if e.get("ready_sec") is not None:
                m.log_metric("session/ready_sec", float(e["ready_sec"]))
        elif t == "span":
            for k, v in (e.get("tokens") or {}).items():
                self._tokens[k] = self._tokens.get(k, 0) + v
        elif t == "metrics":
            values = {k: float(v) for k, v in e["values"].items() if isinstance(v, numbers.Real) and not isinstance(v, bool)}
            values.update(self._running_totals())
            self._last_step = int(e["step"])
            m.log_metrics(values, step=self._last_step)
            self._published["metric_steps"] += 1
        elif t == "rollouts":
            self._traces(int(e["step"]), e.get("items") or [])
        elif t == "artifact":
            self._artifacts += 1
            m.set_tag(f"model_package.{self._artifacts:02d}", f"{e['kind']} {e['arn']} {e.get('description') or ''}"[:MAX_TAG_CHARS])
            self._published["model_packages"] += 1
        elif t == "eval":
            m.log_metrics({"eval/before": float(e["before"]), "eval/after": float(e["after"]), "eval/n": float(e["n"])})
            m.set_tag("srl.eval", f"{e['label']} ({e.get('unit') or 'count'})"[:MAX_TAG_CHARS])
        elif t in ("compare", "reward_check", "policy_check"):
            m.log_dict({k: v for k, v in e.items() if k not in ("v", "seq", "run")}, f"{t}.json")
        elif t == "handled":
            self._handled.append({"code": e["code"], "message": e["message"]})
        elif t == "sandbox":
            if e["state"] == "ready":
                m.set_tag("srl.sandbox", f"{e.get('where')} {e.get('runtime') or ''}".strip()[:MAX_TAG_CHARS])
            elif e["state"] == "batch" and e.get("step") is not None:
                values = {"sandbox/sessions": float(e.get("created") or 0)}
                if e.get("cold_start"):
                    values["sandbox/cold_start_p50_sec"] = float(e["cold_start"]["p50"])
                m.log_metrics(values, step=int(e["step"]))
            elif e["state"] == "closed":
                m.log_dict({k: v for k, v in e.items() if k not in ("v", "seq", "run")}, "sandbox.json")

    def _running_totals(self) -> dict:
        out = {f"tokens/{k}": float(v) for k, v in self._tokens.items()}
        out["time/elapsed_sec"] = self._last_t
        if self.prices:
            out["cost/est_usd"] = sum(self._tokens.get(k, 0) * self.prices[k] for k in ("prefill", "sample", "train")) / 1e6
        return out

    def _start(self, e: dict) -> None:
        try:
            import mlflow
            self._mlflow = mlflow
            mlflow.set_tracking_uri(self.uri)
            exp = mlflow.set_experiment(self.experiment)
            if "mlflow.experimentKind" not in (exp.tags or {}):
                mlflow.MlflowClient().set_experiment_tag(exp.experiment_id, "mlflow.experimentKind", "custom_model_development")
            tags = {"srl.run_id": self.run_name, "srl.scenario": e.get("scenario"), "srl.engine": e.get("engine", "training-session"),
                    "srl.backend": e.get("backend"), "srl.model": e.get("model"), "srl.region": e.get("region"),
                    "srl.app": "Serverless RL InstantStart",
                    **{f"srl.slot.{k}": v for k, v in (e.get("variants") or {}).items()}, **self.extra_tags}
            self._run = mlflow.start_run(run_name=self.run_name, tags={k: str(v)[:MAX_TAG_CHARS] for k, v in tags.items() if v is not None})
            params = _flat(e.get("params") or {})
            if params:
                mlflow.log_params({k: str(v)[:MAX_PARAM_CHARS] for k, v in params.items()})
        except Exception as err:
            self._stopped = True
            self.ev.log(f"MLflow: could not start a run in {self.uri} ({type(err).__name__}: {err}). The run continues "
                        "without MLflow; events.jsonl has everything.", level="warn")
            self.ev.emit("mlflow", state="failed", error=f"{type(err).__name__}: {err}", **self._where())
            return
        self.ev.emit("mlflow", state="publishing", **self._where())

    def _traces(self, step: int, items: list[dict]) -> None:
        """One rollout per group, up to traces_per_step: a sample of what the model wrote, not all of it."""
        picked, seen = [], set()
        for it in items:
            if it.get("group") not in seen:
                seen.add(it.get("group"))
                picked.append(it)
            if len(picked) >= self.traces_per_step:
                break
        m = self._mlflow
        for i, it in enumerate(picked):
            reward = float(it.get("reward") or 0.0)
            with m.start_span(name=f"rollout_step{step}_{i}", span_type="LLM",
                              attributes={"step": step, "group": it.get("group"), "reward": reward}) as span:
                m.update_current_trace(tags={"step": str(step), "reward": f"{reward:.4f}", "srl.run_id": self.run_name})
                span.set_inputs({"prompt": it.get("prompt"), "answer": it.get("answer")})
                span.set_outputs({"completion": it.get("completion"), "reward": reward, "stop_reason": it.get("stop_reason")})
            self._published["traces"] += 1

    def _end(self, status: str | None) -> None:
        if self._run is None:
            return
        m = self._mlflow
        try:
            # Tokens spent after the last step (held-out eval, final saves), so the run ends on the full count.
            m.log_metrics(self._running_totals(), step=self._last_step)
            for name in sorted(os.listdir(".")):
                if (name.endswith(".py") or name == "config.json") and os.path.isfile(name):
                    m.log_artifact(name, artifact_path="code")
            if self._handled:
                m.log_dict(self._handled, "handled.json")
            flush = getattr(m, "flush_trace_async_logging", None)
            if flush:
                flush()
            m.end_run(status=END_STATUS.get(status or "", "FAILED"))
        except Exception as e:
            self._failed("the end of the run", e)
            try:
                m.end_run(status=END_STATUS.get(status or "", "FAILED"))
            except Exception:  # already reported above; nothing more to do
                pass
            return
        self.ev.emit("mlflow", state="ended", status=END_STATUS.get(status or "", "FAILED"), published=self._published,
                     **self._where())
