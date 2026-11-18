"""Multi-turn RL (MTRL) jobs for the run templates: submit, watch, stop.

In MTRL, SageMaker AI runs the training loop and calls your agent for every rollout. The
run's own process only prepares, submits and watches, so a job keeps running if this
process dies; ``MultiTurnRLTrainer.attach(job_name)`` reconnects to it.

``mtrl(config)`` returns ``LiveMtrl`` (SDK ``MultiTurnRLTrainer`` / ``AgentRFTJob``).

Every resource is passed explicitly (model package groups, MLflow app, role): left out,
the SDK would get-or-create default ones, and a demo should not create resources nobody
asked for.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field

TERMINAL = ("Completed", "Failed", "Stopped")

# Our parameter names -> MTRL hyperparameter names (see trainer.hyperparameters.get_info()).
HYPERPARAMETERS = {
    "max_steps": "max_steps", "global_batch_size": "global_batch_size", "group_size": "group_size",
    "loss_fn": "loss_fn", "advantage_method": "advantage_method", "learning_rate": "learning_rate",
    "lora_rank": "lora_rank", "sampling_max_tokens": "sampling_max_tokens", "temperature": "temperature",
    "max_steps_off_policy": "max_steps_off_policy", "rollout_max_concurrency": "rollout_max_concurrency",
}


@dataclass
class JobState:
    status: str
    secondary: str | None
    transitions: list[dict] = field(default_factory=list)  # {status, start, end, message}; wall-clock seconds
    progress: dict | None = None                           # {"current": n, "max": n}
    billing: dict | None = None                            # {"prefill", "sample", "train"}
    output_model_package_arn: str | None = None
    failure_reason: str | None = None


BACKENDS = ("live",)


def backend(config: dict) -> str:
    value = os.environ.get("SRL_BACKEND") or config.get("backend", "live")
    if value not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}, got {value!r}")
    return value


def mtrl(config: dict):
    return LiveMtrl(config)


def _val(x):
    """SDK shapes use an ``Unassigned`` sentinel for missing fields."""
    return None if x is None or type(x).__name__ == "Unassigned" else x


def _ts(x) -> float | None:
    x = _val(x)
    return x.timestamp() if x is not None else None


# -- live ----------------------------------------------------------------------------------

class LiveMtrl:
    def __init__(self, config: dict):
        self.aws = config["aws"]
        self.m = self.aws["mtrl"]
        self.region = self.aws["region"]
        os.environ.setdefault("AWS_DEFAULT_REGION", self.region)
        # The SDK's MLflow client needs the regional endpoint; the sample set it the same way.
        os.environ.setdefault("SAGEMAKER_MLFLOW_CUSTOM_ENDPOINT", f"https://mlflow.sagemaker.{self.region}.app.aws")

    def upload(self, files: dict[str, str], s3_prefix: str) -> dict[str, str]:
        import boto3
        bucket, _, key_prefix = s3_prefix.removeprefix("s3://").partition("/")
        s3 = boto3.client("s3", region_name=self.region)
        uris = {}
        for channel, path in files.items():
            key = f"{key_prefix.rstrip('/')}/{os.path.basename(path)}"
            s3.upload_file(path, bucket, key)
            uris[channel] = f"s3://{bucket}/{key}"
        return uris

    def prepare(self, p: dict, datasets: dict[str, str], output_path: str):
        """Build the trainer and run the SDK's own validation (role permissions, schema)."""
        from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer
        trainer = MultiTurnRLTrainer(
            model=p["model"],
            agent_env=self.m["runtime_arn"],
            training_dataset=datasets["train"],
            validation_dataset=datasets.get("validation"),
            s3_output_path=output_path,
            output_model_package_group=self.m["output_group_arn"],
            intermediate_checkpoint_model_package_group=self.m["checkpoint_group_arn"],
            mlflow_app_arn=self.m["mlflow_app_arn"],
            mlflow_experiment_name=self.m.get("mlflow_experiment", "serverless-rl-instantstart"),
            role=self.m["job_role_arn"],
            accept_eula=True,
        )
        hp = trainer.hyperparameters
        for ours, theirs in HYPERPARAMETERS.items():
            if ours in p:
                setattr(hp, theirs, p[ours])
        trainer.train(dry_run=True)  # raises on what the SDK can check locally (e.g. RoleValidationError)
        return trainer

    def submit(self, trainer) -> "LiveJob":
        return LiveJob(trainer.train(wait=False))

    def agent_logs(self) -> "AgentLogs":
        return AgentLogs(self.m["runtime_arn"], self.region)

    def agent_version(self) -> str | None:
        """The agent runtime's current version. The job calls the DEFAULT endpoint, which follows the latest one."""
        import boto3
        rt = boto3.client("bedrock-agentcore-control", region_name=self.region).get_agent_runtime(
            agentRuntimeId=self.m["runtime_arn"].rsplit("/", 1)[-1])
        return rt.get("agentRuntimeVersion")


class LiveJob:
    def __init__(self, job):
        self._job = job
        self.name = job.job_name
        self.arn = job.job_arn

    def state(self) -> JobState:
        self._job.refresh()
        transitions = []
        for t in _val(self._job.secondary_status_transitions) or []:
            transitions.append({"status": t.status, "start": _ts(t.start_time), "end": _ts(t.end_time),
                                "message": _val(t.status_message)})
        info = self._job.progress_info
        progress = None
        if info:
            progress = {"current": int(info.get("CurrentStep") or 0),
                        "max": int(info.get("MaxSteps") or (info.get("MaxEpoch") or 0) * (info.get("StepsPerEpoch") or 0))}
        usage = self._job.billable_token_usage
        billing = None
        if usage:
            billing = {"prefill": int(usage.get("PrefillTokenCount", 0)), "sample": int(usage.get("SampleTokenCount", 0)),
                       "train": int(usage.get("TrainTokenCount", 0))}
        return JobState(status=self._job.job_status, secondary=_val(self._job.secondary_status),
                        transitions=transitions, progress=progress, billing=billing,
                        output_model_package_arn=self._job.output_model_package_arn,
                        failure_reason=_val(self._job.failure_reason))

    def metrics(self) -> list[dict]:
        """Per-step rows from MLflow: rollout/reward/mean, rollout/turns/mean, training/*."""
        return self._job.get_training_metrics()

    def mlflow_url(self) -> str | None:
        return self._job.get_mlflow_url()

    def stop(self) -> None:
        self._job.stop()


class AgentLogs:
    """Rollouts your agent handled, read from its AgentCore Runtime log group.

    Relies on the line sample/rollout_handler.py logs for every rollout
    ("case=... reward=... decision=x/y coverage=..."). A different agent needs its own
    pattern here.
    """

    LINE = re.compile(r"case=(\S+) reward=(\[[^\]]*\]|\S+) decision=(\S+)/(\S+) coverage=([\d.]+)")
    # Events become searchable seconds after their timestamp (0.1-10.8 s on 2026-11-06), so
    # each poll looks back this far and drops repeats by eventId. Starting at the newest
    # timestamp seen instead missed 3 of 80 rollouts that run.
    LOOKBACK_MS = 120_000

    def __init__(self, runtime_arn: str, region: str, qualifier: str = "DEFAULT", logs_client=None):
        if logs_client is None:
            import boto3
            logs_client = boto3.client("logs", region_name=region)
        self._logs = logs_client
        self.group = f"/aws/bedrock-agentcore/runtimes/{runtime_arn.rsplit('/', 1)[-1]}-{qualifier}"
        self._start_ms = int(time.time() * 1000)
        self._since_ms = self._start_ms
        self._seen: set[str] = set()

    def poll(self, training: bool = True, step: int = 0) -> list[dict]:  # noqa: ARG002 (every agent-log reader has this signature)
        out, token = [], None
        start = max(self._start_ms, self._since_ms - self.LOOKBACK_MS)
        while True:
            kw = {"logGroupName": self.group, "startTime": start, "filterPattern": '"reward="'}
            if token:
                kw["nextToken"] = token
            r = self._logs.filter_log_events(**kw)
            for e in r.get("events", []):
                if e["eventId"] in self._seen:
                    continue
                self._seen.add(e["eventId"])
                m = self.LINE.search(e["message"])
                if not m:
                    continue
                reward = m.group(2)
                value = sum(map(float, reward.strip("[]").split(","))) if reward.startswith("[") else float(reward)
                out.append({"ts": e["timestamp"] / 1000, "case": m.group(1), "reward": value,
                            "decision": m.group(3), "expected": m.group(4), "coverage": float(m.group(5))})
                self._since_ms = max(self._since_ms, e["timestamp"])
            token = r.get("nextToken")
            if not token:
                break
        return sorted(out, key=lambda x: x["ts"])


def rollout_rate_hint(p: dict) -> str:
    n = p["max_steps"] * p["global_batch_size"] * p["group_size"]
    return (f"{p['max_steps']} steps x {p['global_batch_size']} prompts x {p['group_size']} samples = {n:,} rollouts; "
            f"the sample sent ~15.8K characters per rollout (perfect policy), so about {math.ceil(n * 15_801 / 1e6)}M characters of prefill")
