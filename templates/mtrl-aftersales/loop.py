"""Train the after-sales agent with multi-turn RL (MTRL): SageMaker AI runs the loop and calls your agent.

    python loop.py                      # live: uses config.json next to this file

What runs where:
    your process    reward check (8 scripted policies through the real reward and SDK hooks),
                    dataset generation and upload, SDK validation, job submission, watching
    your agent      sample/ (agent loop, tools, reward) in AgentCore Runtime. SageMaker AI
                    calls it for every rollout. It is deployed separately with
                    deploy/setup_aws.sh; this run does not rebuild it.
    SageMaker AI    the training loop: sampling, dispatching rollouts, advantages, updates,
                    checkpoints, the output model package

Based on the 260824 after-sales sample. Its agent, tools, reward and AgentCore deployment
were verified on 2026-08-24. CreateJob was refused for this internal account until it was
allowlisted; the first live job ran on 2026-11-06 with the smallest settings (1 step,
32 prompts x 2 rollouts) and completed. The default settings have not run live.
"""

import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import reward_check
from mtrl_client import TERMINAL, backend, mtrl, rollout_rate_hint
from sample.dataset import generate_cases, to_rows, write_jsonl
from srl_events import install_stop_handlers, load_config, open_log

config = load_config()
p = config["params"]
ev = open_log(config)
install_stop_handlers()
mode = backend(config)
ev.emit("run_started", scenario=config["scenario"], engine="mtrl", backend=mode, model=p["model"],
        region=config["aws"]["region"], total_steps=p["max_steps"], params=p)
POLL_S = 30 if mode == "live" else 3


def fail(message: str, code: int = 1):
    ev.log(message, level="error")
    ev.finished("failed", message)
    sys.exit(code)


try:
    # 1. Reward check: no AWS calls, no spend. Do not submit if the reward cannot tell good from bad.
    with ev.span("client", "reward_check", note=f"8 scripted policies x {p['reward_check_cases']} cases, real reward + SDK hooks"):
        rows, checks = reward_check.run(p["reward_check_cases"], seed=p["dataset_seed"] + 7, region=config["aws"]["region"])
    ev.emit("policy_check", rows=rows, checks=checks, n=p["reward_check_cases"])
    failed = [c["check"] for c in checks if not c["ok"]]
    if failed:
        fail("Reward check failed, so no job was submitted: " + "; ".join(failed))
    ev.handled("reward_before_spend", "The reward was checked against 8 scripted policies before anything was "
               "submitted: the SOP-following policy scores 1.0, the reward-hacking probe stays below 0.5, prose scores 0.")

    # 2. Dataset: train and eval from the same generator with different seeds, four decisions balanced.
    with ev.span("client", "build_dataset") as s:
        data_dir = Path("data")
        splits = {"train": (p["train_size"], p["dataset_seed"]), "validation": (p["eval_size"], p["dataset_seed"] + 1)}
        files, stats = {}, {}
        for split, (n, seed) in splits.items():
            cases = generate_cases(n, seed=seed, balanced=True)
            path = data_dir / f"{split}.jsonl"
            rows_out = list(to_rows(cases, key=None))
            write_jsonl(rows_out, path)
            files[split] = str(path)
            stats[split] = {"n": n, "seed": seed,
                            "decisions": dict(Counter(c["reward_spec"]["ground_truth"]["decision"] for c in cases)),
                            "avg_prompt_chars": round(statistics.fmean(len(r["prompt"]) for r in rows_out))}
        s["note"] = f"{p['train_size']} train / {p['eval_size']} eval, plaintext"
    ev.emit("dataset", splits=stats, rollouts=rollout_rate_hint(p))

    mt = mtrl(config)
    output_path = config["aws"]["mtrl"]["output_prefix"].rstrip("/") + f"/{config['run_id']}/"
    with ev.span("client", "upload_dataset"):
        uris = mt.upload(files, output_path + "data/")
    ev.emit("dataset", uris=uris)

    # 3. Validate with the SDK (role permissions, schema), then submit.
    ev.handled("explicit_resources", "Model package groups, MLflow app and role are passed explicitly. Left out, "
               "the SDK would get-or-create default groups and an MLflow app in your account.")
    with ev.span("client", "validate", note="MultiTurnRLTrainer(...).train(dry_run=True)"):
        prepared = mt.prepare(p, uris, output_path + "output/")
    with ev.span("client", "submit", note="CreateJob, JobCategory AgentRFT"):
        job = mt.submit(prepared)
        agent_version = mt.agent_version()
    ev.emit("job", name=job.name, arn=job.arn, status="Submitted", secondary=None, agent_version=agent_version)
    if agent_version is not None:
        ev.handled("agent_version", "The agent runtime's version is recorded when the job is submitted and checked "
                   "again when it ends. The job calls the DEFAULT endpoint, which follows the latest version, so a "
                   "redeploy during training would change the agent mid-job.")
except KeyboardInterrupt:
    ev.finished("stopped")
    sys.exit(130)
except SystemExit:
    raise
except Exception as e:
    message = str(e)
    if "not allowlisted" in message or "not authorized" in message:
        ev.log("CreateJob was refused at the account level. This is not an IAM or execution-role problem; "
               "the account needs MTRL (AgentRFT) access first. Service message: " + message, level="error")
        ev.finished("failed", message)
        sys.exit(1)
    fail(f"{type(e).__name__}: {message}")

# 4. Watch. The job runs on SageMaker AI whether or not this process does.
logs = mt.agent_logs()
status, error, stopping = "completed", None, False
seen_transitions: set[tuple] = set()
last = (None, None)
last_step = 0
last_billing = None
reported_steps: set[int] = set()
mlflow_url_sent = False
while True:
    try:
        st = job.state()
        if (st.status, st.secondary) != last:
            ev.emit("job", name=job.name, arn=job.arn, status=st.status, secondary=st.secondary,
                    failure_reason=st.failure_reason)
            last = (st.status, st.secondary)
        for t in st.transitions:
            key = (t["status"], t["start"])
            if t["end"] is not None and key not in seen_transitions:
                seen_transitions.add(key)
                ev.record_span("aws", f"job_{t['status'].lower()}", ev.at(t["start"]), t["end"] - t["start"],
                               note=t["message"])
        if st.progress and st.progress["current"] > last_step:
            for step in range(last_step + 1, st.progress["current"] + 1):
                ev.step(step, st.progress["max"])
            last_step = st.progress["current"]
        if st.billing and st.billing != last_billing:
            ev.emit("billing", source="job", tokens=st.billing)
            last_billing = st.billing
        if not mlflow_url_sent and st.status == "InProgress":
            url = job.mlflow_url()
            if url:
                ev.emit("job", name=job.name, arn=job.arn, status=st.status, secondary=st.secondary, mlflow_url=url)
                mlflow_url_sent = True

        training = st.secondary == "Training"
        if training or st.status in TERMINAL:
            for row in job.metrics():
                if row["step"] not in reported_steps:
                    reported_steps.add(row["step"])
                    ev.metrics(row["step"], **{k: v for k, v in row.items() if k != "step" and v is not None})
        records = logs.poll(training=training, step=last_step)
        if records:
            t0, t1 = records[0]["ts"], records[-1]["ts"]
            ev.record_span("client", "agent_rollouts", ev.at(t0), t1 - t0, step=last_step or None,
                           count=len(records), reward_mean=round(statistics.fmean(r["reward"] for r in records), 4),
                           note=f"{len(records)} rollouts handled by your agent, mean reward "
                                f"{statistics.fmean(r['reward'] for r in records):.3f}")
            ev.emit("rollouts", step=last_step, source="agent_logs", items=[
                {"group": None, "prompt": r["case"], "answer": r["expected"], "completion": r["decision"],
                 "reward": r["reward"], "stop_reason": f"coverage {r['coverage']:.2f}"} for r in records[-48:]])

        if st.status in TERMINAL:
            for t in st.transitions:  # close the phase that was open when the job ended
                if t["end"] is None:
                    ev.record_span("aws", f"job_{t['status'].lower()}", ev.at(t["start"]), time.time() - t["start"])
            if st.output_model_package_arn:
                ev.artifact("MODEL", st.output_model_package_arn, "LoRA adapter trained by the MTRL job", last_step)
            if st.status == "Failed":
                status, error = "failed", st.failure_reason or "the job failed without a reason"
            elif st.status == "Stopped":
                status = "stopped"
            break
        time.sleep(POLL_S)
    except KeyboardInterrupt:
        if stopping:
            ev.log("Stopped watching; the job is still stopping on SageMaker AI. Check it on the Capacity page.", "warn")
            status = "stopped"
            break
        stopping = True
        ev.log(f"Stopping job {job.name}; SageMaker AI finishes the current work first.", "warn")
        job.stop()
    except Exception as e:  # keep the outcome visible even if watching breaks
        status, error = "failed", f"Watching the job failed: {type(e).__name__}: {e}. The job may still be running."
        ev.log(error, "error")
        break

try:
    agent_now = mt.agent_version()
    if agent_version is not None and agent_now != agent_version:
        ev.log(f"The agent runtime changed from version {agent_version} to {agent_now} while job {job.name} ran: "
               "rollouts after the update may have run the new agent code.", "warn")
except Exception as e:  # the outcome of the job is known; this check only adds to it
    ev.log(f"Could not read the agent runtime's version after the job ({type(e).__name__}: {e}).", "warn")
ev.finished(status, error)
sys.exit({"completed": 0, "stopped": 130}.get(status, 1))
