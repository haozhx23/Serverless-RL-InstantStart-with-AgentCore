"""The official AWS sample "GRPO on GSM8K", with its training loop in this process.

    python loop.py                      # live: uses config.json next to this file

Source: aws/amazon-sagemaker-examples, commit 3429cb1a55, notebook
sm-training_sessions_1_train_grpo_gsm8k.ipynb (Apache-2.0); the sample the 2026-10-22 launch blog
walks through. official.py holds the notebook's task code verbatim (prompt template, answer parser,
reward, GRPO advantages, datum layout). This file runs the notebook's cells 19-35 in order:

    evaluate() on 30 held-out problems x 4 samples                       baseline
    10 steps of: draw 4 questions, sample 8 responses each, score them,
                 skip all-equal groups, GRPO advantages, forward_backward
                 (ppo) + optim_step, save_weights_and_get_sampling_client,
                 save_state every 5 steps
    evaluate() again                                                    after training
    save_weights_for_sampler(merge_weights=True), save_state, stop

Differences from the notebook, none of them in the algorithm:
  - The session comes from this app's ServiceClient: its execution role, model package group, S3
    prefix and "instantstart" session names, instead of the notebook's SageMakerTrainingSessionRole
    and gsm8k-grpo-training-sessions group.
  - The 40 training and 30 evaluation problems are read from gsm8k/*.jsonl, prepared exactly as the
    notebook's cell 9 prepares them (the first 40 of GSM8K's train split, the first 30 of its test
    split), so nothing is downloaded at run time.
  - Every API call is wrapped in ev.span(...) so the app can draw it on the loop map, count its tokens
    and price it, and prompts are length-checked before they are sent.
  - Waiting for a step's samples and scoring them are two blocks here; the notebook scores each
    question as soon as its responses arrive. The functions and the results are the same.
  - The notebook's constants are parameters, with the notebook's values as defaults.

Published result (AWS blog, 2026-10-22): 10 steps on 40 problems took held-out accuracy from 56% to 75%.
"""

import json
import random
import sys

from sagemaker.train.training_session import AdamParams, SamplingParams

from official import build_prompt, build_training_datum, compute_advantages, compute_reward, extract_answer
from srl_events import install_stop_handlers, load_config, open_log
from ts_client import backend, check_length, finish, service_client, start_session

config = load_config()
p = config["params"]
ev = open_log(config)
install_stop_handlers()
ev.emit("run_started", scenario=config["scenario"], backend=backend(config), model=p["base_model"],
        region=config["aws"]["region"], total_steps=p["steps"], params=p, variants=config.get("variants", {}))


def load_jsonl(path):  # notebook cell 10
    problems = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            problems.append((row["question"], row["ground_truth"]))
    return problems


train_problems = load_jsonl("gsm8k/train.jsonl")
eval_problems = load_jsonl("gsm8k/eval.jsonl")
SAMPLING_PARAMS = SamplingParams(max_tokens=p["max_tokens"], temperature=p["temperature"], top_p=p["top_p"])
ADAM_PARAMS = AdamParams(learning_rate=p["learning_rate"])  # as in the notebook: other Adam values are SDK defaults
ev.handled("official_session", "The session uses this app's execution role and model package group instead of the "
           "sample's SageMakerTrainingSessionRole and gsm8k-grpo-training-sessions group. The training code is the "
           "notebook's.")
ev.handled("adam_sdk_defaults", "As in the official sample, only the learning rate is set; the other Adam parameters "
           "are the SDK defaults (beta2 0.999, weight decay 0.01).")


def evaluate(sampler, tokenizer, label):
    """Notebook cell 19: EVAL_SAMPLES_PER_PROBLEM samples per held-out problem, mean accuracy and its error bar."""
    prompts = [tokenizer.encode(build_prompt(question), add_special_tokens=False) for question, _ in eval_problems]
    for x in prompts:
        check_length(x, p["base_model"], "prompt")
    with ev.span("aws", "sample", calls=len(prompts),
                 note=f"{len(prompts)} held-out problems x {p['eval_samples']}, {label}") as s:
        futures = [sampler.sample(x, num_samples=p["eval_samples"], sampling_params=SAMPLING_PARAMS) for x in prompts]
        results = [f.result().sequences for f in futures]
        s["tokens"] = {"prefill": sum(map(len, prompts)), "sample": sum(len(q.tokens) for seqs in results for q in seqs)}
    rows, scores = [], []
    for (question, ground_truth), seqs in zip(eval_problems, results):
        texts = [tokenizer.decode(sample.tokens) for sample in seqs]
        rewards = [compute_reward(t, ground_truth) for t in texts]
        scores += rewards
        rows.append({"question": question, "ground_truth": ground_truth, "correct": int(sum(rewards)),
                     "answers": [extract_answer(t) for t in texts]})
    n = len(scores)
    accuracy = sum(scores) / n
    std_err = (accuracy * (1 - accuracy) / n) ** 0.5  # binomial standard error
    ev.log(f"{label} accuracy: {accuracy:.0%} +/- {std_err:.0%} ({sum(scores):.0f}/{n} samples)")
    return rows, int(sum(scores)), n


tc = start_session(ev, service_client(config), p["base_model"], p["lora_rank"], p["lora_alpha"])
status, error = "completed", None
try:
    with ev.span("client", "load_tokenizer", note="the service takes token ids, so text is tokenized here"):
        tokenizer = tc.get_tokenizer()
    sampling_client = tc.create_sampling_client()
    before, before_correct, n_eval = evaluate(sampling_client, tokenizer, "Baseline (before training)")

    train_pool = list(train_problems)
    random.shuffle(train_pool)
    updates = 0
    for step in range(p["steps"]):
        ev.step(step, p["steps"])
        with ev.span("client", "make_prompts", step=step, note=f"{p['prompts_per_step']} questions from the shuffled pool"):
            if len(train_pool) < p["prompts_per_step"]:
                train_pool = list(train_problems)
                random.shuffle(train_pool)
            batch = [train_pool.pop() for _ in range(p["prompts_per_step"])]
            prompts = [tokenizer.encode(build_prompt(question), add_special_tokens=False) for question, _ in batch]
            for x in prompts:
                check_length(x, p["base_model"], "prompt")

        with ev.span("aws", "sample", step=step, calls=len(prompts),
                     note=f"{len(prompts)} questions x {p['group_size']} responses") as s:
            # Queue every question's request up front, as the notebook does: they generate concurrently.
            futures = [sampling_client.sample(x, num_samples=p["group_size"], sampling_params=SAMPLING_PARAMS)
                       for x in prompts]
            groups = [f.result().sequences for f in futures]
            s["tokens"] = {"prefill": sum(map(len, prompts)), "sample": sum(len(q.tokens) for g in groups for q in g)}

        with ev.span("client", "reward", step=step):  # notebook cell 21, score_group()
            scored = []
            for (question, ground_truth), x, seqs in zip(batch, prompts, groups):
                scored.append([{"prompt_tokens": x, "response_tokens": sample.tokens, "response_logprobs": sample.logprobs,
                                "reward": compute_reward(tokenizer.decode(sample.tokens), ground_truth)} for sample in seqs])

        with ev.span("client", "advantage", step=step) as s:  # notebook cell 29, inside the step
            datums, rewards_this_step, n_degenerate = [], [], 0
            for group in scored:
                rewards_this_step.extend(r["reward"] for r in group)
                if len({r["reward"] for r in group}) == 1:
                    n_degenerate += 1
                    continue
                group = compute_advantages(group)
                datums.extend(build_training_datum(r) for r in group)
            s["note"] = f"{len(scored) - n_degenerate} of {len(scored)} groups give a gradient, {len(datums)} datums"

        ev.emit("rollouts", step=step, items=[
            {"group": i, "prompt": question[:160], "answer": ground_truth,
             "completion": "…" + tokenizer.decode(r["response_tokens"])[-300:], "reward": r["reward"],
             "stop_reason": str(getattr(seq.stop_reason, "value", seq.stop_reason))}
            for i, ((question, ground_truth), group, seqs) in enumerate(zip(batch, scored, groups))
            for r, seq in zip(group, seqs)])

        accuracy = sum(rewards_this_step) / len(rewards_this_step)
        values = {"batch_accuracy": accuracy, "groups_with_signal": (len(scored) - n_degenerate) / len(scored)}
        if not datums:  # every group all right or all wrong: the notebook skips the update and the rebind
            ev.log(f"Step {step}: every group scored all-equal, so there was no update.", level="warn")
            ev.metrics(step, **values)
            continue

        train_tokens = sum(len(d["modelInput"]["chunks"][0]["tokens"]) for d in datums)
        with ev.span("aws", "forward_backward", step=step, calls=1, tokens={"train": train_tokens},
                     note=f"loss_fn={p['loss_fn']}"):  # notebook cell 27, train_step()
            fwd_bwd_op = tc.forward_backward(datums, loss_fn=p["loss_fn"])
            optim_op = tc.optim_step(ADAM_PARAMS)
            metrics = fwd_bwd_op.result().metrics or {}
        with ev.span("aws", "optim_step", step=step, calls=1):
            optim_op.result()
        updates += 1
        for key, name in (("actor/pg_loss", "pg_loss"), ("actor/ppo_kl", "ppo_kl"), ("actor/pg_clipfrac", "pg_clipfrac")):
            if key in metrics:
                values[name] = metrics[key]
        ev.metrics(step, **values)

        with ev.span("aws", "save_weights_and_rebind", step=step, calls=1,
                     note="saves a MODEL package; the next step samples from the policy just trained"):
            sampling_client = tc.save_weights_and_get_sampling_client(description=f"policy after step {step}")
        ev.artifact("MODEL", sampling_client.model_package_arn, f"Policy after step {step}", step)

        if p["save_state_every"] and (step + 1) % p["save_state_every"] == 0 and step + 1 < p["steps"]:
            with ev.span("aws", "save_state", step=step, calls=1, note="weights + optimizer, resumable"):
                ckpt = tc.save_state(description=f"step {step} checkpoint").result()
            ev.artifact("CHECKPOINT", ckpt.model_package_arn, f"Checkpoint at step {step}", step)

    after, after_correct, _ = evaluate(sampling_client, tokenizer, "After training")  # notebook cell 31

    with ev.span("client", "compare"):
        def shown(row):
            answers = ", ".join(a if a is not None else "—" for a in row["answers"])
            return f"{row['correct']}/{len(row['answers'])} · {answers}"
        items = [{"prompt": b["question"][:160], "expected": b["ground_truth"], "before": shown(b), "after": shown(a),
                  "before_ok": b["correct"] * 2 >= len(b["answers"]), "after_ok": a["correct"] * 2 >= len(a["answers"])}
                 for b, a in zip(before, after)]
    ev.emit("compare", label="Held-out GSM8K problems: correct samples of 4, base model vs trained weights", items=items)
    ev.emit("eval", label="Held-out samples answered correctly (30 problems x 4)", unit="samples", n=n_eval,
            before=before_correct, after=after_correct)

    with ev.span("aws", "save_weights_merged", calls=1, note="merged LoRA, for deployment"):  # notebook cell 33
        deployable = tc.save_weights_for_sampler(description="merged model, end of run", merge_weights=True).result()
    ev.artifact("MODEL", deployable.model_package_arn, "Merged model for deployment", p["steps"])
    with ev.span("aws", "save_state", calls=1, note="weights + optimizer, resumable"):
        state_checkpoint = tc.save_state(description="end of run, resumable").result()
    ev.artifact("CHECKPOINT", state_checkpoint.model_package_arn, "Weights + optimizer state; resume training from here",
                p["steps"])
except KeyboardInterrupt:
    status = "stopped"
except Exception as e:
    status, error = "failed", repr(e)
    ev.log(f"Run failed: {e!r}", level="error")
finally:
    code = finish(ev, tc, status, error)  # notebook cell 35: stop the session

sys.exit(code)
