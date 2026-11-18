"""GRPO-style reinforcement learning on a verifiable task, with the whole loop in this process.

    python loop.py                      # live: uses config.json next to this file

Each step:
    1. save weights and rebind the sampler (every `rebind_every` steps)  SageMaker AI
    2. make prompts                                                      your process
    3. sample a group of completions for every prompt                    SageMaker AI
    4. score them with reward.py                                         your process
    5. turn rewards into advantages with advantage.py                    your process
    6. forward_backward with the chosen RL loss, then optim_step         SageMaker AI

reward.py and advantage.py are the loop's two code slots: swap them for another variant in
the app, edit them, or replace them here. Before any GPU is used, the reward is run on
answers whose correctness is known. Before the first step and after the last one, a fixed
held-out set is answered greedily by the base model and by the trained weights.

Adapted from sample-code/02_rl_grpo_loop.py, which ran live on 2026-10-27 (4 steps of
8 prompts x 4 samples in 180-206 s, about $0.001 on gpt-oss-20b). How many steps it takes
for the reward to rise clearly has not been measured.
"""

import math
import random
import statistics
import sys

from sagemaker.train.training_session import AdamParams, SamplingParams

from advantage import advantages
from reward import reward
from srl_events import install_stop_handlers, load_config, open_log
from task import is_correct, make_problem
from ts_client import (answer_probes, backend, check_length, datum_tokens, decode, encode, finish, make_datum,
                       service_client, start_session)

config = load_config()
p = config["params"]
ev = open_log(config)
install_stop_handlers()
ev.emit("run_started", scenario=config["scenario"], backend=backend(config), model=p["base_model"],
        region=config["aws"]["region"], total_steps=p["steps"], params=p, variants=config.get("variants", {}))

rng = random.Random(p["seed"])
heldout = [make_problem(random.Random(10_000 + i)) for i in range(p["eval_size"])]
adam = AdamParams(learning_rate=p["learning_rate"], beta1=p["adam_beta1"], beta2=p["adam_beta2"],
                  eps=p["adam_eps"], weight_decay=p["weight_decay"])
max_tokens = max(512, p["max_tokens"])
if max_tokens != p["max_tokens"]:
    ev.handled("max_tokens_floor", f"max_tokens raised from {p['max_tokens']} to 512, the service minimum. "
               "Completions still end at the first newline.")
else:
    ev.handled("max_tokens_floor", "max_tokens is at least 512, the service minimum; completions end at the first newline.")
explore = SamplingParams(max_tokens=max_tokens, temperature=p["temperature"], stop=["\n"])
greedy = SamplingParams(max_tokens=max_tokens, temperature=0.0, stop=["\n"])
ev.handled("adam_explicit", "Every Adam parameter is passed explicitly. The SDK defaults (beta2 0.999, "
           "weight decay 0.01) differ from Tinker's, and nothing warns you when a ported script picks them up.")


def check_reward():
    """Score answers whose correctness is known, before any GPU is used. Raises if reward() is broken."""
    cases = {"correct": lambda a: f" {a}", "correct, with commas": lambda a: f" {a:,}",
             "off by 10": lambda a: f" {a + 10}", "wrong": lambda a: f" {a * 2 + 7}", "no number": lambda a: " I don't know."}
    problems = [make_problem(random.Random(500 + i)) for i in range(8)]
    rows = []
    for case, make in cases.items():
        scores = []
        for _, answer in problems:
            r = reward(make(answer), answer)
            if isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r):
                raise ValueError(f"reward({make(answer)!r}, {answer}) returned {r!r}; it must return a finite number")
            scores.append(float(r))
        rows.append({"case": case, "example": make(problems[0][1]).strip(), "reward_mean": statistics.mean(scores)})
    by = {r["case"]: r["reward_mean"] for r in rows}
    checks = [
        {"check": "Correct answers score above wrong ones", "ok": by["correct"] > max(by["wrong"], by["no number"]),
         "why": "Otherwise RL moves the model away from correct answers, or nowhere."},
        {"check": "Not every case scores the same", "ok": len({round(v, 9) for v in by.values()}) > 1,
         "why": "A constant reward makes every group all-equal: zero advantage, no gradient, tokens spent for nothing."},
    ]
    return rows, checks


def evaluate(sampler, tok, label):
    prompts = [encode(tok, q) for q, _ in heldout]
    with ev.span("aws", "sample", note=f"held-out set, {label}", calls=len(prompts)) as s:
        futures = [sampler.sample(x, num_samples=1, sampling_params=greedy) for x in prompts]
        outs = [f.result().sequences[0].tokens for f in futures]
        s["tokens"] = {"prefill": sum(map(len, prompts)), "sample": sum(map(len, outs))}
    return [decode(tok, o).strip() for o in outs]


try:
    with ev.span("client", "reward_check", note="reward.py on known answers, before any GPU is used"):
        rows, checks = check_reward()
except Exception as e:  # nothing to stop yet: no session exists
    ev.log(f"Reward check failed, so no session was created: {e}", level="error")
    ev.finished("failed", f"reward.py: {e}")
    sys.exit(1)
ev.emit("reward_check", rows=rows, checks=checks)
for c in checks:
    if not c["ok"]:
        ev.log(f"Reward check: {c['check'].lower()} failed. {c['why']} The run continues, so you can see the effect.",
               level="warn")

tc = start_session(ev, service_client(config), p["base_model"], p["lora_rank"], p["lora_alpha"])
status, error = "completed", None
try:
    with ev.span("client", "load_tokenizer", note="the service takes token ids, so text is tokenized here"):
        tok = tc.get_tokenizer()
    ev.handled("base_in_session", "The base model is sampled through this session (create_sampling_client with "
               "an empty ARN). ServiceClient.create_sampling_client(base_model=...) would open a second session.")
    before = evaluate(tc.create_sampling_client(""), tok, "base model")

    updates, sampler_at, sampler = 0, None, None
    for step in range(p["steps"]):
        ev.step(step, p["steps"])
        if step % p["rebind_every"] == 0:
            with ev.span("aws", "save_weights_and_rebind", step=step, calls=1,
                         note="saves a MODEL package; the sampler now serves the weights being trained"):
                sampler = tc.save_weights_and_get_sampling_client(description=f"{config['run_id']} rl-arith step {step}")
            sampler_at = updates
            ev.artifact("MODEL", sampler.model_package_arn, f"Sampler weights before step {step}", step)
        sampler = answer_probes(ev, tok, step, f"sampler weights after {sampler_at} updates", sampler=sampler)

        with ev.span("client", "make_prompts", step=step):
            problems = [make_problem(rng) for _ in range(p["prompts_per_step"])]
            prompts = [encode(tok, q) for q, _ in problems]
            for x in prompts:
                check_length(x, p["base_model"], "prompt")

        with ev.span("aws", "sample", step=step, calls=len(prompts),
                     note=f"{len(prompts)} prompts x {p['group_size']}, from weights after {sampler_at} updates") as s:
            # Submit every prompt first, then collect: the requests run concurrently.
            futures = [sampler.sample(x, num_samples=p["group_size"], sampling_params=explore) for x in prompts]
            groups = [f.result().sequences for f in futures]
            s["tokens"] = {"prefill": sum(map(len, prompts)),  # once per prompt, whatever num_samples is
                           "sample": sum(len(q.tokens) for g in groups for q in g)}

        with ev.span("client", "reward", step=step):
            texts = [[decode(tok, q.tokens) for q in g] for g in groups]
            scores = [[reward(t, answer) for t in ts] for ts, (_, answer) in zip(texts, problems)]

        with ev.span("client", "advantage", step=step) as s:
            datums, with_signal = [], 0
            truncated = sum(q.stop_reason == "length" for g in groups for q in g)
            for x, g, adv in zip(prompts, groups, advantages(scores)):
                if adv is None or not any(adv):
                    continue  # no gradient from this group
                with_signal += 1
                datums += [make_datum(x, q.tokens, q.logprobs, a) for q, a in zip(g, adv)]
            s["note"] = f"{with_signal} of {len(prompts)} groups give a gradient, {len(datums)} datums"

        ev.emit("rollouts", step=step, items=[
            {"group": i, "prompt": q.splitlines()[0], "answer": answer, "completion": t.strip(),
             "reward": r, "stop_reason": str(getattr(seq.stop_reason, "value", seq.stop_reason))}
            for i, ((q, answer), ts, rs, g) in enumerate(zip(problems, texts, scores, groups))
            for t, r, seq in zip(ts, rs, g)])

        n = len(prompts) * p["group_size"]
        values = {"reward_mean": statistics.mean(r for rs in scores for r in rs),
                  "groups_with_signal": with_signal / len(prompts), "truncated": truncated / n,
                  "updates_behind": updates - sampler_at}
        if datums:
            with ev.span("aws", "forward_backward", step=step, calls=1,
                         tokens={"train": sum(datum_tokens(d) for d in datums)}, note=f"loss_fn={p['loss_fn']}"):
                fwd_bwd = tc.forward_backward(datums, loss_fn=p["loss_fn"])
                optim = tc.optim_step(adam)  # queued behind forward_backward; both submitted before waiting
                metrics = fwd_bwd.result().metrics
            with ev.span("aws", "optim_step", step=step, calls=1):
                optim.result()
            updates += 1
            # rollout_corr/k3_kl: how far the trainer's logprobs are from the sampler's on the same tokens.
            if "rollout_corr/k3_kl" in metrics:
                values["sampler_trainer_kl"] = metrics["rollout_corr/k3_kl"]
        else:
            ev.log(f"Step {step}: every group scored all-equal, so there was nothing to train on.", level="warn")
        ev.metrics(step, **values)

    with ev.span("aws", "save_weights_and_rebind", calls=1, note="final weights"):
        final = tc.save_weights_and_get_sampling_client(description=f"{config['run_id']} rl-arith final")
    ev.artifact("MODEL", final.model_package_arn, "Final weights for sampling or deployment", p["steps"])
    if p["save_checkpoint"]:
        with ev.span("aws", "save_state", calls=1, note="weights + optimizer, resumable"):
            ckpt = tc.save_state(description=f"{config['run_id']} rl-arith step {p['steps']}").result()
        ev.artifact("CHECKPOINT", ckpt.model_package_arn, "Weights + optimizer state; resume training from here",
                    p["steps"])
    answer_probes(ev, tok, p["steps"], f"final weights, after {updates} updates", sampler=final)
    after = evaluate(final, tok, "trained weights")

    with ev.span("client", "compare"):
        items = [{"prompt": q.splitlines()[0], "expected": str(a), "before": b, "after": t,
                  "before_ok": is_correct(b, a), "after_ok": is_correct(t, a)}
                 for (q, a), b, t in zip(heldout, before, after)]
    ev.emit("compare", label="Held-out problems, greedy answers: base model vs trained weights", items=items)
    ev.emit("eval", label="Held-out problems answered correctly", unit="count", n=len(items),
            before=sum(i["before_ok"] for i in items), after=sum(i["after_ok"] for i in items))
except KeyboardInterrupt:
    status = "stopped"
except Exception as e:
    status, error = "failed", repr(e)
    ev.log(f"Run failed: {e!r}", level="error")
finally:
    code = finish(ev, tc, status, error)

sys.exit(code)
