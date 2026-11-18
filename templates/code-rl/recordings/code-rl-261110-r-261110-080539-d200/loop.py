"""Code RL: the model writes Python functions, an environment sandbox runs their tests, this loop trains on the result.

    python loop.py                      # live: uses config.json next to this file

Each step:
    1. save weights and rebind the sampler                     SageMaker AI
    2. pick training problems                                  your process
    3. sample a group of answers for every problem             SageMaker AI
    4. run the code of every answer against its tests          your code, in AgentCore Runtime
       in a sandbox
    5. score the results with reward.py                        your process
    6. group-normalized advantages, then forward_backward
       and optim_step                                          SageMaker AI

Model-generated code is never run in this process: it holds this machine's AWS credentials and
the training session. Step 4 sends the code of each answer (its last ```python block, task.extract)
to an AgentCore Runtime session (sandbox_client.py), a microVM of its own under the sandbox's
execution role, which can do next to nothing. The `isolation` parameter says how many answers
share a session: one per answer (the default, as in MTRL), or one per group, wiped between
answers, which opens fewer sessions.

reward.py is the loop's code slot. Before the training session exists, the reward is run on
bodies whose behaviour is known (correct, wrong, broken, one that exits before its tests finish,
one that never returns), in the sandbox. That is also the sandbox's smoke test: if it fails,
no GPU has been used. Before the first step and after the last one, eight held-out functions are
written greedily by the base model and by the trained weights; a function counts as working when
every test passes (task.is_correct, independent of reward.py).

Built for this app on the pattern of rl-arith/loop.py; the sandbox side follows the 2026-11-10
validation (docs/261110-agentcore-runtime-sandbox.md). It has not run live yet.
"""

import math
import random
import statistics
import sys

from sagemaker.train.training_session import AdamParams, SamplingParams

import task
from reward import reward
from sandbox_client import SandboxError, sandboxes
from srl_events import install_stop_handlers, load_config, open_log
from ts_client import (answer_probes, backend, check_length, datum_tokens, decode, encode, finish, make_datum,
                       service_client, start_session)

config = load_config()
p = config["params"]
ev = open_log(config)
install_stop_handlers()
ev.emit("run_started", scenario=config["scenario"], backend=backend(config), model=p["base_model"],
        region=config["aws"]["region"], total_steps=p["steps"], params=p, variants=config.get("variants", {}))

rng = random.Random(p["seed"])
adam = AdamParams(learning_rate=p["learning_rate"], beta1=p["adam_beta1"], beta2=p["adam_beta2"],
                  eps=p["adam_eps"], weight_decay=p["weight_decay"])
max_tokens = max(512, p["max_tokens"])
if max_tokens != p["max_tokens"]:
    ev.handled("max_tokens_floor", f"max_tokens raised from {p['max_tokens']} to 512, the service minimum.")
ev.handled("harmony_prompt", "Prompts use gpt-oss's harmony chat format with low reasoning effort, and the solution is "
           "the answer's last ```python block. Asked to continue a plain Python file, gpt-oss-20b wrote prose "
           "reasoning instead of code in every one of 192 completions (live, 2026-11-10).")
explore = SamplingParams(max_tokens=max_tokens, temperature=p["temperature"])
greedy = SamplingParams(max_tokens=max_tokens, temperature=0.0)
ev.handled("adam_explicit", "Every Adam parameter is passed explicitly. The SDK defaults (beta2 0.999, "
           "weight decay 0.01) differ from Tinker's, and nothing warns you when a ported script picks them up.")

# Bodies whose behaviour is known, for the reward check.
CHECK_CASES = {
    "correct": lambda prob: prob["reference"],
    "wrong": lambda prob: prob["wrong"][0],
    "syntax error": lambda prob: "    return (\n",
    "exits before the tests finish": lambda prob: "    import os\n    os._exit(0)\n",
    "never returns": lambda prob: "    while True:\n        pass\n",
}


def grade_groups(groups, step, label, whole=False):
    """Run every (problem, code, seed) of every group in sandboxes; results in the same shape.

    `whole`: the code is a whole file (an answer's code block); otherwise a body for the problem's head.

    isolation "rollout": one session per body, stopped as soon as it is scored. "group": one session
    per group, its work directory wiped (reset) before each body, stopped after the group.
    """
    per_body = p["isolation"] == "rollout"
    jobs = [[item] for g in groups for item in g] if per_body else groups
    boxes = pool.new(len(jobs), label)

    def run(i):
        return [task.grade(boxes[i], prob, code if whole else task.solution(prob, code), seed, p["test_timeout_s"])
                for prob, code, seed in jobs[i]]
    try:
        done = pool.map(run, list(range(len(jobs))), step=step, what=label)
    except BaseException:
        pool.stop(boxes, step, check=False)  # the original error is the one to report
        raise
    pool.stop(boxes, step)
    flat = iter(r for rs in done for r in rs)
    return [[next(flat) for _ in g] for g in groups]


def sandbox_note(s, n_bodies):
    b = pool.last_batch
    cold = b["cold"]
    s["sandbox_calls"] = b["calls"]
    s["note"] = (f"{n_bodies} programs in {len(cold)} {'sessions' if len(cold) != 1 else 'session'}"
                 + (f", cold start p50 {statistics.median(cold):.2f} s" if cold else "")
                 + (f", {b['retries']} throttled and retried" if b["retries"] else ""))


def check_reward():
    """Score bodies whose behaviour is known, in the sandbox, before any GPU is used. Raises if reward() is broken."""
    probs = task.TRAIN[:3]
    groups = [[(prob, make(prob), 500 + i)] for case, make in CHECK_CASES.items() for i, prob in enumerate(probs)]
    results = grade_groups(groups, None, "check")
    rows, it = [], iter(results)
    by = {}
    for case in CHECK_CASES:
        scores = []
        for prob in probs:
            res = next(it)[0]
            r = reward(res)
            if isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r):
                raise ValueError(f"reward({res!r}) returned {r!r}; it must return a finite number")
            scores.append(float(r))
        by[case] = statistics.mean(scores)
        rows.append({"case": case, "example": probs[0]["id"], "reward_mean": by[case]})
    hung = results[len(probs) * 4:]
    checks = [
        {"check": "Correct bodies score above wrong and broken ones",
         "ok": by["correct"] > max(by["wrong"], by["syntax error"]),
         "why": "Otherwise RL moves the model away from working code, or nowhere."},
        {"check": "A body that exits before its tests finish does not score",
         "ok": by["exits before the tests finish"] == 0,
         "why": "os._exit(0) ends the test program with exit code 0 before any result: grading by exit code "
                "(as with pytest's) pays the model for skipping the tests."},
        {"check": "A body that never returns is stopped at the timeout and scores 0",
         "ok": by["never returns"] == 0 and all(r[0]["timed_out"] for r in hung),
         "why": "The sandbox kills the whole process group at the timeout; a hung body must not earn reward."},
        {"check": "Not every case scores the same", "ok": len({round(v, 9) for v in by.values()}) > 1,
         "why": "A constant reward makes every group all-equal: zero advantage, no gradient, tokens spent for nothing."},
    ]
    return rows, checks


def advantages(groups):
    """Group-normalized, as in GRPO: (reward - group mean) / group std; an all-equal group is left out."""
    out = []
    for rewards in groups:
        mean, std = statistics.mean(rewards), statistics.pstdev(rewards)
        out.append(None if std == 0 else [(r - mean) / std for r in rewards])
    return out


def evaluate(sampler, tok, label):
    """Greedy answers for the held-out functions, the code of each, and how it did in the sandbox."""
    prompts = [encode(tok, task.prompt(prob)) for prob in task.HELDOUT]
    with ev.span("aws", "sample", note=f"held-out functions, {label}", calls=len(prompts)) as s:
        futures = [sampler.sample(x, num_samples=1, sampling_params=greedy) for x in prompts]
        outs = [f.result().sequences[0].tokens for f in futures]
        s["tokens"] = {"prefill": sum(map(len, prompts)), "sample": sum(map(len, outs))}
    codes = [task.extract(decode(tok, o)) or "" for o in outs]
    with ev.span("client", "sandbox_run", where=pool.where) as s:
        results = grade_groups([[(prob, c, 900 + i)] for i, (prob, c) in enumerate(zip(task.HELDOUT, codes))],
                               None, "eval", whole=True)
        sandbox_note(s, len(codes))
    return codes, [r[0] for r in results]


try:
    pool = sandboxes(ev, config)
except SandboxError as e:  # nothing to stop: no session exists
    ev.log(f"No sandbox for this run: {e}", level="error")
    ev.finished("failed", str(e))
    sys.exit(1)

tc = None
status, error = "completed", None
try:
    with ev.span("client", "reward_check", where=pool.where) as s:
        rows, checks = check_reward()
        sandbox_note(s, len(rows) * 3)
        s["note"] = "reward.py on known solutions, in the sandbox, before any GPU is used: " + s["note"]
except Exception as e:  # no session yet; the sandboxes are stopped below
    kind = "The sandbox failed" if isinstance(e, SandboxError) else "Reward check failed"
    ev.log(f"{kind}, so no session was created: {e}", level="error")
    status, error = "failed", f"{'sandbox' if isinstance(e, SandboxError) else 'reward.py'}: {e}"
except KeyboardInterrupt:
    status = "stopped"

if status == "completed":
    ev.emit("reward_check", rows=rows, checks=checks)
    for c in checks:
        if not c["ok"]:
            ev.log(f"Reward check: {c['check'].lower()} failed. {c['why']} The run continues, so you can see the effect.",
                   level="warn")
    # If the session never comes up, start_session ends the run; the sandbox summary goes first.
    tc = start_session(ev, service_client(config), p["base_model"], p["lora_rank"], p["lora_alpha"],
                       before_exit=pool.close)

if tc is None:
    pool.close()
    ev.finished(status, error)
    sys.exit({"completed": 0, "stopped": 130}.get(status, 1))

try:
    with ev.span("client", "load_tokenizer", note="the service takes token ids, so text is tokenized here"):
        tok = tc.get_tokenizer()
    ev.handled("base_in_session", "The base model is sampled through this session (create_sampling_client with "
               "an empty ARN). ServiceClient.create_sampling_client(base_model=...) would open a second session.")
    before, before_results = evaluate(tc.create_sampling_client(""), tok, "base model")

    updates, sampler = 0, None
    for step in range(p["steps"]):
        ev.step(step, p["steps"])
        with ev.span("aws", "save_weights_and_rebind", step=step, calls=1,
                     note="saves a MODEL package; the sampler now serves the weights being trained"):
            sampler = tc.save_weights_and_get_sampling_client(description=f"{config['run_id']} code-rl step {step}")
        ev.artifact("MODEL", sampler.model_package_arn, f"Sampler weights before step {step}", step)
        sampler = answer_probes(ev, tok, step, f"sampler weights after {updates} updates", sampler=sampler)

        with ev.span("client", "make_prompts", step=step):
            n = p["prompts_per_step"]
            problems = rng.sample(task.TRAIN, n) if n <= len(task.TRAIN) else [rng.choice(task.TRAIN) for _ in range(n)]
            prompts = [encode(tok, task.prompt(prob)) for prob in problems]
            for x in prompts:
                check_length(x, p["base_model"], "prompt")

        with ev.span("aws", "sample", step=step, calls=len(prompts),
                     note=f"{len(prompts)} functions x {p['group_size']}, from weights after {updates} updates") as s:
            futures = [sampler.sample(x, num_samples=p["group_size"], sampling_params=explore) for x in prompts]
            groups = [f.result().sequences for f in futures]
            s["tokens"] = {"prefill": sum(map(len, prompts)),  # once per prompt, whatever num_samples is
                           "sample": sum(len(q.tokens) for g in groups for q in g)}

        texts = [[decode(tok, q.tokens) for q in g] for g in groups]
        codes = [[task.extract(t) for t in ts] for ts in texts]
        with ev.span("client", "sandbox_run", step=step, where=pool.where) as s:
            # An answer without a code block still runs, as an empty file: its tests fail, a task result.
            work = [[(prob, c or "", rng.randrange(1 << 30)) for c in cs] for prob, cs in zip(problems, codes)]
            results = grade_groups(work, step, f"s{step:03d}", whole=True)
            sandbox_note(s, sum(map(len, texts)))

        with ev.span("client", "reward", step=step):
            scores = [[reward(r) for r in rs] for rs in results]

        with ev.span("client", "advantage", step=step) as s:
            datums, with_signal = [], 0
            for x, g, adv in zip(prompts, groups, advantages(scores)):
                if adv is None or not any(adv):
                    continue  # no gradient from this group
                with_signal += 1
                datums += [make_datum(x, q.tokens, q.logprobs, a) for q, a in zip(g, adv)]
            s["note"] = f"{with_signal} of {len(prompts)} groups give a gradient, {len(datums)} datums"

        ev.emit("rollouts", step=step, items=[
            {"group": i, "prompt": prob["id"], "answer": f"{res['passed']}/{res['total']} tests"
             + (", timed out" if res["timed_out"] else "" if res["reported"] else ", no result"),
             "completion": c.rstrip() if c is not None else "(no code block) …" + t[-300:],
             "reward": r, "stop_reason": str(getattr(seq.stop_reason, "value", seq.stop_reason))}
            for i, (prob, ts, cs, rs, sc, g) in enumerate(zip(problems, texts, codes, results, scores, groups))
            for t, c, res, r, seq in zip(ts, cs, rs, sc, g)])

        flat = [r for rs in results for r in rs]
        values = {"reward_mean": statistics.mean(x for sc in scores for x in sc),
                  "all_pass": sum(task.is_correct(r) for r in flat) / len(flat),
                  "no_result": sum(not r["reported"] for r in flat) / len(flat),
                  "no_code": sum(c is None for cs in codes for c in cs) / len(flat),
                  "groups_with_signal": with_signal / len(prompts),
                  "truncated": sum(q.stop_reason == "length" for g in groups for q in g) / len(flat)}
        if pool.last_batch["cold"]:
            values["sandbox_cold_start_s"] = statistics.median(pool.last_batch["cold"])
        if datums:
            with ev.span("aws", "forward_backward", step=step, calls=1,
                         tokens={"train": sum(datum_tokens(d) for d in datums)}, note="loss_fn=importance_sampling"):
                fwd_bwd = tc.forward_backward(datums, loss_fn="importance_sampling")
                optim = tc.optim_step(adam)  # queued behind forward_backward; both submitted before waiting
                metrics = fwd_bwd.result().metrics
            with ev.span("aws", "optim_step", step=step, calls=1):
                optim.result()
            updates += 1
            if "rollout_corr/k3_kl" in metrics:
                values["sampler_trainer_kl"] = metrics["rollout_corr/k3_kl"]
        else:
            ev.log(f"Step {step}: every group scored all-equal, so there was nothing to train on.", level="warn")
        ev.metrics(step, **values)

    with ev.span("aws", "save_weights_and_rebind", calls=1, note="final weights"):
        final = tc.save_weights_and_get_sampling_client(description=f"{config['run_id']} code-rl final")
    ev.artifact("MODEL", final.model_package_arn, "Final weights for sampling or deployment", p["steps"])
    if p["save_checkpoint"]:
        with ev.span("aws", "save_state", calls=1, note="weights + optimizer, resumable"):
            ckpt = tc.save_state(description=f"{config['run_id']} code-rl step {p['steps']}").result()
        ev.artifact("CHECKPOINT", ckpt.model_package_arn, "Weights + optimizer state; resume training from here",
                    p["steps"])
    answer_probes(ev, tok, p["steps"], f"final weights, after {updates} updates", sampler=final)
    after, after_results = evaluate(final, tok, "trained weights")

    with ev.span("client", "compare"):
        items = [{"prompt": prob["id"], "expected": f"all {len(prob['cases'])} tests pass",
                  "before": b.rstrip() or "(no code block)", "after": a.rstrip() or "(no code block)",
                  "before_ok": task.is_correct(rb), "after_ok": task.is_correct(ra)}
                 for prob, b, a, rb, ra in zip(task.HELDOUT, before, after, before_results, after_results)]
    ev.emit("compare", label="Held-out functions, greedy answers: base model vs trained weights", items=items)
    ev.emit("eval", label="Held-out functions whose tests all pass", unit="count", n=len(items),
            before=sum(i["before_ok"] for i in items), after=sum(i["after_ok"] for i in items))
except KeyboardInterrupt:
    status = "stopped"
except SandboxError as e:
    status, error = "failed", f"sandbox: {e}"
    ev.log(f"The sandbox failed; a sandbox failure is never scored as a reward: {e}", level="error")
except Exception as e:
    status, error = "failed", repr(e)
    ev.log(f"Run failed: {e!r}", level="error")
finally:
    if pool.close() and status == "completed":
        status, error = "failed", "some sandbox sessions could not be stopped; see the log"
    code = finish(ev, tc, status, error)

sys.exit(code)
