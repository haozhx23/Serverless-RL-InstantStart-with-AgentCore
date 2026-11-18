"""Teach the model eight made-up facts with supervised fine-tuning (cross_entropy loss).

    python loop.py                      # live: uses config.json next to this file

What runs where:
    your process    the loop itself, tokenizing the facts in data.py, building datums, checking answers
    SageMaker AI    create the session, forward_backward, optim_step, save_state,
                    save weights + rebind the sampler, sample

The loop is a plain for-loop: two API calls per step. Questions asked from the run console
are answered between steps from a snapshot of the current weights.

Adapted from sample-code/01_sft_quickstart.py, which ran live on 2026-10-27 (176 s,
about $0.0004 on gpt-oss-20b). This version also asks the base model the same questions
first, inside the same session, so the before/after comparison costs no extra quota.
"""

import sys

from sagemaker.train.training_session import AdamParams, SamplingParams

from data import FACTS
from srl_events import install_stop_handlers, load_config, open_log
from ts_client import (answer_probes, backend, check_length, datum_tokens, decode, encode, finish, make_datum,
                       service_client, start_session)

config = load_config()
p = config["params"]
ev = open_log(config)
install_stop_handlers()
ev.emit("run_started", scenario=config["scenario"], backend=backend(config), model=p["base_model"],
        region=config["aws"]["region"], total_steps=p["steps"], params=p, variants=config.get("variants", {}))

adam = AdamParams(learning_rate=p["learning_rate"], beta1=p["adam_beta1"], beta2=p["adam_beta2"],
                  eps=p["adam_eps"], weight_decay=p["weight_decay"])
probe = SamplingParams(max_tokens=512, temperature=0.0, stop=["\n"])
ev.handled("adam_explicit", "Every Adam parameter is passed explicitly. The SDK defaults (beta2 0.999, "
           "weight decay 0.01) differ from Tinker's, and nothing warns you when a ported script picks them up.")
ev.handled("max_tokens_floor", "Probes use max_tokens=512, the service minimum, and stop at the first newline "
           "to keep answers short. Smaller values are rejected by request validation.")


def ask(sampler, tok, label):
    """Ask every fact question once; prompts bill as prefill, answers as sample tokens."""
    prompts = [encode(tok, q) for q, _ in FACTS]
    with ev.span("aws", "sample", note=label, calls=len(prompts)) as s:
        futures = [sampler.sample(x, num_samples=1, sampling_params=probe) for x in prompts]
        outs = [f.result().sequences[0].tokens for f in futures]
        s["tokens"] = {"prefill": sum(map(len, prompts)), "sample": sum(map(len, outs))}
    return [decode(tok, o).strip() for o in outs]


def correct(answer, expected):
    return answer.lower().startswith(expected.strip().lower())


tc = start_session(ev, service_client(config), p["base_model"], p["lora_rank"], p["lora_alpha"])
status, error = "completed", None
try:
    with ev.span("client", "load_tokenizer", note="the service takes token ids, so text is tokenized here"):
        tok = tc.get_tokenizer()
    with ev.span("client", "build_batch"):
        batch = []
        for q, a in FACTS:
            obs, act = encode(tok, q), encode(tok, a) + [tok.eos_token_id]
            check_length(obs + act, p["base_model"], "training example")
            batch.append(make_datum(obs, act))
        train_tokens = sum(datum_tokens(d) for d in batch)

    ev.handled("base_in_session", "The base model is sampled through this session (create_sampling_client with "
               "an empty ARN). ServiceClient.create_sampling_client(base_model=...) would open a second session.")
    before = ask(tc.create_sampling_client(""), tok, "base model")

    for step in range(p["steps"]):
        ev.step(step, p["steps"])
        answer_probes(ev, tok, step, f"weights after {step} updates", snapshot=lambda: tc.save_weights_and_get_sampling_client(
            description=f"{config['run_id']} sft-facts snapshot at step {step}"))
        with ev.span("aws", "forward_backward", step=step, calls=1, tokens={"train": train_tokens},
                     note=f"{len(batch)} datums, loss_fn=cross_entropy"):
            fwd_bwd = tc.forward_backward(batch, loss_fn="cross_entropy")
            optim = tc.optim_step(adam)  # queued behind forward_backward; both submitted before waiting
            loss = fwd_bwd.result().metrics["actor/loss"]
        with ev.span("aws", "optim_step", step=step, calls=1):
            optim.result()
        ev.metrics(step, loss=loss)

    if p["save_checkpoint"]:
        with ev.span("aws", "save_state", calls=1, note="weights + optimizer, resumable"):
            ckpt = tc.save_state(description=f"{config['run_id']} sft-facts step {p['steps']}").result()
        ev.artifact("CHECKPOINT", ckpt.model_package_arn, "Weights + optimizer state; resume training from here",
                    p["steps"])

    with ev.span("aws", "save_weights_and_rebind", calls=1, note="saves a MODEL package and points a sampler at it"):
        sampler = tc.save_weights_and_get_sampling_client(description=f"{config['run_id']} sft-facts weights")
    ev.artifact("MODEL", sampler.model_package_arn, "Trained weights for sampling or deployment", p["steps"])
    answer_probes(ev, tok, p["steps"], f"final weights, after {p['steps']} updates", sampler=sampler)
    after = ask(sampler, tok, "trained weights")

    with ev.span("client", "compare"):
        items = [{"prompt": q.splitlines()[0], "expected": a.strip(), "before": b, "after": t,
                  "before_ok": correct(b, a), "after_ok": correct(t, a)}
                 for (q, a), b, t in zip(FACTS, before, after)]
    ev.emit("compare", label="Same questions, base model vs trained weights", items=items)
    ev.emit("eval", label="Questions answered with the trained fact", unit="count", n=len(items),
            before=sum(i["before_ok"] for i in items), after=sum(i["after_ok"] for i in items))
except KeyboardInterrupt:
    status = "stopped"
except Exception as e:
    status, error = "failed", repr(e)
    ev.log(f"Run failed: {e!r}", level="error")
finally:
    code = finish(ev, tc, status, error)

sys.exit(code)
