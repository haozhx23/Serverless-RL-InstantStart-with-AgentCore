"""Train on preference pairs with a loss the service does not ship: DPO, written in this process.

    python loop.py                      # live: uses config.json next to this file

The service offers four built-in losses. Anything else that depends on the model only
through sequence log-probs can still be trained, with the surrogate trick the cookbook's
KTO recipe uses (preference/kto.py):

    1. forward on every chosen and rejected answer: per-token log-probs, no gradient   SageMaker AI
    2. your loss on those numbers, and its derivative per sequence (preference_loss.py) your process
    3. forward_backward with importance_sampling, the old log-probs from step 1 (ratio
       1) and advantage = -dloss/dlogp on every answer token: the gradient is exactly
       your loss's gradient                                                            SageMaker AI
    4. optim_step                                                                      SageMaker AI

Reference log-probs come from one forward before the first update, while the LoRA delta is
still zero (the cookbook's "precompute" reference source). The cost of a custom loss is one
extra forward per step, billed as train tokens (measured 2026-10-27).

Adapted from the cookbook's preference/train.py and the forward output format measured on
2026-10-27; this template has not run live.
"""

import statistics
import sys

from sagemaker.train.training_session import AdamParams, SamplingParams

from data import PAIRS
from preference_loss import loss_and_grad
from srl_events import install_stop_handlers, load_config, open_log
from ts_client import (answer_probes, backend, check_length, datum_tokens, decode, encode, finish, forward_logprobs,
                       make_datum, service_client, start_session)

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
ev.handled("policy_logprobs_from_trainer", "Policy log-probs come from forward on the training engine, never from "
           "the sampler: the two engines disagree slightly on the same tokens, and the cookbook's preference "
           "recipe uses the trainer's numbers for the side that drives the gradient.")


def ask(sampler, tok, label):
    prompts = [encode(tok, q) for q, _, _ in PAIRS]
    with ev.span("aws", "sample", note=label, calls=len(prompts)) as s:
        futures = [sampler.sample(x, num_samples=1, sampling_params=probe) for x in prompts]
        outs = [f.result().sequences[0].tokens for f in futures]
        s["tokens"] = {"prefill": sum(map(len, prompts)), "sample": sum(map(len, outs))}
    return [decode(tok, o).strip() for o in outs]


def sequence_logprobs(per_token, starts):
    """Sum each datum's log-probs over its answer positions (prompt positions are skipped)."""
    return [sum(lp[k:]) for lp, k in zip(per_token, starts)]


tc = start_session(ev, service_client(config), p["base_model"], p["lora_rank"], p["lora_alpha"])
status, error = "completed", None
try:
    with ev.span("client", "load_tokenizer", note="the service takes token ids, so text is tokenized here"):
        tok = tc.get_tokenizer()
    with ev.span("client", "build_batch", note=f"{len(PAIRS)} pairs, chosen and rejected"):
        seqs = []  # chosen then rejected for every pair: (prompt tokens, answer tokens)
        for q, good, bad in PAIRS:
            obs = encode(tok, q)
            for answer in (good, bad):
                act = encode(tok, answer) + [tok.eos_token_id]
                check_length(obs + act, p["base_model"], "preference example")
                seqs.append((obs, act))
        plain = [make_datum(obs, act) for obs, act in seqs]
        starts = [len(obs) - 1 for obs, _ in seqs]  # first answer position in each datum
        train_tokens = sum(datum_tokens(d) for d in plain)

    ev.handled("base_in_session", "The base model is sampled through this session (create_sampling_client with "
               "an empty ARN). ServiceClient.create_sampling_client(base_model=...) would open a second session.")
    before = ask(tc.create_sampling_client(""), tok, "base model")

    with ev.span("aws", "forward", calls=1, tokens={"train": train_tokens},
                 note="reference log-probs: the weights before any update (LoRA delta still zero)"):
        ref = sequence_logprobs(forward_logprobs(tc, plain), starts)

    for step in range(p["steps"]):
        ev.step(step, p["steps"])
        answer_probes(ev, tok, step, f"weights after {step} updates", snapshot=lambda: tc.save_weights_and_get_sampling_client(
            description=f"{config['run_id']} dpo-facts snapshot at step {step}"))

        with ev.span("aws", "forward", step=step, calls=1, tokens={"train": train_tokens},
                     note="policy log-probs of every answer, no gradient"):
            per_token = forward_logprobs(tc, plain)

        with ev.span("client", "preference_loss", step=step) as s:
            logp = sequence_logprobs(per_token, starts)
            datums, losses, margins, chosen_r, rejected_r = [], [], [], [], []
            for i in range(0, len(seqs), 2):
                c, r = i, i + 1
                margin = (logp[c] - ref[c]) - (logp[r] - ref[r])
                loss, dloss = loss_and_grad(margin, p["beta"])
                losses.append(loss)
                margins.append(margin)
                chosen_r.append(p["beta"] * (logp[c] - ref[c]))
                rejected_r.append(p["beta"] * (logp[r] - ref[r]))
                # dmargin/dlogp is +1 for the chosen answer and -1 for the rejected one; advantage = -dloss/dlogp.
                for j, sign in ((c, 1.0), (r, -1.0)):
                    obs, act = seqs[j]
                    datums.append(make_datum(obs, act, per_token[j][starts[j]:], -dloss * sign))
            s["note"] = f"mean loss {statistics.mean(losses):.4f}, {sum(m > 0 for m in margins)} of {len(margins)} pairs preferred"

        with ev.span("aws", "forward_backward", step=step, calls=1, tokens={"train": train_tokens},
                     note="importance_sampling at ratio 1: the gradient of your loss"):
            fwd_bwd = tc.forward_backward(datums, loss_fn="importance_sampling")
            optim = tc.optim_step(adam)  # queued behind forward_backward; both submitted before waiting
            fwd_bwd.result()
        with ev.span("aws", "optim_step", step=step, calls=1):
            optim.result()
        ev.metrics(step, loss=statistics.mean(losses), pairs_preferred=sum(m > 0 for m in margins) / len(margins),
                   chosen_reward=statistics.mean(chosen_r), rejected_reward=statistics.mean(rejected_r))

    with ev.span("aws", "save_weights_and_rebind", calls=1, note="saves a MODEL package and points a sampler at it"):
        sampler = tc.save_weights_and_get_sampling_client(description=f"{config['run_id']} dpo-facts weights")
    ev.artifact("MODEL", sampler.model_package_arn, "Trained weights for sampling or deployment", p["steps"])
    answer_probes(ev, tok, p["steps"], f"final weights, after {p['steps']} updates", sampler=sampler)
    after = ask(sampler, tok, "trained weights")

    with ev.span("client", "compare"):
        def says(text, answer):
            return text.lower().startswith(answer.strip().lower())
        items = [{"prompt": q.splitlines()[0], "expected": good.strip(), "before": b, "after": t,
                  "before_ok": says(b, good), "after_ok": says(t, good)}
                 for (q, good, _), b, t in zip(PAIRS, before, after)]
    ev.emit("compare", label="Same questions, base model vs trained weights (expected = the chosen answer)", items=items)
    ev.emit("eval", label="Questions answered with the chosen answer", unit="count", n=len(items),
            before=sum(i["before_ok"] for i in items), after=sum(i["after_ok"] for i in items))
except KeyboardInterrupt:
    status = "stopped"
except Exception as e:
    status, error = "failed", repr(e)
    ev.log(f"Run failed: {e!r}", level="error")
finally:
    code = finish(ev, tc, status, error)

sys.exit(code)
