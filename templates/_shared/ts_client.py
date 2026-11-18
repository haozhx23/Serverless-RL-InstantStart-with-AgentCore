"""Client construction and datum helpers shared by the run templates.

``service_client(config)`` returns the real ``ServiceClient`` for a live run. Set
``SRL_BACKEND`` to override the backend config.json names.

Datum layout follows the AWS cookbook and was verified live on 2026-10-27
(sample-code/common.py in the research repo).
"""

from __future__ import annotations

import json
import os
import sys

from sagemaker.train.training_session import Datum, ModelInput, SamplingParams, TensorData, TensorDtype

# Largest input we have seen work per model. gpt-oss-20b was measured (131,071 tokens run,
# 131,072 put the whole session into Failed instead of returning a validation error).
# The others are not measured, so stay well below their published context.
MEASURED_INPUT_LIMIT = {"openai/gpt-oss-20b": 131_071}
CONSERVATIVE_INPUT_LIMIT = 32_768
INBOX = "inbox"  # questions from the run console wait here until the loop answers them
PROBE_SAMPLING = SamplingParams(max_tokens=512, temperature=0.0, stop=["\n"])  # 512 is the service minimum


BACKENDS = ("live",)


def backend(config: dict) -> str:
    value = os.environ.get("SRL_BACKEND") or config.get("backend", "live")
    if value not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}, got {value!r}")
    return value


def service_client(config: dict):
    aws = config["aws"]
    kwargs = dict(
        region=aws["region"],
        role_arn=aws["role_arn"],
        model_package_group_arn=aws["model_package_group_arn"],
        s3_output_path=aws["s3_output_path"],
        session_name_prefix=aws.get("session_name_prefix", "instantstart"),
    )
    from sagemaker.train.training_session import ServiceClient
    return ServiceClient(**kwargs)


def start_session(ev, svc, base_model: str, rank: int, alpha: int, before_exit=None):
    """Open a LoRA training session and record how long it took to reach Running.

    Exits the process (code 1, or 130 when stopped) if the session never comes up; there
    is nothing to stop in that case, but a session interrupted while starting may still
    come up server-side, so the log says where to find it. ``before_exit`` runs first on
    that path (a loop with sandboxes closes them there, before the final event).
    """
    ev.session("creating")
    try:
        with ev.span("aws", "create_session") as s:
            tc = svc.create_lora_training_client(base_model=base_model, rank=rank, alpha=alpha)
    except BaseException as e:
        interrupted = isinstance(e, KeyboardInterrupt)
        ev.session("stopped" if interrupted else "failed")
        if interrupted:
            ev.log("Stopped while the session was starting. If it came up anyway it is listed on the "
                   "Sessions page; stop it there so it does not hold quota.", level="warn")
        else:
            ev.log(f"Could not create the training session: {e!r}", level="error")
        if before_exit is not None:
            before_exit()
        ev.finished("stopped" if interrupted else "failed", error=None if interrupted else repr(e))
        sys.exit(130 if interrupted else 1)
    ev.session("running", tc.training_session_name, ready_sec=s["sec"])
    ev.handled("stop_in_finally", "The session is stopped in a finally block, on success, error and Stop. "
               "An idle session bills no tokens but keeps holding one of your concurrency slots.")
    return tc


def finish(ev, tc, status: str, error: str | None) -> int:
    """Stop the session, write the final event and return the process exit code."""
    try:
        with ev.span("aws", "stop_session"):
            tc.stop()
        ev.session("stopped", tc.training_session_name)
    except BaseException as e:  # still record the outcome; the session may be left running
        ev.log(f"Stopping session {tc.training_session_name} failed: {e!r}. Stop it on the Sessions page.",
               level="error")
        status, error = "failed", error or repr(e)
    ev.finished(status, error)
    return {"completed": 0, "stopped": 130}.get(status, 1)


def input_limit(base_model: str) -> tuple[int, bool]:
    """(limit, measured) for the longest input we will send to this model."""
    if base_model in MEASURED_INPUT_LIMIT:
        return MEASURED_INPUT_LIMIT[base_model], True
    return CONSERVATIVE_INPUT_LIMIT, False


def check_length(tokens: list[int], base_model: str, what: str) -> None:
    """Refuse an over-long input locally: the service fails the whole session instead of rejecting it."""
    limit, _ = input_limit(base_model)
    if len(tokens) > limit:
        raise ValueError(f"{what} has {len(tokens)} tokens, above the {limit}-token input limit for {base_model}; "
                         "sending it would put the training session into Failed")


def make_datum(obs_tokens, act_tokens, act_logprobs=None, advantage=None) -> Datum:
    """One training example: observation (prompt) + action (completion).

    tokens = obs + act, model input = tokens[:-1], target = tokens[1:]. The first
    len(obs) - 1 positions predict prompt tokens and get weight 0; the remaining
    len(act) positions are trained. RL datums also carry the sampler's per-token
    logprobs and one scalar advantage for the whole completion.
    """
    tokens = list(obs_tokens) + list(act_tokens)
    inputs, targets = tokens[:-1], tokens[1:]
    n, k = len(inputs), len(obs_tokens) - 1
    loss_fn_inputs = {
        "target_tokens": TensorData(targets, dtype=TensorDtype.INT64, shape=[n]),
        "weights": TensorData([0.0] * k + [1.0] * len(act_tokens), shape=[n]),
    }
    if act_logprobs is not None:
        if len(act_logprobs) != len(act_tokens):
            raise ValueError(f"{len(act_logprobs)} logprobs for {len(act_tokens)} action tokens")
        loss_fn_inputs["logprobs"] = TensorData([0.0] * k + list(act_logprobs), shape=[n])
        loss_fn_inputs["advantages"] = TensorData([0.0] * k + [float(advantage)] * len(act_tokens), shape=[n])
    return Datum(model_input=ModelInput.from_ints(inputs), loss_fn_inputs=loss_fn_inputs)


def datum_tokens(datum: Datum) -> int:
    """Billable train tokens for one datum: the length of its model input.

    Matches the measured billing: 8 SFT datums with 128 input tokens in total billed
    1,024 train tokens over 8 forward_backward steps.
    """
    return datum.model_input.length


def encode(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False)


def decode(tok, tokens) -> str:
    return tok.decode(tokens, skip_special_tokens=True)


def forward_logprobs(tc, datums: list, loss_fn: str = "cross_entropy") -> list[list[float]]:
    """Per-token log-probs of every datum's targets under the current weights, no gradient.

    ``forward`` returns them in ``loss_fn_outputs``. Measured on 2026-10-27 (16 datums,
    7,250 model-input tokens): a single entry whose ``logprobs`` held 7,266 values, i.e. all
    datums concatenated with one extra position each. The cookbook's preference recipe
    parses the same shape and also accepts one entry per datum; so does this. Each list
    returned has one value per model-input position, aligned with ``target_tokens``.
    """
    result = tc.forward(datums, loss_fn=loss_fn).result()
    lfo = result.loss_fn_outputs
    entries = lfo if isinstance(lfo, list) else ([lfo] if lfo else [])
    lens = [d.model_input.length for d in datums]

    def values(entry) -> list[float]:
        lp = entry.get("logprobs") if isinstance(entry, dict) else None
        if lp is None:
            raise RuntimeError(f"forward returned no per-token 'logprobs' (keys: {sorted(entry) if isinstance(entry, dict) else type(entry).__name__})")
        return [float(x) for x in (lp["data"] if isinstance(lp, dict) else lp)]

    if len(entries) == len(datums):
        return [values(e)[:n] for e, n in zip(entries, lens)]
    if len(entries) == 1:
        flat = values(entries[0])
        for extra in (1, 0):
            if sum(n + extra for n in lens) == len(flat):
                out, offset = [], 0
                for n in lens:
                    out.append(flat[offset:offset + n])
                    offset += n + extra
                return out
        raise RuntimeError(f"forward returned {len(flat)} log-probs, which do not split into {len(datums)} datums "
                           f"of input lengths {lens} (+0 or +1 each)")
    raise RuntimeError(f"forward returned {len(entries)} loss_fn_outputs for {len(datums)} datums")


def pending_probes() -> list[dict]:
    """Questions the run console has left in inbox/ since the last call, oldest first."""
    if not os.path.isdir(INBOX):
        return []
    out = []
    for name in sorted(n for n in os.listdir(INBOX) if n.endswith(".json")):
        path = os.path.join(INBOX, name)
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
        os.makedirs(os.path.join(INBOX, "answered"), exist_ok=True)
        os.replace(path, os.path.join(INBOX, "answered", name))
    return out


def answer_probes(ev, tok, step: int, weights: str, sampler=None, snapshot=None):
    """Answer the questions asked from the run console, between two steps of the loop.

    The console only writes a file into inbox/. This process decides when to answer, so
    every call to the service is still made here and shows up as a span. ``weights`` says
    which weights answer ("after 3 updates"). A loop with no sampler at hand (SFT has none
    until the end) passes ``snapshot``, which saves the current weights and binds a sampler
    to them: about 20 s and one MODEL package (measured 2026-10-27). Returns the sampler used.
    """
    probes = pending_probes()
    if not probes:
        return sampler
    if sampler is None:
        with ev.span("aws", "save_weights_and_rebind", step=step, calls=1, role="probe",
                     note="snapshot of the current weights, to answer your question"):
            sampler = snapshot()
        ev.artifact("MODEL", sampler.model_package_arn, f"Snapshot taken to answer a question at step {step}", step)
    prompts = [encode(tok, f"Q: {p['question'].strip()}\nA:") for p in probes]
    with ev.span("aws", "sample", step=step, calls=len(prompts), role="probe",
                 note="your question" if len(prompts) == 1 else f"{len(prompts)} of your questions") as s:
        futures = [sampler.sample(x, num_samples=1, sampling_params=PROBE_SAMPLING) for x in prompts]
        outs = [f.result().sequences[0].tokens for f in futures]
        s["tokens"] = {"prefill": sum(map(len, prompts)), "sample": sum(map(len, outs))}
    for p, o in zip(probes, outs):
        ev.emit("probe", id=p["id"], question=p["question"], answer=decode(tok, o).strip(), step=step, weights=weights)
    return sampler
