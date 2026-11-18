"""The task-specific functions of the official AWS sample, copied verbatim. Do not edit them here.

Source: aws/amazon-sagemaker-examples, commit 3429cb1a55 (2026-10-14),
"     model_customization/sm-training_sessions_grpo_gsm8k/sm-training_sessions_1_train_grpo_gsm8k.ipynb"
(the directory name starts with five spaces), code cells 15, 17, 23, 25, in that order. The sample is the one the
2026-10-22 launch blog walks through. Licensed under the Apache License 2.0 (LICENSE.apache-2.0 here).

loop.py runs the notebook's training loop around these functions, unchanged except that every API call is
recorded for the loop map and the meter. What else differs from the notebook is listed in loop.py.
"""

def build_prompt(question: str) -> str:
    """Render the question in the GPT-OSS harmony chat format, and ask for step-by-step
    reasoning with the final answer after '#### ' so the reward function can parse it.
    Prompting the model in its trained format is what makes it emit the marker (and stop)
    reliably -- a raw 'Question:/Answer:' string does not.

    This template is specific to GPT-OSS. Changing BASE_MODEL means rewriting it for
    that model's chat format (or building it with the tokenizer's chat template);
    left as-is, the harmony control tokens become literal text and the model will
    neither reason nor emit the '#### ' marker the reward function looks for."""
    instruction = (
        f"{question}\n\n"
        "Solve this step by step, then give the final numeric answer on its own "
        'line after "#### ".'
    )
    return (
        "<|start|>system<|message|>You are a helpful assistant.\n"
        "Reasoning: high\n"
        "# Valid channels: analysis, commentary, final. "
        "Channel must be included for every message.<|end|>"
        f"<|start|>user<|message|>{instruction}<|end|>"
        "<|start|>assistant"
    )


import re


def extract_answer(text: str) -> str | None:
    # The final answer lives at the end of the response, so look only at the tail and
    # take the LAST "#### <number>" match -- robust to the model reasoning with "####"
    # earlier or repeating the answer.
    tail = text[-300:]
    solutions = re.findall(r"####\s*\$?\s*(-?[0-9\.,]+)", tail)
    if not solutions:
        return None
    return solutions[-1].replace(",", "").strip()


def compute_reward(response_text: str, ground_truth: str) -> float:
    predicted = extract_answer(response_text)
    return 1.0 if predicted == ground_truth else 0.0


import numpy as np


def compute_advantages(group):
    """Center (and normalize) rewards within a group -- classic GRPO baseline."""
    rewards = np.array([r["reward"] for r in group])
    advantages = rewards - rewards.mean()
    std = rewards.std()
    if std > 1e-8:
        advantages = advantages / std
    for response, advantage in zip(group, advantages):
        response["advantage"] = float(advantage)
    return group


def build_training_datum(response: dict) -> dict:
    # Next-token prediction shifts inputs and targets by one, so the prompt occupies
    # len(prompt_tokens) - 1 positions of the shifted sequence: the prompt's final token
    # is an input whose target is the response's first token -- the first position we
    # train on.
    prompt_len = len(response["prompt_tokens"]) - 1
    tokens = response["prompt_tokens"] + response["response_tokens"]

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    response_len = len(input_tokens) - prompt_len

    logprobs = [0.0] * prompt_len + list(response["response_logprobs"])
    advantages = [0.0] * prompt_len + [response["advantage"]] * response_len
    weights = [0.0] * prompt_len + [1.0] * response_len  # 0 = prompt, 1 = response

    seq_len = len(input_tokens)
    return {
        "modelInput": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
        "lossFnInputs": {
            "target_tokens": {"data": target_tokens, "dtype": "int64", "shape": [seq_len]},
            "logprobs": {"data": logprobs, "dtype": "float32", "shape": [seq_len]},
            "advantages": {"data": advantages, "dtype": "float32", "shape": [seq_len]},
            "weights": {"data": weights, "dtype": "float32", "shape": [seq_len]},
        },
    }
