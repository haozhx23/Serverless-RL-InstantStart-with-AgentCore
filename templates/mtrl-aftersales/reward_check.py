"""Prove the reward separates good from bad trajectories before spending on a training job.

Eight scripted policies (local/policies.py) stand in for the model, each one a real failure
mode, and run through the real rollout path: the SDK's @sagemaker_rft_handler,
make_inference_headers and RolloutFeedbackClient all execute, against a local stand-in for
the Job Runtime Service (local/mock_runtime.py). generate_token signs locally with this
machine's AWS credentials and sends nothing.

Adapted from the 260824 sample's local/run_local.py; on 2026-10-28 it reproduced the
sample's numbers exactly (perfect 1.0000, skip_logistics 0.6169, always_reject 0.3767).
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import sys
from typing import Any

from local.mock_runtime import endpoint_of, make_server
from local.policies import POLICIES
from sample.dataset import generate_cases

DESCRIPTIONS = {
    "perfect": "Follows the SOP exactly",
    "tool_spam": "Calls the same tool repeatedly before submitting",
    "double_submit": "Right decision, submitted twice in parallel",
    "text_protocol": "Writes <tool_call> text blocks instead of native tool calls",
    "skip_logistics": "Skips the delivery lookup and counts the window from the order date",
    "blind_exchange": "Always exchanges quality claims without checking stock",
    "always_reject": "Calls every tool, then always rejects (reward-hacking probe)",
    "malformed": "Answers in prose, never calls a tool",
}


def run_policy(name: str, cases: list[dict[str, Any]], endpoint: str, region: str) -> dict[str, Any]:
    os.environ["ROLLOUT_MODEL"] = name
    sys.modules.pop("sample.rollout_handler", None)  # it reads ROLLOUT_MODEL at import
    from sample.rollout_handler import handle_rollout

    rewards, exact, coverage = [], [], []
    for i, case in enumerate(cases):
        payload = {
            "instance": {"prompt": json.dumps(case, ensure_ascii=False, separators=(",", ":"))},
            "metadata": {"job_arn": f"arn:aws:sagemaker:{region}:000000000000:job/local-{name}",
                         "trajectory_id": f"{name}-{i:04d}", "endpoint": endpoint, "region": region},
            "inferenceParams": {"temperature": 1.0, "topP": 1.0, "maxTokens": 512},
        }
        result = handle_rollout(payload)
        r = result["reward"]
        rewards.append(sum(r) if isinstance(r, list) else r)
        m = result.get("metrics") or {}
        exact.append(m.get("exact_match", 0.0))
        coverage.append(m.get("required_tool_coverage", 0.0))
    return {"policy": name, "description": DESCRIPTIONS.get(name, ""), "n": len(cases),
            "reward_mean": round(statistics.fmean(rewards), 4),
            "reward_std": round(statistics.pstdev(rewards), 4) if len(rewards) > 1 else 0.0,
            "exact_match": round(statistics.fmean(exact), 4), "coverage": round(statistics.fmean(coverage), 4)}


def assertions(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Each check: what it guards against, and whether it held."""
    p = rows["perfect"]["reward_mean"]
    checks = [
        ("perfect scores 1.0", rows["perfect"]["reward_mean"] > 0.999,
         "If the SOP-following policy cannot score full marks, the tools do not expose enough to solve the task."),
        ("malformed scores 0", rows["malformed"]["reward_mean"] == 0.0,
         "An unparsable answer must get no partial credit, or a lenient parser becomes a reward hack."),
        ("always_reject stays below 0.5", rows["always_reject"]["reward_mean"] < 0.5,
         "Calling every tool must not earn a high score on its own; the outcome has to be right."),
        ("skip_logistics scores well below perfect", rows["skip_logistics"]["reward_mean"] < p - 0.1,
         "Skipping a required lookup must cost reward."),
        ("blind_exchange scores well below perfect", rows["blind_exchange"]["reward_mean"] < p - 0.1,
         "Guessing the branch without checking stock must cost reward."),
        ("tool_spam scores below perfect", rows["tool_spam"]["reward_mean"] < p,
         "Repeated tool calls must cost something, or the model learns to spam (and prefill grows)."),
        ("double_submit = 0.8 x perfect", abs(rows["double_submit"]["reward_mean"] - 0.8 * p) < 1e-6,
         "Submitting several answers to hit one must be penalized."),
        ("text_protocol = perfect", abs(rows["text_protocol"]["reward_mean"] - p) < 1e-6,
         "The text tool-call fallback must work as well as native tool calls."),
        ("some policy has reward variance", any(r["reward_std"] > 0 for k, r in rows.items() if k != "perfect"),
         "With zero variance inside every group, advantages are zero and there is no gradient."),
    ]
    return [{"check": c, "ok": bool(ok), "why": why} for c, ok, why in checks]


def run(cases: int, seed: int, region: str) -> tuple[list[dict], list[dict]]:
    for name in ("sagemaker", "botocore", "boto3", "urllib3", "httpx", "openai", "sample"):
        logging.getLogger(name).setLevel(logging.WARNING)
    data = generate_cases(cases, seed=seed, balanced=True)
    server, _ = make_server()
    try:
        endpoint = endpoint_of(server)
        rows = {name: run_policy(name, data, endpoint, region) for name in POLICIES}
    finally:
        server.shutdown()
    return list(rows.values()), assertions(rows)
