"""List prices for training-session tokens and the cost estimate shown while a run is live.

USD per 1M tokens (prefill, sample, train) from the AWS Price List API, checked
2026-10-27; the same price in every region where the model is offered. The estimate is
tokens x list price; the authoritative figure is DescribeTrainingSession.BillableTokenUsage,
which lags by tens of seconds to minutes.
"""

from __future__ import annotations

from .i18n import L, pick

PRICES_AS_OF = "2026-10-27"
PRICES_SOURCE = "AWS Price List API"

# base model id (as list_base_models() returns it) -> Hub content name used in prices
HUB_NAME = {
    "openai/gpt-oss-20b": "openai-reasoning-gpt-oss-20b",
    "google/gemma-4-31b-it": "huggingface-vlm-gemma-4-31b-it",
    "Qwen/Qwen3.6-27B": "huggingface-vlm-qwen3-6-27b",
    "Qwen/Qwen3.8-27B": "huggingface-vlm-qwen3-8-27b",
}
PRICES = {
    "openai-reasoning-gpt-oss-20b": {"prefill": 0.12, "sample": 0.30, "train": 0.36},
    "huggingface-vlm-gemma-4-31b-it": {"prefill": 1.24, "sample": 3.73, "train": 3.73},
    "huggingface-vlm-qwen3-6-27b": {"prefill": 1.24, "sample": 3.73, "train": 3.73},
    "huggingface-vlm-qwen3-8-27b": {"prefill": 1.24, "sample": 3.73, "train": 3.73},
}


# MTRL jobs, keyed by the Hub content name the MTRL SDK uses. From the same Price List
# fetch: usage types "MultiTurnRFT:Finetuning-LORA-{Prefill,Sample,Train}-<model>". That MTRL
# training jobs bill under these usage types is inferred from the names, not documented.
MTRL_PRICES = {
    "openai-reasoning-gpt-oss-20b": {"prefill": 0.12, "sample": 0.30, "train": 0.36},
    "huggingface-vlm-gemma-4-31b-it": {"prefill": 1.24, "sample": 3.73, "train": 3.73},
    "huggingface-vlm-qwen3-6-27b": {"prefill": 1.24, "sample": 3.73, "train": 3.73},
    "nova-textgeneration-lite-v2": {"prefill": 0.49, "sample": 2.00, "train": 1.47},
}
MTRL_PRICES_BASIS = L("MultiTurnRFT Finetuning-LORA usage types (mapping to MTRL jobs inferred from the names)",
                      "MultiTurnRFT Finetuning-LORA 计费项（根据名称推断其与 MTRL 任务的对应关系）")


# AgentCore Runtime, the environment sandbox and MTRL agents: USD per vCPU-hour and per GB-hour of memory, from
# the Price List API (AmazonBedrockAgentCore, usage types Runtime:Consumption-based:vCPU / Memory, "-v2" for
# platform V2), read on 2026-11-10. CPU is billed only while used; memory all the time a session is alive.
# Other regions were not read, so they have no price here (never a guess).
AGENTCORE_PRICES_AS_OF = "2026-11-10"
_AGENTCORE = {"V1": {"vcpu_h": 0.0895, "gb_h": 0.00945}, "V2": {"vcpu_h": 0.1276, "gb_h": 0.0169}}
AGENTCORE_PRICES = {"us-west-2": _AGENTCORE, "us-east-1": _AGENTCORE}


def agentcore_price(region: str, platform: str | None) -> dict | None:
    return AGENTCORE_PRICES.get(region, {}).get(platform or "")


def price_for(model_or_hub: str, engine: str = "training-session") -> dict | None:
    if engine == "mtrl":
        return MTRL_PRICES.get(model_or_hub)
    return PRICES.get(HUB_NAME.get(model_or_hub, model_or_hub))


def estimate(model_or_hub: str, tokens: dict, engine: str = "training-session") -> float | None:
    """USD at list price, or None when the model has no known price (never a guess)."""
    price = price_for(model_or_hub, engine)
    if price is None:
        return None
    return sum(tokens.get(k, 0) * price[k] for k in price) / 1e6


def table(lang: str = "en") -> dict:
    return {"as_of": PRICES_AS_OF, "source": PRICES_SOURCE, "unit": "USD per 1M tokens",
            "models": {m: PRICES[h] for m, h in HUB_NAME.items()},
            "mtrl": {"models": MTRL_PRICES, "basis": pick(MTRL_PRICES_BASIS, lang)},
            "agentcore": {"as_of": AGENTCORE_PRICES_AS_OF, "unit": "USD per vCPU-hour and per GB-hour",
                          "regions": AGENTCORE_PRICES}}
