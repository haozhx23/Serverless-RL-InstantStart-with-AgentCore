"""训练集与评估集生成器。

三条设计约束直接来自 MTRL 的数据接口特性：

1. **服务只读一列字符串**，不解析不校验。所以整个任务对象（含标准答案、
   环境快照、工具配置）都要序列化进 ``prompt`` 列。列名必须显式叫 ``prompt``：
   服务的选列规则是"有 prompt 列就用它，否则用第一列"，写成 ``question``
   之类不会报错，而是静默读错列。
2. **训练集与评估集由同一段代码生成**，只换随机种子和数量。评估集格式、
   加密方式、schema 与训练集完全一致，否则 Agent 在评估任务里会解析失败。
3. **四类裁决均衡**。恒选单一类别的策略拿不到高分（见 ``reward.py``）。

环境快照随 prompt 下发，是本样例对"有状态业务系统如何隔离"的答案：
不需要托管沙箱，把该条任务需要的全部数据行塞进 prompt，Agent 侧用
in-memory SQLite 重建。代价是 prompt 变长（本样例约 1.2 KB），换来的是
每条 rollout 完全可复现、无副作用、无跨 rollout 污染。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

from sample.rules import decide_from_snapshot

DATA_SOURCE = "ecom-aftersales-v1"
ENV_CLASS = "aftersales"
ABILITY = "policy-compliance-decision"

CATEGORIES = {
    # category: (无理由窗口, 质量窗口, 支持换货, 需原包装)
    "consumer_electronics": (7, 180, 1, 1),
    "apparel": (7, 30, 1, 0),
    "fresh_food": (0, 3, 0, 0),
    "home_appliance": (7, 365, 1, 1),
    "books": (7, 15, 0, 0),
}

DECISION_CLASSES = ("refund", "exchange", "reject", "escalate")

SYSTEM_PROMPT = """你是电商平台的售后裁决助手。你必须严格按标准操作流程（SOP）逐步调用工具核实事实，然后提交裁决。

SOP 判定顺序（必须按此优先级）：
1. 先查客户售后历史。若客户被标记为售后滥用，直接 escalate，reason_code=ABUSE_REVIEW。
2. 再查物流。若订单未签收（delivery_status 不是 delivered），escalate，reason_code=NOT_DELIVERED。
3. 若物流已上报运输破损，refund 全额（订单金额减去已用优惠券），reason_code=DAMAGED_IN_TRANSIT。
4. 查该类目售后政策，计算签收后天数。**所有窗口都从签收日期 delivered_at 起算，不是下单日期。**
   - 质量问题且在质量窗口内：该类目支持换货时须查库存，库存大于 0 则 exchange（reason_code=QUALITY_EXCHANGE）；
     不支持换货或无库存则 refund（reason_code=QUALITY_DEFECT）。
   - 无理由退货且在无理由窗口内：refund，reason_code=NO_REASON_RETURN。
5. 其余情况 reject，reason_code=WINDOW_EXPIRED。

退款金额一律为 order_amount - coupon_amount（优惠券不退）；非退款场景 amount 填 0。
核实完毕后调用 submit_decision 提交，且只能调用一次。

调用工具的方式：优先使用系统提供的 function calling。如果你无法发起 function call，
就在回复里输出下面这种文本块，一次一个工具：

<tool_call>{"name": "get_order", "arguments": {"order_id": "SO-12345678"}}</tool_call>

每一轮只做一件事：要么调用一个工具，要么提交裁决。不要在同一轮里输出多个工具调用。"""

CLAIM_TEXT = {
    "quality": "收到的商品有质量问题，希望处理一下。",
    "no_reason": "买回来不太合适，想申请退货。",
}


def _make_snapshot(rng: random.Random, request_day: date) -> tuple[dict[str, Any], str, str]:
    """随机构造一份环境快照。返回 (snapshot, claim_type, request_date)。"""
    category = rng.choice(list(CATEGORIES))
    no_reason_days, quality_days, exchange_ok, packaging = CATEGORIES[category]

    order_id = f"SO-{rng.randint(10**7, 10**8 - 1)}"
    customer_id = f"CU-{rng.randint(10**6, 10**7 - 1)}"
    sku = f"SKU-{category[:3].upper()}-{rng.randint(1000, 9999)}"

    order_amount = round(rng.uniform(29.0, 4999.0), 2)
    # 约一半订单用过券，用于检验模型是否记得扣券。
    coupon_amount = round(rng.choice([0.0, 0.0, 20.0, 50.0, 100.0]), 2)
    coupon_amount = min(coupon_amount, round(order_amount * 0.5, 2))

    # 下单到签收刻意间隔 2~9 天：按下单日算窗口会在部分样本上得到不同结论。
    ship_lag = rng.randint(2, 9)
    days_since_delivery = rng.choice([0, 1, 2, 3, 5, 8, 12, 20, 40, 100, 200, 400])
    delivered_day = request_day - timedelta(days=days_since_delivery)
    ordered_day = delivered_day - timedelta(days=ship_lag)

    delivery_status = rng.choices(["delivered", "in_transit", "lost"], weights=[85, 10, 5])[0]
    damage_reported = 1 if rng.random() < 0.12 else 0

    snapshot = {
        "orders": [
            {
                "order_id": order_id,
                "customer_id": customer_id,
                "sku": sku,
                "category": category,
                "order_amount": order_amount,
                "coupon_amount": coupon_amount,
                "ordered_at": ordered_day.isoformat(),
            }
        ],
        "logistics": [
            {
                "order_id": order_id,
                "delivery_status": delivery_status,
                "delivered_at": delivered_day.isoformat() if delivery_status == "delivered" else None,
                "damage_reported": damage_reported,
            }
        ],
        "policies": [
            {
                "category": cat,
                "no_reason_window_days": v[0],
                "quality_window_days": v[1],
                "exchange_supported": v[2],
                "requires_original_packaging": v[3],
            }
            for cat, v in CATEGORIES.items()
        ],
        "customer_history": [
            {
                "customer_id": customer_id,
                "aftersales_count_90d": rng.randint(0, 9),
                "abuse_flag": 1 if rng.random() < 0.10 else 0,
            }
        ],
        "inventory": [{"sku": sku, "available_qty": rng.choice([0, 0, 3, 12, 40])}],
    }
    claim_type = rng.choice(["quality", "no_reason"])
    return snapshot, claim_type, request_day.isoformat()


def _build_case(rng: random.Random, request_day: date, case_id: str) -> dict[str, Any]:
    """构造一条任务对象。标准答案一律由 rules.decide_from_snapshot 推导。"""
    snapshot, claim_type, request_date = _make_snapshot(rng, request_day)
    gt = decide_from_snapshot(snapshot, claim_type, request_date)
    order = snapshot["orders"][0]

    user_msg = (
        f"客户提交售后申请。\n"
        f"申请日期：{request_date}\n"
        f"订单号：{order['order_id']}\n"
        f"诉求类型：{'质量问题' if claim_type == 'quality' else '七天无理由退货'}\n"
        f"客户描述：{CLAIM_TEXT[claim_type]}\n\n"
        f"请按 SOP 核实后提交裁决。"
    )

    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "ability": ABILITY,
        "env_class": ENV_CLASS,
        "reward_spec": {
            "style": "rule",
            "ground_truth": {
                "decision": gt.decision,
                "amount": gt.amount,
                "reason_code": gt.reason_code,
                "required_tools": list(gt.required_tools),
            },
        },
        "extra_info": {
            "case_id": case_id,
            "claim_type": claim_type,
            "request_date": request_date,
            "env_snapshot": snapshot,
            "max_turns": 10,
            # 便于人工核对，不参与打分；上生产前应确认是否需要移除。
            "rationale": gt.rationale,
        },
    }


def generate_cases(
    n: int, seed: int, balanced: bool = True, request_day: date | None = None
) -> list[dict[str, Any]]:
    """生成 n 条任务对象。

    Args:
        balanced: 是否对四类裁决做拒绝采样均衡。关闭后分布会明显偏向
            reject/refund，模型容易学到"恒选多数类"的捷径。
    """
    rng = random.Random(seed)
    day = request_day or date(2026, 8, 20)
    cases: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    attempts = 0
    max_attempts = n * 2000

    if balanced:
        # 精确配额：前 n % 4 个类别多拿一条，四类合计正好 n。
        k = len(DECISION_CLASSES)
        quota = {c: n // k + (1 if i < n % k else 0) for i, c in enumerate(DECISION_CLASSES)}
    else:
        quota = None

    while len(cases) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"拒绝采样未能在 {max_attempts} 次内凑齐 {n} 条样本"
                f"（目标分布 {quota}，当前 {dict(counts)}）。"
                " exchange 类天然稀有，请调大 max_attempts 或调整 _make_snapshot 的分布参数。"
            )
        case = _build_case(rng, day, case_id="pending")
        decision = case["reward_spec"]["ground_truth"]["decision"]
        if quota is not None and counts[decision] >= quota[decision]:
            continue
        counts[decision] += 1
        case["extra_info"]["case_id"] = f"case-{len(cases):05d}"
        cases.append(case)

    if quota is not None:
        # 逐类比较：Counter 只含出现过的键，不能直接和含 0 配额的 dict 比。
        mismatch = {c: (counts[c], quota[c]) for c in DECISION_CLASSES if counts[c] != quota[c]}
        if mismatch:
            raise RuntimeError(f"均衡失败，(实际, 目标) 为 {mismatch}")
    rng.shuffle(cases)
    return cases


# ---- 序列化：明文 / AES-GCM 加密 ----------------------------------------


def encrypt_task(payload: str, key: bytes) -> str:
    """AES-256-GCM 加密并 base64 编码。

    文档明确指出 base64 只是编码、不构成保护。敏感数据要真加密，密钥由
    Agent 侧从 Secrets Manager 取。这里返回 ``nonce || ciphertext`` 的 base64。
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:  # 不静默降级为明文
        raise RuntimeError(
            "请求了加密但未安装 cryptography；不会退回明文写出。"
            " 请 pip install cryptography 后重试。"
        ) from e
    if len(key) != 32:
        raise ValueError(f"AES-256 需要 32 字节密钥，收到 {len(key)} 字节")
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, payload.encode("utf-8"), None)
    return base64.b64encode(nonce + ct).decode("ascii")


def decrypt_task(blob: str, key: bytes) -> str:
    """解密 ``encrypt_task`` 的输出。Agent 侧使用。"""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    raw = base64.b64decode(blob)
    return AESGCM(key).decrypt(raw[:12], raw[12:], None).decode("utf-8")


def to_rows(cases: list[dict[str, Any]], key: bytes | None) -> Iterator[dict[str, str]]:
    """把任务对象转成只有一列 ``prompt`` 的数据行。"""
    for case in cases:
        payload = json.dumps(case, ensure_ascii=False, separators=(",", ":"))
        yield {"prompt": encrypt_task(payload, key) if key else payload}


def write_jsonl(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_parquet(rows: list[dict[str, str]], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError(
            "请求了 parquet 输出但未安装 pyarrow。大数据集建议 parquet，"
            " 小规模试运行可改用 --format jsonl。"
        ) from e
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def main() -> None:
    p = argparse.ArgumentParser(description="生成售后裁决 MTRL 训练/评估数据集")
    p.add_argument("--out-dir", default="data", type=Path)
    p.add_argument("--train-size", type=int, default=640, help="建议 >= 10 x global_batch_size")
    p.add_argument("--eval-size", type=int, default=160)
    p.add_argument("--train-seed", type=int, default=20260824)
    p.add_argument("--eval-seed", type=int, default=99991, help="必须与 train-seed 不同")
    p.add_argument("--format", choices=["jsonl", "parquet"], default="jsonl")
    p.add_argument("--no-balance", action="store_true")
    p.add_argument(
        "--encrypt-key-hex",
        default=None,
        help="32 字节密钥的 hex 表示（64 个 hex 字符）。给定则对 prompt 列做 AES-GCM 加密。",
    )
    args = p.parse_args()

    if args.train_seed == args.eval_seed:
        raise SystemExit("train-seed 与 eval-seed 相同会导致评估集与训练集重叠，已拒绝执行。")

    key = bytes.fromhex(args.encrypt_key_hex) if args.encrypt_key_hex else None
    balanced = not args.no_balance
    writer = write_jsonl if args.format == "jsonl" else write_parquet
    ext = "jsonl" if args.format == "jsonl" else "parquet"

    for split, size, seed in (
        ("train", args.train_size, args.train_seed),
        ("eval", args.eval_size, args.eval_seed),
    ):
        cases = generate_cases(size, seed=seed, balanced=balanced)
        dist = Counter(c["reward_spec"]["ground_truth"]["decision"] for c in cases)
        rows = list(to_rows(cases, key))
        out = args.out_dir / f"{split}.{ext}"
        writer(rows, out)
        avg_len = sum(len(r["prompt"]) for r in rows) / len(rows)
        print(f"{split}: {len(rows)} 条 -> {out}")
        print(f"  裁决分布 {dict(dist)}")
        print(f"  prompt 列平均长度 {avg_len:.0f} 字符，加密={'是' if key else '否'}")


if __name__ == "__main__":
    main()
