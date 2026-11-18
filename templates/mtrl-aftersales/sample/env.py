"""售后裁决任务的环境沙箱与工具实现。

设计要点（这是 MTRL 项目里客户必须自己解决、官方材料没有覆盖的部分）：

1. 每条 rollout 从 prompt 中携带的 ``env_snapshot`` 重建一个 in-memory SQLite，
   rollout 结束即销毁。因此同一个 prompt 在任意训练步、任意并发下看到的环境完全一致，
   且不同 rollout 之间不会互相污染，也不会写到任何真实业务系统。
2. 工具调用全程记录（名称、参数、是否命中、是否重复），reward 侧据此计算流程合规分。
3. ``submit_decision`` 是唯一的终止动作，重复提交只认第一次。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from typing import Any

SCHEMA = """
CREATE TABLE orders (
    order_id      TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL,
    sku           TEXT NOT NULL,
    category      TEXT NOT NULL,
    order_amount  REAL NOT NULL,
    coupon_amount REAL NOT NULL,
    ordered_at    TEXT NOT NULL
);
CREATE TABLE logistics (
    order_id        TEXT PRIMARY KEY,
    delivery_status TEXT NOT NULL,   -- delivered | in_transit | lost
    delivered_at    TEXT,            -- NULL 表示未签收
    damage_reported INTEGER NOT NULL -- 0 | 1
);
CREATE TABLE policies (
    category                   TEXT PRIMARY KEY,
    no_reason_window_days      INTEGER NOT NULL,
    quality_window_days        INTEGER NOT NULL,
    exchange_supported         INTEGER NOT NULL,
    requires_original_packaging INTEGER NOT NULL
);
CREATE TABLE customer_history (
    customer_id         TEXT PRIMARY KEY,
    aftersales_count_90d INTEGER NOT NULL,
    abuse_flag          INTEGER NOT NULL
);
CREATE TABLE inventory (
    sku           TEXT PRIMARY KEY,
    available_qty INTEGER NOT NULL
);
"""

# 工具的 OpenAI function-calling 定义。训练与生产共用同一份，避免 schema 漂移。
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "按订单号查询订单详情：客户号、SKU、类目、订单金额、已用优惠券金额、下单日期。",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_logistics",
            "description": "查询物流状态：是否已签收、签收日期、是否有运输破损上报。退货窗口必须从签收日期起算。",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_return_policy",
            "description": "按商品类目查询售后政策：无理由退货窗口天数、质量问题窗口天数、是否支持换货、是否要求原包装。",
            "parameters": {
                "type": "object",
                "properties": {"category": {"type": "string"}},
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_history",
            "description": "查询客户近 90 天售后次数以及是否已被标记为售后滥用。滥用客户必须转人工。",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "查询 SKU 可换货库存。只有库存大于 0 时才能裁决为换货。",
            "parameters": {
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_decision",
            "description": "提交最终裁决并结束任务。只能调用一次。",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["refund", "exchange", "reject", "escalate"],
                    },
                    "amount": {"type": "number", "description": "退款金额；非退款场景填 0。"},
                    "reason_code": {
                        "type": "string",
                        "enum": [
                            "ABUSE_REVIEW",
                            "NOT_DELIVERED",
                            "DAMAGED_IN_TRANSIT",
                            "QUALITY_EXCHANGE",
                            "QUALITY_DEFECT",
                            "NO_REASON_RETURN",
                            "WINDOW_EXPIRED",
                        ],
                    },
                },
                "required": ["decision", "amount", "reason_code"],
            },
        },
    },
]

READ_TOOLS = frozenset(
    {"get_order", "get_logistics", "get_return_policy", "get_customer_history", "check_inventory"}
)
TERMINAL_TOOL = "submit_decision"


@dataclass
class ToolCall:
    """一次工具调用的记录，reward 侧用它判断流程合规性。"""

    name: str
    args: dict[str, Any]
    ok: bool
    turn: int


@dataclass
class Decision:
    decision: str
    amount: float
    reason_code: str


@dataclass
class RolloutTrace:
    """一条 rollout 的完整过程记录。"""

    calls: list[ToolCall] = field(default_factory=list)
    decision: Decision | None = None
    submit_attempts: int = 0
    turns: int = 0
    invalid_tool_calls: int = 0
    malformed_output: bool = False
    error: str | None = None
    #: 有多少轮是靠文本协议解析出动作的（模型没发原生 tool_calls）。
    text_protocol_turns: int = 0

    def called_names(self) -> list[str]:
        return [c.name for c in self.calls]

    def read_tools_called(self) -> set[str]:
        return {c.name for c in self.calls if c.name in READ_TOOLS}


class ToolError(Exception):
    """工具参数非法或查不到数据。会作为观测回给模型，不中断 rollout。"""


class AfterSalesEnv:
    """从 snapshot 重建的一次性售后环境。

    Args:
        snapshot: 由数据生成器写入 prompt 的环境快照。
        request_date: 客户提交售后申请的日期（ISO 日期串），窗口计算的"今天"。
    """

    def __init__(self, snapshot: dict[str, Any], request_date: str) -> None:
        self.request_date = request_date
        self.trace = RolloutTrace()
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._load(snapshot)

    def _load(self, snapshot: dict[str, Any]) -> None:
        for table in ("orders", "logistics", "policies", "customer_history", "inventory"):
            rows = snapshot.get(table) or []
            if not rows:
                continue
            cols = list(rows[0].keys())
            placeholders = ",".join("?" for _ in cols)
            sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
            self._conn.executemany(sql, [tuple(r[c] for c in cols) for r in rows])
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AfterSalesEnv":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- 工具分发 -------------------------------------------------------

    def call(self, name: str, args: dict[str, Any], turn: int) -> str:
        """执行一次工具调用，返回给模型的观测字符串（JSON）。"""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            self.trace.invalid_tool_calls += 1
            self.trace.calls.append(ToolCall(name, args, ok=False, turn=turn))
            return json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False)
        try:
            result = handler(args)
        except ToolError as e:
            self.trace.calls.append(ToolCall(name, args, ok=False, turn=turn))
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        self.trace.calls.append(ToolCall(name, args, ok=True, turn=turn))
        return json.dumps(result, ensure_ascii=False)

    def _one(self, sql: str, param: Any, missing: str) -> sqlite3.Row:
        row = self._conn.execute(sql, (param,)).fetchone()
        if row is None:
            raise ToolError(missing)
        return row

    def _tool_get_order(self, args: dict[str, Any]) -> dict[str, Any]:
        oid = args.get("order_id")
        if not isinstance(oid, str) or not oid:
            raise ToolError("order_id 必须是非空字符串")
        r = self._one("SELECT * FROM orders WHERE order_id=?", oid, f"订单不存在: {oid}")
        return dict(r)

    def _tool_get_logistics(self, args: dict[str, Any]) -> dict[str, Any]:
        oid = args.get("order_id")
        if not isinstance(oid, str) or not oid:
            raise ToolError("order_id 必须是非空字符串")
        r = self._one("SELECT * FROM logistics WHERE order_id=?", oid, f"无物流记录: {oid}")
        out = dict(r)
        out["damage_reported"] = bool(out["damage_reported"])
        return out

    def _tool_get_return_policy(self, args: dict[str, Any]) -> dict[str, Any]:
        cat = args.get("category")
        if not isinstance(cat, str) or not cat:
            raise ToolError("category 必须是非空字符串")
        r = self._one("SELECT * FROM policies WHERE category=?", cat, f"无该类目政策: {cat}")
        out = dict(r)
        out["exchange_supported"] = bool(out["exchange_supported"])
        out["requires_original_packaging"] = bool(out["requires_original_packaging"])
        return out

    def _tool_get_customer_history(self, args: dict[str, Any]) -> dict[str, Any]:
        cid = args.get("customer_id")
        if not isinstance(cid, str) or not cid:
            raise ToolError("customer_id 必须是非空字符串")
        r = self._one(
            "SELECT * FROM customer_history WHERE customer_id=?", cid, f"无客户记录: {cid}"
        )
        out = dict(r)
        out["abuse_flag"] = bool(out["abuse_flag"])
        return out

    def _tool_check_inventory(self, args: dict[str, Any]) -> dict[str, Any]:
        sku = args.get("sku")
        if not isinstance(sku, str) or not sku:
            raise ToolError("sku 必须是非空字符串")
        r = self._one("SELECT * FROM inventory WHERE sku=?", sku, f"无该 SKU 库存记录: {sku}")
        return dict(r)

    def _tool_submit_decision(self, args: dict[str, Any]) -> dict[str, Any]:
        self.trace.submit_attempts += 1
        decision = args.get("decision")
        amount = args.get("amount")
        reason = args.get("reason_code")

        valid_decisions = {"refund", "exchange", "reject", "escalate"}
        if decision not in valid_decisions:
            raise ToolError(f"decision 必须是 {sorted(valid_decisions)} 之一，收到 {decision!r}")
        if not isinstance(amount, (int, float)) or isinstance(amount, bool):
            raise ToolError(f"amount 必须是数字，收到 {amount!r}")
        if not isinstance(reason, str) or not reason:
            raise ToolError("reason_code 必须是非空字符串")

        # 只认第一次提交；重复提交由 reward 侧扣分。
        if self.trace.decision is None:
            self.trace.decision = Decision(decision, float(amount), reason)
        return {"accepted": True, "terminal": True}


def days_between(start: str, end: str) -> int:
    """两个 ISO 日期之间的天数（end - start）。"""
    return (date.fromisoformat(end) - date.fromisoformat(start)).days
