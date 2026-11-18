"""The task for the code RL scenario: write a Python function so that its tests pass.

problems.json holds 24 training and 8 held-out functions, each with test cases whose expected
values were computed from a reference body. The prompt shows the function's signature and
docstring, with one example, in gpt-oss's harmony chat format, and asks for the whole function
in one ```python block; the solution is the last such block of the answer.

Why harmony: the first live run (2026-11-10) used a plain Python file to continue, as rl-arith
does with "Q: ... A:". gpt-oss-20b answered every one of 192 prompts with prose reasoning instead
of code, so no test ever passed and no group gave a gradient. The official GSM8K sample notes the
same about its prompt (official.py, build_prompt). Like it, this format is specific to gpt-oss.

The body is never run in this process. grade() sends it, the cases and runner.py to a sandbox
(sandbox_client.Sandbox: an AgentCore Runtime session when live) and runs runner.py there as a
program of its own. It returns what happened: tests passed out of total, whether the program
timed out, its exit code, and whether it reported a result at all. Whether that is worth a
reward is reward.py's decision. is_correct() is what the held-out comparison counts: every test
passed in time. It deliberately does not use reward.py, so a reward you swap or edit cannot
change what counts as a working function.
"""

from __future__ import annotations

import json
import re
import secrets
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROBLEMS = json.loads((_HERE / "problems.json").read_text(encoding="utf-8"))["problems"]
TRAIN = [p for p in PROBLEMS if p["split"] == "train"]
HELDOUT = [p for p in PROBLEMS if p["split"] == "heldout"]
RUNNER = (_HERE / "runner.py").read_text(encoding="utf-8")

INSTRUCTION = "Complete this Python function. Reply with the whole function in one ```python code block."
CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def head(p: dict) -> str:
    """The function's signature and docstring, with its first test case as the example."""
    c = p["cases"][0]
    call = f"{p['id']}({', '.join(repr(a) for a in c['args'])})"
    return f'{p["signature"]}\n    """{p["doc"]}\n\n    >>> {call}\n    {c["expected"]!r}\n    """\n'


def prompt(p: dict) -> str:
    """The request in the gpt-oss harmony chat format, with low reasoning effort so the code fits in max_tokens."""
    return ("<|start|>system<|message|>You are a helpful assistant.\n"
            "Reasoning: low\n"
            "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
            f"<|start|>user<|message|>{INSTRUCTION}\n\n```python\n{head(p)}```<|end|>"
            "<|start|>assistant")


def extract(completion: str) -> str | None:
    """The last ```python block of an answer (the final channel comes after the reasoning), or None."""
    blocks = CODE_BLOCK.findall(completion)
    return blocks[-1] if blocks else None


def solution(p: dict, body: str) -> str:
    """A file made of the function's head and a body: how the reward check builds its known solutions."""
    return head(p) + body


def grade(box, p: dict, code: str, seed: int, timeout_s: float) -> dict:
    """Run `code` against p's cases in sandbox `box`; a fresh work directory each time."""
    nonce = secrets.token_hex(8)  # the result line must carry it, so text the solution prints cannot pass for one
    box.reset({"solution.py": code, "runner.py": RUNNER,
               "cases.json": json.dumps({"name": p["id"], "cases": p["cases"]})}, seed)
    r = box.run(["{python}", "runner.py", nonce], timeout_s)
    marker = f"SRL-RESULT {nonce} "
    line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith(marker)), None)
    out = {"total": len(p["cases"]), "passed": 0, "timed_out": r.timed_out, "exit_code": r.exit_code,
           "reported": line is not None, "error": None, "sec": r.sec}
    if line is not None:
        got = json.loads(line[len(marker):])
        out.update(passed=int(got["passed"]), error=got.get("error"))
    elif r.timed_out:
        out["error"] = f"no result: still running after {timeout_s:g} s"
    else:
        tail = (r.stderr.strip().splitlines() or [""])[-1][:200]
        out["error"] = f"no result: exit code {r.exit_code}" + (f", {tail}" if tail else "")
    return out


def is_correct(result: dict) -> bool:
    """What the held-out check counts as a working function: it reported, in time, and passed every test."""
    return result["reported"] and not result["timed_out"] and result["passed"] == result["total"]
