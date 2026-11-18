"""Runs one solution against its test cases, inside the sandbox.

task.grade() uploads this file with solution.py and cases.json and starts it there as a program
of its own (python runner.py <nonce>); it never runs in the training loop's process. It prints
one line last, "SRL-RESULT <nonce> {"passed": n, ...}", and exits 0 only when every test passed,
like pytest. A solution that ends the program first (os._exit, a crash, an endless loop cut off
by the timeout) leaves no result line.

What the solution prints is swallowed, so it cannot crowd out the result line. The solution runs
in this same process, so a determined one can still tamper with this file's work: the sandbox
protects the machine and its credentials, not the reward. Check solutions by hand when the
reward rises suspiciously fast.
"""

import contextlib
import copy
import importlib.util
import io
import json
import math
import sys


def same(got, expected) -> bool:
    """Equal as JSON would compare them (tuples are lists); floats within 1e-9; bool is not int."""
    if isinstance(expected, float) or isinstance(got, float):
        return (isinstance(got, (int, float)) and isinstance(expected, (int, float)) and not isinstance(got, bool)
                and math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-9))
    if isinstance(expected, list):
        return isinstance(got, (list, tuple)) and len(got) == len(expected) and all(map(same, got, expected))
    if isinstance(expected, dict):
        return isinstance(got, dict) and got.keys() == expected.keys() and all(same(got[k], expected[k]) for k in got)
    return type(got) is type(expected) and got == expected


def main() -> int:
    nonce = sys.argv[1]
    with open("cases.json", encoding="utf-8") as f:
        spec = json.load(f)
    out = {"passed": 0, "failed": [], "error": None}
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            loader = importlib.util.spec_from_file_location("solution", "solution.py")
            module = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(module)
            fn = getattr(module, spec["name"])
        except BaseException as e:  # noqa: BLE001 - a broken solution is a result, not a crash of the runner
            out["error"] = f"{type(e).__name__}: {e}"[:300]
        else:
            for i, case in enumerate(spec["cases"]):
                try:
                    ok = same(fn(*copy.deepcopy(case["args"])), case["expected"])
                except BaseException as e:  # noqa: BLE001
                    ok = False
                    out["error"] = out["error"] or f"case {i}: {type(e).__name__}: {e}"[:300]
                if ok:
                    out["passed"] += 1
                elif len(out["failed"]) < 3:
                    out["failed"].append(i)
    print(f"SRL-RESULT {nonce} {json.dumps(out)}", flush=True)
    return 0 if out["passed"] == len(spec["cases"]) else 1


if __name__ == "__main__":
    sys.exit(main())
