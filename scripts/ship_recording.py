#!/usr/bin/env python3
"""Copy recordings of live runs into their card, templates/<scenario>/recordings/, so they come with the project.

    python3 scripts/ship_recording.py data/recordings/<name> [...]   # one or more
    python3 scripts/ship_recording.py --all                            # every recording in data/recordings/

A shipped recording is replayed on machines that never ran it, and its live runs are what the
cost estimate there is based on (backend/srl/estimate.py). It lives in its scenario's card, so a
release that leaves the card out leaves its recordings out too. What only means something in the
account that recorded it is changed or left out:

    account IDs in ARNs and bucket names   replaced with 111122223333, the AWS documentation's example
    mlflow events, config.json "mlflow"    removed: they point at that account's MLflow app
    stdout.log, run.json, caches, inbox/   not copied (a replay writes its own stdout.log)
    copies of templates/_shared/ files     not copied: a replay gets this installation's (backend/srl/runs.py)
    README.md                              not copied: generated for an export; a replay writes its own

recording.json gets a "scrubbed" entry saying so. Afterwards every file is searched for the
original IDs and for any other 12-digit account in an ARN; finding one is an error, and the
copy is removed. An existing shipped recording of the same name is an error unless --replace.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
PLACEHOLDER = "111122223333"
NEVER_REAL = {PLACEHOLDER, "000000000000"}  # the documentation example, and an all-zero placeholder
SKIP_FILES = {"stdout.log", "run.json"}
SKIP_DIRS = {"__pycache__", "inbox", ".pytest_cache"}
GENERATED = {"README.md"}
ACCOUNT_IN_ARN = re.compile(r"arn:aws[a-z-]*:[a-z0-9-]+:[a-z0-9-]*:(\d{12})")
ACCOUNT_IN_BUCKET = re.compile(r"sagemaker-[a-z]{2}(?:-[a-z]+)+-\d-(\d{12})")


def accounts_in(text: str) -> set[str]:
    return {m.group(1) for rx in (ACCOUNT_IN_ARN, ACCOUNT_IN_BUCKET) for m in rx.finditer(text)} - NEVER_REAL


def text_files(d: Path):
    for f in sorted(d.rglob("*")):
        if f.is_file() and not SKIP_DIRS & set(f.relative_to(d).parts):
            yield f


def ship(src: Path, replace: bool) -> Path:
    if not (src / "recording.json").is_file() or not (src / "events.jsonl").is_file():
        sys.exit(f"{src}: not a recording (needs recording.json and events.jsonl)")
    first = json.loads((src / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    if first.get("type") != "run_started" or first.get("backend") != "live":
        sys.exit(f"{src}: only live runs are shipped; its first event says backend={first.get('backend')!r}")
    scenario = json.loads((src / "recording.json").read_text(encoding="utf-8")).get("scenario")
    if not scenario or not (TEMPLATES / scenario / "scenario.json").is_file():
        sys.exit(f"{src}: recording.json names scenario {scenario!r}, which is not a card in {TEMPLATES}")
    dst = TEMPLATES / scenario / "recordings" / src.name
    if dst.exists():
        if not replace:
            sys.exit(f"{dst} already exists; pass --replace to overwrite it")
        shutil.rmtree(dst)
    ids = set()
    for f in text_files(src):
        if f.name not in SKIP_FILES:
            ids |= accounts_in(f.read_text(encoding="utf-8", errors="replace"))

    def scrub(text: str) -> str:
        for account in ids:
            text = text.replace(account, PLACEHOLDER)
        return text

    dst.mkdir(parents=True)
    dst.parent.chmod(0o755)  # recordings/ itself, when this is the card's first
    removed_mlflow = 0
    platform = {p.name for p in (TEMPLATES / "_shared").iterdir()} if (TEMPLATES / "_shared").is_dir() else set()
    left_out = sorted(f.name for f in src.iterdir() if f.is_file() and f.name in platform | GENERATED)
    # recording.json last: it reports what was changed in the other files
    for f in sorted(text_files(src), key=lambda f: (f.relative_to(src).as_posix() == "recording.json", str(f))):
        rel = f.relative_to(src)
        if rel.name in SKIP_FILES or f.suffix in (".tmp", ".pyc") or rel.as_posix() in left_out:
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        text = f.read_text(encoding="utf-8")
        if rel.as_posix() == "events.jsonl":
            kept = []
            for line in text.splitlines():
                e = json.loads(line)
                if e["type"] == "mlflow":
                    removed_mlflow += 1
                    continue
                kept.append(json.dumps({**e, "seq": len(kept)}, ensure_ascii=False))
            text = "\n".join(kept) + "\n"
        elif rel.as_posix() == "config.json":
            config = json.loads(text)
            config.pop("mlflow", None)
            text = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        elif rel.as_posix() == "recording.json":
            info = json.loads(text)
            removed = [f"{removed_mlflow} mlflow events", "stdout.log"]
            removed += [f"{len(left_out) - ('README.md' in left_out)} copies of templates/_shared files"] if left_out else []
            removed += ["README.md"] if "README.md" in left_out else []
            info["scrubbed"] = {"date": date.today().isoformat(), "account_id": PLACEHOLDER, "removed": removed}
            text = json.dumps(info, indent=2, ensure_ascii=False) + "\n"
        out.write_text(scrub(text), encoding="utf-8")

    problems = []
    for f in text_files(dst):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            left = [a for a in ids if a in line] or sorted(accounts_in(line))
            if left:
                problems.append(f"{f.relative_to(ROOT)}:{n}: account {', '.join(left)} is still there")
    if problems:
        shutil.rmtree(dst)
        sys.exit("not shipped, scrubbing missed something:\n  " + "\n  ".join(problems[:20]))
    # templates/ is mounted read-only into the app's container, whose user is not this one: whatever this
    # shell's umask (0077 on some hosts), the shipped copy must be readable by others.
    for d in [dst, *[x for x in dst.rglob("*") if x.is_dir()]]:
        d.chmod(0o755)
    for f in (x for x in dst.rglob("*") if x.is_file()):
        f.chmod(0o644)
    print(f"{src.name}: shipped to {dst.relative_to(ROOT)} ({len(ids)} account ID(s) replaced, "
          f"{removed_mlflow} mlflow events removed)")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recordings", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true", help="every recording in data/recordings/")
    ap.add_argument("--replace", action="store_true", help="overwrite shipped recordings of the same name")
    args = ap.parse_args()
    sources = sorted((ROOT / "data" / "recordings").iterdir()) if args.all else args.recordings
    if not sources:
        ap.error("name at least one recording, or pass --all")
    for src in sources:
        ship(src.resolve(), args.replace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
