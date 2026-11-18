"""Re-emit a recorded live run's events with their original timing. Makes no AWS calls.

    python srl_replay.py <recorded events.jsonl> [--speed 4]

A replay goes through the same pipeline as a live run (this process writes events.jsonl,
the backend streams it), so the UI shows it exactly as it was recorded. Every event keeps
its recorded values; run_started is marked backend="replay" with the recording date so
the UI never presents a replay as a live run.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

from srl_events import install_stop_handlers, load_config, open_log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed; 4 plays four times faster")
    args = ap.parse_args()
    if args.speed <= 0:
        sys.exit("--speed must be positive")

    with open(args.recording, encoding="utf-8") as f:
        recorded = [json.loads(line) for line in f if line.strip()]
    if not recorded or recorded[0]["type"] != "run_started":
        sys.exit(f"{args.recording} does not start with a run_started event")
    if recorded[0].get("backend") != "live":
        sys.exit(f"{args.recording} is a {recorded[0].get('backend')!r} run; only live runs can be replayed")

    config = load_config()
    ev = open_log(config)
    ev.echo = False
    install_stop_handlers()
    recorded_at = datetime.fromtimestamp(recorded[0]["ts"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    start = time.monotonic()
    try:
        for e in recorded:
            wait = e["t"] / args.speed - (time.monotonic() - start)
            if wait > 0:
                time.sleep(wait)
            fields = {k: v for k, v in e.items() if k not in ("v", "seq", "ts", "t", "run", "type")}
            if e["type"] == "run_started":
                fields.update(backend="replay", recorded_run=e["run"], recorded_at=recorded_at, speed=args.speed)
            ev.emit(e["type"], **fields)
    except KeyboardInterrupt:
        ev.emit("run_finished", status="stopped", error=None, note="replay stopped")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
