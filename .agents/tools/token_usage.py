#!/usr/bin/env python3
"""Print the token/cost usage of an OpenCode session time window.

Used to stamp every operator-level commit/push with the wall-clock period and
the tokens consumed while that operator was being optimized (project rule).

Usage:
  python3 .agents/tools/token_usage.py --since "2026-10-17 16:03"
  python3 .agents/tools/token_usage.py --since "2026-10-17 16:03" --until "2026-10-17 16:26"
  python3 .agents/tools/token_usage.py --since ... --session ses_xxx

Defaults: --until = now; --session = the most recently updated root session
whose directory is the workspace root.  Reads the local OpenCode DB read-only.
"""

import argparse
import datetime
import json
import os
import sqlite3
import sys

DB = os.path.expanduser("~/.local/share/opencode/opencode.db")


def parse_dt(s: str) -> datetime.datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"unrecognized time: {s}")


def ms(dt: datetime.datetime) -> int:
    return int(dt.timestamp() * 1000)


def latest_root_session(con: sqlite3.Connection, directory: str) -> str:
    row = con.execute(
        "select id from session_v2 where parent_id is null and directory=? "
        "order by time_updated desc limit 1",
        (directory,),
    ).fetchone()
    if row:
        return row[0]
    row = con.execute(
        "select id from session_v2 where parent_id is null order by time_updated desc limit 1"
    ).fetchone()
    if not row:
        sys.exit("no session found in the OpenCode DB")
    return row[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="window start, local time")
    ap.add_argument("--until", default=None, help="window end (default: now)")
    ap.add_argument("--session", default=None, help="session id (default: latest root)")
    ap.add_argument("--workspace", default="/mnt/workspace")
    args = ap.parse_args()

    start = parse_dt(args.since)
    end = parse_dt(args.until) if args.until else datetime.datetime.now()
    if not os.path.exists(DB):
        sys.exit(f"OpenCode DB not found: {DB}")

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    sid = args.session or latest_root_session(con, args.workspace)

    tot = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cost": 0.0, "msgs": 0}
    for (data,) in con.execute(
        "select data from session_message where session_id=? and type='assistant'", (sid,)
    ):
        d = json.loads(data)
        t = d.get("tokens")
        if not t:
            continue
        ts = d.get("time", {}).get("completed") or d.get("time", {}).get("created") or 0
        if ms(start) <= ts < ms(end):
            tot["input"] += t.get("input", 0)
            tot["output"] += t.get("output", 0)
            tot["reasoning"] += t.get("reasoning", 0)
            tot["cache_read"] += t.get("cache", {}).get("read", 0)
            tot["cost"] += d.get("cost", 0) or 0
            tot["msgs"] += 1

    fmt = "%Y-%m-%d %H:%M"
    # Two lines shaped for a commit message body.
    print(f"period: {start.strftime(fmt)}–{end.strftime(fmt)} (+{end.astimezone().strftime('%z')[1:3]}:00)")
    print(
        f"tokens: input={tot['input']:,} output={tot['output']:,} "
        f"reasoning={tot['reasoning']:,} cache-read={tot['cache_read']:,} "
        f"cost=${tot['cost']:.4f}"
    )
    print(f"source: session={sid} assistant_messages={tot['msgs']}", file=sys.stderr)


if __name__ == "__main__":
    main()
