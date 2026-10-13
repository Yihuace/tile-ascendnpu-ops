"""kb_stale_check.py -- toolchain-staleness detection for KB entries (E-5).

"Entries are auto-downgraded to pending re-verification after toolchain
changes" was a prose-only convention; this tool makes it mechanical. It never
modifies library files -- it only reports.

Current stamp detection (best effort, override with --stamp):
  tilelang: import tilelang and read __version__ + repo HEAD if this repo is
            a dev root (git rev-parse --short HEAD)
  CANN:     $ASCEND_HOME_PATH/../version.cfg or /usr/local/Ascend/ascend-
            toolkit/latest/version.cfg (Version= line)

Comparison: an entry is "stale-candidate" when both its toolchain stamp and
the current stamp parse to a tilelang commit (7-40 hex) or version tuple and
they differ; entries whose stamp cannot be parsed are listed as "unknown"
(manual check). Only hex-commit mismatches are reported by default
(--include-unknown to also list unparsable ones).

Usage:
  python3 .agents/tools/kb_stale_check.py [--stamp "tilelang X + CANN Y"]
                                          [--include-unknown] [--json]
Exit 0 always (advisory); consumes the JSON "stale" list in conductor start /
optimizer Phase 0 as the pre-task re-verification checklist. Repro-backed
entries can be mechanically re-verified with:
  python3 .agents/tools/repro_runner.py --filter stale
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 2)))
KB_GLOBS = [
    os.path.join(
        REPO_ROOT,
        ".agents",
        "skills",
        "tilelang-op-optimize",
        "references",
        "pattern-library",
        "*.md",
    ),
    os.path.join(REPO_ROOT, ".agents", "evolution", "queue.md"),
]
COMMIT_RE = re.compile(r"\b([0-9a-f]{7,40})\b")
VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")


def detect_current_stamp() -> dict:
    stamp = {"tilelang": None, "tilelang_commit": None, "cann": None, "raw": ""}
    try:
        import io
        import contextlib

        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            import tilelang  # noqa: PLC0415
        stamp["tilelang"] = getattr(tilelang, "__version__", None)
    except Exception:  # noqa: BLE001 -- advisory tool
        pass
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short=10", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if head.returncode == 0:
            stamp["tilelang_commit"] = head.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    for cand in (
        os.environ.get("ASCEND_HOME_PATH", "").rstrip("/") + "/../version.cfg",
        "/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
    ):
        try:
            with open(cand) as f:
                text = f.read()
            m = re.search(r"Version\s*=\s*(\S+)", text)
            if m:
                stamp["cann"] = m.group(1)
                break
        except OSError:
            continue
    stamp["raw"] = "tilelang {} (commit {}) + CANN {}".format(
        stamp["tilelang"] or "?", stamp["tilelang_commit"] or "?", stamp["cann"] or "?"
    )
    return stamp


def parse_entries(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    rel = os.path.relpath(path, REPO_ROOT)
    entries = []
    parts = re.split(r"(?m)^---\s*$", text)
    for i in range(1, len(parts) - 1, 2):
        fm_raw = parts[i]
        m = re.search(r"^toolchain:\s*(.+)$", fm_raw, re.M)
        mid = re.search(r"^id:\s*(\S+)", fm_raw, re.M)
        if m and mid:
            entries.append(
                {"id": mid.group(1), "toolchain": m.group(1).strip(), "path": rel}
            )
    # queue.md VP entries carry '- toolchain_stamp:'
    for m in re.finditer(r"(?m)^## (VP-\d{4}-\d{4})\s*$([\s\S]*?)(?=^## |\Z)", text):
        mm = re.search(r"^- toolchain_stamp:\s*(.+)$", m.group(2), re.M)
        if mm:
            entries.append(
                {"id": m.group(1), "toolchain": mm.group(1).strip(), "path": rel}
            )
    return entries


def classify(entry_stamp: str, current: dict) -> str:
    e_commits = COMMIT_RE.findall(entry_stamp or "")
    if e_commits:
        cur = current.get("tilelang_commit")
        if (
            cur
            and cur not in e_commits
            and not any(c.startswith(cur) or cur.startswith(c) for c in e_commits)
        ):
            return "stale"
        return "fresh"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stamp",
        default=None,
        help="override current stamp, e.g. 'tilelang 0.1.2+abcdef123 + CANN 8.5.0'",
    )
    ap.add_argument("--include-unknown", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.stamp:
        current = {
            "raw": args.stamp,
            "tilelang_commit": (COMMIT_RE.findall(args.stamp) or [None])[0],
            "cann": None,
            "tilelang": None,
        }
    else:
        current = detect_current_stamp()

    stale, unknown = [], []
    for pattern in KB_GLOBS:
        for path in sorted(glob.glob(pattern)):
            if not os.path.isfile(path):
                continue
            for e in parse_entries(path):
                verdict = classify(e["toolchain"], current)
                if verdict == "stale":
                    stale.append({**e, "verdict": "stale"})
                elif verdict == "unknown":
                    unknown.append({**e, "verdict": "unknown"})

    report = {
        "current_stamp": current["raw"],
        "checked": "pattern-library topic files + queue.md",
        "stale_count": len(stale),
        "stale": stale,
        "unknown_count": len(unknown),
        "note": "stale = entry toolchain commit differs from current; "
        "re-verify repro-backed entries via: "
        "repro_runner.py --filter stale",
    }
    if args.include_unknown:
        report["unknown"] = unknown
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"current stamp: {current['raw']}")
        print(f"stale entries: {len(stale)}")
        for e in stale:
            print(f"  [stale] {e['id']}  ({e['path']})  stamp: {e['toolchain'][:80]}")
        print(
            f"unknown (unparsable stamp, manual check): {len(unknown)} "
            f"(--include-unknown to list)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
