"""kb_lint.py -- mechanical lint for the TileLang knowledge base (K-4 + ED-F).

Independent, reusable checks for every KB write path (optimizer in-task
write-back, evolver merge, and PR review -- wired into tilelang-review-skill
at the same level as standards_check.py). Checks:

  pattern-library/ topic files (front-matter entries):
    KB-FM-REQUIRED    entry must carry id / kind / status / origin_task /
                      toolchain / repro front-matter fields (K-1 schema)
    KB-FM-ENUM        kind in {pattern, trap, case, constant};
                      status in {verified, stale, overturned}
    KB-FM-ID-UNIQUE   entry ids unique across the library
    KB-INDEX-COVERED  every entry id appears in INDEX.md routing table
    KB-BUDGET         byte budget: INDEX <= 12KB, topic <= 16KB (WARNING --
                      overrun triggers consolidate at next distill, merge
                      itself is not blocked)
    KB-REPRO-EXISTS   repro field (unless repro-missing/none) resolves to a
                      file under pattern-library/repro/ (ED-F i)
    KB-REPRO-HEADER   repro file header (docstring) contains the entry id and
                      a version stamp line (ED-F ii)
    KB-REPRO-COMPILE  repro file passes py_compile (ED-F iii)
    KB-REPRO-DECOUPLE repro file must not reference task-workspace paths like
                      examples/{project}/ (ED-F iv; provenance in headers is
                      exempt)
    KB-PROC-REF       entry lines referencing UNTRACKED (process-workspace)
                      examples/ paths must carry a provenance marker
                      (溯源/provenance/session-local/工作区/归档) or point to
                      the knowledge-domain repro -- process files do not
                      merge to main, so unmarked load-bearing references
                      silently rot (WARNING, ED-A)
    KB-IMPERATIVE     entry text free of agent-facing imperative instructions
                      (narrow heuristic, WARNING -- ED-F port of the evolver
                      D2 anti-injection check)

  queue.md:
    KB-QUEUE-SCHEMA   proposal entries carry well-formed id / type / status /
                      confirmations fields
    KB-QUEUE-REPRO    PENDING entries' repro field must be a knowledge-domain
                      repro path, a reproduction CONDITION, or repro-missing
                      -- NOT a command into task workspaces (examples/...,
                      ./relative, /tmp/...) which die with the workspace
                      (ED-A; Decided archives are exempt and immutable)

Provenance exemption (ED-A / ED-F v): origin_task and prose path references
inside entries are NOT existence-checked (process files are allowed to die);
only repro paths are checked.

Usage:
  python3 .agents/tools/kb_lint.py                 # full library
  python3 .agents/tools/kb_lint.py --files A B     # only these files
Exit code 0 = clean (warnings allowed), 1 = failures found.
"""

import argparse
import glob
import json
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 2)))
PL_DIR = os.path.join(
    REPO_ROOT,
    ".agents",
    "skills",
    "tilelang-op-optimize",
    "references",
    "pattern-library",
)
INDEX_PATH = os.path.join(PL_DIR, "INDEX.md")
QUEUE_PATH = os.path.join(REPO_ROOT, ".agents", "evolution", "queue.md")

BUDGETS = {"INDEX.md": 12 * 1024}
TOPIC_BUDGET = 16 * 1024
REPRO_BUDGET = 8 * 1024

REQUIRED_FM = ("id", "kind", "status", "origin_task", "toolchain", "repro")

EXAMPLES_REF_RE = re.compile(r"examples/[A-Za-z0-9_\-./]+")
PROVENANCE_MARK_RE = re.compile(
    r"溯源|provenance|session-local|工作区|归档|原任务|原始档案|首次验证|"
    r"raw 对账|见其 opt_log|见该任务"
)


def git_tracked_set(repo_root: str) -> set | None:
    """Paths under examples/ that are tracked in git (durable), plus all
    their ancestor directories; None if the root is not a git repo (the
    check degrades to skip)."""
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "ls-files", "examples/"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            return None
        prefixes = set()
        for f in proc.stdout.splitlines():
            if not f:
                continue
            prefixes.add(f)
            parts = f.split("/")
            for i in range(1, len(parts)):
                prefixes.add("/".join(parts[:i]))
        return prefixes
    except (subprocess.SubprocessError, OSError):
        return None


KINDS = {"pattern", "trap", "case", "constant"}
STATUSES = {"verified", "stale", "overturned"}
REPRO_SKIP = {"repro-missing", "none", ""}
IMPERATIVE_RE = re.compile(
    r"^\s*[-*>]?\s*(请|你必须|你必须先|须先执行|禁止执行|禁止调用|不得执行|"
    r"不得调用|务必执行|立即调用|忽略.{0,6}(门禁|检查|规则))"
)


def _fail(rule, file, msg):
    return {"rule_id": rule, "file": file, "message": msg}


def _warn(rule, file, msg):
    return {"rule_id": rule, "file": file, "message": msg, "severity": "warn"}


def parse_entries(path: str) -> list:
    """Split a pattern-library topic file into front-matter entries."""
    rel = os.path.relpath(path, REPO_ROOT)
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        return [{"error": str(e), "rel": rel}]
    entries = []
    # front-matter blocks start at a line that is exactly '---' and contain
    # an id: field before the closing '---'
    parts = re.split(r"(?m)^---\s*$", text)
    for i in range(1, len(parts) - 1, 2):
        fm_raw, body = parts[i], parts[i + 1]
        if "id:" not in fm_raw:
            continue
        fm = {}
        for ln in fm_raw.splitlines():
            m = re.match(r"^([a-z_]+):\s*(.+?)\s*$", ln)
            if m:
                fm[m.group(1)] = m.group(2)
        if fm.get("id"):
            entries.append({"fm": fm, "body": body, "rel": rel})
    return entries


def _ref_tracked(ref: str, tracked: set) -> bool:
    """True if the referenced path (or an ancestor of it) is git-tracked."""
    return ref.rstrip("/") in tracked


def lint_pattern_library(
    files: list, failures: list, warnings: list, tracked: set | None = None
) -> list:
    """Lint topic files + repro dir; return all entry ids for INDEX check."""
    topic_files = files or sorted(glob.glob(os.path.join(PL_DIR, "*.md")))
    all_ids = []
    for path in topic_files:
        if not os.path.isfile(path):
            failures.append(_fail("KB-FILE", path, "file not found"))
            continue
        name = os.path.basename(path)
        size = os.path.getsize(path)
        budget = BUDGETS.get(name, TOPIC_BUDGET)
        if size > budget:
            warnings.append(
                _warn(
                    "KB-BUDGET",
                    path,
                    f"{name} is {size} bytes > budget {budget} bytes -- "
                    "consolidate at next distill (update-first, no deletions)",
                )
            )
        for entry in parse_entries(path):
            if "error" in entry:
                failures.append(_fail("KB-FILE", entry["rel"], entry["error"]))
                continue
            fm, rel = entry["fm"], entry["rel"]
            eid = fm["id"]
            all_ids.append((eid, rel))
            missing = [f for f in REQUIRED_FM if not fm.get(f)]
            if missing:
                failures.append(
                    _fail(
                        "KB-FM-REQUIRED",
                        rel,
                        f"entry {eid}: missing front-matter fields {missing}",
                    )
                )
                continue
            if fm["kind"] not in KINDS:
                failures.append(
                    _fail(
                        "KB-FM-ENUM",
                        rel,
                        f"entry {eid}: kind '{fm['kind']}' not in {sorted(KINDS)}",
                    )
                )
            if fm["status"] not in STATUSES:
                failures.append(
                    _fail(
                        "KB-FM-ENUM",
                        rel,
                        f"entry {eid}: status '{fm['status']}' not in "
                        f"{sorted(STATUSES)}",
                    )
                )
            if re.search(
                r"工具链|版本戳缺失", fm["toolchain"]
            ) is None and not re.search(r"\d{4}|build|commit|CANN", fm["toolchain"]):
                warnings.append(
                    _warn(
                        "KB-FM-STAMP",
                        rel,
                        f"entry {eid}: toolchain stamp lacks date/build/commit "
                        f"marker: '{fm['toolchain']}'",
                    )
                )
            for m in IMPERATIVE_RE.finditer(entry["body"]):
                warnings.append(
                    _warn(
                        "KB-IMPERATIVE",
                        rel,
                        f"entry {eid}: agent-facing imperative suspected: "
                        f"{m.group(0)[:40]}... (rewrite as fact statement)",
                    )
                )
            if tracked is not None:
                for m in EXAMPLES_REF_RE.finditer(entry["body"]):
                    ref = m.group(0).rstrip(".,;:!?、。】）)")
                    line = next(
                        (ln for ln in entry["body"].splitlines() if ref in ln), ""
                    )
                    if PROVENANCE_MARK_RE.search(line):
                        continue  # marked provenance -- allowed to rot
                    if _ref_tracked(ref, tracked):
                        continue  # durable (git-tracked) example
                    warnings.append(
                        _warn(
                            "KB-PROC-REF",
                            rel,
                            f"entry {eid}: references untracked process path "
                            f"'{ref}' without a provenance marker -- mark it "
                            "(溯源/工作区/session-local) or point to the "
                            "knowledge-domain repro (ED-A decoupling)",
                        )
                    )
            repro = fm["repro"]
            if repro in REPRO_SKIP:
                continue
            repro_rel = repro.removeprefix("repro/")
            repro_path = os.path.join(PL_DIR, "repro", repro_rel)
            if not os.path.isfile(repro_path):
                failures.append(
                    _fail(
                        "KB-REPRO-EXISTS",
                        rel,
                        f"entry {eid}: repro '{repro}' not found under "
                        "pattern-library/repro/ (mark 'repro-missing' until "
                        "backfilled)",
                    )
                )
                continue
            lint_repro(repro_path, eid, failures, warnings)
    return all_ids


def lint_repro(path: str, eid: str, failures: list, warnings: list) -> None:
    rel = os.path.relpath(path, REPO_ROOT)
    size = os.path.getsize(path)
    with open(path, encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    if size > REPRO_BUDGET or n_lines > 100:
        warnings.append(
            _warn(
                "KB-BUDGET",
                rel,
                f"repro {eid}: {size} bytes / >100 lines exceeds the ED-B "
                "minimization budget (8KB / 100 lines) -- not yet minimized",
            )
        )
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
    except OSError as e:
        failures.append(_fail("KB-REPRO-HEADER", rel, str(e)))
        return
    header = src.split('"""')[1] if '"""' in src else src[:800]
    if eid not in header:
        failures.append(
            _fail(
                "KB-REPRO-HEADER",
                rel,
                f"repro header docstring must name its entry id '{eid}'",
            )
        )
    if not re.search(r"First verified|首次验证|verified:", header):
        failures.append(
            _fail(
                "KB-REPRO-HEADER",
                rel,
                f"repro {eid}: header lacks a 'First verified' version stamp line",
            )
        )
    # decoupling: repro must not read/import task workspace paths (ED-F iv);
    # provenance mentions inside the header docstring are exempt.
    code = src.split('"""', 2)[-1] if src.count('"""') >= 2 else src
    m = re.search(r"examples/[A-Za-z0-9_{}\[\]]+/", code)
    if m:
        failures.append(
            _fail(
                "KB-REPRO-DECOUPLE",
                rel,
                f"repro {eid}: code references task workspace path "
                f"'{m.group(0)}...' -- repro must be self-contained (provenance "
                "goes in the header docstring only)",
            )
        )
    try:
        import ast

        ast.parse(src)
    except SyntaxError as e:
        failures.append(
            _fail("KB-REPRO-COMPILE", rel, f"repro {eid}: syntax check failed: {e}")
        )


def lint_repro_orphans(failures: list, warnings: list) -> None:
    """repro files not referenced by any entry are orphans (warning)."""
    referenced = set()
    for path in sorted(glob.glob(os.path.join(PL_DIR, "*.md"))):
        for entry in parse_entries(path):
            if "fm" in entry and entry["fm"].get("repro"):
                referenced.add(entry["fm"]["repro"].removeprefix("repro/"))
    for path in sorted(glob.glob(os.path.join(PL_DIR, "repro", "*.py"))):
        name = os.path.basename(path)
        if name not in referenced:
            warnings.append(
                _warn(
                    "KB-REPRO-ORPHAN",
                    path,
                    f"repro file not registered by any entry front-matter "
                    f"(repro: repro/{name})",
                )
            )


def lint_index(all_ids: list, failures: list, warnings: list) -> None:
    if not os.path.isfile(INDEX_PATH):
        failures.append(_fail("KB-FILE", INDEX_PATH, "INDEX.md not found"))
        return
    with open(INDEX_PATH, encoding="utf-8") as f:
        text = f.read()
    for eid, rel in all_ids:
        if eid not in text:
            warnings.append(
                _warn(
                    "KB-INDEX-COVERED",
                    rel,
                    f"entry {eid} missing from INDEX.md routing table",
                )
            )


def lint_extra_kb_files(tracked: set | None, failures: list, warnings: list) -> None:
    """Line-level KB-PROC-REF scan for the other KB prose files whose entries
    are heading-based rather than front-matter based."""
    files = [
        os.path.join(
            REPO_ROOT,
            ".agents",
            "skills",
            "tilelang-op-optimize",
            "references",
            "bottleneck-patterns.md",
        ),
        os.path.join(REPO_ROOT, ".agents", "evolution", "capability-gaps.md"),
    ]
    for path in files:
        if not os.path.isfile(path) or tracked is None:
            continue
        with open(path, encoding="utf-8") as f:
            for ln in f.read().splitlines():
                for m in EXAMPLES_REF_RE.finditer(ln):
                    ref = m.group(0).rstrip(".,;:!?、。】）)")
                    if PROVENANCE_MARK_RE.search(ln):
                        continue
                    if _ref_tracked(ref, tracked):
                        continue
                    warnings.append(
                        _warn(
                            "KB-PROC-REF",
                            path,
                            f"references untracked process path '{ref}' "
                            "without a provenance marker -- mark it "
                            "(溯源/工作区/session-local) or point to the "
                            "knowledge-domain repro (ED-A decoupling)",
                        )
                    )
                for m in re.finditer(r"/tmp/opencode/\S+", ln):
                    if PROVENANCE_MARK_RE.search(ln):
                        continue
                    warnings.append(
                        _warn(
                            "KB-PROC-REF",
                            path,
                            f"references session-local probe "
                            f"'{m.group(0)}' without a provenance marker "
                            "(session-local/工作区) -- probes die with the "
                            "session; register a knowledge-domain repro",
                        )
                    )


def lint_queue(failures: list, warnings: list) -> None:
    if not os.path.isfile(QUEUE_PATH):
        return
    with open(QUEUE_PATH, encoding="utf-8") as f:
        text = f.read()
    # ED-A: pending entries' repro must not be a task-workspace command
    # (process files do not merge to main). Decided archives are immutable
    # (queue-schema section 4) and therefore exempt.
    pending_region = text.split("## Decided")[0]
    for m in re.finditer(
        r"(?m)^## (VP-\d{4}-\d{4})\s*$([\s\S]*?)(?=^## |\Z)", pending_region
    ):
        eid, body = m.group(1), m.group(2)
        if not re.search(r"^- status: pending\b", body, re.M):
            continue
        rm = re.search(r"^- repro: (.+)$", body, re.M)
        if not rm:
            continue
        repro_val = rm.group(1)
        inline = re.match(r"^python3? -c ", repro_val)
        if not inline and re.search(r"examples/|/tmp/|^\./|cd examples", repro_val):
            failures.append(
                _fail(
                    "KB-QUEUE-REPRO",
                    QUEUE_PATH,
                    f"{eid}: repro field is a task-workspace command "
                    f"'{repro_val[:80]}' -- process files do not merge to "
                    "main; use a knowledge-domain repro path, a "
                    "reproduction condition, or repro-missing (ED-A)",
                )
            )
    entries = re.split(r"(?m)^## (VP-\d{4}-\d{4})\s*$", text)
    for i in range(1, len(entries), 2):
        eid, body = entries[i], entries[i + 1]
        if not re.search(r"^- type: [DPRC]", body, re.M):
            failures.append(
                _fail(
                    "KB-QUEUE-SCHEMA",
                    QUEUE_PATH,
                    f"{eid}: missing '- type:' field (D/P/R/C)",
                )
            )
        if not re.search(r"^- status: \w+", body, re.M):
            failures.append(
                _fail(
                    "KB-QUEUE-SCHEMA", QUEUE_PATH, f"{eid}: missing '- status:' field"
                )
            )
        if not re.search(r"^- confirmations: (\d/2|-/-)", body, re.M):
            warnings.append(
                _warn(
                    "KB-QUEUE-SCHEMA",
                    QUEUE_PATH,
                    f"{eid}: confirmations field malformed (expect n/2 or -/-)",
                )
            )


def main() -> int:
    global REPO_ROOT, PL_DIR, INDEX_PATH, QUEUE_PATH
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="restrict to specific files (repo-relative or absolute paths)",
    )
    ap.add_argument(
        "--root", default=None, help="override repo root (tests / apply-preview checks)"
    )
    ap.add_argument("--json", action="store_true", help="JSON output")
    args = ap.parse_args()

    if args.root:
        REPO_ROOT = os.path.abspath(args.root)
        PL_DIR = os.path.join(
            REPO_ROOT,
            ".agents",
            "skills",
            "tilelang-op-optimize",
            "references",
            "pattern-library",
        )
        INDEX_PATH = os.path.join(PL_DIR, "INDEX.md")
        QUEUE_PATH = os.path.join(REPO_ROOT, ".agents", "evolution", "queue.md")

    files = [
        f if os.path.isabs(f) else os.path.join(REPO_ROOT, f)
        for f in (args.files or [])
    ]
    failures, warnings = [], []
    tracked = git_tracked_set(REPO_ROOT)
    all_ids = lint_pattern_library(files, failures, warnings, tracked)
    if not files:
        lint_extra_kb_files(tracked, failures, warnings)
        lint_repro_orphans(failures, warnings)
        lint_index(all_ids, failures, warnings)
        lint_queue(failures, warnings)

    if args.json:
        print(
            json.dumps(
                {"failures": failures, "warnings": warnings},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for f in failures:
            print(f"[FAIL] {f['rule_id']}: {f['file']}: {f['message']}")
        for w in warnings:
            print(f"[WARN] {w['rule_id']}: {w['file']}: {w['message']}")
        print(f"\nkb_lint: {len(failures)} failures, {len(warnings)} warnings")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
