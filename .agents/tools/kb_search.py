"""kb_search.py -- unified BM25 retrieval over the TileLang knowledge base
(E-1 / K-3).

One entry point for the experience layer (docs/ stays routed by AGENTS.md
keyword mapping as the authoritative semantic source):

  pattern-library/ topic files   (entries with front-matter: pattern/trap/
                                  case/constant, incl. cases + constants)
  bottleneck-patterns.md         (BP_* headings)
  algorithm-candidates.md        (ALG-* entries)
  capability-gaps.md             (CG-* headings, open gaps)
  evolution/queue.md             (VP-* proposals -- pending knowledge counts)

Pure-python BM25 (no external dependencies). CJK text is tokenized as
bigrams; latin text as words. Every query and its returned entry ids are
appended to .agents/evolution/kb_search_log.jsonl so hit statistics become
measurable (E-2 automation; stats.md consumes it at distill time).

Usage:
  python3 .agents/tools/kb_search.py "attention 块宽 L1 上限"
  python3 .agents/tools/kb_search.py "fp16 精度失败 golden" --top 5 --json
  python3 .agents/tools/kb_search.py --index   # dump the entry inventory
"""

import argparse
import datetime
import glob
import json
import math
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 2)))
SOURCES = [
    (
        "pattern-library",
        os.path.join(
            REPO_ROOT,
            ".agents",
            "skills",
            "tilelang-op-optimize",
            "references",
            "pattern-library",
            "*.md",
        ),
    ),
    (
        "bottleneck-patterns",
        os.path.join(
            REPO_ROOT,
            ".agents",
            "skills",
            "tilelang-op-optimize",
            "references",
            "bottleneck-patterns.md",
        ),
    ),
    (
        "algorithm-candidates",
        os.path.join(
            REPO_ROOT,
            ".agents",
            "skills",
            "tilelang-op-design",
            "references",
            "algorithm-candidates.md",
        ),
    ),
    (
        "capability-gaps",
        os.path.join(REPO_ROOT, ".agents", "evolution", "capability-gaps.md"),
    ),
    ("queue", os.path.join(REPO_ROOT, ".agents", "evolution", "queue.md")),
]
LOG_PATH = os.path.join(REPO_ROOT, ".agents", "evolution", "kb_search_log.jsonl")

BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str) -> list:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_\-]{2,}", text)
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    tokens += [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return tokens


def parse_frontmatter_entries(path: str, source: str) -> list:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    rel = os.path.relpath(path, REPO_ROOT)
    entries = []
    parts = re.split(r"(?m)^---\s*$", text)
    for i in range(1, len(parts) - 1, 2):
        fm_raw, body = parts[i], parts[i + 1]
        fm = {}
        for ln in fm_raw.splitlines():
            m = re.match(r"^([a-z_]+):\s*(.+?)\s*$", ln)
            if m:
                fm[m.group(1)] = m.group(2)
        if not fm.get("id"):
            continue
        title = (body.strip().splitlines() or [""])[0].lstrip("# ").strip()
        entries.append(
            {
                "source": source,
                "id": fm["id"],
                "title": title[:120],
                "path": rel,
                "body": body,
                "facets": {
                    k: fm.get(k, "")
                    for k in ("kind", "family", "apis", "dtype", "status", "repro")
                },
            }
        )
    return entries


def parse_heading_entries(path: str, source: str, pattern: str) -> list:
    """Split by markdown headings matching a pattern (BP_/CG-/VP- style)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    rel = os.path.relpath(path, REPO_ROOT)
    headings = list(re.finditer(r"(?m)^(#{2,3})\s+(" + pattern + r"[^\n]*)$", text))
    entries = []
    for i, m in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[m.end() : end]
        title = m.group(2).strip()
        eid = re.match(r"([A-Za-z]+-[\d\-a-zA-Z]+)", title)
        entries.append(
            {
                "source": source,
                "id": eid.group(1) if eid else title[:40],
                "title": title[:120],
                "path": rel,
                "body": body,
                "facets": {"status": "pending" if source == "queue" else "open"},
            }
        )
    return entries


def build_index() -> list:
    entries = []
    for source, pattern in SOURCES:
        for path in sorted(glob.glob(pattern)):
            if not os.path.isfile(path):
                continue
            if source == "pattern-library":
                if os.path.basename(path) == "INDEX.md":
                    continue
                entries += parse_frontmatter_entries(path, source)
            elif source == "bottleneck-patterns":
                entries += parse_heading_entries(path, source, r"BP_")
            elif source == "algorithm-candidates":
                entries += parse_frontmatter_entries(path, source)
            elif source == "capability-gaps":
                entries += parse_heading_entries(path, source, r"CG-")
            elif source == "queue":
                entries += parse_heading_entries(path, source, r"VP-")
    for e in entries:
        e["_tokens"] = tokenize(e["id"] + " " + e["title"] + " " + e["body"])
    return entries


def bm25(query_tokens: list, entries: list, top: int) -> list:
    n = len(entries)
    if not n or not query_tokens:
        return []
    avgdl = sum(len(e["_tokens"]) for e in entries) / n
    df = {}
    for e in entries:
        for t in set(e["_tokens"]):
            df[t] = df.get(t, 0) + 1
    scored = []
    for e in entries:
        tf = {}
        for t in e["_tokens"]:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for qt in query_tokens:
            if qt not in df:
                continue
            idf = math.log(1 + (n - df[qt] + 0.5) / (df[qt] + 0.5))
            f = tf.get(qt, 0)
            score += (
                idf
                * f
                * (BM25_K1 + 1)
                / (f + BM25_K1 * (1 - BM25_B + BM25_B * len(e["_tokens"]) / avgdl))
            )
        if score > 0:
            scored.append((score, e))
    scored.sort(key=lambda x: -x[0])
    return scored[:top]


def best_snippet(entry: dict, query_tokens: list, width: int = 160) -> str:
    for ln in entry["body"].splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("---"):
            continue
        low = ln.lower()
        if any(qt in low for qt in query_tokens):
            return ln[:width]
    return (entry["body"].strip().splitlines() or [""])[0][:width]


def log_query(query: str, results: list) -> None:
    if not results:
        return
    rec = {
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "query": query,
        "returned": [r[1]["id"] for r in results],
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", nargs="*", help="search query")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--index", action="store_true", help="dump entry inventory and exit"
    )
    args = ap.parse_args()

    entries = build_index()
    if args.index:
        inv = [
            {"source": e["source"], "id": e["id"], "path": e["path"]} for e in entries
        ]
        print(
            json.dumps(inv, ensure_ascii=False, indent=2)
            if args.json
            else "\n".join(f"{e['source']:22s} {e['id']}" for e in entries)
        )
        return 0

    query = " ".join(args.query).strip()
    if not query:
        print("usage: kb_search.py <query> [--top N] [--json]")
        return 2
    results = bm25(tokenize(query), entries, args.top)
    log_query(query, results)

    if args.json:
        out = [
            {
                "id": e["id"],
                "source": e["source"],
                "title": e["title"],
                "path": e["path"],
                "facets": e["facets"],
                "score": round(s, 3),
                "snippet": best_snippet(e, tokenize(query)),
            }
            for s, e in results
        ]
        print(
            json.dumps({"query": query, "results": out}, ensure_ascii=False, indent=2)
        )
        return 0
    if not results:
        print(f"kb_search: no entries matched '{query}'")
        return 0
    print(f"kb_search '{query}' -> top {len(results)} of {len(entries)} entries")
    for rank, (s, e) in enumerate(results, 1):
        facets = " ".join(f"{k}={v}" for k, v in e["facets"].items() if v)
        print(f"\n[{rank}] {e['id']}  ({e['source']}, score={s:.2f})")
        print(f"    {e['title']}")
        if facets:
            print(f"    facets: {facets}")
        print(f"    {e['path']}")
        print(f"    {best_snippet(e, tokenize(query))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
