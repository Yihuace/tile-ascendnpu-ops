#!/usr/bin/env python3
"""Report drift between the delivery tree (single source of truth) and a dev tree.

Doctrine
--------
``tilelang-ascend-npu-ops`` is the **single source of truth**: an optimisation is
only "done" when (a) the kernel lives there, (b) ``pytest tests/ops`` passes
there, and (c) the official ``tileops-report`` run is produced from there.  A dev
tree (e.g. ``opdev-repo``) may run ahead; whatever it has is *unmerged work*, not
delivered work.  This script makes that gap explicit so it cannot be lost again.

Background: on 2026-11-04 the GELU family was optimised to 101.9 us / 85.94 %
(block_N=16384 + tanh chain) but only in the dev tree; the delivery tree still
carried the fp32 polynomial erf with block_N=2048 (494 us / 46.89 %).  Re-doing
the work from the delivery tree reproduced 57 % before the recipe was recovered.

Usage
-----
    python3 tools/diff_dev_vs_delivery.py \
        --dev ~/opdev-repo/examples/TileOPs \
        --delivery . \
        --json /tmp/drift.json

Exit code is 1 when drift is found (usable as a pre-delivery gate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

KERNEL_GLOBS = ("perf_opt",)


def _sha(path: str) -> str | None:
    try:
        with open(path, "rb") as fh:
            return hashlib.sha1(fh.read()).hexdigest()
    except OSError:
        return None


def _walk(root: str):
    """Yield paths (relative to *root*) of python files under any perf_opt dir."""
    for base, _dirs, files in os.walk(root):
        if not any(g in base.split(os.sep) for g in KERNEL_GLOBS):
            continue
        for name in files:
            if name.endswith(".py"):
                full = os.path.join(base, name)
                yield os.path.relpath(full, root)


_HINTS = (
    "compute_dtype",
    "block_N",
    "block_size",
    "vtanh",
    "tanh",
    "erf",
    "Pipelined",
    "num_cores",
)
_WORDY = {"erf", "tanh", "vtanh"}  # match as whole words only ("perf_opt" is not erf)


def _hint(path: str) -> str:
    """Short fingerprint of the tuning-relevant tokens in a kernel file."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
    except OSError:
        return ""
    import re

    found = []
    for token in _HINTS:
        import re as _re

        pattern = rf"\b{token}\b" if token in _WORDY else token
        if _re.search(pattern, text):
            found.append(token)
    m = re.search(r"block_(?:N|size)\s*=\s*(\d+)", text)
    if m:
        found.append("bn=" + m.group(1))
    return ",".join(found[:6])


def compare(dev: str, delivery: str) -> dict:
    """Compare kernel files.  Direction is deliberately NOT inferred from mtime:
    a dev tree can be an older snapshot, and local edits make delivery files look
    "newer" even when they carry the un-optimised implementation.  Every differing
    file needs a content/measurement decision."""
    dev_files = set(_walk(dev))
    del_files = set(_walk(delivery))
    differing = []
    for rel in sorted(dev_files & del_files):
        pd, pl = os.path.join(dev, rel), os.path.join(delivery, rel)
        hd, hl = _sha(pd), _sha(pl)
        if hd == hl:
            continue
        differing.append(
            {
                "file": rel,
                "dev_sha": hd[:8],
                "delivery_sha": hl[:8],
                "dev_hint": _hint(pd),
                "delivery_hint": _hint(pl),
            }
        )
    return {
        "dev": os.path.abspath(dev),
        "delivery": os.path.abspath(delivery),
        "identical": len(dev_files & del_files) - len(differing),
        "differing": differing,
        "only_in_dev": sorted(dev_files - del_files),
        "only_in_delivery": sorted(del_files - dev_files),
    }


def _print(report: dict) -> None:
    print(f"dev      : {report['dev']}")
    print(f"delivery : {report['delivery']}  (single source of truth)")
    print(f"identical: {report['identical']}")
    print(f"DIFFERING: {len(report['differing'])}   (direction must be decided per file)")
    print(
        f"only in dev / only in delivery: {len(report['only_in_dev'])} / {len(report['only_in_delivery'])}"
    )
    if report["differing"]:
        print("\n--- differing kernels: <file> | dev hint | delivery hint ---")
        for e in report["differing"]:
            print(f"  {e['file']}")
            print(f"      dev      : {e['dev_hint']}")
            print(f"      delivery : {e['delivery_hint']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dev", required=True, help="dev tree root (e.g. ~/opdev-repo/examples/TileOPs)"
    )
    parser.add_argument(
        "--delivery", default=".", help="delivery tree root (single source of truth)"
    )
    parser.add_argument("--json", help="write the full drift report as JSON")
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="exit 0 even when drift is found (report-only mode)",
    )
    args = parser.parse_args(argv)

    report = compare(os.path.expanduser(args.dev), args.delivery)
    _print(report)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nreport written to {args.json}")

    drifted = report["differing"]
    if drifted and not args.allow_drift:
        print(
            "\nFAIL: kernel drift detected (see the list above).  Decide each file "
            "by content / same-session A/B -- never by mtime -- merge into the "
            "single source of truth, then re-run."
        )
        return 1
    print("\nOK: no unresolved drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
