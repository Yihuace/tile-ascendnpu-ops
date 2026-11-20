#!/usr/bin/env python3
"""Sync ``examples/TileOPs`` from the ops repo into this working tree and
re-apply the LeakyRelu / Hardtanh registration patch.

Why this script exists
----------------------
The ops repo (``tilelang-ascend-npu-ops``) carries the team's TileOPs content,
but as of ``d014e76c`` it still does **not** register ``LeakyReluFwdOp`` /
``HardtanhFwdOp`` — their kernel, op, workload, tests and ``perf_opt`` all
exist, yet all four registry surfaces lack them, so

    from tileops.ops.elementwise import LeakyReluFwdOp    # ImportError

A plain ``rsync`` from the ops repo overwrites the local fix.  This script
re-does both steps in one shot and is safe to re-run (idempotent).

What it does
------------
1. ``rsync -a`` (no ``--delete``) so locally added files survive:
   ``tools/`` and ``docs/op_performance_methodology.md`` live inside TileOPs.
2. Re-applies four registration edits, skipping any that are already present:
   - ``tileops/kernels/elementwise/__init__.py``   (import + __all__)
   - ``tileops/ops/elementwise/__init__.py``       (import + __all__)
   - ``tileops/workloads/elementwise.py``          (two workload classes)
   - ``tileops/manifest/elementwise.yaml``         (two manifest entries)
3. Verifies the registries import and the manifest parses.

Usage
-----
    python3 tools/sync_from_ops.py                 # sync + patch + verify
    python3 tools/sync_from_ops.py --dry-run       # show what would change
    python3 tools/sync_from_ops.py --skip-sync     # only re-apply the patch
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

DEFAULT_SOURCE = "/mnt/workspace/tilelang-ascend-npu-ops/examples/TileOPs"
EXCLUDES = ["__pycache__", "*.pyc", ".pkl_memoize_py3"]

# --- patch specs -----------------------------------------------------------
# Each entry: (anchor_line_inclusive, lines_to_insert_after_anchor, marker)
# ``marker`` is searched case-insensitively in the file; when present the whole
# file patch is skipped (idempotency).

KERNELS_INIT = [
    (
        "from tileops.kernels.elementwise.hardswish import HardswishFwdKernel",
        ["from tileops.kernels.elementwise.hardtanh import HardtanhFwdKernel"],
        "hardtanh",
    ),
    (
        "from tileops.kernels.elementwise.isnan import IsnanFwdKernel",
        ["from tileops.kernels.elementwise.leaky_relu import LeakyReluFwdKernel"],
        "leaky_relu",
    ),
    (
        '    "HardswishFwdKernel",',
        ['    "HardtanhFwdKernel",'],
        "HardtanhFwdKernel",
    ),
    (
        '    "IsnanFwdKernel",',
        ['    "LeakyReluFwdKernel",'],
        "LeakyReluFwdKernel",
    ),
]

OPS_INIT = [
    (
        "from tileops.ops.elementwise.hardswish import HardswishFwdOp",
        ["from tileops.ops.elementwise.hardtanh import HardtanhFwdOp"],
        "hardtanh",
    ),
    (
        "from tileops.ops.elementwise.isnan import IsnanFwdOp",
        ["from tileops.ops.elementwise.leaky_relu import LeakyReluFwdOp"],
        "leaky_relu",
    ),
    (
        '    "HardswishFwdOp",',
        ['    "HardtanhFwdOp",'],
        "HardtanhFwdOp",
    ),
    (
        '    "IsnanFwdOp",',
        ['    "LeakyReluFwdOp",'],
        "LeakyReluFwdOp",
    ),
]

WORKLOAD_CLASSES = '''
class HardtanhWorkload(ShapedRandnWorkload):
    """Workload definition for HardtanhFwdOp (shape + dtype, randn input).

    The clamp bounds ``min_val`` / ``max_val`` are op parameters, not tensors,
    so the workload only generates the single input tensor.
    """


class LeakyReluWorkload(ShapedRandnWorkload):
    """Workload definition for LeakyReluFwdOp (shape + dtype, randn input).

    ``negative_slope`` is an op parameter, not a tensor, so the workload only
    generates the single input tensor.
    """


'''

MANIFEST_BLOCK = '''LeakyReluFwdOp:
  ref_api: "torch.nn.functional.leaky_relu"
  family: elementwise
  status: implemented

  torch_compile_fullgraph: true
  signature:
    inputs:
      input: {dtype: "float16 | bfloat16 | float32"}
    outputs:
      output: {dtype: "same_as(input)"}
    params:
      negative_slope: {type: float, default: 0.01}
      inplace: {type: bool, default: false}
    shape_rules:
      - "output.shape == input.shape"

  workloads:
    - {input_shape: [1048576], dtypes: [float32, float16, bfloat16], label: "smoke-1m"}
    - {input_shape: [4000000], dtypes: [float16, bfloat16], label: "full-4m"}

  roofline:
    vars:
      N: "product(input.shape)"
    # compare-and-select(1) + mul(1) = 2 per elem.
    flops: "2 * N"
    bytes: "2 * N * elem_bytes"

  source:
    kernel: tileops/kernels/elementwise/leaky_relu/leaky_relu.py
    kernel_map:
      leaky_relu: LeakyReluFwdKernel
    op: tileops/ops/elementwise/leaky_relu.py
    test: tests/ops/test_leaky_relu.py
    bench: benchmarks/ops/bench_leaky_relu.py
    bench_manifest_driven: true


HardtanhFwdOp:
  ref_api: "torch.nn.functional.hardtanh"
  family: elementwise
  status: implemented

  torch_compile_fullgraph: true
  signature:
    inputs:
      input: {dtype: "float16 | bfloat16 | float32"}
    outputs:
      output: {dtype: "same_as(input)"}
    params:
      min_val: {type: float, default: -1.0}
      max_val: {type: float, default: 1.0}
      inplace: {type: bool, default: false}
    shape_rules:
      - "output.shape == input.shape"

  workloads:
    - {input_shape: [1048576], dtypes: [float32, float16, bfloat16], label: "smoke-1m"}
    - {input_shape: [4000000], dtypes: [float16, bfloat16], label: "full-4m"}

  roofline:
    vars:
      N: "product(input.shape)"
    # a two-sided clamp collapses to 1 compare-and-select per output elem.
    flops: "N"
    bytes: "2 * N * elem_bytes"

  source:
    kernel: tileops/kernels/elementwise/hardtanh/hardtanh.py
    kernel_map:
      hardtanh: HardtanhFwdKernel
    op: tileops/ops/elementwise/hardtanh.py
    test: tests/ops/test_hardtanh.py
    bench: benchmarks/ops/bench_hardtanh.py
    bench_manifest_driven: true


'''


def apply_edits(path: str, edits: list[tuple[str, list[str], str]], dry_run: bool) -> str:
    """Insert ``lines`` after each anchor line. Skip when any marker is present."""
    with open(path) as handle:
        lines = handle.readlines()
    blob = "".join(lines)

    pending = []
    for anchor, inserts, marker in edits:
        if marker.lower() in blob.lower():
            continue
        pending.append((anchor, inserts))

    if not pending:
        return "already patched"

    for anchor, inserts in pending:
        try:
            pos = next(i for i, line in enumerate(lines) if line.strip() == anchor.strip())
        except StopIteration:
            return f"ANCHOR MISSING: {anchor!r}"
        lines[pos + 1 : pos + 1] = [f"{text}\n" for text in inserts]

    if not dry_run:
        with open(path, "w") as handle:
            handle.writelines(lines)
    return f"inserted {sum(len(ins) for _, ins in pending)} line(s)"


def apply_block(path: str, anchor: str, block: str, marker: str, dry_run: bool) -> str:
    """Insert ``block`` immediately before the line matching ``anchor``."""
    with open(path) as handle:
        lines = handle.readlines()
    if marker.lower() in "".join(lines).lower():
        return "already patched"
    try:
        pos = next(i for i, line in enumerate(lines) if line.strip() == anchor.strip())
    except StopIteration:
        return f"ANCHOR MISSING: {anchor!r}"
    lines[pos:pos] = [block]
    if not dry_run:
        with open(path, "w") as handle:
            handle.writelines(lines)
    return "inserted block"


def sync(source: str, dest: str, dry_run: bool) -> str:
    cmd = ["rsync", "-a", "--stats"]
    for pattern in EXCLUDES:
        cmd += ["--exclude", pattern]
    if dry_run:
        cmd.append("--dry-run")
    cmd += [source + "/", dest + "/"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"rsync failed: {result.stderr.strip()[:200]}"
    for line in result.stdout.splitlines():
        if "Number of regular files transferred" in line:
            return line.strip()
    return "rsync ok"


def verify(dest: str) -> list[str]:
    problems = []
    sys.path.insert(0, os.path.dirname(dest))
    try:
        from tileops.kernels.elementwise import HardtanhFwdKernel, LeakyReluFwdKernel  # noqa: F401
        from tileops.ops.elementwise import HardtanhFwdOp, LeakyReluFwdOp  # noqa: F401
        from tileops.workloads.elementwise import HardtanhWorkload, LeakyReluWorkload  # noqa: F401
    except Exception as exc:  # pragma: no cover
        problems.append(f"registry import failed: {type(exc).__name__}: {exc}")
    try:
        import yaml

        manifest = yaml.safe_load(open(os.path.join(dest, "tileops/manifest/elementwise.yaml")))
        for name in ("LeakyReluFwdOp", "HardtanhFwdOp"):
            if name not in manifest:
                problems.append(f"manifest missing {name}")
    except Exception as exc:  # pragma: no cover
        problems.append(f"manifest parse failed: {exc}")
    for rel in ("tools/op_status.py", "tools/ratio_vs_peak.py",
                "tools/calibration_910B3.json", "docs/op_performance_methodology.md"):
        if not os.path.exists(os.path.join(dest, rel)):
            problems.append(f"locally added file lost: {rel}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--dest", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-sync", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        print(f"source not found: {args.source}", file=sys.stderr)
        return 1

    if not args.skip_sync:
        print(f"sync : {sync(args.source, args.dest, args.dry_run)}")

    targets = [
        ("tileops/kernels/elementwise/__init__.py", "edits", KERNELS_INIT),
        ("tileops/ops/elementwise/__init__.py", "edits", OPS_INIT),
    ]
    for rel, _, edits in targets:
        path = os.path.join(args.dest, rel)
        print(f"patch: {rel}: {apply_edits(path, edits, args.dry_run)}")

    path = os.path.join(args.dest, "tileops/workloads/elementwise.py")
    print(f"patch: tileops/workloads/elementwise.py: "
          f"{apply_block(path, 'class ClampMinWorkload(ShapedRandnWorkload):', WORKLOAD_CLASSES, 'class HardtanhWorkload', args.dry_run)}")

    path = os.path.join(args.dest, "tileops/manifest/elementwise.yaml")
    print(f"patch: tileops/manifest/elementwise.yaml: "
          f"{apply_block(path, 'IsfiniteFwdOp:', MANIFEST_BLOCK, 'LeakyReluFwdOp:', args.dry_run)}")

    if args.dry_run:
        print("dry-run: verification skipped")
        return 0

    problems = verify(args.dest)
    if problems:
        print("\nVERIFY FAILED:")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("\nVERIFY OK: registries import, manifest parses, local files intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
