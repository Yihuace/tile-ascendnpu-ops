"""[repro] TRAP-threads-kwarg-noop -- pattern-library/traps-compiler.md

Phenomenon: `T.Kernel(..., threads=)` is a no-op on the npuir target. The NPU
launch branch in src/ir.cc (KernelLaunch, guarded by is_npu_kernel_frame)
only consumes grid_size and binds cid/vid/bx/by from blockIdx -- it never
generates any threadIdx binding for the threads kwarg (the Python signature
still accepts and forwards it). GPU-source migrations should simply drop
`threads=` and fold threads/npt into a single block_size passed via wrapper
args.

Assertion (trap, source-semantics check): within src/ir.cc, the NPU kernel
launch branch (from `is_npu_kernel_frame` to the `Launch GPU Kernel` else
branch) contains no threadIdx/thread binding, and the grid-derived bindings
(cid/vid/bx/by via blockIdx) are present.

First verified: tilelang-mlir-dev dev root build 2026-09-07 source read
(evidence: examples/lerp_tensor/_make_lerp_tensor_kernel/REVIEW.md note #28).
origin_task (provenance, may rot): lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z
Re-verification log (append-only):
  2026-09-10 PASS (formalized from the documented grep; repo src/ir.cc).
"""

import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 6)))
SRC = os.path.join(REPO_ROOT, "src", "ir.cc")


def main():
    if not os.path.isfile(SRC):
        print(f"SKIP: {SRC} not found (run inside the tilelang-mlir-dev repo)")
        sys.exit(0)
    with open(SRC, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"is_npu_kernel_frame\) \{.*?// Launch GPU Kernel", text, re.S)
    assert m, "NPU kernel launch branch not found in src/ir.cc"
    branch = m.group(0)
    assert "threadIdx" not in branch and "thread_idx" not in branch, (
        "NPU launch branch must not bind threadIdx (threads= kwarg is a no-op)"
    )
    for kw in ("blockIdx.x", "blockIdx.y", "blockIdx.z"):
        assert kw in branch, f"grid-derived binding {kw} missing from NPU branch"
    print(
        "TRAP-threads-kwarg-noop: PASS "
        "(NPU launch branch consumes grid only, no threadIdx binding)"
    )


if __name__ == "__main__":
    main()
