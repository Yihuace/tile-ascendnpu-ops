"""[repro] TRAP-load-nd2nz-strided -- pattern-library/traps-runtime.md

Phenomenon: T.load_nd2nz silently flat-misreads strided GM regions (e.g. a
[S, D] tile of a [B, S, H, D] tensor, whose dim-1 stride is H*D) as "flat
contiguous memory from the base address" -- no error, no warning,
deterministic wrong data. The slice form of T.copy is bit-exact on the same
strided region and is the verified workaround (CG-2026-0004; the base+size
form of T.copy was also wrong on the same strided region in the original
2026-09-07 session, diff 3.4).

Assertion (trap, workaround-pass): a strided GM tile loaded and stored back
via slice-form T.copy matches the source bit-exact (max |diff| == 0 per
head). The flat-misread hypothesis was verified pointwise in the original
session (got[i,j] == flat_q[i] * flat_k[j]); reproducing the wrong-load side
requires the historical base+size form which the current parser rejects on
indexed sources, so this repro locks the workaround only.

First verified: tilelang dev root build 2026-09-07 (HEAD 21586b5) + CANN 8.5.0
+ Ascend910B2C (attention expert migration session, debug_log D1/D2).
origin_task (provenance, may rot): multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
Re-verification log (append-only):
  2026-09-10 PASS (formalized from /tmp/opencode/probe_stridecopy.py, this toolchain).
"""

import os

import numpy as np
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

B, H, S, D = 1, 4, 64, 64
RM = 32


@tilelang.jit(out_idx=[-1], target="npuir")
def stride_copy_probe():
    @T.prim_func
    def main(
        q: T.Tensor((B, S, H, D), "float32"),
        din: T.Tensor((1,), "float32"),
        stg: T.Tensor((B, S, H, D), "float32"),
    ):
        with T.Kernel(4, is_npu=True) as (kernel_id, vid):
            with T.Scope("Vector"):
                ub = T.alloc_ub([RM, D], "float32")
                with T.rs("PIPE_MTE2"):
                    # workaround: slice form, bit-exact on strided regions
                    T.copy(q[0, 0:RM, kernel_id, 0:D], ub[0:RM, 0:D])
                with T.rs("PIPE_MTE3"):
                    # validated mix-form strided write
                    T.copy(ub[0:RM, 0:D], stg[0, 0:RM, kernel_id, 0:D])

    return main


def main():
    torch.manual_seed(3)
    f = stride_copy_probe()
    q = torch.randn(B, S, H, D, dtype=torch.float32, device="npu")
    din = torch.zeros(1, dtype=torch.float32, device="npu")
    (stg,) = f(q, din)
    torch.npu.synchronize()
    ok = True
    for hh in range(4):
        got = stg.reshape(B, S, H, D)[0, :RM, hh, :].cpu().numpy()
        exp = q[0, :RM, hh, :].cpu().numpy()
        m = float(np.abs(got - exp).max())
        ok = ok and m == 0.0
        print(f"head{hh} strided-copy max|diff|={m:.2e}")
    assert ok, "slice-form T.copy must be bit-exact on strided regions"
    print("TRAP-load-nd2nz-strided: PASS (slice-form workaround bit-exact)")


if __name__ == "__main__":
    main()
