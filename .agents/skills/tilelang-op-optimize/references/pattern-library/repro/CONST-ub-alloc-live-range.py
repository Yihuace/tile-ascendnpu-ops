# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""[repro] CONST-ub-alloc-live-range -- pattern-library/constants.md
Entry ID: CONST-ub-alloc-live-range. Origin: TileOPs-_gqa_sw_fwd_varlen_kernel-20261105T011012Z (round 13).
Assertion: disjoint compiles (4th 64 KB aliased to slot 0; 3 x 64 KB = 196,608 B); concurrent fails with
  ub overflow, requires 2097152 bits while 1572864 bits available
Compile-only, no device run.
First verified: 2026-11-05, tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3. Re-verification log (append-only): 2026-11-05 PASS.
Run: python repro/CONST-ub-alloc-live-range.py {disjoint,concurrent}

Round-13 UB allocator-semantics probe (NO kernel change, compile-only).

Purpose: resolve the contradiction recorded across rounds --
  (a) round 12: "BishengIR allocates the static sum (no lifetime reuse)",
      157,952 B user allocs == the reported footprint for this kernel;
  (b) Bwd (BN=64) case: static sum 240 KB > 192 KB yet it compiled because
      "the allocator reuses by lifetime".

Design: 4 x [64,256] f32 UB buffers = 65,536 B each => static sum 262,144 B
(256 KB, 1.33x the 192 KB UB).  Two lifetime patterns of the same allocations:

  * ``disjoint``  -- each buffer is consumed and stored before the next is
                     written; a lifetime-reusing planner would need 64 KB only.
  * ``concurrent``-- all four are written before any is read (true peak 256 KB);
                     must fail if any capacity check exists.

Verdict logic
  disjoint FAIL  + concurrent FAIL  -> static sum (no reuse)
  disjoint PASS  + concurrent FAIL  -> lifetime reuse
  disjoint PASS  + concurrent PASS  -> no compile-time check at all

Run: python _gqa_sw_fwd_varlen_kernel_round13_ubprobe.py {disjoint,concurrent}
"""
import sys

import tilelang
import tilelang.language as T

M, N = 64, 256
DT = "float32"
NBUF = 4

_PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
}


@tilelang.jit(out_idx=[-1], target="npuir", pass_configs=_PASS_CFG)
def build_disjoint():
    @T.prim_func
    def main(A: T.Tensor((M, N), DT), C: T.Tensor((M, N), DT)):
        with T.Kernel(1, is_npu=True) as (kid, sid):
            with T.Scope("Vector"):
                b0 = T.alloc_ub([M, N], DT)
                b1 = T.alloc_ub([M, N], DT)
                b2 = T.alloc_ub([M, N], DT)
                b3 = T.alloc_ub([M, N], DT)
                with T.rs("PIPE_MTE2"):
                    T.copy(A, b0)
                with T.rs("PIPE_V"):
                    T.vmul(b0, 2.0, b0)
                with T.rs("PIPE_MTE3"):
                    T.copy(b0, C)
                with T.rs("PIPE_MTE2"):
                    T.copy(C, b1)
                with T.rs("PIPE_V"):
                    T.vmul(b1, 2.0, b1)
                with T.rs("PIPE_MTE3"):
                    T.copy(b1, C)
                with T.rs("PIPE_MTE2"):
                    T.copy(C, b2)
                with T.rs("PIPE_V"):
                    T.vmul(b2, 2.0, b2)
                with T.rs("PIPE_MTE3"):
                    T.copy(b2, C)
                with T.rs("PIPE_MTE2"):
                    T.copy(C, b3)
                with T.rs("PIPE_V"):
                    T.vmul(b3, 2.0, b3)
                with T.rs("PIPE_MTE3"):
                    T.copy(b3, C)

    return main


@tilelang.jit(out_idx=[-1], target="npuir", pass_configs=_PASS_CFG)
def build_concurrent():
    @T.prim_func
    def main(A: T.Tensor((M, N), DT), C: T.Tensor((M, N), DT)):
        with T.Kernel(1, is_npu=True) as (kid, sid):
            with T.Scope("Vector"):
                b0 = T.alloc_ub([M, N], DT)
                b1 = T.alloc_ub([M, N], DT)
                b2 = T.alloc_ub([M, N], DT)
                b3 = T.alloc_ub([M, N], DT)
                with T.rs("PIPE_MTE2"):
                    T.copy(A, b0)
                    T.copy(A, b1)
                    T.copy(A, b2)
                    T.copy(A, b3)
                with T.rs("PIPE_V"):
                    T.vadd(b0, b1, b2)
                    T.vadd(b2, b3, b0)
                with T.rs("PIPE_MTE3"):
                    T.copy(b0, C)

    return main


if __name__ == "__main__":
    # no-arg default = "disjoint" (the passing variant), so the ED-D repro suite
    # (`repro_runner.py`, which invokes repros without arguments) can execute it.
    which = sys.argv[1] if len(sys.argv) > 1 else "disjoint"
    if which == "disjoint":
        build_disjoint()
    elif which == "concurrent":
        build_concurrent()
    else:
        raise SystemExit(f"unknown variant {which}")
    print(f"UBPROBE_{which.upper()}_COMPILE_OK")
