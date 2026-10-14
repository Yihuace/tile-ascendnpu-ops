# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Repro: TRAP-sync-self-set-wait  (pattern-library/traps-runtime.md)

Phenomenon: In an Expert MixCV depth-2 handshake, a "slot free" prologue ``sync_block_set``
  placed on the PRODUCER's own engine (Cube) and waited on by that same engine deadlocks:
  the kernel never retires and the device reports "aicore execution times out" (acl 507014).
  ``sync_block_set/wait`` is an inter-engine (AIC<->AIV) event; a set is delivered to the
  OTHER engine, not looped back, so a same-core self set->wait never retires.
Fix: run the free-slot prologue on the CONSUMER engine (Vector) so the producer (Cube) wait
  crosses engines (mirrors PL-1.12 in _ssd_chunk_scan_fwd_kernel.py).
Assertion: the workaround (consumer-side prologue) runs and matches the CPU reference; the
  failing producer-side variant is left commented below and NOT executed (it hangs).
First verified: 2026-10-14 conv1d _conv1d_pointwise_kernel Stage 3 (design v1 deadlocked).
Toolchain: tilelang 0.1.2+417dae99 + Ascend910B3 (get_aicore_num=20) + CANN npu-smi 25.5.0
Run: python repro/TRAP-sync-self-set-wait.py
"""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T
from tilelang.utils import NPUUtils

DTYPE, ACCUM = "float16", "float32"
BM, BN, BK = 64, 128, 128
M, N, K = 384, 1280, 16
SLOTS = 2
NK = NPUUtils.get().get_aicore_num()
GX, GY = N // BN, M // BM
NUM_LOGICAL = GY * GX  # 60 -> 3 tasks/core, exercises slot reuse (task_id % 2)

@tilelang.jit(out_idx=[2], target="npuir")
def build():
    @T.prim_func
    def main(
        A: T.Tensor((M, K), DTYPE),
        B: T.Tensor((K, N), DTYPE),
        out: T.Tensor((M, N), ACCUM),
        ws: T.Tensor((NK, SLOTS, BM, BN), ACCUM),
    ):
        with T.Kernel(NK, is_npu=True) as (kernel_id, _):
            num_local_tasks = T.ceildiv(NUM_LOGICAL - kernel_id, NK)
            with T.Scope("Cube"):
                l1_a = T.alloc_L1([BM, BK], DTYPE)
                l1_b = T.alloc_L1([BK, BN], DTYPE)
                l0_c = T.alloc_L0C([BM, BN], ACCUM)
                # WRONG variant (deadlocks): prologue set(1)/set(3) would sit here on
                # the Cube (producer) engine == same-core self set->wait.
                for task_id in T.serial(num_local_tasks):
                    cid = T.min(task_id * NK + kernel_id, NUM_LOGICAL - 1)
                    by, bx = cid // GX, cid % GX
                    with T.rs("PIPE_MTE2"):
                        T.copy(A[by * BM : by * BM + BM, 0:K], l1_a[0:BM, 0:K])
                        T.copy(B[0:K, bx * BN : bx * BN + BN], l1_b[0:K, 0:BN])
                    with T.rs("PIPE_C"):
                        T.gemm(l1_a, l1_b, l0_c, initC=True, size=[BM, K, BN])
                    slot = task_id % SLOTS
                    with T.rs("PIPE_FIX"):
                        T.sync_block_wait(2 * slot + 1)  # slot free (WAR)
                        T.copy(l0_c, ws[kernel_id, slot, 0:BM, 0:BN])
                        T.sync_block_set(2 * slot)  # ws ready
            with T.Scope("Vector"):
                ub = T.alloc_ub([BM, BN], ACCUM)
                # CORRECT: free-slot prologue on the CONSUMER (Vector) engine.
                with T.rs("PIPE_MTE2"):
                    T.sync_block_set(1)
                    T.sync_block_set(3)
                for task_id in T.serial(num_local_tasks):
                    cid = T.min(task_id * NK + kernel_id, NUM_LOGICAL - 1)
                    by, bx = cid // GX, cid % GX
                    slot = task_id % SLOTS
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(2 * slot)  # ws ready
                        T.copy(ws[kernel_id, slot, 0:BM, 0:BN], ub)
                    with T.rs("PIPE_MTE3"):
                        T.copy(ub, out[by * BM : by * BM + BM, bx * BN : bx * BN + BN])
                        T.sync_block_set(2 * slot + 1)  # slot free

    return main

def run():
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16)
    b = torch.randn(K, N, dtype=torch.float16)
    ref = a.float() @ b.float()
    ws = torch.empty((NK, SLOTS, BM, BN), dtype=torch.float32, device="npu")
    out = build()(a.to("npu"), b.to("npu"), ws)
    torch.npu.synchronize()
    assert out.cpu().shape == (M, N)
    max_abs = float((out.cpu() - ref).abs().max().item())
    print(f"max_abs_err = {max_abs:.4e} (tol 5e-2)")
    assert max_abs < 5e-2, f"handshake wrong: {max_abs}"
    print("[TRAP-sync-self-set-wait] PASS: consumer-side prologue handshake correct")


if __name__ == "__main__":
    run()
