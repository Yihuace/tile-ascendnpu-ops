"""Isolate why two-loop / bidirectional handshakes fail with
'multiple entry functions detected (previous: main_mix_aiv, this: main_mix_aic)'.

02 (single Cube->Vector handshake, no loops) works.
12 (Cube -> Vector -> Cube) fails.
14 (Cube loop + Vector loop, bidirectional) fails.

Variants:
  V1  no loops, bidirectional (Cube set(0)+wait(1); Vector wait(0)+set(1))
  V2  Cube loop(2) + Vector loop(2), UNIdirectional only (Cube sets, Vector waits)
  V3  no loops, unidirectional (Cube set(0); Vector wait(0))  -- control, must pass
"""

import os

os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # MUST force: .bashrc exports Developer

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

C = 16


@tilelang.jit(out_idx=[1], target="npuir")
def build_v1():
    @T.prim_func
    def main(x: T.Tensor((C, C), "float16"), out: T.Tensor((C, C), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                l1 = T.alloc_L1([C, C], "float16")
                l0 = T.alloc_L0C([C, C], "float32")
                T.copy(x, l1)
                T.gemm(l1, l1, l0, size=[C, C, C], b_transpose=True, initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(l0, out)
                    T.sync_block_set(0)
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(1)
            with T.Scope("Vector"):
                ub = T.alloc_ub((C, C), "float32")
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(0)
                    T.copy(out, ub)
                with T.rs("PIPE_MTE3"):
                    T.sync_block_set(1)

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def build_v2():
    @T.prim_func
    def main(x: T.Tensor((2, C, C), "float16"), out: T.Tensor((2, C, C), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                l1 = T.alloc_L1([C, C], "float16")
                l0 = T.alloc_L0C([C, C], "float32")
                for ck in T.serial(2):
                    T.copy(x[ck, 0:C, 0:C], l1)
                    T.gemm(l1, l1, l0, size=[C, C, C], b_transpose=True, initC=True)
                    with T.rs("PIPE_FIX"):
                        T.copy(l0, out[ck, 0:C, 0:C])
                        T.sync_block_set(0)
            with T.Scope("Vector"):
                ub = T.alloc_ub((C, C), "float32")
                for ck in T.serial(2):
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(0)
                        T.copy(out[ck, 0:C, 0:C], ub)

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def build_v3():
    @T.prim_func
    def main(x: T.Tensor((C, C), "float16"), out: T.Tensor((C, C), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                l1 = T.alloc_L1([C, C], "float16")
                l0 = T.alloc_L0C([C, C], "float32")
                T.copy(x, l1)
                T.gemm(l1, l1, l0, size=[C, C, C], b_transpose=True, initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(l0, out)
                    T.sync_block_set(0)
            with T.Scope("Vector"):
                ub = T.alloc_ub((C, C), "float32")
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(0)
                    T.copy(out, ub)

    return main


torch.manual_seed(0)
x1 = torch.randn(C, C, dtype=torch.float16).npu()
x2 = torch.randn(2, C, C, dtype=torch.float16).npu()

for name, builder, arg in (("V1 bidirectional, no loop", build_v1, x1),
                           ("V2 two loops, unidirectional", build_v2, x2),
                           ("V3 unidirectional, no loop (control)", build_v3, x1)):
    try:
        builder()(arg)
        torch.npu.synchronize()
        print(f"{name:38s} COMPILED+RAN OK")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        key = ("multiple entry functions" if "multiple entry functions" in msg
               else msg.replace("\n", " ")[:90])
        print(f"{name:38s} FAIL {key}")
