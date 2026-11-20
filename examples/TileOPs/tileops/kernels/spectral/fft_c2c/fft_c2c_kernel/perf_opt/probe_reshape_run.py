"""Determine T.reshape(src,dst) semantics on npuir shared buffers.

Input x = [0..7].  A <- x ; T.reshape(A,B) with B (2,4) ; B += 1 ; read A back.
  A == x+1  -> alias (free reshape)
  A == x    -> copy  (dst is separate storage)
"""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

N = 8


@tilelang.jit(out_idx=[1], target="npuir")
def build():
    @T.prim_func
    def main(x: T.Tensor((N,), "float32"),
             y: T.Tensor((N,), "float32")) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            A = T.alloc_shared((N,), "float32")
            B = T.alloc_shared((N // 2, 2), "float32")
            T.copy(x[0:N], A)
            T.reshape(A, B)
            for i, j in T.Parallel(N // 2, 2):
                B[i, j] = B[i, j] + 1.0
            T.copy(A, y[0:N])
    return main


k = build()
x = torch.arange(N, dtype=torch.float32)
out = k(x.npu())
print("num outs:", len(out))
y = out[0].cpu()
print("y.shape:", tuple(y.shape))
print("input :", x.tolist())
print("out_A :", y.tolist())
print("ALIAS" if torch.allclose(y, x + 1) else
      ("COPY" if torch.allclose(y, x) else "OTHER"))
