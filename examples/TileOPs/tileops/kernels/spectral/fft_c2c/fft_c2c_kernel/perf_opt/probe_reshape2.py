import os
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa
N = 16
H = N // 2


@tilelang.jit(out_idx=[1], target="npuir")
def k_r2():
    @T.prim_func
    def main(x: T.Tensor((N,), "float32"), y: T.Tensor((N,), "float32")) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            A = T.alloc_shared((1, N), "float32")
            Bc = T.alloc_shared((2, 1, H), "float32")
            A2 = T.alloc_shared((2, H), "float32")
            B3 = T.alloc_shared((N,), "float32")
            T.copy(x[0:N], A[0, 0:N])
            T.reshape(A, A2)
            for g, i in T.Parallel(1, H):
                Bc[0, g, i] = A2[0, i] + A2[1, i]
            for g, i in T.Parallel(1, H):
                Bc[1, g, i] = A2[0, i] - A2[1, i]
            T.reshape(Bc, B3)
            T.copy(B3, y[0:N])
    return main


@tilelang.jit(out_idx=[1], target="npuir")
def k_r3():
    """no reshape: direct 2-D alloc of the source, 3-D dest."""
    @T.prim_func
    def main(x: T.Tensor((N,), "float32"), y: T.Tensor((N,), "float32")) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            A2 = T.alloc_shared((2, H), "float32")
            Bc = T.alloc_shared((2, 1, H), "float32")
            B3 = T.alloc_shared((N,), "float32")
            T.copy(x[0:N], A2[0, 0:H])
            T.copy(x[H:N], A2[1, 0:H])
            for g, i in T.Parallel(1, H):
                Bc[0, g, i] = A2[0, i] + A2[1, i]
            for g, i in T.Parallel(1, H):
                Bc[1, g, i] = A2[0, i] - A2[1, i]
            T.reshape(Bc, B3)
            T.copy(B3, y[0:N])
    return main


x = torch.arange(N, dtype=torch.float32).npu()
for name, f in (("r2", k_r2), ("r3", k_r3)):
    try:
        y = f()(x); torch.npu.synchronize()
        print(name, y.cpu().tolist())
    except Exception as e:
        print(name, "ERR", str(e)[:300])
xc = torch.arange(N, dtype=torch.float32)
print("ref", torch.cat([xc[:H] + xc[H:], xc[:H] - xc[H:]]).tolist())
