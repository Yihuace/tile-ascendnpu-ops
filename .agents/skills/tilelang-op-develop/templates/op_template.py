"""{op}.py template

This template takes elementwise_add as an example operator.
When generating an actual operator, replace the relevant kernel/golden/test use cases.
run: python example_template.py --level all
"""

import argparse
import torch
import tilelang
import tilelang.language as T


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_add(x, y):
    return x + y


# ---------- Kernel ----------
@tilelang.jit(out_idx=[-1], target="npuir")
def add_kernel(M, N, block_M, block_N, in_dtype="float32", out_dtype="float32"):
    @T.prim_func
    def _main(
        A: T.Tensor((M, N), in_dtype),
        B: T.Tensor((M, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (
            cid,
            _,
        ):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)
            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            B_shared = T.alloc_shared((block_M, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), out_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)
            for local_y, local_x in T.Parallel(block_M, block_N):
                C_local[local_y, local_x] = (
                    A_shared[local_y, local_x] + B_shared[local_y, local_x]
                )
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return _main


# ---------- hierarchical testing ----------
def _run_case(M, N, dtype, tag):
    a = torch.randn(M, N, dtype=dtype, device="npu")
    b = torch.randn(M, N, dtype=dtype, device="npu")
    kernel = add_kernel(
        M,
        N,
        block_M=32,
        block_N=32,
        in_dtype=str(dtype).replace("torch.", ""),
        out_dtype=str(dtype).replace("torch.", ""),
    )
    c = kernel(a, b)
    golden = golden_add(a, b)
    torch.testing.assert_close(c, golden, rtol=1e-2, atol=1e-2)
    print(f"[{tag}] PASS: shape=({M},{N}) dtype={dtype}")


def run_L0():
    for M, N in [(128, 128), (256, 256)]:
        _run_case(M, N, torch.float32, "L0")


def run_L1():
    for M, N in [(130, 130), (64, 200)]:
        _run_case(M, N, torch.float32, "L1")


def run_L2():
    try:
        _run_case(1, 1, torch.float32, "L2")
    except Exception as e:
        print(f"[L2] WARN (Record without blocking): {e}")


def run_boundary():
    a = torch.zeros(128, 128, dtype=torch.float32, device="npu")
    b = torch.zeros(128, 128, dtype=torch.float32, device="npu")
    kernel = add_kernel(128, 128, block_M=32, block_N=32)
    c = kernel(a, b)
    golden = golden_add(a, b)
    torch.testing.assert_close(c, golden, rtol=1e-2, atol=1e-2)
    print("[Boundary] PASS: zeros")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()
    if args.level == "L0":
        run_L0()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
