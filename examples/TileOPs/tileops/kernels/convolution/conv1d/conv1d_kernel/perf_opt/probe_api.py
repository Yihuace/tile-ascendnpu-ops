#!/usr/bin/env python3
"""API probes for the general Conv1d perf-opt round (all in one NPU process).

P1  gemm with K = kernel_l (7 / 10): does npuir accept a non-16-multiple K in
    ``size=[M, K, N]`` and does it produce the right numbers?  If yes, the whole
    tap-packed K axis needs no zero padding.
P2  ``T.clear`` on an Expert-mode Vector UB buffer: does it lower, and does it
    actually zero the buffer (so a padded weight tile can be zero-filled
    in-kernel instead of relying on host-zeroed GM)?
P3  ``b_transpose=True`` with B stored as [N, K] (contiguous GM run): does the
    Cube read it correctly?  If yes, a GM im2col stored as [out_l, K] can be
    fetched with ONE contiguous DMA instead of K row-strided descriptors.

Run:  python3 probe_api.py            (prints PROBE lines + PASS/FAIL summary)
"""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

DT = "float16"
ACC = "float32"


def _tol(dtype):
    return (1e-2, 1e-2) if dtype == "float16" else (1.6e-2, 1.6e-2)


def gemm_k(M, K, N, kcl):
    """A[M,K] @ B[K,N] -> C[M,N], L1 tiles sized [M,kcl] / [kcl,N]."""

    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def main(
            A: T.Tensor((M, kcl), DT),
            B: T.Tensor((kcl, N), DT),
            C: T.Tensor((M, N), DT),
        ):
            with T.Kernel(1, is_npu=True) as (kid, _):
                with T.Scope("Cube"):
                    l1a = T.alloc_L1([M, kcl], DT)
                    l1b = T.alloc_L1([kcl, N], DT)
                    l0c = T.alloc_L0C([M, N], ACC)
                    T.copy(A[0:M, 0:kcl], l1a[0:M, 0:kcl])
                    T.copy(B[0:kcl, 0:N], l1b[0:kcl, 0:N])
                    T.gemm(
                        l1a, l1b, l0c,
                        initC=True, b_transpose=False, size=[M, K, N],
                    )
                    T.copy(l0c[0:M, 0:N], C[0:M, 0:N])

        return main

    return _f


def gemm_bt(M, K, N, kcl):
    """A[M,K] @ B^T with B stored [N,K] (b_transpose=True)."""

    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def main(
            A: T.Tensor((M, kcl), DT),
            B: T.Tensor((N, kcl), DT),
            C: T.Tensor((M, N), DT),
        ):
            with T.Kernel(1, is_npu=True) as (kid, _):
                with T.Scope("Cube"):
                    l1a = T.alloc_L1([M, kcl], DT)
                    l1b = T.alloc_L1([N, kcl], DT)
                    l0c = T.alloc_L0C([M, N], ACC)
                    T.copy(A[0:M, 0:kcl], l1a[0:M, 0:kcl])
                    T.copy(B[0:N, 0:kcl], l1b[0:N, 0:kcl])
                    T.gemm(
                        l1a, l1b, l0c,
                        initC=True, b_transpose=True, size=[M, K, N],
                    )
                    T.copy(l0c[0:M, 0:N], C[0:M, 0:N])

        return main

    return _f


def gemm_mslice(M, K, N, kcl, n_m):
    """A resident as one wide L1 tile [n_m*M, kcl]; gemm on an M-offset slice."""

    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def main(
            A: T.Tensor((n_m * M, kcl), DT),
            B: T.Tensor((kcl, N), DT),
            C: T.Tensor((M, N), DT),
        ):
            with T.Kernel(1, is_npu=True) as (kid, _):
                with T.Scope("Cube"):
                    l1a = T.alloc_L1([n_m * M, kcl], DT)
                    l1b = T.alloc_L1([kcl, N], DT)
                    l0c = T.alloc_L0C([M, N], ACC)
                    T.copy(A[0:n_m * M, 0:kcl], l1a[0:n_m * M, 0:kcl])
                    T.copy(B[0:kcl, 0:N], l1b[0:kcl, 0:N])
                    T.gemm(
                        l1a[M:2 * M, 0:kcl], l1b, l0c,
                        initC=True, b_transpose=False, size=[M, K, N],
                    )
                    T.copy(l0c[0:M, 0:N], C[0:M, 0:N])

        return main

    return _f


def run_mslice(name, M, K, N, kcl, n_m):
    torch.manual_seed(0)
    with torch.no_grad():
        a = torch.randn(n_m * M, kcl, dtype=torch.float16, device="npu")
        a[:, K:] = 0
        b = torch.randn(kcl, N, dtype=torch.float16, device="npu")
        b[K:, :] = 0
        ref = a[M:2 * M].cpu().float() @ b.cpu().float()
    try:
        k = gemm_mslice(M, K, N, kcl, n_m)()
        torch.npu.synchronize()
        err = float((k(a, b).cpu().float() - ref).abs().max().item())
        ok = err < 1.0
        print(f"PROBE {name} K={K} kcl={kcl}: {'PASS' if ok else 'FAIL'} "
              f"max_abs={err:.4e}")
        return ok
    except Exception as exc:  # noqa: BLE001
        print(f"PROBE {name} K={K} kcl={kcl}: EXC {type(exc).__name__}: "
              f"{str(exc)[:300]}")
        return False


def clear_probe(M, K):
    """T.clear on a Vector UB tile -> copy out; expect all zeros."""

    @tilelang.jit(out_idx=[0], target="npuir")
    def _f():
        @T.prim_func
        def main(C: T.Tensor((M, K), DT)):
            with T.Kernel(1, is_npu=True) as (kid, _):
                with T.Scope("Vector"):
                    ub = T.alloc_ub([M, K], DT)
                    T.clear(ub)
                    T.copy(ub[0:M, 0:K], C[0:M, 0:K])

        return main

    return _f


def run_gemm(name, fn_builder, M, K, N, kcl, bt=False):
    torch.manual_seed(0)
    with torch.no_grad():
        a = torch.randn(M, kcl, dtype=torch.float16, device="npu")
        a[:, K:] = 0
        if bt:
            b = torch.randn(N, kcl, dtype=torch.float16, device="npu")
            b[:, K:] = 0
            ref = (a.cpu().float() @ b.cpu().float().T)
        else:
            b = torch.randn(kcl, N, dtype=torch.float16, device="npu")
            b[K:, :] = 0
            ref = (a.cpu().float() @ b.cpu().float())
    try:
        k = fn_builder(M, K, N, kcl)()
        torch.npu.synchronize()
        c = k(a, b).cpu().float()
        err = float((c - ref).abs().max().item())
        ok = err < 1.0  # fp16 accumulate-free reference, values ~ sqrt(K)*1
        print(f"PROBE {name} K={K} kcl={kcl}: "
              f"{'PASS' if ok else 'FAIL'} max_abs={err:.4e}")
        return ok
    except Exception as exc:  # noqa: BLE001
        print(f"PROBE {name} K={K} kcl={kcl}: EXC {type(exc).__name__}: "
              f"{str(exc)[:300]}")
        return False


def main():
    import sys

    M, N = 64, 256
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    probes = {
        "k7": lambda: run_gemm("gemm", gemm_k, M, 7, N, 7),
        "k10": lambda: run_gemm("gemm", gemm_k, M, 10, N, 10),
        "k16pad7": lambda: run_gemm("gemm", gemm_k, M, 7, N, 16),
        "k16pad10": lambda: run_gemm("gemm", gemm_k, M, 10, N, 16),
        "bt_k16": lambda: run_gemm("gemm_bt", gemm_bt, M, 16, N, 16, bt=True),
        "mslice16": lambda: run_mslice("gemm_mslice", M, 16, N, 16, 4),
        "clear": probe_clear,
        "taprow7x512": lambda: run_taprow("taprow", M, 7, 512),
        "taprow10x128": lambda: run_taprow("taprow", M, 10, 128),
        "taprow8x512": lambda: run_taprow("taprow", M, 8, 512),
        "taprow11x512": lambda: run_taprow("taprow", M, 11, 512),
    }
    names = list(probes) if which == "all" else [which]
    results = {}
    for name in names:
        try:
            results[name] = bool(probes[name]())
        except Exception as exc:  # noqa: BLE001
            print(f"PROBE {name}: EXC {type(exc).__name__}: {str(exc)[:200]}")
            results[name] = False
    print("PROBE SUMMARY " + " ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                     for k, v in results.items()))


def taprow_probe(M, K, N, L):
    """Per-tap row packing: copy a 1-row GM slice into row kw of an L1 B tile."""

    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def main(
            A: T.Tensor((M, K), DT),
            S: T.Tensor((1, L), DT),
            C: T.Tensor((M, N), DT),
        ):
            with T.Kernel(1, is_npu=True) as (kid, _):
                with T.Scope("Cube"):
                    l1a = T.alloc_L1([M, K], DT)
                    l1b = T.alloc_L1([K, N], DT)
                    l0c = T.alloc_L0C([M, N], ACC)
                    T.copy(A[0:M, 0:K], l1a[0:M, 0:K])
                    for kw in T.serial(K):
                        T.copy(S[0:1, kw:kw + N], l1b[kw:kw + 1, 0:N])
                    T.gemm(
                        l1a, l1b, l0c,
                        initC=True, b_transpose=False, size=[M, K, N],
                    )
                    T.copy(l0c[0:M, 0:N], C[0:M, 0:N])

        return main

    return _f


def run_taprow(name, M, K, N):
    L = N + K - 1
    torch.manual_seed(0)
    with torch.no_grad():
        a = torch.randn(M, K, dtype=torch.float16, device="npu")
        s = torch.randn(1, L, dtype=torch.float16, device="npu")
        bm = torch.stack([s[0, kw:kw + N] for kw in range(K)], dim=0)
        ref = a.cpu().float() @ bm.cpu().float()
    try:
        k = taprow_probe(M, K, N, L)()
        torch.npu.synchronize()
        err = float((k(a, s).cpu().float() - ref).abs().max().item())
        ok = err < 1.0
        print(f"PROBE {name} K={K} N={N}: {'PASS' if ok else 'FAIL'} "
              f"max_abs={err:.4e}")
        return ok
    except Exception as exc:  # noqa: BLE001
        print(f"PROBE {name} K={K} N={N}: EXC {type(exc).__name__}: "
              f"{str(exc)[:300]}")
        return False


def probe_clear():
    with torch.no_grad():
        out = clear_probe(64, 16)()
        torch.npu.synchronize()
    mx = float(out.abs().max().item())
    ok = mx == 0.0
    print(f"PROBE clear: {'PASS' if ok else 'FAIL'} max_abs={mx:.4e}")
    return ok


if __name__ == "__main__":
    main()
