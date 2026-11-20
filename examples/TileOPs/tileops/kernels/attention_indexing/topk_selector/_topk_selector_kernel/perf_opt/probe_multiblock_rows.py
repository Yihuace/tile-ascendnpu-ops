"""Probe: are multi-block GM writes reliable for THIS op's count kernel?

Stage 3 concluded "multi-block GM writes lose rows" from a probe of a different
kernel shape.  Before writing off row parallelism (the only remaining structural
lever -- the algorithm is 40 host launches by construction), re-measure it on the
count kernel's exact body and three grid forms:

    grid1    T.Kernel(1) + T.serial over rows        (the delivered form)
    gridM    T.Kernel(M), one block per row          (the suspect form)
    grid2    T.Kernel(2), rows strided over blocks   (grid-stride form)

The body is INLINED three times on purpose: a Python helper called from inside a
`T.prim_func` does not work (the parser consumes the loop frames incorrectly --
`'ForFrame' object is not iterable`), same family as the documented "no nested
def inside T.prim_func" rule.

Each row gets a DIFFERENT threshold, so a block that writes the wrong row's value
is visible (not just a missing row).

Run:  python probe_multiblock_rows.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

M, N = 8, 1024
TILE = N


def build_grid1():
    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def main(x: T.Tensor((M, N), "float32"),
                 thr: T.Tensor((M,), "float32"),
                 cnt: T.Tensor((M,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                row = T.alloc_ub((1, TILE), "float32")
                one = T.alloc_ub((1, TILE), "float32")
                zero = T.alloc_ub((1, TILE), "float32")
                sel = T.alloc_ub((1, TILE), "float32")
                diff = T.alloc_ub((1, TILE), "float32")
                mask = T.alloc_ub((1, TILE), "uint8")
                red = T.alloc_ub((1, 1), "float32")
                acc = T.alloc_ub((1, 1), "float32")
                ts = T.alloc_ub((1, 1), "float32")
                nts = T.alloc_ub((1, 1), "float32")
                for j in T.Parallel(TILE):
                    one[0, j] = 1.0
                    zero[0, j] = 0.0
                for r in T.serial(M):
                    ts[0, 0] = thr[r]
                    T.vmul(ts, -1.0, nts)
                    T.clear(acc)
                    T.copy(x[r, 0:TILE], row[0, 0:TILE])
                    T.vadd(row, nts[0, 0], diff)
                    T.vcmp(diff, zero, mask, "ge")
                    T.vselect(mask, one, zero, sel)
                    T.reduce(sel, red, dims=1, reduce_mode="sum")
                    T.vadd(acc, red, acc)
                    cnt[r] = acc[0, 0]
        return main
    return _f()


def build_gridM():
    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def main(x: T.Tensor((M, N), "float32"),
                 thr: T.Tensor((M,), "float32"),
                 cnt: T.Tensor((M,), "float32")):
            with T.Kernel(M, is_npu=True) as (pid, _):
                row = T.alloc_ub((1, TILE), "float32")
                one = T.alloc_ub((1, TILE), "float32")
                zero = T.alloc_ub((1, TILE), "float32")
                sel = T.alloc_ub((1, TILE), "float32")
                diff = T.alloc_ub((1, TILE), "float32")
                mask = T.alloc_ub((1, TILE), "uint8")
                red = T.alloc_ub((1, 1), "float32")
                acc = T.alloc_ub((1, 1), "float32")
                ts = T.alloc_ub((1, 1), "float32")
                nts = T.alloc_ub((1, 1), "float32")
                for j in T.Parallel(TILE):
                    one[0, j] = 1.0
                    zero[0, j] = 0.0
                ts[0, 0] = thr[pid]
                T.vmul(ts, -1.0, nts)
                T.clear(acc)
                T.copy(x[pid, 0:TILE], row[0, 0:TILE])
                T.vadd(row, nts[0, 0], diff)
                T.vcmp(diff, zero, mask, "ge")
                T.vselect(mask, one, zero, sel)
                T.reduce(sel, red, dims=1, reduce_mode="sum")
                T.vadd(acc, red, acc)
                cnt[pid] = acc[0, 0]
        return main
    return _f()


def build_grid2():
    @tilelang.jit(out_idx=[2], target="npuir")
    def _f():
        @T.prim_func
        def main(x: T.Tensor((M, N), "float32"),
                 thr: T.Tensor((M,), "float32"),
                 cnt: T.Tensor((M,), "float32")):
            with T.Kernel(2, is_npu=True) as (pid, _):
                row = T.alloc_ub((1, TILE), "float32")
                one = T.alloc_ub((1, TILE), "float32")
                zero = T.alloc_ub((1, TILE), "float32")
                sel = T.alloc_ub((1, TILE), "float32")
                diff = T.alloc_ub((1, TILE), "float32")
                mask = T.alloc_ub((1, TILE), "uint8")
                red = T.alloc_ub((1, 1), "float32")
                acc = T.alloc_ub((1, 1), "float32")
                ts = T.alloc_ub((1, 1), "float32")
                nts = T.alloc_ub((1, 1), "float32")
                for j in T.Parallel(TILE):
                    one[0, j] = 1.0
                    zero[0, j] = 0.0
                for q in T.serial(M // 2):
                    ts[0, 0] = thr[pid + q * 2]
                    T.vmul(ts, -1.0, nts)
                    T.clear(acc)
                    T.copy(x[pid + q * 2, 0:TILE], row[0, 0:TILE])
                    T.vadd(row, nts[0, 0], diff)
                    T.vcmp(diff, zero, mask, "ge")
                    T.vselect(mask, one, zero, sel)
                    T.reduce(sel, red, dims=1, reduce_mode="sum")
                    T.vadd(acc, red, acc)
                    cnt[pid + q * 2] = acc[0, 0]
        return main
    return _f()


def main():
    torch.manual_seed(0)
    base = torch.randperm(N, dtype=torch.float32) * 7.0
    x = base.unsqueeze(0).repeat(M, 1) + torch.arange(M, dtype=torch.float32).unsqueeze(1)
    q = torch.linspace(0.1, 0.9, M)
    thr = torch.quantile(base, q).float()
    want = (x >= thr.unsqueeze(1)).sum(dim=1).float()
    print(f"want = {want.tolist()}", flush=True)

    for name, build in (("grid1", build_grid1), ("gridM", build_gridM), ("grid2", build_grid2)):
        print(f"--- {name} ---", flush=True)
        try:
            ker = build()
            got = ker(x.npu(), thr.npu()).cpu()
        except Exception as exc:  # noqa: BLE001
            print(f"  BUILD/RUN FAILED: {type(exc).__name__}: {str(exc).splitlines()[0]}", flush=True)
            continue
        ok = torch.equal(got, want)
        print(f"  got  = {got.tolist()}", flush=True)
        print(f"  {'PASS' if ok else 'FAIL'}", flush=True)

    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
