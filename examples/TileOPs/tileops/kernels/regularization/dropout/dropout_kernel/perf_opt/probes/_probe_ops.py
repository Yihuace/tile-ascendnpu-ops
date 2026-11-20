#!/usr/bin/env python3
"""NPUIR v-op form probe for the dropout UB-diet candidates.

Each probe builds a minimal 1-D persistent kernel exercising one operand form
and reports compile/run status.  Used to decide which scalar/in-place forms are
legal under the current toolchain before writing a candidate.

run: python _probe_ops.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import functools  # noqa: E402

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

N = 8192
BS = 2048
CORES = 40


def build(body, block_size=BS):
    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(block_size):
        @T.prim_func
        def main(x: T.Tensor((N,), "float32"), y: T.Tensor((N,), "float32")):
            with T.Kernel(CORES, is_npu=True) as (cid, _):
                a = T.alloc_ub((block_size,), "int32")
                b = T.alloc_ub((block_size,), "int32")
                c = T.alloc_ub((block_size,), "int32")
                f = T.alloc_ub((block_size,), "float32")
                for i in T.serial(T.ceildiv(N // block_size, CORES)):
                    task = i * CORES + cid
                    if task < N // block_size:
                        off = task * block_size
                        body(a, b, c, f, off)

        return main

    return kernel(block_size)


def run(name, body, check=None):
    try:
        fn = build(body)
        x = torch.ones(N, dtype=torch.float32).npu()
        y = fn(x)
        y = y.cpu()
        if check is not None:
            ok, msg = check(y)
            print(f"[{name}] RUN {'OK' if ok else 'WRONG'} {msg}")
        else:
            print(f"[{name}] RUN OK")
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] FAIL: {type(e).__name__}: {str(e)[:220]}")


def p_fill_off(a, b, c, f, off):
    # fill(h, off) + vadd(lane, h, h)  (in-place, dst == src2)
    lane = T.alloc_ub((BS,), "int32")
    T.arange(lane, [1], 0)
    T.fill(c, off)
    T.vadd(lane, c, c)
    T.vcast(c, f, round_mode="rint")
    T.copy(f, f)


def p_vabs_int32(a, b, c, f, off):
    T.fill(a, T.cast(-5, "int32"))
    T.vabs(a, c)
    T.vcast(c, f, round_mode="rint")
    T.copy(f, f)


def p_vclamp_int32_scalar(a, b, c, f, off):
    T.fill(a, T.cast(-3, "int32"))
    T.vclamp(a, c, 0, 1)
    T.vcast(c, f, round_mode="rint")
    T.copy(f, f)


def p_vsub_scalar(a, b, c, f, off):
    T.fill(a, T.cast(100, "int32"))
    T.vsub(a, 7, c)
    T.vcast(c, f, round_mode="rint")
    T.copy(f, f)


def p_vxor_scalar(a, b, c, f, off):
    T.fill(a, T.cast(5, "int32"))
    T.vxor(a, 9, c)
    T.vcast(c, f, round_mode="rint")
    T.copy(f, f)


def p_vand_scalar(a, b, c, f, off):
    T.fill(a, T.cast(5, "int32"))
    T.vand(a, 9, c)
    T.vcast(c, f, round_mode="rint")
    T.copy(f, f)


def p_vadd_runscalar(a, b, c, f, off):
    lane = T.alloc_ub((BS,), "int32")
    T.arange(lane, [1], 0)
    T.vadd(lane, off, c)
    T.vcast(c, f, round_mode="rint")
    T.copy(f, f)


def p_fill_int32_ub(a, b, c, f, off):
    T.fill(c, T.cast(3, "int32"))
    T.vcast(c, f, round_mode="rint")
    T.copy(f, f)


PROBES = [
    ("fill(h,off)+vadd(lane,h,h)", p_fill_off, None),
    ("vabs(int32)", p_vabs_int32, lambda y: (float(y[0]) == 5.0, f"y0={float(y[0])}")),
    ("vclamp(int32,scalar lo/hi)", p_vclamp_int32_scalar,
     lambda y: (float(y[0]) == 0.0, f"y0={float(y[0])}")),
    ("vsub(t,scalar,t)", p_vsub_scalar, lambda y: (float(y[0]) == 93.0, f"y0={float(y[0])}")),
    ("vxor(t,scalar,t)", p_vxor_scalar, lambda y: (float(y[0]) == 12.0, f"y0={float(y[0])}")),
    ("vand(t,scalar,t)", p_vand_scalar, lambda y: (float(y[0]) == 1.0, f"y0={float(y[0])}")),
    ("vadd(lane,off_runtime,c)", p_vadd_runscalar, None),
    ("fill(int32_ub)", p_fill_int32_ub, lambda y: (float(y[0]) == 3.0, f"y0={float(y[0])}")),
]


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    for name, body, check in PROBES:
        run(name, body, check)
