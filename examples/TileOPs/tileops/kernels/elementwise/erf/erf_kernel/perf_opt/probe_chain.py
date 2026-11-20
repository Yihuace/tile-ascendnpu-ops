"""Diagnostic op-cost decomposition probe (NOT a correctness-gated candidate).

Same structure as the baseline kernel but with compile-time `variant` switches
that drop / replace specific ops, so msprof Task Duration differences reveal
the marginal vector cost of each op group.  Outputs are intentionally not
guaranteed correct -- this file is never used as a perf_opt candidate and is
never precision-gated.

variant:
  full         -- exact baseline chain
  no_exp       -- T.vexp -> T.vmul
  no_rec       -- T.vrec -> T.vmul
  short_horner -- 9-op Horner -> 1 vmul
  no_sign      -- drop vcmp + vselect + vmul(sign)
  only_copy    -- copy in, copy out (chain removed)
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

_BLOCK_SIZE = 2048
_CORE_NUM = 40
_COMPUTE_DTYPE = "float32"

_P = 0.3275911
_A1 = 0.254829592
_A2 = -0.284496736
_A3 = 1.421413741
_A4 = -1.453152027
_A5 = 1.061405429


@tilelang.jit(out_idx=[1], target="npuir")
def erf_fwd_kernel(N, dtype, output_dtype=None, block_size=_BLOCK_SIZE, variant="full"):
    out_dtype = output_dtype or dtype
    f32 = _COMPUTE_DTYPE
    need_cast = dtype != f32
    n_num = T.ceildiv(N, block_size)

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(_CORE_NUM, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)
            xf = T.alloc_ub((block_size,), f32)
            xa = T.alloc_ub((block_size,), f32)
            t = T.alloc_ub((block_size,), f32)
            poly = T.alloc_ub((block_size,), f32)
            e = T.alloc_ub((block_size,), f32)
            ones = T.alloc_ub((block_size,), f32)
            negones = T.alloc_ub((block_size,), f32)
            zero = T.alloc_ub((block_size,), f32)
            cond = T.alloc_ub((block_size,), "bool")

            T.fill(ones, T.cast(1.0, f32))
            T.fill(negones, T.cast(-1.0, f32))
            T.fill(zero, T.cast(0.0, f32))

            xc = xf if need_cast else x_ub

            for i in T.serial(T.ceildiv(n_num, _CORE_NUM)):
                task = i * _CORE_NUM + cid
                if task < n_num:
                    off = task * block_size
                    tail = T.min(block_size, shape - off)
                    T.copy(x[off : off + tail], x_ub[0:tail])

                    if need_cast:
                        T.vcast(x_ub, xf, round_mode="rint")

                    if variant == "only_copy":
                        T.copy(xc[0:tail], y[off : off + tail])
                    else:
                        T.vcmp(xc, zero, cond, "ge")
                        T.vabs(xc, xa)
                        T.vmul(xa, _P, t)
                        T.vadd(t, T.cast(1.0, f32), t)
                        if variant == "no_rec":
                            T.vmul(t, _P, t)
                        else:
                            T.vrec(t, t)
                        if variant == "short_horner":
                            T.vmul(t, _A5, poly)
                        else:
                            T.vmul(t, _A5, poly)
                            T.vadd(poly, _A4, poly)
                            T.vmul(poly, t, poly)
                            T.vadd(poly, _A3, poly)
                            T.vmul(poly, t, poly)
                            T.vadd(poly, _A2, poly)
                            T.vmul(poly, t, poly)
                            T.vadd(poly, _A1, poly)
                            T.vmul(poly, t, poly)
                        T.vmul(xa, xa, xa)
                        T.vmul(xa, T.cast(-1.0, f32), xa)
                        if variant == "no_exp":
                            T.vmul(xa, _P, e)
                        else:
                            T.vexp(xa, e)
                        T.vmul(poly, e, poly)
                        T.vsub(ones, poly, xc)
                        if variant == "no_sign":
                            T.vmul(xc, ones, xc)
                        else:
                            T.vselect(cond, ones, negones, t)
                            T.vmul(xc, t, xc)

                        if need_cast:
                            T.vcast(xc, y_ub, round_mode="rint")
                            T.copy(y_ub[0:tail], y[off : off + tail])
                        else:
                            T.copy(xc[0:tail], y[off : off + tail])

    return main
