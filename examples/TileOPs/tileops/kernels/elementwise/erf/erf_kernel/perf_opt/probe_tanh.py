"""Diagnostic probe for the tanh-based odd approximation (NOT precision-gated).

erf(x) ~= tanh(x * Q(x^2)),  Q degree k.  No sign / vabs / vrec / vexp needed
(tanh of an odd polynomial is odd).  Measures task duration and (separately)
device accuracy against torch.erf.

mode: "tanh" | "no_tanh" (vtanh replaced by vmul, to price vtanh).
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

_CORE_NUM = 40
_F32 = "float32"

# d coeffs ascending power of x^2
_COEF = {
    4: [1.12835817913, 0.102932893381, -0.000496822903435,
        -0.000415134821557, 3.22441004229e-05],
    6: [1.12837971696, 0.102765404596, -0.000184245624304, -0.000625819034681,
        8.97450474202e-05, -5.99043039611e-06, 1.59209199977e-07],
}


@tilelang.jit(out_idx=[1], target="npuir")
def erf_fwd_kernel(N, dtype, output_dtype=None, block_size=2048, degree=4, mode="tanh",
                   clamp=0.0):
    out_dtype = output_dtype or dtype
    need_cast = dtype != _F32
    n_num = T.ceildiv(N, block_size)
    coef = _COEF[degree]
    k = degree

    @T.prim_func
    def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype), shape: T.int32):
        with T.Kernel(_CORE_NUM, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)
            xf = T.alloc_ub((block_size,), _F32)
            z = T.alloc_ub((block_size,), _F32)
            q = T.alloc_ub((block_size,), _F32)
            xc = xf if need_cast else x_ub
            for i in T.serial(T.ceildiv(n_num, _CORE_NUM)):
                task = i * _CORE_NUM + cid
                if task < n_num:
                    off = task * block_size
                    tail = T.min(block_size, shape - off)
                    T.copy(x[off : off + tail], x_ub[0:tail])
                    if need_cast:
                        T.vcast(x_ub, xf, round_mode="rint")
                    T.vmul(xc, xc, z)                       # z = x^2
                    T.fill(q, T.cast(coef[k], _F32))                      # Horner from c_k
                    if k >= 6:
                        T.vmul(q, z, q)
                        T.vadd(q, coef[5], q)
                    if k >= 5:
                        T.vmul(q, z, q)
                        T.vadd(q, coef[4], q)
                    T.vmul(q, z, q)
                    T.vadd(q, coef[3], q)
                    T.vmul(q, z, q)
                    T.vadd(q, coef[2], q)
                    T.vmul(q, z, q)
                    T.vadd(q, coef[1], q)
                    T.vmul(q, z, q)
                    T.vadd(q, coef[0], q)
                    T.vmul(xc, q, q)                        # P = x*Q
                    if clamp > 0.0:
                        T.vmin(q, clamp, q)                 # keep vtanh arg <= clamp
                    if mode == "no_tanh":
                        T.vmul(q, coef[0], z)
                    else:
                        T.vtanh(q, z)                       # y = tanh(P)
                    if need_cast:
                        T.vcast(z, y_ub, round_mode="rint")
                        T.copy(y_ub[0:tail], y[off : off + tail])
                    else:
                        T.copy(z[0:tail], y[off : off + tail])

    return main
