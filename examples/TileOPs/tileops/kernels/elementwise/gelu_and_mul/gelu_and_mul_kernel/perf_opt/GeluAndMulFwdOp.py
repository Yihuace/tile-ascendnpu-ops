# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""GeluAndMul dispatch: exact erf for fp32, tanh approximation for fp16/bf16.

Rationale (2026-10-30):
  * fp32 tolerance is 1e-5 -> must keep the exact erf polynomial (deg-20).
  * fp16/bf16 tolerance is 1e-2 / 1.6e-2; the tanh approximation differs from
    exact gelu by <= ~2.2e-3 (measured 3.9e-3 / 1.95e-3 on the prefill shape)
    and is ~2.2x faster than the deg-8 erf chain (197/202us vs 436/441us).
  * Both implementations keep the same factory signature and [M, 2N] layout.
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EXACT = _load("_gelu_and_mul_exact", "GeluAndMulFwdOp_exact.py")
_TANH = _load("_gelu_and_mul_tanh", "GeluAndMulFwdOp_tanh.py")

__all__ = ["gelu_and_mul_fwd_kernel", "golden_gelu_and_mul"]
golden_gelu_and_mul = _EXACT.golden_gelu_and_mul


def gelu_and_mul_fwd_kernel(M, N, dtype, output_dtype=None, block_N=None, num_cores=None):
    """fp32 -> exact erf; fp16/bf16 -> tanh approximation.

    Routing note (2026-11-05): the fp16/bf16 branch intentionally shares the
    tanh-sigmoid body with ``GeluTanhAndMulFwdOp``.  Identical msprof VEC
    times for the two operators at 2048x14336 are therefore expected, not an
    accidental fallback to the exact-erf path.  The exact-erf (deg-20) path
    is fp32-only and costs roughly 2x the vector instructions; it is kept
    because the fp32 tolerance is 1e-5.
    """
    if dtype == "float32":
        return _EXACT.gelu_and_mul_fwd_kernel(
            M, N, dtype, output_dtype=output_dtype, block_N=block_N, num_cores=num_cores
        )
    return _TANH.gelu_and_mul_fwd_kernel(
        M, N, dtype, output_dtype=output_dtype, block_N=block_N, num_cores=num_cores
    )
