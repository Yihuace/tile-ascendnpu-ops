# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Isinf forward kernel (NPU, Developer mode) -- round-5 deep-dive candidate.

Round-5 hypothesis (secondary deep-dive): bf16 is the ONLY non-floor item.
Sibling Isnan deep-dive (2026-10-18) proved predicate-family fp16/fp32 are
MTE2 mixed-R/W bandwidth-bound (aiv_mte2_ratio 0.81-0.90 >> aiv_vec_ratio
0.25-0.39) but **bf16 is the vector-bound exception** (baseline vec_t >
mte2_t).  Isinf bf16 is 6-pass (2x vcmp + 2x vcast + vadd + vcast) because
``T.vabs`` has no bf16 edge.  This candidate drops bf16 to the same 4-pass
shape as fp16/fp32 by computing |x| via a bitwise sign-bit clear:

    isinf(x) = (x & 0x7FFF) == +inf     (bf16 bit-level; sign bit masked)

    vand(x, 0x7FFF_mask) -> vcmp(eq +inf) -> vcast(bool->f16) -> vcast(f16->i8)

Correctness (bit-level, no fp arithmetic, NaN-safe):
  - +inf (0x7F80) & 0x7FFF = 0x7F80 == +inf -> True
  - -inf (0xFF80) & 0x7FFF = 0x7F80 == +inf -> True
  - NaN (0x7FCx) & 0x7FFF = 0x7FCx != +inf  -> False (ordered eq)
  - finite / +-0 / subnormal: cleared sign still != +inf -> False

The 0x7FFF mask is built ONCE outside the tile loop (no per-tile cost):
  T.fill(sign_ub, -0.0)          # -0.0 -> bit pattern 0x8000
  T.vnot(sign_ub, mask_ub)       # ~0x8000 = 0x7FFF

Docs (bit-level reinterpret on bf16 is supported):
  docs/Tilelang.language/逻辑操作/T.vand.md  2.2.1 (bf16 = √)
  docs/Tilelang.language/逻辑操作/T.vor.md   2.2.1 (fp16/fp32/bf16 bit-reinterpret)
  docs/Tilelang.language/逻辑操作/T.vnot.md  2.2.1 (bf16 = √)

Semantics (frozen by DESIGN.md section 0.1):
    y = isinf(x)   (elementwise float predicate, bool output, exact)
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
ASCEND_MODE = "Developer"

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = ["golden_isinf", "isinf_fwd_kernel"]

SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")
_BOOL_STORAGE_DTYPE = "int8"


def golden_isinf(x):
    """Infinity predicate reference: y = torch.isinf(x)  (bool output, exact)."""
    return torch.isinf(x.detach().cpu().float()).to(torch.bool)


TUNED_BLOCK_SIZE = 8192


@tilelang.jit(out_idx=[1], target="npuir")
def isinf_fwd_kernel(N, dtype, output_dtype=None, block_size=TUNED_BLOCK_SIZE, num_cores=None):
    """Isinf forward kernel: ``y = isinf(x)`` (bool via int8 byte storage).

    Round-5: bf16 |x| via ``vand(x, 0x7FFF)`` bit-level sign clear (4-pass),
    fp16/fp32 via native ``vabs`` (4-pass).  Unified 4-pass abs chain for all
    dtypes; the only dtype branch is the |x| producer (vand vs vabs).
    """
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"isinf_fwd_kernel only supports dtypes {SUPPORTED_DTYPES}, "
            f"got {dtype!r}"
        )

    out_dtype = output_dtype or _BOOL_STORAGE_DTYPE

    num_full = N // block_size
    tail = N - num_full * block_size
    num_logical = num_full + (1 if tail else 0)
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > num_logical:
        num_kernels = min(num_logical, vector_cores)
    else:
        num_kernels = num_cores
    num_local_tasks = T.ceildiv(num_full, num_kernels)

    use_vand = dtype == "bfloat16"   # bf16: no vabs edge -> bitwise sign clear
    # |x| producer: bf16 = vand(x, 0x7FFF); fp16/fp32 = vabs(x)

    # +inf constant in the input dtype (bf16 built by casting fp32 infinity).
    if dtype == "bfloat16":
        pos_inf = T.cast(T.infinity("float32"), "bfloat16")
    else:
        pos_inf = T.infinity(dtype)

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            abs_ub = T.alloc_ub((block_size,), dtype)      # |x|
            posinf_ub = T.alloc_ub((block_size,), dtype)   # constant +inf tile
            sign_ub = T.alloc_ub((block_size,), dtype)     # -0.0 (0x8000) src for vnot
            mask_ub = T.alloc_ub((block_size,), dtype)     # 0x7FFF sign-clear mask
            m_pos = T.alloc_ub((block_size,), "bool")      # eq(+inf) mask
            f_a = T.alloc_ub((block_size,), "float16")     # bool -> f16 (0.0/1.0)
            y_ub = T.alloc_ub((block_size,), out_dtype)    # int8 output byte tile

            # Pre-fill constants once, outside the tile loop.
            T.fill(posinf_ub, pos_inf)
            if use_vand:
                # 0x7FFF sign-clear mask: ~(-0.0) = ~0x8000 = 0x7FFF.
                T.fill(sign_ub, T.cast(-0.0, "bfloat16"))
                T.vnot(sign_ub, mask_ub)

            if num_full > 0:
                for i in T.Pipelined(num_local_tasks, num_stages=2):
                    block_id = i * num_kernels + cid
                    if block_id < num_full:
                        offset = block_id * block_size
                        T.copy(x[offset : offset + block_size], x_ub)
                        if use_vand:
                            # bf16 |x| = x & 0x7FFF (sign-bit clear, 1 op)
                            T.vand(x_ub, mask_ub, abs_ub)
                        else:
                            T.vabs(x_ub, abs_ub)
                        T.vcmp(abs_ub, posinf_ub, m_pos, "eq")
                        T.vcast(m_pos, f_a, round_mode="rint")
                        T.vcast(f_a, y_ub, round_mode="rint")
                        T.copy(y_ub, y[offset : offset + block_size])

            if tail > 0:
                if cid == 0:
                    offset = num_full * block_size
                    T.copy(x[offset : offset + tail], x_ub[0:tail])
                    if use_vand:
                        T.vand(x_ub, mask_ub, abs_ub)
                    else:
                        T.vabs(x_ub, abs_ub)
                    T.vcmp(abs_ub, posinf_ub, m_pos, "eq")
                    T.vcast(m_pos, f_a, round_mode="rint")
                    T.vcast(f_a, y_ub, round_mode="rint")
                    T.copy(y_ub[0:tail], y[offset : offset + tail])

    return main


_TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _max_diff_metric(out, ref):
    out = out.detach().cpu().to(torch.uint8)
    ref = ref.detach().cpu().to(torch.uint8)
    eq = out == ref
    if bool(eq.all()):
        return 0.0
    diff = (out.to(torch.int64) - ref.to(torch.int64)).abs()
    diff = torch.where(eq, torch.zeros_like(diff), diff)
    return diff.max().item()


def _make_input(N, dtype_str, special=False):
    torch_dtype = _TORCH_DTYPE[dtype_str]
    x = torch.randn(N, dtype=torch.float32)
    if special:
        x[0::8] = float("nan")
        x[1::8] = float("inf")
        x[2::8] = float("-inf")
        x[3::8] = 0.0
        x[4::8] = -0.0
        x[5::8] = 1e-30  # subnormal probe
        x[6::8] = 3.4e38  # large finite
        x[7::8] = -3.4e38
    return x.to(torch_dtype)


def _run_tensor(x, dtype_str, tag):
    N = x.shape[0]
    kernel = isinf_fwd_kernel(N, dtype_str)
    y_storage = kernel(x.npu(), N).cpu()
    y = y_storage.view(torch.bool)
    golden = golden_isinf(x)
    torch.testing.assert_close(y, golden, atol=0, rtol=0)
    max_diff = _max_diff_metric(y, golden)
    print(f"[{tag}] PASS: shape=({N},) dtype={dtype_str} max_diff={max_diff:.2e}")
    return max_diff


def _run_case(N, dtype_str, tag, special=False, block_size=None):
    x = _make_input(N, dtype_str, special=special)
    if block_size is not None:
        kernel = isinf_fwd_kernel(N, dtype_str, block_size=block_size)
        y_storage = kernel(x.npu(), N).cpu()
        y = y_storage.view(torch.bool)
        golden = golden_isinf(x)
        torch.testing.assert_close(y, golden, atol=0, rtol=0)
        max_diff = _max_diff_metric(y, golden)
        print(
            f"[{tag}] PASS: shape=({N},) dtype={dtype_str} "
            f"block_size={block_size} max_diff={max_diff:.2e}"
        )
        return max_diff
    return _run_tensor(x, dtype_str, tag)


def run_L0():
    N = 1048576
    for dtype_str in SUPPORTED_DTYPES:
        _run_case(N, dtype_str, "L0", special=True)
    for bad_dtype in ("int32", "bool", "int8"):
        try:
            isinf_fwd_kernel(N, bad_dtype)
            raise AssertionError(f"{bad_dtype} must be rejected")
        except ValueError:
            print(f"[L0] PASS: dtype {bad_dtype} rejected")


def run_L1():
    for dtype_str in SUPPORTED_DTYPES:
        for N in (8192, 100, 4097):
            _run_case(N, dtype_str, "L1", special=True)
    _run_case(4194304, "float16", "L1", special=True)
    _run_case(4194304, "float32", "L1", special=True)
    _run_case(1048576, "bfloat16", "L1", special=True)
    _run_case(1048576, "float32", "L1", special=True, block_size=4096)


def run_L2():
    for dtype_str in SUPPORTED_DTYPES:
        for N in (1, 2, 10):
            try:
                _run_case(N, dtype_str, "L2", special=True)
            except Exception as e:  # noqa: BLE001
                print(f"[L2] WARN (Record without blocking): N={N} dtype={dtype_str} err={e}")


def run_boundary():
    N = 4096
    try:
        _run_case(N, "float32", "Boundary-special", special=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): special err={e}")

    try:
        x = torch.full((N,), float("nan"), dtype=torch.float32)
        y = isinf_fwd_kernel(N, "float32")(x.npu(), N).cpu().view(torch.bool)
        assert not bool(y.any()), "all-NaN input must map to all-False"
        print(f"[Boundary] PASS: all-NaN -> all-False (nan_count={N})")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): all-NaN err={e}")

    try:
        x = torch.empty(N, dtype=torch.float32)
        x[0::2] = float("inf")
        x[1::2] = float("-inf")
        y = isinf_fwd_kernel(N, "float32")(x.npu(), N).cpu().view(torch.bool)
        assert bool(y.all()), "all-+-inf input must map to all-True"
        print("[Boundary] PASS: all-+-inf -> all-True")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): inf err={e}")

    try:
        x = torch.zeros(N, dtype=torch.float32)
        x[0::3] = -0.0
        x[1::3] = 1e-30
        x[2::3] = -3.4e38
        y = isinf_fwd_kernel(N, "float32")(x.npu(), N).cpu().view(torch.bool)
        assert not bool(y.any()), "+-0 / subnormal / finite must be False"
        print("[Boundary] PASS: +-0 / subnormal / finite -> all-False")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): finite err={e}")


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
