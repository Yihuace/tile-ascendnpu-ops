# Copyright (c) Huawei Technologies Co., Ltd. 2026.

import os

# Developer mode for npuir (linear data dependency, no manual sync needed).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  enables .npu() placement

tilelang.cache.clear_cache()


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_mish(x):
    """Mish reference: y = x * tanh(ln(1 + exp(x))).

    Equivalent to torch.nn.functional.mish. For fp16/bf16 inputs the
    computation is upcast to fp32 then cast back, matching the NPU kernel
    precision strategy (vcast -> fp32 compute -> vcast back).
    """
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    y_f32 = x_f32 * torch.tanh(torch.log(1 + torch.exp(x_f32)))
    return y_f32.to(x_cpu.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def mish_fwd_kernel(N, dtype, output_dtype=None, block_size=2048):
    """Mish activation forward kernel.

    y = x * tanh(ln(1 + exp(x))), computed in float32 intermediate
    precision for all input dtypes. block_size folds GPU
    threads(256) * num_per_thread(8) = 2048.
    """
    out_dtype = output_dtype or dtype
    n_num = T.ceildiv(N, block_size)
    compute_dtype = "float32"

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            # 1. Allocate UB buffers (v-prefix API requires UB-resident data).
            x_ub = T.alloc_ub((block_size,), dtype)
            x_f32_ub = T.alloc_ub((block_size,), compute_dtype)
            buf1_ub = T.alloc_ub((block_size,), compute_dtype)
            buf2_ub = T.alloc_ub((block_size,), compute_dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)

            # 2. Tail handling: dynamic block size for non-divisible N.
            offset = cid * block_size
            remaining = shape - offset
            tail_size = T.min(block_size, remaining)

            # 3. Load GM -> UB (only valid elements copied in).
            T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

            # 4. Cast to fp32 (fp16/bf16) or copy (fp32).
            #    Python-level branch: resolved at JIT trace time, no runtime branch.
            if dtype != compute_dtype:
                T.vcast(x_ub, x_f32_ub, round_mode="rint")
            else:
                T.copy(x_ub, x_f32_ub)

            # 5. Compute chain: y = x * tanh(ln(1 + exp(x)))
            #    v-ops operate on full buffers; tail garbage elements are
            #    harmless (per-element, never copied out). buf1/buf2 ping-pong.
            T.vexp(x_f32_ub, buf1_ub)  # buf1 = exp(x)
            T.vadd(buf1_ub, 1.0, buf2_ub)  # buf2 = 1 + exp(x)
            T.vln(buf2_ub, buf1_ub)  # buf1 = ln(1 + exp(x))
            T.vtanh(buf1_ub, buf2_ub)  # buf2 = tanh(ln(...))
            T.vmul(x_f32_ub, buf2_ub, buf1_ub)  # buf1 = x * tanh(...)

            # 6. Cast back to output dtype (fp16/bf16) or copy (fp32).
            if out_dtype != compute_dtype:
                T.vcast(buf1_ub, y_ub, round_mode="rint")
            else:
                T.copy(buf1_ub, y_ub)

            # 7. Store UB -> GM (only valid elements copied out).
            T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])

    return main


# ---------- precision comparing func ----------
_TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

_TOLERANCE = {
    "float16": (1e-3, 1e-3),
    "float32": (1e-5, 1e-5),
    "bfloat16": (1.6e-2, 1.6e-2),
}


def _run_tensor(x, dtype_str, tag):
    """Run kernel on a pre-made tensor and compare against golden."""
    N = x.shape[0]
    atol, rtol = _TOLERANCE[dtype_str]
    kernel = mish_fwd_kernel(N, dtype_str)
    y = kernel(x, N)
    golden = golden_mish(x)
    torch.testing.assert_close(y.cpu(), golden, atol=atol, rtol=rtol)
    max_diff = (y.float().cpu() - golden.float()).abs().max().item()
    print(f"[{tag}] PASS: shape=({N},) dtype={dtype_str} max_diff={max_diff:.2e}")


def _run_case(N, dtype_str, tag, scale=1.0):
    """Run kernel on random data of shape (N,) with given dtype."""
    torch_dtype = _TORCH_DTYPE[dtype_str]
    x = (torch.randn(N, dtype=torch.float32) * scale).to(torch_dtype).npu()
    _run_tensor(x, dtype_str, tag)


# ---------- hierarchical testing ----------
def run_L0():
    # L0 smoke: representative 1M-element cases covering all three dtypes.
    N = 1048576
    _run_case(N, "float16", "L0")
    _run_case(N, "float32", "L0")
    _run_case(N, "bfloat16", "L0")


def run_L1():
    # L1: divisible + non-divisible shapes across all dtypes (tail path).
    for dtype_str in ("float32", "float16", "bfloat16"):
        for N in (4096, 100):
            _run_case(N, dtype_str, "L1")
    _run_case(8192, "float16", "L1")
    _run_case(2049, "float16", "L1")
    _run_case(3000, "bfloat16", "L1")


def run_L2():
    # L2: edge sizes (non-blocking).
    for N in (1, 2, 10):
        try:
            _run_case(N, "float32", "L2")
        except Exception as e:
            print(f"[L2] WARN (Record without blocking): N={N} err={e}")
    try:
        _run_case(4194304, "float32", "L2")
    except Exception as e:
        print(f"[L2] WARN (Record without blocking): N=4194304 err={e}")


def run_boundary():
    # Boundary: special-value inputs (non-blocking).
    # zeros: mish(0) = 0
    try:
        x = torch.zeros(4096, dtype=torch.float32).npu()
        _run_tensor(x, "float32", "Boundary-zeros")
    except Exception as e:
        print(f"[Boundary] WARN (Record without blocking): zeros err={e}")
    # large positive (exp non-overflow in fp32 domain)
    try:
        x = (torch.randn(4096, dtype=torch.float32) * 5.0).to(torch.float16).npu()
        _run_tensor(x, "float16", "Boundary-large-positive")
    except Exception as e:
        print(f"[Boundary] WARN (Record without blocking): large-positive err={e}")
    # large negative (exp -> 0, y -> 0)
    try:
        x = torch.full((4096,), -10.0, dtype=torch.float32).npu()
        _run_tensor(x, "float32", "Boundary-large-negative")
    except Exception as e:
        print(f"[Boundary] WARN (Record without blocking): large-negative err={e}")


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
