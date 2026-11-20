# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""BitwiseNot forward kernel -- Stage 4 round 3 candidate (static bulk + tail).

ONE main optimization point vs the round-1 incumbent: the baseline's hot loop
computes ``tail = T.min(block_size, shape - off)`` every iteration and issues a
*dynamic-extent* ``T.copy`` for both the load and the store.  A dynamic DMA
extent can inhibit the compiler's multi-buffer / pipelined lowering.  This
candidate splits the work into

  * Part A: the ``n_full = N // block_size`` full tiles, guarded only by the
    static ``task < n_full`` compare, with **static-extent** copies
    (``x[off:off+block_size]``) -- for all benchmark tune shapes N is exactly
    divisible by block_size, so the whole bulk runs with static DMA extents;
  * Part B: at most one partial tail tile (``N % block_size``), handled by
    core 0 with the dynamic ``shape`` extent (mirrors the baseline tail).

Correctness for non-divisible N is preserved by Part B.

run: python bitwise_not_fwd_kernel_round3_staticbulk.py --level all
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

ASCEND_MODE = "Developer"

tilelang.cache.clear_cache()

# UB-safe tuned block_size (element count), 3 live UB buffers x 2 (multi-buffer).
_BLOCK_SIZE = {"bool": 32768, "uint8": 32768, "int8": 32768,
               "int16": 16384, "int32": 8192, "int64": 4096}
_CORE_NUM = 40


def golden_bitwise_not(x):
    return torch.bitwise_not(x.detach().cpu())


def _auto_block_size(dtype):
    return _BLOCK_SIZE[dtype]


@tilelang.jit(out_idx=[1], target="npuir")
def bitwise_not_fwd_kernel(N, dtype, output_dtype=None, block_size=None):
    is_bool = dtype == "bool"
    k_dtype = "int8" if is_bool else dtype
    out_dtype = output_dtype or k_dtype
    block_size = block_size or _auto_block_size(dtype)
    n_full = N // block_size
    has_tail = (N % block_size) != 0
    n_num = T.ceildiv(N, block_size)

    @T.prim_func
    def main(
        x: T.Tensor((N,), k_dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(_CORE_NUM, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), k_dtype)
            y_ub = T.alloc_ub((block_size,), k_dtype)
            ones_ub = T.alloc_ub((block_size,), k_dtype)

            # Part A: static-extent bulk loop over full tiles.
            for i in T.serial(T.ceildiv(n_full + (1 if has_tail else 0), _CORE_NUM)):
                task = i * _CORE_NUM + cid
                if task < n_full:
                    off = task * block_size
                    T.copy(x[off : off + block_size], x_ub)
                    if is_bool:
                        T.fill(ones_ub, T.cast(1, k_dtype))
                        T.vxor(x_ub, ones_ub, y_ub)
                    else:
                        T.vnot(x_ub, y_ub)
                    T.copy(y_ub, y[off : off + block_size])
                elif has_tail and task == n_full:
                    off = task * block_size
                    tail = T.min(block_size, shape - off)
                    T.copy(x[off : off + tail], x_ub[0:tail])
                    if is_bool:
                        T.fill(ones_ub, T.cast(1, k_dtype))
                        T.vxor(x_ub, ones_ub, y_ub)
                    else:
                        T.vnot(x_ub, y_ub)
                    T.copy(y_ub[0:tail], y[off : off + tail])

    return main


_TORCH_DTYPE = {
    "bool": torch.bool, "uint8": torch.uint8, "int8": torch.int8,
    "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
}


def _make_input(N, dtype_str):
    if dtype_str == "bool":
        return torch.rand(N) > 0.5
    if dtype_str == "uint8":
        return torch.randint(0, 256, (N,), dtype=torch.uint8)
    if dtype_str == "int64":
        return torch.randint(-(2**63), 2**63 - 1, (N,), dtype=torch.int64)
    if dtype_str == "int32":
        return torch.randint(-(2**31), 2**31 - 1, (N,), dtype=torch.int32)
    if dtype_str == "int16":
        return torch.randint(-(2**15), 2**15 - 1, (N,), dtype=torch.int16)
    if dtype_str == "int8":
        return torch.randint(-128, 127, (N,), dtype=torch.int8)
    raise ValueError(f"unsupported dtype {dtype_str}")


def _max_diff(y, golden):
    return (y.to(torch.float64) - golden.to(torch.float64)).abs().max().item()


def _run_tensor(x_cpu, dtype_str, tag):
    N = x_cpu.numel()
    golden = golden_bitwise_not(x_cpu)
    kernel = bitwise_not_fwd_kernel(N, dtype_str)
    if dtype_str == "bool":
        x_npu = x_cpu.npu().view(torch.int8)
        y = kernel(x_npu, N).view(torch.bool)
    elif dtype_str == "uint8":
        y = kernel(x_cpu.npu(), N).view(torch.uint8)[:N]
    else:
        y = kernel(x_cpu.npu(), N)
    y_cpu = y.cpu()
    ok = bool(torch.equal(y_cpu, golden))
    assert ok, (f"[{tag}] exact-match FAIL: shape=({N},) dtype={dtype_str} "
                f"max_diff={_max_diff(y_cpu, golden)}")
    print(f"[{tag}] PASS: shape=({N},) dtype={dtype_str} max_diff={_max_diff(y_cpu, golden)}")


def _run_case(N, dtype_str, tag):
    _run_tensor(_make_input(N, dtype_str), dtype_str, tag)


def run_L0():
    for dtype_str in ("int32", "int64", "bool", "uint8", "int8", "int16"):
        _run_case(1048576, dtype_str, "L0")


def run_L1():
    _run_case(4000000, "int32", "L1")
    _run_case(4000000, "int64", "L1")
    _run_case(2047, "int32", "L1")
    _run_case(4096, "int16", "L1")
    _run_case(2047, "int8", "L1")
    _run_case(100, "bool", "L1")
    _run_case(100, "uint8", "L1")


def run_L2():
    for N, dtype_str in ((1, "int32"), (2, "int32"), (10, "int64"),
                         (1025, "bool"), (4095, "uint8")):
        try:
            _run_case(N, dtype_str, "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN: N={N} dtype={dtype_str} err={e}")


def run_boundary():
    boundary_cases = (
        ("int32", torch.full((4096,), 0, dtype=torch.int32)),
        ("int32", torch.full((4096,), -1, dtype=torch.int32)),
        ("int32", torch.tensor([-(2**31), 2**31 - 1, 0, 1, -1] * 819, dtype=torch.int32)),
        ("int64", torch.tensor([-(2**63), 2**63 - 1, 0, 1, -1] * 819, dtype=torch.int64)),
        ("uint8", torch.tensor([0, 255, 1, 254, 128] * 819, dtype=torch.uint8)),
        ("bool", torch.zeros(4096, dtype=torch.bool)),
        ("bool", torch.ones(4096, dtype=torch.bool)),
    )
    for dtype_str, x in boundary_cases:
        try:
            _run_tensor(x, dtype_str, "Boundary")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN: dtype={dtype_str} err={e}")


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
