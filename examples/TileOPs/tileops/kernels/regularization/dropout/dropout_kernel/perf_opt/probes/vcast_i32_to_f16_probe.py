"""Probe: is `hivm.hir.vcast` int32 -> fp16/bf16 lowering available on this
toolchain?  (capability-gap evidence for CG entry "vcast int32->half")

Runs four cases and prints the FIRST failing diagnostic line of each:

  1. int32 -> f32   (control: known-supported HW cast)
  2. int32 -> f16   rint     <- the gap dropout hits
  3. int32 -> f16   round / floor / ceil / default
  4. int32 -> bf16  rint     (BishengIR has a SCALAR two-hop template for this)

All cases are TENSOR -> TENSOR casts over UB buffers, i.e. the tensor path of
`hivm.hir.vcast` (the scalar path is a different op/pattern).

STATUS 2026-11-03 (see PATCH-vcast-i32-to-f16.diff + capability-gaps CG-2026-0027):
a source-level patch (whitelist + i32->f32->half decompose) was built and
propagated into bishengir-compile / bishengir-opt / libtilelangir.so /
libtilelang_module.so / the bundled libBiShengIRAggregateCAPI.so, yet this
probe STILL fails: the rejecting gate lives in CANN's prebuilt
`hivmc` (cann-9.0.0/tools/bishengir/bin/hivmc), which bishengir-compile
invokes for the final `.mlir -> bin` stage.  Even if it were reachable, that
patch only implements a two-hop decompose -- identical to what the dropout
kernel already writes by hand -- so it would save 0 ops.  Keep this probe as
the regression check for a future CANN build that has a NATIVE i32->half cast.

Run:  python vcast_i32_to_f16_probe.py
"""

import os
import re
import traceback

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

N = 4096
# Order matters: the real diagnostic ("currently don't support cast X_to_Y_mode")
# must win over the echoed IR line that also contains "hivm.hir.vcast".
ERROR_RE = re.compile(r"don't support cast\s+(\S+)")
FALLBACK_RE = re.compile(r"(Unsupported|error: NPU IR opt failed|error:)")


def first_diagnostic(msg):
    m = ERROR_RE.search(msg)
    if m:
        return f"currently don't support cast {m.group(1)}"
    for line in msg.splitlines():
        if FALLBACK_RE.search(line):
            return line.strip()[:300]
    tail = [ln for ln in msg.splitlines() if ln.strip()]
    return tail[-1][:300] if tail else "<no output>"


def build(dst_dtype, round_mode):
    @tilelang.jit(target="npuir")
    def factory():
        @T.prim_func
        def main(src: T.Tensor((N,), "int32"), dst: T.Tensor((N,), dst_dtype)):
            with T.Kernel(1, is_npu=True) as (_b, _v):
                s_ub = T.alloc_ub((N,), "int32")
                d_ub = T.alloc_ub((N,), dst_dtype)
                T.copy(src, s_ub)
                if round_mode is None:
                    T.vcast(s_ub, d_ub)
                else:
                    T.vcast(s_ub, d_ub, round_mode=round_mode)
                T.copy(d_ub, dst)

        return main

    return factory()


CASES = [
    ("control  int32 -> f32  rint", "float32", "rint"),
    ("GAP      int32 -> f16  rint", "float16", "rint"),
    ("GAP      int32 -> f16  round", "float16", "round"),
    ("GAP      int32 -> f16  floor", "float16", "floor"),
    ("GAP      int32 -> f16  ceil", "float16", "ceil"),
    ("GAP      int32 -> f16  default", "float16", None),
    ("GAP      int32 -> bf16 rint", "bfloat16", "rint"),
]


TORCH_DTYPE = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}


def check_numeric(dtype, mode, kernel):
    """rint == round-to-nearest-even, so the result must be bit-exact vs torch.

    Data covers the three regimes that matter:
      * 0/1 dropout-mask pattern (the value the kernel actually carries)
      * general integers in [-2048, 2048] (exact in f16's 11-bit mantissa)
      * boundary magnitudes (0, +/-1, +/-2048, +/-2049)
    """
    import torch
    import torch_npu  # noqa: F401

    torch.manual_seed(0)
    quarter = N // 4
    pattern = torch.arange(quarter, dtype=torch.int32) % 2          # 0/1 mask
    general = torch.randint(-2048, 2049, (quarter,), dtype=torch.int32)
    boundary = torch.tensor([0, 1, -1, 2, -2, 2048, -2048, 2049, -2049, 1000, -1000, 30000],
                            dtype=torch.int32).repeat(quarter // 12 + 1)[:quarter]
    src = torch.cat([pattern, general, boundary, general]).to("npu")
    assert src.numel() == N, f"probe data length {src.numel()} != N={N}"

    dst = torch.empty(N, dtype=getattr(torch, TORCH_DTYPE[dtype]), device="npu")
    kernel(src, dst)
    torch.npu.synchronize()

    exp = src.cpu().to(getattr(torch, TORCH_DTYPE[dtype]))
    got = dst.cpu()
    exact = torch.equal(got, exp)
    max_ulp_gap = (got.float() - exp.float()).abs().max().item() if not exact else 0.0
    return exact, max_ulp_gap


def main():
    print(f"probe: T.vcast(tensor int32 -> tensor <dst>)  N={N}, UB->UB\n")
    verdicts = {}
    kernels = {}
    for label, dtype, mode in CASES:
        try:
            k = build(dtype, mode)
        except Exception as exc:  # noqa: BLE001 - diagnostic text is the payload
            text = str(exc)
            verdicts[(dtype, mode)] = "FAIL"
            print(f"[FAIL] {label}")
            print(f"       {first_diagnostic(text)}")
        else:
            verdicts[(dtype, mode)] = "OK"
            kernels[(dtype, mode)] = k
            print(f"[ OK ] {label}")

    print("\n--- summary ---")
    for label, dtype, mode in CASES:
        print(f"  {verdicts[(dtype, mode)]:4s}  {label}")

    assert verdicts[("float32", "rint")], \
        "control case failed -> the probe itself is broken, fix it first"
    print("\ncontrol passes => harness is sound; the f16/bf16 rows are the gap report")

    print("\n--- numeric (rint == round-to-nearest-even; must be bit-exact) ---")
    for dtype in ("float32", "float16", "bfloat16"):
        key = (dtype, "rint")
        if key not in kernels:
            print(f"  SKIP  int32 -> {dtype}: not compilable")
            continue
        exact, gap = check_numeric(dtype, "rint", kernels[key])
        print(f"  {'EXACT' if exact else 'DIFF '} int32 -> {dtype}"
              f"  (max |gap| = {gap})")
        assert exact, f"int32->{dtype} is not bit-exact vs torch: gap={gap}"


if __name__ == "__main__":
    main()
