"""G3 A/B app: measure the latency delta of ONE removed PRNG v-op.

Same workload/shape/block_size/dtype as the r9_final tune case
(tokens-1k-hidden-4k-fp16 = [1024,4096] fp16, bs=8192), so the only
difference is the kernel variant selected by env ``VAR``:

  VAR=baseline  -> perf_opt/_make_dropout_kernel.py          (r9_final, 5 PRNG ops)
  VAR=minus_op  -> probes/_make_dropout_kernel_minus_op.py   (4 PRNG ops, timing only)

Timing must be read with msprof op (wall clock is unusable on this host):

  cd /tmp/opwork && VAR=baseline msprof op --kernel-name=main \
      --launch-count=5 --warm-up=2 --output=./ab_base python prng_ab_app.py

The minus_op variant's mask distribution is intentionally invalid; this app
exists only to answer "does removing one op even show up in Task Duration?".
"""

import importlib.util
import os
import sys

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
VAR = os.environ.get("VAR", "baseline")
MODULE = {
    "baseline": os.path.join(HERE, "..", "_make_dropout_kernel.py"),
    "minus_op": os.path.join(HERE, "_make_dropout_kernel_minus_op.py"),
}[VAR]

N = 1024 * 4096          # tokens-1k-hidden-4k
BS = 8192                # tune winning block size
DTYPE, P, SEED = "float16", 0.5, 0


def load_module(path):
    spec = importlib.util.spec_from_file_location(f"dropout_ab_{VAR}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load_module(MODULE)
    factory = mod._make_dropout_kernel(N, DTYPE, P, SEED)
    kernel = factory(BS)

    torch.manual_seed(0)
    x = torch.randn(N, dtype=torch.float16, device="npu")
    for _ in range(30):
        y = kernel(x)
    torch.npu.synchronize()
    print(f"[app] VAR={VAR} done; out[0]={float(y.reshape(-1)[0]):.4f} "
          f"(value is meaningless for minus_op -- timing only)")


if __name__ == "__main__":
    main()
