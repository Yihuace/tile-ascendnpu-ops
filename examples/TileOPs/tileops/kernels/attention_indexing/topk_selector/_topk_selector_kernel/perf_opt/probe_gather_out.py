"""Probe: what does the gather select actually write?

K=1 passes, K>1 fails -> the per-slot loop is suspect (mask clearing or the
scratch store).  Tiny case for readability: N=16, x=0..15, thr=12, so the answer
is the set {12,13,14,15} in ANY order.

Run:  python probe_gather_out.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import importlib.util  # noqa: E402

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, tag):
    spec = importlib.util.spec_from_file_location(tag, os.path.join(HERE, name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    mod = load("_topk_selector_kernel_round3_gather_select.py", "r3")
    m, n, k = 1, 16, 4
    x = torch.arange(n, dtype=torch.float32).unsqueeze(0)
    thr = torch.tensor([12.0])
    print(f"x    = {x.tolist()}", flush=True)
    print(f"thr  = {thr.tolist()}  expect set {{12,13,14,15}}", flush=True)

    ker = mod._topk_select_gather_kernel(m, n, k)
    out = ker(x.npu(), thr.npu()).cpu()
    print(f"got mask-grid = {out.tolist()}", flush=True)

    # also: does the count kernel see 4 there?
    kc = mod._topk_count_kernel(m, n)
    cnt = kc(x.npu(), thr.npu()).cpu()
    print(f"count = {cnt.tolist()}  expect [4.0]", flush=True)

    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
