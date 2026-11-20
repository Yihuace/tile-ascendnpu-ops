"""Probe: does the in-kernel bisection BODY build, or only multi-body forms?

Builds the round1 candidate's fused count kernel with rounds=1 and rounds=2 and
checks the produced threshold against the exact k-th largest value.  A crash at
rounds=1 means the body's op mix is the problem; a crash only at rounds>=2 means
the multi-body/unroll structure is.

Run:  python probe_inbranch_rounds.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

import importlib.util  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    path = os.path.join(HERE, name)
    spec = importlib.util.spec_from_file_location("probe_mod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    mod = load("_topk_selector_kernel_round1b_serial_bisect.py")
    m, n, k = 2, 256, 8
    base = torch.randperm(n, dtype=torch.float32) * 7.0
    x = base.unsqueeze(0).repeat(m, 1) + torch.arange(m, dtype=torch.float32).unsqueeze(1)
    want = torch.topk(x, k, dim=-1).values[:, -1]

    for rounds in (1, 2):
        print(f"--- rounds={rounds} ---", flush=True)
        ker = mod._topk_count_kernel(m, n, k, rounds=rounds)
        got = ker(x.npu()).cpu()
        print(f"  built+ran OK; thr={got.tolist()}", flush=True)
        print(f"  exact kth={want.tolist()}  match={torch.equal(got, want)}", flush=True)

    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
