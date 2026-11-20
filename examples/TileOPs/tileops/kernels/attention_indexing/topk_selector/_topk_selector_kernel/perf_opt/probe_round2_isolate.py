"""Probe: isolate which kernel breaks under row parallelism (round2).

round2 (grid = one block per row) passes only the k=N case.  This probe takes the
host bisection out of the picture -- the threshold is computed exactly on the host
-- and forces `grid` to 1 or M by monkeypatching the module's `_grid_for`, so each
kernel form is checked on its own:

    count  grid=1 / grid=M   -> per-row count vs golden
    select grid=1 / grid=M   -> per-row index SET vs golden

Run:  python probe_round2_isolate.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import importlib.util  # noqa: E402

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, tag):
    path = os.path.join(HERE, name)
    spec = importlib.util.spec_from_file_location(f"probe_{tag}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    mod = load("_topk_selector_kernel_round2_rowparallel.py", "round2")
    real_grid_for = mod._grid_for

    for m, n, k in ((2, 256, 8), (8, 1024, 16)):
        base = torch.randperm(n, dtype=torch.float32) * 7.0
        x = base.unsqueeze(0).repeat(m, 1) + torch.arange(m, dtype=torch.float32).unsqueeze(1)
        thr = torch.topk(x, k, dim=-1).values[:, -1].contiguous()
        want_cnt = (x >= thr.unsqueeze(1)).sum(dim=1).float()
        want_idx = torch.topk(x, k, dim=-1).indices
        print(f"=== M={m} N={n} K={k} ===", flush=True)

        for label, g in (("grid=1", 1), ("grid=M", m)):
            mod._grid_for = lambda mm, cap=48, _g=g: _g
            try:
                kc = mod._topk_count_kernel(m, n)
                got_cnt = kc(x.npu(), thr.npu()).cpu()
                cnt_ok = torch.equal(got_cnt, want_cnt)
                print(f"  count  {label}: {'PASS' if cnt_ok else 'FAIL'} got={got_cnt.tolist()} want={want_cnt.tolist()}", flush=True)

                ks = mod._topk_select_kernel(m, n, k)
                got_idx = ks(x.npu(), thr.npu()).cpu()
                sel_ok = mod.set_equal(got_idx, want_idx)
                print(f"  select {label}: {'PASS' if sel_ok else 'FAIL'}", flush=True)
                if not sel_ok:
                    print(f"    got  row0 = {sorted(got_idx[0].tolist())}", flush=True)
                    print(f"    want row0 = {sorted(want_idx[0].tolist())}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  {label}: BUILD/RUN FAILED: {type(exc).__name__}: {str(exc).splitlines()[0]}", flush=True)

    mod._grid_for = real_grid_for
    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
