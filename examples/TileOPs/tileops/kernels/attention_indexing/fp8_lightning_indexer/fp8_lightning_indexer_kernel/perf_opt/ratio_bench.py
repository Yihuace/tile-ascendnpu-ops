"""Official-Ratio bench: reproduce the benchmark's ``--aic-metrics=Roofline`` run.

Measures one (module, workload, config) with ``msprof op --aic-metrics=Roofline``
(the exact mode used by ``tileops/benchmark/benchmark_base.py`` to produce the
``Ratio(%)`` report field) and prints:

* ``Task Duration(us)`` from ``OpBasicInfo*.csv`` (the Stage-4 latency metric),
* the profiler roofline point (``Ratio``/``AI``/``Performance``/``Computility``),
* the same model recomputed from the per-block CSVs
  (``msprof_roofline.compute_from_csvs``) so the ops/bytes split
  (cube vs vector vs GM traffic) is visible.

Usage::

    python ratio_bench.py --module _fp8_lightning_indexer_kernel.py --workload s8k \
        --config '{"block_N":64,"block_Q":8,"num_stages":2}' --candidate-id r0_base \
        --phase baseline --round 0 --launch-count 2
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "examples", "TileOPs")):
    ROOT = os.path.dirname(ROOT)
REPO = ROOT
sys.path.insert(0, os.path.join(REPO, "examples", "TileOPs"))

from tileops.benchmark.msprof_roofline import (  # noqa: E402
    DEFAULT_CONFIG,
    compute_from_csvs,
    parse_bin_file,
)

WORKLOADS = {
    "smoke": dict(batch=1, seq_len=1024, heads=8, index_dim=64, seq_len_kv=1024, kv_group=1,
                  label="lightning-indexer-smoke-s1k-h8-d64"),
    "s8k": dict(batch=1, seq_len=8192, heads=32, index_dim=64, seq_len_kv=8192, kv_group=1,
                label="lightning-indexer-bf16-s8k-h32-d64"),
    "s4k": dict(batch=1, seq_len=4096, heads=32, index_dim=64, seq_len_kv=8192, kv_group=1,
                label="lightning-indexer-bf16-s4k-h32-d64"),
}

KERNEL_ID = ("examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/"
             "fp8_lightning_indexer_kernel/_fp8_lightning_indexer_kernel.py"
             "::_fp8_lightning_indexer_kernel::_fp8_lightning_indexer_main")

INNER = '''\
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import torch, torch_npu
import importlib.util
torch.npu.set_device(0)
_spec = importlib.util.spec_from_file_location("cand_mod", {module!r})
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CASE = {case!r}
CONFIG = {config!r}
torch.manual_seed(0)
dev = "npu"


def _causal_windows(seq_len, seq_len_kv, avg=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    total = seq_len
    n = max(1, total // avg * 2)
    cu = torch.randint(0, avg * 2, (n,), generator=g)
    reached = torch.where(cu.cumsum(0) >= total)[0]
    if len(reached) > 0:
        cu = cu[:reached[0]]
    if int(cu.sum()) < total:
        cu = torch.cat([cu, torch.tensor([total - int(cu.sum())])])
    cum = torch.cumsum(cu, 0)
    ks = torch.zeros(total, dtype=torch.int32)
    ke = torch.zeros(total, dtype=torch.int32)
    start = 0
    for i in range(len(cu)):
        m = int(cu[i]); base = int(cum[i] - cu[i])
        ks[start:start + m] = base
        ke[start:start + m] = torch.minimum(
            torch.arange(1, m + 1, dtype=torch.int32) + base,
            torch.tensor(int(cum[i]), dtype=torch.int32))
        start += m
    return ks.clamp_(0, seq_len_kv), ke.clamp_(0, seq_len_kv)


ks, ke = _causal_windows(CASE["seq_len"], CASE["seq_len_kv"])
print("VISIBLE_FRAC", float(((ke - ks).double().clamp(min=0).sum())
                            / (CASE["seq_len"] * CASE["seq_len_kv"])))
ks, ke = ks.to(dev), ke.to(dev)
idx_q = torch.randn(CASE["batch"], CASE["seq_len"], CASE["heads"], CASE["index_dim"],
                    device=dev, dtype=torch.bfloat16)
idx_k = torch.randn(CASE["batch"], CASE["seq_len_kv"], CASE["kv_group"], CASE["index_dim"],
                    device=dev, dtype=torch.bfloat16)
w = torch.randn(CASE["seq_len"], CASE["heads"], device=dev, dtype=torch.float32)
scale = torch.rand(CASE["batch"], CASE["seq_len_kv"], CASE["kv_group"], device=dev,
                   dtype=torch.float32) + 0.5
logits = torch.empty(CASE["batch"], CASE["seq_len"], CASE["seq_len_kv"], CASE["kv_group"],
                     device=dev, dtype=torch.float32)
fac = getattr(_mod, {factory!r})
fn = fac(CASE["batch"], CASE["seq_len"], CASE["heads"], CASE["index_dim"],
         CASE["seq_len_kv"], CASE["kv_group"], True, True)(**CONFIG)
args = (idx_q.view(CASE["batch"], CASE["seq_len"] * CASE["heads"], CASE["index_dim"]),
        idx_k, scale, logits, w, ks, ke)
torch.npu.synchronize()
for _ in range({warm!r}):
    fn(*args)
torch.npu.synchronize()
for _ in range({launch!r}):
    fn(*args)
torch.npu.synchronize()
print("INNER_DONE")
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    ap.add_argument("--factory", default="_fp8_lightning_indexer_kernel")
    ap.add_argument("--config", default="{}")
    ap.add_argument("--candidate-id", default="candidate")
    ap.add_argument("--phase", default="candidate")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--parent-id", default="")
    ap.add_argument("--launch-count", type=int, default=2)
    ap.add_argument("--warm-up", type=int, default=2)
    ap.add_argument("--records", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    case = dict(WORKLOADS[args.workload])
    case.pop("label")
    config = json.loads(args.config)
    module = args.module if os.path.isabs(args.module) else os.path.join(HERE, args.module)
    tag = args.tag or f"{args.workload}_{args.candidate_id}"
    log_dir = os.path.join(HERE, "logs", args.phase)
    os.makedirs(log_dir, exist_ok=True)
    inner = os.path.join(log_dir, f"{tag}_inner.py")
    with open(inner, "w") as f:
        f.write(INNER.format(module=module, case=case, config=config,
                             warm=args.warm_up, launch=args.launch_count, factory=args.factory))

    prof_dir = os.path.join(HERE, "profiles", args.phase, args.workload, args.candidate_id)
    if os.path.isdir(prof_dir):
        subprocess.run(["rm", "-rf", prof_dir], check=False)
    os.makedirs(prof_dir, exist_ok=True)
    for d in (prof_dir, os.path.dirname(prof_dir), os.path.dirname(os.path.dirname(prof_dir))):
        try:
            os.chmod(d, 0o700)
        except OSError:
            pass

    cmd = ["msprof", "op", "--kernel-name=_fp8_lightning_indexer_main",
           f"--output={prof_dir}", f"--launch-count={args.launch_count}",
           f"--warm-up={args.warm_up}", "--dump=off", "--aic-metrics=Roofline",
           sys.executable, inner]
    log_path = os.path.join(log_dir, f"{tag}.log")
    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
    elapsed = time.time() - t0

    durations = []
    for c in glob.glob(os.path.join(prof_dir, "**", "OpBasicInfo*.csv"), recursive=True):
        with open(c, newline="") as f:
            for row in csv.DictReader(f):
                v = (row.get("Task Duration(us)") or "").strip()
                if v:
                    try:
                        durations.append(float(v))
                    except ValueError:
                        pass
    if not durations:
        print(f"NO_DURATION rc={proc.returncode} log={log_path}")
        with open(log_path) as lf:
            print(lf.read()[-4000:])
        raise SystemExit(2)
    durations.sort()
    median = durations[len(durations) // 2]

    roofs = []
    for b in glob.glob(os.path.join(prof_dir, "**", "visualize_data.bin"), recursive=True):
        try:
            data = parse_bin_file(b)
        except Exception as e:  # pragma: no cover
            print("BIN_PARSE_FAIL", b, e)
            continue
        for e in data.get("roofline_entries", []):
            if e.get("title") == "GM/L2" and "GM Read + Write" in e.get("bw_name", ""):
                roofs.append(dict(ratio=e["ratio"] * 100, AI=e["AI"], perf=e["performance"],
                                  computility=e["computility"], bw=e["bw"],
                                  pinfo=data.get("base_info", {})))
    opprof = sorted(glob.glob(os.path.join(prof_dir, "OPPROF_*")))
    detail = {}
    if opprof:
        try:
            detail = compute_from_csvs(opprof[0], DEFAULT_CONFIG)
        except Exception as e:  # pragma: no cover
            print("CSV_DETAIL_FAIL", e)

    out = dict(candidate_id=args.candidate_id, workload=args.workload, config=config,
               duration_us=median, min_us=durations[0], max_us=durations[-1],
               samples=len(durations), elapsed_s=round(elapsed, 1),
               raw_profile_dir=os.path.relpath(opprof[0] if opprof else prof_dir, HERE))
    if roofs:
        out["roofline"] = roofs[0]
        print("RATIO " + json.dumps(out))
    else:
        print("RATIO_UNAVAILABLE " + json.dumps(out))
    keep = ("cube_fops", "vec_fops", "total_fops", "cube_computility", "vec_computility",
            "combined_computility", "total_bytes", "AI", "performance", "computility",
            "ratio", "freq", "block_dim", "sum_cube_ratio", "sum_vec_cycles",
            "fp32_prop", "misc_prop")
    print("DETAIL " + json.dumps({k: detail[k] for k in keep if k in detail}))
    record = dict(round=args.round, candidate_id=args.candidate_id,
                  parent_id=args.parent_id, kernel_id=KERNEL_ID,
                  workload_id=WORKLOADS[args.workload]["label"], phase=args.phase,
                  duration_us=float(median),
                  msprof_raw_path=out["raw_profile_dir"],
                  timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if args.records:
        with open(os.path.join(HERE, args.records), "a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
