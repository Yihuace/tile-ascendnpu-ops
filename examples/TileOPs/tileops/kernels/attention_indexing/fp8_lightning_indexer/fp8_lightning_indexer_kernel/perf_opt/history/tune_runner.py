"""Stage 4 experiment runner for FP8LightningIndexerOp NPU kernels.

Measures a single (candidate file, workload, tiling config) triple with
``msprof op`` (the only latency metric used by this Stage) and appends one
augmented record to a JSONL file (append-only, fields per
``_shared/standards/signal-registry.md`` §5).

Usage::

    python tune_runner.py --module <candidate.py> --workload s8k \
        --block_N 64 --block_Q 8 --num_stages 2 \
        --round 1 --candidate-id r1_bq8 --phase candidate \
        --records perf_records.jsonl --launch-count 5

The inner launch script is written under ``perf_opt/logs/<phase>/`` so all
stdout/stderr stays out of the ``perf_opt/`` top level.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_DIR = os.path.dirname(HERE)

# Benchmark workload ids (from benchmarks/ops/bench_fp8_lightning_indexer.py
# manifest parameterisation -- deduped on shape, so the fp8 row collapses into
# the bf16 s8k row and is not a separate case).
WORKLOADS = {
    "smoke": dict(batch=1, seq_len=1024, heads=8, index_dim=64, seq_len_kv=1024, kv_group=1,
                  label="lightning-indexer-smoke-s1k-h8-d64"),
    "s8k": dict(batch=1, seq_len=8192, heads=32, index_dim=64, seq_len_kv=8192, kv_group=1,
                label="lightning-indexer-bf16-s8k-h32-d64"),
    "s4k": dict(batch=1, seq_len=4096, heads=32, index_dim=64, seq_len_kv=8192, kv_group=1,
                label="lightning-indexer-bf16-s4k-h32-d64"),
}

AUX_WORKLOADS = {
    "clean_s8k": dict(batch=1, seq_len=8192, seq_len_kv=8192, kv_group=1,
                      label="clean-logits-s8k-g1", window="real"),
    "clean_s4k": dict(batch=1, seq_len=4096, seq_len_kv=8192, kv_group=1,
                      label="clean-logits-s4k-g1", window="real"),
    "clean_s8k_full": dict(batch=1, seq_len=8192, seq_len_kv=8192, kv_group=1,
                           label="clean-logits-s8k-g1-fullwin", window="full"),
}

KERNEL_ID = ("examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/"
             "fp8_lightning_indexer_kernel/_fp8_lightning_indexer_kernel.py"
             "::_fp8_lightning_indexer_kernel::_fp8_lightning_indexer_main")
AUX_KERNEL_ID = ("examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/"
                 "fp8_lightning_indexer_kernel/clean_logits_.py"
                 "::clean_logits_::clean_logits_kernel")

INNER_TEMPLATE = '''\
import os
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import torch
import torch_npu  # noqa: F401
import importlib.util

torch.npu.set_device(0)

_spec = importlib.util.spec_from_file_location("cand_mod", {module!r})
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CASE = {case!r}
CONFIG = {config!r}
WARM, LAUNCH = {warm!r}, {launch!r}

torch.manual_seed(0)
dev = "npu"


def _causal_windows(seq_len, seq_len_kv, avg=512, seed=0):
    """Faithful host-side copy of ``generate_random_cu_seqlens`` (cp=1, kv_stride=1)."""
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
        m = int(cu[i])
        base = int(cum[i] - cu[i])
        ks[start:start + m] = base
        ke[start:start + m] = torch.minimum(
            torch.arange(1, m + 1, dtype=torch.int32) + base,
            torch.tensor(int(cum[i]), dtype=torch.int32))
        start += m
    return ks.clamp_(0, seq_len_kv), ke.clamp_(0, seq_len_kv)


if CASE.get("window", "real") == "full":
    ks = torch.zeros(CASE["seq_len"], device=dev, dtype=torch.int32)
    ke = torch.full((CASE["seq_len"],), CASE["seq_len_kv"], device=dev, dtype=torch.int32)
else:
    ks, ke = _causal_windows(CASE["seq_len"], CASE["seq_len_kv"])
    ks, ke = ks.to(dev), ke.to(dev)
print("WINDOW", CASE.get("window", "real"),
      "visible_frac", float(((ke - ks).double().clamp(min=0).sum())
                            / (CASE["seq_len"] * CASE["seq_len_kv"])))

if "factory" in CASE:
    # clean_logits_ style factory(block_N, G_MAX) -> prim_func
    fn = getattr(_mod, CASE["factory"])(**CONFIG)
    logits = torch.zeros(CASE["batch"], CASE["seq_len"], CASE["seq_len_kv"], CASE["kv_group"],
                         device=dev, dtype=torch.float32)
    args = (logits, ks, ke)
else:
    idx_q = torch.randn(CASE["batch"], CASE["seq_len"], CASE["heads"], CASE["index_dim"],
                        device=dev, dtype=torch.bfloat16)
    idx_k = torch.randn(CASE["batch"], CASE["seq_len_kv"], CASE["kv_group"],
                        CASE["index_dim"], device=dev, dtype=torch.bfloat16)
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
for _ in range(WARM):
    fn(*args)
torch.npu.synchronize()
for _ in range(LAUNCH):
    fn(*args)
torch.npu.synchronize()
print("INNER_DONE")
'''


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_metrics(raw_dir: str) -> dict:
    """Aggregate the msprof per-block CSV rows into comparable scalars."""

    def _mean(stem: str, cols) -> dict:
        files = glob.glob(os.path.join(raw_dir, "**", f"{stem}*.csv"), recursive=True)
        if not files:
            return {}
        tot, cnt = {c: 0.0 for c in cols}, 0
        for path in files:
            try:
                with open(path, newline="") as f:
                    for row in csv.DictReader(f):
                        cnt += 1
                        for c in cols:
                            try:
                                tot[c] += float(row.get(c) or 0.0)
                            except (TypeError, ValueError):
                                pass
            except OSError:
                continue
        if not cnt:
            return {}
        return {c: tot[c] / cnt for c in cols}

    out = {}
    out.update(_mean("PipeUtilization",
                     ["aic_cube_ratio", "aiv_vec_ratio", "aic_mte2_ratio", "aiv_mte3_ratio",
                      "aic_scalar_ratio", "aiv_scalar_ratio", "aic_fixpipe_ratio"]))
    out.update(_mean("Memory",
                     ["aiv_main_mem_read_bw", "aiv_main_mem_write_bw", "aic_main_mem_read_bw",
                      "write_main_memory_datas(KB)", "read_main_memory_datas(KB)"]))
    return out


def measure(module: str, workload: str, config: dict, *, factory: str,
            phase: str, candidate_id: str, launch: int = 5, warm: int = 3,
            aux: bool = False) -> dict:
    cases = AUX_WORKLOADS if aux else WORKLOADS
    case = dict(cases[workload])
    label = case.pop("label")
    case.setdefault("window", "real")
    if aux:
        case["factory"] = factory

    run_id = f"{workload}_{candidate_id}"
    log_dir = os.path.join(HERE, "logs", phase)
    os.makedirs(log_dir, exist_ok=True)
    inner = os.path.join(log_dir, f"{run_id}_inner.py")
    with open(inner, "w") as f:
        f.write(INNER_TEMPLATE.format(module=module, case=case, config=config, warm=warm,
                                      launch=launch, factory=factory))

    prof_dir = os.path.join(HERE, "profiles", phase, workload, candidate_id)
    if os.path.isdir(prof_dir):
        subprocess.run(["rm", "-rf", prof_dir], check=False)
    os.makedirs(prof_dir, exist_ok=True)
    for d in (prof_dir, os.path.dirname(prof_dir), os.path.dirname(os.path.dirname(prof_dir))):
        try:
            os.chmod(d, 0o700)
        except OSError:
            pass

    kernel_name = "_fp8_lightning_indexer_main" if not aux else "clean_logits_kernel"
    cmd = ["msprof", "op", f"--kernel-name={kernel_name}", f"--output={prof_dir}",
           f"--launch-count={launch}", f"--warm-up={warm}", "--dump=off",
           "--aic-metrics=Default", sys.executable, inner]
    log_path = os.path.join(log_dir, f"{run_id}.log")
    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
    elapsed = time.time() - t0

    durations, op_names = [], []
    csvs = glob.glob(os.path.join(prof_dir, "**", "OpBasicInfo*.csv"), recursive=True)
    for c in csvs:
        with open(c, newline="") as f:
            for row in csv.DictReader(f):
                v = (row.get("Task Duration(us)") or "").strip()
                if v:
                    try:
                        durations.append(float(v))
                    except ValueError:
                        pass
                n = (row.get("Op Name") or "").strip()
                if n:
                    op_names.append(n)
    if not durations:
        raise RuntimeError(f"no OpBasicInfo durations under {prof_dir} (rc={proc.returncode}); "
                           f"see {log_path}")

    durations.sort()
    median = durations[len(durations) // 2]
    raw_root = os.path.dirname(csvs[0]) if csvs else prof_dir
    opprof = sorted(glob.glob(os.path.join(prof_dir, "OPPROF_*")))
    metric_root = opprof[0] if opprof else raw_root
    return dict(
        duration_us=median,
        sample_count=len(durations),
        min_us=durations[0],
        max_us=durations[-1],
        captured_op_name=sorted(set(op_names))[0] if op_names else "",
        raw_profile_dir=os.path.relpath(metric_root, HERE),
        command=" ".join(cmd),
        profile_status="valid",
        elapsed_s=round(elapsed, 1),
        metrics=_parse_metrics(metric_root),
    )


def append_record(records_path: str, *, round_, candidate_id, parent_id, kernel_id,
                  workload_id, phase, artifact_path, duration_us, raw_profile_path,
                  extra=None) -> dict:
    """Append one measurement row (append-only; never rewrite existing rows)."""
    art_abs = artifact_path if os.path.isabs(artifact_path) else os.path.join(HERE, artifact_path)
    rec = dict(
        round=round_,
        candidate_id=candidate_id,
        parent_id=parent_id,
        kernel_id=kernel_id,
        workload_id=workload_id,
        phase=phase,
        artifact_path=os.path.basename(artifact_path),
        artifact_sha256=_sha256(art_abs),
        duration_us=float(duration_us),
        l0_pass=bool((extra or {}).get("l0_pass", True)),
        msprof_raw_path=raw_profile_path,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(records_path, "a") as f:
        f.write(json.dumps(rec, sort_keys=False) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="candidate kernel .py path")
    ap.add_argument("--workload", required=True, choices=sorted(list(WORKLOADS) + list(AUX_WORKLOADS)))
    ap.add_argument("--factory", default="_fp8_lightning_indexer_kernel")
    ap.add_argument("--config", default="{}", help="JSON dict of prim_func config")
    ap.add_argument("--block_N", type=int, default=None)
    ap.add_argument("--block_Q", type=int, default=None)
    ap.add_argument("--num_stages", type=int, default=None)
    ap.add_argument("--hp_sub", type=int, default=None)
    ap.add_argument("--G_MAX", type=int, default=None)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--candidate-id", default="baseline")
    ap.add_argument("--parent-id", default="")
    ap.add_argument("--phase", default="candidate")
    ap.add_argument("--launch-count", type=int, default=5)
    ap.add_argument("--warm-up", type=int, default=3)
    ap.add_argument("--l0-pass", default="true")
    ap.add_argument("--records", default=os.path.join(HERE, "perf_records.jsonl"))
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    config = json.loads(args.config)
    for key in ("block_N", "block_Q", "num_stages", "G_MAX", "hp_sub"):
        val = getattr(args, key)
        if val is not None:
            config[key] = val

    aux = args.workload.startswith("clean_")
    module = args.module
    if not os.path.isabs(module):
        if os.path.exists(module):
            module = os.path.abspath(module)
        else:
            module = os.path.join(HERE, module)
    res = measure(module, args.workload, config, factory=args.factory, phase=args.phase,
                  candidate_id=args.candidate_id, launch=args.launch_count, warm=args.warm_up,
                  aux=aux)

    print(json.dumps(dict(candidate_id=args.candidate_id, workload=args.workload,
                          config=config, **res), indent=2))

    if not args.no_record:
        rec = append_record(
            args.records, round_=args.round, candidate_id=args.candidate_id,
            parent_id=args.parent_id,
            kernel_id=AUX_KERNEL_ID if aux else KERNEL_ID,
            workload_id=(AUX_WORKLOADS if aux else WORKLOADS)[args.workload]["label"],
            phase=args.phase, artifact_path=module, duration_us=res["duration_us"],
            raw_profile_path=res["raw_profile_dir"],
            extra=dict(l0_pass=args.l0_pass.lower() == "true"))
        print("RECORD " + json.dumps(rec))


if __name__ == "__main__":
    main()
