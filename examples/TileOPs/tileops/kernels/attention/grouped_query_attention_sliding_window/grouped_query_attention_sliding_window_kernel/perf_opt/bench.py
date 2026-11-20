"""Stage-4 lean msprof driver: kernel-only Task Duration for the fixed GQA-SW kernel.

Roles (same shape as the varlen sibling bench.py, adapted to the fixed op):

  python bench.py launch --cfg cfg.json      # child: build + launch the kernel
  python bench.py run    --cfg cfg.json      # parent: spawn `msprof op`, parse

Why a private driver (DESIGN 12.1 / opt_log 10.1 item 2): `Op.forward()` runs a
host-side validation operator and allocates the workspaces on every call, which
pollutes the msprof capture window (report `invalid_profile_count>0`).  This
driver calls the JIT'ed prim_func closure directly with pre-allocated
workspaces, so the captured op list contains only `_gqa_sw_fwd_main_mix_aic`.

Metric contract: `Task Duration(us)` from `OpBasicInfo*.csv`, median of
`--launch-count` launches after `--warm-up` warm-ups.  This is the only latency
caliber used by Stage 4 (no NPU events, no end-to-end timing).

Round-20 caliber repair (why the R19 "blind spot" was a *parser* bug, not a
toolchain limit): `--aic-metrics=Default` on CANN 9.0.0 / msprof 25.5.0 *does*
emit `PipeUtilization_*.csv` / `ArithmeticUtilization_*.csv` / `Memory_*.csv`
(the R19 raw dirs contain them -- one row per (block_id, sub_block_id), values
already averaged over the 20 launches).  The old code only walked
`OpBasicInfo*.csv` and therefore always read `None` for every `aic_*`/`aiv_*`
key.  This version requests an explicit metrics list (the one proven on the
same 910B3/CANN by the GQA-bwd Stage-4 run) and parses every table, grouping by
`sub_block_id` (`cube0` / `vector0` / `vector1`) and taking the median over the
20 cores.

  python bench.py parse --cfg cfg.json       # parent: re-parse an existing dir
"""

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

PERF_DIR = os.path.dirname(os.path.abspath(__file__))

# Explicit metric list -- identical to the list proven on this 910B3 / CANN
# 9.0.0 by the GQA-bwd Stage-4 pipeline.  `PipeUtilization` is the table that
# carries `aic_*` / `aiv_*` ratios and the `*_scalar_wait_id*` flag breakdown.
MSPROF_METRICS = "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,Roofline"

# msprof op table stems (file = <stem>_<timestamp>.csv)
METRIC_TABLES = (
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "ResourceConflictRatio",
)

# flag ids whose wait time we break out (0..9 cover both slots for L_max<=5;
# flags 10..15 exist in the 16-id budget but are unused by this kernel)
FLAG_IDS = tuple(range(10))

# benchmark workloads (examples/TileOPs/tileops/manifest/attention.yaml)
CASES = {
    "8b-long": dict(batch=2, seq=2048, heads=32, heads_kv=8, dim=128,
                    causal=True, wl=1024, wr=-1),
    "70b-long": dict(batch=1, seq=2048, heads=64, heads_kv=8, dim=128,
                     causal=True, wl=1024, wr=-1),
    "8b-short": dict(batch=4, seq=512, heads=32, heads_kv=8, dim=128,
                     causal=True, wl=256, wr=-1),
    "70b-short": dict(batch=2, seq=512, heads=64, heads_kv=8, dim=128,
                      causal=True, wl=256, wr=-1),
    "smoke": dict(batch=2, seq=512, heads=8, heads_kv=2, dim=64,
                  causal=True, wl=-1, wr=-1),
    # Round-20 diagnostic probe (NOT a manifest workload, never recorded as a
    # tune workload): S=1024/wl=512 sits between the short (S=512/wl=256,
    # avg band L=1.5) and long (S=2048/wl=1024, avg L=3.75) families and is
    # used only to validate the per-task + per-band-iter cost model.
    "probe-mid": dict(batch=4, seq=1024, heads=32, heads_kv=8, dim=128,
                      causal=True, wl=512, wr=-1),
}

WORKLOAD_ID = {
    ("8b-long", "float16"): "llama-3.1-8b-long-w1024-fp16",
    ("8b-long", "bfloat16"): "llama-3.1-8b-long-w1024-bf16",
    ("70b-long", "float16"): "llama-3.1-70b-long-w1024-fp16",
    ("70b-long", "bfloat16"): "llama-3.1-70b-long-w1024-bf16",
    ("8b-short", "float16"): "llama-3.1-8b-short-w256-fp16",
    ("8b-short", "bfloat16"): "llama-3.1-8b-short-w256-bf16",
    ("70b-short", "float16"): "llama-3.1-70b-short-w256-fp16",
    ("70b-short", "bfloat16"): "llama-3.1-70b-short-w256-bf16",
    ("smoke", "float16"): "gqa-sw-fwd-smoke-s512-h8-kv2-d64-fp16",
    ("smoke", "bfloat16"): "gqa-sw-fwd-smoke-s512-h8-kv2-d64-bf16",
}


# ---------------------------------------------------------------- child side
def _launch(cfg):
    import importlib.util

    import torch
    import torch_npu  # noqa: F401

    spec = importlib.util.spec_from_file_location("impl_mod", cfg["impl"])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["impl_mod"] = mod
    spec.loader.exec_module(mod)

    c = CASES[cfg["case"]]
    torch_dtype = getattr(torch, cfg["dtype"])
    dt_str = "float16" if torch_dtype is torch.float16 else "bfloat16"
    torch.manual_seed(0)
    q = torch.randn((c["batch"], c["seq"], c["heads"], c["dim"]),
                    dtype=torch_dtype, device="npu")
    k = torch.randn((c["batch"], c["seq"], c["heads_kv"], c["dim"]),
                    dtype=torch_dtype, device="npu")
    v = torch.randn((c["batch"], c["seq"], c["heads_kv"], c["dim"]),
                    dtype=torch_dtype, device="npu")

    fn = mod._gqa_sw_fwd_kernel(
        c["batch"], c["heads"], c["heads_kv"], c["seq"], c["dim"],
        c["causal"], c["wl"], c["wr"], dt_str,
    )
    # (64, 64, 1) -> the file's design default (64, 256) substitution path.
    wrapped = fn(64, 64, 1)
    t0 = time.perf_counter()
    out, lse = wrapped(q, k, v)   # first call: JIT compile
    torch.npu.synchronize()
    print(f"[bench] first call (compile) {time.perf_counter() - t0:.1f}s", flush=True)

    # Reach into the closure to reuse the compiled kernel + pre-allocate the
    # workspaces once (avoids per-call torch.empty inside the capture window).
    cells = {n: cl.cell_contents for n, cl in
             zip(wrapped.__code__.co_freevars, wrapped.__closure__)}
    kernel = cells["kernel"]
    meta = {
        "ws_s_shape": list(cells["ws_s_shape"]),
        "ws_p_shape": list(cells["ws_p_shape"]),
        "ws_o_shape": list(cells["ws_o_shape"]),
        "torch_ws_s": cells["torch_ws_s"],
        "torch_ws_o": cells["torch_ws_o"],
        "torch_dtype": cells["torch_dtype"],
        "bn_eff": int(cells["bn_eff"]),
    }
    dev = q.device
    ws_s = torch.zeros(meta["ws_s_shape"], dtype=meta["torch_ws_s"], device=dev)
    ws_p = torch.zeros(meta["ws_p_shape"], dtype=meta["torch_dtype"], device=dev)
    ws_o = torch.zeros(meta["ws_o_shape"], dtype=meta["torch_ws_o"], device=dev)
    zbuf = torch.zeros((meta["bn_eff"], c["dim"]), dtype=meta["torch_dtype"], device=dev)

    warmup, launch = cfg.get("warmup", 5), cfg.get("launch", 20)
    for _ in range(warmup):
        out, lse = kernel(q, k, v, ws_s, ws_p, ws_o, zbuf)
    torch.npu.synchronize()
    for _ in range(launch):
        out, lse = kernel(q, k, v, ws_s, ws_p, ws_o, zbuf)
    torch.npu.synchronize()
    print("LAUNCH_OK", tuple(out.shape), out.dtype, tuple(lse.shape),
          "finite:", bool(torch.isfinite(out).all().item()),
          "meta:", json.dumps(meta), flush=True)
    if cfg.get("dump"):
        path = cfg["dump"]
        torch.save({"out": out.float().cpu(), "lse": lse.cpu()}, path)
        print("[bench] dumped", path, flush=True)


# --------------------------------------------------------------- parent side
def _table_rows(d, stem):
    """All rows of every `<stem>_<ts>.csv` under `d` (recursive)."""
    rows = []
    pat = re.compile(re.escape(stem) + r"_.*\.csv")
    for root, _dirs, files in os.walk(d):
        for f in files:
            if not pat.fullmatch(f):
                continue
            path = os.path.join(root, f)
            with open(path) as fh:
                lines = [ln for ln in fh.read().strip().splitlines() if ln]
            if len(lines) < 2:
                continue
            header = lines[0].split(",")
            rows.extend(dict(zip(header, ln.split(","))) for ln in lines[1:])
    return rows


def _num(s):
    if s is None:
        return None
    s = s.strip()
    if s in ("", "NA"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _group_median(rows):
    """`sub_block_id` -> {column: median over block rows}.

    msprof writes one row per (block_id, sub_block_id) with the value already
    averaged over the launches of the capture; the median over the 20 cores is
    the per-core representative value.
    """
    groups = {}
    for r in rows:
        groups.setdefault(r.get("sub_block_id", "-"), []).append(r)
    out = {}
    for sb, g in groups.items():
        cols = {}
        for k in g[0]:
            if k in ("block_id", "sub_block_id"):
                continue
            vals = [v for r in g if (v := _num(r.get(k))) is not None]
            if vals:
                cols[k] = round(statistics.median(vals), 4)
        out[sb] = cols
    return out


def _flag_wait(cols, side):
    """Sum of the per-flag-id wait columns plus the per-id breakdown."""
    per_id = {}
    for i in FLAG_IDS:
        v = cols.get(f"{side}_scalar_wait_id{i}_time(us)")
        if v is not None:
            per_id[f"id{i}"] = v
    return round(sum(per_id.values()), 4), per_id


def _tables(d):
    return {stem: _group_median(_table_rows(d, stem)) for stem in METRIC_TABLES}


def _summarize(d):
    """Flat + nested view of one raw msprof op profile directory."""
    ob = _table_rows(d, "OpBasicInfo")
    durs = [v for r in ob if (v := _num(r.get("Task Duration(us)"))) is not None]
    out = {
        "n_launch": len(durs),
        "median_us": round(statistics.median(durs), 2) if durs else None,
        "min_us": round(min(durs), 2) if durs else None,
        "max_us": round(max(durs), 2) if durs else None,
        "mean_us": round(statistics.fmean(durs), 2) if durs else None,
        "block_dim": ob[0].get("Block Dim") if ob else None,
    }
    tables = _tables(d)
    out["pipe"] = tables["PipeUtilization"]
    out["arith"] = tables["ArithmeticUtilization"]
    out["mem"] = tables["Memory"]

    cube = out["pipe"].get("cube0", {})
    vec = out["pipe"].get("vector0", {})
    dur = out["median_us"] or 0.0

    # ---- AIC side (cube0) -------------------------------------------------
    out["aic_cube_ratio"] = cube.get("aic_cube_ratio")
    out["aic_mte1_ratio"] = cube.get("aic_mte1_ratio")
    out["aic_mte2_ratio"] = cube.get("aic_mte2_ratio")
    out["aic_fixpipe_ratio"] = cube.get("aic_fixpipe_ratio")
    out["aic_scalar_ratio"] = cube.get("aic_scalar_ratio")
    out["aic_cube_time(us)"] = cube.get("aic_cube_time(us)")
    out["aic_mte1_time(us)"] = cube.get("aic_mte1_time(us)")
    out["aic_mte2_time(us)"] = cube.get("aic_mte2_time(us)")
    out["aic_fixpipe_time(us)"] = cube.get("aic_fixpipe_time(us)")
    out["aic_scalar_time(us)"] = cube.get("aic_scalar_time(us)")
    aic_flag, aic_per_id = _flag_wait(cube, "aic")
    out["aic_flag_wait_sum(us)"] = aic_flag
    out["aic_flag_wait_by_id"] = aic_per_id
    out["aic_flag_wait_pct_vs_dur"] = (
        round(100.0 * aic_flag / dur, 2) if dur else None)

    # ---- AIV side (vector0; vector1 carries the identical program) -------
    vec1 = out["pipe"].get("vector1", {})
    out["aiv_vec_ratio"] = vec.get("aiv_vec_ratio")
    out["aiv_mte2_ratio"] = vec.get("aiv_mte2_ratio")
    out["aiv_mte3_ratio"] = vec.get("aiv_mte3_ratio")
    out["aiv_scalar_ratio"] = vec.get("aiv_scalar_ratio")
    out["aiv_vec_time(us)"] = vec.get("aiv_vec_time(us)")
    out["aiv_mte2_time(us)"] = vec.get("aiv_mte2_time(us)")
    out["aiv_mte3_time(us)"] = vec.get("aiv_mte3_time(us)")
    out["aiv_scalar_time(us)"] = vec.get("aiv_scalar_time(us)")
    out["aiv_scalar_wait_time(us)"] = vec.get("aiv_scalar_wait_time(us)")
    out["aiv_scalar_wait_pct_vs_dur"] = (
        round(100.0 * vec["aiv_scalar_wait_time(us)"] / dur, 2)
        if dur and vec.get("aiv_scalar_wait_time(us)") is not None else None)
    aiv_flag, aiv_per_id = _flag_wait(vec, "aiv")
    out["aiv_flag_wait_sum(us)"] = aiv_flag
    out["aiv_flag_wait_by_id"] = aiv_per_id
    out["aiv_flag_wait_pct_vs_dur"] = (
        round(100.0 * aiv_flag / dur, 2) if dur else None)
    out["aiv_scalar_vector_stall_time(us)"] = vec.get("aiv_scalar_vector_stall_time(us)")
    out["aiv_scalar_mte2_stall_time(us)"] = vec.get("aiv_scalar_mte2_stall_time(us)")
    # vector1 must mirror vector0 (Developer-mode duplicate AIV program)
    out["aiv_vec_ratio_v1"] = vec1.get("aiv_vec_ratio")

    # ---- GM traffic (Memory table, KB per core per capture) --------------
    cm = out["mem"].get("cube0", {})
    vm = out["mem"].get("vector0", {})
    out["gm_to_l1(KB)"] = cm.get("GM_to_L1_datas(KB)")
    out["gm_to_ub(KB)"] = vm.get("GM_to_UB_datas(KB)")
    out["l0c_to_gm(KB)"] = cm.get("L0C_to_GM_datas(KB)")
    out["l0c_to_l1(KB)"] = cm.get("L0C_to_L1_datas(KB)")
    out["ub_to_gm(KB)"] = vm.get("UB_to_GM_datas(KB)")
    out["read_main_mem(KB)"] = cm.get("read_main_memory_datas(KB)")
    out["write_main_mem(KB)"] = cm.get("write_main_memory_datas(KB)")
    out["gm_to_l1_bw_rate(%)"] = cm.get("GM_to_L1_bw_usage_rate(%)")
    out["gm_to_ub_bw_rate(%)"] = vm.get("GM_to_UB_bw_usage_rate(%)")

    # ---- arithmetic (ArithmeticUtilization) ------------------------------
    ca = out["arith"].get("cube0", {})
    va = out["arith"].get("vector0", {})
    out["aic_cube_fops"] = ca.get("aic_cube_fops")
    out["aiv_vec_fops"] = va.get("aiv_vec_fops")
    out["aic_cube_ratio_arith"] = ca.get("aic_cube_ratio")
    out["aiv_vec_ratio_arith"] = va.get("aiv_vec_ratio")
    return out


def _parse_op_basic(d):
    return _table_rows(d, "OpBasicInfo")


def _emit(cfg, out):
    tag = cfg.get("tag")
    logdir = cfg.get("logdir")
    if logdir:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, f"{tag}.json"), "w") as f:
            json.dump(out, f, indent=1)
    print("RESULT " + json.dumps(out))


def _parse(cfg):
    out = _summarize(cfg["outdir"])
    out["tag"] = cfg.get("tag")
    out["workload_id"] = WORKLOAD_ID.get((cfg["case"], cfg["dtype"]), cfg["case"])
    out["msprof_raw_path"] = cfg["outdir"]
    _emit(cfg, out)
    return 0 if out["median_us"] else 1


def _run(cfg):
    msprof = shutil.which("msprof")
    outdir = cfg["outdir"]
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    os.chmod(outdir, 0o700)
    cfg_path = os.path.join(outdir, "cfg.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    cmd = [
        msprof, "op",
        f"--kernel-name={cfg.get('kname', '_gqa_sw_fwd_main')}",
        f"--output={outdir}",
        f"--launch-count={cfg.get('launch', 20)}",
        f"--warm-up={cfg.get('warmup', 5)}",
        f"--aic-metrics={cfg.get('metrics', MSPROF_METRICS)}",
        "--dump=off",
        sys.executable, os.path.join(PERF_DIR, "bench.py"), "launch",
        "--cfg", cfg_path,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.getcwd(), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    logdir = cfg["logdir"]
    os.makedirs(logdir, exist_ok=True)
    tag = cfg["tag"]
    with open(os.path.join(logdir, f"{tag}.log"), "w") as f:
        f.write(" ".join(cmd) + "\n\n--- STDOUT ---\n" + res.stdout
                + "\n--- STDERR ---\n" + res.stderr)

    out = _summarize(outdir)
    out["tag"] = tag
    out["workload_id"] = WORKLOAD_ID.get((cfg["case"], cfg["dtype"]), cfg["case"])
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    out["msprof_raw_path"] = outdir
    if not out["n_launch"]:
        out["error"] = "no OpBasicInfo"
        out["stderr_tail"] = res.stderr[-2500:]
        _emit(cfg, out)
        return 1
    _emit(cfg, out)
    return 0


if __name__ == "__main__":
    which = sys.argv[1]
    with open(sys.argv[3]) as f:
        cfg = json.load(f)
    if which == "launch":
        _launch(cfg)
        sys.exit(0)
    if which == "parse":
        sys.exit(_parse(cfg))
    sys.exit(_run(cfg))
