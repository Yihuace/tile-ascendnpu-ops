"""Stage-4 lean msprof driver: kernel-only Task Duration for the varlen GQA-SW kernel.

Roles (same shape as the sibling argmax bench.py, adapted to the varlen op):

  python bench.py launch --cfg cfg.json      # child: build + launch the kernel
  python bench.py run    --cfg cfg.json      # parent: spawn `msprof op`, parse

Why a private driver (DESIGN 12.1 / opt_log 10.1 item 2): `Op.forward()` runs a
host-side validation operator and allocates the workspaces on every call, which
pollutes the msprof capture window (report `invalid_profile_count>0`).  This
driver calls the JIT'ed prim_func closure directly with pre-allocated
workspaces, so the captured op list contains only
`_gqa_sw_fwd_varlen_main_mix_aic`.

Metric contract: `Task Duration(us)` from `OpBasicInfo*.csv`, median of
`--launch-count` launches after `--warm-up` warm-ups.  This is the only latency
caliber used by Stage 4 (no NPU events, no end-to-end timing).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time

PERF_DIR = os.path.dirname(os.path.abspath(__file__))

# benchmark workloads (examples/TileOPs/benchmarks/ops/bench_grouped_query_attention_sliding_window_varlen.py)
CASES = {
    "8b-long": dict(batch=4, q=2048, k=2048, heads=32, heads_kv=8, dim=128,
                    causal=True, wl=1024, wr=-1),
    "70b-long": dict(batch=4, q=2048, k=2048, heads=64, heads_kv=8, dim=128,
                     causal=True, wl=1024, wr=-1),
    "8b-short": dict(batch=4, q=512, k=512, heads=32, heads_kv=8, dim=128,
                     causal=True, wl=256, wr=-1),
    "70b-short": dict(batch=4, q=512, k=512, heads=64, heads_kv=8, dim=128,
                      causal=True, wl=256, wr=-1),
    "smoke": dict(batch=2, q=128, k=128, heads=8, heads_kv=2, dim=64,
                  causal=True, wl=-1, wr=-1),
}

WORKLOAD_ID = {
    ("8b-long", "float16"): "llama-3.1-8b-long-w1024-float16",
    ("8b-long", "bfloat16"): "llama-3.1-8b-long-w1024-bfloat16",
    ("70b-long", "float16"): "llama-3.1-70b-long-w1024-float16",
    ("70b-long", "bfloat16"): "llama-3.1-70b-long-w1024-bfloat16",
    ("8b-short", "float16"): "llama-3.1-8b-short-w256-float16",
    ("8b-short", "bfloat16"): "llama-3.1-8b-short-w256-bfloat16",
    ("70b-short", "float16"): "llama-3.1-70b-short-w256-float16",
    ("70b-short", "bfloat16"): "llama-3.1-70b-short-w256-bfloat16",
    ("smoke", "float16"): "gqa-sw-varlen-fwd-smoke-float16",
    ("smoke", "bfloat16"): "gqa-sw-varlen-fwd-smoke-bfloat16",
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
    batch, d = c["batch"], c["dim"]
    sq, sk = [c["q"]] * batch, [c["k"]] * batch
    total_q, total_k = sum(sq), sum(sk)
    cu_q = torch.tensor([0] + list(torch.tensor(sq).cumsum(0).tolist()), dtype=torch.int32)
    cu_k = torch.tensor([0] + list(torch.tensor(sk).cumsum(0).tolist()), dtype=torch.int32)
    torch.manual_seed(0)
    q = torch.randn((total_q, c["heads"], d), dtype=torch_dtype, device="npu")
    k = torch.randn((total_k, c["heads_kv"], d), dtype=torch_dtype, device="npu")
    v = torch.randn((total_k, c["heads_kv"], d), dtype=torch_dtype, device="npu")
    cu_q_npu, cu_k_npu = cu_q.npu(), cu_k.npu()

    fn = mod._gqa_sw_fwd_varlen_kernel(
        batch, c["heads"], c["heads_kv"], total_q, total_k, d,
        c["causal"], c["wl"], c["wr"], dt_str,
    )
    # (64, 64, 1) -> the file's design default (64, 256) substitution path.
    wrapped = fn(64, 64, 1)
    msq = max(sq)
    t0 = time.perf_counter()
    wrapped(q, k, v, cu_q_npu, cu_k_npu, msq)   # first call: JIT compile
    torch.npu.synchronize()
    print(f"[bench] first call (compile) {time.perf_counter() - t0:.1f}s", flush=True)

    # Reach into the closure to reuse the compiled kernel + pre-allocate the
    # workspaces once (avoids per-call torch.empty inside the capture window).
    cells = {n: cl.cell_contents for n, cl in
             zip(wrapped.__code__.co_freevars, wrapped.__closure__)}
    kernel, meta = cells["_compile"](int(msq))
    dev = q.device
    ws_s = torch.zeros(meta["ws_s_shape"], dtype=meta["torch_ws_s"], device=dev)
    ws_p = torch.zeros(meta["ws_p_shape"], dtype=meta["torch_dtype"], device=dev)
    ws_o = torch.zeros(meta["ws_o_shape"], dtype=meta["torch_ws_o"], device=dev)
    zbuf = torch.zeros((meta["bn_eff"], d), dtype=meta["torch_dtype"], device=dev)

    warmup, launch = cfg.get("warmup", 5), cfg.get("launch", 20)
    for _ in range(warmup):
        out, lse = kernel(q, k, v, cu_q_npu, cu_k_npu, ws_s, ws_p, ws_o, zbuf)
    torch.npu.synchronize()
    for _ in range(launch):
        out, lse = kernel(q, k, v, cu_q_npu, cu_k_npu, ws_s, ws_p, ws_o, zbuf)
    torch.npu.synchronize()
    print("LAUNCH_OK", tuple(out.shape), out.dtype, tuple(lse.shape),
          "finite:", bool(torch.isfinite(out).all().item()),
          flush=True)
    if cfg.get("dump"):
        path = cfg["dump"]
        torch.save({"out": out.float().cpu(), "lse": lse.cpu()}, path)
        print("[bench] dumped", path, flush=True)


# --------------------------------------------------------------- parent side
def _parse_op_basic(d):
    rows = []
    for root, _dirs, files in os.walk(d):
        for f in files:
            if not re.fullmatch(r"OpBasicInfo.*\.csv", f):
                continue
            path = os.path.join(root, f)
            with open(path) as fh:
                lines = [ln for ln in fh.read().strip().splitlines() if ln]
            if len(lines) < 2:
                continue
            header = lines[0].split(",")
            rows.extend(dict(zip(header, ln.split(","))) for ln in lines[1:])
    return rows


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
        f"--kernel-name={cfg.get('kname', '_gqa_sw_fwd_varlen_main')}",
        f"--output={outdir}",
        f"--launch-count={cfg.get('launch', 20)}",
        f"--warm-up={cfg.get('warmup', 5)}",
        "--aic-metrics=Default",
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

    rows = _parse_op_basic(outdir)
    if not rows:
        print(json.dumps({"error": "no OpBasicInfo", "tag": tag,
                          "stderr": res.stderr[-2500:], "stdout": res.stdout[-1500:]}))
        return 1
    durs = sorted(float(r["Task Duration(us)"]) for r in rows if r.get("Task Duration(us)"))
    r0 = rows[0]

    def g(k):
        try:
            return float(r0[k])
        except Exception:
            return None

    out = {
        "tag": tag,
        "workload_id": WORKLOAD_ID.get((cfg["case"], cfg["dtype"]), cfg["case"]),
        "n": len(durs),
        "median_us": durs[len(durs) // 2],
        "min_us": durs[0],
        "max_us": durs[-1],
        "mean_us": round(sum(durs) / len(durs), 2),
        "wall_s": round(time.perf_counter() - t0, 1),
        "block_dim": r0.get("Block Dim"),
        # pipe occupancy ratios / times (msprof "Default" aic metrics)
        "aic_mte2_ratio": g("aic_mte2_ratio"),
        "aic_cube_ratio": g("aic_cube_ratio"),
        "aic_fixpipe_ratio": g("aic_fixpipe_ratio"),
        "aic_scalar_ratio": g("aic_scalar_ratio"),
        "aic_mte2_time(us)": g("aic_mte2_time(us)"),
        "aic_cube_time(us)": g("aic_cube_time(us)"),
        "aic_fixpipe_time(us)": g("aic_fixpipe_time(us)"),
        "aic_scalar_time(us)": g("aic_scalar_time(us)"),
        "aiv_vec_ratio": g("aiv_vec_ratio"),
        "aiv_scalar_ratio": g("aiv_scalar_ratio"),
        "gm_to_l1(KB)": g("gm_to_l1(KB)"),
        "l1_to_l0a(KB)": g("l1_to_l0a(KB)"),
        "l1_to_l0b(KB)": g("l1_to_l0b(KB)"),
        "l0c_to_gm(KB)": g("l0c_to_gm(KB)"),
        "gm_to_ub(KB)": g("gm_to_ub(KB)"),
        "aic_mte2_instructions": g("aic_mte2_instructions"),
        "aic_fixpipe_instructions": g("aic_fixpipe_instructions"),
        "aic_cube_instructions": g("aic_cube_instructions"),
    }
    print("RESULT " + json.dumps(out))
    return 0


if __name__ == "__main__":
    which = sys.argv[1]
    with open(sys.argv[3]) as f:
        cfg = json.load(f)
    if which == "launch":
        _launch(cfg)
        sys.exit(0)
    sys.exit(_run(cfg))
