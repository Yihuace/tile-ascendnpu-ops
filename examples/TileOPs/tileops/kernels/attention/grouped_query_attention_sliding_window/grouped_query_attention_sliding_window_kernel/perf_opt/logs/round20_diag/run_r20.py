"""Round-20 diagnostic runner: repair the msprof metric caliber, then re-collect
3 workloads on the *current best* kernel (no kernel change).

Usage:
  python run_r20.py <impl_path> <tag_prefix> <out_json> <case:dtype> [...]

Every measurement: `msprof op`, `--aic-metrics=BasicInfo,PipeUtilization,
ArithmeticUtilization,Memory,Roofline`, `--launch-count=20 --warm-up=5`,
`Task Duration(us)` median of the 20 launch rows -- the only latency caliber.

Per-case wallclock is bounded (COMPILE_LAUNCH_TIMEOUT_S); a stuck compile is
killed by the outer `subprocess.run(timeout=)` of bench.py's msprof call plus
this runner's own hard cap.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PERF = os.path.dirname(os.path.dirname(HERE))  # .../<op>/perf_opt

CASE_TIMEOUT_S = 900


def main():
    impl, tag_prefix, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    cases = []
    for item in sys.argv[4:]:
        case, dt = item.split(":")
        cases.append((case, dt))
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for case, dt in cases:
        key = f"{case}/{dt}"
        if res.get(key, {}).get("median_us"):
            print(f"skip {key} (already {res[key]['median_us']:.2f} us)", flush=True)
            continue
        tag = f"{tag_prefix}_{case}_{dt}"
        cfg = dict(
            impl=os.path.abspath(impl), case=case, dtype=dt,
            outdir=os.path.join(PERF, "profiles", "round20_diag", tag),
            logdir=HERE, tag=tag, warmup=5, launch=20,
            kname="_gqa_sw_fwd_main",
        )
        path = os.path.join(HERE, f"cfg_{tag}.json")
        with open(path, "w") as f:
            json.dump(cfg, f)
        try:
            out = subprocess.run(
                [sys.executable, os.path.join(PERF, "bench.py"), "run", "--cfg", path],
                capture_output=True, text=True, timeout=CASE_TIMEOUT_S,
            )
            stdout, stderr, rc = out.stdout, out.stderr, out.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = f"timeout after {CASE_TIMEOUT_S}s"
            rc = -1
        got = None
        for ln in stdout.splitlines():
            if ln.startswith("RESULT "):
                got = json.loads(ln[7:])
        if got is None or not got.get("median_us"):
            print(f"FAIL {key} rc={rc}:\n{stdout[-1200:]}\n{stderr[-1200:]}", flush=True)
            continue
        res[key] = got
        with open(out_json, "w") as f:
            json.dump(res, f, indent=1)
        print(f"{key}: median {got['median_us']:.2f} us "
              f"[{got['min_us']:.2f},{got['max_us']:.2f}] n={got['n_launch']} "
              f"wall {got['wall_s']}s | aic_cube {got['aic_cube_ratio']} "
              f"aic_mte2 {got['aic_mte2_ratio']} aiv_vec {got['aiv_vec_ratio']} "
              f"aic_wait {got['aic_flag_wait_sum(us)']} aiv_wait {got['aiv_flag_wait_sum(us)']}",
              flush=True)


if __name__ == "__main__":
    main()
