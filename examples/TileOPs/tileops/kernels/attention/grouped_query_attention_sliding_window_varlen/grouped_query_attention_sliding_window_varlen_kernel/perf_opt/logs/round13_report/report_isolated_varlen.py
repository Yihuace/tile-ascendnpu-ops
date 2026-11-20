"""Process-isolated equivalent of ``tileops.reporting.cli run --op ... --prof-mode msprof``.

Why this exists (toolchain defect, not a report change): this kernel's
``bishengir-compile``/JIT path is only stable when a process compiles ONE
trace.  In one pytest process the 14th correctness case and the 5th benchmark
case deadlock / SIGSEGV (documented in logs/round18b + Round 18/18b of
opt_log.md; the identical cases pass standalone).  This driver keeps the
official pipeline code (same collectors, analyzer, report renderer) and only
changes the pytest GRANULARITY: one process per collected test id, then the
per-case JUnit XML / benchmark JSON artefacts are merged byte-honestly into
the same layout ``run_operator`` produces.

Nothing is synthesised: every latency record is what the per-case pytest run
emitted through ``BenchmarkReport.dump_json`` (real ``msprof op`` numbers).

usage: python report_isolated.py [--timeout 900]
"""

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

OP = "GroupedQueryAttentionSlidingWindowVarlenFwdOp"
TEST_FILE = "tests/ops/test_grouped_query_attention_sliding_window_varlen.py"
BENCH_FILE = "benchmarks/ops/bench_grouped_query_attention_sliding_window_varlen.py"
REPORTS_DIR = "reports/tileops"


def _root() -> Path:
    return Path(__file__).resolve().parents[8]


def _collect(root: Path, target: str, extra: list[str]) -> list[str]:
    cmd = [sys.executable, "-m", "pytest", target, "--collect-only", "-q", *extra]
    out = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=600)
    ids = [ln.strip() for ln in out.stdout.splitlines() if "::" in ln]
    if not ids:
        print(out.stdout[-3000:], out.stderr[-3000:])
        raise SystemExit(f"no cases collected from {target}")
    return ids


def _run_case(root: Path, target: str, nodeid: str, xml_path: Path, env_extra: dict,
              timeout: int) -> int:
    env = dict(os.environ)
    env.update(env_extra)
    cmd = [
        sys.executable, "-m", "pytest", target, "-sv",
        f"--junitxml={xml_path}", nodeid,
    ]
    log_path = xml_path.with_suffix(".log")
    # start_new_session=True lets us kill the whole process tree
    # (pytest -> python -> bishengir-compile) on timeout.  A bare
    # subprocess.run(timeout=) only kills the direct child and leaves an
    # orphan bishengir-compile holding the device/CPU, which poisons the
    # following cases (observed on case 8: 26 min orphan).
    proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, start_new_session=True)
    try:
        out_b, err_b = proc.communicate(timeout=timeout)
        code = proc.returncode
        out = out_b.decode("utf-8", "replace") if isinstance(out_b, bytes) else (out_b or "")
        err = err_b.decode("utf-8", "replace") if isinstance(err_b, bytes) else (err_b or "")
        log_path.write_text(out + "\n--- STDERR ---\n" + err)
        return code
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            out_b, err_b = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            out_b, err_b = b"", b""
        out = out_b.decode("utf-8", "replace") if isinstance(out_b, bytes) else (out_b or "")
        err = err_b.decode("utf-8", "replace") if isinstance(err_b, bytes) else (err_b or "")
        log_path.write_text(out + "\n--- STDERR ---\n" + err + "\n--- TIMEOUT ---\n")
        print(f"  TIMEOUT after {timeout}s (process group killed): {nodeid}", flush=True)
        return 124


def _xml_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return False
    cases = list(root.iter("testcase"))
    if not cases:
        return False
    return all(c.find("failure") is None and c.find("error") is None for c in cases)


def _run_case_retry(root: Path, target: str, nodeid: str, xml_path: Path, env_extra: dict,
                    timeout: int, retries: int) -> int:
    code = 124
    for attempt in range(1, retries + 1):
        if attempt > 1:
            print(f"  retry {attempt}/{retries}: {nodeid.split('::')[-1]}", flush=True)
        code = _run_case(root, target, nodeid, xml_path, env_extra, timeout)
        # retry only transient failures: timeout / signal / compiler crash.
        # rc==1 is a real test failure, do not retry.
        if code in (0, 1):
            return code
        if code >= 0 and code not in (137, 139):
            return code
    return code


def _merge_junit(parts: list[Path], out_path: Path) -> None:
    testsuites = ET.Element("testsuites")
    cases = []
    for p in parts:
        if not p.exists():
            continue
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError:
            continue
        cases.extend(list(root.iter("testcase")))
    failures = sum(1 for c in cases if c.find("failure") is not None)
    errors = sum(1 for c in cases if c.find("error") is not None)
    skipped = sum(1 for c in cases if c.find("skipped") is not None)
    testsuites.set("tests", str(len(cases)))
    testsuites.set("failures", str(failures))
    testsuites.set("errors", str(errors))
    testsuites.set("skipped", str(skipped))
    for case in cases:
        testsuites.append(case)
    ET.ElementTree(testsuites).write(out_path, encoding="utf-8", xml_declaration=True)


def _merge_benchmark(parts: list[Path], out_path: Path, md_path: Path) -> dict:
    merged = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "profiling_mode_requested": "msprof",
        "environment": {},
        "records": [],
    }
    md_chunks = []
    for p in parts:
        if not p.exists():
            print(f"  MISSING benchmark json: {p}", flush=True)
            continue
        data = json.loads(p.read_text())
        if not merged["environment"]:
            merged["environment"] = data.get("environment") or {}
        merged["records"].extend(data.get("records") or [])
        md = p.with_suffix(".md")
        if md.exists():
            md_chunks.append(md.read_text())
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    md_path.write_text("\n\n".join(md_chunks))
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--bench-timeout", type=int, default=1800)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    root = _root()
    sys.path.insert(0, str(root))
    from tileops.reporting.analyzer import analyze_run
    from tileops.reporting.collector import parse_junit_report
    from tileops.reporting.report import write_reports
    from tileops.reporting.runner import list_operators
    from tileops.reporting.session_timing import attach_session_timing
    from tileops.reporting.setup_info import collect_setup_info

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = (root / REPORTS_DIR / f"{stamp}_{OP}").resolve()
    pytest_dir = run_dir / "pytest"
    artifact_dir = run_dir / "artifacts" / "msprof"
    pytest_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- correctness: one process per collected id -------------
    corr_ids = _collect(root, TEST_FILE, [])
    print(f"[isolated-report] correctness: {len(corr_ids)} cases, 1 process each", flush=True)
    corr_parts = []
    corr_codes = []
    for i, nodeid in enumerate(corr_ids, 1):
        xml = pytest_dir / f"correctness_{i:02d}.xml"
        if args.resume and _xml_ok(xml):
            print(f"  [{i}/{len(corr_ids)}] skip (cached) {nodeid.split('::')[-1]}", flush=True)
            corr_codes.append(0)
            corr_parts.append(xml)
            continue
        code = _run_case_retry(root, TEST_FILE, nodeid, xml, {}, args.timeout, args.retries)
        corr_codes.append(code)
        corr_parts.append(xml)
        print(f"  [{i}/{len(corr_ids)}] rc={code} {nodeid.split('::')[-1]}", flush=True)
    correctness_xml = pytest_dir / "correctness.xml"
    _merge_junit(corr_parts, correctness_xml)
    correctness = parse_junit_report(correctness_xml)
    correctness_code = 0 if all(c == 0 for c in corr_codes) else 1

    # ---------------- benchmark: one process per non-smoke case -------------
    # The manifest carries no pytest marker for the smoke label (only
    # bench_skip_reason), so the official runner's `-m "not smoke"` filters
    # nothing: run all collected cases, like the official pipeline does.
    bench_ids = _collect(root, BENCH_FILE, [])
    print(f"[isolated-report] benchmark: {len(bench_ids)} cases, 1 process each", flush=True)
    bench_parts, bench_json_parts, bench_codes, bench_logs = [], [], [], []
    for i, nodeid in enumerate(bench_ids, 1):
        xml = pytest_dir / f"benchmark_{i:02d}.xml"
        bench_md = pytest_dir / f"benchmark_{i:02d}.md"
        env_extra = {
            "TILEOPS_PROF_MODE": "msprof",
            "TILEOPS_MSPROF_KEEP_OUTPUT": "1",
            "TILEOPS_MSPROF_ARTIFACT_ROOT": str(artifact_dir),
            "TILEOPS_BENCHMARK_REPORT_PATH": str(bench_md),
        }
        if args.resume and _xml_ok(xml) and bench_md.with_suffix(".json").exists():
            print(f"  [{i}/{len(bench_ids)}] skip (cached) {nodeid.split('::')[-1]}", flush=True)
            bench_codes.append(0)
            bench_parts.append(xml)
            bench_json_parts.append(bench_md.with_suffix(".json"))
            bench_logs.append(nodeid.split("::")[-1])
            continue
        code = _run_case_retry(root, BENCH_FILE, nodeid, xml, env_extra,
                               args.bench_timeout, args.retries)
        bench_codes.append(code)
        bench_parts.append(xml)
        bench_json_parts.append(bench_md.with_suffix(".json"))
        bench_logs.append(nodeid.split("::")[-1])
        print(f"  [{i}/{len(bench_ids)}] rc={code} {bench_logs[-1]}", flush=True)
    benchmark_xml = pytest_dir / "benchmark.xml"
    _merge_junit(bench_parts, benchmark_xml)
    benchmark_tests = parse_junit_report(benchmark_xml)
    benchmark_code = 0 if all(c == 0 for c in bench_codes) else 1
    benchmark = _merge_benchmark(
        bench_json_parts, run_dir / "benchmark.json", run_dir / "benchmark.md"
    )
    benchmark["status"] = "present" if benchmark["records"] else "missing"

    lat = [(r.get("label"), r.get("result", {}).get("latency_us"),
            r.get("result", {}).get("Ratio(%)")) for r in benchmark["records"]]
    for label, us, ratio in lat:
        print(f"  RECORD {label}: latency_us={us} Ratio(%)={ratio}", flush=True)

    # ---------------- assemble the official run.json / report.md -----------
    generated_at = datetime.now().astimezone().isoformat()
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
            timeout=10,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        git_commit = None
    metadata = {
        "run_id": run_dir.name,
        "generated_at": generated_at,
        "project_root": str(root),
        "git_commit": git_commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "test_file": TEST_FILE,
        "benchmark_file": BENCH_FILE,
        "prof_mode_requested": "msprof",
        "kernel_name_requested": None,
        "commands": {
            "correctness": f"per-case pytest ({len(corr_ids)} processes)",
            "benchmark": f"per-case pytest ({len(bench_ids)} processes)",
        },
        "exit_codes": {"correctness": correctness_code, "benchmark": benchmark_code},
        "case_granularity": "one process per pytest id (single-process multi-trace deadlock)",
    }
    operator_catalog = list_operators()
    setup = collect_setup_info(
        scope=OP,
        run_id=run_dir.name,
        generated_at=generated_at,
        profiler="msprof",
        git_commit=git_commit,
        test_target=TEST_FILE,
        benchmark_target=BENCH_FILE,
    )
    run = analyze_run(
        operator=OP,
        correctness=correctness,
        correctness_exit_code=correctness_code,
        benchmark=benchmark,
        benchmark_exit_code=benchmark_code,
        benchmark_tests=benchmark_tests,
        metadata=metadata,
        setup=setup,
        operator_catalog=operator_catalog,
    )
    attach_session_timing(run, root=root, operator_catalog=operator_catalog, enabled=False)
    write_reports(run, run_dir)
    latest = run_dir.parent / "latest.json"
    latest.write_text(json.dumps({"run_id": run_dir.name,
                                  "run_json": str(run_dir / "run.json")}, indent=2))

    print(f"[isolated-report] status={run.get('status')} run_dir={run_dir}", flush=True)
    return 0 if run.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
