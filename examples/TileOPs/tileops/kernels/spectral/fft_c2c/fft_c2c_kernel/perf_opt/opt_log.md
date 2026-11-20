# FFT C2C (`_fft_c2c_kernel`) — Stage 4 调优日志

- project: `fft_c2c` / operator: `_fft_c2c_kernel` (op class `FFTC2COp`)
- baseline (read-only): `examples/TileOPs/tileops/kernels/spectral/fft_c2c/fft_c2c_kernel/_fft_c2c_kernel.py`
- perf artifact: `perf_opt/_fft_c2c_kernel.py` (final, == baseline kernel body, see §Decision)
- bench entry: `examples/TileOPs/benchmarks/ops/bench_fft.py::test_fft_bench`
- performance target: `best_effort`; primary metric: `msprof op Task Duration(us)` (kernel-only)
- tuning mode: `full`, narrow-scope continuation (single structural route + minimal localisation)
- hardware / toolchain: Ascend910_9362 (`npu-smi -i 6`), Aicore Usage 0% before and after every collection (no external card contention)

---

## 1. Performance Test Data

All durations are the **median of 20 `msprof op` Task Duration samples** (`--launch-count=20 --warm-up=5`).

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | profile_status | raw_profile_dir |
|---|---|---|---|---:|---|---|
| `..._fft_c2c_kernel.py::_fft_c2c_kernel::main` | `fft-4k-c64-unbatched-complex64` | baseline | `main` | `main` | 717.680 | valid | `perf_opt/profiles/baseline/round0/fft-4k-c64-unbatched_baseline/OPPROF_20261016224448_CWDVOLGCGCBRURVF` |
| `..._fft_c2c_kernel.py::_fft_c2c_kernel::main` | `fft-4k-c64-b64-complex64` | baseline | `main` | `main` | 1498.320 | valid | `perf_opt/profiles/baseline/round0/fft-4k-c64-b64_baseline/OPPROF_20261017020209_GJXLNFPMMRTDTOOF` |
| `..._fft_c2c_kernel.py::_fft_c2c_kernel::main` | `fft-4k-c64-unbatched-complex64` | r1_serialvec | `main` | `main` | 2040.120 | valid | `perf_opt/profiles/round1/serialvec_4k_unbatched/OPPROF_20261017020002_YMDNBAZLOCYSUVOR` |
| `..._fft_c2c_kernel.py::_fft_c2c_kernel::main` | `fft-4k-c64-unbatched-complex64` | final | `main` | `main` | 719.530 | valid | `perf_opt/profiles/final/fft-4k-c64-unbatched_final/OPPROF_20261017020320_PMNQDKIUQJEUZMXG` |
| `..._fft_c2c_kernel.py::_fft_c2c_kernel::main` | `fft-4k-c64-b64-complex64` | final | `main` | `main` | 1497.340 | valid | `perf_opt/profiles/final/fft-4k-c64-b64_final/OPPROF_20261017020401_FPIKPAMZVJBKIBQK` |

Excluded workloads:

- `fft-64-c64-smoke-complex64` (**smoke**): manifest label carries the standalone `smoke` token → no perf baseline and no winner; used for the final precision regression only (`L0-1` PASS, `max_diff=2.249e-06`).
- `fft-4k-c128-b64-complex128` (**skipped**): the benchmark body calls `pytest.xfail` (CG-2026-0019 — npuir has no complex dtype and no fp64 vector elementwise ops). The case cannot run, so it has no perf record and no precision record. Recorded in `workload_inventory.json` as `skipped` so the exclusion is disclosed.

### 1.1 Baseline bottleneck (measured, not inferred)

| metric | fft-4k-c64-unbatched | fft-4k-c64-b64 |
|---|---:|---:|
| `aiv_vec_ratio` | 1.197e-04 | 1.157e-04 |
| `aiv_scalar_ratio` | 0.99884 | 0.99848 |
| `aiv_mte2_ratio` | 5.92e-04 | 1.057e-03 |
| `aiv_mte3_ratio` | 3.889e-04 | 3.764e-04 |
| `aiv_ub_read_bw_vector (GB/s)` | 0.0423 | 0.0417 |
| `aiv_time(us)` (per core, mean) | 718.37 | 743.30 |

The kernel is **99.9 % scalar** and UB vector bandwidth is ~0.04 GB/s — i.e. not bandwidth-limited, purely issuance/ALU-limited by scalar butterfly execution. Per the FFT-DESIGN this is the single dominant bottleneck (SPEC §1.6.2), so it defines the search direction.

### 1.2 Design estimate vs measured (`DESIGN.md` §1.6.0 / §1.6.2)

`DESIGN.md` §1.6.2 states the vectorisation conclusion *"逐元素计算（#1–#5）**已全部向量化**"* and §1.6.3 fixes the layout with *"内层轴 `i`（连续段）为向量化轴"*. The measured `aiv_vec_ratio` of 1.2e-04 shows the butterfly network is **not** vectorised at all: the design's vectorisation pre-condition does not hold on this toolchain. This measured contradiction is the core finding of this round.

### Phase 1 workload ordering (rationale)

`tune` workloads are ordered by measured cost: `fft-4k-c64-unbatched-complex64` (717.68 us) first, then `fft-4k-c64-b64-complex64` (1498.32 us). Both run the same Stage 3 kernel `_fft_c2c_kernel::main` (one FFT row per block, `T.Kernel(batch, is_npu=True)`), so a structural fix applies to both; the unbatched case is also the one with the highest per-row cost (1/40 core occupancy), hence first.

---

## 2. Iteration Log

### Round 0 — baseline (current best = unchanged Stage 3 kernel)

Baseline collected for both `tune` workloads (table §1). current best = `perf_opt/_fft_c2c_kernel.py`, sha256 `623fd321e2befb95d08a3a2afc03f10c1d95d3de01b0e1fa6307c7e3c668c552` (kernel body byte-identical to the read-only Stage 3 baseline `_fft_c2c_kernel.py`, verified by `diff` over lines 1–205; only the `--case` bench harness is appended).

### Round 1 — candidate vs current best

Phenomenon read from the current-best profile: 99.9 % scalar, `aiv_vec_ratio` 1.2e-04, UB vector bandwidth 0.04 GB/s. Diagnostic signal rule (scalar share > 50 % → mandatory re-axis/relayout evaluation) fires, so this round evaluates structural relayouts only.

Candidate optimisations generated from the same current best, one primary change each:

- **r1_vecviews** — re-view the ping-pong rows as `[ng,2,h]` / `[2,ng,h]` so each butterfly becomes four single-store loops whose store indices are bare loop Vars (targets the npuir vectoriser's store-index rule directly).
- **r1_serialvec** — replace the `T.Parallel(ng,h)` butterfly with `T.serial(ng)` + `T.vectorized(h)` single-store loops (TIR `kVectorized` loop → standard TVM `VectorizeLoop`, a vectorisation path independent of the npuir pass and of store-index shape).

**Candidate vs current best comparison table** (Task Duration, AICore utilisation, memory metrics, L0 result):

| candidate_id | primary change | Task Duration(us) | AICore util (`aiv_vec`/`aiv_scalar_ratio`) | memory (`aiv_ub read/write GB/s`, `aiv_mte2_ratio`) | L0 | verdict |
|---|---:|---|---|---|---|---|
| current best (baseline) | — | 717.680 | 1.20e-04 / 0.99884 | 0.0423 / 0.000592 | PASS | reference |
| r1_vecviews | per-stage `T.view` relayout, 4 single-store loops | — (no valid profile: compile fails) | — | — | not reached | **blocked (compile)** |
| r1_serialvec | `T.serial(g)` + `T.vectorized(i)`, 4 single-store loops | 2040.120 | 4.60e-05 / 0.99957 | 0.0150 / 0.000200 | PASS | **regression 2.84x, rollback** |

`r1_vecviews` produced no valid profile, so per the collection contract it is not written to `perf_records.jsonl`; its structural evidence is recorded below.

#### r1_vecviews — blocked at compile time (root cause localised)

`perf_opt/_fft_c2c_kernel_round1_vecviews.py` fails at `tvm::tir::RemoveNoOp`:

```
InternalError: Check failed: (!def_count_.count(v)) is false:
variable S3i has already been defined, the Stmt is not SSA
```

Root cause (localised with minimal probes, see §2.1): inside a `@T.prim_func`, `X = T.view(buf, shape)` is an assignment whose RHS is a `Buffer`, so the parser calls `IRBuilder.name("X", buffer)` (`tilelang/language/parser/parser.py` `bind_assign_value`, line 149). Because `T.view` builds the alias on the **same data pointer** (`tilelang/language/customize.py` `reshape`/`view` → `T.Buffer(shape, dtype, src.data)`), every re-binding of the same identifier across the inlined stage macros renames/rebinds the same underlying var → SSA violation. The stage count is a compile-time constant and every stage needs a different `[ng,2,h]` shape, so this is unavoidable in the required code shape.

#### r1_serialvec — compiles and is numerically correct, but does not vectorise and is 2.84x slower

`perf_opt/_fft_c2c_kernel_round1_serialvec.py` passes the full L0 suite (`L0-1..L0-6` PASS, `L0-7` disclosed SKIP) with `max_diff` ≤ 6.4e-05 (tolerance 1e-4). But the profile shows `aiv_vec_ratio` **fell** to 4.6e-05 (`aiv_scalar_ratio` 0.99957) and Task Duration rose to 2040.12 us — the `TIR kVectorized` loop is not lowered to vector instructions on npuir and additionally loses the block-level `T.Parallel` parallelism. **Rollback**: current best unchanged.

### 2.1 Minimal localisation of the blocking constraint (probe set)

The exact compiler constraint was read from source and then confirmed by minimal probes. `src/transform/npu_loop_vectorize.cc`:

- `VectorizeSingleStatement` (line 1184): the innermost `T.Parallel` body must be exactly one `BufferStore`, and **every store index must be a bare `Var` or a scalar constant** (`if (!idx.as<VarNode>() && !IsScalar(idx)) return …`).
- `IsSupportedVectorLoadIndex` (line 251): every load index must simplify to either a loop-invariant expression or a **single-loop-var unit-stride** expression.

The baseline butterfly `D[g*h+i] = …` / `D[g*h+i+nh] = …` violates the store rule (affine index), which forces the relayout — and the relayout needs a per-stage buffer alias. Alias availability was then probed exhaustively on npuir (all minimal, single-process, `n=8`):

| probe form | result |
|---|---|
| `A2 = T.view(A,[N//2,2]); T.copy(A2, y)` — the repo's own official view pattern (`testing/python/language/test_tilelang_language_view.py` form) | FAIL — `Check failed: (GetVarValue(op->buffer_var.get()) == mlir::Value{})` (npuir codegen) |
| `A2 = T.Buffer([N//2,2], "float32", A.data, scope="shared")` (and `scope="local"`) | FAIL — same npuir codegen diagnostic |
| single `T.view` + one `T.Parallel` store | FAIL — SSA (`variable B2 has already been defined`) |
| two different-shaped `T.view`s of the same buffer | FAIL — SSA (`variable A_v1 has been used before definition`) |
| `T.reshape(A, B)` | **not an alias** — on npuir `T.reshape` resolves to `tilelang.language.customize_npuir.npuir_reshape(src, dst)`, the `tl.npuir_reshape` **operation**, called with a shape list it raises `'list' object has no attribute 'buffer'`; used as an op it compiles but perturbs the JIT output contract (returned 8 scalar tensors for a 1-D output) |
| `for g in T.serial(ng): for i in T.vectorized(h): …` | compiles, but stays scalar (see above) |

Conclusion: on this toolchain npuir has **no usable shared/UB buffer aliasing primitive** (`T.view`, same-data `T.Buffer`, `T.reshape` all unusable). Without aliasing, a per-stage geometry change can only be expressed by allocating a distinct physical buffer per stage, which needs `2·log2(n)·2` row buffers (for `n=4096`: 12 stages × 2 ping-pong × 32 KB re+im ≈ 768 KB ≫ 192 KB UB). The butterfly therefore cannot be given bare-Var store indices, and the vectoriser cannot fire. This is a **compiler capability gap**, not a tiling/parameter problem.

### 2.2 Merge / non-regression check and winner

- No candidate produced a valid improvement, so nothing was merged; current best is unchanged.
- `perf_opt/_fft_c2c_kernel.py` re-measured on the final file for both `tune` workloads (table §1) and re-verified for precision: `--level all` → `L0-1..L0-6` PASS, `L0-7` disclosed SKIP, `L1` (10 cases incl. `n=8192`) PASS, `L2` (8 cases) PASS, `Boundary` (impulse/constant/zeros) PASS, max_diff ≤ 8.53e-05 (< 1e-4 tolerance).
- `tuning_status`: `no_gain` for both `tune` workloads.

### Round 1 termination

`stop_reason = blocked`: the remaining structural candidates (`T.view` relayout, `kVectorized` loops, and every buffer-alias construction) are all refuted with measured/document-level evidence on the current toolchain. Budget was respected (1 adopted-candidate attempt + minimal localisation probes; no further divergence).

---

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/spectral/fft_c2c/fft_c2c_kernel/_fft_c2c_kernel.py::_fft_c2c_kernel::main | fft-4k-c64-unbatched-complex64 | 717.680 | 719.530 | final |
| examples/TileOPs/tileops/kernels/spectral/fft_c2c/fft_c2c_kernel/_fft_c2c_kernel.py::_fft_c2c_kernel::main | fft-4k-c64-b64-complex64 | 1498.320 | 1497.340 | final |

- Both `final_us` values come from the same final candidate (`perf_opt/_fft_c2c_kernel.py`,
  sha256 `623fd321e2befb95d08a3a2afc03f10c1d95d3de01b0e1fa6307c7e3c668c552`), re-measured on the final file.
- Improvement is within msprof noise (≤ 0.4 %): **no net gain** (−0.26 % and +0.07 %).
- The final file's kernel body is byte-identical to the read-only Stage 3 baseline; verified by `diff` over the shared region (lines 1–205) — only the appended `--case` bench harness differs.
- smoke final precision: `fft-64-c64-smoke-complex64` PASS (`max_diff=2.249e-06`), no perf record.
- skipped: `fft-4k-c128-b64-complex128` — `pytest.xfail`, CG-2026-0019, no perf and no precision record.

### Decision

**No adoptable candidate; the tuning loop is converged (blocked).** `perf_opt/_fft_c2c_kernel.py` is kept as the final best (drop-in, unchanged kernel). The measured 99.9 % scalar profile persists because the required relayout is not expressible on the current toolchain — see §2.1 and `perf_opt/perf_feedback.md`.

---

## 3. Skill Retrospective

### 3.1 Process findings

1. **The vectoriser's acceptance rule is a conjunction, not the single store-index rule.** The round's premise ("single store + bare-Var index is the only constraint") is necessary but not sufficient: `IsSupportedVectorLoadIndex` (`src/transform/npu_loop_vectorize.cc:251`) additionally rejects a load index that mixes two loop vars (`S[g*2h+i]` maps to `-2` in `FindMappedLoopVar`). Any future FFT-style relayout planning must check **both** the store-index and the load-index rules before committing to a code shape.
2. **`T.view` / `T.Buffer(..., data=...)` must be treated as unsupported on npuir.** The repo's own `T.view` unit test does not run under `target="npuir"`, so its passing status is no evidence for npuir. Recommend a trap entry (T-class) with the exact diagnostics.
3. **The mechanical workload classifier cannot represent imperative `pytest.xfail`.** `gate_lint._perf_records_workload_lint` derives `kind` from `"skip" in marks` or a standalone `smoke` token; `pytest.xfail()` inside a benchmark body leaves no marker, so a manifest-only-xfail case would be forced into `tune` and then fail the baseline/final coverage check. Worked around by recording the case as `skipped` with an explicit mechanism note in `reason`; a cleaner rule would treat benchmark-level `xfail` as `skipped`.

### 3.2 Value point proposals

> Scope note: this run is a narrow-scope continuation with an explicit "write only into `perf_opt/`" boundary, so the `pattern-library/` write-back required by Phase 4 is **not** performed here (no files outside `perf_opt/` were touched). The entries below are stated inline with toolchain stamps and self-contained repro so the terminal `tilelang-skill-evolver` can file them; `repro-missing` is not claimed because the repro bodies are given verbatim below.

**D-1 (实测数据, vp_type=D)** — FFT C2C scalar-bound baseline, Ascend910_9362 / CANN 9.0.0 / tilelang dev-root build (`/home/developer/tilelang-mlir-ascend/build`), origin_task `fft_c2c-_fft_c2c_kernel` 2026-10-17.
Evidence: `aiv_vec_ratio=1.20e-04`, `aiv_scalar_ratio=0.99884`, `aiv_ub_read_bw_vector=0.0423 GB/s`, Task Duration 717.68 us (median of 20, `--launch-count=20 --warm-up=5`) for `[4096]` complex64; profile `perf_opt/profiles/baseline/round0/fft-4k-c64-unbatched_baseline/OPPROF_20261016224448_CWDVOLGCGCBRURVF`.
Repro: `perf_opt/_fft_c2c_kernel.py --case fft-4k-c64-unbatched` under `msprof op`.

**D-2 (实测数据, vp_type=D)** — TIR `T.vectorized` loops do **not** vectorise on npuir and additionally lose `T.Parallel` parallelism.
Evidence: `T.serial(g)+T.vectorized(i)` variant compiles, passes L0 (max_diff 6.4e-05) yet measures `aiv_vec_ratio=4.6e-05`, Task Duration 2040.12 us vs 717.68 us baseline (−2.84x); profile `perf_opt/profiles/round1/serialvec_4k_unbatched/OPPROF_20261017020002_YMDNBAZLOCYSUVOR`.
Repro (minimal, self-contained, NPU-runnable):
```python
# before: for g, i in T.Parallel(ng, h): D[g*h+i] = S[g*2*h+i] + S[g*2*h+h+i]
# after:  for g in T.serial(ng):
#             for i in T.vectorized(h):
#                 D[g*h+i] = S[g*2*h+i] + S[g*2*h+h+i]
```
Optimisation insight: the npuir vectoriser is not the TIR `kVectorized`/`VectorizeLoop` path; swapping to the TIR form scalarises and serialises.

**P-1 (模式/否定, vp_type=P)** — npuir has no usable shared/UB buffer aliasing: `T.view`, `T.Buffer(shape, dtype, src.data, scope=...)` and `T.reshape` are all unusable.
Evidence (all on `n=8`, single process):
- `A2 = T.view(A,[N//2,2]); T.copy(A2, y)` → `Check failed: (GetVarValue(op->buffer_var.get()) == mlir::Value{})`.
- `A2 = T.Buffer([N//2,2], "float32", A.data, scope="shared"|"local")` → same diagnostic.
- `A2 = T.view(A,[N//2,2])` used in one `T.Parallel` store → `variable B2 has already been defined, the Stmt is not SSA`.
- two shapes on one buffer → `variable A_v1 has been used before definition`.
- `T.reshape(A, B)` → on npuir resolves to `npuir_reshape(src, dst)` (`tl.npuir_reshape` op), not an alias.
Impact: any per-stage geometry change in-kernel (FFT/Stockham, stencils, multi-stage tiling) cannot be expressed without allocating a distinct buffer per stage — a hard blocker, quantified for `n=4096` at ~768 KB UB needed vs 192 KB available.
repro (minimal, exact):
```python
@T.prim_func
def main(x: T.Tensor((8,), "float32"), y: T.Tensor((8,), "float32")) -> None:
    with T.Kernel(1, is_npu=True) as (cid, _):
        A = T.alloc_shared((8,), "float32")
        T.copy(x[0:8], A)
        A2 = T.view(A, [4, 2])          # <- npuir codegen CHECK fires here
        T.copy(A2, y[0:4, 0:2])
```

**C-1 (案例索引, vp_type=C)** — `fft_c2c / _fft_c2c_kernel` (TileOPs optimize, complex64 spectral): scalar-bound kernel whose DESIGN §1.6.2/§1.6.3 assumed `T.Parallel` vectorisation; measured `aiv_vec_ratio=1.2e-04` refutes it. Reference the D-1/P-1 entries before planning any in-kernel relayout for FFT-family kernels.

**R-1 (流程, vp_type=R)** — mechanical workload classification vs imperative `pytest.xfail` (see §3.1 item 3). Proposed rule change for `tilelang-skill-evolve` queue (Tier 2, needs approval): in `_perf_records_workload_lint`, treat a benchmark-level `pytest.xfail` case as `skipped`, or require the inventory's `marks` field to be able to carry `xfail`.

---

## 4. Files

| path | role |
|---|---|
| `perf_opt/_fft_c2c_kernel.py` | **final best** (kernel body identical to the read-only Stage 3 baseline; sha256 `623fd321…c552`) |
| `perf_opt/_fft_c2c_kernel_round1_vecviews.py` | experiment branch r1_vecviews — does not compile (SSA), kept as evidence |
| `perf_opt/_fft_c2c_kernel_round1_serialvec.py` | experiment branch r1_serialvec — compiles/L0 PASS, 2.84x slower, rolled back |
| `perf_opt/probe_*.py` | minimal localisation probes (alias forms, vectorized loops, view forms) |
| `perf_opt/workload_inventory.json` | benchmark workload inventory |
| `perf_opt/perf_records.jsonl` | append-only per-measurement records |
| `perf_opt/profiles/` | raw `msprof op` OPPROF data (baseline / round1 / final) |
| `perf_opt/logs/` | experiment stdout/stderr (baseline / round1 / final) |
| `perf_opt/perf_feedback.md` | `[DESIGN_LIMIT]` feedback artifact |

## 8. Route C (4-step FFT via Cube GEMM) — feasibility verdict (2026-11-04)

`repro/gemm64_efficiency_probe.py`：每 block 连做 500 次 64×64×64 GEMM（20 block = 全部
AI core），`msprof op` Task Duration 取均值；数值 rel_err 1.9e-5 / 2.3e-5 通过。

| kernel | Task Duration | achieved | 相对 Cube 峰值 |
|---|---:|---:|---:|
| `gemm64_fp16` | 96.56 µs | 54.30 TOps/s | **18.4 %**（峰值 294.912） |
| `gemm64_fp32` | 167.66 µs | 31.27 TOps/s | **21.2 %**（峰值 147.456） |

**判定：C 路线按「每级 64 个 64×64 复 GEMM」的形态达不到 Ratio 60%。**
提交口径下 `Ratio ≈ GEMM 效率`（Perf / computility），18–21% 的效率对应 Ratio
≈18–21%；要到 60% 需要 **≥65% 效率，即 3× 差距**。

**原因**：64×64×64 对 Cube 太小（分形块 16×16，只剩 4×4×4 个），K=64 无法摊薄
accumulator/ L0A-L0B 装载开销。

**未证伪的后续**：把 64 行 batch 进同一个 GEMM（M=4096, K=64, N=64）可把每-tile 开销
摊薄 64 倍，效率可能显著高于 18%；或合并级数把 K 做大。两者都需再一轮探针才能定，
在此之前不得宣称 C 路线可达 60%。

## 9. 折叠形态向量化落地（2026-11-04）—— 718 µs → 12.62 µs（56.9×）

在放弃 C 路线后，按「形状折叠 + 分组并行 + 三阶融合」把标量实现改写为向量化实现。
**每一版都与 `torch.fft.fft` 对拍（含 DIF bit-reversal 重排），数值全部通过。**

| 版本 | 机制 | 每链耗时 | 相对基线 | 数值 |
|---|---|---:|---:|---|
| baseline | 现网标量实现 | 717.68 µs | 1.0× | — |
| v1 `fft_fullchain_proto.py` | 形状折叠（`(nb,2,m)` + IntImm 首维），单核 `T.serial(nb)` | 673.3 µs | 1.07× | rel 3.18e-07 |
| v2 `fft_fullchain_par.py` | 同上，改为 `T.Kernel(nb)` 分组并行到 40 AIV | **299.9 µs** | **2.4×** | rel 3.18e-07 |
| v3 `fft_fullchain_radix8.py` | 最后 3 级融合成 radix-8（512 块 × 8 元素，旋转因子编译期常量） | **59.33 µs** | **12.1×** | rel 3.18e-07 |
| v4 `fft_fullchain_v4.py` | 通用三阶融合模板，stage 7–9 也融合（64 块 × 64 元素） | **20.04 µs** | **35.8×** | rel 3.31e-07 |
| v5 `fft_fullchain_v5.py` | 再加 fused(512) 覆盖 stage 4–6 → **6 kernel/链** | **12.62 µs** | **56.9×** | rel 3.85e-07 |

### v5 逐 kernel 分解（6 kernel/链，每链 12.62 µs）

| kernel | 内容 | 耗时 |
|---|---|---:|
| 1–3 | stage 1–3（nb = 1/2/4） | 2.24 / 2.50 / 1.78 µs |
| 4 | fused 4–6（8 块 × 512 元素） | 2.14 µs |
| 5 | fused 7–9（64 块 × 64 元素） | 2.24 µs |
| 6 | fused 10–12（512 块 × 8 元素） | 1.72 µs |

**目前每个 kernel 都落在 1.7–2.5 µs，即"内核固定开销"地板**（最便宜的内核实测 1.72 µs）。
所以 v5 的 12.62 µs ≈ 6 × 2.1 µs，算术部分已可忽略。

### 地板与剩余空间（实测界定）

| kernel 数 | 地板时延 | 相对基线 |
|---|---:|---:|
| 6（现状 v5） | ~12.6 µs | **56.9×** |
| 4 | ~8.4 µs | ~85× |
| 2 | ~4.2 µs | ~170× |
| 1 | ~2.1 µs | ~340× |

继续压缩**只有减少 kernel 数一条路**：
- stage 1–3（nb = 1/2/4）不能套融合模板——模板的级内循环是 `T.serial`（标量），
  bs0=4096 时 m0=2048 会退化成 34 µs 级，必须保留向量化的单级 kernel；
- 要再降就得把多级塞进单个 kernel，而每级需要不同的 `(nb, 2, m)` 缓冲形状，
  跨级复用同一块 UB 需要 **buffer view/alias**——这正是 `CG-2026-0002` 附近的已知缺口
  （`T.view` / 同 data `T.Buffer` 均不可用）。属于编译器能力问题，不是本算子能解。

### v4 逐 kernel 分解（8 kernel/链）

| kernel | 内容 | 耗时 |
|---|---|---:|
| 1–5 | stage 1–5 | 2.00 / 2.36 / 1.86 / 2.14 / 2.70 µs |
| 6 | stage 6 | 3.80 µs |
| 7 | fused 7–9（64 块 × 64 元素） | 2.28 µs |
| 8 | fused 10–12（512 块 × 8 元素） | 2.58 µs |
| | **合计** | **20.04 µs** |

### 三条关键机制（对后续同类算子可复用）

1. **形状折叠**是唯一能绕过「store 下标必须是裸 Var/IntImm」的写法：把 `D[g*h+i]` 改成
   多维缓冲 + IntImm 首维（`D[0,i]`），向量化即触发（vec 1.2e-4 → 0.26）。
2. **operand rank 必须一致**：2D 索引操作数 × 1D twiddle 会在 `codegen_npuir_dev.cc:166`
   的广播维度校验上失败（成对出现时静默算错 21.1，拆成单循环时直接 crash）。把 twiddle
   也做成同 rank（`(2,m)` 或 `(1,1,m)`）即 bit-exact。
3. **融合比自己并行更值钱**：每级折半会让末段批量宽度 m→1，单级内核再多也只能做到
   2.4×；把级数按块融合（三阶 = radix-8 常量旋转因子）把 245 µs 压到 2.2 µs。

### 剩余空间（未做）

v4 的 20 µs 里，**8 次 kernel 发射的固定开销约占 16 µs**（每个 ~2 µs），真正的计算只剩
~4 µs。继续压缩的唯一方向是**减少发射次数**：把 stage 1–6 继续融合（例如 stage 1–3
一组、4–6 一组）可望到 ~12–14 µs（约 55×）；彻底方案是整链折叠进单个 kernel。

注意：v1–v4 都是**原型**（host 驱动多 kernel），尚未并入 `_fft_c2c_kernel.py` 的真实
kernel 与精度门禁（L0/L1/L2/boundary 用例）；对外报数前必须过完整门禁。
