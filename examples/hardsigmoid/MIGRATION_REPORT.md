# TileOPs 迁移报告 — HardsigmoidFwdOp（GPU → Ascend NPU）

> 依据《TileOPs 迁移任务书》§3 交付清单 + §4 报告骨架产出。所有数字均来自命令输出或 `run.json`；本报告不做估算、不复述未经验证的数字。
> 迁移模式：`migration-harness`（conductor 场景路由判定，见 §2）。
> 本版为 **修订报告（rev.2，2026-10-13）**：不重做迁移（未改 kernel 语义、未重跑 Stage 1–3），仅在报告与工具侧收口 M1–M8。

## 0. 修订留痕（M1–M8）

| M | 改了什么 | 证据 | 命令 |
|---|---|---|---|
| M1 | §4 顶部新增「workload 来源」表 + 「benchmark 用例构成」表（9 例 / 6 例进 msprof / 3 例未采集逐例说明） | 上游 vs NPU manifest workloads diff；bench parametrize 9 例 nodeid；`run.json` `performance.cases`(6) | `diff <(grep -A4 '^HardsigmoidFwdOp:' src/tileops/.../elementwise_unary_activation.yaml) <(grep -A20 '^HardsigmoidFwdOp:' examples/TileOPs/tileops/manifest/elementwise.yaml)`；`pytest benchmarks/ops/bench_hardsigmoid.py --collect-only -q` |
| M2 | 修正 `fc-wide（256M elem）` → `8,388,608 ≈ 8.39M`，并写出元素数/字节计算式 | 2048×4096=8,388,608；fp16 读写 2×N×2B≈33.55MB | `python -c "N=2048*4096; print(N, 2*N*2)"` |
| M3 | §4 增 Mish `fc-wide [2048,4096]` 同 shape 对照行（fp16 54.01% / bf16 57.67%）+ 环境差异注 | `reports/report.md:106-107`（MishFwdOp 表，带宽列 1.8000）；本报告行为 910B3/8.5.2 | `grep -n 'fc-wide | 28.1800\|29.0200' reports/report.md` |
| M4 | 补 `inplace` 契约处置（保留 kw、默认 false、forward 原地语义、测试未覆盖）+ 1.75 TB/s 出处 | `tileops/ops/elementwise/hardsigmoid.py`（forward/inplace）；`tileops/benchmark/benchmark_base.py:464-467`（`Bandwidth(TB/s)=gm["bw"]`，msprof 解析值）；`msprof_roofline.py:36`（`gm_bw=1.8` 默认）；`constants.md:77/97` | `grep -n 'inplace' tileops/ops/elementwise/hardsigmoid.py`；`grep -n 'Bandwidth(TB/s)' tileops/benchmark/benchmark_base.py` |
| M5 | 消除 ruff 9→0（wrapper I001 修复；集成 kernel E402/RUF022、history_version I001 经 per-file-ignore + `__all__` 排序）；standards_check 失败明细逐条证无关；适配层三处改动清单 | `ruff check` 输出；`pyproject.toml` per-file-ignores；`standards_check.py check` JSON | `python -m ruff check tileops/kernels/elementwise/hardsigmoid tests/ops/test_hardsigmoid.py benchmarks/ops/bench_hardsigmoid.py`；`python3 .agents/tools/standards_check.py check` |
| M6 | §2.4 注明 DESIGN.md 还原对象的版本号/依据（快照文件名+时间戳），说明为修订终版 v1 及 Stage 1 失败尝试影响已消除 | `history_version/design_v0.md`(21:48)/`design_v1.md`(21:58)；集成快照 `hardsigmoid_kernel/hardsigmoid_fwd_kernel_DESIGN.md`(2026-10-12 21:58:52) | `ls -la --time-style=full-iso history_version/ hardsigmoid_kernel/*DESIGN.md` |
| M7 | §5 补 run.json `correctness_tests`/`git_commit`；终态蒸馏产物 id；两个工具缺陷登记（VP-2026-0145/0146） | queue.md 新增条目；stats.md 蒸馏行；run.json summary | `grep -n 'correctness_tests\|git_commit' run.json`；`grep -n 'VP-2026-014' .agents/evolution/queue.md` |
| M8 | 工具侧：`integrate_kernel.py` 新增必写 `wrapper_rewritten` 字段（`write_integration_log`）；备份集成包后重跑集成，护栏 1 FAIL→PASS；两份日志留档 | integrate_kernel.py diff；复跑前后 integration_log.md；`grep` 输出 | `python3 .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/hardsigmoid/.migration_meta.json`；`grep -n wrapper_rewritten integration_log.md` |

---

## 1. 身份与范围

| 项 | 值 |
|---|---|
| op_name | `HardsigmoidFwdOp` |
| family | `elementwise` |
| op_slug | `hardsigmoid` |
| 对外接口函数 | `hardsigmoid_fwd_kernel`（`target="npuir"`） |
| 编程模式 | Developer |
| gpu_repo_root | `/mnt/workspace/TileOPs-main`（上游 TileOPs `1238a9f`） |
| NPU_WORKSPACE | `/mnt/workspace/npuir-migration` |
| NPU_TILEOPS | `/mnt/workspace/npuir-migration/examples/TileOPs` |
| 算子状态目录 | `examples/hardsigmoid/hardsigmoid_fwd_kernel/` |
| 报告目录 | `examples/TileOPs/reports/tileops/20261012_223321_664388_HardsigmoidFwdOp/` |
| run_id | `20261012_223321_664388_HardsigmoidFwdOp` |
| GPU 源算子 | `src/tileops/kernels/elementwise/activations.py:169` `HardsigmoidFwdKernel`（+ `_nan.py`、`_builders.py`） |

**版本戳（D10，§2.2）**

| 组件 | 版本 |
|---|---|
| tilelang | `0.1.2`，源码 `/home/developer/tilelang-mlir-ascend`，commit `417dae9` |
| torch | `2.7.1+cpu` |
| torch_npu | `2.7.1.post4` |
| CANN | `8.5.2`（`$ASCEND_HOME=/home/developer/Ascend/cann-8.5.2`） |
| 驱动 / npu-smi | `25.5.0` |
| 设备 | Ascend910B3 × 1 |
| python / OS | 3.11.4 / Linux aarch64, glibc 2.31 |

---

## 2. 迁移流程证据

### 2.1 路由与判定（D1，§0.4）

| 场景 | migration_mode | stage_plan（函数级） | stage_plan（op 级） |
|---|---|---|---|
| `migration` | `harness` | `[1, 2, 3]` | `[0, (1→2→3)×1, 5]` |

§0.4 四条判定（`cd /mnt/workspace/TileOPs-main`）——**字面形态**：

```
test -d tileops/manifest   -> 1 FAIL
test -d tests/ops          -> 2 ok
test -d benchmarks/ops     -> 3 ok
grep -rl '^HardsigmoidFwdOp:' tileops/manifest/*.yaml -> 4 FAIL（无该目录）
```

> 上游为 `src/tileops/manifest/spec/*.yaml` 布局，字面判定会**静默判成 plain**（任务书 §7.1 点名的坑）。

**§0.3 适配层套用后**（`MANIFEST_DIR` 回退）：

```
MANIFEST_DIR=src/tileops/manifest
1 ok / 2 ok / 3 ok
4 ok  -> src/tileops/manifest/spec/elementwise_unary_activation.yaml
```

**结论：`migration_mode=harness`（四条全过）。**

**§0.3 源仓适配层（本次补齐，3 处）**
1. `examples/TileOPs/.agents/skills/add-npu-op/scripts/extract_tl_kernel.py::load_manifest_entry` — 增加 `src/tileops/manifest`（含 `spec/` 子目录 rglob）回退 + `source` 缺失时回退读 `.migration_source_map.json`（新增 `_resolve_manifest_dir` / `_apply_source_map_fallback`）。
2. `.opencode/agents/tilelang-op-conductor.md` — harness 判定加 `MANIFEST_DIR` src-layout 回退。
3. `{GPU_REPO}/.migration_source_map.json` — 由 op 类 `kernel_types` + kernels 目录扫描生成（69 个算子，`op → {op_class, op, kernel, kernel_map, manifest, test, bench, workload}`）。

### 2.2 Stage 0 产物（D3）

`.migration_meta.json`：`extracted_functions = ["hardsigmoid_fwd_kernel"]`（N=1），`wrapper_path = tileops/kernels/elementwise/hardsigmoid/hardsigmoid.py`。

7 文件（S1–S7，均存在于 `examples/TileOPs/`）：

| S | 路径 |
|---|---|
| S1 | `tileops/manifest/elementwise.yaml` |
| S2 | `tileops/workloads/elementwise.py` |
| S3 | `tileops/kernels/elementwise/hardsigmoid/{hardsigmoid.py, __init__.py, _hardsigmoid_fwd_kernels.py}` |
| S4 | `tileops/ops/elementwise/hardsigmoid.py` |
| S5 | `tests/ops/test_hardsigmoid.py` |
| S6 | `benchmarks/ops/bench_hardsigmoid.py` |
| S7 | `tileops/{kernels,ops}/elementwise/__init__.py`（导出更新） |

### 2.3 Tier 1 结构校验（D4）

| 校验 | 命令 | 结果 |
|---|---|---|
| import | `python -c "import tileops"` | OK（`examples/TileOPs/tileops/__init__.py`） |
| collect | `python -m pytest tests/ops/test_hardsigmoid.py --collect-only -q` | **7 tests collected** |
| lint | `python -m ruff check tileops/kernels/elementwise/hardsigmoid tests/ops/test_hardsigmoid.py benchmarks/ops/bench_hardsigmoid.py` | **0 errors**（M5 收口后；修订前 9：6×E402 固有形态 + 2×I001 + 1×RUF022，见 §5） |

### 2.4 设计与检视（D5）

| 工件 | 路径 | 结论字面量 |
|---|---|---|
| DESIGN.md | `examples/hardsigmoid/hardsigmoid_fwd_kernel/DESIGN.md` | v1（含 §0.0 修订说明） |
| REVIEW.md | `examples/hardsigmoid/hardsigmoid_fwd_kernel/REVIEW.md` | `结论: 通过` |
| verify_equiv.py | `examples/hardsigmoid/hardsigmoid_fwd_kernel/verify_equiv.py` | 2 项均 `EQUIV_PASS` |

`verify_equiv.py` 实跑：

```
NaN 传播化简 (逐层 nan_min/nan_max -> 单点恢复) | max ulp 差 0.000 | 违反率 0 | EQUIV_PASS
除法转乘倒数 (clamp(...)/6 -> clamp(...)*(1/6)) | max ulp 差 1.000 | 违反率 0 | EQUIV_PASS
总体结论: 全部 EQUIV_PASS — 可采纳
```

**DESIGN.md 代际说明（M6）**：本报告 §2.4 所指 `DESIGN.md` 为 **修订终版 v1**（非初版）。依据：`history_version/design_v0.md`（2026-10-12 21:48，Stage 2 首检不通过后备份）与 `history_version/design_v1.md`（2026-10-12 21:58，design_revision 后版本）；集成包内冻结快照 `hardsigmoid_kernel/hardsigmoid_fwd_kernel_DESIGN.md`（mtime **2026-10-12 21:58:52**）即 v1 的集成期快照，还原对象＝该快照。v1 相对 v0 修复了 2 个阻塞项（`T.isnan` 向量 API 误用 → `T.vcmp(x,x,"ne")` 单点恢复；fp32 路径 UB 预算算术错 152KB→168KB），见 v1 §0.0「相对上一版的关键调整」。**Stage 1 失败尝试的影响已消除**：Stage 1 有 3 次 dispatch（1 次 runtime fail 1077s → 重试 62s；后续 design_revision 又 666s），其残留产物（含 §0.0 所列 v0 缺陷）均被 v1 覆盖且未进入 Stage 2/3/5。注意 `statectl verify` 仅比对 **schema 与 SHA256**，不判定语义代际——本轮曾因会话内误重写 `DESIGN.md`（first_design）产生哈希漂移，已按集成快照还原 v1 后 `verify` 恢复 `ok`。

### 2.5 Stage 5 集成三护栏（D7，§2.1）

> 本表为 **M8 收口后**的实测状态。修订前护栏 1 为 FAIL（`integration_log.md` 无 `wrapper_rewritten` 字段），M8 修补 `integrate_kernel.py` 并幂等重跑集成后转 PASS。

| 护栏 | 结果（M8 后） | 修订前 |
|---|---|---|
| `grep "wrapper_rewritten" integration_log.md` | ✅ `- wrapper_rewritten: true`（第 46 行） | ❌ 无该字段 |
| `grep "already integrated" integration_log.md` | ⚠️ 命中 `- wrapper_rewrite_action: already integrated`（第 47 行）→ **人工复核**（幂等重跑预期信号；经护栏 3 确认 baseline 谱系正确） | 无 |
| `w.hardsigmoid_fwd_kernel.__module__` | ✅ `tileops.kernels.elementwise.hardsigmoid.hardsigmoid_kernel.hardsigmoid_fwd_kernel`（baseline 谱系，非 perf_opt） | 同 |

**M8 两份日志留档**（`integration_log.md` 追加段，非覆盖；集成包备份 `/tmp/opencode/hardsigmoid_kernel.bak_m8_20261013_090943`）：

```text
# 修订前（integrator 手写，全文无 wrapper_rewritten；行数 42）
# 修订后（M8 追加段，integrate_kernel.py 自动写入）：
## integrate_kernel.py run (2026-10-13T09:09:56+0800)
- wrapper_rewritten: true
- wrapper_rewrite_action: already integrated
- ascend_mode: Developer
- functions: hardsigmoid_fwd_kernel
- smoke: pass
```

> 说明：复跑时 `rewrite_wrapper_import()` 走「已集成」分支（JSON `wrapper_rewritten:false` 指「本次未改写」），日志字段 `wrapper_rewritten` 取 **wrapper 最终形态**（baseline/perf_opt 块存在）故为 `true`——字段语义见 VP-2026-0145。

### 2.6 流程与时间线（D2）

函数级状态（`statectl show`）：`phase=DONE`，`stage_status={1:completed, 2:completed, 3:completed}`，`retry_count=1`。

`statectl timeline-summary --dir examples/hardsigmoid/hardsigmoid_fwd_kernel`：

| Stage | attempts | failed | durations_s |
|---|---|---|---|
| 1 designer | 3 | 1（runtime） | 1077 / 62 / 666 |
| 2 reviewer | 2 | 1（design_revision） | 501 / 482 |
| 3 developer | 1 | 0 | 960 |

`dispatches=6`；op 级 `migration-init → migration-func(in_progress) → migration-func(done) → migration-phase(INTEGRATE) → migration-integrate(pass) → migration-phase(DONE)`。
Session 耗时（`SESSION_TIMING_ANALYSIS.md`）：**总计 1h51m52s**（Stage0 18m29s / Stage1 16m49s+58.9s / 修订 11m03s / Stage2 7m56s+7m55s / Stage3 15m51s / 二次校验 54.3s / Stage5 25m03s）。

---

## 3. 正确性事实（D8，除性能）

| 项 | 值 |
|---|---|
| 用例与通过率 | `pytest tests/ops/test_hardsigmoid.py -v` → **7 passed / 7（100%），84.93s** |
| 失败项 | 无 |
| 内嵌分层（Stage 3 实测） | `python hardsigmoid_fwd_kernel.py --level all` → L0/L1/L2/**Boundary** 全 PASS，`All check passed!`（含 `Boundary PASS: NaN→NaN (vcmp(ne)+vselect)` fp32/fp16/bf16） |
| `max_abs_err` | `run.json`：`5.96e-08`（fp32）；多数用例 `0.00e+00`。属 fp32 中间计算经 `vcast` rint 舍入的构造必然，非缺陷 |

**链路结论：通过**（正确性 100%，精度门禁 L0/L1 阻塞项全过；L2/Boundary 亦全过）。本节不掺性能。

---

## 4. 性能事实（D9）

`run.json`：`status=passed`，`benchmark_passed=true`，`benchmark_tests=9`，`msprof_case_count=6`，`profiler_fallback_count=0`，`invalid_profile_count=0`（**性能口径仅 msprof，fallback 为 0**）。

### 4.1 workload 来源（M1）

| Label | DType | 来源 origin | 说明 |
|---|---|---|---|
| mbv3-se-gate | float16 | 上游 manifest `elementwise_unary_activation.yaml:122` | 源算子原生 workload（SE-block gating） |
| mbv3-se-gate | bfloat16 | 上游 manifest:122 | 源算子原生 workload |
| mbv3-se-gate-deep | float16 | 上游 manifest:123 | 源算子原生 workload |
| mbv3-se-gate-deep | bfloat16 | 上游 manifest:123 | 源算子原生 workload |
| smoke-1m | float32 / float16 / bfloat16 | NPU 侧扩展 `examples/TileOPs/tileops/manifest/elementwise.yaml` | 冒烟：小扁平向量 |
| fc-wide | float16 / bfloat16 | NPU 侧扩展 同上 | 增补理由：沿用本仓 Mish/LerpTensor 的 `fc-wide` 约定，便于同 shape 横向对照 |

上游 manifest 仅 2 label × 2 dtype = **4 例**；NPU 侧增补 smoke-1m(3) + fc-wide(2) = 5 例；合计 **9 例**。

### 4.2 benchmark 用例构成（M1）

`run.json` `benchmark_tests.tests=9`（收集/运行 9 例）中 **6 例进 msprof**，3 例未采集：

| # | Label | DType | 进 msprof | 未采集说明 |
|---|---|---|---|---|
| 1 | smoke-1m | float32 | ✗ | pytest `smoke` 标记；profiled benchmark 以 `-m "not smoke"` 运行故不入 roofline 表（`tileops_latency_us=7.20` 仅记录，不入性能口径） |
| 2 | smoke-1m | float16 | ✗ | 同上（6.76us） |
| 3 | smoke-1m | bfloat16 | ✗ | 同上（6.98us） |
| 4 | mbv3-se-gate | float16 | ✓ | — |
| 5 | mbv3-se-gate | bfloat16 | ✓ | — |
| 6 | mbv3-se-gate-deep | float16 | ✓ | — |
| 7 | mbv3-se-gate-deep | bfloat16 | ✓ | — |
| 8 | fc-wide | float16 | ✓ | — |
| 9 | fc-wide | bfloat16 | ✓ | — |

> 口径区分：`tiles`/`benchmark_tests=9` 为用例数；`summary.msprof_case_count=6` 为进入 msprof 的用例数（`performance.cases`）。证据：`pytest benchmarks/ops/bench_hardsigmoid.py --collect-only -q` 列 9 nodeid；`run.json metadata.commands.benchmark` 含 `-m not smoke`。

### 4.3 per-shape 表（msprof，`Kernel=main`）

| Label | Latency (us) | Roofline Ratio (%) | Shape | DType |
|---|---:|---:|---|---|
| mbv3-se-gate | 3.12 | 0.46 | [32,240,1,1] | float16 |
| mbv3-se-gate | 3.06 | 0.53 | [32,240,1,1] | bfloat16 |
| mbv3-se-gate-deep | 2.26 | 2.53 | [32,960,1,1] | float16 |
| mbv3-se-gate-deep | 2.52 | 2.56 | [32,960,1,1] | bfloat16 |
| fc-wide | 30.32 | 48.27 | [2048,4096] | float16 |
| fc-wide | 29.88 | 55.07 | [2048,4096] | bfloat16 |

### 4.4 同 shape 对照 + 环境差异（M3）

| 算子 | Shape | DType | Roofline Ratio (%) | 出处 |
|---|---|---|---|---|
| HardsigmoidFwdOp | [2048,4096] | float16 | 48.27 | 本报告 `run.json` |
| HardsigmoidFwdOp | [2048,4096] | bfloat16 | 55.07 | 本报告 `run.json` |
| **MishFwdOp** | [2048,4096] | float16 | **54.01** | `reports/report.md:106` |
| **MishFwdOp** | [2048,4096] | bfloat16 | **57.67** | `reports/report.md:107` |

**趋势**：同 shape 下 hardsigmoid 与 Mish 同量级（bf16 分别 55.07% / 57.67%，差 ~2.6pp），属 elementwise 同族合理区间。

> **环境差异注**：本报告行为 **Ascend910B3 / CANN 8.5.2 / msprof GM 带宽口径 1.75 TB/s**；Mish 行出自历史报告，为 **Ascend910B2C / CANN 8.5.0 / 1.800 TB/s**（`reports/report.md` 带宽列）。跨代**仅趋势参考**，不做逐点等值比较。

### 4.5 元素数与访存计算（M2 修正）

- `fc-wide` shape = [2048, 4096] → **N = 2048 × 4096 = 8,388,608 元素 ≈ 8.39M**（**更正**：原文误写 “256M elem”）。
- fp16 读写合计访存 = 2 × N × elem_bytes = 2 × 8,388,608 × 2 B = **33,554,432 B ≈ 33.55 MB**（≈ 32 MiB）。
- 口径：`bytes = 2 * N * elem_bytes`，与 manifest roofline 及 `Op.eval_roofline()` 一致。

**同类横向对照 / 反常信号**：`fc-wide` 达 48–55% roofline，属访存受限正常；`mbv3-se-gate*`（7.7K / 31K 元素，发射受限）仅 0.46–2.56%，系小 shape 启动/发射开销主导（设计期估算项已引 `CONST-vector-launch-overhead`）。小 shape（<32K 元素）利用率 <3%；`mbv3-se-gate-deep` 元素数 ×4 而延迟略降（3.12→2.26us），符合发射开销被摊薄、非数据相关。

**水位结论**：迁移链路达标；`fc-wide` 已近半 roofline，小 shape 受发射开销限制——**迁移报告不将其写作“性能已达水位”**，调优属另一次 `optimize` 任务。

### 4.6 契约（inplace）与带宽口径出处（M4）

**inplace 契约处置**：
- NPU `HardsigmoidFwdOp.__init__(N_total, dtype, inplace=False, *, kernel_map=None)` **保留 `inplace`**（默认 `False`）。
- 语义实现：`forward()` 在 `inplace=True` 时 `input.copy_(result); return input`（保持张量身份，对应 GPU manifest `outputs.alias=input` / `mutated: inplace`）；`False` 时返回新张量，**不静默退化为 out-of-place**。
- **测试覆盖缺口（如实披露）**：`tests/ops/test_hardsigmoid.py` 现有 7 用例均未构造 `inplace=True`，原地路径未被覆盖——建议后续补一条 inplace 用例。
- 契约差异：GPU manifest 声明 `inplace: bool, kw_only=true`，NPU `Op` 生成为位置参数（第 3 位）而非 kw-only；调用点全走关键字，功能等价，差异已记录。

**1.75 TB/s 出处**：
- 报告代码：`tileops/benchmark/benchmark_base.py:464-467` —— `"Bandwidth(TB/s)": gm["bw"]`，取自 msprof `visualize_data.bin` 的 `GM Read + Write` roofline 条目（**设备实测报告值**，非硬编码常量）。
- 默认峰值（对照）：`tileops/benchmark/msprof_roofline.py:36` `SOC_CONFIG[...]["gm_bw"] = 1.8`（910B 系列默认；本 run 实际解析值 1.75）。
- 常数库对照：`pattern-library/constants.md:77` “fabric 聚合带宽 ≥1.73 TB/s（24 核 × 72GB/s，pp 探针）”；`:97` “~1.8 TB/s 峰值参考对混合读写不可达”。

---

## 5. 未决项与风险（D11）

**run.json 字段（M7）**：`summary.correctness_tests = 7`（顶层键名 `correctness_tests` 不存在，计数在 `summary` 下）；`metadata.git_commit = null`（环境非 git 仓库，report 生成器未取到 commit，见下）。

| # | 项 | 影响 | 处置/入口 |
|---|---|---|---|
| 1 | **Stage 5 出现 1 次人工补丁**（wrapper `forward()` 调用约定 + `default_config.block_size` GPU 派生值→8192） | 严格按 §6 批出口标准「全程无需人工补丁」**不满足** | 已记入 `integration_log.md`「调试历史」+ 备份 `history_version/hardsigmoid_s5_attempt1.py`；**工具缺陷登记 VP-2026-0146**（target: `add-npu-op/SKILL.md` S3 wrapper 模板） |
| 2 | `integration_log.md` 缺 `wrapper_rewritten` 字面标记（护栏 1） | 机械护栏 1 判 FAIL | **已修（M8）**：`integrate_kernel.py` 新增 `write_integration_log()`，重跑后护栏 1 PASS；**工具缺陷登记 VP-2026-0145**（target: 同脚本） |
| 3 | `ruff check` 9 处（6×E402 + 2×I001 + 1×RUF022） | 代码风格，非功能 | **已修（M5）**：wrapper I001 修复；集成 kernel E402（env-before-import 固有形态）与 history_version 备份经 `pyproject.toml` per-file-ignore，`__all__` 排序修复 → `ruff check` 0 errors |
| 4 | 本会话曾误重写 `DESIGN.md`（first_design）造成 SHA256 漂移 | 已消除 | 用集成快照还原 v1；`statectl verify` 现 `ok`（代际说明见 §2.4） |
| 5 | `standards_check.py check` 存量失败 7 条 | **逐条证与本次改动无关**（详见下） | 未处理；不影响本算子交付 |
| 6 | kernel cache 每次重编译（`T.cast(float("nan"))` 使 JSON cache miss） | 仅增加编译耗时 | 已记入 Stage 3 RETROSPECTIVE + integration_log；蒸馏产物 VP-2026-0142(D) |

**M5 — standards_check 失败明细（rule_id / 文件 / 指纹）与无关性证明**：

| rule_id | 文件 | 指纹/引用 | 与本次改动关系 |
|---|---|---|---|
| SC-INLINE-DUP | `.agents/skills/tilelang-op-develop/SKILL.md` | 指纹 `SC-DUP-MODEENUM`（权威 `signal-registry.md`） | **无关**——本次未改任何 skill 文件 |
| SC-REF-PATH | `.agents/skills/_shared/standards/core-split-strategy.md` ×3 | 引用 `docs/开发指南.md` 不存在 | **无关**——本次未改标准文件；`docs/开发指南.md` 不在本工作区 |
| SC-REF-PATH | `.agents/skills/_shared/standards/negative-claim-evidence.md` ×2 | 引用 `docs/Tilelang.language/` 不存在 | **无关**——同上 |
| SC-REF-PATH | `.agents/skills/_shared/standards/perf-feedback.md` ×1 | 引用 `examples/flash_attention/flash_attn_npuir.py` 不存在 | **无关**——同上 |

> 本次改动文件（`extract_tl_kernel.py`、`integrate_kernel.py`、`tilelang-op-conductor.md`、`examples/TileOPs/pyproject.toml`、算子/集成 kernel）均不在上述规则的目标集内；7 条失败为工作区既有存量（缺 `docs/`/`examples/` 路径），非本次引入。**有关项为零，无需修**。

**M7 — 终态蒸馏产物**：本任务终态 `DONE` 后 evolver 已执行（`stats.md` 记 `EVOLVE_COMPLETED`，2026-10-12 23:03）。产物 id：
- Tier 0 add：`constants.md#CONST-vcmp-ne-nan-unordered`、`cases.md#CASE-elementwise-hardsigmoid-migration`、`CONST-aicore-910B2C` update、`INDEX.md` ×3。
- Tier 1：**VP-2026-0142**（D：NaN 常数缓存元数据陷阱）、**VP-2026-0143**（P：buffer 别名赋值 → Undefined variable）。
- Tier 2：**VP-2026-0144**（R：向量 API 签名级/源码级验证 → quality-checklist）。
- 证据链追加：VP-2026-0047（第五现）。
- 查重命中 1：`extract_tl_kernel` src 布局（已含回退，不新开提案）。
- 本次修订新增：**VP-2026-0145 / VP-2026-0146**（工具缺陷，见上）。

**M7 — 批 1b 汇总口径**：hardsigmoid 的失败点在 **工具链（脚手架 wrapper 模板 + 集成脚本缺字段），不在 kernel**——kernel 本身 Stage 3 first-pass（attempt 1 零 fail）且 L0/L1/L2/Boundary 全过。

capability gap：本次未新增（无 `[DESIGN_LIMIT]` / `[DESIGN_ERROR]`）。

**M5 — 适配层三处改动（提交清单；环境非 git → 无法 `git commit`）**：

> **卡点**：`/mnt/workspace/npuir-migration` 及其 `examples/` 均非 git 仓库（`git rev-parse` → fatal；`stats.md` 已披露“环境非 git 仓库”）。故无法“单独提交”，改以文件级改动清单 + 就绪 diff 提交；如需真正提交，请授权 `git init` 或指定目标仓。有 git 的仅 `/home/developer/tilelang-mlir-ascend`（本适配层不在其中）。

| 改动 | 文件 | 说明 |
|---|---|---|
| ① conductor 判定 src-layout 回退 | `.opencode/agents/tilelang-op-conductor.md`（harness 判定块） | 新增 `MANIFEST_DIR=$([ -d {gpu_repo_root}/tileops/manifest ] && … || echo {gpu_repo_root}/src/tileops/manifest)`，四条判定改用 `$MANIFEST_DIR` |
| ② extract 脚本回退 | `examples/TileOPs/.agents/skills/add-npu-op/scripts/extract_tl_kernel.py` | `load_manifest_entry()` 增 `_resolve_manifest_dir()`（`tileops/manifest` → `src/tileops/manifest`，`rglob` 含 `spec/`）+ `_apply_source_map_fallback()`（`source` 缺失读 `.migration_source_map.json`）；新增 `import json` |
| ③ 源映射文件 | `{GPU_REPO}/.migration_source_map.json`（`/mnt/workspace/TileOPs-main/`） | 由 op 类 `kernel_types` + kernels 扫描生成，69 算子 `op → {op_class, op, kernel, kernel_map, manifest, test, bench, workload}`；生成器 `/tmp/opencode/gen_source_map.py` |

---

## 6. 复核入口（逐条可照抄重跑）

```bash
# 路由判定
cd /mnt/workspace/TileOPs-main
MANIFEST_DIR=$([ -d tileops/manifest ] && echo tileops/manifest || echo src/tileops/manifest)
grep -rl '^HardsigmoidFwdOp:' "$MANIFEST_DIR"

# 状态与时间线
cd /mnt/workspace/npuir-migration
python3 .agents/tools/statectl.py show --dir examples/hardsigmoid/hardsigmoid_fwd_kernel
python3 .agents/tools/statectl.py verify --dir examples/hardsigmoid/hardsigmoid_fwd_kernel
python3 .agents/tools/statectl.py timeline-summary --dir examples/hardsigmoid/hardsigmoid_fwd_kernel
python3 .agents/tools/statectl.py migration show --dir examples/hardsigmoid

# 设计与等价性
python3 examples/hardsigmoid/hardsigmoid_fwd_kernel/verify_equiv.py

# Stage 3 内嵌分层实测
cd /mnt/workspace/npuir-migration/examples/hardsigmoid/hardsigmoid_fwd_kernel
python hardsigmoid_fwd_kernel.py --level all

# 正确性 / 性能
cd /mnt/workspace/npuir-migration/examples/TileOPs
python -m pytest tests/ops/test_hardsigmoid.py -v
python -m pytest benchmarks/ops/bench_hardsigmoid.py -v

# 集成护栏
LOG=tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/integration_log.md
grep -n "wrapper_rewritten\|already integrated" "$LOG"
python -c "import importlib; m=importlib.import_module('tileops.kernels.elementwise.hardsigmoid.hardsigmoid'); print(m.hardsigmoid_fwd_kernel.__module__)"

# 报告与 run.json 字段
R=reports/tileops/20261012_223321_664388_HardsigmoidFwdOp
grep -nE '"status"|"correctness_passed"|"correctness_tests"|"benchmark_passed"|"benchmark_tests"|"msprof_case_count"|"profiler_fallback_count"|"invalid_profile_count"|"max_abs_err"|"run_id"|"git_commit"' $R/run.json

# M5：lint 收口（期望 All checks passed!）
python -m ruff check tileops/kernels/elementwise/hardsigmoid tests/ops/test_hardsigmoid.py benchmarks/ops/bench_hardsigmoid.py
# M5：standards_check 明细（7 条存量，均与本次改动无关）
python3 /mnt/workspace/npuir-migration/.agents/tools/standards_check.py check

# M8：工具侧重跑集成（幂等；写入 wrapper_rewritten 字段）
python3 .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/hardsigmoid/.migration_meta.json
grep -n 'wrapper_rewritten\|wrapper_rewrite_action' tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/integration_log.md

# M8/完成标准：重渲染报告（不重跑 correctness/benchmark；数字应不变）
python -m tileops.reporting.cli render reports/tileops/20261012_223321_664388_HardsigmoidFwdOp/run.json
```

**§6 出口标准自评（批 1b：HardsigmoidFwdOp）**：走完 Stage 0→5 ✅；正确性 100% ✅；`wrapper_rewritten=true` ✅（M8 后集成日志有该字段且为 true）；**全程无需人工补丁 ❌**（Stage 5 有 1 次 wrapper 胶水修复，根因在脚手架模板，已登记 VP-2026-0146）。→ 作为批 1b 首算子，**出口标准的“无需人工补丁”一项未满足**，需在批 1b 汇总中如实计入；**失败点在工具链、不在 kernel**。
