# Integration Log -- ArgmaxFwdOp

- 分支: `op/reduction/argmax-lvpanhao`
- base: `integration/npu-ops` @ `da8b5dc`
- family: `reduction`
- 集成方式: 上游基线已含六区（commit `93891e3` "Renew report and add argmax kernel (#193)"，
  文件清单见下），**本轮不改算子代码**，补齐验收证据并实跑验证。
- 算子文件: `tileops/ops/reduction/argmax.py`、`tileops/kernels/reduction/argmax/argmax.py`、
  `tileops/kernels/reduction/argmax/argmax_kernel/_argreduce_kernel.py`、
  `tests/ops/test_argmax.py`、`benchmarks/ops/bench_argmax.py`
- 运行环境: Ascend910B3 x1 / CANN 9.0.0 / driver 25.5.0 / torch 2.7.1 /
  torch_npu 2.7.1.post4 / tilelang 0.1.2（dev root build `/home/developer/tilelang-mlir-ascend/build`）
- 关键环境变量: `PYTHONPATH=/home/developer/tilelang-mlir-ascend:.`
  —— 必须使用源码树；site-packages 里的 tilelang 安装包缺 `tilelang/utils/npu_utils.py`。

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261025_224554_735584_ArgmaxFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261025_224554_735584_ArgmaxFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261025_224554_735584_ArgmaxFwdOp/report.html`
- report_status: **partial**
- correctness: **pass**（42/42；`summary.correctness_passed=true`，avg max abs error 0.00e+00）
- benchmark: **partial**（8 workload 中 2 个失败；3 个非 smoke 用例 msprof 有效，
  `profiler_fallback_count=0`，`invalid_profile_count=0`）

### benchmark 数值（本机实测，取自 run.json `performance.cases`）

| label | dtype | shape | latency (us) | roofline (%) |
|---|---|---|---:|---:|
| hidden-state-argmax | float16 | [2048, 4096] | 30.8200 | 31.11 |
| hidden-state-argmax | bfloat16 | [2048, 4096] | 52.7200 | 21.79 |
| 3d-non-last-axis-argmax | float16 | [4, 128, 4096] | 13.7200 | 24.60 |

## 调试历史

- attempt 1 —— 环境缺模块
  - 症状：`pytest tests/ops/test_argmax.py -m smoke` 32 个用例全 `ModuleNotFoundError`；
    实际报 `No module named 'tilelang.utils.npu_utils'`（`_argreduce_kernel.py:54`）。
  - 定位：site-packages 中的 tilelang 安装包 `utils/` 下缺 `npu_utils.py`（也无 `npu_arch.py`），
    而源码树 `/home/developer/tilelang-mlir-ascend/tilelang/utils/` 有。
  - 修复：`PYTHONPATH=/home/developer/tilelang-mlir-ascend:.`，让 tilelang 走源码树 +
    dev root build。修后 32/32 smoke 通过（224.07s）。

- attempt 2 —— 全量精度 + 报告
  - `tileops.reporting.cli run --op ArgmaxFwdOp` → correctness 42/42 通过，
    但 benchmark 有 2 个用例失败：`RuntimeError: NPU IR opt failed: ...
    error: custom op 'syn_instr_mode' is unknown`。

- attempt 3 —— 失败根因双向验证
  - `syn_instr_mode` 由 `src/target/codegen_npuir.cc:1053` 为 `hivm.hir.sync_block_set`
    生成；`tilelang/utils/npu_utils.py:362 get_npucompiler_path()` 用
    `shutil.which('bishengir-compile')`，实际命中 CANN 9.0.0 自带版本，而非
    `BISHENGIR_ROOT_PATH` 指向的 `3rdparty/AscendNPU-IR/build/install`。
  - PATH 前置 3rdparty 版本后：解析阶段通过（`AscendNPU IR compile success`），
    但下游 `hivmc` 报 `Unknown command line argument '--enable-cpu-trace-intrinsic=false'`
    与 `'--enable-lir-compile=true'`，`kernel.o` 未生成。
  - 结论：两条路径版本互斥，属环境级工具链缺口，非算子缺陷 →
    登记 `.agents/evolution/capability-gaps.md` **CG-2026-0018**。

- 数据质量处置
  - 首次 benchmark 与另一并发任务（另一工作树的 `AbsFwdOp` benchmark）同抢同一张 NPU，
    时延数据不可信，**该轮报告已删除并重跑**；
  - 本文所有数值来自 NPU 空闲（`npu-smi info` 无运行进程）状态下的一轮采集。

## 已知限制

- `lm-head-argmax`（`[4, 102400]`，fp16/bf16）在本机无法编译，原因见 CG-2026-0018。
  该两档属 kernel tiled 路径（`_argreduce_kernel.py:180` / `:368`，bm=1 强制）。
- 本轮未重新测量 baseline。`perf_opt/opt_log.md` 中的历史 baseline 来自
  CANN 8.5.0 / Ascend910B2C（48 Vector 核），与本机（CANN 9.0.0 / 910B3）不可直接比较，
  **故本轮不声明 baseline → optimized 的提升倍数**，只交付实测终态数据。

## issues

- `run.json` warnings 含 `correctness passed, but the benchmark stage did not produce
  valid records` 与 `benchmark pytest reported 2 failed/error cases`，均对应上方已知限制。
- 未降低任何测试容差、未新增 skip、未删除用例。
