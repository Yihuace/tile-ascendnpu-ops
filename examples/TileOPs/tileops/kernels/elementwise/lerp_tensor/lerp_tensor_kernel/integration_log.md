# Integration Log -- LerpTensorFwdOp

- 分支: `op/elementwise/lerp-tensor-lvpanhao`
- base: `integration/npu-ops` @ `da8b5dc`
- family: `elementwise`
- 集成方式: 六区在上游基线已到位（`ref_api: torch.lerp`，Tensor-weight 三输入变体），
  **本轮不改算子代码**，补齐验收证据并实跑验证。
- 分派说明: `tileops/kernels/elementwise/lerp_tensor/lerp_tensor.py:62` 默认导入
  `lerp_tensor_kernel/perf_opt/_make_lerp_tensor_kernel.py`，即**本轮实测的是 Stage 4 调优版**；
  基线版导入在第 60 行被注释保留。
- 运行环境: Ascend910B3 x1 / CANN 9.0.0 / driver 25.5.0 / torch 2.7.1 /
  torch_npu 2.7.1.post4 / tilelang 0.1.2（dev root build `/home/developer/tilelang-mlir-ascend/build`）
- 关键环境变量: `PYTHONPATH=/home/developer/tilelang-mlir-ascend:.`
  —— 必须使用源码树；site-packages 里的 tilelang 安装包缺 `tilelang/utils/npu_utils.py`。
- 采集条件: `npu-smi info` 确认无其他进程占用后单独采集，无并发干扰。

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261025_230253_045981_LerpTensorFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261025_230253_045981_LerpTensorFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261025_230253_045981_LerpTensorFwdOp/report.html`
- report_status: **passed**
- correctness: **pass**（8/8）
- benchmark: **pass**（9/9 用例；7 个非 smoke 用例 msprof 全部有效，
  `profiler_fallback_count=0`，`kernel_filter_fallback_count=0`，`invalid_profile_count=0`）
- warnings: 无

### benchmark 数值（本机实测，取自 run.json `performance.cases`）

| label | dtype | shape | latency (us) | roofline (%) |
|---|---|---|---:|---:|
| elementwise-16m | float16 | [4096, 4096] | 71.4800 | 80.47 |
| elementwise-16m | bfloat16 | [4096, 4096] | 74.4200 | 77.29 |
| elementwise-16m | float32 | [4096, 4096] | 152.6400 | 80.37 |
| elementwise-64m | float16 | [8192, 8192] | 378.1200 | 73.46 |
| elementwise-64m | bfloat16 | [8192, 8192] | 381.9800 | 72.87 |
| elementwise-256m | float16 | [16384, 16384] | 1772.8000 | 67.58 |
| elementwise-256m | bfloat16 | [16384, 16384] | 1802.3800 | 66.48 |

### P0 裁决（按 `_infra_ACCEPTANCE.md` §4 规则）

- 剔除 smoke 档（`smoke-1m`）。
- 剔除 cache-resident 档：256m 档等效带宽约 1.2 TB/s < 1.75 TB/s 峰值，不属于缓存塌陷档，保留。
- 取元素数最大档 → **elementwise-256m**。
- 收敛判定：P0（67.58% / 66.48%）与次大档 elementwise-64m（73.46% / 72.87 %）
  相差 **5.88pp ≥ 3pp** → **未收敛**，如实记录。

## 调试历史

- 本轮无失败用例，无需修复。
- 环境准备沿用 ArgmaxFwdOp 任务已定位的结论：安装态 tilelang 缺
  `tilelang/utils/npu_utils.py`，必须 `PYTHONPATH=/home/developer/tilelang-mlir-ascend:.`
  让 tilelang 走源码树 + dev root build。
- 单次 full 报告耗时约 20 分钟（含 256m 档 msprof 采样），期间独占 NPU。

## 已知限制

- 未在本机重测 torch.lerp 基线。`perf_opt/opt_log.md` 中「256M ~2827us」的对照值来自
  CANN 8.5.0 / Ascend910B2C（48 Vector 核），与本机工具链不同，**故本轮不声明提升倍数**。
- elementwise-256m 与 64m 差 5.88pp 未收敛（见上），属既有调优结论的延续，非本轮引入。

## issues

- 无。未降低测试容差、未新增 skip、未删除用例。
