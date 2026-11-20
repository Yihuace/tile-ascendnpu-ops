# Integration Log -- RopeLlama31Op

- 分支: `op/position_encoding/rope-llama31-lvpanhao`
- base: **`op/position_encoding/rope-neox-lvpanhao`（PR #55，stacked）**，最终目标 `integration/npu-ops`
- family: `position_encoding`
- 依赖说明: `position_encoding` 族骨架（`manifest/position_encoding.yaml`、
  `workloads/position_encoding.py`、族的两个 `__init__.py`）由 PR #55 引入且尚未合并。
  为保证「一个 PR 只包含一个算子」，本分支 stacked 在 #55 之上；#55 合入后 GitHub 会自动
  把 base 切回 `integration/npu-ops`。
  与 `RopeLongRopeOp`（PR #61）为**兄弟分支**，两者都基于 #55，互不依赖。
- 源算子: GPU 仓 `tileops/ops/rope.py` 的 `RopeLlama31Op`。kernel 侧与 RopeNeox 共用
  `_make_rope_neox_1d` / `_make_rope_neox_2d`；Llama 3.1 的差异（分段缩放频率表）全部落在 Op 层。
- 迁移来源: `/home/developer/tilelang-mlir-ascend/examples/TileOPs`（旧框架）的
  `tileops/kernels/position_encoding/rope_llama31/`、`tileops/ops/position_encoding/rope_llama31.py`；
  测试与 benchmark 由旧仓 `tests/ops/test_rope.py`、`benchmarks/ops/bench_rope.py` 的
  Llama 3.1 部分拆出。
- 清理动作: 未纳入 `__pycache__`、`.migration_meta.json`、`history_version/`、
  `perf_opt/logs/`、`.stage_state.json`、`.task_timeline.jsonl`、`integration_report.json`。
- 运行环境: Ascend910B3 x1 / CANN 9.0.0 / driver 25.5.0 / torch 2.7.1 /
  torch_npu 2.7.1.post4 / tilelang 0.1.2（dev root build `/home/developer/tilelang-mlir-ascend/build`）
- 关键环境变量: `PYTHONPATH=/home/developer/tilelang-mlir-ascend:.`
  —— 安装态 tilelang 缺 `tilelang/utils/npu_utils.py`，必须走源码树。

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261026_000429_461405_RopeLlama31Op/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261026_000429_461405_RopeLlama31Op/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261026_000429_461405_RopeLlama31Op/report.html`
- report_status: **passed**
- correctness: **pass**（11/11）
- benchmark: **pass**（6/6 用例；3 个非 smoke 用例 msprof 全部有效，
  `profiler_fallback_count=0`，`invalid_profile_count=0`）
- warnings: 无

### benchmark 数值（本机实测，run.json `performance.cases`）

| label | dtype | shape | latency (us) | roofline (%) | torch-ref (us) | 相对参考倍数 |
|---|---|---|---:|---:|---:|---:|
| llama31-1d-2k-d128-fp16 | float16 | [2048, 128] | 3.98 | 15.06 | 361.69 | 90.88× |
| llama31-1d-8k-d128-bf16 | bfloat16 | [8192, 128] | 6.28 | 38.17 | 357.14 | 56.87× |
| llama31-2d-b1-s8k-h32-d128-fp16 | float16 | [1, 8192, 32, 128] | 81.52 | 54.78 | 547.52 | 6.72× |

> `torch-ref` 为测试内同一次运行采集的**纯 PyTorch 参考实现**时延（同 `bm.profile` 口径），
> 不是调优过的 torch 算子；倍数仅用于同 run 内对照。

### P0 裁决（按 `_infra_ACCEPTANCE.md` §4 规则）

- 剔 smoke 档（`llama31-1d-256-d64-smoke`）。
- 剔 cache-resident 档：2d 档等效带宽 ≈1.67 TB/s < 1.75 TB/s 峰值，保留。
- 取元素数最大档 → **`llama31-2d-b1-s8k-h32-d128-fp16`**（33.5M 元素），**Ratio 54.78%**。
- 收敛判定：与次大档（1d-8k，38.17%）相差 **16.61pp ≥ 3pp** → **未收敛**，如实记录。

## 调试历史

- 本轮无失败用例：补齐六区与族注册后，模块导入与 `load_manifest()` 一次通过，
  精度测试首跑即 11/11（113.73s），未做容差调整。
- 报告生成（精度 + benchmark，含 4 档 msprof）约 13 分钟；采集期间 NPU 无其他进程竞争
  （`npu-smi info` 已确认）。

## 已知限制

- 本分支依赖 PR #55 先合入（stacked）；#55 合入前 base 不是 `integration/npu-ops`。
- 与 PR #61（RopeLongRopeOp）会修改同一批族文件，合并时后合并者需处理冲突。
- 未做额外调优轮次：kernel 沿用旧框架已调优的 `perf_opt` 版本
  （`perf_opt/_make_rope_neox_1d.py`、`_2d.py`），本轮只做迁移 + 实跑取证，
  `perf_records.jsonl` 记录该版本的实测终态。
- `llama31-1d-2k-d128-fp16` 仅 15.06%，属小 shape 固定开销主导，未据此下结论。

## issues

- 无。未降低测试容差、未新增 skip、未删除用例。
