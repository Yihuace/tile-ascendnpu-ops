# Integration Log -- RopeLongRopeOp

- 分支: `op/position_encoding/rope-longrope-lvpanhao`
- base: **`op/position_encoding/rope-neox-lvpanhao`（PR #55，stacked）**，最终目标 `integration/npu-ops`
- family: `position_encoding`
- 依赖说明: `position_encoding` 族的骨架文件（`manifest/position_encoding.yaml`、
  `workloads/position_encoding.py`、两个族的 `__init__.py`）由 PR #55 引入且尚未合并。
  为保证「一个 PR 只包含一个算子」，本分支 stacked 在 #55 之上；#55 合入后 GitHub 会自动
  把本 PR 的 base 切回 `integration/npu-ops`。
- 源算子: GPU 仓 `tileops/ops/rope.py` 的 `RopeLongRopeOp`（LongRoPE 变体）。
  kernel 侧与 RopeNeox 共用 `_make_rope_neox_1d` / `_make_rope_neox_2d`（GPU 仓无
  `_make_rope_longrope_*`）；LongRoPE 的差异（`rescale_factors` + `scaling_factor`）
  全部落在 Op 层。
- 迁移来源: `/home/developer/tilelang-mlir-ascend/examples/TileOPs`（旧框架）
  的 `tileops/kernels/position_encoding/rope_longrope/`、`tileops/ops/position_encoding/rope_longrope.py`，
  按 NPU 单算子包布局重新落位；测试与 benchmark 由旧仓 `tests/ops/test_rope.py`、
  `benchmarks/ops/bench_rope.py` 中的 LongRoPE 部分拆出。
- 清理动作: 未纳入 `__pycache__`、`.migration_meta.json`、`history_version/`、
  `perf_opt/logs/`、`.stage_state.json`、`.task_timeline.jsonl`（均为本地产物或历史状态）。
- 运行环境: Ascend910B3 x1 / CANN 9.0.0 / driver 25.5.0 / torch 2.7.1 /
  torch_npu 2.7.1.post4 / tilelang 0.1.2（dev root build `/home/developer/tilelang-mlir-ascend/build`）
- 关键环境变量: `PYTHONPATH=/home/developer/tilelang-mlir-ascend:.`
  —— 安装态 tilelang 缺 `tilelang/utils/npu_utils.py`，必须走源码树。

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261025_234635_085088_RopeLongRopeOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261025_234635_085088_RopeLongRopeOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261025_234635_085088_RopeLongRopeOp/report.html`
- report_status: **passed**
- correctness: **pass**（11/11）
- benchmark: **pass**（6/6 用例；3 个非 smoke 用例 msprof 全部有效，
  `profiler_fallback_count=0`，`invalid_profile_count=0`）
- warnings: 无

### 精度覆盖

`test_rope_longrope.py`：`1d`/`2d` 两种 layout × fp16/bf16/fp32，外加 `seq_len=1`、
`head_dim=16`、`seq_len=512` 等边界，共 11 例。参考实现独立计算 LongRoPE 频率表
（`divisor = rescale_factors[k] * base^(2k/d)`，`scaling_factor = sqrt(1+log(scale)/log(orig_max_pos))`），
不复用 Op 内部实现。

### benchmark 数值（本机实测，run.json `performance.cases`）

| label | dtype | shape | latency (us) | roofline (%) | torch-ref (us) | 相对参考倍数 |
|---|---|---|---:|---:|---:|---:|
| longrope-1d-2k-d128-fp16 | float16 | [2048, 128] | 3.92 | 15.29 | 381.83 | 97.40× |
| longrope-1d-8k-d128-bf16 | bfloat16 | [8192, 128] | 6.30 | 38.04 | 368.14 | 58.44× |
| longrope-2d-b1-s8k-h32-d128-fp16 | float16 | [1, 8192, 32, 128] | 81.58 | 54.74 | 519.99 | 6.37× |

> `torch-ref` 为测试内同一次运行采集的**纯 PyTorch 参考实现**时延（同 `bm.profile` 口径），
> 不是调优过的 torch 算子；倍数仅用于同 run 内对照，对外引用需注明该口径。

### P0 裁决（按 `_infra_ACCEPTANCE.md` §4 规则）

- 剔 smoke 档（`longrope-1d-256-d64-smoke`）。
- 剔 cache-resident 档：2d 档等效带宽 ≈1.67 TB/s < 1.75 TB/s 峰值，保留。
- 取元素数最大档 → **`longrope-2d-b1-s8k-h32-d128-fp16`**（33.5M 元素），**Ratio 54.74%**。
- 收敛判定：与次大档（1d-8k，38.04%）相差 **16.70pp ≥ 3pp** → **未收敛**，如实记录。

## 调试历史

- 唯一一次失败：`NameError: name 'Optional' is not defined`
  —— 追加 `RopeLongRopeOpWorkload` 时用到 `Optional[torch.Tensor]`，而
  `workloads/position_encoding.py` 原本没有该导入。补 `from typing import Optional` 并规范
  导入顺序后，模块导入与 manifest 解析一次通过。
- 精度测试首跑即 11/11 通过（112.04s），未做容差调整。
- 报告生成（精度 + benchmark，含 4 档 msprof）耗时约 10 分钟，NPU 独占（采集前确认无其他进程）。

## 已知限制

- 本分支依赖 PR #55 先合入（stacked）。在 #55 合入前，本 PR 的 base 不是 `integration/npu-ops`。
- 未做额外的调优轮次：kernel 沿用旧框架已调优的 `perf_opt` 版本
  （`perf_opt/_make_rope_neox_1d.py`、`_2d.py`，`rope_longrope.py:90-91` 默认导入），
  本轮只做迁移 + 实跑取证，`perf_records.jsonl` 记录的是该版本的实测终态。
- `longrope-1d-2k-d128-fp16` 仅 15.29%，属小 shape 固定开销主导，未据此下结论。

## issues

- 无。未降低测试容差、未新增 skip、未删除用例。
