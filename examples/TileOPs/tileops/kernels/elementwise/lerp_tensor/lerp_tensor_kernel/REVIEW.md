# _make_lerp_tensor_kernel 检视报告（LerpTensorFwdOp 迁移复审）

## 检视结论

结论: **通过**

- 精度: **通过**（8/8，`correctness_passed=true`）
- 性能: **通过**（9/9 benchmark 用例全过，7 个非 smoke msprof 用例全部 `profiler_valid=true`，
  `profiler_fallback_count=0`，`invalid_profile_count=0`，无告警）
- 阻塞项: 0

## 检视元信息

- 文档: `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel_DESIGN.md`
  （`DESIGN.md` 为同内容顶层副本，用于满足提交检查的 basename 约定）
- 任务类型: 迁移（上游基线算子接入 + 实跑验收取证）
- 本轮复审范围: 可运行性、分派正确性、证据完整性、性能口径。本轮**不改算子代码**。
- 检视维度: 4
- 阻塞级问题数: 0
- 建议级问题数: 1（见末尾「建议」）

## 检视详情

### 1. 可运行性 — pass

- Ascend910B3 上精度 8/8 通过，覆盖 fp16/bf16/fp32 三 dtype 与 smoke/16m/64m/256m shape，
  含三输入 broadcast 与非连续输入路径（见 `tests/ops/test_lerp_tensor.py`）。
- benchmark 9/9 通过，7 个非 smoke 用例全部产出有效 msprof 记录。

### 2. 分派正确性 — pass

- `tileops/kernels/elementwise/lerp_tensor/lerp_tensor.py` **默认导入的是调优版**：
  `from .lerp_tensor_kernel.perf_opt._make_lerp_tensor_kernel import _make_lerp_tensor_kernel`
  （第 62 行），基线版导入被注释保留（第 60 行），因此本轮实测即为 perf_opt 调优版，
  与 `perf_records.jsonl` 记录的 `artifact_sha256` 一致。
- 复核 `_BLOCK_SIZE_FP32 = 8192`（Stage 4 tuned，注释记为 2048→8192、16M fp32 −3.1%）与
  `opt_log.md` 的调优结论一致，无「调优代码已写但未接线」的情况。

### 3. 证据完整性 — pass

- 六区: Op / Kernel / Manifest / Workload / 精度测试 / Benchmark 均在基线到位。
- 六证据: `DESIGN.md`、`REVIEW.md`、`integration_log.md`、`RETROSPECTIVE.md`、`run.json`、
  `report.md` 齐全。
- 优化证据: `perf_opt/opt_log.md`（历史调优记录）+ `perf_opt/perf_records.jsonl`（本轮实测）。

### 4. 性能口径 — pass

- 报告 `status=passed`，Ratio 区间 66.48%–80.47%，达到验收文档对带宽受限型「≥65% 为合格、
  ≥73% 为良好」的下限要求。
- 按 P0 裁决规则（剔 smoke、剔 cache-resident、取最大元素数档）：裁决档为
  **elementwise-256m**，fp16 **67.58%** / bf16 **66.48%**；与次大档 elementwise-64m
  （73.46% / 72.87%）相差 **5.88pp**，**按规则判定「未收敛」**，如实记录，不美化为收敛。
- 256m 档等效带宽约 1.2 TB/s < 1.75 TB/s 峰值，不属于缓存塌陷档，作为裁决档成立。

## 建议

1. elementwise-256m 与 64m 差 5.88pp 未收敛，说明大 shape 下仍有取数/发射开销未摊平
   （与 `opt_log.md` 中 256M 档的调优结论一致）。若后续要提升，优先在 256m 档做一轮
   针对性调优，而不是继续压小 shape。
2. 本轮未重测 torch.lerp 基线。`opt_log.md` 中「256M ~2827us」来自 CANN 8.5.0 / 910B2C，
   与本机（CANN 9.0.0 / 910B3）不可直接比较，故本轮不声明提升倍数。
