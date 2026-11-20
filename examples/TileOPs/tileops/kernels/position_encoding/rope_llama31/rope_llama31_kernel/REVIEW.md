# RopeLlama31Op 迁移检视报告

## 检视结论

结论: **通过**

- 精度: **通过**（11/11，`correctness_passed=true`）
- 性能: **通过**（benchmark 6/6 用例通过，3 个非 smoke msprof 用例全部 `profiler_valid=true`，
  `profiler_fallback_count=0`、`invalid_profile_count=0`，无告警）
- 阻塞项: 0

## 检视元信息

- 设计文档: `rope_llama31_kernel/DESIGN.md`（与 `_make_rope_neox_1d_DESIGN.md` 同内容）
  以及 `_make_rope_neox_1d_DESIGN.md` / `_make_rope_neox_2d_DESIGN.md`
- 任务类型: 迁移（GPU TileOPs → NPU npuir），六区新建
- 源算子: GPU 仓 `tileops/ops/rope.py` 的 `RopeLlama31Op`（Llama 3.1 变体，
  kernel 侧复用 `_make_rope_neox_1d/_2d`）
- 检视维度: 5（语义一致性 / 六区与注册 / 精度覆盖 / 性能口径 / 依赖与流程）
- 阻塞级问题数: 0
- 建议级问题数: 1（见末尾）

## 检视详情

### 1. 语义一致性 — pass

- Llama 3.1 RoPE 与 neox 共用同一旋转，差异只在频率表的分段缩放：
  高频维（`wavelen < high_freq_wavelen`）保留原频率；低频维（`wavelen > low_freq_wavelen`）
  除以 `scale_factor`；中间维按 `smooth` 线性混合两者。
- 该差异由 **Op 层**实现，kernel 只做旋转，与源算子分层一致，迁移未改变算法归属。
- 测试侧用独立实现的 `_compute_llama31_freqs`（对照 Meta Llama 3.1 参考实现）生成 cos/sin，
  不复用 Op 内部实现，避免「自己验证自己」。

### 2. 六区与注册 — pass

- Op: `tileops/ops/position_encoding/rope_llama31.py`
- Kernel: `tileops/kernels/position_encoding/rope_llama31/`（`rope_llama31.py` 为包装，
  实际分派 `perf_opt/_make_rope_neox_1d.py` / `_2d.py`，见第 4 节）
- Manifest: `position_encoding.yaml` 新增 `RopeLlama31Op`（`variant_of: RopeNeoxOp`，
  带 `scale_factor` / `low_freq_factor` / `high_freq_factor` / `original_max_position` 参数）
- Workload: `workloads/position_encoding.py` 新增 `RopeLlama31OpWorkload`
- 测试: `tests/ops/test_rope_llama31.py`；Benchmark: `benchmarks/ops/bench_rope_llama31.py`
- 注册已核对：族内两个 `__init__.py` 均导出新符号；`load_manifest()` 实跑返回
  `['RopeNeoxOp', 'RopeLlama31Op']`。

### 3. 精度覆盖 — pass

11 个用例覆盖 `1d`/`2d` 两种 layout、fp16/bf16/fp32 三种 dtype，以及 `seq_len=1`、
`head_dim=16`、`seq_len=512` 等边界。容差按 dtype 分层（fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2），
未放宽。用例额外覆盖分段缩放的三个区间（高频保持、低频缩放、中间混合）。

### 4. 性能口径 — pass

- 报告 `status=passed`，Ratio 区间 15.06%–54.78%。
- 按 P0 裁决规则（剔 smoke、剔 cache-resident、取元素数最大档）：
  裁决档 **`llama31-2d-b1-s8k-h32-d128-fp16`**（`[1, 8192, 32, 128]`，33.5M 元素）
  → **54.78%**（等效带宽 ≈1.67 TB/s < 1.75 TB/s 峰值，不属缓存塌陷档）。
- 收敛判定：P0 与次大档 `llama31-1d-8k-d128-bf16`（38.17%）相差 **16.61pp ≥ 3pp**
  → **未收敛**，如实记录。
- 同 run 内 `torch-ref`（纯 PyTorch 参考实现）对照：2d 档 **6.72×**、1d-8k 档 **56.87×**、
  1d-2k 档 **90.88×**。该倍数只对纯 PyTorch 参考实现成立，不是对调优算子的加速，已在正文注明。
- kernel 生效版本已核对：`rope_llama31.py:93-94` 导入
  `rope_llama31_kernel/perf_opt/_make_rope_neox_1d.py` / `_2d.py`（基线版导入被注释），
  故 `perf_records.jsonl` 的 `artifact_sha256` 记录的是 perf_opt 版本。

### 5. 依赖与流程 — pass（附一条流程注意）

- 本分支 stacked 在 `op/position_encoding/rope-neox-lvpanhao`（PR #55）之上，与
  `RopeLongRopeOp`（PR #61）是**兄弟**关系，互不依赖。
- 三者都会改同一批族文件（manifest / workloads / 两个 `__init__.py`），因此#55 → #61 →
  本 PR 或 #55 → 本 PR → #61 的合并顺序需要在审核时串行处理，后合并者需处理族文件的冲突。
- 进度只改 `team/progress/吕盼豪.yaml` 中 `RopeLlama31Op` 一条。

## 建议（不阻塞）

1. P0 与次大档差 16.61pp 未收敛，`llama31-1d-2k-d128-fp16` 仅 15.06%（小 shape 固定开销主导）。
   若后续要提 Ratio，优先看 2d 大 shape 档。
2. 三个 Rope 变体共用同一套 neox kernel，族文件的串行合并顺序建议在团队内约定成文，
   减少后续同类冲突。

## 阻塞项

无。
