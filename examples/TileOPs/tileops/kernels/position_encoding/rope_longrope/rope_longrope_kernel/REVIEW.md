# RopeLongRopeOp 迁移检视报告

## 检视结论

结论: **通过**

- 精度: **通过**（11/11，`correctness_passed=true`）
- 性能: **通过**（benchmark 6/6 用例通过，3 个非 smoke msprof 用例全部 `profiler_valid=true`，
  `profiler_fallback_count=0`、`invalid_profile_count=0`，无告警）
- 阻塞项: 0

## 检视元信息

- 设计文档: `rope_longrope_kernel/DESIGN.md`（与 `_make_rope_neox_1d_DESIGN.md` 同内容）
  以及 `_make_rope_neox_1d_DESIGN.md` / `_make_rope_neox_2d_DESIGN.md`
- 任务类型: 迁移（GPU TileOPs → NPU npuir），六区新建
- 源算子: GPU 仓 `tileops/ops/rope.py` 中 `RopeLongRopeOp`（LongRoPE 变体，
  kernel 侧复用 `_make_rope_neox_1d/_2d`，无独立 `_make_rope_longrope_*`）
- 检视维度: 5（语义一致性 / 六区与注册 / 精度覆盖 / 性能口径 / 依赖与流程）
- 阻塞级问题数: 0
- 建议级问题数: 2（见末尾）

## 检视详情

### 1. 语义一致性 — pass

- LongRoPE 与 RopeNeox 共用 neox 旋转（`y = x*cos + rotate_half(x)*sin`），
  差异只在频率表：`divisor = rescale_factors[k] * base^(2k/d)`，且 `cos/sin` 乘上
  `scaling_factor = sqrt(1 + log(scale)/log(orig_max_pos))`（`scale > 1` 时）。
- 该差异由 **Op 层**（`_longrope_freqs`）实现，kernel 只做旋转 —— 与源算子结构一致，
  移植未改变算法归属。
- 测试侧用独立实现的 `_compute_longrope_freqs`（对照 TVM `rope_freq_longrope`）生成参考
  cos/sin，不复用 Op 内部实现，避免「自己验证自己」。

### 2. 六区与注册 — pass

- Op: `tileops/ops/position_encoding/rope_longrope.py`
- Kernel: `tileops/kernels/position_encoding/rope_longrope/`（`rope_longrope.py` 为包装，
  实际分派 `perf_opt/_make_rope_neox_1d.py` / `_make_rope_neox_2d.py`，见第 4 节）
- Manifest: `position_encoding.yaml` 新增 `RopeLongRopeOp`（`variant_of: RopeNeoxOp`）
- Workload: `workloads/position_encoding.py` 新增 `RopeLongRopeOpWorkload`
- 测试: `tests/ops/test_rope_longrope.py`
- Benchmark: `benchmarks/ops/bench_rope_longrope.py`
- 注册已核对：`kernels/position_encoding/__init__.py`、`ops/position_encoding/__init__.py`
  均导出新符号；`load_manifest()` 实跑返回 `['RopeNeoxOp', 'RopeLongRopeOp']`。

### 3. 精度覆盖 — pass

11 个用例覆盖 `1d`/`2d` 两种 layout、fp16/bf16/fp32 三种 dtype、以及 `seq_len=1`、
`head_dim=16` 等边界；容差按 dtype 分层（fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2），
未放宽。实测最大误差见 `run.json`。

### 4. 性能口径 — pass

- 报告 `status=passed`，Ratio 区间 15.29%–54.74%。
- 按 P0 裁决规则（剔 smoke、剔 cache-resident、取元素数最大档）：
  裁决档 **`longrope-2d-b1-s8k-h32-d128-fp16`**（`[1, 8192, 32, 128]`，33.5M 元素）
  → **54.74%**（等效带宽 ≈1.67 TB/s < 1.75 TB/s 峰值，不属缓存塌陷档）。
- 收敛判定：P0 与次大档 `longrope-1d-8k-d128-bf16`（38.04%）相差 **16.70pp ≥ 3pp**
  → **未收敛**，如实记录。
- 本轮 benchmark 同时采到了 `torch-ref`（纯 PyTorch 参考实现）时延，同一次运行内可比：
  2d 档 **6.37×**、1d-8k 档 **58.44×**、1d-2k 档 **97.40×**。这是与**纯 PyTorch 参考实现**
  的比值，不是与调优过的 torch 算子比，正文中已注明口径。
- kernel 实际生效版本已核对：`rope_longrope.py:90-91` 导入的是
  `rope_longrope_kernel/perf_opt/_make_rope_neox_1d.py` / `_2d.py`（基线版导入被注释），
  故 `perf_records.jsonl` 的 `artifact_sha256` 记录的是 perf_opt 版本。

### 5. 依赖与流程 — pass（附一条流程注意）

- 本分支 stacked 在 `op/position_encoding/rope-neox-lvpanhao`（PR #55）之上：`position_encoding`
  族的骨架（manifest 文件、workloads 文件、两个 `__init__.py`）由 #55 引入。
  若直接从 `integration/npu-ops` 拉分支，本 PR 的 diff 会混入 #55 的全部 RopeNeox 改动，
  违反「一个 PR 只包含一个算子」，故采用 stacked 并在 PR 描述中显式声明依赖。
- 本分支未修改其他人的进度文件，进度只改 `team/progress/吕盼豪.yaml` 中 `RopeLongRopeOp` 一条。

## 建议（不阻塞）

1. P0 与次大档差 16.70pp 未收敛，且 `longrope-1d-2k-d128-fp16` 仅 15.29% —— 小 shape 下
   固定开销占比高（3.92us 量级），若后续要提 Ratio 应优先看 2d 大 shape 档。
2. `torch-ref` 基线是纯 PyTorch 逐元素参考，比值（6×–97×）会被参考实现的低效放大，
   对外引用时建议与业界 RoPE 实现对比，而不是引用这里的倍数。

## 阻塞项

无。
