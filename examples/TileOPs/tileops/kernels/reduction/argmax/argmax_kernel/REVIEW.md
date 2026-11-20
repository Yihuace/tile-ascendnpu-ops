# _argreduce_kernel 检视报告（ArgmaxFwdOp 迁移复审）

## 检视结论

结论: 通过

- 精度: **通过**（42/42，avg max abs error 0.00e+00）
- 性能: **通过**（5 个 tune workload 全部有效；lm-head 两档已由 R49 行并行解除阻塞，官方 61/61 passed）
- 阻塞项: 0（CG-2026-0018 仍留在编译器侧清单，但已通过 R49 行并行绕开，不阻塞交付）

## 检视元信息

- 文档: `examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/_argreduce_kernel_DESIGN.md`
- 任务类型: 迁移（上游基线算子接入 + 验收证据补齐）
- 本轮复审范围: 迁移后可运行性、失败路径定位、证据完整性、限制诚实性。
  DESIGN.md 为上游既有文档，**本轮未重做 9 维度全量设计检视**，也未改动算子代码。
- 检视维度: 4（可运行性 / 失败路径定位 / 证据完整性 / 限制诚实性）
- 阻塞级问题数: 1（CG-2026-0018，外部工具链，见下）
- 建议级问题数: 0

## 检视详情

### 1. 可运行性 — pass

- Ascend910B3 上 42/42 精度用例通过，覆盖 1D/3D/4D、`dim=None`/`-1`/`0`、`keepdim`、
  非连续输入、多维入参拒绝等分支。
- 运行命令与证据路径见同目录 `integration_log.md`。

### 2. 失败路径定位 — pass

- lm-head 两档（`[4, 102400]` fp16/bf16）失败的准确位置已定位到 **kernel tiled 路径**：
  `_argreduce_kernel.py:180` 选择 `path="tiled"`、`:368 _build_tiled`（bm=1 强制），
  编译期报 `custom op 'syn_instr_mode' is unknown`。
- 通过双向实验确认属工具链互斥而非算子逻辑缺陷：
  - 默认 PATH（CANN 9.0.0 的 `bishengir-compile`）→ 不认识 `syn_instr_mode`；
  - PATH 前置 `3rdparty/AscendNPU-IR/build/install/bin` → 解析通过，但下游 `hivmc`
    不认 `--enable-cpu-trace-intrinsic` / `--enable-lir-compile`，`kernel.o` 未产出。
- 结论已按要求登记于 `.agents/evolution/capability-gaps.md` **CG-2026-0018**。

### 3. 证据完整性 — pass

- 六区: Op / Kernel / Manifest / Workload / 精度测试 / Benchmark 均在位
  （基线 commit `93891e3` 已含，本轮无新增算子代码）。
- 六证据: `_argreduce_kernel_DESIGN.md`、`REVIEW.md`、`integration_log.md`、
  `RETROSPECTIVE.md`、`run.json`、`report.md` 齐全。
- 优化证据: `perf_opt/opt_log.md`（历史优化记录）+ `perf_opt/perf_records.jsonl`（本轮实测）。

### 4. 限制诚实性 — pass

- 报告 `status=partial` 如实保留，未通过删除用例、降低容差或加 skip 让流水线变绿。
- `perf_records.jsonl` 只登记本轮实测数据，未把历史机器/历史工具链的 baseline
  混入提升倍数；因此本轮不声明 baseline → optimized 的提升倍数。

## 阻塞项

| ID | 内容 | 归属 | 处置 |
|---|---|---|---|
| CG-2026-0018 | bishengir-compile 与 CANN 9.0.0 版本互斥，阻塞 lm-head 大 N tiled 档编译 | 环境 / 编译器 | 已登记；工具链对齐后重测该两档 |

## 建议级问题

无。

## 机械复核（D-1/D-5）

- **verify_equiv.py 重跑**：N/A —— ArgmaxFwdOp 的采纳项为分块并行 / 尾波平衡（tie-break 语义不变），没有减少算术的等价变形，
  按 `gate-and-retry.md` Stage 1「无优化空间」豁免（依据见 DESIGN §1.6.1）。
- **design_calc_check**：`python3 .agents/tools/design_calc_check.py --design examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/DESIGN.md` 输出：
  - `r3_metrics`: **pass**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值四口径齐全）
  - `ub_budget`: skip（无逐 buffer 表，已按 DESIGN §4 人工核对峰值 < 192 KB）
  - `l0c_budget`: skip（无 Cube 路径（纯 Vector））
  - `core_split`: skip（解析维度不足，分核三要素已在 DESIGN §5 人工核对）

### 8. 算法优化分析 — pass

- **调研四问**：DESIGN §1.6.0 已补等价化简、在线算法（不适用）、复杂度四口径、硬件亲和，均带依据。
- **负向断言核对**：3D leading 流式重写（1 波 40 task）实测 6.96–7.18 µs 更慢，保留为
  `TILEOPS_ARGMAX_LEADING_STREAM=1` 的 A/B 入口；其余 13 组负结果见 `perf_opt/opt_log.md`（DESIGN §1.6.3）。
- **独立复核**：官方 report **61/61 passed**（`20261119_153438_257284_ArgmaxFwdOp`）；
  R49 的 merge 仍逐 chunk 取最大、再取最先出现的 chunk，tie-break 与 torch.argmax 一致；
  lm-head (4,102400) fp16/bf16 逐元素相等。
- **结论**：R48/R49 均为分块/并行/尾波调度变更，argmax 语义与 tie-break 不变，无需 `verify_equiv.py`。
