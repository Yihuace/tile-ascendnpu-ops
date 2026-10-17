# 优化报告：AddFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `AddFwdOp`；单 kernel / 单 func / 纯 Vector（AIV），`y = a + alpha * b` |
| kernel | `tileops/kernels/elementwise/add/add_kernel/` |
| 调优产物 | `.../add_kernel/perf_opt/AddFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`） |
| 口径 | `msprof op --kernel-name=main` 的 `Task Duration(us)` + `visualize_data.bin` 的 GM Read+Write SOL ratio |
| 工具链 | tilelang `0.1.2` dev root（AscendNPU-IR `31f6903`）/ **CANN 9.0.0**（bishengir 1.1.0 + hivmc 0.2.0）/ Ascend910B3（20 AIC / 40 AIV）/ 2026-10-17 |
| benchmark | `benchmarks/ops/bench_add.py::test_add_manifest_bench`（11 tune + 1 smoke） |
| 过程日志 | `.../add_kernel/perf_opt/opt_log.md`、`perf_records.jsonl`、`logs/` |

## 1. 结论

- **11 个 tune case 中 9 个 ≥70% SOL**；明确的时延收益两处：
  `hidden-state-prefill bf16 −3.85%`（25.74→24.75 µs，SOL 73.9→78.0）、
  `llama-7b-ffn fp32 −1.86%`（63.34→62.16 µs）。
- **`linear-bias` fp16/bf16 未达标**：官方单次 final 为 66.9% / 68.7%（14.57 / 13.96 µs，
  比同批 baseline 慢 3.5%）。但该差异 **< ±5% 单进程噪声底**，且 R2/R3/A-B 全部样本的池化中位数显示
  调优版反而快 **2.4–2.9%**（≈70.0/70.5%）——**结论：该 case 在噪声内，70% 未被稳健达成**，
  既不宣称达标，也不宣称回退。
- 采纳结构：**满块静态 extent + 单尾块 epilogue + `T.Pipelined(num_stages=2)` + grid-stride 40 核**。
  该结构在 Add 上整体中性偏正（11 case 时延和 −0.32%），在大 shape 上收益最大。

## 2. 所做改动

仅在 `perf_opt/` 内（基准 kernel 与 wrapper 未动）：

1. **静态满块 + 尾块分离**：`num_full = N // block_size`，热循环不再逐 tile 计算 `T.min` / 动态 copy extent，
   只保留一个 `tail` epilogue（core 0 处理，≤block_size 元素）。
2. **`T.Pipelined(num_local_tasks, num_stages=2)`**：下一 tile 的 MTE2 装载与当前 tile 的 VEC 链重叠。
3. `TUNED_BLOCK_SIZE = 8192` 作为模块级常量暴露，供 Stage-5 wrapper 切换块成对引用。
4. 保留 fp32 中转与 `alpha==1` 折叠语义不变（精度契约未动）。

**否决项**（源码保留在 `perf_opt/candidates/`，可复跑）：
- `block_size` 12288/16384：40 核上 tile 数 342/256 → 9/8 或 7/6 分配，关键路径 +7.5~9.4%，实测 +5~7%；
- 输出缓冲别名 `y_ub = a_ub`：prim_func 内 if/else 不折叠 → `Undefined variable: y_ub`（TRAP-tvm-parser-rules）；
- 每核等元素连续分块：破坏 grid-stride 的 HBM 聚集窗口，−3%；
- `T.Pipelined` @12288：UB 超限，采集失败。

## 3. 优化结果（msprof，baseline/final 相邻交错，各 1 次独立进程）

| label | dtype | base µs | opt µs | base SOL% | opt SOL% | 判定 |
|---|---|---:|---:|---:|---:|:--:|
| hidden-state-prefill | float32 | 47.08 | 46.94 | 81.5 | 82.5 | ✅ |
| hidden-state-prefill | bfloat16 | 25.74 | **24.75** | 73.9 | **78.0** | ✅ |
| cnn-feat-broadcast | float32 | 71.37 | 71.28 | 82.4 | 84.3 | ✅ |
| cnn-feat-broadcast | float16 | 37.22 | 36.90 | 77.8 | 80.0 | ✅ |
| cnn-feat-broadcast | bfloat16 | 37.37 | 37.46 | 78.7 | 78.3 | ✅ |
| llama-7b-ffn | float32 | 63.34 | **62.16** | 81.1 | **82.7** | ✅ |
| llama-7b-ffn | float16 | 32.71 | 33.01 | 80.2 | 78.4 | ✅ |
| llama-7b-ffn | bfloat16 | 32.64 | 32.86 | 79.5 | 78.3 | ✅ |
| linear-bias | float32 | 24.28 | 24.12 | 78.8 | 80.1 | ✅ |
| **linear-bias** | **float16** | 14.07 | 14.57 | 69.2 | **66.9** | ❌ 噪声内 |
| **linear-bias** | **bfloat16** | 13.47 | 13.96 | 70.3 | **68.7** | ❌ 噪声内 |

smoke：`hidden-state-prefill fp16`（N=8,388,608）精度 PASS（max_diff=0），不参与性能判定。

### 精度

- `python perf_opt/AddFwdOp.py --level all` → **L0 / L1 / L2 / Boundary 全 PASS**（`All check passed!`）；
- `python perf_opt/validate_final.py AddFwdOp.py --block-size 8192` → smoke + linear-bias 两 dtype PASS，max_diff=0；
- 注：首次 `--level all` 的 18 个 L1 失败是「运行期间编辑源文件」导致 JIT 源映射失效的假失败，非 kernel 缺陷
  （失败日志 `logs/final_level_all.log` vs 稳定文件全绿 `logs/final_level_all_v2.log`）。

## 4. 时间与 token

| 项 | 值 |
|---|---|
| 开始 / 结束（+08:00） | 2026-10-17 16:03 → 17:50 |
| 墙钟时长 | **≈ 107 min**（含 3 轮实验、A/B 裁决、全量复测） |
| token input/output/reasoning | 195,429 / 59,649 / 88,000 |
| cache-read / cost | 41,289,216 / **$0.2418** |

（来源：`.start_utc`；`opencode.db` 同上会话窗口求和。）

## 5. 产物

- `.../add_kernel/perf_opt/AddFwdOp.py`（final，sha256 见 `perf_records.jsonl` final 行）
- `.../add_kernel/perf_opt/opt_log.md`（全过程 + 每轮「候选 vs current best」表 + Skill Retrospective）
- `.../add_kernel/perf_opt/perf_records.jsonl`（11 baseline / 40 candidate / 11 final）
- `.../add_kernel/Optimize.md`（一页摘要）
- 基线报告（迁移期）：`reports/tileops/20261013_162043_457669_AddFwdOp/`
- gate 4：`pass`（0 failures / 0 warnings）；state `phase=DONE`
- 提交：`5a4367e perf(add): Stage-4 tuned drop-in + full 11-case retest`
