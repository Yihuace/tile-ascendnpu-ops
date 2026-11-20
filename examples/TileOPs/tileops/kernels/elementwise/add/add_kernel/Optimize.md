# AddFwdOp Stage-4 调优摘要（Optimize.md）

> 本页只摘要 `perf_opt/opt_log.md`，完整过程与证据以该文件为准。
> 工具链：CANN 9.0.0（bishengir 1.1.0 / hivmc 0.2.0）· tilelang 0.1.2 dev root · Ascend910B3（40 AIV）·
> 2026-10-17（R1–R4）/ **2026-10-21（R5 复核，仓库搬迁后重采）**。

## 结论（R5 终态）

- **11 个 tune case 中 10 个 ≥70% SOL**；未过线的是 `linear-bias bf16`：**14.050µs / 69.17%**
  （同 shape fp16 为 13.420µs / 73.41%）。
- 采纳（R5 未改动，仍为 R4 winner）：**满块静态 extent + 尾块 epilogue + `T.Pipelined(2)` +
  grid-stride 40 核 + fp16(alpha==1) 原生 vadd**（`perf_opt/AddFwdOp.py`，`TUNED_BLOCK_SIZE = 8192`）。
- 全部 11 case 时延和 **404.818 → 396.310 µs（−2.10%，R5 同版复测）**。
- R5 五个候选（bf16 原生 vadd / fp32 直通 / pipe3 / bs4096 / bs12288）**全部 tie 或更慢**，无 winner。
- `linear-bias bf16` 定性：**结构 + bf16 dtype 成本地板**（MTE2 ratio 0.925；同结构 bf16 比 fp16 慢 6.2%，
  4/4 对一致）。CANN 9.0 的 bf16 `T.vadd` 被后端降级为 fp32 仿真链（`aiv_vec_time` 5.986 vs 中转 6.019µs），
  故「bf16 原生」不存在可复现收益。

## 最终对账（官方 msprof，round 10 final，逐 workload 独立进程）

| workload | baseline µs | final µs | Δ | base SOL% | final SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| cnn-feat-broadcast-bfloat16 | 38.421 | 37.290 | −2.94% | 77.43 | 80.57 | ✅ |
| cnn-feat-broadcast-float16 | 38.531 | 36.050 | −6.44% | 76.34 | 81.15 | ✅ |
| cnn-feat-broadcast-float32 | 71.661 | 71.040 | −0.87% | 82.26 | 84.03 | ✅ |
| hidden-state-prefill-bfloat16 | 25.761 | 25.520 | −0.94% | 75.19 | 75.08 | ✅ |
| hidden-state-prefill-float32 | 47.321 | 46.700 | −1.31% | 81.97 | 81.25 | ✅ |
| linear-bias-bfloat16 | 13.780 | 14.050 | +1.96% | 69.47 | 69.17 | ❌ |
| linear-bias-float16 | 13.350 | 13.420 | +0.52% | 72.63 | 73.41 | ✅ |
| linear-bias-float32 | 24.570 | 24.230 | −1.38% | 78.13 | 79.63 | ✅ |
| llama-7b-ffn-bfloat16 | 34.181 | 32.690 | −4.36% | 74.85 | 79.08 | ✅ |
| llama-7b-ffn-float16 | 33.871 | 32.280 | −4.70% | 75.51 | 80.82 | ✅ |
| llama-7b-ffn-float32 | 63.371 | 63.040 | −0.52% | 81.51 | 81.92 | ✅ |

smoke：`hidden-state-prefill fp16`（N=8,388,608）精度 PASS（max_diff=0），不参与性能判定。

## 有效优化点

| 优化点 | 实测 |
|---|---|
| 满块静态 extent + 尾块 epilogue | 相邻对比 4/4 优于基线（fp16 −1.7~−4.6%） |
| `T.Pipelined(num_stages=2)` | 大 shape 明确收益（llama fp16 −5.37%，A/B 池化 −2.4~−2.9%） |
| fp16(alpha==1) 原生 vadd | `linear-bias fp16` −2.5%（A4→A6），`--level all` max_diff=0 |
| fp32 直通 vadd（R5-B2 验证） | 与 base tie（−0.22%），证明 3 次 UB copy 已被隐藏；未采纳 |

## 否决项（源码在 `perf_opt/candidates/`、`perf_opt/AddFwdOp_round5_*.py`）

| 候选 | 原因 |
|---|---|
| block 12288/16384 | 40 核 tile 不均（关键路径 +5.5~9.4%）；R5 复核 bs12288 直接 UB overflow（245,760B > 196,608B） |
| `y_ub = a_ub` 别名 | prim_func 内 if/else 不折叠 → `Undefined variable`（TRAP-tvm-parser-rules） |
| 每核等元素连续分块 | 破坏 grid-stride 聚集窗口，−3% |
| bf16 原生 vadd | 逐位相等但零收益（VEC 时间 5.986 vs 6.019µs，后端仿真），R5 6 对 A/B tie |
| `num_stages=3`（R5-B4） | 6 对 A/B −1.33% 但方向 3/6，tie；`aiv_mte2_ratio` 未变 |
| `block_size=4096`（R5） | 4 对 A/B +2.43%，候选更慢 |

## 精度

- `python perf_opt/AddFwdOp.py --level all`（R5 复核，2026-10-21）：**L0/L1/L2/Boundary 全 PASS**
  （`All check passed!`，`perf_opt/logs/r5/final_level_all.log`）
- bf16 原生 vs fp32 中转：`linear-bias` 全量 + 3 组对抗样本 **逐位相等（max_diff=0）**
- `python perf_opt/validate_final.py AddFwdOp.py --block-size 8192`：smoke + linear-bias 两 dtype PASS，max_diff=0

## 遗留

> **2026-10-21 R6 跟进**：遗留 2 的「唯一量级路径」已实现并实测（`REPORT_ADD_BCAST.md`）：
> `linear-bias` 三 dtype **−30.0% / −35.8% / −43.2%**，`cnn feat fp32` **−41.0%**；
> 12 case 时延和 **419.16 → 370.38 µs（−11.64%）**，逐 workload 见该报告 §5。
> 注意报表 `Ratio(%)` 由 msprof 的 GM **读**字节反推，删冗余读必然使该列下降（§6），
> 不能再用它判这类优化。

1. **`linear-bias bf16` 69.17% 未达 70%**：与阈值差 0.6–0.8pp，小于跨进程噪声（±5%）与 A/B 可分辨度（≈1.5%），
   在 MTE2 ratio 0.925 的 DMA 受限前提下属于结构 + dtype 成本地板。
2. **唯一量级路径（需主会话决策）**：`linear-bias` 的 `other=[1,4096]` 在 Op 层被物化成 `[1024,4096]`，
   kernel 把同一行读了 1024 次（1/3 冗余流量）。行复用探针实测 **14.050 → 8.980µs（−36.1%，逐位相等）**，
   但需改 Op 层 `_expand_flat`、kernel 签名与 `eval_roofline` 字节口径（3N → 2N+row，否则 ratio 会显示 106.8%）。
   详见 `perf_opt/opt_log.md` §R5 与 `perf_opt/REPORT_ADD_R5_DATA.md` §5.3。
3. `cnn bf16` 单轮 +2.06%（历史）为噪声，不做特判。
4. Stage 5 wrapper 切换：`from .add_kernel.perf_opt.AddFwdOp import add_fwd_kernel`，
   默认 block 用 `AddFwdOp.TUNED_BLOCK_SIZE`。

## 证据

`perf_opt/opt_log.md`、`perf_opt/REPORT_ADD_R5_DATA.md`、`perf_opt/perf_records.jsonl`、
`perf_opt/logs/r5/`、`perf_opt/ab_r5/`、`perf_opt/candidates/`、
`reports/optimize/AddFwdOp/OPTIMIZE_REPORT.md`
