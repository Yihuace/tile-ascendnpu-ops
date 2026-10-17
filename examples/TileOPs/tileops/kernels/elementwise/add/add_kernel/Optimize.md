# AddFwdOp Stage-4 调优摘要（Optimize.md）

> 本页只摘要 `perf_opt/opt_log.md`，完整过程与证据以该文件为准。
> 工具链：CANN 9.0.0（bishengir 1.1.0 / hivmc 0.2.0）· tilelang 0.1.2 dev root · Ascend910B3（40 AIV）· 2026-10-17。

## 结论

- **11 个 tune case 全部 ≥70% SOL**；`linear-bias` 两 dtype 均过线：
  fp16 **72.30%**、bf16 **70.80%**。
- 采纳：**满块静态 extent + 尾块 epilogue + `T.Pipelined(2)` + grid-stride 40 核
  + fp16(alpha==1) 原生 vadd**（`perf_opt/AddFwdOp.py`，`TUNED_BLOCK_SIZE = 8192`）。
- 全部 11 case 时延和 **404.818 → 400.138 µs（−1.16%）**；
  明确收益 `llama fp16 −5.37%`、`llama bf16 −3.34%`、`hidden bf16 −2.64%`。
- bf16 保持 fp32 中转：原生 bf16 可编译且逐位正确，但性能方向不一致（探针 −2%、官方 +3.8%），不采纳。

## 最终对账（官方 msprof，逐 workload 相邻交错）

| workload | baseline µs | final µs | Δ | base SOL% | final SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| cnn-feat-broadcast-bfloat16 | 38.421 | 39.211 | +2.06% | 77.42 | 73.81 | ✅ |
| cnn-feat-broadcast-float16 | 38.531 | 38.361 | -0.44% | 76.34 | 79.05 | ✅ |
| cnn-feat-broadcast-float32 | 71.661 | 71.201 | -0.64% | 82.26 | 82.56 | ✅ |
| hidden-state-prefill-bfloat16 | 25.761 | 25.081 | -2.64% | 75.19 | 76.45 | ✅ |
| hidden-state-prefill-float32 | 47.321 | 46.751 | -1.20% | 81.97 | 81.87 | ✅ |
| linear-bias-bfloat16 | 13.780 | 13.550 | -1.67% | 69.47 | 70.80 | ✅ |
| linear-bias-float16 | 13.350 | 13.260 | -0.67% | 72.63 | 72.30 | ✅ |
| linear-bias-float32 | 24.570 | 24.430 | -0.57% | 78.13 | 78.45 | ✅ |
| llama-7b-ffn-bfloat16 | 34.181 | 33.041 | -3.34% | 74.85 | 77.93 | ✅ |
| llama-7b-ffn-float16 | 33.871 | 32.051 | -5.37% | 75.51 | 82.00 | ✅ |
| llama-7b-ffn-float32 | 63.371 | 63.201 | -0.27% | 81.51 | 81.30 | ✅ |

smoke：`hidden-state-prefill fp16`（N=8,388,608）精度 PASS（max_diff=0），不参与性能判定。

## 有效优化点

| 优化点 | 实测 |
|---|---|
| 满块静态 extent + 尾块 epilogue | 相邻对比 4/4 优于基线（fp16 −1.7~−4.6%） |
| `T.Pipelined(num_stages=2)` | 大 shape 明确收益（llama fp16 −5.37%） |
| fp16(alpha==1) 原生 vadd | `linear-bias fp16` −5.6%（14.01→13.23µs 单轮），`--level all` max_diff=0 |

## 否决项（源码在 `perf_opt/candidates/`）

| 候选 | 原因 |
|---|---|
| block 12288/16384 | 40 核 tile 不均，关键路径 +5.5~9.4%，实测 +5~7% |
| `y_ub = a_ub` 别名 | prim_func 内 if/else 不折叠 → `Undefined variable`（TRAP-tvm-parser-rules） |
| 每核等元素连续分块 | 破坏 grid-stride 聚集窗口，−3% |
| bf16 原生 vadd | 可编译且 max_diff=0，但时延方向不一致（−2% ~ +3.8%），不采纳 |

## 精度

- `python perf_opt/AddFwdOp.py --level all`：**L0/L1/L2/Boundary 全 PASS**（`All check passed!`）
- `python perf_opt/validate_final.py AddFwdOp.py --block-size 8192`：smoke + linear-bias 两 dtype PASS，max_diff=0

## 遗留

1. `cnn bf16` 单轮 +2.06% 属噪声（round 6 同 case 为 −1.40%），不做特判。
2. bf16 `linear-bias` 70.80% 仍贴线；bf16 原生收益不可靠，已按“噪声内”处理。
3. Stage 5 wrapper 切换：`from .add_kernel.perf_opt.AddFwdOp import add_fwd_kernel`，
   默认 block 用 `AddFwdOp.TUNED_BLOCK_SIZE`。

## 证据

`perf_opt/opt_log.md`、`perf_opt/perf_records.jsonl`、`perf_opt/logs/`、`perf_opt/candidates/`、
`reports/optimize/AddFwdOp/OPTIMIZE_REPORT.md`
