# AddFwdOp Stage-4 调优摘要（Optimize.md）

> 本页只摘要 `perf_opt/opt_log.md`，完整过程、原始数据与证据链以该文件为准。
> 工具链：CANN 9.0.0（bishengir 1.1.0 / hivmc 0.2.0）· tilelang 0.1.2 dev root · Ascend910B3（40 AIV）· 2026-10-17。

## 结论

- **11 个 tune case 中 9 个稳定 ≥70% SOL**；`linear-bias` fp16/bf16 **未达标（66.85% / 68.67%）**，
  但差异落在 **±5% 单进程噪声**内，池化中位数反而比基线快 2.4–2.9%（≈70.0/70.5%）——
  正确表述是「**在噪声内，70% 未被稳健达成**」。
- 采纳结构：**满块静态 extent + 单尾块 epilogue + `T.Pipelined(num_stages=2)` + grid-stride 40 核**
  （`perf_opt/AddFwdOp.py`，`TUNED_BLOCK_SIZE = 8192`）。
- 明确收益：`hidden-state-prefill bf16` **−3.85%**、`llama-7b-ffn fp32` **−1.86%**；
  全部 11 case 时延和 **−0.32%**（整体小幅正向，仍在噪声内）。

## 最终对账（官方单次 msprof，11 个 tune）

| workload | baseline µs | final µs | Δ | baseline SOL% | final SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| hidden-state-prefill fp32 | 47.081 | 46.941 | −0.30% | 81.49 | 82.54 | ✅ |
| hidden-state-prefill bf16 | 25.741 | **24.750** | **−3.85%** | 73.86 | 78.00 | ✅ |
| cnn-feat-broadcast fp32 | 71.371 | 71.281 | −0.13% | 82.40 | 84.29 | ✅ |
| cnn-feat-broadcast fp16 | 37.221 | 36.901 | −0.86% | 77.79 | 80.00 | ✅ |
| cnn-feat-broadcast bf16 | 37.371 | 37.461 | +0.24% | 78.71 | 78.25 | ✅ |
| llama-7b-ffn fp32 | 63.341 | **62.161** | **−1.86%** | 81.15 | 82.69 | ✅ |
| llama-7b-ffn fp16 | 32.711 | 33.011 | +0.92% | 80.16 | 78.36 | ✅ |
| llama-7b-ffn bf16 | 32.641 | 32.861 | +0.67% | 79.47 | 78.31 | ✅ |
| linear-bias fp32 | 24.280 | 24.120 | −0.66% | 78.84 | 80.09 | ✅ |
| **linear-bias fp16** | 14.070 | 14.570 | +3.55% | 69.17 | **66.85** | ❌ |
| **linear-bias bf16** | 13.470 | 13.960 | +3.64% | 70.28 | **68.67** | ❌ |

smoke：`hidden-state-prefill fp16`（N=8,388,608）精度 PASS（max_diff=0），不做性能调优。

## 有效优化点

| 优化点 | 实测 |
|---|---|
| 满块静态 extent（去掉每个 tile 的 `T.min` 动态尾块） | 相邻对比 4/4 优于基线（fp16 −1.7~−4.6%） |
| `T.Pipelined(num_stages=2)` 重叠 MTE2/VEC | 池化中位 fp16 −2.4% / bf16 −2.9%；大 shape 明确收益（hidden bf16 −3.85%） |

## 否决项（证据保留在 `perf_opt/candidates/`）

| 候选 | 否决原因 |
|---|---|
| block_size 12288 / 16384 | 40 核上 tile 数 342/256 → 9/8 或 7/6 分配，关键路径 +5.5~9.4%，实测 +5~7% |
| 输出缓冲别名 `y_ub = a_ub` | prim_func 内 if/else 不折叠，`Undefined variable: y_ub`（TRAP-tvm-parser-rules） |
| 每核等元素连续分块 | 破坏 grid-stride HBM 聚集窗口，−3% |
| pipe2 @12288 | UB 超限，采集失败 |

## 精度与验收

- `python perf_opt/AddFwdOp.py --level all`：**L0/L1/L2/Boundary 全 PASS**，`All check passed!`
- `python perf_opt/validate_final.py AddFwdOp.py --block-size 8192`：smoke + linear-bias 两 dtype PASS（max_diff=0）
- 注：首次 `--level all` 的 18 个 L1 失败是「运行期间编辑源文件」导致 JIT 源映射失效的假失败，非 kernel 缺陷。

## 遗留与建议

1. `linear-bias` 的 70% 判定落在噪声内：若要补足，需要 **≥7 对**交错 A/B 或降低单进程双态（当前 `ab_test.py` n=5 的符号检验下限 p=0.25，分辨不了 2–3%）。
2. 已排除参数/映射/流水方向；进一步空间只剩「原生 fp16 计算」（会破坏 fp32 opmath golden，精度契约不允许）。
3. Stage 5 切换 wrapper 时，import 应为 `from .add_kernel.perf_opt.AddFwdOp import add_fwd_kernel`，
   默认 block_size 用 `AddFwdOp.TUNED_BLOCK_SIZE`（8192）。

## 证据

- `perf_opt/opt_log.md`（全过程 + 迭代表 + 复盘）
- `perf_opt/perf_records.jsonl`（baseline/candidate/final 全记录，append-only）
- `perf_opt/logs/`（msprof 过程日志、`ab_*.json`、`final_level_all_v2.log`）
- `perf_opt/candidates/`（全部候选源码）
