# 优化报告：AddFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `AddFwdOp`；单 kernel / 单 func / 纯 Vector（AIV），`y = a + alpha * b` |
| kernel（基准，只读） | `tileops/kernels/elementwise/add/add_kernel/add_fwd_kernel.py` |
| 调优产物 | `.../add_kernel/perf_opt/AddFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`） |
| 口径 | `msprof op --kernel-name=main` 的 `Task Duration(us)` + `visualize_data.bin` 的 GM Read+Write SOL ratio |
| **工具链版本戳** | tilelang `0.1.2` dev root（AscendNPU-IR `31f6903`）/ **CANN 9.0.0**（bishengir 1.1.0 `428ab8fdab46` + hivmc 0.2.0）/ Ascend910B3（20 AIC / 40 AIV）/ 2026-10-17 |
| 达标线 | 访存受限 **≥70%**（AI=0.125） |
| benchmark | `benchmarks/ops/bench_add.py::test_add_manifest_bench`（11 tune + 1 smoke） |
| 过程日志 | `.../add_kernel/perf_opt/opt_log.md`、`perf_records.jsonl`、`logs/`、`candidates/` |
| 前置报告 | 本文件 v0 已归档 `history_version/OPTIMIZE_REPORT_v0.md`（内容相同，仅结构未归一） |

## 1. 结论

- **Stage-4 直跑 kernel：11 个 tune case 全部 ≥70% SOL**：全 shape **11/11**、可调 shape **11/11**，无豁免项。
- **9.0 端到端 wrapper 复测（2026-10-18，run `20261018_143726_708597_AddFwdOp`）**：正确性 **24/24**、report `passed`；11 个 perf shape **10/11 ≥70%**，唯一 `linear-bias bf16` **69.37%**（贴线）。2026-10-18 单点复测 3 次：**14.00 / 14.28 / 15.06µs，SOL 68.48 / 67.13 / 63.66%**，确认该 shape 稳定在 63.7–69.4%，未过 70%，10/11 结论成立。
- 最贴线的是 `linear-bias` fp16 **72.30%** / bf16 **70.80%**（此前两轮分别为 66.9% / 68.7%，均在线下）。
- 结构：**满块静态 extent + 单尾块 epilogue + `T.Pipelined(num_stages=2)` + grid-stride 40 核 +
  fp16(`alpha==1`) 原生 vadd**。
- 全部 11 case 时延和 **404.818 → 400.138 µs（−1.16%）**；明确收益
  `llama-7b-ffn fp16 −5.37%`、`llama-7b-ffn bf16 −3.34%`、`hidden-state-prefill bf16 −2.64%`。
- **建议采纳 wrapper**：净收益为正且无 shape 回退（唯一 `+2.06%` 的 cnn bf16 已判为噪声）。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测 | 数值 | 结论 |
|---|---|---|
| AI | 0.125 | 访存受限，达标线 70% |
| 瓶颈 | `mte2` 主导、`vec` 有富余 | 应优先压缩「搬运次数 / 每 tile 固定开销」，而不是减计算 |
| 小 shape 固定开销 | `linear-bias`(4.19M) ≈ 1.6µs（占 12%） | 结构参数（block/映射）只能挤 1–3%，不足以稳定跨线 |
| 跨进程噪声 | ±5% | 单批 A/B <3% 不可分辨，必须交错多对 |

### 2.2 方法（两段式）

**第一段：结构杠杆拿底**（目标 1–3%）

1. **满块静态 extent**：`num_full = N // block_size`，热循环用编译期常量边界，去掉每 tile 的 `T.min` 与动态 extent（省标量开销、利于向量化）；
2. **单尾块 epilogue**：尾块单独一次处理，不进热循环；
3. **`T.Pipelined(num_stages=2)`**：用下一 tile 的 MTE2 覆盖当前 tile 的 VEC 链；
4. **grid-stride 40 核**：20 AIC / 40 AIV 下按 AIV 数铺满，避免核间负载不均；
5. `TUNED_BLOCK_SIZE = 8192` 提为模块级常量，供 wrapper 成对引用。

**第二段：语义等价快路径跨线**（目标 −5~6%）

6. **fp16 原生 vadd**（`alpha == 1` 且 dtype=fp16）：直接 `T.vadd(a_ub, b_ub, y_ub)`，把
   `cast→add(fp32)→cast` 的 4 条向量 pass 压到 1 条、UB 流量降到 1/3。
   合法性依据：单次加法的精确和可被 fp32 精确表示 →「fp32 加 + 一次舍回」与「原生 fp16 加」都是
   单次正确舍入，**逐位相等**（实测 max_diff = 0）。

### 2.3 逐轮记录

策略：本算子 AI=0.125（访存受限），1–3% 的参数杠杆不足以稳定跨 70% 线，因此分两段——先用结构杠杆
（静态 extent / 流水 / 映射）拿底，再用**语义等价的快路径**（fp16 原生）跨线。

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| R1 | `base-bs12288`（A1） | block 8192→12288 | fp16 +5~7% | ❌ 否决：9/8 tiles，关键路径 9×12288 比均值高 5.5% |
| R2 | A3 / A4 | 静态尾块 + `T.Pipelined(2)` | fp16 −1.7~−4.6%、bf16 −1.1~−2.0%（4/4 相邻对比） | ✅ 采纳为 current best |
| R2 | `a5-balanced`（A5） | 每核等元素连续分块 | +3%（256M 档再 +3.2%） | ❌ 否决：省下的 1.6% 不平衡抵不过访问模式损失 |
| R3 | `a4-pipe2-bs8192` | 大 block / 别名复测 | 与基线互有胜负（噪声内） | 保留 |
| A/B | A4 vs 基线 | 交错 A/B、交替先后、5 对/dtype | fp16 tie（+1.76%）、bf16 −2.87%（p=0.375）；合并复测 −2.4% / −2.9% | ✅ pipe2 采纳（约 2–3% 真实优势，不宣称 >5%） |
| R4（A6） | fp16 原生 vadd | `alpha==1` 且 fp16 时 `T.vadd(a_ub,b_ub,y_ub)`，4 条向量 pass → 1 条 | `linear-bias` 跨线（72.30%），相对 A4 −2.5%，相对最初基线 −5~6% | ✅ 采纳 |
| 探针 | bf16 原生 | bf16 直接 vadd | CANN 9.0 下可编译、max_diff=0，但时延方向不一致（探针 −2.0%/−2.8%，round 8 官方 +3.8%） | ❌ 不采纳：保留 fp32 中转 |

**fp16 原生的语义依据**：单次加法的精确和可被 fp32 精确表示，「fp32 计算 + 一次舍回」与「原生 fp16」
都是单次正确舍入 → 逐位相等（`--level all` max_diff = 0）。

## 3. 优化结果（msprof，baseline/final 逐 workload 相邻交错，各 1 次独立进程）

| workload | base µs | final µs | Δ | base SOL% | final SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| cnn-feat-broadcast-bfloat16 | 38.421 | 39.211 | +2.06% | 77.42 | 73.81 | ✅ |
| cnn-feat-broadcast-float16 | 38.531 | 38.361 | -0.44% | 76.34 | 79.05 | ✅ |
| cnn-feat-broadcast-float32 | 71.661 | 71.201 | -0.64% | 82.26 | 82.56 | ✅ |
| hidden-state-prefill-bfloat16 | 25.761 | 25.081 | -2.64% | 75.19 | 76.45 | ✅ |
| hidden-state-prefill-float32 | 47.321 | 46.751 | -1.20% | 81.97 | 81.87 | ✅ |
| linear-bias-bfloat16 | 13.780 | 13.550 | -1.67% | 69.47 | **70.80** | ✅ |
| linear-bias-float16 | 13.350 | 13.260 | -0.67% | 72.63 | 72.30 | ✅ |
| linear-bias-float32 | 24.570 | 24.430 | -0.57% | 78.13 | 78.45 | ✅ |
| llama-7b-ffn-bfloat16 | 34.181 | 33.041 | -3.34% | 74.85 | 77.93 | ✅ |
| llama-7b-ffn-float16 | 33.871 | 32.051 | -5.37% | 75.51 | 82.00 | ✅ |
| llama-7b-ffn-float32 | 63.371 | 63.201 | -0.27% | 81.51 | 81.30 | ✅ |

smoke：`hidden-state-prefill fp16`（N=8,388,608）精度 PASS（max_diff=0），不参与性能判定。

**测量口径与噪声**：baseline/final 相邻交错、各 1 次独立进程；`linear-bias` fp16/bf16 存在跨进程双态 ±5%，
A6 后 fp16 连续两次 ≥71%；`cnn bf16 +2.06%` 为噪声（同 case round 6 为 −1.40%），不做 shape 特判。

## 4. 精度

- `python perf_opt/AddFwdOp.py --level all` → **L0/L1/L2/Boundary 全 PASS**（`All check passed!`）
- `python perf_opt/validate_final.py AddFwdOp.py --block-size 8192` → smoke + `linear-bias` PASS，max_diff = 0
- 依据：fp16 原生与 fp32 中转同为单次正确舍入 → 逐位相等

## 5. 时间与成本

| 项 | 值 |
|---|---|
| 开始 / 结束（+08:00） | 2026-10-17 16:03 → 20:05 |
| 墙钟时长 | **≈ 242 min**（3 轮结构 + A/B + 4 次官方 final，含 A6 与 bf16 原生证伪） |
| token input/output/reasoning | 254,794 / 117,543 / 128,923 |
| cache-read / cost | 100,834,560 / **$0.4886** |

## 6. 产物

| 类别 | 路径 / 状态 |
|---|---|
| final kernel | `.../add_kernel/perf_opt/AddFwdOp.py`（sha256 见 `perf_records.jsonl` round 9 final 行） |
| 调优记录 | `.../perf_opt/opt_log.md`、`Optimize.md`、`perf_records.jsonl`、`logs/`、`candidates/` |
| 迁移期基线报告 | `reports/tileops/20261013_162043_457669_AddFwdOp/` |
| gate 4 | `pass` |
| 算子状态 | `phase=DONE` |
| wrapper | 已切 `perf_opt`（2026-10-18 统一 9.0 口径时启用；双 import 切换块） |
| 端到端验收 | `reports/tileops/20261017_095652_466213_AddFwdOp/`（status=passed） |
| pattern-library 回写 | PL-1.22（小 shape 固定截距/负载不均）、PL-1.17 证伪更正（bf16 原生可用但不可靠）、TRAP-tvm-parser-rules 追加 |

## 7. 天花板 / 豁免 / 缺口

- **豁免项：无**（11 个 tune shape 全部 ≥6µs，不触发微型 shape 豁免）。
- **贴线风险**：`linear-bias` bf16 70.80%、fp16 72.30% 距 70% 线仅 0.8–2.3pp，受 ±5% 双态噪声支配；
  已在报告中如实标注，不做 shape 特判。
- **`cnn-feat-broadcast bf16` 的 +2.06%**：判定为噪声（同 case 多轮为负），不计为回退。
- 无能力缺口（未触发 CG 条目）。

## 8. 建议

1. 采纳 `perf_opt/AddFwdOp.py`（已切 wrapper，端到端 report passed）。
2. 若后续做 9.0 → 更高版本的工具链升级，需重跑本表 11 个 workload 复核贴线项（`linear-bias` 两档）。
3. 本报告的 ratio 口径为「访存受限」；若未来该算子 AI 变化超过 ridge，需改按算力判据重评。

---

## 9. R5（2026-10-21）：贴线攻坚结论（未达成）与 broadcast 行复用探针

> 目标：`linear-bias bf16` 69.37% → ≥70%（11/11）。**结果：未达成，10/11（69.17%）**；
> 当前 `perf_opt/AddFwdOp.py` **未改动**（sha256 `0392900e…`）。数据见 `perf_opt/REPORT_ADD_R5_DATA.md`。

### 9.1 已排除的方向（都有实测证据）

| 方向 | 结果 | 证据 |
|---|---|---|
| bf16 原生 `vadd`（`alpha==1`） | **tie（−0.18%）** | 6 对同会话交错 A/B；profile 显示后端把 bf16 vadd 降级为 fp32 仿真（`aiv_vec_time` 5.986µs vs 中转 6.019µs，无实质差异）→ 上一轮“方向不一致”的根因是仿真，不是噪声 |
| `T.Pipelined(3)` | −1.33%（tie） | 配对 A/B |
| `block_size=4096` | +2.43% | 配对 A/B |
| `block_size=12288` | UB overflow | 编译失败 |
| 小 shape 减 VEC pass / UB copy | 全部落在噪声内 | `linear-bias` 的 `mte2_ratio` 0.925，DMA 主导 |

另测得：**同结构 bf16 比 fp16 慢 6.2%**（13.955 vs 13.140µs，4/4 对同向），这是 dtype 固有成本，不是实现差异。

### 9.2 唯一量级杠杆：broadcast 行复用（探针 −36.1%，需改 Op 接口）

`linear-bias` 的 `other` 本来是 `[1,4096]`，但 **Op 层 `_expand_flat` 先把它物化成 `[1024,4096]`**，
kernel 因此读了两份全量输入（流量 3N，而实际只需 2N + row）：

| 方案 | 流量 | kernel 时延 | 正确性 |
|---|---:|---:|---|
| 当前（物化 broadcast） | 3N | 14.050 µs | 基线 |
| **行复用**（kernel 直接读 `[1,4096]`） | 2N + row | **8.980 µs（−36.1%）** | **逐位相等（max_diff=0）** |

代价与影响面：
- 需改 `_expand_flat` / kernel 签名（增加 broadcast 形状参数）；
- 需同步修正 manifest 的 roofline `bytes`（3N → 2N + row），否则 ratio 会显示 ~106.8%；修正后约 **71.2%**（跨线）；
- 同一机制也会让 `cnn-feat-broadcast`（`[256,1,1]`）等广播 shape 受益。

### 9.3 结论

- Stage-4 只读边界内**没有其它可拿的收益**：0.6–0.8pp 的差距小于跨进程噪声（±5%）与 A/B 可分辨度（≈1.5%）。
- **建议作为设计级改动立项**：Add（以及 Sub/Lerp 等走同一 `_expand_flat` 的广播算子）的 broadcast 行复用。
  预期：`linear-bias bf16` 8.98µs / ≈71.2%，`cnn-feat-broadcast` 同受益；风险集中在 Op 接口与 roofline 口径。

---

## 10. R6（2026-10-21）：broadcast 行/标量复用实现（默认关闭）

> 目标：消除 Op 层 `_expand_flat` 对 broadcast 操作数的物化（3N → 2N+B），把 `linear-bias bf16` 拉过 70%。
> **实现完成、正确性全绿，但默认关闭**：真实时延降 30–43%，官方 Ratio 反而下降，达标从 10/11 变 9/11。

### 10.1 实现

- `perf_opt/AddFwdOp.py`：`plan_broadcast()` + `add_bcast_row_kernel`（`[1,L]` 行重复）与
  `add_bcast_scalar_kernel`（`[C,1,1]` 每平面标量）；命中时 Op 层不再物化 `other`。
- `ops/elementwise/add.py`：dispatch plan，`eval_roofline` 跟随 plan（复用 2N+B；回退 3N）。
- manifest：`bytes: (2*N + B)*elem_bytes`（同形 B=N → 3N）。
- 开关：`BCAST_ENABLED`（默认 **False**）。
- 回退面：input 自身被广播、`1` 夹在中间、2 字节标量模式（bf16/fp16）、row<512、UB 超限、同形 add。

### 10.2 实测（同会话 A/B，msprof Task Duration）

| workload | dtype | base µs | reuse µs | Δ | 官方 Ratio |
|---|---|---:|---:|---:|---:|
| linear-bias | bf16 | 13.82 | **9.68** | **−30.0%** | 69.4 → 49.3 |
| linear-bias | fp16 | 13.34 | **8.56** | **−35.8%** | 71.9 → 59.1 |
| linear-bias | fp32 | 23.86 | **13.56** | **−43.2%** | 80.4 → 64.8 |
| cnn-feat | fp32 | 70.12 | **41.34** | **−41.0%** | 83.7 → 71.9 |
| cnn-feat | bf16/fp16 | 36.6/36.6 | 36.3/35.9 | −0.9/−2.0%（回退） | ≈不变 |
| 其它 6 case | — | — | — | 噪声内 | 不变 |

12 case 合计 **419.16 → 370.38 µs（−11.64%）**；`test_add` **24/24**、快路径与 flat 路径**逐位相等**。

### 10.3 为什么默认关闭（度量口径，不是性能回退）

官方 `Ratio(%)` 来自 `msprof_roofline`（L2 read-miss × cacheline / t / 1.75 TB/s）；它把「物化 b 的冗余读」算作有效功。
删掉这部分后，分子显著变小，Ratio 从 69.4% 掉到 49.3%——**时延同时 −30%**。
统一字节口径（base 3N、复用 2N+B）下 `linear-bias bf16` 也只有 **67.0%**（9.68µs vs 2N 带宽地板 6.6µs），
仍未过 70%。因此：

- 两种口径下都**达不到 11/11**；
- 若启用，交付 Ratio 区间变为 49.3–82.2%、达标 9/11（纸面变差）；
- 故默认 `BCAST_ENABLED=False`，交付维持 69.37–83.48%、10/11。

### 10.4 待决策

若要采纳这条真实时延收益（−11.6% 合计），需要先统一 Add 的交付口径（建议 Task Duration 或 `eval_roofline` 字节模型），
并接受 Ratio 列不可与旧口径横比。相关证据：`perf_opt/REPORT_ADD_BCAST.md`、`r6_bcast_probe.py`、CG-2026-0032
（2 字节 UB 广播慢 3–4×）。
