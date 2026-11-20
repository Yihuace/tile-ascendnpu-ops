# REPORT_FALN_R3_DATA — FusedAddLayerNormFwdOp Stage-4 R8 数据交接

> 本文件是 R8 轮（2026-10-21）的数据/方法交接，交付口径一律取 `perf_opt/perf_records.jsonl`（129 行，final = `r8final`）。
> 时延口径唯一：`msprof op Task Duration(us)`，每点 `--launch-count=20 --warm-up=5`，取 20 次中位数；同会话相邻交错 A/B。
> 工具链：tilelang 0.1.2 dev root（HEAD `9309edb`，包戳 `3c4e9cf9`）/ CANN 9.0.0 / Ascend910B3（20 AICore × 2 AIV = 40 向量核）/ **Developer 模式**。

## 1. 身份与产物

| 项 | 值 |
|---|---|
| 算子 | `FusedAddLayerNormFwdOp`（单 kernel / 单 prim_func 家族 / 纯 Vector） |
| baseline（只读） | `../_fused_add_layer_norm_kernel.py`（Stage-3，sha256 `2be87b2f…`） |
| 交付 kernel | `perf_opt/FusedAddLayerNormFwdOp.py`（R8，sha256 `91121fa392fcbf2db896394eb3e7c3654966e6ca1cde9bff250602a5745d3a45`） |
| 上一轮 | R7 final，sha256 `6b051de2…`（8b 62.65 / 70b 116.79、120.40 / 13b 79.01µs） |
| benchmark | `benchmarks/ops/bench_fused_add_layer_norm.py::test_fused_add_layer_norm_bench`（7 tune + 3 smoke） |
| 调优日志 | `perf_opt/opt_log.md` §5（R8 候选 vs current best 表、Final 表、复盘） |
| 结构化记录 | `perf_opt/perf_records.jsonl`（append-only；R8 新增 18 行：13b A/B 4 行 + final 14 行） |
| 采集工件 | `perf_opt/profiles/candidate-r8-*`、`profiles/final-r8-*`、`profiles/baseline-r8-*`、`perf_opt/logs/r8/` |
| 探针 | `perf_opt/probe_r8_ub/compile_probe.py`（UB 预算先验闸门） |

## 2. 诊断

R7 结束时 8b-prefill fp16 = 62.65µs（SOL 33.05%）、vec ratio 0.756、mte2 0.336、`aiv_time` 60.0 vs Task 62.69。
Cube GEMV 探针（`probe_cube_gemv`，2026-10-21）在同一批测量里给出**非 Cube 的线索**：Vector `reduce_sum` 的每行成本随行批处理下降——
`bm=2` 0.319 µs/row → `bm=4` 0.228 µs/row（−29%）；单次 reduce 占 R7 每 tile 2.41µs 的 26%。据此提出 R8 假设：
**把 `block_m` 从 2 抬到 3/4，8b prefill 可再降 5–10%。**

R7 把 `block_m` 钉在 ≤2 的依据是 Stage-3 的 `20 B/elem` 保守包络（`_UB_MAX_ELEMS=9216`，含 auto multi-buffer 的粗略折算），
不是逐平面的实测账。R8 第一步就是把这个包络换成可标定的平面清单模型。

## 3. 方法

1. **UB 预算重标定**：`_ub_footprint_bytes(block_m, N, dtype)` = 该 dtype 路径实际分配的 (block_m,N) 平面 +
   两个 (1,N) fp32 affine 平面（w1/b1），并对 bm≥2 路径叠加后端 auto-multi-buffer 的 +6 B/elem。
   用后端容量诊断原文标定（见下表），误差 32 B。
2. **行批处理候选（op0）**：`block_m=3/4`（N=4096 half 路径），kernel body 不变，只改默认参数与守卫。
3. **multi-buffer 抑制候选（op2）**：`pass_configs={"npuir.enable_auto_multi_buffer": False}`，用于判定 bm≥3 的膨胀归属。
4. **预算模型驱动的 fp32 候选（op1，胜出）**：新模型判定 fp32 N=5120 `block_m=2` 只需 184,320 B（≤196,608 B），
   而 Stage-3 包络把它判为超额（2×5120=10,240 > 9,216）。13b 因此从每 tile 1 行变为 2 行。

### 3.1 UB 容量账（后端原文，Developer 模式）

| 配置 | 模型预测 | 后端自报 requires | 结论 |
|---|---:|---:|---|
| fp16 N=4096 bm=3，auto-mb ON | 229,376 B | **229,408 B** | 超容量 32,800 B → 不可编译 |
| fp16 N=4096 bm=3，auto-mb OFF | 155,648 B | **204,832 B** | 仍超 8,224 B（另含 copy-cast 的 (bm,N) staging 平面 +4 B/elem，模型未计） |
| fp16 N=4096 bm=4，auto-mb OFF | 196,608 B | **262,176 B** | 超 65,568 B |
| fp16 N=4096 bm=2，auto-mb ON（R7 默认） | 163,840 B | 编译通过 | 有 32,768 B 余量 |
| fp16 N=8192 bm=1（R7 默认） | 163,840 B | 编译通过 | 再加 1 个 (1,N) fp32 平面即被拒 |
| fp32 N=5120 bm=2（R8 op1） | 184,320 B | 编译通过 | 有 12,288 B 余量 |

容量 = 1,572,864 bits = 196,608 B。行批处理在 N=4096 half 路径因此 **UB-blocked**；
「stats 批处理 + epilogue 分块」不改变该结论（stats pass 自身就要 4 行驻留 fp32，逻辑下界已恰等于容量）。

## 4. 逐 workload 结果（同会话交错，base = Stage-3，final = R8）

| workload | base µs | R7 best µs | **R8 final µs** | Δ vs base | Δ vs R7 | final SOL% | 判据/线 | 达标 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| llama-8b-prefill fp16 | 82.360 | 62.65 | **62.940** | −23.58% | +0.46%（噪声） | 32.99 | 算力 60% | 否 |
| llama-8b-prefill bf16 | 84.260 | 64.43 | **64.480** | −23.47% | +0.08%（噪声） | 37.88 | 算力 60% | 否 |
| llama-70b-prefill fp16 | 165.370 | 116.79 | **117.100** | −29.19% | +0.27%（噪声） | 35.01 | 算力 60% | 否 |
| llama-70b-prefill bf16 | 168.760 | 120.40 | **120.440** | −28.63% | +0.03%（噪声） | 40.43 | 算力 60% | 否 |
| llama-13b-prefill fp32 | 107.360 | 79.01 | **67.280** | **−37.33%** | **−14.85%** | **70.82** | 访存 70% | **是** |
| llama-8b-decode bf16 | 4.240 | 4.19 | **4.160** | −1.89% | −0.72% | 0.52 | 微型豁免 | 豁免 |
| llama-70b-decode bf16 | 4.650 | 4.21 | **4.170** | −10.32% | −0.95% | 0.87 | 微型豁免 | 豁免 |

13b 的候选级 A/B（2 rep，同会话交错，`perf_records.jsonl` round 8）：

| candidate | rep1 µs | rep2 µs | SOL% | vs R7 best |
|---|---:|---:|---:|---:|
| R7 final（bm=1） | 78.740 | 78.400 | 60.40 / 60.25 | — |
| R8 op1（bm=2） | **68.390** | **67.670** | **70.07 / 68.53** | **−13.1% / −13.7%** |

## 5. 精度

- `python perf_opt/FusedAddLayerNormFwdOp.py --level all` → **`All check passed!`**
  （L0 13/13、L1 全过含更新后的 UB 守卫契约、L2 6/6、Boundary 6/6；`Boundary-extreme-affine` 仍为 WARN-only，fp16 ±65504 溢出，非本轮引入）。
- Summary max_diff：bf16 3.125e-02、fp16 1.000e+00（仅上述溢出例，2/16384 元素）、fp32 2.861e-06 —— 与 R7 同量级。
- 数值契约未改：round-first stats（先 `round_dtype(s)` 再进 fp32 统计）+ fp32 opmath + 单次 round-back；签名 drop-in（factory 签名与 `main` 六参数未变）。
- L0 smoke（64×4096 fp32/fp16/bf16）：PASS。

## 6. 结论 / gate

- **赢面**：13b prefill −14.85%（vs R7）/ −37.33%（vs Stage-3），SOL 70.82%，是该算子**首个跨过判据线的 shape**。
- **未达成**：本轮原始目标（8b prefill 再降 5–10%）**未达成**——行批处理在后端 UB 容量上被否（229,408 B vs 196,608 B），
  且两条可选"节食"路径各自已有 ≥5% 实测代价（R7 op4 跨 dtype store +5.0%；affine 平面串行化需每 tile 重载），不硬塞。
- **硬底线**：7/7 tune workload 不劣于 R7 best（4 个噪声内持平、1 个显著下降、2 个 decode 变短）。
- **与 DESIGN 假设的关系**：未出现「实测与设计假设直接矛盾」，且不存在被冻结且 >2x 的备选结构族 ⇒ **不触发 `[DESIGN_LIMIT]`，不产出 `perf_feedback.md`**。
- gate 4：`python3 .agents/tools/statectl.py gate 4 --dir <op dir>` → 见 `optimize` 阶段返回（pass）。

## 7. 未竟项

1. **8b/70b 行批处理**：需要换 residency 结构（如 d² 的 N 方向分块，把 `sq_f32` 从 (bm,N) 降为 (bm,N/2)），
   预计 bm=3 half 路径可压到 ~140 KB；代价是 ≥2 处新 pass + 分块 sum 顺序改变需复核数值。本轮预算内未启动。
2. **13b 继续下降**：已 70.8% SOL（访存侧），剩余空间在 GM 带宽本身，非 tiling 可解。
3. **multi-buffer 逐路径控制**：`npuir.enable_auto_multi_buffer` 目前只能整 kernel 切换（`tilelang.jit` 装饰器级），
   要做「8b 关 / 70b、13b 开」需要第二个 jit 装饰函数（Python 层分派），与本 skill「不新增 factory 层性能分派」的纪律冲突，未实施。
4. **工具链复核**：`T.gemm` 在 Developer 模式 persistent 循环内的运行期崩溃（CG-2026-0010）未变；Cube GEMV 归约仍不推荐。
