# REPORT_FALN_R4_DATA — R9（N 方向 d² 分块 / bm=3-4）数据交接

> 算子：`FusedAddLayerNormFwdOp`（单 kernel / 纯 Vector AIV，4 入 2 出）
> 轮次：**R9**（2026-10-22），上一轮 R8（交付态）
> 工具链：tilelang `0.1.2+a2d5db1` dev root（`/mnt/workspace/tilelang-mlir-ascend`）/ CANN 9.0.0 / Ascend910B3（20 AICore × 2 AIV = 40 向量核）
> 口径：`msprof op Task Duration(us)`（唯一时延口径，`--launch-count=20 --warm-up=5`，取 20 次中位数）+ GM Read+Write SOL ratio
> 运行模式：**`TILELANG_ASCEND_MODE=Developer` 必须显式导出**（`tileops_env.sh` 默认 Expert，Expert 下 L0 全挂）
> 结论：**TUNING_COMPLETED，无 winner，保留 R8**（sha256 `91121fa3…` 未改一行）

---

## 1. 本轮目标与诊断

### 1.1 目标（conductor 下达）

8b prefill（N=4096）通过 **N 方向 d²（sum-of-squares）分块**腾出 UB，使 `block_m=3/4` 落地，Task Duration 再降 ≥5%；
70b（N=8192）/13b（N=5120）不得回退；硬底线 kernel 必须变短。

### 1.2 诊断（R8 交付态）

| 观测 | 数值 | 结论 |
|---|---|---|
| 8b prefill fp16/bf16 | 62.94 / 64.48 µs，SOL 33.0 / 37.9% | 停在设计族上限（需算力 60%） |
| pipe 画像（R7 画像，8b fp16） | `aiv_vec 0.756` / `mte2 0.336` / `mte3 0.170`，`aiv_time 60.0` vs Task 62.7 | 向量链 + 每 task 依赖延迟主导 |
| 每 tile 成本（bm=2，N=4096） | 2.41 µs / tile，其中 row-reduce 0.637 µs（26%） | reduce 是 bm 的主要摊薄对象 |
| R8 结论 | bm=2 为 UB 上限：bm=3 需 229,408 B（auto-mb ON）/204,832 B（OFF），容量 196,608 B | 缺口 32,800 / 8,224 B |

### 1.3 立项前的收益标定（本轮新增探针，未改 kernel）

R8 源码原样 + `probe_r9_ub/bench_shape.py`，取 **M=2048 / N=2048 / fp16**（该 shape 下 bm≤5 都在 UB 内，
用来测「行批处理的真实收益上限」，不代表 workload 数值）：

| block_m | Task Duration(µs) | vs bm=2 | task/core | 说明 |
|---:|---:|---:|---:|---|
| 2 | 44.400 | — | 25.6 | 1024 logical blocks |
| 3 | 37.850 | **−14.8%** | 17.1 | 683 blocks（含 5.6% wave 不齐） |
| 4 | 33.840 | **−23.8%** | 12.8 | 512 blocks（wave 最齐） |

拟合 `T = tasks·(a + b·bm)`：**a ≈ 0.90 µs/task（固定开销）**、**b ≈ 0.42 µs/row**。
外推到 N=4096（每行数据翻倍 → b≈0.8）：bm=3 预计 **−7~9%**，含 wave 不齐约 **−5~8%**。
→ 行批处理是真杠杆，本轮失败在**容量**而非收益。

---

## 2. UB 字节账（后端 `requires ... bits` 原文，capacity 196,608 B）

| 布局 | bm | auto-mb | 后端自报 | 字节 | 判定 |
|---|---:|---|---:|---:|---|
| R8 copy-cast（现行） | 2 | ON | — | 163,840 | 编译通过（现行默认） |
| R8 copy-cast | 3 | ON | 1,835,264 bits | **229,408** | 缺口 **32,800 B** |
| R8 copy-cast | 3 | OFF | 1,638,656 bits | **204,832** | 缺口 **8,224 B** |
| op2 显式 2 平面 staging | 3 | ON | 1,835,264 bits | **229,408** | 与 R8 **一字节不差** |
| op2 显式 2 平面 staging | 3 | OFF | 1,638,656 bits | **204,832** | 与 R8 **一字节不差** |
| R8（bm=1，N=8192，70b） | 1 | ON | — | 163,840 | 编译通过（bm=2 需 327,680 → 早已否决） |

**逐平面拆解（bm=3，N=4096，auto-mb OFF）**：`stage(2) + x_f32(4) + sq_f32(4) + w1/b1(2.67) + 隐式 cast scratch(4) ≈ 16.67 B/elem = 204,832 B`。
要到 196,608 B 必须再降 **4 B/elem（= 一个 `(bm,N)` fp32 平面或等价物）**。

---

## 3. 方法：为什么"把 sq_f32 降为 (bm,N/2)"在本工具链写不出来

`probe_r9_ub/colslice_probe2.py` / `colslice_probe3.py`（每个 primitive 独立编译 + 数值校验，N=4096、bm=3）：

| primitive | 形态 | 结果 |
|---|---|---|
| `vmul` | 列半切片 src + 全宽 `(bm,N/2)` dst | **OK**，`max_diff < 2e-3` |
| `reduce_sum` | 全宽 `(bm,N/2)` fp32 → `(bm,1)`，两半 `vadd` 合并 | **OK**，`max_diff 4.9e-04` |
| `vbrc` | `(1,N/2)` fp32 → `(bm,N/2)` fp32 | **OK** |
| `vadd`/`vmul` | **列半切片做 in-place dst** | **FAIL** `Check failed: buffer_shape0[i] == buffer_shape1[i]`（`codegen_npuir_dev.cc:175`） |
| `vadd`/`vmul` | **行切片做 dst**（per-row affine，免 `vbrc`） | **FAIL** 同上（`rowslice_probe.py`） |
| `vcast` | 列半切片 src → 半宽 dst | **FAIL** 同上 |
| `vbrc` | dtype src → fp32 dst | **FAIL** `'hivm.hir.vbrc' op requires the same element type for all operands` |

**一般规则（8 种形态枚举）**：v-op 的 **dst 必须是整 buffer**（不能是 region）；src 可以是 region（`vmul`/`reduce`）；
`vcast` 连 src 的 region 形状都不认；`vbrc` 要求同 dtype。因此「逐行 affine 顺带免掉 `vbrc`（省 2 pass + 可能的广播 scratch）」
这条也在编译期被堵死。

**推论（决定性）**：
1. 残差加法必须有**两个全宽 fp32 平面**：`vcast` 不能把 dtype 平面写进半宽 fp32，`vadd` 不能 in-place 写列半切片；
   所以 `sq` 一旦降到 `(bm,N/2)`，`s = x + residual` 就没有合法落地面。
2. d²/affine **读**列半切片合法，但 affine 必须写回一个全宽平面 → 再加一个 `(bm,N/2)` fp32 平面（+2 B/elem，等于没省）。
3. w/b 压成 dtype 需 `vbrc` 跨 dtype cast（被拒）；改用「dtype 平面 + 共用 `(1,N)` fp32 scratch」只省 8,192 B ⇒ 196,640 B，
   比容量**多 256 bits**，仍被后端拒绝。

被否决的其它路径：

| 路径 | 结果 | 依据 |
|---|---|---|
| auto-mb OFF（全局，省 24,576 B） | **invalid：13b +28.5%** | 67.44 → 86.65 µs，SOL 71.97 → 55.16%（§5 A/B 表） |
| 显式 staging 取代 copy-cast（op2） | blocked：字节不变 | §2 两行 229,408/204,832 与 R8 完全相同 |
| 两遍半宽（pass1 stats + pass2 重读 `residual_out`） | 纸面否决 | 流量 8 → 12 B/elem（**+50%**）；向量 pass 13 → ~16 全宽当量（**+45%**）；对 AI 3.0 的算力受限 shape 净收益不可信 |
| Cube GEMV / bm=4 硬塞 / 全局关 auto-mb | 早已否决（R8） | 探针慢 1.8–6.7×；262,176 B；见 §2 |

---

## 4. 逐 workload：base / final（交付态未变）

final = R8 merged（`perf_opt/FusedAddLayerNormFwdOp.py`，sha256 `91121fa392fcbf2db896394eb3e7c3654966e6ca1cde9bff250602a5745d3a45`），
7 条 `phase=final`（`r8final`，round 8，同一候选）；`R9 同会话对照` = 本轮 round 9 `r9cur`（**同一 artifact**，`phase=candidate`）。

| workload | base µs（Stage-3） | R9 同会话对照 µs | final µs（R8） | vs base | final SOL% | 判据/线 |
|---|---:|---:|---:|---:|---:|---|
| llama-8b-prefill-float16 | 82.360 | 62.870 | **62.940** | −23.58% | 32.99 | 算力 60% ❌ |
| llama-8b-prefill-bfloat16 | 84.260 | 64.420 | **64.480** | −23.47% | 37.88 | 算力 60% ❌ |
| llama-70b-prefill-float16 | 165.370 | 116.710 | **117.100** | −29.19% | 35.01 | 算力 60% ❌ |
| llama-70b-prefill-bfloat16 | 168.760 | 120.720 | **120.440** | −28.63% | 40.43 | 算力 60% ❌ |
| llama-13b-prefill-float32 | 107.360 | 67.440 | **67.280** | −37.33% | **70.82** | 访存 70% ✅ |
| llama-8b-decode-bfloat16 | 4.240 | 4.370 | **4.160** | −1.89% | 0.52 | 微型豁免 |
| llama-70b-decode-bfloat16 | 4.650 | 4.340 | **4.170** | −10.32% | 0.87 | 微型豁免 |

- **R9 无交付变更**：表中 final 与 base 的差值全部来自 R6/R7/R8 的历史收益；本轮 7 点对照与 final 偏差 ≤3%（噪声内）。
- 达标数（不变）：可调 prefill shape **1/3**（13b 70.8% ≥ 70%）；全 shape 1/7（decode 2 例微型豁免）。
- **硬底线**：7/7 tune workload 不劣于 R8。

### 4.1 A/B（R9 分支 op3 `mb0`，同会话逐 workload 交错）

| workload | R8（`r9cur`）µs | `mb0` µs | Δ | 判定 |
|---|---:|---:|---:|---|
| llama-13b-prefill-float32 | 67.44 | 86.65 | **+28.5%** | invalid |
| llama-8b-prefill-float16 | 62.87 | 62.90 | +0.05% | tie |
| llama-8b-prefill-bfloat16 | 64.42 | 65.06 | +1.0% | tie |
| llama-70b-prefill-float16 | 116.71 | 116.94 | +0.2% | tie |
| llama-70b-prefill-bfloat16 | 120.72 | 120.40 | −0.3% | tie |
| llama-8b-decode-bfloat16 | 4.37 | 4.20 | −3.9% | 微型豁免 |
| llama-70b-decode-bfloat16 | 4.34 | 4.18 | −3.7% | 微型豁免 |

→ `auto-multi-buffer` 是 **fp32 N=5120 bm=2** 路径的 28.5% 来源（本轮新实测常数）；half 路径 ≤±1%。
该 flag 为模块级（`_JIT_KWARGS`），不能按 shape 分派 ⇒ 永久不可用作"只给 8b 省字节"。

---

## 5. 精度

- 交付 artifact 未改一行 ⇒ 数值契约不变：**round-first stats（先 `round_dtype(s)` 再进 fp32 统计）+ fp32 opmath + 单次 round-back**；签名 drop-in。
- 本轮在交付 artifact 上实跑 `python FusedAddLayerNormFwdOp.py --level all` → **`All check passed!`**（rc=0，2026-10-22 10:40）：
  `[L0] ALL PASS (13 items, 13 runs)`（5.7 s）、L1 全过（125.6 s）、L2（17.9 s）、Boundary（8.7 s）；
  日志 `logs/r9/level_all_r9final.log`（150 行）。本轮分支 `FusedAddLayerNormFwdOp_r9_mboff.py` 的 L0 亦 PASS。
- 最大值差量级与 R8 一致（bf16 3.1e-02 / fp16 1.0e+00 仅 `Boundary-extreme-affine` 的 ±65504 fp16 溢出例、
  fp32 2.9e-06）；该 Boundary WARN 为 R6 同源、非本轮引入、非阻塞。

---

## 6. sha256 / 记录 / gate

| 项 | 值 |
|---|---|
| final artifact | `perf_opt/FusedAddLayerNormFwdOp.py` = `91121fa392fcbf2db896394eb3e7c3654966e6ca1cde9bff250602a5745d3a45`（与 R8 相同，未修改） |
| 分支文件 | `FusedAddLayerNormFwdOp_r9_mboff.py`（op3，已测 invalid）、`FusedAddLayerNormFwdOp_r9_stage2.py`（op2，bm=3 被 UB 拒） |
| perf_records | `perf_records.jsonl` 143 行；round 9 新增 **14 行**（`r9cur`×7 + `r9mb0`×7，`phase=candidate`）；final 仍为 `r8final`×7 |
| 探针工件 | `perf_opt/probe_r9_ub/`（`colslice_probe{,2,3}.py`、`bench_shape.py`、`run_bm_curve.sh`、`probe_r9_build.py`、`profiles/`、`logs/`） |
| raw profile | `perf_opt/profiles/candidate-r9-*`（14 个 workload 点）、`probe_r9_ub/profiles/probe-bmcurve-*` |
| 过程日志 | `perf_opt/logs/r9/`（`level_all_r9final.log`）、`perf_opt/logs/candidate-r9-*.log` |
| gate 4 | **pass**（`failures: []`；1 warning `S4-JIT`：`target="npuir"` 经模块常量 `_JIT_KWARGS` 传入，功能等价） |

---

## 7. 未竟项（跨过 UB 墙需要什么）

1. **bm=3 缺 8,224 B**，只能靠"少一个全宽 fp32 平面"。需要下列任一编译器能力：
   - ① `vcast` 支持区域形状语义（列半切片 src → 半宽 dst）；或
   - ② 列半切片可做 in-place dst；或
   - ③ `(bm,N)` fp32 平面可按 `(bm,N/2)` 逻辑视图复用。
   任一条成立 → `sq` 降半宽，bm=3 = 180,256 B（容量内 16 KB 余量）。
2. 退路：**`vbrc` 支持跨 dtype（cast+broadcast）** → w/b 压 dtype，bm=3 = 188,448 B（容量内 8 KB 余量）。
3. **bm=4**（探针 −23.8%）在 N=4096 需 ≤12 B/elem，必须同时做到 (1)+(2)，属设计级新立项。
4. 8b 若要跨 60% 算力线，单靠行批处理（上限 ≈ −9%）仍不够；需要**降低每行 vector pass 数**
   （当前 13 个全宽当量：3 vcast + vadd + vsub + 2 vmul + 2 reduce + 2 vbrc + 2 arith + 1 round-back），
   而 R7 op5 已证"去 2 个 vbrc pass"为 tie（vbrc 被后端与 vmul/vadd 融合），故 pass 削减方向需换结构（如把 stats 移出 Vector 链）。
5. 能力缺口按 AGENTS.md 记入 `.agents/evolution/capability-gaps.md`（v-op 区域形状语义 + `vbrc` 跨 dtype 两条）。

---

## 8. 复现命令（全部经 `npu_lock.sh`）

```bash
cd /mnt/workspace/tile-ascendnpu-ops && source /mnt/workspace/tileops_env.sh
D=examples/TileOPs/tileops/kernels/norm/fused_add_layer_norm/fused_add_layer_norm_kernel/perf_opt

# 1) 收益标定：bm 曲线（不改 kernel）
.agents/tools/npu_lock.sh --wait 900 --wait-idle 60 --timeout 7200 -- \
  bash -c "cd $D/probe_r9_ub && TILELANG_ASCEND_MODE=Developer bash run_bm_curve.sh 2 3 4"

# 2) 字节账（后端 requires 原文）
.agents/tools/npu_lock.sh --wait 900 --wait-idle 60 --timeout 7200 -- \
  bash -c "cd $D/probe_r9_ub && TILELANG_ASCEND_MODE=Developer \
    python3 probe_r9_build.py FusedAddLayerNormFwdOp_r9_stage2.py 2048 4096 float16 3 0"

# 3) v-op 视图语义（4 个 primitive）
.agents/tools/npu_lock.sh --wait 900 --wait-idle 60 --timeout 7200 -- \
  bash -c "cd $D/probe_r9_ub && TILELANG_ASCEND_MODE=Developer python3 colslice_probe3.py"

# 4) A/B（R9 分支 + 交付态，逐 workload 交错）
.agents/tools/npu_lock.sh --wait 900 --wait-idle 60 --timeout 7200 -- \
  bash -c "cd $D && TILELANG_ASCEND_MODE=Developer python3 run_experiments.py \
    --experiments experiments_r9_b1.json --all-tune --order workload --round 9 --phase candidate"

# 5) gate
python3 .agents/tools/statectl.py gate 4 --dir examples/TileOPs/tileops/kernels/norm/fused_add_layer_norm/fused_add_layer_norm_kernel
```
