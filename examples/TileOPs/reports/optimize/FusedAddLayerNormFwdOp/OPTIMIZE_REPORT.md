# 优化报告：FusedAddLayerNormFwdOp（Stage-4 optimize）

> **口径**：本文件是**过程记录**，交付数据一律取 `perf_data/canonical.json`（CANN 9.0.0）。
> 2026-10-20 **R7**（bm=1 staging + fp32 外提）与 2026-10-21 **R8**（UB 平面模型 + fp32 行批处理）
> 两轮结构深挖；§3 的 R8 表为当前交付态来源，R6/R7 表仅作历史对照。

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `FusedAddLayerNormFwdOp`；单 kernel / 单 func / 纯 Vector（AIV），4 输入 2 输出（`y` + pre-norm `residual_out`） |
| kernel（基准，只读） | `tileops/kernels/norm/fused_add_layer_norm/fused_add_layer_norm_kernel/`（Stage-3 `_fused_add_layer_norm_kernel.py`） |
| 调优产物 | `.../perf_opt/FusedAddLayerNormFwdOp.py`（R8 merged，sha256 `91121fa3…`）+ R7/R8 分支文件 |
| 口径 | `msprof op Task Duration(us)`（唯一时延口径）+ GM Read+Write SOL ratio；数据来自 `perf_opt/perf_records.jsonl`（129 行，final = `r8final`） |
| **工具链版本戳** | CANN 9.0.0 / tilelang 0.1.2 dev root（`a2d5db1`）/ Ascend910B3（20 AICore × 2 AIV = 40 向量核）/ R8 实测 2026-10-21 |
| **运行模式** | `TILELANG_ASCEND_MODE=Developer` **必须显式导出**（`tileops_env.sh` 默认 Expert；`setdefault` 不会覆盖，Expert 下 L0 报 `hivm.hir.store only support ub to gm`） |
| 达标线 | prefill 算力 ≥60%（AI 3.0–3.8）；13b fp32 访存 ≥70%（AI 1.02）；decode 4–5µs 微型豁免 |
| benchmark | `bench_fused_add_layer_norm.py::test_fused_add_layer_norm_bench`（7 tune + 3 smoke） |
| 过程日志 | `perf_opt/opt_log.md`（§4 R7 / §5 R8）、`perf_records.jsonl`、`profiles/`、`logs/`、`REPORT_FALN_R2_DATA.md`、`REPORT_FALN_R3_DATA.md`、`probe_cube_gemv/PROBE_CUBE_GEMV.md` |
| 端到端验收 | `reports/tileops/20261021_095838_127621_FusedAddLayerNormFwdOp/`（status=passed，正确性 14/14） |
| 前置版本 | v0 / R6 / R7 报告已归档 `history_version/` |

## 1. 结论

- **一句话**：R7 用 bm=1 per-input staging + fp32 affine 外提把 70b prefill 打到 −17%、13b −27%；
  R8 把 UB 预算从「20 B/elem 包络」换成**可标定的逐平面模型**，让 fp32 N=5120 的 `block_m=2` 重新合法，
  **13b fp32 再降 −14.85%（vs R7），SOL 70.82% — 本算子首个跨过达标线的 shape**；
  8b/70b 的行批处理以精确 UB 字节账判 blocked，保持 R7 水平（噪声内）。
- **本轮（R8）目标**：8b bm=4 行批处理再降 5–10% → **未达成**（UB 容量否决，非参数可解）；
  替代赢面 13b −14.85% 已落地。
- **达标数（双分母）**：
  - 可调 shape 达标数：**1 / 5**（13b fp32 71.37% ≥70%；8b/70b 仍 33–40% <60%）。
  - 全 shape：7 tune 中 2 个 decode 为微型豁免；smoke 3 例仅精度。
- **硬底线**：**7/7 tune workload 不劣于 R7 best**（13b −14.85%，其余在 ±0.5% 噪声内）。
- **wrapper**：R5 后已切 `perf_opt` 并保持激活；R8 只更新 `perf_opt/`，无需再翻转。
- **是否建议保留采纳**：**是**（13b 达标 + 其余不回退，精度零回归，签名 drop-in）。

## 2. 优化方法与过程

### 2.1 诊断

#### R5/R7 阶段（历史）

| 观测 | 数值 | 结论 |
|---|---|---|
| 判据归类 | prefill AI 3.0–3.8；13b AI 1.02 | prefill 算力 60%；13b 访存 70% |
| R5 前瓶颈 | MTE2/VEC 交替串行（单 `stage` 四角色复用） | loads-first |
| R5 后瓶颈 | 15+ 向量 pass + 2 次 row-reduce；`advice=latency bound:pipeline caused` | 结构级 |
| R7 起点 profile | `aiv_time 60.0` / Task 62.7（4%）、vec 0.756、mte2 0.336、核间 imbalance 1.6% | 依赖链上限 |
| R7 两处缺陷 | bm=1 未走 loads-first（UB 差 256 bits）；fp32 路径 weight/bias 未外提 | 已修 |

#### R8 阶段（本轮）

| 观测 | 数值 | 结论 |
|---|---|---|
| Cube GEMV 探针（2026-10-21） | 同 tile 1.17 vs 0.637µs（1.8×）；N=8192 4.27 vs 0.637（6.7×） | **否决**（M=1/2 浪费 15/16 fractal） |
| 探针线索 | `bm=4,N=4096` reduce 0.228 vs bm=2 0.319 µs/row（−29%） | 行批处理值得验证 |
| UB 包络 | `20 B/elem`、`_UB_MAX_ELEMS=9216` | 把 fp32 N=5120 的 bm=2 误判为超额 |
| 逐平面模型（实测标定） | 三平面 + auto-mb +6 B/elem；误差 32 B | 13b bm=2 实际 **184,320 B** < 196,608 B |
| bm=4 精确字节账 | bm=3 auto-mb ON **229,408 B** / OFF 204,832 B；bm=4 OFF **262,176 B**；容量 196,608 B | **8b bm≥3 不可行** |
| 70b bm=2 | **327,680 B** | 不可行 |
| auto-mb 关闭 | 仅省 24,576 B（dtype 平面双缓冲），且是 kernel 级全局开关 | 会波及 70b/13b，不采纳 |

### 2.2 方法

**R3/R5（历史）**：weight/bias 循环不变量外提；loads-first 双输入前置装载（按 block_m 编译期分支）。

**R7**：`block_m==1` 复用 `wstage` 作第二 dtype 平面（UB 零增量）取回 loads-first（70b −17%）；
fp32 直通路径补做 affine 外提（13b −27%，GM 流量 −33%）。

**R8（本轮）**：

1. **UB 预算模型替换（唯一落地改动）**
   - 改了什么：`_UB_BYTES_PER_ELEM=20` 的包络改为**逐 dtype 平面清单 + auto-multi-buffer +6 B/elem**
     的可标定模型（误差 32 B）；`_default_block_m` 与 host guard 同步按新模型判定。
   - 为什么有效：旧模型把 fp32 N=5120 的 `block_m=2` 判为超额（白丢 −13.9%）；新模型实测 184,320 B 合法。
   - 结果：13b fp32 **78.74/78.40 → 68.39/67.67µs（−13.1%/−13.7%）**；落地 67.28µs，SOL **70.82%**。
   - 正确性：算术序列与数值契约未变；L0/L1 守卫改按新模型判定并通过。
2. **行批处理（bm=3/4）**：以精确字节账判 blocked（见 §2.1），未落地。
3. **保留 R7 全部改动**，7/7 不回退。

**被否决/未启动**：

| 候选 | 结果 | 原因 |
|---|---|---|
| bm=3/4（8b） | blocked | 229,408 / 262,176 B > 196,608 B |
| bm=2（70b） | blocked | 327,680 B |
| auto-mb 全局关闭 | 弃 | 只省 24,576 B，波及 70b/13b |
| Cube GEMV | 否决 | 慢 1.8–6.7×（探针） |
| `T.Pipelined(2)` / pass 削减 / store 折叠 | tie / +5% | R7 已证 |
| N 方向 d² 分块（sq_f32 降为 (bm,N/2)） | 未启动 | 需 ≥2 处新 pass + 数值复核，超本轮预算 |

### 2.3 逐轮记录

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| R2 | `block_m=1` 分核 | 降低 block_m | 慢 25% | ❌ |
| R3 | `hoist-wb` | weight/bias 外提 | 70b −14%、8b −7% | ✅ |
| R5 | `loadsfirst` | 双输入前置装载 | 8b −18% | ✅ |
| R6 | merged | hoist + loadsfirst | 8b 63.25/64.55、70b 140.87/144.73、13b 108.70 | ✅ R6 |
| R7-op1/op2 | `loadsfirst_bm1` / `fp32hoist` | bm=1 复用 wstage；fp32 外提 | 70b −17%、13b −27% | ✅ |
| R7-final | `r7final` | 合并 | 见历史表 | ✅ R7 |
| **R8-op1** | `r8_ubplane` | 逐平面 UB 模型 → fp32 bm=2 | 13b −14.85%（2 rep 同向） | ✅ **采纳** |
| R8-op2/3/4 | `r8_bm3/bm4/mb0` | 行批处理 / 关 auto-mb | 229–262KB 超限；mb0 仅省 24KB | ❌ blocked |
| **R8-final** | `r8final` | op1 + R7 全部改动 | 见 §3 | ✅ final（gate pass） |

**收束依据（budget_exhausted）**：8b 的行批处理在 UB 容量上被精确否决；13b 赢面已落地；剩余方向需 ≥2 处新 pass，超预算。

## 3. 优化结果

### 3.1 R8 final（当前交付态，同会话）

| workload | base µs | R7 best µs | **R8 final µs** | vs R7 | final SOL% | 判据/线 |
|---|---:|---:|---:|---:|---:|---|
| llama-8b-prefill fp16 | 82.36 | 62.65 | **62.94** | +0.46%（噪声） | 32.99 | 算力 60% ❌ |
| llama-8b-prefill bf16 | 84.26 | 64.43 | **64.48** | +0.08%（噪声） | 37.88 | 算力 60% ❌ |
| llama-70b-prefill fp16 | 165.37 | 116.79 | **117.10** | +0.27%（噪声） | 35.01 | 算力 60% ❌ |
| llama-70b-prefill bf16 | 168.76 | 120.40 | **120.44** | +0.03%（噪声） | 40.43 | 算力 60% ❌ |
| llama-13b-prefill fp32 | 107.36 | 79.01 | **67.28** | **−14.85%** | **70.82** | 访存 70% ✅ |
| llama-8b-decode bf16 | 4.24 | 4.19 | 4.16 | −0.72% | 0.52 | 微型豁免 |
| llama-70b-decode bf16 | 4.65 | 4.21 | 4.17 | −0.96% | 0.87 | 微型豁免 |

- final 记录：`round=8`、`candidate_id=r8final`、`phase=final`，7 条同一候选（sha256 `91121fa3…`）。
- 端到端 `tileops-report`（交付口径）：8b 63.26/64.74µs（32.80/37.68）、70b 117.56/121.26µs（35.04/39.97）、
  13b **67.20µs（71.37）**、decode 4.78/4.66µs；**可调 shape ratio 32.80–71.37%**。

### 3.2 历史数据（仅作对照）

| 版本 | 8b fp16/bf16 | 70b fp16/bf16 | 13b fp32 |
|---|---|---|---|
| Stage-3 baseline | 82.45 / 84.01 | 164.23 / 168.28 | 107.64 |
| R6 final | 63.25 / 64.55 | 140.87 / 144.73 | 108.70 |
| R7 final | 62.65 / 64.43 | 116.79 / 120.40 | 79.01 |
| **R8 final** | **62.94 / 64.48** | **117.10 / 120.44** | **67.28** |

**测量口径与噪声**：R8 采用同会话 A/B/A/B 交错（每点独立 msprof 进程）；8b/70b 的 +0.03~0.46% 在 ±3% 噪声内，
不判回退。13b 的两个独立 rep 同向 −13.1%/−13.7%。

## 4. 精度

### 4.1 R8 最终回归（conductor 亲自执行，2026-10-21）

- 命令：`TILELANG_ASCEND_MODE=Developer python perf_opt/FusedAddLayerNormFwdOp.py --level all`
  → **`All check passed!`**
  - L0 13/13 PASS；L1 14 项 PASS；L2 6/6 PASS；Boundary 6/6 PASS。
  - Summary max_diff：bf16 3.125e-02、fp16 2.000（仅 extreme-affine overflow 例）、fp32 2.861e-06。
- smoke 3 例：PASS。
- **端到端**：正确性 **14/14（100%）**，avg max_abs_err 3.29e-03，status=**passed**。
- 数值契约未改：round-first stats + fp32 opmath + 单次 round-back；签名 drop-in。

### 4.2 历史

R6/R7 的 `--level all` 全过；Boundary `extreme-affine` 为 fp16 溢出（非本轮引入）。

## 5. 时间与成本

| 项 | R7 | R8 |
|---|---|---|
| 起止 | 2026-10-20 19:30→20:30 | 2026-10-21 08:27→10:03 |
| 总墙钟 | ≈60 min | **≈112 min**（4 分支 + control + 精确字节探针 + A/B）；conductor 验收另 ≈20 min |
| 实验分支 | 5 | 5（ubplane + bm3/bm4/mb0 + control） |
| 采集参数 | `msprof op`，launch-count=20 / warm-up=5 | 同左；A/B/A/B 交错 |
| 预算 | ≤180 min | ≤120 min，未超 |

## 6. 产物

| 类别 | 路径 / 状态 |
|---|---|
| final kernel | `perf_opt/FusedAddLayerNormFwdOp.py`（sha256 `91121fa392fcbf2db896394eb3e7c3654966e6ca1cde9bff250602a5745d3a45`） |
| 分支文件 | `perf_opt/FusedAddLayerNormFwdOp_r7_*.py`、`FusedAddLayerNormFwdOp_r8_*.py` |
| 调优记录 | `perf_opt/opt_log.md`（§4 R7 + §5 R8 + 复盘） |
| 结构化记录 | `perf_opt/perf_records.jsonl`（129 行，append-only；final 7 条 = `r8final`） |
| 分类清单 | `perf_opt/workload_inventory.json`（7 tune + 3 smoke） |
| 采集工件 | `perf_opt/profiles/`、`perf_opt/logs/`、`perf_opt/probe_r8_ub/`（UB 字节账探针） |
| 数据交接 | `REPORT_FALN_R2_DATA.md`（R7）、`REPORT_FALN_R3_DATA.md`（R8） |
| Cube 探针 | `perf_opt/probe_cube_gemv/PROBE_CUBE_GEMV.md`（否决证据） |
| gate 4 | **pass**（failures []；1 warning `S4-JIT`：`target="npuir"` 经模块常量 `_JIT_KWARGS` 传入，lint 正则未识别，功能等价） |
| 端到端 report | `reports/tileops/20261021_095838_127621_FusedAddLayerNormFwdOp/`（passed） |
| wrapper | 已激活 `perf_opt`（R5 后翻转，本轮未改） |

> **记录例外披露**：gate 4 要求每个 tune workload 恰一条 `phase=final`。R8 收束时把 R7 的 7 条旧
> `phase=final` 降级为 `phase=merged`（仅标签；round / candidate_id / duration / sha256 / profile 路径未改），
> 降级前快照 `/tmp/opencode/perf_records_before_r8_final.jsonl`。

## 7. 天花板 / 豁免 / 缺口

- **豁免项**：`llama-8b-decode bf16`（4.16/4.78µs）、`llama-70b-decode bf16`（4.17/4.66µs）— 微型 shape；smoke 仅精度。
- **8b prefill 仍停在本设计族上限**（62.94/64.48µs，SOL 33.0/37.9%）：
  - 行批处理 bm=3/4 被 UB 容量精确否决（229,408 / 262,176 B vs 196,608 B）；70b bm=2 需 327,680 B；
  - 关 auto-mb 只省 24,576 B 且全局波及；
  - Cube GEMV 探针否决（慢 1.8–6.7×）；
  - 剩余为每 task 依赖链（2 次 row-reduce + ~16 链接），R7 的 op3/op4/op5 已证不可削。
- **13b 已达标**（71.37%），R8 的 UB 平面模型是唯一改动。
- **0/4 其余 prefill 达 60% 线**：要跨线需 N 方向 d² 分块（sq_f32 降为 (bm,N/2)，需 ≥2 处新 pass）
  或换结构，均属设计级。
- **`[DESIGN_LIMIT]`**：不触发——本轮仍拿到实现级收益（13b −14.85%），不存在「被冻结且 >2x」的备选结构。

## 8. 建议

1. **保留采纳**：R8 final 已激活，13b 首次达标（71.37%），其余 6 个 workload 不回退，精度零回归。
2. **不再投入**：bm=3/4（UB 精确否决）、Cube GEMV（探针否决）、auto-mb 关闭（收益/影响比差）、
   pass 削减 / task 流水 / store 折叠（R7 已证）。
3. **如要继续**：唯一剩余方向是 **N 方向 d² 分块**（把 `sq_f32` 降为 `(bm, N/2)` 以腾出 bm=4 的空间），
   但需 ≥2 处新 pass 与数值复核，属设计级新立项。
4. **运行注意**：本 kernel 必须显式 `TILELANG_ASCEND_MODE=Developer`；`tileops_env.sh` 的 Expert 默认值
   不会覆盖 kernel 的 `setdefault`，会导致 L0 全挂。
5. **工具链升级时复核**：UB 平面模型的 +6 B/elem auto-mb 常数；以及 CG-2026-0030（cbuf NZ padding 诊断）。

---

## 11. R9（2026-10-22）：N 方向 d² 分块 / bm=3–4 —— 结论 blocked

> 目标：把 `sq_f32` 降为 `(bm, N/2)` 腾 UB，让 8b prefill（N=4096）的 block_m 从 2 提到 3/4，再降 ≥5%。
> **结果：不可落地；`FusedAddLayerNormFwdOp.py` 保持 R8 不变（sha256 `91121fa3…`），零回退。**

### 11.1 收益上限是真实的（bm 曲线）

邻近可容 shape 的微基准（同结构、只改 bm）：

| block_m | Task Duration | 相对 bm=2 |
|---|---:|---:|
| 2 | 44.40 µs | — |
| 3 | 37.85 µs | **−14.8%** |
| 4 | 33.84 µs | **−23.8%** |

⇒ 行批处理确实是 8b 的正确杠杆，问题只在“能否放得下”。

### 11.2 为什么放不下（精确字节账）

| 布局 | bm | auto-mb | 需要 B | 可用 B |
|---|---:|---|---:|---:|
| R8（copy-cast） | 3 | ON / OFF | **229,408 / 204,832** | 196,608 |
| 显式 2 平面 staging | 3 | ON / OFF | **229,408 / 204,832**（与 R8 一字节不差） | 196,608 |
| 8b 现行 | 2 | ON | 163,840 | 196,608 |
| bm=4（OFF） | 4 | OFF | 262,176 | 196,608 |
| 70b bm=2 | 2 | ON | 327,680 | 196,608 |

- “显式 2 平面 staging”与 R8 字节完全相同 ⇒ **R8 §5.7 的“d² 降半宽可省 UB”假设被证伪**：省不下。
- `mb0`（全局关 auto-multi-buffer）只在 8b/70b 上 tie（≤±1%），却让 13b fp32 **67.44 → 86.65µs（+28.5%）**，且只省 24,576 B，仍不够 bm=3 ⇒ 不可采纳。

### 11.3 编译器能力缺口（新登记）

- **CG-2026-0033**：v-prefix op 的 dst 必须是整 buffer；`vcast` 不接受 region/slice；因此无法把 N 拆半后分别写进同一平面的两半。
- **CG-2026-0034**：`vbrc` 不允许跨 dtype，`fp32` 标量广播到 `bf16` 平面无法直接表达。
- 这两条同时满足前，bm=3/4 的“半宽 d²”路径在语言层不可写，不是 UB 预算问题。

### 11.4 结论与建议

- **R9 无 winner**，保留 R8：8b prefill 62.94/64.48µs（SOL 33.0/37.9%）、13b 67.28µs（SOL 70.8%）、70b 117.10/120.44µs。
- 要拿到 bm=3/4 的 −15~24%，需先补 CG-2026-0033（region dst）或 CG-2026-0034（跨 dtype vbrc）；
  在工具链修复前不再重复尝试。
- 证据：`perf_opt/opt_log.md` §6、`perf_records.jsonl` round 9、CG-2026-0033/0034。
