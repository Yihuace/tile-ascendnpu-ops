# 优化报告：ClampScalarFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `ClampScalarFwdOp`（复用同 kernel 者：`ClampMinFwdOp`）；family elementwise；**单 kernel / 单 func / 纯 Vector（AIV）**，非 Mixed；语义为标量上下界 clamp（原文以 `vclamp`/`vmax`/`vmin` 链表述，公式形式原文未记录） |
| kernel（基准，只读） | `examples/TileOPs/tileops/kernels/elementwise/clamp_scalar/clamp_scalar_kernel/clamp_scalar_fwd_kernel.py` |
| 调优产物 | `.../clamp_scalar_kernel/perf_opt/clamp_scalar_fwd_kernel.py` |
| 口径 | `msprof op Task Duration(us)`（唯一） |
| **工具链版本戳** | tilelang `0.1.2+417dae99`（dev root）/ **CANN 上一代（升级前）** / Ascend910B3 / torch_npu 2.7.1.post4 / 日期 `2026-10-15`（取自 `.start_utc`） |
| 达标线 | 原文按 **≥60%** 判定（原文「全部 ≥60% 目标」）；AI 与归类依据原文未记录 |
| benchmark | 原文未记录 benchmark 文件::函数与 tune/smoke 计数（原文仅记 `tileops.reporting.cli run --op ClampScalarFwdOp`） |
| 过程日志 | `.../clamp_scalar_kernel/perf_opt/opt_log.md`、`perf_records.jsonl` |
| 前置报告 | 本文件 v0 归档 `history_version/OPTIMIZE_REPORT_v0.md`（内容相同，仅结构未归一） |

## 1. 结论

- 一句话：**去除被实测证伪为冗余的 NaN 恢复**（向量链 5→3 op），所有测量项带宽利用率由
  45–63% 提升至 **71.9%–82.8%**，延迟降 **12%–39%**。
- **达标数**：全 shape **11/11**、可调 shape **11/11**、豁免 **0**
  （原文明确「本算子（ClampScalar/ClampMin）**无微型 shape**」）。
- 关键收益（代表值）：
  - `16M` fp16 **42.1 → 26.1 µs（−38.0%）**，ratio **45.5% → 73.5%**；
  - `16M` bf16 **42.3 → 25.9 µs（−38.7%）**，ratio **51.7% → 74.0%**；
  - `16M` fp32 **6上一代工具链 → 46.3 µs（−32.4%）**，ratio **56.0% → 82.8%**；
  - 最大 shape `256M` fp16 **891.4 → 751.4 µs（−15.7%）**，ratio **62.4% → 73.9%**。
- **是否建议采纳 wrapper**：✅ 建议采纳（原文记录：wrapper 切换块已翻至 perf_opt；接入后
  `tileops.reporting.cli run --op ClampScalarFwdOp` 正确性 **11/11**、`status=passed`）。
  收益来自纯 op 数削减、无 shape 回退，判据余量充足（最低 71.9% vs 60% 线）。
- **9.0 端到端复测（2026-10-18，run `20261018_131331_663371_ClampScalarFwdOp`）**：正确性 **11/11**、report `passed`，但 ratio 降到 **58.21–70.16%（1/11 ≥70%）**，相对本报告 上一代工具链 的 71.9–82.8% 明显回退。**A/B 定性（2026-10-18，baseline run `20261018_160303_880572`）**：同链路 9.0 baseline 仅 **45.35–64.37%**，tuned 仍比 baseline 快 **7.6–40%** ⇒ 回退是 9.0 工具链/环境整体下移，不是调优失效；**保留 perf_opt**，上一代工具链 的绝对值不再跨版本引用。
- **ClampMin（同 kernel）9.0 端到端（run `20261018_153708_107121_ClampMinFwdOp`）**：正确性 **14/14**、ratio **61.77–68.84%（0/7）**；同受 9.0 clamp 链回退影响。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测 | 数值 | 结论 |
|---|---|---|
| 瓶颈 pipe（16M fp16，40 vector core） | `aiv_vec_ratio = 0.72`（最高）、`aiv_mte2_ratio = 0.56`、`aiv_mte3_ratio = 0.28`、`aiv_scalar_ratio = 0.08` | **向量单元是瓶颈，不是 DMA**（`mte2_ratio ≪ 0.95`，计算链未隐藏；符合 pattern-library PL-1.6 判据） |
| 单核有效带宽 | `aiv_mte2_active_bw ≈ 39 GB/s/core` | 搬运不是限制 → 杠杆是「减向量 op」，不是调 block/core |
| 配置扫描 | block_size ∈ {2048, 4096, 8192} × num_cores ∈ {auto, 20, 40} | 默认已最优，**config 调参无收益** |
| 计算链构成 | `vcast(f16→f32) → vclamp/vmax/vmin → vcmp(NaN) → vselect(NaN) → vcast(f32→f16)`（5 条向量 op） | NaN 恢复（`vcmp`+`vselect`）占 2/5；若冗余即为纯开销 |
| 设计侧来源 | NaN 恢复是设计 R2 因「Ascend `vmax`/`vmin` 未文档化 NaN 传播」而加 | 该假设可被实测证伪，是首选优化点 |
| 噪声底 | 原文未记录独立噪声测量 | 以「全 11 项一致大幅下降 + Boundary 全绿」作为判定支撑 |

### 2.2 方法（做了什么、为什么这样做）

1. **删除 `vcmp` + `vselect`（NaN 恢复两点）**，向量链
   `vcast → vclamp/vmax/vmin → vcast`（**5 → 3 条向量 op**）。
   - 为什么有效：瓶颈 pipe 是 AIV VEC（0.72），直接削掉 40% 的向量 op，等于按比例抬升带宽利用率；
     实测 ratio 45–63% → 71.9–82.8% 与「op 数削减」一致。
   - 正确性依据（证伪实验，本工具链实测）：
     - `vmax`/`vmin`/`vclamp` 在 **fp32 域已传播 NaN**——3 种边界形态（min-only / max-only / both）
       × fp16 × **586 个 NaN 点**，全部 `nan_ok=True`；
     - `vclamp` 有限值 `max_diff = 0.00e+00`；`inf/-inf` 与 torch 一致。
2. **`nan-bound` 分支的 `vdiv` 改原地写 `buf1`**：`buf2` 不再使用，少一次中间 buffer 占用/搬运。
3. 改动**只落在 `perf_opt/`**，基线文件未动；产物仅 `perf_opt/clamp_scalar_fwd_kernel.py`。

### 2.3 逐轮记录

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| 预检 | — | 读 `conductor-scenarios/optimize.md`、`tilelang-op-optimize` skill、pattern-library INDEX/hardware-context/elementwise.md | 确定口径与手法 | — |
| R0 baseline | baseline | `tileops-report run --op ClampScalarFwdOp` + 单例 `PipeUtilization.csv` | 16M fp16 **42.1µs / 45.5%**；向量瓶颈（vec 0.72 ≫ mte2 0.56） | 定为杠杆点 |
| R1 | config 扫描 | block_size ∈ {2048,4096,8192} × num_cores ∈ {auto,20,40} | 默认已最优，无收益 | ❌ 否决：config 调参方向 |
| R2 | NaN 证伪实验 | 3 形态 NaN/inf 传播实测 + 有限值 max_diff | NaN 已原生传播 → 恢复冗余，`vclamp` 有限值 max_diff=0 | ✅ 判定恢复两点可删 |
| R3 | `perf_opt` 变体 | 去 NaN 恢复（5→3 op）+ `vdiv` 原地写 | 全 11 项延迟 −12%~−39%、ratio 71.9–82.8% | ✅ 采纳为 current best |
| R4 | 回归 | `perf_opt/...py --level all` | **All check passed**（L0/L1/L2/Boundary 全绿，含 NaN→NaN 512 点、NaN-bound→全 NaN、inf/−inf、min>max） | ✅ 通过精度门禁 |
| R5 | 采纳 + 验收 | 翻 wrapper 切换块 → perf_opt；`tileops-report run` | 正确性 **11/11**、`status=passed` | ✅ 收束 |

## 3. 优化结果（msprof op，baseline → perf_opt）

| workload | base µs | final µs | Δ | base SOL% | final SOL% | ≥60% |
|---|---:|---:|---:|---:|---:|:--:|
| 16M / float16 | 42.1 | 26.1 | −38.0% | 45.5 | 73.5 | ✅ |
| 16M / bfloat16 | 42.3 | 25.9 | −38.7% | 51.7 | 74.0 | ✅ |
| 16M / float32 | 6上一代工具链 | 46.3 | −32.4% | 56.0 | 82.8 | ✅ |
| 256M / float16 | 891.4 | 751.4 | −15.7% | 62.4 | 73.9 | ✅ |
| 256M / bfloat16 | 884.6 | 773.3 | −12.6% | 62.8 | 71.9 | ✅ |
| 16M-min-only / float16 | 39.3 | 26.2 | −33.5% | 48.7 | 73.3 | ✅ |
| 16M-min-only / bfloat16 | 38.7 | 24.7 | −36.1% | 49.5 | 77.5 | ✅ |
| 16M-min-only / float32 | 64.0 | 49.4 | −22.7% | 60.0 | 77.6 | ✅ |
| 16M-max-only / float16 | 38.6 | 26.2 | −32.2% | 49.7 | 73.3 | ✅ |
| 16M-max-only / bfloat16 | 37.5 | 25.6 | −31.8% | 51.1 | 75.0 | ✅ |
| 16M-max-only / float32 | 63.3 | 47.6 | −24.8% | 60.6 | 81.0 | ✅ |

**结论（原文）**：所有 shape 的带宽利用率由 45–63% 提升至 71.9%–82.8%，全部 ≥60% 目标；延迟降 12%–39%。

**测量口径**：`msprof op` 单次进程，`Task Duration(us)` 为唯一时延口径；原文未记录同一 workload 的
交错复测次数与噪声带。收益最小两项为 `256M bf16`（−12.6%）与 `256M fp16`（−15.7%）。

## 4. 精度

- `python perf_opt/clamp_scalar_fwd_kernel.py --level all` → **All check passed**
  （L0 / L1 / L2 / Boundary 全绿）。
- 关键边界用例：**NaN→NaN 512 点**、**NaN-bound → 全 NaN**、**inf / −inf**、**min > max**（原文列出的形态）；
  subnormal / −0 的逐项结果原文未记录。
- 数值依据：`vclamp` 有限值 `max_diff = 0.00e+00`；NaN 传播 `nan_ok=True`（586 点 ×3 形态）。
- 端到端：接入 wrapper 后 `tileops.reporting.cli run --op ClampScalarFwdOp` 正确性 **11/11**、`status=passed`。

## 5. 时间与成本

| 项 | 值 | 来源 |
|---|---|---|
| 开始（UTC） | 2026-10-15T11:37:04Z | `reports/optimize/ClampScalarFwdOp/.start_utc` |
| 结束（UTC） | 2026-10-15T12:50:54Z | `date -u` |
| 墙钟时长 | **≈ 73.8 min** | 相减 |
| token input | 23,391 | `~/.local/share/opencode/opencode.db`（`message.data.tokens`，窗口内求和） |
| token output | 7,638 | 同上 |
| token reasoning | 20,067 | 同上 |
| token 合计 | **51,096** | input+output+reasoning |
| cost | $0.0510 | 同上 |

> 取数窗口 = 开始时刻→结束时刻；`time_created` 为 epoch ms。token 数为该窗口内 `message` 表求和，
> 受 db 记录粒度影响，供量级参考。

## 6. 产物

| 类别 | 路径 / 状态 |
|---|---|
| final kernel | `tileops/kernels/elementwise/clamp_scalar/clamp_scalar_kernel/perf_opt/clamp_scalar_fwd_kernel.py` |
| 调优记录 | `tileops/kernels/elementwise/clamp_scalar/clamp_scalar_kernel/perf_opt/opt_log.md` |
| 结构化记录 | `tileops/kernels/elementwise/clamp_scalar/clamp_scalar_kernel/perf_opt/perf_records.jsonl` |
| baseline 报告 | `reports/tileops/20261015_195138_601192_ClampScalarFwdOp/` |
| perf_opt 报告 | `reports/tileops/20261015_202825_275014_ClampScalarFwdOp/`（status=passed） |
| gate 4 | 原文未记录 gate 编号/结论（原文仅记录 `status=passed` 与正确性 11/11） |
| 算子状态 | 原文未记录（`clamp_scalar_kernel/` 无 `.stage_state.json`） |
| wrapper 切换 | 已切：`clamp_scalar.py` 的 baseline/perf_opt import 块已翻至 perf_opt |
| pattern-library 回写 | 原文建议作为候选回写（kind: pattern，family: elementwise，token 待 distill）；实际写入状态原文未记录 |

## 7. 天花板 / 豁免 / 缺口

- **豁免项：无**。原文明记「本算子（ClampScalar/ClampMin）无微型 shape，故本报告不涉及豁免项」。
- **族内微型 shape 豁免政策（记录项，不适用于本算子）**：用户已同意微型 shape 豁免 60% 目标并标注
  「物理不可达」。判据（以 Hardsigmoid `mbv3-se-gate`、shape `[32,240,1,1]` fp16 为例）：
    - 数据量 ≈ 30 KB；按设备地板 1.26 TB/s 理论搬运 ≈ **24 ns**；实测 kernel ≈ **2.86 µs**；
    - kernel 有与数据量无关的固定成本（派发/调度 + 流水填充排空 + 收尾同步）≈ 2 µs 级；
    - 即便搬运瞬间完成，利用率上限 ≈ `30KB/(2µs×1.75TB/s) ≈ 0.86%`。
  ⇒ 此类 shape **60% 在数学上不可达**，正确指标是绝对延迟（2–4 µs），不是带宽利用率。
- **贴线风险**：原文按 60% 口径判定无贴线项（最低 71.9%）。若按族内 70% 访存线复核，
  `256M bf16` 71.9% 为最近的一项（距 70% 线 1.9pp）——该复核口径原文未记录。
- **ceiling 证据**：原文未记录 roofline advice / pipe ratio 上限分析；本算子未声明结构上限。
- **能力缺口**：原文未记录 CG 编号；NaN 传播结论依赖本工具链（CANN 上一代工具链）实测，跨版本需复验。

## 8. 建议

1. **采纳** `perf_opt/clamp_scalar_fwd_kernel.py`（wrapper 已切换，端到端 report `passed`、正确性 11/11）。
2. `ClampMinFwdOp` 复用同 kernel，**同步受益**（原文）——两算子应一起复核。
3. 该优化点（去除被证伪的冗余 NaN 恢复）机制清晰，原文建议作为 pattern-library 候选回写：
   纯 Vector 逐元素算子若 NaN 已由 `vmax`/`vmin`/`vclamp` 原生传播，则 `vcmp + vselect` 恢复是纯开销。
4. 跨 CANN 版本复核：本报告为 CANN 上一代工具链 口径，升级后需先复验 NaN 传播再沿用结论。
5. **复核入口**（原文）：

```bash
cd examples/TileOPs
# 正确性
python -m pytest tests/ops/test_clamp_scalar.py tests/ops/test_clamp_min.py -q
# 分层回归（perf_opt）
python tileops/kernels/elementwise/clamp_scalar/clamp_scalar_kernel/perf_opt/clamp_scalar_fwd_kernel.py --level all
# 端到端报告
python -m tileops.reporting.cli run --op ClampScalarFwdOp --prof-mode msprof
```

---

## 9. R2（2026-10-21）：9.0 结构重试 —— 原生 dtype clamp + A4

> 本轮把 9.0 的 58.21–70.16%（1/11）拉回 **69.62–80.53%（10/11）**；ClampMin 同 kernel **7/7**。
> 口径：CANN 9.0.0 / Ascend910B3 / 2026-10-21；数据 `perf_opt/REPORT_CLAMP_R2_DATA.md`、`perf_records.jsonl`。

### 9.1 诊断：9.0 回退的根因不是“环境整体下移”

R1 的 3-op 链在 9.0 上仍是：

```
T.copy(GM→UB) → vcast(f16→f32) → vclamp/vmax/vmin(fp32) → vcast(f32→f16) → T.copy(UB→GM)
                 \__ 2 条无信息类型搬运 __/
```

- pipe：`vec_ratio 0.72 ≫ mte2_ratio 0.56`，`mte2_active_bw≈39 GB/s/core` ⇒ **瓶颈是向量 pass 数，不是带宽**。
- 语义：clamp 输出只可能是输入值或标量边界，fp32 opmath 不产生新中间值；舍入单调 ⇒
  `rn(clamp(x,lo,hi)) == clamp(x,rn(lo),rn(hi))`。**两条 vcast 是纯开销**（R1 只删了 NaN 恢复的 2 条）。
- 结构：R1 每 tile 重算 `T.min(block_size, shape-offset)` 与动态 copy extent，流水拿不到静态长度。

### 9.2 方法（3 个单点改动）

1. **N1 原生 dtype clamp**：fp16/fp32 直接 `vmax/vmin`，删 2 条 vcast；边界按 dtype 网格预舍入（round-half-even）。
   bf16 无原生向量算术，保留 fp32 中转（文档 `T.vclamp.md` §2.2.1 明示 bf16 ×）。
2. **N2 A4 结构**：满块静态 extent + 单尾块 epilogue（core 0）+ `T.Pipelined(2)`。
3. **N3**：显式 `vmax→vmin`（只出现 `dst==src1`，避免 `dst==src2` 的 alias 税）。

**被否决**：`block_size=16384`、`num_stages=3`（均 `config_no_gain`）；
同 dtype `vcast`（fp32→fp32）在 9.0 **不报错但结果错**，直通必须 `T.copy`（已作为陷阱记录）。

### 9.3 结果（端到端，交付口径）

| 维度 | R1（9.0） | **R2（9.0）** |
|---|---:|---:|
| ratio 区间 | 58.21–70.16% | **69.62–80.53%** |
| 达标（70%） | 1/11 | **10/11** |
| kernel Task Duration | — | **18/18 变短，均值 −27.0%**（最大 −40.7%） |

逐 workload 见 `perf_opt/REPORT_CLAMP_R2_DATA.md` §3；唯一未达 `16M-min-only bf16` 69.62%
（同计算姊妹 75.02%，时延稳定，判为 ratio 估计器随会话摆动，未做 shape 特判）。

### 9.4 精度与产物

- `perf_opt/ClampScalarFwdOp.py --level all` → **All check passed**；586 点 NaN 探针 `max_diff=0`（逐位）。
- 端到端：ClampScalar **11/11 passed**（run `20261021_172100`），ClampMin **14/14 passed**（run `20261021_173421`）。
- 产物：`perf_opt/ClampScalarFwdOp.py`（sha256 `760579b0…`）+ 转发层 `perf_opt/clamp_scalar_fwd_kernel.py`；
  wrapper import 未改。gate 4 **pass**。

### 9.5 结论

9.0 的 clamp 回退**可以通过实现级修复拿回**：原生 dtype 去掉 2 条无信息 vcast + A4 静态化，
即从 1/11 提升到 10/11（ClampMin 0/7 → 7/7）。R1 报告中“环境整体下移、不重优化”的结论**已被本轮推翻**。
