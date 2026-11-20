# 算子性能分析方法论（判定优化状态的标准流程）

> **本文档解决什么问题**：给出「这个算子在当前机器上优化到什么程度了 / 还要不要继续投入」的
> 可执行判定流程。
>
> **与其它文档的分工**：
>
> | 文档 | 职责 |
> |---|---|
> | [roofline_metrics_analysis.md](roofline_metrics_analysis.md) | Ratio / AI / Perf / Computility **怎么算**（含 msopprof 源码级依据） |
> | [roofline_types_and_latency_analysis.md](roofline_types_and_latency_analysis.md) | bin 里有哪几种 roofline、advice 怎么用 |
> | **本文档** | **怎么拿这些指标判「到位没到位」**：天花板标定、分档、判据、采数纪律、陷阱 |
>
> 全文数据均实测于 **Ascend910B3（20 AIC / 40 AIV，1800 MHz，CANN 9.0.0，Developer 模式）**。

---

## 0. 一页速查

```
采数（逐 shape） → 三个数 → 两条规则 → 结论
```

**三个数**：`Task Duration`、`Roofline Ratio`、`主导流水线占比`
**两条规则**：逐 shape 判"瓶颈是否打满"（SATURATED / HEADROOM）；整体判"平均 Ratio ≥ 60%"

一条命令出结果：

```bash
cd examples/TileOPs
PYTHONPATH=$(git rev-parse --show-toplevel):$PWD \
  python3 tools/op_status.py --opprof ~/op_profiles/<op>/<shape>/OPPROF_xxx
```

---

## 1. Ratio 的准确口径（引用前必须确认三件事）

```
Ratio = Perf ÷ min(AI × bw, Computility)

  AI          = total_fops ÷ total_bytes      # total_bytes = L2 read miss × 128B，只算读
  Perf        = total_fops ÷ Task Duration
  bw          = msprof SoC 查表的标称带宽      # 910B3 = 1.75 TB/s；910B2C = 1.8 TB/s
  Computility = 向量/Cube 峰值算力             # 910B3 向量 ≈ 4.6 TOps/s（64 ppc × 1.8 GHz × 40）
```

化简后分两个分支，**含义完全不同**：

| 分支 | 判据 | Ratio 实际含义 | 实用上限 |
|---|---|---|---|
| **带宽受限** | `AI × bw < Computility` | `实际带宽 ÷ bw` | 见 §3 分档表（73~85%） |
| **算力受限** | `AI × bw ≥ Computility` | **向量流水利用率** | ~85~90%（流水不可能 100% 占满） |

⚠️ **引用注意**：

1. `bw` 随 SoC 变。910B2C 是 1.8，910B3 是 1.75，写文档/对比时必须带设备型号。
2. 报告里的 `Bandwidth(TB/s)` 列**不是实测带宽**，就是这个标称 `bw`。
3. 报告每次跑两遍（算子 + `torch` 基线），**求平均只能取算子侧的行**。Erf 那次 20 行日志，
   全取平均得 55.58%，只取算子侧 10 行得 59.02% —— 差 3.4pt，正好跨过 60% 线。

### 1.1 口径决定（2026-11-09 定案）

**报告 Ratio 一律用 `1.75` 当分母。** 机器可读版本见
`tools/calibration_910B3.json` 的 `caliber_notes`。

| 候选值 | 结论 | 理由 |
|---|---|---|
| **1.75** | ✅ **采用** | msprof 在 910B3 上实际报的 `bw`（从 `visualize_data.bin` 读回），任何人可复现；仓库 82 份历史报告都是这个分母，可与历史比 |
| 1.64 | ❌ 不采用 | 只出现在一份**未提交的孤儿文档**（`docs/optimize_results_20261030.md`），声称 `DEVICE_GM_BW_TBPS = 1.64`、`SOC_CONFIG["Ascend910B3"]`。实测 HEAD 与工作区**都没有**这段代码，`CONST-ascend910-9362-profile` 也未在 pattern-library 登记 |
| 1.8 | ❌ 不采用 | 这是 **910B2C** 的值（见 `roofline_metrics_analysis.md`），设备不对 |

**若将来要切到 1.64，必须先完成三件事**，否则不可执行：

1. 把 `DEVICE_GM_BW_TBPS` / `SOC_CONFIG["Ascend910B3"]` 真正写入
   `benchmark_base.py` 与 `msprof_roofline.py` 并提交；
2. 用新口径**重跑历史报告**（否则 82 份存量数据与新报告不可比）；
3. 把 `CONST-ascend910-9362-profile` 登记进 pattern-library。

**切换的代价**：`Ratio` 整体放大 `1.75 ÷ 1.64 = 1.067`（+6.7%），会翻转临界结论 ——
`ErfFwdOp` 从 59.02% 变成 63.0%（由不达标变达标）。**一次报告内绝不混用两种口径。**

> ⚠️ 但注意：1.75 与 1.64 都只是**标称**分母，都不能回答"还剩多少空间"。
> 判断 head-room 永远用本文件 §3 的**实测可达值**（带宽 1.062~1.490 TB/s、算力 2.97/3.494 TOp/s）。

### 1.2 平均口径（2026-11-09 定案）

一个算子只报一个通过/不通过数字时，按下面三条执行（机器可读版在
`tools/calibration_910B3.json` 的 `caliber_notes.averaging_rule`）：

**① 排除 `smoke` 用例**

`smoke-1m` 系列带约 **3.75 µs 固定启动开销**（占其总时延 36%，实测见 §4.1），
会把每个算子的均值压低 **8~12pt**。仓库自己的调优流程本来就排除它们：

> `perf_opt/opt_log.md`：smoke 排除出性能调优，仅作精度回归。

实测影响：含 smoke 时 20 个算子只有 **5** 个达标，排除后为 **8** 个。

**② 同一算子存在多份数据时，取较高值**

团队报告与本机实测可能因**用例集不同**（如 `Expm1FwdOp` 团队报告 6 个用例、当前
manifest 10 个）而给出不同均值。此时逐算子取较高者，并在表格里注明取自哪一侧
及报告日期，保证可追溯。

**③ `smoke` 的表现必须单列**

排除 ≠ 隐藏。它们代表真实的小 shape 场景，报告里要保留一列观测值，只是不参与评分。

---

## 2. 为什么不能用 `bw` 当分母算"还剩多少空间"

`bw` 是不可达的标称值。本机实测：任何 kernel、任何 shape、任何优化版本，
**Ratio 从未超过 85.3%**。用 `bw` 当分母会得出"还有 25% 空间"的错误结论。

正确做法是**先在目标 shape 上标定可达带宽**，再算达成率：

```
达成率 = 实测 Ratio ÷ 该 shape 档位的 Ratio 天花板
```

---

## 3. 天花板标定

### 3.1 本机实测天花板（直接可用）

用 3 读 1 写 + 1 个 vadd 的探针测得，每档 8 轮取中位：

| 元素数 | 工作集 | 可达带宽 | **Ratio 天花板** |
|---|---:|---:|---:|
| 1M | 12 MB | 1.062 TB/s | **60.7%** |
| 8M | 96 MB | 1.442 TB/s | **82.4%** |
| 16M | 192 MB | 1.490 TB/s | **85.3%** |
| 64M | 768 MB | ~1.37 TB/s（时长反推） | **~78%** |

> ⚠️ 64M 那档的 bin 数据**不可用**：`L2Cache.csv` 报「768 MB 流量只有 0.5 MB DRAM 读」，
> 导致 `Ratio` 算成 0.08%。该行的 1.37 TB/s 是用 `768 MB ÷ 558.6 µs` 从时长反推的。
> 遇到这种量级明显异常的数据，一律用 `声明流量 ÷ 时长` 交叉验证后再采信（见 §7 陷阱 7）。

**天花板呈倒 U 型**：小 shape 被启动开销压住，大 shape 被 MTE2 退化压住，中间最高。

### 3.2 标定探针的写法

必须有**至少 1 个向量 op**，结构上与目标算子同流量比：

```python
@T.prim_func
def main(A: T.Tensor((N), dtype), B: T.Tensor((N), dtype), C: T.Tensor((N), dtype)):
    with T.Kernel(T.ceildiv(N, BS), is_npu=True) as (cid, _):
        a = T.alloc_ub((BS), dtype); b = T.alloc_ub((BS), dtype); c = T.alloc_ub((BS), dtype)
        s = cid * BS
        T.copy(A[s:s+BS], a); T.copy(B[s:s+BS], b)
        T.vadd(a, b, c)
        T.copy(c, C[s:s+BS])
```

```bash
msprof op --aic-metrics=Roofline --kernel-name=main \
  --output=$HOME/ceil/<N> --launch-count=8 --warm-up=4 python3 probe.py <N>
```

### 3.3 ❌ 纯 copy 探针不能用

只做搬运、没有向量 op 时 `fops ≈ 0` → `AI ≈ 0` → `Ratio = Perf ÷ (AI×bw)` 分母坍塌，
实测会得出 33.56% / 43.71% 这种**退化值**，不是天花板。

（纯 copy 仍可用于**标定可达带宽**，但不能用来标定 Ratio 上限。）

---

## 4. 启动开销与 shape 分档

### 4.1 实测启动开销 ≈ 3.75 µs

用 1M 实测时延减去 16M 的 1/16 等比值求得（Erf，三种 dtype 高度一致）：

| dtype | 1M 实测 | 16M ÷ 16 | 固定开销 | 占比 |
|---|---:|---:|---:|---:|
| fp32 | 10.08 µs | 6.36 µs | 3.72 µs | 36.9% |
| fp16 | 10.48 µs | 6.72 µs | 3.76 µs | 35.9% |
| bf16 | 10.68 µs | 6.84 µs | 3.83 µs | 35.9% |

这部分是 kernel 前导 + wave 量化的固定成本，**不随数据量缩小，也不该指望通过优化消掉**。
它把 1M 的 Ratio 从"本可以到的 ~65%"压到 ~41%。

### 4.2 排除规则（可机械执行）

```python
固定开销占比 = (t_small - t_large × N_small/N_large) / t_small

> 30%  → 排除出「达标判定」，但必须保留在「观测报告」里
≤ 30%  → 参与达标判定
```

**三个前置条件**，否则就变成挑数据：

1. 排除依据是**固定开销占比**，不是"这个用例分数低"
2. 排除的用例**必须单独报告**，不能从报告里消失
3. 规则**写死、机械可判**，不能每次人工判断

### 4.3 这条规则救不了 decode 类算子

`DSA-Decode`（0.35%）、`MHA-DecodePaged`（0.01%）的**所有**用例本身都是小 batch
（`single-batch-mainstream`、`long-cache` 全是同一量级）。排除 smoke 前后平均不变。

原因在分母：`Computility` 按 40 核满负荷计算，而 batch=1 的 decode 根本铺不满核，
Ratio 上限在结构上就只有几个百分点。**这类算子需要换指标**（如 token 时延、
或对比 AscendC 参考实现），不是继续压 Ratio。

---

## 5. 判定的两条轨道

### 5.1 逐 shape 轨：算子内相对判定（不需要外部标定）

这是日常最常用的，**全部是算子内相对量**，所以可以批量跑：

```
主导引擎 = argmax(mte2_ratio, vec_ratio, scalar_ratio)      # AIV 时间占比
核间极差 = (max_core_time - min_main_core_time) / mean_main_core_time
           （min 只统计 ≥ 峰值一半的核，剔除尾部/残缺记录）
```

| 判定 | 条件 | 含义 | 动作 |
|---|---|---|---|
| **SATURATED** | 主导引擎 ≥ 0.85 且 核间极差 ≤ 30% | 瓶颈引擎已饱和 | **停止优化** |
| **HEADROOM** | 所有引擎 < 0.65 | 没有引擎饱和，时间花在别处 | **查启动/标量/wave** |
| **CHECK** | 0.65 ~ 0.85 | 边界 | 人工看 |

实测样例（`BitwiseNotFwdOp` int32）：

| shape | mte2 | vec | scalar | 核间极差 | 判定 |
|---|---:|---:|---:|---:|---|
| 1M | 0.54 | 0.08 | **0.32** | 53% | **HEADROOM**（启动/标量主导） |
| 16M | **0.91** | 0.10 | 0.04 | 17% | **SATURATED** |
| 64M | **0.97** | 0.05 | 0.01 | 4% | **SATURATED** |

**同一算子三个 shape、两种结论** —— 这就是为什么必须逐 shape 看，不能只看一个。

### 5.2 整体轨：绝对达标判定

| 项 | 规则 |
|---|---|
| 用例范围 | manifest 登记的全部用例，**不可挑选** |
| 数据取侧 | **只取算子侧行**，排除 `torch` 基线段 |
| 聚合方式 | **算术平均**（对齐"平均 Roofline Ratio"的原话） |
| 达标线 | **算术平均 ≥ 60%** |
| 必带项 | 同时报**最差用例**，防止均值掩盖短板 |

`60%` 对带宽受限算子意味着"达到该档位天花板的 70%"，是合理门槛；
对 1M 档则是"99% 天花板"，所以 1M 档应单独定线（约 45%）或按 §4.2 排除。

---

## 6. 采数纪律（不遵守则数据作废）

| # | 纪律 | 不遵守的后果 |
|---|---|---|
| 1 | **必须 `--kernel-name=<kernel>`** | 全捕获会把 `ZerosLike` 等辅助 op 当目标，实测出现 **0.01%~0.65%** 的伪影量级 |
| 2 | **多轮取中位，噪声阈值 3%** | 实测 6 轮变异系数 1.13%、极差 2.95%；<3% 的"提升"不可信，必须交错 A/B |
| 3 | **采集前充分 warmup** | 空闲时 `Aicore curFreq` 降到 **800 MHz**，冷启动会测到低频数据 |
| 4 | **缓存隔离** | `TILELANG_CACHE_DIR` 默认全局共享，两个工作区会串味（改了不生效 / 测到旧 kernel） |
| 5 | **模式固定** | `TILELANG_ASCEND_MODE=Developer`；Expert/Developer 影响可用 API 与生成代码 |
| 6 | **区分 op 侧与 torch 基线段** | 见 §1 引用注意第 3 条，混算会差 3~4pt |

---

## 7. 陷阱清单（全部有实测证据）

| # | 陷阱 | 证据 |
|---|---|---|
| 1 | **纯 copy 探针标定 Ratio 上限无效** | `fops≈0` 导致分母坍塌，实测 33.56%/43.71% 为退化值 |
| 2 | **Ratio 跨环境会漂移** | 同一用例同一时延（101.8 vs 103.2 µs），历史 64.55% / 本次 39.02%，差 25.5pt —— 因 `fops` 计数不同。**Ratio 只适合同环境内前后比** |
| 3 | **历史报告用例集与当前 manifest 不同** | Erf 历史 46 用例、当前 10 用例；BitwiseNot 历史 46、当前 6。不可直接比 |
| 4 | **`0.01%` 量级基本都是采集伪影** | `GQA-SlidingWindowVarlen` 历史 0.01%，重测实为 **14.12%** |
| 5 | **单点比全集平均高 5~13pt** | LerpTensor 单点 81.52% → 全集 68.66%；Clamp 77.74% → 70.72%；Elu 65.15% → 54.61% |
| 6 | **小 shape 受 L2 驻留影响虚高** | 8M fp32 工作集 96 MB < L2 192 MB，重放后部分命中，实测带宽被视为 1.44 TB/s |
| 7 | **大 shape 的 `L2Cache.csv` 计数可能失效** | 64M 实测报"768 MB 流量只有 0.5 MB DRAM 读"，Ratio 0.08% 为假数据，需用 `时长×流量` 交叉验证 |
| 8 | **`ratio` 字段为 `None` 不等于缺数据** | JSON 只存 latency/带宽；Ratio 在 `.log` 的 markdown 表里由 bin 解析得到 |

---

## 8. 工具

### 8.1 `tools/op_status.py`

```bash
# 完整分析（有 OPPROF 目录：时延+Ratio+流水线+核间均衡+判定）
python3 tools/op_status.py --opprof ~/op_profiles/<op>/<shape>/OPPROF_xxx --label <op>/<shape>

# 只有 profile_run 日志（时延 + Ratio）
python3 tools/op_status.py --log profile_run_msprof_<ts>.log --label <op>
```

输出示例：

```
=== BitwiseNotFwdOp/16m ===
  latency       : 48.02 us   block_dim 40   freq 1800
  roofline Ratio: 79.86 %   AI 0.1251   binding bandwidth
  pipelines     : mte2 0.91  vec 0.10  scalar 0.04
  core balance  : 39/40 main cores  39.8..47.4 us  spread 17%
  DRAM reads    : 524289 cachelines
  VERDICT       : SATURATED
  reason        : mte2 engine pegged at 0.91; little structural head-room
```

### 8.2 ⚠️ 留存 OPPROF 目录

`msprof op` 不指定 `--output` 时数据落在 `/tmp` 并被清理，之后就**只剩 Ratio、拿不到流水线归因**。
批量采集时务必显式落盘：

```bash
msprof op --output=$HOME/op_profiles/<op>/<shape> ...
```

---

## 9. 标准工作流（从零分析一个算子）

```
1. 定口径
   export TILELANG_ASCEND_MODE=Developer
   export TILELANG_CACHE_DIR=~/.tilelang/cache-<your-workspace>
   msprof --version  &&  npu-smi info          # 记录设备/工具链，写进报告

2. 采数（逐 shape，落盘）
   pytest benchmarks/ops/bench_<op>.py -q      # 官方口径
   确认 OPPROF 目录被保留

3. 逐 shape 出事实
   python3 tools/op_status.py --opprof <dir> --label <op>/<shape>
   记录：时延 / Ratio / 绑定项 / 主导引擎 / 核间极差

4. 标定该 shape 天花板（若尚无基准）
   用 §3.2 的探针，同流量比、带 1 个向量 op

5. 逐 shape 判
   达成率 = Ratio ÷ 天花板
   SATURATED 或 达成率 ≥ 90%  → 该 shape 到位
   HEADROOM                   → 按主导引擎归因后继续

6. 整体判
   算术平均（只取算子侧）≥ 60% ？
   最差用例的达成率？有空间的 shape 有几个、瓶颈是否同一类？

7. 收敛确认
   读 perf_opt/opt_log.md 的 stop_reason / [DESIGN_LIMIT]
   success + 无剩余结构差异 → 停止；否则按 opt_log 记录的否决清单继续
```

---

## 10. 附录：本机基准（可复现）

| 项 | 值 | 来源 |
|---|---|---|
| 设备 | Ascend910B3，20 AIC / 40 AIV，1800 MHz，L2 192 MB，HBM 64 GB | `torch.npu.get_device_properties` |
| 工具链 | CANN 9.0.0（`prof_version 2.0`），torch 2.7.1+cpu，torch_npu 2.7.1.post4 | `npu-smi` / `version.info` |
| `bw`（msprof 报） | **1.75 TB/s**（GM Read+Write），7.78 TB/s（L2 Read+Write） | bin 的 `bw` 字段 |
| `Computility` | 向量 ≈ 4.6 TOps/s（64 ppc × 1.8 GHz × 40 核，fp32 口径） | bin 的 `computility` 字段 |
| 天花板 | 1M 60.7% / 8M 82.4% / 16M 85.3% / 64M ~78% | §3.1 探针 |
| 启动开销 | ≈ 3.75 µs（占 1M 档 36%） | §4.1 |
| 测量噪声 | 8M 探针 6 轮：变异系数 **1.13%**、极差 2.95%；16M 探针 8 轮：变异系数 **0.44%**、极差 1.33% | 实测两处不同 probe |
