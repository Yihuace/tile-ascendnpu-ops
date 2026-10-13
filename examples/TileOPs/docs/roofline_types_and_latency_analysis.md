# MindStudio Insight Roofline 多类型分析与 Latency Bound 判定

## 一、bin 中多种 Roofline 的分类与使用场景

### 硬件存储层次

Ascend 910B 的存储层次是**多级金字塔结构**，数据从 GM 到达计算单元需要经过多级搬运。每个 roofline 条目回答的是**"算子在某一级存储/通路/流水线上是否是带宽瓶颈"**。

```
GM (HBM, 1.8 TB/s)
  └── L2 Cache (8.0 TB/s)
       ├── Cube 通路:
       │    L1 (20.1 TB/s) → L0A (10.1 TB/s) / L0B (5.0 TB/s) → Cube单元 → L0C (5.0 TB/s) → L1
       └── Vector 通路:
            UB (40.2 TB/s) → Vector单元 → UB
```

数据每经过一级，带宽峰值递增、延迟递减。不同级别的 total_bytes 不同（越靠近计算单元，复用越多，bytes 越大）。

### 三个 Tab 的视角差异

| Tab | 视角 | 回答的问题 | bw 含义 | total_bytes 含义 |
|-----|------|-----------|---------|-----------------|
| **Memory Unit** | 按存储硬件单元 | "L1/UB/L0C 这一级存储的带宽是否是瓶颈？" | 该存储单元的读写峰值带宽 | 该存储单元的实际访问字节数 |
| **Memory Transfer** | 按数据搬运路径 | "GM→L0A 这条搬运路径的带宽是否是瓶颈？" | 该路径的峰值带宽 | 该路径上实际搬运的字节数 |
| **Pipeline** | 按流水线阶段 | "MTE2/MTE3 这个搬运引擎的带宽是否是瓶颈？" | 该流水线引擎的峰值带宽 | 该引擎实际处理的字节数 |

### Mix Kernel 数据实例解读

#### 1. GM/L2（最顶层，全局视角）

| 通路 | BW | AI | Perf | Comp | Ratio | 瓶颈类型 |
|------|-----|-----|------|------|-------|---------|
| **GM Read + Write** | 1.8 | 2798.6 | 89.8 | 359.4 | 25.0% | **算力瓶颈**（AI×BW=5037 >> Comp=359） |
| **L2 Read + Write** | 8.0 | 30.1 | 89.8 | 359.4 | 37.3% | **算力瓶颈**（AI×BW=241 < Comp=359，L2带宽瓶颈） |

GM/L2 条目是 Cube+Vector 的合并视角，用于判断**整个算子是算力瓶颈还是内存瓶颈**。

- GM Read + Write：ratio=25%，`AI×BW=5037 >> Comp=359`，说明算子到达了算力瓶颈区，GM 带宽不是限制因素
- L2 Read + Write：ratio=37.3%，`AI×BW=241 < Comp=359`，L2 带宽是更紧的约束，说明 L2 Cache 命中率不够高，导致 L2 带宽成为瓶颈

**使用场景**：快速判断算子整体是 compute-bound 还是 memory-bound。

#### 2. Memory Unit(Cube)（Cube 通路各级存储）

| 存储单元 | BW | AI | Ratio | 解读 |
|---------|-----|-----|-------|------|
| L1 Read + Write | 20.1 | 30.8 | 25.0% | L1 带宽充足 |
| Read from L1 | 10.1 | 61.4 | 25.0% | L1 读带宽充足 |
| Write to L0A | 10.1 | 170.7 | 25.0% | L0A 写带宽非常充足 |
| Write to L0B | 5.0 | 96.0 | 25.0% | L0B 写带宽充足 |
| Read from L0C | 5.0 | 172.0 | 25.0% | L0C 读带宽充足 |

所有 Cube 存储单元的 ratio 都是 25%（=Cube Perf/Cube Comp），说明**Cube 计算单元的算力是瓶颈，而不是任何一级 Cube 存储的带宽**。

**使用场景**：当 Cube 算子的 ratio 在某一级存储突然下降（低于 compute ratio），说明那一级存储的带宽不足。例如如果 L1 Read + Write 的 ratio 是 15% 而 compute ratio 是 25%，说明 L1 带宽是瓶颈。

#### 3. Memory Unit(Vector)（Vector 通路各级存储）

| 存储单元 | BW | AI | Ratio | 解读 |
|---------|-----|-----|-------|------|
| UB Read + Write | 40.2 | 0.098 | **37.9%** | UB 带宽最紧的约束 |
| Vector Read UB | 40.2 | 0.187 | 27.4% | UB 读带宽 |
| Write to UB | 10.1 | 2.92 | 27.4% | UB 写带宽 |
| Read from UB | 10.1 | 4.25 | 27.4% | UB 读带宽（单口） |

注意 **UB Read + Write 的 ratio=37.9%** 高于 Vector 的 compute ratio=27.4%，说明 **Vector 算子受到了 UB 读写带宽的限制**。这是 UB 总带宽（40.2 TB/s）不够导致的。

**使用场景**：当 Vector 算子性能不达标时，检查 UB 各级 ratio，如果某级 ratio 明显高于 compute ratio，说明该级存储是瓶颈。

#### 4. Memory Transfer（数据搬运路径）

| 搬运路径 | BW | 解读 |
|---------|-----|------|
| GM/L1 to L0A | 10.3 | GM/L1 到 Cube 输入 A 的搬运带宽 |
| GM/L1 to L0B | 4.9 | GM/L1 到 Cube 输入 B 的搬运带宽 |
| L0C to GM | 4.9 | Cube 输出到 GM 的搬运带宽 |
| L1 to GM | 4.7 | L1 到 GM 的回写带宽 |
| GM to UB | 5.2 | GM 到 UB 的搬运带宽 |
| UB to GM | 4.4 | UB 到 GM 的回写带宽 |

**使用场景**：当发现某级存储是瓶颈后，进一步定位是**读路径还是写路径**的问题。例如如果 "GM to UB" 的 ratio 低但 "UB to GM" 的 ratio 高，说明回写带宽是瓶颈，应优化输出数据布局。

#### 5. Pipeline（流水线引擎）

| 引擎 | BW | 解读 |
|-----|-----|------|
| MTE1 | 6.8 | Cube 的 MTE1 引擎（L1→L0A/L0B 搬运） |
| MTE2 | 8.0 | Cube 的 MTE2 引擎（GM→L1 搬运） |
| MTE3 | 4.7 | Cube 的 MTE3 引擎（L0C→L1/GM 回写） |
| FIXP | 4.9 | Cube 的 FIXP 引擎（定点后处理） |
| MTE2 vector | 4.4 | Vector 的 MTE2 引擎（GM→UB 搬运） |
| MTE3 vector | 5.2 | Vector 的 MTE3 引擎（UB→GM 回写） |

注意 **Pipeline(Cube) MTE2 的 ratio=33.8%**，明显高于其他 Cube 条目的 25%，说明 **MTE2 引擎（GM→L1 搬运）是 Cube 通路中相对最紧的瓶颈**。

**使用场景**：最细粒度的瓶颈定位。当 Memory Unit 或 Memory Transfer 发现问题时，Pipeline tab 可以精确到是哪个搬运引擎（MTE1/MTE2/MTE3/FIXP）的问题。

### 使用决策树

```
Step 1: 看 GM/L2 tab
  ├── ratio 低 + AI×BW < Comp → 内存瓶颈 → Step 2
  └── ratio 低 + AI×BW > Comp → 算力瓶颈 → 优化计算逻辑/减少指令数

Step 2: 看 Memory Unit tab
  ├── 哪一级 ratio 最高(最紧) → 该级存储是瓶颈 → Step 3
  └── 所有级 ratio 相近 → 最外层(GM/L2)是瓶颈

Step 3: 看 Memory Transfer tab
  ├── 哪条路径 ratio 最高 → 该搬运路径是瓶颈 → Step 4
  └── 所有路径 ratio 相近 → 该级存储整体带宽不足

Step 4: 看 Pipeline tab
  └── 哪个引擎 ratio 最高 → 该搬运引擎是瓶颈
      → 针对性优化: 调整 tiling/数据布局/L2 cache命中率
```

### Mix Kernel 诊断结论

```
GM/L2:  ratio=25%  → 算力瓶颈区（AI×BW >> Comp）
  ├── Cube:  所有存储 ratio=25%, 仅 MTE2=33.8%  → MTE2引擎(GM→L1搬运)是Cube次要瓶颈
  └── Vector: compute ratio=27.4%, 但 UB Read+Write=37.9%  → UB读写带宽是Vector瓶颈
```

**优化建议**：
- Cube 侧：MTE2 引擎（GM→L1）负载最重，可考虑提升 L2 Cache 命中率减少 GM 读取
- Vector 侧：UB 读写带宽是瓶颈，可考虑增大 tiling 减少 UB 访问次数，或减少 Vector 计算量

---

## 二、"latency bound:compute caused" 的分析逻辑

### advice 的来源

这个提示是 **msopprof 工具**在生成 `visualize_data.bin` 时计算并写入 JSON 的 `advice` 字段，MindStudio Insight 只是原样展示。

源码位置：`modules/compute/src/components/detail/Roofline/Index.tsx:211`

```tsx
{data?.advice?.length > 0 && <Hit text={data.advice} type={'alarm'} data-testId={'rooflineAdvice'}/>}
```

bin 文件中的 JSON 结构：
```json
{
  "advice": "latency bound:compute caused",
  "multiple_rooflines": [...]
}
```

### msopprof 中的 5 种 advice 类型

从 `msopprof` 二进制 strings 提取：

```
"compute bound"
"memory bound"
"latency bound:compute caused"
"latency bound:memory caused"
"latency bound:pipeline caused"
```

以及 2 个阈值常量：

```
"compute usage lower than 20%"
"bandwidth utilization lower than 80% when active"
```

### 判定流程（三步分类法）

#### Step 1: 判断是否 latency bound

检查**所有** roofline 通路的 ratio 是否都低于某个阈值（推测为 0.8 或 1.0）：

```
if max(all_ratios) < threshold → latency bound (所有通路都没跑满)
else if 某条内存通路 ratio ≈ 1.0 → memory bound
else if compute ratio ≈ 1.0    → compute bound
```

Mix kernel 的所有 25 条 roofline 的 ratio 最大值仅 37.9%，远低于阈值 → **latency bound**。

#### Step 2: 判断 latency 的根因（compute / pipeline / memory）

对比 **计算单元利用率** 与 **搬运引擎利用率**（来自 `PipeUtilization.csv`）：

| 判定条件 | advice | 含义 |
|---------|--------|------|
| `compute_ratio < mte_ratio` | `compute caused` | 计算单元利用率最低，是短板 |
| `compute_ratio > mte_ratio` | `pipeline caused` | 搬运引擎利用率最低，数据搬运是短板 |
| 内存带宽接近饱和 | `memory caused` | 内存带宽不足导致延迟 |

#### Step 3: 额外阈值检查

- `compute_ratio < 20%` → 触发 `"compute usage lower than 20%"` 警告
- 带宽利用率 `< 80% when active` → 触发带宽警告

### Mix Kernel 具体数据

```
所有 roofline ratio < 38%  → latency bound ✓

Cube 通路（dominant）:
  compute_ratio = 14.7%    ← 最低！
  mte2_ratio    = 35.2%    ← 搬运引擎更忙
  14.7% < 35.2% → compute 是瓶颈单元 → "compute caused" ✓
  14.7% < 20%   → 触发 "compute usage lower than 20%" 警告

Vector 通路:
  compute_ratio = 31.1%
  mte2_ratio    = 11.7%
  31.1% > 11.7% → vector 侧 pipeline 是瓶颈
```

由于 Cube 的计算量占主导（88.3 vs 1.5 TOps/s），而 Cube 计算利用率仅 14.7%（远低于搬运引擎 35.2%），所以整体判定为 **"latency bound:compute caused"**。

### 对比：Vector-only kernel 为何是 "pipeline caused"

```
所有 roofline ratio < 39%  → latency bound ✓

Vector 通路:
  compute_ratio = 61.2%    ← 较高
  mte2_ratio    = 18.1%    ← 最低！
  61.2% > 18.1% → pipeline 是瓶颈单元 → "pipeline caused" ✓
```

计算单元利用率（61.2%）明显高于搬运引擎（18.1%），说明计算单元在等数据，瓶颈在搬运流水线。

### 完整决策树

```
                        所有 roofline ratio < 阈值?
                       /              \
                     否                是
                    /                    \
           某通路 ratio ≈ 1.0?        latency bound
           /        \                    |
         否          是            compute_ratio vs mte_ratio?
         |           |              /        |        \
    compute      memory        compute <   compute >   带宽
     bound        bound        mte         mte         接近饱和
                               |           |           |
                          compute       pipeline     memory
                          caused        caused       caused
```

### 数据来源

- **Profiling 数据**: `/home/tilelang/zuochuanuong/prof/OPPROF_20260814025420_ZEBQXREXAQJUQYSA`（Mix kernel）
- **Profiling 数据**: `/tmp/tileops_msprof_2fun3kfv/msprof_output/OPPROF_20260813122049_IDYAVJINHLQBOPMQ`（Vector kernel）
- **msopprof 工具**: `/usr/local/Ascend/cann-9.1.0-beta.1/tools/msopprof/bin/msopprof`
- **MindStudio Insight 源码**: `/home/tilelang/zuochuanuong/msinsight`
