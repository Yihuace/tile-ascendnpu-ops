# MindStudio Insight Roofline 指标计算分析

## 核心结论

msprof 采集工具（即 `msopprof` 二进制程序，属于 CANN 工具链）负责计算这 4 个指标并写入 `visualize_data.bin`。其源码位于 CANN 内部的 `msopprof/csrc/` 目录，未包含在 MindStudio Insight 仓库中。通过**逆向分析 bin 文件和 CSV 原始数据**，验证了每个指标的计算公式。

## 源码位置（从 msopprof 二进制中提取的调试路径）

| 源文件 | 职责 |
|--------|------|
| `msopprof/csrc/op_profiling/profiling/device/data_visualize/roofline.cpp` | Roofline 核心计算逻辑 |
| `msopprof/csrc/op_profiling/profiling/device/data_visualize/pmu_calculator.cpp` | fops 计算（PMU 指标） |
| `msopprof/csrc/op_profiling/profiling/device/data_visualize/storage_access.cpp` | 内存访问数据计算 |
| `msopprof/csrc/op_profiling/profiling/device/data_parse/pmu_calculate.cpp` | PMU 原始数据解析 |

C++ 类继承结构（从 RTTI 信息提取）：

```
Visualize::RoofLine (基类)
├── Visualize::RoofLineOf910B   ← Ascend910B2C 走这个
├── Visualize::RoofLineOfA5
└── Visualize::RoofLineOf310P
```

## 4 个指标的计算公式（均已用数据验证）

### 前置数据：fops（总运算次数）

```
fops = Σ(FP32_instructions × 64 + MISC_instructions × 32)  对所有 block 求和
```

- **64** = FP32 每条指令的运算量（256B SIMD 宽度 / 4B per FP32 = 64 elements）
- **32** = MISC 每条指令的运算量

验证：`393728 × 64 + 8704 × 32 = 25,477,120`（与 `aiv_vec_fops` 之和精确匹配）

---

### 1. Bandwidth（带宽）= 1.8 TB/s

```
bw = 硬件理论峰值带宽（SoC 固有规格，非采集计算）
```

- **GM Read + Write**: 1.8 TB/s（Ascend 910B2C 的 GM/HBM 理论带宽）
- **L2 Read + Write**: 8.0 TB/s（L2 Cache 理论带宽）

源码中通过 SoC 类型查表获取，二进制中的错误信息印证了这一点：

```
"Missing theoretical bandwidth for soc %s, using default value"
"Missing theoretical L2 bandwidth for soc %s, using default value"
"Failed to get max bandwidth of type : %d"
```

`RoofLineOf910B` 类中硬编码了 910B 系列各 SoC 的理论带宽表。

---

### 2. Arithmetic Intensity（算力强度）= 3.037 Ops/Byte

```
AI = total_fops / total_bytes

total_bytes = Σ(L2_read_miss) × cacheline_size(128B)
```

- `L2_read_miss` = L2 Cache 读未命中数（即需要从 GM 读取的 cacheline 数）
- **128** = cacheline 大小（字节）

验证（从 `L2Cache.csv` 提取）：

```
total_L2_read_miss = 65537
total_bytes = 65537 × 128 = 8,388,736 bytes
AI = 25,477,120 / 8,388,736 = 3.037063  ✓（与 bin 文件中 3.0370631217956543 匹配）
```

`StorageAccess910B` 类负责从硬件 PMU 计数器计算各级存储的访问字节数。

---

### 3. Performance（实际性能）= 1.55 TOps/s

```
Performance = total_fops / task_duration
```

- `task_duration` = 算子任务总耗时（微秒），来自 `OpBasicInfo`

验证（从 `visualize_data.bin` 的 `DETAILS_BASE_INFO` 块提取）：

```
total_fops = 25,477,120
task_duration = 16.440658569335938 μs
Performance = 25,477,120 / 16.440658569335938e-6 = 1.549641 × 10¹² Ops/s = 1.5496 TOps/s  ✓
```

`PmuCalculator910B` 类负责 fops 计算，`OpBasicInfo` 类提供 task_duration。

---

### 4. Computility（峰值算力）= 5.4698 TOps/s

```
computility = (peak_FP32_ppc × fp32_proportion + peak_MISC_ppc × misc_proportion) × freq × num_cores
```

| 参数 | 值 | 含义 |
|------|-----|------|
| `peak_FP32_ppc` | 64 | FP32 每周期每核峰值运算量 |
| `peak_MISC_ppc` | 32 | MISC 每周期每核峰值运算量 |
| `fp32_proportion` | 97.837% | FP32 活跃周期占向量总活跃周期比例 |
| `misc_proportion` | 2.163% | MISC 活跃周期占比 |
| `freq` | 1800 MHz | AI Core 频率 |
| `num_cores` | 48 | 910B2C 的 AI Core 数量 |

其中：

```
fp32_proportion = Σ(fp32_ratio × total_cycles) / Σ(vec_ratio × total_cycles)
misc_proportion = Σ(misc_ratio × total_cycles) / Σ(vec_ratio × total_cycles)
```

验证：

```
(64 × 0.97837151 + 32 × 0.02162850) × 1800 × 48 = 5.4698 TOps/s  ✓
```

`computility_name` 字段（`"Vec_FP32(97.837151%),Vec_MISC(2.162850%)"`）就是这个比例分解的可视化展示。二进制中的 `theoryTfops` 字符串对应此计算。

---

### 5. Performance Ratio（性能占比）= 28.347%

```
ratio = Performance / min(AI × bw, computility)
```

这是标准 Roofline 模型的利用率公式：

- 当 `AI × bw < computility`（**内存瓶颈**）：`ratio = Performance / (AI × bw)`
- 当 `AI × bw ≥ computility`（**算力瓶颈**）：`ratio = Performance / computility`

验证（使用 bin 文件中的精确值）：

| 通路 | AI × bw | computility | min | ratio = Perf / min | 预期 ratio | 匹配 |
|------|---------|-------------|-----|-------------------|-----------|------|
| GM (内存瓶颈) | 3.037 × 1.8 = 5.467 | 5.470 | 5.467 | 1.5496 / 5.467 = **0.283469** | 0.283469 | ✓ 精确 |
| L2 (算力瓶颈) | 2.980 × 8.0 = 23.84 | 5.470 | 5.470 | 1.5496 / 5.470 = **0.283308** | 0.283308 | ✓ 精确 |

---

## 完整计算流程图

```
硬件 PMU 计数器
    │
    ├── ArithmeticUtilization.csv ──→ aiv_vec_fops, aiv_vec_fp32_ratio, aiv_vec_misc_ratio
    ├── L2Cache.csv ────────────────→ L2 read miss count
    ├── Memory.csv ─────────────────→ GM_to_UB_datas, UB_to_GM_datas
    ├── OpBasicInfo.csv ────────────→ task_duration, cur_freq, block_dim
    │
    ▼
msopprof (roofline.cpp)
    │
    ├── fops = Σ(FP32_instr×64 + MISC_instr×32)         ← PmuCalculator
    ├── total_bytes = Σ(L2_read_miss) × 128              ← StorageAccess
    ├── Performance = fops / task_duration                ← point[1]
    ├── AI = fops / total_bytes                           ← point[0]
    ├── bw = SoC理论带宽表查表                            ← RoofLineOf910B
    ├── computility = weighted_peak_ppc × freq × cores   ← theoryTfops
    └── ratio = Performance / min(AI×bw, computility)     ← ratio
    │
    ▼
visualize_data.bin (DETAILS_ROOFLINE 数据块, type=13)
    │
    ▼
MindStudio Insight (只读取+格式化展示, 不做计算)
```

## 数据来源说明

- **Profiling 数据**: `/tmp/tileops_msprof_2fun3kfv/msprof_output/OPPROF_20260813122049_IDYAVJINHLQBOPMQ/`
- **MindStudio Insight 源码**: `/home/tilelang/zuochuanuong/msinsight`
- **msopprof 工具**: `/usr/local/Ascend/cann-9.1.0-beta.1/tools/msopprof/bin/msopprof`

### MindStudio Insight 侧仅做展示处理

- `server/src/modules/source/parser/RooflineParserImpl.cpp:77-90` — 解析 JSON 字段，无计算
- `modules/compute/src/components/detail/Roofline/RooflineChart.tsx:360-374` — tooltip 展示，唯一"计算"为 `100 * Number(ratio)` 转百分比
- `modules/lib/src/utils/Common.tsx:210-224` — `formatDecimal` 保留 3 位有效小数
