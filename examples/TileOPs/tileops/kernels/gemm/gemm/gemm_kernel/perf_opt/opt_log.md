# GemmOp Stage 4 调优日志（`_gemm_kernel`）

> 口径：`msprof op` 的 `Task Duration(us)`（唯一时延口径）。
> 达标线：**`Ratio ≥ 60%`**（msprof 原生口径：AI/流量取自 L2 读未命中，**只计 GM 读、不含写**，上限 100%）。
> 目标用例集：`benchmark_manifest: examples/TileOPs/tileops/manifest/gemm.yaml::GemmOp.workloads`，
> 16 项（14 `tune` + 2 `smoke`），分类见 `workload_inventory.json`（**不得**用 DESIGN.md 的 shape 替代）。
> 权威入口：`cd examples/TileOPs && python3 -m tileops.reporting.cli run --op GemmOp --prof-mode msprof`

---

## 0. 结论（定稿）

| 项 | 值 |
|---|---|
| 定稿内核 | `perf_opt/GemmOp.py`（规范名 = Stage 4 gate 硬编码的 `perf_opt/{operator_name}.py`；sha256 `4120c870…`） |
| 定稿配置 | `block_M=128 / block_N=256 / block_K=256 / num_stages=2`（M/N 方向另有形状相关策略，见 ⑤） |
| 关键内核改动 ① | `_tile_size()` 由「固定候选表（上限 128）」改为「从调用方 cap 向下扫描 16 的倍数」——基线会把 `block_N=256` **静默截断回 128**，不改这里任何 >128 的 tile 都无效 |
| 关键内核改动 ② | 新增 `_mn_aspect()` 形状相关 M/N 方向策略：`B = N×K` 超过 128 MB 时改用 `(block_M, block_N) = (256, 128)` |
| wrapper | `gemm.py` 已切 `perf_opt`（`_gemv_kernel` 不在范围，仍用 Stage-3 基线）|
| 达标 | **6 / 14**（`Ratio ≥ 60%`）|
| 加速比 | 中位 **2.34×**，最大 2.83×，最小 1.74× |
| 权威报告 | `examples/TileOPs/reports/tileops/20261016_111604_041135_GemmOp/`（correctness 17/17，benchmark 16 case）|

---

## 1. 基线（round 0，未调优）

`cfg=(128,128,64,3)`，见 `perf_records.jsonl` 的 `phase=baseline` 行
（权威复测：`reports/tileops/20261015_164502_081440_GemmOp/`）。

| workload | 延迟 (µs) | Ratio | Mem项 | Compute项 |
|---|--:|--:|--:|--:|
| square-1k-nn / float16 | 36.48 | 21.78% | 2.40% | 21.78% |
| square-1k-nn / bfloat16 | 34.90 | 22.76% | 2.50% | 22.76% |
| ds-v3-decode-gate-up | 155.40 | 9.23% | 2.87% | 9.23% |
| ds-v3-decode-down | 77.66 | 17.90% | 5.34% | 17.90% |
| ds-v3-prefill-gate-up | 1949.02 | 23.54% | 0.68% | 23.54% |
| ds-v3-prefill-down | 1544.51 | 28.80% | 0.45% | 28.80% |
| ds-v3-prefill-attn-proj / float16 | 3019.06 | 29.47% | 0.83% | 29.47% |
| ds-v3-prefill-attn-proj / bfloat16 | 3001.24 | 29.64% | 0.73% | 29.64% |
| k-dominant-7168x16384 | 14064.82 | 25.30% | 5.94% | 25.30% |
| wide-n-24576 | 4237.40 | 27.00% | 2.32% | 27.00% |
| mid-m16-attn | 106.20 | 7.71% | 7.71% | 3.27% |
| mid-m32-attn | 113.04 | 7.27% | 7.27% | 6.15% |
| mid-m64-down | 58.38 | 11.91% | 7.05% | 11.91% |
| mid-m96-gate-up | 141.88 | 7.58% | 3.10% | 7.58% |

**共性**：全部 14 个用例都是**算力受限**（访存项普遍 < 10%，`Ratio = Compute 项`），瓶颈在 Cube 利用率（基线 cube util ≈ 0.344）。

---

## 2. 轮次与「候选 vs current best」

### Round 1 — tile 形状扫描（在 `ds-v3-prefill-down` 上，1 变量法）

| 候选 cfg | Task Duration (µs) | AICore **cube** util | mte2 | mte1 | scal | L2 hit |
|---|--:|--:|--:|--:|--:|--:|
| (128,128,**64**,3) ← baseline | 1551.8 | 0.344 | 0.949 | 0.323 | 0.030 | 98.5% |
| (128,128,128,3) | 1046.4 | 0.470 | 0.758 | 0.400 | 0.032 | 98.5% |
| (128,128,**256**,2) | 849.1 | 0.558 | 0.596 | 0.515 | 0.028 | 98.5% |
| (128,**256**,64,3) | 1136.1 | 0.437 | 0.953 | 0.333 | 0.025 | 99.0% |
| (**256**,128,64,3) | 1176.7 | 0.425 | 0.948 | 0.269 | 0.024 | 98.0% |
| (128,256,128,2) | 756.6 | 0.630 | 0.770 | 0.465 | 0.025 | 99.0% |
| (176,176,64,3) | 1546.3 | 0.344 | 0.949 | 0.323 | 0.030 | 98.5% |
| (128,192,64,3) | 1544.8 | 0.345 | 0.949 | 0.324 | 0.030 | 98.5% |

**读法**：`block_K` 是主旋钮（64→256 把 cube util 从 0.344 拉到 0.558）；`block_N` 256 再叠加到 0.630；
`block_M=256`、`(176,176)`、`(128,192)` 均为**静默回退**到与基线相同的实际 tile（说明 `_TILE_CANDIDATES` 上限 128 在生效）。

### Round 1b — 继续加压

| 候选 cfg | Task Duration (µs) | cube util |
|---|--:|--:|
| (128,256,128,2) | 755.3 | 0.629 |
| **(128,256,256,2)** ← winner | **643.3** | **0.722** |
| (128,256,128,3) | 755.4 | 0.629 |
| (256,128,256,2) | 649.8 | 0.718 |
| (128,128,512,2) | 753.0 | 0.627 |
| (128,256,512,1) | **FAIL**（L1 装不下，无 OpBasicInfo） | — |
| (128,224,256,2) | 765.0 | 0.611 |
| (64,256,256,2) | 931.9 | 0.540 |

### Round 2 — 把 winner 与次优跑满全部 14 个 tune 用例

| candidate_id | 14 用例 min / max (µs) | L0 |
|---|--:|:--:|
| `r2_bk256_bn256` = (128,**256**,**256**,2) ← winner | 19.1 / 5998.1 | PASS |
| `r2_bk256_bn128` = (128,128,256,2) | 27.1 / 7103.2 | PASS |

（逐用例数据见 `perf_records.jsonl` 的 `round=2, phase=candidate` 行。）

### Round 3 — 独立复核 + 参数空间穷尽性验证

独立复核（`msprof op` 直跑，未走 harness）＋ 收尾实验：

| 实验 | 结果 | 结论 |
|---|---|---|
| 复现 winner：flextile + (128,256,256,2) | square-1k 19.33 µs（上轮 19.30）；prefill-down 643.25（上轮 644.28） | winner **可复现** |
| 交叉复核 (128,128,256,2) | prefill-down 847.43（上轮 847.42） | 两套测量**逐位一致** |
| `block_K=512` × `block_N=256` | 三种 m 全部 **FAIL**（L1 容量） | 该方向封死 |
| `block_K=512` × `block_N=128` | k-dominant 6846.9；square-1k 28.4 | 劣于 winner |
| `block_K=384 / 320` | 384 全部 FAIL；320 k-dominant 8244.3、gate-up 927.5 | 劣于 winner |
| `num_stages = 1 / 2 / 3` | square-1k 19.09 / 19.11 / 19.26；prefill-down 642.75 / 642.05 / 643.70 | **等价**（流水深度非瓶颈）|
| `block_M` 缩小（切 M 维补核数） | decode-gate-up 55.4→73.1(bm64)→90.8(bm32)；mid-m96 50.3→68.3(bm48) | **更差** |
| `block_N` 缩小（凑 20 核整数倍块数） | decode-gate-up 54.8(bn256/11块) vs 71.5(bn128/22块)；mid-m16 61.4 vs 66.5(bn128) / 67.2(bn64) | **更差** |

> 结论：**K 块深 256、M/N 取大块**是参数层最优；剩余差距不在 tile 参数。

### Round 3b — M/N 方向的形状相关策略（新增 ⑤）

在 Round 3 的穷尽结果上发现一个例外：**`k-dominant`（B = N×K = 224 MB ≫ L2）**下，
把 L0C 预算 `32768` 从 `(128,256)` 重新分配到 `(256,128)` 会快 12.8%：

| 用例 | B = N×K×2B | (128,256) µs | (256,128) µs | 结论 |
|---|--:|--:|--:|---|
| **k-dominant-7168x16384** | **224 MB** | 6021.1 | **5247.9** | **(256,128) 胜 −12.8%** |
| ds-v3-prefill-gate-up | 30 MB | **717.9** | 782.3 | (128,256) 胜 +9.0% |
| ds-v3-prefill-down | 29 MB | **642.8** | 648.6 | (128,256) 平/略胜 |
| ds-v3-prefill-attn-proj | 59 MB | **1199.7** | 1229.4 | (128,256) 胜 |
| wide-n-24576 | 75 MB | 1761.3 | **1759.2** | 平 |
| square-1k-nn | 2 MB | **19.22** | 22.98 | (128,256) 胜 +19.6% |
| ds-v3-decode-gate-up | 30 MB | **55.09** | 70.67 | (128,256) 胜 |
| ds-v3-decode-down | 29 MB | 38.54 | **36.75** | (256,128) 略胜 |
| mid-m64-down | 29 MB | 33.31 | **29.73** | (256,128) 略胜 |
| mid-m96-gate-up | 30 MB | **50.27** | 63.30 | (128,256) 胜 |

**机理**：B 远超 L2 时两个操作数都从 GM 流式读，**更高的 M tile 把 B 的重复遍数减半**；
B 能驻留 L2 时，被重复读的是 A，此时**更宽的 N tile** 更优 —— 这也解释了为什么除 `k-dominant` 外全部形状选 (128,256)。

**落地**：内核内 `_mn_aspect()`，阈值 `B > 128 MB` 时切 `(256,128)`，否则 `(128,256)`；
两候选都恰用满 `block_M×block_N = 32768`（L0C 上限），并保留「不超调用方预算」的收缩保护。

> ⚠️ **置信度声明**：该阈值是**拟合**的 —— 目标集里只有一个用例（`k-dominant`，224 MB）超过阈值，
> 其余全部 ≤ 75 MB，两者分离充分但**只有单点支撑**。该规则只在 B > 128 MB 时触发（推理形状罕见），
> 故对一般负载风险低；后续若接入 B 在 75~224 MB 之间的形状，**须重新标定阈值**。

### Round 4 权威复测（harness 全量，定稿内核）

| workload | baseline (µs) | **final (µs)** | Ratio | 达标 | 加速 |
|---|--:|--:|--:|:--:|--:|
| ds-v3-prefill-attn-proj / float16 | 3019.06 | 1199.60 | 74.17% | ✅ | 2.52× |
| ds-v3-prefill-attn-proj / bfloat16 | 3001.24 | 1200.92 | 74.09% | ✅ | 2.50× |
| ds-v3-prefill-down | 1544.51 | 642.07 | 69.28% | ✅ | 2.41× |
| k-dominant-7168x16384 | 14064.82 | 5152.20 | 69.07% | ✅ | 2.73× |
| ds-v3-prefill-gate-up | 1949.02 | 718.69 | 63.83% | ✅ | 2.71× |
| wide-n-24576 | 4237.40 | 1856.20 | 61.63% | ✅ | 2.28× |
| mid-m16-attn | 106.20 | 57.66 | 56.80% | | 1.84× |
| mid-m32-attn | 113.04 | 58.30 | 56.39% | | 1.94× |
| mid-m64-down | 58.38 | 33.60 | 48.98% | | 1.74× |
| ds-v3-decode-down | 77.66 | 38.08 | 43.60% | | 2.04× |
| square-1k-nn / float16 | 36.48 | 19.12 | 41.55% | | 1.91× |
| square-1k-nn / bfloat16 | 34.90 | 19.04 | 41.72% | | 1.83× |
| mid-m96-gate-up | 141.88 | 50.10 | 35.10% | | 2.83× |
| ds-v3-decode-gate-up | 155.40 | 55.64 | 32.06% | | 2.79× |

**L0/L1 精度回归**：全过（correctness 17/17，含 smoke 与 gemv 边界）。两处内核改动（`_tile_size` 上限放宽、
`_mn_aspect` 方向策略）都只影响 **tile 尺寸/方向选择**，不改变 K 累加顺序与 fp32 累加域。

> 测量噪声：同一配置的批间波动实测约 ±5%（如 wide-n 1760→1856），故「边缘差 0.5pp 内」的结论不稳健。

### Round 5 — 真 split-K 探针（P2；结论：**证伪，未采纳**）

**先纠正一个被混用的旧结论**：赛前清单里的「GemmOp split-K 无效」实测内容是 `block_N` 256→128（块数 11→22，
**块更碎**），不是 K 维切分。本轮实现的是真 split-K：`block_M/N/K` 与每块 B tile（256×256）完全不变，
只把每个输出块的 K 区间切成 `split_k` 段，段数 × 逻辑块数 = 发射块数。

**实现**（`_gemm_kernel_round5_splitk.py`，留档；基准 `GemmOp.py` 未改动）

- `CG-2026-0011`（无 L0C 原子累加）→ 两段式：`_gemm_splitk_partial` 每 (块, 段) 写 fp32 workspace
  `W[split_k*m, n]`（静态 extent，段的 K 偏移是动态 offset，合法），`_splitk_reduce` 逐段 `T.vadd` 求和后
  一次 `T.vcast` 落到 bf16。
- `T.atomic_add`（GM，支持 bf16/fp32）评估后**否决**：算子契约要求输出 dtype == 输入 dtype，fp32 原子累加
  仍需第二个 cast kernel（发射数不减），却额外引入同块 `split_k` 路写竞争。
- 分派门禁：`k % (block_K*split_k) == 0`、`m % bm == 0`、`n % bn == 0`（保证两个 kernel 全是静态 extent）。
  目标形状 7168 = 256×28，故 `split_k ∈ {2,4,7,14,28}` 均可整除；`split_k=7` 是唯一能填满 20 核的档
  （16 块 × 7 = 112 块 = 5.6 波/核）。

**候选 vs current best 对比表**（本机 chip1 / Phy-ID 13，`Ascend910_9362`：20 AIC / 40 AIV / L2 168 MB；
A/B 交错 3~5 次取中位；`ratio` 按仓库口径 `L2 读未命中字节 ÷ 时长 ÷ 1.8 TB/s` 复算，已用官方 56.80%/57.659µs 逐位校准）

| workload | 分支 | 发射块数 | main (µs) | reduce (µs) | **总时延 (µs)** | cube util | MTE2 util | 单核读 BW (GB/s) | 聚合读 BW (TB/s) | ratio (%) | L0 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| mid-m16 | **sk=1（current best）** | 16 | **60.68** [57.30,70.36] | — | **60.68** | 0.131 | 0.974 | 65.5 | 1.02 | 53.97 | PASS |
| mid-m16 | sk=2 | 32 | 61.87 | 2.30 | 64.20 (+5.8%) | — | — | — | — | 51.02 | PASS |
| mid-m16 | sk=7 | 112 | 58.88 [57.90,62.96] | 4.08 | **63.04 (+3.9%)** | 0.111 | 0.875 | 55.7 | 0.99 | 51.95 | PASS |
| mid-m16 | sk=28 | 448 | 73.72 | 10.08 | 83.81 (+38%) | — | — | — | — | 39.09 | PASS |
| mid-m16 | bn=128 对照（缩块加块数） | 32 | 65.23 | — | 65.23 (+7.5%) | 0.117 | 0.969 | 61.8 | 1.04 | 50.34 | PASS |
| mid-m32 | **sk=1（current best）** | 16 | **59.45** [57.86,61.45] | — | **59.45** | 0.209 | 0.972 | 69.2 | 1.14 | 55.30 | PASS |
| mid-m32 | sk=2 | 32 | 64.78 | 3.08 | 67.85 (+14.1%) | — | — | — | — | 48.47 | PASS |
| mid-m32 | sk=7 | 112 | 59.48 [58.42,60.68] | 5.66 | **65.14 (+9.6%)** | 0.172 | 0.863 | 56.8 | 1.10 | 50.47 | PASS |
| mid-m32 | sk=28 | 448 | 75.64 | 16.42 | 92.07 (+55%) | — | — | — | — | 35.71 | PASS |
| mid-m32 | bn=128 对照 | 32 | 67.26 | — | 67.26 (+13.1%) | — | — | — | — | 48.88 | PASS |

**判定表：结构性分支是否带来有效提升（噪声阈值 3%）**

- **有效分支：无**。`sk=2/7/28` 与 `bn=128` 四条结构性分支**全部劣于或等于** current best，最接近的
  `sk=7`（+3.9% / +9.6%）也是回退。
- **winner：无，current best 保持 `r3b_aspect_policy`（`perf_opt/GemmOp.py` 未改动，sha256 `4120c870…`）**。
- **本轮无合并动作**，故不涉及「其他 tune workload 非回退检查」；全 14 个 tune 用例的定稿数值仍是 §3 末尾 `Final Performance Test Data` 表（Round 4 权威复测）。

**决定性证据：K 维切分把 20 个核全部填满，但 GEMM 时间一点没降**

| workload | 16 块 / 16 核（sk=1） | 112 块 / 20 核（sk=7） | main 时长差 |
|---|--:|--:|--:|
| mid-m16 | 60.68 µs | 58.88 µs | −3.0%（落在 ±8% 批间噪声内） |
| mid-m32 | 59.45 µs | 59.48 µs | **+0.05%（逐位相同）** |

即「16 块 < 20 核 → 4 个核空转」**不是**这个形状的真实瓶颈。profile 直接给出旁证：

- sk=1 时每核 **MTE2 占用 97.4%**（流水满负荷）、cube 仅 12~21%，msprof 结论行是
  `latency bound: pipeline caused`；
- 把块数从 16 拉到 112，**聚合 GM 读带宽仍停在 0.99~1.14 TB/s**（单核读 BW 反而从 65.5 降到 55.7 GB/s，
  因为每块只剩 4 次流水迭代、斜坡占比升高）。
- 两条独立的加块途径（K 维切 7 段 / N 维切 128 宽）都得到同一个结论：**这块 die 上该访存模式的
  聚合 GM 读带宽在 ~1.0–1.14 TB/s 饱和，与块数/核占用无关**；56.8% = 1.02/1.8，剩下的 44% 是这一模式的
  带宽天花板，不是核数。

**归约 + launch 开销分解（实测）**

两段式的第二段 `reduce`（fp32 workspace 求和 + 单次 vcast）实测：

| 分支 | workspace (MB) | reduce 实测 (µs) | 线性模型 1.73 + 1.15×MB |
|---|--:|--:|--:|
| m16 sk=2 | 0.5 | 2.30~2.36 | 2.31 |
| m16 sk=7 | 1.8 | 4.04~4.16 | 3.80 |
| m16 sk=28 | 7.3 | 10.08~10.10 | 10.13 |
| m32 sk=7 | 3.7 | 5.66~5.76 | 5.95 |

⇒ **固定 launch ≈ 1.7 µs + 边际 ≈ 1.15 µs/MB（有效归约带宽 ≈ 0.87 TB/s）**。m16/sk=7 的 4.0 µs
= 1.7 launch + 2.3 流量，占 GEMM 时长 6.9%。即使 GEMM 侧有 7% 的收益也会被这一项吃平；
而实测 GEMM 侧收益为 0。

**精度**：split-K 路径 fp32 workspace 累加（每段 fp32 L0C，段间 fp32 相加，末端一次 RINT 到 bf16），
L0 全部 11 例 PASS（含 sk=2/7 的 m16/m32、m128、NN fp16），worst max_diff = 1.0（bf16 ≈ 2 ulp，与单 kernel 路径同量级）。

**口径说明（不得用来过线）**：官方 harness 用 `--kernel-name=main` 采集，split-K 路径下它**只看得到 GEMM 核**
（57.9~59.5 µs），reduce 的 4~5.7 µs 与 workspace 写流量都不会计入 —— 那会把「更慢」显示成「持平」。
本表所有 split-K 时延与 ratio 均为 **main + reduce 两核合计**，未采用任何缩口径的写法。

**结论**

1. **真 split-K 在 m16/m32 上证伪**：不能把 Ratio 推过 60%（最好 51.95% / 50.47%），总时延也不降
   （+3.9% / +9.6%）。与「缩 bn 加块数」的旧实验结果（本轮同样复测：65.23 / 67.26 µs）同向 —— **两条加块途径都劣于现状**，
   故「真 split-K 也无效」成立，后人不必再试。
2. 机理不是「核没占满」，而是「聚合 GM 读带宽已饱和」；因此 `sk` 再大（28）只会把单块工作压到 1 次流水迭代，
   斜坡开销爆炸（main 73.7 / 75.6 µs）。
3. 其余 6 条未达标用例（短 K / 小方块）按任务边界**未做**，其归因（流水斜坡 / 结构性）不受本轮结果影响。

---

## 3. 未达标 8 例的归因（参数空间已穷尽 + Round 5 修正）

| workload | Ratio | 结构性原因 |
|---|--:|---|
| mid-m16-attn | 56.80% | m=16 → 1 个 M 行块、16 块；**Round 5 修正**：真实瓶颈是聚合 GM 读带宽 ~1.0 TB/s 饱和（112 块铺满 20 核后 GEMM 时长 0 变化），不是「4 核空转」 |
| mid-m32-attn | 56.39% | m=32 → 同上；**Round 5 修正**：同上（112 块 vs 16 块，main 59.48 vs 59.45 µs）|
| mid-m64-down | 48.98% | K=2048 短（bk256 仅 8 次迭代）+ m=64 |
| ds-v3-decode-down | 43.60% | m=128 → 1 个 M 行块；K=2048 短 |
| square-1k-nn（双 dtype） | ~41.6% | M=N=K=1024 三者皆小，K 仅 4 次迭代，流水斜坡占比高 |
| mid-m96-gate-up | 35.10% | m=96、n=2112 → 逻辑块 11 个；访存项 35.10% 与 m16/m32 同属「该模式读带宽上限」量级（**未实测 split-K，属外推**）|
| ds-v3-decode-gate-up | 32.06% | m=128、n=2112 → 逻辑块 11 个；同上（**未实测 split-K，属外推**）|

**已实测排除的参数方向**：`block_K` 384/512/320（L1 封死或更差）、`num_stages` 1~4（等价）、
`block_M` 缩小（更差）、`block_N` 缩小（更差）、`block_M/N` 放大（L0C 128KB 上限 `block_M×block_N ≤ 32768`）、
M/N 方向重分配（仅对 B≫L2 的 `k-dominant` 有效，已并入 ⑤）。

**结论（Round 5 后更新）**：剩余差距来自两方面——（a）短 K / 小方块用例的**流水斜坡与固定开销**；
（b）小 M 用例的**聚合 GM 读带宽饱和**（~1.0~1.14 TB/s，与核占用无关）。两者都不是参数层可解，
且**都不是「加块数 / 切 K 维」能补的**（Round 5 已用两条独立途径证伪）。剩余可选方向只剩
persistent + 核内多任务（Developer 模式已被 `CG-2026-0010` / `TRAP-DEVMODE-PERSIST-GEMM` 封死）
或换编程模式（Expert，属设计层改动）。

> 未产出 `perf_feedback.md`：`[DESIGN_LIMIT]` 触发条件两项**不同时**满足——① 8 例的剩余差距本身估计在
> 1.06×~1.9× 区间（均 < 2×），Round 5 之后 m16/m32 的可得上限进一步收窄到 **1.11×**（56.80% → 60% 需 1.06×，
> 而实测聚合带宽已贴着该模式上限）；②「逻辑块 < 物理核数」是 Stage-3 设计 §5.5 已接受的情形，Round 5
> 推翻的是**归因**（不是 DESIGN 假设）——DESIGN 并未假设「填满核数即可提速」，故不构成「与设计假设直接矛盾」。
> 按契约条件不满足时禁止产出该文件，此处仅留档归因与证伪证据。

---


### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|--:|--:|---|
| tileops/kernels/gemm/gemm/gemm_kernel/_gemm_kernel.py::_gemm_kernel::main | square-1k-nn-float16 | 35.739 | 19.120 | r3b_aspect_policy |
| tileops/kernels/gemm/gemm/gemm_kernel/_gemm_kernel.py::_gemm_kernel::main | square-1k-nn-bfloat16 | 35.819 | 19.040 | r3b_aspect_policy |
| tileops/kernels/gemm/gemm/gemm_kernel/_gemm_kernel.py::_gemm_kernel::main | ds-v3-decode-gate-up-bfloat16 | 154.397 | 55.639 | r3b_aspect_policy |
| tileops/kernels/gemm/gemm/gemm_kernel/_gemm_kernel.py::_gemm_kernel::main | ds-v3-decode-down-bfloat16 | 75.418 | 38.079 | r3b_aspect_policy |
| tileops/kernels/gemm/gemm/gemm_kernel/_gemm_kernel.py::_gemm_kernel::main | ds-v3-prefill-gate-up-bfloat16 | 1937.681 | 718.686 | r3b_aspect_policy |
| tileops/kernels/gemm/gemm/gemm_kernel/_gemm_kernel.py::_gemm_kernel::main | ds-v3-prefill-down-bfloat16 | 1546.629 | 642.067 | r3b_aspect_policy |
| tileops/kernels/gemm/gemm/gemm_kernel/_gemm_kernel.py::_gemm_kernel::main | ds-v3-prefill-attn-proj-float16 | 3003.060 | 1199.596 | r3b_aspect_policy |
| tileops/kernels/gemm/gemm/gemm_kernel/_gemm_kernel.py::_gemm_kernel::main | ds-v3-prefill-attn-proj-bfloat16 | 2988.080 | 1200.916 | r3b_aspect_policy |
| tileops/kernels/gemm/gemm/gemm_kernel/_gemm_kernel.py::_gemm_kernel::main | k-dominant-7168x16384-bfloat16 | 13521.629 | 5152.197 | r3b_aspect_policy |
| tileops/kernels/gemm/gemm/gemm_kernel/_gemm_kernel.py::_gemm_kernel::main | wide-n-24576-bfloat16 | 4265.285 | 1856.203 | r3b_aspect_policy |
| tileops/kernels/gemm/gemm/gemm_kernel/_gemm_kernel.py::_gemm_kernel::main | mid-m16-attn-bfloat16 | 108.318 | 57.659 | r3b_aspect_policy |
| tileops/kernels/gemm/gemm/gemm_kernel/_gemm_kernel.py::_gemm_kernel::main | mid-m32-attn-bfloat16 | 111.178 | 58.299 | r3b_aspect_policy |
| tileops/kernels/gemm/gemm/gemm_kernel/_gemm_kernel.py::_gemm_kernel::main | mid-m64-down-bfloat16 | 58.399 | 33.599 | r3b_aspect_policy |
| tileops/kernels/gemm/gemm/gemm_kernel/_gemm_kernel.py::_gemm_kernel::main | mid-m96-gate-up-bfloat16 | 135.917 | 50.099 | r3b_aspect_policy |

## 4. Skill Retrospective（复盘）

**本轮最有价值的可迁移经验**

1. **「配置改了没效果」先怀疑被静默截断**：基线 `_tile_size` 用固定候选表（≤128），任何 `block_N=256`
   请求都被吞掉且**不报错**——我最初用基线内核测 `bn256` 得到的「无变化」结论是假的。
   教训：调参前先验证「参数是否真的生效」（对比生成代码 / 打印实际 tile）。
2. **同一 kernel 的两套测量可以互相标定**：我用自建 `msprof op` 探针测 (128,128,256,2) 得 847.43 µs，
   另一套 harness 得 847.423 µs——逐位一致。**先做交叉标定再大规模扫参**，能避免整轮扫参白跑。
3. **`msprof op` 单配置探测 ≈ 20~40 s**（对比整轮 harness 9m21s）：调参阶段自建探针，
   只在定稿后跑全量 harness。
4. **大 tile 的复用收益 > 核数利用率**：11 块（bn256）胜过 22 块（bn128），
   「块数 < 物理核数就缩 tile」是错的直觉——先测再改。
5. **L0C 预算是硬耦合**：`block_M×block_N ≤ 32768` 让 `bm256` 必须配 `bn128`，
   于是「M/N 方向」成为一个**离散的二选一**；方向的最优解与 `B` 是否驻留 L2 相关（⑤）。
6. **边缘结论要防噪声**：实测批间波动约 ±5%，「差 0.5pp 达标」这类结论必须重复测量后再下判断。
7. **「瓶颈是核没占满」这类归因必须先做「填充实验」再采信**（Round 5）：16 块 < 20 核这个观察是**真的**，
   但由此推出「切 K 填满核就能提速」是**错的**——按定义把块数做到 112（= 5.6 波/核，20 核全满）后，
   GEMM 时长逐位不变（m32：59.48 vs 59.45 µs）。教训：**占用率是现象，不是瓶颈证明**；
   判据应是「增加并发单元后聚合吞吐是否上升」。本轮的旁证形态（每核 MTE2 占用 97% + 聚合读 BW 不随块数变化）
   可复用于任何「块数 < 核数」的 Mem-bound 用例。
8. **两段式（split + reduce）结构性分支必须把第二段计入判据**：归约固定开销实测 1.73 µs launch
   + 1.15 µs/MB（≈0.87 TB/s 有效带宽），在 58 µs 的 GEMM 上就是 6.9%。且必须显式声明
   「harness 用 `--kernel-name=main` 只统计第一段」这一口径陷阱，不得用它过线。

**待登记的能力缺口**

- `msprof op` 在 `block_K > 256`（配合 `block_N=256`）、`block_K=384` 等配置下**直接无产物**，
  报错是「No OpBasicInfo CSV」而非容量/L1 约束提示 —— 诊断体验差，建议 harness 侧补容量预检或明确报错。
- `tileops.benchmark.msprof.bench_kernel_msprof` 对**模块级 callable** 路径在本环境不可用（Op 路径正常）。

---

## 5. 产物

- 定稿内核：`perf_opt/GemmOp.py`（含 `_tile_size` 上限放宽 + `_mn_aspect` 方向策略）
- 实验件（留档）：`perf_opt/_gemm_kernel_round1_flextile.py`、`perf_opt/_gemm_kernel_round1_stagevar.py`、
  `perf_opt/_gemm_kernel_round5_splitk.py`（真 split-K 两段式探针，证伪留档）
- 结构化记录：`perf_opt/perf_records.jsonl`（round 0 baseline / round 2 candidate / round 4 final / round 5 split-K 探针）
- 用例分类与结果：`perf_opt/workload_inventory.json`（14 tune 全部 `winner_merged`，2 smoke `precision_pass=true`）
- 调优脚本：`perf_opt/sweep.py`、`run_perf.py`、`bench_case.py`（新增 `--splitk`）、
  `perf_opt/probe_splitk.py`（split-K A/B 探针 + msprof ratio 复算）、`queue*.sh`
- 基线快照：`history_version/20261021_p3p2_round1/`（Round 5 未采纳任何内核改动，定稿文件 sha256 不变）
- 原始日志：`perf_opt/logs/`（按仓库 `.gitignore` 约定不入库，本地保留）

---

## 6. Gate 4 合规修复（2026-10-21，内核未改动）

本节的改动**只涉及文件命名与引用**，`GemmOp.py` 的内核实现与性能数值（`perf_records.jsonl` 的
`duration_us` / `msprof_raw_path` / `timestamp` / `candidate_id` 等）**未做任何修改**，未重跑调优。

- **现象**：`statectl gate 4` 报 `S4-EXISTS`（"kernel 文件不存在或为空"，找 `perf_opt/GemmOp.py`）
  与 14 条 `S4-PERF-RECORDS-RECON`（"final 未测最终 perf_opt 文件"）。
- **根因**：`statectl.py::artifact_paths()` 把 Stage 4 定稿路径硬编码为 `perf_opt/{operator_name}.py`
  （本例 `GemmOp.py`），而集成侧沿用了内部函数名 `_gemm_kernel.py`；`RECON` 要求每条
  `phase="final"` 记录的 `artifact_path`（相对 `perf_opt/` 解析）等于该定稿路径、且
  `artifact_sha256` 与其当前 sha256 一致。
- **改动**：
  1. `git mv perf_opt/_gemm_kernel.py perf_opt/GemmOp.py`（内容零改动，sha256 仍为
     `4120c8700b3ab5e1b18fe8ae9d2f8f1e73d98c4da2e55600aae803f015ac36c8`）；
  2. wrapper `gemm.py` 的 perf_opt 切换块 import 改为 `.gemm_kernel.perf_opt.GemmOp`；
  3. `perf_records.jsonl` 中 14 条 `phase="final"` 行：`artifact_path` 由绝对路径改为规范相对名
     `GemmOp.py`，`artifact_sha256` 复核为新文件实际 sha256（**值未变**，仅因改名后重算同一内容）；
     其余字段与其余 74 行（baseline/candidate）逐字未动，总行数仍为 88。
  4. 实验件 `_gemm_kernel_round1_*.py` / `_gemm_kernel_round5_splitk.py` / `probe_splitk.py` **保持原名**
     （它们是留档实验件，不是定稿，且不 import 定稿模块）。
- **口径说明（避免后续误读）**：`perf_records.jsonl` 与 `workload_inventory.json` 里的 `kernel_id`
  仍是 inventory 字面量 `tileops/kernels/gemm/gemm/gemm_kernel/_gemm_kernel.py::_gemm_kernel::main`
  （函数名 `_gemm_kernel` 是内核函数符号，与文件规范名解耦，保持稳定以便逐条对账）；
  `artifact_path` 才是定稿文件路径，现为规范名 `GemmOp.py`。
- **未改动**：`gemm_kernel/__init__.py` 仍 `from ._gemm_kernel import _gemm_kernel`——它导出的是
  **Stage-3 基线**同名模块 `gemm_kernel/_gemm_kernel.py`（基线文件不改名），与 perf_opt 定稿无关。
- **刻意保留的一处陈旧自引用**：`GemmOp.py` 模块 docstring 第 60 行仍写 `Run: python _gemm_kernel.py --level ...`。
  不改的原因：任何字节改动都会改变定稿 sha256，使 `perf_records.jsonl` 里已采集的 final 记录失去对账基础
  （须重新测量），故按「内核实现不动」原则保留；实际运行命令见本节验收行。
- **改前快照**：`history_version/20261021_gatefix/`（含 `SNAPSHOT.sha256`）。
- **验收**：`statectl gate 4` → `ok:true / pass:true`；`python perf_opt/GemmOp.py --level all` L0/L1 全过；
  wrapper 导入冒烟通过。
