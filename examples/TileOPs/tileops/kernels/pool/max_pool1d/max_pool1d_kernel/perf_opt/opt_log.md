# Stage 4 性能调优日志 — `MaxPool1dFwdOp` / `_max_pool1d_kernel`

- task_id: `max_pool1d-_max_pool1d_kernel-20261016T090104Z`
- scenario / mode: `optimize` / `full`
- operator dir: `examples/TileOPs/tileops/kernels/pool/max_pool1d/max_pool1d_kernel/`
- baseline kernel (read-only): `_max_pool1d_kernel.py`
  SHA256 `0e3e21346f49b53584c2248f121c92304cf6027a66073410ae8fec0b4389d4f7`
- deliverable: `perf_opt/_max_pool1d_kernel.py`
  SHA256 `f8ae961c015f0e9b71d4cd6f6492333b40ca54d3adb975e3516f9301dadc7a54`
- 唯一时延口径: `msprof op` → `OpBasicInfo.csv` 的 `Task Duration(us)`（median over 20 launches，
  `--launch-count=20 --warm-up=5 --dump=off`；实验分支用 `--launch-count=20`）。

## 0. Environment / toolchain stamp（首轮必记）

| Item | Value |
|---|---|
| npu | Ascend910B3 × 1（`NPUUtils.get().get_aicore_num()` 实查 **20** → 纯 Vector 核数 `20×2 = 40`） |
| UB | 196608 B / AIV（实查 `ascend` 口径 192 KiB） |
| tilelang | `0.1.2+c3c39ba6929f77c131a29c7cde8126ac245e104d`（git `c3c39ba`，dev root build） |
| torch / torch_npu | `2.9.0+cpu` / `2.9.0.post8` |
| CANN | 9.0.0（`msprof` = `/usr/local/Ascend/cann-9.0.0/bin/msprof`） |
| `TILELANG_ASCEND_MODE` | Developer |
| 迁移基线工具链（`CASE-pool-maxpool1d-migration`） | `ddf4255a / 417dae9` — **与本轮不同** ⇒ 迁移期结论按版本戳规则全部视为待重验，本任务全部重新实测 |

`kb_stale_count = 265`（`/tmp/kb_stale_20261016.json`）。本任务引用的 pattern-library 条目
（`PL-1.4` / `PL-1.21` / `PL-1.22` / `PL-1.23` / `CONST-msprof-task-floor-910B3` /
`CONST-copy-floor-dma-granularity` / `TRAP-transpose-ub-gm-roundtrip` 等）均标注 origin_task，
其中 `PL-1.1-transpose-chain` 在 INDEX 中已是 `stale`，未作决策依据。

## 1. Benchmark workload inventory（Phase 1）

入口：`benchmarks/ops/bench_pool.py::test_max_pool1d_bench`（manifest-driven，
`workloads_to_params("MaxPool1dFwdOp", include_extra=True)`，pytest id = `f"{label}-{dtype_str}"`）。
manifest：`tileops/manifest/pool.yaml::MaxPool1dFwdOp.workloads`。实际展开 **6 个参数化案例**：

| # | workload_id | kind | shape | params | dtype | 说明 |
|---|---|---|---|---|---|---|
| 1 | `smoke-1x8x32-k3s2p1-float16` | smoke | [1,8,32] | k3 s2 p1 d1 | fp16 | manifest 首条，label 独立含 `smoke` → 仅精度 |
| 2 | `smoke-1x8x32-k3s2p1-bfloat16` | smoke | [1,8,32] | k3 s2 p1 d1 | bf16 | 同上 |
| 3 | `smoke-1x8x32-k3s2p1-float32` | smoke | [1,8,32] | k3 s2 p1 d1 | fp32 | 同上 |
| 4 | `sincnet-speaker-local-float16` | **tune** | [32,80,4096] | k3 s3 p0 d1 | fp16 | 一般网格；最大流量 27.96 MB |
| 5 | `textcnn-global-float16` | **tune** | [64,256,128] | k128 s1 p0 d1（`L_out=1`） | fp16 | **C4 整线归约**路径 |
| 6 | `ecg-cnn-dilated-bfloat16` | **tune** | [16,128,2048] | k5 s2 p2 d2 ceil | bf16 | 一般网格；膨胀重叠窗 |

无 skipped 案例（bench 全部 6 例可运行、`msprof` 有效；迁移期的 `tuple(int)` TypeError 已修复）。
完整分类与 `tuning_status` 见 `perf_opt/workload_inventory.json`。

### 1.1 Profile 有效性（Phase 1 校验项）

- 每个 `kernel_id + workload_id` 的 raw dir 下均有 `OPPROF_*/main/<i>/{OpBasicInfo,PipeUtilization,
  ArithmeticUtilization,Memory,L2Cache,MemoryUB,MemoryL0,ResourceConflictRatio}.csv`
  （每 launch 一份，`launches=20`）。
- `captured_op_name` = `main`，来自 `--kernel-name=main` + 实际候选文件路径（命令见 §2 表），
  可追溯到目标 TileLang kernel；非 Cast/Mul/Random 类框架小算子。
- 所有记录 `profile_status = valid`。

### 1.2 调优顺序与理由（Phase 1 排序）

按「实测负担 × 结构收益机会」排序，**不跳过任何 tune workload**：

1. `sincnet-speaker-local-float16`（302 µs，流量最大、一般路径 op 数最多）；
2. `ecg-cnn-dilated-bfloat16`（155 µs，一般路径 + 膨胀重叠窗）；
3. `textcnn-global-float16`（12.7 µs，C4 退化解码路径，绝对量级最小但结构独立）。

## 2. Performance Test Data — baseline（round 0）

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | profile_status | raw_profile_dir |
|---|---|---|---|---:|---|---|---|
| `.../_max_pool1d_kernel.py::_max_pool1d_kernel::main` | `sincnet-speaker-local-float16` | `_max_pool1d_kernel` | `main` | `main` | **301.936** | valid | `perf_opt/profiles/baseline/round0/max_pool1d_kernel_aaec86f2/sincnet-speaker-local-float16_653b8152` |
| 同上 | `textcnn-global-float16` | `_max_pool1d_kernel` | `main` | `main` | **12.710** | valid | `.../textcnn-global-float16_f5206ca3` |
| 同上 | `ecg-cnn-dilated-bfloat16` | `_max_pool1d_kernel` | `main` | `main` | **154.653** | valid | `.../ecg-cnn-dilated-bfloat16_76510648` |

命令模板（每条记录实际执行的命令，写入 `perf_opt/logs/`）：

```bash
msprof op --kernel-name=main --output=profiles/<stage>/round<N>/<candidate>/<workload> \
  --launch-count=20 --warm-up=5 --dump=off \
  --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio \
  /workspace/venv/bin/python3 bench.py --impl <candidate>.py --case <workload_id> --iters 25
```

`msprof op` 全程**串行**；stdout/stderr 统一写入 `perf_opt/logs/`（不在 `perf_opt/` 顶层散落日志）。

### 2.1 baseline 逐 workload 画像（`aiv/vec/mte2/mte3` 单位为 µs，`GM<->UB` 为每核每 launch KB）

| workload | Task | aiv_time | vec | mte2 | mte3 | scalar | GM_to_UB | UB_to_GM | 官方 Ratio(%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sincnet | 301.936 | 281.189 | **205.498** | 45.146 | 33.808 | 111.490 | 842.0 | 584.4 | 13.727 |
| textcnn | 12.710 | 9.860 | 5.961 | 2.584 | 0.868 | 1.148 | 96.0 | 0.75 | 18.334 |
| ecg | 154.653 | 150.634 | **111.112** | 20.386 | 13.930 | 60.405 | 416.0 | 305.5 | 12.229 |

### 2.2 综合口径（`Ratio` 的定义与判读，本任务实测复现）

conductor 给的口径是「带宽 roofline（achieved_bw ÷ 1.75 TB/s）」。本轮按官方 report 的
`tileops.benchmark.msprof_roofline.compute_from_csvs`（文档说明与 `visualize_data.bin`
Method 1 一致）**实测复现**了 report 的 `Ratio(%)`，其定义是：

```text
perf = total_fops / TaskDuration          # total_fops = ArithmeticUtilization 的 aiv_vec_fops 和
AI   = total_fops / total_bytes           # total_bytes = L2 read-miss 数 × cacheline(128B)
peak = min(AI × gm_bw(1.8 TB/s), computility)     # computility = 实测 vec 峰值加权
Ratio(%) = perf / peak
```

关键结论（决定了「70%」对应的真实时延线）：

| workload | AI | AI×1.8 | computility | 模型判定 | 70% 对应 Task |
|---|---:|---:|---:|---|---:|
| sincnet | 20.21 | 36.4 | 3.806 TOps/s | **计算受限** | `105.98M / (0.7×3.806e12)` = **39.8 µs**（若 fops 保持 baseline 153.4M 则为 57.6 µs） |
| ecg | 32.49 | 58.5 | 3.934 TOps/s | **计算受限** | **24.7 µs** |
| textcnn | 2.03 | 3.66 | 5.480 TOps/s | **带宽受限** | `4.194MB / (0.7×1.8e12)` = **3.33 µs** |

> 判读纪律（profile-collection.md §4.1）：池化族的 Ratio 是**比较/归约 op 的记账密度**，不是硬件
> 利用率。这里进一步给出可操作的推论：`Ratio = (fops/T)/computility` ⇒ **结构改造在减少 fops 的
> 同时会拉低分子**——sincnet 实测即为此列（T −37.6%，但 fops 153.4M→106.0M，Ratio 只从 13.7% 升到 14.8%）。
> 因此 Ratio 不是「越少的 op 越好」，而是「同样的 op 数量下更低的时延」。

### 2.3 设计估算 vs 实测偏差行（D-2 回填，`DESIGN.md §1.6.0`）

`DESIGN.md §1.6.0` 对三个 dispatch 代表 workload 给出的估算下界 vs 实测 baseline：

| workload | 设计估算下界 | 实测 baseline Task | 偏差 | 设计已识别的失准项 | 本轮实测归因 |
|---|---:|---:|---:|---|---|
| sincnet | **22.2 µs**（「流量主导」，按 1.26 TB/s HBM） | 301.936 | **13.6×** | §1.6.0 自标「发射项口径 = **未文档化假设**」并提示「发射项可能成为实际主导项」 | 命中：窄向量 op 的发射/地址代价（实测 ≈14.4 ns/窄 op/核）主导，见 §3 |
| textcnn | 3.35 µs | 12.710 | 3.8× | 同上（13 op/单元） | C4 路径的 5 条全平面 pass + 固定开销 |
| ecg | 9.98 µs | 154.653 | 15.5× | 同上 | 同 sincnet（膨胀窗 + 322 op/单元） |

**该偏差行是 §6 `[DESIGN_LIMIT]` 归因的量化证据**（hardware-cost-model.md §3）。

## 3. 现象分析与候选优化点（Phase 2）

### 3.1 根因定位（baseline profile 事实）

- **一般路径（sincnet / ecg）**：`vec` = 205.5 / 111.1 µs，占 `aiv_time` 的 73% / 74%；`scalar`
  = 111.5 / 60.4 µs。结构上每个逻辑单元执行 `2·k·BL + 12` 条宽 **CH 元素**的向量 op：
  `T.serial(k) × T.serial(BL)` 内每个 (kw, owl) 各一条 `vmax` + 一条 `vmul`（NaN 掩码），
  配 1 条标量 index/guard 计算 ⇒ sincnet `2·3·26 = 156` 条窄 op + 12 条宽 op = 168 条/单元，
  全 kernel `3392 × 168 = 570K` 条窄向量 op。实测 ≈ **14.4 ns/窄 op/核**（570K/40 核 ÷ 205 µs），
  与 `CONST-vector-launch-overhead` 的「宽块 ~0.5 µs/op 不可外推到窄 op」方向一致。
- **通道分块尾部的 lane 浪费**：sincnet `C=80`、`CH=64`（`_CH_MAX=64`）⇒ `num_cblk=2`，
  第 2 块 `real_c=16`，但窗口 op 仍按 **CH=64** 跑 ⇒ **37.5% 的 vector lane 是 padding**，且
  `num_logical` 翻倍（3392）。
- **DMA 突发长度受限**：`T.copy(x[nb, cb:cb+ch, s0:s0+LSPAN], ...)` 的行长 = `LSPAN`，
  行间距 = `L_IN`。sincnet `LSPAN=78`（156 B）/ 合并版 `64`（128 B），即每次 DMA 只有
  128–156 B 连续段（见 §5 的 `probe_tr` 实测：同形 80 行 × 128 B、stride 8 KB 的读只有
  **≈0.38 TB/s**，而 `CONST-copy-floor-dma-granularity` 的 16 KB 档为 1.06 TB/s）。
- **C4 路径（textcnn）**：`vec=5.96 µs`，5 条全平面 pass（`vcast`/`reduce_max`/`vcmp`/`vcast`/
  `reduce_min`）+ 2 条 `(CH,1)` pass；`GM_to_UB=96 KB` 与理想 4.19 MB/40 核 = 105 KB 同量级
  （无放大），说明该路径的 DMA 是健康的，瓶颈在 `reduce` 链的 pass 数。
- **`TRAP-transpose-ub-gm-roundtrip` 复核（未命中）**：baseline 的 `GM_to_UB`（842 KB/核）
  = `85 单元 × 9984 B` 恰为 span 载入量（**无额外读**），UB→GM 584 KB/核 高于理想 174 KB/核；
  但 `probe.py variant=drop_out_transpose`（删输出转置、仅计时的探针）实测 294.126 µs
  vs 301.886 µs（**仅 −2.6%**），且 `UB_to_GM` 完全不变 ⇒ **转置不是主因**，不作为主优化点。

### 3.2 本轮候选优化点

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| R1a | 一般路径 37.5% padding lane + `num_logical` 翻倍 | `_CH_MAX` 64 → 128（CH 可等于 C） | `_general_bytes` 容量不等式不变；sincnet C=80、ecg C=128 均可单通道块 | `_CH_MAX_TUNED = 128`，容量不等式驱动 BL 收缩 | Task / `num_logical` / L0 | improved（合并） |
| R1b | C4 路径 5 条全平面 pass、UB 单价高 | 等代价 NaN 追踪：`isnan(reduce_sum(x*x))` 原位平方 | `PL-1.22-maxreduce-nan-tracking` ⑧（origin_task `adaptive_max_pool2d-...-20261015`，verified）；原位平方 32 vs 掩码链 96–128 B/元素 | 删 `eq_mask/eq_f32/nmin`，改 `T.vmul(flat32,flat32,flat32)` + `reduce_sum` + `vcmp(nsum,nsum,"eq")` 直送 `vselect`；`_C4_BYTES_PER_ELEM=4` | Task / `vec` / L0 | improved（合并） |
| R2-a | 一般路径 156 条窄 op/单元 | 折叠 `owl` 循环（位置主序 dilate-max + gather 抽取） | 见 §4 探针（否决） | `probe_dmg.py` | Task | family_no_gain（实测否决，见 §5） |
| R2-b | 同上 | 位置主序 + 3D 转置抽取 | 见 §4 探针 | `probe_pm.py` | Task | family_no_gain（实测否决，见 §5） |
| R2-c | 算子族最优实现对照 | 参照结构 diff（morph ladder 轻量版） | T-1 要求 | 见 §6.2 参照锚定 | — | 已完成（无窗口类参照） |

**回滚规则**：任何分支 L0 失败或本 kernel 任一 tune workload 出现可信回退即回滚该分支。

## 4. Iteration Log

### Round 1（`max_rounds=10` / `max_experiments=30` 内）

**Current best（本轮 base）** = `../_max_pool1d_kernel.py`（baseline，round 0 profile）。

**候选 vs current best 对比表**（B2 结构化回流；数据来自 `perf_records.jsonl` 与 raw profile）

| candidate | workload | L0 | Task Duration(us) | Δ vs best | aiv_time | vec | mte2 | mte3 | GM_to_UB(KB) | UB_to_GM(KB) | 官方 Ratio(%) | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | sincnet | pass | 301.936 | — | 281.189 | 205.498 | 45.146 | 33.808 | 842.0 | 584.4 | 13.727 | best |
| R1a `chfull` | sincnet | pass | **187.964** | **−37.7%** | 185.149 | 113.350 | 55.270 | 34.380 | 520.0 | — | — | improved |
| R1b `c4sq` | sincnet | pass | 302.016 | +0.03% | 281.320 | 205.498 | 45.104 | 33.867 | 842.0 | — | — | config_no_gain（该路径不受 R1b 影响，符合预期） |
| baseline | textcnn | pass | 12.710 | — | 9.860 | 5.961 | 2.584 | 0.868 | 96.0 | 0.75 | 18.334 | best |
| R1a `chfull` | textcnn | pass | 12.740 | +0.2% | 9.870 | 5.961 | 2.562 | 0.867 | 96.0 | — | — | config_no_gain |
| R1b `c4sq` | textcnn | pass | **11.770** | **−7.4%** | 9.131 | 5.284 | 2.586 | 0.866 | 96.0 | — | — | improved |
| baseline | ecg | pass | 154.653 | — | 150.634 | 111.112 | 20.386 | 13.930 | 416.0 | 305.5 | 12.229 | best |
| R1a `chfull` | ecg | pass | **113.002** | **−26.9%** | 107.632 | 73.818 | 25.267 | 21.085 | 480.0 | — | — | improved |
| R1b `c4sq` | ecg | pass | 154.533 | −0.08% | 150.562 | 111.112 | 20.313 | 13.967 | 416.0 | — | — | config_no_gain（同 sincnet） |

- 分支文件：`perf_opt/_max_pool1d_kernel_round1_chfull.py`（R1a）、
  `perf_opt/_max_pool1d_kernel_round1_c4sq.py`（R1b）。
- R1a 实测说明：`num_logical` 3392 → 2080（CH=80 单通道块 + BL 26→21），
  padding lane 归零；`vec` 205.5 → 113.4 µs。**L0/L1 全过、三 dtype 逐位精确。**
- R1b 实测说明：`eq_mask/eq_f32/nmin` 三平面消失，`vec` 5.96 → 5.28 µs；
  R1a 使 textcnn 的 `_CH_MAX` 变 128 时 C4 足迹（eb+4 档）仍可容纳 CH=128，
  故合并版把它的 C4 路径保住了（见 Round 2）。

### Round 2（合并版）

R1a 与 R1b 作用于**不同路径**（一般网格 vs C4），因此按「单点已证明有效后再组合」合入：

- 合并文件 `perf_opt/_max_pool1d_kernel_round2_merged.py`。
- **该 kernel 全部 tune workload 的合并非回退检查 + smoke 精度**：

| merged_candidate | workload | current_best_us | merged_us | precision | status |
|---|---|---:|---:|---|---|
| `round2_merged` | sincnet | 187.964 | 188.394 | bit-exact（L0/L1） | pass（−0.2%，噪声内） |
| `round2_merged` | textcnn | 11.770 | 11.860 | bit-exact | pass（+0.8%，噪声内） |
| `round2_merged` | ecg | 113.002 | 113.892 | bit-exact | pass（+0.8%，噪声内） |
| smoke × 3 dtype | `smoke-1x8x32-k3s2p1-{fp16,bf16,fp32}` | — | — | `exact=True max_diff=0.000e+00` | pass（仅精度，不参与性能门禁） |

`round2_merged` 成为新的已合并 current best（三个 workload 均无可信回退，全部落在 ±1% 噪声带内）。

### Round 3（收束 + `phase=final` 重采）

- 选择 `perf_opt/_max_pool1d_kernel.py` = `round2_merged` 的逐字节副本
  （SHA256 `f8ae961c...`，与所有 `phase=final` 记录的 `artifact_sha256` 一致）。
- 在**最终文件**上对全部 3 个 tune workload 重新采集 `phase=final` 记录（`perf_records.jsonl`，
  append-only）；smoke 仅做精度回归。

## 5. 结构探针（决定性实验，`probe_*`；非候选分支）

这些探针**不进入 `perf_records.jsonl`**（非 tune workload 的候选记录），其数据用于
「折叠 owl 循环」这一类结构方向的实测裁决与 §6 的地板归因：

| 探针 | 结构 | 结果 | 结论 |
|---|---|---|---|
| `probe.py variant=base` | baseline 结构复刻 | 301.886 µs（vs 301.936） | harness 自校验通过 |
| `probe.py variant=drop_out_transpose` | 删输出转置（仅计时） | 294.126 µs（**−2.6%**），`UB_to_GM` 不变 | 转置不是主因 |
| `probe_dmg.py` | 通道主序「整 span dilate-max + `T.gather` 抽取」 | **23328 µs**（vec 480 / **scalar 22878**） | `tensor.extract_slice` 的目标行距≠源行距时，bufferization 把它降级为 **UB→GM→UB 物化**并被标量化 ⇒ 该形态不可用 |
| `probe_pm.py` | 位置主序 dilate-max（全部切片连续）+ 3D 转置抽取 + `gather`，BL=13 逐位精确 | **998.92 µs**（vec 771） | 单点列布局改造后仍比 baseline 慢 ⇒ 该结构族否决 |
| `probe_tr.py` | 纯 DMA/转置隔离 | `copy` 55.461 µs vs `trans` 51.311 µs；`trans` 的 `vec` 仅 6.594 µs / 2080 次 (80,64)→(64,80) 转置 | **UB 内 2 轴转置本身近乎免费**；`copy` 档 `mte2=51 µs` 说明**瓶颈是同形 80×128B 行距 8KB 的读只有 ≈0.38 TB/s** |
| `probe_deint.py` | `T.deinterleave` 可行性 | `channel_nums=2` 语义正确；`channel_nums=3` → `bishengir-compile` SIGABRT | 相位数 >2 的相位分解不可用 |

**换轴分析（证伪协议 §2.4）**：诊断驱动项是「窄 op 数 × 发射代价」而非布局轴选择；已按
`layout.md` 评估「原生布局 + 最内连续轴」vs「核内重排布局」两条路线，并对后者做了两个独立
实测分支（DMG / PM），均被实测否决 ⇒ 不构成「未穷举即否证」。

## 6. Final Summary

### Final Performance Test Data（round-1 / Ascend910B3 — 已被 §13.8 取代，数字原样保留作历史）

最终候选：`perf_opt/_max_pool1d_kernel.py`，`candidate_id = _max_pool1d_kernel`，
SHA256 `f8ae961c015f0e9b71d4cd6f6492333b40ca54d3adb975e3516f9301dadc7a54`
（全部 `phase=final` 记录的 `artifact_path`/`artifact_sha256` 与该文件一致）。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---:|---:|---|
| tileops/kernels/pool/max_pool1d/max_pool1d_kernel/_max_pool1d_kernel.py::_max_pool1d_kernel::main | sincnet-speaker-local-float16 | 301.936 | 188.454 | _max_pool1d_kernel |
| tileops/kernels/pool/max_pool1d/max_pool1d_kernel/_max_pool1d_kernel.py::_max_pool1d_kernel::main | textcnn-global-float16 | 12.710 | 11.620 | _max_pool1d_kernel |
| tileops/kernels/pool/max_pool1d/max_pool1d_kernel/_max_pool1d_kernel.py::_max_pool1d_kernel::main | ecg-cnn-dilated-bfloat16 | 154.653 | 113.182 | _max_pool1d_kernel |

### 逐 workload 分解与 profile 证据（final）

每个 tune workload 的 baseline / final 时延与官方 Ratio 对照、对应的 raw profile 目录；
`kernel_id` 与 `perf_opt/workload_inventory.json` 逐字符一致（无缩写、无「同上」）：

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id | Δ | 官方 Ratio baseline→final | raw profile（final） |
|---|---|---:|---:|---|---:|---|---|
| tileops/kernels/pool/max_pool1d/max_pool1d_kernel/_max_pool1d_kernel.py::_max_pool1d_kernel::main | sincnet-speaker-local-float16 | 301.936 | **188.454** | `_max_pool1d_kernel` | **−37.6%** | 13.727% → **14.776%** | `perf_opt/profiles/final/round3/max_pool1d_kernel_aaec86f2/sincnet-speaker-local-float16_653b8152` |
| tileops/kernels/pool/max_pool1d/max_pool1d_kernel/_max_pool1d_kernel.py::_max_pool1d_kernel::main | textcnn-global-float16 | 12.710 | **11.620** | `_max_pool1d_kernel` | **−8.6%** | 18.334% → **20.053%** | `perf_opt/profiles/final/round3/max_pool1d_kernel_aaec86f2/textcnn-global-float16_f5206ca3` |
| tileops/kernels/pool/max_pool1d/max_pool1d_kernel/_max_pool1d_kernel.py::_max_pool1d_kernel::main | ecg-cnn-dilated-bfloat16 | 154.653 | **113.182** | `_max_pool1d_kernel` | **−26.8%** | 12.229% → **15.306%** | `perf_opt/profiles/final/round3/max_pool1d_kernel_aaec86f2/ecg-cnn-dilated-bfloat16_76510648` |

final 逐 workload 分解（µs）：

| workload | Task | aiv_time | vec | mte2 | mte3 | scalar | GM_to_UB(KB) | UB_to_GM(KB) | total_fops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sincnet | 188.454 | 185.486 | 113.350 | 55.482 | 34.721 | 53.919 | 520.0 | 682.5 | 105,976,320 |
| textcnn | 11.620 | 8.925 | 5.284 | 2.526 | 0.886 | 0.913 | 96.0 | 0.75 | 8,530,432 |
| ecg | 113.182 | 107.454 | 73.818 | 25.261 | 21.188 | 33.921 | 480.0 | 574.25 | 68,144,640 |

### 6.2 性能目标达成判定（`DESIGN.md §11`，合格线：3 个 full workload 全部 ≥ 70%）

| workload | 目标 Ratio | 最终 Ratio | 达标 | 70% 对应 Task | 最终 Task | 差距 | 判定依据 |
|---|---:|---:|---|---:|---:|---:|---|
| sincnet | ≥70% | 14.776% | ✗ | 39.8 µs | 188.454 | 4.7× | **实测 DMA 项 `mte2=55.482 µs` 单独已 > 39.8 µs**（§5 `probe_tr`） |
| textcnn | ≥70% | 20.053% | ✗ | 3.33 µs | 11.620 | 3.5× | **3.33 µs < 空 kernel Task 地板 3.37 µs**（`CONST-msprof-task-floor-910B3`）⇒ 口径不可达 |
| ecg | ≥70% | 15.306% | ✗ | 24.7 µs | 113.182 | 4.6× | 实测 `mte2=25.261 µs` 单独已 > 24.7 µs |

**性能目标未达成。** 本任务如实报告：流程完整执行（3 个 tune workload 均有结论、有最终全量
记录、smoke 精度通过），但**流程完成 ≠ 性能目标达成**。已产出的提升为
sincnet −37.6% / ecg −26.8% / textcnn −8.6%，均为逐位精确（`max_diff=0.000e+00`）。

### 6.3 中止原因

`success` 未达成；`budget_exhausted` 未触发（3 轮 ≤ `max_rounds=10`，4 个实验分支 + 5 个探针
≤ `max_experiments=30`）；`blocked` 未触发（剩余结构候选经实测否决，非不可实现）；
**实际中止 = `plateau`**（连续轮次无有效结构增益 + 主要结构候选已实测验证），并且按 T-1
完成了**参照结构 diff 检查**（§6.4）。因两个 §1 触发条件同时满足，另产出
`perf_opt/perf_feedback.md` 并附 `[DESIGN_LIMIT]` 信号。

### 6.4 参照锚定（T-1 morph ladder 轻量版，定稿前强制）

检索同族已知最优实现（`pattern-library/cases.md` 参考实现集 + `examples/` 同类）：

| 参照 | 路径 | 结构形态 | 是否可移植 |
|---|---|---|---|
| 同族全局池化 Stage 4 | `examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d/adaptive_max_pool2d_kernel/perf_opt/`（CASE-pool-adaptivemaxpool2d-migration，320.83→18.86 µs） | **整平面/全局**（`L_out==1`）+ 扁平归约 | **不可移植到窗口类**：其收益来自「整平面单次宽归约」，而窗口类每个输出对应一个变长/重叠窗口，不存在整平面归约；且其 16–19 µs 地板带属 `L_out=1` 域的固定开销带（conductor 已显式提示不可外推）。**textcnn（C4）本就属该域，已按其结构对齐**（单次整线归约 + 等代价 NaN 追踪，见 R1b）。 |
| 同族扁平空间轴归约 | `PL-1.23-flat-spatial-axis-reduce`（origin_task `adaptive_avg_pool2d-...-20261014T172114Z`） | 合并空间轴成宽最内轴免转置 | 不可移植：要求「每个输出 bin 是矩形子区域」，窗口类的输出块在合并轴上不连续（`D=…` 分块 + 窗口重叠）。 |
| 同族窗口池化 2D 迁移 | `examples/TileOPs/tileops/kernels/pool/max_pool2d/max_pool2d_kernel/` | **无 `perf_opt/`**（第 4 个 pool 迁移只到集成） | **不存在窗口类 Stage 4 参照实现** ⇒ 窗口类「当前工具链内的最优结构」在库内无先例可比。 |
| 本算子迁移前作 | `examples/max_pool1d/_max_pool1d_kernel/`（同设计族，`ddf4255a/417dae9` 工具链） | 同结构 | 同族同构，无结构差异可取。 |

**结论**：窗口类（general 路径）**未找到可移植的同族最优参照** ⇒ 按规范显式标注
「未找到参照，天花板结论置信度降级」；仅 textcnn（C4 域）有可对照参照且已对齐。
地板结论限定于**冻结设计族**（`T.serial(k) × T.serial(BL)` 窄向量 op 结构 + 行距 8 KB 的
128–156 B 突发读），不作绝对地板断言。

### 6.5 已知残留（如实披露）

- `perf_opt/_max_pool1d_kernel.py` 相对 baseline 的改动仅为：`_CH_MAX` 常量值与 C4 路径的
  NaN 追踪实现 + `_C4_BYTES_PER_ELEM` 常量 + 相应注释。**无死缓冲 / 无用声明 / 仅初始化无消费者
  的临时量**；`phase=final` 采集前已完成清理，测量后未再修改文件（SHA256 与 final 记录一致）。
- 未被采纳的探针文件（`probe*.py`）保留在 `perf_opt/` 顶层用于溯源，不属于交付候选。

## 7. Skill Retrospective

### 7.1 skill 流程问题

| # | 问题 | 现象 | 建议 |
|---|---|---|---|
| 1 | **「Ratio」目标口径在 conductor 与官方 report 之间不一致，且未在调度 prompt 中给出反解公式** | 调度 prompt 给的是「带宽 roofline = achieved_bw ÷ 1.75 TB/s」；实测复现官方 `compute_from_csvs` 后发现 sincnet/ecg 在模型中是**计算受限**（`Ratio = perf/computility`），textcnn 才是带宽受限，且 `Ratio` 的分子是 **执行到的 vector fops**（会随结构简并而下调） | 该条落为流程建议（C 类）：`signal-registry` 或 optimize 场景 prompt 应给出「目标 Ratio 的反解公式 + 分子是否含 fops/字节口径」，否则执行端会优化错目标（本任务即出现「降时延 37.6% 只换来 Ratio +1.05pt」） |
| 2 | **Phase 1 缺少「目标线 vs 平台地板」的前置机械判据** | 本任务直到 §6.2 才用 `CONST-msprof-task-floor-910B3` 反解出 textcnn 的目标线 3.33 µs **低于**空 kernel 地板 3.37 µs | `perf-feedback.md §1` 已有该机械前置判据，应前移到 Phase 1 的 baseline 之后（零成本、一行比较），避免把不可达 workload 排进迭代 |
| 3 | **窗口类算子缺少「折叠输出位置循环」的可用原语组合** | 四个独立实测分支（DMG/PM/deint/tr）说明：`T.gather` 约 1.8 ns/元素、`T.deinterleave` 仅支持 2 相、`extract_slice` 行距不匹配即降级为 UB→GM→UB 物化 | 该条为 capability gap（见 7.3），非流程问题 |

### 7.2 value point proposals（含 vp_type 与证据三件套）

| vp_id | vp_type | 结论 | evidence（实测） | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-P1 | **D** | 通道分块尾部 lane 浪费是**纯 Vector** 窗口/归约类算子的第一杠杆：`CH=min(C,_CH_MAX)` 而非 `64`。sincnet（C=80）：`num_logical` 3392→2080、窗口 op 的 padding lane 37.5%→0，Task 301.936→187.964 µs（−37.7%，逐位精确）；ecg（C=128）：154.653→113.002 µs（−26.9%）。同族 `PL-1.4-tiling-heuristic` 的容量不等式**不变**，BL 自动收缩 | `examples/TileOps/.../max_pool1d_kernel/perf_opt/{perf_records.jsonl round1 chfull, opt_log.md §4}` | `repro-missing`（最小形态 = 单常量 + 现有 bench） | tilelang `0.1.2+c3c39ba` + CANN 9.0.0 + Ascend910B3 / 2026-10-16 |
| VP-P2 | **D** | 窗口类（general）路径的实测地板由 **DMA 突发长度**决定而非带宽上限：行长 = `LSPAN`、行距 = `L_IN` 时，sincnet 的 80×128 B / stride 8 KB 读实测仅 **≈0.38 TB/s**（`mte2=51 µs / 520 KB/核`），而 `CONST-copy-floor-dma-granularity` 的 16 KB 档为 1.06 TB/s。**`LSPAN` 是目前设计里被当作「halo 度量」而非「DMA 粒度」的量** | `perf_opt/probe_tr.py` + `logs/probe_tr_{copy,trans}.log` + `profiles/probe/tr_*` | `perf_opt/probe_tr.py`（自包含、可重跑；头部说明） | 同上 |
| VP-P3 | **D** | npuir `T.gather`(vgather) 实测 **≈1.8 ns/元素**（2 次 (80,21)←(80,61) gather 使 `vec` 增加 320 µs / 52 单元）；`T.deinterleave` 仅支持 `channel_nums=2`（`=3` 触发 `bishengir-compile` SIGABRT）⇒ **「用 gather/deinterleave 把窗口类改成宽归约」在当前工具链不可行** | `perf_opt/probe_dmg.py`、`probe_pm.py`、`probe_deint.py`（后者的 channel_nums=3 崩溃输出见 `logs/probe_deint.log`） | `perf_opt/probe_deint.py`（自包含最小复现） | 同上 |
| VP-P4 | **D** | `tensor.extract_slice` 源/目标**行距不一致**（如 `(80,61)` 取自 `(80,64)`）会被 bufferization 降级为 **UB→GM→UB 物化并标量化**：sincnet 单分支 Task 301.9→**23328 µs**（`scalar=22878 µs`）。⇒ 结构改造必须保证每个 region 操作数与其 buffer **行距一致（整宽行区间）** | `perf_opt/probe_dmg.py` + `logs/probe_dmg_*.log` | `perf_opt/probe_dmg.py --case sincnet --bl 21 --mode static` | 同上 |
| VP-R1 | **R** | Stage 4 的 Ratio 类目标应在 Phase 1 做一次「目标线 vs 空 kernel Task 地板 + 必需流量地板」的前置比较（`perf-feedback.md §1` 的机械前置判据），并把不可达 workload 尽早标记 | 本任务 §6.2：textcnn 70% ⇒ 3.33 µs < 3.37 µs 地板 | — | 同上 |
| VP-C1 | **C** | 案例索引续档：`CASE-pool-maxpool1d-migration` 的 Stage 4 段（R1a/R1b + 两个被否决的结构族 + 窗口类「无参照实现」结论 + `perf_feedback.md`） | 本文件 | — | 同上 |

（已回写的 pattern-library：本轮结论已写入 `perf_feedback.md`；pattern-library 增量条目与
repro 由任务终态 `tilelang-skill-evolver` 蒸馏时以本表为输入合入。）

### 7.3 capability gaps

| # | 缺失能力 | 编译器层 | 证据 |
|---|---|---|---|
| 1 | 宽区域「跨步抽取」（decimation）无可用向量原语：`T.gather` 是逐元素实现（≈1.8 ns/元素），`T.deinterleave` 仅 2 相，`T.transpose` 抽取需 3D 视图（`vtranspose` 仅 2 轴） | Frontend API（`tilelang/language/customize_npuir.py` 的 `npuir_gather` / `npuir_deinterleave`） | `probe_dmg.py` / `probe_pm.py` / `probe_deint.py` |
| 2 | `vdeinterleave` 的 `channel_num=3` 直接使 `bishengir-compile` 崩溃（backend 段错） | BishengIR | `probe_deint.py` case B 输出（NPU IR 已生成，`bishengir-compile` rc=-6） |
| 3 | bufferization 把「源/目标行距不一致的 `tensor.extract_slice`」降级为 UB→GM→UB 物化，且该物化被标量化（无向量化兜底） | TileLangIR pass / bufferization | `probe_dmg.py`（scalar 22.9 ms） |

## 8. Autotune Log

未使用 autotune：本轮的结构搜索空间（CH/BL/分核）由容量不等式闭式给出（`_choose_ch_bl`），
且结构性候选已由定向探针裁决；按 `autotune.md`「结构性分支通过正确性后再 coarse search」的口径，
本轮未进入参数精搜阶段（结构方向本身被否决）。CH/BL 的取值点记录于 §4 与 §1.2。

---

# 13. Round-4 continuation（2026-10-21，Ascend910B2 / tilelang f947fb1）

> 本节是本轮（第二轮有界调优，`mode=full`、`goal=best_effort`）的全部证据。
> **器件与工具链再次变更**：910B3 → **910B2**（Vector 核 40 → **48**），tilelang
> `c3c39ba` → **`f947fb1`**。上轮台账的绝对时延一律不复用；本节所有比较都在本机同口径重采
> （`msprof op`，20 launch median，`--kernel-name=main --launch-count=20 --warm-up=5 --dump=off`，
> `flock /tmp/tilelang-msprof.lock` 串行）。
>
> **本轮结论（一句话）**：round-1 `perf_feedback.md` 的「窗口域无可移植同族最优参照 / 冻结结构族内
> 不可达」被**实测推翻**——两个优先 workload 各换掉结构族后分别降到 **61.55 µs（−61.3%）** 与
> **37.23 µs（−60.5%）**，逐位精确。

## 13.0 版本戳（本轮实测）

| Item | 本轮 value |
|---|---|
| npu | **Ascend910B2** ×1（`torch.npu.get_device_name(0)`）；`NPUUtils.get().get_aicore_num()` = **24** → 纯 Vector 核 **48** |
| UB | 196 608 B / AIV |
| tilelang | `0.1.2+f947fb18c3022f05bca04af74d2f8222eb3980e7`（git `f947fb1`，dev-root `build/`） |
| CANN / bishengir | 9.0.0 / **1.1.0 `428ab8fdab46`**（llvm 19.1.7 e4ce5939f6e1） |
| torch / torch_npu | `2.9.0+cpu` / `2.9.0.post8` |
| `TILELANG_ASCEND_MODE` | Developer |
| `kb_stale_count` | **391**（预注入条目全部按线索处理，凡作决策依据者均在本节给出本机重验数据） |
| 空 kernel Task 地板（本轮本机实测） | **2.79 µs**（48 核，`probe_floor_r4.py --variant empty`） |
| DMA 突发地板（本轮本机实测） | 21 MB 读+写、行距 8 KB：**126 B 行 → mte2 55.19 µs（0.38 TB/s）**；**1 KiB 行 → mte2 20.29 µs（1.03 TB/s）**（`rows63` / `rows512`） |

`probe_floor_r4.py --variant row4096`（整条 8 KiB 行 × 80 通道）**无法测量**：该 UB 足迹
（640 KB）超出 192 KB 容量，进程内直接失败 ⇒ 本轮的 DMA 突发曲线只由 `rows63` / `rows512` 两点给出。

## 13.1 同设备起点（本轮 round 0 = round-1 交付文件，本机重采）

`perf_opt/_max_pool1d_kernel_round1_frozen.py`（sha `f8ae961c…`，= round-1 交付文件逐字节副本）：

| kernel_id | workload_id | Task(us) | aiv | vec | mte2 | mte3 | scalar | GM_to_UB(KB) | UB_to_GM(KB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `.../_max_pool1d_kernel.py::_max_pool1d_kernel::main` | `sincnet-speaker-local-float16` | **159.01** | 156.083 | 95.833 | 46.847 | 29.574 | 44.226 | 430.0 | 564.75 |
| 同上 | `textcnn-global-float16` | **10.72** | 7.826 | 4.421 | 2.088 | 0.699 | 1.076 | 80.0 | 0.6 |
| 同上 | `ecg-cnn-dilated-bfloat16` | **94.34** | 90.078 | 61.231 | 21.107 | 17.821 | 28.569 | 400.0 | 478.0 |

> 910B3 台账（188.45 / 11.62 / 113.18）**不可比**：核数 40→48、工具链 stamp 不同。本表才是本轮基准。

## 13.2 Current Phenomena（本轮起点 profile 事实 + 可行性复验）

- **P1（sincnet，159.01 µs）**：`vec 95.833` 占 `aiv_time` 的 61%；一般路径每逻辑单元发
  `2*k*bl = 126` 条**窄（1×CH=80）** op，全 kernel `2080 × 126 = 262 080` 条，实测
  **16.3 ns/窄 op/核**（`95.833 µs × 48 / 282 880`）——op **条数**而非字节数是主因。
  `mte2 46.847 µs` 对 20.6 MB 读 = **0.44 TB/s**（行 = `LSPAN=63` → 126 B，行距 8 KB，与
  `rows63` 探针 0.38 TB/s 同档）。**新发现**：`UB_to_GM = 564.75 KB/核 × 48 = 27.1 MB`，而真实
  输出仅 **6.99 MB ⇒ 写侧放大 3.88×**（store 行 = `BL*2 = 42 B`）。
- **P2（ecg，94.34 µs）**：`vec 61.231 + scalar 28.569`；一般路径 `2*k*bl = 140` 条/单元 ×
  `1184` 单元 = **165 760** 条窄 op。读侧 `19.2 MB`（真实 8.39 MB ⇒ 2.29×，行 = `LSPAN=35`
  → 70 B）；写侧 `22.94 MB`（真实 4.19 MB ⇒ **5.47×**）。
- **P3（textcnn，10.72 µs）**：`UB_to_GM = 0.6 KB/核 × 48 = 0.029 MB` vs 真实输出 0.033 MB
  ⇒ **1.0×，无 1 元素行放大**。conductor 提示的 sibling `VP-D5`（扁平 store 拿回 21–25×）在本
  op 上**不适用**（本 op 的 C4 路径写的是 `(real_c,1)` 但 `ch≥64` ⇒ 行宽 = `ch*eb ≥ 128 B`）。
  剩余成本是 5 条全平面 pass（`vcast`/`reduce_max`/`vmul`/`reduce_sum`/`vcmp`）。
- **P4（本轮的两个代数事实，先于 API 判定）**：
  1. **sincnet `s=3,k=3,p=0,d=1`** ⇒ 窗口就是**连续不重叠的三元组** `[3ow, 3ow+3)` ⇒ 整个池化 =
     「把 L 看成 `(out_l, 3)`，对 size-3 轴做归约」。`k*out_l = 4095 ≤ 4096`，**块跨度无 halo、
     无需 clamp**。
  2. **ecg `s=2,p=2,d=2`** ⇒ 每个 tap `iw = 2ow-2+2t` **恒为偶数** ⇒ 整窗落在输入偶相位；
     在相位平面 `ph[q]=x[2q]` 上退化为 **stride 1、k=5 的普通滑窗**（VP-P3 纪律，见 11.2）。
  3. **反例（必须先说清）**：`s_w == 3` 无任何 2 相分解（3 ∤ 2^k）；`s_w == 3` 的 3 相抽取
     仍是硬缺口（`cn=3` SIGABRT，sibling 已双重实测）⇒ sincnet **不能**走相位路线，只能走 P4.1。

## 13.3 Candidate Optimization Points（本轮候选）

| opt_id | 目标现象 | 优化点（单一主改动） | 判断依据（实测） | 改法 | 状态 |
|---|---|---|---|---|---|
| **R4-a** | P2 | **2 相 deinterleave 行批处理**（相位域宽 op）：span→`T.deinterleave(cn=2)`→transpose→`(pp,ch)` 相位块，每 tap **1 条宽 op** | 代数（P4.2）+ `probe` 逐位精确 + sibling `VP-D4` 同族 | 新增 `_phase_plan` + `_func_phase` | **采纳（ecg 主攻）** |
| **R4-b** | P1 | **组归约**（`s_w==k_w`）：span→transpose→`(bl,k,ch)`→`T.reduce_max(dim=1)`，每单元 1 次宽归约替代 `2*k*bl` 条窄 op | 代数（P4.1）+ `probe_sinc3_r4.py` 逐位精确 35.79 µs vs 同 tile 窄 op 90.42 µs | 新增 `_group_plan` + `_func_group` | **采纳（sincnet 主攻）** |
| R4-c | P3 | 退化路径（C4/identity）**按实际路径的足迹**重选 CH，而非沿用一般路径的 CH | `_global_bytes` 每元素 `eb+4` vs 一般路径 `2eb+9` ⇒ textcnn 可 CH=128（2 块）而非 64（4 块） | `_plan` 内 `path in (identity,global)` 时重导 ch | **采纳** |
| R4-d | P1 | 一般路径加大 BL（45→64）拿长 DMA 突发 | `rows63` 0.38 vs `rows512` 1.03 TB/s | — | **不采纳**（被 R4-b 完全覆盖：组归约同 tile 下 35.79 vs 90.42） |
| R4-e | P1/P2 | 用 `vmax` 的 NaN 传播替代显式掩码 | `probe_vmax_nan_r4.py`：**fp16 传播 / bf16 丢弃（−inf）/ fp32 传播** ⇒ 跨 dtype 不一致 | — | **否决（实测）**，保留显式掩码 |
| R4-f | P1/P2 | 组归约改「最内层 size-3 归约」（`(ch,bl,k)` + `dim=2`，省两次 transpose） | 逐位精确但**被标量化**：`Task 2234.01 µs`，`scalar 2215.7` | — | **否决（实测）**，改用中间轴 `dim=1` |
| R4-g | P1 | 组归约用 bf16 直接归约 | `T.reduce_*` 无 bf16 形态（与 round-1 C4 路径同一缺口）| — | **否决（编译失败）** |

## 13.4 Iteration Log（B2 结构化回流：候选 vs current best）

### Round 4 分支表

| candidate | workload | L0 | Task(us) | vs best | aiv | vec | mte2 | mte3 | scalar | GM_to_UB(KB) | UB_to_GM(KB) | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 起点 `_round1_frozen` | sincnet | pass | 159.01 | — | 156.08 | 95.83 | 46.85 | 29.57 | 44.23 | 430.0 | 564.75 | best |
| `_round4_phase`（R4-a） | sincnet | pass | 159.16 | +0.1% | 156.68 | 95.83 | 47.11 | 29.64 | 44.36 | 430.0 | 564.75 | config_no_gain（预期：一般路径未改） |
| 起点 | ecg | pass | 94.34 | — | 90.08 | 61.23 | 21.11 | 17.82 | 28.57 | 400.0 | 478.0 | best |
| `_round4_phase`（R4-a，tile 128×40） | ecg | pass | **39.81** | **−57.8%** | 36.40 | 18.93 | 17.44 | 9.20 | 3.04 | 288.0 | 227.5 | improved |
| `_round4_phase`（R4-a，tile 64×64 = 交付 tile） | ecg | pass | **37.34** | **−60.4%** | 33.83 | 20.59 | 12.51 | 6.98 | 8.05 | 264.0 | 174.6 | **winner** |
| 起点 `_round1_frozen` | textcnn | pass | 10.72 | — | 7.83 | 4.42 | 2.09 | 0.70 | 1.08 | 80.0 | 0.6 | best |
| `_round4_phase` | textcnn | pass | 10.43 | −2.7% | 7.71 | 4.42 | 2.06 | 0.73 | 0.97 | 80.0 | 0.6 | 噪声内 |
| **`_round4_globalch`（R4-c）** | textcnn | pass | **9.15** | **−14.6%** | 7.21 | 4.67 | 1.47 | 0.34 | 0.93 | 96.0 | 0.8 | **winner** |
| **`_max_pool1d_kernel`（R4-a+b+c，交付）** | sincnet | pass | **61.55** | **−61.3%** | 57.89 | 38.71 | 13.88 | 14.50 | 19.27 | 430.0 | 290.8 | **winner（合并）** |
| 同上 | ecg | pass | **37.23** | **−60.5%** | 33.52 | 20.59 | 12.28 | 7.02 | 8.06 | 264.0 | 174.6 | **winner（合并）** |
| 同上 | textcnn | pass | **9.85** | **−8.1%** | 7.53 | 4.55 | 1.85 | 0.57 | 0.95 | 88.0 | 0.7 | **winner（合并）** |

分支文件：`_max_pool1d_kernel_round4_phase.py`（R4-a + R4-b）、
`_max_pool1d_kernel_round4_globalch.py`（+ R4-c）、交付 `_max_pool1d_kernel.py`（= globalch 内容，
sha `2c483e09d5a9496b44505d70c63237ac5efe5de2a2643ef17dc482d2e0853480`）。

### 结构探针（决定性实验，不进 `perf_records.jsonl`）

| 探针 | 结构 | 结果（同机同口径） | 结论 |
|---|---|---|---|
| `probe_sinc3_r4.py --variant reduce3 --bl 64` | transpose + `(bl,k,ch)` + `reduce_max(dim=1)`（**无 NaN 机制**） | **35.79 µs**，`vec 16.09 / mte2 14.29 / mte3 14.61`，逐位精确 | 中间轴归约 = 宽 op，可行 |
| `probe_sinc3_r4.py --variant last --bl 64` | `(ch,bl,k)` + `reduce_max(dim=2)`（最内层组归约） | **2234.01 µs**，`scalar 2215.69`，逐位精确 | **被标量化** ⇒ 该形态否决 |
| `probe_sinc3_r4.py --variant narrow --bl 64` | round-1 一般结构仅把 BL 加到 64（无 NaN 追踪） | **90.42 µs**（`vec 60.17 / scalar 36.22`） | 单靠加大 BL 只得 1.76×；组归约在同 tile 下再快 2.5× |
| `probe_vmax_nan_r4.py` | `vmax` 的 NaN 传播（fp16/bf16/fp32） | fp16 **NaN 胜**；bf16 **NaN 丢**（得 −inf）；fp32 NaN 胜 | 跨 dtype 不一致 ⇒ **不可依赖**，显式掩码保留 |
| `probe_floor_r4.py` | 地板 | empty **2.79 µs**；rows63 mte2 55.19（0.38 TB/s）；rows512 mte2 20.29（1.03 TB/s） | DMA 突发长度是一等 tiling 判据 |

## 13.5 Tile 扫描（Autotune Log）

扫描点为 autotune 数据（`run_phase_sweep_r4.sh` / `run_group_sweep_r4.sh`，`--check` 保证逐位精确），
按 sibling 的约定**只进本节表格、不进 `perf_records.jsonl`**。

### 13.5.1 ecg 相位路径 `(CH, bl)`

| (CH, bl) | pp / rows | 逻辑单元 | Task(us) | 备注 |
|---|---|---:|---:|---|
| (128, 40) | 48 / 69 | 416 | 39.81–39.97 | 最宽 tile（lanes 5120） |
| **(64, 64)** | 72 / 77 | 512 | **37.34** | **winner**（bl 更长 ⇒ 读 288 B / 写 128 B 行） |
| (64, 48) | 56 / 93 | 704 | 未单独测（L0 通过） | — |
| (64, 80) / (64, 96) / (128, 56) / (128, 64) | — | — | 不可用 | 超出后端 UB 预算（152–382 KB naive 不等，非线性） |

规则（`_phase_plan`）：**在「lanes ≥ 0.8×最大 lanes」的可行集合里取最大 bl，再取最大 ch**
（bl ≤ 64）。这条规则由本节实测点确定：lanes 最大点（128,40）不是最快点。

### 13.5.2 sincnet 组归约路径 `(CH, bl)`

| (CH, bl) | lspan | lanes | naive bytes | 结果 |
|---|---|---:|---:|---|
| (80, 64) | 192 | 5120 | 235 520 | **可行且逐位精确 ⇒ winner** |
| (80, 48) | 144 | 3840 | — | 可行且逐位精确（未单独计时） |
| (40, 128) | 384 | 5120 | 235 520 | **设备 fault**（同样字节数下 ch*eb=80 B 不满足 32 B lane 粒度） |
| (16, 160) | 480 | 2560 | 107 520 | **设备 fault**（同上，32 B lane 粒度不足/后端静默超配） |
| (80, 96)/(80, 80) | — | — | 294–353 KB | 超后端 UB 预算 |

规则（`_group_plan`）：**lanes 最大、并列取更大 ch**；硬约束 = `ch` 整除 C（排除 partial c-block）、
`ch*eb % 32 == 0`（否则 VEC “UB address not aligned” fault）、`bl ≤ 96`、非 bf16。

## 13.6 op 数 / GM 字节记账（work conservation，必做项）

### 13.6.1 op 条数（每逻辑单元 × 逻辑单元数）

| workload | 结构 | 逻辑单元 | 每单元 op | 全 kernel op 数 | 窄（1×CH）op 数 | 单 op 宽度 |
|---|---|---:|---|---:|---:|---|
| sincnet | 一般（round-0） | 2080 | `2*3*21 + 10 = 136` | 282 880 | **262 080** | 80 lanes |
| sincnet | **组归约** | 704 | 17（其中 4 条 = 值/NaN 归约 pass，其余为 transpose/reshape/cast/select） | **11 968** | **0** | 15 360 lanes（(64,3,80)） |
| ecg | 一般（round-0） | 1184 | `2*5*14 + 12 = 152` | 179 968 | **165 760** | 128 lanes |
| ecg | **相位** | 512 | ~20（其中 10 条 = tap 宽 op） | **10 240** | **0** | 4 096 lanes（(64,64)） |
| textcnn | global（round-0） | 256 | 11 | 2 816 | 0 | 64×128 lanes |
| textcnn | global + R4-c | **128** | 11 | **1 408** | 0 | 128×128 lanes |

⇒ 「更少、更宽的 op」在数据上成立：sincnet op 发射数 **23.6×** 下降（282 880 → 11 968，且单 op 宽度 80 → 15 360 lanes）、ecg **17.6×** 下降（179 968 → 10 240，宽度 128 → 4 096 lanes），窄 op 归零。

### 13.6.2 GM 字节（每核每 launch 实测 × 48 核；理想值按 shape 计算）

| workload | 结构 | GM 读 | 理想读 | 读放大 | GM 写 | 理想写 | 写放大 |
|---|---|---:|---:|---:|---:|---:|---:|
| sincnet [32,80,4096] fp16 | 一般 | 20.64 MB | 20.97 MB | **0.98×** | 27.11 MB | 6.99 MB | **3.88×** |
| sincnet | **组归约** | 20.64 MB | 20.97 MB | **0.98×** | 13.96 MB | 6.99 MB | **2.00×** |
| ecg [16,128,2048] bf16 | 一般 | 19.20 MB | 8.39 MB | **2.29×** | 22.94 MB | 4.19 MB | **5.47×** |
| ecg | **相位** | 12.68 MB | 8.39 MB | **1.51×** | 8.38 MB | 4.19 MB | **2.00×** |
| textcnn [64,256,128] fp16 | global | 3.84 MB | 4.19 MB | 0.92× | 0.029 MB | 0.033 MB | 0.88× |
| textcnn | global+R4-c | 4.22 MB | 4.19 MB | 1.01× | 0.034 MB | 0.033 MB | 1.02× |

**输出覆盖**：三条路径每个输出元素**恰好写一次**（`bl_real = min(bl, out_l-ow0)` 处理尾块；
`--level all` 对全部 shape 逐位精确，含尾块/ceil/边界用例）⇒ 没有任何“通过少写换来的加速”。
**没有任何一条路径增加 GM 读**（sincnet 0.98×、ecg 1.51× < 一般路径 2.29×）。

## 13.7 合并非回退、smoke 与中止判定

- 合并：R4-a（相位）/R4-b（组归约）/R4-c（退化路径 CH）作用在**互斥的编译期路径**上，
  全部在同一个 `@tilelang.prim_func` 集合内；路径选择是纯代数条件，**没有新增 Python
  factory / op 层 / wrapper 性能分派**；wrapper 契约（`block_m`）未变。
- 交付文件 `--level all`：**L0 / L1 / L2 / Boundary 全 PASS**，`Summary max_diff` =
  bf16/fp16/fp32 **0.000e+00**。
- 本轮新增覆盖（保留全部既有 case id）：`L0-18a…L0-18i`（相位路径：k3d2 / k5d2-ceil / k3d4-fp32 /
  k1 / k·c 尾块 / 三个代数守卫必须回退）、`L0-18j`/`L0-18j2`（相位路径 NaN 与 −inf padding）、
  `L0-19a…L0-19j`（组归约：k3s3 / k2s2-ceil / k4s4-fp32 / k5s5 / 尾块 bl_real<bl / ceil 越界守卫 /
  s≠k 守卫 / bf16 与 32 B lane 粒度守卫 / NaN）。
- **built-path 断言**（本轮新增的防呆）：`_run_case` 不仅断言 `plan["path"]`，还断言内核**实际
  构建**的路径（`_BUILT_PATH`）。后端拒绝特化 tile 时会静默回退到一般路径（结果仍逐位精确），
  没有这条断言时“计划=特化、实际=一般”会被误记为性能结论（本轮实际抓到 3 次）。
- 中止原因：**`success`（best_effort 口径）**——两个优先 workload 的结构族均已被换掉并有实测结论，
  textcnn 亦拿到 −8.1%；未触发 `budget_exhausted`（本轮 formal 分支 3 个 + 12 次 tile 扫描 + 6 次探针，
  ≤ `max_stage4_experiments=40`）。

## 13.8 Final Summary

最终候选：`perf_opt/_max_pool1d_kernel.py`，`candidate_id = _max_pool1d_kernel`，
SHA256 `2c483e09d5a9496b44505d70c63237ac5efe5de2a2643ef17dc482d2e0853480`
（3 条 `phase=final` 记录的 `artifact_sha256` 与该文件一致；每条 2 次独立读数）。

### Final Performance Test Data

`phase=final`，**round 6（2026-10-21，Ascend910B2 / tilelang `f947fb1`，48 Vector 核）**，同一最终候选
（`candidate_id = _max_pool1d_kernel`，artifact `_max_pool1d_kernel.py`，SHA256
`2c483e09d5a9496b44505d70c63237ac5efe5de2a2643ef17dc482d2e0853480`）。
`baseline_us` = 本轮 round 4 同机重采的 baseline 记录（round-0 的 910B3 台账见 §6，两套记录均在
`perf_records.jsonl` 中 append-only 保留；不同器件/工具链，绝对时延不可混用）。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/pool/max_pool1d/max_pool1d_kernel/_max_pool1d_kernel.py::_max_pool1d_kernel::main | sincnet-speaker-local-float16 | 159.01 | 61.55 | _max_pool1d_kernel |
| tileops/kernels/pool/max_pool1d/max_pool1d_kernel/_max_pool1d_kernel.py::_max_pool1d_kernel::main | textcnn-global-float16 | 10.72 | 9.85 | _max_pool1d_kernel |
| tileops/kernels/pool/max_pool1d/max_pool1d_kernel/_max_pool1d_kernel.py::_max_pool1d_kernel::main | ecg-cnn-dilated-bfloat16 | 94.34 | 37.225 | _max_pool1d_kernel |

| kernel_id | workload_id | baseline_us | final_us | Δ | 路径 | final 读数 |
|---|---|---:|---:|---:|---|---|
| `tileops/kernels/pool/max_pool1d/max_pool1d_kernel/_max_pool1d_kernel.py::_max_pool1d_kernel::main` | `sincnet-speaker-local-float16` | 159.01 | **61.55** | **−61.3%** | group(CH80,BL64) | 61.57 / 61.53 |
| 同上 | `textcnn-global-float16` | 10.72 | **9.85** | **−8.1%** | global(CH128) | 9.80 / 9.90 |
| 同上 | `ecg-cnn-dilated-bfloat16` | 94.34 | **37.23** | **−60.5%** | phase(CH64,BL64) | 37.19 / 37.26 |

final 逐 workload 分解（µs）：

| workload | Task | aiv | vec | mte2 | mte3 | scalar | GM_to_UB(KB) | UB_to_GM(KB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sincnet | 61.55 | 57.89 | 38.71 | 13.88 | 14.50 | 19.27 | 430.0 | 290.8 |
| textcnn | 9.85 | 7.53 | 4.55 | 1.85 | 0.57 | 0.95 | 88.0 | 0.7 |
| ecg | 37.23 | 33.52 | 20.59 | 12.28 | 7.02 | 8.06 | 264.0 | 174.6 |

> textcnn 的分支文件读数为 **9.15 µs**（`_round4_globalch`，2 次一致），交付文件两次读数为
> **9.80 / 9.90 µs**（中位 9.85）。两者代码路径相同（仅文件头注释不同），7% 的差属本机已知的
> 跨进程读数漂移（`BP_run_state_bimodality` 家族）⇒ 一律以**交付文件**的两次一致读数为准
> （与 conductor 的「交付文件上取两次一致读数」要求一致）。

smoke（仅精度，不参与性能门禁）：`smoke-1x8x32-k3s2p1-{fp16,bf16,fp32}` 在 `--level all` 中
全部 `max_diff=0.000e+00` ✓。skip：无（manifest 6 例全部可运行）。

### 13.8.1 剩余差距与归因（best_effort）

| workload | final | 剩余主项 | 归因 | 还能做什么 |
|---|---:|---|---|---|
| sincnet | 61.55 | `mte2 13.88 + mte3 14.50` = 46% | 两者都受 **burst 长度**限制：读行 = `3*BL*2 = 384 B`、写行 = `BL*2 = 128 B`（`bl` 被 UB 卡在 64） | 需要更省 UB 的归约形态（例如把 NaN 掩码并进值平面）才能 bl↑ |
| ecg | 37.23 | `vec 20.59` + `mte2 12.28` | 读侧有 **2× 相位冗余**（偶相位有用、奇相位白读）；`vec` 里 10 条 tap 宽 op（值 5 + NaN 掩码 5） | 无跨步 DMA 原语（capability gap CG-2026-0031 仍开）；NaN 掩码可尝试搬进 bf16 域 |
| textcnn | 9.85 | `vec 4.55`（5 条全平面 pass） | `L_out==1` 域；70% 线 3.33 µs **低于本机空 kernel 地板 2.79 µs 加必需读 4.1 µs** | 值+NaN 双归约合并（无向量 argmax 类原语可用） |

### 13.8.2 参照锚定（T-1 morph ladder，定稿前强制）

| 参照 | 路径 | 结构形态 | 本轮结论 |
|---|---|---|---|
| 同 op indices 变体 round-2/3 | `max_pool1d_with_indices/.../perf_opt/opt_log.md` §11–§12 | 2 相 deinterleave 行批处理（ecg）、`L_out==1` 折半归约（textcnn） | **可移植**：相位路线已按 §11 形态落地（§13.4 R4-a）；其 sincnet（s=3）当时未解 |
| max_pool2d round 4 | `max_pool2d/.../perf_opt/PROGRESS.md` | 相位分解 + shift-vmax + ping-pong 行 DMA | 相位形态一致；**其 `s_w==2` 通解未覆盖 `s_w==k_w` 的组归约** |
| 本 op round-1 | 本文件 §6.4 | 一般网格窄 op 扫描 | **已被本轮实测推翻**（159.01→61.55 / 94.34→37.23） |
| `pattern-library` 窗口类 | `CASE-pool-*` | — | 仍无「组归约」条目 ⇒ 本轮新增案例（见 §13.10 VP-D 系列） |

**结论**：round-1 `perf_feedback.md` 的「窗口域无可移植同族最优参照」+「冻结结构族内不可达」
**对本 op 的两个优先 workload 已被证伪**；其存活的硬上限只剩「burst 长度受 UB 约束」这一条。
按 `perf-feedback.md` §3 的 append-only 纪律，本轮**不删除原文**，在 `perf_feedback.md` 追加
「修正附录」，并且**不再发新的 `[DESIGN_LIMIT]` 信号**（设计层的结构天花板已被实测推翻）。

### 13.8.3 已知残留（如实披露）

1. **`_GROUP_BL_SAFE_MAX = 96` / 32 B lane 粒度 / 非 bf16 / ch 整除 C** 四条硬约束来自
   实测（其中两条以**设备 fault** 形式暴露），未定位到编译器根因；它们只影响组归约的候选面，
   不影响正确性（不满足者回退一般路径，仍逐位精确）。
2. **组归约对 bf16 不可用**（`hivm.hir.vreduce` 无 bf16 形态）；bf16 的 `s==k` 形状回退一般路径。
3. **ecg 的 2× 相位冗余读**未消除（无跨步抽取原语）。
4. `probe_floor_r4.py --variant row4096` 不可测（UB 容量）。
5. 交付文件里保留的 tuning 环境变量（`MAXPOOL1D_PHASE_CH/BL`、`MAXPOOL1D_GROUP_CH/BL`、
   `..._UB_BUDGET`）默认全为「自动」，只用于复现扫描，不参与默认决策。
6. 死缓冲/无用声明：三条既有路径与本轮两条新路径的 `T.alloc_ub` 列表均被实际使用
   （`_phase_bytes` / `_group_bytes` 与 alloc 列表逐项对应）；`po`（deinterleave 的奇相位输出）
   必须存在但内容不参与计算，已在注释说明。

## 13.9 Skill Retrospective（本轮增量）

| id | vp_type | 结论 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-D7 | **D** | **`s_w == k_w`（非重叠连续窗）= 组归约**：transpose + `(bl,k,ch)` + `reduce_max(dim=1)` 把 `2*k*out_l*N` 条窄 op 折成每单元一次宽归约；sincnet 159.01→61.55 µs（2.58×），逐位精确 | §13.4 / §13.6 / `perf_records.jsonl` round 4 | `probe_sinc3_r4.py --variant reduce3` | 910B2 / f947fb1 / CANN 9.0.0 |
| VP-D8 | **D** | **归约轴的位置决定成败**：同一 `(ch,bl,k)` 数据，`reduce dim=1`（中间轴）35.79 µs，`dim=2`（最内层组）**2234 µs**（scalar 2215.7 = 标量化） | `probe_sinc3_r4.py` reduce3 vs last | 同上（`--variant last`） | 同上 |
| VP-D9 | **D** | **3-D 归约的 lane 粒度硬约束 = 32 B 的整数倍**：`ch*eb = 16 B` 与 `48 B` 触发「UB address accessed by the VEC instruction is not aligned」设备 fault；32 B / 160 B 正常。**bf16 无 vreduce 形态**（编译失败） | §13.5.2 + L0-19/18 用例 | `_group_plan` 的两条 guard | 同上 |
| VP-D10 | **D** | **`vmax` 的 NaN 语义跨 dtype 不一致**：fp16 传播 NaN、fp32 传播、**bf16 丢弃 NaN**（得 −inf）⇒ 任何「靠 vmax 传播 NaN」的简化都不可移植 | `probe_vmax_nan_r4.py` | `probe_vmax_nan_r4.py`（自包含） | 同上 |
| VP-D11 | **D** | **`s_w==2`+偶 `p/d` 的相位路线在**纯值**op 上仍成立**（sibling 是 indices 变体）：ecg 94.34→37.23 µs（2.53×），且 tile 最优点从「最大 lanes」移到「更长 bl」（(64,64) 37.34 < (128,40) 39.81） | §13.5.1 | sibling `probe_phase_ecg.py` / 本节扫描脚本 | 同上 |
| VP-D12 | **D** | **写侧放大是本族被忽视的第一成本项**：sincnet 一般路径 `UB_to_GM` 是真实输出的 **3.88×**（store 行 = `BL*2`），ecg **5.47×**；结构改造后降到 2.00× / 2.00× | §13.2 P1/P2 + §13.6.2 | `perf_records.jsonl` 的 GM 列 | 同上 |
| VP-R3 | **R** | **plan 与 built 必须分开断言**：特化 tile 被后端拒绝时会**静默**回退一般路径（结果仍正确），只断言 `plan["path"]` 会把回退误记为性能结论。本轮因此抓出 3 次假阳性（sincnet/bf16/小 ch） | §13.7 built-path 断言 + `logs/round4/l0_group*.log` | — | — |
| VP-R4 | **R** | **planner 重构必须复算所有「循环变量派生量」**：本轮把「循环内构造 best」改成「先收集 feasible 再选」，导致 `nob` 泄漏为最后一个候选值 ⇒ `num_logical` 偏大 ⇒ 尾单元越界 ⇒ **设备 fault**（表现为「某个 tile 不稳定」，误导了两轮排查） | §13.4 与 `l0_phbl*.log` 的对比 | — | — |
| VP-C3 | **C** | 案例索引续档：`CASE-pool-maxpool1d-migration` 的 round-4 段（组归约成立、相位对纯值 op 成立、reduce 轴位置、32 B lane 粒度、bf16 vreduce 缺口、vmax NaN 跨 dtype 不一致、写放大账） | 本节全文 | — | — |

### 13.9.1 skill 流程问题

| # | 问题 | 证据 | 建议 |
|---|---|---|---|
| S1 | Phase 2 的「tile 扫描点不进 `perf_records.jsonl`」是 sibling 的**约定**而非 skill 文本，gate 只校验 formal 分支 ⇒ 不同执行者的账本口径可能不一致 | 本节 13.5 与 `perf_records.jsonl` 的差异 | skill 明示「autotune/扫描点只进 opt_log 的 Autotune Log，不进 perf_records」 |
| S2 | `perf_records.jsonl` 的 append-only 与「同一轮内 final 被自己替换」冲突（本轮先对旧文件写了 3 条 final，随后补 R4-c 又重采） | `perf_records_superseded.jsonl` 的两段 `_reason` | `signal-registry` §5 增「同轮替换」的机械约定（建议：superseded 文件的表头行格式固定化） |
| S3 | 「设备 fault」类失败与「tile 性能差」在 skill 里没有区分通道：fault 需要立即从候选面剔除并可复现，否则会被误读为性能波动 | §13.5.2 的 (40,128)/(16,160) | 迭代诊断增一节「设备 fault（不是性能失败）」的处置：记录 errorStr、剔除 + 加 guard、并写进 L0 守卫用例 |
