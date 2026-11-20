# Stage 4 调优日志 — AmaxFwdOp（reduction / amax）

- project: TileOPs（`examples/TileOPs`）；operator: **AmaxFwdOp**（manifest PascalCase 键）
- 算子目录: `examples/TileOPs/tileops/kernels/reduction/amax/amax_kernel/`
- 定稿产物: `perf_opt/_simple_reduce_kernel.py`（resident 路径）+ `perf_opt/_simple_reduce_kernel_tiled.py`（streaming/tiled 路径），二者顶层定义同名工厂，与基准同签名
- mode: `full`（scenario=optimize）；**2026-10-19 追加 Round 7 定向二轮**（只冲 hidden bf16，未重跑已达标/结构地板 case）
- 性能口径: **`msprof op` Task Duration(us)**（唯一 kernel 时延口径）；Roofline Ratio = `GM/L2 GM Read + Write` 条目 ratio（`parse_bin_file`），与 `tileops.reporting` 的 `Ratio(%)` 同源同口径（峰值参考 1.75 TB/s）
- 目标: 每个非 smoke case 的 Roofline Ratio ≥ 60%；未达标须给出原因
- 测量方式: 调优口径 = `_measure.py`（直接构建目标 kernel、按 wrapper 实际选择的 `(block_m, tile_n)` 采集 `msprof op`），launch-count=10 / warm-up=5、取中位；**生产口径 = `python -m tileops.reporting.cli run --op AmaxFwdOp --prof-mode msprof`，每 case 仅 1 次 launch（`sample_count=1`）**，Round 7 报告两种口径
- 工具链戳: tilelang 0.1.4（dev root `build/`，git ab62f46）+ CANN 8.5.2 + 设备 **Ascend910B3**（`torch_npu.npu.get_device_name` 实查；20 AICore → 纯 Vector **40 核**；`supports_native_bf16=False`）+ torch 2.7.1 (npu)。知识预注入 `PL-1.12-task-pipeline-depth2`（origin=attention/Expert）在本 op 的向量归约上**未复现收益**（见 Round 3）；`kb_stale_check` 报 stale_count=133（均为 attention 族，本轮未引用其余 pattern-library 条目）

## Workload Inventory（benchmark 展开，详见 `workload_inventory.json`）

benchmark 入口：`examples/TileOPs/benchmarks/ops/bench_amax.py::test_amax_bench`（`workloads_to_params("AmaxFwdOp", include_extra=True)`）。smoke 被 `-m "not smoke"` 排除。

| workload_id | dtype | shape | dim | 路径/配置（wrapper `ReduceKernel`） | kind |
|---|---|---|---|---|---|
| smoke-amax-float16 / bfloat16 / float32 | fp16/bf16/fp32 | (32,256) | -1 | resident bm=8 | smoke（仅精度回归） |
| hidden-state-reduce-float16 | fp16 | (2048,4096) | -1 | `_simple_reduce_kernel`，wrapper bm=4 | tune |
| hidden-state-reduce-bfloat16 | bf16 | (2048,4096) | -1 | `_simple_reduce_kernel`，wrapper bm=4（bf16 走 fp32 vcast 工作缓冲） | tune |
| 3d-multidim-reduce-float16 | fp16 | (4,128,4096) | [0,2] | op `permute(1,0,2).contiguous()` → (M=128,N=16384)；`_simple_reduce_kernel`，wrapper bm=1（核内 promoted→2） | tune |
| long-seq-reduce-bfloat16 | bf16 | (64,32768) | -1 | `_simple_reduce_kernel_tiled`，wrapper tile_n=16384（核内 clamped→8192） | tune |

> 说明：`3d-multidim-reduce` 的 `permute+contiguous` 是 op 层的一次设备内拷贝（不计入 kernel-only 口径）；kernel 侧 M=128、N=16384、流量 4.19 MB。

## Performance Test Data — Baseline（round 0, phase=baseline）

| workload_id | kernel | Task Duration(us) | Ratio(%) | Block Dim | aiv_vec_ratio | aiv_scalar_ratio | aiv_mte2_ratio | aiv_mte3_ratio | profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| hidden-state-reduce-float16 | baseline resident bm=4 | **14.82** | 64.69 | 40 | 0.26 | 0.30 | 0.58 | **0.64** | profiles/REC_base_hs_fp16_1 |
| hidden-state-reduce-bfloat16 | baseline resident bm=4 | **19.72** | 48.62 | 40 | 0.48 | 0.26 | 0.33 | **0.31** | profiles/REC_base_hs_bf16_1 |
| 3d-multidim-reduce-float16 | baseline resident bm=1 | **8.64** | 27.74 | 40 | 0.14 | 0.12 | 0.30 | **0.66** | profiles/REC_base_3d_fp16_2 |
| long-seq-reduce-bfloat16 | baseline tiled bm=1 tn=8192† | **9.30** | 25.77 | 40 | 0.35 | 0.20 | 0.40 | 0.07 | profiles/REC_base_lseq_bf16_1 |

† **集成基线对 long-seq 直接失败**（wrapper 传 `tile_n=16384`，baseline `assert 2*tile_n*B(bf16=6) <= 184KB` 抛 `AssertionError`，见 `integration_log.md`）。为给出可对账的 baseline 记录，此处用**同一 baseline kernel 在预算合法 tile_n=8192** 下实测（即 baseline 结构在合法配置下的能力）。集成态（16384）= crash。

> **现象（Round 0 诊断）**：三例 resident baseline 的 `aiv_mte3_ratio` 高达 0.31–0.66 —— 每次 persistent 任务迭代都单独发一次 UB→GM 写出（`T.copy(out_ub, out)`），单次写出无论 payload 大小都耗 ~0.4–0.9 µs MTE3 busy，成为小 workload 的主导项；同时 baseline 用 `min(num_kernels,40)` 网格步进（非整除时 40 核），`mte2_ratio` 仅 0.30–0.58，DMA 未饱和。

### 设计估算 vs 实测偏差（D-2 回填）

`DESIGN.md §1.6.0` 设计期 roofline 估算下界（per-core 30.2 GB/s × 40 核 = 1.2 TB/s 外推档）：

| workload | 设计估算下界(µs) | baseline 实测(µs) | 偏差 | 失准项 |
|---|---:|---:|--:|---|
| hidden-state fp16 | ≈14 | 14.82 | 1.06× | 无（估算准） |
| hidden-state bf16 | ≈14–52 | 19.72 | 落在区间 | 发射项串行上界 |
| 3d-multidim fp16 | ≈3.5 | 8.64 | 2.47× | 发射/写出项（MTE3 每任务写出）被低估 |
| long-seq bf16 | ≈6.4（tiled 单元估算） | 9.30 | 1.45× | 发射项串行上界未达 |

## Iteration Log

### Round 1 — 结构优化：批量化写出 + 整除精确网格 + 连续分片 + reduce 目标切片（S1–S4）

- **base**：baseline（round 0）。
- **现象**：三例 resident baseline 由 MTE3（每任务 UB→GM 写出）主导（0.31–0.66），DMA 未饱和。
- **候选优化点（从同一 base 派生多个分支）**：
  - **S3 批量化输出写出**：每核把各任务的行极值暂存在扁平 UB buffer，**每核只发 1 次** UB→GM 拷贝（原为每任务 1 次）。
  - **S4 reduce 目标切片**：连续任务的行极值直接落在同一个 `(L*bm,1)` buffer 的连续切片，形状收缩每核 1 次。
  - **S1 整除精确 persistent 网格**：网格取「logical 块数 ≤40 的**最大因子**」（静态 trip count、无动态 guard）。
  - **S2 连续、静态均衡的每核行区间**：核 `cid` 拥有连续任务区间（输出行连续）。
  - （tiled 路径）**S3′ 首个满 tile 提升出循环**。
- **实验结果**（Task Duration / Ratio）：

| 分支 | hs fp16 | hs bf16 | 3d fp16 | L0 |
|---|---:|---:|---:|---|
| baseline（round 0 current best） | 14.82 / 64.7% | 19.72 / 48.6% | 8.64 / 27.7% | PASS |
| S1–S4 合并（perf_opt） | **13.28 / 72.2%** | **17.30 / 55.4%** | **5.30 / 45.2%** | PASS |

- **winner**：S1–S4 合并版，核内合并通过（见 Round 2/3 的全 tune workload 非回退检查 + smoke 精度）；更新已合并 current best。

### Round 2 — 网格规模 / block_m 精搜（候选 vs current best）

- **base**：Round 1 合并版（current best）。
- **候选优化点**：① 用满 40 核（full-grid，带 guard）替代整除因子 32 核；② bf16 `bm` 精搜（1/2/4）；③ 3d `bm` 精搜（1/2/4）。
- **每分支只改一个主优化点**，同 base 派生；全部 interleaved 多次复测（噪声阈值 3%）。

| 候选（vs current best） | workload | Task Duration(us) mean | Ratio(%) | AICore/网格 | aiv_mte2_ratio | L0 | 结论 |
|---|---|---:|---:|---|---:|---|---|
| current best | hs fp16 | 13.19 | 72.7 | 32 核整除 | 0.92 | PASS | — |
| full-40 网格 | hs fp16 | 13.40 | 71.5 | 40 核 guard | 0.71 | PASS | **no gain**（≤3% 平区） |
| current best | hs bf16 | 17.02 | 57.0 | 32 核整除 | 0.68 | PASS | — |
| full-40 网格 | hs bf16 | 17.04 | 57.0 | 40 核 guard | 0.71 | PASS | **no gain** |
| bm=2 | hs bf16 | 17.00 | 56.4 | 32 核 | — | PASS | **no gain**（bm=2≈bm=4） |
| bm=1 | hs bf16 | 40.92 | 23.4 | 32 核 | — | PASS | 回退（每任务开销） |
| current best | 3d fp16 | 5.36 | 44.9 | 32 核（bm→2） | 0.65 | PASS | — |
| full-40 网格 | 3d fp16 | 5.31 | 45.2 | 40 核 guard | 0.62 | PASS | **no gain** |
| bm=1（不 promote） | 3d fp16 | 5.62 | 42.6 | 32 核 L=4 | — | PASS | 回退 |
| bm=4（手动） | 3d fp16 | — | — | — | — | — | **UB overflow**（`2*4*16384*2=256KB>192KB`，BiShengIR 编译失败） |

- **结论**：full-40 在多次复测下与整除 32 核**打平**（首轮单次 59.3% 为噪声，被重复测量推翻 → 证伪更正记录）；`bm` 已被核内 `1→2` promotion 取到最优。current best 不变。`config_no_gain` 范围 = full-40 网格 + bf16 bm 精搜 + 3d bm 精搜。

### Round 3 — 显式软件流水（T.Pipelined）候选（vs current best）

- **base**：Round 2 current best。
- **现象**：resident 路径 `aiv_time` ≈ 各 pipe busy 之和（3d：mte2 2.35 + vec 0.75 + scalar 0.80 ≈ 4.1 µs，而 `aiv_mte2_ratio` 仅 0.65）→ DMA 与向量链**零重叠**；bf16 路径有整宽 fp32 `vcast`。
- **候选优化点**：① 任务循环 `T.Pipelined(num_stages=2/3)`（fp16/bf16）；② bf16 去 vcast（同域 `reduce_max(bf16)`）；③ tiled 内层 tile 循环流水 / 独立 partial 槽。

| 候选（vs current best） | workload | Task Duration(us) | Ratio(%) | aiv_vec/mte2 ratio | L0 | 结论 |
|---|---|---:|---:|---|---|---|
| current best | hs fp16 | 13.19 | 72.7 | 0.30 / 0.92 | PASS | — |
| task-loop pipe2 | 3d fp16 | 5.34 | 44.9 | — | PASS | **no gain** |
| current best | hs bf16 | 17.02 | 57.0 | 0.62 / 0.68 | PASS | — |
| task-loop pipe2（bf16） | hs bf16 | 16.90 | 56.7 | 0.62 / 0.69 | PASS | **no gain**（≤3%） |
| bf16 去 vcast（同域 reduce） | hs bf16 | — | — | — | — | **compile fail**（`T.reduce_max` src/dst shape 校验失败 → bf16 同域 reduce 本工具链**不支持**，DESIGN §9.1 约束复验成立） |
| tiled full-40 / inner-pipe / partial-slot | lseq bf16 | 9.18 / 8.84 / 8.80 | 26.1 / 27.1 / 27.2 | — | PASS | **no gain**（≤3%） |

- **结论**：`T.Pipelined` 在本工具链的该形态**未产生 DMA/向量重叠**（与 `PL-1.12` 的 attention task-pipeline 结论不同族，向量归约未复现）；bf16 同域 reduce 确不支持（保持 fp32 vcast）。current best 不变。

### Round 4 — tiled tile_n 包络收紧（S0，正确性修复）

- **base**：Round 3 current best。
- **现象**：`_simple_reduce_kernel_tiled --level all` 的 L0-2b（128×65536 fp32, tile_n=16384）**间歇失败**（随机单元素错，2/5）；**baseline 同款 kernel 同样复现**（说明是工具链层 auto-multi-buffer 在 64KB 单级 staging（128KB doubled 足迹）上的 read-after-write 竞态，非本迁移引入）。
- **候选优化点**：把 tiled kernel 的 doubled-buffer 包络从 184 KiB 收紧到 **96 KiB**（`2*tile_n*B <= 96 KiB`），tile_n 仍按「N 的 256 倍数因子优先」下取。
- **结果**：fp32 128×65536 tile_n 16384→**clamped 8192**，0/8 失败（原 2/5）；bf16 long-seq 的 tile_n 仍为 8192（`2*8192*6=96KiB`，不变），benchmark 性能无回退；tiled `--level all`（L0/L1/L2/Boundary）**全过**。
- **winner**：S0 包络收紧并入 current best。

### Round 7 — 定向二轮：满 40 核网格（S1′）+ 静态行块 extent（S6）

> 本轮为针对 `hidden-state-reduce / bf16`（生产路径 54.97%，距 60% 差 5.0pp）的**定向**二轮调优；hidden fp16 已达标、3d/lseq 已判小任务结构地板，不重跑其结构探索。base = Round 4 current best（整除精确 32 核 + 动态行 extent）。

- **现象**（base profile，hs bf16，32 核）：logical 块数 512，`_choose_grid` 取「≤40 的最大整除因子」= **32**，**8 个 vector core 空闲**；`aiv_vec_time` 9.38 µs、`aiv_scalar_time` 5.41 µs、`aiv_mte2_time` 10.63 µs，`aiv_mte2_ratio` 0.676。同族 `AminFwdOp` 在同形状上用满 40 核把 bf16 做到 58.60%（生产口径），因此**重新裁定** Round 2 的「full-40 打平」结论。
- **候选优化点（同一 base 派生，每分支只改一个主点）**：
  - **S1′ 满 40 核 persistent 网格**：`nk = min(logical, 40)` + 连续均衡区间（`cid*nl//nk`，带 guard）。512 块 → 32 核各 13 块 / 8 核各 12 块（原 32 核各 16 块）。
  - **S6 静态行块 extent**：`M % block_m == 0` 时用编译期常量 extent 的 `T.copy` 取代 `real_m = min(bm, M-off_m)` 动态切片（去掉热循环内每任务的标量 extent 计算）。
  - 探索性分支（均不改变 Round 2/3 已验证的 bf16 提升链）：任务循环 `T.Pipelined(2)`、手工双缓冲预取、固定 reduce 目标 + 每任务小拷贝、跨 dtype GM→fp32 直拷、半宽 pair-fold。

- **交错 A/B：满 40 核 vs 整除 32 核（同结构、均含 S6，T-2 协议）**

  `msprof op Task Duration(us)`，launch-count=10 / warm-up=5，A/B/A/B×5 交错，同 session：

  | pair | 32 核（r7c, us） | 40 核（r7b, us） | 胜者 |
  |---:|---:|---:|---|
  | 1 | 16.72 | **15.80** | 40 |
  | 2 | 16.76 | **15.80** | 40 |
  | 3 | 16.58 | **15.60** | 40 |
  | 4 | 17.20 | **16.12** | 40 |
  | 5 | 17.46 | **15.84** | 40 |
  | **中位数** | **16.76** | **15.80** | **40（5/5 配对胜，-5.7%）** |

  补充采样：40 核 + 动态 extent 中位数 16.32 µs（5 run：16.32/16.40/16.44/15.58/16.14）；40 核 + S6 全量 22 run 中位数 **15.83 µs（Ratio 60.59%）**；32 核 + S6 中位数 16.76 µs（57.20%）。**结论：full-40 在 amax 上真实有效（非 Round 2 的 run-state 噪声），且 S6 再贡献约 -3.5%。**

- **候选 vs current best（B2 结构化对比表）**

  | 候选（vs current best） | workload | Task Duration(us) 中位/代表 | Ratio(%) | vec/mte2 ratio | L0 | 结论 |
  |---|---|---:|---:|---|---|---|
  | current best（32 exact + 动态 extent） | hs bf16 | 16.94（Round-4 final） | 54.97（生产） | 0.60 / 0.68 | PASS | — |
  | S1′ full-40（动态 extent） | hs bf16 | 16.32 | 58.46 | 0.53 / 0.70 | PASS | **有效（+3.8%）** |
  | S1′+S6 full-40 + 静态 extent | hs bf16 | **15.83**（22 run 中位） | **60.59** | 0.53 / 0.70 | PASS | **winner（较 base -6.6%）** |
  | S6 only（32 + 静态 extent） | hs bf16 | 16.76 | 57.20 | 0.62 / 0.69 | PASS | 网格为主效应，S6 单独收益小 |
  | full-40 + `T.Pipelined(2)` | hs bf16 | 16.20 | 59.18 | — | PASS | no gain（npui r 无软件流水 pass，IR 等价） |
  | full-40 + 手工双缓冲预取（关 auto multi-buffer） | hs bf16 | 16.68 | 57.48 | mte2 10.18 | PASS | no gain（后端同步早于预取生效） |
  | full-40 + 固定 reduce 目标 + 每任务小拷贝 | hs bf16 | 19.66 | 48.76 | — | PASS | 回退（每任务额外向量写+同步） |
  | full-40 + vcast 后半宽 vmax 折叠 | hs bf16 | 16.88 | 56.79 | — | PASS | 回退（多一趟 vmax 的代价 > vreduce 减半收益） |
  | full-40 + **融合 cast+pair-max**（两个 `T.cast` 同趟） | hs bf16 | 13.14 | 72.96 | — | **FAIL**（max_diff=1.72） | **不可采纳**：lowering 把两个 cast 提升成 `tensor.empty()` 后丢失 vcast（编译器缺陷，见 Issues）。**2026-11-02 复核更正**：该 13.14/72.96 是 cast 被整段删除后的假数（内核实际没做 bf16→fp32），缺陷修复后同形态实测 20.12 µs / 45.69%，不采纳，见附录 A |
  | full-40 + 跨 dtype GM→fp32 直拷（去 x_ub+vcast） | hs bf16 | 15.87（12 run 中位） | 60.44 | — | PASS | 与 S6 版持平（未采纳，保留显式 vcast 语义） |
  | full-40 + bm=5 | hs bf16 | 17.36 | 55.22 | — | PASS | 回退（410 块 / 40 → 10–11 任务不均衡） |
  | full-40 + bm=6/8 | hs bf16 | — | — | — | — | 编译失败（UB overflow，192KB+32B / 262KB） |

- **合并与非回退检查**：S1′+S6 并入同一 `@T.prim_func`（无 Python 侧性能分派；`_choose_grid` 为纯编译期网格规则，`static_m` 为编译期布尔）。合并版在最终文件上复测全部 4 个 tune workload（各 3 run）：hs fp16 13.42 / hs bf16 15.90 / 3d 5.40 / lseq 8.86 µs；smoke 精度通过。
- **生产路径（`tileops.reporting.cli run --op AmaxFwdOp --prof-mode msprof`）**：该路径每 case **只采 1 次 launch**（`sample_count=1`），共跑 3 次：

  | run | hs fp16 | hs bf16 | 3d fp16 | lseq bf16 |
  |---|---:|---:|---:|---:|
  | 1 | 12.94 / 74.09% | **15.72 / 60.99%** | 5.32 / 45.05% | 9.06 / 26.45% |
  | 2 | 13.56 / 70.70% | 16.68 / 57.48% | 5.32 / 45.05% | 9.12 / 26.28% |
  | 3 | 13.74 / 69.77% | 16.70 / 57.41% | 5.18 / 46.27% | 8.86 / 27.05% |
  | 中位 | 70.70% | **57.48%** | 45.05% | 26.45% |

  bf16 单发口径分布 57.41–60.99%（launch-count=10 的暖机中位 60.29–60.59%），**恰跨在 60% 线上**。

- **winner**：S1′（full-40）+ S6（静态 extent）并入 current best；`perf_opt/_simple_reduce_kernel.py` 定稿，tiled 文件不变。

### Round 8 — bf16 归约路径重构：A/B/C 三路线全实测（2026-10-26）

- **base**：Round 7 定稿 `final_r7`（`perf_opt/_simple_reduce_kernel.py`，bf16 = 整块 `T.vcast` 升 fp32 后 `reduce_max`）。本轮改前备份 `history_version/20261026_pre_bf16/`（两个模块 + wrapper + records/log/inventory）。
- **现象（base profile，hs bf16，同会话）**：`aiv_time` 15.10 µs = `aiv_vec_time` **8.21** + `aiv_mte2_time` 6.61 + `aiv_scalar_time` 4.66（近串行、几乎不重叠）；对比 fp16 的 `vec` 2.77 µs，bf16 的全部额外成本来自「bf16→fp32 vcast（2B 读 + 4B 写）+ fp32 vreduce（4B 读）」= **10 B/元素**片上流量。**故本轮唯一有价值的方向 = 把片上流量从 10 B/元素降下来**，而不是继续调 tiling。
- **知识预注入与首轮必查项**：`CONST-reduce-dims0-skinny`（dim0 归约灾难性慢，本轮不适用：归约轴为最内连续轴）、《PL reduction 族条目》与 `TRAP-vecop-slice-noop-and-nan-branch`（lane-local 折叠树只有 lane 0 生效 + 逐 op 成本 ~0.8–2 µs）**直接否决了"手工 vmax 折叠树"的乐观估计**；`kb_stale_check` 报 133 条 stale（attention 族），本轮引用的 3 条均在当前工具链上**重验**（见下 A/B 路线的实测指纹，未沿用旧结论）。

#### 三条路线的实测指纹（全部实测，不再重复踩）

| 路线 | 形态 | 结果 | 指纹（实测） |
|---|---|---|---|
| **A** bf16 原生归约 | `T.vmax(a_bf16,b_bf16,c_bf16)` 逐元素 | **可用且 bit-exact** | 4096 元素 bit-mismatch = 0（文档 §2.2.1 标 bf16 `×`，**文档滞后**） |
| A′ | bf16 `T.reduce_max`（跨 lane） | **不可用** | (64,512) 编译期 `error: ub overflow, requires 2097152 bits while 1572864 bits available`（= 4× tile，内部物化 fp32）；缩到能装下则运行期 `vector core timeout`（rtStreamSynchronize 507034） |
| A″ | 手工 `T.vmax` 折叠树（K 块 in-place，`T.serial`） | **状态丢失** | K=4 `max_diff=1.891`、K=8 `max_diff=2.156`；K=2（单次折叠、无 loop-carried）`max_diff=0` 但 **21.68 µs > 18.14** 基准（多一趟 copy + 一次 vmax 的固定开销 > 省下的 2 B/元素） |
| A‴ | 同一 buffer 内的**切片折叠**（`vmax(x_ub[:,0:W], x_ub[:,W:2W], p0)`，省掉额外 copy） | **编译失败** | `TVMError: Check failed: (*as_const_int(buffer_shape0[i]) == *as_const_int(buffer_shape1[i]))` —— vmax 形状校验拒绝 (bm,N) buffer 的 (bm,W) 子区域 |
| **B** int16 位模式归约 | `x` 声明 int16（bf16 张量按位直传）+ `reduce_max`/`reduce_min` + `vcmp/vselect` 符号修复 | **正确性达成、性能否决** | bit-exact 覆盖 随机 + 全负行 + ±inf + ±0 + +NaN（全负行靠 min 修复）；但 **int16 vreduce 被标量仿真**：max-only 即 27.94 µs（`aiv_scalar_time` **14.82** vs fp32 路径 4.66）；加上 min + 修复后 **48.24 µs**（修复一步 +20 µs） |
| B′ | 位模式映射（`u ^ (sign?0x7FFF:0)` + 单次 max） | 未采纳 | 需要 3 个 int16 逐元素 op（12 B/元素）> 双 reduce 的 4 B/元素；且 int16 `vcmp/vselect` 同样标量仿真（B 的实测） |
| **C** pair-max 绕 cast 提升 | = A″ 的 K=2 折叠（bf16 两两 max 后升 fp32） | **回退** | 21.68 µs vs 18.14（同会话），`max_diff=0`。2026-11-02 编译器缺陷修复后复测同结论：20.12 µs vs 定稿 17.58 µs（见附录 A） |
| B 的额外风险 | 位模式全序的 NaN 语义 | 需显式说明 | 位模式全序无法把 +NaN 与 −NaN 同时排到顶端（−NaN 符号位使其沉底）；而现定稿 fp32 提升路径两种 NaN 都传播（实测 bf16 行内 −NaN → 输出 NaN，与 torch 一致）→ B 路线另有一处隐性正确性回归 |
| 附：`T.vbitcast` bf16→int16 | 位重解释 | **不可用** | 被 lower 成 vcast：`error: 'hivm.hir.vcast' op currently don't support cast bfloat16_t_to_int16_t_rintmode`；fp16→int16 同形态**静默错**（4096 中 4092 个 bit 不符） |
| 附：标量修复 | `T.serial` + `if_then_else` / `^ & >>` | **崩溃** | 编译期 SIGSEGV（rc=139），与 `TRAP-*` 的"标量循环 + 位运算"族一致 |

#### 候选 vs current best（B2 结构化对比表，hs bf16，同会话同口径）

| 候选 | Task Duration(us) | aiv_time | vec | scalar | mte2 | Ratio(%) | L0 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **current best**（fp32 提升，`final_r7`） | **18.14**（7 run 中位 16.84–18.42） | 15.10 | 8.21 | 4.66 | 6.61 | 51.4 | PASS | — |
| r8_int16_max（B，省维护） | 27.94 | 25.93 | 5.07 | **14.82** | 6.13 | 33.4 | PASS | 回退 54%：vec 降 38% 但 scalar 涨 3.2× |
| r8_int16_maxmin_sel（B 完整） | 48.24 | 45.88 | 10.31 | **30.11** | 5.97 | 19.3 | PASS | 回退 166%：符号修复 +20 µs |
| r8_fold2_vmax（A/C） | 21.68 | — | — | — | — | 43.0 | PASS | 回退 19% |
| r8_fold4 / fold8（A″） | — | — | — | — | — | — | **FAIL**（1.891 / 2.156） | 多折叠丢 loop-carried 状态 |
| r8_foldslice2/4（A‴） | — | — | — | — | — | — | 编译失败 | vmax 形状校验拒绝子区域 |

> 结论：**三条路线全部实测失败**（A 缺跨 lane reduce、手工折叠亏本且 >1 次折叠状态丢失；B 正确但 int16 vreduce 标量仿真 + 修复灾难性；C 即 A″ 的 K=2，回退 19%）。current best 保持 `final_r7` 的 fp32 提升路径不变——它已经是本工具链上 bf16 amax 的可达最优（10 B/元素是 bf16 唯一可用归约域的固有流量）。

#### 合并与非回退检查

- 本轮**没有改动最终 kernel 体**：`perf_opt/AmaxFwdOp.py`（gate-4 契约定稿）与 `final_r7` 的 kernel 内容**逐字节相同**（sha256 `f5ec7347…` 为新增的模块头/约束注释；`@T.prim_func` 体未改）。故不存在回退风险；四个 tune workload 在定稿文件上重测：hs fp16 **13.40**、hs bf16 **18.14**、3d fp16 **5.32**、lseq bf16 **9.08**（各 3+ 次 msprof，launch-count=10/warm-up=5）。
- **fp16 未回退**：hs fp16 13.40 vs Round 7 的 13.42（同码路，会话内噪声 <1%）；3d 5.32 vs 5.40；lseq 9.08 vs 8.86（+2.5%，同码路 = 单发/会话抖动，Round 7 已记录 ±3%）。
- **smoke 精度**：`AmaxFwdOp.py --level all` → L0 12 例 / L1 10 例 / L2 4 例 / Boundary 5 例 **全过**（`max_diff=0`，含 smoke fp16/bf16/fp32）。`perf_opt/_simple_reduce_kernel_tiled.py --level all` 同样全过。
- **工件重排（本轮 gate 契约变更）**：定稿模块改名为 `perf_opt/AmaxFwdOp.py`（真实 `@T.prim_func` 定义所在），`perf_opt/_simple_reduce_kernel.py` 退化为 import 别名（同一函数对象），`perf_opt/_simple_reduce_kernel_tiled.py` 保持原名与内容。**wrapper `amax.py` 恢复 perf_opt 双 import 切换块**并指向 `perf_opt.AmaxFwdOp`。

## Autotune Log

未使用 autotune 框架；`block_m`/`tile_n`/网格规模均按「wrapper 选择 + 核内受控 promotion」手动粗搜（Round 2/7），并经 interleaved 多次复测裁决。搜索空间：网格 ∈ {整除 ≤40, 满 40}；bf16/3d `block_m` ∈ {1,2,4,5,6,8}；tiled `tile_n` ∈ {4096, 8192, 16384}。最优：**满 40 核 + 静态行块 extent + bm promotion(1→2) + tile_n clamp 8192**。

> Round 7 说明：hidden bf16 的 `block_m` 由 wrapper 传入 4（`SHARED_MEMORY_BUDGET_BYTES=48KiB` 按 2B/元素算出），核内不再改；`bm=5` 因 410 个逻辑块在 40 核上 10/11 任务不均衡而变慢，`bm≥6` UB overflow。

## Final Summary

- 定稿（Round 8 gate 契约重排）：**`perf_opt/AmaxFwdOp.py`**（candidate `final_r8`，真实 `@T.prim_func` 定义所在，sha256 `f5ec73477b84d9cba1cbd56e3c8cda06c8106812469b60fcdb43900ac3700a93`）；`perf_opt/_simple_reduce_kernel.py` 退化为 import 别名（同一函数对象，保留历史名），`perf_opt/_simple_reduce_kernel_tiled.py` 保持原名与内容（streaming 工厂由 `AmaxFwdOp.py` re-export）。两文件各自 `--level all` **L0/L1/L2/Boundary 全过**。
- **kernel 内容自 Round 7 起未改**：Round 8 只重排了工件与 wrapper 接线。Round 7 的调度改动仍为定稿内容：**S1′ 满 40 核 persistent 网格** + **S6 静态行块 extent**；bf16 仍走「整块 vcast 升 fp32 → reduce_max」。
- Round 8 结论：bf16 归约重构的 A/B/C 三路线**全部实测失败**（指纹见 Round 8 表），其中 **B 路线（int16 位模式）正确性完全达成**（随机 + 全负行 + ±inf + ±0 + +NaN 逐元素 bit-exact）但因 int16 vreduce 被标量仿真（`aiv_scalar_time` 14.82 µs vs fp32 的 4.66 µs）而慢 54%，完整修复形态慢 166%。current best 因此保持不变。
- 最终四例在定稿文件上以 `_measure.py`（launch-count=10 / warm-up=5）各采 3+ 次、取中位（`phase=final`，candidate=`final_r8`）；L0 逐例 `max_diff=0`，`perf_records.jsonl` 的 `artifact_path`=`AmaxFwdOp.py`、`artifact_sha256` 与磁盘文件一致。
- **final 记录口径说明**：Round 7 的 4 条 `phase=final`（candidate `final_r7`）按本轮要求**降级为 `phase=merged` 并加 note**（历史与数值原样保留，未删除）；baseline/candidate 行未被改写。`kernel_id` 沿用历史字符串（`perf_opt/_simple_reduce_kernel.py::<factory>::main`）作为 workload 的稳定标识，定稿工件改由 `artifact_path` 指向 `perf_opt/AmaxFwdOp.py`——`_simple_reduce_kernel.py` 现在是同一 kernel 的 import 别名，两者不是两份实现。
- wrapper 集成路径：本轮修复了 `amax.py`（见 Issues）——恢复 **perf_opt 双 import 切换块**（指向 `perf_opt.AmaxFwdOp`）+ **`npub::amax_reduce_*` 命名空间**（原先在 commit `843498c` 被回退成 GPU 参考模块 + 共享 `npub::reduce_*` 名，导致 amin 抢占 amax 的 dispatch、且 `_func()` 报 `missing 1 required positional argument: 'threads'`）。修复后 fp16/bf16/fp32 三 dtype 逐元素 bit-exact。
- 达标结论（`_measure.py` 暖机中位，本会话）：**hs fp16 13.40 µs / 69.6% ✅**、hs bf16 **18.14 µs / 51.4% ❌（Round 7 同码路在快态会话为 15.90 µs / 60.3%）**、3d fp16 5.32 µs / 43.8% ❌（小任务地板）、lseq bf16 9.08 µs / 25.7% ❌（小任务地板）。hs bf16 本轮**未新增达标**：三条结构性路线实测均劣于现状，缺口已归因为工具链能力（int16 vreduce 标量仿真、bf16 reduce 无跨 lane 实现），非 tiling/参数可解。
- 中止原因：`blocked`（bf16 剩余候选优化点全部因编译器/运行时能力失败或实测回退；Round 8 预算内三条路线均已完整验证，无未验证的结构候选）。

### Final Performance Test Data

Deliverable `perf_opt/AmaxFwdOp.py`（resident 工厂 + streaming 工厂 re-export）+ `perf_opt/_simple_reduce_kernel_tiled.py`（candidate `v3_s7_reverify`），每个 tune workload 恰一条 `phase=final` 记录，`artifact_path` 均为 `AmaxFwdOp.py`。`baseline_us` = round-0 `phase=baseline` 同口径记录；两列均取自 `perf_records.jsonl`。

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/reduction/amax/amax_kernel/perf_opt/_simple_reduce_kernel.py::_simple_reduce_kernel::main | hidden-state-reduce-float16 | 14.82 | 13.80 | v3_s7_reverify |
| examples/TileOPs/tileops/kernels/reduction/amax/amax_kernel/perf_opt/_simple_reduce_kernel.py::_simple_reduce_kernel::main | hidden-state-reduce-bfloat16 | 19.72 | 17.52 | v3_s7_reverify |
| examples/TileOPs/tileops/kernels/reduction/amax/amax_kernel/perf_opt/_simple_reduce_kernel.py::_simple_reduce_kernel::main | 3d-multidim-reduce-float16 | 8.64 | 5.40 | v3_s7_reverify |
| examples/TileOPs/tileops/kernels/reduction/amax/amax_kernel/perf_opt/_simple_reduce_kernel.py::_simple_reduce_kernel_tiled::main | long-seq-reduce-bfloat16 | 9.30 | 6.52 | v3_s7_reverify |

> **v3_s7_reverify 接管说明（candidate `v3_s7_reverify`，2026-10-30 12:31–12:46 同会话 3 次 op 级 report 中位）**：`final_us` 全部来自同一候选在同一次复核会话内的 3 次 `tileops.reporting.cli run --prof-mode msprof` 中位（`reports/tileops/20261030_1231*`/`1234*`/`1237*_AmaxFwdOp`），`perf_records.jsonl` 的 4 条 `phase=final`（`candidate_id=v3_s7_reverify`）与之一一对应；上一轮 `v3_smalltask_s7` 的 4 条 `phase=final` 已按 signal-registry §5 降级为 `phase=merged`（数值原样保留 + note）。
>
> **v3_20261101_smalltask 原始接管说明（candidate `v3_smalltask_s7`）**：上表已按本轮定稿重测更新（同一最终候选、同会话 2–7 次 `msprof op` 中位；`perf_records.jsonl` 中 r8/r9 的旧 `phase=final` 行按 signal-registry §5「二次进入 Phase 3」条款降级为 `phase=merged` 并加 `note`，数值原样保留）。本轮**只改 streaming（tiled）模块**：long-seq 的 2 行流水段（S7）；resident 模块 `AmaxFwdOp.py` 的 sha256 与任务开始时的备份**逐字节相同**（`f5ec73477b84d9cba1cbd56e3c8cda06c8106812469b60fcdb43900ac3700a93`），因此 hs/3d 的差异只可能是会话态（见文末「本轮会话态」）。

> long-seq 集成基线为 crash（wrapper tile_n=16384 被 baseline kernel 拒绝）；上表 9.30 = 同一 baseline kernel 在合法 tile_n=8192 下实测（可对账记录见 `perf_records.jsonl` phase=baseline）。

### 逐 case 结果（生产 CLI 3 次单发 + 暖机口径）

生产口径 = `python -m tileops.reporting.cli run --op AmaxFwdOp --prof-mode msprof`（每 case **单发 1 次**，`sample_count=1`；report 的 Bandwidth 常数 1.8 TB/s、只计 GM 读）。本会话共跑 3 次（`reports/tileops/20261026_113306 / 113537 / 113750_AmaxFwdOp`）；正确性 **14/14 通过**（Avg Max Abs Error = 0.00e+00）。

| case | dtype | 基线 Ratio | 定稿 Ratio（生产单发中位） | 3 次单发 | 暖机 Ratio（`_measure.py` 10-launch 中位） | ≥60% | 未达标原因 |
|---|---|--:|--:|--:|--:|:-:|---|
| hidden-state-reduce | fp16 | 64.7% | **69.77%** | 69.45 / 69.77 / 71.81 | 69.6% | ✅ | — |
| hidden-state-reduce | bf16 | 48.6% | **55.68%** | 54.00 / 55.68 / 56.35 | 51.4%（本会话暖机）/ 60.3%（Round 7 快态会话，同码路） | ❌（临界） | bf16 无原生跨 lane 最大值归约（`npur_arch.supports_native_bf16=False`；`T.reduce_max` bf16 → UB overflow / vector-core timeout），必须整块升 fp32：片上流量 10 B/元素（fp16 为 4）。Round 8 实测的三条替代路线（bf16 vmax 折叠 / int16 位模式 / pair-max）**全部更慢或不可用**，见 Round 8 表。 |
| 3d-multidim-reduce | fp16 | 27.7% | 42.52% | 42.21 / 42.52 / 42.84 | 43.8% | ❌ | 4 MB 小体量（理想 2.4 µs）+ 128 行仅够每核 2–4 行，DMA 与向量链无法重叠；与 Round 7 的 45.05% 差异属单发/会话抖动（同码路）。 |
| long-seq-reduce | bf16 | 25.8% | 25.22% | 25.22 / 25.22 / 24.95 | 25.7% | ❌ | 与 3d 同因（4 MB 小体量 + 64 行），单行还需按 8192 列分 4 段串行规约；tiled 文件本轮未改动。 |

> **会话对照（同码路、不同会话）**：Round 7 记录 hs bf16 生产单发中位 57.48%（暖机 60.29–60.59%）；本会话同一条 kernel 为 55.68%（暖机 51.4%，7 次暖机跑 16.84–18.42 µs 双态）。**本轮没有改动 kernel 体**，差异全部来自 run-state/会话（pattern-library `BP_run_state_bimodality`）——这也是本轮所有候选必须同会话交错对比的原因。

## Skill Retrospective

- **流程问题 1（多 kernel 工件的单文件对账假设）**：`signal-registry.md §5` 与 `gate_lint._perf_records_workload_lint` 假定一个算子只有**一个** `perf_opt/{op}.py`（`final_artifact_path` 单值 + 每条 final 记录 `artifact_path` 必须与它相等）。TileOPs 集成算子的 Stage 4 共享一个 `perf_opt/` 目录却可含**多个函数文件**（本 op：resident + tiled，wrapper 双 import），此时次函数的 final 记录在单值对账下必然被判为「未测最终 perf_opt 文件」。建议：`kernel_id` 粒度对账（每个 kernel_id 允许各自的 final 文件）或 harness/optimize 多函数场景显式传入 `final_artifact` 集合。
- **流程问题 2（baseline 不可运行的 workload 无有效 baseline 记录）**：gate 要求每个 tune workload 有 baseline 记录，但集成态对 long-seq 直接 crash。本日志以「同一 baseline kernel + 预算合法 tile_n」补一条可对账 baseline，并在 reason 注明；建议标准明确此类情形的 baseline 口径。
- **证伪更正记录**：Round 2 首次单次测量显示 full-40 网格把 hs bf16 从 56.5% 提到 59.3%，一度疑似结构性收益；interleaved 3 次复测后 exact-32 56.99/56.86/57.34 vs full-40 56.53/56.66/59.03 → 判定为 run-state 双态噪声（<5% 差异须走多次复测），维持整除网格。这正是 BP_run_state_bimodality 的场景。
- **Round 7 证伪更正（推翻上面的 Round 2 结论）**：本轮用「同结构 + S6」的 5 对交错 A/B 得到 full-40 **5/5 配对胜**（中位 15.80 vs 16.76 µs，-5.7%），与同族 `AminFwdOp` 的 4 次交错 A/B（bf16 +7.2%）方向一致 → **Round 2 的「打平」是单/少次测量落在 run-state 慢态的误判**。教训：<5% 平区裁决必须按 T-2 做**同结构、多配对、取中位**的 A/B；不同结构下的历史单次数字不可跨轮直接比较（本轮 S6 静态 extent 单独也只值 ~3.5%，与网格效应叠加后才到 -6.6%）。
- **流程问题 3（生产口径与调优口径的采样数不一致）**：`tileops.reporting.cli run --prof-mode msprof` 每 case 只采 **1 次 launch**（`benchmark.json: sample_count=1`），而 Stage 4 的 `_measure.py` 用 launch-count=10/warm-up=5 取中位。同一最终 kernel 的生产单发分布在 57.41–60.99%（3 次），暖机中位 60.29–60.59% —— 单发口径的 ±3% 抖动足以跨越 60% 门槛。建议：性能目标口径显式声明 launch 次数（或要求 ≥5 次取中位），否则「是否达标」会被采样噪声决定。本日志同时给出两种口径的全部数字。
- **流程问题 4（final 记录与定稿工件绑定的 append-only 冲突）**：gate 4 要求每个 tune workload 恰有一条 final、且哈希等于当前定稿；当二轮调优改写定稿时，旧 final 行必然哈希失配。本轮把旧 4 条 final 原样归档到 `perf_records_prev_final_archive.jsonl` 后再写入新 final，并保留全部 baseline/candidate/merged 行。建议标准明确「final 行可被同一 workload 的新 final 取代（历史入 archive）」。
- **编译器能力缺口（Round 7 实证，应回写 `capability-gaps.md`）**：`T.Parallel` 体内对同一 shared buffer 的两个 `T.cast`（`x_ub[ii,jj]` / `x_ub[ii,jj+N/2]`）会被 NpuLoopVectorize 提升成两个整块 fp32 `tensor.empty()`，但**不下发对应的 vcast**，生成内核读未初始化 UB（`max_diff=1.72`）。~~该形态本可把 bf16 的 UB 流量从 8B/元素降到 4B/元素、实测掉到 13.14 µs（72.96%），是 bf16 突破 60% 的唯一结构性路径；当前只能放弃。~~ **2026-11-02 更新：缺陷已就地修复（CG-2026-0051，status=fixed）。修复后同形态 bit-exact，但实测 20.12 µs / 45.69%，仍慢于定稿 17.58 µs / 50.93%；13.14 µs / 72.96% 是 cast 被删除后的假数，"突破 60% 的唯一结构性路径"不成立。详见附录 A。** 层级：TileLangIR pass（NpuLoopVectorize / 向量化 cast 提升）。

### Round 8 复盘（流程 + 证伪更正）

- **证伪更正（重要，推翻文档与旧结论）**：①`T.vmax` **支持 bf16**（`docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1 标 `×`，实测 4096 元素 bit-mismatch=0）——"文档缺项 ≠ 不支持"，本 op 因此重开了路线 A；②`T.reduce_max` **int16/uint16 文档标 `√` 但被标量仿真**（`aiv_scalar_time` 14.82 µs vs fp32 4.66 µs）——"文档支持 ≠ 可用/快"，与 `PL-1.22` 的"文档缺项 ≠ 不支持 ≠ 有收益"互为反向。两条都说明：**dtype 级 API 结论必须按 (功能 × 性能) 双维实测**，只测功能会误判可用性。
- **流程问题 5（首轮必查项缺"API 性能维度"）**：Phase 0 的首轮必查项要求重估"向量化轴与布局"，但**没有要求重估"归约原语的 dtype 性能谱"**。本 op 的 bf16 天花板恰恰只在归约原语的 dtype 维度上——若首轮就先测「同 shape 下 fp16/fp32/int16/bf16 四种 reduce 的 vec/scalar 分解」，可在 1 小时内判定整类方向，而不是先实现再测。建议：把"归约/规约类算子的 dtype 原语性能谱（vec + scalar 分解）"加入 reduction 族的首轮必查项。
- **流程问题 6（`T.serial` 内 in-place 向量 op 的 loop-carried 状态）**：`TRAP-vselect-inplace-carried-state` 只登记了 `vselect`；本轮新增实证 **`vmax(dst==src0)` 在 `T.serial` 多迭代同样丢状态**（K=4 折叠 `max_diff=1.891`、K=8 `2.156`，K=2 正常）。已按 ED-C 回写 `traps-compiler.md`（见下）。
- **流程问题 7（`--level all` 不覆盖"多迭代折叠"形态）**：本轮 K>2 的错误之所以能被抓到，是因为实测 kernel 直接跑了 `max_diff` 检查；若只跑 L0（其用例都是单次 reduce），K=4 折叠的静默错会漏到 benchmark。建议：**kernel 内的归约是"跨迭代累加"形态时，L0 必须含 ≥3 个分段的 case**（本 op 的 tiled 路径已有，resident 路径没有）。
- **`BP_run_state_bimodality` 再次成立**：同一条 kernel（sha 相同）跨会话 15.90 vs 18.14 µs（+14%），会话内 7 次跑 16.84–18.42。本轮所有裁决都用**同会话**数字（current best 18.14 vs 候选 21.68/27.94/48.24），未跨会话比较。

### 价值点提案（供 `tilelang-skill-evolver` 蒸馏）

- **P / vp_type=P（优化模式，正向）——「reduce 类逐任务写出 → 每核一次批量化写出」**：机制 = persistent reduce 核每任务迭代单独发一次 UB→GM 写出时，MTE3 单次写出固定开销（~0.4–0.9 µs，与 payload 无关）随任务数线性累积，成为小/中 workload 主导项；把各任务结果暂存扁平 UB 并在核尾**每核一次**写出，`aiv_mte3_ratio` 由 0.64→0.01。证据：本 op hs fp16 mte3 0.64→0.01、3d 0.66→0.10，Task Duration 14.82→13.28 / 8.64→5.30。repro：`_variants_r2.py::_contig_bal`（before = baseline 逐任务写 / after = out_acc 批量写）。toolchain_stamp: tilelang dev-build + CANN 8.5.2 + Ascend910_9362。回写位置：本日志 + `_variants_r2.py`。
- **D / vp_type=D（实测数据）——「bf16 同域 reduce_max 在 CANN 8.5.2 仍不支持」**：机制 = `T.reduce_max` 对 bf16 src/dst 触发 `Check failed: buffer_shape` TVMError，必须 `vcast` 到 fp32。证据：`_variants_r4b.py::simple_bf16_nocast` 编译失败。repro：同文件。toolchain_stamp 同上。
- **D / vp_type=D（实测数据）——「大 staging tile 的 auto-multi-buffer 竞态阈值」**：64 KiB 单级（128 KiB doubled）fp32 staging 在 fp16/fp32 tiled reduce 路径间歇单元素错（2/5），32 KiB 稳定（0/5）；doubled 包络收到 96 KiB 即消除。证据：baseline 与 perf_opt 同复现、clamp 后 0/8。repro：`_simple_reduce_kernel_tiled.py`（S0 前后）。toolchain_stamp 同上。
- **R / vp_type=R（流程规则）——多函数 TileOPs 算子的 perf_records 单文件对账假设**：见「流程问题 1」；建议改 `kernel_id` 粒度对账或传入 final 工件集合。
- **P / vp_type=P（优化模式，正向）——「persistent reduce：满核网格 + 编译期静态行块 extent」**：机制 = ① 当 logical 块数不是核数的整数倍时，取「≤N 的最大整除因子」会让部分 vector core 空闲（本 shape 512 块只用 32/40 核），改用满核 + `cid*nl//nk` 连续均衡区间并保留 guard 可再拿 4–6%；② `M % block_m == 0` 时把 `real_m = min(block_m, M-off_m)` 换成编译期常量 extent 的 `T.copy`，去掉热循环每任务的标量 extent 运算与 masked-DMA 形态，再拿 ~3.5%。证据：hs bf16 16.76→15.80 µs（5/5 配对）、16.32→15.83 µs（22 run 中位）；hs fp16 13.28→13.42 持平（暖机）而生产单发 71.65→74.09。repro：`_variants_r7c.py`（32 核对照）vs `_variants_r7b.py`（40 核 + 静态 extent），以及 `_simple_reduce_kernel.py` 内 S1′/S6 注释。toolchain_stamp: tilelang 0.1.4 + CANN 8.5.2 + Ascend910B3。回写位置：本日志 + `_variants_r7b.py`/`_variants_r7c.py`。
- **D / vp_type=D（实测数据）——「bf16 提升链的量化代价（Ascend910B3, CANN 8.5.2）」**：同一 hidden 形状下 fp16 vs bf16 的 profile 对照（均满 40 核）：`aiv_vec_time` 2.77 → 7.50 µs（+4.73），`aiv_time` 11.21 → 14.10 µs（+2.89），`aiv_mte2_time` 10.03 → 9.83（−0.20），Task 13.42 → 15.90 µs。即 bf16 的全部额外成本来自「整宽 bf16→fp32 vcast（4B/元素写）+ fp32 vreduce（4B/元素读）」这 8B/元素的 UB 流量，DMA 侧无差异。证据：`profiles/FIN7_hs_fp16_1`、`profiles/FIN7_hs_bf16_1`。repro：`_simple_reduce_kernel.py` bf16/else 两个 prim_func 的对照。toolchain_stamp 同上。回写位置：本日志。
- **D / vp_type=D（编译器缺陷，已修复）——「NpuLoopVectorize 丢失 T.Parallel 内的双路 T.cast」**：两个同 buffer、不同偏移的 `T.cast` 被提升成 `tensor.empty<pure fp32 tile>` 但不下发 vcast → 读未初始化（max_diff 1.72）。**2026-11-02：根因是 `LoopCarriedVarCollector` 未收集 `tl.npuir_cast` 的结果，cast 变量没有进 `scf.for` 的 iter_args 被当死代码删掉；已在 codegen 就地修复（新增 `tl.npuir_cast` 分支 + VcastCodegen 支持子区域）。修复后 bit-exact，但实测 20.12 µs / 45.69%，慢于定稿 17.58 µs / 50.93% —— 13.14 µs / 72.96% 是 cast 缺失时的假数，不构成性能路线。** 证据：`_variants_r7q.py`（修复前 max_diff=1.72 / 修复后 0，性能见附录 A）。repro：`_variants_r7q.py`。toolchain_stamp: 发现 CANN 8.5.2 + Ascend910B3；修复验证 CANN 9.0.0 + Ascend910_9362 + 2026-11-02。回写位置：本日志 + `.agents/evolution/capability-gaps.md` CG-2026-0051（status=fixed）。

- **D / vp_type=D（实测数据）——「int16/uint16 vreduce 文档标支持但被标量仿真」**：`T.reduce_max`/`T.reduce_min` 对 int16（同 uint16）**功能正确且 bit-exact**，但 lowering 走标量路径：hs bf16 (2048×4096, bm=4, 40 核) 上 `aiv_scalar_time` **14.82 µs**、`aiv_vec_time` 5.07 µs，合计 27.94 µs；同结构 fp32 reduce 为 scalar 4.66 / vec 8.21 / 18.14 µs。推论：**int16 位模式归约（IEEE 单调映射）在本工具链上不可能赢**，即使省下 60% 片上流量。证据：`profiles/cost_v_max_only/PipeUtilization.csv` vs `profiles/ab_r8_old_1/…`。repro：`_probe_r8_variants.py::v_max_only`（`python _measure.py --kernel-file _probe_r8_variants.py --factory v_max_only --M 2048 --N 4096 --dtype bfloat16 --block-m 4`）。toolchain_stamp: tilelang 0.1.2+da4f6a4（dev build）+ CANN 9.0.0 + Ascend910_9362。
- **D / vp_type=D（实测数据）——「bf16 逐元素 vmax 可用（文档标不支持）」**：`T.vmax(a_bf16,b_bf16,c_bf16)` 编译通过且 bit-exact（4096 元素 0 失配）。但 bf16 **没有跨 lane reduce**：`T.reduce_max` bf16 在 (64,512) 报 `ub overflow, requires 2097152 bits while 1572864 available`（= 4× tile），缩到可容纳后运行期 `vector core timeout`（507034）。→ bf16 amax 只能"逐元素 vmax 折叠 + 另一域收尾"，而折叠实测亏本。证据：`_probes_r8.py` A/A′。repro：`_probes_r8.py`（A 段 kernel）。toolchain_stamp 同上。
- **D / vp_type=D（实测数据）——「bf16 vmax 折叠的代价曲线」**：(2048×4096, bm=4, 40 核) 上，单次半宽折叠后升 fp32 为 **21.68 µs**（bit-exact），比不折叠的 18.14 µs 慢 19%；多折叠（K=4/8）因 in-place `vmax` 的 loop-carried 丢状态而**数值错**（1.891 / 2.156），改用子区域切片折叠则编译失败（`vmax` 形状校验拒绝 (bm,N)→(bm,W)）。→ "折叠树比内置 reduce 慢"（`PL-1.21 §22` 的结论）在 bf16 上方向一致，本轮补齐绝对数字。repro：`_probe_r8_variants.py::v_fold2`。toolchain_stamp 同上。
- **P / vp_type=P（优化模式，反向）——「矢量归约的 dtype 性能谱必须先行」**：优化"某 dtype 的归约"时，第一步应实测 `{fp32, fp16, int32, int16, bf16}` 五种 dtype 的 `(vec, scalar)` 分解，再决定结构；本例 int16 的 scalar 是 fp32 的 3.2×，若先测可省掉整条 B 路线的实现。repro：`_probe_r8_variants.py` 的 5 个 factory 即该谱的最小骨架。toolchain_stamp 同上。
- **R / vp_type=R（流程规则）——reduction 族首轮必查项补"归约原语 dtype 性能谱"**：见复盘「流程问题 5」。

## Issues（记录，不阻塞）

- 集成基线 long-seq 崩溃（wrapper `_tile_n_for_block_m` 用 `_primitives.compute_tile_n`（bf16 2B/元素）给出 tile_n=16384，而迁移 tiled kernel 按 B=6 预算）——本 Stage 4 在 **perf_opt tiled kernel 内**以 S0 clamp 使两侧口径一致，case 恢复可运行；未改 wrapper（conductor 采纳动作）。
- 多函数工件单文件对账假设（见 Skill Retrospective 流程问题 1）。
- hs bf16 仍未稳定 ≥60%：生产单发中位 57.48%（3 次 57.41–60.99%），暖机中位 60.29–60.59%。根因是 bf16 无原生 reduce 导致的强制 fp32 提升（+4.73 µs `aiv_vec_time`），唯一可破解的融合 cast+pair-max 被 NpuLoopVectorize 的 cast 丢失缺陷阻断（见 Skill Retrospective「编译器能力缺口」）。**不构成 `[DESIGN_LIMIT]`**：算法/设计层无更优数学等价式（max 已是单遍精确选择），缺口在编译器 lowering，且预估可达收益 ~17%（13.9 µs 量级）< 2× 门槛，故不产出 `perf_feedback.md`。
- 探索性变体保留在 `perf_opt/_variants_r7*.py`（r7=40 动态、r7b=40 静态 winner、r7c=32 静态对照、r7d/r7f=流水、r7j=关 auto multi-buffer、r7k=手工双缓冲、r7m=固定 reduce 目标、r7n/r7o/r7p/r7q=pair-fold 系列、r7r=cast+半宽折叠、r7s=标量提升、r7t=半宽双 cast），供后续工具链修复后复验；`logs/r7/` 保存本轮 L0/L1 与生产 CLI 日志。

---

# v3_20261101_smalltask -- 小任务 case（long-seq / 3d）专项提速

> 任务边界：纯 Vector 类平均 59.85 → ≥60，本任务只贡献 `long-seq`（25%）与 `3d`（43%）两个小任务 case。
> 约束：模式 `full`，只读 pattern-library / queue / stats；新价值点**因冻结暂缓入队**，证据只写本日志。
> 环境：独占 Chip0（`ASCEND_RT_VISIBLE_DEVICES=0`）、`TILEOPS_MSPROF_TMPDIR=/tmp/opencode/msprof_amx/tmp`、每次 msprof 前 `env -u TILEOPS_MSPROF_OUTPUT_DIR`。
> 备份：`history_version/20261101_smalltask/`（AmaxFwdOp.py / _simple_reduce_kernel_tiled.py / _simple_reduce_kernel.py / opt_log.md / perf_records.jsonl / workload_inventory.json / _measure.py；amin 同）。
> 复用上一轮遗留的 20 个 `_probe_*` 探针与 `_variants_s4.py` / `_variants_s5.py`（未重写），新增 `_run_batch.py` + `cases_v3*.txt` 批处理 runner（T-4）。

## Performance Test Data（baseline，Round 0，本会话重新采集于当前工具链）

口径：`_measure.py`（`msprof op` Task Duration，launch-count=10 / warm-up=5，暖机中位），L0 为 kernel 内嵌 golden 精确比对（`max_diff=0`）。

| 算子 | workload | shape / dtype | 配置（wrapper 传入） | Task Duration(us) | Ratio(%) | L0 |
|---|---|---|---:|---:|---|:-:|
| AmaxFwdOp | hidden-state | 2048×4096 fp16 | bm=4 | 13.180 | 69.46 | PASS |
| AmaxFwdOp | hidden-state | 2048×4096 bf16 | bm=4 | 16.720 | 54.70 | PASS |
| AmaxFwdOp | long-seq | 64×32768 bf16 | bm=1, tile_n=16384→clamp 8192 | **9.200** | **25.00** | PASS |
| AmaxFwdOp | 3d | 128×16384 fp16 | bm=1→promote 2 | **5.500** | **41.46** | PASS |
| AminFwdOp | hidden-state | 2048×4096 fp16 | bm=4 | 16.260 | 56.15 | PASS |
| AminFwdOp | hidden-state | 2048×4096 bf16 | bm=4 | 16.740 | 54.89 | PASS |
| AminFwdOp | long-seq | 64×32768 bf16 | bm=1, tile_n=16384→clamp 8192 | **9.040** | **24.74** | PASS |
| AminFwdOp | 3d | 128×16384 fp16 | bm=1→promote 2 | **5.380** | **41.32** | PASS |

> 会话态说明（`BP_run_state_bimodality`）：本会话 amax hs fp16 69.46 / amin hs fp16 56.15（同一晚报告为 72 / 67），差值来自 run-state 而非代码。**所有裁决只用同会话数字**。

## Round 1 -- long-seq 分段数与 3d 任务数（首轮实验）

现象（base profile，同会话）：long-seq bf16 4 MB 体量、32/40 核、每核 8 个 16 KB 分段（4 列段 × 2 行）串行；3d fp16 每核 64 KB DMA ×2 任务，`aiv_time = mte2 2.05 + vec 0.82 + scalar 0.81`，三管串行。

| 分支 | 改动 | Task Duration(us) | Ratio(%) | L0 | 结论 |
|---|---|---:|---:|:-:|---|
| base | — | 9.200 | 25.00 | PASS | current best |
| **R1a** `_tiled_rows` bm=2 tn=8192 | 每段 2 行 × 8192 列（段宽 32 KB），每核 reduce 数 8→4，32 任务/32 核 exact L=1 | **6.540** | **33.48** | PASS | **winner（-28.9%）** |
| R1b `_tiled_rows` bm=2 tn=16384 | 分段 4→2 | — | — | — | **blocked-compile**：`CodeGenTileLangNPUIRDEV::VcastCodegen` `Check failed: buffer_shape0[i]==buffer_shape1[i]`（chunked `T.vcast(x_ub[0:bm,0:chw], x_work)` 子区域被形状校验拒绝，与 R8 的 vmax 子区域同族） |
| R1c `_tiled_rows` bm=1 tn=32768 | 分段 4→1 | — | — | — | 同上（nch=2 需 chunked vcast） |
| R1d `AmaxFwdOp` bm=4（3d） | 任务 64→32，32 核 exact | — | — | — | **blocked-compile**：`ub overflow, requires 1573120 bits while 1572864`（128 KB stage + 64 KB reduce scratch = 192 KB + 32 B） |
| R1e `_res_rows` bm=2（3d） | 每核一次整块 copy + 一次 reduce | 5.220 | 42.68 | PASS | 与 base 打平（详见 Round 3 交错复测） |
| R1f `_res_rows` bm=4（3d） | 同上 bm=4 | — | — | — | 同上 ub overflow |

## Round 2 -- 手工 depth-2 ping-pong（S5，波前重叠 MTE2/Vector）

动机：base profile 显示三管**串行**（`aiv_time ≈ mte2 + vec + scalar`），pattern-library 常数 `CONST-manual-pingpong-bw-9362` 记录手写 2 深流水可拿 1.24–1.33× 有效带宽。`_variants_s5.py` 用 `T.rs("PIPE_MTE2")/T.set_flag/T.wait_flag` 发出 s0..s3 四阶段。

| 分支 | 改动 | 结果 | L0 | 结论 |
|---|---|---|---|:-:|
| S5-a `_tiled_pp` bm=2 tn=8192 | long-seq 4 段 2 深 ping-pong | 运行期 `507034 Vector core execution timed out`（多核同时超时，`subErrType:4`） | FAIL | **blocked-runtime**：手工 flag 协议在本 kernel 上死锁 |
| S5-b `_res_pp` bm=4 split=4 | 3d 4 阶段 ping-pong | 同上形态（进程未产出 msprof 记录） | FAIL | blocked-runtime |

> 结论：手工 ping-pong 在本 op 上不可用（死锁）；不再投入预算重试（与 R8 的「vmax 折叠 + 标量修复」族同类失败）。

## Round 3 -- 3d 任务数候选的交错 A/B（3 reps，同会话）

判据：`msprof op` Task Duration 中位；<5% 差异按 T-2 走多次交错复测。

| 候选 | r0 | r1 | r2 | 中位(us) | 中位 Ratio(%) | L0 | 结论 |
|---|---:|---:|---:|---:|---:|:-:|---|
| **base**（AmaxFwdOp，bm=1→2，64 任务/40 核） | 5.32 | 5.42 | 5.22 | **5.32** | 42.68 | PASS | 保持 |
| bm4_tn8k（`_tiled_rows`，32 任务/32 核 exact，2 段） | 5.20 | 5.38 | 5.26 | 5.26 | 42.37 | PASS | -1.1%，平区内 |
| bm2_tn16k（`_tiled_rows`，1 段） | 5.50 | 5.72 | 5.66 | 5.66 | 39.63 | PASS | 回退 6.4% |
| res_bm2（`_res_rows`，每核 1 copy + 1 reduce） | 5.34 | 5.42 | 5.32 | 5.34 | 42.84 | PASS | 平区（+0.4%，tie） |
| long-seq bm2_tn8k 复测 | 6.50 | 6.48 | 6.48 | **6.48** | 34.68 | PASS | 稳定复现 |

> **3d 结论**：任务数（64 → 32）不是主导项——三种任务/分段结构全部落在 5.26–5.66 µs 平区。地板探针 `PROBE_s4_floor`（40 核、每核写 1 元素）实测 **2.80 µs**，`PROBE_s4_dma3d`（只搬不算）实测 **4.84 µs** ⇒ 3d 的可达下界 ≈ 2.8（launch）+ 2.04（4 MB DMA，≈2.0 TB/s 聚合，已在硬件峰值）≈ **4.84 µs = 44.0%**。目标 50%（4.66 µs）**低于「launch floor + DMA 峰值」之和**，本工具链不可达；实际值见 Final Summary。

## Round 4 -- 采纳：long-seq 2 行流水段（S7）合入 streaming 模块

合入内容（`perf_opt/_simple_reduce_kernel_tiled.py`，`AmaxFwdOp.py` 逐字节不变）：
1. `_stage_fits(bm, tn, dtype)`：`stage + work + stage//2 ≤ 190 KiB`（measured reduce scratch = 0.5×stage）；
2. factory 期常量 `promote_bm = (M ≥ 2) and (M % 2 == 0) and _stage_fits(2, tile_n, dtype)`，`bm = 2 if promote_bm`；
3. 两个 prim_func 的 stage 从 `(1, tile_n)` 泛化为 `(bm, tile_n)`，`acc/tile 输出/out_acc` 同步为 `(bm, …)`；
4. **`pass_configs={"npuir.enable_auto_multi_buffer": False}` 只在 `promote_bm` 时生效**——2 行 bf16 段 = 32 KiB stage + 64 KiB fp32 work，双缓冲包络会到 192 KiB，实测直接 `507034 vector core timeout`（首次尝试即撞上，见 logs/v3/tiled_level_all.log 的失败版本）；关闭后 L0/L1/L2 全过。`bm == 1` 时 pass config 与旧版本完全一致。

合入前后交错 A/B（**6/6 配对全胜**，同会话，3 对 A/B/A/B）：

| 算子 | 旧（bm=1, tn=8192） | 新（bm=2, tn=8192） | Δ | L0 |
|---|---:|---:|---:|:-:|
| AmaxFwdOp | 9.12 / 9.08 / 9.06（中位 **9.08**，24.58%） | 6.46 / 6.46 / 6.62（中位 **6.46**，35.31%） | **−28.8%** | PASS |
| AminFwdOp | 9.12 / 9.20 / 9.16（中位 **9.16**，24.53%） | 6.50 / 6.42 / 6.54（中位 **6.50**，34.99%） | **−29.0%** | PASS |

非回退检查（本 kernel 全部 tune workload + smoke 精度）：
- `_simple_reduce_kernel_tiled.py --level all` → **L0 12 例 / L1 7 例 / L2 7 例 / Boundary 全过（max_diff=0.00e+00）**，含 (64,32768) bf16（新 S7 路径）、(128,65536) fp16/fp32、(16,32769)、(2,262144)、(129,65536) 等非整除/尾块形态（`logs/v3/tiled_level_all.log`）。
- `AmaxFwdOp.py --level all` 与 `AminFwdOp.py --level all` 全过（resident 路径未改，见下）。
- 3d / hs 的 kernel 代码 sha256 与任务开始时的备份相同 → 无回退可能；本会话实测 hs fp16 14.54–16.68（双态）、hs bf16 16.66–16.74、3d 5.26–5.66（均落在历史 ±3% 内，且为**同码路**）。

### 本轮会话态（`BP_run_state_bimodality` 第 4 次实证）

同一 `AmaxFwdOp.py`（sha `f5ec7347…`）在同一会话内：Round 0 测 hs fp16 **13.18**（69.46%）、Round 4 测 14.54–14.70（62–63%）、Round 5 final 批测 16.28–16.68（53.9–55.5%），随后追加 3 次又回到 14.54–14.70。**±18% 的会话内漂移**，与代码无关。故本轮所有采纳裁决只用**同会话交错 A/B**（上表 6/6 配对），绝对值取同一 workload 全部同会话样本的中位。

## Final Summary（v3_20261101_smalltask）

- 定稿：`perf_opt/AmaxFwdOp.py`（sha256 `f5ec7347…`，本轮**未改**）+ `perf_opt/_simple_reduce_kernel_tiled.py`（**S7 2 行流水段**，本轮唯一改动）。
- **long-seq-reduce bf16：9.08 → 6.46 µs（−28.8%），Ratio 24.6% → 35.3%**（同会话 3 对交错 A/B，6/6 配对胜）。
- **3d-multidim-reduce fp16：不改**（目标 ≥50% 不可达，见下「3d 天花板」）。
- **hidden-state fp16/bf16：不改**（resident 代码 sha 相同）。
- 两算子 `--level all` 全过（`max_diff=0`），smoke 精度通过；`perf_records.jsonl` 的 4 条 `phase=final` 均来自同一候选 `v3_smalltask_s7`、`artifact_path=AmaxFwdOp.py`、哈希与磁盘一致。
- 中止原因：`success`（本任务范围内的可行改动已落地；long-seq 目标 40% 未达，报实际 35.3%）。

### 3d 天花板（目标 50% 不可达，报实际值）

| 探针 | 形态 | Task Duration(us) | 换算下界 |
|---|---|---:|---|
| `_variants_s4.py::_probe_floor` | 40 核、每核写 1 元素（launch 地板） | **2.80** | — |
| `_variants_s4.py::_probe_dma` | 只搬不算（4 MB DMA） | **4.84** | 4 MB / (4.84−2.80) µs = **1.96 TB/s 聚合**（已在硬件峰值之上） |
| 目标 50% | — | 4.66 | **低于「launch 地板 + DMA 峰值」之和 4.84 µs** |

四个结构候选（bm=4/32 核 exact、`_res_rows` 每核 1 copy+1 reduce、tiled 1 段、base 2 段）实测中位 5.26–5.66 µs，全部落在平区（±7%），**任务数不是主导项**。故 3d 报实际 **41–43%**（本会话中位 5.41 µs / 40.77%）。

## Round 5 -- 生产 CLI 复测（官方口径，`msprof`）

`python -m tileops.reporting.cli run --op AmaxFwdOp --prof-mode msprof` → `reports/tileops/20261030_115006_351053_AmaxFwdOp/`，status=**passed**（`tests/ops/test_amax.py` + `benchmarks/ops/bench_amax.py`，正确性 14/14）。

| case | dtype | v2 报告 | v3 本次 CLI | Latency(us) | 说明 |
|---|---|--:|--:|--:|---|
| hidden-state-reduce | fp16 | 69.77 | 67.64 | 13.700 | 代码未改（sha 相同），会话抖动 |
| hidden-state-reduce | bf16 | 55.68 | 56.02 | 16.660 | 代码未改 |
| long-seq-reduce | bf16 | 25.22 | **36.76** | **6.440** | **S7 改动** |
| 3d-multidim-reduce | fp16 | 42.52 | 42.22 | 5.340 | 代码未改 |
| 平均 | — | 48.91 | **50.66** | — | +1.75 pp |

回归命令与日志（本轮全部重跑）：
- `python perf_opt/AmaxFwdOp.py --level all` → L0/L1/L2/Boundary 全过（`logs/v3/resident_level_all.log`）
- `python perf_opt/_simple_reduce_kernel_tiled.py --level all` → L0/L1/L2/Boundary 全过（`logs/v3/tiled_level_all.log`）

> 冻结自证：本任务**未回写** `pattern-library/`、`.agents/evolution/queue.md`、`stats.md`（任务约定覆盖 skill 的 ED-C 回写义务）；新价值点（P/D/R 三条，见 AminFwdOp opt_log 的「Skill Retrospective（v3 追加）」）**因冻结暂缓入队**，证据只落在本目录日志与 `_variants_*.py` / `profiles/v3_*`。`python3 .agents/tools/frozen_check.py` → PASS（62/62 登记文件与基线一致，repro 无新增文件）。

## Round 6 -- S7 复核（先证伪再采纳）：定位 / 机制隔离 / 7 对交错 A/B / 3 次 op 级复测

**触发**：上游复核提出两点质疑：① `perf_opt/_simple_reduce_kernel_tiled.py` 工作区 sha256 与 HEAD 逐字节相同（`2edcaa13…`），若"改动从未发生"，则 long-seq 的 25%→36% 只能归因 run-state；② `perf_records.jsonl` 似乎未追加（仍 41 行）。本轮按"先证伪再采纳"重做整条证据链，**不沿用上一轮任何结论性数字**。

### 6.1 定位：S7 到底在不在？——在，且差异行可逐行指出

| 事实 | 证据 |
|---|---|
| **本轮起点 HEAD（`abab30a`）不含 S7** | `git show abab30a:.../perf_opt/_simple_reduce_kernel_tiled.py \| sha256sum` = `241b3d96…`，与 `perf_opt/_variants_v3_oldtiled.py` **逐字节相同**（该文件即 pre-S7 基线） |
| 含 S7 的工作区文件产生于本轮 | `stat` mtime = **2026-10-30 11:19:31**；docstring L38–46、`_stage_fits` L171–180、`promote_bm` L224、双 `_jit` 分支 L225–231、`bm` 泛化 L236 + 提升 L247–248、`num_logical=_ceildiv(M,bm)` L250 / `static_m` L253 |
| 该编辑在 **11:46:12** 的备份提交 `e8d3157` 里被入库，故"现在的 HEAD 已含 S7" | `git show e8d3157:...` = `2edcaa13…`（= 工作区）；提交时间 11:46:12 **晚于**文件 mtime 11:19:31 |
| `perf_records.jsonl` 的 4 条 `phase=final` 已在 **11:49:06** 追加 | 41 行中的末 4 行，`timestamp=2026-10-30T11:49:06+08:00`，`candidate_id=v3_smalltask_s7` |

⇒ **结论（驳回"改动从未发生"）**：「工作区 sha256 == HEAD」是**备份提交晚于编辑**的必然结果，不能作为"没改"的证据。S7 真实落地，差异集中在：`_stage_fits()` 新增、`promote_bm` 工厂期常量、按 `promote_bm` 切换 `pass_configs`、`bm` 由常数 1 泛化为 `max(1,min(block_m,M))` 并在 `M` 偶数且装得下时提升为 2、4 个 prim_func 的 `x_ub/x_work/x_tail/acc/out_acc` 由 `(1,…)` 泛化为 `(bm,…)` 且 copy 分 `static_m` / 动态 `real_m` 两种形态。

> **代码注释与本文数字的关系**：`_simple_reduce_kernel_tiled.py` 文件头与 `promote_bm` 处的注释写的是上一轮单次交错视图的 `9.20 → 6.48 µs（−29.6 %）`；本轮 7 对复核值为 **9.34 → 6.62 µs（−29.1 %，7/7）**。**故意不改注释**，以保持被测产物的 sha256（`2edcaa13…`）与实测二进制严格对应；数字以本文为准。

### 6.2 生效路径：trace 级可复现（不是推测）

1. wrapper 传入 `block_m=1, tile_n=16384`（生产配置，见 `reports/tileops/20261030_120108_759894_AmaxFwdOp/benchmark.json`）；
2. `_clamp_tile_n(32768,'bfloat16',16384)` 用 `_B_ELEM['bfloat16']=6`（x_ub 2 + fp32 work 4）把 16384 **收紧到 8192**（`2*16384*6 = 192 KiB > 96 KiB` 包络）；
3. `_stage_fits(2, 8192, 'bfloat16') = 32768 + 65536 + 16384 = 114688 ≤ 190 KiB` → **promote_bm = True**，`bm` 提升为 2，且 `enable_auto_multi_buffer=False` 生效。
4. TIR（打桩 `tilelang.jit` 后 trace，`_probe` 级证据）：
   - 旧：`x_ub (1, 8192)`，`Block Dim=32`、每核 2 个任务 × (1 + 3) 段 = **8 次 (copy→cast→reduce→max)**；
   - 新：`x_ub (2, 8192)`，每核 1 个任务 × 4 段（每次 copy 2×8192、`npuir_reduce` 跨 2 行）= **4 次**。
5. 设备二进制：旧 `aicore_binary.o` = 3376 B `15c96f43…`；新 = 4408 B `63a8542d…`；**生产 CLI 报告里 long-seq 的 artifact（`main_e36c4c0p`）= 4408 B `63a8542d…`** ⇒ 生产跑的就是 S7 版本。

### 6.3 机制隔离：收益不是来自 pass config

单点控制变量（PROBE `_probe_s7isol_bm1_nomb.py` = 保留 bm 泛化但**禁止提升**，同时**仍然**关 `enable_auto_multi_buffer`）：

| 分支 | 构造 | 3 次测量（µs） | 中位 | Ratio |
|---|---|---:|---:|---:|
| old | bm=1，multi-buffer 默认（on） | 9.24 / 9.28 / 9.36 | 9.34 | 24.63% |
| **isol（PROBE）** | bm=1，但强制关 multi-buffer | 9.30 / 9.28 / 9.24 | **9.28** | 24.74% |
| new | bm=2 + 关 multi-buffer | 6.48 … 6.66 | **6.62** | 34.47% |

⇒ 单独关 multi-buffer 落在噪声内（9.34 → 9.28，−0.6%）；**−29% 来自 2 行 stage 本身**（每核 reduce/DMA 段数减半 + DMA 宽度 16→32 KB）。

### 6.4 同会话交错 A/B（唯一裁决，7 对，逐对交替顺序）

`_measure.py`（`msprof op`，launch-count=10 / warm-up=5），A/B、B/A 交替以抵消慢漂移：

| 算子 | 旧（bm=1，7 次） | 中位 | 新（bm=2，S7，7 次） | 中位 | Δ | 配对胜 |
|---|---|---:|---|---:|---:|---:|
| AmaxFwdOp | 9.24 / 9.28 / 9.36 / 9.34 / 9.36 / 9.22 / 9.34（24.63%） | **9.34** | 6.48 / 6.64 / 6.50 / 6.66 / 6.50 / 6.62 / 6.64（34.47%） | **6.62** | **−29.1%** | **7/7**（符号检验 p=0.0078） |

3d 对照（resident 模块，S7 未触及，**两臂是同一文件** `AmaxFwdOp.py`，M=512/N=4096/fp16/bm=1 = 生产 3d 配置）：旧臂 6.66/6.42/6.78/6.54/6.42（中位 6.54）vs 新臂 6.62/6.76/6.54/6.72/6.56（中位 6.62），2/5 胜、p=0.81、Δ −1.2% ⇒ **无方向、±3% 抖动带**，确认 3d 未被牵连。

> **否证「run-state 双态」假设**：若 25%/36% 是会话态，则交替执行的 14 次测量不可能 7/7 同向；实测每对 Δ 恒定在 2.60–2.86 µs。会话态在本任务里确实存在（同码路 hs fp16 曾 13.18↔16.7），但它解释不了**同一分钟内交替**出现的稳定 2.7 µs 差。

### 6.5 3 次 op 级报告复测（最终达标判定数字）

`python -m tileops.reporting.cli run --op AmaxFwdOp --prof-mode msprof` × 3（`reports/tileops/20261030_123135 / 123426 / 123719_AmaxFwdOp`，均 status=passed，正确性 14/14）：

| case | dtype | run1 | run2 | run3 | **中位 Task Duration(µs)** | **中位 Ratio** |
|---|---|---:|---:|---:|---:|---:|
| hidden-state-reduce | fp16 | 13.80 | 13.50 | 14.06 | **13.80** | 67.06% |
| hidden-state-reduce | bf16 | 18.10 | 17.52 | 16.80 | **17.52** | 53.20% |
| **long-seq-reduce** | bf16 | 6.68 | 6.52 | 6.46 | **6.52** | **35.74%** |
| 3d-multidim-reduce | fp16 | 5.40 | 5.44 | 5.30 | **5.40** | 43.48% |
| **平均** | — | — | — | — | — | **49.87%** |

**与上一轮声明的差异（如实更正）**：long-seq 由"36.76% / 6.44 µs"更正为 **35.74% / 6.52 µs**；算子平均由"50.66%"更正为 **49.87%**（上一轮是**单次** CLI 单发，本轮是 3 次复测中位）。**机制结论不变**（S7 带来 −29%），但绝对数字以本轮 3 次复测为准。

### 6.6 回归

- `python perf_opt/AmaxFwdOp.py --level all` → **All check passed**（`logs/v3_s7verify/level_all_AmaxFwdOp.log`）
- `python perf_opt/_simple_reduce_kernel_tiled.py --level all` → **All check passed**（含 L0-1 `(64,32768) bf16` = S7 路径、L0-8 `(4,13000) bf16` = S7 + 尾块）（`logs/v3_s7verify/level_all_amax_tiled.log`）
- 3 次 op 级报告均 `status=passed`、正确性 14/14、`Avg Max Abs Error = 0.00e+00`。

## Final Summary（v3_s7_reverify）

- 定稿文件未变：`perf_opt/AmaxFwdOp.py`（sha256 `f5ec7347…`，本轮未改）+ `perf_opt/_simple_reduce_kernel_tiled.py`（sha256 `2edcaa13…`，S7）。本轮**只重做证据链与数字复核，未改动任何 kernel 代码**。
- long-seq-reduce bf16：同会话 7 对交错 A/B **9.34 → 6.62 µs（−29.1%，7/7，p=0.0078）**；3 次 op 级复测中位 **6.52 µs / 35.74%**。
- 算子平均 Ratio **49.87%**（3 次复测逐 case 中位后平均）；4 例中 1 例达标（hs fp16 67.06%），hs bf16 53.20%、3d 43.48%、long-seq 35.74% 未达 60% 线。
- 中止原因：`success`（复核范围内结论闭合；S7 有效且已生效，数字已按本轮实测校正）。

## 附录 A（2026-11-02）：编译器缺陷 CG-2026-0051 已修复，pair-fold 的 ~73% 预期被否证

### A.1 缺陷根因与修复

- **根因**：`T.Parallel` 内同一 shared buffer 的两路 `T.cast` 被 NpuLoopVectorize 分解进循环后，`LoopCarriedVarCollector` 没有处理 `tl.npuir_cast`，cast 的结果变量没有被收进 `scf.for` 的 `iter_args`；MLIR canonicalize 因此把"没人用"的 cast 当死代码删除，后续 vmax 实际读的是 `tensor.empty()` 的未初始化 fp32 片（`max_diff=1.72`）。
- **修复 1**（`src/target/codegen_npuir_dev_common.cc:1386`）：循环携带变量收集器新增 `tl.npuir_cast` 分支，把 cast 的 src/dst 都声明为携带变量。
- **修复 2**（`src/target/codegen_npuir_dev.cc:873`，`VcastCodegen`）：vcast 支持 `src_range`/`dst_range` 子区域，按 `extract_slice → vcast → insert_slice` 生成；`src_range` 含动态 extent 时回退整块语义。整块 cast 走 `NeedGenInsertSlice` 的直通分支（`codegen_npuir_dev.cc:689`），生成的 IR 与修复前一致，不影响已定稿内核。
- **验证**：修复后 raw NPUIR 可见 `scf.for ... iter_args(...)` 与 `extract_slice [1,128] → vcast → insert_slice`；64×256 小形状与 2048×4096 bm=4 生产形状均 `max_diff=0`。
- **回归**：`testing/npuir/parallel_ops/test_cast_parallel.py` 新增 3 个 pair-cast 用例（bf16 4×256、bf16 2×1024、fp16 4×512），全部 bit-exact，该文件 10 项全过；生产 `AmaxFwdOp.py --level all` 31 项全过。

### A.2 13.14 µs / 72.96% 是假数

那次测量时 cast 已被整段删除：内核**根本没有做** bf16→fp32，等于跳过了 8 B/元素的片上流量，所以又快又错。缺陷修复后同一写法必须真正付出这笔流量，**"修好 bug 就能到 ~73%" 不成立**。

### A.3 修复后的同会话实测（2048×4096 bf16，bm=4，msprof 10 launch / 5 预热取中位）

> 工具链：本附录测量为 **CANN 9.0.0 + Ascend910_9362**（与正文 Round 0–8 的 CANN 8.5.2 + Ascend910B3 不同），因此与历史绝对值不可直接比较，只在附录内部同会话对比。

| 形态 | latency (µs) | ratio (%) | 说明 |
|---|---:|---:|---|
| Stage-3 baseline（`amax_kernel/_simple_reduce_kernel.py`） | 20.56 / 20.70 / 20.68 | ~44.5 | 无 S1′/S6 |
| **定稿 `perf_opt/AmaxFwdOp.py`（S1′+S6）** | **17.58** | **50.93** | 仍是最快 |
| `perf_opt/_variants_r7q.py` pair-fold（修复后） | 20.16 / 19.94 / 20.12，中位 **20.12** | **45.69** | bit-exact，比定稿慢 **14.4%** |
| 同会话 fp16 参考（`_simple_reduce_kernel.py`） | 14.24 | 64.19 | bf16 与 fp16 仍差 ~3.3 µs |

- `_variants_r7q.py` 是在定稿同款 Stage 4 结构上只替换 bf16 分支（其余 S1′ 满 40 核 / S6 静态 extent 调度相同），因此 20.12 vs 17.58 的差可归为 **pair-fold 自身的净成本**（多一趟整宽 vmax 折叠 + 两个半宽 fp32 缓冲的拆分写入），约 2.5 µs，正好把调度收益吃回去。
- 测量顺序为两次交错 A/B（Stage-3 baseline ↔ pair）后再测定稿，同一编译器（修复后）、同一会话，可直接对比。
- 结论：**bf16 天花板仍是 ~51–53%，pair-fold 不采纳；定稿 `AmaxFwdOp.py` 保持现状不变。**

### A.4 为什么 pair-fold 在 bf16 上必然亏

- 逐元素 max 若留在 bf16 域，会被 `LegalizeNpuirBF16` 再拆成"fp32 cast + fp32 vmax"，流量与显式提升等价，省不掉（另有一个 probe 触发 bishengir dominance 报错）。
- 显式路径的流量：读 bf16 整宽 2 B + 写两个半宽 fp32 各 2 B = 6 B/元素，再 vreduce 读 4 B/元素；原路径是整宽 vcast 写 4 B + vreduce 读 4 B。**并没有真正减半**，反而多一趟 vmax 与一次半宽读。
- UB 预算：NpuLoopVectorize 给 cast 临时缓冲按**源**形状 `(bm, N)` 分配，bm=4 / N=4096 时两个整宽 fp32 temp = 2×64 KiB，再加 `x_ub` 32 KiB 与 `m_all/out_acc` 等小缓冲后超 192 KiB 上限（`ub overflow, requires 1574656 bits while 1572864 bits available`，只差 224 B）。`_variants_r7q.py` 只能用显式半宽目标缓冲（`tmp_lo/tmp_hi/x_half`）绕过。

### A.5 留档

- `_variants_r7q.py --level all`：31 PASS，`All check passed!`。
- `AmaxFwdOp.py --level all`：31 PASS，`All check passed!`（本次修复后复跑）。
- 缺陷记录：`.agents/evolution/capability-gaps.md` **CG-2026-0051 → status=fixed**。
