# Stage 4 调优日志 — `_engram_gate_conv_bwd_kernel`

- 场景: `scenario=optimize`, `mode=full`
- 项目 / 算子: `engram_gate_conv_bwd` / `_engram_gate_conv_bwd_kernel`
- 算子目录: `examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel/`
- 基准 kernel（只读）: `_engram_gate_conv_bwd_kernel.py`（sha256 `668a00d5f012a89697fb04dd36c00121915217fca995b46b6c47a4b749964543`）
- 调优产物: `perf_opt/_engram_gate_conv_bwd_kernel.py`（drop-in，同工厂签名 `(M, seq_len, d, eps, dtype)`，返回闭包调度 K1/K2/K3）
- 性能口径: **`msprof op` `Task Duration(us)`（kernel-only）**，唯一口径。该算子无 `T.gemm`，非 MixCV；工厂返回闭包按序启动 **3 个独立 jit kernel**（`k1`/`k2`/`k3`），故一个 benchmark workload 登记为 3 个 `(kernel_id, workload_id)` 对，`target_kernel_name` = `k1`/`k2`/`k3`（`OpBasicInfo.csv` 实测 `Op Name=k1/k2/k3`, `Op Type=vector`，`Block Dim` = `M·num_tiles` 或 6）
- 噪声阈值: 3%；<5% 差异一律走 `perf_opt/ab_interleaved.py` 交错 A/B/A/B 多 run 协议（`BP_run_state_bimodality` 对策）
- `max_rounds` = 10（实际 4 轮）；实验分支 8 个（无效/负向分支不写 `perf_records.jsonl`，见各轮说明）
- 工具链戳: tilelang `0.1.2+90908d5694f131fa5538eeb3546d740c4b50bec4`（`/mnt/workspace/Agentkj/build`, USE_NPUIR=ON）/ CANN 9.0.0 / Ascend910B3 x1（20 AIC / 40 AIV）/ 2026-10-17

---

## 0. Phase 0 诊断（采数前强制项）

### 0.1 算子类型与首轮必查项

- **算子类型 = 纯 Vector**（无 `T.gemm`；K1/K2 各含行向/列向 `reduce_sum`；K3 为 grid=`KS+2` 的 combine）。与 DESIGN §5.1/§5.5 一致，非 MixCV，无 CV 同步。
- **向量化轴/布局重估（首轮必查项 ①）**：DESIGN §1.6.3 主选 A（核内布局 = I/O 行主序、向量化轴 = 尾轴 `d`、`seq_len` 走 block 级分块 + 因果 halo、行内水平 `reduce_sum(dim=1)`、token 轴 `reduce_sum(dim=0)`）。本轮实测**未发现需要 repack 的信号**：`memref.copy` 描述符全部为整行连续区间，`GM_to_UB/UB_to_GM` 每 block ≤ 64 KB（见各轮 profile），无跨步搬运热点；`aiv_vec_ratio` 在 OP1/OP2 后升到 0.40–0.58，残余瓶颈是**逐 op 发射/同步**而非轴质量。→ 保留主选 A（结论见 §4）。
- **陷阱条目版本戳核对（首轮必查项 ②）**：工具链相对设计期已变更（DESIGN 探针期 `2562795b` → 本次 `90908d56`）。本算子自身引用的 4 条 npuir 陷阱在任务开始均为 stale；本轮按同口径复验其**结论是否仍成立**：
  - `TRAP-npuir-parallel-runtime-minmax`（`T.Parallel` 内运行时 `min/max` 标量 store）→ **仍成立**：本次未改动掩码形态（unroll 标量语句），改动前后 L0 均通过；未做独立重验（沿用 DESIGN 探针 P6a/P6b 结论，标注"本工具链未重验"）。
  - `TRAP-npuir-parallel-scalar-store-race` → **仍成立**：K3 的 slot 版 combine **去掉**了 `(d,)` unroll fill（改为 `T.copy` 直写 `(1,d)` 区域），未引入 `T.Parallel` 标量 staging；`L0-9-determinism` 双跑 bit 一致通过。
  - `TRAP-npuir-vcast-shape-mismatch` → **仍成立**：OP1/OP2 的全部 `T.vcast` 均保持 src/dst 同 shape（`(R,d)→(R,d)`、`(blk_s,d)→(blk_s,d)`、`(1,d)→(1,d)`）。
  - `TRAP-npuir-per-copy-ub-staging-sum` / `TRAP-UB-multibuffer-inflation` → **仍成立**：K1/K2 新增 raw staging 后仍可编译（d=512/blk_s=8 亦可编译），说明 UB 未溢出；但 **OP6 的 tap 展开把 3 个 `T.copy` site 变成 12 个**（每 site 一份 staging），与"按 copy site 求和"一致地表现为负向（见 Round 4）。
  - `TRAP-npuir-2d-gm-cast-bufferize` → **仍然成立**：K1/K2 的 `conv_w` 依旧走 `T.copy` + `T.vcast`，未回退为 2-D 元素 `T.cast`。
- **DESIGN §1.6.3 实验裁决（首轮必做项）**：DESIGN 的裁决对象是「主选 A（3-kernel）vs 备选 B（全融合 2-kernel）」，判定阈值「B < A × 0.9 才翻转」。本轮**独立复核**：备选 B 需同一 block 内同时驻留 K1 与 K2 的 UB 常驻集（`≈118 KB + ≈80 KB > 192 KB`，§4.5 实测复用后仍超）且需重算右 halo 的 conv/silu（+40% 计算）；而**先做的 OP1/OP2/OP3 已把 K1/K2/K3 降到 3.6/6.4/1.7 µs 量级**，A 侧的"3 次 launch + `d_silu` 27 KB 往返"绝对值已 ≤ 1.5 µs——B 的结构性上限远小于阈值 0.9× 所需。→ **主选 A 保留**，B 不实现（判定依据 = 实测时长 + UB 预算，不依赖未实证常数；该裁决不构成设计层天花板）。

### 0.2 IR 级标量化定位（`VP-2026-0286` 操作条，采纳 conductor 预注入线索 1）

`TILELANG_DUMP_IR=1` 编译基线最小 shape（`M1,S24,d256,fp16`，避免命中编译缓存；日志 `logs/phase0/dump_ir_baseline_s24.log`），对 3 个 kernel 的 MLIR 分别 grep `scf.for` / `memref.load` / `tensor.extract` / `tensor.insert`。

**基线（改前）逐面统计**

| kernel | scf.for | memref.load | memref.copy | tensor.extract | `hivm.hir.v*` |
|---|---:|---:|---:|---:|---:|
| k1 | 8 | 3 | 2 | 3 | 25 |
| k2 | 27 | 11 | 8 | 15 | 105 |
| k3 | 4 | 0 | 3 | 3 | 9 |

**关键 IR 片段（k1，改前）**

```mlir
// vhat 因果窗载入 —— 嵌套标量循环（每 block R×d = 11×256 次标量 load）
%24 = scf.for %arg16 = %c0_i32 to %c11_i32 ... iter_args(%arg17 = %2) -> (tensor<11x256xf32>) {
  %47 = scf.for %arg18 = %c0_i32 to %c256_i32 ... iter_args(%arg19 = %arg17) -> (tensor<11x256xf32>) {
    %54 = memref.load %reinterpret_cast_5[%c0, %52, %53] : memref<1x24x256xf16, strided<[6144, 256, 1]>>
    %extf = arith.extf ... : f16 to f32
    %inserted = tensor.insert %55 into %arg19[%56, %53] : tensor<11x256xf32>
// dY own-rows 载入 —— 同形态嵌套标量循环（8×256）
%33 = scf.for %arg16 = %c0_i32 to %c8_i32 ... { %47 = scf.for %arg18 = ... %c256 ... { memref.load ...  } }

// conv tap 累加 —— **已向量化**（非标量）
%32 = scf.for %arg16 = %c0_i32 to %c4_i32 ... {
  %50 = hivm.hir.vbrc ins(%extracted_slice : tensor<1x256xf32>) outs(%47 : tensor<8x256xf32>)
  %51 = hivm.hir.vmul ins(%50, %extracted_slice_8 : tensor<8x256xf32>, ...)
  %52 = hivm.hir.vadd ins(%arg17, %51 : tensor<8x256xf32>, ...)
```

同形态在 k2 中出现在 5 处：`vhat_l / h_l / k_l / v_l` 各 `scf.for(8)×scf.for(256)`（每处 2048 次标量 load）+ `dY` 残差 `scf.for(8)×scf.for(256)`。

**结论（对 DESIGN §12.2 预注入线索 1 的独立验证 + 证伪/修正）**

- ⚠️ **证伪（对预注入线索 1 的本算子适用性）**：DESIGN §12.2 与 `VP-2026-0283` 指认的「`T.Parallel` 内元素级 loop-carried 累加」**在本算子并非标量化热点**。IR 实测：K1 的 `conv[i,j] += cw[p,j]*vnw[i+p,j]`（kernel L260–262）与 K2 的 `dvn[i,j] += cw[p,j]*dsw[..]`（L385–387）lower 为 **整 tile 的 `hivm.hir.vbrc`+`vmul`+`vadd`**（累加对象是 `tensor<blk_s,d>`），K1 的 `dconv_w` 部分和同理（整 tile）。→ 与同族正向 Stage 4 的独立结论一致（`CASE-engram-gate-conv-fwd-migration` / `perf_opt/opt_log.md` §0.2 第 53–54 行）：**同为 `+=` 写法，lowering 结果取决于累加粒度（元素 vs 整 tile）**；`VP-2026-0283` 的真实形态是逐元素 `tensor.extract`+`addf`+`insert`。
- ✅ **真正的标量化热点（三 kernel 共 7 处）**：**GM→UB 载入路径上带 `T.cast` 的元素级 `T.Parallel` 读取**（K1：vhat 窗口 `(R,d)` + dY own-rows；K2：4 个 own-rows 张量 + dY 残差）。机制：`T.Parallel` 内**元素级 `T.cast` 读 GM**（2-D GM region + 运行时 clamp 行索引）被 lower 为 `scf.for(R)×scf.for(d)` 的 `memref.load`+`extf`+`tensor.insert`，与 `TRAP-npuir-2d-gm-cast-bufferize` 同族（本条为其**性能面**）。
- ✅ **次要标量化点（k3）**：grid=1 combine 的 `for j in range(d_padded): stg[j] = acc0[0,j]` 被 lower 为 `scf.for(256)` + `tensor.extract`（每 output 一次，共 6 次 = 1536 条标量语句/次调用）。
- 量化核对：基线 K1 标量时间 = `0.62 × 223.9 ≈ 139 µs`，每 block 标量元素数 = `11×256 + 8×256 = 4864`；16 blocks → 77824 次标量 load，与实测同量级（`aiv_scalar_time` 139.9 µs，s128）。

### 0.3 初始现象（baseline profile，`phase=baseline`）

| workload | kernel | Task Duration(us) | aiv_time(us) | vec% | **scalar%** | mte2% | block_dim | GM→UB(KB) | UB→GM(KB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bwd-b1-s32-d256-fp16` | k1 | 50.90 | 50.30 | 2.9 | **98.6** | 0.6 | 4 | 2.5 | 12 |
| | k2 | 112.98 | 112.32 | 2.8 | **97.5** | 2.4 | 4 | 13 | 14 |
| | k3 | 26.20 | 25.64 | 1.4 | **85.9** | 9.6 | 1 | 24 | 6 |
| `bwd-b2-s64-d512-fp16` | k1 | 59.52 | 58.89 | 2.9 | **98.0** | 1.1 | 32 | 5 | 16 |
| | k2 | 140.92 | 140.31 | 2.5 | **98.5** | 1.9 | 32 | 18 | 16 |
| | k3 | 51.98 | 51.40 | 4.9 | **88.6** | 9.8 | 1 | 384 | 12 |
| `bwd-b1-s128-d256-bf16` | k1 | 225.56 | 224.89 | 28.9 | **62.2** | 0.1 | 16 | 2.5 | 12 |
| | k2 | 497.70 | 497.09 | 28.4 | **61.8** | 0.6 | 16 | 13 | 14 |
| | k3 | 25.98 | 25.44 | 3.6 | **90.3** | 8.1 | 1 | 96 | 6 |

- **P1（主导）**：`aiv_scalar_ratio` 0.62–0.99、`aiv_vec_ratio` 0.014–0.29 → **标量执行主导**（`VP-2026-0286` 的 `>50%` 强制触发换轴/定位分析）；`aiv_main_mem_read_bw` 极低（`GM→UB ≤ 24 KB/block`）→ **非带宽受限**，是纯发射/标量受限。
- **P2**：`s32`（4 blocks）与 `s128`（16 blocks）每 block 工作量相同（`R=11, blk_s=8, d=256`），但 `s128` 的 K1 是 `s32` 的 4.4×（225.6 vs 50.9）。单核工作量不变而 wall 随并发 block 数超线性增长 → 标量逐元素 GM 读**延迟受限**，并发升高时 L2 命中下降（与同族正向 Stage 4 的 P2 同型）。（迁移基线报告的 4 档数字 50.92/51.04/59.60/225.64 与本轮 K1 实测 50.90/59.52/225.56 逐一吻合，交叉验证采数口径一致。）
- **P3**：`s64-d512` 的 K3 = 51.98 µs（grid=1、`nchunks=2`、`GM→UB 384 KB`）→ grid-1 串行 + 逐 output unroll 标量填充是第二大热点。
- **P4**：`aiv_scalar_wait_time`（K2 s128）= 266.8 µs ≈ 0.54×wall → 标量侧在等向量结果，说明**依赖链/同步点密集**（op 数 ≈104/block），后续轮次以"减 op / 减同步点"为方向。

---

## 1. Performance Test Data

### 1.1 tune workload baseline（`phase=baseline`, `round=0`）

`kernel_id` 前缀统一为 `examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel.py::_engram_gate_conv_bwd_kernel::`（下表以 `…::kN` 简写，完整串见 `perf_records.jsonl` 与 §5 Final 表）。

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | profile_status | raw_profile_dir |
|---|---|---|---|---:|---|---|---|
| `…::k1` | `bwd-b1-s32-d256-fp16` | baseline | k1 | k1 | 50.90 | valid | `profiles/baseline/round0/baseline_8ba8496a/bwd-b1-s32-d256-fp16_8a09fbc4__k1/` |
| `…::k2` | `bwd-b1-s32-d256-fp16` | baseline | k2 | k2 | 112.98 | valid | `profiles/baseline/round0/baseline_8ba8496a/bwd-b1-s32-d256-fp16_8a09fbc4__k2/` |
| `…::k3` | `bwd-b1-s32-d256-fp16` | baseline | k3 | k3 | 26.20 | valid | `profiles/baseline/round0/baseline_8ba8496a/bwd-b1-s32-d256-fp16_8a09fbc4__k3/` |
| `…::k1` | `bwd-b2-s64-d512-fp16` | baseline | k1 | k1 | 59.52 | valid | `profiles/baseline/round0/baseline_8ba8496a/bwd-b2-s64-d512-fp16_dbdbe5d7__k1/` |
| `…::k2` | `bwd-b2-s64-d512-fp16` | baseline | k2 | k2 | 140.92 | valid | `profiles/baseline/round0/baseline_8ba8496a/bwd-b2-s64-d512-fp16_dbdbe5d7__k2/` |
| `…::k3` | `bwd-b2-s64-d512-fp16` | baseline | k3 | k3 | 51.98 | valid | `profiles/baseline/round0/baseline_8ba8496a/bwd-b2-s64-d512-fp16_dbdbe5d7__k3/` |
| `…::k1` | `bwd-b1-s128-d256-bf16` | baseline | k1 | k1 | 225.56 | valid | `profiles/baseline/round0/baseline_8ba8496a/bwd-b1-s128-d256-bf16_3eec799d__k1/` |
| `…::k2` | `bwd-b1-s128-d256-bf16` | baseline | k2 | k2 | 497.70 | valid | `profiles/baseline/round0/baseline_8ba8496a/bwd-b1-s128-d256-bf16_3eec799d__k2/` |
| `…::k3` | `bwd-b1-s128-d256-bf16` | baseline | k3 | k3 | 25.98 | valid | `profiles/baseline/round0/baseline_8ba8496a/bwd-b1-s128-d256-bf16_3eec799d__k3/` |

命令模板（每个 `(kernel, workload)` 一次独立采集，全局串行）：

```bash
msprof op --kernel-name={kN} --output=perf_opt/profiles/{phase}/round{N}/{candidate_id}/{workload_id}__{kN} \
  --launch-count=20 --warm-up=5 --dump=off \
  --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio \
  python perf_opt/bench_runner.py --impl {kernel_file} --case {workload_id}
```

有效 profile 判据（全部 102 次 `msprof op` 采集均满足）：`OpBasicInfo.csv` 的 `Op Name` == 本次 `--kernel-name`；`Op Type=vector`；`Task Duration(us)` 为正、非 Cast/Mul/ZerosLike 等框架小算子（首次无 filter 的 nameprobe 捕到 `ZerosLike_...` 框架算子，已据此固定 `--kernel-name` 并重采）。输出目录与父目录均 `chmod 700`。

### 1.2 smoke / skipped 排除

| workload_id | kind | 原因 | 精度 |
|---|---|---|---|
| `bwd-smoke-b1-s16-d256-fp16` | smoke | benchmark `_engram_gate_conv_bwd_params()` 的 `i=0` → `pytest.mark.smoke`（label 亦含独立 `smoke` 词）；不做性能 baseline、不选 winner | **pass**（`L0-1-smoke`（1,16,256,fp16）maxdiff 7.629e-06；最终 `--level all` 复跑 PASS） |

无 `skipped` 项（benchmark 4 个参数化案例全部可运行：1 smoke + 3 full）。

> 说明：该算子无 `full`×`smoke` 标记冲突；3 个 `tune` workload 与 inventory 的 `kind=tune` 一一对应，全部有 baseline 与 final 记录。

### 1.3 设计估算 vs 实测偏差行（D-2 回填）

DESIGN §1.6.0 设计期估算（代表 workload W3 = `M2,S64,d512,fp16`）：下界 ≈ `max(流量项 2.5 µs, 发射项 51 µs, 0) ≈ **51 µs**`，**发射项主导**（每 block 向量 op 数 ≈103 × `0.5 µs/op`）。

实测：W3 baseline 三 kernel 合计 = `59.52 + 140.92 + 51.98 = **252.42 µs**` → 偏差 **4.9×（低估）**。失准项 = **发射项口径**：`CONST-vector-launch-overhead`（0.5 µs/op）已知在 `VP-2026-0285` 中被证伪（仅适用逐 op 串行小向量链），本算子真实成本来自**元素级 GM 载入的标量 `memref.load` 延迟**（`aiv_scalar_time` 主导），非 op 数 × 常数。

补充：改后的同一 workload 实测 **18.42 µs**，已**低于**设计估算下界 51 µs —— 即该常数在 910B3 上系统性**高估**发射成本。该偏差行是本轮 `[DESIGN_LIMIT]` 归因的量化证据（最终**未**触发反向反馈：问题由参数/结构级可解，且矛盾项属已证伪常数的数据修正，非设计层天花板，见 §5 判定）。

---

## 2. Iteration Log

> 每轮固定 `base = 当前 current best`；每个实验分支只改一个主要优化点；每个 valid 分支先跑 L0 精度回归、再用 `msprof op` 采集；`<5%` 或表观相反的差异一律走 `perf_opt/ab_interleaved.py` 交错 A/B/A/B 协议。分支文件由 `perf_opt/probe/make_variant.py`（baseline + 命名补丁链）确定性生成，故每个分支的 sha256 可复算。

### Round 1 — OP1/OP2：向量化 GM 载入（同一机制的 K1 / K2 分支）⭐ 决定性

**现象**：P1（scalar 0.62–0.99）+ §0.2 IR 证据（7 处嵌套标量 `scf.for` 载入）。

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP1 | K1 标量 0.62（s128 139.9 µs） | K1 的 GM 载入改「raw dtype 整块 `T.copy` + 整 tile `T.vcast`」 | §0.2 IR `memref.load` 嵌套循环 | vhat 窗拆 `[halo(3) \| own(blk_s)]` 两段连续拷贝（own 精确 region `t0:t0+blk_s`、halo 用 clamp base `hb`），dY own-rows 精确 region，权重 1-D raw staging | `memref.load`→1、`aiv_scalar_ratio`、Task Duration | improved |
| OP2 | K2 标量 0.62（s128 307.3 µs） | K2 的 5 处 GM 载入同法（4 own-rows 共用 1 个 raw staging + dsw own/halo 两段 + dY） | 同上 | 同左（`raw` 单缓冲顺序复用，仿同族正向 `hr` 复用） | 同上 | improved |

**「候选 vs current best」对比表（Round 1）**

| branch | base(current best) | opt_id | workload | kernel | Task Duration(us) | aiv_time(us) | vec% | scalar% | mte2% | L0 | vs current best |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| baseline | — | — | s32 | k1 | 50.90 | 50.30 | 2.9 | 98.6 | 0.6 | pass | — |
| baseline | — | — | s64 | k1 | 59.52 | 58.89 | 2.9 | 98.0 | 1.1 | pass | — |
| baseline | — | — | s128 | k1 | 225.56 | 224.89 | 28.9 | 62.2 | 0.1 | pass | — |
| `r1_op1_k1vec` | baseline | OP1 | s32 | k1 | **3.72** | 3.48 | 53.9 | 49.7 | 28.5 | pass | **−92.7%** |
| `r1_op1_k1vec` | baseline | OP1 | s64 | k1 | **5.34** | 5.07 | 51.9 | 53.7 | 43.2 | pass | **−91.0%** |
| `r1_op1_k1vec` | baseline | OP1 | s128 | k1 | **4.90** | 4.19 | 51.3 | 62.2 | 34.3 | pass | **−97.8%** |
| `r1_op2_k2vec` | baseline | OP2 | s32 | k2 | **6.76** | 6.13 | 53.3 | 47.3 | 24.9 | pass | **−94.0%** |
| `r1_op2_k2vec` | baseline | OP2 | s64 | k2 | **8.72** | 8.13 | 58.1 | 58.4 | 29.7 | pass | **−93.8%** |
| `r1_op2_k2vec` | baseline | OP2 | s128 | k2 | **7.52** | 5.63 | 55.5 | 52.1 | 23.4 | pass | **−98.5%** |
| `r1_op12_merged` | baseline | OP1+OP2 | s32 | k1/k2/k3 | 3.60 / 6.36 / 25.14 | — | 53.7/56.3/1.5 | 49.1/43.7/89.4 | 27.3/24.6/6.9 | pass | **新 current best** |
| `r1_op12_merged` | baseline | OP1+OP2 | s64 | k1/k2/k3 | 5.88 / 8.84 / 52.08 | — | 48.6/54.1/4.9 | 57.9/60.9/88.6 | 45.1/38.8/9.9 | pass | 新 current best |
| `r1_op12_merged` | baseline | OP1+OP2 | s128 | k1/k2/k3 | 4.74 / 8.16 / 26.30 | — | 51.9/55.4/3.5 | 59.9/53.8/89.3 | 29.1/27.1/9.1 | pass | 新 current best |

**判断**：三 workload 全部远超 3% 噪声阈值 → 直接采纳，无需 ab 协议。合并版 = `r1_op12_merged`（`round1_op12_merged.py`）。逐 workload 合计：`s32 190.08 → 35.10 µs`、`s64 252.42 → 66.80 µs`、`s128 749.24 → 39.20 µs`。
**归因**：K1/K2 的 `memref.load` 由 14/11 降至 1/4（残余为掩码/own-row 标量语句，规模 O(R+blk_s)），K1 sigmoid 前的 `conv`/`dconv` 段未动（IR 已证明是整 tile 向量形态）。
**IR 复检（改后）**：k1 `scf.for` 8→4、`memref.load` 3→1、`memref.copy` 2→5；k2 `scf.for` 27→6、`memref.load` 11→4、`memref.copy` 8→10——嵌套 `scf.for(R)×scf.for(d)` 全部消失。
**副产物**：`bwd-b1-s128-d256-bf16/k2` 由 `497.70 µs` 降到 `8.16 µs`（61.0×），单此一项即解释了关注 shape 的大部分收益。

### Round 2 — OP3：K3 combine 的 slot 并行化（+ 统一 workspace）

**现象**（Round 1 后重新分析）：K3 成为**最大项**（25.1 / 52.1 / 26.3 µs，占三 kernel 合计的 39–78%）；profile = `grid=1`、`scalar 0.86–0.89`、`mte2 0.07–0.10`；IR = 6 组 `[T.clear + 串行 chunk `T.copy` + `reduce_sum` + `vadd` + **`scf.for(d)` unroll 标量填充** + `T.copy`]`（`d`=256/512 条标量语句 × 6）。

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP3 | grid=1 串行 6 次归约 + 6 次 `(d,)` unroll 标量填充 | K3 改 `grid = KS + 2`「一 block 一输出槽」的统一归约；K1/K2 部分和写入**单一 workspace** 的 slot；combine 输出写 `(KS+2, d)` 缓冲、由闭包切片成 3 个契约输出 | §0.2 k3 IR 的 `scf.for(256)+tensor.extract`；grid=1 只用 1/40 核 | `ws` 形状 `((KS+2)*nb_pad, d)`（slot 0=`drms_w_h`、1=`drms_w_v`、`2+p`=tap p）；K3 `base = cid*nb_pad` + `T.serial(nchunks)` 累加 + `T.copy(acc0, out_all[cid:cid+1, 0:d])`；闭包 `out_all[0]/out_all[1]/out_all[2:2+KS]` 作视图返回 | Task Duration / `aiv_scalar_ratio` / block_dim | improved |

**「候选 vs current best」对比表（Round 2）**

| branch | base(current best) | opt_id | workload | kernel | Task Duration(us) | aiv_time(us) | vec% | scalar% | mte2% | L0 | vs current best |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| `r1_op12_merged` | baseline | — | s32 | k3 | 25.14 | 24.63 | 1.5 | 89.4 | 6.9 | pass | current best |
| `r1_op12_merged` | baseline | — | s64 | k3 | 52.08 | 51.49 | 4.9 | 88.6 | 9.9 | pass | current best |
| `r1_op12_merged` | baseline | — | s128 | k3 | 26.30 | 25.73 | 3.5 | 89.3 | 9.1 | pass | current best |
| `r2_op3_k3slots` | `r1_op12_merged` | OP3 | s32 | k3 | **1.66** | 1.07 | 9.6 | 43.0 | 39.7 | pass | **−93.4%** |
| `r2_op3_k3slots` | `r1_op12_merged` | OP3 | s64 | k3 | **2.52** | 1.90 | 30.7 | 50.0 | 60.9 | pass | **−95.2%** |
| `r2_op3_k3slots` | `r1_op12_merged` | OP3 | s128 | k3 | **1.82** | 1.14 | 16.9 | 49.1 | 38.4 | pass | **−93.1%** |
| `r2_op3_k3slots` | `r1_op12_merged` | OP3 | s32/s64/s128 | k1 | 4.22 / 5.94 / 4.42 | — | 49.7/53.8/52.2 | 58.2/57.8/59.4 | 36.3/42.2/30.2 | pass | +15%/+1%/−7%（见下） |
| `r2_op3_k3slots` | `r1_op12_merged` | OP3 | s32/s64/s128 | k2 | 6.42 / 9.48 / 7.88 | — | 56.4/48.2/54.9 | 43.5/63.0/52.8 | 27.9/41.8/25.8 | pass | +1%/+7%/−3%（见下） |
| `r2_op3_k3slots` | `r1_op12_merged` | OP3 | — | 合计 | **12.30 / 17.94 / 14.12** | — | — | — | — | pass | **−65%/−73%/−64%** |

**判断**：
- K3 三档 −93% 起，远超阈值 → 直接采纳。
- K1/K2 的 ±1–15% 波动**不来自 OP3 的负载**（K1/K2 的算法未改，仅 `ws_conv`→`ws` 的地址表达式与 K3 输出切片变了），属**同候选内多 kernel 采集的 run-to-run 抖动 + 峰值 block 指标口径**（`aiv_*` 取 per-block max）。为免把抖动当回退，本轮以 **合并版三 kernel 全量复测**（9 对）为准：合计 35.10→12.30 / 66.80→17.94 / 39.20→14.12，全部大额改善。`r2_op3_k3slots` 收入 current best。
- 顺带收益：`ws` 由 3 个 `torch.zeros` 合为 1 个，闭包主机侧分配/清零少 2 次（不进 msprof 口径，仅为真实收益备注）。

### Round 3 — OP4/OP5：seq-tile 宽度（`BLK_S`）扫描 + 按 shape 分派

**现象**（Round 2 后）：K3 已消除，**K1/K2 恢复为全部耗时**（合计 10.6–15.4 µs，占 86–91%）；三档 block_dim 仅 4/6/8/16/32，远低于 40 AIV；`aiv_time ≈ dur`、`vec ≈ 0.40–0.58`、`scalar ≈ 0.40–0.64` → 单核 wall = 固定 per-block 开销 + 与 tile 宽成正比的工作量，**wall 对 block 数不敏感（8/16/32 blocks 同量级）** ⇒ tile 越窄，单核工作量越小（但 halo 冗余 3/(blk+3) 升高）。

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP4 | s32 仅 4 blocks、单核 wall 高 | d≤256 的 `BLK_S` 8→4（blocks 翻倍） | DESIGN §5.2 明示 `BLK_S` 为 Stage-4 杠杆 | `BLK_4_4`（d256=4, d512=4） | Task Duration（**交错 A/B**） | **分 shape 有效** |
| OP5 | s64-d512 单核 wall 最高（k2 9.2 µs） | d=512 的 `BLK_S` 4→8（halo 冗余 43%→27%） | 同上 | `BLK_4_8`（d512=8） | Task Duration（**交错 A/B**） | **rejected** |
| — | — | 按 shape 分派 `BLK_S`（合入 OP4 的有效域） | 上述 A/B 的相反结论 | `TUNED_BLK_CAP_D256/D512` + `TUNED_TARGET_BLOCKS=8`：capacity tile 产生 <8 blocks 时逐级减半（保持 `blk_s ≥ HALO` 且 `seq_len % blk_s == 0`） | 全量 9 对复测 | **merged** |

**「候选 vs current best」对比表（Round 3）**

| branch | base(current best) | opt_id | workload | kernel | Task Duration(us) | vec% | scalar% | mte2% | L0 | 交错 A/B 裁决 | vs current best |
|---|---|---|---|---|---:|---:|---:|---:|---|---|---|
| `r2_op3_k3slots` | baseline | — | s32/s64/s128 | k1/k2/k3 | 4.22/6.42/1.66｜5.94/9.48/2.52｜4.42/7.88/1.82 | — | — | — | pass | — | current best |
| `r3_op4_blk4d256` | `r2_op3_k3slots` | OP4 | s32 | k1/k2/k3 | 3.74 / 5.50 / 1.64 | 39.5/46.0/13.4 | 41.5/51.0/48.9 | 34.9/32.1/44.1 | pass | **s32/k2：B −17.72%，3/3 → adopt_b** | **−11.5%（合计）** |
| `r3_op4_blk4d256` | `r2_op3_k3slots` | OP4 | s128 | k1/k2/k3 | 4.92 / 7.56 / 2.12 | 41.7/38.3/22.9 | 60.8/65.0/44.7 | 45.4/25.9/53.0 | pass | **s128/k1：B +15.22%，0/3 → adopt_a** | **+3.4%（合计）** |
| `r3_op5_blk8d512` | `r2_op3_k3slots` | OP5 | s64 | k2 | 9.76（A/B median） | — | — | — | pass | **s64/k2：B +8.69%，0/3 → adopt_a** | rejected |
| `r3_op5_blk8d512` | `r2_op3_k3slots` | OP5 | s64 | k1 | 5.68（A/B median） | — | — | — | pass | s64/k1：+4.70%，2/3 → inconclusive | rejected（净负） |
| `r3_op4_dispatch` | `r2_op3_k3slots` | 分派 | s32 | k1/k2/k3 | **3.68 / 5.46 / 1.68** | 41.8/46.1/13.3 | 43.5/51.0/50.4 | 32.1/34.7/45.8 | pass | — | **新 current best（合计 10.82，−12.0%）** |
| `r3_op4_dispatch` | `r2_op3_k3slots` | 分派 | s64 | k1/k2/k3 | 6.00 / 9.18 / 2.52 | 50.5/51.2/31.0 | 58.7/61.2/48.1 | 47.2/37.7/59.6 | pass | — | 17.70（−1.3%） |
| `r3_op4_dispatch` | `r2_op3_k3slots` | 分派 | s128 | k1/k2/k3 | 4.52 / 7.68 / 1.76 | 50.9/54.0/17.2 | 60.4/52.8/51.9 | 30.5/25.3/43.2 | pass | — | 13.96（−1.1%） |

**交错 A/B 裁决摘要**（`perf_opt/profiles/ab/`，3 pairs、交替次序、按配对中位 + 配对胜数判据）

| 决策 | A | B | case/kernel | A median | B median | B−A | paired B wins | verdict | 采纳 |
|---|---|---|---:|---:|---:|---:|---|---|
| OP4 是否全域生效 | `r2_op3_k3slots` | `r3_op4_blk4d256` | s32 / k2 | 6.66 | 5.48 | **−17.72%** | 3/3 | adopt_b | d256 在 s32 用 `BLK_S=4` |
| OP4 在 s128 是否回退 | `r2_op3_k3slots` | `r3_op4_blk4d256` | s128 / k1 | 4.60 | 5.30 | **+15.22%（B 更差）** | 0/3 | adopt_a | s128 保留 `BLK_S=8` |
| OP5 是否有效 | `r2_op3_k3slots` | `r3_op5_blk8d512` | s64 / k2 | 8.98 | 9.76 | **+8.69%（B 更差）** | 0/3 | adopt_a | d512 保留 `BLK_S=4` |
| OP5 附带 | `r2_op3_k3slots` | `r3_op5_blk8d512` | s64 / k1 | 5.96 | 5.68 | −4.70% | 2/3 | inconclusive | 不足以翻转 |

**判断**：OP4 在 `s32` 有效（A/B 3/3）、在 `s128` 反向（A/B 0/3）→ **按 shape 分派**是正确形态（DESIGN §5.2/§1.6.3 已允许"按 workload 分派而非全局二选一"，且分派发生在**编译期常量**层：`blk_s` 由 `(M, seq_len, d)` 决定，工厂 `lru_cache` 逐 shape 特化，**不是**运行时分派）。`TUNED_TARGET_BLOCKS` 取 8：`s32`（cap 8 → 4 blocks）触发减半、`s128`（cap 8 → 16 blocks）不触发；取 16 对全部 manifest shape 结论相同（故该常数不敏感）。OP5 全域否决。

**副作用**：`TUNED_BLK_CAP_D256/D512`、`TUNED_TARGET_BLOCKS` 以模块级常量暴露（wrapper 的 baseline/perf_opt 切换块可成对引用调优默认值）。

### Round 4 — OP6：tap 循环显式展开（负向，已回滚）

**现象**（Round 3 后）：`scalar ≈ 0.44–0.64` 与 `vec ≈ 0.42–0.55` 并存、`aiv_scalar_wait_time` 高 → 疑为「运行时 `scf.for` + 动态 UB 子视图偏移」（`TRAP-C13`）导致的地址计算/依赖开销；同族正向 Stage 4 的 Round 8 曾以同类展开取得 4.7–5.3% 收益。

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP6 | K1 `conv`/`dconv` 与 K2 转置的 `for p in range(KS)` 是运行时 `scf.for` + `%49` 动态行偏移，且 `T.clear` 独立成 pass | 3 个 tap 循环显式展开为 4 段静态偏移语句；首个 tap 用 `=` 折掉 `T.clear`；tap 权重经 `T.vbrc(cw[p:p+1], bro_s)` 广播 | §0.2 IR（`tensor.extract_slice %30[%49, 0]`，动态偏移）；`TRAP-C13` | 展开 12 段（K1 conv 4 + K1 dconv 4 + K2 转置 4） | Task Duration（**交错 A/B**） | **rejected** |

**「候选 vs current best」对比表（Round 4）**

| branch | base(current best) | opt_id | workload | kernel | Task Duration(us) | vec% | scalar% | mte2% | L0 | 交错 A/B 裁决 | vs current best |
|---|---|---|---|---|---:|---:|---:|---:|---|---|---|
| `r3_op4_dispatch` | — | — | s32/s64/s128 | k1 | 3.68 / 6.00 / 4.52 | 41.8/50.5/50.9 | 43.5/58.7/60.4 | 32.1/47.2/30.5 | pass | — | current best |
| `r3_op4_dispatch` | — | — | s32/s64/s128 | k2 | 5.46 / 9.18 / 7.68 | 46.1/51.2/54.0 | 51.0/61.2/52.8 | 34.7/37.7/25.3 | pass | — | current best |
| `r4_op6_tapunroll` | `r3_op4_dispatch` | OP6 | s32 | k1 | 3.50 | 36.8 | 43.1 | 31.7 | pass | s32/k1：−1.11%，2/3 → **tie** | 未采纳 |
| `r4_op6_tapunroll` | `r3_op4_dispatch` | OP6 | s128 | k1 | 5.02 | 47.7 | 59.0 | 31.2 | pass | **s128/k1：+8.14%（B 更差），0/3 → adopt_a** | **rejected** |
| `r4_op6_tapunroll` | `r3_op4_dispatch` | OP6 | s32 | k2 | 5.36 | 46.4 | 51.5 | 38.1 | pass | — | 未采纳 |
| `r4_op6_tapunroll` | `r3_op4_dispatch` | OP6 | s64 | k1/k2 | 6.02 / 9.14 | 46.2/58.0 | 57.7/60.5 | 52.5/29.8 | pass | — | −0.3%/−0.4%（噪声内） |
| `r4_op6_tapunroll` | `r3_op4_dispatch` | OP6 | s128 | k2 | 7.66 | 54.2 | 52.0 | 26.1 | pass | — | −0.3%（噪声内） |

**判断**：**回滚 OP6**。`s128/k1` 的 −8.14% 经 0/3 的 A/B 确认为真实回退，`s32/k1` 为 tie，其余在噪声内。
**机制归因（回滚理由，非"方向无效"断言）**：展开把 `T.copy` 的 **copy site 数**从 3 增到 12（K1 的 dconv 部分和 4 个 `T.copy(red0, ws…)` 由运行时循环内的 1 个 site 变为 4 个），按 `TRAP-npuir-per-copy-ub-staging-sum`「staging 按 copy site 逐个分配、按总和计」UB 常驻集上升；同时 `T.clear` 的消除收益（同族正向 OP8 的 4.7–5.3%）在本算子被**每个 tap 新增的 `T.vbrc` 广播**抵消。→ 记录为「本 workload / 本工具链 / 本结构下的实测负向」，不外推为"展开无用"。

### Round 1–4 汇总「候选 vs current best」对比表（B2 结构化回流）

> 数据来源 `perfopt/perf_records.jsonl` + raw profile。时延列为单次 run `msprof op` median（A/B 行除外，标 `ab`）；`vec/scalar/mte2` 为 per-block max（关键路径口径）。`合计` = 同 workload 三 kernel 之和。

| round | candidate_id | workload | kernel | Task Duration(us) | vec% | scalar% | mte2% | L0 | vs 上一 current best | 状态 |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| 0 | baseline | s32 | k1/k2/k3 | 50.90 / 112.98 / 26.20（合计 190.08） | 2.9/2.8/1.4 | 98.6/97.5/85.9 | 0.6/2.4/9.6 | pass | — | baseline |
| 0 | baseline | s64 | k1/k2/k3 | 59.52 / 140.92 / 51.98（合计 252.42） | 2.9/2.5/4.9 | 98.0/98.5/88.6 | 1.1/1.9/9.8 | pass | — | baseline |
| 0 | baseline | s128 | k1/k2/k3 | 225.56 / 497.70 / 25.98（合计 749.24） | 28.9/28.4/3.6 | 62.2/61.8/90.3 | 0.1/0.6/8.1 | pass | — | baseline |
| 1 | `r1_op1_k1vec` | s32/s64/s128 | k1 | 3.72 / 5.34 / 4.90 | 53.9/51.9/51.3 | 49.7/53.7/62.2 | 28.5/43.2/34.3 | pass | **−92.7%/−91.0%/−97.8%** | improved |
| 1 | `r1_op2_k2vec` | s32/s64/s128 | k2 | 6.76 / 8.72 / 7.52 | 53.3/58.1/55.5 | 47.3/58.4/52.1 | 24.9/29.7/23.4 | pass | **−94.0%/−93.8%/−98.5%** | improved |
| 1 | `r1_op12_merged` | s32/s64/s128 | 合计 | 35.10 / 66.80 / 39.20 | — | — | — | pass | **−81.5%/−73.5%/−94.8%** | current best |
| 2 | `r2_op3_k3slots` | s32/s64/s128 | k3 | 1.66 / 2.52 / 1.82 | 9.6/30.7/16.9 | 43.0/50.0/49.1 | 39.7/60.9/38.4 | pass | **−93.4%/−95.2%/−93.1%（对 k3）** | improved |
| 2 | `r2_op3_k3slots` | s32/s64/s128 | 合计 | 12.30 / 17.94 / 14.12 | — | — | — | pass | **−65.0%/−73.1%/−64.0%** | current best |
| 3 | `r3_op4_blk4d256` | s32 | 合计 | 10.88 | 39.5/46.0/13.4 | 41.5/51.0/48.9 | 34.9/32.1/44.1 | pass | −11.5%（A/B 3/3 支持） | 有效（s32 域） |
| 3 | `r3_op4_blk4d256` | s128 | 合计 | 14.60 | — | — | — | pass | **+3.4%（A/B 0/3 反对）** | 回退（s128 域） |
| 3 | `r3_op5_blk8d512` | s64 | k2 / k1 | 9.76ᵃᵇ / 5.68ᵃᵇ | — | — | — | pass | +8.7% / −4.7%（inconclusive） | rejected |
| 3 | `r3_op4_dispatch` | s32/s64/s128 | 合计 | **10.82 / 17.70 / 13.96** | — | — | — | pass | **−12.0%/−1.3%/−1.1%** | **current best** |
| 4 | `r4_op6_tapunroll` | s32 | k1/k2 | 3.50 / 5.36 | 36.8/46.4 | 43.1/51.5 | 31.7/38.1 | pass | −1.1% / −1.8%（A/B tie） | 未采纳 |
| 4 | `r4_op6_tapunroll` | s64 | k1/k2 | 6.02 / 9.14 | 46.2/58.0 | 57.7/60.5 | 52.5/29.8 | pass | +0.3% / −0.4%（噪声内） | 未采纳 |
| 4 | `r4_op6_tapunroll` | s128 | k1/k2 | 5.02 / 7.66 | 47.7/54.2 | 59.0/52.0 | 31.2/26.1 | pass | **+8.1%（A/B 0/3）** / −0.3% | **rejected** |

**当前 kernel 全部 tune workload 合并检查**（每个 winner 都是整体 kernel 的合并版；每次合并后都按同一 winner 复测全部 3 个 tune workload 的三 kernel 与 smoke 精度）：

`r1_op12_merged`（9 对全测）→ `r2_op3_k3slots`（9 对全测）→ `r3_op4_dispatch`（9 对全测）各步均覆盖全部 tune workload，且每步 `--l0` 通过、smoke 精度见 §1.2；**无任何 workload 出现可信回退**（Round 3 中 `s128` 的单点 +11% 已用 A/B 归因到 `blk_s` 域并以分派修复；Round 4 的负向分支未合并）。合并版本的选取与 §5 的 Final 表同源（`final_r3_op4_dispatch` 与 `r3_op4_dispatch` 为同一文件内容 + 文档字符串更新）。

> blocked/invalid 分支说明：本轮无编译失败/精度失败分支；**未写 `perf_records.jsonl` 的分支**仅 `r3_op5_blk8d512` 的 A/B 采集（其 2 行已作为 `phase=candidate` 记录写入，见 §5 交付物索引）。`r3_op4_blk4d256` / `r4_op6_tapunroll` 各自 6 行候选记录已写入。

---

## 3. Autotune Log

未使用 autotune API（本算子无 autotune 注入点；`BLK_S` 与 `TUNED_TARGET_BLOCKS` 是小整数网格，直接手动粗扫 + 交错 A/B 精判更可靠）。

- **搜索空间**：`d ≤ 256`：`BLK_S ∈ {8, 4}`；`d = 512`：`BLK_S ∈ {4, 8}`；`TUNED_TARGET_BLOCKS ∈ {8, 16}`（对全部 manifest shape 决策等价，故不单独扫）。
- **约束**：`blk_s ≥ HALO(=3)`（halo 精确拷贝的对齐前提）、`seq_len % blk_s == 0`（精确 region 前提）、UB 预算（`d=512, blk_s=8` 编译通过）。
- **best**：`blk_cap = 8 (d≤256) / 4 (d=512)`，并按 `TUNED_TARGET_BLOCKS=8` 在 blocks 不足时减半 → 实测落点 `s16:4 / s32:4 / s64-d512:4 / s128:8`。
- **winner 邻域**：`blk_s = 2` 被 `blk_s ≥ HALO` 排除（且 `T.vbrc` 拒空 broadcast 维，同族正向实测编译失败）；`blk_s = 16 (d256)` 因 UB 与 halo 冗余双重压力未提交（`R=19`，K2 `(16,d)` 常驻 buffer ×12 ≈ 196 KB raw，超预算）。
- **最终 winner 复测**：见 §5 Final 表（同一最终候选、`msprof op` 独立采集）与 Round 3 的三组交错 A/B。

---

## 4. 布局与向量化轴复核结论（Phase 0 首轮必查项收口）

- 主选 A（核内布局 = I/O 行主序、向量化轴 = 尾轴 `d`、`seq_len` block 分块 + 因果 halo、`reduce_sum(dim=1)` 行归约 + `reduce_sum(dim=0)` token 归约）经实测确认；**未发现需要核内重排/转置的收益**：`memref.copy` 描述符全为整行连续、`GM→UB ≤ 64 KB/block`（不构成带宽压力），而 `vec` 与 `scalar` 两条 pipe 的忙时都在 1.5–3.3 µs 量级、`aiv_time ≈ dur`，说明瓶颈是**逐 op 发射/同步开销**而非轴质量（换轴不会减少 op 数）。
- 残余画像（final）：`vec 0.36–0.55 / scalar 0.40–0.64 / mte2 0.30–0.65`，`Task Duration(3.6–9.5 µs) ≈ aiv_time`（差 ≤ 0.4 µs，launch 开销被核内工作掩盖）→ **发射 + 小规模 MTE2/vector 混合受限**，非带宽受限。
- 残余标量来源（final IR，全部为 O(R)+O(blk_s) 量级的小规模串行）：K1 掩码 11 + own-mask 8；K2 掩码 11 + own-rows 标量 staging 8（4 次 `(blk_s,1)` 标量读）；K3 chunk 循环控制。这些形态是 ABI 决定的**必要性**约束（`TRAP-npuir-parallel-scalar-store-race` / `TRAP-npuir-parallel-runtime-minmax` 曾重验为成立），不是可继续压缩的冗余。

---

## 5. Final Summary

**最终候选**：`perf_opt/_engram_gate_conv_bwd_kernel.py`（sha256 `6a702c606f9a4343b727af94cf93bea163bd2e84e68be34f050304cbb256d373`）＝ `r3_op4_dispatch`（+ 文档字符串更新）。采纳的优化点：**OP1**（K1 张量化载入）+ **OP2**（K2 张量化载入）+ **OP3**（slot 并行 combine + 统一 workspace）+ **按 shape 的 `BLK_S` 分派**（`TUNED_BLK_CAP_D256=8` / `TUNED_BLK_CAP_D512=4` / `TUNED_TARGET_BLOCKS=8`）。

**精度**：`_engram_gate_conv_bwd_kernel.py --level all` → **L0 PASS / L1 PASS / L2 无告警 / Boundary 仅 `B-S1`（M1,S1,d256）warn**；该 warn 为**既有**问题（已在冻结基线 `../_engram_gate_conv_bwd_kernel.py` 上同口径复现同一 `TVMError`，且 Boundary 为非阻塞级）——原因为 `seq_len=1` 时左 halo clamp base 与单 tile 掩码形态触发 codegen 异常，与本次改动无关。smoke `bwd-smoke-b1-s16-d256-fp16` 精度 **pass**（maxdiff 7.629e-06）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel.py::_engram_gate_conv_bwd_kernel::k1 | bwd-b1-s32-d256-fp16 | 50.90 | 3.58 | final_r3_op4_dispatch |
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel.py::_engram_gate_conv_bwd_kernel::k2 | bwd-b1-s32-d256-fp16 | 112.98 | 5.58 | final_r3_op4_dispatch |
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel.py::_engram_gate_conv_bwd_kernel::k3 | bwd-b1-s32-d256-fp16 | 26.20 | 1.56 | final_r3_op4_dispatch |
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel.py::_engram_gate_conv_bwd_kernel::k1 | bwd-b2-s64-d512-fp16 | 59.52 | 6.34 | final_r3_op4_dispatch |
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel.py::_engram_gate_conv_bwd_kernel::k2 | bwd-b2-s64-d512-fp16 | 140.92 | 9.50 | final_r3_op4_dispatch |
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel.py::_engram_gate_conv_bwd_kernel::k3 | bwd-b2-s64-d512-fp16 | 51.98 | 2.58 | final_r3_op4_dispatch |
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel.py::_engram_gate_conv_bwd_kernel::k1 | bwd-b1-s128-d256-bf16 | 225.56 | 4.50 | final_r3_op4_dispatch |
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel.py::_engram_gate_conv_bwd_kernel::k2 | bwd-b1-s128-d256-bf16 | 497.70 | 7.54 | final_r3_op4_dispatch |
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel.py::_engram_gate_conv_bwd_kernel::k3 | bwd-b1-s128-d256-bf16 | 25.98 | 1.68 | final_r3_op4_dispatch |

**逐 kernel 提升（同一最终候选，`phase=final`，`msprof op Task Duration`）**

| workload | k1 | k2 | k3 | 三 kernel 合计 | 合计提升 |
|---|---|---|---|---|---|
| `bwd-b1-s32-d256-fp16` | −92.97%（14.2×） | −95.06%（20.2×） | −94.05%（16.8×） | 190.08 → **10.72 µs** | **−94.36%（17.7×）** |
| `bwd-b2-s64-d512-fp16` | −89.35%（9.4×） | −93.26%（14.8×） | −95.04%（20.1×） | 252.42 → **18.42 µs** | **−92.70%（13.7×）** |
| `bwd-b1-s128-d256-bf16`（关注 shape） | −98.00%（50.1×） | −98.49%（66.0×） | −93.53%（15.5×） | 749.24 → **13.72 µs** | **−98.17%（54.6×）** |

- **smoke**：`bwd-smoke-b1-s16-d256-fp16` 仅精度（pass），按契约无性能记录。
- **已知外观问题（不影响正确性/性能，未在调优后改动）**：工厂内 `chunk` 上方的注释仍写「three of them must fit inside 192 KB」，那是 **grid-1 combine** 形态的推导依据；OP3 后每 block 只有一个 `(chunk, d)` staging tile，该上界偏保守（`chunk` 公式本身与基线一致、未变）。因该文件的 sha256 已被 §5 Final 记录逐字节对账（`6a702c60…`），本轮**不为注释再改文件**（改则须重采 9 条 final 记录，而 `perf_records.jsonl` 为 append-only、同一 `(kernel_id, workload_id)` 只允许恰一条 final 记录，二者不可同时满足）——故以本注记代替。
- **交叉验证**：迁移基线报告的四档数字（`50.92 / 51.04 / 59.60 / 225.64 µs`）与本轮 K1 的 baseline 实测（`50.90 / 59.52 / 225.56 µs`）逐一吻合 → 迁移期采到的正是 **K1**（闭包首个 kernel），本轮的 3-kernel 分面口径是它的严格超集，不存在口径漂移。
- **总体结论**：`best_effort` 无硬门槛。以「消除元素级 GM 载入标量化」+「K3 从 grid-1 串行改 slot 并行」两项结构改动取得 13.7–54.6×；随后按 shape 分派 `BLK_S` 收敛微调；Round 4 的 tap 展开经 A/B 否决并回滚。**中止原因 = `success`（`best_effort` 语义）**：全部 3 个 tune workload（×3 kernel = 9 对）均有调优结论与同一最终候选的 final 记录、合并处非回退检查通过、smoke 精度通过。结构性候选已穷尽：Round 4 的 tap 展开全域 ≤1% 或负向（A/B 0/3 回滚），剩余方向（K2 代数式外提以减少 `vbrc`/`vmul` 条数）预估收益 ≈3% 已落在噪声阈值内，不值得以精度风险换取。**注**：`plateau` 的机械判据（连续 3 轮无 >3% 提升）**未正式达成**（Round 3 的 tiling 分派在 s32 上仍有 −12.0%），故按 `success` 收束；已按 T-1 完成**参照结构 diff 检查**——对照同族 `examples/engram/engram_bwd.py`（3 kernel / atomic / 语句级 if）、同族正向 Stage 4 最优实现（单 kernel + 张量化载入 + tap 展开 + `cap=4`）与 `kb_search` 命中的 `CASE-engram-decode-stage4`（K 分块外积 + `vreduce`）：本算子 final 的**grid 组织**（`M·num_tiles` + 6-slot combine）、**数据流分相位**（K1/K2/K3 三相位 + `d_silu` GM 交接）、**片上驻留策略**（raw staging + 整 tile `vcast` + 别名复用）、**dtype 路径**（fp16/bf16 → fp32 中转 → 单次 `rint` 回写）均已与该族最优形态一致，**未发现新的结构差异点**（唯一剩余差异 = 备选 B 的 2-kernel 融合，其 UB 预算已实测超限，见 §0.1）。
- **设计层天花板判定**：**未触发 `[DESIGN_LIMIT]`**（故**不产出** `perf_feedback.md`）。依据 `.agents/skills/_shared/standards/perf-feedback.md` §1 双门槛：① 性能不足由**参数/结构级**（载入形态、combine 组织、tile 宽度）可解，非算法/设计层不可解——DESIGN §0.6 的 R1（3-kernel 闭包）、R4（算术掩码）、R5（确定性两段式）三项重设计在实测中均**成立且有效**，无一条被推翻；② 不存在「结构性加速估计 > 2x 不可达」的残留天花板，也无与 DESIGN 假设**直接矛盾**的实测（§1.3 的偏差是**常数口径**修正：`CONST-vector-launch-overhead` 0.5 µs/op 在本设备高估发射，属已知 `VP-2026-0285` 证伪域的**数据修正**，且方向是"实际更快"而非"设计上界不可达"）。

---

## 6. Skill Retrospective

### 6.1 Skill 流程问题与建议修改

**问题 1（Optimize-flow，高价值）**：`bottleneck-patterns.md` 的 `BP_scalarized_loop_carried_accum`（来自 `VP-2026-0283`）**已在同族正向 Stage 4 被证伪一次**（`CASE-engram-gate-conv-fwd-migration`：整 tile 累加已向量化），本轮**再次**以同一形态在 K1 `conv`、K1 `dconv`、K2 转置三处命中并再次证伪（IR = `hivm.hir.vbrc`+`vmul`+`vadd`）。该条目的触发信号至今仍是「`T.Parallel` 内 `+=`」，**缺少累加粒度判别器**，导致每个同族后续任务都要重走一次证伪路径（`kb_search` 返回该条目时 DESIGN §12.2 即以"高危点"预注入，进一步抬高了误判成本）。
→ 建议 update `BP_scalarized_loop_carried_accum`：触发信号改为「`TILELANG_DUMP_IR=1` 后 `scf.for` body 内出现 `tensor.extract`/`tensor.insert` **逐元素**形态」；`T.Parallel` 内 `+=` **不再单独构成触发条件**。
（建议目标文档 `bottleneck-patterns.md` + `pattern-library/INDEX.md` 的该条目一句话；`vp_type=R`，证据 = 本 log §0.2 + 同族正向 `perf_opt/opt_log.md` §0.2。）

**问题 2（Optimize-flow）**：`profile-collection.md` 的 msprof 命令模板假定"一个 workload ↔ 一个目标 kernel"，且工具链只捕获**首个**匹配 kernel——本算子的工厂返回**闭包调度 3 个独立 jit kernel**，无 filter 采集时捕到的是框架算子 `ZerosLike_...`（`torch.zeros` 生成），若不核查会得到完全错误的 baseline（首采即踩中）。当前 skill 未给出"一个案例调用多个目标 kernel"时的 `--kernel-name` 逐 kernel 采集与命名约定。
→ 建议 update `profile-collection.md` §1/§3：增补「工厂返回闭包/多 kernel 调度时，`(kernel_id, workload_id)` 按 kernel 展开、`target_kernel_name` 取 jit kernel 名（非 prim_func 名）、无 filter 采集仅作 nameprobe 定位（须先剔除框架算子）」。
（建议目标文档 `profile-collection.md`；`vp_type=R`，证据 = 本 log §1.1 的 nameprobe 记录与 `profiles/nameprobe/` 原始数据。）

**问题 3（Optimize-flow）**：`profile-collection.md` 与 `.agents/tools/ab_test.py` 的 `OpBasicInfo` 路径都假定 `OPPROF_*/…/*/OpBasicInfo_*.csv`（多层 + 时间戳），本工具链（CANN 9.0.0 / 910B3）实际产出 `OPPROF_*/OpBasicInfo.csv`（1 层、无时间戳），且逐指标 CSV 为 **per-block 长表**（`block_id` 多行），`ab_test.py` 因此报 `no OpBasicInfo rows`。同族正向 Stage 4 已报告同一问题（其 §6.1 问题 2），**本轮再次复现**（本目录自写 `ab_interleaved.py`/`extract_metrics.py` 绕过）。
→ 建议 update：`ab_test.py` 的 CSV 定位改为 `**` 递归 glob 且兼容两种命名；`profile-collection.md` 同步，并注明 per-block 长表的聚合口径（median = 典型 block，max = 关键路径）。
（建议目标文档 `.agents/tools/ab_test.py` + `profile-collection.md`；`vp_type=R`，证据 = 本 log §1.1/§3 与 `perf_opt/ab_interleaved.py` 头注释。）

**问题 4（Optimize-flow，方法学）**：per-kernel 的**单次 run `msprof op` Task Duration 抖动可达 ±10%**（本轮实测：同一文件同一 workload 的 K1 在 4.60–5.32 µs 间波动；`r2` 与 `r3_op4` 的 s128/k1 单次对比一度给出 +11% 的假回退，经 3-pair 交错 A/B 复判为 0/3 的**真实**回退；而 s32/k1 的 −4.9% 单次表观值经 A/B 判为 **tie**）。当前 SKILL 的 `<5% 走 ab_test` 门槛在本量级（3–9 µs 小 kernel）**偏松**：±10% 的抖动使 5–15% 的表观差异仍不可信。
→ 建议：小 kernel（Task Duration < 10 µs）把交错 A/B 的触发门槛从 5% 收紧到 10%，或在候选对比表旁**强制标注单次 run 的复测次数**。
（建议目标文档 `SKILL.md`「先确认测量分辨力」节 + `iteration-diagnosis.md`；`vp_type=R`，证据 = 本 log Round 3/4 的 A/B 与单次 run 对照表。）

### 6.2 Pattern-library 回写内容（本任务写域为 `perf_opt/`，物理回写待终态蒸馏落地）

> 本任务的调度约束把产物限制在 `perf_opt/`（`references/pattern-library/` 属 skill 目录，本 Agent 不写入），故以下为**待回写内容 + repro**（内容自包含，`vp_type` 与证据已就绪）。

- **`traps-compiler.md`（add）**：新条目「**npuir `T.Parallel` 内元素级 `T.cast` 读 GM（含运行时 clamp 行索引）→ 嵌套标量 `scf.for`**」。现象：`T.Parallel(R,d): vnw[r,j] = T.cast(vhat[bid,gc,j], f32)` lower 为 `scf.for(R)×scf.for(d)` 的 `memref.load`+`extf`+`tensor.insert`；每 block 规模 `R·d`。绕法：raw dtype 整块 `T.copy`（own 精确 region + halo clamp-base 连续拷贝）+ 整 tile `T.vcast`。与 `TRAP-npuir-2d-gm-cast-bufferize` 互链（本条为其**性能面**：前者是正确性失败，本条是 62–99% 的标量时间）。实测单算子 **13.7×–54.6×**（三 workload 合计）。repro：`perf_opt/probe/dump_ir.py`（`TILELANG_DUMP_IR=1` 最小 shape 编译 + 逐 kernel 统计，可复现 `memref.load` 由 14→1）。
- **`bottleneck-patterns.md`（update）**：`BP_scalarized_loop_carried_accum` 补「累加粒度 = 元素 vs 整 tile」判别器（见 6.1 问题 1；本任务是**第二次**同族证伪，证据强度已足）。
- **`layout.md` / `cases.md`（add）**：新 C ASE「`CASE-engram-gate-conv-bwd-stage4`」——首个 **sequence_modeling 反向（staged 3-pass）** Stage 4 档案：`aiv_scalar_ratio 0.62–0.99 → 0.40–0.64`；两项结构改动（张量化载入 / K3 由 grid-1 改 `grid=KS+2` slot 并行 + 统一 workspace + `(KS+2,d)` 输出切片）；`BLK_S` **按 shape 分派**的 A/B 正反证据（s32 −17.7% / s128 +15.2%，同 d256 不同 seq_len 结论相反）；tap 展开回滚（`TRAP-npuir-per-copy-ub-staging-sum` 的 copy-site 反作用）。
- **`constants.md`（update/add）**：回填 D-2 实测——W3 设计估算下界 51 µs（发射项主导）vs baseline 实测 252.42 µs（**4.9× 低估**），且 final 18.42 µs **已低于设计估算下界**；佐证 `VP-2026-0285` 对 `CONST-vector-launch-overhead`（0.5 µs/op）的口径收窄，并给出 910B3 上的方向性结论（该常数在 910B3 上**高估**发射成本）。
- **`repro/`（add）**：① `perf_opt/probe/make_variant.py`（baseline + 命名补丁链 → 分支文件，可复算 sha，是本任务全部 8 个分支与两个 A/B 对照的**最小形态**：OP1/OP2 的 old→new 片段即"慢→快关键代码更改"）；② `perf_opt/probe/dump_ir.py`（IR 标量化定位，`TILELANG_DUMP_IR=1` + 逐 kernel 统计）；③ `perf_opt/ab_interleaved.py`（3-kernel 版交错 A/B 协议）。
- **`capability-gaps.md`**：本轮**无**新增能力缺口（`T.atomic_add` 崩溃、多 `T.Kernel` 不可用等仍为**已登记**条目且本轮未触发）；`seq_len=1`（B-S1）的 codegen `TVMError` 属**既有**行为（基线同样复现），不新增条目，仅在 opt_log 留痕。

### 6.3 Value Point Proposals

| id | vp_type | 标题 | 证据三件套 |
|---|---|---|---|
| **VP-stage4-egcb-vec-load** | **P** | npuir `T.Parallel` 内**元素级 `T.cast` 读 GM**（2-D region + 运行时 clamp 行索引）被 lower 为嵌套标量 `scf.for`（`memref.load`+`extf`+`tensor.insert`）；改「raw dtype 整块 `T.copy`（own 精确 region + halo clamp-base）+ 整 tile `T.vcast`」后标量段消除，三 workload 合计 **13.7×–54.6×**（K1 225.6→4.5 µs、K2 497.7→7.5 µs 于 `bwd-b1-s128-d256-bf16`；910B3 / CANN 9.0.0 / tilelang `0.1.2+90908d56`） | evidence：本 log §0.2（IR 前后对照）+ §2 Round 1 + `perf_records.jsonl` `r1_op1_k1vec`/`r1_op2_k2vec`；repro：`perf_opt/probe/make_variant.py`（K1_LOADS/K2_LOADS 补丁片段）+ `perf_opt/probe/dump_ir.py`；toolchain：tilelang `0.1.2+90908d5694f131fa5538eeb3546d740c4b50bec4` / CANN 9.0.0 / Ascend910B3 |
| **VP-stage4-egcb-k3-slot-parallel** | **P** | 确定性两段式权重梯度的 **combine 段**：grid=1 串行 6 次归约 + 6 次 `(d,)` unroll 标量填充 → **统一 workspace（slot = 输出）+ `grid = KS+2` 一 block 一槽 + `(KS+2,d)` 输出缓冲由闭包切片**，单 kernel **−93.1%…−95.2%**（25.1/52.1/26.3 → 1.7/2.5/1.8 µs），且**去掉所有 `(d,)` unroll 标量 staging**（消除 `TRAP-npuir-parallel-scalar-store-race` 的绕法成本） | evidence：本 log §2 Round 2 + `perf_records.jsonl` `r2_op3_k3slots`（k3 三档）+ §0.2 k3 IR；repro：`perf_opt/probe/make_variant.py` 的 `OP3_K3UNIFIED` 补丁（7 处替换，含 K1/K2 写入槽位与 K3 全量重写）；toolchain 同上 |
| **VP-stage4-egcb-tiling-dispatch** | **D** | 「窄 tile 提并行度」不是全域有效：同为 `d=256`，`seq_len=32`（cap 8→4 blocks 减半为 4 → 8 blocks）**−17.7%（A/B 3/3）**，而 `seq_len=128`（16→32 blocks）**+15.2%（A/B 0/3）**；`d=512` 由 4→8（32→16 blocks）**+8.7%（0/3）**。根因：wall = 固定 per-block 开销 + O(blk_s) 工作，blocks 远少于物理核（40）时减半有效，blocks 已 ≥16 时 halo 冗余（3/(blk+3)）与总 op 数翻倍反噬 ⇒ 需按 `(M, seq_len, d)` 编译期分派（落点 `s32:4 / s128:8 / s64-d512:4`） | evidence：本 log §2 Round 3 三组 A/B JSON（`profiles/ab/`）+ `perf_records.jsonl` `r3_op4_blk4d256`/`r3_op5_blk8d512`/`r3_op4_dispatch`；repro：`perf_opt/probe/make_variant.py` 的 `TILING`/`BLK_*` 补丁 + `perf_opt/ab_interleaved.py`；toolchain 同上 |
| **VP-stage4-egcb-tap-unroll-negative** | **D** | 反向否证：把 `for p in range(KS)`（运行时 `scf.for` + 动态 UB 子视图）显式展开为静态偏移语句，**在同族正向 Stage 4 是 4.7–5.3% 的正向**，在本算子却 `seq_len=128/k1` **−8.1%（A/B 0/3）**、其余 ≤1%。归因：展开把 K1 dconv 的 `T.copy` **copy site** 由 3 增至 12（`TRAP-npuir-per-copy-ub-staging-sum`：staging 按 site 逐个分配）且每 tap 新增 `T.vbrc`，抵消了 `T.clear` 消除收益 ⇒ 「同族先例的优化项不可照搬」再次实证 | evidence：本 log §2 Round 4 + A/B JSON（`profiles/ab/bwd-b1-s128-d256-bf16__k1/`）+ `perf_records.jsonl` `r4_op6_tapunroll`；repro：`perf_opt/probe/make_variant.py` 的 `OP6_TAPUNROLL` 补丁（3 处替换）+ `_engram_gate_conv_bwd_kernel_round4_op6_tapunroll.py`（负向分支文件保留）；toolchain 同上 |
| **VP-stage4-egcb-multikernel-profile** | **D** | 工厂返回**闭包调度多 kernel** 时，`msprof op` 无 `--kernel-name` 只会捕到**首个** launch（可能为 `torch.zeros` 产生的 `ZerosLike_*` 框架算子）→ baseline 会完全错；正确口径 = 按 jit kernel 名逐 kernel 采集并把一个 workload 登记为多个 `(kernel_id, workload_id)` 对 | evidence：本 log §1.1（nameprobe 记录）+ `profiles/nameprobe*/OPPROF_*/OpBasicInfo.csv`（捕获 `ZerosLike_…`, `Task Duration 1.2/1.18/1.68 µs`）+ 本 log §6.1 问题 2；repro：`perf_opt/bench_runner.py` + `perf_opt/run_experiments.py`（`--kernel k1,k2,k3`）；toolchain 同上 |

---

## 7. 交付物索引

| 文件 | 说明 |
|---|---|
| `perf_opt/_engram_gate_conv_bwd_kernel.py` | 最优 drop-in 版本（sha256 `6a702c606f9a4343b727af94cf93bea163bd2e84e68be34f050304cbb256d373`）；模块级暴露 `TUNED_BLK_CAP_D256/D512`、`TUNED_TARGET_BLOCKS` |
| `perf_opt/opt_log.md` | 本文件 |
| `perf_opt/perf_records.jsonl` | append-only 结构化性能记录（65 行：baseline 9 / candidate 20 / merged 27 / final 9；每行字段契约见 `signal-registry.md` §5） |
| `perf_opt/workload_inventory.json` | workload 清单（1 smoke×3 个 kernel + 3 tune×3 个 kernel = 12 项，`tuning_status=winner_merged`） |
| `perf_opt/profiles/` | 各阶段 raw `msprof op` 数据（`baseline/`、`candidate/`、`merged/`、`final/`、`nameprobe*/`、`ab/`） |
| `perf_opt/logs/` | 实验 stdout/stderr、L0 日志、msprof 命令日志（`logs/phase0/`、`logs/msprof/`、`logs/l0/`、`logs/final_level_all.log`） |
| `perf_opt/run_experiments.py` / `bench_runner.py` / `ab_interleaved.py` / `extract_metrics.py` | 采数、交错 A/B 裁决、指标提取 runner/helper |
| `perf_opt/probe/make_variant.py` / `probe/dump_ir.py` / `probe/compile_check.py` | 分支确定性生成（补丁链）/ IR 标量化定位 / 编译可行性门（UB 探针） |
| `perf_opt/_engram_gate_conv_bwd_kernel_round{1,2,3,4}_*.py` | 各轮实验分支（含负向分支 OP6 的保留证据） |
