# Stage 4 调优日志 — `_engram_gate_conv_fwd_kernel`

- 场景: `scenario=optimize`, `mode=full`
- 项目 / 算子: `engram_gate_conv_fwd` / `_engram_gate_conv_fwd_kernel`
- 算子目录: `examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_fwd/engram_gate_conv_fwd_kernel/`
- 基准 kernel（只读）: `_engram_gate_conv_fwd_kernel.py`
- 调优产物: `perf_opt/_engram_gate_conv_fwd_kernel.py`（drop-in，同工厂签名 `(M, seq_len, d, eps, dtype)`，`out_idx=[6,7,8,9,10,11]`）
- 性能口径: **`msprof op` `Task Duration(us)`（kernel-only）**，唯一口径；`--kernel-name=main`（`OpBasicInfo.csv` 实测 `Op Name=main`, `Op Type=vector`）
- 噪声阈值: 3%；`<5%` 差异走交错 A/B/A/B 多 run 协议（`perf_opt/ab_interleaved.py`）
- `max_rounds` = 10；实验分支实际使用 9 个（无效/阻塞分支不写 `perf_records.jsonl`）
- 工具链戳: tilelang `0.1.2+bafbd81b1538684f98dec570c7d7cdc4863f5d02`（commit 013dbbf, `/mnt/workspace/Agentkj/build`, USE_NPUIR=ON）/ CANN 9.0.0 / Ascend910B3 x1 / 2026-10-16

---

## 0. Phase 0 诊断（采数前强制项）

### 0.1 算子类型与首轮必查项

- **算子类型 = 纯 Vector**（无 `T.gemm`；4 处行内 `reduce_sum`；grid = `M·num_tiles`）。与 DESIGN §5.1/§5.5 一致。
- **向量化轴/布局重估**：DESIGN §1.6.3 主选 A（I/O 行主序、尾轴 `d` 向量化、`seq_len` block 级分块、UB 内因果滑窗）经本轮实测确认无更优布局（见 §4 结论）；未发现需要 repack 的信号。
- **陷阱条目版本戳核对**：本算子自身 4 条陷阱在任务开始时均被判 **stale**（工具链已变更：设计期 `2562795b` → 当前 `bafbd81b`）。已按同口径重验（`python3 .agents/tools/repro_runner.py --filter <ID>`）：
  - `TRAP-npuir-parallel-runtime-minmax` → **PASS**（1-D `T.Parallel` 标量 store 内运行时 `min/max` 仍 SIGSEGV）
  - `TRAP-npuir-parallel-scalar-store-race` → **PASS**（unroll fill + 单次 `T.copy` 仍确定）
  - `TRAP-npuir-vcast-shape-mismatch` → **PASS**（`T.vcast` 仍要求 src/dst 同 shape）
  - `TRAP-npuir-ifthenelse-parallel-segv` → repro 文件不在 `repro/`（该条在 `traps-compiler.md`，见下）
  - **结论：本轮引用这些条目时已重验，可按 `status=verified`（本工具链戳）使用**，非"待重验"。

### 0.2 IR 级标量化定位（`VP-2026-0286` 操作条）

`TILELANG_DUMP_IR=1` 编译最小 shape（`M1,S16,d256,fp16`，日志 `logs/phase0/dump_ir_b1_s16_d256.log`），grep `scf.for` / `tensor.extract` / `tensor.insert` / `memref.load`：

**基线 IR 最终形态（关键片段）**

```mlir
// H / k / v 窗口载入 —— 嵌套标量循环（每张量 R×d = 11×256 次标量 load）
%28 = scf.for %arg21 = ... to %c11_i32 ... iter_args(%arg22 = %2) -> (tensor<11x256xf32>) {
  %84 = scf.for %arg23 = ... to %c256_i32 ... iter_args(%arg24 = %arg22) -> (tensor<11x256xf32>) {
    %91 = memref.load %reinterpret_cast[%c0, %89, %90] : memref<1x16x256xf16, ...>
    %92 = arith.extf %91 : f16 to f32
    %inserted = tensor.insert %92 into %arg24[%93, %90] : tensor<11x256xf32>
    ...
// 权重载入 —— 标量循环
%24 = scf.for ... to %c256_i32 { memref.load ...; tensor.insert ... }

// conv tap 累加 —— 已被向量化（非标量！）
%75 = scf.for ... to %c4_i32 iter_args(%arg22 = %74) -> (tensor<8x256xf32>) {
  %92 = hivm.hir.vmul ins(%91, %extracted_slice_26 : tensor<8x256xf32>, ...)
  %93 = hivm.hir.vadd ins(%arg22, %92 : tensor<8x256xf32>, tensor<8x256xf32>)
```

**结论（对 DESIGN §11.2 预注入线索的独立验证）**

- ⚠️ **证伪**：DESIGN §11.2 预判 L322–326 的 `conv[i,j] += cb[i,j]*prod[i+p,j]` 是"与 `VP-2026-0283` 同形态的元素级 loop-carried 累加 → 被 lower 成标量 `scf.for`"。**实测不成立**：该 `T.Parallel` 的累加是 **(blk_s,d) 整 tile 的 `vmul`+`vadd`**（loop-carried `iter_args` 是整 tile，非元素），conv 段**已向量化**。
  → 归因：`VP-2026-0283` 的标量化形态是「**元素级** loop-carried 累加（`tensor.extract`+`addf`+`insert` 逐元素）」，而本处累加粒度是整 tile（`hivm.hir.vadd`）。**同为 `+=` 写法，lowering 结果取决于累加粒度/是否触发 `iter_args` 元素化**——`VP-2026-0283` 的识别信号需补「累加粒度 = 元素 vs 整 tile」的区分器（见 Skill Retrospective）。
- ✅ **真正的标量化热点**：**H/k/v 窗口载入**（`T.Parallel(R, d_padded)` 内 `T.cast(H[bid, gc, j], f32)`，`gc` 为运行时 clamp 行索引）被 lower 为**嵌套标量 `scf.for`**（`memref.load`+`extf`+`tensor.insert`，每张量 `R×d` 次）+ 权重载入 2×`d` 次。
  → 机制：`T.Parallel` 内的 **元素级 `T.cast` 读 GM** + 运行时行索引使向量化器退化为标量逐元素；与 `TRAP-npuir-2d-gm-cast-bufferize`（2-D GM 元素 `T.cast`）同族。

### 0.3 初始现象（baseline profile，`phase=baseline`）

| workload | Task Duration(us) | aiv_time(us) | vec% | **scalar%** | mte2% | GM_to_UB(KB) | read(KB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fwd-b1-s32-d256-fp16` | 92.93 | 91.53 | 2.3 | **97.5** | 0.5 | 2.0 | 42.2 |
| `fwd-b2-s64-d512-fp16` | 136.42 | 131.80 | 1.7 | **98.3** | 0.4 | 4.0 | 50.2 |
| `fwd-b1-s128-d256-bf16` | 412.38 | 409.70 | 28.5 | **61.8** | 0.1 | 2.0 | 37.2 |

- **P1（主导）**：`aiv_scalar_ratio` 0.62–0.98、`aiv_vec_ratio` 0.02–0.29 → **标量执行主导**（`VP-2026-0286` 的 `>50%` 触发换轴/定位）；`aiv_main_mem_read_bw ≈ 0.44 GB/s`（搬运可忽略）→ 纯发射/标量受限，非带宽受限。
- **P2**：`s128` 的绝对标量时间（0.618×409.7 ≈ 253 µs）约为 `s32`（0.975×91.5 ≈ 89 µs）的 2.8×，但每 block 工作量相同（R=11、blk_s=8、d=256）且 block 数 16 ≤ 40 物理核 → 标量逐元素 GM 读为**延迟受限**，并发 block 增多时 L2 命中下降，代价超线性。这解释了"`s16`≈`s32` 但 `s128` 涨 4.4×"的固定/工作量混合现象（DESIGN §11.1 提示）。
- **P3**：任务语义要求的 `T.Parallel` 内 `+=`（`VP-2026-0258` 反侧）在本算子**未触发错值**（correctness 4/4 稳定）——本处累加为整 tile 向量形态，非标量累加进 shared。

---

## 1. Performance Test Data

### 1.1 tune workload baseline（`phase=baseline`, `round=0`）

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | profile_status | raw_profile_dir |
|---|---|---|---|---:|---|---|---|
| `...::_engram_gate_conv_fwd_kernel::main` | `fwd-b1-s32-d256-fp16` | baseline | main | main | 92.93 | valid | `profiles/baseline/round0/baseline_8ba8496a/fwd-b1-s32-d256-fp16_6b128b61/` |
| `...::_engram_gate_conv_fwd_kernel::main` | `fwd-b2-s64-d512-fp16` | baseline | main | main | 136.42 | valid | `profiles/baseline/round0/baseline_8ba8496a/fwd-b2-s64-d512-fp16_1430caf3/` |
| `...::_engram_gate_conv_fwd_kernel::main` | `fwd-b1-s128-d256-bf16` | baseline | main | main | 412.38 | valid | `profiles/baseline/round0/baseline_8ba8496a/fwd-b1-s128-d256-bf16_d8007dbe/` |

命令模板（`msprof op --kernel-name=main --output=<dir> --launch-count=20 --warm-up=5 --dump=off --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio python perf_opt/bench_runner.py --impl <file> --case <workload_id>`）。

### 1.2 smoke / skipped 排除

| workload_id | kind | 原因 | 精度 |
|---|---|---|---|
| `fwd-smoke-b1-s16-d256-fp16` | smoke | benchmark `i=0` → `pytest.mark.smoke`；不做性能 baseline、不选 winner | **pass**（maxdiff 8.869e-04，`_run_case(1,16,256,fp16)`） |

无 `skipped` 项（benchmark 4 个参数化案例全部可运行）。

### 1.3 设计估算 vs 实测偏差行（D-2 回填）

DESIGN §1.6.0 设计期估算（W3 `M2,S64,d512,fp16`）：下界 ≈ 23 µs，**发射项主导**（`向量 op 数 ≈46 × 0.5 µs`）。实测 baseline W3 = **136.42 µs** → 偏差 **5.9×**（低估）。失准项：**发射项口径**——`CONST-vector-launch-overhead`（0.5 µs/op）在 `VP-2026-0285` 中已被证伪（仅适用逐 op 串行小向量）；本算子真实成本来自**标量逐元素 GM 读**（`memref.load` 延迟受限），非 op 数×常数。该偏差行即本轮 `[DESIGN_LIMIT]` 归因的量化证据（最终仍**未**触发反向反馈，因问题参数/结构可解，见 §5）。

---

## 2. Iteration Log

> 每轮固定 `base = current best`；每个实验分支只改一个主要优化点；每个 valid 分支跑 L0 精度回归后再用 `msprof op` 采集。`<5%` 差异一律走 `perf_opt/ab_interleaved.py` 交错多 run 协议。

### Round 1 — OP1：向量化 H/k/v/权重载入  ⭐ 决定性

**现象**：P1（标量 97.5%/98.3%/61.8%）；IR 证据 = §0.2。

**候选优化点**

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP1 | 标量占比 62–98%，`memref.load` 逐元素 | H/k/v + 权重载入改「raw dtype 整块 `T.copy` + 整 tile `T.vcast`」 | IR 嵌套标量 `scf.for`；`memref.load=4` | 窗口拆 `[halo(3) \| own(blk_s)]` 两段连续 `T.copy`（own 用 `t0:t0+blk_s` 精确 region，halo 用 clamp base `hb`），再 `T.vcast` 整 tile | `memref.load`→0、`aiv_scalar_ratio` 下降、Task Duration | improved |

**实验分支（vs current best = baseline）**

| branch | base | opt_id | major_change | L0 | Task Duration(us) | aiv_time | vec% | scalar% | mte2% | GM→UB(KB) | profile | result |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| baseline | — | — | — | pass | 92.93 / 136.42 / 412.38 | 91.53/131.8/409.7 | 2.3/1.7/28.5 | 97.5/98.3/61.8 | 0.5/0.4/0.1 | 2/4/2 | valid | — |
| `round1_op1` | baseline | OP1 | 向量化载入 | pass | **5.90 / 7.94 / 6.94** | 5.27/6.28/5.68 | 42.5/38.9/39.8 | 22.5/34.8/31.8 | 41.1/36.1/33.8 | 19.5/27/19.5 | valid | **improved 15.7×–59.4×** |

**判断**：三 workload 全部远超 3% 噪声阈值 → 无需 ab 协议，直接采纳为 current best。相对基线 **−93.7% / −94.2% / −98.3%**（15.7× / 17.2× / 59.4×）。IR 复检：`memref.load=0`，`memref.copy=18`，残余 `scf.for` 仅 mask(11)/own-rows staging(4×8)/conv tap(4)。

### Round 2 — OP1：BLK_S 粗扫（大/小两方向）

**现象**：Round 1 后标量已降到 23–35%；剩余结构自由度 = `BLK_S`（DESIGN 指定的 Stage-4 杠杆；`cap = 8 if d_padded<=256 else 4`）。

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP2a | block 数少、halo 冗余 | 增大 `BLK_S`（d256:8→16；d512:4→8） | DESIGN §5.5 杠杆 | `cap=16/8` | Task Duration / block 数 | config_no_gain |
| OP2b | 并行度不足 | 减小 `BLK_S`（d256:8→4） | block 数翻倍（s32: 4→8） | `cap=4/4` | Task Duration | improved(小) |

| branch | base | opt_id | L0 | s32(us) | s64(us) | s128(us) | result |
|---|---|---|---:|---:|---:|---|
| `round2_r2op1_bigs` | round1_op1 | OP2a | pass | 9.08 | 8.24 | 7.50 | **config_no_gain（全回退）** |
| `round2_r2op2_smalls` | round1_op1 | OP2b | pass | 5.46 | 8.08 | 6.92 | improved(s32 −7.5%) |

**判断**：OP2a 全线回退（block 数减少 → 核占用下降，`s32` 仅 2 block）；OP2b 对 `s32` 有效（+并行度），对 `s128/s64` 落在噪声内。**`BLK_S` 越大越慢 → 该算子受 block 级并行度主导**。保留 OP2b 的 `cap=4`（d256）进入后续组合。

### Round 3 — OP1：隐式跨 dtype `T.copy` / OP-AB：两 Pass 裁决（DESIGN §1.6.3 强制）

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP3 | 载入多一次 `vcast` + `hr` 暂存 | 去掉 raw staging，用 `T.copy` 隐式 f16/bf16→f32 | `TRAP-C12-copy-dtype-cast` | `T.copy(H[...], ht)`（f32 dst） | L0 精度 | **invalid** |
| OP-AB | DESIGN §1.6.3 实验裁决（主选 A vs 备选 B 两 Pass） | 两 `T.Kernel` 块：Pass1 写 `vhat/rrms_*`，Pass2 读回 halo 重算 `vhat_norm` 卷积 | DESIGN §1.6.3 | 同 prim_func 内两 grid 发射（同 GPU 源结构，无额外 workspace） | 可运行性 + Task Duration | **blocked** |

| branch | base | opt_id | L0 | 结果 |
|---|---|---|---|---|
| `round3_op1_implicitcast` | round1_op1 | OP3 | **fail** | 52.5% 元素错（`Y` max_abs 2.27 @ (0,12,195)）→ `T.copy` 跨 dtype **不做数值转换**，确认 `TRAP-C12` |
| `round3_opAB_twopass` | round1_op1 | OP-AB | **blocked** | 编译通过（IR 两 `func.func @main`）但**运行期 launcher 不支持多 `T.Kernel`**：`ValueError: Failed to evaluate grid expression 'None'`（`jit_npu.py:1267` `_calcu_grid`） |

**A/B 裁决结论（DESIGN §1.6.3 必做项）**：候选 A（单 kernel 序列分块）**胜**。证据：① **备选 B 在 npuir Developer 不可运行**——单个 `@T.prim_func` 内两个 `T.Kernel` 块编译通过但 launcher 无法求 grid（上述 traceback）；② B 的替代实现（两个 jit 工厂 + 闭包）需经 GM 传 `vhat_norm` workspace，而冻结的 drop-in 契约（`out_idx=[6..11]`、`register_fake` 6 张量）无 workspace 输出位，B 不能作为 drop-in 表达；③ 结构性方向：B 额外增加 1 次 kernel 启动 + `vhat_norm` GM 往返，而 A 已实测相对基线 15.7×–59.4×，B 的结构性上限远小于该量级。**A 的实测数字即本裁决的固化依据**（设计期判定：主选 A 保留，B 弃选）。

### Round 4 — OP4：conv 权重 hoist

**现象**：Round 1 profile 中 `mte2%` 达 33–41%（P1 已消除）；conv 段每 block 做 4 次 (1,d) 卷积权重 GM 行拷贝。

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP4 | conv 权重 4 次小拷贝/block | hoist 成 1 次 (4,d) 区域拷贝 + 1 次 vcast，逐 tap 从 UB 行广播 | profile `mte2%`；设计 §6.3 | `T.copy(conv_w,cww_d)` + `T.vcast` + `T.vbrc(cww[p:p+1,:],cb)` | `mte2%` / Task Duration | improved |

| branch | base | opt_id | L0 | s32(us) | s64(us) | s128(us) | result |
|---|---|---|---:|---:|---:|---|
| `round4_op1_convwhoist` | round1_op1 | OP4 | pass | 5.54 | 8.06 | 6.54 | improved(s32 −6.1%, s128 −5.8%) |

### Round 5 — OP5 = OP2b + OP4 组合

| branch | base | opt_id | L0 | s32(us) | s64(us) | s128(us) | result |
|---|---|---|---:|---:|---:|---|
| `round5_op1_combo` | round4_op1 | OP2b+OP4 | pass | **5.04** | 8.04 | **6.44** | improved → 新 current best |

**交错 A/B 复核（s64，`ab_r5_s64`，3 pairs，交替次序）**：A=`round1_op1` median 8.00 µs vs B=`round5_op1_combo` median 7.54 µs → **B −5.75%**（配对 B 胜 2/3）。单次 run 曾显示 s64 +1.3%，交错协议翻转为 −5.75% → `BP_run_state_bimodality` 实证，采纳以交错中位为准。

### Round 6 — OP6：`BLK_S=2`（d256）

| branch | base | opt_id | L0 | 结果 |
|---|---|---|---|---|
| `round6_op1_blk2` | round5_op1 | OP6 | **fail** | `S=13` 编译失败（`hivm.hir.vbrc ... empty broadcast dims array`）；且 `S=4/10` 等 `seq_len%4!=0` 形态下 `t0∈{2,…}<halo` → **halo 错位** |

**重要副产物（正确性硬化）**：本轮暴露 Round 1 引入的 tensorized 载入隐含约束——**`blk_s >= halo` 才保证每 tile 起点 `t0` 为 0 或 ≥ halo**，否则左 halo 的 clamp-base 连续拷贝与期望 token 错位。原工厂 `for cand in (cap, cap//2, 2)` 的 `2` 分支（`seq_len ≡ 2 (mod 4)`，如 6/10/14）会静默错位。**已在 Round 7 加 guard 修复。**

### Round 7 — OP7：`blk_s >= halo` 正确性 guard（无性能变化，硬化为必须）

**改法**：`for cand in (cap, cap//2): if cand >= halo and seq_len % cand == 0` → 候选低于 halo 一律排除，此类 shape 回退单 tile（`blk_s = seq_len`）。已用 `S=6/10/14` 实测验证（`maxdiff ≤ 3.7e-3`，全部通过）。

### Round 8 — OP8：把 `T.clear(conv)` 折入首个 tap

**现象**：conv 段先 `T.clear`（一次 (blk_s,d) 零写）再 4 次 FMA；`for p in range()` 在 prim_func 内是**运行时 `scf.for`**（`TRAP-C13`），无法用 `if p==0` 编译期分支 → 改**显式展开** 4 个 tap（首个用 `=`，后三个 `+=`）。

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP8 | conv 段一次多余的全 tile 零写 | 折 clear 入首 tap | IR `T.clear` 独立 pass；`TRAP-C13` | 显式展开 4 tap，p=0 用 `=` | Task Duration | improved |

| branch | base | opt_id | L0 | s32(us) | s64(us) | s128(us) | result |
|---|---|---|---:|---:|---:|---|
| `round8_op1_noclear` | round7_op1 | OP8 | pass | 5.12 | 7.18 | 6.70 | candidate winner |

**交错 A/B 复核（`ab_r7v8_*`，3 pairs，交替次序；A = `round7_op1_guard`，B = `round8_op1_noclear`）**

| workload | A median(us) | B median(us) | B−A | paired B wins | verdict |
|---|---:|---:|---:|---:|---|
| `fwd-b1-s32-d256-fp16` | 5.08 | **4.84** | **−4.72%** | 3/3 | **adopt_b** |
| `fwd-b2-s64-d512-fp16` | 7.38 | 7.52 | +1.90% | 1/3 | tie |
| `fwd-b1-s128-d256-bf16` | 6.76 | **6.40** | **−5.33%** | 3/3 | **adopt_b** |

**判断**：OP8 在优先 shape `s128`（−5.33%，3/3）与 `s32`（−4.72%，3/3）胜出，`s64` 为 tie → 采纳为最终 winner（`round8_op1_noclear`）。单次 run 的 s32/s128 表观回退（+1.6%/+4.0%）被交错协议推翻——再次印证 `BP_run_state_bimodality`。

### Round 2–8 汇总「候选 vs current best」对比表（B2 结构化回流）

> 数据来源 `perf_records.jsonl` + raw profile（`perf_opt/profiles/`）。时延列为单次 run `msprof op` median；决策列以交错协议为准。

| round | candidate_id | workload_id | Task Duration(us) | aiv_time(us) | vec% | scalar% | mte2% | GM→UB(KB) | L0 | vs current best | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | `r1_op1_vec_load` | s32 | 5.90 | 5.27 | 42.5 | 22.5 | 41.1 | 19.5 | pass | −93.7% | improved |
| 1 | `r1_op1_vec_load` | s64 | 7.94 | 6.28 | 38.9 | 34.8 | 36.1 | 27.0 | pass | −94.2% | improved |
| 1 | `r1_op1_vec_load` | s128 | 6.94 | 5.68 | 39.8 | 31.8 | 33.8 | 19.5 | pass | −98.3% | improved |
| 2 | `r2_op1_bigblk` | s32 / s64 / s128 | 9.08 / 8.24 / 7.50 | - | 38.9/48.4/49.3 | 20.8/27.7/24.5 | 31.7/35.6/33.2 | 31.5/39/31.5 | pass | +54%/+4%/+8% | config_no_gain |
| 2 | `r2_op2_smallblk` | s32 / s64 / s128 | 5.46 / 8.08 / 6.92 | - | 36.4/39.4/34.0 | 24.1/38.8/32.6 | 44.4/32.5/38.7 | 13.5/27/13.5 | pass | −7.5%/+1.8%/−0.3% | improved(s32) |
| 4 | `r4_op1_convwhoist` | s32 / s64 / s128 | 5.54 / 8.06 / 6.54 | - | 46.4/40.7/41.3 | 25.2/40.5/32.0 | 33.5/26.6/30.4 | 19.5/27/19.5 | pass | −6.1%/+1.5%/−5.8% | improved |
| 5 | `r5_op1_combo` | s32 / s64 / s128 | 5.04 / 8.04 / 6.44 | - | 41.6/41.9/36.8 | 28.1/38.2/34.2 | 34.4/28.1/33.2 | 13.5/27/13.5 | pass | **current best** | improved |
| 8 | `r8_op1_noclear` | s32 / s64 / s128 | 5.12 / 7.18 / 6.70 | - | 40.0/44.1/34.0 | 28.1/35.5/38.9 | 38.1/29.7/31.3 | 13.5/27/13.5 | pass | ab 裁决 winner | improved |
| 3 | `r3_op1_implicitcast` | — | — | — | — | — | — | — | **fail** | — | invalid（跨 dtype T.copy 错值） |
| 3 | `r3_opAB_twopass` | — | — | — | — | — | — | — | **blocked** | — | blocked（多 T.Kernel launcher 不支持） |
| 6 | `r6_op1_blk2` | — | — | — | — | — | — | — | **fail** | — | invalid（blk_s<halo 错位 + S=13 编译失败） |

> 注：blocked/invalid 分支按契约**不写** `perf_records.jsonl`（其 `duration_us` 不得为 null）；结构性不可行证据保留于本表与本轮小节。

**当前 kernel 全部 tune workload 合并检查**（每个 winner 都是整体 kernel 的合并版；逐 workload 均按同一 winner 复测全部 tune workload 与 smoke 精度）：
`r1→r2_op2→r4→r5→r7→r8` 各步均对全部 3 个 tune workload 采集、并每次合并后跑 smoke 精度（`fwd-smoke-b1-s16-d256-fp16`，pass）；无 workload 出现可信回退（`s64` 的全部 <5% 差异均以交错协议复核，`r5`/`r8` 未回退）。

---

## 3. Autotune Log

未使用 autotune API。`BLK_S` 采取**手动粗扫**（`cap ∈ {16,8,4,2}` 两方向，Round 2/6）：搜索空间 `{blk_s = cap, cap//2}`（受 `seq_len` 整除与 `blk_s>=halo` 约束），best = `cap=4`（d256）/`4`（d512）。winner 邻域：`blk_s=2` 因 halo 错位被否；`blk_s=16/8` 因 block 数下降回退。最终 winner 以独立 `msprof op` + 交错协议复测（§2 Round 8）。

---

## 4. 布局与向量化轴复核结论（Phase 0 首轮必查项收口）

- 主选 A（尾轴 `d` 向量化 + `seq_len` block 分块 + UB 内因果滑窗）经实测确认；**未发现需要核内重排/转置的收益**：消除标量化后 `aiv_vec_ratio` 升到 34–46%，残余 `aiv_scalar_ratio` 26–39% 对应 mask(11)/own-rows staging(4×8)/conv tap(4) 等**必要的小规模串行**（`TRAP-npuir-parallel-scalar-store-race` 重验为 verified → staging 必须用 unroll + 单 `T.copy`）。
- 残余瓶颈画像（final，见 §5）：`vec ≈ 40% / mte2 ≈ 30–37% / scalar ≈ 27–38%`，`Task Duration(5–8µs) > aiv_time`（差 ≈ 0.6–1.3 µs 为 launch 开销）→ **发射 + 小规模 MTE2/vector 混合受限**，非带宽受限（`GM→UB ≤ 27 KB`）。

---

## 5. Final Summary

**最终候选**：`perf_opt/_engram_gate_conv_fwd_kernel.py`（= `round8_op1_noclear`，采纳的优化点：OP1 向量化载入 + OP2b `cap=4`(d256) + OP4 conv 权重 hoist + OP7 `blk_s>=halo` guard + OP8 折 clear 入首 tap）。
**精度**：L0 PASS / L1 PASS / L2 ok / Boundary ok（`B-S1` 为既有 warn——`blk_s==1` 被 `T.vbrc` 拒，非本改动引入）；smoke `fwd-smoke-b1-s16-d256-fp16` pass（maxdiff 8.869e-04）。`S=6/10/14` 附加回归 pass（guard 硬化验证）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_fwd/engram_gate_conv_fwd_kernel/_engram_gate_conv_fwd_kernel.py::_engram_gate_conv_fwd_kernel::main | fwd-b1-s32-d256-fp16 | 92.93 | 4.80 | final_r8_op1_noclear |
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_fwd/engram_gate_conv_fwd_kernel/_engram_gate_conv_fwd_kernel.py::_engram_gate_conv_fwd_kernel::main | fwd-b2-s64-d512-fp16 | 136.42 | 7.70 | final_r8_op1_noclear |
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_fwd/engram_gate_conv_fwd_kernel/_engram_gate_conv_fwd_kernel.py::_engram_gate_conv_fwd_kernel::main | fwd-b1-s128-d256-bf16 | 412.38 | 7.12 | final_r8_op1_noclear |

逐项提升（同一最终候选，`phase=final`）：`s32` **−94.8%**（19.4×）、`s64` **−94.4%**（17.7×）、`s128` **−98.3%**（57.9×）。交错协议复核（3 pairs）显示 winner 相对前一 best（`r7`）在 `s32` −4.72%（3/3）、`s128` −5.33%（3/3），`s64` tie。

- **smoke**：`fwd-smoke-b1-s16-d256-fp16` 仅精度（pass），无性能记录（契约）。
- **总体结论**：`best_effort` 无硬门槛；以消除「元素级 GM 载入标量化」为核心的单轮结构优化即取得 15.7×–59.4×，随后 BLK_S/权重 hoist/clear 折叠等微调再收敛。中止原因 = **success**（`best_effort`：全部 3 个 tune workload 均有调优结论、最终全量复测与 smoke 精度通过；Round 6–8 已把剩余结构候选逐一验证——`BLK_S` 双向扫描、跨 dtype/两 Pass 载入替代、conv 权重 hoist、clear 折叠均已覆盖，残余 `scalar%` 27–38% 对应 mask/staging/conv-tap 等**必要的小规模串行**（受重验为 verified 的 `TRAP-npuir-parallel-scalar-store-race` 约束），micro 候选穷尽）。
- **设计层天花板判定**：**未触发 `[DESIGN_LIMIT]`**。依据 `.agents/skills/_shared/standards/perf-feedback.md` §1 双门槛：① 性能不足由**参数/结构级**（载入形态）可解，非算法/设计层不可解；② 无「结构性加速估计 > 2x 不可达」或「实测与 DESIGN 假设直接矛盾」的**残留**天花板——DESIGN 的发射项估算失准（§1.3）已被**本轮结构改动消除**（该误估源自已被 `VP-2026-0285` 证伪的 `0.5µs/op` 常数，属数据修正而非设计层天花板）。故**不产出** `perf_feedback.md`。

---

## 6. Skill Retrospective

### 6.1 Skill 流程问题与建议修改

**问题 1（Optimize-flow，高价值）**：`bottleneck-patterns.md` 的 `BP_scalarized_loop_carried_accum`（来自 `VP-2026-0283`）触发信号以 `aiv_scalar_ratio > 0.9` + 「热点 `T.Parallel` lower 为标量 `scf.for`」为准，但**未区分累加粒度**。本任务实测：`T.Parallel` 内**整 tile** loop-carried 累加（`conv[i,j] += cb[i,j]*prod[i+p,j]`，`i,j` 是 tile 元素但累加对象是整 `tensor<blk_s,d>`）**已向量化**为 `hivm.hir.vmul`+`vadd`；而**元素级**逐元素 `tensor.extract`+`addf`+`insert`（`VP-2026-0283` 的真实形态）才标量化。识别信号需补「累加粒度 = 元素 vs 整 tile」判据，否则会把已向量化的 `+=` 误判为标量化热点、错过真正的热点（本例真正的热点是 `T.Parallel` 内**元素级 `T.cast` 读 GM**）。
→ 建议 update `bottleneck-patterns.md` `BP_scalarized_loop_carried_accum`：触发信号加「`TILELANG_DUMP_IR=1` 后确认 `scf.for` body 内为 `tensor.extract`/`insert` **逐元素** 而非 `hivm.hir.vmul`/`vadd` **整 tile**」。
（建议目标文档 `bottleneck-patterns.md`，`vp_type=R`。）

**问题 2（Optimize-flow）**：`profile-collection.md` 的 msprof 命令模板与 `.agents/tools/ab_test.py` 均假定 `OPPROF_*/.../*/OpBasicInfo_*.csv`（3 层）路径，而本工具链（CANN 9.0.0 / Ascend910B3）`msprof op` 实际产出 `OPPROF_*/OpBasicInfo.csv`（1 层）。导致 `ab_test.py` 直接报 `no OpBasicInfo rows`。本轮自写 `perf_opt/ab_interleaved.py` 复刻协议（≥3 pairs 交替次序 + 配对符号检验）绕过。
→ 建议 update：`ab_test.py` 的 CSV 定位改为 `**` 递归 glob（或兼容两种布局）；`profile-collection.md` 命令模板同步。
（建议目标文档 `.agents/tools/ab_test.py` + `profile-collection.md`，`vp_type=R`。）

### 6.2 Pattern-library 回写内容（本任务范围约束下待终态蒸馏落地）

> 说明：本任务的调度约束为「只写 `perf_opt/`」（算子目录外均标注只读），故 `references/pattern-library/` 的物理回写**未在本任务内执行**，改由任务终态 `tilelang-skill-evolver` 按 `distill` 流程落地。以下为**待回写内容 + repro**（`vp_type` 与证据已就绪，内容自包含）。

- **`traps-compiler.md`**（`add`）：新条目「**npuir `T.Parallel` 内元素级 `T.cast` 读 GM（运行时 clamp 行索引）→ 嵌套标量 `scf.for`**」。现象：`T.Parallel(R,d): ht[i,j]=T.cast(H[bid,gc,j],f32)` 被 lower 为 `scf.for(R)×scf.for(d)` 的 `memref.load`+`extf`+`tensor.insert`。绕法：改「raw dtype 整块 `T.copy`（own 精确 region + halo clamp-base 连续拷贝）+ 整 tile `T.vcast`」→ `memref.load` 4→0。与 `TRAP-npuir-2d-gm-cast-bufferize` 互链（同族，本条为其**性能面**）。实测单算子 15.7×–59.4×。
- **`bottleneck-patterns.md`**（`update`）：细化 `BP_scalarized_loop_carried_accum` 的**累加粒度判据**（见 6.1 问题 1）。
- **`constants.md`**（`update`）：回填 D-2 实测——W3 baseline 136.42 µs vs 设计发射项估算 23 µs（5.9× 低估），佐证 `VP-2026-0285` 对 `CONST-vector-launch-overhead` 的口径收窄。
- **`repro/`**（`add`）：`perf_opt/probe/load_variants.py`（5 种载入形态最小对照，`variant_a`=标量 vs `variant_c/d`=向量化，`TILELANG_DUMP_IR=1` 可复现 `memref.load` 计数差）；`_engram_gate_conv_fwd_kernel_round3_opAB_twopass.py`（两 `T.Kernel` launcher 失败）；`_engram_gate_conv_fwd_kernel_round3_op1_implicitcast.py`（`T.copy` 跨 dtype 错值）；`_engram_gate_conv_fwd_kernel_round6_op1_blk2.py`（`blk_s<halo` 错位）。

### 6.3 Value Point Proposals

| id | vp_type | 标题 | 证据三件套 |
|---|---|---|---|
| **VP-stage4-engram-gcf-vec-load** | **P** | npuir `T.Parallel` 内**元素级 `T.cast` 读 GM**（尤其带运行时 clamp 行索引）被 lower 为嵌套标量 `scf.for`——改「raw dtype 整块 `T.copy`（own + halo 两段）+ 整 tile `T.vcast`」后标量段消除，实测 15.7×–59.4×（3 workload，Ascend910B3/CANN 9.0.0） | evidence：`opt_log.md` §0.2（IR 对照）+ §2 Round 1 + profiles baseline/candidate；repro：`perf_opt/probe/load_variants.py`；toolchain：tilelang `0.1.2+bafbd81b...`（013dbbf）/CANN 9.0.0/910B3 |
| **VP-stage4-engram-gcf-accum-granularity** | **P** | 修正 `BP_scalarized_loop_carried_accum`：`T.Parallel` 内 `+=` 的 lowering **取决于累加粒度**——整 tile 累加 → `hivm.hir.vmul`+`vadd`（已向量化），元素级累加 → 标量 `scf.for`；识别须 dump IR 看 body | evidence：`opt_log.md` §0.2（conv tap 段 IR = `vmul`+`vadd`，证伪 DESIGN §11.2 预注入）；repro：`perf_opt/probe/load_variants.py` variant_a（元素级）+ 本 kernel conv 段（整 tile）；toolchain 同上 |
| **VP-stage4-engram-gcf-blk-halo-guard** | **D** | tensorized 载入（own 精确 region + halo clamp-base 连续拷贝）要求 **`blk_s >= halo`**；`blk_s=2`（原工厂 `seq_len ≡ 2 mod 4` 分支）会静默 halo 错位，且 `S=13` 时 `T.vbrc empty broadcast dims` 编译失败 | evidence：`opt_log.md` §2 Round 6/7 + `S=6/10/14` guard 回归；repro：`_engram_gate_conv_fwd_kernel_round6_op1_blk2.py`（fail）vs `round7_op1_guard.py`（pass）；toolchain 同上 |
| **VP-stage4-engram-gcf-ab-two-kernel-blocked** | **D** | npuir Developer 下单个 `@T.prim_func` 内**两个 `T.Kernel` 块**可编译但 launcher 无法求 grid（`ValueError: Failed to evaluate grid expression 'None'`, `jit_npu.py:1267`）——多 grid 发射的 drop-in 结构不可行 | evidence：`opt_log.md` §2 Round 3（traceback）；repro：`_engram_gate_conv_fwd_kernel_round3_opAB_twopass.py`；toolchain 同上 |
| **VP-stage4-engram-gcf-implicit-copy** | **D** | `T.copy` 跨 dtype（f16 GM→f32 UB）**不做数值转换**，静默产错值（≠`TRAP-C12` "静默转换" 的中性描述；实测 52.5% 元素错）——勿用 `T.copy` 替代 `T.vcast` | evidence：`opt_log.md` §2 Round 3（L0 fail max_abs 2.27）；repro：`_engram_gate_conv_fwd_kernel_round3_op1_implicitcast.py`；toolchain 同上 |

---

## 7. 交付物索引

| 文件 | 说明 |
|---|---|
| `perf_opt/_engram_gate_conv_fwd_kernel.py` | 最优 drop-in 版本（sha256 `a22cd619a17da818c93e577eb101b6620c1884848f3305e7682dd8e0722f0c07`） |
| `perf_opt/opt_log.md` | 本文件 |
| `perf_opt/perf_records.jsonl` | append-only 结构化性能记录（30 行） |
| `perf_opt/workload_inventory.json` | workload 清单（1 smoke + 3 tune） |
| `perf_opt/profiles/{baseline,candidate,final}/` | 各阶段 raw `msprof op` 数据 |
| `perf_opt/logs/` | 实验 stdout/stderr + msprof 日志 |
| `perf_opt/run_experiments.py` / `bench_runner.py` / `ab_interleaved.py` / `extract_metrics.py` | 采数与裁决 runner/helper |
| `perf_opt/probe/load_variants.py` / `dump_ir_*.py` | 载入形态最小对照 + IR dump helper |
| `perf_opt/_engram_gate_conv_fwd_kernel_round{1,2,3,4,5,6,7,8}_*.py` | 各轮实验分支（含 invalid/blocked 证据） |
