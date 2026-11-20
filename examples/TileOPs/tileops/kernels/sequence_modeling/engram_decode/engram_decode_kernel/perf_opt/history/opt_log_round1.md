# `_engram_decode_kernel` Stage 4 调优日志（optimize / mode=full）

- task_id: `engram_decode-_engram_decode_kernel-20261016T104225Z`
- scenario: `optimize` ｜ mode: `full`
- 算子目录: `examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/`
- 基准 kernel（只读，永不修改）: `_engram_decode_kernel.py`
- 最优版本: `perf_opt/_engram_decode_kernel.py`
- 性能口径（唯一）: `msprof op` → `OpBasicInfo*.csv` 的 `Task Duration(us)`（median of 20 launches）
- 性能目标: `best_effort`（无硬性数值门槛）；噪声阈值 3%；最大迭代 10；预算无限制
- 关注 shape（仅影响优先级）: `decode-b4-dmem1024-d512-fp16`
- 环境: Ascend910B3（`npu-smi` 25.5.0，实查 1 die）/ CANN 9.0.0 / tilelang dev root build `0.1.2+76e45509964c313ceca43930cceff43c7820551c (commit 013dbbf)`（`/mnt/workspace/Agentkj/build`，USE_NPUIR=ON）

---

## Phase 0 — 上下文加载与前置核对

### 0.1 算子类型判定：**Vector（纯 AIV），非 MIX**

读 `_engram_decode_kernel.py` 确认：kernel 内**不含 `T.gemm`**（无 Cube 段）；结构为 `T.Kernel(batch, is_npu=True)` 一维 grid（每 batch 一 block）+ `T.alloc_shared`（UB）+ v-prefix 向量链 + `T.reduce_sum` on UB。
`msprof op` 也证实：`Op Type = vector`、`Mix Block Dim = NA`。

> `_engram_decode_kernel_DESIGN.md` §1.6.0 / §3.3 / §10 仍描述 Cube `T.gemm` 主选（R1）；据 `CG-2026-0022`，Stage 3 实际落地的是**向量 FMA 回退形态**。**以 kernel 实际代码为准**，DESIGN 的 Cube 章节仅作历史决策参照（本日志 §2.2 记录了 Cube 路线的重验）。

### 0.2 版本失效核对（E-5）

- `python3 .agents/tools/kb_stale_check.py --json` → `stale_count = 257`，`current_stamp = tilelang 0.1.2+76e4550 + CANN ?`。
- 本任务引用的条目中，绑定本算子的 `TRAP-npuir-vec-scalar-broadcast`、`TRAP-npuir-fp32-vector-reduce`、`TRAP-outidx-call-convention`、`CONST-flag-id-budget` 其工具链戳为 `tilelang 0.1.2+a2357253662 / CANN 9.0.0 / Ascend910`，与本环境 `0.1.2+76e4550 / CANN 9.0.0` 不同 → **标注「待重验」**。
- **本任务实测重验结果（同口径）**：`TRAP-npuir-vec-scalar-broadcast`（裸 out-of-place 标量广播 codegen 段错误 EXIT=139）在 `/tmp/opencode/t8.py` LEVEL0 构造（`acc[0,j] = e_n[0] + w_f[0,j]`，仅 1 个向量操作数）**复现**（进程 exit=139，无诊断）→ 该条目在新工具链下**存活**；`TRAP-npuir-fp32-vector-reduce`／`TRAP-outidx-call-convention` 未单独重验，本轮未作为决策依据。
- `CONST-vector-launch-overhead`（0.5µs/op）与 `CONST-L1-port-bw` 的 staleness 只影响 DESIGN 的 D-2 估算，本轮改用**实测**反解（见 §4），不依赖陈旧常数。

### 0.3 迁移基线「固定台阶」假设的验证/证伪（调度的首要诊断线索）

调度 prompt 与 Stage 5 RETROSPECTIVE 记录「四档 latency 远高于 roofline 且呈固定台阶，疑似 host/调度固定开销主导」。**本轮实测证伪该假设**：

| workload | d_mem | d | 实测 `msprof Task Duration(us)` | 与 d_mem·d 的比例 |
|---|---:|---:|---:|---:|
| decode-smoke-b1-dmem256-d256-fp16 | 256 | 256 | 2996.16（迁移期） | 1.00× |
| decode-b1-dmem512-d256-fp16 | 512 | 256 | **5988.77（本轮复测）** | 1.997× |
| decode-b8-dmem512-d256-bf16 | 512 | 256 | **5988.63（本轮复测）** | 1.997× |
| decode-b4-dmem1024-d512-fp16 | 1024 | 512 | **23668.15（本轮复测）** | 7.90× |

即 latency **严格 ∝ d_mem·d**（batch 1→8 不变、L 9→18 不变）→ 与 host/调度固定开销假设**矛盾**，瓶颈确定在 **GEMV（`O(d_mem)` 串行链，占 manifest FLOPs 99.4%）**。后续 profile 直接证实（§2.1）。

### 0.4 benchmark workload inventory（Phase 1）

`benchmarks/ops/bench_engram_decode.py::test_engram_decode_bench` 经 `_engram_decode_params()` **直读 manifest workloads**（`tileops/manifest/sequence_modeling.yaml::EngramDecodeOp.workloads`），展开 4 个参数化案例（1 smoke + 3 full）：

| kernel_id | workload_id | marks | kind | 排除/参与原因 |
|---|---|---|---|---|
| `..._engram_decode_kernel.py::_engram_decode_kernel::main` | decode-smoke-b1-dmem256-d256-fp16 | `smoke` | **smoke** | label 含独立 `smoke` 词 → 仅精度回归 |
| 同上 | decode-b1-dmem512-d256-fp16 | `full` | **tune** | 进入 baseline/调优/回退门禁 |
| 同上 | decode-b4-dmem1024-d512-fp16 | `full` | **tune** | 关注 shape（绝对耗时最大） |
| 同上 | decode-b8-dmem512-d256-bf16 | `full` | **tune** | 唯一 bf16 workload，进入门禁 |

inventory 落盘：`perf_opt/workload_inventory.json`。无 skipped 案例（benchmark 无 skip 标记）。

---

## 1. Performance Test Data

### 1.1 tune workload baseline（Phase 1，`msprof op`，launch-count=20，warm-up=5）

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | profile_status | raw_profile_dir | command |
|---|---|---|---|---|---:|---|---|---|
| `..._engram_decode_kernel.py::_engram_decode_kernel::main` | decode-b1-dmem512-d256-fp16 | baseline | `main` | `main` | 5988.770 | valid | `profiles/baseline/decode-b1-dmem512-d256-fp16_baseline/OPPROF_20261016190039_PFZHALXZWGSJRBID` | `msprof op --kernel-name=main --output=... --launch-count=20 --warm-up=5 --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio python perf_opt/bench_driver.py --module _engram_decode_kernel.py --case decode-b1-dmem512-d256-fp16` |
| 同上 | decode-b4-dmem1024-d512-fp16 | baseline | `main` | `main` | 23668.151 | valid | `profiles/baseline/decode-b4-dmem1024-d512-fp16_baseline/OPPROF_20261016190228_PXWNPKZETNVMCMGE` | 同上，`--case decode-b4-dmem1024-d512-fp16` |
| 同上 | decode-b8-dmem512-d256-bf16 | baseline | `main` | `main` | 5988.630 | valid | `profiles/baseline/decode-b8-dmem512-d256-bf16_baseline/OPPROF_20261016190417_IKZLYYWOATWMHJYX` | 同上，`--case decode-b8-dmem512-d256-bf16` |

三档 baseline 与迁移期报告 `20261015_063432_610459_EngramDecodeOp`（5988.04 / 23667.86 / 5987.24）**逐档一致（偏差 <0.05%）**，确认测量口径与基线可复现。

### 1.2 smoke 排除

`decode-smoke-b1-dmem256-d256-fp16` 按分类规则归 **smoke**：不采集性能 baseline、不选 winner、不参加性能回退门禁；仅在每次合并后与最终做精度回归（结果见 §5.3，L0 通过）。

### 1.3 调优顺序与理由

1. **`decode-b4-dmem1024-d512-fp16`**（关注 shape；绝对耗时 23668us 最大，仅 4 个 block、单 block 工作量最大，结构性收益最高）→ 先做结构诊断与首个候选。
2. **`decode-b1-dmem512-d256-fp16`**（迭代快、单 block，用于候选分支快速排序与 BK 扫描）。
3. **`decode-b8-dmem512-d256-bf16`**（唯一 bf16 路径 + 最大 batch，验证 fp32 中转与 UB 预算在大 batch 下不回退）。

排序只影响先后，三个 tune workload 全部完成调优与最终复测。

---

## 2. Iteration Log

### Round 1 — 现象、诊断与候选

#### 2.1 Diagnostic Context

- kernel_id: `..._engram_decode_kernel.py::_engram_decode_kernel::main`
- workload_id: decode-b1-dmem512-d256-fp16（首个诊断对象；结论在 b4/b8 上复现）
- current best（本轮 base）: 基准 `_engram_decode_kernel.py`

**Profile Facts（baseline, valid）**

- Task Duration: **5988.77 us**；Block Dim `1`；Op Type `vector`；Current/Rated Freq 1800/1800 MHz
- `aiv_time = 5988.48 us`；`aiv_total_cycles = 10,779,260`
- **`aiv_scalar_ratio = 0.9789`（97.9%）**、`aiv_vec_ratio = 0.00164`（0.16%）、`aiv_mte2_ratio = 0.0226`
- `aiv_scalar_time = 5862.19 us`；`aiv_vec_time = 9.83 us`；`aiv_mte2_time = 135.21 us`
- `aiv_mte2_active_bw = 3.69 GB/s`；`aiv_ub_read_bw_scalar = 1.31 GB/s`；`aiv_ub_read_bw_vector = 0.17 GB/s`
- `read_main_memory_datas = 527.25 KB`（= W_K+W_V+e+h，与理论一致）
- ResourceConflictRatio / L2Cache 无异常

**理论估算**

- W 读流量 @b1 = 2 × 512×256×2 B = 512 KB；实测 527 KB ✓（即**流量口径本身健康，无冗余 GM 往返**）
- 内存地板 ≈ 527 KB / ~1 TB/s ≈ **0.5 µs**；实测 5988.8 µs → **gap ≈ 12000×**
- 向量 op 数 ≈ 50 × 0.5 µs ≈ 25 µs（DESIGN D-2 估算）→ 实测仍高 240×
- Block Dim / AI Core Count = 1 / 40 → 核利用率 2.5%（batch=1 的固有特征）

**Code Structure**

- dispatch: 无（单 kernel，grid = batch）
- 并行粒度: `T.Kernel(batch)`，block 内 `T.serial(d_mem)` 串行 GEMV 链 + `T.Parallel(d_padded)` 逐元素链
- 搬运路径: `T.copy(W_K[i,:] → shared)` 逐行 MTE2；`T.copy` 兼做 fp16→fp32 转换（`TRAP-C12` 用法，基线既有）
- 片上 buffer: 全部 `T.alloc_shared`（UB），~20 个 `(d_padded,)` f32 + GEMV 内 2 个
- 冗余访问: 无中间 GM 往返

**IR 级根因（决定性证据，`TILELANG_DUMP_IR=1`）**

`for i in T.serial(d_mem)` 内的 `for j in T.Parallel(d_padded): k_local[j] += e_local[i] * W_K_local[j]` 被 lower 成**完全标量**的嵌套 `scf.for`：

```
%141 = scf.for %arg22 = 0 to 256 step 1 iter_args(%arg23 = %arg20) {
  %extracted_75 = tensor.extract %arg23[%arg22]   ; 累加器标量读
  %extracted_76 = tensor.extract %56[%arg19]      ; e[i] 标量读（每 lane 重复）
  %extracted_77 = tensor.extract %136[%arg22]     ; W[i,j] 标量读
  %145 = arith.mulf / %146 = arith.addf
  %inserted_78 = tensor.insert %146 ...
}
```

即 **`T.Parallel` 元素级累加（loop-carried 累加器）阻止了 npuir 向量化**：每次 (i,j) 产生 3 次 `tensor.extract` + 1 `mulf` + 1 `addf` + 1 `insert`。b1 规模 = 512(i) × 256(j) × 2(投影) × ~5 标量 op ≈ **1.3 M 标量 op**，与 5862 µs / 41 cycles-per-op 完全吻合。同一 kernel 中**不参与 loop-carried 累加的** `T.Parallel`（如 `hsq_2d[0,j] = h_local[j]*h_local[j]`）均正常 lower 为 `hivm.hir.vmul`/`vbrc` —— 证实瓶颈是**累加形态**而非 API。

#### 2.2 Cube 路线（CG-2026-0022）重验 —— 结论与处置

按调度 prompt §8.1 的强制重验要求执行（工具链已由 `tilelang 0.1.1.10 + CANN 9.0.0` 升级到 `0.1.2+76e4550 + CANN 9.0.0`）：

- **原 repro 脚本缺失**：`examples/engram_decode/_engram_decode_kernel/repro/TRAP-npuir-mix-gemm-vector-reduce.py` 在仓库中**不存在**（该目录仅有 `TRAP-npuir-fp32-vector-reduce.py`、`TRAP-npuir-vec-scalar-broadcast.py`）。本轮自行重建探针：`perf_opt/probe/cg2026_0022_mix_gemm_vector_reduce.py`、`perf_opt/probe/gemv_cube_probe.py`。
- **实测结果（新工具链）**：不含向量消费的 **Cube-only** `T.gemm` GEMV 在把结果从 `c_frag`→`(d,1)` shared→`(d,)` shared 时被拒：`'hivm.hir.copy' op Unsupported copy from cbuf to cbuf!`（**新形态，与 CG 记录的 "cbuf to cc" 不同**）。而含「`T.gemm` GEMV + shared 向量规约 + 标量广播消费」的**完整 MIX 组合**在 `mix_full` 变体下 **`COMPILE_OK`**；但同一结构的另一变体（`T.Pipelined` + fragment 累加 + 向量消费，`/tmp/opencode/t4.py`）在 BiShengIR 阶段 `err code: 1` 失败。**结论：该 MIX 组合在新工具链下仍不稳定（同结构不同 staging 形态结果不一致），不能作为可靠路线**，且复现脚本缺失使原缺口无法逐条同口径比对 → **记为「待重验/证据不足」**，不据其翻转设计，也不产出 `[DESIGN_LIMIT]`。
- **处置**：**Cube GEMV 不纳入本轮候选**。理由不是「缺口仍成立」，而是：(a) 重验证据自相矛盾（不能证明已修复）；(b) 更重要——**向量路线修复后已达 30–78 µs，与 DESIGN D-2 的 25 µs 估算同量级**（§4），Cube 「O(d_mem/BK) 硬件归约」相对此收益 <2×，不满足 [perf-feedback.md](../../../../../../.agents/skills/_shared/standards/perf-feedback.md) §1 的结构性加速门槛。
- 供后续任务参考的**新实证**（回写 pattern-library，见 §6）：`T.vcast` 对 2-D buffer 被前端拒（`load->indices.size() == ndim` InternalError），而 `T.copy` 的隐式跨 dtype 转换对 2-D 可用。

#### 2.3 Current Phenomena

- **P1**：`aiv_scalar_ratio = 97.9%`，`aiv_vec_ratio = 0.16%`；IR 证实 GEMV 累加被 lower 为标量 `scf.for`（§2.1）。
- **P2**：`aiv_time` 与 `d_mem·d` 严格成正比（§0.3），瓶颈唯一定位在 GEMV。
- **P3**：无 GM 冗余流量（527 KB ≈ 理论值）、无非连续搬运、无 tail/空 block；排除布局/搬运类候选。

#### 2.4 Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP1 | P1（每 lane 重复 `tensor.extract e[i]`） | 把标量广播显式化为 `T.vbrc` | IR 中 `e_local[i]` 的 extract 在 j 循环内每 lane 重复；显式 vbrc 提到 j 循环外 | 内层改 `T.vbrc(e_local[i], e_brc)` + `T.vmul` + `T.vadd` | Task Duration / `aiv_scalar_ratio` / L0 | improved（部分） |
| OP2 | P1（loop-carried 元素级累加被标量化） | **K 分块外积 + `T.reduce_sum(dim=0, clear=False)` 硬件归约**，彻底消除 loop-carried 元素级累加 | IR 证明累加形态是根因；`hivm.hir.vreduce` 是向量单元原生归约 | 见 §2.5 代码骨架；引入 `BK` 编译期常量 | Task Duration / `aiv_vec_ratio` / L0 | **improved（胜出）** |
| OP3 | P1 | Cube `T.gemm` GEMV（DESIGN R1） | 设计主选 | 见 §2.2 | — | **blocked/defer**（重验证据自相矛盾；且收益 <2×） |
| OP4 | — | `T.Pipelined(nk, num_stages=2)` 叠加在 OP2 上 | MTE2 ratio 0.66 且无重叠 | 仅把 `T.serial(nk)` 换为 `T.Pipelined(nk,2)` | Task Duration | config_no_gain |
| OP5 | UB 预算 | 乘积原地复用（`w_f[kk,j] *= e_n[kk]`）以放大 BK | 省一半 UB → 允许 BK=32 | 原地 `T.Parallel` 乘 + `reduce_sum(w_f,...)` | Task Duration / UB | config_no_gain（大 shape 回退） |
| OP6 | P1 | BK 扫描（4/8/16/32/64） | BK 为 OP2 唯一关键参数 | 仅改 `BK` | Task Duration | **improved（BK=16 胜出）** |

#### 2.5 OP2 关键代码更改（慢 → 快最小形态）

```python
# BEFORE (baseline): T.Parallel 元素级累加 -> 标量 scf.for（97.9% scalar）
k_local = T.alloc_shared((d_padded,), accum_dtype)
T.clear(k_local)
for i in T.serial(d_mem):
    W_K_local = T.alloc_shared((d_padded,), accum_dtype)
    T.copy(W_K[i, :], W_K_local)                    # 隐式 fp16->fp32
    for j in T.Parallel(d_padded):
        k_local[j] += e_local[i] * W_K_local[j]     # <-- 标量化根因

# AFTER (winner, BK=16): K 分块外积 + 硬件归约，无 loop-carried 元素级累加
nk = d_mem // BK
w_f = T.alloc_shared((BK, d_padded), accum_dtype)
e_n = T.alloc_shared((BK,), accum_dtype)
prod = T.alloc_shared((BK, d_padded), accum_dtype)
acc_k = T.alloc_shared((1, d_padded), accum_dtype)
T.clear(acc_k)
for k0 in T.serial(nk):
    T.copy(e_t[bid, k0 * BK : (k0 + 1) * BK], e_n)
    T.copy(W_K[k0 * BK : (k0 + 1) * BK, :], w_f)    # 2-D 隐式 dtype 转换 copy
    for kk, j in T.Parallel(BK, d_padded):
        prod[kk, j] = w_f[kk, j] * e_n[kk]          # 行广播（非累加 -> 可向量化）
    T.reduce_sum(prod, acc_k, dim=0, clear=False)   # -> hivm.hir.vreduce
```

机制归因：npuir 向量化器对 `T.Parallel` 内**写回同一 buffer 的元素级累加**（loop-carried `iter_args`）退化为标量 `scf.for`；改成「不累加的乘积 tile + `vreduce` 累加」后，乘积走 `vmul`、累加走 `vreduce`，标量段消失（`ub_read_bw_vector` 从 0.17 → 153.9 GB/s）。

#### 2.6 实验分支与「候选 vs current best」对比表（Round 1）

base（current best）= 基准 `_engram_decode_kernel.py`。所有分支从同一 base 派生，每分支只改一个主要优化点。

| branch | opt_id | 主要改动 | L0 | Task Duration(us) b1 | b4 | b8 | AICore 利用率（b1 vec/mte2/scalar ratio） | memory 指标 | 结论 |
|---|---|---|---|---:|---:|---:|---|---|---|
| `perf_opt/_engram_decode_kernel_round1_op1.py` | OP1 | `vbrc` 显式广播 | pass | **3487.90** | — | — | 0.0084 / 0.0388 / 0.9569 | GM_to_UB 527 KB（不变） | improved（部分，1.72×） |
| `perf_opt/_engram_decode_kernel_round1_op2e.py` | OP2+OP6 | K 分块 vreduce，BK=8 | pass | **34.59** | **86.07** | — | — | GM_to_UB 527 KB | improved（173×/275×） |
| `perf_opt/_engram_decode_kernel_round1_op2f.py` | OP2+OP6 | K 分块 vreduce，**BK=16** | pass | **30.10** | **78.56** | **31.39** | 0.5729 / 0.6763 / 0.6419 | GM_to_UB 527 KB | **improved（199×/301×/191×）← winner** |
| `perf_opt/_engram_decode_kernel_round1_op2g.py` | OP6 | BK=32（非原地） | **fail** | 28.16 | invalid | 29.57 | — | **UB overflow**：d=512 档 `requires 1606656 bits while 1572864 bits available` | blocked（UB） |
| `perf_opt/_engram_decode_kernel_round1_op2k.py` | OP4 | BK=16 + `T.Pipelined(2)` | pass | 30.18 | 78.43 | — | — | 同 op2f | config_no_gain（与 op2f 打平，<1%，噪声内） |
| `perf_opt/_engram_decode_kernel_round1_op2l.py` | OP5 | BK=32 + 原地乘积 | pass | 28.78 | **99.23** | 30.20 | — | w_f 64 KB（省一半） | config_no_gain（大 shape −26% 回退） |
| `perf_opt/_engram_decode_kernel_round1_op2m.py` | OP5+OP6 | BK=64 + 原地乘积 | **fail** | 32.58 | — | — | — | UB overflow | blocked（UB） |
| `perf_opt/_engram_decode_kernel_round1_op2.py`、`_op2b/c/d.py` | OP2 早期形态 | (2-D `T.vcast` / `(BK,1)` vcast / 非原地 vadd) | — | 未编译 | 未编译 | 未编译 | — | — | invalid（前端 `load->indices.size() == ndim` InternalError，见 §2.2 末） |

> **本轮 winner 判定**：`decode-b4`（关注 shape）上 op2f=78.56 µs 显著优于 op2l=99.23 µs（−21%）且 op2g/BK=32 溢出不可用；`decode-b1/b8` 上 op2g/op2l 名义更快（28.2/28.8 vs 30.1），但相对 base 已是 199× 量级、差异 4–7%，而**同一 `BK` 常量须服务全部 shape**（不允许在 Python factory 内做按 shape 性能分派）→ 取 **BK=16（op2f）为全局 winner**。
> 落盘：`TUNED_BK = 16` 模块级常量 + 工厂默认 `BK=TUNED_BK`，供 wrapper 切换块成对引用。

#### 2.7 核内合并检查（Round 1）

winner 的改动是**无条件结构性替换**（不引入任何 shape/dtype 条件分支），直接合入 `perf_opt/_engram_decode_kernel.py`，并在合并版上复测**本 kernel 全部 tune workload + smoke**：

| merged_candidate | kernel_id | workload_id | current_best_us | merged_us | precision | status |
|---|---|---|---:|---:|---|---|
| `perf_opt/_engram_decode_kernel.py`（= op2f） | `...::_engram_decode_kernel::main` | decode-b1-dmem512-d256-fp16 | 5988.77 | **30.13** | pass (L0) | pass |
| 同上 | 同上 | decode-b4-dmem1024-d512-fp16 | 23668.15 | **78.42** | pass (L0) | pass |
| 同上 | 同上 | decode-b8-dmem512-d256-bf16 | 5988.63 | **31.35** | pass (L0) | pass |
| 同上 | 同上 | decode-smoke-b1-dmem256-d256-fp16（smoke，仅精度） | — | 不采集 | pass (L0) | pass |

无回退 → 更新已合并 current best 为 `perf_opt/_engram_decode_kernel.py`。

#### 2.8 Iteration Winner

- winner: `perf_opt/_engram_decode_kernel.py`（候选来源 `_engram_decode_kernel_round1_op2f.py`，candidate_id `r1_op2f`）
- reason: 三个 tune workload 的 `Task Duration` 全部最优（b4 显著领先，b1/b8 与名义最优差 4–7% 且同 BK 约束下不可分派），且合并版在本 kernel 全部 tune workload 与 smoke 上精度通过、性能无回退
- new current best: `perf_opt/_engram_decode_kernel.py`
- rollback / 未采纳: op1（仅部分改善）、op2e（BK=8 次优）、op2g/op2m（UB overflow）、op2k（打平）、op2l（大 shape 回退）、op2/op2b/op2c/op2d（前端不可编译）

#### 2.9 终止条件判定

- Round 1 已达到 199×–301× 量级提升，全部 tune workload 均有结论并完成全量复测；**无 tune workload 处于 `merge_blocked`**。
- 结构性候选已充分验证：OP1/OP2/OP3/OP4/OP5/OP6 六个方向各有 valid/blocked/no_gain 结论；Cube 路线经重验后因证据矛盾 + 收益 <2× 明确弃选。
- 参照结构 diff 检查（T-1 morph ladder 轻量版）：同族参照 `examples/engram/engram_decode.py`（源串行 GEMV 基线，本已复刻）、`examples/norm/rms_norm.py`（单遍行归约）、`examples/gemv/example_gemv.py`（Cube GEMV，不可用）。已识别且已消费的结构差异 = **「元素级累加 → vreduce」**，无进一步未消费的结构差异。
- 中止原因：**success/best_effort 收束**（1 轮内完成结构性主优化与参数扫描；剩余差异均 <3% 噪声阈值或受 UB 硬约束）。无 `[DESIGN_LIMIT]`（见 §4）。

---

## 3. Autotune Log

未使用 autotune 框架；OP2/OP6 的关键参数 `BK` 采用手动粗搜 + winner 邻域（等价于 coarse search），`T.Pipelined` 深度手动扫：

| 结构 | 搜索空间 | 实测 best | 备注 |
|---|---|---|---|
| K 分块 + vreduce | BK ∈ {4, 8, 16, 32, 64} × {原地, 非原地} × {serial, Pipelined(2)} | **BK=16, 非原地, serial** | BK=32/64 非原地 → UB overflow（d=512）；原地大 shape 回退；Pipelined(2) 打平 |
| BK=8 邻域 | 8 | 34.59（b1）/ 86.07（b4） | 次优 |

winner（BK=16）的 `msprof op` 复测即 §2.6 的 op2f 行与 §5.1 的 final 行（两次独立采集 30.10 / 30.13，偏差 0.1%，落在噪声内）。

---

## 4. 设计估算 vs 实测（D-2 回填）

`DESIGN.md` §1.6.0 给出 D-2 roofline 估算（发射项主导）：W3（B4,d_mem1024,d512）下界 ≈ **25 µs**；W1 ≈ 12 µs。

| workload | 设计估算下界 | baseline 实测 | 偏差 | 修复后实测 | 与估算偏差 |
|---|---:|---:|---:|---:|---:|
| decode-b1-dmem512-d256-fp16 | ≈12 µs | 5988.77 µs | **499×** | 30.13 µs | 2.5× |
| decode-b4-dmem1024-d512-fp16 | ≈25 µs | 23668.15 µs | **947×** | 78.42 µs | 3.1× |

**失准项归因**：设计 D-2 的「发射项 = 向量 op 数 × 0.5 µs」假设**没有预见到 GEMV 会被 lower 成标量 `scf.for`**（发射项被低估 ~1000×，且实际瓶颈是标量执行而非发射）。修复后实测收敛到估算的 **2.5–3.1×** 以内，说明 D-2 对「一个**向量化正确**的实现」的估算量级是可信的（误差来自 `T.copy` 的 fp16→fp32 中转流量与 MTE2 小块开销，未计入 D-2）。

**`[DESIGN_LIMIT]` 判定（`.agents/skills/_shared/standards/perf-feedback.md` §1，两项须同时满足）**：

1. 「性能天花板由算法/设计层决定」→ **不满足**。本轮已把主瓶颈（GEMV 标量化）在**不改变算法族**（仍是 fp32 向量 FMA/vreduce）的前提下消除；剩余 2.5–3.1× 与估算的差距归因于 kernel 级工程项（T.copy 中转 dtype、MTE2 小块开销、block 数=batch 导致低核利用），属 tiling/搬运可继续优化范畴，非设计假设错误。
2. 「结构性加速估计 >2×」→ **不满足**。唯一潜在的结构性替代是 Cube GEMV，其相对当前 30–78 µs 的估计收益 <2×（且经重验证据矛盾）。

→ **不产出 `perf_opt/perf_feedback.md`；不返回 `[DESIGN_LIMIT]`。**

> 附带的设计层事实（供 conductor 参考，非 `[DESIGN_LIMIT]`）：DESIGN §1.6.3 候选 B 的结论「向量分块 GEMV 性能远劣（op 数 768，发射下界 ~0.38 ms）」被本轮实测**证伪**——同一算法的正确实现（K 分块 + vreduce）实测 30–78 µs，比其估算快 5–12×。差异根因：估算按「每 op 0.5 µs 串行发射」计，忽略了外积的 `vmul`/`vreduce` 在向量单元上可整块执行。

---

## 5. Final Summary

### 5.1 相对 baseline 的改进（`msprof op Task Duration`）

| workload_id | baseline_us | final_us | 加速比 | 降幅 |
|---|---:|---:|---:|---:|
| decode-b1-dmem512-d256-fp16 | 5988.770 | 30.130 | 198.8x | -99.50% |
| decode-b4-dmem1024-d512-fp16 | 23668.151 | 78.420 | 301.8x | -99.67% |
| decode-b8-dmem512-d256-bf16 | 5988.630 | 31.350 | 191.0x | -99.48% |

## Final Performance Test Data

`kernel_id = examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/_engram_decode_kernel.py::_engram_decode_kernel::main`；`final_candidate_id = r1_op2f`；final 时延在最终文件 `perf_opt/_engram_decode_kernel.py`（sha256 `d30e11a98c225aa08b532ce9e95017e8055628c1de01c33f185635baa5d7a234`）上重新采集，launch-count=20、warm-up=5。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/_engram_decode_kernel.py::_engram_decode_kernel::main | decode-b1-dmem512-d256-fp16 | 5988.770 | 30.130 | r1_op2f |
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/_engram_decode_kernel.py::_engram_decode_kernel::main | decode-b4-dmem1024-d512-fp16 | 23668.151 | 78.420 | r1_op2f |
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/_engram_decode_kernel.py::_engram_decode_kernel::main | decode-b8-dmem512-d256-bf16 | 5988.630 | 31.350 | r1_op2f |

### 5.2 profile 证据（final，`profiles/final/`）

| workload | global | raw_profile_dir | `aiv_vec_ratio` | `aiv_mte2_ratio` | `aiv_scalar_ratio` | `ub_read_bw_vector` | Block Dim |
|---|---|---:|---:|---:|---:|---:|
| decode-b1-dmem512-d256-fp16 | valid | `profiles/final/decode-b1-dmem512-d256-fp16_final/OPPROF_20261016202433_RAKEQXGGLPHELNUQ` | 0.573 | 0.676 | 0.642 | 153.9 GB/s | 1 |
| decode-b4-dmem1024-d512-fp16 | valid | `profiles/final/decode-b4-dmem1024-d512-fp16_final/OPPROF_20261016202538_PITBIHRZJBYNSWVQ` | 0.726 | 0.737 | 0.523 | 240.4 GB/s | 4 |
| decode-b8-dmem512-d256-bf16 | valid | `profiles/final/decode-b8-dmem512-d256-bf16_final/OPPROF_20261016202642_JGGPGEJDTYKABICV` | — | — | — | — | 8 |

（b1 baseline 对照：`vec_ratio` 0.0016 / `mte2_ratio` 0.0226 / `scalar_ratio` 0.9789 / `ub_read_bw_vector` 0.17 GB/s。final 的三项 ratio 已重叠（>0.5 且和 >1），说明 MTE2/VEC 真正并行工作；剩余标量占比为循环/地址计算，非数据路径。）

### 5.3 smoke / skip 排除与精度

| workload | kind | 处理 | 最终精度 |
|---|---|---|---|
| decode-smoke-b1-dmem256-d256-fp16 | smoke | 仅精度回归（不采集性能、不选 winner、不进门禁） | **pass**：`y_maxdiff=2.441e-04, st_maxdiff=1.953e-03`（与 baseline 逐位一致） |

无 skipped 案例。

### 5.4 最终精度回归（`perf_opt/_engram_decode_kernel.py --level all`）

- **L0（阻塞）PASS**：4/4 manifest 用例 max diff 与 baseline **逐位一致**（2.441e-04 / 1.221e-04 / 9.766e-04 / 7.812e-03）
- **L1（阻塞）PASS**：`L1-multistep`(17 步) / `L1-bf16` / `state-shift bit-exact (0.0e+00)` / `gate-corner finite` / `L1-dil3` / `L1-dil5` / `L1-seed5` 全过
- L2（告警不阻塞）：`L2-fp32` 告警（**继承 baseline 的既有 fp32 GM 输入不可编译限制**，非本轮回退；`TRAP-npuir-fp32-vector-reduce`）；`L2-seed101/202` 通过
- Boundary（告警不阻塞）：`B-minL` / `B-w2` / `B-w1` / `B-b1` 全过
- 日志：`perf_opt/logs/final_L_all.log`、`perf_opt/logs/round1_*_L0.log`

### 5.5 总体结论与中止原因

- 单一结构性根因（`T.Parallel` 元素级 loop-carried 累加被 lower 为标量 `scf.for`）占据 baseline 97.9% 的执行时间；改为 K 分块外积 + `vreduce` 硬件归约后，三个 tune workload 分别提速 **198.8× / 301.8× / 191.0×**。
- 中止原因：**success（best_effort 收束）**——结构性主优化完成、关键参数（BK/pipeline/原地）已扫描、全量 tune workload 有结论并复测、smoke 精度通过。
- 未采纳 / 受限项：Cube GEMV（CG-2026-0022 重验证据矛盾，且收益 <2×）；BK>16（UB 硬上限）；原地乘积（大 shape 回退）；Pipelined（打平）。
- 遗留（非阻塞，交后续）：剩余 2.5–3.1× 与 D-2 估算的差距主要来自 (a) `T.copy` 的 fp16→fp32 中转使 UB 写入量翻倍、(b) MTE2 小块搬运开销（24–35 GB/s 有效带宽）、(c) block 数 = batch（b1/b8 仅 1/8 个 block）导致核利用率低——后者需 split-K + 跨核归约（结构级改动）。

---

## 6. Skill Retrospective

### 6.1 Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Optimize-flow | **「98% scalar_ratio」缺一条直接的诊断脚本入口**：本算子从 baseline 到 IR 级根因（`T.Parallel` 元素级累加 → 标量 `scf.for`）需要人工 `TILELANG_DUMP_IR=1` + 读 1171 行 MLIR；skill 只给了「标量占比 >50% 触发换轴分析」的方向，无「dump IR 并定位标量化指令形态」的操作步骤 | 本轮 §2.1；`TILELANG_DUMP_IR=1` 输出中被标量的 `scf.for`/`tensor.extract` 形态 | `tilelang-op-optimize/references/iteration-diagnosis.md` Step 2 增补操作条：「`aiv_scalar_ratio > 50%` 时先 `TILELANG_DUMP_IR=1` 编译最小 shape，grep `scf.for` + `tensor.extract` 定位被标量化的 `T.Parallel`，判定其是否为 loop-carried 累加」 | R |
| Optimize-flow | **micro-benchmark 探针在本工具链上比全 kernel 更易触发 codegen 段错误**，导致「先做最小复现隔离」的常规手法反而更慢（本轮 6 次 micro 探针中 3 次 EXIT=139、3 次前端 InternalError，而同一改动放进全 kernel 一次通过） | `/tmp/opencode/t8.py`（LEVEL0 即段错误）、`t7.py`（6/6 全 fail，harness 自身可疑）；对照：`perf_opt/_engram_decode_kernel_round1_op2f.py` 首编即过 | `tilelang-op-optimize` Phase 2 增补：「npuir 下 micro-benchmark 探针的可用性低于全 kernel 上下文（标量广播段错误 / 前端 InternalError 在裸 harness 中更易触发）；优先在真实 kernel 内做单变量分支」 | R |
| Kernel-flow | `T.vcast` **不支持 2-D buffer**（`load->indices.size() == ndim` InternalError），skill/pattern-library 无此条；`TRAP-npuir-vcast-shape-mismatch` 只记了 shape 必须相同 | `/tmp/opencode/t8.py` LEVEL2 单变量隔离；绕法 `T.copy` 隐式跨 dtype 或逐行 1-D `vcast` 均可用（`t9.py` 三形态全 OK） | `pattern-library/traps-compiler.md` 新增条目（含 repro） | D |

### 6.2 Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **npuir 下 `T.Parallel` 内写回同一 buffer 的元素级累加（loop-carried `iter_args`）被 lower 为完全标量的 `scf.for`（`tensor.extract`/`mulf`/`addf`/`tensor.insert`）**；改为「不累加的乘积 tile + `T.reduce_sum(dim=0, clear=False)`（`hivm.hir.vreduce`）」后标量段消失，本算子三个 workload 提速 191–302×（`aiv_scalar_ratio` 0.979→0.642，`ub_read_bw_vector` 0.17→153.9 GB/s）。识别信号：`aiv_scalar_ratio > 0.9` 且 `aiv_vec_ratio < 0.01` | P | 本轮 §2.1（`TILELANG_DUMP_IR=1` MLIR 证据 + `msprof` 指标）；`perf_opt/profiles/baseline/...` vs `perf_opt/profiles/final/...` | `perf_opt/probe/gemv_variants.py`（慢/快对照：`gv0` 累加形态 vs K 分块 vreduce）；最小 delta 见 `opt_log.md` §2.5 | tilelang 0.1.2+76e4550（commit 013dbbf） / CANN 9.0.0 / Ascend910B3 / 2026-10-16 | `pattern-library/elementwise.md` 或 `layout.md`（新增「loop-carried 累加 → vreduce 消除」模式条） |
| **`T.vcast` 对 2-D buffer 不可用**（前端 `InternalError: Check failed: (load->indices.size() == ndim)`）；绕法 = `T.copy` 隐式跨 dtype 转换，或逐行 1-D `T.vcast`（`w_h[kk,:] → w_f[kk,:]`） | D | `/tmp/opencode/t8.py` LEVEL2 单变量（仅 2-D vcast 失败，1-D vcast 通过）；`t9.py` 三绕法全 OK | `perf_opt/probe/cg2026_0022_mix_gemm_vector_reduce.py` 同族探针（vcast 形态） | tilelang 0.1.2+76e4550 / CANN 9.0.0 / Ascend910B3 / 2026-10-16 | `pattern-library/traps-compiler.md`（新条目，与 `TRAP-npuir-vcast-shape-mismatch` 交叉引用） |
| **DESIGN 的「向量分块 GEMV 性能远劣（~0.38 ms）」结论被实测证伪**：同一算法族（K 分块外积 + 归约）的正确向量实现实测 **30–78 µs**，比设计估算快 5–12×；根因是估算按「每 op 0.5 µs 串行发射」计，忽略了 `vmul`/`vreduce` 在向量单元上可整块执行 | D | 本轮 §4；`DESIGN.md` §1.6.3 候选 B 行 vs `perf_opt/profiles/final/*` | `perf_opt/_engram_decode_kernel_round1_op2f.py` | tilelang 0.1.2+76e4550 / CANN 9.0.0 / Ascend910B3 / 2026-10-16 | `pattern-library/constants.md`（修正「向量 op 发射成本」在 **成块 `vmul`/`vreduce`** 下的口径；`CONST-vector-launch-overhead` 0.5 µs/op 仅适用于逐 op 串行小向量链） |
| CG-2026-0022 重验未收敛：新工具链下 `T.gemm` GEMV + shared 向量规约消费的 MIX **部分形态可编译**（`mix_full` COMPILE_OK），但同结构另一 staging 形态仍 BiShengIR 失败；且原 repro 脚本在仓库中**缺失** | C | 本轮 §2.2；`perf_opt/probe/cg2026_0022_mix_gemm_vector_reduce.py`、`gemv_cube_probe.py` | 同上 | tilelang 0.1.2+76e4550 / CANN 9.0.0 / Ascend910B3 / 2026-10-16 | `.agents/evolution/capability-gaps.md` CG-2026-0022（补 versions 记录 + 标注「repro 脚本缺失，需重建」；`occurrences` 不增量——本轮未得出确定性结论） |

### 6.3 Transferable Lessons

- **`aiv_scalar_ratio > 90%` 应首先怀疑 `T.Parallel` 元素级累加被标量化**，而不是「标量开销大」；用 `TILELANG_DUMP_IR=1` grep `tensor.extract` + `scf.for` 可 5 分钟内定位。同一 kernel 内**不累加**的 `T.Parallel` 会正常向量化，这是最强对照。
- **把「归约」交给 `T.reduce_sum(..., clear=False)`**（→ `hivm.hir.vreduce`）而不是手写元素级 `+=`：前者是向量单元原生带累加归约，后者退化为标量链。
- **npuir 下 micro-benchmark 探针的可靠性低于全 kernel 上下文**：bare harness 更易触发标量广播段错误（EXIT=139）与前端 InternalError；隔离单变量时优先在真实 kernel 内做，且每次只改一处。
- **同一 kernel 的 `BK` 常量不能做 Python 级按 shape 分派**（skill 约束）：因此参数扫描要按「全局最劣 shape 优先」取舍——本轮 b1/b8 上 BK=32 名义快 4–7%，但 b4 上不合法（UB overflow）且原地变体在 b4 回退 26%，故取 BK=16 全局最优。
- **D-2 估算失准的典型形态**：估算「发射项主导」而实测「标量执行主导」时，偏差可达 ~1000×；修复后若能收敛到估算的 2–3× 以内，说明估算方法本身可继续用（误差来自未计入的 dtype 中转与小块搬运）。

### 6.4 本轮未执行/未回写项

- **pattern-library 回写**：本轮尝试向 `pattern-library/traps-compiler.md`（2-D `T.vcast`）与 `elementwise.md`/`layout.md`（loop-carried 累加 → vreduce）回写条目，但**该目录属 skill 资源目录**；按本 Agent 的只读约束（只写 `perf_opt/`），本轮把条目内容与 repro 位置完整记录于本节，交任务终态 `tilelang-skill-evolver` 蒸馏合入。
- **CG-2026-0022 的 `occurrences` 未增量**：本轮未得出「仍成立/已修复」的确定结论（证据自相矛盾 + 原 repro 缺失），按 `AGENTS.md` 编译器能力缺口规则第 5 条应先在统一 repro 下复验再决定，故不改写 gap 条目，仅记录。
- **未产出 `perf_opt/perf_feedback.md`**：§4 两项触发条件均不满足。

---

## 7. 工件清单

| 文件 | 说明 |
|---|---|
| `perf_opt/_engram_decode_kernel.py` | **最优版本**（drop-in，同工厂签名；`TUNED_BK = 16`） |
| `perf_opt/opt_log.md` | 本文件 |
| `perf_opt/perf_records.jsonl` | 17 行（baseline 3 + candidate 11 + final 3），append-only |
| `perf_opt/workload_inventory.json` | 4 workloads（1 smoke + 3 tune） |
| `perf_opt/bench_driver.py` | 单 workload 启动器（msprof op 口径） |
| `perf_opt/run_prof.sh` | msprof op 采集 + OpBasicInfo 解析封装 |
| `perf_opt/probe/` | CG-2026-0022 重验探针（`cg2026_0022_mix_gemm_vector_reduce.py`、`gemv_cube_probe.py`、`gemv_variants.py`） |
| `perf_opt/_engram_decode_kernel_round1_op*.py` | 实验分支（op1/op2e/op2f/op2g/op2k/op2l/op2m 及早期形态） |
| `perf_opt/profiles/` | raw `msprof op` 数据（baseline / round1 / final） |
| `perf_opt/logs/` | 实验 stdout/stderr |
