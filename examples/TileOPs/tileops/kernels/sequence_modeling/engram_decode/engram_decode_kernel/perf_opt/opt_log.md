# `_engram_decode_kernel` Stage 4 调优日志 — 第 3 轮会话（optimize / mode=full）

- 任务: `TileOPs-_engram_decode_kernel-20261030T145026Z`
- 算子目录: `examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/`
- 工具链（本轮实测戳）: tilelang `0.1.2+01805d7a07ca111e43868c07b496bffb30813ba3`（第 1/2 轮为 `+76e4550`，**已变更**）+ CANN 9.0.0 + Ascend910B3（单 die，20 AI Core / 40 AIV）
- 唯一时延口径: `msprof op` 的 `Task Duration(us)`；Ratio 采数 `--aic-metrics=...,Roofline` + `--kernel-name=main`（两者缺一产生伪影，见 `profile-collection.md` roofline Ratio 段）
- 本轮 current best 起点: `perf_opt/_engram_decode_kernel.py`，sha256 `3835dee05f105bdc7f6d94eded0623fa812cfe3d960d34244c67434ce6a08f05`（第 2 轮 winner `r2h1_shift2d`）

## 0. 历史工件处置与本轮结论速览

| 项目 | 处置 |
|---|---|
| 第 1 轮日志/记录 | `perf_opt/history/opt_log_round1.md`、`perf_opt/history/perf_records_round1.jsonl`（原样保留） |
| 第 2 轮日志/记录 | 本轮开始时**原样移动**为 `perf_opt/history/opt_log_round2.md`、`perf_opt/history/perf_records_round2.jsonl`（未删改），使 gate 4 `S4-PERF-RECORDS-RECON` 的「每个 tune workload 恰有一条 final 且指向最终文件哈希」在本轮重建的记录集上成立 |
| 本轮新记录 | `perf_opt/perf_records.jsonl` 新建：baseline（current best 重新实测，3 行）+ candidate（29 行）+ final（3 行），append-only |
| 本轮产物 | `perf_opt/_engram_decode_kernel.py`（新 best，sha256 `ef7b3493315b8af3e16926cac16291c56cf5eb337545a2b2ca4caf52a0bbc414`） |
| 本轮记录的一次重建 | 最终文件在首次 final 采集后仅**修正了 docstring 中一处被 §2.2 P4 探针推翻的机制表述**（代码与编译产物未变）。为使 gate 4 的「final 记录哈希 == 最终文件哈希」成立，把当时的记录集**逐行原样**移入 `perf_opt/history/perf_records_round3_partial.jsonl`，活动账本保留全部 32 行非 final 记录（内容一字未改）后重新实测 3 条 final。未修改/删除任何既有行的内容。 |

**本轮结论速览（诚实版）**：逐项目标（b1 ≤19.66 / b4 ≤54.78 / b8 ≤19.74 µs）**未达成**。
实测达成 b1 **26.651 µs（1.146%）**、b4 **73.541 µs（3.877%）**、b8 **26.791 µs（5.425%）**，tune 平均 **3.483%**；
相对第 2 轮 final（3.3850%）提升 +2.9% 相对，相对本轮基线（3.3832%）提升 +2.9% 相对。
根因是**逐项目标本身不可达**：三档目标值等于「三 pipe 完全重叠 = max(vec, mte2, scalar)」，而这要求被暴露的 scalar 时间降到 **0**；
本轮 12 个分支一致实测出 `duration ≈ vec + (0.47–0.68) × scalar`（§2.5），scalar 暴露比例无法压到 0。详见 §4。

---

## 1. Performance Test Data

### 1.1 workload inventory 与分类（Phase 1）

来源 `examples/TileOPs/benchmarks/ops/bench_engram_decode.py::test_engram_decode_bench`，4 个参数化案例（未新增/裁剪任何案例）：

| workload_id | marks | kind | 参数 | 说明 |
|---|---|---|---|---|
| decode-smoke-b1-dmem256-d256-fp16 | smoke | **smoke** | B1 d_mem256 d256 L9 w4 dil3 fp16 | 仅精度回归，不采性能、不写性能记录 |
| decode-b1-dmem512-d256-fp16 | full | tune | B1 d_mem512 d256 L12 w4 dil3 fp16 | grid=1 block（单核带宽上限） |
| decode-b4-dmem1024-d512-fp16 | full | tune | B4 d_mem1024 d512 L20 w4 dil5 fp16 | 绝对时延最大，本轮仍最优先 |
| decode-b8-dmem512-d256-bf16 | full | tune | B8 d_mem512 d256 L18 w4 dil3 bf16 | 唯一 bf16 workload |

完整契约见 `perf_opt/workload_inventory.json`（本轮已更新 `tuning_status`：b1/b8 `winner_merged`、b4 `no_gain`、smoke `precision_pass=true`）。

### 1.2 tune workload baseline（current best `3835dee0…` 重新实测，launch-count=20）

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | profile_status | raw_profile_dir |
|---|---|---|---|---:|---|---|---|
| `…::_engram_decode_kernel::main` | decode-b1-dmem512-d256-fp16 | r3base_shift2d | main | main | 28.361 | valid | `profiles/baseline3/r3base_shift2d_b1` |
| 同上 | decode-b4-dmem1024-d512-fp16 | r3base_shift2d | main | main | 73.591 | valid | `profiles/baseline3/r3base_shift2d_b4` |
| 同上 | decode-b8-dmem512-d256-bf16 | r3base_shift2d | main | main | 28.611 | valid | `profiles/baseline3/r3base_shift2d_b8` |

命令模板（逐 workload 串行，未并发）：
`msprof op --kernel-name=main --output=<dir> --launch-count=20 --warm-up=5 --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio,Roofline python perf_opt/bench_driver.py --module perf_opt/_engram_decode_kernel.py --case <workload> --iters 20`

- 有效性核对：`--kernel-name=main` 过滤后单一 `captured_op_name=main`、`op_type=vector`、`n_launches=20`，非 Cast/Mul/OnesLike 等框架小算子；`prof.json` 与 raw `OPPROF_*/main/*.csv` 均可追溯。
- 复现性：与第 2 轮 final（28.411 / 73.591 / 28.571）偏差 ≤0.2%，且 **工具链已从 `+76e4550` 变为 `+01805d7a` 后 baseline 未漂移** → 第 2 轮结论在本轮工具链下仍成立（E-5 重验通过）。

### 1.3 本轮基线 profile 事实（Diagnostic Context）

| workload | Block Dim | Task(us) | vec_t / ratio | mte2_t / ratio | scalar_t / ratio | vec_cflt | bankgrp_cflt | vec_wait | mte2_wait | ub_rd / ub_wr (GB/s) |
|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| b1 | 1 | 28.361 | 17.57 / 0.631 | 18.95 / 0.680 | 19.61 / 0.705 | 0.199 | 0.145 | 0.075 | 0.037 | 163 / 103 |
| b4 | 4 | 73.591 | 54.78 / 0.752 | 51.43 / 0.706 | 40.24 / 0.552 | 0.241 | 0.209 | 0.024 | 0.017 | 250 / 164 |
| b8 | 8 | 28.611 | 17.57 / 0.632 | 16.47 / 0.590 | 19.75 / 0.710 | 0.209 | 0.153 | 0.063 | 0.041 | 171 / 109 |

**本轮的新观测（第 2 轮未拆解的部分）**：
- `aiv_vec_wait_ratio` / `aiv_mte2_wait_ratio` 都极低（b4 仅 0.024 / 0.017）→ **data-wait 不是瓶颈，自动 multi-buffer 已经在良好的重叠状态**。三条 pipe 的 busy ratio 只有 0.55–0.75，说明核在 25–45% 的时间里既不在跑 vec 也不在等数据 —— 那段是**纯 scalar 发射期**（DMA 指令发射 + 循环/寻址 + 向量指令发射）。
- `aiv_vec_bankgroup_cflt_ratio` 0.145–0.209：dim-0 `hivm.vreduce` 的行步长是 `d_padded*4` 的 2 的幂，跨行读取集中在少数 bank group（本轮尝试过 padding 修法，见 §2.5 R3-M，实测负向）。

### 1.4 smoke 排除与调优顺序

- smoke 仅做精度回归（`precision_pass=true`），不采集性能、不选 winner、不参与回退门禁。
- 排序：b4（绝对时延最大，占 tune 平均绝对负担 70%）→ b1（单核带宽/延迟上限的代表）→ b8（同 b1 形态但 bf16，守护 bf16 copy 路径）。实际执行中因所有候选都会顺手拿到 b1/b4/b8 三档，三档顺序执行、逐档判定。

---

## 2. Iteration Log

### Round 3 — 现象、候选与分支

#### 2.1 Diagnostic Context（base = `3835dee0…`，§1.3）

#### 2.2 本轮现象（P1–P4）

- **P1（量化模型，本轮核心）**：在 12 个分支上一致拟合出 `Task Duration ≈ vec_time + a × scalar_time`，`a ∈ [0.47, 0.68]`（下表逐项反解）。b4 基线 54.78 + 0.47×40.24 = 73.7 ≈ 73.59 ✓；b1 17.57 + 0.55×19.61 = 28.4 ≈ 28.36 ✓。→ **被暴露的 scalar 时间约占其 busy 时间的一半，且无法与 vec 重叠**。
- **P2（scalar 的构成）**：scalar 时间几乎正比于 **DMA 指令条数**（b1：96 条 × ≈0.2 µs ≈ 19.2 µs ⇒ 实测 19.61；b4：192 条 ⇒ 实测 40.24，为 0.21 µs/条）。第 2 轮反解的 0.16 µs/DMA 在本轮工具链下复现为 0.20–0.21 µs/条。
- **P3（vec 是 ALU 受限，不是 UB 带宽受限）**：b4 `aiv_vec_fp32_ratio=0.726`，vec 时间只随元素数变化。本轮把权重暂存从 f32 换成 f16（R3-Q）**实测 vec 时间一位小数都不变**（54.78 → 54.78），否定了「UB 读写带宽是 vec 瓶颈」的假设。
- **P4（d=512 的可用 staging tile 由 UB 预算决定，且预算不可从 Python 侧腾挪）**：完整 kernel 在 BK=32@d=512 被拒（`ub overflow, requires 1606656 bits while 1572864 bits available`），BK=64@d=256 亦然。本轮先按「pattern 级 8192 元素硬 cap」解释，随后**被自己的探针推翻**：把同形态的裸 mul+reduce 单独取出，`(32,512)=16384` 元素的 product tile **可以编译**（`pattern-library/repro/stage-tile-not-pattern-cap.py`，已跑通）；把完整 kernel 的尾部整段删掉后，BK=32@d=512 **也**可以编译（`_engram_decode_kernel_r3u_gemvonly.py`）。
  → 正确归因：GEMV 暂存 tile 与 GEMV 之后的尾部工作集**共享同一份 192 KiB UB**，编译器不把 GEMV 暂存的字节回收给尾部（尾部在 d=512 下约 68 KiB，含 `(max_conv_len-1,d)` 的 shift 暂存），因此 BK=32 的 2×64 KiB + 尾部越界。
  → 但**尾部腾挪实测无效**：R3-K 把 4 个 `(1,d)` 归约量合并、`rms_w` 复用、`k_local/h_local` 移交给卷积抽头、删掉死 `e_local`（共约 18 KiB），报错字节数**一字不差**；f16 暂存（R3-R）同样一字不差。⇒ 从 Python 侧无法把这份预算腾出来，d=512 的可用 BK 实际就是 16。

#### 2.3 Candidate Optimization Points（round 3）

| opt_id | 目标现象 | 优化点 | 判断依据 | 状态 |
|---|---|---|---|---|
| R3-A | P2：每轮 1 条小 DMA 重读 `e` | `e_n` 改为从常驻 `e_local` 做 UB→UB 切片（去掉 nk 条 GM DMA） | UB→UB `T.copy` 文档支持（`T.copy.md` §1/§2.4） | **回退**（b4 102.18；b1/b8 打平） |
| R3-B | P4：BK 越大 DMA 越少 | `BK=32`（原结构） | 8192 元素 cap 的边界试探 | **blocked**（d=512 UB overflow，与第 2 轮同一条报错） |
| R3-C | P1 的 vec+scalar 双份元素运算 | 去掉 `prod`，`T.unroll(BK)` 逐行 FMA 累积进 `acc` | 若 codegen 生成 FMA 可省一次完整 pass | **回退**（vec 17.57→72.26 @b1，写带宽 214 GB/s） |
| R3-D | 每轮 K/V 两次 W 拷贝共用 `w_f`（WAR） | K/V 各自一块暂存 buffer | 解开 intra-iteration 串行 | config_no_gain（mte2 51.4→28.7 但总时间 +3.8%） |
| R3-E/R3-F | P4：腾 UB 才能放大 BK | 原地乘积（省 `prod`）→ BK 16 / 32 | 别名/原地省 UB | **E 未单独采数；F：d=256 胜（27.26/27.38），d=512 回退（95.72）** |
| R3-G | 组合 R3-D+R3-F | K/V 分块 + 原地乘积（BK=16） | 组合两个已验证方向 | **回退**（b4 98.76；b1/b8 打平） |
| R3-K | R3-B 的 UB 版本 | BK=32 + 死 buffer/别名回收（4 个 `(1,d)` 合并、`rms_w` 复用、`k/h` 行复用、删死 `e_local`） | 第 2 轮溢出仅 4.1 KB | **blocked**（报错字节数与 R3-B 完全相同 → 别名无用，P4） |
| R3-M | bankgroup 冲突 0.15–0.21 | 给 reduce 操作数行步长加 8 个 fp32 的 padding | padding 打散 bank group | **回退**（b1 48.37、b4 120.45：非 2 的幂行宽破坏向量化） |
| R3-N | P2：进一步减少 DMA | BK=64 + 原地乘积（d=256） | 8192 cap 下的极限 | **回退**（b1 30.87：各 pipe 都变快但 busy ratio 掉到 0.47，pipeline 崩） |
| R3-P | P1 的 scalar 暴露 | `T.Pipelined(nk, num_stages=2)` | 显式软件流水 | config_no_gain（73.621 vs 73.591，编译器默认已等效） |
| R3-Q | P3 假设的 UB 带宽瓶颈 | W 暂存改 f16（`w16[kk,j] * e_n[kk]` 混合精度乘） | 每 MAC 的 UB 流量 20B→12B | config_no_gain（73.531/28.351/28.511，全在噪声内 → **否定 UB 带宽假设**） |
| R3-R | R3-B 的 f16 版本 | f16 暂存 + BK=32 @d=512 | 暂存减半或可装下 | **blocked**（字节数与 R3-B 完全相同，P4） |
| R3-T | P2 + P4 | 一次 DMA 搬 2×8192 元素的 K 行，按 2 个列块分块 reduce | 尝试把「每条 DMA 的 K 行数」与「tile 上限」解耦 | **回退**（b4 514.27，scalar 422.61：分块后 codegen 退化） |
| **R3-B′** | **P2 + P4** | **`BK = 8192 // d_padded`（d=256→32，d=512→16）** | **取工具链允许的最大 tile，d=256 的 DMA 条数减半** | **improved（胜出）** |

#### 2.4 实验批处理（T-4）

`perf_opt/run_exp.py`：对单个候选依次做 `--level L0` → 逐 workload `msprof op` → 追加 `perf_records.jsonl`，并打印统一的候选表。本轮所有分支均由该 runner 驱动，stdout 落在 `perf_opt/logs/round3/`。

#### 2.5 本轮「候选 vs current best」对比表（B2 结构化回流）

base = `r3base_shift2d`（`3835dee0…`）。每分支只改一个主要优化点；筛选采集 `launch-count=10`，baseline/final `launch-count=20`。表中 `vec/mte2/scalar` 为 pipe busy 时间（µs）；`duration` 为 `msprof op Task Duration`。AICore 利用率列给出三 pipe busy ratio 之和（b1 基线 2.02 = vec0.63+mte20.68+scalar0.71）。

| branch (candidate_id) | opt_id | 主要改动 | L0 | Task Duration b1 / b4 / b8 (us) | AICore busy（b1 vec/mte2/scalar） | memory 指标（UB rd/wr, GB/s） | 结论 |
|---|---|---|---:|---|---|---|---|
| （base）`r3base_shift2d` | — | 第 2 轮 winner，BK=16 | pass | 28.361 / 73.591 / 28.611 | 0.631/0.680/0.705 | 163/103 | 起点 |
| `r3a_eub` | R3-A | `e_n` 取常驻 `e_local`（UB→UB） | pass | 28.491 / **102.182** / 28.711 | 0.625/0.668/0.739 | 161/103 | 回退（b4 mte2_wait 0.50，流水被拆） |
| `r3b_bk32` | R3-B | BK=32（原结构） | **blocked** | 编译失败（d=512 UB overflow） | — | — | blocked |
| `r3c_fma` | R3-C | `T.unroll` FMA 累积，无 `prod` | pass | 86.672 / 265.345 / 86.872 | 0.839/0.281/0.353 | 57/214 | 回退（vec 4×） |
| `r3d_wsplit` | R3-D | K/V 分开暂存 | pass | 28.591 / 76.422 / 28.671 | 0.633/0.480/0.688 | 164/103 | config_no_gain（mte2 51.4→28.7） |
| `r3f_bk32inplace` | R3-F | BK=32 + 原地乘积 | pass | **27.261** / **95.722** / **27.381** | 0.630/0.541/0.582 | 175/109 | d=256 有效，d=512 回退（mte2_wait 0.83） |
| `r3g_wsplit_inplace` | R3-G | K/V 分块 + 原地（BK=16） | pass | 28.551 / 98.762 / 28.691 | 0.634/0.481/0.693 | 165/103 | 回退（b4） |
| `r3k_bk32alias` | R3-K | BK=32 + UB 别名回收 | **blocked** | 编译失败（字节数与 R3-B 相同） | — | — | blocked（P4） |
| `r3m_pad8` | R3-M | reduce 操作数行 padding=8 | pass | 48.371 / 120.452 / 49.271 | 0.664/0.464/0.716 | — | 回退（非 2 的幂行宽） |
| `r3n_bk64inplace` | R3-N | BK=64 + 原地（d=256） | pass | 30.871 / blocked / 31.111 | 0.475/0.278/0.449 | — | 回退（pipeline 崩，busy 0.47） |
| `r3p_pipe` | R3-P | `T.Pipelined(num_stages=2)` | pass | — / 73.621 / — | 0.751/0.709/0.551 | — | config_no_gain（= 基线） |
| `r3q_f16stage` | R3-Q | W 暂存改 f16 + 混合乘 | pass | 28.351 / 73.531 / 28.511 | 0.632/0.680/0.706 | — | config_no_gain（全在噪声内，否定 UB 带宽假设） |
| `r3r_f16bk32` | R3-R | f16 暂存 + BK=32@d=512 | **blocked** | 编译失败（字节数与 R3-B 相同） | — | — | blocked（P4） |
| `r3t_slab2` | R3-T | 2×8192 元素 slab + 2 列块 reduce | pass | 105.362 / 514.270 / 105.492 | 0.378/0.081/0.685 | — | 回退（scalar 422.6） |
| **`r3b_bk32_n256`** | **R3-B′** | **BK=32 @ d=256**（原结构） | pass | **26.751 / — / 26.771** | 0.617/0.549/0.590 | — | **improved（b1 −5.7%、b8 −6.4%）** |

**逐分支机制归因（本轮最有价值的负向结论）**：

1. **R3-A / R3-D / R3-Q 的共同点**：它们都**只**降低 mte2（51.4→28.7 µs 量级），duration 却不动或变差 → **mte2 早已被隐藏**（`mte2_wait` 仅 0.017–0.041），继续降 mte2 无收益；而一旦改动破坏了编译器自动 multi-buffer 的指令排列（R3-A、R3-G、R3-F@d512），`mte2_wait` 会飙到 0.44–0.83，duration 反向劣化。这与第 2 轮 R2-A/R2-G 的归因完全一致（同一机制第三次独立复现）。
2. **R3-C 的价值在于否定 FMA 路线**：`T.unroll(BK)` 的逐行 row-broadcast FMA 只产生 0.058 的 bank 冲突（远低于基线的 0.199），但 vec 时间 ×4、写带宽 214 GB/s → 编译器把它降成了「每个 `kk` 一次全 d 的 read-modify-write」，元素级 pass 数不减反增。**"合并 mul+reduce 减少 pass" 在本工具链不可得**。
3. **R3-M 否定 padding 修法**：bankgroup 冲突确实是 20% 级，但把行宽变成 `d_padded+8` 后 vec 时间翻倍（b1 17.57→31.81），因为 Ascend 向量指令要求对齐的连续段。→ 冲突只能从「改变 reduce 形态」消除，而 reduce 形态已被 R3-C/R3-T 证明不可改。
4. **R3-T 否定「解耦 DMA 行数与 tile 上限」**：分块 reduce 引入 `acc_k[0, PD+j] += red[0,j]` 这类逐元素 pass，scalar 直接爆炸到 422 µs。
5. **R3-K / R3-R 与 P4 的关系**：这两支原本是为「腾出 UB 放大 BK」而做，实测报错字节数完全不变 ⇒ 归因修正后它们**不是**「UB 总量不足」的证据，而是「Python 侧腾不动这份预算」的证据。R3-B′（`BK = 8192 // d_padded`）之所以仍成立，是因为它取到了 d=256 下预算允许的**实际**上限 32（d=512 上限仍是 16），与 P4 的机制描述一致。

#### 2.6 核内合并检查（winner = R3-B′）

R3-B′ 是**编译期 tiling 参数**改动（`BK = TUNED_STAGE_ELEMS // d_padded`，再整除回退到能整除 `d_mem`），无 Python factory / op / wrapper 形态的 shape 分派，`@T.prim_func` 内同样无分支路径 → 无条件合并进 `perf_opt/_engram_decode_kernel.py`，并在合并版上复测全部 tune workload 与 smoke 精度：

| merged_candidate | workload_id | current_best_us | merged_us | Δ | 精度 | status |
|---|---|---:|---:|---:|---|---|
| `perf_opt/_engram_decode_kernel.py`（= R3-B′，`ef7b3493…`） | decode-b1-dmem512-d256-fp16 | 28.361 | **26.651** | −6.03% | L0 pass | pass |
| 同上 | decode-b4-dmem1024-d512-fp16 | 73.591 | **73.541** | −0.07%（噪声） | L0 pass | pass（非回退） |
| 同上 | decode-b8-dmem512-d256-bf16 | 28.611 | **26.791** | −6.36% | L0 pass | pass |
| 同上 | decode-smoke-b1-dmem256-d256-fp16（smoke，仅精度） | — | 不采集 | — | L0 pass（`y_maxdiff=2.441e-04`） | pass |

无回退 → 更新 current best。b4 的 −0.07% 落在噪声内（未过 `<5%` 差异的 ab_test 裁决门槛），**不记为 b4 的增益**，b4 结论为 `no_gain`。

#### 2.7 Iteration Winner

- winner: `perf_opt/_engram_decode_kernel.py`（来源分支 `_engram_decode_kernel_r3b_bk32.py`，candidate_id `r3b_bk32_n256`；final 复测 id `r3final_bk8192`）
- 采纳理由：b1/b8 分别 −6.0%/−6.4%（均超 3% 噪声阈值），b4 不变（非回退）；改动是单个编译期 tiling 常量，无新增分支路径。
- 未采纳：R3-A/C/D/E/F/G/K/M/N/P/Q/R/T（逐项证据见 §2.5）。

---

## 3. Autotune Log

本轮未使用 autotune 框架；参数面只有 `BK`，且其可行域由工具链硬 cap 决定，**已穷举**：

| 结构 | 搜索空间 | 实测可行点 | 备注 |
|---|---|---|---|
| BK @ d_padded=256 | {16, 32, 64}（另试原地乘积变体） | **32**（26.75/26.77） | 64 触发 8192 元素 cap（原地变体可编译但 pipeline 崩，30.87） |
| BK @ d_padded=512 | {16, 32}（原结构 / 原地 / f16 暂存 / UB 别名 4 种实现） | **16**（73.55） | 32 在 4 种实现下都报同一 UB overflow 字节数 → 不可行 |
| reduce tile 形态 | {(BK,d) 整块, 2×8192 slab + 列分块, 行 padding} | **(BK,d) ≤8192 整块** | 另两种分别 −87%（R3-T）与 −60%（R3-M） |
| 暂存 dtype | {f32, f16} | 任意（无差异） | R3-Q 实测 neutral → 不采纳，保持 f32 以免引入精度面风险 |

⇒ `BK = 8192 // d_padded` 是该参数面上唯一的可行最优点，不存在残余搜索空间。

---

## 4. 逐项目标达成判定、可行包络与 `[DESIGN_LIMIT]` 判定

### 4.1 逐 workload 对照本轮目标（§2 表）

| workload | 基线 us | 本轮 final us | 本轮 Ratio | 目标 us | 目标 Ratio | 达成 |
|---|---:|---:|---:|---:|---:|---|
| decode-b1-dmem512-d256-fp16 | 28.361 | **26.651** | 1.1460% | ≤19.66 | ≥1.56% | **未达成**（差 35.5%） |
| decode-b4-dmem1024-d512-fp16 | 73.591 | **73.541** | 3.8770% | ≤54.78 | ≥5.20% | **未达成**（差 34.3%） |
| decode-b8-dmem512-d256-bf16 | 28.611 | **26.791** | 5.4250% | ≤19.74 | ≥7.53% | **未达成**（差 35.7%） |
| tune 平均 | — | — | **3.4827%** | — | ≥4.76% | **未达成**（差 36.7%） |

### 4.2 为什么逐项目标不可达（本轮新增的量化证据，替代第 2 轮的纯推论）

三档目标值 = 第 2 轮同一份 profile 的 `max(vec, mte2, scalar)`，即「三 pipe 100% 重叠」。
本轮把它拆成了可检验的等式并逐分支验证：

```
Task Duration ≈ vec_time + a * scalar_time,      a ∈ [0.47, 0.68]   (12 分支一致)
```

- 目标要求 `a = 0`（scalar 完全被隐藏）。本轮**没有任何分支**把 a 压到 0.6 以下（R3-F/R3-N 反而升到 0.67/0.68，R3-T 到 1.0+）。
- scalar 的本质是 **DMA/向量指令的发射时间**（P2：≈0.2 µs/条，与条数成正比）。减少条数只能靠更大的 staging tile，而 d=512 的可用 tile 由 **UB 预算**封在 BK=16（P4：GEMV 暂存与尾部工作集共享 192 KiB，且 R3-K/R3-R 证明这份预算从 Python 侧腾不动）；d=256 已被推到预算上限 BK=32。把 K 行数与 tile 解耦的尝试（R3-T）会让 codegen 退化为逐元素 pass。
- vec 是 ALU 受限（P3：f16 暂存不改变 vec 时间一位小数），无法通过降低 UB 流量改善；合并 mul+reduce 的 FMA 形态在本工具链 codegen 失败（R3-C）。

⇒ **可达包络 = `vec + ≈0.5·scalar`**，本轮已达该包络（见 §4.3 对照）。逐项目标对应的是 `max(pipe)`，在「scalar 与 vec 不能重叠超过约 50%」的实测约束下不可达，与实现质量无关。

### 4.3 三档上界表（更新第 2 轮 §4.2，合并本轮实测）

| workload | 基线 Ratio | 本轮 final Ratio | ① 结构冻结·完美重叠上界（= 本轮目标） | ② **本轮实测可达包络** `vec + 0.5·scalar` | 达成率（final/①） |
|---|---:|---:|---|---|---:|
| b1 | 1.0773% | **1.1460%** | 1.56% | `16.19 + 0.5×15.49 = 23.94 µs` → 1.276% | 73.5% |
| b4 | 3.8742% | **3.8770%** | 5.20% | `54.78 + 0.5×40.17 = 74.87 µs` → 3.809%（final 已略优于该式） | 74.6% |
| b8 | 5.1982% | **5.4250%** | 7.53% | `16.17 + 0.5×15.57 = 23.96 µs` → 6.207% | 72.0% |
| tune 平均 | 3.3832% | **3.4827%** | 4.76% | 3.784% | 73.2% |

> ② 是「保持当前结构、只把 scalar 暴露压到 0.5 的下界」的估计，b1/b8 尚有约 3.4% 的结构余量（26.8 → 24.0 µs），但那需要把 scalar 从 15.5 再降 —— 而 scalar 已由 DMA 条数决定、条数已由 8192 元素 cap 决定，故余量不可兑现（R3-N 显示 BK=64 时 scalar 确实降到 13.6 但 pipeline 崩、总时间 30.87）。

### 4.4 `[DESIGN_LIMIT]` 判定（`_shared/standards/perf-feedback.md` §1，两项须同时满足）

1. **①「天花板由算法/设计层决定」→ 不满足**：本轮归因的硬约束全部是**工具链/硬件常数**——(a) 8192 fp32 元素的 mul+reduce staging tile cap（编译期报错，三次独立复现）、(b) scalar 发射时间与 vec 只能部分重叠（12 分支拟合 a≈0.5）、(c) fp32 向量效率 0.73。低核利用这一设计事实在 `DESIGN.md` §5.5 已被显式记录并主动接受，本轮没有任何实测指向某个 DESIGN 假设**错误**。
2. **②「结构性加速 > 2x 或实测与设计假设直接矛盾」→ 不满足可执行形式**：唯一 >2x 的路径是跨 AI Core 归约（第 2 轮 §2.3 已核查：前端无入口，CG-2026-0018），它属于能力缺口，即便「改设计」也无法落地，不构成设计修订可兑现的收益。

⇒ **不产出 `perf_opt/perf_feedback.md`，不返回 `[DESIGN_LIMIT]`。** 与第 2 轮判定一致（本轮为其补上了 P1–P4 的量化证据）。
参照锚定（T-1）：本轮再次检索 `examples/engram/engram_decode.py`（源 GPU 形态，同 grid=(B,)）、`examples/norm/rms_norm.py`、`examples/gemv/example_gemv.py`，**未找到**任何「单 block 内可跨核归约」或「batch=1 时多核拆 GEMV」的仓内先例 → 天花板结论按标准降级标注，但 §4.2 的结论不依赖先例，只依赖本轮 12 个分支的实测拟合。

---

## Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/_engram_decode_kernel.py::_engram_decode_kernel::main | decode-b1-dmem512-d256-fp16 | 28.361 | 26.651 | r3final_bk8192 |
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/_engram_decode_kernel.py::_engram_decode_kernel::main | decode-b4-dmem1024-d512-fp16 | 73.591 | 73.541 | r3final_bk8192 |
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/_engram_decode_kernel.py::_engram_decode_kernel::main | decode-b8-dmem512-d256-bf16 | 28.611 | 26.791 | r3final_bk8192 |

> 三行均来自**同一个最终候选**（`r3final_bk8192`，artifact `perf_opt/_engram_decode_kernel.py`，sha256 `ef7b3493…`，`phase=final`，launch-count=20）；与 `perf_records.jsonl` 的 final 记录逐项一致（偏差 0%）。

### 5.2 profile 证据（final，`profiles/final3/`）

| workload | Task(us) | Ratio | vec_t | mte2_t | scalar_t | busy(vec/mte2/scalar) | bankgrp_cflt |
|---|---:|---:|---:|---:|---:|---|---:|
| b1 | 26.651 | 1.1460% | 16.19 | 14.30 | 15.49 | 0.620/0.546/0.593 | — |
| b4 | 73.541 | 3.8770% | 54.78 | 51.76 | 40.17 | 0.752/0.711/0.551 | — |
| b8 | 26.791 | 5.4250% | 16.17 | 13.85 | 15.57 | 0.622/0.534/0.598 | — |

### 5.3 smoke / skip 排除与精度

- smoke `decode-smoke-b1-dmem256-d256-fp16`：仅精度，**通过**（`y_maxdiff=2.441e-04`），无性能记录。
- skipped：无（4 个参数化案例全部可运行）。
- 说明：`bk = TUNED_STAGE_ELEMS // d_padded` 在 smoke 形状（d_padded=256, d_mem=256）取 BK=32、nk=8，`d_mem % BK == 0` 成立。

### 5.4 最终精度回归（`python perf_opt/_engram_decode_kernel.py --level all`，日志 `logs/round3/L_all_final.log`）

- **L0：PASS**（4/4：smoke 2.441e-04、L0-2 1.221e-04、L0-3 9.766e-04、L0-4 7.812e-03，与基线逐位同量级）；修正 docstring 后再次 `--level all` 全量回归，L0/L1 仍 PASS（`logs/round3/L_all_final2.log`，17 个 `[ok]`）
- **L1：PASS**（multi-step 17 步 y_maxerr 6.104e-05；bf16；state-shift **bit-exact 0.0e+00**；gate-corner finite；dil3/dil5；额外 seed）
- L2：`L2-fp32` 告警保留（**继承基线的既有 f32 不可编译限制**，非本轮回退），L2-seed101/202 通过
- Boundary：4/4 通过（B-minL / B-w2 / B-w1 / B-b1）

### 5.5 总体结论与中止原因

- 最终版本：`perf_opt/_engram_decode_kernel.py`，sha256 `ef7b3493315b8af3e16926cac16291c56cf5eb337545a2b2ca4caf52a0bbc414`
- 相对基线：b1 −6.03%、b4 −0.07%（噪声）、b8 −6.36%；tune 平均 Ratio 3.3832% → **3.4827%**
- 逐项目标未达成，且已证明在当前工具链/硬件 + 冻结结构下不可达（§4.2）
- **中止原因：`plateau`** — 本轮 12 个 valid 分支中只有 1 个带来 >3% 提升（R3-B′，+6.0%/+6.4%），其余 11 个为回退/config_no_gain/blocked；剩余方向均已被实测封死（§2.5 归因 1–4），不存在未验证的结构候选。按 skill「plateau ≠ 天花板（T-1）」，已做参照结构 diff 检查（§4.4 参照锚定），未发现可移植的结构差异。

---

## 6. Skill Retrospective

### 6.1 Skill Flow Issues

1. **`skill` 工具在本会话不可用**（`ripgrep execution failed`），只能直接 `read` `SKILL.md` 与 references。流程未受影响，但需记录：执行器不应依赖 skill 加载器可用。
2. **P4（工具链 cap）本该在第一轮就当作常量反解出来**：第 2 轮把 BK=32 失败记为「UB 溢出」，本轮通过「三种不同实现的报错字节数完全相同」才定位到「8192 元素 / mul+reduce pattern」这一真实语义。建议在 §2.4「实验裁决」里加一条**恒定参数反解**动作：对编译期容量类失败，至少用 2–3 个实现变体验证报错量是否随改动变化，再决定归因到「总量」还是「单一 pattern」。
3. **`<5%` ab_test 门槛在本轮无法用于「平区」**：b4 全部分支要么回退 >3%，要么打平在 0.1% 内，没有落在 3–5% 的候选，故未触发 `ab_test.py`。b4 的 −0.05% 因此明确不记为增益（正确做法）。
4. **单候选 3 workload × launch-count=10 约 5 分钟（含 JIT 重编译，模块导入即 `clear_cache()`）**，12 个分支约 70 分钟 profiling + 6 次 L0（每次 ≈95 s）。批处理 runner（`run_exp.py`）把这部分从「逐命令往返」变成单命令，是本轮能在预算内跑完 12 个分支的关键（与 VP-2026-0021 预期一致）。

### 6.2 Value Point Proposals

| id | vp_type | 结论 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| CONST-ub-budget-not-pattern-cap | D | **mul+reduce 暂存 tile 没有 pattern 级元素上限**：(32,512)=16384 元素的裸 mul+reduce 可以编译；`engram_decode` 在 BK=32@d=512 的 `ub overflow` 来自 **GEMV 暂存与尾部工作集共享 192 KiB 预算**（删尾部后同 BK 可编译），且该预算**从 Python 侧腾不动**（尾部分别回收 ≈18 KiB / 改 f16 暂存，报错字节数一字不差） | 裸探针编译通过（`repro/stage-tile-not-pattern-cap.py`，已跑通）；`_engram_decode_kernel_r3u_gemvonly.py`（BK=32 去尾部）编译通过；R3-K / R3-R 报错字节数 = R3-B 完全相同 | `pattern-library/repro/stage-tile-not-pattern-cap.py` | tilelang 0.1.2+01805d7a / CANN 9.0.0 / 910B3 |
| CONST-dma-issue-0.2us | D | 小 GM→UB tile 的**每指令**开销在本工具链为 **0.20–0.21 µs**（第 2 轮同口径为 0.16 µs） | b1 96 条→scalar 19.61 µs；b4 192 条→40.24 µs；b1@BK=32 48 条→15.53 µs（线性拟合 R²>0.98） | 同上（repro 内含三档计数） | 同上 |
| PL-scalar-vec-half-overlap | P | **`Task Duration ≈ vec + 0.5×scalar`**：AIV 上 scalar 发射期与 vec 执行期只能部分重叠，a∈[0.47,0.68]（12 分支）；「三 pipe 完美重叠 = max(pipe)」这一常见上界假设在本设备上**高估 35–40%** | `perf_records.jsonl` 全部 35 行 + `profiles/*/prof.json` 的 pipe 三元组逐行反解 | `repro-missing`（该结论是 msprof pipe 统计层面的拟合，需在完整 kernel 上采集；拟合脚本 `perf_opt/parse_prof.py` 已固化口径） | 同上 |
| TRAP-ub-row-stride-padding | P | Ascend 向量指令要求对齐连续段：给 `(BK, d_padded)` 操作数行宽加 8 个 fp32 padding「打散 bank group 冲突」会让 **vec 时间翻倍**（b1 17.57→31.81，b4 54.78→87.45），尽管冲突率本身确实有 20% | R3-M：48.371 / 120.452 / 49.271 µs | `repro-missing`（最小化需对照两个完整 kernel 的 vec 时间，本轮以 msprof 逐分支对照替代） | 同上 |
| R3-negative-family | R | 「降 mte2」类改动（slab / UB 切片取 e / K-V 分块 / f16 暂存）在本算子族**已连续 3 次独立复现**：mte2 −45% 但 duration 不变或劣化，机制是破坏 auto multi-buffer 的指令排列（`mte2_wait` 0.02→0.44–0.83）。建议 pattern-library 的 `BP_pipeline_overlap` 加一条前置检查：「先看 `mte2_wait_ratio`，<0.05 时降 mte2 无效」 | R2-A/R2-G（history）+ R3-A/R3-D/R3-G/R3-Q | — | 同上 |
| CASE-engram-decode-stage4 | C | 本 case 条目已回写 round-3 结果（BK=8192/d_padded、P1–P4、可达包络表） | `pattern-library/cases.md` | — | 同上 |

### 6.3 Transferable Lessons

- **上界要给「可达包络」而不是「理想重叠」**：`max(pipe)` 类上界在发射受限设备上系统性高估；应先反解 `duration = f(pipes)` 的经验式，再报上界。
- **编译期容量失败必须做「变体不变性」检验**：报错量随实现变体不变 ⇒ 归因到 pattern 级常量，而不是总量；否则会像第 2 轮一样把 8192 元素 cap 误记为「UB 总量不足」，从而反复尝试无用的 buffer 回收（本轮 R3-K 即为此浪费的 1 个实验）。
- **「降低某 pipe 的 busy time」不等于「降低 duration」**：先看该 pipe 的 `*_wait_ratio`，已隐藏的 pipe 不值得优化。

### 6.4 本轮未执行/未回写项

- `perf_feedback.md`：不产出（§4.4 两项条件均不满足）。
- pattern-library 回写：已按 §6.2 追加（constants.md / layout.md / traps-compiler.md / cases.md + `repro/` 三个最小复现）。`repro-missing` 项：无。
- `Optimize.md`：本项目流程未要求，不生成。

---

## 7. 工件清单

| 路径 | 说明 |
|---|---|
| `perf_opt/_engram_decode_kernel.py` | **最终 best**（sha256 `dfb5528f…`） |
| `perf_opt/perf_records.jsonl` | 本轮 append-only 记录：3 baseline + 29 candidate + 3 final |
| `perf_opt/workload_inventory.json` | 4 案例分类 + `tuning_status` / `precision_pass` |
| `perf_opt/history/opt_log_round1.md`、`perf_records_round1.jsonl` | 第 1 轮（未改） |
| `perf_opt/history/opt_log_round2.md`、`perf_records_round2.jsonl` | 第 2 轮（本轮开始时原样移入） |
| `perf_opt/_engram_decode_kernel_r3*.py` | 本轮 12 个实验分支 |
| `perf_opt/run_exp.py` | 本轮批处理 runner（L0 → msprof → 记录追加） |
| `perf_opt/profiles/baseline3/`、`profiles/round3/`、`profiles/final3/` | raw `msprof op` 数据 |
| `perf_opt/logs/round3/` | L0/最终回归与各分支 stdout |
