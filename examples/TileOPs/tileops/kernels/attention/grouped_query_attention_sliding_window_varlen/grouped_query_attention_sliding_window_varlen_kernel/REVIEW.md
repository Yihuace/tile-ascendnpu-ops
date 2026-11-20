# `_gqa_sw_fwd_varlen_kernel` 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/grouped_query_attention_sliding_window_varlen/_gqa_sw_fwd_varlen_kernel/DESIGN.md
- 任务类型: 迁移（scenario=migration, migration_mode=harness；**增量设计**，骨架 = 同族定长版 `examples/grouped_query_attention_sliding_window/_gqa_sw_fwd_kernel/`）
- 源算子: `/mnt/workspace/TileOPs/tileops/kernels/attention/gqa_sliding_window_varlen_fwd.py`（工厂 `_gqa_sw_fwd_varlen_kernel` L105–241 / prim_func L140–239 / 掩码工厂 `_make_apply_mask` L33–102 / wrapper L244–282）+ Op `TileOPs/tileops/ops/attention/gqa.py::GroupedQueryAttentionSlidingWindowVarlenFwdOp`（L1682/L1761/L1847）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 5（见维度 2/7/8 说明，均不阻塞）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: **PASS**（`python3 verify_equiv.py`，torch CPU，2026-10-16，退出码 0，**11/11 EQUIV_PASS**）。执行输出与 `DESIGN.md` §1.6.1 内嵌结果表**逐行一致**（M1' max_ulp 6 / M2' 1 / M3' 0 / M5'/M6' 0 / M4' 0 / M8'/M9' 2147798666※ / M10' 0 / E2E-fp16 2147798666※ / E2E-fp16-golden 2178593504※ / E2E-bf16 734 / E2E-bf16-golden 2192168490※；全部 `vr_ref=vr_base=0.0`；末行 `OVERALL: ALL EQUIV_PASS`）。※ 容差类诊断值口径与设计一致（判定以违反率 0 为准）。E2E-bf16-golden 实测 tier-1 率 5.8e-4、tier-2 率 0，与设计 §8.2 所述 tier-3 分类一致。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design …` JSON checks 摘要：
  - `r3_metrics`: **pass** — "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"。
  - `ub_budget`: **skip** — "no per-level buffer tables parsed from the memory plan"（§4.5 表格级标记为 `### 4.5 UB …`，checker 的 level 正则未命中）→ **人工已核**：§4.5 逐项 16+32+8+16+16（固定 5 项）+ diff 32 + pen 32 + 行向量 ~1.1 + scales 0.77 ≈ **154 KB ≤ 192 KB ✓**（原位复用口径见维度 2 说明）。
  - `l0c_budget`: **skip** — "no block_M x block_N values parsed" → **人工已核**：`l0_c [64,256] f32 = 64 KB ≤ 128 KB ✓`（`CONST-capacity-910B2C` L0C=128KB）。
  - `core_split`: **skip** — "insufficient parsed dims {bm:64}; declared = None"（设计用 `max_seqlen_q` 而非 `M`，checker 无该口径）→ **人工已核**：逻辑核数 = `ceildiv(max_seqlen_q,bm)×H×B`，8b-long 4096 / 70b-long 8192 » 物理核数 20 → 极大规模 → `T.Kernel(20)` persistent 核内 `T.serial`（§5.5 三要素齐全）。
- **设计期探针复跑（独立重跑，非纸面）**：
  - **P-V1**（`history_version/design_probe_varlen_structure.py`，`source env_npuir.sh`）：**复现成功** — A Developer persist+gemm（无 vec）**PASS** / B Developer + `v-prefix` **BLOCKED**（`out_of_range` / `map::at` / `ERR99999`）/ C Expert 双 Scope **PASS**；V1 ragged 直接 GM 标量读 `cu[bz]`（动态 slice 起点/extent/循环界/输出地址 + ghost 块）**PASS maxdiff 0.0**；V2 copy cu→UB 再标量索引 **BLOCKED**。与设计 §2.2/§0.6 V1/§10.4 结论**逐项一致**。
  - **P-V2**（`history_version/design_probe_varlen_mask.py`）：**复现成功** — 主选 offset 平移双侧算术惩罚 + per-row 列 OOB + 常开尾带 **7/7 配置 bit-exact（valid mask 逐位 + `S'=S+pen·(−1e38)` sentinel 逐位 + NaN-free）**，`scalar-threshold warning=False`。含运行期 offset∈{0,32,64,96} 与运行期 `tn_real`。与设计 §0.6 V2/V4/§1.6.3 一致。
- **物理核数实查**: `NPUUtils.get()` → `get_arch()=Ascend910B3`，`get_aicore_num()=20`，`get_aivector_core_num()=40`（2026-10-16 实查）。与 §5.5/§0.9 声明（20 AIC / 40 AIV）**一致**。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**亲自 Read 源算子全文（gqa_sliding_window_varlen_fwd.py 606 行）并与 §0 逐项核对。**

- **语义理解正确性（pass）**：§0.1 四条件掩码（causal / 左窗 / 右窗 / q_oob / k_oob）与源码 `_make_apply_mask` L61–100 四编译期路径逐条对应（`k_idx·block_n+j >/< bx·block_m+i+offset±w`、`bx·block_m+i ≥ q_len`、`k_idx·block_n+j ≥ kv_len`）；`offset = kv_len − q_len`（§0.1/§0.2）与源码 L177 一致；`offset=0` 标准 prefill、`offset>0` KV-cache 与源码 docstring L6–11 一致。**dtype 语义**（S/统计量 fp32、P 量化回输入 dtype、lse fp32）与源码 L158–165/L220/L149 一致。**规约语义**（per-sample 沿 kv 轴、online 按 kv 块序）与 L206–237 一致。
- **角点识别正确（pass，强证据）**：§0.1 角点说明指出「源 `is_causal and has_window` 分支（L61–72）在 causal 下只施加 causal|left、忽略 right，且 `k_end` 不含 `wr`（L186–188）」——源头核对**属实**（L61–72 分支无 `right_mask`；L186–188 causal 分支 `min(kv_len, offset+(bx+1)bm)` 无 `+wr`）。设计据此按实际行为分派 trace 并在 §9.3 记明该组合不承诺——**属真实读源码的证据，非臆造**。
- **算法解读完整性（pass）**：§0.3 步骤表 S1–S13 覆盖源码 L140–239 全部计算语句（grid/边界标量读/Q 装载/循环范围/mask 预填/gemm1/online softmax/量化/PV/epilogue 守卫），无遗漏、无臆造；§0.3 host 侧逻辑（`lru_cache`、`heads % heads_kv` 校验、`make_log2e_scale`、`total_q/total_k` 运行期解析）与 L105–123/L314–331 一致。
- **I/O 契约（pass）**：§0.1/§0.2 表逐项与源码 prim_func L140–150 核对（q `(total_q,H,D)`、k/v `(total_k,H_kv,D)`、cu_seqlens_q/k `(B+1,)` int32、max_seqlen_q int32 grid 标量、output `(total_q,H,D)` out_idx=[6,7] 第 6 号、lse `(H,total_q)` fp32 第 7 号）；§0.2「Op wrapper 丢弃 lse」经 `gqa.py` L1847 `output, _ = self.kernel.forward(...)` **亲自核对属实**。
- **优化手段覆盖（pass）**：§0.4 OP1–OP16 覆盖源码全部可识别优化（flash online softmax / 带状裁剪 / ghost 空转 / SMEM tiling / coalesced / mma / Pipelined / fragment 累加 / warp 归约 / FullRow / 三维 grid / exp2 折叠 / FAST_MATH / lru_cache / bf16 直连 / per-sample 运行期长度）；算法层与实现层标注正确（OP9/OP10/OP13 属实现/结构层）。
- **耦合性判定合理性（pass）**：§0.5 表体 20 行四态处置，逐行核对——**保留 7 / 等价替换 7 / 重新设计 4 / 舍弃 2 = 20，每行恰计一次 ✓**（保留下含纯算法层的 OP2/OP2b/OP16/OP1/S7 clamp/q 驻留；重新设计 = OP6/OP7/OP12/OP10；舍弃 = OP9/OP13，均为 CUDA 专属且给出依据）。舍弃/重设计依据与 GPU→NPU 能力映射（一维 Kernel、L0C 出数仅 GM、无 warp、无 CUDA 编译链）一致，无矛盾。
- **NPU 重设计可行性（pass）**：§0.6 V1–V8 每项含「源方案 → NPU 新算法 → 语义保持论证」三要素，且满足 NPU 硬件约束（一维 persistent `T.Kernel(20)`、L0C 出数必经 GM ws（`store_fixpipe` 仅 GM）、分形下限、L1/L0C/UB 容量）。V1（ragged 任务解码）、V2（offset 平移掩码 + per-row 列 OOB）、V3（运行期带状范围）、V4（常开尾带）、V5（运行期 flag 预算 + 两相位）、V6（packed 寻址）、V7（lse 布局）、V8（接口）覆盖 varlen 全部新增轴；V1/V2/V4 有设备证据（P-V1/P-V2）。
- **设计与决策一致性（pass）**：§1–§7 描述的是 §0.5/§0.6 决策后的 NPU 算法——§1.4 选定算法 = §1.6.0 #2 两相位；§3.1 公式拆解以 §1.6.1 优化后公式为输入；§3.2 API 与 V1–V7 对应；§4/§5/§6/§7 与 V1–V8 落地一致；无「§0 判重设计、§3 照抄源方案」脱节。
- **golden 独立性（pass）**：§8.1 golden 以 §0.1 源语义为唯一依据（per-sample 物化 softmax + 源四条件掩码 + 源 P 量化点），**无 online softmax / 无带状裁剪 / 无 e 域换底 / 无 f16 S/O 物化 / 无算术惩罚掩码**——与 §0.6 NPU 算法结构正交，未复刻。与源仓测试 `test_grouped_query_attention_sliding_window_varlen.py::ref_program`（L39，offset 右下对齐语义）语义一致，仅补 lse；独立推演成立。

**维度 0 结论：pass**（语义/算法/优化/耦合性/重设计/一致性/golden 独立性 7 项全 pass）。

### 1. API 可行性 — pass

- §3.2/§3.4 列出的 API 均有佐证：`T.gemm`（`docs/Tilelang.language/线性代数操作/T.gemm.md` 存在）、`T.reduce`（`docs/…/规约操作/T.reduce.md`）、`T.vcast`、`T.arange`/`vbrc`/`vmax`/`vmin`/`vsub`/`vmul`/`vadd`、`T.sync_block_set/wait`（`docs/…/同步管道操作/`）、`T.store_fixpipe`（`docs/…/内存操作/`）——**引用文档路径逐一存在（亲自 Glob/Read 核对）**；同族定长版母本 + MHA 母本为生产先例。
- **本算子最高新颖度 API = 核内 GM 标量读 `cu[i]`（int32 → 运行期 PrimExpr 用于 slice 起点/extent/循环界/输出地址）**：本仓 `examples/` 无先例、docs 未文档化，但**设备探针 P-V1 实测 PASS（maxdiff 0.0）+ 本检视独立复跑复现**；备选「copy cu→UB 再标量索引」实测 BLOCKED（P-V1 VARLEN_COPY）。判定成立。
- v-prefix 优先：设计链全部用 v-prefix（vsub/vmax/vmin/vmul/vadd/vexp/vdiv/vlog2/vbrc/vcast），未出现 npuir_xxx 遗留形态。模式匹配：Expert 模式用显式分层 + 手动 `sync_block_set/wait`（§7.1），符合 Expert 口径。

### 2. 内存层级规划 — warn

- **搬运路径完整（pass）**：§4.4 给出完整 GM → L1/UB → L0C 路径，无 GM→L0 越级；S/O 跨引擎经 GM ws 三族（`store_fixpipe` L0C→GM 强制，`CONST-store-fixpipe-gm-only`）。
- **UB 预算（pass，但余量薄）**：§4.5 逐项 = 16 + 32 + 8 + 16 + 16（固定 5 项）+ ub_diff 32 + ub_pen 32 + 行向量 ~1.1 + scales 0.77 ≈ **154 KB ≤ 192 KB ✓**（`CONST-capacity-910B2C`）；`auto-multi-buffer=false` 显式 alloc 结构实测 ~0.98–1.0×（`CONST-capacity-910B2C` r9 反例 / CG-2026-0008），非 1.7× 悲观口径，判定合理。
- **L0C 容量（pass）**：`l0_c [64,256] f32 = 64 KB ≤ 128 KB ✓`。
- **Buffer 层级标注（pass）**：§4.3 逐 buffer 标注 L1/L0C/GM/UB。
- **建议（非阻塞）**：§4.5 的「原位复用」把 varlen 新增的 `rowi/thr/t1/t2/t3`（另 ~64KB）折叠进 diff/pen 区；但 diff 与 pen 是掩码链全生命期活跃的 buffer，逐元素核对后**掩码链至少需 diff + pen + 1 个 scratch（t1/t2/t3 串行复用）= 3×[32,256]f32 ≈ 96KB**，而非设计所述 2×（64KB）——修正后 UB ≈ 186KB（仍 ≤ 192KB，但余量仅 ~6KB）。**叠加 `TRAP-npuir-per-copy-ub-staging-sum` 口径（每处 `T.copy(GM→UB)` 单独分配 staging、按 copy site 求和，不做 liveness 合并）**，varlen 的 S/O 两处 UB-dest copy + 收紧后的预算存在触边风险。设计已给 R-5 回退（UB diet −32KB 或 bn_eff→128），且同族定长版同结构 22/22 通过，故非阻塞；**建议 §4.5 显式补录 mask scratch 与 per-copy staging 计数，并在 §9.1 引 `TRAP-npuir-per-copy-ub-staging-sum`（同 910B3/CANN9 工具链）**。

### 3. Tiling 策略 — pass

- **Block 划分（pass）**：§5.2 给出 `bm=64` / `bn_eff=256` 具体值 + 依据（UB 预算、wrapper 默认、L0C 余量、发射开销定律 PL-1.9-hardlimits、flag 预算守卫）。
- **对齐约束（pass）**：`bn_eff`/`dim` 为 16 倍数（守卫 `ceil16` 保证；dim∈{64,128}），尾轴 256·2B=512B ≥ 32B；分形 M/N ≥ 16、K ≥ 32 由尾钳位保证。
- **非整除处理（pass）**：§6.4 尾块全套（`tmc/tnc` 尾钳位 + 常开尾带 per-row 列 OOB + N 载体零初始化 + `zbuf` + 行截断 + ghost 空转），运行期 ragged 机制常开。
- **分核策略三要素（pass）**：① 逻辑核数 = `ceildiv(max_seqlen_q,bm)×H×B`（8b-long 4096 / 70b-long 8192，§5.5 表）；② 物理核数 = **`NPUUtils.get().get_aicore_num()` 实查 20**（本检视独立复跑确认 20/40；混合算子直接用返回值，非假设值/经验值）✓；③ 规模判定 = **极大规模**（4/5 workload 逻辑任务 1024–8192 » 20），方案 = 固定启动 20 核 + 核内 `T.serial` 串行（`num_local = T.ceildiv(num_logical − kernel_id, 20)`）。核内串行循环边界由编译期常量 + kernel_id 派生（非运行期 shape/核数）——P-V1 实测可 lower/运行。核数与 block 取值自洽。**smoke 欠载（32 任务 / 20 核，2.0× 不均衡）**已如实记录并给 Stage 4 处置，未掩盖。

### 4. 技术约束检测 — pass

逐项核对 §3.5.1 强制 5 项（+ 附加已知陷阱）：三维 Kernel→V1 一维 persistent ✓；L0C 128KB ✓；L1 96KB≤512KB / UB ~154KB≤192KB ✓；GEMM 非整除（运行期 `real_m<16`/`tn_real<32`/`kv_len%bn≠0`）→ 尾钳位 + 常开尾带 + 行截断 + ghost 空转 ✓；`store_fixpipe` L0C 出数仅 GM → ws 三族 ✓。另：Developer MIX 不可用→Expert（P-V1 复现 + CG-2026-0010 + PL-1.16）✓；`T.sync_block_set` CANN9.0.0 编译失败→运行时 shim（CG-2026-0023，§7.4，逐语句沿用定长版）✓；`T.floordiv/ceildiv` 负分子截断（CG-2026-0024）→两段式分界用无除法整数比较（`k_start` 的 `max(0,·)` 在所有除法之前使分子非负）✓；`vcmp` int16 标量化（PL-1.11）→算术惩罚式（P-V2 零 warning）✓；stale UB 尾列 NaN（`TRAP-npuir-stale-ub-tail-columns-nan`）→N 载体零初始化 + 全宽 per-row 列 OOB + `l1_b` zbuf ✓。**VP-2026-0097 双判据纪律**：设计以**源头零初始化**（stale 列恒有限）根治，使加法掩码不再承担清 NaN 职责——比 vselect 双判据更强，符合该 VP 根因。

### 5. 循环与同步 — pass

- **循环结构（pass）**：§6.1 明确 `T.serial`（任务维/带内块 pass-1/pass-2，Cube+Vector 各 2 个）；元素级走 v-prefix 链 + `T.reduce`（无逐元素标量循环）；明确不用 `T.Pipelined`（Expert 手动两相位）。
- **同步与模式匹配（pass）**：§7.2 flag id 分配表（带内块 l 三次握手 + `FLAG_TASKDONE = L_max_bound`），总 id = `L_max_bound+1 ≤ 16`（T.set_flag.md §2.1）；Expert 手动 `sync_block_set/wait` + `T.rs("PIPE_*")`，与 §2 模式一致。varlen flag 复用安全论证（两侧同 task 解码 + 同 cu_seqlens 读 ⇒ 同 loop_count/l 序列）成立。**2-AIV 聚合语义（R-1）**单 wait 依定长版母本，附回退（按 subid 拆双 id）。
- **尾块处理（pass）**：§6.4 说明 shape 不整除时的尾块逻辑（常开），含数值例（smoke `kv_len=128%256≠0`）。
- **跨核同步（pass）**：无；核内原语适用范围（T.set_flag.md §1 / T.sync_block_set.md §1）已核。

### 6. 验证方案 — pass

- **Golden 函数（pass）**：§8.1 含完整 PyTorch 参考实现草案（packed 逐样本切片 → expand KV heads → 带 offset 四条件掩码 → softmax → @v → 拼回），独立于 NPU 算法（见维度 0）。
- **L0 测试计划（pass）**：§8.2 含 L0 门槛（L0-1 fp16 smoke offset=0 / L0-2 bf16 / L0-3 KV-cache offset>0 / L0-4 混合长度 ragged + 契约校验），具体规则 shape、dtype、按类别精度标准齐全；L0 首验顺序明确。
- **精度标准（pass）**：按 dtype 给 atol/rtol（fp16 5e-3/1e-5 双门 + 2-ulp 回退；Op 门 atol=1e-2, rtol=1e-2 与源仓测试 L229 冻结值一致）；lse 1e-3/1e-3。

### 7. 完整性与一致性 — pass

- **章节齐全（pass）**：§0（0.1–0.7 齐全）/§1–§11 齐全；§1.6 含 1.6.0/1.6.1/1.6.2/1.6.3。
- **无占位符（pass）**：全文无未填写占位符（`⬜ 待实现` 为交付状态标记，非占位符；已逐行检索确认无遗留模板标记）。
- **同类引用真实（pass）**：§3.5.3/§0.7 引用路径（定长版姊妹、MHA 母本、flash_attn_npuir.py、devmode 探针）**逐一存在**；§3.4 引用的 docs 路径 14 条**全部存在**。
- **内部一致（pass，一处小瑕）**：API 映射/内存规划/Tiling 三处相互一致。**建议（非阻塞）**：§3.3 的 `k_end` 伪代码只写了 causal / else 两支（`else → ceildiv(kv_len, bn_eff)`），而权威的 §0.6 V3（L220）含非 causal 右窗 `+wr` 支；若按 §3.3 字面实现非 causal 右窗域（L2 覆盖），band 不右裁会使 `loop_count` 达 `ceildiv(kv_len,bn)` 而超出 `L_max_bound`（flag id 溢出）。target manifest 5 workload 全 causal（`wr=-1`），L0/L1 不受影响；**建议 §3.3 补齐 `+wr` 支，与 §0.6 V3 对齐**。

### 8. 算法优化分析 — warn

**以 §1.6 为对象逐项核对，并交叉核对 §3.1/§6 与 §1.6 的双向一致性；数学等价性与调研结论独立推演/复核。**

- **§1.6 章节存在性（pass）**：四个子节齐全。
- **算法调研存在性与完整性（pass）**：§1.6.0 四问齐全——R1 候选表 15 行**含基线 #0（源算法）** + 每条等价性初判与收益方向；R2 明确「**有**」在线算法（结构依据：源本身即 online softmax + 选定 #2 保留 online 递推，单遍扫描）；R3 四口径表（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值，**基线在表**，口径写清含/不含）；R4 逐候选对照 6 项检查清单。**对照 `algorithm-candidates.md` 命中行**：`ALG-attention`（R1 scale 融入 Q/K、softmax 倒数乘法化；R2 flash/online；R3 materialize vs flash；R4 Cube GEMM + Vector online softmax 两相位）与 `ALG-softmax`（R2 online softmax）——命中行候选**均已评估**（#9 scale 融入 q 经精度论证否决；#2 flash 两相位选定），无漏候选。
- **调研结论正确性（独立复核）（pass / 见下）**：
  - **等价性**：verify_equiv 独立重跑 11/11 EQUIV_PASS（与内嵌表一致）；本检视另**独立代数推演 M5' 的 offset 平移**（`k≤q+off ⟺ (n_lo+j)≤(bx·bm+row0+i)+off ⟺ j−i ≤ (bx·bm+row0+off)−n_lo = K_up` ✓）与 **M2' lse 域**（`log2(ℓ)+m·LOG2E`，e 域 m=prescale·m_score ⇒ 等价源 `log2(ℓ)+m_score·dim^{−1/2}·LOG2E` ✓）。
  - **复杂度算术逐项独立复算（✅ 全部可复算，无自相矛盾）**：Σloop_count=120（逐 bx：bx0–3→1、4–7→2、8–11→3、12–15→4、16–19→5、20–31→5〔k_end−k_start，非线性单调，经逐 bx 代入 k_start=`max(0,bx·64−1024)//256` 与 k_end=`ceildiv((bx+1)·64,256)` 验证〕）×（H·B=128）= **E=15360 ✓**；全 causal Σk_end=4·(1+…+8)=144×128=**18432 ✓**，削减 (18432−15360)/18432=**−16.7% ✓**；FLOPs 每样本 attended=524800+1049600=1574400、total=6297600、`4·32·6297600·128=1.0318e11=103.2 GFLOP ✓`（全 causal 137.5、−24.9% ✓）；HBM `2·(33.55M+8.39M)·2B=167.8MB ✓`；fabric **#2 选定**候选 KV tile 2.01GB + ws 往返（S32+P32+O_f16 16=80KB×2×15360=2.52GB）≈ **4.5GB ✓**（**中间缓冲 GM 往返已计入**）；D-2 ≈2.0–3.3ms（发射项 `15360×17÷40×0.5µs≈3.26ms` + 保守外推 1279.54×(128/64)≈2.56ms ✓）。**无漏算、选定候选口径自洽**。
  - **负向断言复核**：R2 为正向（有在线变体，无负向断言）；R4 的 defer #6 KV-split（跨核合并无原语，T.set_flag.md §1/T.sync_block_set.md §1 核内语义——文档已核）与 #11 f16 softmax 链（defer Stage 4）均给量化理由，非「未考虑」；#1/#9/#13/#14/#15 否决附实测（2.31×）/精度/契约/结构依据。**未与候选库/源码证据矛盾**。
  - **建议级瑕疵（非阻塞，不影响结论）**：① §1.6.0 R3 的 #2「per-core 单槽 ws ≈1.05MB（总量 20 核 ≈21MB）」与其**自身行内公式**（`pad_band·bm·f16·2 + dim·L_max·bm·f16 = 2·1536·64·2 + 128·6·64·2 = 480KB ≈0.47MB`）**不符**，且与 §4.3 权威值（合计 **≈9.8MB**，逐项 3.93+3.93+1.97）**约 2× 矛盾**；② §1.6.0 R3 的 **#0/#1 基线行** fabric「`E×160KB=2.52GB`；合计 ≈4.7GB」与其自列的「S32+P32+O f32 32KB」（应 192KB/往返 ≈2.95GB、合计 ≈5.0GB）**不符**（本行是比对行，非选定候选）。均属非权威单元格的算术自相矛盾，**不改变「ws 入 L2 ✓」与选型结论**（选型依据 = 2.31× 实测，非 fabric 字节）。
- **调研→设计一致性（pass）**：§1.4 算法描述 = §1.6.0 选定 #2；§3.1 公式拆解以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致（无 §1.6.2 未论证的逐元素标量循环）；迁移任务 §0.5/§0.6 处置与调研选族一致，源优化手段意图承接逐项（OP1→保留、OP2/OP2b→V3/V1、OP16→核内标量读、OP6→两相位、OP7→ws 物化、OP12→算术惩罚…），**无静默丢弃**。
- **数学等价优化完整性（pass）**：§1.6.1 M1'–M14' 逐项四要素（原式→优化后→等价性论证→收益量化）；采纳 12 项（M1'/M2'/M3'/M4'/M5'/M6'/M7'/M8'/M9'/M10'/M12'/M13'），否决 M11'（精度降级）/M14'（无 API）；无优化空间项有依据。
- **等价性论证正确性（pass）**：逐项独立推演成立（M4' 全掩码块 = online 递推恒等元 α=1/Δℓ=0/Δo=0 经 verify_equiv 实证 `alpha==1`；M5' 端点代入见上；M8'/M9' 容差内等价 f16 2^-11 与 §8.2 精度门一致）。**机器验证重跑结果与内嵌表一致**（D-1）。
- **向量化替代完整性（pass）**：§1.6.2 覆盖 19 个计算点，与 §3.3 伪代码、§6 循环结构交叉核对**无遗漏**；15 项已向量化（替代方案 API 均有 docs 佐证，优先 v-prefix）；掩码 1 项由源 `T.Parallel` 谓词改为全 ALU 算术惩罚链（P-V2 设备 7/7 bit-exact + 零 warning）。
- **不可替代理由充分性（pass）**：保留 4 类——#16 运行期 cu_seqlens 标量读（block 级索引/任务映射，每任务 O(1)，非逐元素热点，源同形态）；#17 block 索引与运行期范围（每任务 ~20 次整数标量运算）；#18 tile 级顺序依赖/块分类控制流；#19 host 元数据。理由具体到本算子，非「实现简单/照源码写」。
- **替代 API 可行性（pass）**：替代 API 在 docs/examples 有佐证（见维度 1），v-prefix。
- **向量化轴与布局决策完整性（pass）**：§1.6.3 Vector 侧候选矩阵 4 行（#1 行主序 tile 列轴【主选】/#2 [bn,rm] 列主序【转置毒化 2.6× 实测否决，核对 `TRAP-transpose-epilogue-poison` 实为 2.6x ✓】/#3 单 AIV 全行【UB 192KB 超限否决 ✓】/#4 GPU warp 轴【轴系不对应】），逐候选量化评分；掩码式选择表（算术惩罚 vs vselect）含量化理由；**弃选方案的代价机制前提与所引文档/pattern-library 实测一致**（无「把 I/O 契约当内部布局依据」）；Cube 侧分形/tensorgemm 决策清晰。
- **实验裁决模式完整性（pass）**：§1.6.3 给出三件套——主选（算术惩罚式）+ 备选（vselect 形态，结构完整：掩码链替换 + 载体零初始化 + 其余结构不变）+ 裁决计划（代表 shape smoke+8b-short/long×fp16、msprof median-of-20、**判定阈值 0.95**、回写路径）；判定依赖未知常数（vmin/vmax 吞吐 vs vcmp 标量化税，跨 SoC 未重验）已显式标注。
- **布局决策与实现一致（pass）**：§1.6.3 选定行主序 [rm,bn]/[rm,dim]、列连续轴与 §3.3 伪代码、§4.3/§4.5 buffer 形状、【H,total_q,1] lse 视图、§6 循环（累加循环内层向量维=列轴）三方一致；无核内重排、无 transpose（VP-D6 规避）。
- **与下游章节一致（pass）**：§3.1 用优化后公式（含 `pen`/`K_up`/`K_lo`/`thr` 与 e 域 α/P）；§6 无 §1.6.2 未论证的标量循环。

**维度 8 结论：warn**（调研/等价/向量化/布局实质全 pass，仅 §1.6.0 R3 两处非权威单元格算术自相矛盾——建议 Stage 1 顺手订正；不构成阻塞）。

## 检视小结

9 维度无阻塞级 fail：维度 0（源算子理解与迁移分析，亲自读源码 606 行逐项核对）、1、3、4、5、6、7 为 pass；维度 2、7、8 各含建议级瑕疵（UB 余量薄 + per-copy staging 口径未录；§3.3 `k_end` 缺 `+wr` 支；§1.6.0 R3 两处算术自相矛盾）。上述均不导致 Stage 3 编译/运行/精度失败，且有文档化回退或属非权威单元格，按风险优先判为**建议级**。核心新增轴（ragged 运行期 cu_seqlens 标量读、offset 右对齐掩码、per-row 列 OOB、常开尾带）均有**设备探针 + 等价性机器验证**双证据，且本检视**独立复跑全部复现**（verify_equiv 11/11、P-V1 A/C/V1 PASS·B/V2 BLOCKED、P-V2 7/7）。结论：**通过**。
