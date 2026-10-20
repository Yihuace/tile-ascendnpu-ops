结论: 通过

# _conv1d_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_kernel_DESIGN.md
- 任务类型: 迁移（harness，GPU TileOPs-main → NPUIR Ascend）
- 源算子: /mnt/workspace/TileOPs-main/src/tileops/kernels/convolution/conv1d.py（`_conv1d_kernel` builder L33-148；`Conv1dKernel` 类 L745-871）；op 层 `/mnt/workspace/TileOPs-main/src/tileops/ops/convolution.py`（`Conv1dFwdOp` L75-167，`Conv1dKernel` 注册 L80）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 4（均为可直接在 Stage 3 收敛的实现细化项，不构成阻塞）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: 不适用（无采纳优化项）**——§1.6.1 结论为「无优化空间」（Winograd / bias 折叠 / 转置 GEMM / split-K 四候选均无正收益，不采纳；§1.6.1 表 + §1.6.1 优化结论）。已确认 `examples/conv1d/_conv1d_kernel/verify_equiv.py` **不存在**（`ls` 返回 `No such file or directory`），按 gate `S1-EQUIV-EXEC` 豁免规则合法。注：§1.6.0 R1 候选 6（polyphase，stride>1 专用）是**精确恒等的结构分解**（算法路线，不改变 FLOPs），归 §1.6.0 采纳而非 §1.6.1 公式级采纳项，其恒等性由本检视**独立推演**（见维度 8），故不触发脚本生成。
- **design_calc_check**: 任务书写 `--dir`，脚本实际接口为 `--design`（`python3 .agents/tools/design_calc_check.py -h` 确认仅 `--design`；传 `--dir` 报 `error: the following arguments are required: --design`）。以脚本自身接口运行 `python3 .agents/tools/design_calc_check.py --design examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_kernel_DESIGN.md`，输出：
```json
{
  "design": "examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_kernel_DESIGN.md",
  "capacity_constants": {"ub_kb": 192.0, "ub_inflation": 1.7, "l0c_kb": 128.0, "source": "pattern-library/constants.md CONST-capacity-910B2C"},
  "checks": [
    {"check": "l1_budget", "status": "pass", "detail": "L1: 2 buffer entries sum = 48.0 KB; capacity = 512.0 KB", "within_inflation_budget": null},
    {"check": "l0c_budget", "status": "pass", "detail": "block 64x64 x fp32 accum = 16.0 KB vs L0C 128 KB"},
    {"check": "core_split", "status": "skip", "detail": "insufficient parsed dims {'N': 1, 'bm': 64, 'bn': 128}; declared logical cores = 64"},
    {"check": "r3_metrics", "status": "pass", "detail": "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"}
  ]
}
```
  - `l1_budget` pass（48KB ≤ 512KB），与 §4.5 一致。
  - `l0c_budget` pass，但机械把 `block_M×block_N` 欠解析为 `64×64`（16KB）；设计实际 `64×128×4 = 32768 B = 32KB`（§4.5）——两者均 ≤ 128KB，**人工复核通过**。
  - `core_split` **skip**（解析到的 dims 不足，且把与核数无关的 `64` 误当 declared logical cores）→ **人工复核通过**：whisper-conv1 `20×24=480`、encodec-deep `4×5=20`、wav2vec2-layer1 `8×25=200`、smoke `1×1=1`（物理核 20 实查）四档自洽（见维度 3）。
  - `r3_metrics` pass（四口径齐全）。
  - 机械未输出 `ub_budget`（无可机械解析的 UB 条目）→ 由维度 2 人工复核覆盖（§4.5：weight 前置段 80KB、bias 段 80KB，各 ≤192KB；合并口径见维度 2 建议项）。
- **门禁自检**（`python3 .agents/tools/statectl.py gate 2 --dir examples/conv1d/_conv1d_kernel`）：
```json
{"action": "gate", "failures": [], "ok": true, "pass": true, "stage": 2, "warnings": []}
```

## 检视详情

### 0. 源算子理解与迁移分析 — pass

已亲自 Read 源算子全文（`conv1d.py` `_conv1d_kernel` builder L33-148、`Conv1dKernel` L745-871；`ops/convolution.py` `Conv1dFwdOp` L75-167），并与 §0.1–§0.7 逐项核对，结论与源码一致且完整：

- **语义理解正确**：§0.1 公式 `out[n,c_o,l]=bias[c_o]+Σ_{k_w,c_i} W[c_o,c_i,k_w]·x[n,c_i,l·s_W+k_w·d_W-p_L]` 与源码 B 装载 `il = ol*stride_l + kw*dilation_l - pad_left`（`conv1d.py:94`）、`kw=k_idx//c_in; ci=k_idx%c_in`（`:92-93`）、`T.gemm(weight_shared, data_shared, out_local)`（`:105`）一致。`L_out` 公式（§0.1）与源码 `out_l=(l_in+pad_left+pad_right-dilation_l*(kernel_l-1)-1)//stride_l+1`（`:47`）、`k_total=c_in*kernel_l`（`:48`）逐字对应。**K 序 `(k_w,c_i)`（`c_i` 最内）** 与源码注释「k runs over (kernel_l, c_in)…kw = k_idx // c_in」（`:89-92`）一致。
- **规约/dtype 语义正确**：§0.1 记「唯一规约轴 K=C_in·K_W、K 分块 fp32 累加、无跨 L 规约」；源码 `accum_dtype="float"`（`:46`）、`out_local=T.alloc_fragment(...,accum_dtype)`（`:69`）、`T.clear(out_local)`（`:73`）、`T.Pipelined(T.ceildiv(k_total,block_k))`（`:83`）→ 一致。bias 在 **fp32 域**相加后统一 cast：源码 `T.cast(out_local[i,j] + T.cast(bias[oc], accum_dtype), dtype)`（`:113`/`:227`）与 §0.1 dtype 语义一致。
- **I/O 与源码签名一致**：§0.2 `x(n,c_in,l_in)` / `weight_flat(c_out,k_total)` / `out(n,c_out,out_l)` / `bias(c_out)` 与 `_conv1d_bias_main`（`:127-134`）/`_conv1d_main`（`:138-144`）逐字对应；`out_idx=[2]` 与源码 `@tilelang.jit(out_idx=[2],...)`（`:50`）一致。op 层 `weight` 原生 `[C_out,C_in/groups,K_W]`（`ops/convolution.py:129`），groups=1（`Conv1dKernel.refusal` 要求 `groups==1`，`conv1d.py:754`）——§0.2 已注明；host `weight.permute(0,2,1).contiguous().view(c_out,k_total)`（`:870`）已在 §0.1/§0.3 步骤 1 记录。
- **算法步骤覆盖完整、无臆造**：§0.3 九步覆盖 host 重排（`:870`）、三维 grid（`:61-66`）、`tile_spatial_full` 快/慢路径判定（`:75-81`）、`T.clear`（`:73`）、`T.copy(weight_flat...)`（`:84`）、B 快路径 `T.copy` / 慢路径逐元素掩码 gather（`:86-103`，`in_bound=(k_idx<k_total)&(ol<out_l)&(il>=0)&(il<l_in)` `:98`）、`T.gemm`（`:105`）、epilogue 掩码 + bias + cast（`:107-121`）、`T.copy(out_shared, out[...])`（`:123`）；host 侧 `launch` 已列出。无遗漏、无臆造步骤。
- **优化手段识别完整**：§0.4 覆盖 SMEM tiling（`:67-68`）、tensor-core mma（`:105`）、`T.Pipelined(num_stages)`（`:83`）、fragment fp32 累加（`:69`）、implicit-GEMM K 物化 host 重排（`:870`）、尾块掩码 gather（`:86-103`/`:107-121`）、coalesced 连续 `T.copy`（`:84`/`:96`）、`T.use_swizzle(CONV_SWIZZLE_PANEL)`（`:72`）、`threads`（`:61-65`）、空间完整快路径（`:95`）；未把实现层优化误判为算法层，亦未遗漏（源码无 warp shuffle / online / persistent / 手工 TMA，§0.4 末注已声明）。
- **耦合性判定合理**：§0.5 每条目四态（保留 / 等价替换 / 重新设计 / 舍弃）+ 依据；10 条优化项处置与 GPU→NPU 能力映射不矛盾——`T.gemm`→Cube、`alloc_shared`→`alloc_L1`、fragment→L0C、`T.Pipelined`→L1 多缓冲为等价替换；三维 grid→一维 persistent、host 重排→核内 `T.transpose`、跨步 im2col→polyphase/per-tap slice、bias epilogue→MixCV Vector 为重设计；swizzle / threads 舍弃理由（NPU 无 bank conflict / `T.Kernel(threads=)` noop）写明。
- **NPU 重设计可行且语义保持**：§0.6 R1（三维 grid+threads → 一维 `T.Kernel(num_kernels,is_npu=True)` + `cid` 解码 + persistent，双射+idle 钳位论证）、R2（掩码 gather → 分形 clamp `T.max(16,ceil16(tail))` + slice 尾块 + 越界零填充 + 裁剪 store）、R3（host 重排 → 核内 Vector `T.copy`+`T.transpose([0,2,1])`+`T.copy`）、R4（stride=1 per-tap slice / stride>1 polyphase 恒等分解）、R5（fragment bias → MixCV Vector + 深度 2 双槽 4-flag 握手）五处置，每项含「源方案 → NPU 新算法 → 语义保持论证」三要素；新算法满足一维 Kernel / 内存层级 / 分形 ≥16 / 对齐 / 容量约束（§5.3/§4.5 复算）。语义保持论证成立（任务映射双射；补 0 项对累加无贡献；裁剪列不落 GM；fp32 累加序不变）。
- **§1–§7 与迁移决策一致**：§3.1/§3.2/§3.3、§4、§5、§6、§7 描述的是 §0.5/§0.6 决策后的 NPU 算法，无「§0 判重设计、§3 照抄源方案」的脱节。
- **golden 独立性成立**：§8.1 以 §0.1 语义为唯一依据移植源仓参考实现 `torch.nn.functional.conv1d`（在 fp32 域计算后 cast 回输入 dtype），**不复刻 §0.6 的 Cube/persistent/polyphase 数据流**——验证独立性成立。

### 1. API 可行性 — pass

§3.2/§3.4 所列 API 均有佐证且优先 v-prefix：
- `T.Scope("Cube")`/`T.Scope("Vector")`、`T.rs("PIPE_*")`、`T.sync_block_set/wait` — `_gqa_prefill_fwd_kernel.py`（`with T.Scope` `:274`/`:388`）、`_ssd_chunk_scan_fwd_kernel.py`（Cube/Vector 双 Scope + 4-flag 握手 `:253-255`/`:398-399`）；`docs/Tilelang.language/同步管道操作/T.sync_block_set.md`、`T.sync_block_wait.md` 签名一致（§2.4 规范示例即「Cube 写前 wait / 写后 set、Vector 读前 wait / 消费后 set」的双向形态）。
- `T.gemm(a,b,c,size,initC,b_transpose)` — `docs/.../线性代数操作/T.gemm.md`（`size=[a,b,c]`；fp16 输入 dst 须 fp32）与设计 `size=[tmc,tkc,tnc]`、`l0_c` fp32 一致；探针 `examples/conv1d/probe/conv1d_geomm_probe.py:100` 实测。
- `T.copy` slice 形态（GM→L1 自动 ND2NZ、L0C→GM 自动 fixpipe、静态 1 维自动折叠、动态 extents 尾块、GM 侧 min_rank=2）— `docs/.../内存操作/T.copy.md` §2.2.2 规则 1/2/3 + §2.3 条目 6、§2.4 示例 4 吻合；探针 `:88-108` 实测（per-tap slice 承载重叠/跨步 im2col）。
- `T.transpose(src,dst,permutation)`（UB 级、**仅两轴交换**、fp16/fp32/bf16 ✓、`size` 参数触发 IndexError 须留空）— `docs/.../创建操作/T.transpose.md` §2.3；设计 `permutation=[0,2,1]`（交换轴 1/2）合规。
- `T.alloc_L1`/`T.alloc_L0C`/`T.alloc_ub`、`T.vbrc`/`T.vadd`/`T.vcast` — docs + gqa/ssd 先例（ssd `:439` vbrc、`:443` vadd、`:421/:450` vcast）。
- `NPUUtils.get().get_aicore_num()` 实查（`tilelang.utils`）— §5.5 附查询代码与返回值 20。
- `T.load_nd2nz` 存在但**主动避开**（跨步误读，`TRAP-load-nd2nz-strided` / CG-2026-0004），改用 slice 形态 `T.copy`——与探针实测绕法一致。
- 编程模式：Expert（§2），与 `T.gemm`/L1/L0C/双 Scope 要求匹配；Developer 在 persistent+gemm 下崩溃（`TRAP-DEVMODE-PERSIST-GEMM`）。

建议项（warn，不阻塞）：设计支持 `bfloat16`，而 pattern-library `algorithm-candidates.md` 的 `ALG-transpose` R4 记「T.transpose dtype 矩阵（fp16/fp32 ✓，bf16/整型 ×）」，与 `T.transpose.md` §2.2.1（bf16 √）**不一致**。按项目「API 文档为准、examples/候选库次之」的路由规则，以文档 bf16 支持为准（bf16 路径判定可行）；但两处矛盾属实测/文档待澄清项，建议 Stage 3 在 bf16 用例上显式验证 `T.transpose`。

### 2. 内存层级规划 — pass

- **搬运路径完整**（§4.4）：无 bias 纯 Cube `GM→L1→L0C→GM`；bias `GM→L1→L0C→GM[ws]→UB→GM[out]`；前置段 `GM[weight]→UB→GM[w_flat]`（Vector）、stride>1 `GM[x]→GM[y_poly]`。无 GM→L0 越级。
- **L0C 容量**：`64×128×4 = 32768 B = 32KB ≤ 128KB`（`CONST-capacity-910B2C`）✓。
- **L1 容量**：单槽 `(64×128 + 128×128)×2 = 49152 B = 48KB ≤ 512KB`；双缓冲 96KB ✓。
- **UB 容量**：weight 前置段 `2×16×c_in×K_W×eb`——whisper-conv1（`c_in=80,K_W=3`）15KB、encodec-deep（`128×10`）80KB；bias 段 `ub_f32(32KB)+ub_f16(16KB)+bias_brc(32KB)+bias_16(64B)+bias_col(256B) ≈ 80KB`。各段 ≤192KB（§4.5）✓。
- **Buffer 层级标注完整**（§4.3：L1/L0C/GM/UB 逐项标注）；跨引擎 `ws` 位于 GM（`20×2×64×128×4 = 1.31MB`），不计入片上预算。
- 机械 `l1_budget`/`l0c_budget` pass；`ub_budget` 未机械输出 → 人工复核（见机械复核段）。

建议项（warn，不阻塞）：**合并口径 UB**。weight 前置段与 bias 段分属两个 `T.Scope("Vector")` 块（§3.3 Phase A / Phase C），§4.5 假设二者不共驻留、可复用缓冲。但编译器是否跨两个 Vector Scope 复用 UB 未文档化；若**按和**计，encodec-deep 最坏为 `80KB + 80KB = 160KB ≤ 192KB`（`CONST-capacity-910B2C` 显式 alloc 结构实测 ratio ≈0.98–1.12 → ≤179KB，仍达标），而按 auto-multi-buffer 的 ~1.7×膨胀口径则为 272KB（超限）——基线为 `T.serial`（未启用 auto-multi-buffer），1.7× 不适用。结论：当前 `block_M=64` 下预算达标；建议 Stage 3 以显式 alloc 实测核对两段 UB 的实际分配（是否合并），并在 Stage 4 若增大 `block_M` 前按 §4.5/§9.1 复算。

### 3. Tiling 策略 — pass

- **Block 划分明确**：§5.2 `block_M/N/K = 64/128/128`，承接源 `default_config`（`conv1d.py:823`）并说明理由；`bm_w=16`（weight 前置 co 分块）。
- **对齐约束**：`block_N=128` fp16 尾轴 256B、`block_K=128` 256B（32B 对齐 ✓）；K/N/M 尾轴分形 ≥16（`ceil16` clamp `tmc/tkc/tnc = T.max(16, T.ceildiv(tail,16)*16)`，§3.3）。
- **非整除处理**：§0.6 R2 + §6.4 三重非整除（`c_out%block_M`、`out_l%block_N`、`k_total%block_K`）——分形 clamp + slice 尾块 + 越界零填充 + 裁剪 store；禁止 base+size 形态（`TRAP-T-copy-region-semantics`）。
- **分核策略三要素齐全**（§5.5）：① 逻辑核数 `num_logical = GY×GX×n`；② 物理核数 `NPUUtils.get().get_aicore_num()` **实查 = 20**（附查询代码与返回值；Ascend910B3，与 `CONST-aicore-910B2C` 记录的 24 不同，按 `CONST-aicore-910B2C` 注记「以实查为准」），Cube/混合算子**直取不翻倍**（`core-split-strategy.md` §1 要素②）；③ 统一 persistent 结构 + `T.serial(num_local_tasks)`，`cid=T.min(task_id*num_kernels+kernel_id, num_logical-1)` 钳位，边界静态。**独立复算**（本检视）：whisper-conv1 `GY=ceil(1280/64)=20, GX=ceil(2998/128)=24 → 480 = 20×24`；encodec-deep `4×5=20`；wav2vec2-layer1 `8×25=200`；smoke `1×1=1`——均与 §5.5 一致，物理核 20 使负载均匀/无超发。
- 机械 `core_split` skip → 人工复核通过（见机械复核段）。

### 4. 技术约束检测 — pass

- 三维 Kernel：源三维 `T.Kernel(bx,by,bz)`（`:61-66`）已由 §0.6 R1 改为一维 `T.Kernel(num_kernels,is_npu=True)` + cid 解码（§3.5.1）。
- GEMM 非整除：已在维度 3 处理（§6.4）。
- L0C 溢出：已在维度 2 处理（32KB）。
- `T.load_nd2nz` 跨步误读（`TRAP-load-nd2nz-strided` / CG-2026-0004）已识别并改用 slice `T.copy`（§3.5.1、§9.1/§9.2）。
- Expert 通用路径跨 dtype（`TRAP-C12-copy-dtype-cast` / `T.copy.md` §2.3 条目 2）已识别——bias 路径 fp32→fp16 由显式 `T.vcast` 承担、输入 dtype→fp32 由 `T.copy`(同 dtype)+`T.vcast` 两步承担（§4.4/§9.1/§9.2）。
- 尾块越界写（`TRAP-T-copy-region-semantics`）已识别，一律双显式 slice（§6.4/§9.2）。
- 跨引擎 `ws` 传输依赖 `CONST-store-fixpipe-gm-only`（L0C→GM 只能 fixpipe），已在 §0.6 R5/§4.4 声明。

### 5. 循环与同步 — pass

- **循环结构清晰**（§6.1/§6.2）：核级并行 `T.Kernel(num_kernels)` / 核内 `T.serial(num_local_tasks)`（persistent 静态边界）/ K 分块 `T.serial(ceil(k_total/block_K))`（`T.gemm(initC=(k==0))` 累加）/ K 块内逐 tap `T.serial(kernel_l)` slice B / Cube `T.gemm` / Vector `T.transpose`+`vbrc`+`vadd`+`vcast`；无逐元素标量循环（polyphase 物化索引除外，§1.6.2 #7 已论证）。
- **同步与模式匹配**：Expert 手动同步，`with T.rs("PIPE_*")` 标注各流水段；跨引擎用 `T.sync_block_set/wait`。
- **【关键复核】跨引擎握手协议无「自 set→自 wait」且 race-free**（本检视独立复推）：
  - 角色：Cube = ws **生产者**（L0C→fixpipe→GM ws），Vector = ws **消费者**（读 ws + 加 bias + cast → out）。flag 语义（§3.3 注 4/§6.2/§7.2 单一来源）：`2*slot` = ws 就绪（Cube 写后 `set` / Vector 读前 `wait`）；`2*slot+1` = 槽空闲（Vector 消费后 `set` / Cube 写前 `wait`），`slot=task_id%2`。
  - **「槽空闲」前导 `set(1)/set(3)` 置于消费者（Vector）Scope**（§3.3 伪代码 L487-489 在 `with T.Scope("Vector")` 内；§7.2 明列）。据此各引擎私有 flag 集**互补无交集**：Cube 置 {0,2}、等 {1,3}；Vector 置 {1,3}、等 {0,2}——**任一引擎均不对自己置位的 flag 发 wait**，无 `TRAP-sync-self-set-wait` 同核自环。
  - 独立时间线复推：前导 `set(1);set(3)` 后，Cube task0/task1 的写前 `wait(1)/wait(3)` 立即通过 → 写 `ws[0]/ws[1]` → `set(0)/set(2)`；Vector task0 `wait(0)` 通过 → 读 `ws[0]` → 处理 → 写 out → `set(1)`。Cube task2（slot0）的 `wait(1)` 须等 Vector task0 归还后才通过——生产者至多领先消费者 2 个任务，**Cube task(t+2) 覆盖 `ws[slot]` 必晚于 Vector task(t) 读毕（WAR 闭合）**、**Vector 读 `ws[slot]` 必晚于 Cube 同任务写毕（RAW 闭合）**；每 flag id 严格 set/wait 交替，深度 2 使 Cube(T) 与 Vector(T-1) 并发而 slot 不冲突，无死锁。
  - 对齐已验证先例与修正：`_ssd_chunk_scan_fwd_kernel.py` §235-342 即「深度 2 双槽 4-flag + 生产者写前 `wait(2*slot+1)`（其注释明确 placing the wait BEFORE any ws[slot] write prevents the off-by-one WAR race）+ 消费者前导 `set`」形态；ssd 中该前导位于 Cube（因 ssd 里 Cube 是槽空闲置位方），本设计角色镜像后前导移至 Vector（消费者）——**与函数 1 Stage 3 实测修正一致**（`examples/conv1d/_conv1d_pointwise_kernel/repro/TRAP-sync-self-set-wait.py`：同核自 set→自 wait 不 retire、aicore timeout 507014，修法即消费者侧前导）。
  - 前置就绪 flag `4`（Vector 写后 `set` / Cube 读前 `wait`）同样互补；flag id 用量 {0,1,2,3,4} ≤ 15/核（`CONST-flag-id-budget`）。
- **尾块处理**（§6.4）：`tm/tn/tk=T.min(block, remain)`、slice 只搬有效区、clamp store、K 块内 tap 行偏移 clamp 到 `[0,block_K)`——完整。

### 6. 验证方案 — pass

- **Golden 函数**（§8.1）：以 §0.1 语义为依据移植 `torch.nn.functional.conv1d`（groups=1，`weight[C_out,C_in,K_W]`），在 **fp32 域**计算后 cast 回输入 dtype——与源累加域（`accum_dtype="float"`）一致，且**不复刻** §0.6 的 NPU 算法（golden 独立性成立）。
- **L0 门槛测试计划**（§8.3）：覆盖 `whisper-conv1`（stride=1）、`wav2vec2-layer1`（stride=5，**polyphase 路径**）、smoke（stride=2,pad=2，**stride>1+padding**），并含 `has_bias=False/True` 两条路径；L0 必测点含非整除尾块、batch>1、dilation>1、stride>1+padding。
- **精度标准**（§8.2）：fp16 1e-2 / fp32 1e-4 / bf16 1.6e-2，与源仓 `convolution_verification` 容差口径一致。

### 7. 完整性与一致性 — pass

- 章节齐全（§0 含 0.1–0.7 子节，§1–§10 齐全）；未检出模板占位符、未完成标记。
- §3.5.3 引用的 examples 路径**均真实存在**（已 Glob 核对）：`examples/conv1d/probe/conv1d_geomm_probe.py`、`.../attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`、`.../mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/_ssd_chunk_scan_fwd_kernel.py`、`examples/TileOPs/tileops/kernels/convolution/conv1d/_conv1d_fwd_kernels.py`、`examples/TileOPs/tests/ops/test_conv1d.py`。
- API 映射（§3.2）、内存规划（§4.3/§4.4/§4.5）、Tiling（§5）、同步协议（§3.3/§6.2/§7.2）四处交叉一致，无矛盾。
- 建议项（warn，不阻塞）：§4.3/§3.3 将 `w_flat` 声明为 3D `(c_out, K_W, c_in)`，而 §3.2 步骤 3/§3.3 主循环对其按 **2D** 平坦 K 轴切片 `w_flat[by*BM : +tm, k0:k0+tk]`（`T.copy.md` §2.2.2 仅自动折叠**静态大小 1** 的维度，`c_in` 非 1 故不能据此折叠）。二者存在声明/使用口径不一致，建议统一为源算子同形的 **2D `(c_out, k_total)`**（K 序 `(k_w,c_i)` 的扁平即此），或写明 3D→2D 的显式视图；属实现细化，不改变 K 序与算法。

### 8. 算法优化分析 — pass

- **§1.6 四子节齐全**（1.6.0 调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代 / 1.6.3 向量化轴与布局）。
- **调研完整性（§1.6.0）**：R1 候选表**含基线**（候选 0 = implicit GEMM）且覆盖 `ALG-conv` 命中行（im2col+GEMM 候选 1、Winograd 候选 2、implicit GEMM 即基线 0）及恒等变形空间（FFT / split-K / bias 折叠 / polyphase / 转置 GEMM）；R2 在线算法给**明确结论**「无独立在线变体」（结构依据：输出无跨位置依赖、唯一规约 K 分块 running 累加即单遍，无 running max/sum 重缩放）；R3 复杂度**四口径**齐全（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值，基线在表，口径标注）；R4 逐候选硬件亲和（单元匹配/容量/对齐/静态边界/流水/跨核）；调研结论含选定算法与「无更优替代」调研范围（§1.6.0 末段）。
- **调研结论正确性（独立复核/复算）**：
  - **负向断言「无更优等价化简」**：对照 `algorithm-candidates.md` `ALG-conv` 命中行（R1: im2col+GEMM / implicit GEMM / Winograd）**逐条复核**——implicit GEMM 即基线（候选 0）；im2col+GEMM（候选 1）在 stride=1 仅增 `O(C_in·K_W·L_out)` 物化访存、stride>1 由候选 6（polyphase）更省；Winograd（候选 2）对 `K_W≤10` 小核净亏（变换/逆变换向量发射开销，引用 `CONST-vector-launch-overhead ~0.5µs/op`）；**无被漏掉的更优等价候选**。断言成立。
  - **负向断言「无在线变体」**：与 `ALG-conv` R2「滑窗天然流式」及源算子 §0.3/§0.4（单遍 K 分块累加、无 online 手段）**一致**，无归约/online-softmax 类被漏掉。断言成立。
  - **复杂度独立复算**：whisper-conv1 FLOPs `2·1280·2998·80·3 = 1.842 GFLOP` ✓（§1.6.0 记 1.84 GFLOP）；流量 `x 0.48MB + W 0.61MB + w_flat 写 0.61MB + w_flat 读 0.61MB + out 7.67MB ≈ 10.0MB` ✓；`10.0MB / 0.62TB/s ≈ 16.1µs` ✓；L1→L0 操作数流 `480·((64+128)·240)·2 ≈ 44.2MB ÷ ≈150GB/s/核 ÷ 20 ≈ 14.9µs` ✓。数字自洽、未发现漏计的中间缓冲 GM 往返（`w_flat` 写读已计入 D-2）。
  - **亲和性淘汰依据**：R4/§1.6.1 引用的 `CONST-capacity-910B2C`（UB 192KB / L1 512KB / L0C 128KB）、`CONST-L1-port-bw`（≈148–154 GB/s）、`CONST-vector-launch-overhead`（~0.5µs/op）、`CONST-mte2-degradation`、`CONST-store-fixpipe-gm-only`、`CONST-aicore-910B2C` 条目**均已亲自打开核对存在**，判定与实测条目不矛盾。
- **弃选论证前提核对（亲自查证）**：§1.6.3 候选 #2/#3/#4/#5 的 repack/代价机制前提与文档一致——核内 `T.transpose` 二轴交换链实测 **~5.4µs/4 次**（`layout.md` `PL-1.1-transpose-chain` L27，且 L28 明确「仅支持二轴交换、reshape→transpose 链 mis-compile、须在自然形状 buffer 间转置」），host permute 实测 **106–147µs**（`layout.md` `PL-1.3-host-permute` L65），epilogue 活跃源 transpose 毒化 **2.6x**（`traps-compiler.md` `TRAP-transpose-epilogue-poison`）；引用的 `TRAP-C10-strided-gather`、`TRAP-C12-copy-dtype-cast`、`TRAP-T-copy-region-semantics`、`TRAP-load-nd2nz-strided`、`PL-1.12-task-pipeline-depth2`、`PL-1.16-expert-dualscope-bypass`、`PL-1.15-broadcast-perf-tax`、`TRAP-sync-self-set-wait`（函数 1 repro）路径**均核对存在**。设计把「核内重排」写成 UB 级、`T.transpose` 限定二轴、host permute 单列量化——**前提与文档/实测一致，无「GM 级 transpose」类矛盾**；负向论断均有 `docs/`/pattern-library 实测佐证。
- **polyphase 换算独立推演**（stride>1 路径的等价性，本检视亲自推导）：令 `c = kw·d_W − p_L`，源 `il = ol·s_W + c`。设计 `y_poly[bn,ci,r,q]=x[bn,ci,q·s_W+r−p_L]`、`r_kw=(kw·d_W)%s_W`、`q_kw=(kw·d_W−r_kw)//s_W`。则 `y_poly[...,r_kw,l0+j+q_kw]=x[...,(l0+j+q_kw)·s_W+r_kw−p_L]=x[...,ol·s_W + (q_kw·s_W+r_kw) − p_L]=x[...,ol·s_W + kw·d_W − p_L]`（因 `q_kw·s_W+r_kw = kw·d_W` 恒等）——**与源逐元素 gather 逐项精确对应**，是精确恒等分解（非容差近似）。`Lg = ceil((l_in+p_L+(K_W−1)·d_W)/s_W)+2` 覆盖所需 q 上界。**ND2NZ 连续性**：`y_poly` 最内 `q` 维连续，故 `T.copy(y_poly[bz,0:c_in,r_kw,off:off+tn], ...)` 在 GM→L1 slice 上满足「区域最内维连续」约束（stride=1 的 per-tap slice 同理，最内 `l` 连续）——设计对「stride>1 直接 slice 会静默平坦误读」的判定与 `TRAP-load-nd2nz-strided`/CG-2026-0004 及探针注释一致。
- **调研→设计一致性**：选定算法（基线 implicit GEMM + optional bias；stride>1 叠加 polyphase）实际驱动 §1.4、§3.1（以优化后=基线公式为输入）、§6（K 分块 `T.serial` + `T.gemm(initC=(k==0))`；B 按 K 序 `(k_w,c_i)` 逐 tap slice）；迁移任务 §0.5/§0.6 处置与选族一致，源优化手段（§0.4）意图承接逐项说明（§0.6 末段），无「调研选 A 设计做 B」，无静默丢弃。
- **数学等价优化完整性**：§1.6.1 逐项评估（Winograd / bias 折叠 / 转置 / split-K）均无正收益，写明「无优化空间」的充分依据（纯 im2col-GEMM+加法，无除法/超越函数/公共子表达式/稳定化需求）；采纳项为空，等价性独立推演无对象，`verify_equiv.py` 豁免合法。
- **向量化替代完整性**：§1.6.2 覆盖 §3.3/§6 全部循环/标量点（11 项）——逐元素点（weight 重排 `T.transpose`、bias `vbrc`/`vadd`、输出 cast `T.vcast`/fixpipe）已向量化（API 有 ssd/docs 佐证，优先 v-prefix），矩阵乘落 Cube；保留 4 类标量点（polyphase 跨步 gather 索引 / cid 解码 / 尾块尺寸 / cid 钳位）均属 block 级任务映射或动态边界/跨步 gather API 缺失（`TRAP-C10-strided-gather`），非逐元素热点，理由充分。与 §6 交叉核对一致（§6 无 §1.6.2 未论证的逐元素标量循环）。
- **向量化轴与布局决策完整性**：§1.6.3 候选矩阵按 Cube 类枚举（K 序 `(k_w,c_i)` / `(c_i,k_w)` / 转置 GEMM / host permute / per-channel 分组 共 5 候选），逐候选量化评分（整除性/尾 lane 浪费/累加链/repack 代价/UB 影响），弃选均给量化理由（B 物化访存、host permute 违规 ascend-constraints §4 且净亏、per-tap 分组拷贝数爆炸）；明确 I/O layout 是契约、核内布局（NZ 分形）是设计变量，并对照源码隐式轴选择（GPU 轴为输入非结论）。选定 #1（A=`w_flat[C_out,k_total]`、B=`im2col[k_total,L_out]`、dst=L0C`[C_out,L_out]`，A 一次性 `T.transpose`、B 零 repack）与 §3.3 伪代码、§4 L0C tile 形状、§6 内层 `size=[M,K,N]` 三方一致；布局决策与 §4 UB 预算（§4.5 已按前列布局计）一致。
- **与下游章节一致**：§3.1 以优化后（=基线）公式为输入；§6 与 §1.6.2 向量化结论一致；§1.6.3 选定轴与 §6 内层分形轴一致。

建议项（warn，不阻塞）：§1.6.0 R1 可显式补一行 **direct-conv** 候选（`ALG-conv` 的 R3 提及「direct vs im2col」；源仓另有 `_conv1d_direct_kernel` 即该形态），并说明其无 Cube 利用率、FLOPs 同阶而算力利用更低故被取代——以完全覆盖命中行语境下的候选枚举。该候选非「更优」，不影响 R1 负向断言成立。

## Review Result
- stage: 2
- project: conv1d
- operator: _conv1d_kernel
- output: examples/conv1d/_conv1d_kernel/REVIEW.md
- is_migration: true
- source_op_path: /mnt/workspace/TileOPs-main/src/tileops/kernels/convolution/conv1d.py
- conclusion: 通过
- validation: pass
- validation_details:
  - 结论行存在: pass
  - 结论一致: pass
  - 维度完整（迁移 9）: pass
  - 维度 0 有源码证据: pass
  - 维度 8 有推演证据: pass
  - 建议完整: n/a（通过，无阻塞级问题）
  - 无占位符: pass
- blocking_issues: 0
- migration_findings: 语义/pass 算法/pass 优化/pass 耦合性/pass 重设计/pass 一致性/pass golden独立性/pass
- algo_opt_findings: 调研完整性/pass 调研结论正确性/pass 调研→设计一致性/pass 优化完整性/pass 等价论证/pass（无采纳项，polyphase 恒等性独立推演成立） 向量化覆盖/pass 不可替代理由/pass 替代API佐证/pass 与§3§6一致性/pass
- suggestion_count: 4
- skills_consulted: .agents/skills/tilelang-design-review/SKILL.md; .agents/skills/_shared/standards/gate-and-retry.md; .opencode/agents/tilelang-design-reviewer.md
- summary: 亲自读源算子（conv1d.py L33-148/L745-871 + ops/convolution.py）逐项核对 §0；跨引擎 bias 握手确为深度 2 双槽 4-flag、前导置于消费者（Vector）侧、各引擎 flag 集互补，独立复推 race-free 且无自 set→自 wait（函数 1 TRAP 已吸收）；polyphase 恒等分解与复杂度四口径独立复算成立；9 维度全 pass，4 项实现细化建议不阻塞——结论通过。
- issues: none
