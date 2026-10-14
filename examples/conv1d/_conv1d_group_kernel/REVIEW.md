# _conv1d_group_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/conv1d/_conv1d_group_kernel/DESIGN.md
- 任务类型: 迁移（harness，GPU TileOPs-main → NPUIR Ascend，Expert Cube）
- 源算子: /mnt/workspace/TileOPs-main/src/tileops/kernels/convolution/conv1d.py（`_conv1d_group_kernel` builder，L405-539；`GroupConv1dKernel` 类，L929-1071；op 层 `src/tileops/ops/convolution.py` `Conv1dFwdOp`，L75-163）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 3（均非阻塞，见维度 0/8 内注）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: 不适用（无采纳优化项）**——§1.6.1「优化结论」明确为「无优化空间」（block-diagonal / Winograd / bias 折叠 / 转置 GEMM / split-K 五候选经 §1.6.0 R1/R3/R4 评估均无正收益，不采纳；§1.6.1 第 6 行「不适用」），符合 gate `S1-EQUIV-EXEC` 豁免规则。已确认 `examples/conv1d/_conv1d_group_kernel/verify_equiv.py` **不存在**（`ls` 返回 `No such file or directory`），豁免合法；v1 未新增任何采纳的公式级优化项，故无等价性机器验证对象。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design examples/conv1d/_conv1d_group_kernel/DESIGN.md`）输出：
```json
{
  "design": "examples/conv1d/_conv1d_group_kernel/DESIGN.md",
  "capacity_constants": {
    "ub_kb": 192.0,
    "ub_inflation": 1.7,
    "l0c_kb": 128.0,
    "source": "pattern-library/constants.md CONST-capacity-910B2C"
  },
  "checks": [
    {
      "check": "l1_budget",
      "status": "pass",
      "detail": "L1: 2 buffer entries sum = 48.0 KB; capacity = 512.0 KB",
      "within_inflation_budget": null
    },
    {
      "check": "l0c_budget",
      "status": "pass",
      "detail": "block 64x64 x fp32 accum = 16.0 KB vs L0C 128 KB"
    },
    {
      "check": "core_split",
      "status": "skip",
      "detail": "insufficient parsed dims {'N': 1, 'bm': 64, 'bn': 128}; declared logical cores = 64"
    },
    {
      "check": "r3_metrics",
      "status": "pass",
      "detail": "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"
    }
  ]
}
```
  - `l1_budget` **pass**：单槽 `l1_a(64×128)+l1_b(128×128)` fp16 = 48KB ≤ 512KB（§4.5）。双槽 ping-pong 96KB（Stage 4）亦 ≤ 512KB，人工核对通过。
  - `l0c_budget` **pass**：机械把 `block_M×block_N` 欠解析为 `64×64`（16KB）；设计实为 `64×128×4 = 32768 B = 32KB`（§4.5）——两者均 ≤ 128KB，人工复核通过。
  - `core_split` **skip**（解析到的 `N/bm/bn` 不足，并把与核数无关的 `64` 误当 declared logical cores）→ **人工复核通过**：本算子 `num_logical = GY_g·GX·n·groups`（§5.5 ①）。代表 shape `full-groups2-k3-fp16`：`GY_g=ceil(32/64)=1`、`GX=ceil(128/128)=1`、`n=1`、`groups=2` → `1×1×1×2 = 2`；`full-groups3-coutg24-fp16` → `1×1×1×3 = 3`；说明性大 shape → `8×32×1×2 = 512 = 20×25+12`。三档均自洽，`num_local_tasks` 对 `kernel_id` 为编译期常量、`cid` 钳位使 idle 核重算末 tile（bitwise 一致）。
  - `r3_metrics` **pass**：R3 四口径（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）齐备（§1.6.0）。
- **门禁自检**（`python3 .agents/tools/statectl.py gate 2 --dir examples/conv1d/_conv1d_group_kernel`）：
```json
{"action": "gate", "failures": [], "ok": true, "pass": true, "stage": 2, "warnings": []}
```

## 检视详情

### 0. 源算子理解与迁移分析 — pass

已亲自 Read 源算子全文（`conv1d.py` builder L405-539、`GroupConv1dKernel` L929-1071、op 层 `ops/convolution.py` `Conv1dFwdOp` L75-163）与 §0 逐项核对。

- **语义理解正确性（§0.1）**：§0.1 数学语义 `out[n,c_o,l]=bias[c_o]+Σ_{c_i^g,k_w} W[c_o,c_i^g,k_w]·x[n, g(c_o)·C_in^g+c_i^g, l·s_W+k_w·d_W-p_L]`，`g(c_o)=c_o//C_out^g`——与源码 A/B gather（`oc=group_id*c_out_g+oc_g` L460/L489、`x[batch_id, group_id*c_in_g+ci_g, il]` L481）、`L_out` 公式（源码 L423 `(l_in+pad_left+pad_right-dilation_l*(kernel_l-1)-1)//stride_l+1`）逐字一致。dtype 语义（同 dtype I/O、`accum_dtype="float"` fp32、bias 在 fp32 域相加后统一 cast：源码 L494 `T.cast(out_local[i,j]+T.cast(bias[oc],accum_dtype),dtype)`）与 §0.1 一致。**规约语义**：唯一规约轴 = 单 group `K=C_in^g·K_W`，k 序 `(k_w,c_i^g)`（源码 L462-463 `kw=k_idx//c_in_g`、`ci_g=k_idx%c_in_g`），无跨 L/跨 group 规约——与 §0.1「规约语义」一致。**边界语义**：越界补 0（源码 L479-483 `if_then_else((k_idx<k_total)&(ol<out_l)&(il>=0)&(il<l_in),...,0)`）、输出 M/N 尾块掩码（L487-502）、`c_out_g % block_m == 0` 分支防写进下一 group（L504-514）——§0.1 逐项对应。**I/O（§0.2）**：`x(n,c_in,l_in)`、`weight(c_out,c_in_g,kernel_l)`、`out(n,c_out,out_l)`、`bias(c_out,)`，`out_idx=[2]`；与源码 prim_func 签名（L519-533）及 op 层 `weight: [C_out, C_in/groups, k]` 一致。
- **算法解读完整性（§0.3）**：§0.3 九步覆盖源码全部计算语句——三维 grid（L439-444）、`bz//groups`/`bz%groups`/`oc_base`（L453-455）、A gather（L457-468）、B gather（L470-483）、`T.gemm`（L485）、epilogue（L487-502）、输出溢出分支（L504-514）；无遗漏、无臆造。host 侧：`forward` 直调 `launch`（L1065-1071）、`_build_program` 校验 `c_in%groups==0 and c_out%groups==0`（L1009-1015）、`_check_config` 要求 `block_m/block_k` 为 16 倍数（L1033-1040）、`default_config` 的 `block_m=min{c∈(16,32,64,128):c≥c_out_g}`（L1042-1055）——均已在 §0.2/§0.3 列出。
- **优化手段覆盖完整性（§0.4）**：§0.4 12 项覆盖 SMEM tiling、tensor-core mma、`T.Pipelined`、fragment fp32 累加、**group 折叠进 grid z**（L453-455）、**A 按 K 序 gather**（源码注释 L473-475 明言权重存储序非该序）、B 逐元素掩码 gather、**输出 group 溢出保护**（`c_out_g%block_m` 分支）、coalesced 访问、`T.use_swizzle`（L450）、`threads`（L443）、`num_stages`/`enable_rasterization`——未把实现层（swizzle/threads）误判为算法层，逐个标注硬件依赖。**关键差异识别正确**：group builder **无 host `weight.permute`**（对比 `Conv1dKernel.forward` L870 有 `weight.permute(0,2,1)`，而 `GroupConv1dKernel.forward` L1065-1071 直接 `launch(self,x,weight,bias=bias)`）——§0.1/§0.4 正确指出此点并据此判 A 需 §0.6 R3 前置。
- **耦合性判定合理性（§0.5）**：每条含四态处置（保留/等价替换/重新设计/舍弃）+ 依据。抽检：Tensor core→Cube `T.gemm`（等价替换，有 L0A/L0B/L0C）；SMEM→L1（等价替换）；register 累加→L0C fp32（等价替换）；A/B 逐元素 gather→**重新设计**（Expert Vector 无逐元素条件写）；三维 Kernel→一维 persistent（重新设计）；swizzle/threads→**舍弃**（NPU 无 bank conflict / `T.Kernel(threads=)` 无效）。处置与 `migration-analysis` §5.3 映射及 ascend-constraints 无矛盾，舍弃项理由写明。
- **NPU 重设计可行性（§0.6）**：R1–R5 均有「源方案 → NPU 新算法 → 语义保持论证」三要素，且满足 NPU 硬件约束。**Extra-focus 逐项复核**：
  - **group 轴解码（R1）**：`num_kernels=`（实查 20）一维 `T.Kernel`，`cid=min(task*num_kernels+kernel_id, num_logical-1)`，`num_logical=GY_g·GX·n·groups`；`bz=cid//(GY_g·GX)` → `batch_id=bz//groups`、`group_id=bz%groups`，`oc_base=group_id·c_out_g+by·block_M`。**独立复核**：该映射是与源 (bx,by,bz) 的双射（`bz` 保持源 batch-major/group-minor 序），钳位仅令 idle 核重算末 tile（同输入同指令序，bitwise 一致）；group 间输出通道不相交，无数据依赖。语义保持论证成立。
  - **per-group weight/bias 索引（R3/R5）**：A 片 `w_flat[group_id, kw, by·block_m:+tm, kc·block_k:+kcl]`、B 片 `x_gpad[batch_id, group_id, ...]`、bias 片 `bias[oc_base:+tm]`、out 片 `out[batch_id, oc_base:+tm, ...]`，`group_id` 从不出现在核内 tile 布局——**独立核对**：`w_flat[g,kw,co,ci]=weight[g·C_out^g+co, ci, kw]` 与源 gather（L466 `weight[oc,ci_g,kw]`，`oc=group_id*c_out_g+oc_g`）逐元素恒等；两次 `T.transpose`（`[0,2,1]`→`[1,0,2]`）是纯数据重排，不改变 K 序 `(k_w,c_i^g)` 与累加值。索引正确。
  - **group 通道段 im2col 边界（R4）**：`x_gpad[bn,g,ci,p_L+i]=x[bn,g·c_in_g+ci,i]`，stride=1 时 `off=bx·block_N+kw·d_W`，恒在界内（`off+tn-1≤(out_l-1)+(K_W-1)·d_W=l_in+p_L+p_R-1<l_pad`）；stride>1 用 polyphase `xdec_g`，`base=r_kw·Mdec+bx·block_N+q_kw`。**独立核对**（含 stride>1 相位）：由 `xdec_g[...,rho·Mdec+m]=x_gpad[...,m·s_W+rho]` 及 `x_gpad[...,col]=x[...,col-p_L]`，要求读 `il=ol·s_W+kw·d_W-p_L`，推得 `m·s_W+rho=ol·s_W+kw·d_W`（p_L 自动抵消），故 `rho=(kw·d_W) mod s_W`、索引 `r_kw·Mdec+ol+q_kw`——**与 §3.1 步骤 6 / §3.3 L535-536 的实现完全一致**，且 `l_pad` 为 `s_W` 倍数保证 `s_W·Mdec=l_pad` 无缺口。边界正确。
  - **K-extent 处理（对照函数 3 的已修缺陷）**：本算子 **K 循环为「每 tap 单独一次 `T.gemm`」**——内层 `kw` 每步 A 装 `l1_a[0:tm,0:kcl]`、B 装 `l1_b[0:kcl,0:tn]`，`T.gemm(size=[tmc, kcl, tnc])`，**装入宽度与 gemm K 逐字相等**（`kcl`）；`kcl=min(block_k, c_in_g_pad-kc·block_k)` 因 `c_in_g_pad`、`block_k` 均 16 倍数而为 16 倍数。即本设计**结构上不存在**函数 3（`_conv1d_unit_stride_kernel`）v0 的「装入多 tap、gemm 只取单 tap 宽度」缺陷（§1.6.2 #8、§3.3 L524-549 一致）。K 维总覆盖 = `kernel_l × c_in_g_pad`，其中补零通道（`w_flat` 补零列×`x_gpad` 补零行 = 0×0）对输出无贡献——独立核对通过。
  - **MixCV bias（R5）**：深度 2 双槽 + 4-flag 双向握手，`2*slot`=ws 就绪、`2*slot+1`=槽空闲，**「槽空闲」前导 `set(1)/set(3)` 置于消费者（Vector）`_bias_epilogue` Scope**（§3.3 L571-573；吸收函数 1 Stage 3 `TRAP-sync-self-set-wait` 修正）；bias 经同 dtype `T.copy` + 显式 `T.vcast` 到 fp32 再 `vbrc`+`vadd`+`vcast`（吸收 `TRAP-C12-copy-dtype-cast`）。与 `_ssd_chunk_scan_fwd_kernel.py` 先例及函数 2 实现（`_conv1d_kernel.py` L322-364）同构。
- **设计与决策一致性（§1–§7 ↔ §0.5/§0.6）**：§1.4/§1.6.3/§3/§6 描述的均为一维 persistent + `cid` 解码（含 group）+ `w_flat`/`x_gpad` 前置 + per-tap slice + 分形 clamp + MixCV bias，无「§0 判重设计、§3 照抄源三维 grid / 逐元素 gather」的脱节。§3.3 注 3（group 轴解码单一来源）、注 4（握手单一来源）保证跨章节一致。
- **golden 独立性（§8.1）**：以 §0.1 语义为唯一依据移植源仓基准 `torch.nn.functional.conv1d(groups=groups)`，在 fp32 域计算后 cast——**不复刻** §0.6 的 per-group Cube/persistent/polyphase 数据流，验证独立性成立（源仓 `examples/TileOPs/tests/ops/test_conv1d.py` `test_conv1d_grouped_matches_torch`）。

> 建议级（非阻塞）注 1：§0/§0.7 的源码行号略有漂移（DESIGN 记 builder「L407-560」实际 L406-539、类「L929-1051」实际 L929-1071、`forward`「L1049-1051」实际 L1065-1071）。语义核对全部成立，不影响结论，建议 Stage 1 修订时对齐行号。

### 1. API 可行性 — pass

- §3.2/§3.4 列出的每条 API 均能在本仓 `examples/` 或 `docs/Tilelang.language/` 找到佐证：`T.Scope("Cube"/"Vector")`、`T.alloc_L1`/`T.alloc_L0C`/`T.alloc_ub`、`T.gemm(size=,initC=,b_transpose=)`（`docs/.../T.gemm.md` 确认 `size=[a,b,c]`→src1[a,b]/src2[b,c]/dst[a,c]、fp16 输入 dst 须 fp32）、`T.copy` slice 形态、`T.transpose(permutation)`（`docs/.../创建操作/T.transpose.md` §2.3 确认仅支持两轴交换、>2 轴分解为交换链）、`T.gather`、`T.vbrc`/`T.vadd`/`T.vcast`、`T.rs`、`T.sync_block_set/wait`、`NPUUtils.get().get_aicore_num()`。两处 Cube 迁移先例（`_gqa_prefill_fwd_kernel.py`、`_ssd_chunk_scan_fwd_kernel.py`）与同族已实现 `_conv1d_kernel.py` 逐一印证（后者 L180-364 与本设计 §3.3 伪代码结构同构，仅缺 group 轴解码）。
- 模式匹配：Expert（`T.Scope` + 手动 `T.rs`），与 §2.2 选型一致（Developer/persistent+gemm 有 `TRAP-DEVMODE-PERSIST-GEMM`/`PL-1.16` 阻塞）。
- v-prefix 优先：Vector 段 `T.vcast`/`T.vbrc`/`T.vadd` 均 v-prefix，符合 AGENTS.md API 约定。

### 2. 内存层级规划 — pass

- 搬运路径完整（§4.4）：GM[weight]→UB[wsrc/wt/wdst]→GM[w_flat]；GM[x]→UB[ub2]→GM[x_gpad]（GM 侧偏移）；GM[w_flat]/GM[x_gpad|xdec_g]→L1[l1_a/l1_b]（ND2NZ）→L0C[l0_c]；L0C→GM[out]（fixpipe f32→f16）或 L0C→GM[ws]（f32→f32）；GM[ws]→UB→GM[out]。无 GM→L0 越级路径。
- 每个 buffer 标注层级（§4.3）。
- L0C 预算：`64·128·4=32768 B=32KB ≤ 128KB`（§4.5）✓。L1 单槽 48KB ≤ 512KB ✓（机械 l1_budget pass）。UB：weight 前置段代表 shape ≈4.6KB、`x_gpad` 前置 ≤32KB、polyphase ≤96KB、bias 段 ≈80KB，各段 ≤192KB 且标注为「不同时驻留/可复用」；按 `CONST-capacity-910B2C` ratio 0.98–1.12 口径留量充分（§4.5）。GM 工作区（`w_flat`/`x_gpad`/`xdec_g`/`poly_idx`/`ws`）不计入片上容量。

### 3. Tiling 策略 — pass

- Block 划分明确（§5.2）：`block_M=64, block_N=128, block_K=128`，理由（承接源 default_config 量级、L0C 32KB 留余量、L1 单槽 48KB）写明。
- 对齐约束（§5.3）：K/N 尾轴分形 ≥16（`ceil16` clamp）、`block_N=128` fp16 尾轴 256B（32B 对齐 ✓）、`c_in_g_pad` 16 倍数 ⇒ `kcl` 16 倍数。
- 非整除处理（§5.4/§6.4）：`c_out_g%block_M`、`out_l%block_N`、`c_in_g%block_K` 由分形 clamp + slice 尾块 + 裁剪 store 处理；`c_in_g_pad` 补零通道行/列恒 0。
- **分核策略三要素（§5.5）**：① 逻辑核数 `GY_g·GX·n·groups`（三档实例）；② 物理核数**实查** `NPUUtils.get().get_aicore_num()=20`（附查询代码与返回值，并注明与 `CONST-aicore-910B2C`=24 不同、以实查为准；Cube/混合算子直取不翻倍）；③ 规模判定：代表 shape `num_logical≤20`（无需适配）、大 shape 512=20×25+12（固定启动 20 核 + `T.serial` 核内串行，边界静态）。核数与 block 取值自洽，与 `core-split-strategy.md` §1 一致。
- **group 轴正确性**：group 只进入并行维（`num_logical` 乘 `groups`），不进入核内 tile 布局（§1.6.3 候选 6 弃选），核对通过。

### 4. 技术约束检测 — pass

- 三维 Kernel：涉及，§0.6 R1 给出 `cid` 解码（含 group）方案 ✓。
- GEMM 非整除：在 §5.3/§5.4 处理（分形 clamp + slice 尾块）✓。
- L0C 不溢出：§4.5 32KB ≤ 128KB ✓。
- `T.load_nd2nz` 跨步误读：§3.5.1 标注涉及，改用 slice 形态 `T.copy`（`TRAP-load-nd2nz-strided`/CG-2026-0004）✓。
- Expert Vector 无逐元素 `T.Parallel`：源码 A/B 逐元素 gather 全部改为 slice `T.copy`（R3/R4 前置），无逐元素 Vector 写 ✓。

### 5. 循环与同步 — pass

- 循环结构（§6.1）：`T.Kernel`（核级并行）、`T.serial(num_local_tasks)`（persistent 核内串行）、`T.serial(n_kc)`/`T.serial(kernel_l)`（K 分块累加）、Vector 前置串行——选择明确。
- 同步与模式匹配（§7.1/§7.2）：Expert 手动同步；`T.rs("PIPE_MTE2"/"PIPE_C"/"PIPE_FIX"/"PIPE_MTE3")` 包裹流水段；跨引擎用 `T.sync_block_set/wait`。
- **同步 race-free（Extra-focus）独立复核**：前置就绪 flag 4（Vector 写后 set / Cube 读前 wait，块内）；bias 路径深度 2 双槽 4-flag——初始 `set(1)/set(3)`（Vector 侧）→ 每个 task：Cube `wait(2*slot+1)`→写 ws→`set(2*slot)`；Vector `wait(2*slot)`→读 ws→`set(2*slot+1)`。逐 flag 计数器核验：slot0 序列 `set(1)[前导]→wait(1)[Cube t0]→set(0)→wait(0)[Vec t0]→set(1)→wait(1)[Cube t2]→…` 严格 set/wait 交替，深度 2 使 Cube(t) 与 Vector(t-1) 并发而 slot 不冲突；WAR（Cube 写前 wait）与 RAW（Vector 读前 wait）双向覆盖。与函数 2 已验证实现（`_conv1d_kernel.py` L302-364）同构 → race-free ✓。
- 尾块处理（§6.4）：`tm/tn` 有效区、slice 只搬有效区、`size=[tmc,kcl,tnc]` 分形 clamp、输出裁剪写出、禁止 base+size 形态（`TRAP-T-copy-region-semantics`）✓。

### 6. 验证方案 — pass

- Golden 函数（§8.1）：PyTorch 参考草案 `golden_conv1d_group`（`F.conv1d(groups=groups)`，fp32 域计算后 cast），独立于 NPU 算法 ✓。
- L0 门槛测试计划（§8.3）：覆盖代表 workload（`full-groups2-k3-fp16`/`full-groups3-coutg24-fp16`/grouped smoke）、**stride>1+padding**（polyphase）、dilation>1、batch>1、`has_bias` 两条路径；L0 必测点含非整除尾块、batch>1 且 groups>1、**group 边界不溢出**（`c_out_g` 非 block_M 整除时输出 slice 恰止于 group 末行）✓。
- 精度标准（§8.2）：fp16 1e-2 / fp32 1e-4 / bf16 1.6e-2，与源仓 tolerance 口径一致；已说明分形累加顺序差异落在容差内 ✓。

### 7. 完整性与一致性 — pass

- 章节齐全：迁移类 §0（0.1–0.7 齐全）+ §1–§10（含 §1.6.0/1.6.1/1.6.2/1.6.3），模板要求满足。
- 无占位符：无 `{placeholder}`/`TODO`/`待补充`（§10.2「⬜ 待实现（Stage 3）」为计划内交付物标记，非内容占位）。
- 同类引用真实：§3.5.3 全部 `examples/` 路径已 Glob 核验存在（含 gqa/ssd 先例、`_conv1d_fwd_kernels.py`、probe、四份同族 DESIGN）。
- 内部一致：API 映射、内存规划、Tiling、循环、同步五处描述相互一致（§3.3 注 3/注 4 为 group 解码与握手的单一来源）。

### 8. 算法优化分析 — pass

- **§1.6 章节存在**：§1.6.0 算法调研、§1.6.1 数学等价优化、§1.6.2 向量化替代、§1.6.3 向量化轴与布局决策四子节齐全。
- **算法调研存在性与完整性（§1.6.0）**：R1 候选表**含基线**（候选 0 = per-group implicit GEMM，即源算法），并评估 `algorithm-candidates.md` `ALG-conv` 命中行的 im2col+GEMM / implicit GEMM / Winograd + 分组特有 block-diagonal dense 等共 10 项；R2 在线算法**明确结论「无独立在线变体」**并有结构依据（单遍窗口 K 分块、无 running 统计量）；R3 复杂度表覆盖**四口径**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值），基线在表、口径写清；R4 逐候选六维亲和性评估（计算单元匹配/容量/对齐/静态边界/流水/跨核），并列候选（基线 vs polyphase）有裁决路径。调研范围写明。
- **调研结论正确性（独立复核）**：
  - ① 等价性：选定算法（基线 per-group implicit GEMM + stride>1 polyphase）与源语义等价性**独立推演成立**；polyphase 为精确恒等分解（`(q,r)` 对 `kw·d_W` 唯一），无近似；基线无容差内近似。
  - ② 复杂度算术独立复算：基线 FLOPs `2·C_out·L_out·C_in^g·K_W`——代表 shape 复算 `2·64·128·16·3 = 786432 ≈ 0.79 MFLOP`，与设计一致；block-diagonal dense 为 `groups×基线`（跨 group 全 0 块白算），设计数字正确；设计已计入 `x_gpad`/`w_flat`/`xdec_g` 物化的 GM 往返并按 L2 吸收口径给出 compulsory 下界，未发现漏计中间缓冲 GM 往返。
  - ③ 亲和性判断：与 ascend-constraints、pattern-library 实测条目无矛盾（`CONST-mte2-degradation`/`CONST-L1-port-bw`/`CONST-vector-launch-overhead`/`CONST-capacity-910B2C`/`CONST-aicore-910B2C` 均存在）。
  - ④ **负向断言主动复核**：R2「无在线变体」对照 `algorithm-research.md` §2 结构判据（无 running 统计量的逐窗口映射即天然单遍在线）与 `algorithm-candidates.md` `ALG-conv`（R2「滑窗天然流式」）——**不矛盾，成立**；「无更优等价化简」复核 R1 候选 1–9（含分组特有 block-diagonal），未发现被漏掉的已知候选（Winograd/FFT 已评估并淘汰）。
  - ⑤ 并列候选裁决：基线（stride=1）与 polyphase（stride>1）按 stride 分支裁决（§3.3 JIT 期常量分支），非纸面拍板。
- **调研→设计一致性**：§1.4 算法描述、§3.1 公式拆解、§6 循环结构与 §1.6.0 选定算法（基线 + group 轴并行 + stride>1 polyphase）一致；迁移任务 §0.5/§0.6 处置与调研选族一致，源优化手段意图承接在 §0.6 末段逐项说明（SMEM→L1、tensor core→Cube、Pipelined→L1 多缓冲、fragment→L0C、#5→R1、#6→R3、#7→R4、coalesced→MTE、#10/#11 舍弃）。
- **数学等价优化完整性（§1.6.1）**：六行逐项审视（block-diagonal / Winograd / bias 折叠 / 转置 GEMM / split-K / 除法降代换类「不适用」），每项含原式→优化后→等价论证→收益；**结论「无优化空间」**，依据成立（纯 im2col-GEMM + 加法，无除法/超越函数/公共子表达式/稳定化需求）——符合 gate `S1-EQUIV-EXEC` 豁免。
- **等价性论证正确性（独立推演）**：§1.6.1 五项否决项均无「采纳的变形」，无破坏语义的等价论证；polyphase（结构分解，非公式化简）的等价性独立推演成立（见维度 0 R4 复核）。
- **向量化替代完整性（§1.6.2，与 §3.3/§6 交叉核对）**：12 个循环/标量点全部盘点。逐元素点（A 重排、`x_gpad`/polyphase 构造、bias 相加、输出 cast）**全部向量化**（`T.transpose`/`T.copy`/`T.gather`/`vbrc`/`vadd`/`vcast`）；保留 4 类标量点（cid/group 解码、尾块尺寸与 group 基址、cid 钳位、地址算术）均属 block/tile 级任务映射/动态边界，非逐元素热点。**与 §6 一致**：§6 无逐元素标量循环、无 Vector 逐元素 `T.Parallel`。
- **不可替代理由充分性**：4 类标量点的理由具体到本算子（类别①block 级索引守卫/类别④动态边界），非泛泛而谈。
- **替代 API 可行性**：`T.transpose`/`T.gather`/`T.vbrc`/`T.vadd`/`T.vcast` 均在 `examples/`/`docs/` 有佐证（维度 1 口径一致），v-prefix ✓。
- **向量化轴与布局决策完整性（§1.6.3，含弃选论证前提核对）**：候选矩阵 6 行（K 序 `(k_w,c_i^g)`+group 外层 / K 序 `(c_i^g,k_w)` / 转置 GEMM / host permute / B 轴选择 / group 并入 M 轴），逐候选量化评分（整除性/尾 lane/累加链/repack/UB 影响）。**弃选前提亲自核对**：`T.transpose` 的「核内 UB 级两轴交换链」前提与 `docs/.../T.transpose.md` §2.3 一致（非 GM 级 transpose）；候选 4「host permute 违反 ascend-constraints.md §4（host 禁改输入内容）」——已打开该文档确认 §4 原文禁止 host 侧改动输入张量内容，前提正确；候选 2/5/6 的代价机制（B per-tile gather / 指令数爆炸 `CONST-mte2-degradation` / block-diagonal FLOPs groups 倍）均有量化依据且与 pattern-library 实测无矛盾。**group 轴归属**明确：作为并行维、不进入 tile 布局。迁移任务已对照源码隐式轴选择（GPU 轴为输入非结论）。
- **实验裁决模式**：§1.6.3 判定不依赖未实证常数（布局/轴选择为结构性论证；唯一「未文档化假设」标注在 §1.6.0 R4 Winograd 估算，属已淘汰候选，不影响选定方案）→ 无需三件套。
- **布局决策与实现一致**：§1.6.3 选定布局（A=`w_flat[groups,K_W,C_out^g,C_in^g_pad]`、B=im2col per-tap 连续矩形、dst=L0C[C_out^g,L_out]）与 §3.3 伪代码、§4 内存规划（buffer 形状/UB 预算）、§6 循环结构三方一致。
- **与下游章节一致**：§3.1 以 §1.6.1 优化后公式（=基线）为输入；§6 循环结构与 §1.6.2 结论一致（判定向量化的点无对应逐元素标量循环）。

> 建议级（非阻塞）注 2：§0.6 R4 的 stride>1 等价式书写有笔误——文中写作 `ol·s_W + kw·d_W - p_L = (ol+q_kw)·s_W + r_kw`，其 LHS 多了一个 `-p_L`（`x_gpad` 已含 `p_L` 偏移，该等式应针对 `x_gpad` 列 `ol·s_W+kw·d_W` 而非 `il`）。**实际实现索引（§3.1 步骤 6 / §3.3 L535-536 `r_kw·Mdec + bx·block_N + q_kw`）经独立推演正确、与函数 2 已验证实现逐字一致**，不影响功能；建议 Stage 1 修订时删除该 `-p_L`。
>
> 建议级（非阻塞）注 3：§1.6.3 类别候选矩阵聚焦「布局/轴」变体；卷积类的三种算法族候选（直接跨步窗口 / UB im2col / implicit GEMM）已在 §1.6.0 R1（候选 9 / 候选 1 / 基线）完整评估并给出量化淘汰理由，覆盖成立；建议在 §1.6.3 增设一行交叉引用以免读者误判遗漏。
