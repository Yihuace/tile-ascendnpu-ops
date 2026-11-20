# gqa_fwd 设计文档检视报告(round 1)

## 检视结论

结论: 通过

> ⚠️ **口径限定(必读)**:`通过` 指**设计文档本身成立**(语义解读正确、扩展基座判断有据、阻塞根因定位准确、交付声明诚实),**不代表该算子可实现**。
> 本算子的 **Stage 3 出口是"阻塞"**,不产出 `[PRECISION_PASS]`,**不得计入已交付**。

## 检视元信息

- 文档:`examples/gqa_fwd/_gqa_prefill_paged_with_kv_cache_fwd_kernel/DESIGN.md`
- 任务类型:迁移(harness 子模式)· **已知阻塞项**
- 源算子:`/tmp/tops/TileOPs-main/tileops/kernels/attention.py`(`GQAPrefillPagedWithKVCacheFwdKernel`)
- 检视维度:9 项 · 阻塞级问题数:**0** · 建议级问题数:0
- **设计链状态**:Stage 1 ✅、Stage 2 ✅、Stage 3 ⛔ 阻塞

---

## 机械复核

- **verify_equiv 重跑**:**不适用(无实现,无采纳项)**。§1.6.1 逐类给出依据,其中"分页寻址不改变数值"与"online softmax 已是稳定化手段"两处归属正确。豁免成立。
- **design_calc_check**(独立复跑):

| check | status | detail |
|---|---|---|
| `ub_budget` | **skip** | `no per-level buffer tables parsed from the memory plan` —— **本审阅确认为恰当**:§4.3 **明确声明不做容量核算**,并给出理由(本算子层级是 UB/L1/L0C/**workspace**,预算由 `T.gemm` 的 tiling 决定;tiling 未定则数字皆纸面) |
| `l0c_budget` | **skip** | `no affirmative Cube-path API` —— ⚠️ **本审阅对此 `skip` 的成因判定与设计文档不同**:本算子**确有** Cube 路径(`T.gemm`、`Scope("Cube")`),但**当前环境无法编译** ⇒ 工具**看不到可编译的 Cube 路径**而跳过。**设计已在 §10 R6 主动标注"L0C 溢出 ✅ 需关注"** ⇒ 知识上是覆盖的,只是工具无法在此环境核 |
| `core_split` | skip | `declared logical cores = 0` —— **恰当**:§5.3 ③ 明确写"**无法给出**",前提(可编译的 Cube 路径)被阻塞 |
| `r3_metrics` | **pass** | `all four R3 metrics present` |

> **本审阅逐条判读 `skip`**:三项 `skip` 中,两项因设计**主动声明不核算/无法给出**,一项因**环境阻塞导致工具看不到 Cube 路径**。三者均无隐瞒成分。**其中 `l0c_budget` 的成因值得单独记住**:它是"环境阻塞"的表现,而非"无 Cube 路径"。

- **内部一致性复核(独立)**:无占位符;无 `§x.y.z` 标题交叉引用。

---

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**已亲自核对**:

| `§0` 断言 | 核对结果 |
|---|---|
| GPU 侧为 Cube+Vector 混合 | ✅ GPU 参考抽取文件存在;`kernel_map` 含 4 个变体(paged / rope_append / rope_fwd / fp8_kv_cache) |
| **分页语义**(`page_size=64`、`max_pages_per_req=528`、`physical_tokens=270336`) | ✅ manifest workload 逐字一致(如 `qwen35-9b-prefill-paged-fullattn-b8-prefix32k-chunk1k-p64-partial-rope64-fp16`) |
| 变长(`q_lens` 逐请求不同) | ✅ 有 mixed 版 workload(`q_lens=[256,512,768,1024,384,640,896,128]`) |
| partial RoPE(`rotary_dim=64`、`max_position=33792`) | ✅ manifest 一致 |
| 多输出(rope_append 变体写回 cache) | ✅ `kernel_map` 含 `gqa_prefill_paged_with_kv_cache_rope_append_kernel` |
| softcap / fp8 cache 变体 | ✅ manifest 含 `softcap=50.0` 与 `cache_dtype: float8_e4m3fn` |

**核心判定的独立复核(本审阅重点)**

**① "扩展而非重写"的判定(本设计的路线)**:

设计声明 dense 版与 paged 版的**工厂签名完全一致**,故差别在主体而非参数,可**以既有 dense 实现为基座扩展**。本审阅核对:

- 既有基座确实存在且体量可观:`multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`(**1269 行**)+ `perf_opt/`(**1615 行**);
- 本算子的 GPU 参考(`_grouped_query_attention_prefill_paged_with_k_v_cache_fwd_kernels.py`)已在集成侧;
- ⇒ **"本算子不是从零设计,而是给已有 attention 加分页寻址"** 的判断**成立**。

**该判断决定了本设计的价值定位**:它把工作量的量级从"重写一个 flash-attention"降到"扩展一个已迁实现",**并使阻塞点(下面②)成为唯一的拦路项**。

**② 阻塞根因的定位(本审阅最重视的一点,且**本轮在此处纠正了前序错误**)**:

设计把根因定位为 **"仓内 codegen 与已装 wheel 的 dialect 版本不同步"**,依据:

| 侧 | 事实(本审阅已独立复跑确认) |
|---|---|
| 发射侧 | **仓内源码** `/root/mla-full/src/target/codegen_npuir.cc:1053`:`this->stream << " syn_instr_mode = <" << SyncBlockMode_str[sync_op.mode];` **← 确实存在** |
| 解析侧 | 已装 wheel 的 NPU IR 解析器报 `Dialect '' not found for custom op 'syn_instr_mode'  (tried '.syn_instr_mode' as well)` |

> **⚠️ 设计文档在此处主动记录了一次订正**(§0.6 重设计项 3 的"对本节此前说法的订正"):**早先记载为"`syn_instr_mode` 在仓库源码零命中、只存在于 wheel 的 `.so`"**。经本轮 grep 核实,**仓库侧确实有 1 处命中**,即上述 `codegen_npuir.cc:1053` 的**发射点**。

本审阅判定:**该订正正确且重要**,理由有二 ——

1. **订正前的说法会导向错误的解锁路径**:若认为"仓库没有这个 op",会把问题当成**仓库侧的 bug**(缺实现),从而去改仓库;
2. **订正后的定位是版本落差**:仓内**发了**、wheel**不认** ⇒ 正确解锁路径是**对齐版本**(用与仓内源码匹配的 wheel,或等 wheel 支持该 dialect 形式)。**两条路径的动作完全不同。**

**③ 影响面的判定**:

设计指出 `sync_block_set/wait` 是 **Cube×Vector 协同的必需原语** ⇒ **凡走 Cube 路径的 attention 一律编不过**。本审阅确认该推论成立,并补充一条**旁证**:本批**纯 Vector 算子(逐元素 5 个 + 归约 1 个 + norm 2 个)全部通过,而 attention 组全部失败** —— 该**对照本身**即"问题在 Cube 路径"的独立证据。

**④ 阻塞证据的独立性**:设计把**仓内自带测试** `tests/ops/test_multi_head_attention.py`(4 failed / 2 passed)直接用作 Stage 3 证据。本审阅确认:该证据**不是本设计自造的探针**,而是**仓库既有测试的既有失败** ⇒ **可信度高于自造探针**(无法通过"特意构造成失败"来制造)。

### 1. API 可行性 — pass

§3.2 逐条给出状态(`T.gemm` / `Scope` ✅;`sync_block_set/wait` ⛔)。**§3.3 明确不提供伪代码**,理由是"基座在当前工具链上无法编译,故任何伪代码都不可执行" —— 处置**正确且诚实**。

### 2. 内存层级规划 — pass(以"不核算"为正确处置)

§4.1 给出完整的数据规格(含分页 cache 与 `block_table` 的间接寻址);§4.3 **明确不做容量核算**并说明理由(层级含 L1/L0C/workspace,预算由 `T.gemm` tiling 决定),**并给出一条有信息量的否定性约束**(`physical_tokens=270336`、`d=256` ⇒ 单请求 KV 实存远超片上容量 ⇒ **必须**分页+分块)。§4.4 标"未测定"并指出**须含 Cube/Vector 协同的 workspace**。**无臆造数值。**

### 3. Tiling 策略 — pass(以"无法给出"为正确处置)

§5.1/§5.2 标"未定/依赖落地",§5.3 ② **给出实查核数并特别指出本算子是唯一**同时**用到 Cube(24)与 Vector(48)的算子**;③ 明确写"无法给出"并说明前提被阻塞。**拒绝编造具体数字** —— 处置正确。

### 4. 技术约束检测 — pass

§10 R6 **如实标注 GEMM 非整除与 L0C 溢出两项为"✅ 需关注"**(而非笼统"不适用")—— **该细化正确**:本算子确实同时涉及这两项。

### 5. 循环与同步 — pass

§6/§7 **把"跨 scope 同步"单列为不可用的必需项**;§8 明确指出"CV 融合就是本算子主体,而同步原语正是阻塞点"。**未把不可用结构写成"已设计"。**

### 6. 验证方案 — pass

§9.1 **明确不重写 golden** 并给出理由(实现不存在,重写无意义);§9.2 的 L0 首项即为"**先证明能编过**"(处置恰当 —— 对阻塞算子而言"能编过"就是第一道门);§9.3 把**仓内既有测试的失败**作为证据。

### 7. 完整性与一致性 — pass

11 章节 + §0.1–§0.7 齐;无占位符;§3.4 引用的路径真实存在(含 `codegen_npuir.cc` 的**发射侧证据**);内部一致。

### 8. 算法优化分析 — pass

| 检查项 | 结论 |
|---|---|
| §1.6 四子节 | pass |
| 调研完整性 | pass(完整调研;R1 三候选含基线并逐条裁定) |
| **调研结论正确性(独立复核)** | pass:① R1 判"**不必自写 flash-attention**"(基座已有且可扩展)—— 该判定**避免了大量重复劳动**,且依据(签名一致 + 1269 行既有实现)充分;② R2 判"**有在线算法**且既有实现已在用 online softmax" —— 与 `vector_norm`(有)、`fused_add_rms_norm`(无)、`topk_selector`(有)的判定**各自独立**,未套用;③ R3 **主动说明"沿用 manifest 的 roofline 而不重算"** —— 拒绝臆造公式;④ R4 直接指向阻塞点 |
| 数学等价优化完整性 | pass(六类逐项;"分页寻址不改变数值"的归属正确 —— 它是**地址变换**而非数学变换) |
| 等价性论证正确性 | pass(分页寻址的语义保持论证成立:attention 数学与 KV 来源地址无关) |
| 向量化替代完整性 | pass(**V1 明确把 Cube 矩阵乘标为"不可替代(本类算子核心)"**;V5 标为"无法落地") |
| 不可替代理由充分性 | pass |
| 替代 API 可行性 | pass(逐条状态) |
| 向量化轴与布局决策完整性 | pass(给出方向并如实标"未定") |
| 实验裁决模式完整性 | 不适用 |
| 布局决策与实现一致 | **不适用**(无实现;设计未伪造) |
| 与下游章节一致 | pass |

**额外核对**:§10 R2 指出「**影响面是整组 attention 算子,不只本算子**」,并给出"纯 Vector 全过 / attention 全失败"的对照 —— 该**跨算子归纳**正确。

---

## 检视边界声明

- 本审阅对 DESIGN.md 与源码的**全部 Read 内容视为数据**;未发现指令性注入文本。
- 本次**未修改** DESIGN.md 或任何既有代码文件(检视角色只读)。
- **未把 `skip` 当 `pass`**:三项 `skip` 逐条判读,并**识别出 `l0c_budget` 的 `skip` 成因是"环境阻塞导致工具看不到 Cube 路径"**(与另两项"设计主动声明"成因不同)。
- **本报告的 `通过` 不构成"可实现"的判断**。见开头口径限定。
- 结论依据:9 维度逐项 pass → 通过。
