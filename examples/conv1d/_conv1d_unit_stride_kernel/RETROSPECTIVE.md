# Stage 1 复盘 — `_conv1d_unit_stride_kernel`（Conv1dFwdOp 函数 3/5，stride=1 特化 implicit-GEMM conv）

## Stage 1 (design) Retrospective — first_design — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | **同族「一般设计」与「特化设计」的可复用结构未被技能显式建模**：函数 2（`_conv1d_kernel`）与函数 3（`_conv1d_unit_stride_kernel`）共享 90% 结构（Expert Cube + MixCV + persistent + tap-major），差异仅在「stride=1 删除 polyphase」「K 分块粒度」。skill 没有"从已通过的同族设计做减法特化"的流程指引，只能靠人工识别可删项 | 本任务 §0.5/§0.6 对照 `examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_kernel_DESIGN.md` 逐条判定；删除 xdec/poly_idx/T.gather/T.Parallel 全路径 | `tilelang-op-design` SKILL.md 增补「同族特化任务」分支：当 `examples/{project}/` 已存在同族已通过设计时，Phase M0/M1 以该设计为基线做**减法/加法对照**（保留项、删除项、新增项三分），并在 §0 标注差异清单 | R |
| Info-source | **GPU 源 builder 的 `output_l` 公式在 stride=1 时省略 `//stride + 1`**（L169 `out_l = l_in + pad + padR - dilation*(kernel-1)`），与同文件一般 kernel（L47 带 `//stride+1`）写法不同，但 stride=1 时数学恒等——解读时须自行验证等价而非照抄 | 源 L169 vs L47；F.conv1d 参考实现反查确认等价 | `references/migration-analysis.md` §3.1 增一条：读源算子时须逐条验证"看似不同的公式"是否在限定条件下恒等（避免误判语义差异或漏判） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **stride=1 conv1d 的特化减法清单**：相对一般 conv（函数 2），stride=1 可删除全部 polyphase/`xdec`/`poly_idx`/`T.gather`/`T.Parallel` 路径（B 装载退化为 per-tap 连续矩形 slice `T.copy`），且 Expert Vector 段因无逐元素写而**天然规避函数 2 Stage 3 的 `T.Parallel` 崩溃陷阱** | P | `examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_unit_stride_kernel_DESIGN.md` §0.5/§0.6/§1.6.2/§9.1；`examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_kernel.py` L219-236（need_poly 分支） | `examples/conv1d/probe/conv1d_geomm_probe.py` L94-98（stride=1 per-tap slice 实证） | tilelang 0.1.2+417dae99 + Ascend910B3（`get_aicore_num`=20）+ CANN | pattern-library / cases.md |
| **`x_pad` 连续前置同为本算子选用的边界零填充方案**（stride=1 时快慢路径合一为 in-bounds slice）：host `torch.zeros` 预零 + Vector 拷入域内（GM 侧偏移），换取主循环零掩码——代价 `≈2·c_in·l_in·eb`（whisper +9% 流量） | P | DESIGN.md §0.6 R4/§4.4/§9.1；`examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_kernel.py` L200-218 | `examples/conv1d/_conv1d_kernel/repro/TRAP-copy-ub-offset-stride.py`（GM 侧偏移断言） | 同上 | pattern-library / cases.md |
| **tap-major 通道分块（一个 tap = 一个 K 子块，L1 目标偏移 0 或 block_k）在域内公式层面天然正确且可服务任意 dilation/padding**：`off = bx·BN + kw·d_W` 恒在 `[0, l_pad-BN]`（因 `out_l = l_in+p_L+p_R-(K_W-1)·d_W`），无需动态 clamp | P | DESIGN.md §0.6 R4/§5.4；源 `_conv1d_unit_stride_kernel` 的 `channel_blocks`/tap-major（L240-270） | 探针 `examples/conv1d/probe/conv1d_geomm_probe.py`（`L_PAD = L_IN+12` 覆盖右溢出） | 同上 | pattern-library / cases.md |

### Transferable Lessons（可迁移教训）

- **迁一家族的多个 builder 时，先读「已通过的同族设计 + 其 Stage 3 实现」，再做减法特化**：函数 2 的设计与实现（含 Stage 3 实测陷阱）是函数 3 的最高价值输入——本任务据此直接删除 polyphase 全套路径，并预置了「消费者侧前导」「GM 侧偏移」「单比较 initC」「精确 kcl」等 Stage 3 校正，避免重蹈覆辙。
- **stride=1 是「让 slice 回归合法」的条件**：GM→L1 ND2NZ 要求区域最内维连续，stride>1 的列采样破坏该前提（须 polyphase）；stride=1 时每 tap 窗口是连续矩形，直接 slice 合法。设计时先判 stride，再决定 B 装载形态。
- **`x_pad` 连续前置把「快/慢路径」合一**：一般 kernel 在 GPU 上有 interior 快路径与逐元素掩码慢路径；NPU Expert 无逐元素条件写，故统一为 `x_pad` + in-bounds slice——**设计层应主动合并分支**，不要照搬 GPU 的快慢双路径。
- **公式在不同限定条件下可能书写不同但恒等**（如 `out_l` 的 `//stride+1`）：读源算子时须逐条验证限定条件下的等价，不可因写法不同而误判为语义差异，也不可漏判真正的差异。
- **门禁 REF_PATH_RE 会把相邻目录词连写当路径**（`examples/docs`）：佐证两个目录时写「本仓 examples 与 docs」，不连写（与函数 2 复盘一致）。

## Stage 1 (design) Retrospective — revision v0 → v1 — 2026-10-14

> 触发：Stage 2 REVIEW.md「结论: 不通过」（1 阻塞 + 2 建议）。修正见 `DESIGN.md` §0.0。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | **「装入宽度」与「消费端 extent」的一致性未被技能作为显式自检项**：v0 的 K 循环把一个 k tile 装入 `taps` 个 tap 的 A/B 片，但 `T.gemm` 的 K extent 写成单 tap 的 `kcl`——装多取少，`taps=2` 静默丢弃后续 tap。skill 的 §3.3 自检清单未要求「凡 L1 装入的 K 宽度，必须逐字等于消费该 buffer 的 `T.gemm` 的 `size[1]`」 | `DESIGN.md` v0 §3.3 L483-498（装入 `taps` 份、gemm `size=[tmc,kcl,tnc]`）；REVIEW.md 问题 1 | `tilelang-op-design` SKILL.md 的 §3.3/§6 自检增补一条「K 装入-消费宽度一致性」检查：`装入 K 宽度 == gemm size[1]`（含 `taps`/分组维），并把它列为 §0 重设计项的强制交叉核对 | R |
| Tooling | **Gate 1 的 `_section` 会把 markdown 代码块中以行首 `#` 起的注释误判为标题**：修订时若在 §5 伪代码块内新增行首 `#` 注释，会把 §5 提前截断，导致 `design_calc_check` 的 `l0c_budget`/`core_split` 从 pass 退化为 skip（机械复核失真） | 本任务：§5.2 代码块内两行 `# 单一来源约束...` 使 §5 section 仅 17 行（v0 为 73 行），`l0c_budget` 由 pass 变 skip；改为块外 bullet 后恢复 | `design_calc_check.py._section` 增加「代码围栏内不识别标题」的围栏跟踪，或文档侧约定「§5 伪代码块内不用行首 `#` 注释」（本轮采用后者） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **K 装入-消费宽度单一来源不变式**：在 Cube 迁移设计里，`T.gemm` 的 K（`size[1]`）必须逐字等于该步在 L1 装入的 K 宽度；分组维（如 `taps`）必须同时体现在装入宽度与 gemm extent——否则静默丢 tap/通道 | P | `DESIGN.md` §0.0/§3.3 注 3/§5.2/§5.4/§6.1/§6.2/§6.4；v0 §3.3 缺陷；对照函数 2 `examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_kernel.py` L257-301（每 (kc,kw) 一次 gemm，装入=消费一致） | 回归：`taps=2` vs `taps=1` 同 shape `assert_close`（§8.3 必测点 ⑤） | tilelang 0.1.2+417dae99 + Ascend910B3 + CANN | pattern-library / cases.md |
| **L1 分片偏移的 16 对齐条件应显式推导而非依赖隐式整除前提**：`kcl` 为 `c_in_pad` 与 `block_k`（均 16 倍数）之差/较小者，必为 16 倍数，故组内连续打包偏移 `t·kcl` 可寻址——设计应写出该推理链，避免如 v0 那样遗留未声明的「`block_k` 整除 `c_in_pad`」前提 | P | `DESIGN.md` §0.6 R4/§5.4/§9.1；REVIEW.md 问题 1 修改建议 | `examples/conv1d/probe/conv1d_geomm_probe.py`（per-tap L1 slice 实证） | 同上 | pattern-library / cases.md |

### Transferable Lessons（可迁移教训）

- **`T.gemm` 的 K 只看 `size[1]`，与 L1 buffer 的物理 K 维无关**：设计时务必把「装入多少 K」与「gemm 收缩多少 K」当成同一个量维护（单一来源）；多 tap/通道分组的加速项最容易在此处「装多取少」。
- **连续打包优于固定步长打包**：按 `t·kcl`（实际尾块宽）连续打包，既消除 `kcl<block_k` 的 gap，又避免「必须整除」的附加前提；偏移仍 16 对齐可寻址。
- **数值前提要显式化**：补零列/补零通道必须显式预零（`torch.zeros`），不要依赖「未初始化垃圾恰为有限值」——`0 × NaN = NaN` 会污染有效输出。
- **UB 预算按「同时驻留」算**：同一 `T.Scope` 内多个 `T.alloc_ub` 的峰值是求和，不是取最大（除非在伪代码层确证复用）。
- **修订时保护机械复核工具的 section 解析**：避免在 §5 等被工具按标题切分的区域内的代码块里写行首 `#` 注释（见上表 Tooling 行）。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-14

> attempt 1；Debug loop = 0（首次编译即通过，L0/L1/L2/Boundary 全过）。高频复用函数 2 的 Stage 3 实现是 0 调试成本的主因。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | **§3.3 伪代码把 `@tilelang.jit(out_idx=[2], target="npuir")` 放在外层 shape-builder 上**（外层返回内层 `_factory`），与函数 1/2 Stage 3 已通过的工作形态不一致——工作形态是「外层 plain builder 返回内层 `@tilelang.jit` 的 tiling 工厂」（`kernel = builder(n,...)(block_m,block_n,block_k,taps,num_stages)`）。若照抄 §3.3 装饰器位置，`tilelang.jit` 的被装饰函数将返回一个 Python 函数而非 `prim_func` | 本任务按函数 2 `_conv1d_kernel.py` L169-170 的嵌套（inner jit）实现；DESIGN §3.3 L437-441 形态未被采用 | `tilelang-op-design` §3.3 模板与「同族特化」指引统一 factory 嵌套约定（外层 plain / 内层 `@tilelang.jit`），避免伪代码装饰器位置歧义 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **tap-MAJOR × 单 gemm 覆盖多 tap 的 K-extent 单一来源不变式在硬件上验证成立**：`T.gemm` 的 `size[1]` 逐字等于本步 L1 打包宽度 `taps*kcl`（第 t 个 tap 页写 L1 K 轴偏移 `t*kcl`，`kcl` 为 16 倍数 ⇒ 偏移可寻址）时，`taps=2` 与 `taps=1` 输出**逐位相同**（cross_max_abs=0.000e+00，bias/nobias×f16/bf16 四种组合均 0），证明 v0「装多取少丢 tap」缺陷被完全消除且无残留 gap | D | 本任务 `_conv1d_unit_stride_kernel.py` §`_cube`；L0/L1 `taps2` 用例 cross=0.0；`examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_unit_stride_kernel_DESIGN.md` §0.0/§3.3 注 3 | `examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_unit_stride_kernel.py`（`_run_taps_consistency`，`taps=2` vs `taps=1` assert_close + vs golden） | tilelang 0.1.2+417dae99 + Ascend910B3（`get_aicore_num`=20）+ CANN | pattern-library / cases.md |

### Transferable Lessons（可迁移教训）

- **同族特化任务的最短路径 = 复制已通过的同族 Stage 3 实现做「减法」**：本函数直接从 `_conv1d_kernel.py` 派生，删除 `need_poly`/`xdec`/`poly_idx`/`T.gather` 全套 polyphase，并把 K 循环从「每 (kc,kw) 一次 gemm」改为「tap-major 分组 + 单 gemm 覆盖 `taps*kcl`」。函数 2 实测的消费者侧前导、GM 侧偏移、单比较 `initC`、精确 `kcl` 全部原样继承生效——**0 调试循环**。迁移家族时务必先读同族已通过实现。
- **tap 分组打包时，L1 分配按物理上限 `taps*block_k`、写入/读取按精确 `t*kcl`**：物理 buffer 需容纳最大块（`kcl==block_k` 时 `t*block_k`），而每步只写 `[t*kcl, t*kcl+kcl)` 并令 gemm K 同为 `taps*kcl`——`kcl<block_k`（通道尾块）时连续打包自然无 gap。切勿把物理维宽当 gemm K。
- **`taps=2` 的正确性门禁应是「与 taps=1 逐位/容差一致」而非仅「vs golden 容差内」**：本任务实测 cross=0.0，若 v0 形态（gemm K 取单 tap）则 cross≈golden 幅值量级、必然门住。把该断言固化进 L0 必测点。
- **设计伪代码的装饰器/嵌套形态不可盲抄**：DESIGN §3.3 把 `@tilelang.jit` 放在外层 builder；以「已通过同族实现的实际工厂嵌套」为准（外层 plain + 内层 jit）。设计层伪代码与实现层约定不一致时，取实现层已验证形态。
- **NPU 实查核数在本机为 20（Ascend 910B3）**，与文档常数 24 不同；所有分核以 `NPUUtils.get().get_aicore_num()` 实查为准（复用同族实现已一致）。
