# _conv1d_group_kernel — Stage 1 Retrospective

## Stage 1 (design) Retrospective — first_design — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| none | none | none | none | - |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 分组卷积迁移的「group 轴解码」复用模式：一般 conv 迁移设计中把 `n×groups` 折叠进一维 persistent 的 `cid`，逐 tile 解码 `bz=cid//(GY_g·GX)`、`batch_id=bz//groups`、`group_id=bz%groups`，A/B/bias/out 全按 `group_id` 选片（`oc_base=group_id·c_out_g+by·block_M`） | P | `examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_group_kernel_DESIGN.md` §0.6 R1 / §1.6.3 / §3.3 注 3；源 `src/tileops/kernels/convolution/conv1d.py` L453-455 | 设计期即可复用（无需 GPU/NPU 运行） | tilelang 0.1.2+417dae99 + CANN / Ascend 910B3（实查 get_aicore_num=20） | pattern-library / 待定 |
| group 尾块「溢出保护」的 NPU 等价改写：源用 `if c_out_g % block_m == 0: T.copy(...) else 掩码写` 防写进下一 group；NPU 侧以输出 slice extent 天然承载——`tm=min(block_M, c_out_g - by·block_M)`、`oc_base=group_id·c_out_g + by·block_M`，`oc_base+tm ≤ c_out` 恒成立，无需运行时分支 | P | `examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_group_kernel_DESIGN.md` §0.6 R2 / §5.4 / §9.2；源 L527-537 | 设计期即可复用 | tilelang 0.1.2+417dae99 + CANN / Ascend 910B3 | pattern-library / 待定 |

### Transferable Lessons（可迁移教训）

- 分组算子（Grouped Conv / Grouped GEMM）的迁移可**直接复用同族一般算子（ungrouped）的完整设计**，仅在四处加 group 轴：① 逻辑核数乘 `groups` 并把 `n×groups` 折叠进 `cid`；② A 的 K 序重排按 group 分段做（`w_flat[groups, K_W, C_out^g, C_in^g_pad]`，不跨 group 边界转置）；③ B 从「该 group 的通道段」取片（`x_gpad[batch_id, group_id, ...]`，永不跨 group）；④ 输出/bias 用 `oc_base=group_id·c_out_g + by·block_M` 做行基址，M 维尾块按 `c_out_g`（非 `c_out`）计算以避免写进下一 group。
- 源 GPU group builder **可能没有 host `weight.permute`**（A 在 kernel 内按 K 序逐元素 gather，因权重存储序 `(c_i_g,k_w)` ≠ GEMM K 序 `(k_w,c_i_g)`）；NPU Expert 无逐元素条件写，此类 A 必须改为 `w_flat` 前置（`T.copy`+`T.transpose` 链）一次性物化——迁移前先确认源是否已 host 重排，勿默认套用一般 conv 的「host 重排内移」结论。
- 复用同族设计时，**上游函数的 Stage 3 实测修正必须逐条继承**：本设计继承了函数 1/2 的「跨引擎握手前导 `set(1)/set(3)` 置消费者（Vector）侧」（`TRAP-sync-self-set-wait`）、「bias 显式 `T.vcast`」（`TRAP-C12-copy-dtype-cast`）、「`x_gpad` 偏移放 GM 侧」（`TRAP-copy-ub-offset-stride`）、「slice 尾块禁 base+size 形态」（`TRAP-T-copy-region-semantics`）。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| none | none | none | none | - |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 分组 Conv1d（`groups>1`）在 NPU Expert 上可直接以「同族一般 kernel + group 轴解码」一次落地：把 `n×groups` 折进一维 persistent `cid`，逐 tile 解 `bz=cid//(GY_g·GX)`、`batch_id=bz//groups`、`group_id=bz%groups`，`oc_base=group_id·c_out_g+by·block_M`，A 片取 `w_flat[group_id,kw,by·BM:+tm,kc·BK:+kcl]`、B 片取 `x_gpad[batch_id,group_id,...]`、bias 取 `bias[oc_base:+tm]`、out 写 `out[batch_id,oc_base:+tm,...]`；`tm=min(block_M,c_out_g-by·block_M)`（相对 `c_out_g`）使 group 溢出保护结构化 | D | `examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_group_kernel.py` L180-500；实测 L0 12 + L1 40 + L2 16 + Boundary 8 全 PASS（首次尝试，0 调试迭代），配置 groups∈{1,2,3,4,8}、stride 1/2/4、dilation 1/2、pad、kW 1/3/5/7/10、c_out_g∈{8,16,17,24,32,48,50,96,168,256,320} | `examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_group_kernel.py` | tilelang 0.1.2+417dae99 + Ascend 910B3（get_aicore_num=20） | pattern-library / 待定 |

### Transferable Lessons（可迁移教训）

- 分组算子的 Stage 3 实现**只需在已通过验证的同族一般 kernel 上做 4 处增改**即可一次通过（本函数 0 调试迭代）：① `num_logical = gy·gx·n·groups`；② `bz` 后追加 `batch_id/group_id` 解码、`oc_base` 基址；③ weight/bias 前置循环外包一层 `for g in groups`，`weight[g·c_out_g+…]`、`w_flat[g,…]`；④ A/B/bias/out 的 slice 基址全部换成含 `group_id`/`oc_base` 的版本。其余（tap-major K、`T.gemm initC`、尾块 clamp、polyphase、4-flag bias 握手、消费者侧前导）与一般 kernel 逐字相同。
- 复用同族已通过验证实现时，**保持接口形参序列与函数 2 一致**（`w_flat`/`x_gpad`/`xdec_g`/`poly_idx`/`ws` 作为 prim_func 形参由 wrapper 分配）可最大限度继承其正确性；本算子唯一新增的接口差异仅 `weight` 形状 `(c_out, c_in_g, kernel_l)` 与 `w_flat` 增一维 `groups`。
- 「group 溢出保护」在 NPU 侧**不需要运行时分支**：把 M 维尾块 `tm` 相对 `c_out_g`（而非 `c_out`）计算，输出 slice 自然止于本 group 末行——这比移植源 `c_out_g % block_m` 分支更简单且被 L0 `c_out_g=96`（非 `block_M` 整除）与 `c_out_g=24` 两组实测覆盖。
- bf16 分组用例的 `max_abs` 可达 ~0.25（`c_out_g`/`c_in_g` 大、累加 K 大时），但 `assert_close` 的 `rtol`（bf16 1.6e-2）约束相对误差，仍全 PASS——迁移验证不要用裸 `max_abs` 门限，须用 `atol+rtol` 相对判据。
