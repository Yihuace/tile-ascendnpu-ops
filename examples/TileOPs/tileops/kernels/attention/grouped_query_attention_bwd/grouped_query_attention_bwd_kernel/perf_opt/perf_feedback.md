# 性能反馈（[DESIGN_LIMIT]）

## 触发判定

- 设计层归因: `_gqa_bwd_kernel_DESIGN.md` §0.6 **R2** + §5 分核方案（1-D 非持久 `T.Kernel(nblk)`，每 program =
  一个 KV block、`groups` 个 q head 在核内展开，**单个 program 内顺序执行** 5×`T.gemm` → L0C→UB → ~12 条
  v-op → 2 次 cast → 3×`T.gemm`），以及 §4.5 的 tile 上限（`BM=64` 由 L0C 的 `dk_f`/`dv_f` fp32 预算
  = 2×`BM×D` 锁定，`BN` 受 UB 192 KB 限制）。实测：**即使把两项 BSHD 行粒度访问优化到接近零之后**
  （本轮 `dq` 改 rank-2 区域原子，`aiv_mte3` 220.3→4.8 µs/block、`aiv_scalar_mte2_stall` 333→~6 µs/block），
  剩余墙钟仍有 **70% 是 `aiv_scalar`**（`106.1` / `aiv_time` `150.3` µs per block），cube 占比只从 2.3% 升到
  **5.5%**；`aic_scalar_wait_id2` 从 409 降到 24 后，cross-engine 握手不再是主项，但 cube 仍远未连续忙。
  要达 40% 需 cube 占比 **≥53%**，只能靠"单引擎分相位"把 Vector 工作整体移出 cube 的关键路径（多 pass +
  GM workspace），属数据流/分工重设计，**不是 tiling 或参数可解**；`BM` 上探（行数 ∝1/`BM` 的直接杠杆）被
  L0C 128 KB 与 UB 192 KB 双封顶（`BM=128` 需 `dk_f`/`dv_f` 各 64 KB；宽原子暂存在 `BN=64` 上三种写法编译
  均失败，见"实测"第 3 条）。
- 结构性加速估计: **5.0x**（依据: 同算子、同工具链族的已有 Stage-4 交付
  `examples/multi_head_attention/_gqa_bwd_kernel/perf_opt/opt_log.md` §10.4——3-pass 单引擎相位拆分在
  **Ascend910B2 / 24 核 / tilelang `0.1.2+151c62a` / CANN 9.0.0** 上实测 1016.8 µs（8b-short fp16）/
  7518.6 µs（8b-long fp16），本任务同 workload 现为 1995.8 / 14018.2 µs ⇒ 1.96x；而该参照的**融合基线**
  （结构与本 kernel 同类）2011.6 / 13954.2 µs 按核数归一（×20/24）≈ 2413 / 16745 µs，本 kernel 现为
  1995.8 / 14018.2 µs ⇒ 已相当/略优；两条路径合计 5.0x/5.3x 对应 Ratio ≈8.2%，仍只有目标的 1/4.9）。
  同时满足"实测与设计假设直接矛盾"：§4.5 假设 `BN=64` 不可行（静态 UB 加和 240 KB > 192 KB），
  实测可编译且生效；§0.6 R3 假设 dq 逐行原子是可接受成本，实测它是 51% 的墙钟。

## 参照锚定

- 参照实现:
  1. `examples/multi_head_attention/_gqa_bwd_kernel/perf_opt/opt_log.md`（**同一算子** GroupedQueryAttention
     backward 的 Stage-4 交付档案，含 `perf_records.jsonl`；910B2 / 24 核 / tilelang `0.1.2+151c62a` /
     CANN 9.0.0）。winner = **引擎相位拆分**：pass1 `_gqa_bwd_sdw`（Cube only，S=QKᵀ、dW=dOVᵀ → fp32 GM
     workspace）、pass2 `_gqa_bwd_pds`（Vector only，P/dS → dtype workspace）、pass3 `_gqa_bwd_mul`（Mix，
     dv/dk/dq）；KV 轴分块 + `by0: T.int32` 标量入参使 `dk/dv` 保持单写者。
  2. `examples/deepseek_v4/example_sparse_attn_bwd_kernel.py`（同族 attention backward 的 3-kernel 形态：
     预处理 + dKV pass + dQ pass）。
  3. `examples/deepseek_v32/sparse_mla_bwd.py`、`examples/indexer/indexer_bwd.py`（API 级先例，非结构参照）。
- 不可移植论证（逐条结构差异 + 依据，纯推理项已标注）:
  1. **kernel 数量/分工改变 op 契约**：参照把一次 op 调用拆成 3 个 pass kernel × `n_chunks` 次 chunk
     （其 §10.3/§10.4：8b-long 为 16 chunk、workspace 预算 256 MB）。TileOPs 侧
     `GroupedQueryAttentionBwdOp` 的 kernel 组成与 wrapper 是 Stage 5 交付物，Stage 4 对 wrapper 只读
     （本任务边界）；官方 Ratio 采数按单 op 名捕获（`--kernel-name=_gqa_bwd_main_mix_aic`），拆分后的
     "一次调用 = 多 kernel 时延之和"口径与 `bench_grouped_query_attention_bwd.py` 的单 kernel 捕获不一致。
     → 依据：任务约束 + `perf_opt/_bench_kernel.py` 的过滤形态。
  2. **需要跨 pass 的 GM workspace + 标量 chunk 入参**（参照 §10.3 约束 1）：kernel 签名从 9 个张量变为
     "张量 + workspace + `by0` 标量"，属设计层接口决定。
  3. **需要 `T.pipe_barrier("PIPE_ALL")` 关闭一个已知 WAR 竞态**（参照 §10.3 约束 2：无 barrier 时 512 块
     中 91 块 dS 出错；登记 `TRAP-npuir-vector-kernel-store-war-race`）：纯 Vector kernel 内 UB scratch 的
     跨迭代 WAR 不被自动同步覆盖——把该形态引入本 kernel 属新结构，不是本任务可单点验证的变更。
  4. **纯 Vector pass 的并行度模型不同**（参照 §10.3 约束 3：每 block 只占 1 AIV，需 `2×cores` 核）。
  5. **未实证假设（标注）**：参照的 3-pass 形态在本 kernel 的 BSHD 契约下**未实测**；"拆分后叠加宽块
     （`BM/BN` 128–256）可再削行粒度流量"是基于行数 ∝1/`BM` 的实测外推（本轮 `BN` 32→64 的对照实验
     支持"行数不变则两项 pipe 不变"），非实测。
  6. 器件差异（24 核 910B2 vs 20 核 910B3）只影响绝对时延：融合形态上本 kernel 已与参照归一后持平
     （1995.8 vs ≈2413 µs），说明剩余差距来自数据流结构而非器件。

## 假设

- `_gqa_bwd_kernel_DESIGN.md` §0.6 **R2**：`BM=64`（`dk_f`/`dv_f` = 2×`BM×D` fp32 = 64 KB，
  `qkt/dp` + `dq_f` 合计 24 576 ≤ 32 768 元素）、`BN = 32`（`dim=128`），UB ≈104 KB（**静态加和**口径）——
  即"tile 尺寸由片上容量单点锁定"。
- §0.6 **R3**：`dk/dv` 改"核内全量累加 + 单次写（零原子）"、`dq` 保留"跨 kv block 的逐行 `T.atomic_add`"，
  设计期判定该原子是"唯一跨核写"、成本可接受。
- §5 分核方案：1-D 非持久 grid，每 program 负责一个 KV block、`groups` 个 q head 在核内展开；
  q/kv 按 BSHD 切片 `T.copy`（禁用 `load_nd2nz`）。
- 隐含假设：每次迭代的固定成本（5 次 gemm 发射、2 次 fixpipe、~12 条 v-op、逐行原子）可随元素规模摊薄。

## 实测

口径：`msprof op Task Duration(us)`（kernel-only，20 次 launch 中位），`--kernel-name=_gqa_bwd_main_mix_aic`，
`--aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,Roofline`；设备 Ascend910B3
（`cube_core_num=20` / `vector_core_num=40`，rated 1800 MHz，CANN 9.0.0，msprof 25.5.0）。raw profile：
`perf_opt/profiles/baseline/round0/main/*`、`perf_opt/profiles/final/round3/main/*`、
`perf_opt/profiles/roofline/r0_main/8bshort_fp16/`。

1. **baseline 逐块分解（8b-short fp16，`profiles/roofline/r0_main/8bshort_fp16/`，PipeUtilization 每 block 均值）**：
   `aiv_time` 431.3 µs，其中 `aiv_mte3` 220.3（51%，`dq` 逐行原子）、`aiv_mte2` 141.1（33%）、
   `aiv_scalar` 129.0、`aiv_vec` 28.6；cube 侧 `aic_time` 431.0 而 `aic_cube_time` 仅 10.7（2.5%），
   `aic_scalar_wait_id2_time` 409.1（95% 的 cube 墙钟在等 Vector）。
2. **本轮已消除的项（Stage 4 内可解，因此不构成设计层天花板）**：`dq` 改 rank-2 跨步区域原子
   （`(B·S,H·D)` 视图 + 一次 `[BN,dim]` 区域原子）后 `aiv_mte3` 220.3→**4.8**、`aiv_mte2` 141.1→**18.3**
   （原子流量消失后 GM 通道不再被挤占）、`aic_scalar_wait_id2` 409.1→**23.6**；`BN` 32→64（推翻 §4.5 的
   "静态加和 240 KB 不可行"）再削 scalar/fixpipe。合计 `Task Duration` 5865.6→**1995.8 µs**（−66%），
   官方 Ratio 1.4209%→**4.1385%**（8 档几何均值 1.4610%→4.2385%，2.90x）。
3. **剩余天花板（设计层）**：修完后 `aiv_scalar` = 106.1 / `aiv_time` 150.3 = **70%**，`aiv_vec` 18.2、
   `mte2` 18.3、`mte3` 4.8、cube 实算 8.6（**cube 占比 5.5%**）。即瓶颈已从"访存形态"转为
   **每迭代指令发射 + Developer 模式双 AIV 冗余执行**（`vector0`/`vector1` 的 `mte2`/`vec` 计数一致 ⇒ 两份
   相同工作，pattern-library PL-1.13 结论：Developer Mix 下 `subid` 恒 0，无法分片），这两项都不是
   `BM/BN`/`num_stages` 可解。`BM=128`（唯一能同时削减行数的杠杆）需 `dk_f`/`dv_f` 各 64 KB fp32，L0C 超限；
   宽原子暂存（`[BN,groups,dim]` fp32 = 64–128 KB）在 `BN=64` 上三种写法均编译失败
   （`hivm.hir.copy: Unsupported copy from cc to cbuf` / `hivm.hir.load: src and dst shape should be the same`，
   见 `perf_opt/_gqa_bwd_kernel_r2_atom4h.py`）。
4. **参照对照（同算子）**：参照 1016.8 / 7518.6 µs（910B2，24 核，3-pass 拆分）⇒ counted-flops 吞吐
   24.5 / 24.3 TFLOPS ⇒ **Ratio ≈8.2% / 8.1%**；本轮 12.39 / 12.93 TFLOPS ⇒ 4.14% / 4.32%。
   即：再走完"单引擎分相位"这条设计层路径，本算子在同族最优实现下的 Ratio 上界仍只有 **≈8.2%**。
5. **Roofline 口径**：`computility = 299.36 TOps/s`（40 cube 单元口径），counted flops = 24.7 GF（short）/
   181.2 GF（long），`AI = 1182 ops/byte` ⇒ 绑定项为 compute ⇒ `Ratio = counted_flops/(duration×299.36 TOps/s)`。
   同机 fp16 峰值实测 `torch.matmul(4096³)` = **270.1 TFLOPS** ⇒ 理论 Ratio 上限 90.2%；**40% = 持续
   119.7 TFLOPS = 实测峰值的 44%**，而本仓 attention 族最好水平为 70 TFLOPS（Ratio 23.6%，
   pattern-library PL-1.12 终值，同器件族）。

## 影响面

- 直接受影响：`GroupedQueryAttentionBwdOp` 全部 9 个 benchmark 案例（8 tune + 1 smoke）；pipeline 中
  main kernel 占 99.0–99.7%（pre/post 为带宽地板，`no_gain`）。
- 同族算子：所有 BSHD 布局、需跨核归约 fp32 输出的 attention backward（mha / gqa / mla / sparse-attn），
  以及任何在**单个 program 内顺序执行"cube→vector→cube"链**的 Developer 模式算子——都会遇到同一个
  "每迭代固定发射/握手成本 + 双 AIV 冗余"地板。
- 改设计预期影响：需重开 Stage 1（DESIGN）改走"多 pass 单引擎 + 宽块 + GM workspace + 单写者"，涉及
  Stage 1（数据流/分工/workspace 契约）、Stage 3（新增 pass kernel 与 wrapper 组成）、Stage 5（集成口径
  与 Ratio 采数方式），工作量约等于重做一次 migration 实现。

## 反馈结论

本轮已在 Stage 4 内把两个可解的访存形态问题全部消除（`dq` 区域原子 + 被挤占的装载通道，5865.6→1995.8 µs，
官方 Ratio 1.46%→4.24%，2.90x）；剩余墙钟的 70% 是每迭代指令发射 + Developer 模式双 AIV 冗余，cube 占比
仅 5.5%，而 40% 目标需要 cube 占比 ≥53%（= 实测峰值的 44% 持续算力，本仓 attention 族最好水平的 1.7x）。
同算子已有最优实现（3-pass 单引擎相位拆分）在其器件上给出 Ratio ≈8.2% 的上界，仍只有目标的 1/4.9 ⇒
**在冻结的融合单 program + Developer 双 AIV 结构下 40% 不可达**，必须做多 pass 单引擎相位拆分（op 契约/
pipeline 拆分层的设计改动）才可能接近。

建议路由: 附录补记
