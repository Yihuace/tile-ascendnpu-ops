# _gemm_w4a16_kernel 算子设计文档

> 迁移任务（harness 模式，project=`gemm_w4a16`，op=`_gemm_w4a16_kernel`，编程模式 Developer）。
> 源算子：`<gpu_repo>/tileops/kernels/gemm_w4a16.py` 中的 `_gemm_w4a16_kernel`（GPU，W4A16 分组仿射反量化 + A16 Tensor Core GEMM）；GPU op 类 `<gpu_repo>/tileops/ops/gemm.py` 的 `GemmW4A16Op`。
> 迁移决策（§0）驱动 §1–§10 全部设计；算法调研（§1.6.0）先于算法级优化（§1.6.1–§1.6.3）。
> 探针证据（D-3，2 个，见 §0.7/§1.6.2）：`history_version/design_probe_p1_int4_unpack.py`、`history_version/design_probe_p2_dequant_recipe.py`。

## 修订说明（v2 · revision_index=2）

> 本节记录本版相对上一版（`history_version/design_v1.md`）的调整。上一版首轮阻塞项 `num_launched` 已修复（`num_launched = num_logical`，§0.6 R3/§3.3/§5.5 三处一致），本版据 Stage 2 复检结论继续修订下述 5 项；**已成立结论一律保留**（融合反量化 Cube 直接法、cast-only INT4 解包、E1–E4、13/13 `EQUIV_PASS`、`num_launched = num_logical`）。

**相对上一版的关键调整**

1. **§1.6.0 R1 补齐 `ALG-gemv` 命中行必查候选**（复检阻塞项）：补入候选 9「**Vector 规约累加（m=1 gemv 形态）**」，并在 R3 补其四口径（FLOPs 真值 `1.34e12` 无分形填充 / 访存同基线 `361.9 MB` / 扫描遍数 1 / 片上缓冲同量级）、在 R4 补其六维评估（MAC 密集段落 Vector、吞吐劣势、无访存收益、仅 m=1）；给出**量化淘汰依据**（复用 §1.6.0 D-2：m=1 decode 下 GM 流量 ≈287 µs 主导，候选 9 不降低主导项，Cube 填充实算 ≈72–107 µs 本就被流量项掩盖）；同步更新「调研范围」「关键依据」「§1.6.1 其余候选」「§9.1.2-5」。
2. **UB 数值口径全文统一**：§1.6.0 R3/R4/D-2 与 §1.6.3 残留的「UB ≈59 KB / ≈117 KB / ≈131 KB」改为引用 §4.5 逐 buffer 表 —— 2-stage 合计 **150.0 KiB**、膨胀后 **≈165–168 KiB**（< 192 KiB）；同时把 §1.6.0/§4.5/§5.3 的 `L0C 8 KB` 更正为 **4.0 KiB**。
3. **`deq_N` 分块自洽**（§5.2/§5.5/§6.1/§9.1.2/§9.3）：引入不变式 **`deq_N | block_N` 且 `deq_N % 16 == 0`**；明确 `W_sh` 是 `T.gemm` 的 src2、**必须整 `block_N`、不随 deq 分块缩小**；`block_N=208` 配套 `deq_N=16`（`208 // 16 = 13` 段，全覆盖）；**重算**该配置 UB 峰值 **≈94–96 KiB**（含整块 `W_sh` 53,248 B）；删除错误表述「UB 峰值降到 ≈60 KB」并注明其作废原因（`deq_N=32` 仅覆盖 192 行且未计入 `W_sh`）。
4. **§4.5 单位口径注释**：显式声明 `1 KiB = 1024 B`、容量上限按二进制口径（UB 192 KiB = 196,608 B）；全文其余位置只引用 §4.5 汇总值。
5. **§0.3 chunk 循环次数更正**：源码为 `(block_n*block_k//2) % (threads*4) == 0`（整除，L49–L50），默认 `2048 = 4×(128×4)` ⇒ chunk 循环 **4 次**（原写「`== threads*4` ⇒ 1 次」为算术错误）。

**为何不会再犯同一错误**

- **命中行必查**：R1 现在逐候选建行——`algorithm-candidates.md` 的 `ALG-gemv` R1 行两个候选（Cube 分形路径 = 基线 0；Vector 规约累加 = 候选 9）均已进入 R1/R3/R4 量化对比，不再以一句「会改变算法族」带过。
- **口径单一事实源**：UB 数值以 §4.5 逐 buffer 表为唯一事实源，其余章节只引用其汇总值（不再出现本地重算的第二套数字）；单位显式 KiB，避免十进制 KB / 二进制 KiB 混读。
- **配置不变式前置**：`deq_N | block_N` 作为编译期不变式写进 §5.2/§5.5/§6.1，任何 `block_N > 64` 的配置必须同时给出整除的 `deq_N` 并把 `W_sh` 整块单列，杜绝「分块只覆盖部分行 / 缓冲算漏」类错误。
- **源码事实以源码为准**：涉及源实现的描述（如线程分片整数关系）以源码约束条款（L49–L50）为准，算术前先核对等式/整除关系。

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

- **数学语义**：分组仿射量化的 4-bit 权重 × 16-bit 激活矩阵乘（W4A16）。设
  `A ∈ ℝ^{M×K}`（activation）、`W ∈ ℝ^{N×K}`（逻辑权重），权重按 `group_size=128` 沿 K 分组、组内用无符号 INT4 + 仿射参数 `(scale, zero)` 表示：
  `W[n,k] = (q[n,k] − z[n,g]) · s[n,g]`，其中 `g = k // 128`，`q ∈ {0,…,15}`，`z ∈ {0,…,15}`（uint8 存储），`s > 0`（float32 存储）。输出 `out = A · Wᵀ`，`out ∈ ℝ^{M×N}`。
- **INT4 打包约定**：`packed_weight[n, j]`（uint8）承载 k=2j（**低半字节** = 偶数 k）与 k=2j+1（**高半字节** = 奇数 k）两个量化值：
  `packed = q[:, 0::2] | (q[:, 1::2] << 4)`（`examples/TileOPs/tileops/workloads/gemm.py` L77）。
- **I/O 契约**（`examples/TileOPs/tileops/manifest/gemm.yaml` 的 `GemmW4A16Op.signature` / `shape_rules`，L90–L105）：

  | 参数 | Shape | dtype | 说明 |
  |------|-------|-------|------|
  | `activation` | `(M, K)` | **float16**（op 强制，见 §0.2） | 左操作数 A |
  | `packed_weight` | `(N, K/2)` | uint8 | INT4 打包权重（每字节 2 个 nibble） |
  | `weight_scale` | `(N, K/group_size)` | float32 | 组仿射 scale |
  | `weight_zero` | `(N, K/group_size)` | uint8 | 组仿射 zero |
  | `output` | `(M, N)` | same_as(activation) → float16 | ND 连续，无转置布局 |

  shape 规则：`K % group_size == 0`、`packed_weight.shape == (N, K/2)`、`weight_scale.shape == weight_zero.shape == (N, K/group_size)`、`output.shape == (M, N)`。
- **输出 shape（迁移必需字段）**：`(M, N)`。依据：`GemmW4A16Op._infer_output_shapes`（`<gpu_repo>/tileops/ops/gemm.py` L430–L437）返回 `{"output": (activation.shape[0], packed_weight.shape[0])}`，即 `(M, N)`；与 manifest `shape_rules` 末条一致。
- **规约语义**：沿 K 求和；源实现按 `block_k` 分块**顺序累加**（`k = 0, block_k, 2·block_k, …`），累加器为 **fp32**（`output_local = T.alloc_fragment(..., "float")`，L70）。分块加法分组与「一次 fp32 扁平求和」不同，但同在 fp32 域，误差在 fp16 容差内（§0.5 语义层保留 + §8.2 容差）。
- **dtype 语义**：activation 仅 **float16**（`GemmW4A16Op._validate_dtypes` L394–L396 对非 fp16 抛错；manifest `signature` 亦只标 float16）；**累加必须 fp32**（源 L70；NPU 侧同理，见 §3.5.1）；`weight_scale` float32、`weight_zero` uint8 的量化元数据在反量化时才参与、且**反量化结果按源实现落到 activation dtype**（`dequantized_local` 为 `dtype`，L72/L137–L148）。输出为 activation dtype。
- **边界语义**：
  - **非整除 M/N**：源 M/N 尾块在装载时用 `T.if_then_else` 零填充（L92–L96、L104–L109、L112–L121）、在写回时用逐元素 `if` 守卫（L165–L167）；K 恒被 `group_size` 整除、且工厂强制 `group_size % block_k == 0`（L47–L48），故 K 永远整块。
  - **N 整除快路径**：`n % block_n == 0` 时用整块 `T.copy` 装 packed_weight（L98–L102），否则逐元素谓词装载（L103–L109）——纯实现层分支。
  - **`m == 1`**：本算子**没有** GEMV 快路径（对照 `GemmOp._get_kernel` 的 `gemv` 分支，`GemmW4A16Op` 无此分支）；m=1 的 decode 形态直接进入同一 kernel（manifest 6 个 workload 中 4 个是 m=1）。
  - `nan`/`inf`：经 matmul 自然传播，无掩码；量化元数据 `s > 0`（`quantize_weight_int4` L68 用 `clamp_min(1e-12)`），`z ∈ [0,15]`，无除零。
  - **无 0-size 维**：`K % 128 == 0` 且 `K ≥ 128`。
- **host 侧语义**：无输入预处理、无输出后处理（`GemmW4A16Op.forward` 仅校验 + 调 kernel，L477–L508）；`packed_weight`/`weight_scale`/`weight_zero` 由**调用方**在测试/bench 里用 `quantize_weight_int4` 生成（`workloads/gemm.py` L107–L115），属测试装置、非算子契约的一部分。

**语义保持基线**：§8.1 的 golden 以本节语义为唯一依据实现（`torch.matmul(activation, dequantized_weight.T)`，移植源仓 `GemmW4A16Test.ref_program`），未复刻 §0.6 的 NPU 算法。

### 0.2 源算子输入输出

见 §0.1 的 I/O 契约表（依据 manifest `GemmW4A16Op.signature` L90–L99 与 `GemmW4A16Op._validate_shapes` L439–L459）。补充两点源码依据：

- **activation 只支持 float16**：`_validate_dtypes`（`<gpu_repo>/tileops/ops/gemm.py` L394–L397）对 `activation.dtype != torch.float16` 抛错；benchmark/manifest 亦只列 float16。故本迁移**只覆盖 fp16 activation**（与 `GemmOp` 的 fp16/bf16 双 dtype 不同）。
- **metadata 必须精确等于 `(N, K//group_size)`**：`_validate_shapes` L446–L458 强制，且 `GemmW4A16Op.__init__` L368–L371 只接受 `group_size == 128`。本设计据此把 `group_size` 当作编译期常量 128。

### 0.3 实现算法解读（怎么算）

源为工厂函数 `_gemm_w4a16_kernel(m, n, k, dtype, group_size)`，`functools.lru_cache(maxsize=32)` 缓存；内层 `build(block_m=64, block_n=64, block_k=64, num_stages=2, threads=128)` 返回已编译 `main`（L15–L46、L174–L216）。`GemmW4A16Kernel.default_config` 即该组默认值（L196–L204）。

**计算步骤分解**（源码每个计算语句均归入某步骤；行号指 `<gpu_repo>/tileops/kernels/gemm_w4a16.py`）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | Grid 映射：`T.Kernel(ceildiv(n,block_n), ceildiv(m,block_m), threads=128)` → `(bx, by)`（L60–L64） | m,n,block_m,block_n | `m_start=by*block_m`、`n_start=bx*block_n`（L83–L84） | 输出矩阵分块 |
| 2 | 分配：`activation_shared[block_m,block_k]`、`weight_shared[block_n,block_k]`、`packed_weight_shared[block_n,block_k//2]`、`weight_scale_shared[block_n]`、`weight_zero_shared[block_n]`、`output_local[block_m,block_n]`（fp32）、线程级 `packed_local/ dequantized_local / scale_local / zero_local`（L65–L74） | — | 片上缓冲 | 操作数驻留 + 累加器 |
| 3 | `T.annotate_layout`：`make_swizzled_layout(activation_shared / weight_shared)`（L76–L81） | — | SMEM swizzle | 消 bank conflict（实现层） |
| 4 | `T.clear(output_local)`（L86） | — | 累加器清零 | — |
| 5 | K 流水循环 `for kk in T.Pipelined(ceildiv(k,block_k), num_stages=2)`（L88） | — | — | K 分块累加调度 |
| 6 | 装载 activation tile：`T.Parallel(block_m, block_k)` + 谓词零填充 → `activation_shared`（L91–L96） | GM activation | `activation_shared` | 左操作数搬入 |
| 7 | 装载 packed Weight tile：`n % block_n == 0` 时整块 `T.copy(packed_weight[n_start, k_start//2], packed_weight_shared)`，否则逐元素谓词装载（L98–L109） | GM packed_weight | `packed_weight_shared` | 右操作数量化数据的搬入 |
| 8 | 装载组元数据：`T.Parallel(block_n)` 把 `weight_scale`/`weight_zero` 的当前组 `k_start//group_size` 装入 `weight_scale_shared`/`weight_zero_shared`（L111–L121） | GM weight_scale/zero | 组元数据 shared | 反量化参数搬入 |
| 9 | **线程级 chunk 解码与反量化**：`for chunk in T.serial(block_n*block_k//2 // (threads*4))`，每线程 4 字节：取该 chunk 所属行的 `scale/zero` → `for v in T.vectorized(4)` 读 4 个 packed 字节 → `for v in T.serial(8)` 用 `decode_unsigned_int4(4, byte, v%2, dtype)`（即 `_tir_packed_to_unsigned_convert("uint",8)`）解出 8 个 nibble、`(q − zero) * scale` → `for v in T.vectorized(8)` 写回 `weight_shared`（L123–L155） | `packed_weight_shared`、`weight_scale_shared`、`weight_zero_shared` | `weight_shared`（fp16，已反量化） | `W = (q − z)·s` 的物化 |
| 10 | 矩阵乘累加：`T.gemm(activation_shared, weight_shared, output_local, transpose_B=True, policy=FullRow)`（L157–L163） | 两个 shared | `output_local` | `out += A_tile · W_tileᵀ`（K 累加） |
| 11 | Epilogue：`T.Parallel(block_m, block_n)` + M/N 守卫写回 `output`（L165–L167） | `output_local` | GM output | 输出写回（单次舍入到 fp16） |

**数据流与内存访问模式**（源硬件视角）：

```
GM[activation]   --T.copy(分块)--> SMEM activation_shared[bm,bk] --┐
GM[packed_weight] --T.copy--> SMEM packed_weight_shared[bn,bk/2]   │
                                  │ chunk 解码 + 反量化（thread 级） │
                                  ▼                                │
                              SMEM weight_shared[bn,bk] (fp16) ────┴--GMMA(NT, trans_B)--> register fragment output_local[bm,bn] fp32
                                                                                              │ T.Parallel + M/N 守卫
                                                                                              ▼
                                                                                          GM[output]
   K 流水：T.Pipelined(num_stages=2) 环形复用 activation_shared / packed_weight_shared，
           weight_shared 为每迭代重写的解码结果缓冲；swizzle 布局缓解 SMEM bank conflict。
```

**循环与并行结构**：
- 两级并行：grid（`ceildiv(n,block_n) × ceildiv(m,block_m)` 个 block）+ 块内 128 线程（`threads=128`）。
- 块内 K 迭代：`T.Pipelined(ceildiv(k,block_k), num_stages=2)`。
- 解码阶段的线程级三层循环：`T.serial(chunk)`（每线程处理的 4 字节组数）× `T.vectorized(4)`（读）× `T.serial(8)`（nibble 标量解码）× `T.vectorized(8)`（写）。源要求 `(block_n*block_k//2) % (threads*4) == 0`（**整除**而非相等，L49–L50）；默认 `threads=128`、`block_n=block_k=64` 时 `block_n*block_k//2 = 2048 = 4×(128×4)` ⇒ `T.serial(block_n*block_k//2 // (threads*4))`（L123）= `T.serial(4)`，即 chunk 循环 **4 次**（每线程 4 组 × 4 字节），并非 1 次。

**host 侧逻辑**：
- `GemmW4A16Kernel.__init__` 调工厂建 kernel；`forward(activation, packed_weight, weight_scale, weight_zero)` 用 `**self.config` 编译并执行（L206–L216）。
- `GemmW4A16Op.forward` 做 dtype/shape 校验 + `(m,n,k)` 推断 + kernel 缓存，无预处理/后处理（L477–L508）。
- 工厂级 `lru_cache(maxsize=32)` 按 `(m,n,k,dtype,group_size)` 缓存编译产物（L15）。
- 约束校验：`k % group_size == 0`（L23–L24）；`group_size % block_k == 0`（L47–L48）；`(block_n*block_k//2) % (threads*4) == 0`（L49–L50）。

**编译配置**：`@tilelang.jit(out_idx=[-1], pass_configs={TL_ENABLE_FAST_MATH: True}, compile_flags=["-O3","-DENABLE_BF16"])`（L33–L39）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 耦合性初判 |
|---|----------|------|------|-----------------|-----------|
| W1 | **权重流量减半（INT4 打包随算子读入）** | 把 W 的 GM 读流量从 2 B/元素降到 0.5 B/元素（W4 的核心收益） | 保持 `packed_weight` 打包态直到核内，反量化后才进 GEMM | 无（纯布局/契约） | 算法层可移植（**必须承接**） |
| W2 | **融合反量化（不物化到 GM）** | 省去 `W_deq` 的 GM 写+读（decode 形态下主流量项） | 反量化结果只落在 SMEM `weight_shared`，紧接 `T.gemm` 消费 | SMEM + 线程级解码 | 算法层意图可移植；**实现层硬件强相关**（NPU 需 Vector→Cube 交接） |
| W3 | SMEM tiling（`alloc_shared` + 分块搬运） | 减 GM 重复访问 | A / packed W tile 驻留 SMEM | shared memory | 硬件强相关（NPU 有 L1/UB 等价） |
| W4 | 分块搬运（`T.copy`，TMA 形态） | 高带宽块搬运 | 整块 `T.copy`（n 整除快路径） | TMA / cp.async | 硬件强相关（NPU 有 MTE2 等价） |
| W5 | 多级流水环形缓冲（`num_stages=2`） | 搬入与计算重叠 | `T.Pipelined` stage 索引 + 多缓冲 | 异步拷贝 + SMEM | 模式可移植（NPU `T.Pipelined`） |
| W6 | **A16 Tensor Core GEMM（`T.gemm(..., transpose_B=True)`）+ fp32 累加器** | 矩阵算力；反量化后走 16-bit 张量核 | WGMMA 分形 MMA，fp32 累加 | tensor core | 硬件强相关（NPU 有 Cube 等价） |
| W7 | 线程级 chunk 解码 + 向量化访存（`T.serial(chunk)` / `T.vectorized(4)` 读 / `T.vectorized(8)` 写） | nibble 解包与反量化的线程并行 + 合并访存 | 每线程 4 字节 → 8 个 nibble | warp/线程模型 + SMEM | 硬件强相关（NPU 无 thread 绑定，见 `TRAP-threads-kwarg-noop`） |
| W8 | **解码负载均衡（每 chunk = 同一行，scale/zero 标量提升）** | 每线程的 `scale_local/zero_local` 只取一次（每 chunk 4 字节同属一行） | `scale_row = packed_base // (block_k//2)` 行内恒定 | 行主序 + chunk 划分 | 算法层可移植（**必须承接**：避免逐元素取元数据） |
| W9 | `T.gemm` 的 `policy=FullRow` | warp 级 tile 划分策略 | warp 分区 | warp 模型 | 硬件强相关（NPU 无 warp policy） |
| W10 | SMEM swizzle（`make_swizzled_layout`） | 消 SMEM bank conflict | 地址重排 | SMEM bank 结构 | 硬件强相关（NPU 无 bank conflict） |
| W11 | 装载谓词零填充 + epilogue M/N 守卫 | 支持非整除 M/N | 逐元素 `if_then_else` / `if` | — | 算法层可移植 |
| W12 | n 整除快路径（整块 `T.copy` vs 逐元素谓词装载） | 常见形态免谓词开销 | 编译期分支 | — | 可移植（NPU 用动态 extent 切片承接） |
| W13 | 工厂 `lru_cache` + 每 shape 编译 + `default_config` | 降 host 开销、shape 特化 | 缓存编译产物 | — | 可移植 |
| W14 | `TL_ENABLE_FAST_MATH` + `compile_flags=["-O3","-DENABLE_BF16"]` | 快速数学 / nvcc 编译选项 | 编译器 config | GPU nvcc / CUDA fast-math | 硬件强相关（npuir 无对应） |

> 识别不出 ≠ 不存在：以上 14 项为源码中可识别的优化手段，逐项在 §0.5 判定处置，避免静默丢弃。其中 **W1/W2/W8 是 W4A16 的性能本体**（省权重流量、省 GM 往返、免逐元素元数据），迁移中必须承接其意图。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定依据映射表：`tilelang-op-design/references/migration-analysis.md` §5.3；NPU 能力佐证：`docs/开发指南.md` §2/§3、`docs/Tilelang.language/线性代数操作/T.gemm.md`、`docs/Tilelang.language/内存操作/T.alloc_shared.md`、`docs/Tilelang.language/内存操作/T.copy.md`、`examples/gemm/example_gemm.py`、`testing/npuir/bf16_support_ops/test_mix_kernel_bf16.py`。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 语义（`out = A·Wᵀ`、`W=(q−z)s`、fp16 输出、fp32 累加、nibble 打包约定） | 语义 | 无 | — | **保留** | 语义层无条件保留（§0.1） |
| **W1 权重流量减半（INT4 打包保持到核内）** | 算法层 | 无 | — | **保留** | 契约即 INT4 打包；核内反量化后才物化 fp16 → 流量收益保持 |
| **W2 融合反量化（不物化 W_deq 到 GM）** | 算法层（意图）+ 实现层（手段） | SMEM + 线程级解码 | 意图可移植；手段需换 | **重新设计** | → §0.6 R1（Vector 侧反量化 → Cube `T.gemm` 直接消费 UB 缓冲；precedent `test_mix_kernel_bf16.py`） |
| W3 SMEM tiling | 优化 | shared memory | 有（L1 512KB / UB 192KB） | **等价替换** | Developer `T.alloc_shared`（Cube 操作数 → L1；Vector 中间 → UB）；`docs/开发指南.md` §2.1 |
| W4 整块 `T.copy` 搬运 | 优化 | TMA / cp.async | 有（MTE2） | **等价替换** | `T.copy`（GM→L1，自动 ND→NZ）；`docs/Tilelang.language/内存操作/T.copy.md` §2.4 |
| W5 `T.Pipelined(num_stages=2)` | 优化 | 异步引擎 + SMEM | 有 | **等价替换** | `T.Pipelined(k_iters, num_stages=S)`，S 按 L1/UB 容量重算；`docs/开发指南.md` §3.2 |
| W6 A16 Tensor Core GEMM + fp32 累加 | 优化 | tensor core WGMMA | 有（Cube `T.gemm`→`hivm.hir.mmadL1`） | **等价替换** | `T.gemm(A_sh, W_sh, C_local, initC=(kk==0), b_transpose=True)`；dst fp32（`T.gemm.md` §2.3）；NT 形态 |
| **W7 线程级 chunk 解码 / `T.vectorized` 访存** | 优化 | warp/线程模型 + `threads=128` | **无**（npuir `T.Kernel(threads=)` 是 no-op，`TRAP-threads-kwarg-noop`；无 thread binding 语义） | **重新设计** | → §0.6 R2/R4（整块 v-prefix 向量链 / `T.Parallel`，由编译器做 Vector 向量化与 CV 拆分） |
| **W8 解码负载均衡（chunk 行内恒定，scale/zero 标量提升）** | 算法层 | 行主序 + chunk 划分 | 有（可移植） | **保留（意图）** | NPU：每 (n, k-tile) 的 scale/zero 只 cast 一次并 `vbrc` 广播（§1.6.1 E3）；免逐元素元数据取用的意图保持 |
| W9 `policy=FullRow` | 优化 | warp 分区 | **无** | **舍弃** | NPU `T.gemm` 无 warp policy 参数（`T.gemm.md` §2.1 签名无 policy） |
| W10 SMEM swizzle 消 bank conflict | 优化 | SMEM bank | **无此问题** | **舍弃** | NPU 无 SMEM bank conflict；NZ 分形由搬运/框架处理（`references/nz-format.md`；migration-analysis §5.3「swizzle→舍弃」） |
| W11 谓词零填充 + epilogue M/N 守卫 | 算法层 | 无 | 有 | **等价替换** | 动态 extent 切片装载 + 写回守卫（`docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 规则 2） |
| W12 n 整除快路径 | 优化 | 无 | 有 | **等价替换** | 动态 extent 切片（`tm=min(block_M, M−m0)`、`tn=min(block_N, N−n0)`）天然统一快慢路径 |
| W13 `lru_cache` / 每 shape 编译 / `default_config` | 优化（host） | 无 | 有 | **保留** | 保留 shape 特化与缓存；host 只做 shape/stride 元数据 |
| W14 `TL_ENABLE_FAST_MATH` + nvcc flags | 优化（编译配置） | CUDA nvcc / fast-math | **无对应** | **舍弃** | npuir 不需要；`pass_configs` 默认空（源 `-O3/-DENABLE_BF16` 为 nvcc 选项） |
| Stage 0 抽取件的 `_gemm_w4_a16_kernels.py`（工厂 + 源码副本） | helper | — | — | **保留（仅参考）** | 作语义/算法参照（`examples/TileOPs/tileops/kernels/gemm/gemm_w4a16/_gemm_w4_a16_kernels.py`），非实现基线 |
| `tilelang.quantize._tir_packed_to_unsigned_convert`（私有 TIR 内建） | 实现依赖 | 依赖 GPU codegen 的移位 | **无**（探针实测：位运算路径被 verifier/runtime 阻断，见 §1.6.2 与 CG-2026-0018/0019/0025） | **重新设计** | → §0.6 R2（符号无关的 cast-only nibble 提取，探针 P2 实测 bit-exact） |

> 判定统计：**保留 4 项**（语义、W1、W8 意图、W13，另 Stage 0 抽取件仅作参考）、**等价替换 6 项**（W3/W4/W5/W6/W11/W12）、**重新设计 3 项**（W2、W7、私有 TIR 内建；详见 §0.6 R1–R4）、**舍弃 4 项**（W9/W10/W14 + host 无预处理）。所有舍弃项均写明理由，未静默丢弃。

### 0.6 NPU 算法重设计

对照 Phase R 调研结论（§1.6.0）：调研选定 **基线「分块融合反量化的 A16 Cube 直接法」**（与源算法同族，不做算法族替换），但**反量化的实现载体必须重设计**（源用 SMEM+线程、NPU 用 UB+Vector→Cube 交接）。以下逐项给出「源方案 → NPU 新算法 → 语义保持论证」。

**重设计项 R1：融合反量化的载体 —— 线程级 SMEM 解码 → Vector(UB) 反量化 + Cube `T.gemm` 直接消费**

- **源方案**：每线程读 4 个 packed 字节、标量解 8 个 nibble、写回 SMEM `weight_shared`（fp16），紧随 `T.gemm`（L123–L163）。意图 = 「W2 融合反量化」：`W_deq` 不落 GM，省 N·K·2 B 的写+读（decode 形态下这是最大可消除项）。
- **NPU 新算法**：单核 Developer 内核，反量化在 **Vector 侧**用 v-prefix 向量链把 packed tile 解包+仿射变换成 fp16 写入 `T.alloc_shared` 缓冲，**该缓冲直接作为 `T.gemm` 的 src2**（compiler 自动做 CV 拆分与 UB→Cube 交接）。
  - 依据（同形态先例）：`testing/npuir/bf16_support_ops/test_mix_kernel_bf16.py` —— `local_tmp = T.alloc_shared(...)` 由 `T.vadd(local_a, local_b, local_tmp)` 写入后直接被 `T.gemm(local_tmp, local_c, local_out)` 消费，同一 Developer 内核、无显式 `Scope`/手工同步；该测试被标记为通过。
  - 该结构满足 `tilelang-mixcv-skill` 的 Developer 判定（同一 kernel 同时含 `T.gemm` 与 v-prefix 算子）→ 本设计按 **MixCV** 规则处理（§6/§7）。
- **语义保持论证**：反量化的数学不变（`W=(q−z)s`，见 §1.6.1 E4 的机器验证），改变的只是「谁算、算在哪级存储」；K 累加顺序（顺序 k-tile 累加）、fp32 累加域、输出 round 次数均不变。边界语义沿用 R5。
- **不采纳的替代**（详见 §1.6.0 R1）：把 `W_deq` 物化到 GM 再 GEMM（两次 kernel 或 GM 往返）——访存从 0.5 B/元素升到 2.5 B/元素（回退掉 W1/W2 的全部收益），R3/R4 已淘汰。

**重设计项 R2：nibble 解包 —— 私有 TIR 内建的移位/掩码 → 符号无关的 cast-only 向量 recipe**

- **源方案**：`decode_unsigned_int4 = _tir_packed_to_unsigned_convert("uint", 8)`，对 `packed_local[v//2]` 做 `(val >> (pos*4)) & 15` 的标量位运算（L29–L31、L136–L148）。
- **NPU 新算法**：位运算路径在本工具链被阻断（探针 P1 实测，§1.6.2 与 CG-2026-0018/0019），改用**只用 fp16 算术 + 带舍入模式的 cast**的等价 recipe（探针 P2 **实测 bit-exact**）：

  ```
  bf  = T.npuir_cast(pw, bf, round_mode="rint")          # uint8 -> f16（NPU 按 i8 解释，见下）
  t   = T.npuir_mul(bf, 0.0625, t);  h16 = T.npuir_cast(t, h16, round_mode="floor")   # h = floor(bf/16)
  hf  = T.npuir_cast(h16, hf, round_mode="rint")
  lo  = T.npuir_sub(bf, T.npuir_mul(hf, 16.0, hqx), lo)  # lo = bf - 16h  == 低半字节 ∈ [0,15]
  t   = T.npuir_mul(hf, 0.0625, t);  q16 = T.npuir_cast(t, q16, round_mode="floor")   # floor(h/16) ∈ {-1,0}
  hi  = T.npuir_sub(hf, T.npuir_mul(T.npuir_cast(q16, hqx, round_mode="rint"), 16.0, hqx), hi)
  q   = T.npuir_interleave(lo, hi, q)                    # [N, K] fp16，k 序还原（低 nibble=偶 k）
  ```
- **关键事实（探针实测）**：npuir 把 `uint8` 缓冲 lower 成 **有符号 i8**（IR 为 `tensor<64x64xi8>`），故字节 ≥128 经 cast 得到负值（如 172 → −84）。上述 recipe **对符号无关**：因为 `256 ≡ 0 (mod 16)`，`lo = b mod 16` 与 `hi = floor(b/16) mod 16` 在有符号解释下经「floored 取模」还原后与无符号语义**完全一致**（探针 P2：nibble+interleave `bit_exact=True`）。
- **语义保持论证**：`floor`/乘 16/减法是精确整数运算（中间值 ≤255，fp16 精确表示），无舍入损失；`interleave(lo, hi)` 精确还原源打包约定（低 nibble = 偶 k）→ 与源 `decode_unsigned_int4(4, byte, v%2)` 的取值**逐位相同**（P2 机器验证 `bit_exact=True`）。边界：`q ∈ [0,15]` 与 `z ∈ [0,15]` 不变。

**重设计项 R3：2D grid → 一维 `T.Kernel` + cid 分解 + 分核策略**

- **源方案**：`T.Kernel(ceildiv(n,block_n), ceildiv(m,block_m), threads=128)` 二维 grid，`(bx,by)` 直接映射 N/M 块（L60–L64）。
- **NPU 新算法**：`T.Kernel(num_launched, is_npu=True)` 一维；`n_blocks = T.ceildiv(N, block_N)`、`bx = cid % n_blocks`、`by = cid // n_blocks`；分核三要素见 §5.5。
- **语义保持论证**：`cid → (by, bx)` 为双射，每个逻辑输出块的计算完全一致，无重复/遗漏写；`cid` 的取值范围由启动核数保证——`num_launched = num_logical`（**非 persistent**：一个逻辑输出块对应一个核，超出物理核数的部分由运行时串行调度，核内**无** `T.serial` 多任务循环）。核数对齐到物理核整数倍由 host 选择 `block_N` 实现（§5.5），**不**通过截断启动核数，故 `cid ∈ [0, num_logical)` 全覆盖、无遗漏写。

**重设计项 R4：线程级 chunk 解码循环 → 整块 v-prefix 向量操作**

- **源方案**：`for chunk in T.serial(...)` + 线程 `tx * 4` / `tx * 8` 的 per-thread 地址计算（L85、L123–L155）。
- **NPU 新算法**：去掉 thread binding 与 chunk 循环，直接对整个 `[block_N, block_K/2]` packed tile 施加 R2 的向量链（`npuir_*` 逐元素 op 作用于整块），再用 `T.npuir_interleave` 得到 `[block_N, block_K]`；dequant 的 `(q−z)·s` 用 `T.vbrc` 把 `[block_N,1]` 的 `scale/zero` 广播到 `[block_N, block_K/2]`。
- **语义保持论证**：逐元素 op 作用于整块与线程分片作用于同样元素的数学结果相同（同一元素同一算子）；chunk 划分只是并行手段。W8 的「元数据标量提升」意图由 E3（每 (n, k-tile) 只 cast 一次 + `vbrc` 广播）承接。

**重设计项 R5：非整除 M/N 的处理（取代逐元素谓词装载 + epilogue 守卫）**

- **源方案**：装载阶段逐元素 `if_then_else` 零填充（L92–L96、L104–L109、L114–L121），epilogue 逐元素守卫（L165–L167）。
- **NPU 新算法**：host 侧禁止 padding/内容改动（`references/ascend-constraints.md` §4），全部在 kernel 内：
  - `tm = T.min(block_M, M − m0)`、`tn = T.min(block_N, N − n0)`（`tk` 恒等于 `block_K`，因工厂已强制 `group_size % block_K == 0` 且 `group_size | K`，见 R6）；
  - 装载用动态 extent 切片（`T.copy(activation[m0:m0+tm, k0:k0+BK], A_sh[0:tm, 0:BK])`、packed/scale/zero 同理）；
  - **M/N 尾无需清零**：A 的多余行只影响不被写回的 out 行、W 的多余列只影响不被写回的 out 列（matmul 行/列独立性）；写回用动态 extent 守卫 `T.copy(C_local[0:tm,0:tn], out[m0:m0+tm, n0:n0+tn])`；
  - **K 尾零填充**：若 `block_K ∤ K`（本 manifest 不触发，但保持健壮），A/W 的 K 尾需清零（`T.clear`）——零元素不改变点积；A、W 两侧都清以杜绝 `0·NaN`/`0·inf`（§1.6.1 E2）。
- **语义保持论证**：零填充项贡献 `0·x = 0`（有限 x）故不改变点积；M/N 尾只影响不被写回的元素。非整除行为与源算子一致（输出有效区数值相同）；等价性经 §1.6.1 E2 机器验证（`EQUIV_PASS`）。

**重设计项 R6：编译期常量与约束**

- 源 `group_size % block_k == 0` 约束（L47–L48）在 NPU 侧保留并**强化为设计不变式**：选 `block_K ∈ {32, 64, 128}`，`block_K | group_size = 128`。取 `block_K = 64` 时每 k-tile 落在唯一组内 ⇒ `weight_scale/weight_zero` 的组下标 `g = k0 // 128` 在 tile 内恒定（E3 提升的前提）。
- 源 `(block_n*block_k//2) % (threads*4) == 0` 约束（线程分片整除）与 NPU 无关（无 thread 分片）→ 放弃；改为 **UB 容量约束**（§4.5）。
- 源 `threads=128`（L45、L64）→ 移除（npuir 无效，`TRAP-threads-kwarg-noop`）。

### 0.7 标杆实现（golden 依据）

- 源算子路径（语义依据）：`<gpu_repo>/tileops/kernels/gemm_w4a16.py`（`_gemm_w4a16_kernel`，L15–L171）；源 op 类 `<gpu_repo>/tileops/ops/gemm.py`（`GemmW4A16Op`，L352–L512）。
- 源仓参考实现 / 测试基准：`examples/TileOPs/tests/ops/test_gemm.py` 的 `GemmW4A16Test.ref_program`（L138–L147：`torch.matmul(activation, self.dequantized_weight.T)`，其中 `dequantized_weight` 由 `examples/TileOPs/tileops/workloads/gemm.py` 的 `quantize_weight_int4`（L52–L81）在 fp32 域算出后 cast 到 fp16（L114））。
- **golden 以 §0.1 语义为唯一依据**，直接移植上述 `ref_program` + `quantize_weight_int4`，不复刻 §0.6 的 NPU 算法（保证验证独立性）。
- 源仓 Stage 0 抽取件（参考，非实现基线）：`examples/TileOPs/tileops/kernels/gemm/gemm_w4a16/_gemm_w4_a16_kernels.py`。
- NPU 规格/用例/bench：`examples/TileOPs/tileops/manifest/gemm.yaml`（`GemmW4A16Op` 段 L80–L137）、`examples/TileOPs/tests/ops/test_gemm.py`（L138–L212）、`examples/TileOPs/benchmarks/ops/bench_gemm.py`。
## 1. 概述

### 1.1 算子名称

`_gemm_w4a16_kernel`（项目 `gemm_w4a16`；NPU 实现文件为本算子目录下的 `_gemm_w4a16_kernel.py`，由 Stage 3 产出，当前尚未生成）。

### 1.2 功能描述

分组仿射量化的 W4A16 矩阵乘：`output = activation · dequant(packed_weight)ᵀ`，即 4-bit 打包权重（128 分组、仿射 `(scale, zero)`）在核内解包+反量化为 fp16 后，与 fp16 激活做 A16 矩阵乘、fp32 累加、单次舍入输出。支持非整除 M/N（kernel 内动态 extent 处理，无 host padding）。

迁移后为 NPU 侧 Developer 模式 **MixCV** 实现（Vector 侧反量化 + Cube 侧 `T.gemm`，由编译器自动 CV 拆分与交接）。

### 1.3 数学公式

$$
\mathrm{out}[m,n] = \sum_{k=0}^{K-1} A[m,k]\cdot W[n,k],
\qquad W[n,k] = \big(q[n,k] - z[n,g]\big)\,s[n,g],
\qquad g=\Big\lfloor \frac{k}{128}\Big\rfloor
$$

其中 `A:[M,K]` fp16、`W:[N,K]` 为逻辑权重、`q∈{0..15}` 为 INT4 量化值（`packed_weight` 每字节承载 `k=2j`（低 nibble）与 `k=2j+1`（高 nibble））、`s:[N,K/128]` fp32、`z:[N,K/128]` uint8；累加在 fp32 域，输出 `[M,N]` fp16。

### 1.4 算法描述

**迁移决策后的 NPU 侧算法**（与 §1.6.0 调研选定算法一致）：

1. **输出分块**：`out[M,N]` 划分为 `ceil(M/block_M) × ceil(N/block_N)` 个 `block_M×block_N` 输出块；一维 grid 每 slot 处理一个逻辑块（R3）。
2. **沿 K 分块融合反量化 + A16 累加**：对每个输出块，沿 K 以 `block_K` 分块，`T.Pipelined` 流水：
   - **装载**：activation tile `[block_M, block_K]` 与 packed weight tile `[block_N, block_K/2]`（含组元数据 `scale/zero` 的一列）经 `T.copy`（MTE2）进入片上；
   - **反量化（Vector 侧）**：packed tile 经「cast-only nibble 提取」（R2，§1.6.2 配方）得到 `lo/hi` → `T.npuir_interleave` 还原 `[block_N, block_K]` 的 `q` → 与 `vbrc` 广播的 `(q−z)·s` 得 fp16 `W_sh`（E3/E4，§1.6.1）；
   - **矩阵乘（Cube 侧）**：`T.gemm(A_sh, W_sh, C_local, initC=(kk==0), b_transpose=True)`，fp32 累加（R1）。
3. **尾块处理**：M/N 非整除用动态 extent 切片装载 + 写回守卫；K 尾（若 `block_K ∤ K`）零填充（R5）。
4. **写回**：`T.copy(C_local → out)`，fp32→fp16 单次舍入（E1）。

与源算法的**结构差异**（来源）：
- 源反量化在 **SMEM + 128 线程**上完成（§0.4 W7），NPU 改为 **UB + v-prefix 向量链**，反量化结果缓冲由 `T.gemm` 直接消费（§0.6 R1/R4）。
- 源 nibble 提取用 `_tir_packed_to_unsigned_convert` 的**移位/掩码**（§0.4 W7），NPU 改为**符号无关的 cast-only recipe**（§0.6 R2，探针 P2 bit-exact）。
- 源二维 grid + `threads=128`（§0.3 步骤 1、W7）→ NPU 一维 grid + cid 分解 + 分核策略（§0.6 R3、§5.5）。
- 源 SMEM swizzle / `policy=FullRow` / fast-math+nvcc flags（§0.4 W9/W10/W14）→ NPU 全部舍弃（§0.5）。
- **未变**：算法族（分块融合反量化的 A16 直接法）、K 顺序累加、fp32 累加域、INT4 打包态到核内（W1/W2 意图承接）、组元数据一次取用（W8 意图由 E3 承接）。

### 1.5 数据流图

```
GM[activation (M,K) f16] --T.copy(MTE2)--> L1 A_sh[block_M, block_K] ─────────────┐
GM[packed_weight (N,K/2) u8] --T.copy--> UB pw[block_N, block_K/2]                │
GM[weight_scale (N,K/128) f32] --T.copy--> UB s32[block_N,1]                     │
GM[weight_zero (N,K/128) u8] --T.copy--> UB z8[block_N,1]                        │
        │                                                                        │
        ▼  Vector 链（§0.6 R2 / §1.6.2）                                          │
   bf→h→lo / hi → interleave → q[block_N, block_K] f16                          │
        │  E3: cast 一次 + vbrc 广播；E4: fp16 域 (q−z)s                          │
        ▼                                                                        │
   W_sh[block_N, block_K] f16  ──(UB → L1 交接，编译器自动 CV 拆分)───────────────┤
                                                                                 ▼
                              T.gemm(Cube, mmadL1, b_transpose=True, initC=(kk==0)) --> L0C C_local[block_M,block_N] f32
                                                                                 │  (K 流水：T.Pipelined num_stages=S)
                                                                                 ▼
                                    T.copy(单次 VCast fp32→fp16, 动态 extent 守卫) --> GM[output (M,N) f16]
宿主边界：host 仅做 (m,n,k,group_size) 推断、block 启发式、shape/stride 元数据；不触碰张量内容、不 padding、不做反量化。
```

### 1.6 算法调研与优化分析

> 分析对象 = §1.6.0 调研选定算法（分块融合反量化的 A16 Cube 直接法），经 §0.6 重设计落地的 NPU 算法。本节结论是 §3.1 与 §6 的输入。

#### 1.6.0 算法调研（Algorithm Research）

调研深度：**完整调研**（矩阵类算子 + 量化前置变换，方法论见 `tilelang-op-design/references/algorithm-research.md` §2）。检索源：`algorithm-candidates.md`（ALG-gemv 条目，本仓无 W4A16 专条）、`pattern-library/INDEX.md`（PL-1.8/PL-1.13/PL-1.15/PL-1.16、CONST-capacity-910B2C、CONST-mte2-degradation、CONST-vector-launch-overhead、CONST-store-fixpipe-gm-only、CG-2026-0005/0010）、`examples/gemm/`、`examples/mixcv/`、`testing/npuir/`、源算子 §0.3/§0.4、模型知识（GPTQ/AWQ 族反量化 GEMM）。检索入口：`python3 .agents/tools/kb_search.py "gemm dequant int4 cube L1"`。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线：分块融合反量化 A16 直接法**（源算法） | 每个 `(m,n)` 输出块按 `block_K` 分块，核内把 packed W tile 解包+反量化为 fp16，`T.gemm(..., b_transpose=True)` 累加 | — | — | ✅（基线） |
| 1 | 两阶段：先物化 `W_deq` 到 GM，再普通 fp16 GEMM | `W_deq = (q−z)s` 全量落 GM，第二个 kernel 做 `A@W_deqᵀ` | 数学恒等 | ❌**负收益**：W 流量 0.5 B/元素 → 2.5 B/元素（write+read fp16），代表 shape 访存 +2.68 GB（≈8.4×） | ✅（R3 对比，R4 淘汰） |
| 2 | 仿射分解（免减零）`out = s⊙(A·qᵀ) − (s⊙z)⊙(A·1_g)` | 把 `(q−z)s` 拆成 `qs` 与 `zs` 两项，第二项用 A 的分组行和 `SA[M,G]`（`G=K/128`） | 精确算术恒等；**fp16 下有大数相消风险**（`z` 可达 15、`s` 为大值时 `s·z·SA` 与 `s·Aq` 同量级相减） | 省去逐元素 `−z` 与 `⊙s`（每元素 2 op），但需额外 `SA[m,g]` 行和（`M·K` 级 Vector 加法）与 `[M,G]×[G,N]` GEMM，且 **GM 流量不降**（q 仍须物化为 fp16 供 GEMM） | ✅（R3/R4 对比，淘汰） |
| 3 | int8 Cube 路径（A 也量化到 int8） | `q_A int8 × q int8` 走 `T.gemm` int8/int32 路径，再反量化两侧 | **非等价**：activation 契约是 fp16，量化到 int8 引入不可控精度损失（违反 §0.1 dtype 语义） | FLOPs 不变、流量不变（A 变小但 A 非主项） | ❌（语义违反，`docs/Tilelang.language/线性代数操作/T.gemm.md` §2.2.1） |
| 4 | split-K（K 维并行分解 + 归约） | K 拆 P 段部分和再合并 | 数学恒等（加法结合律，fp32 分组变化） | FLOPs 不变；需 `M·N·P·4` partial 落 GM + 二阶段归约（`CONST-store-fixpipe-gm-only`：跨引擎强制 GM 往返） | ✅（R3 对比，R4 淘汰） |
| 5 | 元数据提升（每 (n, 组) 只取一次 scale/zero，广播复用） | `s[n,g]`/`z[n,g]` 每 k-tile 只 cast 一次 → `vbrc` 广播到整 tile | 数学恒等（同一值的广播 vs 逐元素重读同一值） | 免 `N·K` 次 fp32→fp16 元数据 cast（降为 `2·N·(K/128)` 次） | ✅ **采纳（E3）** |
| 6 | 反量化域下移（fp32 反量化 → fp16 反量化） | `(q−z)s` 在 fp16 域算（q、z 小整数在 fp16 精确） | 容差内等价（`s` 的 fp16 舍入 ≲1 fp16 ulp，实测 max abs 3.9e-3 ≪ atol 7e-2） | 免 fp32 中间缓冲与 fp32→fp16 转换（每元素省 1 cast + 一半缓冲字节） | ✅ **采纳（E4）** |
| 7 | epilogue 单次舍入融合 | fp32 累加器 → store 时一次 VCast 到 fp16（不落 fp32 中间） | 数学恒等（同一 fp32 值单次 RNE） | 省 `M·N·4` B 的 fp32 中间 GM 往返 | ✅ **采纳（E1）** |
| 8 | K 尾零填充（免 host padding） | `Σ_{k<K'}`，`K'=ceil(K/block_K)·block_K`，A/W 尾区同时清零 | 数学恒等（新增项 `0·0=0`） | 使 `block_K ∤ K` 时免 host padding（host padding 被 `references/ascend-constraints.md` §4 禁止） | ✅ **采纳（E2）** |
| 9 | **Vector 规约累加（m=1 gemv 形态）** —— `algorithm-candidates.md` 的 `ALG-gemv` 条目 R1 命中行候选（与「GEMM 分形路径」并列的必查项） | 每个输出元素 `out[0,n] = Σ_k A[0,k]·W[n,k]` 用 Vector 归约（横向 K 归约 / `T.reduce_sum`，无 Cube）计算，不再走 Cube 分形 GEMM | 数学恒等（同一 dot product，K 顺序累加；仅执行单元/结构不同） | 省 Cube 的 `block_M=16` 分形填充（实算 `2.15e13`→真值 `1.34e12` FLOPs），但 MAC 密集段落 Vector 单元、无访存收益（GM 流量同基线）；仅对 m=1 形态有意义 | ✅（R3/R4 对比，淘汰） |

**R2 在线算法**：**无独立的「在线」变体适用于本语义——基线本身已是单遍在线形态**。
- 结构依据：GEMM 的 K 累加天然就是一个输出 tile 的 **running accumulator**（`initC=(kk==0)` + 逐 k-tile 累加，中间状态 O(block_M·block_N)），不存在需要跨 tile 重缩放的聚合统计量；反量化是**逐元素映射**（无跨元素依赖）→ 单遍即在线（`algorithm-research.md` §3 R2 的「逐元素映射单遍天然在线」判据）。
- 对照检查：在线 softmax 类的「running max + 重缩放」结构在本算子无对应物（无 max、无归一化分母）。
- 本 manifest 的 m=1 decode 形态下，并行度来自 N 分块（`ceil(N/block_N)=64~128` 个逻辑核），**无需** split-K 补并行度（R1 候选 4 淘汰）。
- 收益口径（若强行 split-K）：扫描遍数 1→1（K 仍单遍，但分段）、中间缓冲 `M·N·P·4` 落 GM、跨核归并代价 ↑ —— 净亏（R4 淘汰）。

**R3 复杂度对比**（四口径；基线在表；口径含/不含已注明。代表 workload = `decode-long-k`：M=1, N=8192, K=81920, fp16）：

| 算法候选 | FLOPs（含口径） | 访存量 Bytes（GM 流量） | 扫描遍数 | 中间缓冲峰值（片上/落盘） | 可并行度 / 跨核代价 |
|---------|----------------|----------------------|---------|------------------------|---------------------|
| **基线（融合反量化 A16 直接法）** | 真值 `2·M·N·K = 1.34e12`；Cube 实算 `2·block_M·N·K = 2.15e13`（`block_M=16` 分形填充） | `M·K·2 + N·K/2 + N·(K/128)·5 + M·N·2 = 361.9 MB`（manifest roofline 口径；**不含** L1/UB 内部重复） | 1（W 打包读 1 遍） | UB ≈150.0 KiB（2-stage 反量化链，§4.5）/ L0C 4 KiB / L1 ≈20 KiB | 高（`N/block_N=128` 块）；无跨核归并 |
| 候选 1（先物化 `W_deq`） | 同基线 + 反量化 ~5 op/元素 | `361.9 MB + 2·N·K·2 = 3.04 GB`（+2.68 GB，8.4×） | 2（W 打包读 1 遍 + `W_deq` 写读各 1 遍） | GM workspace `N·K·2 = 1.34 GB` | 高；**kernel 间同步** |
| 候选 2（仿射分解） | 基线 + `2·M·N·G = 1.05e7`（G=640）+ 行和 `M·K` 加法 | 同基线（q 仍须物化 fp16） | 1 | +`SA[M,G]·4`（可片上） | 高；需额外行和归约 |
| 候选 4（split-K，P 段） | 同基线 | 基线 + `M·N·P·4` partial 写读 | 1（分段） | `M·N·P·4` 落 GM | 更高（P×）；**二阶段归约 / atomic** |
| 候选 9（Vector 规约累加，m=1 gemv） | 真值 `2·M·N·K = 1.34e12`（无分形填充）；MAC 数 `N·K = 6.7e8` | **同基线 `361.9 MB`**（同一输入读 + 无中间物化；Vector 路径不减少任何 GM 流量） | 1（W 打包读 1 遍） | 片上 W tile `block_N·block_K·2` + K 累加器 `block_N·4`（与基线同量级） | 高（N 分块同基线，`N/block_N` 块）；每输出需 K 横向归约（长 K 需分块 + 累加，归约链长） |

> 口径说明：FLOPs 按乘加 = 2；访存为 GM 侧不可消除流量（不把 L1/UB 内部搬运、也不把 tiling 重读计入）；`block_M=16` 的 Cube 填充计入「实算」列。本算子为 **GM 带宽受限**（W 流量占 92.7%），故访存/遍数是主导口径（R3 不得只比 FLOPs）。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附佐证）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **基线（融合反量化 A16 直接法）** | ✅ MAC 密集段 → Cube（`T.gemm`→`mmadL1`）；反量化 → Vector（v-prefix）—— 两侧都有对应单元 | ✅ L0C 4 KiB、L1 ≈20 KiB、UB ≈150.0 KiB（2-stage；§4.5）（`CONST-capacity-910B2C` 上限 128/512/192 KiB）；UB 见 §4.5/§9.1.2 | ✅ `block_M≥16`、`block_K=64`（32B 尾轴 = 16 元素）对齐；`N` 尾由动态 extent 承接 | ✅ M/N/K 与 block 均为编译期常量（每 shape 特化） | ✅ `T.Pipelined` 多缓冲；CV 拆分由编译器完成（`PL-1.16` 记录 Developer 混排边界，`test_mix_kernel_bf16.py` 为同形态先例） | **✅ 选定** |
| 候选 1（先物化 `W_deq`） | ✅ Cube（第二 kernel） | ⚠️ 1.34 GB workspace 需分块，引入额外调度 | ✅ | ✅ | ❌ 两 kernel + GM 往返（`CONST-store-fixpipe-gm-only` 同族约束：跨引擎传输强制 GM） | ❌ 淘汰（访存 8.4×，回退掉 W1/W2 全部收益） |
| 候选 2（仿射分解） | ✅ Cube + Vector | ✅ | ✅ | ✅ | ⚠️ 需 A 分组行和归约（`T.reduce_sum` 沿 K，`CONST-reduce-dims0-skinny` 提示 dims=0 环形归约陷阱，须用 dims=1 形态） | ❌ 淘汰（fp16 相消风险 + 无访存收益 + 额外归约与 GEMM；代价 > 收益） |
| 候选 3（int8 Cube） | ❌ 破坏 A16 契约 | — | — | — | — | ❌ 淘汰（语义违反） |
| 候选 4（split-K） | ✅ Cube | ⚠️ partial 落 GM | ✅ | ✅ | 需两阶段归约 | ❌ 淘汰（本 manifest 并行度已足：N 分块 128 块 ≫ 20 核） |
| 候选 9（Vector 规约累加，m=1） | ❌ **主导 MAC 段落 Vector（`T.reduce_sum`）**——Vector 单元吞吐远低于 Cube A16；R3 已证无访存收益，故不能降低主导的流量项；Cube 的 `block_M=16` 填充实算（≈72–107 µs）本就被流量项掩盖（§1.6.0 D-2），换 Vector 只把同一 MAC 换到更慢单元 | ✅ 片上 W tile + 累加器，与基线同量级 | ⚠️ K 归约轴需向量宽度对齐；`K=81920` 的横向归约链长（需分块累加） | ✅ 编译期常量 | ⚠️ 纯 Vector，无 CV 融合；仅覆盖 m=1，破坏与 m>1 共用同一 kernel 的统一性 | ❌ 淘汰（无访存收益 + Vector 吞吐劣势 + 仅 m=1 形态，见 D-2 量化依据） |
| 候选 5/6/7/8（E1/E2/E3/E4） | ✅ 随基线 | ✅ 无额外驻留（E4 反而减半缓冲字节） | ✅ | ✅ | ✅ 融合进流水 | **✅ 采纳** |

**调研结论**：选定 **基线「分块融合反量化的 A16 Cube 直接法」+ E1（epilogue 单次舍入）+ E2（K 尾零填充）+ E3（元数据提升+广播）+ E4（fp16 域反量化）**。
- **关键依据**：R3 显示基线在访存口径上最优（361.9 MB vs 候选 1 的 3.04 GB）；候选 2 无访存收益且有 fp16 相消风险、候选 3 违反 dtype 语义、候选 4 在本 manifest 并行度下净亏、**候选 9（`ALG-gemv` R1 命中行候选「Vector 规约累加」）无访存收益且把 MAC 密集段落 Vector（吞吐劣势），仅对 m=1 形态有意义**（R4 逐项淘汰）。本算子为 **GM 带宽受限**（W 流量占 92.7%，§R3 口径说明），所有候选均受同一 GM 流量下界约束 ⇒ 能降低主导项的只有「减少 GM 流量」的候选（候选 1 反而增加、候选 9 不减少），故基线在主导口径上最优。R4 六项检查基线全绿，且 `examples/gemm/example_gemm.py`（官方 Cube 形态）+ `testing/npuir/bf16_support_ops/test_mix_kernel_bf16.py`（Vector 写 `alloc_shared` → `T.gemm` 消费）给出 NPU Developer 的最近形态。
- **与基线的结构差异**：算法族**不变**（不替换算法），仅做硬件组织重设计（§0.6 R1–R4）——反量化载体由 SMEM+线程改为 UB+Vector→Cube 交接。
- **调研范围**（「无更优替代」的举证）：`algorithm-candidates.md` 的 ALG-gemv 条目——**其 R1 命中行的两个候选「GEMM 分形路径（Cube，load_nd2nz）」与「Vector 规约累加」均已评估**（前者 = 基线候选 0，后者 = 候选 9，均在 R3/R4 量化对比后淘汰；本仓无 W4A16/反量化 GEMM 专条，`kb_search "gemm dequant int4 cube L1"` 亦未命中专条，已记录）；`pattern-library/INDEX.md` 全部条目 + `constants.md`、`examples/gemm/`（`example_gemm.py`/`matmul.py`/`matmul_dynamic_shape.py`/`example_gemm_int82int32.py`）、`examples/mixcv/`、`examples/TileOPs/tileops/kernels/gemm/`、源算子 §0.3/§0.4、模型知识（GPTQ/AWQ 族反量化 GEMM、split-K、Strassen 类不适用）、互联网检索未执行（本地源已覆盖候选空间；W4A16 反量化 GEMM 的结构选择由访存口径直接决定，无新算法空间）。
- **源算法优化手段承接**（§0.4 逐项，不得静默丢弃）：**W1**（INT4 打包态到核内）→ 契约保持，核内才物化；**W2**（融合反量化、不落 GM）→ §0.6 R1 承接（Vector 反量化 + Cube 消费）；**W8**（元数据标量提升）→ E3 承接（每 k-tile 一次 cast + `vbrc`）；**W3/W4/W5/W6/W11/W12** → §0.5 等价替换；**W7**（线程级解码）→ §0.6 R2/R4 重新设计；**W9/W10/W14** → §0.5 舍弃（附理由）。

**设计期估算（D-2）**：代表 workload `decode-long-k`（M=1, N=8192, K=81920, fp16；`block_M=16, block_N=64, block_K=64, num_stages=2`）。
- **流量项**：理想 GM 流量 `361.9 MB`（§R3 基线口径）。按 `CONST-mte2-degradation`（tilelang 0.1.2+ed787bb / 2026-09-07 build / CANN 8.5.0）256M 档混合读写地板 ~1.26 TB/s ⇒ ≈ **287 µs**；按 `CONST-fabric-bw`（1.73 TB/s 聚合上界，乐观）⇒ ≈ 209 µs。
- **发射项**：见下「未实测项」——Vector 反量化链（~11 op/字节 + 4 op/元素）与 MTE2 指令数（`(N/block_N)·(K/block_K) = 128·1280 ≈ 1.6e5` 次装载）。`CONST-vector-launch-overhead` 的 0.5µs/op 是**依赖串行链的发射延迟模型**，不适用于流水化块链（反证：`PL-1.6-copy-floor` 实测 7-pass 链在 16M 元素上相对 copy-only 仅 +1.1µs → 流水化后向量 pass 可被 MTE2 窗口吸收）。**故发射项标注「未实测假设 + 估算依据」**，其量级不确定，列入 §9.1.2 首位 Stage 4 实测项。
- **容量项**：UB ≈150.0 KiB（2-stage，§4.5）< 192 KiB、L1 ≈20.0 KiB < 512 KiB、L0C 4.0 KiB < 128 KiB；单次 L0C→GM（无跨引擎往返）⇒ ≈ 0。
- **`估算下界: ≈287 µs`（流量 287 + 发射项未定 + 容量 ~0；引用 CONST-mte2-degradation / CONST-fabric-bw）**。
- **Cube 计算项无实测常数**（pattern-library 未覆盖 Cube 吞吐，标注「未实测假设」）：`block_M=16` 分形填充下实算 `2·16·N·K = 2.15e13` FLOPs，按 300 TFLOPS 估 ≈72 µs、按 200 TFLOPS 估 ≈107 µs，均 < 流量项 ⇒ 选 `block_M=16` 是为了**保证算力项不翻转为主导项**（若用源的 `block_M=64`，实算 8.6e13 FLOPs → ≈287–430 µs，将把 m=1 decode 变成算力受限，故不采用）。此判据交 Stage 4 实测对账（`hardware-cost-model` §3）。
- 常数版本戳：tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b / CANN 8.5.0 / Ascend910B2C（`NPUUtils.get().get_aicore_num()=20`，2026-10-13 实查）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定算法。逐项给出「原式 → 优化后公式 → 等价性论证 → 收益量化」，且**采纳项全部经机器验证**（D-1，见本节末结果表）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| E1 | Epilogue 单次舍入融合 | `out_fp32 = A·Wᵀ`（fp32 中间）落 GM 后再 cast 到 fp16 | `out = round_fp16(A·Wᵀ)`，在 L0C→GM store 内**单次** RNE 舍入 | 数学恒等：两式都对同一 fp32 累加器做一次 RNE 到 fp16（重复舍入是幂等的——机器验证 E1，13 项全 `EQUIV_PASS`）；关键前提是**经过 fp32 opmath 域** | 省 `M·N·4` B 的 fp32 中间 GM 写+读（`decode-hbm`：`1·8192·4 = 32,768 B = 32.0 KiB` 往返；`smoke-rect`：`128·256·4 = 131,072 B = 128.0 KiB`）；并省一次向量 cast pass |
| E2 | K 尾零填充 | 对精确 K 求和 `Σ_{k<K}` | 补齐到 `block_K` 倍数的 K' 求和，A/W 的 K 尾区**都**填零：`Σ_{k<K'}`，`K'=ceil(K/block_K)·block_K` | 数学恒等：新增项 `A[m,k]·W[k,n] = 0·0 = 0`（两侧都清零，杜绝 `0·NaN`/`0·inf`）；仅改变 fp32 求和的加零操作（机器验证 E2，4 组边界 K × 2 dtype 全 `EQUIV_PASS`） | 使 `block_K ∤ K` 时免 host padding（host padding 被 `references/ascend-constraints.md` §4 禁止）；clear 代价仅尾区 |
| E3 | 组元数据 cast 提升 + 广播 | 源每 chunk 取一次 `scale/zero`（线程级标量），逐元素参与 `(q−z)·s` | 每 (n, k-tile) **只 cast 一次** `s[n,g]`/`z[n,g]`（fp32→fp16、uint8→fp16）→ `T.vbrc` 广播 `[block_N,1]→[block_N,block_K/2]` | 数学恒等：广播同一常量与逐元素重复取同一常量完全等价（同一元素同一算子；机器验证 E3，2 dtype 全 `EQUIV_PASS`） | 元数据 cast 次数 `N·(K/block_K)·block_K = N·K` → `2·N·(K/block_K)`：`decode-long-k` 从 `6.7e8` 次降到 `2.6e5` 次（约 −99.96%）；同时吸收源 W8 的「chunk 行内恒定」意图 |
| E4 | fp16 域反量化 | `W = (q_fp32 − z_fp32)·s_fp32` → cast fp16（源在 fp32 域算后落 `dequantized_local`(dtype)） | `W = (fp16(q) − fp16(z))·fp16(s)`，全程 fp16 | **容差内等价**：`q,z ∈ [0,15]` 在 fp16 精确；`s` 的 fp16 舍入引入 ≲1 ulp 相对误差，随后一次乘法舍入 → 实测 max abs 3.9e-3、max ulp 4.0，**违反率 0.000000**（对照本算子容差 atol=7e-2 / rtol=5e-2，`examples/TileOPs/tests/ops/test_gemm.py` L181） | 免 fp32 反量化中间缓冲（每 tile `block_N·block_K·4` B → `·2` B，UB 减半）+ 免每元素 fp32→fp16 转换 |

> 其余候选（先物化 `W_deq` / 仿射分解 / int8 Cube / split-K / **Vector 规约累加（候选 9）**）在 R1/R3/R4 已判不采纳，此处不重复论证。R1 候选 5（元数据提升）与候选 7（epilogue 融合）分别由 E3/E1 承接。

**机器验证（D-1）**：`examples/gemm_w4a16/_gemm_w4a16_kernel/verify_equiv.py`（torch CPU，fp64 参照；随机张量 + 角点值 ±inf/NaN/−0/subnormal + 边界 shape K=1/130/200/300、block_K∤K）。执行结果（`python3 examples/gemm_w4a16/_gemm_w4a16_kernel/verify_equiv.py`）：

| 采纳项 | dtype | 最大 ulp 差 | 违反率 | 结论 |
|--------|-------|------------|--------|------|
| E1-epilogue-cast-fusion | torch.float16 | 0.000 | 0.000000 | `EQUIV_PASS` |
| E1-epilogue-cast-fusion | torch.bfloat16 | 0.000 | 0.000000 | `EQUIV_PASS` |
| E2-ktail-zerofill-K130-bk64 | torch.float16 | 0.000 | 0.000000 | `EQUIV_PASS` |
| E2-ktail-zerofill-K130-bk64 | torch.bfloat16 | 0.000 | 0.000000 | `EQUIV_PASS` |
| E2-ktail-zerofill-K200-bk128 | torch.float16 | 0.000 | 0.000000 | `EQUIV_PASS` |
| E2-ktail-zerofill-K200-bk128 | torch.bfloat16 | 0.000 | 0.000000 | `EQUIV_PASS` |
| E2-ktail-zerofill-K1-bk16 | torch.float16 | 0.000 | 0.000000 | `EQUIV_PASS` |
| E2-ktail-zerofill-K1-bk16 | torch.bfloat16 | 0.000 | 0.000000 | `EQUIV_PASS` |
| E2-ktail-zerofill-K300-bk64 | torch.float16 | 0.000 | 0.000000 | `EQUIV_PASS` |
| E2-ktail-zerofill-K300-bk64 | torch.bfloat16 | 0.000 | 0.000000 | `EQUIV_PASS` |
| E3-metadata-cast-hoist | torch.float16 | 0.000 | 0.000000 | `EQUIV_PASS` |
| E3-metadata-cast-hoist | torch.bfloat16 | 0.000 | 0.000000 | `EQUIV_PASS` |
| E4-fp16-domain-dequant | torch.float16 | 4.000 | 0.000000 | `EQUIV_PASS` |

（E4 的 criterion 为容差判定——`examples/TileOPs/tests/ops/test_gemm.py` L181 的 atol=7e-2/rtol=5e-2；在该容差下违反率 0.000000，max |cand−golden| = 3.906e-3。）

**优化结论**：采纳 4 项（E1/E2/E3/E4），全部 `EQUIV_PASS`。**优化后公式**（供 §3.1 使用）：

$$
\mathrm{out}[m,n] \;=\; \mathrm{round_{fp16}}\!\Big(\sum_{k=0}^{K'-1} A[m,k]\cdot W_{16}[n,k]\Big),\quad
W_{16}[n,k] = \big(\mathrm{fp16}(q[n,k]) - \mathrm{fp16}(z[n,g])\big)\cdot \mathrm{fp16}(s[n,g])
$$

其中 `K' = ceil(K/block_K)·block_K`，`k≥K` 时 `A = W = 0`（E2）；`q` 由「符号无关 cast-only recipe」从 packed 字节复原（§0.6 R2 / §1.6.2）；`z/s` 每 (n, k-tile) 只 cast 一次并广播（E3）；`round_fp16` 在 store 内单次舍入（E1）；反量化在 fp16 域（E4）。除上述外，本算法无其他代数可约结构（候选 1/2/3/4 已淘汰，见 §1.6.0）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态（源） | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------------|-------------|---------|------------------------------|
| 1 | K 维累加 | `T.Pipelined` 内 `T.gemm`（tensor core） | Cube `T.gemm`（矩阵单元，非逐元素） | ✅（Cube 承载） | — |
| 2 | activation/packed/scale/zero tile 装载 | `T.copy` / 逐元素谓词 `T.Parallel` | `T.copy`（GM→UB/L1，MTE2 向量化，动态 extent 切片） | ✅ | — |
| 3 | **nibble 解包**（源 `T.serial(8)` 内的线程级标量 `decode_unsigned_int4`） | 标量位运算 `(byte >> (pos*4)) & 15` | **整块 v-prefix 向量链**（§0.6 R2 recipe）：`npuir_cast`(u8→f16) → `npuir_mul`(×1/16) → `npuir_cast`(f16→i16 floor) → `npuir_cast`(i16→f16) → `npuir_mul`/`npuir_sub` 得 `lo`；再一轮同构得 `hi`；`T.npuir_interleave(lo, hi, q)` 还原 k 序 | ✅（**探针 P2 实测 bit-exact**） | — |
| 4 | 仿射反量化 `(q−z)·s` | 源逐元素标量（线程级） | `npuir_sub`/`npuir_mul` 作用于整块 `[block_N, block_K]`；元数据 `T.vbrc` 广播 | ✅（E4 + E3，P2 实测 max_rel 7.9e-4） | — |
| 5 | 元数据取用 `s[n,g]/z[n,g]` | 源每 chunk 标量 `scale_local[0]=…`（W8） | 每 k-tile 一次 `npuir_cast` + `T.vbrc` 广播整块 | ✅（E3） | — |
| 6 | K 尾零填充 | —（源无此项） | `T.clear`（向量化 memset） | ✅（E2） | — |
| 7 | 输出写回 + cast | 源 `T.Parallel` 逐元素守卫写 | `T.copy(C_local[0:tm,0:tn], out[…])`（VPipe 向量化 + 单次 VCast） | ✅（E1/R5） | — |
| 8 | 源的 `for chunk in T.serial(block_n*block_k//2 // (threads*4))` 线程分片循环 | 线程级循环 + `T.vectorized(4)`/`T.vectorized(8)` | **整体消除**：无 thread binding 语义（`TRAP-threads-kwarg-noop`），改为整块向量 op | ✅（§0.6 R4） | — |
| 9 | block 起始索引 `m0 = by·block_M`、`n0 = bx·block_N`、`cid→(by,bx)` | 标量乘加/整除 | 无 | ❌ | **block 级索引 / 任务映射计算**：每 block 仅数个标量，非逐元素热点；索引不构成张量，无逐元素向量等价 API |
| 10 | `num_logical`/`num_launched`/`tm`/`tn` 边界 | 标量算术 | 无 | ❌ | **host 侧 / kernel 元数据计算**：编译期常量或每 block 一次的标量边界，非逐元素热点 |
| 11 | K 流水循环本身（`T.Pipelined`） | 迭代 | 无 | ❌ | **tile 级顺序依赖**：`C_local` 跨 K-tile 累加，循环携带依赖；已用多缓冲流水隐藏延迟，无法用归约替代（归约需先物化全 K，破坏 tiling 与 UB 预算） |

**向量化结论**：全部**逐元素**与**标量解码**计算点已向量化（#3–#8）；**矩阵乘由 Cube `T.gemm` 承载**（#1，非 Vector）；装载/写回由 `T.copy`（MTE2/VPipe）承载（#2/#7）。保留 3 处标量 / 循环（#9/#10/#11），理由类别：block 索引 / 任务映射、host 与 kernel 元数据、tile 级顺序依赖——均非逐元素热点。§6 循环结构不含任何逐元素标量循环，与本节一致。

**op 数权衡（与源位运算路径的对照，供 Stage 4 优化）**：

| 解包路径 | 每 packed 字节的向量 op 数 | 状态 |
|---------|--------------------------|------|
| **cast-only recipe（采纳，探针 P2）** | ≈11（解包）+ 4（反量化，摊到 2 元素）+ 1（interleave） | ✅ 探针 P2 机器验证 bit-exact；不依赖任何位运算 |
| int16 位运算路径（`vshr`+`vand`+cast） | ≈6 | ❌ 探针 P1 实测受 `vand`/`vshr` 操作数约束与 `int16→uint8` cast 不支持阻断（CG-2026-0019；`f16→i32`/`int32→half` cast 亦不支持） |
| `npuir_floordiv(f16, 标量)` | ≈8 | ❌ 探针 P1 实测**运行期 Vector 核超时**（CG-2026-0025） |

> 「cast-only recipe」的 11 op/字节看似偏多，但它是**当前工具链下唯一机器验证可行的路径**；位运算路径（更少 op）记为 Stage 4 候选，工具链修复后按 CG-2026-0018/0019 去绕化（`AGENTS.md` 版本戳重验规则）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> 本算子为 **Cube / MixCV 类**：I/O layout（`activation (M,K)`、`packed_weight (N,K/2)`、`weight_scale/zero (N,K/128)`、`output (M,N)`，均 ND 行主序）是**契约**；核内 Cube 分形布局与 **Vector 反量化缓冲的存放/交接方式**是**设计变量**。

**候选矩阵**（全部枚举；弃选给量化理由）：

| # | 方案 | 数据路径 | repack / 交接代价 | 预估收益 / 代价 | 是否采纳 |
|---|------|---------|------------------|----------------|---------|
| L1 | **Developer 统一缓冲**：packed/中间量用 `T.alloc_shared`（UB 侧），反量化结果 `W_sh` 同用 `T.alloc_shared`，由 `T.gemm` 直接消费；框架自动做 CV 拆分与 UB→Cube 交接 | GM→UB(packed) →(Vector 链)→ `W_sh` →(框架交接)→ Cube → L0C →(T.copy VCast)→ GM | 交接由编译器插入（非 GM 级重排；GM 流量不变） | 官方 MixCV 形态（`testing/npuir/bf16_support_ops/test_mix_kernel_bf16.py`：`T.vadd` 写 `alloc_shared` → `T.gemm` 消费，同一 Developer 内核）；零显式同步 | **✅ 采纳（主选）** |
| L2 | Expert 显式双 Scope：`T.Scope("Cube")` + `T.Scope("Vector")`，Vector 写 UB、经 **GM workspace** 中转，Cube 用 `load_nd2nz` 从 GM 取 `W_deq` | GM→UB→(向量链)→GM(ws)→L1(`load_nd2nz`)→Cube | **GM 往返** `N·K·2` B 写+读（`decode-long-k` = +2.68 GB） | 绕开 Developer 混排限制（`PL-1.16-expert-dualscope-bypass`），但直接回退 W1/W2 收益（§1.6.0 候选 1 同因淘汰） | ❌ 弃选（GM 往返 8.4× 流量；仅在主选被证实不可行时作最后手段，见 §9.1.2） |
| L3 | 核内 `T.transpose`/`T.reshape` 重排 packed 为 `[K, N]` 形态再解包 | — | `PL-1.1-transpose-chain` 实测 ~5.4 µs/链；且 `T.transpose` 仅支持二轴交换、dtype 矩阵 fp16 ✓/int8 ×（`docs/Tilelang.language/创建操作/T.transpose.md`） | 无收益：解包在 `[block_N, block_K/2]` 行主序上已是最内连续轴，重排只增加核内开销 | ❌ 弃选（无收益 + dtype 限制） |
| L4 | host 侧预解包/预反量化 | host 把 packed 展开为 fp16 `W_deq` | host permute/物化（`PL-1.3-host-permute` 106–147 µs 量级） | 被 `references/ascend-constraints.md` §4 **明文禁止**（host 不得改动输入张量内容、不得物化）；且流量 8.4× | ❌ 弃选（违约 + 负收益） |
| L5 | epilogue 归属 Vector（L0C→GM→UB 再逐元素处理） | L0C→GM→UB→GM | GM 往返 `M·N·2·2` B | 本算子 epilogue 无逐元素运算（仅裁剪写回 + cast），Vector 处理无收益；`CONST-store-fixpipe-gm-only`：L0C 跨引擎传输强制 GM 往返 | ❌ 弃选（无收益 + 引入 GM 往返） |

**逐候选评分（主选 L1）**：
- **整除 / 对齐**：`block_M=16`（Cube 分形 M≥16 的最小值）、`block_K=64`（32B 尾轴 = 16 元素，`64 % 16 == 0`）、`block_N=64`（N 轴 16 倍数）→ 全绿；`N` 尾由动态 extent 承接。
- **反量化缓冲的 UB/L1 归属**：`W_sh`（`alloc_shared`，Vector 写 + Cube 读）与 packed/中间缓冲的归属由编译器决定；UB 预算按 §4.5 按「全部在 UB」的保守口径核算：逐 buffer 2-stage 合计 **153,600 B = 150.0 KiB**（含 `W_sh`），×1.10–1.12 → **≈165–168 KiB < 192 KiB（196,608 B）**（`CONST-capacity-910B2C`）。
- **repack 代价**：无 GM 级全量重排（正确口径：GM 流量不变，代价在 UB→L1 交接的指令开销与 UB 容量）；L1 的交接开销**未实测**（`PL-1.16` 记录 Developer 混排的结构边界，`test_mix_kernel_bf16.py` 为同形态先例但未给代价数字）→ 属未实证常数。
- **向量轴**：逐元素向量维 = `block_K/2`（packed 轴，`[block_N, block_K/2]` 最内连续，32B 对齐；`block_K/2 = 32` 元素 = 64 B ≥ 2×32B）；反量化后的量化轴 = `block_K`（interleave 输出，`[block_N, block_K]` 最内连续）。

**布局决策结论**：主选 **L1 —— Developer 统一缓冲（`alloc_shared` + v-prefix 向量链 + `T.gemm` 直接消费）**；核内 Cube 布局为 **NZ 分形**（由 GM→L1 搬运隐式转换，`T.gemm` 以 `b_transpose=True` 直接消费 `W_sh[block_N, block_K]`），**epilogue 归属 Cube 侧**（L0C→GM 单次 VCast，L5 弃选）。Vector 向量轴 = `packed` 的 `block_K/2` 轴（解包）/ `block_K` 轴（反量化与 interleave）。该结论已同步 §3.3 伪代码、§4 内存规划（缓冲形状与 UB 预算）、§6 循环结构。

**实验裁决计划（判定裕度依赖未实证常数 → 强制三件套）**：

- **① 主选方案（进 Stage 3）**：L1 —— 单内核 Developer，`alloc_shared` 反量化缓冲 → `T.gemm` 直接消费；`block_M=16`、`block_N=64`、`block_K=64`、`num_stages=2`。
- **② 备选方案（≤1 个，结构完整、达 Stage 4 可直接实现）**：**L2'** —— 同一 Developer 主结构下，**把反量化缓冲的跨引擎交接显式化**：packed/元数据/中间量用 `T.alloc_ub`，反量化结果写入 GM workspace（`T.Tensor` 附加输出，shape `[num_logical?, block_N, block_K]` 或按 k-tile 环形复用），Cube 侧用 `T.alloc_L1` + `T.load_nd2nz` 取回后 `T.gemm`；CV 之间用 `T.sync_block_set/wait` 配对。buffer 形状与 UB 预算、分核三要素、循环结构按本任务粒度重算：UB 侧 `block_N·block_K·2 = 8 KiB/份`（链缓冲单份 ≈67 KiB，无流水膨胀，**不含 `W_sh`**——`W_sh` 经 GM workspace）；GM workspace 按 k-tile 双缓冲 `2·block_N·block_K·2 = 16 KB`（跨 block 复用同一 scratch，需 `num_logical` 个副本 → 改为 per-cid scratch `2·16 KB·num_logical`，`num_logical=128` 时 4 MB）；flag id ≤15/核（`CONST-flag-id-budget`），用 1 对 flag。
- **③ 实验裁决计划**：代表 shape = `decode-long-k`(1,8192,81920)、`decode-hbm`(1,8192,16384)、`smoke-rect`(128,256,256)；指标 = kernel latency（msprof）+ `vec_ratio`/`vec_time`（反解反量化链占用）+ `mte2_active_bw`；**判定阈值 = 备选 L2' latency < 主选 L1 × 0.95 才翻转**（若主选 UB 溢出或编译失败则无条件切备选）；裁决结果回写本小节并同步 pattern-library（`perf-feedback.md` 流程）。shape 特化工厂（`lru_cache`/每 shape 编译）→ 裁决计划**须评估按 shape 分派主选/备选**：decode 形态（m=1，A 极小、反量化占比高）与 prefill 形态（大 M、反量化可被 Cube 掩盖）的最优结构可能不同。
- **设计期探针（D-3，已执行，2 个，均 <10 min）**：`history_version/design_probe_p1_int4_unpack.py`（nibble 提取 op 可行性二分）与 `history_version/design_probe_p2_dequant_recipe.py`（cast-only recipe 正确性）。结论：recipe bit-exact（见 §1.6.2），**未标定** UB→L1 交接常数与 Cube 吞吐——交 Stage 4 实测（本设计不再起第三个探针，遵守 ≤2 预算）。
---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式：Developer**（`op_requirements.programming_mode = developer`，用户显式指定）。

### 2.2 选型理由

- 本算子的两个计算面都落在 Developer 的统一缓冲模型内：Cube 侧 `T.gemm`（`alloc_shared` 操作数 + `alloc_fragment` 累加器）、Vector 侧 v-prefix 向量链（`npuir_cast/npuir_mul/npuir_sub/npuir_interleave/vbrc`）——两者用同一 `alloc_shared` 缓冲衔接，**有同形态先例**：`testing/npuir/bf16_support_ops/test_mix_kernel_bf16.py`（`T.vadd` 写 `local_tmp` → `T.gemm` 消费），无需显式 `Scope`/手工同步。
- 官方最近模板 `examples/gemm/example_gemm.py`（Developer Cube）与 `examples/mixcv/example_mixcv.py`（Developer Cube+Vector 后处理）覆盖本设计的两半（AGENTS.md「Operator Implementation Baseline」要求优先改动既有案例）。
- 源算子的手工线程/线程级 chunk 分片、SMEM swizzle、warp policy 在 NPU 无对应能力（§0.5 W7/W9/W10），Developer 的自动向量化/自动 CV 拆分/自动同步正好替代这些手工部分。
- 源 GPU 的 `T.Kernel(threads=128)` 在 npuir 无效（`TRAP-threads-kwarg-noop`），Developer 模式下无需关心 thread 分片。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（Cube 操作数/反量化缓冲 → L1；packed 与向量中间量 → UB，由编译器按用途判定）；`T.alloc_fragment`（fp32 累加器 → L0C）；**不显式**用 `alloc_L1/alloc_L0C/alloc_ub` |
| 计算方式 | `T.gemm(A_sh, W_sh, C_local, initC=(kk==0), b_transpose=True)`（→ `hivm.hir.mmadL1`）；反量化用 v-prefix 向量链 |
| 数据搬运 | `T.copy`（GM↔shared/fragment，自动 ND→NZ、自动 dtype cast、动态 extent 切片） |
| 同步 | 自动（`T.Pipelined` 多缓冲 + CV 拆分由编译器插入），无手动 barrier（§7） |

> **模式边界风险**：`CG-2026-0010`（codegen：Developer 模式 **persistent + gemm 混排**）+ `PL-1.16-expert-dualscope-bypass` 记录 Developer 侧存在结构级边界。故本设计 Developer 主选**不使用核内 `T.serial` 多任务 persistent 分核**（§5.5 据此选择），persistent 化与显式双 Scope 列为 Expert 备选。

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（E1 单次舍入 + E2 K 尾零填充 + E3 元数据提升 + E4 fp16 域反量化）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `m0 = by·block_M`，`n0 = bx·block_N`；`cid → (by, bx)` | 由一维 `cid` 恢复输出块起点 |
| 2 | `tm = min(block_M, M−m0)`，`tn = min(block_N, N−n0)` | M/N 尾块有效范围 |
| 3 | `k0 = kk·block_K`，`g = k0 // 128`（整块恒定，R6） | K 分块起点与组下标 |
| 4 | `bf = fp16(packed_w[n0:n0+tn, k0/2 : k0/2+BK/2])`；`q ⇐` 符号无关 nibble 还原（§0.6 R2） | 解包（每字节 → 偶 k 低 nibble、奇 k 高 nibble） |
| 5 | `s16 = fp16(scale[n0:n0+tn, g])`，`z16 = fp16(zero[n0:n0+tn, g])`；`W_sh = (q − bcast(z16))·bcast(s16)` | E3 提升 + `vbrc` 广播；E4 fp16 域 |
| 6 | `C_local += A[m0:m0+tm, k0:k0+BK] · W_shᵀ` | A16 Cube 分块累加（fp32） |
| 7 | `out[m0:m0+tm, n0:n0+tn] = round_fp16(C_local)` | 有守卫写回 + 单次舍入（E1） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | Grid + `cid→(by,bx)` | `T.Kernel(num_launched, is_npu=True)` | 一维 grid；`bx = cid % ceil(N/block_N)`、`by = cid // ceil(N/block_N)` | Developer |
| 2 | 分配 Cube 操作数 | `T.alloc_shared((block_M, block_K), dtype)`、`T.alloc_shared((block_N, block_K), dtype)` | 形状（NT 布局，`b_transpose=True`） | Developer |
| 3 | 分配累加器 | `T.alloc_fragment((block_M, block_N), "float32")` | fp32 累加（`T.gemm.md` §2.3） | Developer |
| 4 | K 循环流水 | `for kk in T.Pipelined(T.ceildiv(K, block_K), num_stages=2)` | stage 数按 UB/L1 预算（§4.5） | Developer |
| 5 | 装载 activation / packed / 元数据 | `T.copy(A[m0:m0+tm, k0:k0+block_K], A_sh[0:tm, 0:block_K])`；`T.copy(packed_w[n0:n0+tn, k0//2 : k0//2 + block_K//2], pw)`；`T.copy(scale[n0:n0+tn, g:g+1], s32)`、zero 同理 | 动态 extent 尾块（`T.copy.md` §2.2.2 规则 2）；`g` 为标量组下标 | Developer |
| 6 | uint8 → fp16 提升 | `T.npuir_cast(pw, bf, round_mode="rint")` | 有符号语义（§0.6 R2），recipe 免疫 | Developer |
| 7 | `h = floor(bf/16)` | `T.npuir_mul(bf, 0.0625, t)` + `T.npuir_cast(t, h16, round_mode="floor")` + `T.npuir_cast(h16, hf, round_mode="rint")` | **不能用 `T.npuir_floordiv`（探针 P1：Vector 核超时，CG-2026-0025）**；**不能用 `npuir_shr/and`（CG-2026-0018/0019）** | Developer |
| 8 | `lo = bf − 16h`；`hi = h − 16·floor(h/16)` | `T.npuir_mul(hf, 16.0, hqx)` + `T.npuir_sub(bf, hqx, lo)`；第二轮同构得 `hi` | 精确整数运算（§1.6.1 E4 论证） | Developer |
| 9 | `q = interleave(lo, hi)` | `T.npuir_interleave(lo, hi, q)` | `[block_N, block_K/2] ×2 → [block_N, block_K]`；k 序还原（偶 k = 低 nibble） | Developer |
| 10 | 元数据广播 | `T.vbrc(zf, zb)`、`T.vbrc(sf, sb)` | `[block_N,1] → [block_N, block_K/2]`；**不用隐式列广播**（`PL-1.15-broadcast-perf-tax`：+12~16% 性能税） | Developer |
| 11 | 反量化 `(q−z)·s` | `T.npuir_sub(q, zb, t2)` + `T.npuir_mul(t2, sb, W_sh)` | fp16 域（E4） | Developer |
| 12 | 矩阵乘累加 | `T.gemm(A_sh, W_sh, C_local, initC=(kk == 0), b_transpose=True)` | → `hivm.hir.mmadL1`；NT 形态 | Developer |
| 13 | 写回 + cast | `T.copy(C_local[0:tm, 0:tn], out[m0:m0+tm, n0:n0+tn])` | 自动 VCast fp32→fp16（RNE），动态 extent 守卫（E1/R5） | Developer |
| 14 | K 尾零填充（可选） | `T.clear(A_sh[...])` / `T.clear(W_sh[...])` | 仅当 `block_K ∤ K`；本 manifest 不触发（E2） | Developer |

API 依据：`docs/Tilelang.language/线性代数操作/T.gemm.md`、`docs/Tilelang.language/内存操作/T.copy.md`（§2.2.2 规则 2 / §2.4）、`docs/Tilelang.language/内存操作/T.alloc_shared.md`、`docs/Tilelang.language/内存操作/T.clear.md`、`docs/Tilelang.language/数据类型转换操作/T.vcast.md`、`docs/Tilelang.language/shape操作/T.vbrc.md`、`docs/Tilelang.language/创建操作/T.transpose.md`（仅用于弃选论证）、`examples/gemm/example_gemm.py`、`examples/mixcv/example_mixcv.py`、`testing/npuir/bf16_support_ops/test_mix_kernel_bf16.py`、`testing/npuir/shape_manipulation_ops/test_slice_interleave_dev.py`。

### 3.3 计算伪代码

```python
import tilelang
import tilelang.language as T
from tilelang.utils import NPUUtils

# --- host: 分核策略参数（编译期常量；物理核数实查值 20，见 §5.5）---
@tilelang.jit(out_idx=[-1], target="npuir")
def _gemm_w4a16_kernel(M, N, K, group_size=128, dtype="float16",
                       block_M=16, block_N=64, block_K=64, num_stages=2):
    accum_dtype = "float32"
    n_blocks = T.ceildiv(N, block_N)
    num_logical = T.ceildiv(M, block_M) * n_blocks
    num_physical = NPUUtils.get().get_aicore_num()      # 实查（本机返回 20；仅用于选 block_N 对齐，见 §5.5）
    num_launched = num_logical                          # 非 persistent：每核一逻辑块，超发由运行时串行调度（见 §5.5）

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),                          # activation  fp16
        PW: T.Tensor((N, K // 2), "uint8"),                  # INT4 packed
        WS: T.Tensor((N, K // group_size), "float32"),       # scale
        WZ: T.Tensor((N, K // group_size), "uint8"),         # zero
        OUT: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(num_launched, is_npu=True) as (cid, _):
            by = cid // n_blocks
            bx = cid % n_blocks
            m0 = by * block_M
            n0 = bx * block_N
            tm = T.min(block_M, M - m0)
            tn = T.min(block_N, N - n0)

            A_sh = T.alloc_shared((block_M, block_K), dtype)          # Cube 操作数（L1）
            W_sh = T.alloc_shared((block_N, block_K), dtype)          # 反量化结果（Vector 写 → Cube 读）
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)  # L0C fp32

            # --- Vector 侧工作缓冲（UB）---
            pw = T.alloc_shared((block_N, block_K // 2), "uint8")
            bf = T.alloc_shared((block_N, block_K // 2), dtype)
            t  = T.alloc_shared((block_N, block_K // 2), dtype)
            hf = T.alloc_shared((block_N, block_K // 2), dtype)
            hx = T.alloc_shared((block_N, block_K // 2), dtype)
            lo = T.alloc_shared((block_N, block_K // 2), dtype)
            hi = T.alloc_shared((block_N, block_K // 2), dtype)
            h16 = T.alloc_shared((block_N, block_K // 2), "int16")
            q16 = T.alloc_shared((block_N, block_K // 2), "int16")
            q  = T.alloc_shared((block_N, block_K), dtype)
            t2 = T.alloc_shared((block_N, block_K), dtype)
            zb = T.alloc_shared((block_N, block_K), dtype)
            sb = T.alloc_shared((block_N, block_K), dtype)
            s32 = T.alloc_shared((block_N, 1), "float32")
            z8 = T.alloc_shared((block_N, 1), "uint8")
            sf = T.alloc_shared((block_N, 1), dtype)
            zf = T.alloc_shared((block_N, 1), dtype)

            for kk in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                k0 = kk * block_K
                g = k0 // group_size          # 整块在同一组内（R6）

                T.copy(A[m0:m0 + tm, k0:k0 + block_K], A_sh[0:tm, 0:block_K])
                T.copy(PW[n0:n0 + tn, k0 // 2: k0 // 2 + block_K // 2], pw[0:tn, 0:block_K // 2])
                T.copy(WS[n0:n0 + tn, g:g + 1], s32[0:tn, 0:1])
                T.copy(WZ[n0:n0 + tn, g:g + 1], z8[0:tn, 0:1])

                # --- nibble 解包（符号无关，探针 P2 bit-exact）---
                T.npuir_cast(pw, bf, round_mode="rint")            # u8 -> f16（i8 语义，recipe 免疫）
                T.npuir_mul(bf, 0.0625, t)
                T.npuir_cast(t, h16, round_mode="floor")           # h = floor(bf/16)
                T.npuir_cast(h16, hf, round_mode="rint")
                T.npuir_mul(hf, 16.0, hx)
                T.npuir_sub(bf, hx, lo)                            # lo = 低半字节（偶 k）
                T.npuir_mul(hf, 0.0625, t)
                T.npuir_cast(t, q16, round_mode="floor")           # floor(h/16) ∈ {-1,0}
                T.npuir_cast(q16, hx, round_mode="rint")
                T.npuir_mul(hx, 16.0, hx)
                T.npuir_sub(hf, hx, hi)                            # hi = 高半字节（奇 k）
                T.npuir_interleave(lo, hi, q)

                # --- 组仿射反量化（E3 提升 + E4 fp16 域）---
                T.npuir_cast(s32, sf, round_mode="rint")
                T.npuir_cast(z8, zf, round_mode="rint")
                T.vbrc(zf, zb)
                T.vbrc(sf, sb)
                T.npuir_sub(q, zb, t2)
                T.npuir_mul(t2, sb, W_sh)

                # --- A16 Cube 累加 ---
                T.gemm(A_sh, W_sh, C_local, initC=(kk == 0), b_transpose=True)

            T.copy(C_local[0:tm, 0:tn], OUT[m0:m0 + tm, n0:n0 + tn])

    return main
```

> 上伪代码为**主选方案 L1（Developer 统一缓冲）**。`W_sh` 为 `alloc_shared`，由 Vector 链写入、`T.gemm` 直接消费——其 UB/L1 归属与跨引擎交接由编译器处理（同形态先例 `testing/npuir/bf16_support_ops/test_mix_kernel_bf16.py`）。数据类型用位置参数传入（不用 `dtype=` 关键字），避免误报。
> **备选方案 L2'（显式交接，随 DESIGN 冻结进 Stage 4 A/B）**：把 `pw/中间量` 改 `T.alloc_ub`、`W_sh` 经 GM workspace 中转、Cube 侧 `T.alloc_L1` + `T.load_nd2nz` 取回，CV 间以 `T.sync_block_set/wait` 配对；结构细节与预算见 §1.6.3 ②。备选仅在 Stage 4 实测胜出（< 主选 × 0.95）时启用。

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|----------|---------|
| `T.Kernel(..., is_npu=True)`（一维） | `examples/gemm/example_gemm.py`、`docs/开发指南.md` §3.3 | 官方示例已验证 |
| `T.alloc_shared` / `T.alloc_fragment` | `docs/Tilelang.language/内存操作/T.alloc_shared.md` §2.4（L1 与 UB 双例） | 文档 + 官方示例 |
| `T.gemm(..., initC=, b_transpose=)`（fp16×fp16→fp32） | `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1/§2.3/§2.4 | 文档 + 官方示例 |
| `T.copy`（动态 extent 尾块；GM→UB/L1） | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 规则 2、§2.4 | 文档 |
| `T.Pipelined(..., num_stages=)` | `docs/开发指南.md` §3.2、`examples/gemm/example_gemm.py` | 官方示例已验证 |
| `T.npuir_cast` / `T.npuir_mul` / `T.npuir_sub` / `T.npuir_interleave` / `T.vbrc` | 本设计**探针 P2 实测通过**（`history_version/design_probe_p2_dequant_recipe.py`，bit-exact） | ✅ 实机验证 |
| `T.clear` | `docs/Tilelang.language/内存操作/T.clear.md`、`docs/开发指南.md` §2 示例 | 文档 |
| `NPUUtils.get().get_aicore_num()` | `docs/开发指南.md` §3.3；**实查返回 20** | ✅ 实机实查（2026-10-13） |
| Vector 写 `alloc_shared` → `T.gemm` 消费（CV 交接） | `testing/npuir/bf16_support_ops/test_mix_kernel_bf16.py` | 同形态先例（pytest 通过）；**本任务的完整形态待 Stage 3 验证**（§9.1.2） |
| `T.npuir_and` / `T.npuir_shr` / `T.npuir_floordiv` | 探针 P1 实测 | ❌ **不使用**（verifier/超时阻断，CG-2026-0018/0019/0025） |
| `T.load_nd2nz` / `T.alloc_L1` / `T.alloc_L0C` / `T.store_fixpipe`（备选 L2'） | `docs/Tilelang.language/内存操作/T.alloc_L1.md`、`docs/Tilelang.language/内存操作/T.load_nd2nz.md`、`docs/Tilelang.language/内存操作/T.store_fixpipe.md`、`examples/gemm/matmul.py`、`examples/mixcv/mixcv_mixkernel.py` | 文档 + 示例（备选） |
| `T.Kernel(threads=)` / `policy=` / `make_swizzled_layout` / `TL_ENABLE_FAST_MATH` | 源 GPU 代码（§0.4 W7/W9/W10/W14） | **不使用**（npuir 无对应 / no-op） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | 否（源为二维 grid） | 一维 `T.Kernel` + cid 分解（§0.6 R3） |
| GPU 专用 API 不可用 | 是（thread 分片 / swizzle / warp policy / fast-math+nvcc） | 逐项四态处置（§0.5），舍弃或等价替换 |
| GEMM 要求 M,N 为 block 整数倍 | **是**（M=1、N=7168/8192、block 不整除） | kernel 内动态 extent 切片 + 写回守卫（§0.6 R5）；**不做 host padding** |
| L0C 容量上限 128KB | 是 | `block_M·block_N·4 ≤ 131072` ⇒ `16×64×4 = 4 KB` ✅（`CONST-capacity-910B2C`） |
| 物理核数限制 | 是 | §5.5 分核三要素（实查 20） |
| host 侧禁止改动输入 / padding | 是 | 反量化与尾块处理全在 kernel 内；host 只做 shape/helper 与 block 启发式（`references/ascend-constraints.md` §4） |
| fp16 累加会卡死 | 是 | `C_local` 必须 fp32（`docs/Tilelang.language/线性代数操作/T.gemm.md` §2.3） |
| `T.gemm` 不支持 `int32×int32` | 否（不使用 int8/int32 路径） | — |
| UB auto-multi-buffer 膨胀 | **是** | UB 预算按 ×1.10–1.12 放大核算（§4.5，`CONST-capacity-910B2C`） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend NPU） | 转换方案 |
|--------|----------------|---------------------|---------|
| Kernel 维度 | 二维 grid `T.Kernel(n_blocks, m_blocks, threads=128)` | 一维 + cid 分解 | §0.6 R3 |
| 反量化载体 | 128 线程 chunk 解码写 SMEM | Vector 侧整块向量链写 `alloc_shared` | §0.6 R1/R4 |
| nibble 提取 | `(byte >> (pos*4)) & 15` 标量位运算 | 符号无关 cast-only recipe | §0.6 R2（探针 P2） |
| 组元数据 | 每 chunk 取标量放入线程寄存器 | 每 (n,k-tile) 一次 cast + `vbrc` 广播 | E3 |
| 反量化精度域 | `dequantized_local`(dtype) 承接、计算在 fp32 域 | fp16 域（容差内等价） | E4 |
| GEMM API | `T.gemm(..., transpose_B=True, policy=FullRow)` | `T.gemm(..., b_transpose=True)`（无 policy） | §0.5 W6/W9 |
| 内存层级 | SMEM + register fragment + swizzle | UB/L1（`alloc_shared`）+ L0C（`alloc_fragment`） | §0.5 W3/W10 |
| 非整除 M/N | 逐元素谓词装载 + epilogue 守卫 | 动态 extent 切片 + 写回守卫 | §0.6 R5 |
| 编译配置 | `TL_ENABLE_FAST_MATH` + `-O3 -DENABLE_BF16` | `pass_configs` 默认空 | §0.5 W14 |
| dtype | 仅 fp16 activation（op 已限制） | 同（保持） | 保留 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `testing/npuir/bf16_support_ops/test_mix_kernel_bf16.py` | **最高（本设计核心先例）** | Vector op（`T.vadd`）写 `alloc_shared` 缓冲 → `T.gemm` 直接消费；同一 Developer 内核、无显式 Scope/同步 |
| `examples/gemm/example_gemm.py` | 高 | Developer 模式 Cube GEMM：一维 grid + cid 分解、`alloc_shared`/`alloc_fragment`、`T.Pipelined` + `T.gemm(initC=)`、`T.copy` 写回 |
| `examples/mixcv/example_mixcv.py` | 高 | Cube + Vector 同核（`T.gemm` 后 `T.vexp` on fragment），Developer 模式 MixCV 骨架 |
| `examples/mixcv/mixcv_mixkernel.py` | 中 | Expert 双 Scope（`Scope("Cube")`/`Scope("Vector")`）+ `alloc_L1/alloc_L0C/alloc_ub` + `sync_block_set/wait`（备选 L2' 的结构模板） |
| `examples/gemm/matmul_dynamic_shape.py` | 中 | 动态 extent 尾块（`T.min`）与 Expert 显式 NZ 路径 |
| `examples/gemm/matmul.py` | 中 | Expert `alloc_L1`/`alloc_L0C`/`load_nd2nz`/`store_fixpipe`（备选 L2' 结构） |
| `testing/npuir/shape_manipulation_ops/test_slice_interleave_dev.py` | 中 | `T.npuir_interleave` 的正确形态（含 partial dst 切片） |
| `examples/TileOPs/tileops/kernels/gemm/gemm_w4a16/_gemm_w4_a16_kernels.py` | 高（源/Golden 语义参照） | GPU 抽取件（语义与算法参照，非实现基线） |
| `examples/TileOPs/tests/ops/test_gemm.py` | 高（Golden/容差） | `GemmW4A16Test.ref_program` + 容差 atol=7e-2/rtol=5e-2 |
| `examples/TileOPs/tileops/workloads/gemm.py` | 高（Golden/打包约定） | `quantize_weight_int4`（打包/反量化基准、低 nibble = 偶 k） |
| `examples/gemm/_gemm_kernel/DESIGN.md` | 中（Cube 兄弟设计） | 迁移分析结构、分核策略、D-2 roofline 口径 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `activation` | `(M, K)` | float16 | 左操作数 A（op 仅支持 fp16） |
| `packed_weight` | `(N, K/2)` | uint8 | INT4 打包（低 nibble = 偶 k，高 nibble = 奇 k） |
| `weight_scale` | `(N, K/group_size)` | float32 | 组仿射 scale（`group_size = 128`） |
| `weight_zero` | `(N, K/group_size)` | uint8 | 组仿射 zero |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `output` | `(M, N)` | float16（same_as(activation)） | ND 连续，无转置 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级（预期） | 用途 |
|-----------|-------|-------|-----------------|------|
| `A_sh` | `(block_M, block_K)` | float16 | L1（Cube 操作数） | A tile |
| `W_sh` | `(block_N, block_K)` | float16 | UB 写 / L1 读（编译器交接） | 反量化后的 W tile（`T.gemm` src2） |
| `C_local` | `(block_M, block_N)` | float32 | L0C | fp32 累加器 |
| `pw` | `(block_N, block_K/2)` | uint8 | UB | packed W tile |
| `bf/t/hf/hx/lo/hi` | `(block_N, block_K/2)` | float16 | UB | nibble 解包中间量 |
| `h16/q16` | `(block_N, block_K/2)` | int16 | UB | floor 商（`h`、`floor(h/16)`） |
| `q` | `(block_N, block_K)` | float16 | UB | interleave 还原的量化值 |
| `t2/zb/sb` | `(block_N, block_K)` | float16 | UB | 反量化中间量与广播后的元数据 |
| `s32/z8/sf/zf` | `(block_N, 1)` | float32/uint8/float16 | UB | 组元数据与其 fp16 提升 |

### 4.4 内存搬运路径

```
GM[activation (M,K) f16] --T.copy(MTE2)--> L1 A_sh[bm,bk] ─────────────────────────────┐
GM[packed_weight (N,K/2) u8] --T.copy--> UB pw[bn,bk/2]                               │
GM[weight_scale (N,K/128) f32] --T.copy--> UB s32[bn,1]                               │
GM[weight_zero (N,K/128) u8] --T.copy--> UB z8[bn,1]                                  │
        │  Vector 链 (npuir_cast/mul/sub/interleave; vbrc)                           │
        ▼                                                                             │
   UB W_sh[bn,bk] f16（反量化结果）──（CV 拆分：UB→Cube 交接，编译器插入）─────────────┤
                                                                                      ▼
                       T.gemm(mmADL1, b_transpose=True, initC=(kk==0)) --> L0C C_local[bm,bn] f32
                                                                                      │ T.copy（VPipe 单次 VCast）
                                                                                      ▼
                                                                                GM[output (M,N) f16]
宿主边界：host 仅做 (m,n,k,group_size) 推断、block 启发式、分核参数与 shape/stride 元数据；不触碰张量内容、不 padding、不做反量化。
K 流水：T.Pipelined(num_stages=2) 覆盖上述装载→反量化→GEMM→（隐含）写回链条。
```

### 4.5 UB / L1 / L0C 预算

> 本算子为 **MixCV**：`A_sh`/`W_sh` 为 Cube 操作数（L1 侧），向量链与 packed/元数据在 UB 侧。**保守口径**：把反量化链中间量（含 `W_sh`）全部计入 UB，并按 `CONST-capacity-910B2C` 的 auto-multi-buffer 膨胀系数 ×1.10–1.12 放大（UB 上限 192 KiB / AIV）。
>
> **单位口径（全文统一）**：`1 KiB = 1024 B`；容量上限一律按二进制口径——UB 192 KiB = 196,608 B、L1 512 KiB、L0C 128 KiB。逐 buffer 表给精确 Bytes，汇总列同时给 Bytes 与 KiB（= B/1024）。全文其余位置只引用本节汇总值，不再出现第二套数值。

`block_M=16, block_N=64, block_K=64, num_stages=2`（`block_K/2 = 32`）：

| Buffer | 单份大小 (Bytes) | ×份数 | 小计 (Bytes) |
|--------|-----------------|-------|-------------|
| `pw` (u8, 64×32) | 2048 | 2（流水） | 4096 |
| `bf/t/hf/hx/lo/hi` (f16, 64×32) | 4096 各 | 6×2 | 49152 |
| `h16/q16` (i16, 64×32) | 4096 各 | 2×2 | 16384 |
| `q/t2/zb/sb` (f16, 64×64) | 8192 各 | 4×2 | 65536 |
| `W_sh` (f16, 64×64) | 8192 | 2 | 16384 |
| `s32/z8/sf/zf` (64×1) | ≤256 各 | 4×2 | ~2048 |
| **UB 合计** | | | **153,600 B = 150.0 KiB** |
| 按 ×1.10–1.12 膨胀 | | | **≈165–168 KiB < 192 KiB（196,608 B）✅（余量 ≈12–14%）** |

> UB 余量偏紧（≈12–14%，以膨胀后上界计）。**缓解手段（按优先级）**：① 复用中间缓冲（`t` 与 `hx` 可复用；`q` 可复用 `t2` 的存放）→ 可回收 ≈24 KiB；② `num_stages=2 → 1` 仅对反量化缓冲（把 `T.copy` 的流水与向量链解耦）——需 Stage 3 按实际 lowering 决定；③ **N 向 dequant 分块**（`deq_N`，静态 `T.serial`，**约束 `deq_N | block_N` 且 `deq_N ≥ 16`**，把 `[block_N, block_K/2]` 的链缓冲缩到 `[deq_N, block_K/2]`），是 `block_N` 增大（分核对齐需要）时的**必备机制**（§5.2/§5.5）。**注意**：`W_sh` 是 `T.gemm` 的 src2，必须保持整 `block_N`，**不随 deq 分块缩小**，须单列进 UB 预算。取 `block_N=64, deq_N=64`（默认，不分块）时 2-stage 峰值 = 150.0 KiB（上表）；取 `block_N=208, deq_N=16`（13 段）时 2-stage 峰值 ≈85.5 KiB（含整 `block_N` 的 `W_sh` 53,248 B）→ ×1.10–1.12 ≈ **94–96 KiB < 192 KiB**（§5.5 末重算表）。
> `T.Pipelined` 是否对全部链缓冲做 auto-multi-buffer 属**未文档化**（本表按「全部 ×num_stages」最悲观口径），Stage 3 首验证时按 msprof 的 UB 实际占用对账。

| 层级 | 预算 | 上限 | 状态 |
|------|------|------|------|
| L1（`A_sh + W_sh`） | `(16·64 + 64·64)·2 = 10,240 B` ×2 stages = **20,480 B = 20.0 KiB** | 512 KiB | ✅ 余量充足 |
| L0C（`C_local`） | `16·64·4 = 4,096 B = 4.0 KiB` | 128 KiB | ✅ |
| UB（反量化链，见上表） | **≈165–168 KiB**（膨胀后上界） | 192 KiB | ⚠️ 余量 ≈12–14%（§9.1.2） |

### 4.6 动态轴定义

首次设计按 manifest 的 `(M,N,K)` 编译期特化（每 shape 一个 kernel，工厂 `lru_cache` 缓存），**无运行时动态轴**。若后续需动态 shape，参考 `examples/gemm/matmul_dynamic_shape.py` 的 `T.symbolic` 形态，但 manifest 为静态 shape，不引入。

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| 无 | — | — |

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[-1], target="npuir")
def _gemm_w4a16_kernel(M, N, K, group_size=128, dtype="float16",
                       block_M=16, block_N=64, block_K=64, num_stages=2):
    ...
```

（去除源 GPU 的 `pass_configs={TL_ENABLE_FAST_MATH: True}` 与 `compile_flags=["-O3","-DENABLE_BF16"]`——§0.5 W14；若 Stage 3 遇到 codegen 问题再按 `tilelang-error-fixer` 加最小 pass_config。）

---

## 5. Tiling 策略

### 5.1 计算类型

**类型：MixCV（Cube + Vector）**。
判定依据（`tilelang-mixcv-skill` 的 Developer 判定规则）：同一 kernel 内既有 Cube 侧 `T.gemm`，又有 Vector 侧 v-prefix 算子（`T.npuir_cast/npuir_mul/npuir_sub/npuir_interleave/T.vbrc`）。主导计算为 Cube（A16 矩阵乘，算力/带宽受限），Vector 承担量化前置变换。

### 5.2 Block 划分

```python
block_M = 16     # Cube 分形 M≥16 的最小值；m=1 decode 下把填充浪费压到 1/16（见 §1.6.0 D-2 算力判据）
block_N = 64     # N 轴 ≥16 且 16 倍数；UB 预算内（§4.5）；分核对齐时按 §5.5 扫描（必要时启用 deq 分块）
block_K = 64     # 32B 尾轴 = 16 元素 ✓；block_K | group_size(128) 且 ≤ group_size（组内的 k-tile）
num_stages = 2   # UB 预算内（§4.5）；k_iters = K/64 = 128~1280，流水充分
deq_N = 64       # N 向反量化分块；约束 deq_N | block_N 且 deq_N % 16 == 0（默认 = block_N，即不分块）
num_logical = ceil(M/block_M) * ceil(N/block_N)
```

启发式（Stage 3/4 固化）：`block_M = 16`（`M ≤ 16`，即全部 decode workload）/ `32~64`（`M ≥ 64` 的 smoke 形态，减少 M 维块数）；`block_N` 由 UB 预算与分核对齐共同决定（§5.5）；`block_K ∈ {32, 64, 128}` 且满足 `block_K | 128`；`deq_N` 仅在 `block_N > 64`（分核对齐需要）时小于 `block_N`，且**必须整除 `block_N`**（`for c in T.serial(block_N // deq_N)` 覆盖全部行）——`W_sh` 必须整 `block_N`，不随 `deq_N` 缩小（§4.5/§5.5）。

### 5.3 约束分析

- **对齐约束**：`block_K = 64` ⇒ packed 轴 `block_K/2 = 32` 元素 = 64 B（≥2×32B）✓；`interleave` 输出尾轴 `block_K = 64` 元素 = 128 B ✓；`block_N = 64`、`block_M = 16` 均 ≥16（Cube 分形）✓。
- **L0C 容量**：`16 · 64 · 4 = 4,096 B = 4.0 KiB ≤ 128 KiB` ✓（`CONST-capacity-910B2C`）。
- **L1 容量**：`(block_M·block_K + block_N·block_K)·2 · num_stages = 20,480 B = 20.0 KiB ≤ 512 KiB` ✓。
- **UB 容量**：见 §4.5（膨胀后 ≈165–168 KiB / 192 KiB，余量 ≈12–14%，缓解手段已列）。
- **K 组对齐**：`gcd(manifest K) = 128`（K ∈ {256, 8192, 16384, 20480, 81920}……其中 20480/128 = 160 ✓、81920/128 = 640 ✓），`block_K = 64` 整除全部 K ⇒ **K 尾零填充在本 manifest 不触发**（E2 仍实现以增强健壮性）。
- **group 恒定**：`block_K = 64 ≤ group_size = 128` 且整除 ⇒ 每个 k-tile 落在唯一组内，`g = k0 // 128` 在 tile 内为标量（E3 前提，R6）。

### 5.4 注意事项

- **非整除 M/N**：M=1（4 个 decode workload）与 N ∈ {7168, 8192}（非 64 整数倍？8192/64 = 128 ✓、7168/64 = 112 ✓、256/64 = 4 ✓ —— 本 manifest **N 均被 64 整除**）；M 侧的 m=1 强制走尾块路径（`tm = 1`）⇒ **必须**实现动态 extent 装载 + 写回守卫（§0.6 R5），**禁止 host padding**。
- **零 block 风险**：使用 `T.ceildiv` 而非整除，避免 `M < block_M` 时零 block 启动；`tm/tn ≥ 1`。
- **`tm = 1` 的 A tile**：装载仅 1 行；`A_sh` 其余行保持未初始化——**不影响正确性**（对应输出行不写回），但**不得**让未初始化行参与 NaN 传播到有效行（matmul 行独立，安全）。若 Stage 3 观测到未初始化值产生非有限数传播，改为 `T.clear(A_sh)`（代价可忽略）。
- **N 整除快路径**：源按 `n % block_n == 0` 分支（§0.4 W12）；NPU 统一用动态 extent 切片，无需分支。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1；依据 `docs/开发指南.md` §3.3。

- **① 逻辑核数计算**：`num_logical = ceil(M/block_M) × ceil(N/block_N)`（按算子实际输出网格）。
- **② 物理核数依据**：`NPUUtils.get().get_aicore_num()` **实查值 = 20**（本算子为 Cube/混合算子，直接使用返回值、不翻倍）。查询代码与实际输出：
  ```python
  from tilelang.utils.npu_utils import NPUUtils
  print(NPUUtils.get().get_aicore_num())   # 本机实查输出: 20
  ```
  （设备：Ascend910B2C，2026-10-13 实查。注：`pattern-library/constants.md` 的 `CONST-aicore-910B2C` 记 24 核——该条目绑定的任务/机器与本机不同；**本设计以实查值 20 为准**，并保留「换设备须重查」的提醒。）
- **③ 规模判定与分核方案**：按 `block_M=16, block_N=64` 对 manifest workload 计算逻辑核数——

  | workload | (M,N,K) | `ceil(M/16)` | `ceil(N/64)` | `num_logical` | 判定 |
  |----------|---------|--------------|--------------|---------------|------|
  | `compile-smoke-square-64x64x128` | (64,64,128) | 4 | 1 | **4** | ≤20 → **无需适配** |
  | `compile-smoke-rect-128x256x256` | (128,256,256) | 8 | 4 | **32** | 中等规模 |
  | `decode-l2-resident-ish` | (1,8192,8192) | 1 | 128 | **128** | 中等规模（6.4×物理核数） |
  | `decode-hbm-streaming-threshold` | (1,8192,16384) | 1 | 128 | **128** | 同上 |
  | `decode-non-power2-low-cta` | (1,7168,20480) | 1 | 112 | **112** | 中等规模（5.6×） |
  | `decode-long-k-pressure` | (1,8192,81920) | 1 | 128 | **128** | 同上 |

  三档方案（标准 §1 要素③）：
  1. **小规模（`num_logical ≤ 20`）**：`compile-smoke-square`（4）→ 结论「**无需适配**」；启动 `num_logical` 个核（一核一逻辑块，无负载不均）。
  2. **中等规模（`num_logical` 超过 20，本 manifest 主力：32/112/128）**：通过调整 `block_N` 使内核总数接近物理核数 20 的整数倍（按实查核数取 1×/2×/3× = 20/40/60）。**worked example**：`decode-*` 的 N=8192，取 `block_N = 208`（13×16，16 的整数倍）⇒ `ceil(8192/208) = 40 = 2×20`（恰好 2 波，消除「21 个内核→某核双倍任务」式负载不均）；N=7168 ⇒ `block_N = 176`（11×16）⇒ `ceil(7168/176) = 41 ≈ 2×20`。**注意**：`block_N` 超过 64 时反量化链的 UB 峰值会超限，**必须同时启用 §5.2 的「N 向 dequant 分块」，且满足 `deq_N | block_N` 与 `deq_N % 16 == 0`**（静态 `T.serial(block_N // deq_N)` 边界，把链缓冲缩到 `[deq_N, block_K/2]`）。worked example 取 `deq_N = 16`：`208 // 16 = 13` 段（全覆盖）、`176 // 16 = 11` 段（全覆盖）。**`W_sh` 是 `T.gemm` 的 src2，必须保持整 `block_N`（不随 deq 分块缩小）**，须单列进 UB 预算。

     **UB 重算（`block_N=208, deq_N=16, block_K=64, num_stages=2`，§4.5 口径）**：

     | Buffer（deq_N=16） | 单份 (Bytes) | ×2 stages | 小计 (Bytes) |
     |---|---|---|---|
     | `pw` (16×32 u8) | 512 | 2 | 1,024 |
     | `bf/t/hf/hx/lo/hi` (16×32 f16 ×6) | 1,024 各 | 6×2 | 12,288 |
     | `h16/q16` (16×32 i16 ×2) | 1,024 各 | 2×2 | 4,096 |
     | `q/t2/zb/sb` (16×64 f16 ×4) | 2,048 各 | 4×2 | 16,384 |
     | `W_sh`（208×64 f16，**整 `block_N`，不可 deq 分块**） | 26,624 | 2 | 53,248 |
     | `s32/z8/sf/zf` (16×1) | ≤64 各 | 4×2 | ~512 |
     | **UB 合计** | | | **87,552 B = 85.5 KiB** |
     | 按 ×1.10–1.12 膨胀 | | | **≈94–96 KiB < 192 KiB ✅** |

     即 `block_N=208 + deq_N=16` 的 UB 峰值 ≈94–96 KiB（含整 `block_N` 的 `W_sh` 53,248 B），远低于 192 KiB。旧表述「UB 峰值降到 ≈60 KB」作废——该值取自 `deq_N=32` 且未计入整块 `W_sh` 的错误口径（`208 // 32 = 6` 仅覆盖 192 行，`W_sh` 也无法 deq 分块）。
     - 主选（Developer）**不采用核内 `T.serial` persistent 分核**：`CG-2026-0010`（codegen：Developer 模式 persistent + gemm 混排）+ `PL-1.16-expert-dualscope-bypass` 记录该结构在 Developer 侧存在结构性边界；`block_N` 对齐方案已能把核数压到物理核整数倍附近，无需 persistent。
  3. **极大规模（`num_logical ≫ 物理核数`）**：本 manifest 未出现（最大 128 = 6.4×20，经 `block_N` 对齐可到 40 = 2×20）。若未来 shape 使 `block_N` 无法对齐（受 L0C/UB 上限约束），按标准 §1 要素③ 切 **Expert persistent**：`T.Kernel(num_physical)` + 核内 `for t in T.serial(tasks_per_kernel)`（`tasks_per_kernel` 编译期常量、边界静态）+ `cid = t*num_physical + kernel_id`；结构见 `docs/开发指南.md` §3.3 与备选 L2'。
- **分核方案（主选，可实施）**：
  - `num_logical ≤ 20`：`num_launched = num_logical`，一核一逻辑块；
  - `num_logical > 20`：`num_launched = num_logical`（非 persistent，Developer 安全），每核一个逻辑块，超发由运行时串行调度；同时按上条**优先用 `block_N` 对齐核数**（推荐 `block_N ∈ {64, 208}`，均满足 16 倍数；取 `block_N > 64` 时配套 `deq_N | block_N`（如 `block_N=208 → deq_N=16`）；按 shape 由 host 启发式选择，`lru_cache` 按 shape 特化）。**trade-off**：`block_N=64` 结构最简单（不分块、UB 峰值 165–168 KiB），但 `num_logical=112~128 = 5.6~6.4×20`，尾波负载偏斜；`block_N=208 + deq_N=16` 把核数压到 40 = 2×20（消除偏斜）并降 UB 峰值至 ≈94–96 KiB，代价是 Vector 链多 13 次 `T.serial` 迭代（编译期展开、静态边界）。
  - 不采用核内 `T.serial` 串行多任务（避免 Developer+persistent+gemm 结构边界）；persistent 化留作 Expert 备选（Stage 4 A/B）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化（v-prefix / `T.copy` / `T.interleave` / `T.gemm`）；下表循环仅指 block 级 / tile 级调度循环，**无逐元素标量循环**。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 输出块（M×N） | block 级并行 | `T.Kernel(num_launched, is_npu=True)` + cid 分解 | 每 slot 一个逻辑输出块（§0.6 R3） |
| K 方向 | tile 级流水迭代 | `for kk in T.Pipelined(ceildiv(K, block_K), num_stages=2)` | K 分块累加，MTE2 装载 / Vector 反量化 / Cube 计算重叠（§0.6 R1） |
| （可选）N 向反量化分块 | 静态 `T.serial` | `for c in T.serial(block_N // deq_N)`（约束 `deq_N \| block_N` 且 `deq_N % 16 == 0`） | 仅当 `block_N > 64` 时启用，约束 UB 峰值（§5.2/§5.5）；边界 `block_N // deq_N` 为编译期常量；`W_sh` 仍整 `block_N` |
| tile 内元素 | 向量化（MTE / Vector / Cube） | `T.copy`（装载/写回）、v-prefix 链（解包/反量化）、`T.npuir_interleave`、`T.gemm` | §1.6.2；无标量逐元素循环 |

### 6.2 循环伪代码

见 §3.3（主选 L1）。要点：
- K 循环体内顺序固定为：**装载 → 解包 → 反量化 → `T.gemm`**（数据依赖链，`T.Pipelined` 负责跨迭代多缓冲）；
- 每迭代的 `A_sh`/`W_sh` 均被完整重写（`npuir_sub/npuir_mul` 写满 `W_sh`），无跨迭代残留；
- `C_local` 用 `initC=(kk==0)` 在首迭代初始化，其余迭代累加。

### 6.3 流水线优化

使用 `T.Pipelined(ceildiv(K, block_K), num_stages=2)`：编译器生成 MTE2（装载）与 Vector（反量化）/Cube（`T.gemm`）的重叠流水。`num_stages=2` 依据 §4.5 UB 预算（反量化链缓冲按 ×num_stages 计）；`k_iters = K/64 ∈ {4, 128, 256, 320, 1280}`（manifest），除最小 smoke（K=128 → 2 次迭代）之外远大于 `docs/开发指南.md` §3.2 建议的 4~6，流水可充分填充。与源 GPU 的 `T.Pipelined(num_stages=2)`（§0.4 W5）等价。

### 6.4 尾块处理

- **M 尾块**：`tm = T.min(block_M, M − m0)`（decode 下 `tm = 1`）；装载用动态 extent 切片、写回用 `T.copy(C_local[0:tm,0:tn], …)` 守卫（`docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 规则 2）。
- **N 尾块**：`tn = T.min(block_N, N − n0)`；同上。本 manifest N 均被 64 整除，仍实现以保证健壮性。
- **K 尾块**：工厂强制 `group_size % block_K == 0`、`K % group_size == 0` ⇒ `block_K | K` 恒成立，`tk = block_K`，**不触发**；若未来放宽，则按 E2 对 `A_sh`/`W_sh` 的 K 尾 `T.clear`（A、W 两侧都清）。**本设计不实现 K 尾的逐元素谓词**（避免走源 GPU 的标量谓词路径）。
- **对齐填充**：尾轴 `block_K = 64`、packed 轴 `32` 均 ≥16 元素（32B），编译器无需额外对齐填充。

---

## 7. 同步策略

### 7.1 同步模式

**模式：自动同步**（Developer 模式）。CV（Vector 反量化 → Cube `T.gemm`）之间的交接由编译器的自动 CV 拆分与依赖分析插入同步，**无手写 barrier / flag**。

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|------|----------|------|
| K 流水 MTE2 ↔ Vector ↔ Cube | `T.Pipelined` 编译器自动插入多缓冲同步 | Developer 自动同步；替代源 GPU 的隐式异步拷贝依赖（§0.4 W4/W5） |
| 反量化（UB 写） ↔ `T.gemm`（Cube 读 `W_sh`） | 编译器自动（CV 拆分 + 依赖） | 同形态先例 `testing/npuir/bf16_support_ops/test_mix_kernel_bf16.py`（`T.vadd` 写 → `T.gemm` 读，无显式同步） |
| L0C → GM 写回 | `T.copy` 自动插入依赖同步 | 保证 Cube 累加完成后才 VCast 写出 |
| 跨核 | 无需 | 每个逻辑块独立，无跨核数据交换（无 split-K，§1.6.0 R2） |

无手动同步点（无 `T.barrier_all` / `T.sync_block_set/wait` / `T.pipe_barrier`）。**备选 L2'** 若启用，则需 1 对 `T.sync_block_set/wait` 标志（flag id ≤15/核，`CONST-flag-id-budget`）。

### 7.3 pass_configs 配置

```python
pass_configs = {}   # Developer 模式默认；源 GPU 的 TL_ENABLE_FAST_MATH 不适用 npuir（§0.5 W14）
```

若 Stage 3 出现 codegen/UB 溢出问题，按 `tilelang-error-fixer` 与 `CG-2026-0010`/`PL-1.16` 的经验最小化加入，不在设计期预设。

### 7.4 CV 融合设计

**本算子即 MixCV**（§5.1）：Cube 侧 `T.gemm` + Vector 侧反量化链在同一 kernel 内，按 Developer 统一缓冲模型衔接。分工与交接：
- **Vector 侧**：`pw`（UB）→ 解包/反量化 → `W_sh`；元数据 cast 与 `vbrc` 广播；
- **Cube 侧**：`A_sh`、`W_sh` → `T.gemm` → `C_local`（L0C）→ `T.copy` 写回；
- **交接**：不落 GM（`L1` 主选），由编译器在 UB→Cube 之间插入搬运；**不采用** L0C→Vector 方向（`CONST-store-fixpipe-gm-only`：该方向强制 GM 往返，且本算子 epilogue 无逐元素运算，L5 弃选）。
- **双 AIV 提示**：`PL-1.13-aiv-dup-subid-split` 指出 Mix kernel 的双 AIV 默认**重复执行** Vector 程序（除非按 `subid` 分片）；对本算子的影响是反量化 Vector 工作量可能 ×2（性能而非正确性），列入 §9.1.2 的 Stage 4 诊断项。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（移植源仓 `GemmW4A16Test.ref_program` + `quantize_weight_int4`），**不得复刻 §0.6 的 NPU 算法**——保证验证独立性。

```python
import torch


def quantize_weight_int4(weight: torch.Tensor, group_size: int = 128):
    """组仿射 INT4 量化+打包（源仓参考实现，逐行移植）。

    返回 (packed[N, K/2] uint8, scale[N, G] fp32, zero[N, G] uint8, dequantized[N, K] fp32)。
    打包约定：低 nibble = 偶数 k，高 nibble = 奇数 k。
    """
    n, k = weight.shape
    grouped = weight.float().reshape(n, k // group_size, group_size)
    group_min = grouped.amin(dim=-1).clamp_max(0)
    group_max = grouped.amax(dim=-1).clamp_min(0)
    scale = ((group_max - group_min) / 15.0).clamp_min(1e-12)
    zero = torch.round(-group_min / scale).clamp(0, 15).to(torch.uint8)
    quantized = (
        torch.round(grouped / scale.unsqueeze(-1) + zero.float().unsqueeze(-1))
        .clamp(0, 15)
        .to(torch.uint8)
    )
    unsigned = quantized.reshape(n, k)
    packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    dequantized = (
        (quantized.float() - zero.float().unsqueeze(-1)) * scale.unsqueeze(-1)
    ).reshape(n, k)
    return packed.contiguous(), scale.contiguous(), zero.contiguous(), dequantized


def golden_gemm_w4a16(activation, packed_weight, weight_scale, weight_zero,
                      group_size: int = 128):
    """源算子语义（torch 参考）：out = activation @ dequant(packed_weight)ᵀ。

    activation: (M, K) float16；packed_weight: (N, K/2) uint8；
    weight_scale: (N, K/group_size) float32；weight_zero: (N, K/group_size) uint8
    return: (M, N) float16
    """
    n, packed_k = packed_weight.shape
    k = packed_k * 2
    groups = k // group_size
    q = torch.empty(n, k, dtype=torch.uint8, device=packed_weight.device)
    q[:, 0::2] = packed_weight & 0x0F                       # 低 nibble = 偶数 k
    q[:, 1::2] = (packed_weight >> 4) & 0x0F                # 高 nibble = 奇数 k
    s = weight_scale.repeat_interleave(group_size, dim=1)   # (N, K) fp32
    z = weight_zero.repeat_interleave(group_size, dim=1)    # (N, K) uint8
    # 源实现：fp32 域反量化 → 落 activation dtype（源 workload L78 + L114）
    w_deq = (((q.float() - z.float()) * s).reshape(n, k)).to(activation.dtype)
    del weight_scale, weight_zero
    return torch.matmul(activation, w_deq.T)
```

（与 `examples/TileOPs/tests/ops/test_gemm.py` 的 `GemmW4A16Test.ref_program`（L138–L147）等价；输入由 `examples/TileOPs/tileops/workloads/gemm.py` 的 `GemmW4A16Workload.gen_inputs`（L107–L115）生成。）

### 8.2 精度标准

| dtype | atol | rtol | 依据 |
|-------|------|------|------|
| float16（GemmW4A16） | **7e-2** | **5e-2** | `examples/TileOPs/tests/ops/test_gemm.py` L181（`test_gemm_w4a16`） |
| 通用参照 | 1e-2 | 1e-2 | `docs/开发指南.md` §3 示例 |

> 误差来源：① 权重 INT4 量化的固有近似（golden 与 kernel 共享同一 `packed/scale/zero`，故此项对消）；② 反量化域差异（E4，实测 max abs 3.9e-3）；③ fp32 分块累加分组差异；④ 输出 fp32→fp16 单次舍入。均在 atol=7e-2/rtol=5e-2 内（相对通用容差更宽是因为**参考实现本身也做 INT4 量化**，且 `weight_zero` 参与反量化）。

### 8.3 L0 门槛测试计划（交 `tilelang-op-develop` 执行）

| 档位 | shape (m,n,k,dtype) | 目的 |
|------|---------------------|------|
| smoke | 64, 64, 128, float16 | 最小基线（编译通过 + index 等价 + 1 组） |
| smoke | 128, 256, 256, float16 | 矩形 + 多 block（`num_logical=32`） |
| decode m=1 | 1, 512, 512, float16 | `tm=1` 尾块路径 + 多 k-tile（组内 4 tile） |
| decode m=1 | 1, 8192, 16384, float16 | manifest 代表 decode（带宽主项） |
| decode 非 2 幂 | 1, 7168, 20480, float16 | 非 2 幂 N + 大 K 组数（160 组） |
| 组内多 tile | 16, 1024, 1024, float16 | `block_K=64` 下 2 k-tile/组的组内一致性（E3 前提） |
| 长 K 压力 | 1, 8192, 81920, float16 | 最大 K（640 组）+ 分核负载 |

- 每档：`torch.testing.assert_close(out, golden_gemm_w4a16(...), atol=7e-2, rtol=5e-2)`。
- 完整分层套件（L1/L2/Boundary）交 Stage 3 的 `tilelang-op-develop`，此处不枚举。

---

## 9. 风险点与注意事项

### 9.1 已知约束

**9.1.1 工具链 op 级阻断（探针实测，已规避）**

1. **`uint8` 被 lower 成有符号 i8**（CG-2026-0018，本任务为其第 3 个独立数据点）：`T.npuir_cast(uint8_ub → float16, rint)` 对字节 ≥128 得到**负值**（如 172 → −84），IR 为 `tensor<64x64xi8>`。**本设计规避**：cast-only recipe 对符号无关（`256 ≡ 0 mod 16` ⇒ `lo`、`hi` 经 floored 取模后与无符号语义一致），探针 P2 实测 nibble+interleave **bit-exact**（§0.6 R2/§1.6.2）。
2. **位运算族操作数约束**（CG-2026-0019，本任务为第 3 个数据点）：`T.npuir_and`/`T.npuir_shr` 的标量或 `(1,)` 缓冲右操作数被 MLIR verifier 拒绝（`operand at index 1 is vector-only`）；改全形 int16 缓冲后链尾 `int16→uint8` 的 `vcast` 又不支持（`cast int16_t_to_int8_t_truncmode`），且 `int32→half rintmode` 亦不支持。**本设计规避**：整条解包链路**不使用位运算**。
3. **fp16 `npuir_floordiv` 运行期 Vector 核超时**（CG-2026-0025，本任务首证）：`T.npuir_floordiv(f16_ub, 16.0, out)` 编译通过但运行期 `Vector core execution timed out`（507034）。**本设计规避**：用「乘 0.0625 + `f16→i16 floor` cast + `i16→f16 rint` cast」替代常量整除（探针 P2 实测）。
   → **去绕化条件**：以上三条绑定工具链 `tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b` / CANN 8.5.0；工具链升级后按 `AGENTS.md` 版本戳规则重验，届时**更少 op** 的位运算路径（§1.6.2 对照表）可作为 Stage 4 优化候选。

**9.1.2 性能与结构风险（未实测，交 Stage 4）**

1. **反量化链的 Vector 开销未标定（首要实测项）**：解包链 ≈11 op/字节 + 反量化 4 op/元素 —— 其墙钟占用**未实测**（§1.6.0 D-2 的「发射项」）。`CONST-vector-launch-overhead`（0.5µs/op）是依赖串行链的延迟模型，不适用于流水化块链（反证：`PL-1.6-copy-floor` 实测 7-pass 链在 16M 元素上仅 +1.1µs）。**风险**：若反量化无法被 MTE2 窗口吸收，可能成为主导项（decode 形态下 W 元素数 = N·K 与 MTE2 流量同阶，反量化却是「每字节 ~11 op」）。**步骤**：Stage 4 用 msprof 量 `vec_ratio`/`vec_time`/`mte2_active_bw`；若 Vector 占比 >50%，按 `CONST-vector-launch-overhead` 的「宽块优先」原则增大 `deq` 块宽，或启用位运算去绕化路径（9.1.1 末条）。
2. **UB 余量偏紧（≈12–14%）**：§4.5 保守口径膨胀后 ≈165–168 KiB / 192 KiB，且 `T.Pipelined` 是否对全部链缓冲做 auto-multi-buffer 属未文档化。**缓解**：缓冲复用（可回收 ≈24 KiB）、`deq` 分块（`block_N=208, deq_N=16` 时降到 ≈94–96 KiB，§5.5）、必要时降低对反量化缓冲的多缓冲深度。
3. **CV 交接（Vector 写 `alloc_shared` → Cube `T.gemm`）在本形态未验证**：先例 `test_mix_kernel_bf16.py` 形态相同但规模与 dtype 不同（bf16、单次 vadd）。**风险**：`W_sh` 的 UB/L1 归属判定、跨引擎交接是否引入额外 GM 往返。**步骤**：Stage 3 首验证 smoke 档；若失败 → 切**备选 L2'**（显式双 Scope + `alloc_ub`/`alloc_l1` + GM workspace + `sync_block_set/wait`，§1.6.3 ②）。
4. **双 AIV 重复执行 Vector 程序**（`PL-1.13-aiv-dup-subid-split`）：Mix kernel 的双 AIV 默认重复执行同一 Vector 程序 → 反量化有效工作量 ×2。**步骤**：Stage 4 检查两 AIV 子块指标是否镜像（判据见该条目），必要时按 `subid` 分片。
5. **`block_M=16` 的 Cube 填充浪费**：m=1 时 15/16 的 M 行是填充；`block_M=64` 会把 m=1 decode 变为算力受限（§1.6.0 D-2）。保持 16 以让流量项主导；若 Stage 4 实测 Cube 吞吐远低预期，重新评估。**改走 Vector 规约路径已在 §1.6.0 候选 9 评估并淘汰**（无访存收益 + MAC 段落 Vector 吞吐劣势 + 仅覆盖 m=1，R3/R4 量化依据见 §1.6.0），故不作为备选。
6. **Developer + persistent + gemm 结构边界**（`CG-2026-0010` / `PL-1.16`）：主选已规避（非 persistent）；Expert 备选须按 `PL-1.16-expert-dualscope-bypass` 的显式双 Scope + pass_configs 边界实现。
7. **`group_size` 仅 128**：源 `GemmW4A16Op.__init__`（L368–L371）硬限制；设计把 128 当作编译期常量（R6）。若未来支持其它 group_size，需重新评估 `block_K | group_size` 与 UB 预算。
8. **`decode-non-power2-low-cta`（N=7168）的分核对齐**：`ceil(7168/block_N)` 难以同时满足「UB 预算」与「20 的整数倍」（`block_N=64`→112 块 = 5.6×20；`block_N=176`→41 块 ≈2×20 但需 `deq` 分块）。由 host 启发式按 shape 选择（§5.5），Stage 4 实测收敛。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| L0C 溢出 | `block_M·block_N·4 > 128KB` | segfault / 编译失败 | 保持 `block_M·block_N ≤ 32768`（本设计 16×64 = 1024） |
| UB 溢出 | 反量化链 × `num_stages` 膨胀 > 192KB | 编译失败（`ub overflow…`） | 缓冲复用 / `deq` 分块 / 降 stage 数（`CONST-capacity-910B2C`） |
| L1 溢出 | `(bm·bk + bn·bk)·2·num_stages > 512KB` | 编译失败 | 减小 `block_K`/`block_N`/`num_stages` |
| **uint8 有符号误算** | 直接对 `uint8→fp16` 的 cast 结果做减法/比较 | 字节 ≥128 的元素全错 | 用符号无关 recipe（§0.6 R2），或对 `lo/hi` 做 floored 取模还原 |
| **位运算 verifier 失败** | `T.vand/vshr` 带标量或 `(1,)` 操作数 | TVMError / MLIR verify fail | 本设计不使用位运算；若须用，右操作数须与左同形的全形缓冲 |
| **`npuir_floordiv` 超时** | fp16 向量整除 | Vector core timeout，进程 abort | 用「乘倒数 + `f16→i16 floor` cast」替代（CG-2026-0025） |
| 非整除 M/N 越界写 | `tm<block_M` 时未守卫写回 | 写坏相邻 GM | 动态 extent 写回 `C_local[0:tm,0:tn]` |
| fp16 累加卡死 | `C_local` 用 fp16 | 运行时卡死 | 强制 fp32 累加（`T.gemm.md` §2.3） |
| 隐式列广播性能税 | 用 `vsub/vmul` 的列广播代替 `vbrc` | +12~16% 性能（`PL-1.15`） | 显式 `T.vbrc` 物化广播向量 |
| `threads=` 无效 | 沿用源 `T.Kernel(..., threads=128)` | 静默无效（`TRAP-threads-kwarg-noop`） | 移除 |
| `T.Pipelined` 零迭代 | `block_K > K` 时 `ceildiv(K,block_K)=0` | 未定义行为 | `block_K ∈ {32,64,128}` 均 ≤ 最小 K=128 | 

### 9.3 特殊场景处理

- **`m = 1`（4/6 个 manifest workload）**：`block_M=16` 保证 Cube 分形合法；`tm=1` 走动态 extent 尾块路径；A tile 仅 1 行有效（其余行不写回，安全）。不放宽 `block_M`（放宽会把算力项翻为主导，§1.6.0 D-2）。
- **`n` 极大（8192）与 `k` 极大（81920）**：逻辑核数 128（`block_N=64`）；按 §5.5 以 `block_N=208` + `deq_N=16`（`deq_N | block_N`，13 段）对齐到 40 = 2×20（UB 峰值 ≈94–96 KiB）。K=81920 下 `k_iters = 1280`，流水充分。
- **`K` 非 `block_K` 整数倍**：本 manifest 不发生（`gcd(K,128)=128`）；若发生按 E2 对 A/W 的 K 尾清零（A、W **都**清，杜绝 `0·NaN`）。
- **`n` 非 `block_N` 整数倍**：本 manifest N 均被 64 整除；仍实现动态 extent（`tn`）以保证健壮性。
- **`weight_scale` 极小值**：源 `quantize_weight_int4` 用 `clamp_min(1e-12)`，`s>0`；fp16 下 `1e-12` 下溢为 0 → 对应权重全 0（golden 同路径，二者一致）。
- **bf16**：本算子 activation 仅 fp16（§0.1），无 bf16 路径。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/gemm_w4a16/_gemm_w4a16_kernel/
├── DESIGN.md                                    # 本设计文档
├── verify_equiv.py                              # D-1 等价性机器验证脚本（§1.6.1）
├── history_version/
│   ├── design_probe_p1_int4_unpack.py           # D-3 探针 1：nibble 提取 op 可行性二分
│   └── design_probe_p2_dequant_recipe.py        # D-3 探针 2：cast-only recipe 正确性（PASS）
└── _gemm_w4a16_kernel.py                        # 算子实现（Stage 3 产出）
```

> 项目名 `gemm_w4a16`（由 conductor 传入）；迁移 harness 模式按函数逐目录落盘，本任务实际落在 `examples/gemm_w4a16/_gemm_w4a16_kernel/`。

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 设计文档（本文件） |
| `verify_equiv.py` | 已完成 | §1.6.1 E1–E4 等价性验证（13/13 `EQUIV_PASS`） |
| `history_version/design_probe_p1_int4_unpack.py` | 已完成 | D-3 探针 1（op 可行性；结论见 §9.1.1） |
| `history_version/design_probe_p2_dequant_recipe.py` | 已完成 | D-3 探针 2（recipe bit-exact，`RESULT: PASS`） |
| `RETROSPECTIVE.md` | 已完成 | Stage 1 复盘（自进化钩子） |
| `_gemm_w4a16_kernel.py` | 待实现（Stage 3） | NPU Developer MixCV kernel + golden |

### 10.3 命名规范

- 项目目录：`gemm_w4a16`；算子/函数目录：`_gemm_w4a16_kernel`；实现文件：`_gemm_w4a16_kernel.py`。

### 10.4 实现顺序

1. ✅ 设计文档（`DESIGN.md`）+ 等价性验证（`verify_equiv.py`）+ 2 个设计期探针
2. ⬜ Golden 函数（`golden_gemm_w4a16` + `quantize_weight_int4`，以 §0.1 语义）
3. ⬜ 算子实现（`_gemm_w4a16_kernel.py`）+ §8.3 L0 门槛精度比对（**首验证 smoke 档的 CV 交接**，§9.1.2-3）
4. ⬜ 上层 wrapper 替换 GPU 工厂调用为 NPU kernel（`examples/TileOPs/tileops/kernels/gemm/gemm_w4a16/`）
5. ⬜ Stage 4：实测反量化 Vector 占比 / UB 占用 / CV 交接代价，按 §1.6.3 裁决计划对账主选 L1 vs 备选 L2'
