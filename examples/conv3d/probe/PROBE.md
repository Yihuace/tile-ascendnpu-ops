# Conv3d → npuir 可行性探针报告

- **任务**：迁移前可行性探针（pre-migration FEASIBILITY PROBE），**非**迁移、**非** conductor harness。
- **目标算子**：`Conv3dFwdOp` / bias 变体（`has_bias=True` 的 `Conv3dFwdOp`；基线仓中无独立 `Conv3dBiasFwdOp` 类）。
- **目标平台**：Ascend 910B3（NPU 5，driver 25.5.0），CANN 8.5.2（`/home/developer/Ascend/cann-8.5.2`），`tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b`，`target="npuir"`，Expert 模式。
- **GPU 基线（只读 DATA）**：`/tmp/opencode/base/TileOPs-1238a9f45862dc71e3c38cf79a2873fce9763331/`
  - kernel：`src/tileops/kernels/convolution/conv3d.py`
    （`_conv3d_kernel` L29、`_conv3d_group_kernel` L178、`_conv3d_ndhwc_kernel` L351；
    host `Conv3dKernel` L607、`GroupConv3dKernel` L766、`Conv3dNdhwcKernel` L956）
  - op：`src/tileops/ops/convolution.py`（`Conv3dFwdOp` L263，`forward` L301）
  - manifest：`src/tileops/manifest/spec/convolution.yaml:128`（`Conv3dFwdOp` workloads）
- **最近先例（已迁移并提交）**：Conv1d（`examples/conv1d/`，集成于
  `examples/TileOPs/tileops/kernels/convolution/conv1d/`；探针
  `examples/conv1d/probe/conv1d_geomm_probe.py`）。
- **纪律**：源仓只读；未写任何迁移状态文件（无 `.migration_state.json` / 时间线 / capability-gaps 写入）。

## 0. 结论速览

| 项 | 结论 |
|---|---|
| **总判定** | **FEASIBLE（功能可行）** — 3-D implicit-GEMM 与 MixCV bias 路径均已在真实设备编译并运行通过 |
| 核心机制 | per-tap slice 形态 `T.copy` 承载 3-D 跨步/重叠 im2col → `T.gemm` fp32 累加；in-kernel `T.transpose` 权重重排；Cube→Vector 偏置交接 |
| no-bias `max_abs_err` | **3.81e-05**（纯 Cube，fp32 输出，阈值 5e-2） |
| bias `max_abs_err` | **2.97e-02**（MixCV，fp16 输出，即 fp16 舍入量级，相对误差 1.1e-03） |
| 硬能力 | 5-D GM 张量、3-D 地址算术、slice 形态 GM→L1、`T.gemm(initC=)`、L0C→GM 与 L0C→ws、`T.transpose` 两次轴交换、跨 Scope flag 握手、`vcast/vbrc/vadd` 偏置 |
| 未覆盖（需后续） | `stride>1`（video-down）、`dilation>1`（unet-aspp）、`groups>1`（resnext-grouped）；NDHWC 专用变体的直接移植 |
| 建议下一步 | 以 dense NCDHW 路径（+MixCV bias）作为迁移主实现；stride/dilation 用 Conv1d 的 polyphase/`T.gather` 前导扩展；NDHWC 变体作为独立性能项评估 |

## 1. 源算子与算法

### 1.1 数学语义（`conv3d.py` L381-397 / manifest）

```
out[n, oc, od, oh, ow] = bias[oc]
    + Σ_{ci,kd,kh,kw} x[n, ci, id, ih, iw] · weight[oc, ci, kd, kh, kw]
id = od·sD + kd·dD − pD   (ih, iw 同理)
```

这是 **implicit-GEMM** 形式：把 `(kd,kh,kw,ci)` 展平成 GEMM 的 K 轴
（`k_total = kd·kh·kw·c_in`），把输出空间位置 `(od,oh,ow)` 展平成 N/M 轴。

### 1.2 三个 kernel 变体与职责

| 变体（源） | 行 | 结构 | 备注 |
|---|---|---|---|
| `_conv3d_kernel` / `Conv3dKernel` | L29 / L607 | 稠密 NCDHW implicit-GEMM；`T.Parallel` 逐元素展开 im2col 到 shared，`T.gemm(weight_shared, data_shared)` | 通用实现（`general=True`），服务所有 ungrouped |
| `_conv3d_group_kernel` / `GroupConv3dKernel` | L178 / L766 | 每组一次 implicit-GEMM | `groups>1` |
| `_conv3d_ndhwc_kernel` / `Conv3dNdhwcKernel` | L351 / L956 | 先把 x/weight 物化为 NDHWC/ODHW(I) 布局，再做通道连续 gather 的 implicit-GEMM | 16-bit 专用优化变体（见 §1.4） |

### 1.3 数据流（GPU 视角，dense 路径 L99-149）

```
GM[x]         --T.copy--> SMEM[data_shared]   (T.Parallel: k_idx→(ci,kd,kh,kw), spatial→(od,oh,ow), 越界补0)
GM[weight]    --T.copy--> SMEM[weight_shared]
             --T.gemm--> register[out_local] fp32
             --epilogue(Parallel: +bias, cast, mask)--> SMEM[out_shared] --T.copy--> GM[out]
```

### 1.4 工作负载 → kernel 变体映射（关键）

`Conv3dFwdOp.forward`（`ops/convolution.py:356`）恒以 `kernel_for("conv3d", call)`
分派；同一接口 `"conv3d"` 下注册三个实现（`ops/convolution.py:269-273`），由
`key_for`（`ops/op_base.py:386-420`）按「`refusal` + `preferred_over` + general 最低优先」
选出**唯一**实现。`Conv3dNdhwcKernel.refusal`（`conv3d.py:993-1011`）的门槛为：
`groups==1 ∧ dtype∈{fp16,bf16} ∧ c_in%32==0 ∧ c_out≥64 ∧ output_spatial≥1024 ∧ kernel_volume>1`。

| workload（manifest:149-163）| shape 关键项 | 服务变体 | 判据 |
|---|---|---|---|
| **r3d-stem** | C_in=3, C_out=64, k3, s1, p1, fp16 | **`Conv3dKernel`**（dense NCDHW） | `c_in%32≠0` → Ndhwc 拒绝 |
| **video-down** | C_in=64, C_out=128, k3, **s2**, p1, fp16 | **`Conv3dNdhwcKernel`** | 门槛全过 → 特化变体胜过通用 |
| **unet-encoder** | C_in=32, C_out=64, k3, s1, p1, **bf16** | **`Conv3dNdhwcKernel`** | 门槛全过 |
| unet-aspp | C_in=256, C_out=256, k3, p6, **d6** | `Conv3dNdhwcKernel` | 门槛全过 |
| resnext-grouped | groups=32 | `GroupConv3dKernel` | 唯一服务者 |
| `*-bias` 变体 | 同上去掉/加上 bias | 同上（`has_bias=True`，MixCV 路径） | — |

> **重要**：三个被点名的工作负载中，**r3d-stem 走 dense `Conv3dKernel`，video-down /
> unet-encoder 走 `Conv3dNdhwcKernel`**（通用实现被特化实现覆盖，因为 general 在当前
> 分派规则里排在最末）。因此 §1.4 的 NDHWC 变体可行性直接决定 video-down / unet-encoder。

## 2. 探针设计（`conv3d_probe.py`）

**代表性工作负载**（对齐 r3d-stem / unet-encoder 的 `k=3, stride=1, dilation=1, padding=1`）：

```
N=1, C_in=16, C_out=32, D=H=4, W=16, kd=kh=kw=3, padding=1, fp16
→ out_d=out_h=4, out_w=16, out_dhw=256, KV=kd·kh·kw=27
x 预零填充为 [1,16,6,6,18]（padding=1 的 OOB 处理前置为 staging）
```

**分块**：`BLOCK_M=C_out=32, BLOCK_N=out_w=16, BLOCK_K=C_in=16`，grid=`out_d·out_h=16`
个 block（每 block 一条输出行），每 block 对 27 个 tap 累加 `T.gemm`；无尾块掩码，隔离 3-D gather。

**im2col 承载**：为何逐 tap 跨步 slice 而非扁平展开——3-D 卷积下「一段连续展平输出位置」
不再映射到「一段平移输入窗口」，故 block 须留在一个输出行内；每个 tap 的输入行
`(id,ih)` 读作 `x_pad[0, :, od+kd, oh+kh, kw : kw+BLOCK_N]`（行 stride `DP·HP`，W 连续）。
**slice 形态 `T.copy`** 是跨步 GM→L1 的已文档化、已验证形态
（`docs/Tilelang.language/内存操作/T.copy.md` §2.4 示例4、§2.2.2、§2.3.6；跨步
`T.load_nd2nz` 静默误读见 CG-2026-0004）。

**两个 kernel**：
- **A. no-bias**：纯 Cube，L0C→GM（fp32）fixpipe，隔离 3-D gather。
- **B. with-bias**：MixCV。Cube 把 fp32 L0C tile 经 GM workspace 交给 Vector，Vector
  `vcast → vbrc → vadd`（fp32 域加 bias）→ `vcast` 回 fp16 → 存回。跨 Scope 握手复刻
  Conv1d 已验证模式（producer Cube `sync_block_set`，consumer Vector `sync_block_wait`）。

**golden**：独立纯 torch CPU `torch.nn.functional.conv3d`（padding=1），fp32 比较；
与 NPU 算法无关。

## 3. 硬能力验证（已在真实设备验证）

| # | 能力 | 证据 |
|---|---|---|
| 1 | 5-D GM 张量 `x_pad[N,C,DP,HP,WP]`、`weight[C_out,C_in,KV]` 入 kernel | 编译运行通过 |
| 2 | 3-D 地址算术：per-tap `(kd,kh,kw)` 索引 → 5-D 切片 | `conv3d_probe.py` tap 循环 |
| 3 | slice 形态 `T.copy` 承载跨步/重叠 im2col（GM→L1 自动 ND2NZ） | 27 tap 全部正确 |
| 4 | `T.gemm(initC=(tap==0), size=[M,K,N])` 分 tap 累加 | fp32 结果 max_abs 3.8e-05 |
| 5 | in-kernel 权重重排 `[C_out,C_in,KV]→[KV,C_out,C_in]`：两次 `T.transpose` 轴交换 | Vector prelude 通过 |
| 6 | L0C→GM fixpipe（fp32）| A 输出正确 |
| 7 | 跨 Scope flag 握手（Vector→Cube 权重就绪；Cube→Vector ws 就绪） | A/B 均无死锁 |
| 8 | MixCV 偏置：L0C→GM ws→UB，`vcast/vbrc/vadd`，L0C 以外 dtype cast | B 输出正确（fp16） |
| 9 | L0C→GM fp32→fp16 量化 cast | B 输出 dtype=fp16 |

（`T.transpose` 仅支持交换两个轴——`docs/Tilelang.language/创建操作/T.transpose.md:36`；
本探针的权重重排恰由两次两轴交换组成，符合该限制。）

## 4. 结果（`python conv3d_probe.py`，exit=0）

```
tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b target=npuir mode=Expert
building no-bias kernel ...
  compiled OK in 2.6s
  ran OK in 1.3 ms; out shape=(32, 256) dtype=torch.float32
[A-no-bias] max_abs_err = 3.814697e-05
[A-no-bias] mean_abs_err= 3.571643e-06
[A-no-bias] max_rel_err = 7.161589e-04
[A-no-bias] golden absmax = 75.4773
building bias kernel ...
  compiled OK in 2.2s
  ran OK in 0.7 ms; out shape=(32, 256) dtype=torch.float16
[B-bias] max_abs_err = 2.970123e-02
[B-bias] mean_abs_err= 2.355868e-03
[B-bias] max_rel_err = 1.107027e-03
[B-bias] golden absmax = 75.2624
[PROBE] no-bias max_abs_err=3.815e-05 bias max_abs_err=2.970e-02 -> PASS
```

- no-bias：fp32 累加误差 3.8e-05（量级与 Conv1d 先例 5.7e-05 一致）→ 3-D gather 正确。
- bias：2.97e-02 完全由 fp16 输出舍入贡献（相对误差 1.1e-03 ≈ 2⁻¹⁰），非算法误差。
- 运行退出码 0。

## 5. 覆盖范围与缺口

### 5.1 探针已覆盖
`k=3, stride=1, dilation=1, padding=1`、ungrouped、fp16、dense NCDHW implicit-GEMM、
bias 与 no-bias。

### 5.2 未覆盖（需迁移阶段补齐，均有先例/文档支撑，判定为可扩展）
- **`stride>1`（video-down）**：W 轴的 `iw=ow·sW+kw` 不是整数步进切片。Conv1d 先例用
  polyphase 抽取（`_conv1d_kernel.py` R4：`xdec = T.gather(x_pad, host-index)`，见
  `docs/Tilelang.language/索引与元素操作/T.gather.md`）先把跨步采样物化为连续切片；
  Conv3d 需按轴推广。D/H 轴 stride 只是标量索引，免费。**未实测。**
- **`dilation>1`（unet-aspp）**：`id=od·sD+kd·dD` 同样是标量索引；W 轴需与 stride 一并
  经抽取处理。**未实测。**
- **`groups>1`（resnext-grouped）**：走 `GroupConv3dKernel`，结构同 dense 但每 group 独立
  GEMM，与探针同族。**未实测。**

### 5.3 NDHWC 专用变体（`Conv3dNdhwcKernel`）— 直接移植受阻，但有绕法

`_conv3d_ndhwc_kernel` 的核心（`conv3d.py:470-547`）是**逐元素 `T.Parallel`**：
- im2col 在 shared 内直接读 **GM** 张量：
  `data_shared[i,j] = if_then_else(in_bound, x_ndhwc[m//out_dhw, id_, ih, iw, c], 0)`（L490-518）；
- 输出把 shared 逐元素散射回 **GM**：`out_flat[...] = out_shared[i,j]`（L541-547）；
- `to_channels_last` 宏（L432-468）也是逐元素 `T.Parallel` 跨布局 GM 读写。

**限制条款（负向断言证据）**：
`docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.3
> 「循环体内的变量必须是 alloc 申请出的 UB 上的 Buffer，**不能是直接传入的 GM 上的
> Tensor 变量**」；§2.3 亦规定「`T.Parallel` 内下标不能做变换」（如 `i+1`）。

NDHWC 的 gather/scatter 既**直接读写 GM Tensor**，又使用分解/变换下标
（`m%out_dhw`、`id_`、`c` 等），与上述两条限制冲突，故该 kernel **不能按原样**在
Expert npuir 下表达（该路径还依赖 GPU 的 cp.async 向量化 `T.Parallel`，注释 L417-419）。

**绕法（非阻塞）**：
1. **功能上**该变体与 dense 路径计算同一结果，可直接用 §3 已验证的 dense NCDHW 路径服务
   video-down / unet-encoder（正确性无碍；NDHWC 仅为性能特化）。
2. 若追求该布局的访存收益，可用 Conv1d 已验证的 **UB 前导**重建布局：把 x/weight 经
   `T.copy`(GM→UB) + `T.gather`/`T.transpose`(UB 内) 物化为 NDHWC/ODHW(I)，再由 Cube
   以通道连续 slice 装载——代价是额外 staging。属**性能问题**而非可行性阻断。

## 6. 判定与理由

**FEASIBLE（功能可行，可进入迁移）。**

理由：
1. 3-D implicit-GEMM 的全部**算法硬骨架**（5-D 张量、3-D 地址算术、跨步/重叠 im2col、
   fp32 Cube 累加、in-kernel 权重重排）已在真实 910B3 上编译运行且数值正确
   （§4，max_abs 3.8e-05）。
2. bias 变体的 **MixCV 交接**（跨 Scope flag + `vcast/vbrc/vadd` + cast）同样实测通过
   （§4，误差即 fp16 舍入）。
3. 三个点名负载至少 **r3d-stem 直接由 dense 路径服务**；video-down / unet-encoder 虽被分派到
   NDHWC 变体，但该变体与 dense 路径**同一数学**，dense 路径即可满足功能契约（§5.3）。
4. 未覆盖的 `stride/dilation/groups` 均有 Conv1d 先例或同族结构支撑，判定为**工程扩展**而
   非能力阻断（§5.2）。

## 7. 建议下一步

1. **迁移主实现 = dense NCDHW implicit-GEMM（`Conv3dKernel`）+ MixCV bias**，直接复用本探针与
   Conv1d 结构（slice-form im2col、in-kernel 权重 `T.transpose`、bias 握手）。
2. 扩展探针覆盖 **stride>1 / dilation>1**：按轴引入 Conv1d 的 polyphase `T.gather` 抽取前导。
3. 扩展 **groups>1**：验证 `GroupConv3dKernel` 的 per-group GEMM。
4. **NDHWC 变体**：作为独立性能项评估；先以 dense 路径交付功能，再决定是否投入 UB 布局前导；
   若决定评估，按 AGENTS「Compiler Capability Gap Rule」在迁移阶段登记 §5.3 缺口
   （本探针任务按纪律未写状态文件）。

## 附：文件

- `examples/conv3d/probe/conv3d_probe.py`（探针，运行通过 exit=0）
- `examples/conv3d/probe/PROBE.md`（本报告）
- 运行日志：`/tmp/opencode/conv3d_probe_run.log`
