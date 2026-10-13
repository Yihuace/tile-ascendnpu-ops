# GPU-to-NPU 适配点清单

本文档记录从 TileOPs（GPU/TileLang 后端）到 `TileOPs`（NPU/Ascend 后端）的全部适配点，分为**公共机制**和**算子**两个维度。

**核心前提**：当前环境的 TileLang 同时支持 GPU 和 NPU，因此 TileLang DSL 代码（`@tilelang.jit`、`@T.prim_func`、`T.Kernel`、`T.alloc_shared` 等）在 NPU 上可用，不需要替换为 PyTorch 原生 op。适配集中在 GPU 特定的运行时调用（`torch.cuda.*`、CUDA SM 架构检查、共享内存预算查询）和 GPU 执行模型参数（`threads`、`npt`）。

---

## 一、公共机制适配点

公共机制是指框架级基础设施——所有算子共享的设备相关代码。适配集中在 `tileops/device.py`（单一适配面），其余模块通过它间接获取设备能力。

### 1.1 设备检测与后端抽象

| # | GPU 原始代码 | NPU 适配 | 位置 | 说明 |
|---|---|---|---|---|
| C1 | `torch.cuda.is_available()` | `backend.is_available()` | `device.py` | NPU 通过 `torch_npu` 注册 `torch.npu`；后端抽象自动检测 NPU > CUDA > CPU |
| C2 | `torch.cuda.device_count()` | `backend.device_count()` | `device.py` | NPU 设备数量 |
| C3 | `torch.cuda.current_device()` | `backend.current_device()` | `device.py` | 当前 NPU 设备 |
| C4 | `torch.cuda.get_device_name()` | `backend.get_device_name()` | `device.py` | 返回 `Ascend910B2C` 等 |
| C5 | `torch.cuda.get_device_properties()` | `backend.get_device_properties()` | `device.py` | NPU 属性对象 |
| C6 | `torch.cuda.get_device_capability()` → `(9, 0)` | NPU 无 capability 概念；`get_device_capability()` 返回 `None` | `device.py` | CUDA SM 版本号在 NPU 上不存在 |

**适配方式**：引入 `DeviceBackend` 抽象基类，三个子类 `NPUBackend`/`CUDABackend`/`CPUBackend`。框架代码**从不**直接调用 `torch.cuda.*` 或 `torch.npu.*`，统一走 `get_device_backend()`。

### 1.2 张量设备检查

| # | GPU 原始代码 | NPU 适配 | 位置 | 说明 |
|---|---|---|---|---|
| C7 | `x.is_cuda` | `backend.is_device_tensor(x)` → `x.is_npu` | `ops/reduction/softmax.py:_validate` | Op 层输入校验 |
| C8 | `device="cuda"` 硬编码 | `device=backend.name` → `"npu"` | `workloads/workload_base.py:RandnWorkload.gen_inputs` | 输入张量放置设备 |

### 1.3 同步与缓存管理

| # | GPU 原始代码 | NPU 适配 | 位置 | 说明 |
|---|---|---|---|---|
| C9 | `torch.cuda.synchronize()` | `backend.synchronize()` → `torch.npu.synchronize()` | `device.py`, `benchmark_base.py`, `_primitives.py:tune_by_forward` | 设备同步 |
| C10 | `torch.cuda.empty_cache()` | `backend.empty_cache()` → `torch.npu.empty_cache()` | `device.py`, `benchmark_base.py` | 释放设备缓存 |
| C11 | `torch.cuda.manual_seed_all()` | `backend.manual_seed_all()` → `torch.npu.manual_seed_all()` | `device.py`, conftest | 随机种子 |

### 1.4 基准测试计时

| # | GPU 原始代码 | NPU 适配 | 位置 | 说明 |
|---|---|---|---|---|
| C12 | CUPTI/Kineto 内核级计时（`torch.profiler` + annotation window projection） | 设备 Event 计时（`torch.npu.Event`） | `benchmark_base.py:bench_kernel` | CUPTI 可获得纯内核时间；NPU Event 计时包含 ~50-60μs launch 开销 |
| C13 | L2 缓存刷新：`torch.cuda.get_device_properties(0).L2_cache_size` 大小的 buffer `.zero_()` | NPU 无直接等价物；`backend.cache_flush()` 为 no-op | `device.py:CUDABackend.cache_flush` vs `NPUBackend` | CUDA L2 flush 用于隔离迭代间缓存命中 |
| C14 | CUPTI projection 校验 + CUDA-events 降级 | 移除：NPU 直接使用 Event 计时 | `benchmark_base.py` | 简化计时路径 |
| C15 | `torch.cuda.Event(enable_timing=True)` | `backend.Event(enable_timing=True)` → `torch.npu.Event` | `device.py`, `benchmark_base.py`, `_primitives.py:tune_by_forward` | 计时事件 |
| C16 | tilelang `suppress_stdout_stderr`（dup2 /dev/null） | 移除：无 tilelang profiler 依赖 | `benchmark_base.py` | 原代码用于抑制 TileLang JIT 编译输出 |

### 1.5 环境元数据采集

| # | GPU 原始代码 | NPU 适配 | 位置 | 说明 |
|---|---|---|---|---|
| C17 | `nvidia-smi --query-gpu=driver_version,clocks.current.sm,...` | `npu-smi info -t board -i 0` | `device.py:NPUBackend._query_npu_smi` | 设备信息查询命令不同 |
| C18 | `torch.version.cuda` | `torch_npu.__version__` | `device.py:NPUBackend.env_metadata` | 版本信息来源不同 |

### 1.6 Kernel 基类

| # | GPU 原始代码 | NPU 适配 | 位置 | 说明 |
|---|---|---|---|---|
| C19 | `from tilelang.autotuner import autotune` + `autotune()` 方法 + `_autotune_initial_kwargs` + `_call_autotuned_kernel` + `autotune_supply_prog` + `_AUTOTUNE_PARAM_ALIASES` + `autotune_configs` 属性 + `init_config(config, tune)` 的 `tune` 参数 | **全部移除** — NPU 使用启发式 config 选择，不做 autotune | `kernels/kernel_base.py` | TileLang autotuner 机制在 NPU 上不使用；`init_config` 签名简化为 `init_config(config)` |
| C20 | `supported_archs: Optional[list[int]]` = `[80, 86, 89, 90]`（CUDA SM 版本号） | `supported_archs: Optional[list]` — 类型放宽为 `list`（可容纳 int 或 str）；具体 kernel 类中设为 `None`（全平台支持） | `kernels/kernel_base.py`, `ops/op_base.py:_arch_supported` | 架构标识从 CUDA SM 整数改为通用类型 |
| C21 | `dtype_to_str` → TileLang 内核 dtype 字符串 | **保留不变** — 仍用于 TileLang 内核构建 | `kernels/kernel_base.py` | 用途不变 |

### 1.7 Op 基类

| # | GPU 原始代码 | NPU 适配 | 位置 | 说明 |
|---|---|---|---|---|
| C22 | `from tileops.utils import get_sm_version` — CUDA SM 版本整数 | `from tileops.device import get_device_backend` — 设备后端名 | `ops/op_base.py` | 架构检查入口 |
| C23 | `from .compile_boundary import register_instance` — `torch.compile` 分发边界 | 移除 | `ops/op_base.py` | 原始代码中 `compile_boundary` 为 TileLang JIT 内核的 dynamo 追踪屏障；NPU 版本暂未引入此机制，后续如需 `torch.compile` 可恢复 |
| C24 | `__init_subclass__` manifest 代码gen（`_dtype_codegen`, `_roofline_codegen`） | 移除：具体 Op 直接实现 `eval_roofline` | `ops/op_base.py` | 原 codegen 依赖完整 manifest 机器；NPU 版本保持 manifest 作为 spec 但不做代码生成 |
| C25 | 架构兼容检查：`current_arch not in kernel_type.supported_archs`（整数比较） | `_arch_supported(supported, backend)`（设备名子串匹配，`None` 直通） | `ops/op_base.py` | 适配 NPU 芯片命名 |
| C26 | `Op.autotune()` 方法（遍历所有 kernel 属性调用 `autotune()`） | 移除 | `ops/op_base.py` | NPU 不使用 autotune |

### 1.8 Manifest 加载器

| # | GPU 原始代码 | NPU 适配 | 位置 | 说明 |
|---|---|---|---|---|
| C26 | `tileops.manifest` 包目录下多 YAML 合并 | `tileops.manifest` 包目录下独立 YAML（同一逻辑） | `manifest/__init__.py` | 结构相同，数据源独立 |

### 1.9 Workload 基类

| # | GPU 原始代码 | NPU 适配 | 位置 | 说明 |
|---|---|---|---|---|
| C27 | `torch.randn(*shape, dtype=dtype, device="cuda")` | `torch.randn(*shape, dtype=dtype, device=backend.name)` | `workloads/workload_base.py:RandnWorkload.gen_inputs` | 输入张量设备从硬编码改为后端解析 |

---

## 二、算子适配点（LogSumExpFwdOp）

算子级适配分为三部分：**TileLang kernel 工厂函数**（NPU 实现）、**LogSumExpKernel 类**（GPU 特定部分适配）、**_primitives.py**（GPU 特定部分适配）。

### 2.1 TileLang kernel 工厂函数（NPU 实现）

| # | GPU 原始代码 | NPU 适配 | 位置 | 说明 |
|---|---|---|---|---|
| O1 | `_logsumexp_kernel_single(M, N, dtype)` — `@tilelang.jit` + `@T.prim_func` 构建 CUDA 内核；返回的 callable 接受 `(block_m, threads)` | NPU 实现：`@tilelang.jit(target="npuir")` + `@T.prim_func` 构建 NPUIR 内核；callable 只接受 `block_m`（无 `threads`）；不做 alignment padding，直接用原始 N | `kernels/reduction/logsumexp/logsumexp.py` | NPU 无 `threads` 概念；NPUIR `vsel` 在 mask 与 tensor 形状不一致时报错，故不做 padding |
| O2 | `_logsumexp_kernel_tiled(M, N, dtype, tile_n)` — 多 tile 路径；返回的 callable 接受 `(block_m, threads)` | NPU 实现：`@tilelang.jit(target="npuir")`；callable 只接受 `block_m`（`tile_n` 已在闭包固化）；不做 alignment padding | 同上 | 同上 |
| O3 | `_logsumexp_kernel(M, N, dtype, tile_n=0)` — 分发函数 | NPU 实现：分发到 single 或 tiled | 同上 | 同上 |

### 2.2 LogSumExpKernel 类 — GPU 特定部分适配

| # | GPU 原始代码 | NPU 适配 | 位置 | 说明 |
|---|---|---|---|---|
| O4 | `supported_archs: list[int] = [80, 86, 89, 90]` | `supported_archs: Optional[list] = None` | `kernels/reduction/logsumexp/logsumexp.py` | CUDA SM 版本号 → `None`（全平台支持） |
| O5 | `device_smem_budget(device_index)` 内部调用 `torch.cuda.get_device_properties()` | `device_smem_budget` 适配为通过 `get_device_backend()` 查询设备属性 | `kernels/reduction/_primitives.py:device_smem_budget` | NPU 设备属性可能不暴露 `shared_memory_per_block_optin`，回退到默认 48 KiB |
| O6 | `import tilelang; import tilelang.language as T` | **保留不变** — TileLang 在 NPU 上可用 | `kernels/reduction/logsumexp/logsumexp.py` | 无需适配 |
| O7 | `@torch.library.custom_op("top::logsumexp_fwd", ...)` | 命名空间从 `"top::"` 改为 `"npub::"` | `kernels/reduction/logsumexp/logsumexp.py` | 避免与原始 TileOPs 冲突；custom_op 机制本身保留 |
| O8 | `@_logsumexp_fwd_wrapped.register_fake` | **保留不变** | 同上 | fake tensor 实现不依赖 GPU |
| O9 | custom_op wrapper 和 `register_fake` 签名包含 `threads: int` 参数 | 移除 `threads` 参数 | `kernels/reduction/logsumexp/logsumexp.py` | NPU 无线程概念；custom_op 签名从 `(M, N, dtype_str, block_m, threads, tile_n, x)` 改为 `(M, N, dtype_str, block_m, tile_n, x)` |
| O10 | `default_config` 返回 `{"block_m", "threads": 256, "tile_n"}` | 返回 `{"block_m", "tile_n"}`（移除 `"threads"` 键） | 同上 | NPU 无 threads 默认值 |
| O11 | `__init__` 签名包含 `tune: bool = False`；`init_config(config, tune)` | 移除 `tune` 参数；`init_config(config)` | 同上 | NPU 不使用 autotune |
| O12 | `autotune_configs` 属性 + `_tile_n_candidates` 方法 + `_MAX_TILE_N_CANDIDATES` 常量 | **全部移除** | 同上 | autotune 候选生成逻辑不再需要 |
| O13 | `autotune` 方法（跨 tile_n 分组计时，调用 `tilelang.autotuner.autotune`） | **全部移除** | 同上 | NPU 不使用 TileLang autotuner |
| O14 | `forward` 调用 `_logsumexp_fwd_wrapped` 时传入 `self.config["threads"]` | 移除 `self.config["threads"]` 参数 | 同上 | forward 分发不含 threads |

### 2.3 _primitives.py — GPU 特定部分适配

| # | GPU 原始代码 | NPU 适配 | 位置 | 说明 |
|---|---|---|---|---|
| O15 | `device_smem_budget` 中 `torch.cuda.is_available()` / `torch.cuda.current_device()` / `torch.cuda.get_device_properties()` | 改为 `get_device_backend().is_available()` / `.current_device()` / `.get_device_properties()` | `kernels/reduction/_primitives.py` | 设备属性查询从 CUDA 专用改为后端通用 |
| O16 | `tune_by_forward` 函数（`torch.cuda.synchronize()` / `torch.cuda.Event` 计时选择最优 config） | **移除** | 同上 | autotune 不再使用 |
| O17 | `DEFAULT_ALIGNMENT = 256` | **保留不变** — TileLang T.copy 对齐要求 | 同上 | 无需适配 |
| O18 | `MAX_SINGLE_TILE_COLS = 32512` | **保留不变** — TileLang 向量化器限制 | 同上 | 无需适配 |
| O19 | `align_up`, `compute_tile_n` | **保留不变** — 纯数学计算，设备无关 | 同上 | 无需适配 |
| O20 | T.macro 工厂（`make_reduce_epilogue`, `make_welford_update`, `make_softmax_epilogue`, `make_cumulative_scan`） | **保留不变** — TileLang DSL 代码，NPU 上可用 | 同上 | 无需适配 |

### 2.4 LogSumExpKernel 类 — 保留不变的主体流程

以下部分从 GPU 版本**完整保留**（`threads`/`autotune` 相关部分除外）：

| 组件 | 说明 |
|---|---|
| `__init__` 构造流程 | op_kind 校验 → 属性设置 → smem 预算查询 → tile_n 预计算 → kernel 构建 → config 初始化 → tile_n 修正 |
| `_tile_n_for_block_m` | 共享内存分块启发式（基于 smem 预算和列上限） |
| `default_config` | block_m 选择逻辑（遍历 [1,2,4,8,16] 选最优 tile_n）；返回值不含 `threads` |
| `forward` | 通过 `_logsumexp_fwd_wrapped` custom_op 分发（不传 threads） |
| `_compute_padded_cols` | padding 列数计算 |
| `_elem_bytes` | dtype 字节数查询 |

### 2.5 Op 层

| # | GPU 原始代码 | NPU 适配 | 位置 | 说明 |
|---|---|---|---|---|
| O21 | `x.is_cuda` 设备检查 | `backend.is_device_tensor(x)` → `x.is_npu` | `ops/reduction/softmax.py:_validate` | 设备检查从 CUDA 专用改为后端通用 |
| O22 | `eval_roofline()` 公式 `4 * M * N` | 完全相同 | `ops/reduction/softmax.py:eval_roofline` | Roofline 模型是设备无关的 |
| O23 | `dim` 规范化、`flatten_for_multidim`、`restore_multidim_shape` | 完全相同（纯 shape 操作） | `ops/_multidim.py`, `ops/reduction/softmax.py` | 多维归约逻辑设备无关 |
| O24 | Op `__init__` 签名包含 `tune: bool = False`；`_get_or_create_kernel` 传 `tune=self._tune` | 移除 `tune` 参数；kernel 构造不传 `tune` | `ops/reduction/softmax.py` | NPU 不使用 autotune |

### 2.6 源码结构对比

| GPU 文件 | NPU 文件 | 说明 |
|---|---|---|
| `tileops/kernels/kernel_base.py` | `tileops/kernels/kernel_base.py` | TileLang autotuner **移除**；`supported_archs` 类型放宽；`init_config` 简化 |
| `tileops/kernels/reduction/_primitives.py` | `tileops/kernels/reduction/_primitives.py` | T.macro 工厂保留；`device_smem_budget` 适配为后端通用；`tune_by_forward` **移除** |
| `tileops/kernels/reduction/logsumexp.py` (557 行) | `tileops/kernels/reduction/logsumexp/logsumexp.py` (~420 行) | NPU 工厂实现；`autotune`/`autotune_configs`/`_tile_n_candidates` **移除**；`threads` 全面移除；`supported_archs`、custom_op 命名空间适配 |
| `tileops/ops/compile_boundary.py` | **移除** | torch.compile 分发边界暂未引入 |
| `tileops/ops/_dtype_codegen.py` | **移除** | manifest 代码生成未引入 |
| `tileops/ops/_roofline_codegen.py` | **移除** | 同上 |

---

## 三、算子适配点（LerpTensorFwdOp / elementwise family）

LerpTensorFwdOp 是 elementwise / 多输入算子家族的代表（3 输入张量、广播、1-D flat kernel 分发）。与 LogSumExpFwdOp 的归约流不同，它的 kernel 结构是 flat elementwise map，因此适配点集中在 **`threads` + `npt` → `block_size`** 这一 GPU-to-NPU 参数合并上。

### 3.1 核心概念：`threads * npt` → `block_size`

GPU（CUDA）的 elementwise kernel 将每个 block 处理的元素数分解为两个参数：

| GPU 参数 | 含义 | 作用 |
|---|---|---|
| `threads` | 每个线程块的 CUDA 线程数 | `T.Kernel(grid, threads=threads)` 的线程维度 |
| `npt` (num_per_thread) | 每个线程处理的元素数（向量化宽度） | `T.Parallel(threads, npt)` 的第二维度 |

两者的乘积 `threads * npt` 才是真正影响 kernel 行为的值——每个 block 处理的元素数（决定 UB/fragment 分配大小）。NPU 没有 CUDA 线程概念，不存在 "线程数 × 每线程元素数" 的分解，因此两个参数**合并为单一的 `block_size`**：

| NPU 参数 | 含义 | 对应 GPU 值 |
|---|---|---|
| `block_size` | 每个 block 处理的元素数 | `threads * npt`（乘积保持不变） |

**与 LogSumExp 的 `threads` 移除的区别**：LogSumExp 的有意义参数是 `block_m`（行块大小）和 `tile_n`（列 tile 大小），`threads` 纯粹是 CUDA 执行细节，直接移除即可。Elementwise 的 `threads * npt` 乘积本身是有意义参数（per-block 元素数），因此不是移除而是**合并**为 `block_size`。

### 3.2 TileLang kernel 工厂函数

| # | GPU 原始代码 | NPU 适配 | 位置 | 说明 |
|---|---|---|---|---|
| E1 | `_make_lerp_tensor_kernel(N, dtype, output_dtype=None, is_fp8=False, threads=256, npt=8)` — 工厂签名包含 `threads`、`npt`、`output_dtype`、`is_fp8` 四个 GPU/fp8 特有参数 | `_make_lerp_tensor_kernel(N, dtype)` — 四个参数全部移除；`threads` 和 `npt` 合并为 `block_size`（传入 callable 而非工厂） | `kernels/elementwise/lerp_tensor.py` | 工厂 lru_cache 键从 `(N, dtype, output_dtype, is_fp8, threads, npt)` 简化为 `(N, dtype)` |
| E2 | 返回的 callable 签名 `_func(threads_arg, npt_arg)` — 接收两个 GPU 执行参数 | `_func(block_size)` — 单参数，`threads * npt` 合并为 `block_size` | 同上 | callable 在编译期接收 `block_size`，用于 `T.Kernel(T.ceildiv(N, block_size), is_npu=True)` 的 grid 计算 |
| E3 | `T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx` — grid 含 `threads` 维度 | `T.Kernel(T.ceildiv(N, block_size), is_npu=True) as bx` — 无 `threads` 维度 | 同上 | NPU `is_npu=True` 内核无 CUDA 线程维度 |
| E4 | `T.Parallel(threads_arg, npt_arg)` — 二维并行循环 | `T.Parallel(block_size)` — 一维并行循环 | 同上 | 线程×元素的二维分解合并为一维 |

### 3.3 LerpTensorFwdKernel 类 — GPU 特定部分适配

| # | GPU 原始代码 | NPU 适配 | 位置 | 说明 |
|---|---|---|---|---|
| E5 | `supported_archs: list[int] = [80, 86, 89, 90]` | `supported_archs: Optional[list] = None` | `kernels/elementwise/lerp_tensor.py` | CUDA SM 版本号 → `None`（全平台支持） |
| E6 | 继承 `ParametricUnaryKernel(Kernel)` — GPU 公共基类含 `threads`/`npt` config 逻辑、`autotune`、`_builder_fn`/`_builder_args` 模式 | 直接继承 `Kernel`，`ParametricUnaryKernel` 逻辑内联（NPU 项目无此基类） | 同上 | NPU 项目未移植 `ParametricUnaryKernel`；config 选择、factory 调用、`forward` 分发直接在 `LerpTensorFwdKernel` 中实现 |
| E7 | GPU config `{"threads": 256, "num_per_thread": npt}` | NPU config `{"block_size": threads * npt}` | 同上 | `threads * npt` 合并为 `block_size`；默认值：fp32 → 1024 (256×4)，fp16/bf16 → 2048 (256×8) |
| E8 | `default_config` 返回 `{"threads": self._DEFAULT_THREADS, "num_per_thread": npt}` | 返回 `{"block_size": block_size}` | 同上 | 两个常量 `_NPT_FP32`/`_NPT_NON_FP32` 改为 `_BLOCK_SIZE_FP32`/`_BLOCK_SIZE_NON_FP32`，值为原 `threads * npt` 乘积 |
| E9 | `autotune_configs` 属性 + `autotune()` 方法 + `tune` 参数 | **全部移除** | 同上 | NPU 不使用 autotune |
| E10 | 无 `@torch.library.custom_op` wrapper — GPU 直接调用 `self._compiled_fn(a, b, w)` | **保留不变** — NPU 同样直接调用 | 同上 | LerpTensorFwdKernel 在 GPU 侧也没有 custom_op wrapper（与 LogSumExpKernel 不同） |

### 3.4 Op 层

| # | GPU 原始代码 | NPU 适配 | 位置 | 说明 |
|---|---|---|---|---|
| E11 | `input.is_cuda and end.is_cuda and weight.is_cuda` 设备检查 | `backend.is_device_tensor(input) and ...` | `ops/elementwise/lerp_tensor.py:forward` | 三输入设备检查从 CUDA 专用改为后端通用 |
| E12 | `_OP_REGISTRY` + `_wrapped` custom_op wrapper + `compile_boundary.register_instance` | **全部移除** | 同上 | NPU 暂未引入 `torch.compile` 分发边界；`forward` 直接调用 `_eager_forward` |
| E13 | `__init__` 签名含 `tune: bool = False`；kernel 构造传 `tune=tune` | 移除 `tune` 参数；kernel 构造不传 `tune` | 同上 | NPU 不使用 autotune |
| E14 | `eval_roofline()` 公式 `3 * N_total` flops、`4 * N_total * elem_bytes` bytes | **完全相同** | 同上 | Roofline 模型设备无关 |
| E15 | 3-way broadcast → flatten → kernel dispatch → reshape 流程 | **完全相同** | 同上 | 纯 shape 操作，设备无关 |

### 3.5 源码结构对比

| GPU 文件 | NPU 文件 | 说明 |
|---|---|---|
| `tileops/kernels/elementwise.py` (`_make_lerp_tensor_kernel` + `LerpTensorFwdKernel` in `ParametricUnaryKernel`) | `tileops/kernels/elementwise/lerp_tensor.py` | 工厂 stub（`block_size` 替代 `threads`/`npt`）；`ParametricUnaryKernel` 逻辑内联；`autotune` **移除** |
| `tileops/ops/elementwise/arithmetic.py` (`LerpTensorFwdOp`) | `tileops/ops/elementwise/lerp_tensor.py` | 设备检查改为后端通用；`_OP_REGISTRY`/`_wrapped`/`tune` **移除**；broadcast/flatten/roofline 保留 |
| `workloads/elementwise.py` (`LerpTensorManifestWorkload`) | `tileops/workloads/elementwise.py` | `device="cuda"` → `device=backend.name` |

---

## 四、适配总结

### 无需适配（设备无关）

- TileLang DSL 代码（`@tilelang.jit`、`@T.prim_func`、`T.Kernel`、`T.alloc_shared`、`T.reduce_*`、`T.Parallel` 等）
- Manifest YAML 格式与加载逻辑
- Workload 参数化（`FixtureBase`, `FixtureMeta`）
- Op 的 `dim` 规范化、多维 flatten/restore
- `_multidim.py` 全部函数
- `eval_roofline()` 公式
- `TestBase.check()` 正确性验证逻辑
- `BenchmarkReport` 报告生成逻辑
- `workloads_to_params` / `ManifestBenchmark` 参数化逻辑
- 容差（tolerance）定义
- LogSumExpKernel 类主体流程（config 选择、tiling 启发式、forward 分发）
- LerpTensorFwdKernel 类主体流程（config 选择、forward 分发）
- `_primitives.py` 中的 T.macro 工厂、`align_up`、`compute_tile_n`、常量定义
- LerpTensorFwdOp 的 3-way broadcast → flatten → dispatch → reshape 流程

### 需要适配（设备相关）

| 层次 | 适配点数量 | 核心变化 |
|---|---|---|
| **公共机制** | 28 个 (C1-C28) | 引入 `DeviceBackend` 抽象；CUPTI → Event 计时；L2 flush → no-op；arch 检查从 SM 整数改为通用类型；**autotune 全面移除**（kernel_base + op_base） |
| **算子 (LogSumExp)** | 24 个 (O1-O24) | NPU 工厂实现（3 个）；`supported_archs` 改为 `None`；`device_smem_budget` 适配为后端通用；custom_op 命名空间改为 `npub::`；**`threads` 参数全面移除**；**`autotune` 全面移除**（autotune_configs / _tile_n_candidates / autotune 方法 / tune 参数 / tune_by_forward）；Op 层设备检查改为后端通用 |
| **算子 (LerpTensor)** | 15 个 (E1-E15) | NPU 工厂 stub（`block_size` 替代 `threads`/`npt`）；`ParametricUnaryKernel` 逻辑内联；`supported_archs` 改为 `None`；**`threads` + `npt` 合并为 `block_size`**；**`autotune` 全面移除**；`_OP_REGISTRY`/`_wrapped`/`compile_boundary` 移除；Op 层设备检查改为后端通用 |

### `threads` 参数移除说明

`threads` 是 GPU CUDA 执行模型的参数（每个线程块的线程数），NPU 上没有对应概念。根据算子家族不同，移除方式有两种：

**归约家族（LogSumExp 等）— 直接移除**：`threads` 纯粹是 CUDA 执行细节，kernel 的有意义参数是 `block_m`（行块大小）和 `tile_n`（列 tile 大小），`threads` 不影响 kernel 逻辑，直接移除。

1. **custom_op wrapper 签名**：`(M, N, dtype_str, block_m, threads, tile_n, x)` → `(M, N, dtype_str, block_m, tile_n, x)`
2. **`default_config`**：`{"block_m", "threads": 256, "tile_n"}` → `{"block_m", "tile_n"}`
3. **`forward`**：调用 custom_op 时不再传 `self.config["threads"]`

工厂函数返回的 callable 签名也相应改变：GPU 版 `_func(block_m, threads)` → NPU 版 `_func(block_m)`（single-tile）或 `_func(block_m, tile_n)`（tiled）。

**Elementwise 家族（LerpTensor 等）— 合并为 `block_size`**：`threads * npt` 的乘积是有意义参数（per-block 元素数，决定 UB/fragment 分配），不能直接移除。两个参数**合并为单一的 `block_size = threads * npt`**。

1. **工厂签名**：`_make_lerp_tensor_kernel(N, dtype, ..., threads=256, npt=8)` → `_make_lerp_tensor_kernel(N, dtype)` — `threads` 和 `npt` 从工厂签名移除
2. **callable 签名**：`_func(threads_arg, npt_arg)` → `_func(block_size)` — 两个参数合并为一个
3. **`default_config`**：`{"threads": 256, "num_per_thread": 8}` → `{"block_size": 2048}` — 乘积保持不变（fp32: 256×4=1024，fp16/bf16: 256×8=2048）
4. **`T.Parallel`**：`T.Parallel(threads, npt)` → `T.Parallel(block_size)` — 二维并行循环合并为一维
5. **`T.Kernel`**：`T.Kernel(grid, threads=threads)` → `T.Kernel(grid, is_npu=True)` — 无 threads 维度

### autotune 移除说明

GPU 版本的 autotune 机制通过 TileLang autotuner（`from tilelang.autotuner import autotune`）在编译期探索 `block_m × threads × tile_n` 组合并重新编译内核。NPU 版本**全面移除** autotune，改用启发式 config 选择（`default_config` 中的 block_m 遍历逻辑）。移除涉及：

1. **`kernel_base.py`**：移除 `from tilelang.autotuner import autotune`、`autotune()` 方法、`_autotune_initial_kwargs`、`_call_autotuned_kernel`、`autotune_supply_prog`、`_AUTOTUNE_PARAM_ALIASES`、`autotune_configs` 属性；`init_config` 签名从 `(config, tune)` 简化为 `(config)`
2. **`logsumexp.py`**：移除 `autotune()` 方法、`autotune_configs` 属性、`_tile_n_candidates` 方法、`_MAX_TILE_N_CANDIDATES` 常量、`tune` 参数
3. **`op_base.py`**：移除 `Op.autotune()` 方法
4. **`_primitives.py`**：移除 `tune_by_forward` 函数
5. **Op 层 (`softmax.py`)**：移除 `tune` 参数和 `self._tune` 属性
6. **测试 / 基准测试**：移除 `tune` 参数化维度和 try/except autotune 错误处理

### 新增代码

| 文件 | 用途 |
|---|---|
| `tileops/device.py` | **新增模块**：设备后端抽象（NPU/CUDA/CPU 三后端），所有设备相关代码的单一入口 |
