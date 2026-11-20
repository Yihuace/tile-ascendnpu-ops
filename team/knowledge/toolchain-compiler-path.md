# 工具链编译器路径排查与切换（CANN 9.0.0 / tilelang npuir）

- 适用：自建 `bishengir-compile` / `bishengir-opt` 后，想确认 tilelang npuir 实际调用的是哪一份
- 版本戳：CANN 9.0.0 / Ascend910B3 / tilelang 0.1.2（源码 `9309edb`）/ torch_npu 2.7.1.post4
- 结论先行：**tilelang 不硬编码绝对路径，而是运行期用 `PATH` 解析；`TILELANG_NPU_COMPILER_PATH` 只是 `which` 失败后的兜底。** 因此“只设 env”或“PATH 只放一个二进制”都不够。

## 1. 现象

典型场景：

- 已构建自建编译器，设置 `TILELANG_NPU_COMPILER_PATH=<自建工具链>`，或
  `PATH=<自建 bin>:$PATH`；
- 跑 Conv3d/Conv1d 仍然报同一份 `bishengir-compile` 错误，例如
  `err code: -11`、LLVM crash backtrace、`ConvertHFusionToHIVM` 相关栈；
- 同一份代码在 CANN 自带编译器下可能全过（例如 Conv3dBias 17/17），
  换自建编译器后变成 11 failed / 6 passed。

这类现象优先怀疑**编译器没有真正切过去**，而不是先怀疑 PR 代码。

## 2. 源码里的解析顺序

`tilelang/utils/npu_utils.py`：

```python
@functools.lru_cache()
def get_npucompiler_path():
    npu_compiler_path = shutil.which("bishengir-compile")   # PATH 优先
    if npu_compiler_path is None:                            # 只有找不到才看 env
        npu_compiler_root = os.getenv("TILELANG_NPU_COMPILER_PATH", "")
        npu_compiler_path = os.path.join(npu_compiler_root, "npuc")
    return npu_compiler_path

@functools.lru_cache()
def get_npucompiler_opt_path():
    npu_compiler_opt_path = shutil.which("bishengir-opt")   # 只认 PATH
    if npu_compiler_opt_path is None:
        raise EnvironmentError("Couldn't find executable bishengir-opt.")
    return npu_compiler_opt_path

@functools.lru_cache()
def get_bisheng_path():
    bisheng_path = shutil.which("bisheng")                  # PATH 优先
    if bisheng_path is None:
        npu_compiler_root = os.getenv("TILELANG_NPU_COMPILER_PATH", "")
        bisheng_path = os.path.join(npu_compiler_root, "ccec")
    return bisheng_path
```

| 解析点 | 关键事实 | 影响 |
|---|---|---|
| `get_npucompiler_path()` | `which` 优先；env 只是 fallback；env 语义是 `root/npuc` | 只要 CANN bin 在 PATH，env 不生效；指向 bin 目录也会找错 |
| `get_npucompiler_opt_path()` | 只查 `which("bishengir-opt")`，**没有 env fallback** | 只切主编译器，`bishengir-opt` 仍可能走 CANN |
| `get_bisheng_path()` | `which("bisheng")` 优先；env fallback `root/ccec` | `ccec` / clang 头文件可能仍来自 CANN |
| `@functools.lru_cache()` | 首次解析后缓存 | 同进程改 PATH 无效，必须新起进程 |

`tilelang/jit/jit_npu.py` 里，主编译命令使用 `get_npucompiler_path()`；
`get_bisheng_path()` 还被用于 `extract_device_print_code_from_cann()` 定位
`lib/clang/<ver>/include/ccelib`。所以“主编译器切了、其它没切”仍然可能出问题。

另外，`cannlab-migration/env.sh` 会 `source $CANN_HOME/set_env.sh`，而
CANN 的 `set_env.sh` 会把 CANN 的 `bin` 放进 `PATH` 并设置 `ASCEND_HOME_PATH`。
这就是默认命中 CANN 的原因——路径来自环境，不是代码写死。

## 3. 两种常见做法为什么不够

### 3.1 只设 `TILELANG_NPU_COMPILER_PATH`

```bash
export TILELANG_NPU_COMPILER_PATH=/path/to/custom
```

无效，因为 `shutil.which("bishengir-compile")` 已经命中了 CANN 的二进制，
env 根本不会被读取。并且该 env 期望的是：

```text
$TILELANG_NPU_COMPILER_PATH/npuc    # get_npucompiler_path
$TILELANG_NPU_COMPILER_PATH/ccec    # get_bisheng_path
```

不是 `$TILELANG_NPU_COMPILER_PATH/bishengir-compile`。

### 3.2 PATH 只放一部分二进制

```bash
export PATH=/path/to/custom/bin:$PATH
```

只有 `custom/bin` 里**存在**的二进制会被 `which` 命中。实测：

```text
# 只放 fake bishengir-compile
compile: /tmp/.../fakebin1/bishengir-compile
opt    : /home/developer/Ascend/cann-9.0.0/bin/bishengir-opt
bisheng: /home/developer/Ascend/cann-9.0.0/bin/bisheng

# 放 fake compile + opt
compile: /tmp/.../fakebin2/bishengir-compile
opt    : /tmp/.../fakebin2/bishengir-opt
bisheng: /home/developer/Ascend/cann-9.0.0/bin/bisheng

# 三个都放
compile: /tmp/.../fakebin/bishengir-compile
opt    : /tmp/.../fakebin/bishengir-opt
bisheng: /tmp/.../fakebin/bisheng
```

## 4. 正确切换步骤

1. 先 source CANN 环境（它会 prepend CANN bin）：

   ```bash
   source /home/developer/Ascend/cann-9.0.0/set_env.sh
   ```

2. **在其后**把自建 bin 前置：

   ```bash
   export PATH=/path/to/custom/bin:$PATH
   ```

3. 确认自建 bin 同时包含：

   ```text
   bishengir-compile
   bishengir-opt
   bisheng
   ```

   并且 `bisheng` 旁边有自建工具链的
   `lib/clang/<ver>/include/ccelib`。

4. 每次用**新进程**验证（避开 `lru_cache`）：

   ```bash
   python3 team/knowledge/tools/check_toolchain_path.py \
     --expect-prefix /path/to/custom
   ```

   脚本会打印 `which -a`、env 和 tilelang 实际解析结果；不在期望前缀下会返回非 0。

5. 跑用例，检查失败日志里的 `err cmd:` 行：

   ```text
   err cmd: /实际/使用的/绝对路径/bishengir-compile ...
   ```

   - 是 CANN 路径 → 没有真正切过去；
   - 是自建路径 → 自建二进制本身的问题（或内部又调了 CANN 的 `bishengir-opt`/`hivmc`）。

## 5. 自检清单

```bash
which -a bishengir-compile bishengir-opt bisheng
echo "$PATH"
echo "$ASCEND_HOME_PATH"
echo "$TILELANG_NPU_COMPILER_PATH"
python3 team/knowledge/tools/check_toolchain_path.py --expect-prefix /path/to/custom
```

检查点：

- `which -a` 的**第一个**必须是自建；
- `bishengir-opt`、`bisheng` 也必须是自建；
- `TILELANG_NPU_COMPILER_PATH` 只作为兜底，不要只依赖它；
- 不要在同一个 Python 进程里改 PATH；
- 失败日志的 `err cmd` 是最终判据。

## 6. 错误签名速查

| 签名 | 更可能的原因 |
|---|---|
| `err code: -11` + LLVM crash backtrace | 编译器 SIGSEGV；先确认 `err cmd` 命中哪份编译器 |
| `ConvertHFusionToHIVM` / `getResult(0)` / `mmadL1` | CG-2026-0003 家族（CANN 9.0.0 Cube IR 缺陷） |
| `unknown attribute dma_mode` | 前端产物与 bishengir 版本不配对（见 CG-2026-0003 早期记录） |
| `Unsupported copy from cbuf to cbuf` | `T.copy` L1→L1 方向不支持，不是路径问题 |

## 7. 本次证据

- PR #31（Conv3dBiasFwdOp）：
  - CANN 自带编译器：**17/17 passed**；
  - 自建编译器（方式 1 `TILELANG_NPU_COMPILER_PATH`；方式 2 `PATH`）：**11 failed / 6 passed**，
    报错一致；
  - 结论：环境/工具链阻塞，不是 PR 代码回归；需要工具链侧修复后重测。
- PR #88（Conv1dBiasFwdOp）：CANN 自带编译器下 **28/28 passed**，可作为同链路对照。
- 相关代码：`tilelang/utils/npu_utils.py`、`tilelang/jit/jit_npu.py`、
  `tilelang/env.py::_find_ascend_home`。

## 8. 与 CG-2026-0003 的关系

- CG-2026-0003 是**编译器缺陷本体**（Cube IR / `ConvertHFusionToHIVM` / SIGSEGV）；
- 本文是**工具链路径解析/切换问题**，属于环境配置层；
- 两者可能叠加：先确认路径真的切到自建编译器，再判断自建编译器是否包含 CG-2026-0003 的修复。
  不要把“路径没切过去”误判成“PR 代码引入回归”。

## 9. 维护

- 工具链升级后，重新跑第 4/5 节的自检与 `err cmd` 核对；
- 如果未来 tilelang 改为 env 优先，或 `bishengir-opt` 增加 env 支持，更新本文的解析顺序表；
- 本文只记录路径解析与排障，不复制 CG-2026-0003 的能力缺口正文。
