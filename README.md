# tile-ascendnpu-ops

TileOPs GPU → Ascend NPU 算子迁移与自动开发工作区，聚焦于 `TileLang + npuir` 场景下的算子移植、验证、调优和归档。该仓库不仅包含 NPU 版 TileOPs 工程本体，也提供一套 Agent-driven 的运作流程，用于驱动“设计 → 评审 → 实现 → 优化 → 集成”全过程。

## 1. 项目目标

这个仓库的核心目标有三点：

- 将 GPU 侧 TileOPs 算子迁移到 Ascend NPU，覆盖统一的 `npuir` 路线。
- 通过标准化流程，把迁移过程中的设计、评审、验证和性能数据固化下来。
- 把重复性工作封装到 `.agents/` 与 `.opencode/` 中，尽量减少人工重复劳动。

本仓库不是传统业务应用，而是一个“算子开发与迁移流水线”型工程。

## 2. 仓库结构

```text
.
├── AGENTS.md                    # 仓库总纲：API 约定、文档路由、能力触发规则
├── README.md                   # 项目说明（当前文件）
├── MIGRATION_INDEX.md          # 迁移总索引：状态、正确性、性能、report 指向
├── TileOPs迁移任务书.md        # 算子迁移 spec 与批次/出口标准
├── docs/                       # TileLang/Ascend 相关文档和开发说明
│   ├── Tilelang.language/      # API 文档（优先参考）
│   ├── developer/              # 环境变量和 runtime 文档
│   ├── 开发指南.md
│   ├── 快速入门.md
│   └── Tilelang算子调试指南.md
├── .agents/                    # 共享技能、标准、工具与演化记录
│   ├── skills/                 # tilelang-op-design / review / develop / optimize 等能力
│   ├── tools/                  # statectl / kb_search / standards_check / gate_lint
│   └── evolution/              # capability gaps / proposal queue / stats
├── .opencode/                  # Agent 编排配置
│   ├── agents/                 # conductor、designer、reviewer、developer、optimizer
│   ├── command/                # 命令脚本
│   └── conductor-scenarios/    # new-op / migration / optimize / harness
├── examples/
│   ├── TileOPs/                # 业务工程：NPU TileOPs 标准工程 + benchmark + tests
│   │   ├── tileops/            # device / workloads / ops / kernels / reporting
│   │   ├── tests/              # 正确性测试
│   │   ├── benchmarks/         # 性能 benchmark
│   │   ├── docs/               # TileOPs 适配与 Roofline 文档
│   │   └── pyproject.toml      # Python 项目配置
│   ├── flash_attention/        # 参考实现 / 样例实现
│   ├── hardsigmoid/            # 已完成迁移算子工作区示例
│   └── <op_slug>/              # 每个待迁移算子的任务工作区
├── .gitignore
└── ...
```

## 3. 关键目录说明

### `.agents/`
共享的开发知识与工具目录，包括：

- `skills/`: 各阶段技能，如设计、评审、开发、优化和回顾。
- `tools/`: 状态管理、门禁检查和知识索引工具（例如 `statectl`, `standards_check`, `kb_search`）。
- `evolution/`: 能力提案与演化记录。

### `.opencode/`
这是 Agent 编排入口，负责给任务路由到不同 scenario：

- `new_op`: 新算子开发
- `migration`: GPU -> NPU 迁移
- `optimize`: 性能优化
- `harness`: 批量算子迁移/集成流程

### `examples/TileOPs/`
这是集成侧的标准工程，包含：

- `tileops/`: 核心框架
- `tests/`: 正确性测试
- `benchmarks/`: 性能测试
- `reporting/`: 聚合报告输出
- `manifest/`: 算子 manifest 与声明

### `examples/<op_slug>/`
这是逐算子任务工作区的目录。通常包含：

- 设计文档 `DESIGN.md`
- 评审文档 `REVIEW.md`
- 实现代码
- 历史版本 `history_version/`
- 状态文件 `.stage_state.json`
- 回顾文档 `RETROSPECTIVE.md`
- 报告 / 验证结果

## 4. 设计与工作流

仓库采用的是“Stage 0 -> Stage 5”的工程化流程：

1. Stage 0：脚手架生成
2. Stage 1：设计（算法研究、目标说明）
3. Stage 2：评审（风险检查、正反例评估）
4. Stage 3：开发（实现 kernel / op）
5. Stage 4：优化（性能调优）
6. Stage 5：集成与验证

链路上的状态与质量门禁由 `.agents/tools/statectl.py`、`gate_lint.py` 与 `standards_check.py` 维持，以降低流程漂移和不一致结果的风险。

## 5. 运行方式

### 5.1 运行 TileOPs 工程

```bash
cd examples/TileOPs
pip install -e .[dev]

# 正确性测试
pytest tests/ -v

# 性能 benchmark
pytest benchmarks/ -v

# 统一报告命令
# 列出算子和入口
# tileops-report list

# 运行完整正确性 + benchmark，并生成报告
# tileops-report run --all --prof-mode msprof
```

### 5.2 迁移流程入口

通常在根目录以 Agent 方式驱动，不直接写死硬编码的脚本入口；流程由 `.opencode/agents/` 和 `conductor-*` 决定，以任务目标驱动选择 `new_op` / `migration` / `optimize` / `harness` 场景。

## 6. 当前状态

- 迁移状态总览见 [`MIGRATION_INDEX.md`](MIGRATION_INDEX.md)。
- 本批（任务书 §5/§6）已完成：`HardsigmoidFwdOp`(1b)、`ClampScalarFwdOp`(1a)、`HardswishFwdOp`(1a)。
- 更早批次已完成：`AdaLayerNormFwdOp`、`ArgmaxFwdOp`、`LerpTensorFwdOp`、`LogSumExpFwdOp`、`MishFwdOp`、`MultiHeadAttentionFwdOp`、`SSDChunkScanFwdOp`。
- 迁移工作区示例：`examples/hardsigmoid/`、`examples/clamp_scalar/`、`examples/hardswish/`。

此外，仓库在 `README.md` / `AGENTS.md` 中对以下约定做了明确说明：

- 迁移（baseline）与优化（Stage 4）分开处理
- 状态记录和版本戳必须保留
- `v-prefix` API 优先：例如 `T.vadd`, `T.vmul`, `T.vexp`, `T.vcast`
- 文档路由优先参考 `docs/Tilelang.language/`

## 7. 开发约定

- 优先使用 `v-prefix` API，保持对历史 `npuir_xxx` 兼容。
- 设计、评审和实现必须遵循仓库共享标准。
- 每个迁移任务都需要保留可追溯的状态和报告。
- 对新能力或限制要写入 `capability-gaps` / `evolution` 相关记录。

## 8. 适合的使用场景

适用于以下场景：

- GPU TileOPs 子图迁移到 Ascend NPU
- 逐算子设计与实现
- 性能对比和 benchmark 归档
- Agent 驱动的 kernel 迁移与持续迭代

## 9. 说明与限制

这个仓库的重点在于“算子迁移工程”而非通用业务应用，因此：

- 存在大量生成产物与过程记录
- 强依赖特定环境和工具链版本
- 代码与操作流程耦合较强，适合在研究/开发型场景使用

### 已知缺口（技术）

- **提取器覆盖**：`extract_tl_kernel.py` 对 16 个待迁算子，仅 **2 个**（`HardswishFwdOp`/`HardsigmoidFwdOp`）产出真正自包含的**合并** kernel（pattern `A`）；**3 个**（`GeluAndMulFwdOp`/`GeluTanhAndMulFwdOp`/`FusedAddLayerNormFwdOp`）为 **`A-fallback`**（未合并、只发 raw builder；其中 gated 族引用空的 `op_func_for`/`_OP_FUNCS` 注册表，**非自包含**）；**11 个** pattern `B`。`A-fallback`/`B` 的算子 Stage 0 仍需**人工合并 op-body**（gelu 族、clamp、binary 等）。提取器已对 `A-fallback` 显式告警。
- **工具链游离改动**：`/home/developer/tilelang-mlir-ascend/tilelang/version.py` 有 1 行本地修复（`get_git_commit_id` 加 `cwd=current_dir`，避免版本戳被工作区 git commit 污染），**不在本仓库版本控制内**。
- `standards_check.py check` 现为 `ok`（0 failures / 0 dup）。

## 10. 进一步阅读

- [`AGENTS.md`](AGENTS.md)
- [`MIGRATION_INDEX.md`](MIGRATION_INDEX.md)
- [`examples/TileOPs/README.md`](examples/TileOPs/README.md)
- [`docs/开发指南.md`](docs/开发指南.md)
- [`docs/Tilelang.language/`](docs/Tilelang.language/)

如果你是第一次接触这个仓库，建议按下面顺序阅读：

1. `README.md`
2. `AGENTS.md`
3. `MIGRATION_INDEX.md`
4. `examples/TileOPs/README.md`
5. 具体算子工作区 `examples/hardsigmoid/`

