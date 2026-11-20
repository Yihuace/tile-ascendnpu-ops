# RETROSPECTIVE -- ArgmaxFwdOp 迁移

## 本轮做了什么

1. 确认 `ArgmaxFwdOp` 六区在 `integration/npu-ops` 基线已在位（来自上游 `93891e3`），
   本轮定位为「补齐验收证据 + 实跑验证」，不新增算子代码。
2. 修复本机运行环境：安装态 tilelang 缺 `utils/npu_utils.py`，改用源码树 + dev root build。
3. 实跑精度：42/42 通过。
4. 生成完整报告 `run.json` / `report.md` / `report.html` / `benchmark.json` / pytest xml。
5. 定位 benchmark 2 个失败用例的根因，双向实验证明为工具链版本互斥，
   登记 `CG-2026-0018`。
6. 补齐 `REVIEW.md`、`integration_log.md`、`RETROSPECTIVE.md`、`perf_opt/perf_records.jsonl`。

## 做对了什么

- **先跑失败命令再读代码**：先拿到 `ModuleNotFoundError` 与 `syn_instr_mode` 的真实
  traceback，再回到 `codegen_npuir.cc` / `npu_utils.py` 定位，没有凭猜测改代码。
- **假设做双向验证**：怀疑 PATH 命中旧版 bishengir 后，没有停在「换 PATH 试试」，
  而是把两条路径都跑通对比，确认**两边都不行**，才把它定性为环境缺口而非算子缺陷。
- **识别并作废污染数据**：发现另一任务在同卡并发跑 benchmark 后，删除该轮报告并在
  空闲状态下重测，避免把失真时延写进证据。
- **不美化结论**：`status=partial` 原样保留，已知限制写进 integration_log 与 REVIEW，
  没有删用例、降容差或 skip。

## 做得不够

- 未重做 DESIGN.md 的 9 维度全量设计检视，`REVIEW.md` 只覆盖本轮实际做过的 4 个维度
  （可运行性 / 失败路径 / 证据完整性 / 限制诚实性）。
- 未在本机重测 baseline，因此无法给出可信的提升倍数；`opt_log.md` 的历史数据与本机
  工具链不同，只能作参考。
- 首次 benchmark 在并发环境下白跑一轮（约 7 分钟），排程前应先查 `npu-smi` 占用。

## 后续建议

1. **工具链对齐是解放 lm-head 两档的前提**：需让 `bishengir-compile` 与 CANN 的 `hivmc`
   版本互相兼容（或整体升级 AscendNPU-IR + CANN 到配套版本），之后补测
   `lm-head-argmax` fp16/bf16 两档。
2. 工具链对齐后，在同一套环境下重测 baseline 与 final，补齐可比较的提升倍数，
   再按 P0 裁决规则选出裁决档。
3. 跑 benchmark 前先做 NPU 占用检查，避免并发污染（可做成提交前检查脚本的一步）。
4. `get_npucompiler_path()` 依赖 `shutil.which` 容易命中错误的编译器，建议后续在仓库
   文档或环境自检里显式校验 `bishengir-compile --version` 与 `BISHENGIR_ROOT_PATH` 一致。
