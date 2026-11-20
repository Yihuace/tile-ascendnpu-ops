# 优化工作列表（Conv1dFwdOp · Stage 4）

> 维护约定：`[x]` 完成 / `[~]` 进行中 / `[ ]` 未开始；每项完成后在此行尾补证据路径。
> NPU 纪律：所有上机命令一律经 `.agents/tools/npu_lock.sh`（工作区互斥 + 空闲预检 + 孤儿回收）。

## P0 开工前置

- [x] 工作区分组提交，建立干净基线
  - `43e2acf` CANN 9.0 重定基线证据 + `npu_lock.sh`
  - `094eee9` GDN 迁移 WIP
- [x] NPU 独占工具 `npu_lock.sh`（互斥锁 / 空闲预检 / 超时回收 / 残留告警）
- [x] optimize 状态机初始化（`phase=TUNING`，`stage_plan=[4]`）
- [x] 基线目录 `perf_opt/{logs,profiles,ab_r1}`

## P1 基线采集（msprof op Task Duration，逐 case 串行）

- [x] `whisper-conv1/fp16` — 5443.55 µs，ratio 0.1155%
  - 证据：`perf_opt/logs/bench_whisper-conv1-fp16.json`
- [x] 批量采集 17 个 tune case：**16 通过 / 1 失败**（`encodec-deep-bias/fp16`，CG-2026-0003 SIGSEGV）
  - 证据：`perf_opt/logs/bench_all.{json,md,log}`
- [x] 有效基线 **12** 条（msprof 直采）；4 条批量内 msprof 回退 events，已改逐 case 独立进程重采
  - 重采脚本：`perf_opt/rerun_events_fallback.sh`
- [x] `perf_opt/workload_inventory.json`：18 case（17 tune + 1 smoke）
- [x] `perf_opt/perf_records.jsonl`：12 条 baseline 记录（重采后补 4 条）
- [ ] 校验 profile 有效性（captured_op_count / 无 fallback）

## P2 诊断（首轮已定位，待全量印证）

- [x] `whisper-conv1/fp16` 瓶颈：**Vector prelude**
  - Cube 活跃比 0.15%（`aic_cube_ratio`）
  - AIV MTE2 占 97.1%，有效带宽 **2.39 GB/s**；GM→UB 每核 ~13 MB
  - prelude = 权重重排（两次 UB transpose，20 个 co 块）+ `x_pad` 全量拷贝
  - 每个 kernel core 都重复执行整段 prelude（20 核重复）
- [x] 代码定位：`_conv1d_unit_stride_kernel.py:_prelude`（L184-225）
- [x] 印证 `_conv1d_main`（wav2vec / encodec-init）同构 prelude（`_conv1d_kernel.py:_prelude` L180-218，另有 poly 抽取）
- [x] 分派映射：whisper / encodec-deep → unit_stride；wav2vec / encodec-init / smoke → `_conv1d_main`

## P3 优化实验（每轮只改一个主要点）

- [x] R1-A：`x_pad` 恒等时跳过拷贝 —— **不采纳**（whisper +0.2%，仅 encodec-deep −7~−9%）
  - 证据：`perf_opt/opt_log.md` §4.2、`perf_records.jsonl`（r1a_identity_x）
- [x] R1-B：权重重排按 core 切分 —— **挂起**（分区重排数值错 max_abs=81.38；仅映射版逻辑未证伪）
  - 证据：`perf_opt/candidates/{r1b.py,r1b_map_only.py}`、`opt_log.md` §4.3
- [x] 编译器 flaky 判定：同文件重复编译崩溃点随机 → 单次失败不可归因候选（CG-2026-0003 补充证据）
- [ ] 恢复点：先用「仅映射 + 全量重排」验证映射，再查分区重排数值 bug

## P4 合并与回归（挂起）

- [ ] 逐 workload 合并 winner，全量 tune + smoke 精度复测
- [ ] wrapper 双 import 切换块翻转（仅允许的 wrapper 修改）
- [ ] `tileops-report run --op Conv1dFwdOp --prof-mode msprof` 端到端验收

## P5 收尾（挂起）

- [ ] `perf_opt/opt_log.md` Final Summary + Skill Retrospective
- [ ] `TUNING_COMPLETED` → 精度回归 gate → `DONE`
- [ ] 更新 `MIGRATION_INDEX.md` Conv1d 的 9.0 性能口径

> 2026-10-17 决策：按“先易后难”，Conv1d 暂挂起（Stage 4 保持 TUNING，可恢复），
> 先做 Add → Sub → Isnan/Isinf/Isfinite。恢复入口见 `opt_log.md` §4.3/§4.4。

## 并行不占 NPU 的待办（可穿插）

- [ ] ④ 记录回写：用 §11/§13 的 9.0 数据刷新 `MIGRATION_INDEX.md §C` 与版本戳
- [ ] T6：Gelu「删 NaN 恢复」在 9.0 下重验 + 补 nan/inf 用例（需一次上机，排在 P3 之后）
- [ ] `BatchNormBwd large-spatial/fp16` +22.7% 回归定位（需独立立项）
