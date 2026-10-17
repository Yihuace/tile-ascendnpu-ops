# RETROSPECTIVE — prefill_blocksolve_A_bthd（GDN prefill 函数 2/6）

## Stage 1 (design) Retrospective — first_design — 2026-10-17

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 迁移任务的关键"能不能这么算"判断（Cube 链式回喂、`initC=False` 落盘后累加）在 skill 文档里只有"读 EXPERT_FINDINGS"的间接指引，实际必须自建探针才能定案；探针耗时 ≈ 全 Stage 的 40% | DESIGN.md §3.4/§9；`history_version/design_probe_doubling_cube.py` | 在 `tilelang-op-design/references/migration-analysis.md` §6.2 增加一句："算法候选的**可行性**由 Cube 侧 op 代价与回喂通路决定时，先跑 D-3 探针再排候选" | R |
| Design-flow | `verify_equiv.py` 的"违反率"判据口径未在 skill 中给公式，首版用 fp32 绝对容差导致 3 项假 `EQUIV_FAIL`（源式的 fp16 门表舍入与 fp16 输出量化被误判） | `verify_equiv.py` 头部注释 + DESIGN.md §1.6.1 判据行 | 在 `tilelang-op-design/SKILL.md` Phase 2 第 5 步补充："违反率判据 = `|cand-ref| <= rtol*|ref| + 2*eps_out*max|ref|`（rtol 取输出 dtype 口径）" | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| Expert 下 fp32(HF32) gemm 64×64×64 可用且 err ≈ 3e-8；`initC=False` 可累加进"已 store 过"的 L0C（err 2.9e-8） | D | `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_probe_accum_form.py`（V1/V2/V3） | `python3 design_probe_accum_form.py` | tilelang 0.1.2 dev root + CANN 9.0.0 / 910B3 | pattern-library/constants.md |
| 64×64 fp32 的 gemm/store/load 平均 **0.25 µs/op**（6 op 一级 = 1.52 µs，`reps` 差分法） | D | `history_version/design_probe_doubling_cube.py` PROBE-TIMING | `python3 design_probe_doubling_cube.py` | 同上 | pattern-library/constants.md |
| 回喂读回 **`out_idx` 张量**在 CANN 9.0 上不是"数值错"而是 **bishengir SIGSEGV**（比既有 K2 结论更硬） | D | `history_version/design_probe_doubling_cube.py` 首版（读 `out_t1` 即崩）；`EXPERT_FINDINGS.md` §12.1 K2 | 同上（首版形态） | 同上 | pattern-library/traps-compiler.md（追加到 CONST-store-fixpipe-gm-only 相关条目） |
| L0C(fp32) → GM(**fp16**) → L1 回喂在 64×64、`initC=False` 形态下数值错（err 1.875），与 probe 13 的"可用"结论**不矛盾但形态敏感** | D | `design_probe_accum_form.py` V4 | 同上 | 同上 | pattern-library/traps-runtime.md |
| 三角逆的"依赖性"而非 "FLOPs" 才是 NPU 选型主导项：同一语义下 25 gemm(47 µs) vs 11 gemm(11 µs)，MAC 反而多 8× | P | DESIGN.md §1.6.0 R3/R4 | `verify_equiv.py`（仅数值面） | 同上 | algorithm-candidates.md（三角求解族） |

### Transferable Lessons（可迁移教训）

- **迁移"小矩阵求逆"类算子时，先算 op 数再算 FLOPs**：NPU 上 16³ gemm 与 64³ gemm 都是一次发射，
  MAC 差 64×；源实现"少算乘加"的分块策略在 GPU 上赢、在 NPU 上输。判断依据用"依赖级 gemm 数 ×
  实测 op 代价"，不要用 MAC 数排序候选。
- **log-doubling（`A = Π(I+N^{2^i})`）是三角逆在"无片上回喂"硬件上的正确形态**：11 个 gemm、
  无行串行链、天然 fp32 累加；`initC=False` 让"加 T_j"这一步零成本。凡"下三角/幂零"结构都可套用。
- **`verify_equiv.py` 必须先把容差口径对齐"输出 dtype"**，否则源式自身的舍入（fp16 门表、fp16 中间块）
  会把"更准的候选"判成不等价——首版就因此得出 3 个假 FAIL。
- **探针要"分层二分"**：一次写完整链（Gram+掩码+5 级链）失败时无法定位；本次先二分出
  "读 `out_idx` 张量 = 崩"再拼装，总耗时反而更低。
