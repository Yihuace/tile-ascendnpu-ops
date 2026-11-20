# Integration Log -- BitwiseNotFwdOp

- meta: `examples/TileOPs/tileops/kernels/elementwise/bitwise_not/.migration_meta.json`
- integrated: `bitwise_not_fwd_kernel` -> `tileops/kernels/elementwise/bitwise_not/bitwise_not_kernel/bitwise_not_fwd_kernel.py`
- design_docs: `bitwise_not_fwd_kernel` -> `tileops/kernels/elementwise/bitwise_not/bitwise_not_kernel/bitwise_not_fwd_kernel_DESIGN.md`
- ascend_mode: `Developer` (read from Stage 3 product `ASCEND_MODE`; wrapper `Kernel.ascend_mode` synced, unchanged)
- attempts: 1

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261012_191515_259850_BitwiseNotFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261012_191515_259850_BitwiseNotFwdOp/report.md`
- report_status: passed
- correctness: pass (11/11 cases; `summary.correctness_passed=true`, avg max abs error 0.00e+00)
- benchmark: pass (6/6 msprof cases; `profiler_valid=true`, `profiler_fallback_count=0`, `invalid_profile_count=0`)

### benchmark 数值（来自本次 run.json `performance.cases`）

| label | dtype | shape | latency (us) | roofline (%) | bandwidth (TB/s) |
|---|---|---|---:|---:|---:|
| smoke-1m | int32 | [1048576] | 7.7202 | 30.18 | 1.8000 |
| smoke-1m | int64 | [1048576] | 9.3402 | 49.90 | 1.8000 |
| elementwise-16m | int32 | [4096,4096] | 91.7818 | 42.78 | 1.8000 |
| elementwise-16m | int64 | [4096,4096] | 183.9237 | 54.71 | 1.8000 |
| elementwise-64m | int32 | [8192,8192] | 486.0897 | 50.97 | 1.8000 |
| elementwise-256m | int32 | [16384,16384] | 1932.6387 | 59.18 | 1.8000 |

## 调试历史

- attempt 1:
  - 症状：集成后 `pytest tests/ops/test_bitwise.py` 全量 8 个 `test_bitwise_not` 用例失败，`ValueError: Expected 2 inputs, got 1` 于 `bitwise_not_fwd_kernel(N, dtype_str, block_size=block_size)(x)`（`jit_npu.py:1281`，`total_params=3, num_inputs=2`）。
  - 分类：wrapper 胶水错误（集成 kernel 的 `main(x, y, shape)` 新增动态 `shape` 参数未在 wrapper 调用点传入）；同时暴露 DESIGN 已登记的两个 dtype 胶水缺口（R-1 bool 存储视图、R-2 / CG-2026-0014 uint8 out_idx 输出 dtype 映射缺失）。
  - 修复（仅 wrapper 胶水层 `tileops/kernels/elementwise/bitwise_not/bitwise_not.py`，备份 `history_version/bitwise_not_s5_attempt1.py`）：
    1. 调用 kernel 时补传动态 `shape=N`：`kernel(x, N)`；
    2. bool 分支：host 侧 `x.view(torch.int8)` 传入、结果 `.view(torch.bool)` 返回（R-1，1 字节存储视图，仅 reinterpret，不改内容/stride）；
    3. uint8 分支：`kernel(x, N)` 后 `.view(torch.uint8)[:N]` 回收正确字节（R-2 / CG-2026-0014，out_idx 自动分配缓冲为 float32，内核已写入正确 N 字节）。
  - 结果：`pytest tests/ops/test_bitwise.py` 11/11 通过；重新运行完整单算子 report `status=passed`，correctness 11/11、benchmark 6/6（msprof 全部有效）。

## issues

- report `warnings` 含 6 条 `missing tileops candidate record`（每个 benchmark case 一条）。benchmark 用例仍 6/6 passed、msprof profile 全部 `profiler_valid=true`；该 warning 属 report/record 层提示，未影响正确性与 benchmark 结论，仅记录不修复。
- 无 DESIGN.md 缺失（`[warn] no DESIGN.md` 未出现）。
