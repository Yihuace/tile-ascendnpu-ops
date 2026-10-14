# BatchNormBwdOp → npuir — Feasibility Probe (Stage 0 pre-research)

> Verdict: **FEASIBLE** (real on-device probe passes; no missing capability identified).
> This is a harness feasibility probe, NOT the migrated operator. No migration
> state was written; the GPU repo was read as data only.

## Toolchain / device stamp

| item | value |
|------|-------|
| target | `target="npuir"`, Developer mode (`TILELANG_ASCEND_MODE=Developer`) |
| tilelang | `0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b` |
| CANN | 8.5.2 (task stamp; same toolchain as the `fused_add_layer_norm` precedent) |
| device | Ascend910B3, AI cores = 20 (torch_npu `2.7.1.post4`, torch `2.7.1+cpu`) |
| probe file | `batch_norm_bwd_probe.py` |
| run log | `probe_stdout.log` (exit 0) |
| construct evidence | `_scratch_constructs.py` (all OK), `_scratch_tiled.py` (oversized-S, PASS) |

## 1. Algorithm summary (extracted from source, read as data)

Source: `TileOPs-main/src/tileops/kernels/norm/batch_norm.py`
(`_batch_norm_bwd_kernel` L1487, `_batch_norm_bwd_wide_kernel` L1658,
`_batch_norm_bwd_split_kernel` L1795; host classes L1981/L2044/L2146).
Op interface: `TileOPs-main/src/tileops/ops/norm/batch_norm.py:182` (`BatchNormBwdOp`).
Manifest: `TileOPs-main/src/tileops/manifest/spec/norm.yaml:250`.

Inputs `grad_out, x : [N, C, S]`, `weight, mean, rstd : [C]` fp32.
Outputs `grad_x : [N, C, S]`, `grad_weight (dgamma) : [C]` fp32, `grad_bias (dbeta) : [C]` fp32.

With `L = N*S` and `x_hat = (x - mean) * rstd`, per channel `c`:

```
grad_bias[c]   = Σ_{i∈L} grad_out[c,i]
grad_weight[c] = Σ_{i∈L} grad_out[c,i] * x_hat[c,i]
grad_x[c,i]    = weight[c]*rstd[c]/L * ( L*grad_out[c,i]
                                          - grad_bias[c]
                                          - x_hat[c,i]*grad_weight[c] )
```

* **Reduction axes / extents**: per channel `c`, reduce over the flattened
  `(N, S)` axis (`L = N*S` elements) — i.e. axes `{0, 2}` of the native
  `[N, C, S]` layout. Two independent reductions: `grad_bias` and
  `grad_weight = Σ(grad_out·x_hat)`.
* **Layout subtlety**: the reduction axis set is *not* the trailing axis; the
  channel axis `C` is the middle axis. So the per-channel plane
  `x[:, c, :]` is strided in GM (row stride `C*S`), unlike the plain
  row-reduction template (`fused_add_layer_norm` reduces the trailing dim).
* **dtypes**: `float16`, `bfloat16`, `float32` (op docstring + manifest
  `DType[float32|float16|bfloat16]`); channel grads always fp32.
* **Cross-block reduction (dgamma/dbeta over N)**: the GPU source has three
  variants; only the **split** variant merges across blocks
  (`partial_sum[C, splits]` → a `finalize` kernel sums them). **No atomics are
  used anywhere in the GPU source.** On NPU the chosen design is **one core per
  channel**, so the whole channel reduction is *within one core* — **no
  cross-block merge and no atomics are required**, even for channels too long to
  hold in UB (in-core serial S-tiling, §3). If one instead adopted the GPU split
  strategy, `T.atomic_add` (fp32 supported, `docs/Tilelang.language/原子操作/T.atomic_add.md`)
  or a partial-buffer + finalize two-pass merge are both available — so there is
  no capability blocker in either design.

## 2. Hard capabilities required

1. Per-channel **strided plane gather** from the native `[N, C, S]` layout
   (`grad_out[0:N, c, 0:S]` → `(N,S)` UB), i.e. a non-trailing reduction region.
2. **Two accumulators** over `L = N·S` (two `T.reduce` reductions per channel).
3. **Elementwise grad_x epilogue** with per-channel scalar broadcasts
   (`weight/mean/rstd/grad_bias/grad_weight`).
4. fp32 opmath with **fp16/bf16 transit** (`T.vcast` rint) + single round-back
   (bf16 has no Vector arithmetic).
5. Per-channel scalar outputs `grad_weight`, `grad_bias`.

## 3. Probe design & constructs exercised

Reuses the `fused_add_layer_norm` Developer pattern
(`examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/…`):
`T.Kernel(C, is_npu=True)` (one core per channel), `T.alloc_shared` (UB),
auto-sync, explicit slice copies, `T.vcast` transit.

Reduction implemented as a two-step `T.reduce`:
`(N,S) --dims=[1]--> (N,1) --dims=[0]--> (1,1)`,
per-channel scalars broadcast with `T.vbrc((1,1) → (N,S))`.

The `_scratch_constructs.py` run validates, individually on the device:
strided per-channel plane gather, scalar slice load `weight[c:c+1]`,
two-axis reduce, scalar store, and the epilogue broadcasts (all OK).

**Oversized-S extension** (`_scratch_tiled.py`, `S=512`, `bS=128`): one core per
channel with an in-core `T.serial` S-tile loop — pass 1 accumulates a bounded
`(N,bS)` tile elementwise (bounded UB), one final reduce yields the scalars;
pass 2 re-reads tiles for the `grad_x` epilogue. **No second block, no
cross-block merge, no atomics** — this is the NPU replacement for the GPU
split kernel and covers the `large-spatial` manifest workload (`B=4,C=128,S=1024×1024`).

## 4. Probe results

`python batch_norm_bwd_probe.py` → exit 0, **PROBE PASS**.
Golden = pure-torch CPU `batch_norm_backward` (`TileOPs-main/workloads/norm.py:385`);
tolerances = `reference_tolerance` (`TileOPs-main/workloads/numerics.py:264`),
elementwise `rtol=atol` ∈ {fp16 1e-3, bf16 1.6e-2, fp32 1e-5}.

| case (N, C, S) | dtype | max_abs_err grad_x | grad_weight | grad_bias | tol | result |
|---|---|---|---|---|---|---|
| (8, 64, 1024) | float16 | 4.883e-04 | 3.052e-05 | 1.526e-05 | 1e-3 | PASS |
| (4, 64, 256)  | float16 | 6.104e-05 | 3.052e-05 | 3.815e-06 | 1e-3 | PASS |
| (8, 64, 256)  | bfloat16| 3.052e-05 | 1.526e-05 | 1.907e-06 | 1.6e-2 | PASS |
| (4, 64, 256)  | float32 | 4.768e-07 | 8.583e-06 | 4.768e-06 | 1e-5 | PASS |

`(8,64,1024)` is manifest `resnet-stage1` (`B=8,C=64,S=32×32`).
Worst `max_abs_err` across cases: **4.883e-04** (fp16 grad_x, within 1e-3).

No compile errors and no runtime errors were observed
(`AscendNPU IR compile success`, no workspace allocation, exit 0).

`probe_stdout.log` tail:

```
device: Ascend910B3
[PROBE] (N=8,C=64,S=1024) dtype=float16 max_abs_err grad_x=4.883e-04 grad_weight=3.052e-05 grad_bias=1.526e-05 (rtol=atol=1.0e-03)
[PROBE] (N=4,C=64,S=256) dtype=float16 max_abs_err grad_x=6.104e-05 grad_weight=3.052e-05 grad_bias=3.815e-06 (rtol=atol=1.0e-03)
[PROBE] (N=8,C=64,S=256) dtype=bfloat16 max_abs_err grad_x=3.052e-05 grad_weight=1.526e-05 grad_bias=1.907e-06 (rtol=atol=1.6e-02)
[PROBE] (N=4,C=64,S=256) dtype=float32 max_abs_err grad_x=4.768e-07 grad_weight=8.583e-06 grad_bias=4.768e-06 (rtol=atol=1.0e-05)
worst max_abs_err across cases: 4.883e-04
PROBE PASS
```

## 5. Verdict: FEASIBLE

Every hard capability in §2 was exercised on the real device and matched an
independent CPU golden within the op's tolerance:

* non-trailing per-channel reduction over `N·S` from the native strided layout;
* two-accumulator reduction (`grad_bias`, `grad_weight`);
* `grad_x` elementwise epilogue with per-channel broadcasts;
* fp16/bf16 fp32-transit and fp32 paths, per-channel scalar outputs;
* oversized-S in-core tiling with **no cross-core merge and no atomics**.

**No missing capability.** No entry should be registered in
`.agents/evolution/capability-gaps.md` for this operator.

## 6. Residual risks / notes (not capability gaps)

* **UB budget / block sizing.** The held-plane probe needs
  `(N,S)` fp32 buffers (`g, xh, prod, brc`) + transit `stage`; oversized
  workloads use the tiled path (`§3`). The production wrapper should size
  `bS`/core-split from the measured multi-buffered UB budget
  (`~20 B/resident-elem`, as recorded in the `ada_layer_norm` /
  `fused_add_layer_norm` kernels), a parameterization detail, not a missing
  primitive.
* **Parallel structure.** The probe uses `T.Kernel(C)` (one block per channel,
  oversubscribed); production should apply the precedent persistent core-split
  (`num_kernels = min(logical, 40)`) for the `C > 2×AI-core` workloads. Perf-only.
* **Determinism of dgamma/dbeta.** `T.reduce` lowers to `hivm::VReduceOp`
  (`docs/Tilelang.language/规约操作/T.reduce.md §3`); a fixed reduce order per
  core keeps the sums deterministic. The probe validates values, not bit-exact
  reproducibility across cores — verify at Stage 4 if the op needs it.

## 7. References

* GPU source: `TileOPs-main/src/tileops/kernels/norm/batch_norm.py`
  (L1487/L1658/L1795/L1981/L2044/L2146); op `src/tileops/ops/norm/batch_norm.py:182`;
  manifest `src/tileops/manifest/spec/norm.yaml:250`.
* Golden / tolerance: `TileOPs-main/workloads/norm.py:385`,
  `TileOPs-main/workloads/numerics.py:264`.
* NPU precedent: `examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/`
  (DESIGN.md + integrated kernel).
* npuir docs (routing per `AGENTS.md`):
  * `docs/Tilelang.language/规约操作/T.reduce.md` (§2.2.2 rank/size-1 dim rule)
  * `docs/Tilelang.language/规约操作/T.reduce_sum.md`
  * `docs/Tilelang.language/数学操作/T.vsub.md` (§2.2.2 broadcast: `[M,N]-[M,1]`, `[1,N]-[1,1]`, scalar)
  * `docs/Tilelang.language/数学操作/T.vmul.md`, `T.vadd.md`
  * `docs/Tilelang.language/shape操作/T.vbrc.md` (§2.2.2: `(1,1,K)→(M,N,K)` broadcast)
  * `docs/Tilelang.language/内存操作/T.copy.md` (§2.1 extent inference, §2.2.2 size-1 dim fold, §2.3 GM↔UB)
  * `docs/Tilelang.language/内存操作/T.clear.md`, `docs/Tilelang.language/数据类型转换操作/T.vcast.md`
  * `docs/Tilelang.language/原子操作/T.atomic_add.md` (available, not required)
