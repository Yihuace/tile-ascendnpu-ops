# Pre-migration FEASIBILITY PROBE — `GLADecodeOp` (GLA single-token decode)

- **Target**: npuir / Ascend910B3, CANN 8.5.2, tilelang `0.1.2+417dae99`, torch `2.7.1+cpu`, torch_npu `2.7.1.post4`.
- **Deliverable**: `gla_decode_probe.py` (this directory) — minimal but representative, runnable on the real device against an independent pure-torch CPU golden.
- **Verdict**: **FEASIBLE** (see §5). The one real constraint found (UB capacity at the manifest `D=128`) is a standard K-tiling requirement, already present in the GPU kernel, not a missing npuir capability.

> Discipline: GPU baseline read as DATA only; no migration state file written; no writes to the GPU repo. All artifacts live under `examples/gla_decode/probe/`.

---

## 1. Algorithm summary (read from GPU baseline)

Source (DATA): `TileOPs-1238a9f45862dc71e3c38cf79a2873fce9763331/src/tileops/kernels/linear_attention/gla/recurrent.py`
(`GLADecodeKernel` L157, FP32 variant `GLADecodeFP32Kernel` L380), op
`src/tileops/ops/linear_attention/gla/recurrent.py`, manifest
`src/tileops/manifest/spec/linear_attention.yaml` (`GLARecurrentFwdOp` at :396),
interface `src/tileops/kernels/linear_attention/call_spec.py`
(`GLADecodeCall`, `GLADecodeFwdInterface`).

**Op key at baseline is `GLARecurrentFwdOp`; this task delivers the same interface under the name `GLADecodeOp`.**

Single-step (recurrent / decode) gated linear attention, one step per call:

```
state_new[dk, dv] = exp(gk[dk]) * state[dk, dv] + k[dk] * v[dv]     # decay + rank-1 outer-product
o[dv]             = scale * sum_dk q[dk] * state_new[dk, dv]        # contraction over DK
scale             = DK ** -0.5                                      # non-positive scale -> default
```

- `gk` is a **per-key-dimension** log-space gate `[B,H,DK]` (not per-head scalar).
- The GPU kernel rewrites `o = scale*(q*exp(gk))^T S + scale*(q·k) v` to avoid materialising
  `state_new` for the readout, but materialises `state_new` for the output; the two forms are
  algebraically identical. The probe materialises `state_new` and reads `o` directly off it
  (`o = scale * q · state_new`) — the definition given in the task and the manifest.
- **Layout BHD**: `q,k,gk: [B,H,DK]`, `v: [B,H,DV]`, `state: [B,H,DK,DV]`;
  outputs `o: [B,H,DV]`, `new_state: [B,H,DK,DV]`.
- **dtypes**: `float32 | float16 | bfloat16` inputs with **fp32 accumulation**; the fp32 path uses a
  dedicated no-`T.gemm` kernel to avoid TF32 mantissa truncation compounding over recurrent steps.
- **H (multi-head) batching**: every `(b,h)` pair is an **independent recurrence**; there is no
  cross-head interaction. Manifest workloads use `DK=DV=128`, `H in {8,16,32,48,64}`, `B in {1,8}`
  (all `bfloat16`).

## 2. Hard capabilities and the npuir compute unit that serves each

Every hard capability here is **Vector-unit (UB) elementwise/reduction work**; no Cube / `T.gemm`
is required to be *correct* (a Cube matvec is an optional performance alternative for the contraction).

| Hard capability | Probe op sequence | npuir unit | API doc |
|---|---|---|---|
| per-key elementwise gate `exp(gk)` | `T.vexp(g, g)` | Vector | `docs/Tilelang.language/数学操作/T.vexp.md` |
| broadcast `state * exp(gk)` (+ `q`,`k`,`v` broadcasts) | `T.vbrc` + `T.vmul` | Vector | `docs/Tilelang.language/shape操作/T.vbrc.md`, `docs/Tilelang.language/数学操作/T.vmul.md` |
| rank-1 outer-product accumulate `+ k⊗v` | `T.vbrc`(k),(v) + `T.vmul` + `T.vadd` | Vector | `docs/Tilelang.language/数学操作/T.vadd.md` |
| contraction `o = scale·q·state_new` over DK | `T.vbrc`(q) + `T.vmul` + `T.reduce(sum, dims=[0])` | Vector | `docs/Tilelang.language/规约操作/T.reduce.md` |
| per-`(b,h)` independent recurrence | `T.Kernel(B*H, is_npu=True)` | — (core grid) | — |
| `bf16` inputs (`bf16` has **no** Vector arithmetic) | `T.vcast` fp32 transit (in and out) | Vector | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` |
| zero-init accumulator | `T.clear(o2)` | Vector | `docs/Tilelang.language/内存操作/T.clear.md` |

Note on `bf16`: the API docs mark `bf16` as unsupported for `T.vexp` (`数学操作/T.vexp.md`, DataType
table `bf16 = ×`), `T.vmul` and `T.vadd` (same table). The probe therefore does all arithmetic in
`fp32` and only casts at the load/store boundary (`vcast round_mode="rint"`) — matching the GPU
contract of fp32 accumulation.

## 3. Probe design

`gla_decode_probe.py` provides two builders:

- `gla_decode_probe` — **untiled**: one core per `(b,h)`, whole state `[DK,DV]` in UB. Simple,
  representative for small `D`.
- `gla_decode_tiled_probe` — **K-tiled** (`k_tile`): streams the state in `[k_tile, DV]` blocks
  (exactly the GPU kernel's `dim_k // k_tile` loop), accumulates `o` across K-tiles. This is the
  shape that fits the manifest `D=128`.

Both compute the full recurrence; both are checked against `golden_gla_decode`, an independent
pure-torch CPU implementation (`new_state = state*exp(gk) + k⊗v`; `o = scale·einsum(q,new_state)`),
with per-case `max_abs_err` for `o` and `new_state`. Tolerances: fp32 `1e-4`, fp16 `2e-3`, bf16 `2e-2`.

## 4. Results (real device)

`python gla_decode_probe.py` → exit 0, `PROBE PASS`.

| B | H | DK | DV | dtype | variant | max_abs_err `o` | max_abs_err `new_state` | tol |
|---|---|---|---|---|---|---|---|---|
| 2 | 3 | 16 | 16 | float32 | untiled | 1.192e-07 | 2.384e-07 | 1e-4 |
| 1 | 2 | 32 | 16 | float32 | untiled | 1.192e-07 | 2.384e-07 | 1e-4 |
| 2 | 3 | 16 | 16 | float16 | untiled | 2.226e-04 | 9.766e-04 | 2e-3 |
| 1 | 2 | 32 | 16 | float16 | untiled | 1.603e-04 | 9.263e-04 | 2e-3 |
| 2 | 4 | 32 | 32 | bfloat16 | untiled | 1.930e-03 | 7.665e-03 | 2e-2 |
| 2 | 8 | 64 | 64 | float16 | untiled | 4.801e-04 | 9.713e-04 | 2e-3 |
| 1 | 4 | 128 | 128 | float16 | tiled k=16 | 4.508e-04 | 9.766e-04 | 2e-3 |
| 1 | 4 | 128 | 128 | float16 | tiled k=32 | 4.508e-04 | 9.766e-04 | 2e-3 |
| **8** | **32** | **128** | **128** | **bfloat16** | **tiled k=16** | **3.889e-03** | **1.513e-02** | **2e-2** |

Worst `max_abs_err` across all cases: **1.513e-02** (last row, bf16 — expected bf16 round-off).
The full manifest shape `B=8,H=32,DK=DV=128,bf16` passes.

### Probe stdout (tail)

```
===== case (B=8,H=32,DK=128,DV=128) bfloat16 k_tile=16 =====
AscendNPU IR compile success:
Workspace size: kernel declares no '*_infer_workspace_shape_function' callback, no workspace allocated ...
[PROBE] (B=8,H=32,DK=128,DV=128) dtype=bfloat16 tiled(k=16) max_abs_err o=3.889e-03 new_state=1.513e-02 (rtol=atol=2.0e-02)

worst max_abs_err across cases: 1.513e-02
PROBE PASS
```

## 5. Verdict: **FEASIBLE**

The full GLA decode recurrence runs on-device on Ascend910B3 / npuir and matches an independent
torch golden across fp32/fp16/bf16, including the full manifest shape `B=8,H=32,DK=DV=128,bf16`.
All hard capabilities (gate exp, modulated state decay, rank-1 outer-product accumulate, DK
contraction, multi-head batching) map to existing, documented **Vector-unit** APIs.

### One constraint found (not a capability gap)

- With the whole `[DK,DV]` state resident per core, the manifest `D=128` case **fails to compile**:
  `ub overflow, requires 1839104 bits while 1572864 bits available` (≈224.5 KB needed vs 192 KB UB).
- **Mitigation, proven**: K-tiling over `DK` (`k_tile=16/32`), which the GPU kernel already uses
  (`dim_k // k_tile`), brings the working set into UB and passes at `D=128` (table above).
- Therefore the constraint is a **tiling requirement**, not a missing npuir capability; no
  Frontend/TileLangIR/BiShengIR/tladapter/runtime gap was identified, so no capability-gap entry is warranted.

### Optional performance direction

For the `DK` contraction a Cube-side `T.gemm` (rank-1/ matvec via `l0_c`) is an alternative to the
Vector reduce if profiling shows the reduce is a bottleneck; not needed for correctness.

## 6. Recommended next step

Proceed to the full migration of `GLADecodeOp` (baseline `GLARecurrentFwdOp`) as a batch item:

1. Implement a K-tiled Vector kernel, one core per `(b,h)`, `k_tile` as a tunable (start 16/32),
   whole-state-in-UB fast path for small `D <= 64`.
2. Keep `float32` inputs on the pure-fp32 path; route `fp16/bf16` through an explicit fp32 transit
   with `vcast rint` (bf16 has no Vector arithmetic per `数学操作/T.vexp.md`).
3. Preserve the exact BHD interface and `scale = DK**-0.5` default from `call_spec.py`.
4. Validate against the manifest workloads (`B in {1,8}`, `H in {8,16,32,48,64}`, `DK=DV=128`) and
   benchmark; consider a Cube matvec for the contraction only if needed for the perf target.

## Files

- `gla_decode_probe.py` — probe (untiled + K-tiled builders, independent torch golden, runner).
- `probe_stdout.log` — captured stdout of the run (exit 0, `PROBE PASS`).
- `PROBE.md` — this report.
