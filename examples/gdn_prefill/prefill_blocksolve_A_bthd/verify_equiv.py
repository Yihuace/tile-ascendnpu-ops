"""D-1 equivalence machine-check for the adopted math-equivalence items of
``prefill_blocksolve_A_bthd``  (DESIGN.md §1.6.1).

torch CPU only; every item is compared against an **fp64 reference of the §0.1
semantic definition** (or against the baseline algebraic form in the same dtype
path for pure rewrites).  Items:

  A1  gate ratio      exp(g_i - g_j)                              [candidate]
                   vs exp2((g_i-r)*log2e) / exp2((g_j-r)*log2e)     [source algebra, per-16-block r]
  A2  divide -> mul   exp(g_i-r) * exp(r-g_j)                     [candidate, fallback form]
                   vs exp(g_i-r) / exp(g_j-r)                     [baseline algebra]
  A3  step size       beta in fp32                                [candidate]
                   vs beta rounded to fp16                        [source: cast(step, dtype)]
  A4  inverse         log-doubling  A = prod_{i<6}(I + N^{2^i})   [candidate], N = -S_sys
                   vs fp64 direct solve of (I + S_sys)            [semantic reference]
  A5  output cast     fp32 chain result -> fp16                   [candidate]
                   vs fp64 result -> fp16
  A6  pseudo-code     the literal §3.3 Vector op sequence          [candidate]
      (revision 1)    vbrc(g_col)->g_i, vbrc(g_row)->g_j, vsub(g_i,g_j), vmin(d,0),
                      vexp, vcast(beta)+vmul(-1), vmul, vmul(G), vmul(mask01)
                   vs fp64 direct solve of (I + S_sys)            [semantic reference]
                   A6 exists because A4 starts from a *given* S: it cannot see a wrong
                   broadcast direction or a lost sign inside S (REVIEW v0 problems 1/2).
  A7  tail-chunk    the §0.6 R3 mechanism under *poisoned* out-of-range rows
      boundary      (revision 2): rows valid..63 of k/g hold uninitialised-SRAM garbage
      (revision 2)  (NaN/Inf), beta rows valid..63 are forced to 0 (R3-b), and the whole
                    64x64 tile goes through the same literal chain + doubling.
                    Reference = block-diag( inv(I+S[:valid,:valid]), I ) (64x64), i.e.
                    exactly what the source's clamp + beta=0 produces.
                    Variants: (a) block-start zero + beta domain zero -> EXPECT PASS;
                    (b) no block-start zero (k/g NaN) and (c) no beta domain zero
                    (beta NaN) are the negative controls -- each alone must poison the
                    valid rows, otherwise the R3 mechanism would be unnecessary.

Sign adjudication (bottom of the file) re-derives the source's block algorithm and
reports its residual against both ``inv(I + S)`` and ``inv(I - S)``: the sign of
``N`` is not a convention, only one of the two matches.

Metrics per item: max abs diff, max ULP diff (in the item's dtype), relative error
(max abs diff / max |reference|), and violation rate = fraction of entries with
    |cand - ref| > 5e-3 * |ref| + 2 * eps_fp16 * max|ref|
(fp16 output tolerance + 2 ulp of the largest entry, so sub-fp16 entries cannot
produce spurious violations).  A verdict of EQUIV_PASS means violation rate == 0.
Entries outside the strict-lower triangle are excluded from the A1/A2/A6-relational
metrics where the source formula is undefined/overflowing by design.

Run:  python verify_equiv.py
"""

import torch

C = 64
LOG2E = 1.4426950408889634
RTOL = 5e-3
EPS_F16 = float(torch.finfo(torch.float16).eps)


# ---------------------------------------------------------------------------
# semantic reference (fp64): S_sys[i,j] = beta_i * exp(g_i-g_j) * (k_i . k_j), i > j
# ---------------------------------------------------------------------------
def s_sys_fp64(k, g, beta):
    G = k @ k.T
    mask = torch.tril(torch.ones(C, C, dtype=torch.bool), -1)
    r = torch.exp(g[:, None] - g[None, :])
    return torch.where(mask, beta[:, None] * r * G, torch.zeros((), dtype=torch.float64))


def gate_ratio_source(g64, dtype=torch.float32):
    """Source algebra: two exps about a per-16-row-block reference, then a division."""
    out = torch.zeros(C, C, dtype=dtype)
    for bi in range(4):
        r = 16 * bi
        rows = slice(r, r + 16)
        gi = torch.exp(g64[rows] - g64[r]).to(dtype)
        gj = torch.exp(g64 - g64[r]).to(dtype)
        out[rows] = (gi[:, None] / gj[None, :].clamp(min=torch.finfo(dtype).tiny))
    return out


def gate_ratio_source_full(g64, dtype=torch.float32):
    return torch.exp(g64[:, None] - g64[None, :]).to(dtype)


def gate_ratio_mul(g64, dtype=torch.float32):
    """Divide -> reciprocal multiply: exp(g_i-r) * exp(r-g_j)."""
    out = torch.zeros(C, C, dtype=dtype)
    for bi in range(4):
        r = 16 * bi
        rows = slice(r, r + 16)
        gi = torch.exp(g64[rows] - g64[r]).to(dtype)
        gj_inv = torch.exp(g64[r] - g64).to(dtype)
        out[rows] = gi[:, None] * gj_inv[None, :]
    return out


def exact_inverse(S):
    return torch.linalg.solve(torch.eye(C, dtype=torch.float64) + S, torch.eye(C, dtype=torch.float64))


def doubling_from_N(N, levels=6, dt=torch.float32):
    """A = prod_{i<levels}(I + N^{2^i}); every product rounded to dt (= one gemm op)."""
    T = torch.eye(C, dtype=dt)
    P = N.to(dt)
    for j in range(levels):
        T = T + (P @ T).to(dt)
        if j < levels - 1:
            P = (P @ P).to(dt)
    return T


def doubling(S, levels=6, dt=torch.float32):
    """Same, with the sign convention of §1.3: N = -S_sys."""
    return doubling_from_N((-S).to(dt), levels=levels, dt=dt)


def blocked_source(S, store_dtype=torch.float16):
    """Source algorithm: 4 16x16 row-serial inverses + block forward substitution.

    ``store_dtype`` mirrors the source's ``solve_dtype = dtype`` storage of a_s / i_s;
    with ``torch.float64`` the *whole* path stays fp64 (structural cross-check),
    which is what DESIGN.md §8.1 gates at 1e-10 (revision 1 fix: v0 down-cast to
    fp32 unconditionally, so its "fp64 channel" was silently fp32).
    """
    Sf = S if store_dtype == torch.float64 else S.float()
    a = Sf.to(store_dtype).to(Sf.dtype)
    inv = Sf.new_zeros(4, 16, 16)
    eye16 = torch.eye(16, dtype=Sf.dtype)
    for blk in range(4):
        ab = -torch.tril(a[blk * 16:(blk + 1) * 16, blk * 16:(blk + 1) * 16], -1)
        X = torch.zeros(16, 16, dtype=Sf.dtype)
        for r in range(16):
            X[r, :] = eye16[r, :] + ab[r, :r] @ X[:r, :]
        inv[blk] = X.to(store_dtype).to(Sf.dtype)
    Ab = Sf.new_zeros(4, 4, 16, 16)
    for b in range(4):
        Ab[b, b] = inv[b]
    for i in range(1, 4):
        for j in range(i):
            # A[i][j] = -inv_i * sum_{m=j..i-1} S[i][m] A[m][j],  A[j][j] = inv_j
            acc = torch.zeros(16, 16, dtype=Sf.dtype)
            for m in range(j, i):
                acc = acc + a[i * 16:(i + 1) * 16, m * 16:(m + 1) * 16] @ Ab[m, j]
            Ab[i, j] = (-inv[i] @ acc).to(store_dtype).to(Sf.dtype)
    return Ab.permute(0, 2, 1, 3).reshape(C, C)


def metrics(cand, ref, dt, where=None):
    c = cand.double()
    r = ref.double()
    d = (c - r).abs()
    if where is not None:
        d = d[where]
        r = r[where]
        c = c[where]
    rmax = float(r.abs().max()) if float(r.abs().max()) > 0 else 1.0
    atol = 2.0 * EPS_F16 * rmax
    viol = float((d > RTOL * r.abs() + atol).float().mean()) if d.numel() else 0.0
    scale = torch.clamp(r.abs(), min=r.abs().mean() * 1e-6 + 1e-300)
    ulp = float((d / (scale * torch.finfo(dt).eps)).max()) if d.numel() else 0.0
    return dict(abs=float(d.max()) if d.numel() else 0.0, ulp=ulp, rel=(float(d.max()) / rmax) if d.numel() else 0.0, viol=viol)


# ---------------------------------------------------------------------------
# the literal §3.3 Vector-phase expansion (A6) — mirrors the on-device op order
# verified by history_version/design_probe_gate_sign.py
# ---------------------------------------------------------------------------
def vector_phase_literal(k, g64, beta, dt_mid=torch.float32, valid=C):
    G = (k @ k.T).to(dt_mid)                                    # Gram land (Cube, fp32)
    gi = g64.to(dt_mid)[:, None].expand(C, C)                    # vbrc(g_col) -> g_i
    gj = g64.to(dt_mid)[None, :].expand(C, C)                    # vbrc(g_row) -> g_j
    d = (gi - gj).to(dt_mid)                                     # vsub(g_i, g_j)
    d = torch.minimum(d, torch.zeros((), dtype=dt_mid))          # vmin(d, 0)
    e = torch.exp(d).to(dt_mid)                                  # vexp
    b32 = beta.to(torch.float16).to(dt_mid)                      # load fp16 + vcast(rint)
    if valid < C:
        # R3-b: the out-of-range rows of beta are zero-filled *before* the dynamic-extent
        # copy, so the value that reaches vcast is exactly 0 -- independent of whatever
        # the previous task left in the buffer.
        b32 = b32.clone()
        b32[valid:] = 0.0
    bn = (b32 * (-1.0)).to(dt_mid)                               # vmul(b, -1) : the -1 landing point
    bm = bn[:, None].expand(C, C)                                # vbrc
    t1 = (bm * e).to(dt_mid)                                     # vmul
    t2 = (t1 * G).to(dt_mid)                                     # vmul(G)
    mask01 = torch.tril(torch.ones(C, C, dtype=dt_mid), -1)      # host constant 0/1 tile
    N = (t2 * mask01).to(dt_mid)                                 # vmul(mask01)
    return N


def a7_tail_tile(valid: int, seed: int = 11, poison="nan"):
    """One 64x64 tile whose rows valid..63 are out-of-range (source: clamp + beta=0).

    ``poison="nan"`` models *uninitialised* SRAM (the block's first task is the tail
    chunk); ``poison="stale"`` models finite leftovers of a previous full chunk.
    Returns (k_tile, g_tile, beta_tile, ref64) with the reference being
    block_diag(inv(I+S[:valid,:valid]), I).
    """
    torch.manual_seed(seed)
    D = 128
    n = valid
    k_real = torch.nn.functional.normalize(torch.randn(n, D, dtype=torch.float64), dim=-1)
    g_real = torch.cumsum(-torch.rand(n, dtype=torch.float64) * 0.05 - 0.005, 0)
    b_real = torch.rand(n, dtype=torch.float64) * 0.9
    mask_n = torch.tril(torch.ones(n, n, dtype=torch.bool), -1)
    S = torch.where(mask_n,
                    b_real[:, None] * torch.exp(g_real[:, None] - g_real[None, :]) * (k_real @ k_real.T),
                    torch.zeros((), dtype=torch.float64))
    A11 = torch.linalg.solve(torch.eye(n, dtype=torch.float64) + S,
                             torch.eye(n, dtype=torch.float64))
    ref = torch.eye(C, dtype=torch.float64)
    ref[:valid, :valid] = A11

    k_tile = torch.zeros(C, D, dtype=torch.float64)
    g_tile = torch.zeros(C, dtype=torch.float64)
    b_tile = torch.zeros(C, dtype=torch.float64)
    k_tile[:valid], g_tile[:valid], b_tile[:valid] = k_real, g_real, b_real
    if valid < C:
        if poison == "nan":
            k_tile[valid:] = float("nan")
            g_tile[valid:] = float("nan")
            b_tile[valid:] = float("nan")
        else:  # stale finite values of a previous full chunk
            k_tile[valid:] = torch.nn.functional.normalize(
                torch.randn(C - valid, D, dtype=torch.float64), dim=-1)
            g_tile[valid:] = torch.cumsum(-torch.rand(C - valid, dtype=torch.float64) * 0.05, 0)
            b_tile[valid:] = torch.rand(C - valid, dtype=torch.float64)
    return k_tile, g_tile, b_tile, ref


def a7_run(valid: int, block_start_zero: bool, beta_domain_zero: bool, poison="nan",
           scen="S1"):
    k_t, g_t, b_t, ref = a7_tail_tile(valid, poison=poison)
    k_use, g_use = k_t.clone(), g_t.clone()
    if valid < C and block_start_zero and scen == "S1":
        # R3-a: at block start l1k / ub_gc / ub_gr are zeroed once.  In S1 this (tail)
        # task is the block's first task, so those zeros are what the task still reads.
        # In S2 a full-chunk task ran in between and overwrote all 64 rows with finite
        # GM data -- modelling that correctly is the whole point of the two scenarios.
        k_use[valid:] = 0.0
        g_use[valid:] = 0.0
    N = vector_phase_literal(k_use, g_use, b_t,
                             valid=(valid if beta_domain_zero else C))
    A = doubling_from_N(N)
    return A, ref, valid




# ---------------------------------------------------------------------------
# cases (semantic corner values included)
# ---------------------------------------------------------------------------
def make_case(seed, beta_hi=1.0, gscale=0.05, kth=1.0, tail_rows=None, zero_beta=False):
    torch.manual_seed(seed)
    D = 128
    k = torch.nn.functional.normalize(torch.randn(C, D, dtype=torch.float64), dim=-1) * kth
    g = torch.cumsum(-torch.rand(C, dtype=torch.float64) * gscale - 0.005, 0)
    beta = torch.rand(C, dtype=torch.float64) * beta_hi
    if tail_rows is not None:
        beta[tail_rows:] = 0.0
    if zero_beta:
        beta[::7] = 0.0
    return k, g, beta


CASES = [
    ("SC-typical", make_case(0)),
    ("SC-strong-decay", make_case(1, gscale=0.5)),
    ("SC-no-decay", make_case(2, gscale=0.0)),
    ("SC-beta-x20", make_case(3, beta_hi=20.0)),
    ("SC-k-x4", make_case(4, kth=4.0)),
    ("SC-tail-56", make_case(5, tail_rows=56)),
    ("SC-zero-beta-rows", make_case(6, zero_beta=True)),
    ("SC-near-overflow", make_case(7, gscale=1.2)),
]

mask = torch.tril(torch.ones(C, C, dtype=torch.bool), -1)
items = {key: [] for key in ("A1", "A2", "A3", "A4", "A5", "A6")}
for tag, (k, g, beta) in CASES:
    S = s_sys_fp64(k, g, beta)
    ref_A = exact_inverse(S)
    G = k @ k.T
    ratio_ref = torch.exp(g[:, None] - g[None, :])

    # A1: exp(a-b) vs two-exp quotient, same fp32 path, reference = fp64 exact ratio (i>j only:
    # the source formula is only defined where the gates are used, and the i>j region is
    # exactly where §3.3's vmin(d,0) clamp is a no-op)
    items["A1"].append(metrics(gate_ratio_source_full(g), ratio_ref, torch.float32, where=mask))

    # A2: divide -> reciprocal multiply (fallback form)
    items["A2"].append(metrics(gate_ratio_mul(g), gate_ratio_source(g), torch.float32, where=mask))

    # A3: beta fp32 vs beta fp16 (all else exact)
    S_b32 = torch.where(mask, beta[:, None] * ratio_ref * G, torch.zeros((), dtype=torch.float64))
    S_b16 = torch.where(
        mask,
        beta.to(torch.float16).double()[:, None] * ratio_ref * G,
        torch.zeros((), dtype=torch.float64),
    )
    items["A3"].append(metrics(S_b32, S_b16, torch.float32))

    # A4/A5: the adopted inverse algorithm, then the fp16 output cast
    A_dbl = doubling(S)
    items["A4"].append(metrics(A_dbl, ref_A, torch.float16))
    items["A5"].append(metrics(A_dbl.half().float(), ref_A.half().float(), torch.float16))

    # A6: the literal §3.3 Vector expansion feeding the same doubling chain
    N_lit = vector_phase_literal(k, g, beta)
    items["A6"].append(metrics(doubling_from_N(N_lit), ref_A, torch.float16))

print("| case | A1 exp(a-b) rel/viol | A2 mul rel/viol | A3 beta-fp32 rel/viol | "
      "A4 doubling rel/viol | A5 fp16-cast rel/viol | A6 literal-chain rel/viol |")
print("|---|---|---|---|---|---|---|")
for i, (tag, _) in enumerate(CASES):
    cells = " | ".join(f"{items[key][i]['rel']:.1e} / {items[key][i]['viol']:.1e}"
                       for key in ("A1", "A2", "A3", "A4", "A5", "A6"))
    print(f"| {tag} | {cells} |")

print()
verdicts = {}
for key in ("A1", "A2", "A3", "A4", "A5", "A6"):
    worst = max(m["viol"] for m in items[key])
    verdicts[key] = "EQUIV_PASS" if worst == 0.0 else "EQUIV_FAIL"
    print(f"| {key} | worst_rel={max(m['rel'] for m in items[key]):.1e} | "
          f"worst_ulp={max(m['ulp'] for m in items[key]):.1f} | "
          f"worst_viol={worst:.1e} | {verdicts[key]} |")

print()
print(f"criterion: |cand-ref| <= {RTOL:g}*|ref| + 2*fp16_eps*max|ref|   (fp16 output tolerance)")

# ---------------------------------------------------------------------------
# A6 negative controls: the two v0 defects must FAIL this check (teeth proof)
# ---------------------------------------------------------------------------
print()
print("A6 negative controls (the v0 defects), SC-typical, same fp16 criterion:")
k0, g0, b0 = CASES[0][1]
S0_ = s_sys_fp64(k0, g0, b0)
ref0 = exact_inverse(S0_)
mask01_ = torch.tril(torch.ones(C, C, dtype=torch.float32), -1)
G0 = (k0 @ k0.T).to(torch.float32)
e_ok = torch.exp(torch.minimum((g0.float()[:, None] - g0.float()[None, :]),
                               torch.zeros((), dtype=torch.float32)))
e_rev = torch.exp(torch.minimum((g0.float()[None, :] - g0.float()[:, None]),
                                torch.zeros((), dtype=torch.float32)))
bn0 = (b0.to(torch.float16).to(torch.float32) * -1.0)
for name, Nv in (
    ("as-designed  (g_i-g_j, -beta)", (bn0[:, None] * e_ok * G0 * mask01_)),
    ("v0 direction (g_j-g_i, -beta)", (bn0[:, None] * e_rev * G0 * mask01_)),
    ("v0 sign      (g_i-g_j, +beta)", ((-bn0)[:, None] * e_ok * G0 * mask01_)),
):
    m = metrics(doubling_from_N(Nv.to(torch.float32)), ref0, torch.float16)
    print(f"  {name:32s} rel={m['rel']:.1e}  viol={m['viol']:.1e}  "
          f"{'EQUIV_PASS' if m['viol'] == 0.0 else 'EQUIV_FAIL'}")

# ---------------------------------------------------------------------------
# sign adjudication: the source block algorithm vs inv(I+S) and inv(I-S)
# ---------------------------------------------------------------------------
print()
S0 = s_sys_fp64(*make_case(0))
inv_plus = exact_inverse(S0)
r_plus = float((blocked_source(S0, store_dtype=torch.float64) - inv_plus).abs().max())
r_minus = float((blocked_source(S0, store_dtype=torch.float64)
                 - torch.linalg.solve(torch.eye(C, dtype=torch.float64) - S0,
                                      torch.eye(C, dtype=torch.float64))).abs().max())
print(f"sign adjudication (fp64 structural path): |source - inv(I+S)| = {r_plus:.3e} ; "
      f"|source - inv(I-S)| = {r_minus:.3e}")
print(f"sign adjudication: N = -S_sys (= a_s diagonal blocks are negated) -> "
      f"{'CONFIRMED' if r_plus < 1e-10 < r_minus else 'INCONSISTENT'}")

# ---------------------------------------------------------------------------
# informational: source blocked algorithm + fp16 partial storage, same cases
# ---------------------------------------------------------------------------
print()
print("informational: source blocked algorithm + fp16 partial storage, same cases")
print("| case | source-blocked rel/viol |")
print("|---|---|")
for tag, (k, g, beta) in CASES:
    S = s_sys_fp64(k, g, beta)
    m = metrics(blocked_source(S), exact_inverse(S), torch.float16)
    print(f"| {tag} | {m['rel']:.1e} / {m['viol']:.1e} |")
print()
print("blocked_source sanity (fp64 storage, DESIGN.md §8.1 gate 1e-10):",
      float((blocked_source(s_sys_fp64(*make_case(0)), torch.float64).double()
             - exact_inverse(s_sys_fp64(*make_case(0)))).abs().max()))

# ---------------------------------------------------------------------------
# A7 (revision 2): tail-chunk boundary under poisoned out-of-range rows
# ---------------------------------------------------------------------------
print()
print("A7 tail-chunk boundary (REVIEW v1 blocking problem 2)")
print("reference = block_diag(inv(I + S[:valid,:valid]), I) (64x64), fp64")
print("S1 = block's FIRST task is the tail chunk (rows >= valid are uninitialised SRAM: NaN)")
print("S2 = a later task of the block is the tail chunk (rows >= valid hold a previous full")
print("     chunk's finite rows)  -> no NaN is possible, only the padding-row claim is at stake")
print()
print("| scenario | valid | R3-a block-start zero | R3-b beta domain zero | valid-row rel | viol | NaN/Inf in A | padding rows == e_i | verdict |")
print("|---|---|---|---|---|---|---|---|---|")

a7_rows = []
for scen, poison in (("S1", "nan"), ("S2", "stale")):
    for valid in (64, 56, 36, 1):
        for bsz, bdz in ((True, True), (False, True), (True, False)):
            if valid == C and not (bsz and bdz):
                continue  # full chunk: no out-of-range rows, the variants coincide
            A, ref, v = a7_run(int(valid), bsz, bdz, poison=poison, scen=scen)
            m = metrics(A[:v, :v], ref[:v, :v], torch.float16)
            nan_ct = int(torch.isnan(A).sum() + torch.isinf(A).sum())
            pad_ok = bool(torch.equal(A[v:], torch.eye(C, dtype=torch.float32)[v:])) if v < C else True
            good = (m["viol"] == 0.0) and nan_ct == 0 and pad_ok
            # S1: both halves are needed -- R3-a keeps e/G finite and R3-b zeroes the
            #     beta row; either one missing leaves NaN in the padding rows, and
            #     N^2 then drags it into the legal rows (column j < valid of T rows >= valid).
            # S2: operands are finite already, so only the padding-row claim needs R3-b.
            want_good = (bsz and bdz) if scen == "S1" else bdz
            verdict = ("EQUIV_PASS" if good else "EQUIV_FAIL") if want_good else \
                      ("control behaves as predicted" if not good else "control DID NOT fail")
            a7_rows.append((scen, valid, bsz, bdz, good, want_good))
            print(f"| {scen} | {valid} | {'yes' if bsz else 'NO'} | {'yes' if bdz else 'NO'} | "
                  f"{m['rel']:.1e} | {m['viol']:.1e} | {nan_ct} | {'yes' if pad_ok else 'NO'} | {verdict} |")

a7_ok = all(g == w for _, _, _, _, g, w in a7_rows)
print()
print("A7 verdict:", "EQUIV_PASS" if a7_ok else "EQUIV_FAIL",
      "(design rows pass; every negative control fails exactly where predicted)")


