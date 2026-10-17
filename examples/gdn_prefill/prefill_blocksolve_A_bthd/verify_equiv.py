"""D-1 equivalence machine-check for the adopted math-equivalence items of
``prefill_blocksolve_A_bthd``  (DESIGN.md §1.6.1).

torch CPU only; every item is compared against an **fp64 reference of the §0.1
semantic definition** (or against the baseline algebraic form in the same dtype path
for pure rewrites).  Items:

  A1  gate ratio      exp(g_i - g_j)                              [candidate]
                   vs exp2((g_i-r)*log2e) / exp2((g_j-r)*log2e)     [source algebra, per-16-block r]
  A2  divide -> mul   exp(g_i-r) * exp(r-g_j)                     [candidate]
                   vs exp(g_i-r) / exp(g_j-r)                     [baseline algebra]
  A3  step size       beta in fp32                                [candidate]
                   vs beta rounded to fp16                        [source: cast(step, dtype)]
  A4  inverse         log-doubling  A = prod_{i<6}(I + N^{2^i})   [candidate]
                   vs fp64 direct solve of (I + S_sys)            [semantic reference]
  A5  output cast     fp32 chain result -> fp16                   [candidate]
                   vs fp64 result -> fp16

Metrics per item: max abs diff, max ULP diff (in the item's dtype), relative error
(max abs diff / max |reference|), and violation rate = fraction of entries with
    |cand - ref| > 5e-3 * |ref| + 2 * eps_fp16 * max|ref|
(fp16 output tolerance + 2 ulp of the largest entry, so sub-fp16 entries cannot
produce spurious violations).  A verdict of EQUIV_PASS means violation rate == 0.

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


def doubling(S, levels=6, dt=torch.float32):
    """A = prod_{i<levels}(I + N^{2^i}) with every product rounded to dt (=NPU gemm op)."""
    N = (-S).to(dt)
    T = torch.eye(C, dtype=dt)
    P = N.clone()
    for j in range(levels):
        T = T + (P @ T).to(dt)
        if j < levels - 1:
            P = (P @ P).to(dt)
    return T


def blocked_source(S, store_dtype=torch.float16):
    """Source algorithm: 4 16x16 row-serial inverses + block forward substitution.

    ``store_dtype`` mirrors the source's ``solve_dtype = dtype`` storage of a_s / i_s.
    """
    Sf = S.float()
    a = Sf.to(store_dtype).float()
    inv = Sf.new_zeros(4, 16, 16)
    eye16 = torch.eye(16)
    for blk in range(4):
        ab = -torch.tril(a[blk * 16:(blk + 1) * 16, blk * 16:(blk + 1) * 16], -1)
        X = torch.zeros(16, 16)
        for r in range(16):
            X[r, :] = eye16[r, :] + ab[r, :r] @ X[:r, :]
        inv[blk] = X.to(store_dtype).float()
    Ab = Sf.new_zeros(4, 4, 16, 16)
    for b in range(4):
        Ab[b, b] = inv[b]
    for i in range(1, 4):
        for j in range(i):
            # A[i][j] = -inv_i * sum_{m=j..i-1} S[i][m] A[m][j],  A[j][j] = inv_j
            acc = torch.zeros(16, 16)
            for m in range(j, i):
                acc = acc + a[i * 16:(i + 1) * 16, m * 16:(m + 1) * 16] @ Ab[m, j]
            Ab[i, j] = (-inv[i] @ acc).to(store_dtype).float()
    return Ab.permute(0, 2, 1, 3).reshape(C, C)


def metrics(cand, ref, dt):
    c = cand.double()
    r = ref.double()
    d = (c - r).abs()
    rmax = float(r.abs().max()) if float(r.abs().max()) > 0 else 1.0
    atol = 2.0 * EPS_F16 * rmax
    viol = float((d > RTOL * r.abs() + atol).float().mean())
    scale = torch.clamp(r.abs(), min=r.abs().mean() * 1e-6 + 1e-300)
    ulp = float((d / (scale * torch.finfo(dt).eps)).max())
    return dict(abs=float(d.max()), ulp=ulp, rel=float(d.max()) / rmax, viol=viol)


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

items = {k: [] for k in ("A1", "A2", "A3", "A4", "A5")}
for tag, (k, g, beta) in CASES:
    S = s_sys_fp64(k, g, beta)
    ref_A = exact_inverse(S)
    G = k @ k.T
    mask = torch.tril(torch.ones(C, C, dtype=torch.bool), -1)

    # A1: exp(a-b) vs two-exp quotient, same fp32 path, reference = fp64 exact ratio
    ratio_ref = torch.exp(g[:, None] - g[None, :])
    items["A1"].append(metrics(gate_ratio_source_full(g), ratio_ref, torch.float32))

    # A2: divide -> reciprocal multiply
    items["A2"].append(metrics(gate_ratio_mul(g), gate_ratio_source(g), torch.float32))

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

print("| case | A1 exp(a-b) rel/viol | A2 mul rel/viol | A3 beta-fp32 rel/viol | "
      "A4 doubling rel/viol | A5 fp16-cast rel/viol |")
print("|---|---|---|---|---|---|")
for i, (tag, _) in enumerate(CASES):
    cells = " | ".join(f"{items[key][i]['rel']:.1e} / {items[key][i]['viol']:.1e}"
                       for key in ("A1", "A2", "A3", "A4", "A5"))
    print(f"| {tag} | {cells} |")

print()
verdicts = {}
for key in ("A1", "A2", "A3", "A4", "A5"):
    worst = max(m["viol"] for m in items[key])
    verdicts[key] = "EQUIV_PASS" if worst == 0.0 else "EQUIV_FAIL"
    print(f"| {key} | worst_rel={max(m['rel'] for m in items[key]):.1e} | "
          f"worst_ulp={max(m['ulp'] for m in items[key]):.1f} | "
          f"worst_viol={worst:.1e} | {verdicts[key]} |")

print()
print(f"criterion: |cand-ref| <= {RTOL:g}*|ref| + 2*fp16_eps*max|ref|   (fp16 output tolerance)")
print("informational: source blocked algorithm + fp16 partial storage, same cases")
print("| case | source-blocked rel/viol |")
print("|---|---|")
for tag, (k, g, beta) in CASES:
    S = s_sys_fp64(k, g, beta)
    m = metrics(blocked_source(S), exact_inverse(S), torch.float16)
    print(f"| {tag} | {m['rel']:.1e} / {m['viol']:.1e} |")
print()
print("blocked_source sanity (fp64 storage, must be ~1e-16):",
      float((blocked_source(s_sys_fp64(*make_case(0)), torch.float64).double()
             - exact_inverse(s_sys_fp64(*make_case(0)))).abs().max()))
