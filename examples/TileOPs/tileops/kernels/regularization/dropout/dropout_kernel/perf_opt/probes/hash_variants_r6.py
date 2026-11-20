#!/usr/bin/env python3
"""R6: statistical quality of candidate shorter mask/threshold forms (host model).

The kernel hash is a pure int32 function of (seed, element index); the numpy
int32 model below is bit-exact with the device chain (same op order, same
wrapping constants).  We compare:

  V0  current  : h = ((c*A) ^ ((c*A)<<13)) * B ; keep iff |h| >= 256*tp
                 (kernel: vabs + vsub + vclamp)
  V1  signed   : same hash ; keep iff h >= 0    (kernel: vclamp(h,0,1) only)
  V2  3-op mul+shl+xor : h = (c*A); h ^= h<<13 ; keep iff h >= 0
  V3  3-op shl+xor+mul : h = c ^ (c<<13); h *= B ; keep iff h >= 0
  V4  V1 with 2-bit test window (sanity: |h| < 2^31 form)

Quality gates (same as R3 `_hash_analysis.py`):
  * drop rate within 3 sigma of p over 4 Mi elements
  * block-chi2 over 4096 blocks of 1024 elements (dof 4095)
  * lag-1..4 mask autocorrelation, and lag-2048 block rate spread
"""
import numpy as np

A = np.int32(-1640531527)
B = np.int32(-2048144777)
C31 = np.int32(0x7FFFFFFF)


def hash_v0(c):
    x = (c * A).astype(np.int32)
    return ((x ^ (x << np.int32(13))) * B).astype(np.int32)


def hash_v2(c):
    x = (c * A).astype(np.int32)
    return (x ^ (x << np.int32(13))).astype(np.int32)


def hash_v3(c):
    x = (c ^ (c << np.int32(13))).astype(np.int32)
    return (x * B).astype(np.int32)


def mask_v0(h, tp_shift):
    return (np.abs(h.astype(np.int64)) >= tp_shift).astype(np.int8)


def mask_signed(h):
    return (h >= 0).astype(np.int8)


def report(name, m, p, nblock=1024):
    n = m.size
    rate = m.mean()
    sig = 3.0 * (p * (1 - p) / n) ** 0.5
    rate_ok = abs(rate - p) <= sig
    nb = n // nblock
    blocks = m[: nb * nblock].reshape(nb, nblock).sum(axis=1)
    exp = nblock * p
    chi2 = float(((blocks - exp) ** 2).sum() / (exp * (1 - p)))
    # autocorrelation of the 0/1 sequence at small lags
    mu = m.mean()
    ac = []
    for lag in (1, 2, 3, 4):
        v = float(((m[:-lag] - mu) * (m[lag:] - mu)).mean() / (mu * (1 - mu)))
        ac.append(round(v, 5))
    # block-rate spread over 32 groups of 2048 blocks (~2 Mi per group)
    g = blocks[: (len(blocks) // 32) * 32].reshape(32, -1).mean(axis=1)
    print(f"{name:28s} rate={rate:.6f} dsigma={abs(rate-p)/sig*3:5.2f} "
          f"rate_ok={rate_ok} chi2={chi2:8.1f} autocorr={ac} "
          f"grp_spread={g.std():.5f}")


def main():
    n = 1 << 22
    seed = 42
    for p in (0.5, 0.3, 0.1, 0.7):
        tp = int(round(p * float(1 << 23)))
        tp_shift = 256 * tp - 1
        c = (np.arange(n, dtype=np.int64) + seed).astype(np.int32)
        h0 = hash_v0(c)
        print(f"--- p={p}  tp_shift={tp_shift} ---")
        report("V0 |h|>=256tp (current)", mask_v0(h0, tp_shift), p)
        if p == 0.5:
            report("V1 h>=0 (signed)", mask_signed(h0), p)
            report("V2 3op mul-shl-xor", mask_signed(hash_v2(c)), p)
            report("V3 3op shl-xor-mul", mask_signed(hash_v3(c)), p)
            # bit-31 balance over short windows (queue / lattice artifacts)
            bit = (h0 < 0).astype(np.int8)
            for w in (64, 1024, 8192):
                r = bit[: (n // w) * w].reshape(-1, w).mean(axis=1)
                print(f"    bit31 window={w:5d}: mean={r.mean():.5f} "
                      f"std={r.std():.5f} max={r.max():.4f} min={r.min():.4f}")
            # repeat check: does the mask repeat with period 8192/65536?
            m1 = mask_signed(h0)
            for per in (8192, 65536, 1 << 20):
                eq = float((m1[:-per] == m1[per:]).mean())
                print(f"    period {per:8d}: equality={eq:.5f} (0.5 = independent)")
            # seed sensitivity
            c2 = (np.arange(n, dtype=np.int64) + 99).astype(np.int32)
            print("    seed separation (42 vs 99): "
                  f"{float((m1 != mask_signed(hash_v0(c2))).mean()):.5f}")


if __name__ == "__main__":
    main()
