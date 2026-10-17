"""PL-1.22-small-shape-fixed-overhead -- delta repro (ED-B, model form).

Entry: pattern-library/elementwise.md PL-1.22-small-shape-fixed-overhead
First verified: 2026-10-17, tilelang 0.1.2 dev-root (AscendNPU-IR 31f6903) +
CANN 9.0.0 (bishengir 1.1.0) / Ascend910B3.

40-AIV grid-stride tile imbalance vs block_size.

Model (asserted): tiles = ceil(N/bs); per-core = ceil(tiles/40);
critical path = per_core * bs; overhead = path/(N/40) - 1.

Numbers are the ones quoted in pattern-library/elementwise.md §1.22
(AddFwdOp, N=4,194,304, Ascend910B3 40 AIV, CANN 9.0.0).  Pure Python, no NPU.

Measured, same toolchain (msprof op Task Duration, adjacent pairs):
  bs=8192  -> 512 tiles -> 13/12 -> +1.6% predicted; 13.6-14.2us actual
  bs=12288 -> 342 tiles ->  9/8  -> +7.5% predicted; +5~7% actual (rejected)
  bs=16384 -> 256 tiles ->  7/6  -> +9.4% predicted (rejected, UB also tight)
Counter-example: element-balanced contiguous chunks remove the 1.6% imbalance
but measured -3% (grid-stride HBM locality wins; PL-1.6).
"""

N = 4_194_304
CORES = 40


def model(bs: int):
    tiles = -(-N // bs)
    per_core = -(-tiles // CORES)
    critical = per_core * bs
    avg = N / CORES
    return tiles, per_core, (critical - avg) / avg * 100.0


CASES = [
    (8192, 512, 13, 1.6),
    (12288, 342, 9, 5.5),
    (16384, 256, 7, 9.4),
]

for bs, exp_tiles, exp_per_core, exp_overhead in CASES:
    tiles, per_core, overhead = model(bs)
    assert tiles == exp_tiles, (bs, tiles, exp_tiles)
    assert per_core == exp_per_core, (bs, per_core, exp_per_core)
    assert abs(overhead - exp_overhead) < 0.1, (bs, overhead, exp_overhead)
    print(f"bs={bs:5d} tiles={tiles:3d} per_core={per_core:2d} "
          f"critical_path_overhead={overhead:+.1f}%")

# Fixed-intercept fit quoted in the entry (two measured points, bf16):
#   t(4.19M)=13.7us, t(8.39M)=25.8us -> t = 1.6us + N*2.886ns/1k
slope_us_per_m = (25.8 - 13.7) / (8.39 - 4.19)
intercept = 13.7 - slope_us_per_m * 4.19
assert abs(slope_us_per_m - 2.88) < 0.05
assert abs(intercept - 1.63) < 0.1
print(f"fixed intercept={intercept:.2f}us slope={slope_us_per_m:.2f}us/1M")

# Target check: 70% SOL at N=4.19M needs ~13.70us => ~0.47us above the slope-only point.
print("70% target implies shaving ~0.5us of per-launch/tile overhead")
