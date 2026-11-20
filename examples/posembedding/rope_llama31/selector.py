from dataclasses import dataclass, replace
from typing import Tuple


_CORE_COUNT = 48
_SOURCE_UB_BUDGET = 160 * 1024
_PLAN_UB_BUDGET = 190 * 1024
_BLOCK_ELEMS = {"float16": 16, "bfloat16": 16, "float32": 8}
_LOW_PRECISION_1D_UB = {False: (22, 8), True: (34, 8)}


def _ceil_div(value: int, factor: int) -> int:
    return (value + factor - 1) // factor


def _align_up(value: int, factor: int) -> int:
    return _ceil_div(value, factor) * factor


def _divisors(value: int) -> Tuple[int, ...]:
    return tuple(
        candidate for candidate in range(1, value + 1) if value % candidate == 0
    )


def _ub_bytes(
    rows: int,
    block_heads: int,
    dim: int,
    dtype: str,
    auto_multibuffer: bool,
    compiler_reuse_1d: bool = False,
    table_resident: bool = False,
) -> int:
    half_pad = _align_up(dim // 2, _BLOCK_ELEMS[dtype])
    if dtype != "float32":
        work = rows * block_heads * half_pad
        table = rows * half_pad
        if compiler_reuse_1d:
            work_bytes, table_bytes = _LOW_PRECISION_1D_UB[auto_multibuffer]
            return work_bytes * work + table_bytes * table
        if auto_multibuffer:
            return 40 * work + (16 if table_resident else 8) * table
        return 24 * work + 12 * table
    work = rows * block_heads * half_pad * 4
    table = rows * half_pad * 4
    return 8 * work + 4 * table if auto_multibuffer else 4 * work + 2 * table


def _fits_legacy(
    rows: int,
    block_heads: int,
    dim: int,
    dtype: str,
    work_blocks: int,
    compiler_reuse_1d: bool = False,
) -> bool:
    args = (rows, block_heads, dim, dtype)
    return (
        _ub_bytes(*args, False, compiler_reuse_1d) <= _SOURCE_UB_BUDGET
        and _ub_bytes(*args, work_blocks > _CORE_COUNT, compiler_reuse_1d)
        <= _PLAN_UB_BUDGET
    )


def _select_padded_1d(seq_len: int, dim: int, dtype: str) -> int:
    rows_per_core = _ceil_div(seq_len, _CORE_COUNT)
    if _fits_legacy(
        rows_per_core,
        1,
        dim,
        dtype,
        _ceil_div(seq_len, rows_per_core),
        True,
    ):
        return rows_per_core
    max_rows = max(
        rows
        for rows in range(1, rows_per_core + 1)
        if _fits_legacy(
            rows,
            1,
            dim,
            dtype,
            _ceil_div(seq_len, rows),
            True,
        )
    )
    iterations = _ceil_div(rows_per_core, max_rows)
    rows = _ceil_div(rows_per_core, iterations)
    while not _fits_legacy(rows, 1, dim, dtype, _ceil_div(seq_len, rows), True):
        rows -= 1
    return rows


def _select_legacy_tile(
    tokens: int,
    seq_len: int,
    heads: int,
    dim: int,
    dtype: str,
    compiler_reuse_1d: bool = False,
) -> Tuple[int, int]:
    half_pad = _align_up(dim // 2, _BLOCK_ELEMS[dtype])
    if (
        compiler_reuse_1d
        and seq_len >= _CORE_COUNT
        and dim // 2 % _BLOCK_ELEMS[dtype]
        and (dtype == "float32" or half_pad < 64)
    ):
        return _select_padded_1d(seq_len, dim, dtype), 1

    lanes = max(
        divisor
        for divisor in range(1, min(seq_len, _CORE_COUNT) + 1)
        if seq_len % divisor == 0
    )
    token_tiles = tuple(
        divisor for divisor in range(seq_len // lanes, 0, -1) if seq_len % divisor == 0
    )
    candidates = []
    legacy_candidate = None
    for block_heads in reversed(_divisors(heads)):
        for rows in token_tiles:
            work_blocks = tokens // rows * (heads // block_heads)
            if not _fits_legacy(
                rows,
                block_heads,
                dim,
                dtype,
                work_blocks,
                compiler_reuse_1d,
            ):
                continue
            if not compiler_reuse_1d:
                return rows, block_heads
            waves = _ceil_div(work_blocks, _CORE_COUNT)
            last_wave = work_blocks - (waves - 1) * _CORE_COUNT
            legacy_safe = _fits_legacy(rows, block_heads, dim, dtype, work_blocks)
            candidate = (waves, -last_wave, -rows, rows, block_heads)
            if legacy_safe and legacy_candidate is None:
                legacy_candidate = candidate
            tile_work = rows * block_heads * half_pad
            if not legacy_safe and (
                waves == 1
                or tile_work <= 4096
                and last_wave * half_pad >= 2048
                and (half_pad < 256 or work_blocks >= 96)
            ):
                candidates.append(candidate)
    if legacy_candidate is not None:
        candidates.append(legacy_candidate)
    if not candidates:
        raise ValueError(
            f"no UB-safe tile for S={seq_len}, N={heads}, D={dim}, dtype={dtype}"
        )
    _, _, _, rows, block_heads = min(candidates)
    return rows, block_heads


@dataclass(frozen=True)
class TilePlan:
    route: str
    block_tokens: int
    block_heads: int
    head_blocks: int
    bn_groups: int
    max_units: int
    work_blocks: int
    source_ub_bytes: int
    plan_ub_bytes: int
    merge_copyout: bool = False


def _legacy_plan(
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
    dtype: str,
    compiler_reuse_1d: bool,
) -> TilePlan:
    tokens = batch * seq_len
    rows, block_heads = _select_legacy_tile(
        tokens, seq_len, heads, dim, dtype, compiler_reuse_1d
    )
    head_blocks = _ceil_div(heads, block_heads)
    work_blocks = _ceil_div(tokens, rows) * head_blocks
    return TilePlan(
        "blocked",
        rows,
        block_heads,
        head_blocks,
        0,
        0,
        work_blocks,
        _ub_bytes(rows, block_heads, dim, dtype, False, compiler_reuse_1d),
        _ub_bytes(
            rows,
            block_heads,
            dim,
            dtype,
            work_blocks > _CORE_COUNT,
            compiler_reuse_1d,
        ),
    )


def _core_loads(
    batch: int,
    seq_len: int,
    heads: int,
    half_pad: int,
    rows: int,
    block_heads: int,
):
    seq_blocks = _ceil_div(seq_len, rows)
    head_blocks = heads // block_heads
    work_blocks = batch * head_blocks * seq_blocks
    programs = min(work_blocks, _CORE_COUNT)
    core_work = [0] * programs
    core_table = [0] * programs
    core_tasks = [0] * programs
    tail_rows = seq_len - (seq_blocks - 1) * rows
    for work in range(work_blocks):
        task_rows = tail_rows if work % seq_blocks == seq_blocks - 1 else rows
        core = work % programs
        core_work[core] += task_rows * block_heads * half_pad
        core_table[core] += task_rows * half_pad
        core_tasks[core] += 1
    return (
        seq_blocks,
        head_blocks,
        work_blocks,
        max(core_work),
        max(core_table),
        max(core_tasks),
    )


def _select_arbitrary(
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
    dtype: str,
) -> TilePlan:
    half_pad = _align_up(dim // 2, _BLOCK_ELEMS[dtype])
    candidates = []
    for block_heads in reversed(_divisors(heads)):
        for rows in range(1, seq_len + 1):
            source_ub = _ub_bytes(rows, block_heads, dim, dtype, False)
            if source_ub > _SOURCE_UB_BUDGET:
                break
            (
                seq_blocks,
                head_blocks,
                work_blocks,
                max_core_work,
                max_core_table,
                max_core_tasks,
            ) = _core_loads(batch, seq_len, heads, half_pad, rows, block_heads)
            plan_ub = _ub_bytes(
                rows,
                block_heads,
                dim,
                dtype,
                work_blocks >= _CORE_COUNT,
            )
            if plan_ub > _PLAN_UB_BUDGET:
                continue
            score = (
                max_core_work + max_core_table + max_core_tasks * half_pad * 4,
                max_core_work,
                max_core_table,
                max_core_tasks,
                work_blocks,
                batch * head_blocks,
                -block_heads,
            )
            candidates.append(
                (
                    score,
                    TilePlan(
                        "arbitrary",
                        rows,
                        block_heads,
                        head_blocks,
                        0,
                        0,
                        work_blocks,
                        source_ub,
                        plan_ub,
                    ),
                )
            )
    if not candidates:
        raise ValueError(
            f"no arbitrary-S tile for B={batch}, S={seq_len}, N={heads}, D={dim}, dtype={dtype}"
        )
    return min(candidates, key=lambda item: item[0])[1]


def _select_table(
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
    dtype: str,
) -> TilePlan:
    half_pad = _align_up(dim // 2, _BLOCK_ELEMS[dtype])
    bn_units = batch * heads
    candidates = []
    for bn_groups in range(1, min(bn_units, _CORE_COUNT) + 1):
        max_units = _ceil_div(bn_units, bn_groups)
        for rows in range(1, seq_len + 1):
            source_ub = _ub_bytes(rows, 1, dim, dtype, False, table_resident=True)
            if source_ub > _SOURCE_UB_BUDGET:
                break
            seq_blocks = _ceil_div(seq_len, rows)
            work_blocks = seq_blocks * bn_groups
            plan_ub = _ub_bytes(
                rows,
                1,
                dim,
                dtype,
                work_blocks >= _CORE_COUNT,
                table_resident=True,
            )
            if plan_ub > _PLAN_UB_BUDGET:
                continue
            waves = _ceil_div(work_blocks, _CORE_COUNT)
            max_core_work = waves * rows * max_units * half_pad
            max_core_table = waves * rows * half_pad
            score = (
                max_core_work + max_core_table + waves * half_pad * 4,
                bn_groups * seq_len * half_pad,
                work_blocks,
                -rows,
            )
            candidates.append(
                (
                    score,
                    TilePlan(
                        "table",
                        rows,
                        1,
                        0,
                        bn_groups,
                        max_units,
                        work_blocks,
                        source_ub,
                        plan_ub,
                    ),
                )
            )
    if not candidates:
        raise ValueError(
            f"no table tile for B={batch}, S={seq_len}, N={heads}, D={dim}, dtype={dtype}"
        )
    return min(candidates, key=lambda item: item[0])[1]


def _with_copyout(plan: TilePlan, heads: int, dim: int, dtype: str) -> TilePlan:
    if (
        plan.route == "table"
        or heads == 1
        or plan.block_heads != heads
        or (dtype == "float32" and dim > 128)
        or dim % (16 if dtype == "float32" else 32)
    ):
        return plan
    copyout_ub = (
        plan.block_tokens * plan.block_heads * dim * (4 if dtype == "float32" else 2)
    )
    source_ub = plan.source_ub_bytes + copyout_ub
    plan_ub = plan.plan_ub_bytes + copyout_ub
    if source_ub > _SOURCE_UB_BUDGET or plan_ub > _PLAN_UB_BUDGET:
        return plan
    return replace(
        plan,
        source_ub_bytes=source_ub,
        plan_ub_bytes=plan_ub,
        merge_copyout=True,
    )


def select_tile_plan(
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
    dtype: str,
    layout: str,
) -> TilePlan:
    if layout == "1d":
        batch, heads = 1, 1
    legacy = _legacy_plan(
        batch, seq_len, heads, dim, dtype, compiler_reuse_1d=layout == "1d"
    )
    aligned = dim // 2 % _BLOCK_ELEMS[dtype] == 0
    if layout == "2d":
        table = _select_table(batch, seq_len, heads, dim, dtype)
        keep_table = (
            dtype != "float32"
            or not aligned
            or dim < 128
            or table.bn_groups < batch * legacy.head_blocks
            or legacy.work_blocks < _CORE_COUNT
            and table.work_blocks > legacy.work_blocks
        )
        small_serial_group = (
            table.bn_groups == 1 and table.block_tokens <= 3 and table.max_units >= 16
        )
        if keep_table and (
            not aligned
            and not small_serial_group
            or aligned
            and table.block_tokens >= 20
        ):
            return _with_copyout(table, heads, dim, dtype)
        if not aligned:
            return _with_copyout(legacy, heads, dim, dtype)

    arbitrary = _select_arbitrary(batch, seq_len, heads, dim, dtype)
    fills_first_wave = (
        layout == "1d"
        and dtype == "float32"
        and legacy.block_tokens >= 128
        and legacy.work_blocks < _CORE_COUNT
        and legacy.work_blocks < arbitrary.work_blocks <= _CORE_COUNT
        and 4 * arbitrary.source_ub_bytes <= 3 * legacy.source_ub_bytes
    )
    if (fills_first_wave or 4 * arbitrary.work_blocks <= 3 * legacy.work_blocks) and (
        dtype != "float32" or aligned
    ):
        return _with_copyout(arbitrary, heads, dim, dtype)
    return _with_copyout(legacy, heads, dim, dtype)


__all__ = ["TilePlan", "select_tile_plan"]
