"""Pure-Python scheduling simulation of the §3.3 task loop (DESIGN.md revision 2).

Purpose (REVIEW v1 blocking problem 1): the v1 "one-task-lagged store" lost the
**last task of every block whose task count is NPER-1**, because ``scatter`` sat
inside ``if t < TASKS`` and the epilogue only covered the globally last task.
This script re-implements the §3.3 scheduling *mechanically* and proves, for
every in-scope (T, K, NPER) combination, the three properties the design claims:

  P1  coverage      : every logical task t in [0, TASKS) is scattered exactly once
                      (no duplicate, no miss)  -> no `torch.empty` garbage in `A`
  P2  flag pairing  : for each of the 6 flags, #set == #wait, and every wait is
                      satisfied by a preceding set  -> no deadlock, no extra wait
  P3  WAR order     : per slot, the cross-engine buffers alternate
                      W(t) -> R(t) -> W(t+2K) -> R(t+2K) ...  -> no overwrite of
                      un-consumed data (`gs_G`, `ws_S`, `y_blk`)

P3 is not argued on paper: the two engines are run as **co-operatively scheduled
coroutines** with *blocking* flag waits (a wait on an empty flag suspends the
engine).  The global event log then decides the order.  If both engines are
blocked and neither is finished, the run is reported as a DEADLOCK.

Also checked (lemma used by the "block-start zero init" fix of blocking
problem 2): inside one block, tasks with ``valid < 64`` (the sequence tail chunk)
form a **suffix** of that block's task list -- i.e. every task before them is a
full chunk that overwrote all 64 rows of `l1k` / `ub_gc` / `ub_gr` with finite
GM data.  Without that lemma, stale finite data could not be guaranteed.

Run:  python3 design_schedule_sim.py            # tables + PASS/FAIL
      python3 design_schedule_sim.py --quiet    # exit code only
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field

KMAX = 20  # physical AI cores (NPUUtils.get().get_aicore_num() == 20)

# flag ids exactly as in DESIGN.md §7 (F_G=0, F_S=2, F_A=4, + slot)
F_G, F_S, F_A = 0, 2, 4


# ---------------------------------------------------------------------------
# 1. v1 (defective) and v2 (adopted) scatter schedules, as literal transcriptions
# ---------------------------------------------------------------------------
def schedule_v1(bid: int, TASKS: int, K: int, NPER: int) -> list[int]:
    """revision 1 §3.3: scatter inside `if t < TASKS`, epilogue = global-last only."""
    out = []
    for it in range(NPER):
        t = bid + it * K
        if t < TASKS:
            if it >= 1:  # <- inside the guard
                out.append(t - K)
    if bid + (NPER - 1) * K < TASKS:
        out.append(bid + (NPER - 1) * K)
    return out


def schedule_v2(bid: int, TASKS: int, K: int, NPER: int) -> list[int]:
    """revision 2 §3.3: guard bound to the *scattered* task + per-block `last_it`."""
    out = []
    for it in range(NPER):
        t = bid + it * K
        if it >= 1 and (t - K) < TASKS:  # guard belongs to t-K, not to t
            out.append(t - K)
    last_it = (TASKS - 1 - bid) // K  # this block's own last task
    if last_it == NPER - 1:  # only when the loop stopped one task short
        out.append(bid + last_it * K)
    return out


def block_task_count(bid: int, TASKS: int, K: int) -> int:
    return (TASKS - 1 - bid) // K + 1 if bid < TASKS else 0


# ---------------------------------------------------------------------------
# 2. two-engine blocking simulation (Cube + Vector, §3.3 / §7 program order)
# ---------------------------------------------------------------------------
@dataclass
class Flags:
    counts: dict[int, int] = field(default_factory=dict)
    sets: dict[int, int] = field(default_factory=dict)
    waits: dict[int, int] = field(default_factory=dict)

    def set(self, fid: int) -> None:
        self.counts[fid] = self.counts.get(fid, 0) + 1
        self.sets[fid] = self.sets.get(fid, 0) + 1

    def wait(self, fid: int) -> bool:
        if self.counts.get(fid, 0) > 0:
            self.counts[fid] -= 1
            self.waits[fid] = self.waits.get(fid, 0) + 1
            return True
        return False  # would block


class Engine:
    """A coroutine that yields ('wait', fid) or ('ev', ...) and suspends on wait."""

    def __init__(self, name: str):
        self.name = name
        self.pc = None

    def _gen(self):
        raise NotImplementedError


class Sim:
    def __init__(self, TASKS: int, K: int, NPER: int, head: int, nch: int):
        self.TASKS, self.K, self.NPER, self.head, self.nch = TASKS, K, NPER, head, nch
        self.flags = Flags()
        self.log: list[tuple] = []  # (seq, engine, kind, payload)
        self.scattered: list[int] = []
        self.blocked = {"Cube": False, "Vector": False}
        self.done = {"Cube": False, "Vector": False}

    # -- event recorder -----------------------------------------------------
    def ev(self, engine: str, kind: str, payload) -> None:
        self.log.append((len(self.log), engine, kind, payload))

    def access(self, engine: str, buf: str, slot: int, rw: str, task: int) -> None:
        self.ev(engine, "access", (buf, slot, rw, task))

    # -- engine programs ----------------------------------------------------
    def cube(self, bid: int):
        K, TASKS, NPER = self.K, self.TASKS, self.NPER
        F_G, F_S, F_A = 0, 2, 4

        def gram(t, slot):
            self.access("Cube", "gs_G", slot, "W", t)
            self.ev("Cube", "set", F_G + slot)
            self.flags.set(F_G + slot)

        gram(bid, 0)  # leading gram of the block's first task
        for it in range(NPER):
            t = bid + it * K
            slot = it % 2
            if t < TASKS:
                if it + 1 < NPER and t + K < TASKS:
                    gram(t + K, 1 - slot)  # one-task-ahead Gram prefetch
                yield ("wait", F_S + slot)  # ws_S[slot] ready
                self.access("Cube", "ws_S", slot, "R", t)
                # ---- 6-level log-doubling chain (engine-internal, atomic here)
                self.access("Cube", "y_blk", slot, "W", t)
                self.ev("Cube", "set", F_A + slot)
                self.flags.set(F_A + slot)

    def vector(self, bid: int):
        K, TASKS, NPER = self.K, self.TASKS, self.NPER
        F_G, F_S, F_A = 0, 2, 4

        def scatter(tt, pslot):
            yield ("wait", F_A + pslot)
            self.access("Vector", "y_blk", pslot, "R", tt)
            self.access("Vector", "A_pad", pslot, "W", tt)
            self.scattered.append(tt)

        for it in range(NPER):
            t = bid + it * K
            slot = it % 2
            if t < TASKS:
                yield ("wait", F_G + slot)  # gs_G[slot] ready
                self.access("Vector", "gs_G", slot, "R", t)
                self.access("Vector", "ws_S", slot, "W", t)
                self.ev("Vector", "set", F_S + slot)
                self.flags.set(F_S + slot)
            # revision 2: scatter guard bound to the scattered task, outside the guard
            if it >= 1 and (t - K) < TASKS:
                yield from scatter(t - K, (it - 1) % 2)
        last_it = (TASKS - 1 - bid) // K
        if last_it == NPER - 1:
            yield from scatter(bid + last_it * K, last_it % 2)

    # -- co-operative scheduler --------------------------------------------
    def run(self) -> dict:
        gens = {"Cube": self.cube(self.blockid), "Vector": self.vector(self.blockid)}
        # prime
        pending = {}
        for name, g in gens.items():
            try:
                pending[name] = next(g)
            except StopIteration:
                self.done[name] = True
        order = ["Cube", "Vector"]
        idx = 0
        rounds = 0
        while not all(self.done.values()):
            rounds += 1
            if rounds > 10 * (self.TASKS + 1) * 8 + 1000:
                return {"deadlock": True, "reason": "step budget exhausted"}
            name = order[idx % 2]
            idx += 1
            if self.done[name]:
                continue
            instr = pending.get(name)
            if instr is None:
                continue
            if instr[0] == "wait":
                if not self.flags.wait(instr[1]):
                    self.blocked[name] = True
                    continue
                self.blocked[name] = False
            # step the generator until it blocks/finishes
            try:
                pending[name] = next(gens[name])
            except StopIteration:
                self.done[name] = True
                pending[name] = None
            if all(self.done.values()):
                break
            if all(self.blocked[k] for k in ("Cube", "Vector") if not self.done[k]):
                return {"deadlock": True, "reason": "both engines blocked"}
        return {"deadlock": False, "rounds": rounds}

    def run_for(self, bid: int) -> dict:
        self.blockid = bid
        return self.run()


# ---------------------------------------------------------------------------
# 3. checks
# ---------------------------------------------------------------------------
def check_coverage(TASKS: int, K: int, NPER: int) -> tuple[bool, int, int]:
    got = []
    for bid in range(K):
        got.extend(schedule_v2(bid, TASKS, K, NPER))
    want = list(range(TASKS))
    ok = sorted(got) == want and len(got) == len(want)
    return ok, len(got), len(set(got))


def check_v1_gap(TASKS: int, K: int, NPER: int) -> int:
    got = []
    for bid in range(K):
        got.extend(schedule_v1(bid, TASKS, K, NPER))
    return len(set(range(TASKS)) - set(got))


def check_sim(TASKS: int, K: int, NPER: int, head: int, nch: int) -> dict:
    """Full two-engine run for every block; returns the P2/P3/P1 verdicts."""
    res = {
        "deadlock": False,
        "pairing": True,
        "war": True,
        "coverage": True,
        "detail": {},
    }
    all_scattered = []
    for bid in range(K):
        sim = Sim(TASKS, K, NPER, head, nch)
        out = sim.run_for(bid)
        if out.get("deadlock"):
            res["deadlock"] = True
            res["detail"][bid] = out
            return res
        all_scattered.extend(sim.scattered)
        # P2: every flag set count == wait count
        for fid in (0, 1, 2, 3, 4, 5):
            if sim.flags.sets.get(fid, 0) != sim.flags.waits.get(fid, 0):
                res["pairing"] = False
                res["detail"].setdefault("pairing", []).append(
                    (bid, fid, sim.flags.sets.get(fid, 0), sim.flags.waits.get(fid, 0))
                )
        # P3: per (buffer, slot) the access sequence alternates W,R,W,R with equal task ids
        seq = {}
        for _s, eng, kind, payload in sim.log:
            if kind != "access":
                continue
            buf, slot, rw, task = payload
            if buf == "A_pad":
                continue
            seq.setdefault((buf, slot), []).append((rw, task, eng))
        for (buf, slot), acc in seq.items():
            for i in range(0, len(acc) - 1, 2):
                w, r = acc[i], acc[i + 1]
                if not (w[0] == "W" and r[0] == "R" and w[1] == r[1]):
                    res["war"] = False
                    res["detail"].setdefault("war", []).append((bid, buf, slot, acc))
                    break
            if len(acc) % 2:
                res["war"] = False
                res["detail"].setdefault("war", []).append((bid, buf, slot, "odd-length"))
    if sorted(all_scattered) != list(range(TASKS)):
        res["coverage"] = False
    return res


def check_tail_suffix(TASKS: int, K: int, head: int, nch: int) -> bool:
    """Inside a block, valid<64 tasks (tail chunk) must form a suffix."""
    def valid_of(t):
        chunk = t // head
        base = chunk * 64
        return min(64, nch * 64 - base)  # equal-length single sequence

    for bid in range(K):
        ts = [bid + it * K for it in range((TASKS - 1 - bid) // K + 1)]
        flags = [valid_of(t) < 64 for t in ts]
        seen_tail = False
        for f in flags:
            if f:
                seen_tail = True
            elif seen_tail:
                return False
    return True


# ---------------------------------------------------------------------------
# 4. case tables
# ---------------------------------------------------------------------------
CASES = [
    # label, S, head, key_head, dim_k
    ("L0-a  S=64", 64, 1, 1, 64),
    ("L0-b  S=64", 64, 1, 1, 128),
    ("L0-c  S=64 bf16", 64, 1, 1, 64),
    ("L0-d  S=100", 100, 1, 1, 64),
    ("L0-e  S=64 H=2", 64, 2, 1, 64),
    ("L0-f  S=64", 64, 1, 1, 64),
    ("L0-g  S=128 H=16", 128, 16, 16, 128),
    ("L0-h  S=64", 64, 1, 1, 64),
    ("L0-i  S=1600 H=1", 1600, 1, 1, 64),
    ("L1-a  S=1024 H=4", 1024, 4, 2, 128),
    ("L1-b  S=3000 H=16", 3000, 16, 1, 128),
    ("L2    S=4096 H=16", 4096, 16, 1, 128),
    ("L2    S=12288 H=16", 12288, 16, 16, 64),
    ("L2    S=32768 H=16", 32768, 16, 16, 128),
    ("Bound S=56", 56, 1, 1, 64),
    ("Bound S=64", 64, 1, 1, 64),
    ("Bound S=65", 65, 1, 1, 64),
]


def dims(S: int, head: int):
    nch = math.ceil(S / 64)
    T = nch * head
    K = min(KMAX, T)
    NPER = math.ceil(T / K)
    return nch, T, K, NPER


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--sweep-max", type=int, default=2048)
    args = ap.parse_args()

    ok_all = True

    # ---- table A: in-scope shapes, v1 gap vs v2 coverage -------------------
    rows = []
    for label, S, head, kh, dk in CASES:
        nch, T, K, NPER = dims(S, head)
        gap1 = check_v1_gap(T, K, NPER)
        ok2, n, uniq = check_coverage(T, K, NPER)
        counts = [block_task_count(b, T, K) for b in range(K)]
        tail_ok = check_tail_suffix(T, K, head, nch)
        rows.append((label, T, K, NPER, gap1, ok2, n - uniq, min(counts), max(counts), tail_ok))
        ok_all &= ok2 and gap1 >= 0 and tail_ok
    if not args.quiet:
        print("## A. scatter coverage per in-scope shape (§3.3 loop, K=min(20,T))\n")
        print("| case | T | K | NPER | v1 漏落盘任务数 | v2 覆盖 | v2 重复 | 每 block 任务数 min..max | 尾块为本 block 后缀 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for label, T, K, NPER, gap1, ok2, dup, cmin, cmax, tail_ok in rows:
            print(
                f"| {label} | {T} | {K} | {NPER} | {gap1} | "
                f"{'exactly-once' if ok2 else 'FAIL'} | {dup} | {cmin}..{cmax} | "
                f"{'yes' if tail_ok else 'NO'} |"
            )

    # ---- table B: full two-engine blocking simulation ----------------------
    sim_rows = []
    for label, S, head, kh, dk in CASES:
        nch, T, K, NPER = dims(S, head)
        r = check_sim(T, K, NPER, head, nch)
        sim_rows.append((label, r["deadlock"], r["pairing"], r["war"], r["coverage"]))
        ok_all &= (not r["deadlock"]) and r["pairing"] and r["war"] and r["coverage"]
    if not args.quiet:
        print("\n## B. two-engine blocking simulation (deadlock / flag pairing / WAR order)\n")
        print("| case | 死锁 | #set == #wait（6 flag） | 槽位 W→R 严格交替 | 落盘 exactly-once |")
        print("|---|---|---|---|---|")
        for label, dl, pr, war, cov in sim_rows:
            print(
                f"| {label} | {'NO' if not dl else 'DEADLOCK'} | {'yes' if pr else 'NO'} | "
                f"{'yes' if war else 'NO'} | {'yes' if cov else 'NO'} |"
            )

    # ---- table C: exhaustive (T, K=min(20,T)) sweep ------------------------
    bad = []
    for T in range(1, args.sweep_max + 1):
        K = min(KMAX, T)
        NPER = math.ceil(T / K)
        ok, n, uniq = check_coverage(T, K, NPER)
        if not ok or n != T:
            bad.append((T, K, NPER, n, uniq))
        # per-block task count must be NPER or NPER-1 (load balance claim §5.3)
        counts = [block_task_count(b, T, K) for b in range(K)]
        if max(counts) - min(counts) > 1 or max(counts) > NPER:
            bad.append((T, K, NPER, "load-imbalance", counts))
    ok_all &= not bad
    if not args.quiet:
        print(f"\n## C. exhaustive sweep T = 1..{args.sweep_max} (K=min(20,T), NPER=ceil(T/K))\n")
        print(f"- cases: {args.sweep_max}; violation rows: {len(bad)}")
        print("- each case asserts: scatter multiset == range(T) (exactly once) and "
              "per-block task count ∈ {NPER-1, NPER}")
        if bad:
            print(f"- first violations: {bad[:8]}")

    # ---- table D: the NPER-1 shape the reviewer asked for ------------------
    if not args.quiet:
        print("\n## D. shapes whose blocks carry fewer than NPER tasks (REVIEW 建议 11)\n")
        print("| T | K | NPER | block 任务数分布 | v1 漏 | v2 覆盖 |")
        print("|---|---|---|---|---|---|")
        for T in (10, 20, 21, 25, 39, 40, 41, 64):
            K = min(KMAX, T)
            NPER = math.ceil(T / K)
            counts = [block_task_count(b, T, K) for b in range(K)]
            dist = {c: counts.count(c) for c in sorted(set(counts))}
            print(f"| {T} | {K} | {NPER} | {dist} | {check_v1_gap(T, K, NPER)} | "
                  f"{'exactly-once' if check_coverage(T, K, NPER)[0] else 'FAIL'} |")

    # ---- table E: per-block trace (why the NPER-1 blocks are the broken ones) --
    if not args.quiet:
        print("\n## E. per-block scatter trace, T=25 (K=20, NPER=2) - the NPER-1 shape\n")
        print("| bid | this block's tasks (t = bid + it*K) | v1 scattered | v1 missed | v2 scattered | v2 missed |")
        print("|---|---|---|---|---|---|")
        T, K, NPER = 25, 20, 2
        for bid in range(K):
            tasks = [bid + it * K for it in range(NPER) if bid + it * K < T]
            s1, s2 = schedule_v1(bid, T, K, NPER), schedule_v2(bid, T, K, NPER)
            print(f"| {bid} | {tasks} | {s1} | {sorted(set(tasks) - set(s1))} | "
                  f"{s2} | {sorted(set(tasks) - set(s2))} |")

        print("\n## F. event trace, block 1 of T=2 / K=2 / head=1 (S=100 tail chunk, valid=36)\n")
        sim = Sim(2, 2, 1, head=1, nch=2)
        sim.run_for(1)
        for seq, eng, kind, payload in sim.log:
            if kind == "set":
                print(f"  {seq:3d} {eng:6s} set  f={payload}")
            else:
                buf, slot, rw, task = payload
                print(f"  {seq:3d} {eng:6s} {rw}    {buf}[slot={slot}] task={task}")
        print(f"  -> flags: {sorted(sim.flags.sets.items())} sets / "
              f"{sorted(sim.flags.waits.items())} waits; scattered={sim.scattered}")

    print()
    print("RESULT:", "SCHEDULE_PASS" if ok_all else "SCHEDULE_FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
