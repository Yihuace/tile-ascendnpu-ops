"""G3 verdict: compare the SUBMISSION Ratio (GM roofline) of the A/B pair."""
import csv, glob, os

GM_BW, FREQ, CORES, CL = 1.75e12, 1800e6, 40, 128
PPC = {"aiv_vec_fp16_ratio": 128, "aiv_vec_fp32_ratio": 64,
       "aiv_vec_int32_ratio": 64, "aiv_vec_misc_ratio": 32}

def metrics(root):
    dur_rows = glob.glob(os.path.join(root, "OPPROF_*/main/*/OpBasicInfo_*.csv"))
    arith_rows = glob.glob(os.path.join(root, "OPPROF_*/main/*/ArithmeticUtilization_*.csv"))
    l2_rows = glob.glob(os.path.join(root, "OPPROF_*/main/*/L2Cache_*.csv"))
    dur = []
    for f in dur_rows:
        for r in csv.DictReader(open(f)):
            v = r.get("Task Duration(us)")
            if v not in (None, "", "NA"): dur.append(float(v))
    arith = [r for f in arith_rows for r in csv.DictReader(open(f))]
    l2 = [r for f in l2_rows for r in csv.DictReader(open(f))]
    fops = sum(float(r["aiv_vec_fops"]) for r in arith if r.get("aiv_vec_fops") not in (None,"","NA"))
    misses = sum(float(r[c]) for r in l2 for c in ("aiv_r0_read_cache_miss_allocate","aiv_r1_read_cache_miss_allocate")
                 if r.get(c) not in (None,"","NA"))
    cyc = {c: sum(float(r[c]) * float(r["aiv_total_cycles"]) for r in arith
                  if r.get(c) not in (None,"","NA") and r.get("aiv_total_cycles") not in (None,"","NA"))
           for c in PPC}
    tot = sum(cyc.values())
    comp = sum(PPC[c] * (v/tot) for c, v in cyc.items()) * FREQ * CORES
    # multi-launch runs write one sub-dir per launch -> normalise fops/misses per launch
    n_launch = max(1, len({os.path.dirname(f) for f in l2_rows}))
    fops /= n_launch
    misses /= n_launch
    d = sum(dur)/len(dur)
    ai = fops/(misses*CL)
    perf = fops/(d*1e-6)
    ratio = perf/min(ai*GM_BW, comp)
    return dict(dur=d, fops=fops, comp=comp, ai=ai, perf=perf/1e12, ratio=ratio*100,
                props={k: v/tot for k, v in cyc.items()})

base = metrics("/tmp/opwork/ab_baseline")
minus = metrics("/tmp/opwork/ab_minus_op")
for name, m in (("baseline (5 PRNG ops)", base), ("minus_op (4 PRNG ops)", minus)):
    print(f"{name:24s} dur={m['dur']:7.3f}us fops={m['fops']:11.0f} comp={m['comp']/1e12:6.4f} "
          f"perf={m['perf']:6.4f} Ratio={m['ratio']:6.2f}%")
    print(f"{'':24s} mix: " + ", ".join(f"{k.replace('aiv_vec_','').replace('_ratio','')}={v:.3f}" for k,v in m['props'].items()))

print()
print(f"Δdur   = {(minus['dur']/base['dur']-1)*100:+.2f}%")
print(f"Δfops  = {(minus['fops']/base['fops']-1)*100:+.2f}%")
print(f"Δcomp  = {(minus['comp']/base['comp']-1)*100:+.2f}%")
print(f"ΔRatio = {minus['ratio']-base['ratio']:+.2f} 个百分点")
