import csv, glob, sys
def util(prof, keycols=("aic_time(us)","aic_cube_time(us)","aiv_time(us)","aiv_vec_time(us)","aic_mte2_time(us)","aiv_mte2_time(us)","aiv_mte3_time(us)","aic_fixpipe_time(us)","aiv_scalar_time(us)","aic_scalar_time(us)")):
    files = glob.glob(prof + "/**/PipeUtilization*.csv", recursive=True)
    tot = {c: 0.0 for c in keycols}; n=0
    for f in files:
        for row in csv.DictReader(open(f)):
            n+=1
            for c in keycols:
                try: tot[c]+=float(row.get(c) or 0)
                except ValueError: pass
    n = max(n,1)
    m = {c: tot[c]/n for c in keycols}
    aic, cub, aiv, vec = m["aic_time(us)"], m["aic_cube_time(us)"], m["aiv_time(us)"], m["aiv_vec_time(us)"]
    print(f"{prof.split('/')[-2]:16s} rows={n} aic_blk={aic:7.1f}us cube%={100*cub/max(aic,1e-9):5.1f} fixpipe%={100*m['aic_fixpipe_time(us)']/max(aic,1e-9):5.1f} aic_mte2%={100*m['aic_mte2_time(us)']/max(aic,1e-9):5.1f} aic_scalar%={100*m['aic_scalar_time(us)']/max(aic,1e-9):5.1f} | aiv_blk={aiv:7.1f}us vec%={100*vec/max(aiv,1e-9):5.1f} aiv_mte2%={100*m['aiv_mte2_time(us)']/max(aiv,1e-9):5.1f} aiv_mte3%={100*m['aiv_mte3_time(us)']/max(aiv,1e-9):5.1f} aiv_scalar%={100*m['aiv_scalar_time(us)']/max(aiv,1e-9):5.1f}")
for p in sys.argv[1:]:
    util(p)
