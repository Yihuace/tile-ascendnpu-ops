#!/usr/bin/env bash
# Run one `msprof op` capture and parse it into JSON.
#
# usage: ./run_prof.sh <out_root> <tag> <launch_count> <cmd...>
#   env METRICS   override --aic-metrics (default full set incl. Roofline)
#   env WARMUP    override --warm-up (default 5)
#
# Writes <out_root>/<tag>/ (raw OPPROF_*) and <out_root>/<tag>/prof.json.
set -euo pipefail
out_root="$1"; tag="$2"; lc="$3"; shift 3
METRICS="${METRICS:-BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio,Roofline}"
WARMUP="${WARMUP:-5}"
dir="${out_root}/${tag}"
rm -rf "${dir}"
mkdir -p "${dir}"
chmod 700 "${dir}"
start=$(date +%s)
msprof op --kernel-name=main --output="${dir}" --launch-count="${lc}" --warm-up="${WARMUP}" \
  --aic-metrics="${METRICS}" "$@" > "${dir}/msprof.log" 2>&1 || {
    echo "MSPROF_FAILED ${tag}"; tail -20 "${dir}/msprof.log"; exit 1; }
end=$(date +%s)
prof=$(find "${dir}" -maxdepth 1 -type d -name 'OPPROF_*' | head -1)
python "$(dirname "$0")/parse_prof.py" "${prof}" > "${dir}/prof.json" 2>"${dir}/parse.err"
dur=$(python -c "import json;print(round(json.load(open('${dir}/prof.json'))['duration_us'],3))")
rat=$(python -c "import json;d=json.load(open('${dir}/prof.json'));print(round(d.get('ratio_pct') or 0,4))")
bd=$(python -c "import json;print(json.load(open('${dir}/prof.json'))['block_dim'])")
echo "OK ${tag} duration_us=${dur} ratio_pct=${rat} block_dim=${bd} wall=$((end-start))s"
