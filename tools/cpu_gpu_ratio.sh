#!/bin/bash
# How much can the CPU pool add while the GPU is busy?
#
# The answer is a RATIO -- CPU seconds per record on one core, against GPU
# seconds per record -- and it is strongly length-dependent, because the GPU's
# advantage comes from parallelism across an O(n^2) cell space. At 5601 nt the
# device wins by ~76x; the question is where that lands at lengths people
# actually fold. MIN_GPU_BATCH's break-even (~65 records at 300 nt, ~10 at
# 600 nt, 1 at >=1200 nt) says the ratio moves a lot, so measure it.
#
# Usage: cpu_gpu_ratio.sh [build-tree]
set -u
B=${1:-$HOME/port27head}
BIN=$B/src/bin/RNAfold
W=${TMPDIR:-/tmp}/cpugpu
rm -rf "$W"; mkdir -p "$W"

[ -x "$BIN" ] || { echo "no RNAfold at $BIN" >&2; exit 2; }
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
CORES=$(nproc)
echo "cores: $CORES"
echo

gen() { # gen N LEN OUT
  python3 - "$1" "$2" "$3" <<'PY'
import random, sys
n, L, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
random.seed(1234 + L)
with open(out, "w") as f:
    for i in range(n):
        f.write(">r%04d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(L))))
PY
}

printf '%-7s %10s %12s %12s %9s %10s\n' \
       "len" "cpu s/rec" "gpu s/rec" "ratio/core" "with $CORES" "of GPU"
for L in 300 600 1200 2400; do
  # CPU: a handful of records, accelerator OFF, single threaded
  NC=6
  gen $NC "$L" "$W/cpu_$L.fa"
  t0=$(date +%s.%N)
  "$BIN" --noPS -i "$W/cpu_$L.fa" > /dev/null 2>&1
  t1=$(date +%s.%N)
  cpu=$(echo "($t1 - $t0) / $NC" | bc -l)

  # GPU: enough records that the batch machinery is the thing being measured.
  # Uses the KERNEL phases, not wall, so host build/output do not pollute it.
  NG=64
  gen $NG "$L" "$W/gpu_$L.fa"
  RNA_GPU_CHUNK=0 "$BIN" --noPS -i "$W/gpu_$L.fa" > /dev/null 2> "$W/g_$L.err"
  gpu=$(grep -h "phase timing" "$W/g_$L.err" | tail -1 | \
        sed 's/.*int_loop=\([0-9.]*\) hp_mb=\([0-9.]*\) load_my_c=\([0-9.]*\) modular_decomp=\([0-9.]*\) fetch_mx=\([0-9.]*\).*/\1 \2 \4/' | \
        awk -v n=$NG '{print ($1+$2+$3)/n}')
  [ -n "$gpu" ] || gpu=0

  if [ "$(echo "$gpu > 0" | bc -l)" = "1" ]; then
    ratio=$(echo "$cpu / $gpu" | bc -l)
    add=$(echo "100 * $CORES / $ratio" | bc -l)
    printf '%-7s %10.3f %12.4f %12.1f %9s %9.1f%%\n' "$L" "$cpu" "$gpu" "$ratio" "$CORES cores" "$add"
  else
    printf '%-7s %10.3f %12s %12s %9s %10s\n' "$L" "$cpu" "n/a" "n/a" "-" "-"
    echo "         (no phase line -- did the GPU path run? check $W/g_$L.err)"
  fi
done
echo
echo "'of GPU' is what the CPU pool could ADD to device throughput while the"
echo "device is busy, at this core count -- an upper bound assuming the cores are"
echo "otherwise idle and the split is perfectly balanced."
