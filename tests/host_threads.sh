#!/bin/bash
# Reserve a core, or lower the builders' priority? Measured, not argued.
#
# THE TENSION. The GPU is worth 33-43x one CPU core, so a builder thread that
# delays GPU servicing is expensive. But idle cores are also waste, and on a big
# node that is most of the machine. Two candidate policies:
#
#   RESERVE  give the builders fewer threads than the machine has, leaving one
#            (or more) core free for whatever services the device.
#   PRIORITY let the builders use every core but at lower scheduling priority,
#            so they yield whenever the GPU-facing work is runnable.
#
# AND A MEASURED REASON NOT TO ASSUME THE ANSWER. Flow4 section A showed the
# builder is SUPPOSED to compete: with two chunks, 4.96 s of a 9.65 s build hides
# behind the GPU, and that overlap is the single largest host-side win found so
# far. Starving the builder to protect the device could therefore cost more than
# it saves. That is exactly why this is a test and not a patch.
#
# WHAT IS MEASURABLE TODAY, AND WHAT IS NOT:
#   RESERVE  measurable now -- RNA_BUILD_THREADS sets the builder pool size.
#   PRIORITY NOT measurable yet. `nice` applies to the whole process, so it
#            cannot express "builders below the GPU-facing thread". That needs a
#            per-thread knob (pthread_setschedparam / setpriority on the builder
#            tids) which does not exist. This script measures the reserve axis and
#            prints what the priority arm would need. Do not read its absence as
#            a null result.
#
# Usage: tests/host_threads.sh [build-tree] [records] [length] [reps]
set -u
TREE=${1:-$HOME/lfb}
N=${2:-64}
L=${3:-2400}
REPS=${4:-3}
BIN=$TREE/src/bin/RNAfold
[ -x "$BIN" ] || { echo "no RNAfold under $TREE" >&2; exit 2; }
NPROC=$(nproc)
W=${TMPDIR:-/tmp}/host_threads.$$; mkdir -p "$W"
trap 'rm -rf "$W"' EXIT

FA=$W/n${N}_L${L}.fa
python3 - "$FA" "$N" "$L" <<'PY'
import random, sys
_, path, n, L = sys.argv
random.seed(20260927)
with open(path, "w") as f:
    for i in range(int(n)):
        f.write(">r%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(int(L)))))
PY

# The accelerator must be live, or every row compares the CPU against itself.
RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 RNA_GPU_WORK_FLOOR=0 "$BIN" --noPS -i "$FA" >/dev/null 2>"$W/gate.err"
grep -q 'sweep shape:' "$W/gate.err" || { echo "FATAL: no sweep -- no CUDA or no device"; exit 1; }
echo "host: $NPROC cores; shape ${N} x ${L}; accelerator live"
echo

# One arm: median wall, plus the stage numbers that say WHERE it went.
arm() {
  local label=$1 bt=$2 chunkcap=$3
  local -a ts=()
  local i s e last
  for ((i=0;i<REPS;i++)); do
    s=$(date +%s.%N)
    env RNA_GPU_CHUNK="$chunkcap" RNA_MIN_GPU_BATCH=1 RNA_GPU_WORK_FLOOR=0 RNA_BUILD_THREADS="$bt" \
        "$BIN" --noPS -i "$FA" >"$W/o.out" 2>"$W/o.err"
    e=$(date +%s.%N)
    ts+=("$(echo "$e - $s" | bc)")
    last=$W/o.err
  done
  local med bld ovl
  med=$(printf '%s\n' "${ts[@]}" | sort -g | awk -v n="$REPS" 'NR==int((n+1)/2){print; exit}')
  bld=$(grep -oE 'build=[0-9.]+' "$last" | head -1 | cut -d= -f2)
  ovl=$(grep -oE 'OVERLAPPED [0-9.]+' "$last" | head -1 | awk '{print $2}')
  printf '  %-22s %-8s %9.3f  %8s  %10s  %s\n' \
         "$label" "$bt" "$med" "${bld:--}" "${ovl:--}" \
         "$(cmp -s "$W/o.out" "$W/base.out" 2>/dev/null && echo same || echo DIFF)"
}

# Baseline output for the identity column.
env RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 RNA_GPU_WORK_FLOOR=0 "$BIN" --noPS -i "$FA" >"$W/base.out" 2>/dev/null

echo "  RESERVE axis: how many cores the fold-compound builders may use"
printf '  %-22s %-8s %9s  %8s  %10s  %s\n' arm threads wall_s build_s overlapped identical
arm "all cores"            "$NPROC"                 0
arm "reserve 1"            "$((NPROC-1))"           0
arm "reserve 2"            "$((NPROC>2?NPROC-2:1))" 0
arm "half the machine"     "$((NPROC/2>0?NPROC/2:1))" 0
arm "serial builder"       1                        0
echo
echo "  ... and the same with the chunk capped so the pipeline has something to"
echo "  overlap (Flow4 A: one chunk overlaps 0.00 s, two overlap 4.96 s)"
printf '  %-22s %-8s %9s  %8s  %10s  %s\n' arm threads wall_s build_s overlapped identical
CAP=$(( N/2 > 1 ? N/2 : 1 ))
arm "all cores, 2 chunks"  "$NPROC"                 "$CAP"
arm "reserve 1, 2 chunks"  "$((NPROC-1))"           "$CAP"
arm "half, 2 chunks"       "$((NPROC/2>0?NPROC/2:1))" "$CAP"

cat <<'EOF'

READING IT
  If "reserve 1" beats "all cores", the device was being starved and reserving is
  the policy. If it does not -- and especially if "half the machine" is WORSE --
  then builder throughput matters more than device latency at this shape, and the
  right answer is to give the builders everything and let the scheduler sort it
  out. Watch `overlapped`: a reservation that reduces overlap is paying twice.

  Note the confound this cannot remove: RNA_BUILD_THREADS changes BOTH how much
  parallelism the build has AND how much contention the device sees. A drop at
  "half the machine" could be either. Separating them needs the priority knob.

WHAT THE PRIORITY ARM WOULD NEED (not built)
  A per-thread niceness applied to the builder pool only -- setpriority(PRIO_PROCESS,
  tid, n) on Linux from inside each builder, or pthread_setschedparam with
  SCHED_BATCH. `nice` on the whole process cannot express it, which is why that arm
  is absent here rather than reported as a null. Suggested knob: RNA_BUILD_NICE=n,
  default 0, applied by each builder to itself at start-up so it needs no
  privileges (lowering priority never does).
EOF
