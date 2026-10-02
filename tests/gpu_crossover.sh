#!/bin/bash
# Where does the GPU start to pay? Gate 4's threshold, measured instead of guessed.
#
# WHY THIS EXISTS. Gate 4 used to be a RECORD COUNT (VRNA_MIN_GPU_BATCH = 10), and
# a count is the wrong unit: the device's advantage scales with WORK, which is
# O(n^2) per record in the matrices and O(n^3) in the sweep. So one 5601 nt
# sequence is emphatically worth sending while ten 80 nt sequences are not, and
# the count-only gate got both cases backwards. The gate now also fires on TOTAL
# NUCLEOTIDES (VRNA_MIN_GPU_NT, default 2000) -- and that 2000 is a placeholder
# that this script exists to replace with a measurement.
#
# WHAT IT MEASURES. For a grid of (record count x length), the same input folded
# with the device path forced ON and forced OFF, byte-compared, and timed. The
# crossover is where the GPU arm stops losing. Reported per shape AND collapsed
# onto total nucleotides, because the claim being tested is that TOTAL LENGTH is
# the variable that predicts it -- if it is not, the table will say so and the
# gate needs a different form.
#
# HOW EACH ARM IS FORCED, and why it is not just RNA_MIN_GPU_NT:
#   GPU: RNA_MIN_GPU_NT=0 RNA_MIN_GPU_BATCH=1 RNA_GPU_WORK_FLOOR=0   -- both arms of gate 4 wide open
#   CPU: RNA_GPU_CHUNK unset                    -- gate 3, the master switch
# Using the master switch for the CPU arm means the comparison is device-vs-no-
# device, not one threshold against another.
#
# THE TRAP THIS AVOIDS. A GPU-less build of this binary is SILENT and
# byte-identical to the CPU answer (measured: 2437 bytes of stderr against 0,
# same stdout), so "the GPU arm was not slower" is also what you see when there
# was no GPU arm. Every GPU row therefore asserts `sweep shape:` on stderr, and
# the script refuses to report anything if that never appears.
#
# Usage: tests/gpu_crossover.sh [build-tree] [reps]
set -u
TREE=${1:-$HOME/lfb}
REPS=${2:-3}
BIN=$TREE/src/bin/RNAfold
[ -x "$BIN" ] || { echo "no RNAfold under $TREE" >&2; exit 2; }
W=${TMPDIR:-/tmp}/gpu_crossover.$$; mkdir -p "$W"
trap 'rm -rf "$W"' EXIT

# Is the accelerator even in this binary? Positive evidence only.
python3 - "$W/gate.fa" <<'PY'
import random, sys
random.seed(1)
with open(sys.argv[1], "w") as f:
    for i in range(4):
        f.write(">g%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(400))))
PY
RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 RNA_GPU_WORK_FLOOR=0 RNA_MIN_GPU_NT=0 \
    "$BIN" --noPS -i "$W/gate.fa" > /dev/null 2> "$W/gate.err"
if ! grep -q 'sweep shape:' "$W/gate.err"; then
  echo "FATAL: this build never sweeps -- it has no CUDA, or no device is visible."
  echo "       Every row below would compare the CPU against itself."
  exit 1
fi
echo "gate: the accelerator is live in $BIN"
echo

# median of REPS, in seconds, of one arm
timeit() {
  local out=$1 err=$2; shift 2
  local i t best
  local -a ts=()
  for ((i=0;i<REPS;i++)); do
    local s e
    s=$(date +%s.%N)
    env "$@" "$BIN" --noPS -i "$FA" > "$out" 2> "$err"
    e=$(date +%s.%N)
    ts+=("$(echo "$e - $s" | bc)")
  done
  printf '%s\n' "${ts[@]}" | sort -g | awk -v n="$REPS" 'NR==int((n+1)/2){print; exit}'
}

printf '%6s %6s %10s %10s %10s %9s %s\n' \
       records length total_nt gpu_s cpu_s speedup identical
fail=0
for L in 200 400 800 1600 3200 5601; do
  for N in 1 2 4 8 16; do
    FA=$W/n${N}_L${L}.fa
    python3 - "$FA" "$N" "$L" <<'PY'
import random, sys
_, path, n, L = sys.argv
random.seed(hash((int(n), int(L))) & 0xffff)
with open(path, "w") as f:
    for i in range(int(n)):
        f.write(">r%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(int(L)))))
PY
    g=$(timeit "$W/g.out" "$W/g.err" RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 RNA_GPU_WORK_FLOOR=0 RNA_MIN_GPU_NT=0)
    c=$(timeit "$W/c.out" "$W/c.err" RNA_GPU=0)
    # The GPU arm must actually have swept, or this row means nothing.
    if ! grep -q 'sweep shape:' "$W/g.err"; then
      printf '%6d %6d %10d %10s %10s %9s %s\n' "$N" "$L" "$((N*L))" "-" "-" "-" "NO-SWEEP"
      fail=1; continue
    fi
    same=$(cmp -s "$W/g.out" "$W/c.out" && echo yes || echo "*** NO ***")
    [ "$same" = yes ] || fail=1
    sp=$(echo "scale=2; $c / $g" | bc)
    printf '%6d %6d %10d %10.3f %10.3f %8sx %s\n' \
           "$N" "$L" "$((N*L))" "$g" "$c" "$sp" "$same"
  done
done

echo
echo "Reading it: the crossover is the smallest total_nt whose speedup is >= 1.00"
echo "and stays there. If rows with equal total_nt but different shapes disagree"
echo "sharply, then total nucleotides is NOT the right variable for gate 4 and the"
echo "threshold needs to be per-record length, or a function of both."
echo
[ $fail -eq 0 ] && echo "gpu_crossover: PASS (every GPU arm swept and agreed with the CPU)" \
                || echo "gpu_crossover: FAIL -- see the rows above"
exit $fail
