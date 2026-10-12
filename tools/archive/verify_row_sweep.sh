#!/bin/bash
# Run the CUDA batch sweep under RNA_ROW_VERIFY.
#
# RNA_ROW_VERIFY makes the sweep compare every device cell against the host
# recursion, per row. It implies RNA_GPU_SWEEP=0 (the host sweep) because device
# mode is exactly the mode that does not run the host loops -- with nothing to
# compare against, the verify would print nothing, which reads identically to
# "verified clean".
#
# Requires a built --enable-cuda tree and a GPU.
#
# Usage: verify_row_sweep.sh [build-tree] [n_records] [length]
set -u
export PATH="$HOME/miniforge3/bin:$PATH"

B=${1:-$HOME/port27cuda}
N=${2:-4}
L=${3:-120}
SRC=/mnt/c/Users/lukef/CUDA_RNAFold/tests/upstream/cuda_sweep_harness.c
W=${TMPDIR:-/tmp}/vrna_row_verify
mkdir -p "$W" || exit 2

LIB=$(find "$B/src" -name 'libRNA_conv.a' | head -1)
[ -n "$LIB" ] || { echo "no libRNA_conv.a under $B -- build with --enable-cuda first" >&2; exit 2; }

CUDA_LIBDIR=$(dirname "$(command -v nvcc)")/../lib
echo "=== building the harness against $LIB"
gcc -O1 -o "$W/harness" "$SRC" \
    -I"$B/src" -I"$B/src/ViennaRNA" -I"$B" \
    "$LIB" -L"$CUDA_LIBDIR" -lcudart -lm -lpthread -lstdc++ -fopenmp 2>&1 | head -20
[ -x "$W/harness" ] || { echo "COMPILE FAILED"; exit 2; }

echo
echo "=== GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  none visible"

echo
echo "=== running with RNA_ROW_VERIFY=1 ($N records x $L nt)"
echo "    any per-cell disagreement is reported on stderr by the sweep itself"
cd "$W" || exit 2
RNA_ROW_VERIFY=1 LD_LIBRARY_PATH="$CUDA_LIBDIR" ./harness "$N" "$L" > stdout.log 2> stderr.log
rc=$?

echo
echo "--- harness result (exit $rc) ---"
cat stdout.log
echo
echo "--- sweep diagnostics (stderr), filtered ---"
grep -iE 'verify|mismatch|differ|error|FAIL|row' stderr.log | head -30
echo
# Count only NON-ZERO mismatch reports. "8778 cells checked, 0 mismatching" is a
# clean result and contains the word "mismatching"; grepping for the word alone
# counted every successful check as a failure.
n_bad=$(grep -oE '[0-9]+ mismatching' stderr.log | awk '$1 != 0' | wc -l)
n_bad=$((n_bad + $(grep -ciE 'backtracking failed|CUDA error|Segmentation' stderr.log)))
cells=$(grep -oE '[0-9]+ cells checked' stderr.log | awk '{s+=$1} END{print s+0}')
echo "per-cell verify: $cells cells checked, $n_bad reporting disagreement"
if [ "$rc" -eq 0 ] && [ "$n_bad" -eq 0 ]; then
  echo "RESULT: per-cell verify clean AND end results match upstream"
else
  echo "RESULT: see above"
fi
exit $rc
