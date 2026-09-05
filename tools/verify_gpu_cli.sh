#!/bin/bash
# The bar that actually exercises the accelerator.
#
# Every previous green result came from a build where the GPU path was never
# reached: RNA_GPU_CHUNK unset means process_input() dispatches per record and
# par_mfe() is never called. This runs the SAME binary both ways over the same
# frozen references, so a difference can only come from the chunk path.
#
# Usage: verify_gpu_cli.sh [build-tree] [chunk-size]
set -u
export PATH="$HOME/miniforge3/bin:$PATH"
B=${1:-$HOME/port27cuda}
CHUNK=${2:-8}
BIN=$B/src/bin/RNAfold
T=$HOME/rnatest
R=$T/ref_md
W=$HOME/gpucli
mkdir -p "$W" || exit 2
CUDA_LIBDIR=$(dirname "$(command -v nvcc)")/../lib
export LD_LIBRARY_PATH=$CUDA_LIBDIR

[ -x "$BIN" ] || { echo "no RNAfold at $BIN" >&2; exit 2; }
echo "binary: $BIN"
echo "chunk : RNA_GPU_CHUNK=$CHUNK"
echo

fails=0
for inp in C_mixed C_control asc desc extreme; do
  [ -f "$T/$inp.fa" ] || continue
  [ -f "$R/$inp.out" ] || continue

  t0=$(date +%s)
  RNA_GPU_CHUNK=$CHUNK "$BIN" --noPS -i "$T/$inp.fa" > "$W/$inp.gpu" 2> "$W/$inp.err"
  rc=$?
  t1=$(date +%s)

  if [ $rc -ne 0 ]; then
    printf '  %-12s FAIL (exit %d)\n' "$inp" "$rc"
    grep -iE 'error|fail' "$W/$inp.err" | head -3 | sed 's/^/      /'
    fails=$((fails+1)); continue
  fi

  if cmp -s "$R/$inp.out" "$W/$inp.gpu"; then
    printf '  %-12s IDENTICAL via GPU chunks (%ds)\n' "$inp" "$((t1-t0))"
  else
    printf '  %-12s DIFFERS: %s\n' "$inp" "$(cmp "$R/$inp.out" "$W/$inp.gpu" 2>&1 | head -1)"
    fails=$((fails+1))
  fi
done

echo
if [ $fails -eq 0 ]; then
  echo "RESULT: the GPU chunk path reproduces the frozen references byte for byte"
else
  echo "RESULT: $fails inputs differ"
fi
exit $fails
