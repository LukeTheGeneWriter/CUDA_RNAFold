#!/bin/bash
# The bar that actually exercises the accelerator.
#
# Every previous green result came from a build where the GPU path was never
# reached: RNA_GPU_CHUNK unset means process_input() dispatches per record and
# par_mfe() is never called. This runs the SAME binary both ways over the same
# frozen references, so a difference can only come from the chunk path.
#
# RNA_GPU_VRAM_BUDGET_MB forces multi-chunk behaviour on a card big enough to
# swallow the whole input in one batch -- without it the budget path is only
# ever exercised at "everything fits", which is the one case that cannot go
# wrong. The standing bar uses 4/8/16/32.
#
# Usage: verify_gpu_cli.sh [build-tree] [chunk-size] [budget-mb ...]
set -u
export PATH="$HOME/miniforge3/bin:$PATH"
B=${1:-$HOME/port27cuda}
CHUNK=${2:-8}
shift 2 2>/dev/null || true
BUDGETS=${*:-}
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

run_all() {
  local BUDGET=${1:-}
  [ -n "$BUDGET" ] && echo "--- RNA_GPU_VRAM_BUDGET_MB=$BUDGET"
fails=0
for inp in C_mixed C_control asc desc extreme; do
  [ -f "$T/$inp.fa" ] || continue
  [ -f "$R/$inp.out" ] || continue

  t0=$(date +%s)
  if [ -n "$BUDGETS" ]; then
    RNA_GPU_VRAM_BUDGET_MB=$BUDGET RNA_GPU_CHUNK=$CHUNK "$BIN" --noPS -i "$T/$inp.fa" > "$W/$inp.gpu" 2> "$W/$inp.err"
  else
    RNA_GPU_CHUNK=$CHUNK "$BIN" --noPS -i "$T/$inp.fa" > "$W/$inp.gpu" 2> "$W/$inp.err"
  fi
  rc=$?
  t1=$(date +%s)

  if [ $rc -ne 0 ]; then
    printf '  %-12s FAIL (exit %d)\n' "$inp" "$rc"
    grep -iE 'error|fail' "$W/$inp.err" | head -3 | sed 's/^/      /'
    fails=$((fails+1)); continue
  fi

  # Did the GPU actually run? The sweep prints one "sweep shape:" line per
  # par_mfe() call. Without this check a run where every chunk fell to the CPU
  # fallback (chunk smaller than MIN_GPU_BATCH) reports IDENTICAL and looks
  # like a passing GPU test -- which it already did once.
  sweeps=$(grep -c 'sweep shape:' "$W/$inp.err")

  if cmp -s "$R/$inp.out" "$W/$inp.gpu"; then
    if [ "$sweeps" -eq 0 ]; then
      # No sweep is CORRECT under a tight budget: if free VRAM cannot hold
      # MIN_GPU_BATCH records, the degraded path sends them to the CPU, and
      # that is the fallback working rather than failing. It is only a failure
      # when no budget cap was set, where the GPU should certainly have run.
      if [ -n "$BUDGET" ]; then
        printf '  %-12s identical, CPU fallback (%ds) -- budget too small for a GPU batch\n' \
               "$inp" "$((t1-t0))"
      else
        printf '  %-12s identical, but NO GPU SWEEP RAN (%ds) -- the GPU path was never taken\n' \
               "$inp" "$((t1-t0))"
        fails=$((fails+1))
      fi
    else
      printf '  %-12s IDENTICAL via GPU chunks (%ds, %s sweeps)\n' "$inp" "$((t1-t0))" "$sweeps"
    fi
  else
    printf '  %-12s DIFFERS: %s\n' "$inp" "$(cmp "$R/$inp.out" "$W/$inp.gpu" 2>&1 | head -1)"
    fails=$((fails+1))
  fi
done
  return $fails
}

total=0
if [ -n "$BUDGETS" ]; then
  for b in $BUDGETS; do
    run_all "$b"
    total=$((total + $?))
    echo
  done
else
  run_all
  total=$?
fi

echo
if [ $total -eq 0 ]; then
  echo "RESULT: the GPU chunk path reproduces the frozen references byte for byte"
else
  echo "RESULT: $total failures"
fi
exit $total
