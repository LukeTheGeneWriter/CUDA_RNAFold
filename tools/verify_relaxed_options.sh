#!/bin/bash
# The bar behind relaxing a guard.
#
# A guard is only lifted on evidence, and 12 random sequences is not evidence --
# it is a sample that can miss a data-dependent divergence. This runs the
# newly-permitted options over the FULL reference set (239 records, 20-8001 nt,
# the same inputs the byte-identical bar uses), GPU route versus CPU route, and
# confirms both that the answers match AND that the GPU was actually used.
#
# Usage: verify_relaxed_options.sh [build-tree]
set -u
export PATH="$HOME/miniforge3/bin:$PATH"
B=${1:-$HOME/port27cuda}
BIN=$B/src/bin/RNAfold
T=$HOME/rnatest
CUDA_LIBDIR=$(dirname "$(command -v nvcc)")/../lib
export LD_LIBRARY_PATH=$CUDA_LIBDIR
W=$HOME/relaxed
rm -rf "$W"; mkdir -p "$W" || exit 2

[ -x "$BIN" ] || { echo "no RNAfold at $BIN" >&2; exit 2; }
echo "binary: $BIN"
echo

fail=0
# logML dropped: RNAfold has no --logML flag, so it cannot be barred from the
# CLI at all; it stays declined in the library guard until it has a real bar.
for opt_spec in "uniqML:-p" "noGU:--noGU"; do
  tag=${opt_spec%%:*}
  flag=${opt_spec#*:}
  echo "--- $tag  ($flag)"
  for inp in C_mixed C_control asc desc extreme; do
    [ -f "$T/$inp.fa" ] || continue
    n=$(grep -c '^>' "$T/$inp.fa")

    "$BIN" --noPS $flag -i "$T/$inp.fa" > "$W/$tag.$inp.cpu" 2>/dev/null
    RNA_GPU_CHUNK=0 "$BIN" --noPS $flag -i "$T/$inp.fa" \
        > "$W/$tag.$inp.gpu" 2> "$W/$tag.$inp.err"
    rc=$?
    sweeps=$(grep -c 'sweep shape:' "$W/$tag.$inp.err")

    if [ $rc -ne 0 ]; then
      printf '    %-11s FAIL (exit %d) %s\n' "$inp" "$rc" \
             "$(grep -m1 -iE 'unrecognized|invalid' "$W/$tag.$inp.err" | head -c 60)"
      fail=$((fail+1))
    elif [ ! -s "$W/$tag.$inp.cpu" ]; then
      # empty vs empty is not a match -- see the same guard in
      # verify_option_parity.sh
      printf '    %-11s NO OUTPUT -- option rejected, nothing was tested\n' "$inp"
      fail=$((fail+1))
    elif ! cmp -s "$W/$tag.$inp.cpu" "$W/$tag.$inp.gpu"; then
      printf '    %-11s DIFFERS\n' "$inp"; fail=$((fail+1))
    elif [ "$sweeps" -eq 0 ]; then
      printf '    %-11s identical but NO GPU SWEEP -- still being routed away\n' "$inp"
      fail=$((fail+1))
    else
      printf '    %-11s identical, %s records, %s GPU sweeps\n' "$inp" "$n" "$sweeps"
    fi
  done
  echo
done

if [ $fail -eq 0 ]; then
  echo "RESULT: the relaxed options run ON THE GPU and match the CPU route"
else
  echo "RESULT: $fail failures -- do not relax these guards"
fi
exit $fail
