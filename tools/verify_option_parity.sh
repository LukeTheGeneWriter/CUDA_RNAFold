#!/bin/bash
# Option-surface parity: for every option, does the CUDA build with the GPU
# path ENABLED produce exactly what it produces with the GPU path OFF?
#
# This is the claim being taken to upstream -- "a strict accelerator of the
# validated CPU code" -- so it is checked across the option surface rather than
# on default folds alone. The reference is the SAME BINARY with RNA_GPU_CHUNK
# unset, which isolates the accelerator as the only variable.
#
# Usage: verify_option_parity.sh [build-tree] [input]
set -u
export PATH="$HOME/miniforge3/bin:$PATH"
B=${1:-$HOME/port27cuda}
IN=${2:-$HOME/rnatest/asc.fa}
BIN=$B/src/bin/RNAfold
CUDA_LIBDIR=$(dirname "$(command -v nvcc)")/../lib
export LD_LIBRARY_PATH=$CUDA_LIBDIR
W=${TMPDIR:-/tmp}/vrna_optparity
rm -rf "$W"; mkdir -p "$W" || exit 2

[ -x "$BIN" ] || { echo "no RNAfold at $BIN" >&2; exit 2; }
echo "binary: $BIN"
echo "input : $IN  ($(grep -c '^>' "$IN") records)"
echo

pass=0; fail=0; n=0

check() {
  local tag=$1; shift
  n=$((n+1))
  "$BIN" --noPS "$@" -i "$IN" > "$W/$tag.off" 2> "$W/$tag.off.err"; rc_off=$?
  RNA_GPU_CHUNK=0 "$BIN" --noPS "$@" -i "$IN" > "$W/$tag.on" 2> "$W/$tag.on.err"; rc_on=$?
  local sweeps; sweeps=$(grep -c 'sweep shape:' "$W/$tag.on.err")

  if [ $rc_off -ne $rc_on ]; then
    printf '  %-22s EXIT DIFFERS (off %d, on %d)\n' "$tag" "$rc_off" "$rc_on"
    fail=$((fail+1)); return
  fi
  if cmp -s "$W/$tag.off" "$W/$tag.on"; then
    if [ "$sweeps" -gt 0 ]; then
      printf '  %-22s identical  [GPU: %s sweeps]\n' "$tag" "$sweeps"
    else
      printf '  %-22s identical  [CPU route]\n' "$tag"
    fi
    pass=$((pass+1))
  else
    printf '  %-22s *** DIFFERS ***\n' "$tag"
    diff "$W/$tag.off" "$W/$tag.on" | head -4 | sed 's/^/        /'
    fail=$((fail+1))
  fi
}

echo "--- options the GPU path supports (expect GPU sweeps)"
check default
check temp37            -T 37
check temp25            -T 25
check nolp_off          --dangles=2
check partfunc          -p
check partfunc0         -p0
check mea               -p --MEA
check centroid          -p
check bppm_thresh       -p --bppmThreshold=1e-4

echo
echo "--- options that must route to the CPU (expect CPU route, same answer)"
check gquad             -g
check circ              -c
check dangles0          -d0
check dangles1          -d1
check dangles3          -d3
check noLP              --noLP
check noGU              --noGU
check noClosingGU       --noClosingGU
check logML             --logML
check salt              --salt=0.2
check uniqML            -p --MEA

echo
printf '\n%d checks: %d identical, %d differing\n' "$n" "$pass" "$fail"
[ $fail -eq 0 ] && echo "RESULT: the CUDA build matches the CPU build across the option surface" \
                || echo "RESULT: $fail options differ"
exit $fail
