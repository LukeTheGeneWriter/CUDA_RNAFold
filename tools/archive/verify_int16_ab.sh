#!/bin/bash
# The A/B the gate exists for: one binary, folded both ways, compared against
# ITSELF. Same shape that settled RNA_GPU_SWEEP.
SP=/mnt/c/Users/lukef/AppData/Local/Temp/claude/C--Users-lukef-CUDA-RNAFold/cee8a9b1-e71a-4163-bb4f-ece1313aa24f/scratchpad
. $SP/env.sh
BIN=$REL/src/bin/RNAfold
T=$HOME/rnatest
W=$(mktemp -d); trap 'rm -rf $W' EXIT

fail=0
for inp in C_control asc desc C_mixed extreme; do
  [ -f "$T/$inp.fa" ] || continue
  n=$(grep -c '^>' $T/$inp.fa)

  RNA_GPU_CHUNK=0 $BIN --noPS -i $T/$inp.fa > $W/$inp.off 2> $W/$inp.off.err
  rc1=$?
  RNA_GPU_CHUNK=0 RNA_FML_INT16=1 $BIN --noPS -i $T/$inp.fa > $W/$inp.on 2> $W/$inp.on.err
  rc2=$?

  s1=$(grep -c 'sweep shape:' $W/$inp.off.err); s1=${s1:-0}
  s2=$(grep -c 'sweep shape:' $W/$inp.on.err);  s2=${s2:-0}
  gated=$(grep -c 'RNA_FML_INT16=1' $W/$inp.on.err); gated=${gated:-0}

  if [ $rc2 -ne 0 ]; then
    printf '  %-11s FAIL exit %d\n' "$inp" "$rc2"
    grep -m3 -i 'range\|assert\|error' $W/$inp.on.err | head -3
    fail=$((fail+1)); continue
  fi
  if [ "$gated" -eq 0 ]; then
    printf '  %-11s FAIL -- the gate never announced itself; int16 was NOT active\n' "$inp"
    fail=$((fail+1)); continue
  fi
  if [ ! -s $W/$inp.off ] || [ ! -s $W/$inp.on ]; then
    printf '  %-11s FAIL -- empty output\n' "$inp"; fail=$((fail+1)); continue
  fi
  if cmp -s $W/$inp.off $W/$inp.on; then
    printf '  %-11s IDENTICAL  (%s records, %s sweeps off / %s on)\n' "$inp" "$n" "$s1" "$s2"
  else
    printf '  %-11s DIFFERS\n' "$inp"
    diff $W/$inp.off $W/$inp.on | head -6
    fail=$((fail+1))
  fi
done
echo
[ $fail -eq 0 ] && echo "RESULT: int16 reproduces the int32 path byte for byte" \
                || echo "RESULT: FAILED ($fail)"
exit $fail
