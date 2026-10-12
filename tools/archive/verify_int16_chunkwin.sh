#!/bin/bash
# Arm F's whole premise: at the SAME VRAM budget, does int16's halved footprint
# translate into fewer chunks? If it does not, arm F measures nothing.
SP=/mnt/c/Users/lukef/AppData/Local/Temp/claude/C--Users-lukef-CUDA-RNAFold/cee8a9b1-e71a-4163-bb4f-ece1313aa24f/scratchpad
. $SP/env.sh
BIN=$REL/src/bin/RNAfold
W=$(mktemp -d); trap 'rm -rf $W' EXIT

python3 - "$W" <<'PY'
import sys, random
w = sys.argv[1]; random.seed(23)
with open(f"{w}/u.fa","w") as f:
    for i in range(60):
        f.write(f">u{i}\n{''.join(random.choice('ACGU') for _ in range(900))}\n")
PY

printf '%8s %10s %10s %10s %12s\n' budget int32 int16 ratio identical?
for mb in 64 48 32 24 16; do
  RNA_GPU_CHUNK=0 RNA_GPU_VRAM_BUDGET_MB=$mb $BIN --noPS -i $W/u.fa \
      > $W/a.$mb 2> $W/a.$mb.err
  RNA_GPU_CHUNK=0 RNA_GPU_VRAM_BUDGET_MB=$mb RNA_FML_INT16=1 $BIN --noPS -i $W/u.fa \
      > $W/b.$mb 2> $W/b.$mb.err
  c1=$(grep -c 'sweep shape:' $W/a.$mb.err); c1=${c1:-0}
  c2=$(grep -c 'sweep shape:' $W/b.$mb.err); c2=${c2:-0}
  g=$(grep -c 'RNA_FML_INT16=1' $W/b.$mb.err); g=${g:-0}
  if cmp -s $W/a.$mb $W/b.$mb; then same=IDENTICAL; else same="*** DIFFERS ***"; fi
  [ "$g" -eq 0 ] && same="$same (GATE OFF!)"
  printf '%6sMB %10s %10s %9s %14s\n' "$mb" "$c1 chunks" "$c2 chunks" \
         "$(python3 -c "print(f'{$c1/max($c2,1):.2f}x')")" "$same"
done
