#!/bin/bash
# The bar for BOTH cell -> record lookup changes in int_loop_warp_kernel:
#
#   RNA_INT_LOOP_GRIDY=1    H6 -- blockIdx.y IS the record, so there is no
#                           lookup at all. Needs near-uniform widths and
#                           nfiles <= 65535, so the host declines it otherwise.
#   RNA_INT_LOOP_WSEARCH=1  H7 -- keep the flat grid, make the search 32-ary
#                           and warp-cooperative. Always applicable.
#
# TWO FIXTURES, because they test opposite things:
#   uniform  -- widths equal, the waste guard ACCEPTS, the 2-D grid runs
#   ngu.fa   -- 45..700 nt, the waste guard DECLINES, only H7 can help
#
# The first version of this bar used only ngu.fa, asserted that GRIDY was SET,
# and came back GREEN having never once taken the 2-D path. Assert the
# DECISION, not the knob.
GPU=${GPU:-$HOME/port27fml/src/bin/RNAfold}
REF=${REF:-$HOME/vrna27/ViennaRNA-2.7.2/src/bin/RNAfold}
U=${U:-$HOME/h6/uniform_small.fa}
S=${S:-$HOME/ngubar/ngu.fa}
E=/tmp/il_lookup.err
fail=0

if [ ! -s "$U" ]; then
  mkdir -p "$(dirname "$U")"
  python3 - > "$U" <<'PY'
import random
random.seed(28401)
for k in range(24):
    print(">u%d" % k)
    print("".join(random.choice("ACGU") for _ in range(400)))
PY
fi

surface() {   # $1 = env, $2 = fixture, $3 = what must be proven taken
  echo "=== vs pristine 2.7.2 over the option surface: $1"
  for arm in "" "--noLP" "-d0" "--noGU" "--noClosingGU" "-c" "-g" \
             "--maxBPspan=60" "-T 25" "-p"; do
    $REF --noPS $arm < "$2" > /tmp/il_r.out 2>/dev/null
    env RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 $1 \
        $GPU --noPS $arm < "$2" > /tmp/il_o.out 2> $E
    sw=$(grep -c 'sweep shape:' $E)
    tk=$(grep -c "$3" $E)
    d=$(diff /tmp/il_r.out /tmp/il_o.out | wc -l)
    printf '  %-16s sweeps=%-3s proof=%-3s diff=%s\n' "${arm:-(default)}" "$sw" "$tk" "$d"
    { [ "$d" -ne 0 ] || [ "$sw" -eq 0 ] || [ "$tk" -eq 0 ]; } && fail=1
  done
}

surface "RNA_INT_LOOP_GRIDY=1"   "$U" '2-D, blockIdx.y'
surface "RNA_INT_LOOP_WSEARCH=1" "$S" 'RNA_INT_LOOP_WSEARCH=1'

echo "=== byte-identity across both knobs, cells/block, encoding and chunking"
for fx in "$U" "$S"; do
  base=$(env RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 $GPU --noPS < "$fx" 2>/dev/null |
         sha256sum | cut -c1-12)
  echo "  $(basename "$fx"): reference sha (shipped default) = $base"
  for e in "RNA_INT_LOOP_GRIDY=1" \
           "RNA_INT_LOOP_WSEARCH=1" \
           "RNA_INT_LOOP_GRIDY=1 RNA_INT_LOOP_WSEARCH=1" \
           "RNA_INT_LOOP_WSEARCH=1 RNA_INT_LOOP_BLOCK_SIZE=64" \
           "RNA_INT_LOOP_WSEARCH=1 RNA_INT_LOOP_BLOCK_SIZE=128" \
           "RNA_INT_LOOP_WSEARCH=1 RNA_INT_LOOP_BLOCK_SIZE=256" \
           "RNA_INT_LOOP_WSEARCH=1 RNA_FML_INT16=1" \
           "RNA_INT_LOOP_WSEARCH=1 RNA_GPU_VRAM_BUDGET_MB=64" \
           "RNA_INT_LOOP_WSEARCH=1 RNA_INT_LOOP_WARP=0" \
           "RNA_INT_LOOP_WSEARCH=1 RNA_BUILD_THREADS=1"; do
    a=$(env RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 $e $GPU --noPS < "$fx" 2>/dev/null |
        sha256sum | cut -c1-12)
    printf '    %-52s sha=%s\n' "$e" "$a"
    [ "$a" = "$base" ] || { echo "      *** sha moved"; fail=1; }
  done
done

echo "=== the 2-D grid's waste guard must DECLINE the ragged chunk"
env RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 RNA_INT_LOOP_GRIDY=1 \
    $GPU --noPS < "$S" > /dev/null 2> $E
dec=$(grep -c 'waste guard declined' $E)
printf '  ngu.fa declined=%s\n' "$dec"
[ "$dec" -eq 0 ] && fail=1

echo "result: $([ $fail -eq 0 ] && echo GREEN || echo RED)"
