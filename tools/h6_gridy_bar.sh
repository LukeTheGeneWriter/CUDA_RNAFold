#!/bin/bash
# H6 bar. Two fixtures, because they test opposite things:
#   uniform.fa  -- widths equal, the waste guard ACCEPTS, the 2-D grid runs
#   ngu.fa      -- 45..700 nt, the waste guard DECLINES, the flat grid runs
# The first bar I wrote used only ngu.fa and came back GREEN having never once
# taken the new path. Assert the DECISION, not the knob.
GPU=$HOME/port27fml/src/bin/RNAfold
REF=$HOME/vrna27/ViennaRNA-2.7.2/src/bin/RNAfold
U=$HOME/h6/uniform_small.fa   # 24 x 400 nt: uniform widths so the guard
                              # ACCEPTS, small enough that pristine 2.7.2 can
                              # fold ten option arms of it in reasonable time
S=$HOME/ngubar/ngu.fa
E=/tmp/h6bar.err
fail=0

if [ ! -s "$U" ]; then
  mkdir -p "$HOME/h6"
  python3 - > "$U" <<'PY'
import random
random.seed(28401)
for k in range(24):
    print(">u%d" % k)
    print("".join(random.choice("ACGU") for _ in range(400)))
PY
fi

echo "=== the 2-D grid vs pristine 2.7.2, option surface (uniform fixture)"
for arm in "" "--noLP" "-d0" "--noGU" "--noClosingGU" "-c" "-g" "--maxBPspan=60" "-T 25" "-p"; do
  $REF --noPS $arm < $U > /tmp/r.out 2>/dev/null
  env RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 RNA_INT_LOOP_GRIDY=1 \
      $GPU --noPS $arm < $U > /tmp/o.out 2> $E
  sw=$(grep -c 'sweep shape:' $E)
  g2=$(grep -c '2-D, blockIdx.y' $E)
  d=$(diff /tmp/r.out /tmp/o.out | wc -l)
  printf '  %-16s sweeps=%-3s took-2D=%s diff=%s\n' "${arm:-(default)}" "$sw" "$g2" "$d"
  { [ "$d" -ne 0 ] || [ "$sw" -eq 0 ] || [ "$g2" -eq 0 ]; } && fail=1
done

echo "=== byte-identity across cells/block, encoding and chunking (uniform)"
base=$(env RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 $GPU --noPS < $U 2>/dev/null |
       sha256sum | cut -c1-12)
echo "  reference sha (flat grid, shipped default) = $base"
for e in "RNA_INT_LOOP_GRIDY=1" \
         "RNA_INT_LOOP_GRIDY=1 RNA_INT_LOOP_BLOCK_SIZE=64" \
         "RNA_INT_LOOP_GRIDY=1 RNA_INT_LOOP_BLOCK_SIZE=128" \
         "RNA_INT_LOOP_GRIDY=1 RNA_INT_LOOP_BLOCK_SIZE=256" \
         "RNA_INT_LOOP_GRIDY=1 RNA_FML_INT16=1" \
         "RNA_INT_LOOP_GRIDY=1 RNA_GPU_VRAM_BUDGET_MB=64" \
         "RNA_INT_LOOP_GRIDY=1 RNA_INT_LOOP_WARP=0" \
         "RNA_INT_LOOP_GRIDY=1 RNA_BUILD_THREADS=1"; do
  a=$(env RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 $e $GPU --noPS < $U 2>$E |
      sha256sum | cut -c1-12)
  g2=$(grep -c '2-D, blockIdx.y' $E)
  printf '  %-52s sha=%s took-2D=%s\n' "$e" "$a" "$g2"
  [ "$a" = "$base" ] || { echo "    *** sha moved"; fail=1; }
done

echo "=== the waste guard must DECLINE a ragged chunk, and still be right"
$REF --noPS < $S > /tmp/rr.out 2>/dev/null
env RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 RNA_INT_LOOP_GRIDY=1 \
    $GPU --noPS < $S > /tmp/or.out 2> $E
d=$(diff /tmp/rr.out /tmp/or.out | wc -l)
sw=$(grep -c 'sweep shape:' $E)
dec=$(grep -c 'waste guard declined' $E)
printf '  ngu.fa (45..700 nt)  sweeps=%-3s declined=%s diff=%s\n' "$sw" "$dec" "$d"
{ [ "$d" -ne 0 ] || [ "$sw" -eq 0 ] || [ "$dec" -eq 0 ]; } && fail=1

echo "result: $([ $fail -eq 0 ] && echo GREEN || echo RED)"
