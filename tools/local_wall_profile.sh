#!/bin/bash
# Local wall decomposition, for sizing the build/GPU pipeline (option B).
#
# This box is the OPPOSITE corner from Colab: 12 cores and a slow, throttled
# laptop GPU, where Colab's T4 instances are core-starved with a faster device.
# The pipeline's prize is min(build, gpu) per chunk, so the two hosts bracket the
# design -- if B pays here it pays everywhere, and the ratio tells us how deep the
# pipeline needs to be.
#
# Usage: local_wall_profile.sh [build-tree]
set -u
B=${1:-$HOME/port27head}
BIN=$B/src/bin/RNAfold
W=${TMPDIR:-/tmp}/lwp
rm -rf "$W"; mkdir -p "$W"

nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv,noheader
echo "cores: $(nproc)"
echo

gen() {
  python3 - "$1" "$2" "$3" <<'PY'
import random, sys
n, L, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
random.seed(777 + L)
with open(out, "w") as f:
    for i in range(n):
        f.write(">r%04d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(L))))
PY
}

run() { # run N LEN [budget_mb]
  local n=$1 L=$2 bud=${3:-} tag="${1}x${2}${3:+_b$3}"
  gen "$n" "$L" "$W/$tag.fa"
  if [ -n "$bud" ]; then
    RNA_GPU_CHUNK=0 RNA_GPU_VRAM_BUDGET_MB="$bud" "$BIN" --noPS -i "$W/$tag.fa" \
      > /dev/null 2> "$W/$tag.err"
  else
    RNA_GPU_CHUNK=0 "$BIN" --noPS -i "$W/$tag.fa" > /dev/null 2> "$W/$tag.err"
  fi
  python3 - "$W/$tag.err" "$tag" "$n" <<'PY'
import re, sys
err = open(sys.argv[1], errors="replace").read()
tag, n = sys.argv[2], int(sys.argv[3])
ph = re.search(r"int_loop=([\d.]+) hp_mb=([\d.]+) load_my_c=([\d.]+) "
               r"modular_decomp=([\d.]+) fetch_mx=([\d.]+)", err)
st = re.search(r"build=([\d.]+) prepare=([\d.]+) prefill=([\d.]+) backtrack=([\d.]+) "
               r"output=([\d.]+) gpuinit=([\d.]+) teardown=([\d.]+) free=([\d.]+)", err)
chunks = len(re.findall(r"sweep shape:", err))
if not (ph and st):
    print("  %-14s NO TIMER LINES -- did the GPU path run?" % tag); sys.exit()
il, hm, lm, md, fx = map(float, ph.groups())
bd, pr, pf, bt, ou, gi, td, fr = map(float, st.groups())
gpu = il + hm + md
host = bd + bt + ou + gi + pr + td + fr
tot = gpu + lm + fx + host
print("  %-14s chunks=%-3s  GPU=%6.2f  build=%6.2f  backtrack=%5.2f  "
      "other_host=%5.2f  sum=%6.2f" % (tag, chunks, gpu, bd, bt, host-bd-bt, tot))
print("  %-14s   build/GPU = %.2f   per chunk: build=%.2f gpu=%.2f  "
      "-> pipeline could hide %.2f s (%.0f%% of sum)"
      % ("", bd/gpu if gpu else 0, bd/max(chunks,1), gpu/max(chunks,1),
         (chunks-1)*min(bd/max(chunks,1), gpu/max(chunks,1)) if chunks else 0,
         100*(chunks-1)*min(bd/max(chunks,1), gpu/max(chunks,1))/tot if chunks and tot else 0))
PY
}

echo "== single chunk (no pipeline possible, for the ratio) =="
run 24 2000
run 24 3000
echo
echo "== forced multi-chunk: this is where a pipeline pays =="
run 48 2000 900
run 48 2000 500
run 64 1500 400
