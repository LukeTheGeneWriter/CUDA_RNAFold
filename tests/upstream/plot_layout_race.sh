#!/bin/bash
# Defect E reproducer: RNAfold -j with structure plots races in vrna_plot_layout().
# The default layout (naview) keeps its working state in file-scope statics, and RNAfold
# computes the layout OUTSIDE the lock that serialises the plot write
# (src/bin/RNAfold.c, postscript_layout() and ImFeelingLucky()).
# Nothing here is CUDA_RNAFold code: point it at any RNAfold.
#
# Usage: plot_layout_race.sh [path-to-RNAfold] [runs]
#
# Measured 2026-10-14, WSL, 12 hardware threads, pristine 2.7.2, this script's fixture
# (300 records 400-800 nt), 10 runs per arm:
#   j8_plots   7/10 clean (2 segfaults, 1 abort; 75-265 of 300 plots written)
#   j8_noPS   10/10 clean -- control: threads, no layout
#   j1_plots  10/10 clean -- control: layout, no threads
# The race is timing-dependent: another fixture of the same shape crashed 5 of 5. With the
# layout moved inside the lock, all three arms are 10/10 clean on the same fixture.
set -u
U=${1:-$HOME/vrna27/ViennaRNA-2.7.2/src/bin/RNAfold}; N=${2:-10}
[ -x "$U" ] || { echo "no RNAfold at $U"; exit 2; }
W=${TMPDIR:-/tmp}/vrna_plot_layout_race; rm -rf "$W"; mkdir -p "$W"; cd "$W" || exit 2
awk 'BEGIN { srand(20261014); for (i = 0; i < 300; i++) { L = 400 + int(rand() * 401); s = "";
             for (k = 0; k < L; k++) s = s substr("ACGU", 1 + int(rand() * 4), 1); printf(">d%d\n%s\n", i, s) } }' > in.fa
"$U" --version
"$U" --noPS -j1 < in.fa | sort > ref.sorted
for arm in "j8_plots|-j8" "j8_noPS|-j8 --noPS" "j1_plots|-j1"; do
  IFS='|' read -r lab args <<< "$arm"; ok=0; fails=""
  for k in $(seq 1 "$N"); do
    rm -rf o && mkdir o && cd o
    timeout 300 "$U" $args < ../in.fa > ../out 2> ../err; rc=$?; cd ..
    if [ $rc -eq 0 ] && sort out | cmp -s - ref.sorted; then ok=$((ok + 1))
    else fails="$fails rc=$rc/plots=$(ls o | wc -l)"; fi
  done 2> /dev/null
  printf "%-9s %d/%d clean %s\n" "$lab" "$ok" "$N" "$fails"
done
