#!/bin/bash
# One piece of the default-path coverage workload (PORT_CLEANUP_PLAN.md §1).
# Counts accumulate in the .gcda files across pieces, so run one piece per job --
# a laptop's WSL memory reaper stops long jobs.
#
# usage: part.sh reset | parity | <fixture>
#   COV_TREE    coverage tree (default ~/covq)
#   COV_FIX     fixture directory holding <fixture>.fa (default ~/rnatest)
#   COV_BUDGET  sets RNA_GPU_VRAM_BUDGET_MB, to force a multi-chunk run on a small card
#   COV_CPU     non-empty: the CPU arm (RNA_GPU=0); use a separate tree or reset between arms
# Every other RNA_* variable is unset: this is the DEFAULT path.
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniforge | paste -sd:)
for v in $(env | grep -o "^RNA_[A-Z0-9_]*"); do unset $v; done
[ -n "$COV_BUDGET" ] && export RNA_GPU_VRAM_BUDGET_MB=$COV_BUDGET
[ -n "$COV_CPU" ] && export RNA_GPU=0
REPO=$(cd "$(dirname "$0")/../.." && pwd)
B=${COV_TREE:-~/covq}; R=$B/src/bin/RNAfold; T=${COV_FIX:-~/rnatest}; W=$B/covrun
mkdir -p "$W"; cd "$W"
case "$1" in
  reset)  find "$B/src" -name "*.gcda" -delete; rm -f "$W"/*; echo reset;;
  parity) bash "$REPO/tools/verify_option_parity.sh" "$B" > parity.log 2>&1; echo "parity rc $?"; tail -1 parity.log
          echo "gcov version mismatches: $(grep -c 'Version mismatch' parity.log)";;
  *) f=$1; [ -f "$T/$f.fa" ] || { echo "no $T/$f.fa"; exit 1; }
     $R --noPS < "$T/$f.fa" > $f.out 2> $f.err; a=$?
     $R --noPS -j4 < "$T/$f.fa" > $f.j4.out 2> $f.j4.err; b=$?
     echo "$f rc $a/$b sweeps $(grep -c "sweep shape" $f.err) same_j4 $(cmp -s $f.out $f.j4.out && echo yes || echo NO) gcov_err $(cat $f.err $f.j4.err | grep -c 'profiling error')";;
esac
