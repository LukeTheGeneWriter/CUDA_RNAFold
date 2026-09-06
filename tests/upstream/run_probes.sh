#!/bin/bash
# Build and run the three upstream-defect probes against a pristine ViennaRNA
# 2.7.2 tree. Nothing here is CUDA_RNAFold code -- every probe uses only the
# public RNAlib API, so it can be handed to the maintainers as-is.
#
# Usage: run_probes.sh [path-to-ViennaRNA-2.7.2] [path-to-gcc]
#
# Expected verdicts on 2.7.2 (measured 2026-09-05, WSL, 8 hardware threads):
#   A  aux_index_probe   BUG CONFIRMED (deterministic)
#   B1 race_mixed        RACE CONFIRMED, ~15% of tables wrong
#   B2 race_same         no mismatch  -- identical model details are unaffected
#   C1 race_window       CONFIRMED,    ~97% of parameter sets carry another
#                                      thread's window settings
#   C2 race_window1      no mismatch  -- single-threaded control, must be clean
set -u

V=${1:-$HOME/vrna27/ViennaRNA-2.7.2}
GCC=${2:-$HOME/miniforge3/bin/gcc}
HERE=$(cd "$(dirname "$0")" && pwd)
LIB=$V/src/ViennaRNA/libRNA.a
INC="-I$V/src -I$V/src/ViennaRNA -I$V"
W=${TMPDIR:-/tmp}/vrna_upstream_probes
mkdir -p "$W" || exit 2

if [ ! -f "$LIB" ]; then
  echo "no libRNA.a under $V -- build the pristine tree first" >&2
  exit 2
fi

build() { # build <output> <source> [extra flags...]
  local out=$1 src=$2; shift 2
  "$GCC" -O1 -o "$W/$out" "$HERE/$src" "$@" $INC "$LIB" -lm -lpthread -lstdc++ -fopenmp
}

build aux_index    aux_index_probe.c              || exit 2
build race_mixed   params_race_probe.c            || exit 2
build race_same    params_race_probe.c   -DNTEMP=1   || exit 2
build race_window  params_window_probe.c          || exit 2
build race_window1 params_window_probe.c -DNTHREAD=1 || exit 2

echo "###### A. auxiliary grammar rules receive the wrong i (mfe/mfe.c:502)"
"$W/aux_index"

echo
echo "###### B. SPEEDUP_PARAMS cache: concurrent callers, DIFFERENT model details"
"$W/race_mixed"

echo
echo "###### B control: concurrent callers, IDENTICAL model details (RNAfold -j)"
"$W/race_same"

echo
echo "###### C. SPEEDUP_PARAMS cache: same energy model, different window settings"
"$W/race_window"

echo
echo "###### C control: single-threaded, must report 0%"
"$W/race_window1"
