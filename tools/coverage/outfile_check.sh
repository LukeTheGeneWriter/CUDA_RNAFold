#!/bin/bash
# Output-file bar: what RNAfold WRITES, not only its stdout. Every other bar runs --noPS
# without --outfile, so none of them reached these paths (found 2026-10-14) -- and the
# backtrack pipeline (N5a P2) moves output dispatch into a callback, after the next sweep.
#   modes: per-record --outfile, one --outfile=NAME, PostScript plots (no --noPS), and
#          per-record --outfile with -j4
#   arms:  CPU (RNA_GPU=0), GPU default, GPU multi-chunk, multi-chunk + RNA_BT_PIPELINE=1,
#          the same with half RAM/VRAM/thread budgets, and RNA_BT_PIPELINE_NEGCTL=1, which must
#          CHANGE the files (a bar that cannot fail proves nothing)
# Every arm must write the same SET of files with the same contents as the CPU (PostScript
# %%CreationDate/%%Creator lines aside) and the same stdout. Positive evidence: the files
# exist in the expected number, the device swept (and in several chunks where asked), the
# pipeline traced chunks beside a sweep where it may, and it says it stayed sequential
# where the output path builds compounds of its own (plots).
# usage: TREE=<build> [W=<workdir>] [COV_FIX=<fixtures>] outfile_check.sh
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniforge | paste -sd:)
for v in $(env | grep -o "^RNA_[A-Z0-9_]*"); do unset $v; done
[ -n "${BARS_ENV:-}" ] && export $BARS_ENV   # an experimental arm's variables (bars.sh), after the scrub
R=${TREE:-$HOME/pyq}/src/bin/RNAfold; T=${COV_FIX:-$(cd "$(dirname "$0")/../.." && pwd)/tests/fixtures}
W=${W:-$HOME/outfile_check}; rm -rf "$W"; mkdir -p "$W"
FA=$T/C_mixed.fa; NREC=$(grep -c '^>' "$FA")
MULTI="RNA_GPU_CHUNK_CELLS=300000 RNA_GPU_WORK_FLOOR=0"
ARMS=("cpu|RNA_GPU=0" "gpu|" "multi|$MULTI" "pipe|$MULTI RNA_BT_PIPELINE=1"
      "pipe_half|$MULTI RNA_BT_PIPELINE=1 RNA_RAM_BUDGET=half RNA_VRAM_BUDGET=half RNA_THREAD_BUDGET=half"
      "negctl|$MULTI RNA_BT_PIPELINE=1 RNA_BT_PIPELINE_NEGCTL=1")
MODES=("per_rec|--noPS --outfile|$NREC" "one_file|--noPS --outfile=all|1" "ps||$NREC" "per_rec_j4|--noPS --outfile -j4|$NREC")
bad=0

digest() {   # every file in a directory, by name and content; PostScript's date and creator lines aside
  (cd "$1" && for f in $(ls | sort); do
     printf "%s %s\n" "$f" "$(grep -av '^%%CreationDate\|^%%Creator' "$f" | sha1sum | cut -c1-12)"
   done) | sha1sum | cut -c1-12
}

for m in "${MODES[@]}"; do
  IFS='|' read -r mode margs nwant <<< "$m"
  ref_d=""; ref_o=""
  for a in "${ARMS[@]}"; do
    IFS='|' read -r arm aenv <<< "$a"
    d=$W/$mode/$arm; mkdir -p "$d"
    ( cd "$d" && env RNA_GPU_VERBOSE=1 $aenv "$R" $margs < "$FA" > ../$arm.stdout 2> ../$arm.err ); rc=$?
    dg=$(digest "$d"); os=$(sha1sum < $W/$mode/$arm.stdout | cut -c1-12); nf=$(ls "$d" | wc -l)
    e=$W/$mode/$arm.err; sw=$(grep -c 'sweep shape' "$e"); ev=""; v=ok
    [ -z "$ref_d" ] && { ref_d=$dg; ref_o=$os; }
    if [ "$arm" = negctl ]; then   # the bar must be able to fail: a zeroed retained triangle must show
      { [ "$dg" != "$ref_d" ] || [ "$os" != "$ref_o" ]; } && ev="bites" || { v=FAIL; ev="DOES NOT BITE"; }
    else
      { [ $rc -eq 0 ] && [ "$dg" = "$ref_d" ] && [ "$os" = "$ref_o" ] && [ "$nf" -eq "$nwant" ]; } || v=FAIL
    fi
    case $arm in
      cpu) [ "$sw" -eq 0 ] || { v=FAIL; ev="the CPU arm swept"; } ;;
      gpu) [ "$sw" -ge 1 ] || { v=FAIL; ev="no sweep"; } ;;
      *)   [ "$sw" -ge 2 ] || { v=FAIL; ev="not multi-chunk ($sw sweeps)"; } ;;
    esac
    case $arm in pipe*|negctl)
      if [ "$mode" = ps ]; then
        grep -q "backtracks stay sequential" "$e" || { v=FAIL; ev="$ev; plots did not refuse the pipeline"; }
      else
        n=$(grep -o '[0-9]* traced beside the next sweep' "$e" | grep -o '^[0-9]*')
        [ "${n:-0}" -ge 1 ] || { v=FAIL; ev="$ev; nothing traced beside a sweep"; }
        ev="$ev traced-beside ${n:-0}"
      fi ;;
    esac
    case $arm in pipe_half)
      grep -q "budget: RAM" "$e" || { v=FAIL; ev="$ev; no budget line"; } ;;
    esac
    printf "%-11s %-10s rc %d files %3d/%-3d dir %s stdout %s sweeps %3d %s [%s]\n" \
           "$mode" "$arm" $rc "$nf" "$nwant" "$dg" "$os" "$sw" "$ev" "$v"
    [ "$v" = ok ] || bad=$((bad + 1))
  done
done
echo "FAILURES: $bad"
[ $bad -eq 0 ]
