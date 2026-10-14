#!/bin/bash
# Output-file bar: what RNAfold WRITES, not only its stdout. The other bars run --noPS
# without --outfile, so none of them reached these paths (found on Lukes_Flow_Batching
# 2026-10-14; this is its Finished_Port form).
#   modes: per-record --outfile, one --outfile=NAME, PostScript plots (no --noPS), and
#          per-record --outfile with -j4
#   arms:  CPU (RNA_GPU=0), GPU default, GPU in several chunks (RNA_GPU_CHUNK=6), and a
#          negative control: the GPU at -T 36, which must CHANGE the files (a bar that
#          cannot fail proves nothing). This branch has no backtrack pipeline, so the control
#          is on the comparison, not on a retained set.
# Every arm must write the same SET of files with the same contents as the CPU (PostScript
# %%CreationDate/%%Creator lines aside) and the same stdout. Positive evidence: the files
# exist in the expected number, and the device swept (in several chunks where asked).
# usage: TREE=<build> FA=<fixture> [W=<workdir>] tools/outfile_check.sh
#   FA: any multi-record FASTA; Lukes_Flow_Batching's tests/fixtures/C_mixed.fa (40 records,
#   300-800 nt) is the one the bar was written against.
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniforge | paste -sd:)
for v in $(env | grep -o "^RNA_[A-Z0-9_]*"); do unset $v; done
R=${TREE:?TREE=<build tree>}/src/bin/RNAfold; FA=${FA:?FA=<fixture>}
W=${W:-$HOME/outfile_check_fp}; rm -rf "$W"; mkdir -p "$W"
NREC=$(grep -c '^>' "$FA")
MULTI="RNA_GPU_CHUNK=6 RNA_GPU_WORK_FLOOR=0"
ARMS=("cpu|RNA_GPU=0|" "gpu||" "multi|$MULTI|" "negctl|$MULTI|-T 36")
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
    IFS='|' read -r arm aenv aargs <<< "$a"
    d=$W/$mode/$arm; mkdir -p "$d"
    ( cd "$d" && env $aenv "$R" $margs $aargs < "$FA" > ../$arm.stdout 2> ../$arm.err ); rc=$?
    dg=$(digest "$d"); os=$(sha1sum < $W/$mode/$arm.stdout | cut -c1-12); nf=$(ls "$d" | wc -l)
    e=$W/$mode/$arm.err; sw=$(grep -c 'sweep shape' "$e"); ev=""; v=ok
    [ -z "$ref_d" ] && { ref_d=$dg; ref_o=$os; }
    if [ "$arm" = negctl ]; then   # the comparison must be able to fail: different answers must show in the FILES
      # (not stdout: with --outfile it is empty in every arm)
      [ "$dg" != "$ref_d" ] && ev="bites" || { v=FAIL; ev="DOES NOT BITE"; }
      [ $rc -eq 0 ] && [ "$nf" -eq "$nwant" ] || v=FAIL
    else
      { [ $rc -eq 0 ] && [ "$dg" = "$ref_d" ] && [ "$os" = "$ref_o" ] && [ "$nf" -eq "$nwant" ]; } || v=FAIL
    fi
    case $arm in
      cpu) [ "$sw" -eq 0 ] || { v=FAIL; ev="the CPU arm swept"; } ;;
      gpu) [ "$sw" -ge 1 ] || { v=FAIL; ev="no sweep"; } ;;
      *)   [ "$sw" -ge 2 ] || { v=FAIL; ev="$ev; not multi-chunk ($sw sweeps)"; } ;;
    esac
    printf "%-11s %-7s rc %d files %3d/%-3d dir %s stdout %s sweeps %3d %s [%s]\n" \
           "$mode" "$arm" $rc "$nf" "$nwant" "$dg" "$os" "$sw" "$ev" "$v"
    [ "$v" = ok ] || bad=$((bad + 1))
  done
done
echo "FAILURES: $bad"
[ $bad -eq 0 ]
