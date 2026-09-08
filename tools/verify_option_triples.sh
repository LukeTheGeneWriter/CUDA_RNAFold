#!/bin/bash
# One-off THREE-way probe. The pairwise audit is committed and green at 36/36;
# this asks the question that audit explicitly leaves open.
#
# Focused, not exhaustive: every triple here contains a FLOW dimension
# (RNA_SLOT_FLOW or a forced chunk split), because flow is where per-slot state
# lives and where both known interactions appeared -- int16+slotflow (stale
# baselines) and noLP+slotflow (a refill wiping live slots). A triple that
# breaks is far likelier to be there than among the answer-neutral knobs.
#
# Measure first, tool later: if this finds nothing, the matrix stays pairwise.
set -u
export PATH="$HOME/miniforge3/bin:$PATH"
B=${1:-$HOME/port27head}
IN=${2:-$HOME/rnatest/asc.fa}
BIN=$B/src/bin/RNAfold
export LD_LIBRARY_PATH=$(dirname "$(command -v nvcc)")/../lib
W=$HOME/triples; rm -rf $W; mkdir -p $W
NREC=$(grep -c '^>' "$IN")

cli_of() { case $1 in
  T25) echo "-T 25";; noGU) echo "--noGU";; salt) echo "--salt=0.2";; noLP) echo "--noLP";;
esac; }
env_of() { case $1 in
  slotflow) echo "RNA_SLOT_FLOW=2";; chunk12) echo "RNA_GPU_CHUNK=12";;
  minbatch1) echo "RNA_MIN_GPU_BATCH=1";; contflow) echo "RNA_CONTINUOUS_FLOW=1";;
esac; }

pass=0; fail=0; n=0
# CPU reference per CLI combination, computed once and reused.
cpu_ref() {
  local key=$1; shift
  [ -s "$W/ref_$key.out" ] && return
  env -u RNA_GPU_CHUNK -u RNA_SLOT_FLOW -u RNA_CONTINUOUS_FLOW -u RNA_MIN_GPU_BATCH \
      "$BIN" --noPS $* -i "$IN" > "$W/ref_$key.out" 2>/dev/null
}

check() {   # check "cliTagsCSV" "envTagsCSV"
  local ctags=$1 etags=$2
  local cli="" envs="" key=""
  for t in ${ctags//,/ }; do cli="$cli $(cli_of $t)"; key="${key}_$t"; done
  for t in ${etags//,/ }; do envs="$envs $(env_of $t)"; done
  local tag="${ctags},${etags}"
  n=$((n+1))
  cpu_ref "$key" $cli
  if [ ! -s "$W/ref_$key.out" ]; then printf '  %-34s CPU REF EMPTY\n' "$tag"; fail=$((fail+1)); return; fi
  env -u RNA_FML_INT16 -u RNA_SLOT_FLOW -u RNA_CONTINUOUS_FLOW -u RNA_MIN_GPU_BATCH \
      RNA_GPU_CHUNK=0 $envs "$BIN" --noPS $cli -i "$IN" > "$W/g.out" 2> "$W/g.err"
  local rc=$?
  local sw; sw=$(grep -c 'sweep shape:' "$W/g.err")
  if [ $rc -ne 0 ]; then printf '  %-34s *** EXIT %d ***\n' "$tag" $rc; fail=$((fail+1)); return; fi
  if [ "$(grep -c '^>' $W/g.out)" != "$NREC" ]; then
    printf '  %-34s *** SHORT OUTPUT ***\n' "$tag"; fail=$((fail+1)); return; fi
  if [ "$sw" -eq 0 ]; then
    printf '  %-34s *** NO SWEEPS -- folded on the CPU ***\n' "$tag"; fail=$((fail+1)); return; fi
  if cmp -s "$W/ref_$key.out" "$W/g.out"; then
    printf '  %-34s identical  [%s ch]\n' "$tag" "$sw"; pass=$((pass+1))
  else
    local nd; nd=$(paste <(grep -oE '\( *-?[0-9]+\.[0-9]+\)$' "$W/ref_$key.out") \
                        <(grep -oE '\( *-?[0-9]+\.[0-9]+\)$' "$W/g.out") \
                  | tr -d '()' | awk '{if($1+0!=$2+0)d++} END{print d+0}')
    printf '  %-34s *** DIFFERS on %s of %s records ***\n' "$tag" "$nd" "$NREC"
    fail=$((fail+1))
  fi
}

CLI=(T25 noGU salt noLP)
echo "=== two options + one flow switch ==="
for ((i=0;i<4;i++)); do for ((j=i+1;j<4;j++)); do
  for e in slotflow chunk12; do check "${CLI[i]},${CLI[j]}" "$e"; done
done; done

echo
echo "=== one option + two switches (flow x flow) ==="
for c in "${CLI[@]}"; do
  check "$c" "slotflow,chunk12"
  check "$c" "slotflow,minbatch1"
  check "$c" "slotflow,contflow"
done

echo
printf '%d triples: %d ok, %d failing\n' "$n" "$pass" "$fail"
exit $fail
