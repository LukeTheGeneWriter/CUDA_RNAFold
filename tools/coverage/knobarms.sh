#!/bin/bash
# Cleanup bar for the OFF-default arms that survive a phase (PORT_CLEANUP_PLAN.md §6).
# refcheck.sh unsets every RNA_* knob, so it only ever sees the default path; this
# folds each fixture again with one arm's knobs set, requires the output to equal the
# same frozen reference, and requires stderr to show the arm took the path it claims
# (a knob that is silently ignored would pass the byte comparison by construction).
# usage: knobarms.sh <fixture> [budget_mb]
#   REF_DIR, RNAFOLD, COV_FIX  as in refcheck.sh
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniforge | paste -sd:)
for v in $(env | grep -o "^RNA_[A-Z0-9_]*"); do unset $v; done
f=$1; [ -n "$2" ] && export RNA_GPU_VRAM_BUDGET_MB=$2
R=${RNAFOLD:-~/pyq/src/bin/RNAfold}; O=${REF_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/tests/fixtures/cleanup_ref}; T=${COV_FIX:-$(cd "$(dirname "$0")/../.." && pwd)/tests/fixtures}
tag=$f${2:+_b$2}
[ -f "$O/$tag.out" ] || { echo "$tag: no frozen reference"; exit 2; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
RING='RNA_C_RING ACTIVE'; LY='RNA_INT_LOOP_LYNGSO ACTIVE'
SP="RNA_MD_SPARSE ACTIVE"; ST="RNA_MD_SPARSE_SELFTEST ACTIVE"; ST0="compared, 0 MISMATCHING"
# label | env | lines that must appear | lines that must not   (lists split on ';')
ARMS=(
  "default||$RING;$LY;$SP|$ST"
  "ring0|RNA_C_RING=0|$LY|$RING"
  "ly0|RNA_INT_LOOP_LYNGSO=0|$RING|$LY"
  "ring0_ly0|RNA_C_RING=0 RNA_INT_LOOP_LYNGSO=0||$RING;$LY"
  # the dense md, cut to one width in C6 (D1): its fallback arm and the selftest that uses it as the reference
  "sparse0|RNA_MD_SPARSE=0|$RING;$LY|$SP;$ST"
  "selftest|RNA_MD_SPARSE_SELFTEST=1|$ST;$ST0|$SP"
)
fail=0
for a in "${ARMS[@]}"; do
  IFS='|' read -r label envs want nowant <<< "$a"
  env $envs RNA_GPU_VERBOSE=1 $R --noPS < "$T/$f.fa" > "$W/o" 2> "$W/e"; rc=$?
  same=$(cmp -s "$W/o" "$O/$tag.out" && echo yes || echo NO)
  ev=yes
  IFS=';' read -ra L <<< "$want";   for p in "${L[@]}"; do grep -q "$p" "$W/e" || ev="NO(missing: $p)"; done
  IFS=';' read -ra L <<< "$nowant"; for p in "${L[@]}"; do grep -q "$p" "$W/e" && ev="NO(present: $p)"; done
  sw=$(grep -c 'sweep shape' "$W/e")
  ok=PASS; [ $rc = 0 ] && [ $same = yes ] && [ "$ev" = yes ] && [ "$sw" -ge 1 ] || { ok=FAIL; fail=1; }
  echo "$tag $label $ok rc $rc same $same path $ev sweeps $sw"
done
exit $fail
