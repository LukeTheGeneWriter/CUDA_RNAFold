#!/bin/bash
# Byte-identical bar: fold the six frozen reference inputs and compare with cmp.
#
# Run this against every binary that claims to be behaviour-preserving. It is
# deliberately separate from the build so a killed build does not lose the
# evidence, and it takes the binary as an argument so the tarball build, the
# git build and any future CUDA build are all measured the same way.
#
# Usage: verify_ref_md.sh [path-to-RNAfold] [extra RNAfold args...]
set -u
B=${1:-$HOME/port27git/src/bin/RNAfold}
shift 2>/dev/null || true
T=$HOME/rnatest
R=$T/ref_md
W=${TMPDIR:-/tmp}/vrna_ref_md
mkdir -p "$W" || exit 2

[ -x "$B" ] || { echo "no RNAfold at $B" >&2; exit 2; }

echo "binary : $B"
echo "built  : $(date -r "$B" '+%Y-%m-%d %H:%M:%S')"
echo "size   : $(stat -c %s "$B") bytes"
echo

fails=0
checked=0
for inp in C_mixed C_control asc desc extreme F_extreme; do
  [ -f "$T/$inp.fa" ] || { printf '  %-12s SKIP (no input)\n' "$inp"; continue; }
  [ -f "$R/$inp.out" ] || { printf '  %-12s SKIP (no reference)\n' "$inp"; continue; }

  t0=$(date +%s)
  "$B" --noPS "$@" -i "$T/$inp.fa" > "$W/$inp.out" 2>"$W/$inp.err"
  rc=$?
  t1=$(date +%s)
  n=$(grep -c '^>' "$T/$inp.fa")
  checked=$((checked+1))

  if [ $rc -ne 0 ]; then
    printf '  %-12s FAIL (exit %d)\n' "$inp" "$rc"; fails=$((fails+1)); continue
  fi
  if cmp -s "$R/$inp.out" "$W/$inp.out"; then
    printf '  %-12s IDENTICAL  (%s records, %ds)\n' "$inp" "$n" "$((t1-t0))"
  else
    printf '  %-12s DIFFERS    (%s)\n' "$inp" "$(cmp "$R/$inp.out" "$W/$inp.out" 2>&1 | head -1)"
    fails=$((fails+1))
  fi
done

echo
if [ $fails -eq 0 ]; then
  echo "RESULT: $checked/$checked inputs byte-identical to the frozen references"
else
  echo "RESULT: $fails of $checked inputs DIFFER"
fi
exit $fails
