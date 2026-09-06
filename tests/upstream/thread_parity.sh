#!/bin/bash
# Does pristine 2.7.2's own threading produce the same answers as -j1?
#
# Probe B in PORT_UPSTREAM_PROPOSAL.md only showed that vrna_params() is not
# observably corrupted when every thread uses identical model details. That is a
# statement about one cache, not about the tool. This runs the whole binary.
#
# Reference: ~/rnatest/ref_md/*.out, verified byte-identical to pristine 2.7.2
# single-threaded in Phase 0 (PORT_PHASE0.md 0b). The CPU pool does not preserve
# input order, so records are compared as a sorted MULTISET of
# (structure, energy) lines -- the same compromise tests/verify_guards.sh makes.
#
# Usage: thread_parity.sh [path-to-ViennaRNA-2.7.2] [reps]
set -u
V=${1:-$HOME/vrna27/ViennaRNA-2.7.2}
REPS=${2:-3}
B=$V/src/bin/RNAfold
T=$HOME/rnatest
R=$T/ref_md
W=${TMPDIR:-/tmp}/vrna_thread_parity
mkdir -p "$W" || exit 2
cd "$W" || exit 2

[ -x "$B" ] || { echo "no pristine RNAfold at $B" >&2; exit 2; }

# sorted multiset of the answer lines, order- and header-independent
body() { grep -E '^[.()&,<>{}|]+ +\( *-?[0-9]+\.[0-9]+\)$' "$1" | sort; }

fails=0
for inp in C_mixed C_control asc desc extreme F_extreme; do
  [ -f "$T/$inp.fa" ] || { echo "  $inp: SKIP (no input)"; continue; }
  [ -f "$R/$inp.out" ] || { echo "  $inp: SKIP (no reference)"; continue; }

  nref=$(body "$R/$inp.out" | wc -l)
  for rep in $(seq 1 "$REPS"); do
    t0=$(date +%s)
    "$B" --noPS -j8 -i "$T/$inp.fa" > "$W/$inp.j8.$rep.out" 2>"$W/$inp.j8.$rep.err"
    rc=$?
    t1=$(date +%s)
    n=$(body "$W/$inp.j8.$rep.out" | wc -l)

    if [ $rc -ne 0 ]; then
      printf '  %-12s rep %d  FAIL (exit %d)\n' "$inp" "$rep" "$rc"; fails=$((fails+1)); continue
    fi
    if [ "$n" -ne "$nref" ]; then
      printf '  %-12s rep %d  FAIL (%d records, reference has %d)\n' "$inp" "$rep" "$n" "$nref"
      fails=$((fails+1)); continue
    fi
    if diff <(body "$R/$inp.out") <(body "$W/$inp.j8.$rep.out") > "$W/$inp.$rep.diff"; then
      printf '  %-12s rep %d  PASS  (%d records, %ds)\n' "$inp" "$rep" "$n" "$((t1-t0))"
    else
      printf '  %-12s rep %d  FAIL  (%d differing lines, see %s)\n' \
             "$inp" "$rep" "$(grep -c '^[<>]' "$W/$inp.$rep.diff")" "$W/$inp.$rep.diff"
      fails=$((fails+1))
    fi
  done
done

echo
if [ $fails -eq 0 ]; then
  echo "RESULT: pristine 2.7.2 -j8 agrees with the single-threaded reference on every record"
else
  echo "RESULT: $fails FAILURES -- 2.7.2's own threading is NOT answer-preserving here"
fi
exit $fails
