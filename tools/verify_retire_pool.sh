#!/bin/bash
# The Phase 4b bar: retiring a record's CPU half on a worker pool must not move
# the answer.
#
# 4b splits on_retire_cb() in two. The FETCH stays on the sweep's thread because
# refill_slot2() overwrites exactly those device cells later in the same loop
# iteration; the FINISH (exterior loop, MFE, backtrack) goes to a worker. So the
# one bug the change can introduce is an ordering one -- a fetch that is allowed
# to drift past the refill -- and the second bug is an incomplete drain, where a
# worker is still writing Structure[] when the output is read.
#
# TWO ARMS, because no single fixture reaches both:
#
#   mixed   30 records, every one a different length, 200..1360 nt.
#           CORRECTNESS. Uniform lengths retire every slot on the same iteration,
#           so a handover defect lands when all the neighbours are themselves
#           starting fresh and is invisible -- the --noLP slot-flow defect hid
#           behind exactly that for a session (see verify_nolp_parity.sh). Only a
#           mixed file makes slots hand over at staggered, contended moments.
#
#   u2000   24 records, all 2000 nt. CONCURRENCY REACHABILITY. The opposite
#           property: uniform lengths retire every slot on the SAME iteration, so
#           several records are in flight at once. The mixed arm measures peak 1
#           in flight -- it hands off and the worker is done before the next
#           handover arrives -- which means a green mixed run says nothing about
#           whether two records were ever finished concurrently. This arm asserts
#           peak > 1, so the pool is known to have been exercised and not merely
#           present.
#
# Both arms are also run against the serial control (RNA_BACKTRACK_THREADS=0),
# which is the old pre-4b path, so a disagreement is attributable to the pool
# rather than to anything else in the tree.
#
# NEGATIVE CONTROL, run by hand 2026-09-23 and recorded here because the script
# cannot run it: with the fetch moved into the worker -- the exact ordering
# violation above -- BOTH arms went DIFF against upstream. The check can reach
# its own subject.
#
# Trees: $1 = CUDA build, $2 = no-CUDA build (the upstream oracle).
CUDA=${1:-$HOME/lfb}
NOCUDA=${2:-$HOME/port27git}
for d in "$CUDA" "$NOCUDA"; do
  [ -x "$d/src/bin/RNAfold" ] || { echo "no RNAfold under $d" >&2; exit 2; }
done
GPU=$CUDA/src/bin/RNAfold
CPU=$NOCUDA/src/bin/RNAfold
W=$HOME/dfprobe/retirebar; mkdir -p $W
fail=0
say() { printf '  %-52s %s\n' "$1" "$2"; [ "$2" != ok ] && fail=1; return 0; }

python3 - $W/mixed.fa <<'PY'
import random, sys
random.seed(20260923)
with open(sys.argv[1], "w") as f:
    for i in range(30):                     # 200..1360 nt, every record different
        n = 200 + 40*i
        f.write(">m%d_L%d\n%s\n" % (i, n, "".join(random.choice("ACGU") for _ in range(n))))
PY
python3 - $W/u2000.fa <<'PY'
import random, sys
random.seed(7)
with open(sys.argv[1], "w") as f:
    for i in range(24):                     # one length, so slots retire together
        f.write(">u%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(2000))))
PY

nlen=$(grep -v '^>' $W/mixed.fa | awk '{print length}' | sort -n | uniq | wc -l)
say "mixed: the fixture really is mixed-length ($nlen distinct)" \
    "$([ "$nlen" -gt 10 ] && echo ok || echo FAIL)"
ulen=$(grep -v '^>' $W/u2000.fa | awk '{print length}' | sort -n | uniq | wc -l)
say "u2000: the fixture really is uniform ($ulen distinct)" \
    "$([ "$ulen" -eq 1 ] && echo ok || echo FAIL)"

run() {  # run <tag> <fixture> <slotflow> <threads-env>
  local tag=$1 fa=$2 k=$3
  shift 3
  env RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 RNA_SLOT_FLOW=$k "$@" \
      $GPU --noPS -i $fa 2> $W/$tag.err > $W/$tag.out
}

for fx in mixed u2000; do
  $CPU --noPS -i $W/$fx.fa 2>/dev/null > $W/$fx.cpu

  for k in 2 4; do
    run $fx.k$k.par $W/$fx.fa $k
    run $fx.k$k.ser $W/$fx.fa $k RNA_BACKTRACK_THREADS=0

    # The slot-flow arm must actually hand slots over. `peak` counts RESIDENT
    # records; peak < nrec is the proof slots were shared, and without a
    # handover there is no retirement to overlap and nothing here is tested.
    nrec=$(grep -c '^>' $W/$fx.fa)
    peak=$(grep -o 'peak/iteration [0-9]* records' $W/$fx.k$k.par.err | head -1 | awk '{print $2}')
    say "$fx k=$k: slots really were shared (peak ${peak:-0} < $nrec)" \
        "$([ "${peak:-0}" -gt 0 ] && [ "${peak:-0}" -lt "$nrec" ] && echo ok || echo FAIL)"

    # The pool must really have had workers, or the "threaded" arm is the serial
    # path under another name and both comparisons below are vacuous.
    # grep -o, NOT a sed capture: `s/.*\([0-9]\+\) retire workers.*/\1/` has a
    # greedy .* that eats the leading digit, so "12 retire workers" parses as 2.
    # The >1 assertion below then passed for the wrong reason -- a check quietly
    # reading the wrong number is worse than no check.
    nw=$(grep -o '[0-9]\+ retire workers' $W/$fx.k$k.par.err | head -1 | awk '{print $1}')
    say "$fx k=$k: the threaded arm really had a pool ($nw workers)" \
        "$([ "${nw:-0}" -gt 1 ] && echo ok || echo FAIL)"
    nw1=$(grep -o '[0-9]\+ retire workers' $W/$fx.k$k.ser.err | head -1 | awk '{print $1}')
    say "$fx k=$k: the control really was serial ($nw1 worker)" \
        "$([ "${nw1:-0}" -eq 1 ] && echo ok || echo FAIL)"

    say "$fx k=$k: threaded identical to upstream" \
        "$(cmp -s $W/$fx.cpu $W/$fx.k$k.par.out && echo ok || echo FAIL)"
    say "$fx k=$k: threaded identical to the serial control" \
        "$(cmp -s $W/$fx.k$k.ser.out $W/$fx.k$k.par.out && echo ok || echo FAIL)"

    # Every record retired, and the pool shut down clean. retire_pool_drain()
    # prints RETIRE POOL ERROR if a job was fetched and never finished or a
    # scratch never came back -- either is a silently unwritten structure.
    say "$fx k=$k: all $nrec records retired" \
        "$(grep -q 'SCHEDULE ERROR' $W/$fx.k$k.par.err && echo FAIL || echo ok)"
    say "$fx k=$k: the pool drained clean" \
        "$(grep -q 'RETIRE POOL ERROR' $W/$fx.k$k.par.err && echo FAIL || echo ok)"
  done
done

# The reachability assertion the mixed arm cannot make. Asserted on u2000 only,
# and deliberately NOT on mixed, where peak 1 is the expected and correct
# outcome -- asserting it there would be a check that fails for the right answer.
inf=$(sed -n 's/.*peak \([0-9]\+\) in flight.*/\1/p' $W/u2000.k2.par.err | head -1)
say "u2000: the pool really overlapped records (peak ${inf:-0} in flight > 1)" \
    "$([ "${inf:-0}" -gt 1 ] && echo ok || echo FAIL)"

echo
[ $fail -eq 0 ] && echo "verify_retire_pool: PASS" || echo "verify_retire_pool: FAIL"
exit $fail
