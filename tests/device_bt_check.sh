#!/bin/bash
# N5 P2d bar 1-3 (PORT_DEVICE_BACKTRACK.md): the device trace, RNA_DEVICE_BT=1 and =verify,
# against the CPU reference (RNA_GPU=0), byte-identical, with POSITIVE evidence that the
# device traced every record (the ACTIVE / VERIFY counts), never only that the answer is
# right -- a refused or failed-over chunk is byte-identical by construction.
#
#   1. arms: fixtures random / A/U-only / G-rich / long; fML int32 and int16; pipeline off
#      and on; one chunk and several (RNA_GPU_CHUNK_CELLS). Multi-chunk x pipeline OFF is
#      the arm that found P2c's upload race -- the others could not see it. Keep it.
#   2. refusal arms: each option REFUSED with its reason and identical to the host, or
#      routed to the host before any sweep (said so -- that arm then proves less).
#   3. the NEGCTL bites (structures change, energies do not; verify + NEGCTL exits non-zero),
#      an unrecognised value is IGNORED, and unset prints nothing new.
#
# usage: TREE=<build tree> [W=<workdir>] [COV_FIX=<fixtures>] tests/device_bt_check.sh
set -u
TREE=${TREE:-$HOME/pyqc}
R=$TREE/src/bin/RNAfold
ROOT=$(cd "$(dirname "$0")/.." && pwd)
T=${COV_FIX:-$ROOT/tests/fixtures}
W=${W:-$HOME/device_bt_check}
[ -x "$R" ] || { echo "no RNAfold under $TREE" >&2; exit 2; }
for v in $(env | grep -o "^RNA_[A-Z0-9_]*"); do unset $v; done
rm -rf "$W"; mkdir -p "$W"; cd "$W"
GPU="RNA_MIN_GPU_BATCH=1 RNA_GPU_WORK_FLOOR=0"
bad=0

# The device must be live, or every arm compares the CPU with itself.
env $GPU RNA_GPU_VERBOSE=1 $R --noPS < $T/C_mixed.fa > /dev/null 2> gate.err
grep -q 'sweep shape:' gate.err || { echo "FATAL: no sweep -- no CUDA or no device"; exit 1; }

gen() {   # gen <file> <seed> <records> <lo> <hi> <alphabet weights A C G U>
  awk -v seed=$2 -v n=$3 -v lo=$4 -v hi=$5 -v wa=$6 -v wc=$7 -v wg=$8 -v wu=$9 'BEGIN {
    srand(seed); t = wa + wc + wg + wu
    for (r = 0; r < n; r++) {
      L = lo + int(rand() * (hi - lo + 1)); s = ""
      for (i = 0; i < L; i++) { x = rand() * t
        s = s (x < wa ? "A" : x < wa + wc ? "C" : x < wa + wc + wg ? "G" : "U") }
      print ">r" r; print s } }' > $1
}
gen random.fa 20261016 48 60 900 1 1 1 1
gen au.fa     7        40 150 700 1 0 0 1
gen grich.fa  31       40 80 800 1 2 4 1
cp $T/A_longseq_smallbatch.fa long.fa

nrec() { grep -c '^>' $1; }

# One arm: stdout must equal the CPU's, rc 0, and the device must have traced EVERY record
# (verify: and every record compared), with the expected chunk shape and nothing failed over.
arm() {
  local fx=$1 i16=$2 pl=$3 cc=$4 mode=$5
  local tag=${fx%.fa}_i${i16}_p${pl}_c${cc}_$mode
  local n=$(nrec $fx) envs="$GPU RNA_FML_INT16=$i16 RNA_BT_PIPELINE=$pl RNA_DEVICE_BT=$mode"
  [ "$cc" != 0 ] && envs="$envs RNA_GPU_CHUNK_CELLS=$cc"
  env $envs $R --noPS < $fx > $tag.out 2> $tag.err; local rc=$?
  local act=$(grep -o 'RNA_DEVICE_BT ACTIVE: .*' $tag.err)
  local ch=$(echo "$act" | grep -oE '^RNA_DEVICE_BT ACTIVE: [0-9]+' | grep -oE '[0-9]+$')
  local rec=$(echo "$act" | grep -oE '[0-9]+ record\(s\) traced' | grep -oE '^[0-9]+')
  local ref=$(echo "$act" | grep -oE '[0-9]+ chunk\(s\) refused' | grep -oE '^[0-9]+')
  local fo=$(echo "$act" | grep -oE '[0-9]+ failed over' | grep -oE '^[0-9]+')
  local ver=$(grep -oE 'RNA_DEVICE_BT VERIFY: [0-9]+' $tag.err | grep -oE '[0-9]+$')
  local v=PASS why=""
  [ $rc -eq 0 ] || { v=FAIL; why="$why rc=$rc"; }
  cmp -s ref_${fx%.fa}.out $tag.out || { v=FAIL; why="$why DIFF"; }
  [ "${rec:-0}" = "$n" ] || { v=FAIL; why="$why traced ${rec:-0}/$n"; }
  [ "${ref:-x}" = 0 ] && [ "${fo:-x}" = 0 ] || { v=FAIL; why="$why refused=${ref:-?} failed=${fo:-?}"; }
  [ "$mode" = verify ] && { [ "${ver:-0}" = "$n" ] || { v=FAIL; why="$why verified ${ver:-0}/$n"; }; }
  # multi-chunk must BE multi-chunk, or it is the single-chunk arm again
  [ "$cc" != 0 ] && { [ "${ch:-0}" -ge 2 ] || { v=FAIL; why="$why only ${ch:-0} chunk"; }; }
  grep -h 'FAILED\|IGNORED\|REFUSED' $tag.err | sort -u | head -2 | sed 's/^/      /'
  [ $v = FAIL ] && bad=$((bad+1))
  printf "  %-34s %2s chunk(s) %4s rec %4s verified  [%s]%s\n" $tag "${ch:-?}" "${rec:-0}" "${ver:--}" $v "$why"
}

echo "== 1. arms (CPU reference RNA_GPU=0; positive evidence = every record traced on the device)"
for spec in random.fa:2000000 au.fa:600000 grich.fa:1500000 long.fa:20000000; do
  fx=${spec%%:*}; cc=${spec##*:}
  RNA_GPU=0 $R --noPS < $fx > ref_${fx%.fa}.out 2> /dev/null
  for i16 in 1 0; do for pl in 0 1; do for c in 0 $cc; do for mode in 1 verify; do
    arm $fx $i16 $pl $c $mode
  done; done; done; done
done

echo "== 1b. heterogeneous: CPU slice + -j (each side traces its own records)"
# The cores fold their slice through upstream's vrna_mfe(), host fill AND host trace; the device
# trace lives in par_mfe() only. Positive evidence both ways: device-traced + core-folded = all
# records, the slice really took some, and the output is the CPU's.
het() {
  local fx=$1 pl=$2 cc=$3 tag=het_${1%.fa}_p$2
  local n=$(nrec $fx)
  env $GPU RNA_CPU_SLICE=1 RNA_BT_PIPELINE=$pl RNA_DEVICE_BT=1 RNA_GPU_CHUNK_CELLS=$cc RNA_RECORD_TRACE=$W/$tag.csv \
      $R --noPS -j4 < $fx > $tag.out 2> $tag.err; local rc=$?
  local rec=$(grep -oE '[0-9]+ record\(s\) traced' $tag.err | grep -oE '^[0-9]+')
  local sl=$(grep -oE 'cpu slice: [0-9]+ records' $tag.err | grep -oE '[0-9]+')
  # the record trace's per-record route is the authority: device-traced must be exactly its gpu rows
  local tg=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="route") c=i; next} $c=="gpu"{g++} END{print g+0}' $W/$tag.csv 2>/dev/null)
  local tc=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="route") c=i; next} $c=="cpu"{g++} END{print g+0}' $W/$tag.csv 2>/dev/null)
  local v=PASS why=""
  [ $rc -eq 0 ] || { v=FAIL; why="$why rc=$rc"; }
  cmp -s ref_${fx%.fa}.out $tag.out || { v=FAIL; why="$why DIFF"; }
  [ $(( ${rec:-0} + ${sl:-0} )) -eq $n ] || { v=FAIL; why="$why device ${rec:-0} + cores ${sl:-0} != $n"; }
  [ "${rec:-0}" = "${tg:-x}" ] && [ "${sl:-0}" = "${tc:-x}" ] || { v=FAIL; why="$why trace says gpu ${tg:-?} cpu ${tc:-?}"; }
  [ "${sl:-0}" -ge 1 ] || why="$why (the slice took nothing: this arm proved only the device half)"
  grep -q 'RNA_DEVICE_BT FAILED' $tag.err && { v=FAIL; why="$why FAILED"; }
  [ $v = FAIL ] && bad=$((bad+1))
  printf "  %-34s device %4s + cores %4s = %4s of %s  [%s]%s\n" $tag "${rec:-0}" "${sl:-0}" $(( ${rec:-0} + ${sl:-0} )) $n $v "$why"
}
for pl in 0 1; do het random.fa $pl 2000000; het long.fa $pl 20000000; done

echo "== 2. refusal arms (REFUSED with its reason and identical to the host)"
refuse() {   # refuse <tag> <fixture> <reason substring> -- <RNAfold options>
  local tag=$1 fx=$2 want=$3; shift 4
  RNA_GPU=0 $R --noPS "$@" < $fx > rf_$tag.cpu 2> /dev/null
  env $GPU RNA_GPU_VERBOSE=1 RNA_DEVICE_BT=1 $R --noPS "$@" < $fx > rf_$tag.out 2> rf_$tag.err; local rc=$?
  local v=PASS why="" what
  [ $rc -eq 0 ] || { v=FAIL; why="$why rc=$rc"; }
  cmp -s rf_$tag.cpu rf_$tag.out || { v=FAIL; why="$why DIFF"; }
  local rec=$(grep -oE '[0-9]+ record\(s\) traced' rf_$tag.err | grep -oE '^[0-9]+')
  if ! grep -q 'sweep shape:' rf_$tag.err; then
    what="host-routed (no sweep; the refusal was not reached)"
  elif grep -q "RNA_DEVICE_BT REFUSED: .*$want" rf_$tag.err; then
    what="REFUSED: $(grep -o "RNA_DEVICE_BT REFUSED: .*$want" rf_$tag.err | head -1 | cut -c24-)"
    [ "${rec:-0}" = 0 ] || { v=FAIL; why="$why but ${rec} record(s) traced"; }
  else
    what="swept, NOT refused for '$want': $(grep -o 'RNA_DEVICE_BT REFUSED: .*' rf_$tag.err | head -1)"; v=FAIL
  fi
  [ $v = FAIL ] && bad=$((bad+1))
  printf "  %-14s %s  [%s]%s\n" $tag "$what" $v "$why"
}
refuse d0          $T/C_mixed.fa     "(-d)"            -- -d0
refuse noLP        $T/C_mixed.fa     "--noLP"          -- --noLP
refuse circ        $T/C_mixed.fa     "(-c)"            -- -c
refuse gquad       grich.fa          "(-g)"            -- -g
refuse constraint  $T/constrained.fa "(-C)"            -- -C
refuse T25         $T/C_mixed.fa     "(-T)"            -- -T 25
refuse P           $T/C_mixed.fa     "(-P)"            -- -P $ROOT/misc/rna_andronescu2007.par
refuse salt        $T/C_mixed.fa     "(--salt)"        -- --salt=0.2
refuse noClosingGU $T/C_mixed.fa     "--noClosingGU"   -- --noClosingGU
refuse maxBPspan   $T/C_mixed.fa     "(--maxBPspan)"   -- --maxBPspan=100

echo "  binding: fc.mfe() keeps the matrices (refused); RNA.cuda_fold() does not (traced)"
cat > bind.py <<'PY'
import random, RNA
rng = random.Random(14)
seqs = ["".join(rng.choice("ACGU") for _ in range(n)) for n in (400, 233, 517)]
fc_d, fc_h = RNA.fold_compound(seqs[0]), RNA.fold_compound(seqs[0])
d, h = fc_d.mfe(), fc_h.mfe(cpu_only=True)
assert d[0] == h[0] and abs(d[1] - h[1]) < 1e-6, (d, h)
assert fc_d.backtrack(150)[0] == fc_h.backtrack(150)[0]
b = RNA.cuda_batches()
got = RNA.cuda_fold(seqs)
for s, (st, e) in zip(seqs, got):
    fc = RNA.fold_compound(s); ws, we = fc.mfe(cpu_only=True)
    assert st == ws and abs(e - we) < 1e-4, (st, ws, e, we)
print("binding ok, cuda batches", RNA.cuda_batches() - b)
PY
env $GPU RNA_GPU_VERBOSE=1 RNA_DEVICE_BT=1 PYTHONPATH=$TREE/interfaces/Python python3 bind.py > bind.out 2> bind.err; rc=$?
v=PASS
[ $rc -eq 0 ] && grep -q 'binding ok' bind.out || v=FAIL
grep -q 'RNA_DEVICE_BT REFUSED: the caller keeps the matrices' bind.err || v=FAIL
brec=$(grep -oE '[0-9]+ record\(s\) traced' bind.err | grep -oE '^[0-9]+')
[ "${brec:-0}" = 3 ] || v=FAIL
[ $v = FAIL ] && bad=$((bad+1))
printf "  %-14s rc %s, %s; keep-matrices refused: %s; cuda_fold traced %s/3  [%s]\n" binding $rc \
       "$(cat bind.out | tr -d '\n')" "$(grep -c 'keeps the matrices' bind.err)" "${brec:-0}" $v
[ $v = FAIL ] && sed 's/^/      /' bind.err | tail -5

echo "== 3. negative control, IGNORED, unset"
env $GPU RNA_DEVICE_BT=1 RNA_DEVICE_BT_NEGCTL=1 $R --noPS < au.fa > neg.out 2> neg.err; rn=$?
sch=$(paste <(awk 'NR%3==0 {print $1}' ref_au.out) <(awk 'NR%3==0 {print $1}' neg.out) |
      awk '$1 != $2 {c++} END {print c+0}')
egy=$(diff <(grep -oE '\( *-?[0-9.]+\)$' ref_au.out) <(grep -oE '\( *-?[0-9.]+\)$' neg.out) > /dev/null && echo equal || echo CHANGED)
env $GPU RNA_DEVICE_BT=verify RNA_DEVICE_BT_NEGCTL=1 $R --noPS < au.fa > /dev/null 2> negv.err; rv=$?
v=PASS
{ [ $rn -eq 0 ] && [ "$sch" -ge 1 ] && [ $egy = equal ] && [ $rv -ne 0 ] && grep -q 'VERIFY FAILED' negv.err; } || v=FAIL
[ $v = FAIL ] && bad=$((bad+1))
printf "  %-14s %s/%s structures changed, energies %s; verify+NEGCTL rc %s  [%s]\n" negctl "$sch" "$(nrec au.fa)" $egy $rv $v

env $GPU RNA_DEVICE_BT=yes $R --noPS < $T/C_mixed.fa > ig.out 2> ig.err
RNA_GPU=0 $R --noPS < $T/C_mixed.fa > ref_cm.out 2> /dev/null
v=PASS
{ grep -q 'RNA_DEVICE_BT.*IGNORED' ig.err && ! grep -q 'RNA_DEVICE_BT ACTIVE' ig.err && cmp -s ig.out ref_cm.out; } || v=FAIL
[ $v = FAIL ] && bad=$((bad+1))
printf "  %-14s %s  [%s]\n" ignored "$(grep -o 'RNA_DEVICE_BT.*IGNORED.*' ig.err | head -1 | cut -c1-90)" $v

env $GPU $R --noPS < $T/C_mixed.fa > un.out 2> un.err
v=PASS
{ [ "$(grep -c 'RNA_DEVICE_BT' un.err)" = 0 ] && cmp -s un.out ref_cm.out; } || v=FAIL
[ $v = FAIL ] && bad=$((bad+1))
printf "  %-14s %s RNA_DEVICE_BT line(s), output %s  [%s]\n" unset "$(grep -c 'RNA_DEVICE_BT' un.err)" \
       "$(cmp -s un.out ref_cm.out && echo identical || echo DIFF)" $v

echo "FAILURES: $bad"
exit $([ $bad -eq 0 ] && echo 0 || echo 1)
