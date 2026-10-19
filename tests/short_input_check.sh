#!/bin/bash
#
# Short and interactive input: PORT_SHORT_SEQUENCES.md S1 + S2, asserted.
#
# W. B. Langdon, 2026-10-18: a sequence typed at the prompt was answered only after
# Ctrl-D, and a short input paid the GPU's wake-up for a fold the CPU finishes first.
# Both are properties no file-driven bar could see: the first needs a TERMINAL
# (isatty), the second needs proof of something NOT happening.
#
# THE EVIDENCE FOR "THE DEVICE WAS NEVER TOUCHED". libcudart is linked at load
# time, but it dlopen()s the driver, libcuda.so.1, only at the first CUDA call --
# the call that creates the context. So LD_DEBUG=files naming libcuda is positive
# evidence that a context was attempted, and its absence is evidence that it was
# not. Before S2 the probe ran in gpu_batch_begin() whatever the input, so this
# check FAILS on that build, including under RNA_GPU=0 (8 loads each, 2026-10-18):
# the negative control is the previous binary, NEGCTL_TREE below.
#
#   usage: TREE=~/pyqc tests/short_input_check.sh [workdir]
#          NEGCTL_TREE=<pre-S2 tree> additionally demands that tree FAIL case B1.

set -u
TREE=${TREE:-$HOME/lfb}
B=$TREE/src/bin/RNAfold
W=${1:-${TMPDIR:-/tmp}/short_input.$$}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W" && cd "$W" || exit 2
for v in $(env | grep -o '^RNA_[A-Z0-9_]*'); do unset "$v"; done

fail=0
ok()  { printf '  ok      %s\n' "$1"; }
bad() { printf '  FAILED  %s\n' "$1"; fail=$((fail + 1)); }
cuda_loads() { grep -c 'file=.*libcuda\.so' "$1"; }   # an LD_DEBUG=files log

S24=ggggggggggaaaacccccccccc
python3 - <<'PY'
import random
r = random.Random(20261018)
rs = lambda L: "".join(r.choice("ACGU") for _ in range(L))
open("long.seq", "w").write(rs(1500) + "\n")
with open("three100.fa", "w") as f:
    for i in range(3): f.write(">s%d\n%s\n" % (i, rs(100)))
with open("zc12.fa", "w") as f:            # tests/zeroconf_configure.sh's shape
    for i in range(12): f.write(">t%d\n%s\n" % (i, rs((600, 900, 1200)[i % 3])))
with open("mixed.fa", "w") as f:           # held short records, then enough work
    for i in range(30): f.write(">h%d\n%s\n" % (i, rs(24)))
    for i in range(12): f.write(">l%d\n%s\n" % (i, rs((600, 900, 1200)[i % 3])))
with open("big1200.fa", "w") as f:
    for i in range(12): f.write(">d%d\n%s\n" % (i, rs(1200)))
PY
LONG=$(cat long.seq)

echo "=== S1. a terminal is answered before end of input"
python3 "$HERE/tty_answer.py" "$B" "$S24" "$LONG" "$S24" > tty.out 2>&1; r=$?
grep '^answered' tty.out | sed 's/^/          /'
[ $r -eq 0 ] && ok "all three sequences answered before Ctrl-D" \
             || bad "a sequence was held until end of input"
grep -q 'sweep shape' tty.out && ok "the 1500 nt record used the device (sweep shape, at the prompt)" \
                              || bad "the long record did not reach the device"
RNA_GPU=0 python3 "$HERE/tty_answer.py" "$B" "$S24" "$LONG" "$S24" > tty.cpu 2>&1
if [ "$(grep -o 'minimum free energy = *-*[0-9.]*' tty.out)" = \
     "$(grep -o 'minimum free energy = *-*[0-9.]*' tty.cpu)" ] && grep -q 'minimum free' tty.cpu; then
  ok "energies equal RNA_GPU=0 at the terminal"
else
  bad "terminal energies differ from RNA_GPU=0"
fi
grep -q 'fml_j AUTO' tty.out && bad "the AUTO chatter is back at the prompt" \
                             || ok "no fml_j AUTO line at the prompt"

echo "=== S2. no CUDA until the input could earn it"
run() {  # tag env-string input-file [extra args]
  local t=$1 e=$2 in=$3; shift 3
  env $e LD_DEBUG=files "$B" --noPS "$@" < "$in" > "$t.out" 2> "$t.dbg"
  env $e "$B" --noPS "$@" < "$in" > "$t.out2" 2> "$t.err"
  RNA_GPU=0 "$B" --noPS "$@" < "$in" > "$t.cpu" 2> /dev/null
}
echo "$S24" > s24.seq
run b1 "" s24.seq
[ "$(cuda_loads b1.dbg)" -eq 0 ] && ok "B1 one 24 nt record: libcuda never loaded" \
                                 || bad "B1 one 24 nt record loaded libcuda $(cuda_loads b1.dbg) time(s)"
cmp -s b1.out2 b1.cpu && ok "B1 answer equals RNA_GPU=0" || bad "B1 answer differs"
[ -s b1.err ] && { bad "B1 wrote to stderr:"; head -3 b1.err | sed 's/^/          /'; } \
              || ok "B1 silent on stderr, as stock ViennaRNA is"

run b2 "RNA_GPU=0" s24.seq
[ "$(cuda_loads b2.dbg)" -eq 0 ] && ok "B2 RNA_GPU=0: libcuda never loaded" \
                                 || bad "B2 RNA_GPU=0 loaded libcuda"
grep -q 'GPU acceleration OFF: RNA_GPU=0' b2.err && ok "B2 RNA_GPU=0 still says so" \
                                                 || bad "B2 RNA_GPU=0 is no longer announced"

run b3 "" three100.fa
[ "$(cuda_loads b3.dbg)" -eq 0 ] && ok "B3 3 x 100 nt: libcuda never loaded" \
                                 || bad "B3 3 x 100 nt loaded libcuda"
cmp -s b3.out2 b3.cpu && ok "B3 equals RNA_GPU=0" || bad "B3 differs"

run b4 "RNA_RECORD_TRACE=$W/b4.tr" zc12.fa
grep -q 'GPU acceleration ON' b4.err && ok "B4 12 x 600-1200 nt: the device was reached" \
                                     || bad "B4 the device was not reached"
grep -q 'sweep shape' b4.err && ok "B4 swept" || bad "B4 no sweep"
n=$(tail -n +2 b4.tr | cut -d, -f3 | grep -c gpu)
[ "$n" -eq 12 ] && ok "B4 all 12 records routed to the device" || bad "B4 only $n of 12 on the device"
cmp -s b4.out2 b4.cpu && ok "B4 equals RNA_GPU=0" || bad "B4 differs"
for l in 'fml_j AUTO' 'build pipeline AUTO' 'build threads'; do
  grep -q "$l" b4.err && bad "B4 default run still prints '$l'" || ok "B4 default run does not print '$l'"
done
"$B" --noPS --verbose < zc12.fa > /dev/null 2> b4v.err
grep -q 'fml_j AUTO' b4v.err && ok "B4 --verbose prints fml_j AUTO" || bad "B4 --verbose lost fml_j AUTO"

run b5 "RNA_RECORD_TRACE=$W/b5.tr" mixed.fa
n=$(grep -c '^>' b5.out2)
[ "$n" -eq 42 ] && ok "B5 held-then-reached: all 42 records written" || bad "B5 wrote $n of 42"
cmp -s b5.out2 b5.cpu && ok "B5 equals RNA_GPU=0, in input order" || bad "B5 differs"
grep -q 'sweep shape' b5.err && ok "B5 swept once the held work earned it" || bad "B5 no sweep"

run b6 "" three100.fa -d3
[ "$(cuda_loads b6.dbg)" -eq 0 ] && ok "B6 -d3 small: libcuda never loaded" || bad "B6 -d3 small loaded libcuda"
grep -q 'acceleration OFF' b6.err && bad "B6 -d3 small announced a decline nobody lost" \
                                  || ok "B6 -d3 small: no decline line"
run b7 "" big1200.fa -d3
grep -q 'GPU acceleration OFF: dangle model' b7.err && ok "B7 -d3 12 x 1200: the unused device is named" \
                                                   || bad "B7 -d3 large: no decline line"
cmp -s b7.out2 b7.cpu && ok "B7 equals RNA_GPU=0" || bad "B7 differs"

run b8 "RNA_GPU_WORK_FLOOR=0" s24.seq
grep -q 'sweep shape' b8.err && ok "B8 RNA_GPU_WORK_FLOOR=0 still forces the device" \
                             || bad "B8 a forced floor no longer reaches the device"
cmp -s b8.out2 b8.cpu && ok "B8 equals RNA_GPU=0" || bad "B8 differs"

echo "=== the wake-up, timed (reported; asserted loosely)"
med() { sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }
for arm in cpu dflt; do
  for k in $(seq 9); do
    [ $arm = cpu ] && E="RNA_GPU=0" || E=""
    s=$(date +%s%N); env $E "$B" --noPS < s24.seq > /dev/null 2>&1; e=$(date +%s%N)
    echo $(( (e - s) / 1000000 ))
  done | med > t.$arm
done
echo "          median one-shot: RNA_GPU=0 $(cat t.cpu) ms, default $(cat t.dflt) ms"
[ "$(cat t.dflt)" -le $(( $(cat t.cpu) + 30 )) ] && ok "default within 30 ms of RNA_GPU=0" \
                                                 || bad "default still pays a wake-up"

if [ -n "${NEGCTL_TREE:-}" ]; then
  echo "=== negative control: $NEGCTL_TREE must FAIL B1"
  echo "$S24" | LD_DEBUG=files "$NEGCTL_TREE/src/bin/RNAfold" --noPS 2> nc.dbg > /dev/null
  [ "$(cuda_loads nc.dbg)" -gt 0 ] && ok "negctl loads libcuda $(cuda_loads nc.dbg) time(s) -- the check bites" \
                                  || bad "negctl does NOT load libcuda: the check cannot fail"
fi

echo
[ $fail -eq 0 ] && { echo "RESULT: short and interactive input behave as stock ViennaRNA"; exit 0; }
echo "RESULT: $fail check(s) FAILED -- logs in $W"; exit 1
