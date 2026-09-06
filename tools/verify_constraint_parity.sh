#!/bin/bash
#
# Hard structure constraints (-C) on the CUDA build.
#
# WHAT THIS DEFENDS. The library guard in mfe/cuda/engine.c declines any fold
# compound carrying hard constraints, so -C folds on upstream's own path and
# gives upstream's own answer. That guard exists because the sweep was measured
# getting it wrong, not because anyone assumed it would:
#
#   12 x 80 nt, a forced-unpaired block over positions 21-40
#     CPU route : -10.00, and the structure re-evaluates to -10.00
#     GPU sweep : -14.30, and the structure re-evaluates to  -5.40
#
# The fill and the backtrack did not agree with each other. Note also that
# RNA_ROW_VERIFY reports "58128 cells checked, 0 mismatching" on exactly these
# folds: the fork's host sweep is wrong the same way as its device sweep, so
# device-against-host cannot see this. Only upstream's fill_arrays can.
#
# EXPECTATION. By default this asserts constraints are ROUTED: same answer as
# the CPU, and zero GPU sweeps. Pass --expect-accelerated once the sweep
# genuinely honours hc->mx, and it will instead require sweeps > 0. Running it
# in the wrong mode is meant to fail loudly -- a guard that quietly stopped
# declining is exactly the regression worth catching.
#
# Usage: verify_constraint_parity.sh [--expect-accelerated] [build-tree] [input.fa]

set -u

EXPECT=routed
if [ "${1:-}" = "--expect-accelerated" ]; then EXPECT=accelerated; shift; fi

TREE=${1:-$HOME/port27rel}
SRC=${2:-$HOME/rnatest/asc.fa}
BIN=$TREE/src/bin/RNAfold
EVAL=$TREE/src/bin/RNAeval
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "binary   : $BIN"
echo "input    : $SRC"
echo "expecting: constraints are $EXPECT"
echo

[ -x "$BIN" ]  || { echo "FAIL: no RNAfold at $BIN";  exit 1; }
[ -x "$EVAL" ] || { echo "FAIL: no RNAeval at $EVAL"; exit 1; }

run () {  # run <outfile> <errfile> <gpu:0|1> [args...]
  local out=$1 err=$2 gpu=$3; shift 3
  if [ "$gpu" = 1 ]; then
    RNA_GPU_CHUNK=0 "$BIN" --noPS "$@" > "$out" 2> "$err"
  else
    env -u RNA_GPU_CHUNK "$BIN" --noPS "$@" > "$out" 2> "$err"
  fi
}

# ---------------------------------------------------------------------------
# The free fold first: the constraints are derived FROM it, so that every shape
# is guaranteed to bite. A fixed pattern does not do this -- a '|' (force
# paired) block over bases the MFE already pairs changes nothing, passes every
# comparison, and proves nothing. That happened on the first version of this
# script.
# ---------------------------------------------------------------------------
run "$WORK/free.out" "$WORK/free.err" 0 -i "$SRC"
[ -s "$WORK/free.out" ] || { echo "FAIL: the unconstrained fold produced no output"; exit 1; }

python3 - "$SRC" "$WORK/free.out" "$WORK" <<'PY'
import sys, os
src, freeout, work = sys.argv[1], sys.argv[2], sys.argv[3]

seqs = []
hdr, buf = None, []
for line in open(src):
    line = line.strip()
    if line.startswith(">"):
        if hdr is not None: seqs.append((hdr, "".join(buf)))
        hdr, buf = line, []
    elif line: buf.append(line)
if hdr is not None: seqs.append((hdr, "".join(buf)))

# RNAfold prints header, sequence, then "structure ( energy)"
lines = [l.rstrip("\n") for l in open(freeout)]
structs = []
for i, l in enumerate(lines):
    if l.startswith(">"):
        structs.append(lines[i + 2].split()[0])
assert len(structs) == len(seqs), (len(structs), len(seqs))

def emit(name, fn):
    with open(os.path.join(work, name), "w") as f:
        for (h, s), st in zip(seqs, structs):
            f.write(f"{h}\n{s}\n{fn(s, st)}\n")

# control: no constraint at all -- must reproduce the free fold exactly
emit("dots.fa", lambda s, st: "." * len(s))

# force unpaired EXACTLY the bases the free MFE pairs: maximal bite by
# construction, and impossible for the free structure to satisfy
emit("xpaired.fa", lambda s, st: "".join("x" if c != "." else "." for c in st))

# force a pair the free fold does NOT contain: first and last pairable bases
CAN = {("G","C"),("C","G"),("A","U"),("U","A"),("G","U"),("U","G")}
def far_pair(s, st):
    c = ["."] * len(s)
    for lo in range(0, len(s) // 2):
        for hi in range(len(s) - 1, lo + 4, -1):
            if (s[lo].upper(), s[hi].upper()) in CAN and not (st[lo] != "." and st[hi] != "."):
                c[lo], c[hi] = "(", ")"
                return "".join(c)
    return "".join(c)
emit("farpair.fa", far_pair)

print(f"  {len(seqs)} records, {min(len(s) for _,s in seqs)}-{max(len(s) for _,s in seqs)} nt;"
      f" constraints derived from the free fold so each one bites")
PY
[ $? -eq 0 ] || { echo "FAIL: could not build constrained inputs"; exit 1; }
echo

# ---------------------------------------------------------------------------
# Self-consistency: does the energy a route REPORTS match the structure it
# RETURNS? This compares the binary against itself and needs no reference
# implementation. It is the check that found the defect this guard now
# prevents, and it would have found it with no CPU route to compare against.
# ---------------------------------------------------------------------------
selfconsistent () {  # selfconsistent <RNAfold output file> ; echoes bad-record count
  python3 - "$1" "$EVAL" <<'PY'
import sys, subprocess
out, ev = sys.argv[1], sys.argv[2]
lines = [l.rstrip("\n") for l in open(out)]
bad = 0
for i, l in enumerate(lines):
    if not l.startswith(">"): continue
    seq = lines[i+1]
    st, _, en = lines[i+2].partition(" (")
    try: reported = float(en.strip(" )"))
    except ValueError: continue
    r = subprocess.run([ev, "--noconv"], input=f"{seq}\n{st}\n",
                       capture_output=True, text=True)
    try:
        got = float(r.stdout.splitlines()[1].split("(")[-1].strip(" )"))
    except (IndexError, ValueError):
        bad += 1; continue
    if abs(got - reported) > 0.011:
        bad += 1
print(bad)
PY
}

pass=0; fail=0

for shape in dots xpaired farpair; do
  fa=$WORK/$shape.fa

  run "$WORK/$shape.cpu" "$WORK/$shape.cpu.err" 0 -C -i "$fa"
  run "$WORK/$shape.gpu" "$WORK/$shape.gpu.err" 1 -C -i "$fa"

  # `grep -c` already prints 0 when it matches nothing, and exits 1 while doing
  # so. An `|| echo 0` fallback therefore appends a SECOND zero, and the "$ns"
  # comparisons below become `[: 0\n0: integer expected` -- an assertion that
  # errors instead of running, while the script still reports a pass.
  ns=$(grep -c "sweep shape:" "$WORK/$shape.gpu.err" 2>/dev/null)
  ns=${ns:-0}

  # 0 -- both arms produced output. Empty == empty is not a match, it is a
  #      test that never ran.
  if [ ! -s "$WORK/$shape.cpu" ] || [ ! -s "$WORK/$shape.gpu" ]; then
    echo "  $shape: FAIL -- an arm produced NO OUTPUT"
    sed -n '1,3p' "$WORK/$shape.gpu.err"
    fail=$((fail+1)); continue
  fi

  # 1 -- the two routes agree
  if ! cmp -s "$WORK/$shape.cpu" "$WORK/$shape.gpu"; then
    echo "  $shape: FAIL -- GPU route DIFFERS from CPU route"
    diff "$WORK/$shape.cpu" "$WORK/$shape.gpu" | head -6
    fail=$((fail+1)); continue
  fi

  # 2 -- routing expectation
  if [ "$EXPECT" = routed ] && [ "$ns" -ne 0 ]; then
    echo "  $shape: FAIL -- expected constraints to be DECLINED, but the sweep"\
         "ran $ns times. Either the guard regressed or this should be run with"\
         "--expect-accelerated."
    fail=$((fail+1)); continue
  fi
  if [ "$EXPECT" = accelerated ] && [ "$ns" -eq 0 ]; then
    echo "  $shape: FAIL -- expected the sweep to run, but it folded on the CPU;"\
         "the agreement above therefore tested nothing"
    fail=$((fail+1)); continue
  fi

  # 3 -- the constraint had teeth (dots is the control and must NOT)
  same_as_free=1
  cmp -s "$WORK/$shape.cpu" "$WORK/free.out" || same_as_free=0
  if [ "$shape" = dots ]; then
    if [ "$same_as_free" -ne 1 ]; then
      echo "  $shape: FAIL -- an all-dots constraint changed the answer"
      diff "$WORK/free.out" "$WORK/$shape.cpu" | head -6
      fail=$((fail+1)); continue
    fi
  elif [ "$same_as_free" -eq 1 ]; then
    echo "  $shape: FAIL -- the constrained answer equals the UNCONSTRAINED one."\
         "The constraint changed nothing, so this comparison proves nothing."
    fail=$((fail+1)); continue
  fi

  # 4 -- self-consistency of the answer actually returned
  bad=$(selfconsistent "$WORK/$shape.gpu")
  if [ "${bad:-1}" -ne 0 ]; then
    echo "  $shape: FAIL -- $bad records report an energy that is not their own"\
         "structure's energy"
    fail=$((fail+1)); continue
  fi

  if [ "$shape" = dots ]; then
    note="control: reproduces the free fold"
  else
    note="bites: $(diff "$WORK/free.out" "$WORK/$shape.cpu" | grep -c '^<') lines differ from the free fold"
  fi
  echo "  $shape: ok, $ns sweeps, energies self-consistent  [$note]"
  pass=$((pass+1))
done

echo
echo "$((pass+fail)) shapes: $pass pass, $fail fail"
if [ "$fail" -ne 0 ]; then
  echo "RESULT: FAILED"
  exit 1
fi
if [ "$EXPECT" = routed ]; then
  echo "RESULT: hard constraints are correctly DECLINED and fold on upstream's path"
else
  echo "RESULT: the GPU path honours hard structure constraints"
fi
exit 0
