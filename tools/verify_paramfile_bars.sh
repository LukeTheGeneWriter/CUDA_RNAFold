#!/bin/bash
# The four -P / --paramFile bars from PORT_NSP_PARAMFILE_SCOPE.md 2.4.
#
# WHY THIS FILE EXISTS. `-P` CANNOT BE GUARDED: vrna_params_load() mutates
# library globals, so by fold-compound time it has left no flag for the routing
# guard to test. Every assumption the CUDA path makes about the parameter
# tables therefore has to be checked by running it, not by declining it. One of
# them (MAX_NINIO, a #define of 300 "checked" by assert(300 == 300)) was a LIVE
# SILENT WRONG ANSWER on 9 of 12 records, found exactly this way.
#
# The reference is always THE SAME BINARY with the accelerator off
# (RNA_GPU_CHUNK unset), so the accelerator is the only variable -- a real
# reference implementation would be comparing two builds and two parameter
# loaders at once.
#
#   bar 1  stock file re-read from disk   -- isolates the LOADING path
#   bar 2  -P DNA, with and without salt  -- also closes --helical-rise /
#                                            --backbone-length, which have never
#                                            been given an input where they bite
#   bar 3  perturbed NINIO maximum        -- the regression test for the fix
#   bar 4  perturbed stack row + int16    -- the int16 offset bound is B/2 x 340,
#                                            where 340 is the most negative
#                                            DEFAULT stack37 entry. A file
#                                            replaces stack37, so the bound is
#                                            an assumption, not a proof.
#
# Usage: verify_paramfile_bars.sh [build-tree] [input]
set -u
B=${1:-$HOME/port27fml}
IN=${2:-}
BIN=$B/src/bin/RNAfold
PAR=$B/misc/rna_turner2004.par

[ -x "$BIN" ] || { echo "no RNAfold at $BIN" >&2; exit 2; }
[ -f "$PAR" ] || { echo "no stock parameter file at $PAR" >&2; exit 2; }

# NOT /tmp: under WSL it can be cleared between invocations, and a probe whose
# fixtures have vanished reports 0 differences for every arm -- which reads as
# "nothing bites" rather than "nothing ran". Cost one wrong conclusion.
W=${VRNA_BAR_DIR:-$HOME/parbars}
rm -rf "$W"; mkdir -p "$W" || exit 2

# MIXED lengths on purpose. The --noLP + RNA_SLOT_FLOW defect was invisible to
# uniform-length fixtures, and every fixture in this project has been mixed
# since.
if [ -z "$IN" ]; then
  IN=$W/in.fa
  python3 - "$IN" <<'PY'
import random, sys
random.seed(20260910)
with open(sys.argv[1], "w") as f:
    for i, n in enumerate((62, 91, 130, 177, 224, 260, 301, 344, 375, 401, 208, 155)):
        f.write(">r%d_%d\n%s\n" % (i, n, "".join(random.choice("ACGU") for _ in range(n))))
PY
fi

echo "binary: $BIN"
echo "input : $IN  ($(grep -c '^>' "$IN") records, mixed length)"
echo

pass=0; fail=0

# ---------------------------------------------------------------------------
# run TAG EXPECT_ROUTE [env VAR=VAL ...] -- OPTIONS...
# Compares accelerator-on against accelerator-off, byte for byte, and asserts
# the on-run actually took the route it claims.
# ---------------------------------------------------------------------------
run_bar() {
  local tag=$1 expect=$2; shift 2
  local -a envs=()
  while [ "${1:-}" != "--" ]; do envs+=("$1"); shift; done
  shift

  env "${envs[@]}"                     "$BIN" --noPS "$@" -i "$IN" \
      > "$W/$tag.off" 2> "$W/$tag.off.err"; local rc_off=$?
  env "${envs[@]}" RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 \
      "$BIN" --noPS "$@" -i "$IN" \
      > "$W/$tag.on"  2> "$W/$tag.on.err";  local rc_on=$?

  local sweeps; sweeps=$(grep -c 'sweep shape:' "$W/$tag.on.err")

  if [ $rc_off -ne $rc_on ]; then
    printf '  %-34s EXIT DIFFERS (off %d, on %d)\n' "$tag" "$rc_off" "$rc_on"
    fail=$((fail+1)); return 1
  fi
  # Two empty outputs are not a match, they are a test that never ran.
  if [ ! -s "$W/$tag.off" ]; then
    printf '  %-34s NO OUTPUT from either side (exit %d)\n' "$tag" "$rc_off"
    grep -m1 -iE 'unrecognized|invalid|error' "$W/$tag.off.err" | sed 's/^/      /'
    fail=$((fail+1)); return 1
  fi

  if ! cmp -s "$W/$tag.off" "$W/$tag.on"; then
    printf '  %-34s *** DIFFERS *** on %d line(s)\n' "$tag" \
           "$(diff "$W/$tag.off" "$W/$tag.on" | grep -c '^[<>]')"
    fail=$((fail+1)); return 1
  fi

  if [ "$expect" = gpu ] && [ "$sweeps" -eq 0 ]; then
    printf '  %-34s *** SILENT CPU ROUTE *** right answer, GPU never ran\n' "$tag"
    fail=$((fail+1)); return 1
  fi
  if [ "$expect" = cpu ] && [ "$sweeps" -ne 0 ]; then
    printf '  %-34s *** ACCELERATED *** expected a CPU route\n' "$tag"
    fail=$((fail+1)); return 1
  fi

  printf '  %-34s identical  [%s route]\n' "$tag" "$expect"
  pass=$((pass+1)); return 0
}

# bites TAG BASELINE -- OPTIONS...   : does this option change the answer AT ALL?
# A parity test between two identical answers proves nothing. --helical-rise and
# --backbone-length sat in PORT_OPTION_STATUS.md 4 as "accelerated and
# identical, but DID NOT BITE" for exactly this reason.
bites() {
  local tag=$1 base=$2; shift 2
  "$BIN" --noPS "$@" -i "$IN" > "$W/$tag.bite" 2>/dev/null
  local d; d=$(diff "$base" "$W/$tag.bite" 2>/dev/null | grep -c '^[<>]')
  if [ "$d" -eq 0 ]; then
    printf '  %-34s DOES NOT BITE -- the parity result above is vacuous\n' "$tag"
    return 1
  fi
  printf '  %-34s bites (%d lines differ from the baseline)\n' "$tag" "$d"
  return 0
}

# ---------------------------------------------------------------------------
echo "--- baseline"
run_bar baseline gpu -- || true
BASE=$W/baseline.off

echo
echo "--- bar 1: the stock file, re-read from disk (isolates the LOADING path)"
run_bar stock_reread gpu -- -P "$PAR" || true
if cmp -s "$BASE" "$W/stock_reread.off"; then
  echo "  stock -P == no -P at all, as it must"
else
  echo "  *** the stock file re-read DIFFERS from no -P -- the loader is not neutral"
  fail=$((fail+1))
fi

echo
echo "--- bar 2: -P DNA  (also closes --helical-rise / --backbone-length)"
# -P DNA moves helical_rise, backbone_length and saltDPXInitFact
# (gengetopt_helpers.c:set_salt_DNA). The first two feed the SALT model only, so
# at default salt they cannot bite -- which is precisely why they have never
# been tested. The salted arm is the input that makes them bite.
run_bar dna          gpu -- -P DNA || true
bites   dna_bites    "$BASE" -P DNA || fail=$((fail+1))
run_bar dna_salt     gpu -- -P DNA --salt 0.2 || true
run_bar dna_salt_hi  gpu -- -P DNA --salt 1.5 || true
# And the geometry options themselves, under salt, where they can bite.
# MEASURED: --helical-rise 3.4 (the DNA value) does NOT bite at these lengths --
# 2.8 -> 3.4 moves no integer energy. 10 and 100 do (24 lines each). So the
# option IS wired, and the biting arm has to use a value that bites; -P DNA
# therefore closes --backbone-length but NOT --helical-rise.
run_bar rise_salt    gpu -- --salt 0.2 --helical-rise 10 || true
run_bar backbone_slt gpu -- --salt 0.2 --backbone-length 6.76 || true
"$BIN" --noPS --salt 0.2 -i "$IN" > "$W/salt_base" 2>/dev/null
bites  rise_bites     "$W/salt_base" --salt 0.2 --helical-rise 10     || fail=$((fail+1))
bites  backbone_bites "$W/salt_base" --salt 0.2 --backbone-length 6.76 || fail=$((fail+1))

echo
echo "--- bar 3: perturbed NINIO maximum (the MAX_NINIO regression test)"
sed 's/^# NINIO/# NINIO/' "$PAR" > "$W/ninio.par"
python3 - "$PAR" "$W/ninio.par" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().split("\n")
out, i = [], 0
while i < len(lines):
    out.append(lines[i])
    if lines[i].strip() == "# NINIO":
        # "# NINIO" is followed by a comment line then "<m> <dm> <max>"
        j = i + 1
        while j < len(lines) and (not lines[j].strip() or lines[j].lstrip().startswith("/*")):
            out.append(lines[j]); j += 1
        f = lines[j].split()
        f[-1] = "80"                      # the maximum, 300 -> 80
        out.append(" " + "  ".join(f))
        i = j
    i += 1
open(dst, "w").write("\n".join(out))
PY
grep -A3 '^# NINIO' "$W/ninio.par" | sed 's/^/      /'
run_bar ninio80 gpu -- -P "$W/ninio.par" || true
bites   ninio80_bites "$BASE" -P "$W/ninio.par" || fail=$((fail+1))

echo
echo "--- bar 4: perturbed stack row under RNA_FML_INT16=1"
# The int16 fML offset bound is B/2 x 340 = 32 x 340 = 10880 against an int16
# ceiling of 32766, where 340 is the most negative DEFAULT stack37 entry. A
# parameter file REPLACES stack37, so a sufficiently negative entry can push the
# offset past the ceiling. -1100 already exceeds 32766/32 = 1024.
python3 - "$PAR" "$W/stack.par" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().split("\n")
out, i, done = [], 0, False
while i < len(lines):
    out.append(lines[i])
    if lines[i].strip() == "# stack" and not done:
        out.append(lines[i + 1])                      # the /* CG GC ... */ header
        for k in range(i + 2, i + 9):                 # 7 rows
            f = lines[k].split()
            vals, tail = f[:7], " ".join(f[7:])
            vals = [("-2000" if v.lstrip("-").isdigit() and int(v) < 0 else v)
                    for v in vals]
            out.append("  " + "  ".join("%5s" % v for v in vals) + "    " + tail)
        i += 8
        done = True
    i += 1
open(dst, "w").write("\n".join(out))
PY
grep -A9 '^# stack$' "$W/stack.par" | head -10 | sed 's/^/      /'
# int32 first: the perturbed file must be VALID and must bite, or bar 4 is
# testing nothing.
run_bar stack_i32 gpu -- -P "$W/stack.par" || true
bites   stack_bites "$BASE" -P "$W/stack.par" || fail=$((fail+1))
# Then the same file with the int16 encoding on. THIS is the bar.
run_bar stack_i16 gpu RNA_FML_INT16=1 -- -P "$W/stack.par" || true
# Control: int16 with the STOCK table must still be identical, so a failure
# above is attributable to the parameter file and not to int16 in general.
run_bar stock_i16 gpu RNA_FML_INT16=1 -- -P "$PAR" || true


# A guard that shuts on EVERYTHING also passes every parity test. Assert the vet
# fired on the perturbed table and did NOT fire on the stock one -- the same
# "did it over-tighten?" check the --nsp guard carries.
echo
echo "--- bar 4b: the vet must decline the perturbed table and ONLY that one"
for t in stock_i16:0 stack_i16:1; do
  tag=${t%%:*}; want=${t##*:}
  got=$(grep -c 'RNA_FML_INT16 DECLINED' "$W/$tag.on.err" 2>/dev/null || true)
  req=$(grep -c 'RNA_FML_INT16=1: fml_j is 16-bit' "$W/$tag.on.err" 2>/dev/null || true)
  if [ "$req" -eq 0 ]; then
    printf '  %-34s int16 was never REQUESTED -- the arm proves nothing
' "$tag"
    fail=$((fail+1))
  elif [ "$got" -eq "$want" ]; then
    printf '  %-34s declined=%s, as required
' "$tag" "$got"
    pass=$((pass+1))
  else
    printf '  %-34s *** declined=%s, expected %s ***
' "$tag" "$got" "$want"
    fail=$((fail+1))
  fi
done

echo
echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ] || echo "RESULT: at least one -P assumption does not hold."
exit $(( fail != 0 ))
