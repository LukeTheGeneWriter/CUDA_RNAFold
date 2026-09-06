#!/bin/bash
#
# Salt corrections on the GPU path.
#
# Three things have to hold, and the first two are not the same claim:
#
#   1. The FROZEN VECTORS still reproduce. tests/salt/salt_reference.txt was
#      taken from pristine ViennaRNA 2.7.2 before any of this was written, so
#      it is the only check here that is anchored outside our own tree. It runs
#      on 4 records, which is below MIN_GPU_BATCH -- deliberately: this arm is
#      about upstream semantics, not about the device.
#   2. The GPU route matches the CPU route, on a batch big enough to actually
#      reach the device, and with the sweep counted so a CPU fallback cannot
#      masquerade as a passing GPU test.
#   3. Salt CHANGES the answer. A run where the correction was silently dropped
#      on both routes would satisfy 2 perfectly.
#
# Salt is worth this care because it is not a rounding effect: at 0.1 M the MFE
# moves ~20% against the 1.021 M default.
#
# Usage: verify_salt_parity.sh [build-tree]

set -u

TREE=${1:-$HOME/port27rel}
BIN=$TREE/src/bin/RNAfold
REF=$TREE/tests/salt
[ -d "$REF" ] || REF=$(dirname "$0")/../tests/salt
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "binary   : $BIN"
echo "reference: $REF/salt_reference.txt"
echo

[ -x "$BIN" ] || { echo "FAIL: no RNAfold at $BIN"; exit 1; }
[ -f "$REF/salt_test.fa" ] || { echo "FAIL: no frozen salt input at $REF"; exit 1; }

fail=0

# ---------------------------------------------------------------------------
echo "--- 1. frozen vectors from pristine 2.7.2 (CPU semantics)"
# ---------------------------------------------------------------------------
checked=0
while read -r salt idx want; do
  case "$salt" in \#*|"") continue;; esac
  got=$(env -u RNA_GPU_CHUNK "$BIN" --noPS --salt "$salt" -i "$REF/salt_test.fa" 2>/dev/null \
        | awk '/^>/{n++} n=='"$((idx+1))"' && /\(/{ if (match($0,/\(\s*-?[0-9.]+\)$/)) {
                 e=substr($0,RSTART+1,RLENGTH-2); gsub(/ /,"",e); print e; exit } }')
  checked=$((checked+1))
  if [ "$got" != "$want" ]; then
    echo "    salt=$salt record $idx: got '$got', frozen reference says '$want'"
    fail=$((fail+1))
  fi
done < "$REF/salt_reference.txt"

if [ "$checked" -eq 0 ]; then
  echo "    FAIL: parsed ZERO reference rows -- this arm verified nothing"
  fail=$((fail+1))
else
  echo "    $checked frozen energies checked, $fail mismatching"
fi
echo

# ---------------------------------------------------------------------------
echo "--- 2. GPU route vs CPU route (and 3. does salt bite?)"
# ---------------------------------------------------------------------------
# A batch large enough to clear MIN_GPU_BATCH, built from the frozen sequences
# so the arms stay comparable with arm 1.
python3 - "$REF/salt_test.fa" "$WORK/big.fa" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
recs, hdr, buf = [], None, []
for line in open(src):
    line = line.strip()
    if line.startswith(">"):
        if hdr is not None: recs.append((hdr, "".join(buf)))
        hdr, buf = line, []
    elif line: buf.append(line)
if hdr is not None: recs.append((hdr, "".join(buf)))
with open(dst, "w") as f:
    for r in range(6):
        for i, (h, s) in enumerate(recs):
            f.write(f"{h}_{r}_{i}\n{s}\n")
print(f"    batch: {6*len(recs)} records")
PY

default_out=""
for salt in 0.05 0.1 0.2 0.5 1.021 2.0 5.0; do
  env -u RNA_GPU_CHUNK "$BIN" --noPS --salt "$salt" -i "$WORK/big.fa" \
      > "$WORK/cpu.$salt" 2>/dev/null
  RNA_GPU_CHUNK=0 "$BIN" --noPS --salt "$salt" -i "$WORK/big.fa" \
      > "$WORK/gpu.$salt" 2> "$WORK/err.$salt"

  ns=$(grep -c "sweep shape:" "$WORK/err.$salt" 2>/dev/null)
  ns=${ns:-0}

  if [ ! -s "$WORK/cpu.$salt" ] || [ ! -s "$WORK/gpu.$salt" ]; then
    echo "    salt=$salt: FAIL -- an arm produced no output"
    sed -n '1,2p' "$WORK/err.$salt"
    fail=$((fail+1)); continue
  fi

  if ! cmp -s "$WORK/cpu.$salt" "$WORK/gpu.$salt"; then
    echo "    salt=$salt: FAIL -- GPU differs from CPU"
    diff "$WORK/cpu.$salt" "$WORK/gpu.$salt" | head -4
    fail=$((fail+1)); continue
  fi

  if [ "$ns" -eq 0 ]; then
    echo "    salt=$salt: FAIL -- 0 GPU sweeps; folded on the CPU fallback, so"\
         "the agreement above tested nothing"
    fail=$((fail+1)); continue
  fi

  # teeth: every non-default salt must move the answer away from the default
  if [ "$salt" = 1.021 ]; then
    default_out=$WORK/gpu.$salt
    # and the default must be identical to no --salt at all
    RNA_GPU_CHUNK=0 "$BIN" --noPS -i "$WORK/big.fa" > "$WORK/gpu.nosalt" 2>/dev/null
    if ! cmp -s "$WORK/gpu.$salt" "$WORK/gpu.nosalt"; then
      echo "    salt=$salt: FAIL -- the DEFAULT salt changed the answer;"\
           "the correction is being applied when it should be zero"
      fail=$((fail+1)); continue
    fi
    echo "    salt=$salt: identical, $ns sweeps  [default: matches a no --salt run]"
  else
    echo "    salt=$salt: identical, $ns sweeps"
  fi
done

# arm 3, once the default output is known
if [ -n "$default_out" ]; then
  bit=0
  for salt in 0.05 0.1 0.2 0.5 2.0 5.0; do
    cmp -s "$WORK/gpu.$salt" "$default_out" || bit=$((bit+1))
  done
  if [ "$bit" -ne 6 ]; then
    echo "    FAIL -- only $bit of 6 non-default salt values changed the answer."\
         "A correction that changes nothing would pass every check above."
    fail=$((fail+1))
  else
    echo "    all 6 non-default salt values move the answer [the correction bites]"
  fi
else
  echo "    FAIL -- the default-salt arm never ran, so teeth could not be checked"
  fail=$((fail+1))
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "RESULT: the GPU path reproduces ViennaRNA 2.7.2's salt corrections"
  exit 0
fi
echo "RESULT: FAILED ($fail)"
exit 1
