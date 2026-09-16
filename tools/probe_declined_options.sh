#!/bin/bash
#
# Which DECLINED options are declined out of NECESSITY, and which out of caution?
#
# PORT_ACCELERATION_SCOPE.md's "Tier 0 -- may already work; needs TESTING, not
# building" has paid five times: noGU, uniq_ML, --nsp, -g and -d0 all moved from
# DECLINED to ACCELERATED once someone lifted the check and compared. Each of
# those measurements was made in a scratch build, which is both slow and exactly
# the shape of the stale-binary traps this project keeps writing down.
#
# This does it from a SHIPPED binary, using the RNA_ENGINE_ALLOW test hook
# (mfe/cuda/engine.c). For each declined option it runs the same binary twice --
# once down the CPU route it takes today, once with the guard lifted so the fold
# reaches the device -- and reports:
#
#   AGREES     every record byte-identical. A CANDIDATE, not a verdict: agreeing
#              on one fixture is what --nsp did for a year before a bigger one
#              caught 28 wrong answers. It says "worth a real bar", nothing more.
#   DIFFERS    the device does not reproduce this option. The guard is earning
#              its keep, and the DELTA tells you how badly.
#   TRAPPED    VRNA_CUDA_BACKSTOP fired -- gate 3 caught what gates 1 and 2 let
#              through. That is the tripwire working, and it is a result too.
#   SKIPPED    the option could not be exercised here (input not constructible),
#              so nothing is claimed.
#
# Two things are checked per case, not one: agreement with the CPU route, AND
# self-consistency -- does the energy the GPU route reports match the structure
# it returns, re-evaluated by RNAeval? The -C defect of 2026-09-06 was caught by
# the second check alone (fill said -14.30, backtrack's structure re-evaluated
# to -5.40), and it needed no reference implementation at all.
#
# Usage: probe_declined_options.sh [build-tree]

set -u

TREE=${1:-$HOME/port27fml}
BIN=$TREE/src/bin/RNAfold
EVAL=$TREE/src/bin/RNAeval
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "binary : $BIN"
[ -x "$BIN" ]  || { echo "FAIL: no RNAfold at $BIN";  exit 1; }
[ -x "$EVAL" ] || { echo "FAIL: no RNAeval at $EVAL"; exit 1; }
echo

# --- fixture ---------------------------------------------------------------
# Short and few: this asks "does the device reproduce the option at all", and a
# disagreement shows up on the first record or not at all. Length variety is
# deliberate -- a uniform fixture cannot detect a per-record quantity being
# broadcast as a scalar, which is exactly the --maxBPspan defect.
N=8
python3 - "$WORK/seq.fa" <<'PY'
import random, sys
random.seed(4242)
with open(sys.argv[1], "w") as f:
    for i, L in enumerate((80, 95, 110, 128, 150, 173, 200, 240)):
        f.write(">s%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(L))))
PY

cpu () { env -u RNA_GPU_CHUNK RNA_MIN_GPU_BATCH=1 "$BIN" --noPS "$@" 2>/dev/null; }
gpu () { # gpu <allow-id> [args...]
  local id=$1; shift
  RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 RNA_ENGINE_ALLOW="$id" "$BIN" --noPS "$@" 2>"$WORK/gpu.err"
}

# Energy column only, for a delta that means something when structures differ.
energies () { grep -oE '\(\s*-?[0-9]+\.[0-9]+\)$' "$1" | tr -d '() ' ; }

CAND=0; NEED=0

probe () {  # probe <label> <allow-id> <input.fa> [args...]
  local label=$1 id=$2 fa=$3; shift 3

  cpu -i "$fa" "$@" > "$WORK/cpu.out" 2>"$WORK/cpu.err"
  if [ ! -s "$WORK/cpu.out" ]; then
    printf '  %-22s SKIPPED  (the CPU route itself produced nothing: %s)\n' \
           "$label" "$(head -1 "$WORK/cpu.err" | cut -c1-60)"
    return
  fi

  # IS THE REFERENCE ANSWER NON-DEGENERATE? Two empty structures compare equal,
  # and an option that forbids every pair produces exactly that. --energyModel
  # on an ACGU fixture does it: no pair is legal in its alphabet, both routes
  # return all-dots at 0.00, and the first version of this script called that
  # AGREES -- while the SAME option on its own alphabet has the GPU returning
  # 0.00 against the CPU's -45.10.
  if ! grep -qE '^[.()]*\(' "$WORK/cpu.out"; then
    printf '  %-22s SKIPPED  the CPU reference folds NOTHING here (all-dots) --\n' "$label"
    printf '  %-22s          two empty structures always compare equal\n' ""
    return
  fi

  # DOES THE OPTION BITE? An option that changes nothing makes both routes agree
  # for a reason that has nothing to do with the device. The first version of
  # this script reported the theophylline motif as AGREES on a sequence that
  # does not contain the aptamer -- eleven checks in this project have failed
  # exactly this way, and it is cheap to rule out: fold the same input WITHOUT
  # the option and require the answer to move.
  cpu -i "$fa" > "$WORK/plain.out" 2>/dev/null
  if cmp -s "$WORK/cpu.out" "$WORK/plain.out"; then
    printf '  %-22s SKIPPED  the option does not BITE on this input -- it changes\n' "$label"
    printf '  %-22s          nothing on the CPU either, so agreement proves nothing\n' ""
    return
  fi

  gpu "$id" -i "$fa" "$@" > "$WORK/gpu.out"
  local rc=$?

  # Gate 3's message does not contain the word "backstop" -- it says what it
  # caught and that gate 2 should have caught it first. Match on that, not on
  # the name of the macro, or every trip reads as a generic crash.
  if grep -q 'should have declined this' "$WORK/gpu.err"; then
    printf '  %-22s TRAPPED  gate 3: %s\n' "$label" \
           "$(grep -m1 'should have declined this' "$WORK/gpu.err" | cut -c1-72)"
    NEED=$((NEED+1)); return
  fi
  if [ $rc -ne 0 ] || [ ! -s "$WORK/gpu.out" ]; then
    printf '  %-22s TRAPPED  the forced run died (rc=%d): %s\n' \
           "$label" "$rc" "$(tail -1 "$WORK/gpu.err" | cut -c1-50)"
    NEED=$((NEED+1)); return
  fi

  # Did it actually reach the device? A guard lifted in gate 2 but not gate 1
  # sends the fold down the CPU path, where of course it agrees -- which is the
  # single most common way a check in this project has lied.
  local swept="no"
  grep -q 'sweep shape:' "$WORK/gpu.err" && swept="yes"

  if cmp -s "$WORK/cpu.out" "$WORK/gpu.out"; then
    if [ "$swept" = "no" ]; then
      # TWO DIFFERENT THINGS LOOK THE SAME HERE, and only one is a broken probe.
      # If the engine announced the lift, gate 1 and the NAMED gate-2 check both
      # opened -- so a fold that still did not sweep was declined by a DIFFERENT
      # gate-2 check, which is a result: the option is refused for a reason of
      # its own rather than because nobody looked. A ligand motif is that case --
      # vrna_sc_add_hi_motif() installs a SOFT constraint, so `fc->sc != NULL`
      # declines it whatever the "motif" id does.
      if grep -q 'RNA_ENGINE_ALLOW: lifting' "$WORK/gpu.err"; then
        printf '  %-22s DECLINED gate 2 refused it on another check, and the answer\n' "$label"
        printf '  %-22s          matches the CPU route -- the guard is doing its job\n' ""
        NEED=$((NEED+1)); return
      fi
      printf '  %-22s SKIPPED  identical, but NO SWEEP -- it never reached the device\n' "$label"
      return
    fi
    printf '  %-22s AGREES   %d/%d records identical, swept\n' "$label" "$N" "$N"
    CAND=$((CAND+1))
  else
    local d
    d=$(paste <(energies "$WORK/cpu.out") <(energies "$WORK/gpu.out") \
        | awk '{n++; if ($1!=$2) {m++; s+=($2-$1)}} END {printf "%d of %d differ, sum %+.2f kcal", m, n, s}')
    printf '  %-22s DIFFERS  %s (swept: %s)\n' "$label" "$d" "$swept"
    NEED=$((NEED+1))

    # WHICH KIND of wrong? Two failure modes, and they want different fixes.
    # If the forced run reproduces the PLAIN fold byte for byte, the device did
    # not crash or approximate -- it silently dropped the option, which is the
    # exact failure the guard exists to prevent.
    if cmp -s "$WORK/gpu.out" "$WORK/plain.out"; then
      printf '  %-22s          ...and it is byte-identical to the fold with NO option:\n' ""
      printf '  %-22s          the device SILENTLY IGNORES this one\n' ""
    fi
    selfcheck "$label" "$@"
  fi
}

# Does the energy a route REPORTS match the structure it RETURNS, re-evaluated?
# This needs no reference implementation, and on 2026-09-06 it is what caught the
# -C defect on its own: the fill said -14.30 and the backtrack's structure
# re-evaluated to -5.40. A route that is merely folding a DIFFERENT (legal)
# problem stays self-consistent; a route whose fill and backtrack disagree does
# not, and the two failures want different fixes.
selfcheck () {  # selfcheck <label> [model args...]
  local label=$1; shift
  awk '/^[ACGUTNRYKMSWBDHVabcdefghijklmnopqrstuvwxyz]+$/ {s=$0; next}
       /^[.()]+[ \t]*\(/ {print s; print $1}' "$WORK/gpu.out" > "$WORK/pairs.txt" 2>/dev/null
  [ -s "$WORK/pairs.txt" ] || return 0

  # RNAeval takes the model flags, not the fold flags: -C, --shape and friends
  # change what is SEARCHED, never what a given structure costs.
  local m=()
  for a in "$@"; do
    case "$a" in
      --noconv|--energyModel*|-d*|-T*|--temp*|-P*|--salt*|--noGU|--noLP|-4) m+=("$a") ;;
    esac
  done

  "$EVAL" "${m[@]+"${m[@]}"}" < "$WORK/pairs.txt" 2>/dev/null > "$WORK/eval.out" || return 0
  local bad
  bad=$(paste <(energies "$WORK/gpu.out") <(energies "$WORK/eval.out") \
        | awk '{ if (($1-$2)^2 > 0.0001) n++ } END { print n+0 }')
  if [ "${bad:-0}" != "0" ]; then
    printf '  %-22s          ...and %s of its own structures re-evaluate to a\n' "" "$bad"
    printf '  %-22s          DIFFERENT energy: fill and backtrack disagree\n' ""
  fi
}

# --- the cases -------------------------------------------------------------
echo "declined options, guard lifted one id at a time:"
echo

# (1) HARD CONSTRAINTS. Derived FROM the free fold, so they are guaranteed to
# bite: forcing unpaired exactly what the MFE pairs. A fixed pattern is the
# first version of this bar, and it compared two identical outputs.
cpu -i "$WORK/seq.fa" > "$WORK/free.out"
python3 - "$WORK/seq.fa" "$WORK/free.out" "$WORK/con.fa" <<'PY'
import sys
seqs = [l.strip() for l in open(sys.argv[1]) if not l.startswith(">")]
db   = [l.split()[0] for l in open(sys.argv[2]) if l and l[0] in ".()"]
out  = open(sys.argv[3], "w")
for i, (s, d) in enumerate(zip(seqs, db)):
    # force unpaired the first paired block we find; that is the shape that
    # made the 2026-09-06 defect visible.
    c = ["." for _ in s]
    hit = [k for k, ch in enumerate(d) if ch == "("][: max(4, len(s)//10)]
    for k in hit:
        c[k] = "x"
    out.write(">c%d\n%s\n%s\n" % (i, s, "".join(c)))
PY
probe "-C hard constraints" hc "$WORK/con.fa" -C

# (2) NON-DEFAULT ENERGY SET, ON ITS OWN ALPHABET.
#
# This one needs its own fixture and it is the lesson of the whole script.
# --energyModel 1/2 means "A pairs B, C pairs D": on an ACGU input NO pair is
# legal, both routes return all-dots at 0.00, and a comparison of two empty
# folds reports perfect agreement. On the ABCD alphabet the same option has the
# CPU finding -45.10 and the device returning 0.00 -- the opposite verdict from
# the same binary, decided entirely by the fixture's alphabet.
python3 - "$WORK/abcd.fa" <<'PY'
import random, sys
random.seed(2026)
with open(sys.argv[1], "w") as f:
    for i, L in enumerate((80, 95, 110, 128, 150, 173, 200, 240)):
        f.write(">a%d\n%s\n" % (i, "".join(random.choice("ABCD") for _ in range(L))))
PY
probe "--energyModel 1" energy_set "$WORK/abcd.fa" --noconv --energyModel 1
probe "--energyModel 2" energy_set "$WORK/abcd.fa" --noconv --energyModel 2

# (3) SOFT CONSTRAINTS via SHAPE reactivities. One record only: --shape applies
# one data file to the fold.
# The LONGEST record, not the first: an 80 nt fold often has no interior loop
# with unpaired bases on both sides, and the motif case needs one to copy.
tail -2 "$WORK/seq.fa" > "$WORK/one.fa"
cpu -i "$WORK/one.fa" > "$WORK/one.free" 2>/dev/null
python3 - "$WORK/one.fa" "$WORK/shape.dat" <<'PY'
import sys, random
random.seed(7)
seq = [l.strip() for l in open(sys.argv[1]) if not l.startswith(">")][0]
with open(sys.argv[2], "w") as f:
    for i, ch in enumerate(seq, 1):
        f.write("%d %s %.3f\n" % (i, ch, random.uniform(0.0, 1.5)))
PY
N_SAVE=$N; N=1
probe "--shape (soft)" soft "$WORK/one.fa" --shape="$WORK/shape.dat"

# (4) LIGAND MOTIF, DERIVED FROM THE FOLD'S OWN INTERIOR LOOP.
#
# A ligand motif binds a SEQUENCE *and* a STRUCTURE -- that is the whole point
# of the option: a protein that binds a known site and stabilises it. So a motif
# taken from the documentation cannot bite on a random sequence, and three
# attempts to make one bind that way failed, including a synthetic four-pair
# motif at -30 kcal/mol.
#
# The construction that works is to read BOTH halves off a real fold: find an
# interior loop in the free MFE structure -- a closing pair (i,j) with an
# enclosed pair (p,q) and unpaired bases on both sides -- and emit exactly that
# sequence and that dot-bracket as the motif. It is then guaranteed to be
# present and formable, and the bonus shows up as an energy shift: measured
# -21.80 -> -29.80 for a -8.0 motif, same structure, ligand bound.
python3 - "$WORK/one.fa" "$WORK/one.free" "$WORK/motif.arg" <<'PY' || echo "  (motif derivation failed)"
import sys
seq = [l.strip() for l in open(sys.argv[1]) if not l.startswith(">")][0]
db  = [l.split()[0] for l in open(sys.argv[2]) if l and l[0] in ".()"][0]

st, pairs = [], {}
for k, c in enumerate(db):
    if c == "(":
        st.append(k)
    elif c == ")":
        a = st.pop(); pairs[a] = k; pairs[k] = a

best = None
for i in sorted(x for x in pairs if pairs[x] > x):
    j = pairs[i]
    p = next((k for k in range(i + 1, j) if db[k] == "(" and pairs[k] < j), None)
    if p is None:
        continue
    q = pairs[p]
    u1, u2 = p - i - 1, j - q - 1
    if 1 <= u1 <= 6 and 1 <= u2 <= 6:
        best = (i, j, p, q, u1, u2)
        break

if best:
    i, j, p, q, u1, u2 = best
    open(sys.argv[3], "w").write("%s&%s,%s&%s,-8.0" % (
        seq[i:p+1], seq[q:j+1], "(" + "."*u1 + "(", ")" + "."*u2 + ")"))
PY
cpu -i "$WORK/one.fa" > "$WORK/one.free" 2>/dev/null
if [ -s "$WORK/motif.arg" ]; then
  probe "--motif (ligand)" motif "$WORK/one.fa" --motif="$(cat "$WORK/motif.arg")"
else
  printf '  %-22s SKIPPED  no interior loop in the free fold to build a motif from\n' "--motif (ligand)"
fi

# (5) A COMMAND FILE. Commands can add either hard or soft constraints, so this
# is checked as its own route rather than assumed to be one of the two above.
# DERIVED, like every other shape in this script: prohibit pairing for bases the
# free MFE actually pairs. A fixed "P 5 0 3" changes nothing on most inputs, and
# an option that does not bite proves nothing about the device.
python3 - "$WORK/one.free" "$WORK/cmd.txt" <<'PY'
import sys
db = [l.split()[0] for l in open(sys.argv[1]) if l and l[0] in ".()"][0]
paired = [k + 1 for k, c in enumerate(db) if c != "."][:6]
with open(sys.argv[2], "w") as f:
    for k in paired:
        f.write("P %d 0 1\n" % k)
PY
probe "--commands" cmds "$WORK/one.fa" --commands="$WORK/cmd.txt"
N=$N_SAVE

# (6) DANGLE MODELS 1 AND 3. These are the ONLY options gate 3 still watches,
# and the expectation is explicit: they should be caught, here, loudly. An
# AGREES on either of these would be the surprise of the year.
probe "-d1" dangles "$WORK/seq.fa" -d1
probe "-d3" dangles "$WORK/seq.fa" -d3

echo
echo "candidates (AGREES, worth a real bar): $CAND"
echo "confirmed necessary (DIFFERS/TRAPPED) : $NEED"
echo
echo "AGREES is not a licence to lift a guard. It means the option earns a"
echo "verify_*_parity.sh of its own, on a fixture big enough to disagree."
