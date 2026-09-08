#!/bin/bash
# PAIRWISE option-combination audit -- the matrix owed at the end of
# PORT_FEATURE_AUDIT.md.
#
# WHY A MATRIX AND NOT A LIST. tools/verify_option_parity.sh walks options
# SINGLY. That is not enough, and we know it is not enough because two refusals
# were found on 2026-09-07 that no per-option check could have predicted:
#
#   RNA_FML_INT16 + --noLP        near-INF finite values in fML exceed the
#                                 16-bit per-block offset. Each half is
#                                 individually correct and verified.
#   RNA_FML_INT16 + RNA_SLOT_FLOW a slot handover leaves stale baselines.
#
# Both are PAIRWISE properties. Running each option alone proves nothing about
# them, and the accepted set is now six options plus five orthogonal env
# switches -- far more pairs than have ever been run together.
#
# WHAT EACH PAIR MUST SATISFY. From PORT_FEATURE_AUDIT.md, three things a
# per-option check cannot assert, plus one this file adds:
#
#   1. the pair produces the same answer as the CPU route;
#   2. the pair produces the same answer as the OTHER ORDER of enabling;
#   3. where a pair is refused, it is refused AT INIT rather than folding
#      something plausible;
#   4. (added here) the run took the ROUTE it claims to have taken, and folded
#      every record on it. verify_option_parity.sh prints the route and never
#      asserts it, so an option that quietly stopped being accelerated -- or
#      accelerated 20 records and CPU-folded the other 10 -- still scores a
#      pass there. That is the defect family that made int16 look like a
#      regression for a whole session, one level up.
#
# AND A PAIR THAT CANNOT BITE IS NOT A PASS. Every pair is checked to produce
# a DIFFERENT answer from the plain default fold. PORT_NOLP_SPEC.md's bar #4:
# a run that matches the unconstrained answer has not applied its options, and
# comparing two identical outputs is the shape of check that has fooled this
# project repeatedly (the --logML arm that compared two empty files; the
# uniq_ML arm that was `-p --MEA` wearing a label).
#
# Usage: verify_option_matrix.sh [build-tree] [input]
set -u
export PATH="$HOME/miniforge3/bin:$PATH"
B=${1:-$HOME/port27cuda}
IN=${2:-$HOME/rnatest/asc.fa}
BIN=$B/src/bin/RNAfold
. "$(dirname "$0")/bar_preflight.sh"
bar_preflight "$BIN" "$B" || exit 2
CUDA_LIBDIR=$(dirname "$(command -v nvcc)")/../lib
export LD_LIBRARY_PATH=$CUDA_LIBDIR
W=${TMPDIR:-/tmp}/vrna_optmatrix
rm -rf "$W"; mkdir -p "$W" || exit 2

NREC=$(grep -c '^>' "$IN")
echo "input  : $IN  ($NREC records)"
echo

pass=0; fail=0; n=0
declare -a FAILURES=()

note_fail() { fail=$((fail+1)); FAILURES+=("$1"); }

# --------------------------------------------------------------------------
# Primitives.
#
# The GPU path is opt-in: RNAfold.c:1299 sets gpu_enabled only when
# RNA_GPU_CHUNK is set AND non-empty. So the CPU reference is the SAME BINARY
# with that variable unset, which isolates the accelerator as the only
# variable rather than comparing two different builds.

cpu_run() {   # cpu_run TAG CLI...
  local tag=$1; shift
  env -u RNA_GPU_CHUNK -u RNA_FML_INT16 -u RNA_SLOT_FLOW \
      -u RNA_CONTINUOUS_FLOW -u RNA_MIN_GPU_BATCH \
      "$BIN" --noPS "$@" -i "$IN" > "$W/$tag.out" 2> "$W/$tag.err"
  echo $?
}

gpu_run() {   # gpu_run TAG "ENV..." CLI...
  local tag=$1 envs=$2; shift 2
  # shellcheck disable=SC2086 -- envs and CLI are controlled single-word tokens
  env -u RNA_FML_INT16 -u RNA_SLOT_FLOW -u RNA_CONTINUOUS_FLOW \
      -u RNA_MIN_GPU_BATCH RNA_GPU_CHUNK=0 $envs \
      "$BIN" --noPS "$@" -i "$IN" > "$W/$tag.out" 2> "$W/$tag.err"
  echo $?
}

sweeps_of()      { grep -c 'sweep shape:' "$W/$1.err"; }
records_of()     { grep -c '^>' "$W/$1.out"; }

# `peak/iteration N records` is a PEAK, not a total. Under RNA_SLOT_FLOW=2 the
# 30 records occupy 15 slots and it reads 15 -- so summing it says "15 of 30
# fell to the CPU" about a run whose `active record-rows` and `cells` totals
# are IDENTICAL to the non-flow run. It happens to equal the record count when
# no flow is on and every record is resident at once, which is why it has
# looked like a total. (bench v5 sums the same field, and is correct only
# because none of its arms enable flow.)
#
# The CELL total is the honest measure of how much actually folded on the
# device: it is a pure function of the record lengths, so it is invariant
# across options and across flow, and it drops exactly when records are folded
# on the CPU instead.
gpu_records_of() { grep -o 'peak/iteration [0-9]* records' "$W/$1.err" \
                   | awk '{s+=$2} END{print s+0}'; }
# Anchored on `cells; peak` -- the sweep line carries TWO "N cells;" fields, the
# sweep total and the per-iteration PEAK, and a regex that matched both summed
# them. Under flow only the peak moves, so every flow arm came out ~0.1% short
# and the script reported "0% of cells fell to the CPU": a failure whose own
# number contradicted it. Match the total alone.
cells_of()       { grep -oE '[0-9]+ cells; peak' "$W/$1.err" \
                   | awk '{s+=$1} END{print s+0}'; }

# A file that is empty, or short of records, is not an answer. Two empty
# outputs comparing equal is the oldest false pass in this project.
usable() {   # usable TAG -> 0 if it looks like a real answer
  local tag=$1
  [ -s "$W/$tag.out" ] || return 1
  [ "$(records_of "$tag")" = "$NREC" ] || return 1
  return 0
}

# --------------------------------------------------------------------------
# The dimensions.
#
# CLI-reachable ACCEPTED options only. uniq_ML is accepted but has NO usable
# RNAfold flag (--ImFeelingLucky also turns on stochastic backtracking, so it
# cannot be compared byte for byte), so it is out of reach here and stays
# covered by tests/mfe_cuda_fm1.ts. Recording that is the point: an option this
# file cannot reach must be named, not silently omitted.

CLI_TAGS=(T25 noGU salt noLP)
cli_args() {
  case $1 in
    T25)  echo "-T 25" ;;
    noGU) echo "--noGU" ;;
    salt) echo "--salt=0.2" ;;
    noLP) echo "--noLP" ;;
  esac
}

# Orthogonal env switches. chunk12 forces 5 chunks over 30 records, leaving a
# 2-record remainder -- below the stock MIN_GPU_BATCH of 10, so it exercises
# the PARTIAL CPU fallback that a single-chunk run never reaches.
ENV_TAGS=(int16 slotflow contflow minbatch1 chunk12)
env_args() {
  case $1 in
    int16)     echo "RNA_FML_INT16=1" ;;
    slotflow)  echo "RNA_SLOT_FLOW=2" ;;
    contflow)  echo "RNA_CONTINUOUS_FLOW=1" ;;
    minbatch1) echo "RNA_MIN_GPU_BATCH=1" ;;
    chunk12)   echo "RNA_GPU_CHUNK=12" ;;  # applied after the base, last wins
  esac
}

# Pairs KNOWN to be refused. The audit's job for these is not to prove they
# work -- it is to prove they are refused AT INIT and do not emit a plausible
# wrong answer.
is_refused() {   # is_refused TAG_A TAG_B
  case "$1+$2" in
    int16+noLP|noLP+int16)         return 0 ;;
    int16+slotflow|slotflow+int16) return 0 ;;
  esac
  return 1
}

# --------------------------------------------------------------------------
echo "=== 0. references ==="
rc=$(cpu_run default)
if [ "$rc" -ne 0 ] || ! usable default; then
  echo "  PREFLIGHT FAIL: the plain CPU fold did not produce $NREC records"
  exit 2
fi
DEFAULT_SHA=$(sha256sum "$W/default.out" | cut -c1-16)
echo "  plain CPU fold: $NREC records, $DEFAULT_SHA"

# The cell total for a fully-accelerated run of this input. Every GPU arm is
# measured against it; a shortfall means records folded on the CPU.
rc=$(gpu_run defgpu "")
if [ "$rc" -ne 0 ] || ! cmp -s "$W/default.out" "$W/defgpu.out"; then
  echo "  PREFLIGHT FAIL: the plain GPU fold does not match the plain CPU fold"
  exit 2
fi
CELLS_REF=$(cells_of defgpu)
if [ "$CELLS_REF" -le 0 ]; then
  echo "  PREFLIGHT FAIL: no sweep cells reported -- cannot measure the route"
  exit 2
fi
echo "  plain GPU fold: $(sweeps_of defgpu) chunk(s), $CELLS_REF cells = the reference"
echo

# --------------------------------------------------------------------------
# A pair check, in full. Every assertion this file makes lives here.
check_pair() {   # check_pair KIND TAG_A TAG_B
  local kind=$1 a=$2 b=$3
  local tag="${a}+${b}"
  local cli_a="" cli_b="" env_a="" env_b=""
  n=$((n+1))

  case $kind in
    cc) cli_a=$(cli_args "$a"); cli_b=$(cli_args "$b") ;;
    ce) cli_a=$(cli_args "$a"); env_b=$(env_args "$b") ;;
    ee) env_a=$(env_args "$a"); env_b=$(env_args "$b") ;;
  esac
  local cli="$cli_a $cli_b" envs="$env_a $env_b"

  # ---- 3. a refused pair must be refused AT INIT, not answered plausibly.
  if is_refused "$a" "$b"; then
    # shellcheck disable=SC2086
    rc=$(gpu_run "$tag.ref" "$envs" $cli)
    if [ "$rc" -eq 0 ]; then
      printf '  %-24s *** NOT REFUSED *** exit 0 -- it folded something\n' "$tag"
      note_fail "$tag: refusal missing"; return
    fi
    if [ -s "$W/$tag.ref.out" ]; then
      printf '  %-24s *** REFUSED LATE *** exit %d but %d records already emitted\n' \
             "$tag" "$rc" "$(records_of "$tag.ref")"
      note_fail "$tag: refused after emitting output"; return
    fi
    if ! grep -q 'cannot be combined' "$W/$tag.ref.err"; then
      printf '  %-24s *** exit %d but no refusal message -- crashed?\n' "$tag" "$rc"
      note_fail "$tag: nonzero exit without a refusal message"; return
    fi
    printf '  %-24s refused at init (exit %d, no output)\n' "$tag" "$rc"
    pass=$((pass+1)); return
  fi

  # ---- reference: the same CLI options folded entirely on the CPU.
  # shellcheck disable=SC2086
  rc=$(cpu_run "$tag.cpu" $cli)
  if [ "$rc" -ne 0 ] || ! usable "$tag.cpu"; then
    printf '  %-24s *** CPU REFERENCE UNUSABLE *** exit %d, %d records\n' \
           "$tag" "$rc" "$(records_of "$tag.cpu")"
    note_fail "$tag: cpu reference unusable"; return
  fi

  # ---- the pair must BITE -- but only where an ENERGY-MODEL option is in it.
  #
  # A CLI option that returns the plain fold has not been applied, and would
  # then compare equal to anything (PORT_NOLP_SPEC.md bar #4). The env switches
  # are the opposite case: they are performance knobs, and their whole contract
  # is that they do NOT move the answer. So for switch x switch the invariant
  # is inverted -- the answer must EQUAL the default, and a switch pair that
  # changed it is the failure.
  if [ "$kind" = ee ]; then
    if ! cmp -s "$W/$tag.cpu.out" "$W/default.out"; then
      printf '  %-24s *** REFERENCE MOVED *** env switches changed the CPU answer\n' "$tag"
      note_fail "$tag: env switches are not answer-neutral"; return
    fi
  elif cmp -s "$W/$tag.cpu.out" "$W/default.out"; then
    printf '  %-24s *** DOES NOT BITE *** same answer as the plain fold\n' "$tag"
    note_fail "$tag: pair does not change the answer"; return
  fi

  # ---- 1. the accelerated pair must equal the CPU route.
  # shellcheck disable=SC2086
  rc=$(gpu_run "$tag.gpu" "$envs" $cli)
  if [ "$rc" -ne 0 ]; then
    printf '  %-24s *** GPU EXIT %d ***\n' "$tag" "$rc"
    grep -m1 -iE 'error|cannot|refus' "$W/$tag.gpu.err" | sed 's/^/        /'
    note_fail "$tag: gpu exit $rc"; return
  fi
  if ! usable "$tag.gpu"; then
    printf '  %-24s *** GPU OUTPUT UNUSABLE *** %d records\n' \
           "$tag" "$(records_of "$tag.gpu")"
    note_fail "$tag: gpu output unusable"; return
  fi
  if ! cmp -s "$W/$tag.cpu.out" "$W/$tag.gpu.out"; then
    printf '  %-24s *** DIFFERS from the CPU route ***\n' "$tag"
    diff "$W/$tag.cpu.out" "$W/$tag.gpu.out" | head -4 | sed 's/^/        /'
    note_fail "$tag: gpu != cpu"; return
  fi

  # ---- 4. the route. Asserted, not printed.
  local sw gr cl fb
  sw=$(sweeps_of "$tag.gpu"); gr=$(gpu_records_of "$tag.gpu")
  cl=$(cells_of "$tag.gpu")
  fb=$((CELLS_REF - cl))
  if [ "$sw" -eq 0 ]; then
    printf '  %-24s *** NO SWEEPS *** the answer is right but it folded on the CPU\n' "$tag"
    note_fail "$tag: silently took the CPU route"; return
  fi
  local route
  route="$sw ch, $((100 * cl / CELLS_REF))% of cells on GPU, peak $gr"
  if [ "$fb" -gt 0 ]; then
    # Legitimate ONLY where MIN_GPU_BATCH is in play, i.e. a forced-chunk arm
    # at stock threshold. Anywhere else it is the partial fallback that cost a
    # session, and it must not pass silently.
    # ...and NOT when minbatch1 is also set, because defeating exactly this
    # fallback is what RNA_MIN_GPU_BATCH=1 is for. Allowing it there would let
    # the one pair that tests the override pass without the override working.
    if { [ "$a" = chunk12 ] || [ "$b" = chunk12 ]; } &&
       [ "$a" != minbatch1 ] && [ "$b" != minbatch1 ]; then
      route="$route, $((100 * fb / CELLS_REF))% to CPU (MIN_GPU_BATCH remainder)"
    else
      printf '  %-24s *** PARTIAL FALLBACK *** %d%% of cells folded on the CPU\n' \
             "$tag" "$((100 * fb / CELLS_REF))"
      note_fail "$tag: $((100 * fb / CELLS_REF))% of cells fell to the CPU unexpectedly"
      return
    fi
  fi

  # ---- 2. the other order of enabling. Only meaningful where an order exists.
  if [ "$kind" = cc ]; then
    # shellcheck disable=SC2086
    rc=$(gpu_run "$tag.rev" "$envs" $cli_b $cli_a)
    if [ "$rc" -ne 0 ] || ! cmp -s "$W/$tag.gpu.out" "$W/$tag.rev.out"; then
      printf '  %-24s *** ORDER MATTERS *** %s then %s differs from %s then %s\n' \
             "$tag" "$a" "$b" "$b" "$a"
      note_fail "$tag: enabling order changes the answer"; return
    fi
    route="$route, both orders"
  elif [ "$kind" = ee ]; then
    # shellcheck disable=SC2086
    rc=$(gpu_run "$tag.rev" "$env_b $env_a" $cli)
    if [ "$rc" -ne 0 ] || ! cmp -s "$W/$tag.gpu.out" "$W/$tag.rev.out"; then
      printf '  %-24s *** ORDER MATTERS *** env order changes the answer\n' "$tag"
      note_fail "$tag: env order changes the answer"; return
    fi
    route="$route, both orders"
  fi

  printf '  %-24s identical  [%s]\n' "$tag" "$route"
  pass=$((pass+1))
}

# --------------------------------------------------------------------------
echo "=== 1. option x option  (accepted CLI options, both enabling orders) ==="
for ((i = 0; i < ${#CLI_TAGS[@]}; i++)); do
  for ((j = i + 1; j < ${#CLI_TAGS[@]}; j++)); do
    check_pair cc "${CLI_TAGS[i]}" "${CLI_TAGS[j]}"
  done
done

echo
echo "=== 2. option x switch  (each accepted option against each env switch) ==="
for a in "${CLI_TAGS[@]}"; do
  for b in "${ENV_TAGS[@]}"; do
    check_pair ce "$a" "$b"
  done
done

echo
echo "=== 3. switch x switch ==="
for ((i = 0; i < ${#ENV_TAGS[@]}; i++)); do
  for ((j = i + 1; j < ${#ENV_TAGS[@]}; j++)); do
    check_pair ee "${ENV_TAGS[i]}" "${ENV_TAGS[j]}"
  done
done

# --------------------------------------------------------------------------
echo
echo "NOT REACHED by this file, and why:"
echo "  uniq_ML   accepted, but no byte-comparable RNAfold flag exists"
echo "            (--ImFeelingLucky also enables stochastic backtracking)."
echo "            Covered by tests/mfe_cuda_fm1.ts instead."
echo "  logML     declined, and RNAfold has no --logML flag to bar it from."
echo "  triples+  this is a PAIRWISE audit. Both known refusals are pairwise,"
echo "            but nothing here proves a third option cannot interact."
echo
printf '%d pairs: %d ok, %d failing\n' "$n" "$pass" "$fail"
if [ "$fail" -ne 0 ]; then
  echo
  echo "FAILURES:"
  printf '  %s\n' "${FAILURES[@]}"
  echo "RESULT: $fail of $n pairs are not sound"
else
  echo "RESULT: all $n pairs agree with the CPU route, on the route they claim"
fi
exit "$fail"
