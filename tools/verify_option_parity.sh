#!/bin/bash
# Option-surface parity: for every option, does the CUDA build with the GPU
# path ENABLED produce exactly what it produces with the GPU path OFF?
#
# This is the claim being taken to upstream -- "a strict accelerator of the
# validated CPU code" -- so it is checked across the option surface rather than
# on default folds alone. The reference is the SAME BINARY with RNA_GPU_CHUNK
# unset, which isolates the accelerator as the only variable.
#
# Usage: verify_option_parity.sh [build-tree] [input]
set -u
export PATH="$HOME/miniforge3/bin:$PATH"
B=${1:-$HOME/port27cuda}
IN=${2:-$HOME/rnatest/asc.fa}
BIN=$B/src/bin/RNAfold
PARFILE=$B/misc/rna_turner2004.par
. "$(dirname "$0")/bar_preflight.sh"
bar_preflight "$BIN" "$B" || exit 2
CUDA_LIBDIR=$(dirname "$(command -v nvcc)")/../lib
export LD_LIBRARY_PATH=$CUDA_LIBDIR
W=${TMPDIR:-/tmp}/vrna_optparity
rm -rf "$W"; mkdir -p "$W" || exit 2

[ -x "$BIN" ] || { echo "no RNAfold at $BIN" >&2; exit 2; }
echo "binary: $BIN"
echo "input : $IN  ($(grep -c '^>' "$IN") records)"
echo

pass=0; fail=0; n=0

NREC=$(grep -c '^>' "$IN")

# check_sorted TAG EXPECT OPTIONS...
#   As check(), but compares the SORTED outputs. For options whose contract is
#   "same answers, order not guaranteed" -- --unordered emits records in
#   completion order by design, so byte-identity is simply the wrong bar and a
#   plain check() reports a DIFFERS that is the option working correctly.
check_sorted() {
  local tag=$1 expect=$2; shift 2
  n=$((n+1))
  "$BIN" --noPS "$@" -i "$IN" 2> "$W/$tag.off.err" | sort > "$W/$tag.off"
  RNA_GPU_CHUNK=0 "$BIN" --noPS "$@" -i "$IN" 2> "$W/$tag.on.err" | sort > "$W/$tag.on"
  local sweeps; sweeps=$(grep -c 'sweep shape:' "$W/$tag.on.err")

  if [ ! -s "$W/$tag.off" ]; then
    printf '  %-22s NO OUTPUT from either side\n' "$tag"; fail=$((fail+1)); return
  fi
  if ! cmp -s "$W/$tag.off" "$W/$tag.on"; then
    printf '  %-22s *** DIFFERS (sorted) ***\n' "$tag"; fail=$((fail+1)); return
  fi
  if [ "$expect" = gpu ] && [ "$sweeps" -eq 0 ]; then
    printf '  %-22s *** SILENT CPU ROUTE ***\n' "$tag"; fail=$((fail+1)); return
  fi
  printf '  %-22s identical when sorted  [%s route]\n' "$tag" "$expect"
  pass=$((pass+1))
}

# check TAG EXPECT OPTIONS...
#   EXPECT is `gpu` (must be accelerated, and for EVERY record) or `cpu` (must
#   route to the CPU). Added 2026-09-08: this file used to PRINT the route and
#   never assert it, so an option that quietly stopped being accelerated still
#   scored "identical [CPU route]" as a pass, and one accelerated for 20 of 30
#   records scored "identical [GPU: 1 sweeps]". The second is the partial
#   fallback that made int16 look like a regression for a whole session --
#   bench v3 gated on `sweeps == 0`, which catches only a TOTAL fallback.
check() {
  local tag=$1 expect=$2; shift 2
  n=$((n+1))
  set -- "${@//PARAMFILE/$PARFILE}"
  "$BIN" --noPS "$@" -i "$IN" > "$W/$tag.off" 2> "$W/$tag.off.err"; rc_off=$?
  RNA_GPU_CHUNK=0 "$BIN" --noPS "$@" -i "$IN" > "$W/$tag.on" 2> "$W/$tag.on.err"; rc_on=$?
  local sweeps; sweeps=$(grep -c 'sweep shape:' "$W/$tag.on.err")
  local gpurec; gpurec=$(grep -o 'peak/iteration [0-9]* records' "$W/$tag.on.err" \
                         | awk '{s+=$2} END{print s+0}')

  if [ $rc_off -ne $rc_on ]; then
    printf '  %-22s EXIT DIFFERS (off %d, on %d)\n' "$tag" "$rc_off" "$rc_on"
    fail=$((fail+1)); return
  fi

  # Two EMPTY outputs are not a match, they are a test that never ran. A
  # "logML --logML" case sat in this file reporting identical every run,
  # because RNAfold has no --logML flag: both sides exited 1 with no output and
  # cmp was perfectly happy. Refuse to score that as a pass.
  if [ ! -s "$W/$tag.off" ]; then
    printf '  %-22s NO OUTPUT from either side -- option rejected? (exit %d)\n' \
           "$tag" "$rc_off"
    grep -m1 -iE 'unrecognized|invalid|error' "$W/$tag.off.err" | sed 's/^/        /'
    fail=$((fail+1)); return
  fi
  if cmp -s "$W/$tag.off" "$W/$tag.on"; then
    # The answer is right. Now: did it get there the way we claim?
    if [ "$expect" = gpu ]; then
      if [ "$sweeps" -eq 0 ]; then
        printf '  %-22s *** SILENT CPU ROUTE *** right answer, but the GPU never ran\n' "$tag"
        fail=$((fail+1)); return
      fi
      if [ "$gpurec" -ne "$NREC" ]; then
        printf '  %-22s *** PARTIAL FALLBACK *** %d of %d records folded on the CPU\n' \
               "$tag" "$((NREC - gpurec))" "$NREC"
        fail=$((fail+1)); return
      fi
      printf '  %-22s identical  [GPU: %s sweeps, %d/%d records]\n' \
             "$tag" "$sweeps" "$gpurec" "$NREC"
    else
      if [ "$sweeps" -gt 0 ]; then
        printf '  %-22s *** UNEXPECTEDLY ACCELERATED *** %s sweeps for a declined option\n' \
               "$tag" "$sweeps"
        fail=$((fail+1)); return
      fi
      printf '  %-22s identical  [CPU route, as required]\n' "$tag"
    fi
    pass=$((pass+1))
  else
    printf '  %-22s *** DIFFERS ***\n' "$tag"
    diff "$W/$tag.off" "$W/$tag.on" | head -4 | sed 's/^/        /'
    fail=$((fail+1))
  fi
}

echo "--- options the GPU path supports (must be accelerated, for EVERY record)"
echo "--- ACCELERATED: must take the GPU route AND give the same answer"
check default           gpu
check temp37            gpu -T 37
check temp25            gpu -T 25
check dangles2          gpu --dangles=2
check partfunc          gpu -p
check partfunc0         gpu -p0
check mea               gpu -p --MEA
check bppm_thresh       gpu -p --bppmThreshold=1e-4
check betascale         gpu -p --betaScale=1.2
check pfscale           gpu -p --pfScale=1.07
check noLP              gpu --noLP
check noGU              gpu --noGU
check noTetra           gpu -4
check salt              gpu --salt=0.2
check salt_hi           gpu --salt=1.5
# ACCELERATED 2026-09-10 (G3): the sweep now scores quadruplexes into c/fML and
# the bps backtrack renders the box. Was `cpu` in this file until then.
check gquad             gpu -g
# int16 is a run-time gate, not a CLI flag -- set it for this arm only
RNA_FML_INT16=1 check gquad_i16 gpu -g
# ACCELERATED 2026-09-10: Energy()'s 0->7 promotion and rtype[] fixed, guard lifted.
check nsp_sym           gpu --nsp=-GA
check nsp_asym          gpu --nsp=GA
check paramfile         gpu -P PARAMFILE
check helical_rise      gpu --salt=0.2 --helical-rise=10
check backbone_len      gpu --salt=0.2 --backbone-length=6.76
check maxbpspan_full    gpu --maxBPspan=100000
check jobs2             gpu -j2
check_sorted unordered  gpu -j2 --unordered
check noconv            gpu --noconv
check autoid            gpu --auto-id
check idprefix          gpu --auto-id --id-prefix=zz
check noDP              gpu -p --noDP
check verbose           gpu -v
check loglevel          gpu --log-level=3

echo
echo "--- DECLINED: must route to the CPU and give the same answer"
check circ              cpu -c
check dangles0          cpu -d0
check dangles1          cpu -d1
check dangles3          cpu -d3
check noClosingGU       cpu --noClosingGU
check maxbpspan_50      cpu --maxBPspan=50
check energymodel       cpu --energyModel=1
check constraint        cpu -C
check canonicalonly     cpu -C --canonicalBPonly
check enforce           cpu -C --enforceConstraint
[ $fail -eq 0 ] && echo "RESULT: the CUDA build matches the CPU build across the option surface" \
                || echo "RESULT: $fail options differ"
exit $fail
