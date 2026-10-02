#!/bin/bash
#
# The zero-config contract, asserted rather than eyeballed.
#
# WHY THIS EXISTS. An accelerated build cannot detect the defects that matter most
# here, and one of them was live: RNA_RECORD_TRACE landed inside
# `#ifdef VRNA_WITH_CUDA` while being called from unconditional code, so a CPU-only
# build failed with three implicit-declaration errors. That is the bioconda and PyPI
# build -- no CUDA toolkit present -- so it would have broken every such build while
# looking perfectly fine on any machine with a GPU. Nothing but actually configuring
# without a toolkit could find it.
#
# The other half is the opposite failure: a GPU-less build of this tree emits ZERO
# bytes of stderr and byte-identical stdout, so "it ran and the answer was right" is
# not evidence that the device was used. Every accelerated arm here demands POSITIVE
# evidence -- the `sweep shape:` line -- and treats its absence as a failure.
#
# Six cases:
#   1  bare ./configure, toolkit present   -> accelerated       (criterion 1)
#   2  bare RNAfold, no environment at all -> device is used     (criterion 2)
#   3  --disable-cuda                      -> off, still builds
#   4  --enable-cuda, toolkit hidden       -> configure FAILS, does not downgrade
#   5  bare ./configure, toolkit hidden    -> succeeds, CPU-only (criterion 6)
#   6  the two builds' output is identical
#
# HIDING THE TOOLKIT IS THE FIDDLY PART. A first attempt filtered PATH entries whose
# name contained "cuda", which never reached the branch it claimed to test, because
# nvcc is /usr/bin/nvcc here. So case 4 and 5 "passed" while proving nothing. The
# shim below symlinks every executable in the usual bin directories EXCEPT nvcc, and
# then ASSERTS that nvcc is gone and gcc is still there before using it.
#
#   usage: tests/zeroconf_configure.sh [build-dir]    (default: $HOME/lfb)
#
# Leaves the tree configured accelerated, since that is what everything else assumes.

set -u

TREE=${1:-$HOME/lfb}
WORK=${TMPDIR:-$HOME}/zeroconf.$$
COMMON="--without-perl --without-python --without-swig --without-doc
        --without-rnaxplorer --without-forester --without-kinfold --without-rnalocmin"
# Overridable: this runs three full builds back to back, and on a small host that
# is enough memory pressure to get the run killed from outside.
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}

fail=0
pass() { printf '  ok      %s\n' "$1"; }
bad()  { printf '  FAILED  %s\n' "$1"; fail=$((fail + 1)); }

[ -d "$TREE" ] || { echo "no such build tree: $TREE"; exit 2; }
mkdir -p "$WORK"
cd "$TREE" || exit 2

# ---------------------------------------------------------------- the fixture
FA="$WORK/mix.fa"
python3 - "$FA" <<'PY'
import random, sys
rng = random.Random(20260927)
with open(sys.argv[1], "w") as f:
    for i in range(12):
        L = (600, 900, 1200)[i % 3]
        f.write(">t%d\n%s\n" % (i, "".join(rng.choice("ACGU") for _ in range(L))))
PY
[ -s "$FA" ] || { echo "fixture generation failed"; exit 2; }

# ------------------------------------------------------- a PATH with no nvcc
SHIM="$WORK/shimbin"
mkdir -p "$SHIM"
for d in /usr/local/bin /usr/bin /bin /usr/sbin /sbin; do
  [ -d "$d" ] || continue
  for f in "$d"/*; do
    b=$(basename "$f")
    case "$b" in nvcc|nvcc-*|cuda*|nvidia-smi) continue ;; esac
    [ -e "$SHIM/$b" ] || ln -s "$f" "$SHIM/$b" 2>/dev/null
  done
done
# Assert the shim before trusting it. A shim that still exposes nvcc would make
# cases 4 and 5 pass for the wrong reason, which is the whole failure this guards.
if PATH="$SHIM" command -v nvcc >/dev/null 2>&1; then
  echo "the shim still exposes nvcc -- cannot test the no-toolkit cases"; exit 2
fi
PATH="$SHIM" command -v gcc >/dev/null 2>&1 || { echo "shim has no gcc"; exit 2; }
NOCUDA_ENV="env -u CUDA_HOME -u CUDA_PATH -u CUDAToolkit_ROOT -u CONDA_PREFIX PATH=$SHIM"

cuda_on() { grep -q '^#define VRNA_WITH_CUDA 1' "$TREE/config.h"; }

echo "=== 1. bare ./configure with a toolkit present"
if ./configure $COMMON > "$WORK/c1.log" 2>&1; then
  cuda_on && pass "configure enabled CUDA with no flag" \
           || bad "configure did NOT enable CUDA (see $WORK/c1.log)"
  grep -q 'GPU Acceleration' "$WORK/c1.log" \
    && pass "the summary reports the GPU verdict" \
    || bad "the summary does not mention the GPU"
else
  bad "bare configure failed"
fi

if make -j"$JOBS" > "$WORK/m1.log" 2>&1; then
  pass "accelerated build"
else
  bad "accelerated build failed (see $WORK/m1.log)"
fi

echo "=== 2. bare RNAfold: no environment variable of ours set at all"
env -u RNA_GPU_CHUNK -u RNA_MIN_GPU_BATCH -u RNA_MIN_GPU_CELLS -u RNA_GPU_WORK_FLOOR \
    ./src/bin/RNAfold --noPS -i "$FA" > "$WORK/gpu.out" 2> "$WORK/gpu.err"
if [ $? -eq 0 ]; then pass "it ran"; else bad "it exited non-zero"; fi
# POSITIVE evidence. Without this the test passes on a CPU-only build.
if grep -q 'sweep shape:' "$WORK/gpu.err"; then
  pass "the GPU sweep ran with no environment set"
else
  bad "NO SWEEP -- the device was not used, which is the defect this tests for"
fi
grep -q 'GPU acceleration ON' "$WORK/gpu.err" \
  && pass "the run announced that acceleration is on" \
  || bad "the run did not announce its GPU verdict"

echo "=== 3. --disable-cuda"
if ./configure $COMMON --disable-cuda > "$WORK/c3.log" 2>&1; then
  cuda_on && bad "--disable-cuda still defined VRNA_WITH_CUDA" \
           || pass "--disable-cuda turned it off"
else
  bad "--disable-cuda made configure fail"
fi

echo "=== 4. --enable-cuda with no toolkit reachable: must REFUSE"
if $NOCUDA_ENV ./configure $COMMON --enable-cuda > "$WORK/c4.log" 2>&1; then
  bad "configure SUCCEEDED -- an explicit --enable-cuda was silently downgraded"
else
  grep -q 'enable-cuda was requested' "$WORK/c4.log" \
    && pass "configure refused, and said why" \
    || bad "configure failed, but not with the CUDA explanation"
fi

echo "=== 5. bare ./configure with no toolkit reachable: must SUCCEED, CPU-only"
if $NOCUDA_ENV ./configure $COMMON > "$WORK/c5.log" 2>&1; then
  cuda_on && bad "CUDA was enabled although no toolkit was reachable" \
           || pass "configure succeeded with CUDA off"
  grep -q 'CUDA backend not built' "$WORK/c5.log" \
    && pass "it said why, without --verbose" \
    || bad "it went quiet about the downgrade"
else
  bad "bare configure failed when no toolkit was present -- this breaks every CPU-only install"
fi

# THE ONE THAT CAUGHT A REAL DEFECT.
if $NOCUDA_ENV make -j"$JOBS" > "$WORK/m5.log" 2>&1; then
  pass "CPU-only build"
  ./src/bin/RNAfold --noPS -i "$FA" > "$WORK/cpu.out" 2> "$WORK/cpu.err"
  [ $? -eq 0 ] && pass "CPU-only fold ran" || bad "CPU-only fold failed"
  # stock ViennaRNA is silent; a CPU-only build of ours must be too
  if [ "$(wc -c < "$WORK/cpu.err")" -eq 0 ]; then
    pass "CPU-only build is silent on stderr, as stock ViennaRNA is"
  else
    bad "CPU-only build wrote $(wc -c < "$WORK/cpu.err") bytes to stderr"
    head -3 "$WORK/cpu.err" | sed 's/^/          /'
  fi
else
  bad "CPU-ONLY BUILD FAILED -- this is the bioconda/PyPI case (see $WORK/m5.log)"
  grep -iE '\berror\b' "$WORK/m5.log" | head -5 | sed 's/^/          /'
fi

echo "=== 6. do the two builds agree?"
if [ -s "$WORK/cpu.out" ] && [ -s "$WORK/gpu.out" ]; then
  cmp -s "$WORK/cpu.out" "$WORK/gpu.out" \
    && pass "byte-identical output from the CPU-only and accelerated builds" \
    || bad "the two builds disagree"
else
  bad "cannot compare: one of the two runs produced nothing"
fi

echo "=== restoring the accelerated configuration"
./configure $COMMON > "$WORK/c7.log" 2>&1 && cuda_on \
  && pass "tree left configured accelerated" \
  || bad "could not restore the accelerated configuration"
make -j"$JOBS" > "$WORK/m7.log" 2>&1 && pass "and rebuilt" || bad "restore build failed"

echo
if [ "$fail" -eq 0 ]; then
  echo "RESULT: the zero-config contract holds"
  rm -rf "$WORK"
  exit 0
else
  echo "RESULT: $fail check(s) FAILED -- logs kept in $WORK"
  exit 1
fi
