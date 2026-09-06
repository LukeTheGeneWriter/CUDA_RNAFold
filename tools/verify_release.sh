#!/bin/bash
# The whole bar, in one place, for the state that goes to upstream.
#
#   1. builds WITHOUT --enable-cuda   (the accelerator must be absent-able)
#   2. builds WITH    --enable-cuda
#   3. make check both ways
#   4. byte-identical reference set through the GPU chunk path, incl. F_extreme
#   5. option-surface parity, GPU route vs CPU route
#   6. the relaxed guards actually run on the GPU
#
# Verification builds use --enable-asserts: NDEBUG is the release default, and
# the asserts are what turned two device failures in this port into precise
# diagnostics instead of wrong answers.
set -u
export PATH="$HOME/miniforge3/bin:$PATH"
S=${1:-$HOME/port27rel}
R=/mnt/c/Users/lukef/CUDA_RNAFold

for d in "$HOME/port27git"; do
  git -C "$d" fetch -q origin port27
  git -C "$d" reset -q --hard origin/port27
done

rm -rf "$S"
git clone -q --shared -b port27 "$HOME/port27git" "$S" || exit 2
cd "$S" || exit 2
tar -xjf src/dlib-*.tar.bz2 -C src/ 2>/dev/null
tar -xzf src/libsvm-*.tar.gz -C src/ 2>/dev/null
echo "tree: $(git log --oneline -1)"
./autogen.sh > autogen.log 2>&1 || { tail -5 autogen.log; exit 2; }

common="--without-python --without-perl --without-swig --without-doc
        --without-rnaxplorer --without-forester --without-kinfold
        --without-rnalocmin --enable-asserts"

fails=0
for arm in nocuda cuda; do
  extra=""
  [ "$arm" = cuda ] && extra="--enable-cuda"
  echo
  echo "=================== $arm ==================="
  make distclean > /dev/null 2>&1
  ./configure $common $extra PYTHON3="$(command -v python3)" > "cfg.$arm.log" 2>&1 \
    || { grep -m2 'configure: error' "cfg.$arm.log"; fails=$((fails+1)); continue; }
  make -j4 > "mk.$arm.log" 2>&1 \
    || { grep -inE '\berror\b' "mk.$arm.log" | head -6; fails=$((fails+1)); continue; }
  echo "  build ok"

  make check > "ck.$arm.log" 2>&1
  tot=$(grep -E '^# TOTAL' "ck.$arm.log" | tr -dc '0-9')
  bad=$(grep -E '^# (FAIL|ERROR)' "ck.$arm.log" | tr -dc '0-9' | tr -d '\n')
  if [ -z "$tot" ]; then
    echo "  make check DID NOT RUN"; fails=$((fails+1))
  else
    echo "  make check TOTAL=$tot  FAIL+ERROR digits=[$bad]"
    grep -E '^# (TOTAL|PASS|FAIL|ERROR)' "ck.$arm.log" | sed 's/^/    /'
    grep -qE '^# FAIL:  0' "ck.$arm.log" && grep -qE '^# ERROR: 0' "ck.$arm.log" \
      || fails=$((fails+1))
  fi
done

echo
echo "=================== byte-identical bar (GPU chunk path) ==================="
bash "$R/tools/verify_gpu_cli.sh" "$S" 0 || fails=$((fails+1))

echo
echo "=================== option-surface parity ==================="
bash "$R/tools/verify_option_parity.sh" "$S" || fails=$((fails+1))

echo
echo "=================== relaxed guards run on the GPU ==================="
bash "$R/tools/verify_relaxed_options.sh" "$S" || fails=$((fails+1))

echo
[ $fails -eq 0 ] && echo "RELEASE BAR: GREEN" || echo "RELEASE BAR: $fails sections failed"
exit $fails
