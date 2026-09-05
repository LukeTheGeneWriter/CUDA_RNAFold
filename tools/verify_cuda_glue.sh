#!/bin/bash
# Verify the CUDA build glue BOTH ways, because the whole claim is that the
# backend is opt-in and inert:
#
#   without --enable-cuda : must be indistinguishable from before it existed
#   with    --enable-cuda : must build, and must still produce upstream's answer
#                           because the engine declines everything for now
#
# Usage: verify_cuda_glue.sh [branch]
set -u
export PATH="$HOME/miniforge3/bin:$PATH"

S=$HOME/port27cuda
BRANCH=${1:-port27}

rm -rf "$S"
git clone -q --shared -b "$BRANCH" "$HOME/port27git" "$S" || exit 2
cd "$S" || exit 2
tar -xjf src/dlib-*.tar.bz2 -C src/ 2>/dev/null
tar -xzf src/libsvm-*.tar.gz -C src/ 2>/dev/null

echo "###### autogen (picks up the new m4/ac_rna_cuda.m4)"
./autogen.sh > autogen.log 2>&1 || { tail -20 autogen.log; exit 2; }
grep -q 'enable-cuda' configure && echo "  --enable-cuda present in configure" \
  || { echo "  --enable-cuda MISSING from configure"; exit 2; }

common="--without-python --without-perl --without-swig --without-doc
        --without-rnaxplorer --without-forester --without-kinfold
        --without-rnalocmin"

run_arm() {
  local name=$1; shift
  echo
  echo "=================== $name ==================="
  make distclean > /dev/null 2>&1
  ./configure $common PYTHON3="$(command -v python3)" "$@" > "configure.$name.log" 2>&1
  rc=$?
  echo "configure exit=$rc"
  [ $rc -eq 0 ] || { grep -m3 'configure: error' "configure.$name.log"; return 1; }

  grep -iE '^ *\* *CUDA|checking whether .*nvcc' "configure.$name.log" | head -3
  if grep -q 'VRNA_AM_SWITCH_CUDA_TRUE"\]="#"' config.status; then
    echo "  CUDA conditional: OFF"
  else
    echo "  CUDA conditional: ON"
  fi

  make -j4 > "make.$name.log" 2>&1
  rc=$?
  echo "make exit=$rc"
  [ $rc -eq 0 ] || { grep -inE '\berror\b|No rule' "make.$name.log" | head -10; return 1; }

  grep -c 'NVCC' "make.$name.log" | sed 's/^/  nvcc invocations: /'

  make check > "check.$name.log" 2>&1
  echo -n "  "; grep -E '^# (TOTAL|PASS|FAIL|ERROR)' "check.$name.log" | tr '\n' ' '; echo

  echo "  byte-identical bar:"
  bash /mnt/c/Users/lukef/CUDA_RNAFold/tools/verify_ref_md.sh "$S/src/bin/RNAfold" \
    2>&1 | grep -E 'IDENTICAL|DIFFERS|RESULT' | sed 's/^/    /'
}

## ARMS lets a re-run skip the arm that is already known green -- the bar's
## F_extreme record alone is ~7.5 minutes single-threaded.
ARMS=${ARMS:-"nocuda cuda"}

case " $ARMS " in *" nocuda "*) run_arm nocuda ;; esac
case " $ARMS " in *" cuda "*)   run_arm cuda --enable-cuda ;; esac
