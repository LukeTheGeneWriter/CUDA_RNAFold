#!/bin/bash
# Build with NO --without-* exclusions at all, to prove the excluded components
# (RNAforester + its bundled g2, the SWIG bindings, the Sphinx/doxygen manual)
# actually build here -- so that nothing is discovered to be broken later, at a
# point where it would be confused with our own changes.
#
# Deliberately a SEPARATE tree from the working build: this one is allowed to
# fail without costing us a known-good build.
set -u
export PATH="$HOME/miniforge3/bin:$PATH"

S=$HOME/port27full
REPO=/mnt/c/Users/lukef/CUDA_RNAFold
BRANCH=${1:-port27}

rm -rf "$S"
git clone -q --shared -b "$BRANCH" "$HOME/port27git" "$S" 2>/dev/null || \
  git clone -q -b "$BRANCH" "$REPO" "$S" || exit 2
cd "$S" || exit 2

tar -xjf src/dlib-*.tar.bz2 -C src/ 2>/dev/null
tar -xzf src/libsvm-*.tar.gz -C src/ 2>/dev/null

echo "###### autogen"
./autogen.sh > autogen.log 2>&1 || { tail -5 autogen.log; exit 2; }

# conda's perl 5.32 headers still #include <xlocale.h>, which glibc removed in
# 2.26. That is a conda packaging defect, not a ViennaRNA one, and it stops the
# Perl interface building. A one-line shim on the include path fixes it without
# modifying the conda environment, which other builds share.
mkdir -p compat
cat > compat/xlocale.h <<'SHIM'
/* Compatibility shim: glibc >= 2.26 dropped <xlocale.h> and folded its
 * contents into <locale.h>, but conda's perl 5.32 CORE headers still include
 * it. Not part of ViennaRNA. */
#ifndef VRNA_COMPAT_XLOCALE_H
#define VRNA_COMPAT_XLOCALE_H
#include <locale.h>
#endif
SHIM

echo "###### configure (no exclusions)"
./configure PYTHON3="$(command -v python3)" \
            CPPFLAGS="-I$S/compat" > configure.log 2>&1
rc=$?
echo "configure exit=$rc"
if [ $rc -ne 0 ]; then grep -m3 'configure: error' configure.log; exit 2; fi

echo "###### what configure decided to build"
sed -n '/Configure summary/,/^$/p' configure.log | head -60
grep -iE 'RNAforester|Perl 5|Python 3|Reference Manual|SWIG' configure.log | tail -20

echo "###### make"
make -j8 > make.log 2>&1
rc=$?
echo "make exit=$rc"
if [ $rc -ne 0 ]; then
  echo "--- first errors ---"
  grep -inE '\berror\b|No rule to make' make.log | head -15
  exit 2
fi

echo "###### binaries built"
ls src/bin/RNAfold src/RNAforester/RNAforester 2>/dev/null
echo "###### make check"
make check > check.log 2>&1
echo "check exit=$?"
grep -E '^# (TOTAL|PASS|FAIL|SKIP|ERROR)' check.log
