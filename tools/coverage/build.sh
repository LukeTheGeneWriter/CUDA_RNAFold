#!/bin/bash
# Coverage build of an LFB tree (PORT_CLEANUP_PLAN.md §1).
# usage: build.sh [tree]   (default ~/covq; the tree must already be configured)
#
# ONE compiler throughout: gcc-13/g++-13 for C, C++ and link, because nvcc's host
# compiler is gcc 13. Mixing in a newer gcc makes libgcov silently refuse to write
# the .cu objects' counts, and every .cu function then reads 0 %. The last line
# printed must show exactly one "GCC:" version string.
#
# NVCC_ARCH overrides the target (default sm_86, the laptop's RTX 3050).
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniforge | paste -sd:)
T=${1:-~/covq}
A=${NVCC_ARCH:-86}
cd "$T" || exit 1
make -C src clean > /dev/null 2>&1 || true
find "$T/src" -name "*.gcda" -delete
TC='CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13'
NV="-O3 -gencode arch=compute_$A,code=sm_$A -gencode arch=compute_$A,code=compute_$A -I/usr/include --compiler-options=--coverage"
t0=$(date +%s)
if make -j2 -C src $TC CFLAGS="-g -O1 --coverage" CXXFLAGS="-g -O1" LDFLAGS="--coverage" NVCC_FLAGS="$NV" > "$T/cov_make.log" 2>&1; then
  echo "COV-BUILD-OK in $(( $(date +%s) - t0 )) s"
else
  grep -n -B3 -A3 "fatal\|error:" "$T/cov_make.log" | head -30; exit 1
fi
echo "gcno files: $(find "$T/src" -name "*.gcno" | wc -l)"
strings "$T/src/bin/RNAfold" | grep "GCC: (" | sort | uniq -c
