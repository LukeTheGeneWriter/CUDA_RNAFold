#!/bin/bash
# proto_device_bt.sh BUILD_DIR [timing] -- N5 P1 bar (PORT_DEVICE_BACKTRACK.md §4): the device trace prototype.
#
# Builds tools/proto_device_bt.cu against BUILD_DIR (a configured, CUDA-built tree: its libRNA.a, its
# device headers, its NVCC_FLAGS) and checks that the one-warp-per-record device trace reproduces
# upstream's own trace on upstream's own matrices: every f5 cell, every MFE and every structure,
# byte-identical including ties, on random, A/U-only (ties everywhere) and G-rich fixtures. Then the
# negative control: every search's order reversed must change structures on the A/U fixture while
# keeping each at the MFE (so the ties are real and the order is what decides them).
# With "timing", also the per-record device and host trace cost against length (random fixtures).
# Exit 0 iff every check holds.
set -u
B=${1:?usage: proto_device_bt.sh BUILD_DIR [timing]}
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d /tmp/protobt.XXXXXX)
NVF=$(sed -n 's/^NVCC_FLAGS = //p' "$B/src/ViennaRNA/Makefile")
[ -n "$NVF" ] || { echo "no NVCC_FLAGS in $B -- not a CUDA build"; exit 2; }
# nvcc compiles only: libRNA.a carries the tree's own gcc's LTO bytecode, which nvcc's host linker
# cannot read, so the link is the tree's CC, as for the tree's own binaries.
CC=$(sed -n 's/^CC = //p' "$B/src/ViennaRNA/Makefile")
nvcc $NVF -Xcompiler -fopenmp -DNDEBUG -I"$B/src" -I"$B/src/ViennaRNA/mfe/cuda" \
  -c -o "$W/pbt.o" "$HERE/proto_device_bt.cu" > "$W/cc.log" 2>&1 &&
${CC:-gcc} -fopenmp -o "$W/pbt" "$W/pbt.o" "$B/src/ViennaRNA/.libs/libRNA.a" \
  -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcudart -lstdc++ -lm -lpthread >> "$W/cc.log" 2>&1 \
  || { cat "$W/cc.log"; exit 2; }
fail=0
for a in "random 40 400 1" "au 40 400 2" "grich 40 400 3" "random 12 1200 4" "au 12 1200 5" "grich 12 1200 6"; do
  out=$("$W/pbt" $a); rc=$?; echo "$out"
  [ $rc -eq 0 ] || fail=$((fail+1))
done
out=$("$W/pbt" au 40 400 7 1); echo "$out"
echo "$out" | grep -q "NEGCTL.* BITES" || fail=$((fail+1))
echo "$out" | grep -q "0/40 structures differ, 0 failed" || fail=$((fail+1))
if [ "${2:-}" = timing ]; then
  echo "== timing (random; device = one batch, all records at once; host = upstream's f5 + backtrack, serial)"
  for a in "200 64" "400 64" "800 32" "1200 16" "2400 8" "5601 4"; do
    set -- $a; "$W/pbt" random $2 $1 11 | sed 's/^/  /'
  done
fi
rm -rf "$W"
[ $fail -eq 0 ] && echo "RESULT: the device trace equals upstream's on every fixture; the reversed-order control bites" \
                || echo "RESULT: $fail check(s) failed"
exit $fail
