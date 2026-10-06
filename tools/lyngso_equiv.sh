#!/bin/bash
# lyngso_equiv.sh BUILD_DIR -- S0 bar for the Lyngsø int_loop (roadmap N3).
#
# Builds tools/lyngso_equiv.c against BUILD_DIR's libRNA.a (a configured and built tree: this one, or a
# stock 2.7.2) and checks, on upstream's own finished c matrix, that the Lyngsø carry form of the
# interior-loop term equals vrna_mfe_internal() for every (i,j): plain, -C --enforceConstraint with 'x',
# '|' and enforced pairs, --noClosingGU, -g, --noLP and salt. Then both negative controls, which must go
# red: the hard-constraint mask on the carry ignored (under -C), and the carry dropped. Exit 0 iff every
# check holds.
set -u
B=${1:?usage: lyngso_equiv.sh BUILD_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d /tmp/lyngso.XXXXXX)
LIBS="-lm -lpthread -lstdc++"
nm "$B/src/ViennaRNA/.libs/libRNA.a" 2>/dev/null | grep -q cudaMalloc && \
  LIBS="$LIBS -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcudart -lcuda"
gcc -O2 -fopenmp -I"$B/src" -o "$W/eq" "$HERE/lyngso_equiv.c" "$B/src/ViennaRNA/.libs/libRNA.a" $LIBS \
  > "$W/cc.log" 2>&1 || { cat "$W/cc.log"; exit 2; }
fail=0
for a in "400 6 21 plain" "400 6 22 C" "400 6 23 noGU" "400 6 24 g" "400 6 25 noLP" "400 4 26 salt" \
         "1200 2 27 C" "1200 2 28 noGU" "1200 2 29 noLP"; do
  out=$("$W/eq" $a); echo "$out"
  echo "$out" | grep -q ", 0 differ$" || fail=$((fail+1))
done
for a in "400 6 22 C 1" "400 6 21 plain 2" "400 6 22 C 2"; do
  out=$("$W/eq" $a); echo "$out   (negative control: must differ)"
  echo "$out" | grep -q ", 0 differ$" && fail=$((fail+1))
done
rm -rf "$W"
[ $fail -eq 0 ] && echo "RESULT: the Lyngsø carry equals vrna_mfe_internal on every case; both negative controls bite" \
                || echo "RESULT: $fail check(s) failed"
exit $fail
