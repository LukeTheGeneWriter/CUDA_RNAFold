#!/bin/bash
# md_sparse_equiv.sh BUILD_DIR -- S0 bar for PORT_SPARSE_MD.md.
#
# Builds tools/md_sparse_equiv.c against BUILD_DIR's libRNA.a (a configured and built tree: this one,
# or a stock 2.7.2), writes its fixtures, and runs the option matrix twice -- with upstream's full fML
# row as the left operand and with a synthetic row that has only the chain property -- plus both
# negative controls, which must go red. Exit 0 iff every check holds.
set -u
B=${1:?usage: md_sparse_equiv.sh BUILD_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d /tmp/mdsparse.XXXXXX)
LIBS="-lm -lpthread -lstdc++"
nm "$B/src/ViennaRNA/.libs/libRNA.a" 2>/dev/null | grep -q cudaMalloc && \
  LIBS="$LIBS -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcudart -lcuda"
gcc -O2 -fopenmp -I"$B/src" -o "$W/eq" "$HERE/md_sparse_equiv.c" "$B/src/ViennaRNA/.libs/libRNA.a" $LIBS \
  > "$W/cc.log" 2>&1 || { cat "$W/cc.log"; exit 2; }

python3 - "$W" <<'PY'
import random, sys
w = sys.argv[1]; r = random.Random(20261005)
def seq(L, g=0.0):
    s = []
    while len(s) < L:
        s += list("GGGAGGG") if g and r.random() < g else [r.choice("ACGU")]
    return "".join(s[:L])
def fa(name, recs):
    with open("%s/%s.fa" % (w, name), "w") as f:
        for k, (s, c) in enumerate(recs):
            f.write(">%s%d\n%s\n%s" % (name, k, s, (c + "\n") if c else ""))
fa("mix",   [(seq(L), None) for L in (150, 300, 450, 600, 750, 900)])
fa("grich", [(seq(L, 0.08), None) for L in (200, 400, 600)])
# forced-paired '|' positions: up_ml(k) = 0 there, which is what the hc clause is for
pipe = []
for L in (200, 400, 600):
    s = seq(L); c = ["."] * L
    for p in r.sample(range(L), L // 6): c[p] = "|"
    pipe.append((s, "".join(c)))
fa("pipe", pipe)
# an enforced outer pair and forbidden stretches ('x')
enf = []
for L in (300, 500):
    s = "G" + seq(L - 2) + "C"; c = ["."] * L; c[0], c[-1] = "(", ")"
    for p in range(L // 3, L // 3 + 20): c[p] = "x"
    enf.append((s, "".join(c)))
fa("enf", enf)
PY

fail=0
check() {   # $1 = expected outcome (ok|red), rest = args
  local want=$1; shift
  out=$("$W/eq" "$@" 2>&1); rc=$?
  if [ "$want" = ok ] && [ $rc -ne 0 ]; then echo "  FAIL  $* -> $out"; fail=1
  elif [ "$want" = red ] && [ $rc -ne 0 ]; then echo "  FAIL (control did not bite)  $* -> $out"; fail=1
  else echo "  ok    $(echo "$out" | tail -1)   [$*]"; fi
}
for left in --left=full --left=chain; do
  echo "== $left"
  check ok  $left "$W/mix.fa"
  check ok  $left --noLP "$W/mix.fa"
  check ok  $left --circ "$W/mix.fa"
  check ok  $left -d0 "$W/mix.fa"
  check ok  $left --salt=0.2 "$W/mix.fa"
  check ok  $left -T=25 "$W/mix.fa"
  check ok  $left --maxBPspan=150 "$W/mix.fa"
  check ok  $left -g "$W/grich.fa"
  check ok  $left -C "$W/pipe.fa"                  # without --enforce upstream ignores "|" (up_ml stays full)
  check ok  $left -C --enforce "$W/pipe.fa"        # with it, up_ml(k) = 0 at forced-paired k
  check ok  $left -C --enforce "$W/enf.fa"
done
echo "== negative controls"
check red --negctl=scan "$W/mix.fa"
check red --negctl=hc -C --enforce "$W/pipe.fa"
check red --negctl=hc --left=chain -C --enforce "$W/pipe.fa"
rm -rf "$W"
[ $fail -eq 0 ] && echo "md_sparse_equiv: ALL HELD" || echo "md_sparse_equiv: FAILED"
exit $fail
