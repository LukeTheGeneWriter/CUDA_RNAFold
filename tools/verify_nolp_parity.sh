#!/bin/bash
# The --noLP bar. Deliberately NOT an energy comparison: comparing energies is
# exactly what hid this defect for a session (45 of 60 records agreed on energy
# while the structures were 87-300 kcal inconsistent with their own reported
# value). See PORT_NOLP_SPEC.md.
# Trees: $1 = CUDA build, $2 = no-CUDA build (the upstream oracle).
CUDA=${1:-$HOME/port27cuda}
NOCUDA=${2:-$HOME/port27git}
for d in "$CUDA" "$NOCUDA"; do
  [ -x "$d/src/bin/RNAfold" ] || { echo "no RNAfold under $d" >&2; exit 2; }
done
GPU=$CUDA/src/bin/RNAfold
CPU=$NOCUDA/src/bin/RNAfold
W=$HOME/dfprobe/nolpbar; mkdir -p $W
fail=0
say() { printf '  %-46s %s\n' "$1" "$2"; [ "$2" != ok ] && fail=1; return 0; }

for fa in u900 u2000; do
  F=$HOME/dfprobe/$fa.fa
  $CPU --noPS --noLP -i $F 2>/dev/null > $W/$fa.cpu
  RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 $GPU --noPS --noLP -i $F 2>$W/$fa.err > $W/$fa.gpu
  $CPU --noPS        -i $F 2>/dev/null > $W/$fa.plain

  swept=$(grep -c 'sweep shape:' $W/$fa.err)
  say "$fa: the GPU arm actually swept ($swept chunks)" \
      "$([ "$swept" -ge 1 ] && echo ok || echo FAIL)"
  say "$fa: byte-identical to upstream --noLP" \
      "$(cmp -s $W/$fa.cpu $W/$fa.gpu && echo ok || echo FAIL)"
  # A --noLP run that equals the plain fold has not applied the option. It is
  # legitimate per-record (a fold with no lonely pairs is its own noLP answer),
  # so this is asserted over the FILE, where upstream itself differs.
  say "$fa: differs from the plain fold (option is live)" \
      "$(cmp -s $W/$fa.cpu $W/$fa.plain && echo FAIL || echo ok)"

  # Multi-chunk must not move the answer: cc/cc1 are per-record row
  # buffers. 96MB, not 352MB -- at 352MB u900 does not split at all, so
  # the earlier version of this check passed on a path it never took.
  RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 RNA_GPU_VRAM_BUDGET_MB=96 \
      $GPU --noPS --noLP -i $F 2> $W/$fa.chunk.err > $W/$fa.chunk
  nch=$(grep -c 'sweep shape:' $W/$fa.chunk.err)
  say "$fa: the multi-chunk arm really split ($nch chunks)" \
      "$([ "$nch" -gt 1 ] && echo ok || echo FAIL)"
  say "$fa: multi-chunk identical to upstream" \
      "$(cmp -s $W/$fa.cpu $W/$fa.chunk && echo ok || echo FAIL)"

  # noLP + int16 must REFUSE, not silently disagree: noLP puts near-INF
  # finite values into fML that the 16-bit per-block offsets cannot
  # represent. The existing range guard is what found it.
  RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 RNA_FML_INT16=1 \
      $GPU --noPS --noLP -i $F > $W/$fa.i16 2> $W/$fa.i16.err
  rc16=$?
  say "$fa: --noLP + RNA_FML_INT16 refuses at init" \
      "$( [ $rc16 -ne 0 ] && grep -q "cannot be combined" $W/$fa.i16.err && echo ok || echo FAIL)"
done

# Self-consistency and lonely pairs, on the file the diagnosis used.
W=$W python3 - <<'PY'
import os, re, subprocess
W = os.environ["W"]; EVAL = os.path.expanduser("~/port27git/src/bin/RNAeval")
L = open(W+"/u900.gpu").read().splitlines()
seqs, strs, ens = [], [], []
for k in range(0, len(L)-2, 3):
    m = re.match(r"^([.()]+)\s+\(\s*(-?\d+\.\d+)\)\s*$", L[k+2])
    seqs.append(L[k+1].strip()); strs.append(m.group(1)); ens.append(float(m.group(2)))
inp = "".join("%s\n%s\n" % (a,b) for a,b in zip(seqs, strs))
p = subprocess.run([EVAL], input=inp, capture_output=True, text=True)
ev = [float(x) for x in re.findall(r"\(\s*(-?\d+\.\d+)\)", p.stdout)]
bad = sum(1 for a,b in zip(ens, ev) if abs(a-b) > 0.01)
print("  %-46s %s" % ("u900: every structure re-evaluates to its own energy",
                      "ok" if bad == 0 else "FAIL (%d of %d)" % (bad, len(ev))))
def lonely(s):
    pt=[-1]*len(s); st=[]; c=0
    for i,ch in enumerate(s):
        if ch=="(": st.append(i)
        elif ch==")": j=st.pop(); pt[i]=j; pt[j]=i
    for i,ch in enumerate(s):
        if pt[i]<0 or pt[i]<i: continue
        j=pt[i]
        up = (i>0 and j<len(s)-1 and pt[i-1]==j+1)
        dn = (pt[i+1]==j-1) if i+1<len(s) and j-1>=0 else False
        if not up and not dn: c+=1
    return c
tot = sum(lonely(s) for s in strs)
print("  %-46s %s" % ("u900: zero lonely pairs in every structure",
                      "ok" if tot == 0 else "FAIL (%d)" % tot))
PY
echo
[ $fail -eq 0 ] && echo "noLP bar: ALL GREEN" || echo "noLP bar: FAILURES ABOVE"
exit $fail
