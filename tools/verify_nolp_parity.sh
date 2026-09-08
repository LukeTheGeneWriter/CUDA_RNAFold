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

  # --noLP + RNA_ROW_VERIFY must refuse too, and for a different reason: the
  # host new_c loop does not implement noLP, so the verifier both reports ~11%
  # of cells as false mismatches and -- because load_my_c uploads the host's
  # new_C over the device's in verify mode -- returns a fold that is neither
  # the noLP answer nor the plain one. A debug flag must not change the answer.
  RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 RNA_ROW_VERIFY=1 \
      $GPU --noPS --noLP -i $F > $W/$fa.rv 2> $W/$fa.rv.err
  rcrv=$?
  say "$fa: --noLP + RNA_ROW_VERIFY refuses at init" \
      "$( [ $rcrv -ne 0 ] && [ ! -s $W/$fa.rv ] && grep -q "cannot be combined" $W/$fa.rv.err \
          && echo ok || echo FAIL)"
  # ...and the verifier must still WORK without noLP, or the guard above has
  # simply broken it. Zero mismatches over a non-zero cell count, both asserted.
  RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 RNA_ROW_VERIFY=1 \
      $GPU --noPS -i $F > /dev/null 2> $W/$fa.rvp.err
  rvline=$(grep -oE 'new_c: [0-9]+ cells checked, [0-9]+ mismatching' $W/$fa.rvp.err | tail -1)
  rvcells=$(echo "$rvline" | awk '{print $2}')
  rvbad=$(echo "$rvline" | awk '{print $5}')
  say "$fa: RNA_ROW_VERIFY still works without noLP (${rvcells:-0} cells, ${rvbad:-?} bad)" \
      "$( [ "${rvcells:-0}" -gt 0 ] && [ "${rvbad:-1}" -eq 0 ] && echo ok || echo FAIL)"
done

# --------------------------------------------------------------------------
# noLP + RNA_SLOT_FLOW: a slot handover must not move the answer.
#
# THE FIXTURE HAS TO BE MIXED-LENGTH, and that is the whole reason this arm
# sits outside the u900/u2000 loop above with a file of its own.
#
# The defect (2026-09-08, found by tools/verify_option_matrix.sh): refill_gpu3()
# runs at EVERY slot handover and takes no slot argument, so the unguarded
# cc/cc1 prefill inside init_gpu3() INF-filled them for the WHOLE batch, wiping
# the mid-recursion cc1 of every record still running in every OTHER slot.
#
# With UNIFORM lengths every slot holds records of equal row count, so every
# slot retires on the same iteration -- the global wipe lands when all the
# neighbours are themselves starting fresh, and is harmless. Measured on the
# broken binary: u900 (one distinct length) disagreed on 0 of 60 records, while
# a 30-length mixed file disagreed on 17 of 30. An arm built on u900/u2000
# therefore CANNOT fail on this bug, and the first version of this check was
# exactly that arm -- green against a binary with the fix deliberately removed.
#
# Both properties it depends on are asserted below rather than assumed, because
# a fixture that quietly stopped being mixed, or a run that quietly stopped
# sharing slots, would put this check straight back into that state.
MIX=$W/mixed.fa
python3 - "$MIX" <<'PY'
import random, sys
random.seed(20260908)
with open(sys.argv[1], "w") as f:
    for i in range(30):                     # 200..1360 nt, every record different
        n = 200 + 40*i
        f.write(">m%d_L%d\n%s\n" % (i, n, "".join(random.choice("ACGU") for _ in range(n))))
PY
nrec=$(grep -c '^>' $MIX)
nlen=$(grep -v '^>' $MIX | awk '{print length}' | sort -n | uniq | wc -l)
say "mixed: the fixture really is mixed-length ($nlen distinct in $nrec)" \
    "$([ "$nlen" -gt 10 ] && echo ok || echo FAIL)"

$CPU --noPS --noLP -i $MIX 2>/dev/null > $W/mix.cpu
RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1 RNA_SLOT_FLOW=2 \
    $GPU --noPS --noLP -i $MIX 2> $W/mix.err > $W/mix.gpu
peak=$(grep -o 'peak/iteration [0-9]* records' $W/mix.err | head -1 | awk '{print $2}')
# Slot flow with as many slots as records never hands a slot over, and would
# pass the comparison below without once exercising the path it exists to test.
# `peak` counts RESIDENT records, so peak < nrec is the proof slots were shared
# -- it is a PEAK and not a total, which is why it is not compared to the
# record count anywhere else in this project.
say "mixed: the slot-flow arm really shared slots (peak ${peak:-0} < $nrec)" \
    "$([ "${peak:-0}" -gt 0 ] && [ "${peak:-0}" -lt "$nrec" ] && echo ok || echo FAIL)"
say "mixed: --noLP + RNA_SLOT_FLOW identical to upstream" \
    "$(cmp -s $W/mix.cpu $W/mix.gpu && echo ok || echo FAIL)"
# Every returned structure was SELF-CONSISTENT when this bug was live -- the
# energies matched their own structures, they were merely suboptimal. So only a
# comparison against upstream catches it; RNAeval below never would.

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
