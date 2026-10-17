#!/bin/bash
# Backtrack-pipeline self-check (N5a P2, PORT_BACKTRACK_PIPELINE.md): CPU / off / on /
# NEGCTL=1 / NEGCTL=2, one sha per case, plus positive evidence that traces ran beside a
# sweep. usage: TREE=<build> [W=<workdir>] [COV_FIX=<fixtures>] bt_pipeline_check.sh
R=${TREE:-$HOME/pyq}/src/bin/RNAfold; T=${COV_FIX:-$(cd "$(dirname "$0")/../.." && pwd)/tests/fixtures}; W=${W:-$HOME/n5a_p2}; rm -rf $W; mkdir -p $W; cd $W
export RNA_TRI_CHECKSUM=1
bad=0
run() {
  local name=$1 fa=$2 multi=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  env RNA_GPU=0 "${envs[@]}" $R --noPS "$@" < $fa > $name.cpu 2> $name.cpu.err
  env "${envs[@]}" $R --noPS "$@" < $fa > $name.off 2> $name.off.err
  env RNA_BT_PIPELINE=1 RNA_RECORD_TRACE=$W/$name.csv "${envs[@]}" $R --noPS "$@" < $fa > $name.on 2> $name.on.err; local rc=$?
  env RNA_BT_PIPELINE=1 RNA_BT_PIPELINE_NEGCTL=1 "${envs[@]}" $R --noPS "$@" < $fa > $name.n1 2> $name.n1.err; local r1=$?
  env RNA_BT_PIPELINE=1 RNA_BT_PIPELINE_NEGCTL=2 "${envs[@]}" $R --noPS "$@" < $fa > $name.n2 2> $name.n2.err; local r2=$?
  local c=$(sha1sum < $name.cpu|cut -c1-8) o=$(sha1sum < $name.off|cut -c1-8) n=$(sha1sum < $name.on|cut -c1-8)
  local g1=$(sha1sum < $name.n1|cut -c1-8) g2=$(sha1sum < $name.n2|cut -c1-8)
  local rep=$(grep -o "backtrack pipeline:.*" $name.on.err)
  local async=$(echo "$rep" | grep -o "[0-9]* traced beside" | grep -o "^[0-9]*")
  local ov=$(echo "$rep" | grep -o "OVERLAPPED [0-9.]*" | grep -o "[0-9.]*$")
  # Under RNA_DEVICE_BT (BARS_ENV), a chunk the DEVICE traced is never handed off (N5 P2d): when every
  # chunk was, the evidence is ZERO traced beside, and NEGCTL=2 (which acts at the hand-off) has nothing to bite.
  local dact=$(grep -oE 'RNA_DEVICE_BT ACTIVE: [0-9]+ chunk\(s\), [0-9]+ record\(s\) traced on the device; [0-9]+ chunk\(s\) refused' $name.on.err)
  local dchunks=$(echo "$dact" | grep -oE '^RNA_DEVICE_BT ACTIVE: [0-9]+' | grep -oE '[0-9]+$')
  local dref=$(echo "$dact" | grep -oE '[0-9]+ chunk\(s\) refused' | grep -oE '^[0-9]+')
  local devall=0; [ "${dchunks:-0}" -ge 1 ] && [ "${dref:-1}" = 0 ] && devall=1
  local v=PASS
  { [ $rc -eq 0 ] && [ "$c" = "$o" ] && [ "$o" = "$n" ]; } || v=FAIL
  if [ "$multi" = 1 ] && [ $devall = 1 ]; then [ "${async:-x}" = 0 ] || v=FAIL
  elif [ "$multi" = 1 ]; then { [ "${async:-0}" -ge 1 ] && awk "BEGIN{exit !(${ov:-0} > 0)}"; } || v=FAIL; fi
  local b1=bites b2=bites
  { [ "$g1" = "$o" ] && [ $r1 -eq 0 ]; } && { b1="DOES-NOT-BITE"; v=FAIL; }
  if [ "$multi" = 1 ] && [ $devall = 0 ]; then { [ "$g2" = "$o" ] && [ $r2 -eq 0 ]; } && { b2="DOES-NOT-BITE"; v=FAIL; }
  elif [ $devall = 1 ]; then b2="n/a(device-traced)"; else b2="n/a"; fi
  [ $r2 -ne 0 ] && [ $devall = 0 ] && b2="bites(rc$r2)"
  [ $v = FAIL ] && bad=$((bad+1))
  printf "%-16s cpu %s off %s on %s | async %s, OVERLAPPED %s s%s | neg1 %s neg2 %s [%s]\n" "$name" $c $o $n "${async:-0}" "${ov:-0}" \
         "$([ $devall = 1 ] && echo ", device-traced ${dchunks} chunk(s), none handed off")" $b1 $b2 $v
}
run cap_multichunk $T/A_longseq_smallbatch.fa 1 RNA_GPU_CHUNK_CELLS=20000000 --
run default_1chunk $T/A_longseq_smallbatch.fa 0 --
run mixed_cap      $T/C_mixed.fa 1 RNA_GPU_CHUNK_CELLS=2000000 --
run circ           $T/C_mixed.fa 1 RNA_GPU_CHUNK_CELLS=2000000 -- -c
run noLP           $T/C_mixed.fa 1 RNA_GPU_CHUNK_CELLS=2000000 -- --noLP
run gquad          $T/C_mixed.fa 1 RNA_GPU_CHUNK_CELLS=2000000 -- -g
run salt           $T/C_mixed.fa 1 RNA_GPU_CHUNK_CELLS=2000000 -- --salt=0.2
run jobs4          $T/C_mixed.fa 1 RNA_GPU_CHUNK_CELLS=2000000 -- -j4
run hetero_j4      $T/A_longseq_smallbatch.fa 1 RNA_GPU_CHUNK_CELLS=20000000 RNA_CPU_SLICE=auto -- -j4
echo "FAILURES: $bad"
echo "--- refusal check (--unordered must stay sequential and still be right)"
RNA_BT_PIPELINE=1 RNA_GPU_CHUNK_CELLS=2000000 $R --noPS --unordered < $T/C_mixed.fa 2> un.err | sort | sha1sum | cut -c1-8; sort mixed_cap.cpu | sha1sum | cut -c1-8
grep -h "stay sequential\|backtrack pipeline:" un.err
echo "--- cap_multichunk chunk spans (bt_t1 after exit = traced beside the next sweep)"
cat cap_multichunk.csv.chunks.csv
grep -h "backtrack pipeline:\|build pipeline:" cap_multichunk.on.err hetero_j4.on.err
grep -h "CPU slice\|cpu slice" hetero_j4.on.err | head -3
