#!/bin/bash
# Python RNA.fold(list) host-memory bar (PORT_BUDGETS.md decision 3, M2): with a forced
# host reading of 600 MB the windows hold 300 MB of compounds, and on the CPU path
# (RNA_GPU=0) a folded compound keeps its matrices -- so the peak must stay under the
# window plus a base, and every arm must return the same answer. Before 2026-10-13 the
# CPU arm peaked at 1083 MB here (compounds charged unfolded).
# usage: TREE=<built tree> py_window_check.sh      (the CPU arm takes ~4 min)
TREE=${TREE:?set TREE}; F=$(cd "$(dirname "$0")/../.." && pwd)/tests/fixtures/A_longseq_smallbatch.fa
LIMIT_MB=${LIMIT_MB:-420}   # 300 MB window + 120 MB interpreter, module and one fold's scratch
for v in $(env | grep -o '^RNA_[A-Z0-9_]*'); do unset $v; done
command -v /usr/bin/time > /dev/null || { echo "needs /usr/bin/time"; exit 2; }
fail=0; ref=
for arm in "RNA_HOST_AVAIL_MB=600 RNA_GPU=0" "RNA_HOST_AVAIL_MB=600" ""; do
  env $arm PYTHONPATH=$TREE/interfaces/Python /usr/bin/time -v python3 -c "
import RNA, hashlib
seqs=[l.strip() for l in open('$F') if l.strip() and not l.startswith('>')]
print(hashlib.sha1(repr([(s,round(e,2)) for s,e in RNA.fold(seqs)]).encode()).hexdigest()[:12])" > /tmp/pwc.out 2> /tmp/pwc.err
  rss=$(( $(grep 'Maximum resident' /tmp/pwc.err | awk '{print $NF}') / 1024 )); sha=$(cat /tmp/pwc.out)
  v=ok; [ -z "$sha" ] && v=FAIL; [ -n "$ref" ] && [ "$sha" != "$ref" ] && v="FAIL (answer differs)"; ref=${ref:-$sha}
  case "$arm" in *RNA_GPU=0*) [ $rss -le $LIMIT_MB ] || v="FAIL (peak $rss MB > $LIMIT_MB)";; esac
  [ "$v" = ok ] || fail=1
  printf '  %-34s peak %5d MB  sha %s  %s\n' "[$arm]" $rss "$sha" "$v"
done
exit $fail
