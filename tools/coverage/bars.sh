#!/bin/bash
# The standing bars (PORT_CLEANUP_PLAN.md section 6, REVIEW_GUIDE.md), on one build tree,
# DEFAULT environment. One GPU job at a time; about an hour on an RTX 3050 laptop.
# usage: TREE=<build tree> bars.sh <logdir>
#   refcheck (9 fixtures), option parity, option matrix, gpu_cli 4/8/16/32 MB, the Python
#   binding suite, the backtrack-pipeline self-check, the output-file bar, the knob arms, the noLP bar.
# The CPU-only build is its own bar: tests/zeroconf_configure.sh, on a COPY of the tree
# (it reconfigures the tree it is given).
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniforge | paste -sd:)
for v in $(env | grep -o "^RNA_[A-Z0-9_]*"); do unset $v; done
# BARS_ENV="RNA_X=1 ..." runs every bar with an experimental arm on (set after the scrub, here and
# in the sub-bars that scrub their own); BARS_WANT is the banner refcheck must then see.
[ -n "${BARS_ENV:-}" ] && export $BARS_ENV
TREE=${TREE:?set TREE to a configured, built tree}; O=${1:?log directory}; mkdir -p "$O"
D=$(cd "$(dirname "$0")" && pwd); TL=$D/..; ROOT=$D/../..
NOCUDA=${NOCUDA:-$HOME/port27git}   # a no-CUDA build: the noLP bar's upstream oracle
for a in C_mixed C_control asc desc extreme A_longseq_smallbatch constrained "F_lite 64" "C400_lite 256"; do
  RNAFOLD=$TREE/src/bin/RNAfold bash $D/refcheck.sh check $a
done
echo "== option parity"; bash $TL/verify_option_parity.sh $TREE > $O/parity.log 2>&1; echo "rc $?"; tail -2 $O/parity.log
echo "== option matrix"; bash $TL/verify_option_matrix.sh $TREE > $O/matrix.log 2>&1; echo "rc $?"; tail -2 $O/matrix.log
echo "== gpu_cli 4/8/16/32"; bash $TL/verify_gpu_cli.sh $TREE 8 4 8 16 32 > $O/gpucli.log 2>&1; echo "rc $?"; tail -2 $O/gpucli.log
echo "== binding suite"; ( cd $ROOT/tests/python && PYTHONPATH=$TREE/interfaces/Python python3 -m pytest -q -p no:cacheprovider test_RNA-cuda.py > $O/binding.log 2>&1 ); echo "rc $?"; tail -1 $O/binding.log
echo "== backtrack pipeline"; TREE=$TREE W=$O/bt bash $D/bt_pipeline_check.sh > $O/bt.log 2>&1; echo "rc $?"; grep "FAILURES" $O/bt.log
echo "== output files"; TREE=$TREE W=$O/of bash $D/outfile_check.sh > $O/outfile.log 2>&1; echo "rc $?"; grep "FAILURES" $O/outfile.log
echo "== knob arms"; for a in C_mixed extreme "F_lite 64"; do RNAFOLD=$TREE/src/bin/RNAfold bash $D/knobarms.sh $a; done
echo "== noLP"; bash $TL/verify_nolp_parity.sh $TREE $NOCUDA > $O/nolp.log 2>&1; echo "rc $?"; tail -1 $O/nolp.log
