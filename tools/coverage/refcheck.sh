#!/bin/bash
# Cleanup bar: default-path output must not move (PORT_CLEANUP_PLAN.md §6).
# usage: refcheck.sh freeze|check <fixture> [budget_mb]
#   freeze  write <ref>/<tag>.out from this build (run once, at the pre-cleanup tag)
#   check   fold again, plain and -j4, and byte-compare both with the frozen output
#   REF_DIR  reference directory (default tests/fixtures/cleanup_ref)
#   RNAFOLD  binary (default ~/pyq/src/bin/RNAfold)
#   COV_FIX  fixture directory holding <fixture>.fa (default tests/fixtures)
# budget_mb sets RNA_GPU_VRAM_BUDGET_MB to force a multi-chunk run on a small card;
# every other RNA_* variable is unset. One fixture per job (laptop memory reaper).
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniforge | paste -sd:)
for v in $(env | grep -o "^RNA_[A-Z0-9_]*"); do unset $v; done
# BARS_ENV: an experimental arm's variables, set after the scrub (e.g. "RNA_DEVICE_BT=1");
# BARS_WANT: the banner that arm must print, or the check fails (positive evidence it ran).
[ -n "${BARS_ENV:-}" ] && export $BARS_ENV
mode=$1; f=$2; [ -n "$3" ] && export RNA_GPU_VRAM_BUDGET_MB=$3
R=${RNAFOLD:-~/pyq/src/bin/RNAfold}; O=${REF_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/tests/fixtures/cleanup_ref}; T=${COV_FIX:-$(cd "$(dirname "$0")/../.." && pwd)/tests/fixtures}
[ -f "$T/$f.fa" ] || { echo "no $T/$f.fa"; exit 2; }
mkdir -p "$O"; tag=$f${3:+_b$3}; W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
RNA_GPU_VERBOSE=1 $R --noPS < "$T/$f.fa" > "$W/a.out" 2> "$W/a.err"; a=$?
$R --noPS -j4 < "$T/$f.fa" > "$W/b.out" 2> /dev/null; b=$?
sw=$(grep -c 'sweep shape' "$W/a.err")
sha=$(sha256sum < "$W/a.out" | cut -c1-16)
case "$mode" in
  freeze) cp "$W/a.out" "$O/$tag.out"; cp "$W/a.err" "$O/$tag.err"
          echo "$tag FROZEN rc $a/$b sweeps $sw j4_same $(cmp -s "$W/a.out" "$W/b.out" && echo yes || echo NO) sha $sha";;
  check)  [ -f "$O/$tag.out" ] || { echo "$tag: no frozen reference"; exit 2; }
          s1=$(cmp -s "$W/a.out" "$O/$tag.out" && echo yes || echo NO)
          s4=$(cmp -s "$W/b.out" "$O/$tag.out" && echo yes || echo NO)
          sw0=$(grep -c 'sweep shape' "$O/$tag.err")
          ok=PASS; [ $a = 0 ] && [ $b = 0 ] && [ $s1 = yes ] && [ $s4 = yes ] && [ "$sw" = "$sw0" ] || ok=FAIL
          want=""; if [ -n "${BARS_WANT:-}" ]; then
            if grep -q "$BARS_WANT" "$W/a.err"; then want=" | $(grep -o "$BARS_WANT.*" "$W/a.err" | head -1 | cut -c1-90)"
            elif [ "$sw" = 0 ]; then want=" | no sweep, so no '$BARS_WANT' expected"
            else want=" | MISSING '$BARS_WANT'"; ok=FAIL; fi
          fi
          echo "$tag $ok rc $a/$b same $s1 j4_same $s4 sweeps $sw (ref $sw0) sha $sha$want"
          [ $ok = PASS ];;
  *) echo "usage: $0 freeze|check <fixture> [budget_mb]"; exit 2;;
esac
