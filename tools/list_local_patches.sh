#!/bin/bash
# Every change this project makes to an UPSTREAM ViennaRNA file, listed from the
# source itself rather than from a document that can drift.
#
# WHY THIS EXISTS. The CUDA path lives in its own new subdirectory
# (src/ViennaRNA/mfe/cuda/) and that part is easy to describe: upstream can take
# it or leave it. What actually needs defending in a pull request is the handful
# of edits to files upstream already owns. Before this tool those edits were
# discoverable only by diffing against upstream/master and reading 12 000 lines
# of added CUDA to find the 300 that matter.
#
# Every such edit is bracketed in-source:
#
#     /* VRNA-PATCH-BEGIN(<id>, <CLASS>) -- PORT_LOCAL_PATCHES.md
#      * why
#      */
#     ...the change...
#     /* VRNA-PATCH-END(<id>) */
#
# CLASS is one of:
#   DEFECT  an upstream bug we fixed. Submittable on its own, with a reproducer,
#           and it stands whether or not the CUDA work is ever accepted.
#   SEAM    an attachment point the accelerator needs and upstream does not have.
#           Shaped to be useful to upstream on CPU even with no GPU backend.
#   REACH   a capability upstream ALREADY HAS that its public API cannot reach.
#           Narrower than SEAM and usually harder to argue against.
#
# Usage: tools/list_local_patches.sh [repo-root]
#        tools/list_local_patches.sh --check     # pairing only, exit non-zero on error
set -u

ROOT=${1:-$(cd "$(dirname "$0")/.." && pwd)}
[ "$ROOT" = "--check" ] && ROOT=$(cd "$(dirname "$0")/.." && pwd)
CHECK=0
for a in "$@"; do [ "$a" = "--check" ] && CHECK=1; done

cd "$ROOT" || exit 2

# The CUDA subdirectory is ours entirely -- it is not an upstream file and does
# not need marking. Everything else under src/ is upstream's.
FILES=$(grep -rl "VRNA-PATCH-BEGIN(" src/ 2>/dev/null | sort)

fail=0
total_begin=0
total_end=0

printf '%-26s %-8s %-42s %s\n' "ID" "CLASS" "FILE" "LINES"
printf '%s\n' "---------------------------------------------------------------------------------------------"

for f in $FILES; do
  # BEGIN/END must pair, in order, within a file.
  awk -v FNAME="$f" '
    /VRNA-PATCH-BEGIN\(/ {
      match($0, /VRNA-PATCH-BEGIN\(([^,)]+), *([^)]*)\)/, m)
      id = m[1]; cls = m[2]
      if (open_id != "") {
        printf "  *** %s:%d BEGIN(%s) while BEGIN(%s) is still open\n", FNAME, NR, id, open_id
        bad++
      }
      open_id = id; open_cls = cls; open_line = NR
      next
    }
    /VRNA-PATCH-END\(/ {
      match($0, /VRNA-PATCH-END\(([^)]+)\)/, m)
      id = m[1]
      if (open_id == "") {
        printf "  *** %s:%d END(%s) with no matching BEGIN\n", FNAME, NR, id
        bad++
      } else if (id != open_id) {
        printf "  *** %s:%d END(%s) does not match BEGIN(%s)\n", FNAME, NR, id, open_id
        bad++
      } else {
        printf "%-26s %-8s %-42s %d\n", open_id, open_cls, FNAME, NR - open_line - 1
      }
      open_id = ""
      next
    }
    END {
      if (open_id != "") {
        printf "  *** %s: BEGIN(%s) at line %d is never closed\n", FNAME, open_id, open_line
        bad++
      }
      exit (bad > 0)
    }
  ' "$f" || fail=1
  total_begin=$(( total_begin + $(grep -c "VRNA-PATCH-BEGIN(" "$f") ))
  total_end=$((   total_end   + $(grep -c "VRNA-PATCH-END("   "$f") ))
done

echo
echo "$total_begin marked regions across $(echo "$FILES" | grep -c . ) file(s)"

if [ "$total_begin" -ne "$total_end" ]; then
  echo "*** $total_begin BEGIN vs $total_end END -- unbalanced"
  fail=1
fi

# An upstream file changed but NOT marked is the thing this tool exists to
# catch. Compare against upstream/master when it is available.
# v2.7.2, NOT upstream/master. The port is based on the 2.7.2 TAG (which is
# exactly the merge-base), so diffing against master would drag in every
# upstream commit since the release and drown the signal -- the first run of
# this tool reported most of src/Cluster as "NOT MARKED" for exactly that
# reason.
BASE=v2.7.2
if git rev-parse --verify -q "$BASE" > /dev/null 2>&1; then
  echo
  echo "upstream files changed vs $BASE, and whether they are marked:"
  # --diff-filter=M: only files upstream ALREADY HAD that we changed. Added
  # files are ours (mfe/cuda/) or vendored (json/, cthreadpool/, unpacked
  # tarballs) and carry no marker by design -- listing them buried the seven
  # that matter under a hundred that did not.
  # --ignore-cr-at-eol: this repo is checked out on Windows with
  # core.autocrlf=true, so a git run from inside WSL sees CRLF in the working
  # tree against LF in the index and calls EVERY vendored file modified. Without
  # this the tool reported ~100 false positives under WSL and 11 correct ones
  # under Git Bash -- the same command, two answers.
  for f in $(git diff --name-only --diff-filter=M --ignore-cr-at-eol "$BASE" -- src/ 2>/dev/null); do
    case "$f" in
      src/ViennaRNA/mfe/cuda/*) continue ;;   # ours entirely
      */Makefile.am)            continue ;;   # build glue, listed separately
    esac
    [ -f "$f" ] || continue
    if grep -q "VRNA-PATCH-BEGIN(" "$f"; then
      printf '  %-46s marked\n' "$f"
    elif grep -q "VRNA-PATCH-FILE(" "$f"; then
      # A WHOLE-FILE declaration, for files we effectively rewrote rather
      # than patched. src/bin/RNAfold.c is +1333 lines of driver;
      # bracketing every hunk would be noise pretending to be precision.
      printf '  %-46s marked (whole file)\n' "$f"
    else
      printf '  %-46s *** NOT MARKED ***\n' "$f"
      fail=1
    fi
  done
else
  echo
  echo "(no $BASE tag -- skipping the unmarked-file check)"
fi

[ "$CHECK" = 1 ] && exit $fail
exit $fail
