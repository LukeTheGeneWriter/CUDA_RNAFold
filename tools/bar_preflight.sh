# Sourced by the verification bars. Not executable on its own.
#
# WHY THIS EXISTS. On 2026-09-06 three bars were run bare and all reported
# green. They had defaulted to ~/port27cuda -- a different build tree, five
# commits and a day behind the one under test. Nothing in their output made that
# visible: they printed the binary path, but a path alone does not tell you the
# binary is older than the change you are trying to verify.
#
# This is the same family as the stale-binary trap that once let edits to
# fill_arrays*.c rebuild nothing while the bar reported PASS. The lesson each
# time is that a passing check must prove it tested the thing you changed.
#
# Call as:  bar_preflight "$BIN" [source-root]
#
# It prints the binary, its build time and its tree, and FAILS if the binary is
# older than the newest source file under the tree -- because at that point the
# bar is measuring code that is not the code you edited.

bar_preflight() {
  local bin=$1
  local root=${2:-$(cd "$(dirname "$bin")/../.." && pwd)}

  if [ ! -x "$bin" ]; then
    echo "PREFLIGHT FAIL: no executable at $bin" >&2
    return 2
  fi

  local built newest newest_f
  built=$(stat -c %Y "$bin" 2>/dev/null || echo 0)

  echo "binary : $bin"
  echo "built  : $(date -d "@$built" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)"
  echo "tree   : $(git -C "$root" log --oneline -1 2>/dev/null | cut -c1-60)"

  # Newest source that could possibly have changed behaviour.
  #
  # *_cmdl.[ch] are EXCLUDED, and that is not a loosening. They are gengetopt
  # OUTPUT -- gitignored, untracked, regenerated on every build, and emitted in
  # whatever order make happens to reach the 25 .ggo files. On a fresh full
  # build the ones for other tools (ct2db, RNAplot, ...) are written AFTER
  # src/bin/RNAfold links, so this test fired on a binary that was in fact
  # perfectly current, with zero real sources newer than it. That is the mirror
  # of the behind-HEAD hole below: a bar that cries wolf gets bypassed, and a
  # bypassed bar is how you arrive back at a false pass. The .ggo files are the
  # actual sources and ARE scanned.
  newest_f=$(find "$root/src" -type f \
                  \( -name '*.c' -o -name '*.cu' -o -name '*.h' -o -name '*.inc' \
                     -o -name '*.ggo' \) \
                  ! -name '*_cmdl.c' ! -name '*_cmdl.h' \
                  -newer "$bin" -print 2>/dev/null | head -1)

  if [ -n "$newest_f" ]; then
    echo "PREFLIGHT FAIL: $bin is OLDER than $newest_f" >&2
    echo "  The bar would test a binary that predates your change and report a" >&2
    echo "  pass. Rebuild first." >&2
    return 2
  fi

  # ------------------------------------------------------------------
  # AND THE CHECK ABOVE IS NOT ENOUGH. Added 2026-09-08, after it let a stale
  # tree through on the very first run of verify_option_matrix.sh.
  #
  # The newer-than test only asks "is the binary newer than the sources IN
  # THIS TREE". A tree checked out at an OLD COMMIT has old sources, so a
  # binary built from them is newer than all of them and passes cleanly. That
  # is exactly what happened: ~/port27cuda sat at 10acb583 while the work under
  # test was 9d3f63cc, three commits later -- so the bar was about to measure a
  # binary with noLP still DECLINED and RNA_MIN_GPU_BATCH not implemented, and
  # would have reported those absences as failures of the matrix.
  #
  # The reference is the tree the BAR ITSELF came from: if the binary's commit
  # is an ancestor of the bar's commit, the binary predates the bar that is
  # about to judge it. Note the file printed "tree : 10acb583 ..." the whole
  # time -- printing a fact is not checking it, which is the same lesson this
  # file was written for.
  local bar_root bar_head bin_head behind
  bar_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)
  bin_head=$(git -C "$root" rev-parse HEAD 2>/dev/null)
  bar_head=$(git -C "$bar_root" rev-parse HEAD 2>/dev/null)

  if [ -z "$bar_head" ] || [ -z "$bin_head" ]; then
    echo "warn   : could not read a commit for the bar or the build tree --" >&2
    echo "         the behind-HEAD check did not run" >&2
  elif [ "$bar_head" != "$bin_head" ]; then
    if git -C "$root" merge-base --is-ancestor "$bin_head" "$bar_head" 2>/dev/null; then
      behind=$(git -C "$root" rev-list --count "$bin_head".."$bar_head" 2>/dev/null)
      echo "PREFLIGHT FAIL: the build tree is $behind commit(s) BEHIND the bar" >&2
      echo "  build tree $root is at $(git -C "$root" log --oneline -1 2>/dev/null | cut -c1-50)" >&2
      echo "  the bar    $bar_root is at $(git -C "$bar_root" log --oneline -1 2>/dev/null | cut -c1-50)" >&2
      echo "  Its sources are old too, so the newer-than test above passes and" >&2
      echo "  proves nothing. Update and rebuild the build tree first." >&2
      return 2
    elif git -C "$bar_root" merge-base --is-ancestor "$bar_head" "$bin_head" 2>/dev/null; then
      echo "note   : build tree is AHEAD of the bar's tree -- testing newer code" >&2
    else
      echo "warn   : build tree and bar are on DIVERGENT commits, or their" >&2
      echo "         objects are not shared, so neither could be compared" >&2
    fi
  fi

  return 0
}
