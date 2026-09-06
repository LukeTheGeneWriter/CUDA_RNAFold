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
  newest_f=$(find "$root/src" -type f \
                  \( -name '*.c' -o -name '*.cu' -o -name '*.h' -o -name '*.inc' \) \
                  -newer "$bin" -print 2>/dev/null | head -1)

  if [ -n "$newest_f" ]; then
    echo "PREFLIGHT FAIL: $bin is OLDER than $newest_f" >&2
    echo "  The bar would test a binary that predates your change and report a" >&2
    echo "  pass. Rebuild first." >&2
    return 2
  fi

  return 0
}
