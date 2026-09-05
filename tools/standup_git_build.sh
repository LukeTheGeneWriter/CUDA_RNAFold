#!/bin/bash
# Stand up a build directly from the ViennaRNA git tree (not a release tarball).
#
# Upstream's git tracks Makefile.am but NOT configure, Makefile.in, or the
# gengetopt-generated src/bin/*_cmdl.[ch] -- it ships autogen.sh instead. So a
# git-tree build needs a developer toolchain the tarball build does not:
#
#   autoconf automake libtool  (autoreconf -vi -I config)
#   gengetopt                  (25 .ggo files -> src/bin/*_cmdl.c, untracked)
#   help2man                   (man pages)
#   makeinfo / texinfo         (man/RNAlib.info; the tarball ships it prebuilt)
#   doxygen                    (see below -- needed EVEN WITH --without-doc)
#   check                      (the C unit test suite; without it `make check`
#                               silently drops 43 tests and still says PASS)
#
# The doxygen requirement is an upstream build defect rather than a real
# dependency. Passing --without-doc still leaves WITH_REFERENCE_MANUAL true in
# config.status, and doc/doxygen/Makefile.am puts
#
#     noinst_DATA = $(REFERENCE_MANUAL_FILES_XML)      # = xml/*
#
# OUTSIDE the `if WITH_REFERENCE_MANUAL_BUILD` guard that owns the only rule
# able to produce those files. The release tarball ships doc/doxygen/xml, so
# the demand is already satisfied there and the defect is invisible; from git
# with no doxygen installed, `make` stops with
#   "No rule to make target 'xml/*', needed by 'all-am'".
# Installing doxygen is the least invasive workaround -- it leaves the tree
# pristine -- so that is what this script requires.
#
# Usage: standup_git_build.sh [clone-dir] [--fresh]
#   --fresh  re-run autogen.sh and configure even if they have already run
set -u
export PATH="$HOME/miniforge3/bin:$PATH"

D=${1:-$HOME/port27git}
FRESH=${2:-}

cd "$D" || { echo "no tree at $D" >&2; exit 2; }

echo "###### tree: $D  @  $(git log --oneline -1)"

missing=0
for t in autoconf automake libtoolize gengetopt help2man makeinfo doxygen make gcc; do
  command -v "$t" >/dev/null || { echo "MISSING: $t"; missing=1; }
done
[ $missing -eq 0 ] || { echo "install the missing tools first" >&2; exit 2; }

# The bundled third-party sources ship as tarballs in the tree and configure
# refuses to proceed until they are unpacked. The release tarball ships them
# already unpacked, so this step only exists for a git-tree build.
for t in src/dlib-*.tar.bz2 src/libsvm-*.tar.gz; do
  [ -f "$t" ] || continue
  d=$(basename "$t" | sed -e 's/\.tar\.bz2$//' -e 's/\.tar\.gz$//')
  if [ ! -d "src/$d" ]; then
    echo "###### unpacking $t"
    case "$t" in
      *.bz2) tar -xjf "$t" -C src/ ;;
      *.gz)  tar -xzf "$t" -C src/ ;;
    esac
  fi
done

if [ ! -f configure ] || [ "$FRESH" = "--fresh" ]; then
  echo "###### autogen.sh (autoreconf -vi -I config)"
  ./autogen.sh > "$D/autogen.log" 2>&1
  rc=$?
  echo "autogen exit=$rc"
  [ $rc -eq 0 ] || { tail -25 "$D/autogen.log"; exit 2; }
fi

if [ ! -f config.status ] || [ "$FRESH" = "--fresh" ]; then
  echo "###### configure"
  # The same exclusions Phase 0 used: RNAforester's bundled g2 needs X11 headers
  # this machine does not have, and the bindings/docs are not on the MFE path.
  # PYTHON3 is passed explicitly even though the bindings are off. Without it,
  # --without-python leaves $(PYTHON3) EMPTY while doc/source/man/Makefile.am:38
  # still expands
  #     $(man2rst_verbose)$(PYTHON3) ../../man2rst.py -i $< -o $@
  # so make tries to execute man2rst.py directly. It is tracked mode 100644, so
  # that fails with "Permission denied" (exit 126) on every one of the 25 pages.
  # Invisible from the release tarball, which ships the generated .rst files.
  ./configure \
      --without-python --without-perl --without-swig --without-doc \
      --without-rnaxplorer --without-forester --without-kinfold \
      --without-rnalocmin \
      PYTHON3="$(command -v python3)" \
      > "$D/configure.log" 2>&1
  rc=$?
  echo "configure exit=$rc"
  [ $rc -eq 0 ] || { tail -30 "$D/configure.log"; exit 2; }
fi

# The C unit tests are the ones that silently vanish; confirm they are ON.
echo "###### unit test support"
grep -E 'check (not found|found)|Unit tests' "$D/configure.log" | tail -3
grep -n 'WITH_CHECK_TRUE' config.status | head -2

echo "###### make"
make -j8 > "$D/make.log" 2>&1
rc=$?
echo "make exit=$rc"
[ $rc -eq 0 ] || { grep -iE '\berror\b' "$D/make.log" | head -20; exit 2; }

ls -l --time-style=+%H:%M:%S src/bin/RNAfold
echo "     now: $(date +%H:%M:%S)   (RNAfold must have just been linked)"

echo "###### make check"
make check > "$D/check.log" 2>&1
echo "make check exit=$?"
grep -E '^# (TOTAL|PASS|FAIL|SKIP|XFAIL|XPASS|ERROR)' "$D/check.log"
echo
echo "NOTE: the total must be 127 with the mfe_engine suite, 126 without it."
echo "      A summary that says PASS with a smaller total means the C unit"
echo "      tests were dropped -- see PORT_PHASE0.md 0a."
