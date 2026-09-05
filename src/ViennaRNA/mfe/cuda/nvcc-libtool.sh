#!/bin/sh
# Let libtool drive nvcc.
#
# libtool --mode=compile appends the host compiler's PIC flags to whatever
# compiler it is handed:
#
#   libtool: compile:  nvcc -ccbin gcc ... -c device.cu -fPIC -DPIC -o ...
#   nvcc fatal   : Unknown option '-fPIC'
#
# nvcc needs host-compiler flags forwarded explicitly with -Xcompiler. This
# wrapper does that translation and passes everything else through untouched.
#
# Usage: nvcc-libtool.sh <nvcc> [args...]
#
# Invoked as `$(SHELL) nvcc-libtool.sh`, never executed directly, so it does not
# depend on its executable bit surviving a checkout, a tarball, or a copy. That
# is not hypothetical caution: upstream's own doc/man2rst.py is tracked mode
# 100644 and invoked as a program, which breaks every git build (reported in
# PORT_UPSTREAM_PROPOSAL.md, Part 1b).

set -e

if [ $# -lt 1 ]; then
  echo "usage: $0 <nvcc> [args...]" >&2
  exit 2
fi

nvcc=$1
shift

# Rotate the argument list exactly once, translating host-only flags as we go.
# Done with set -- rather than string concatenation so arguments containing
# spaces (build paths, typically) survive intact.
#
# `prev` matters: a flag that is already the ARGUMENT of an nvcc option must be
# left alone. Translating the -fPIC in `--compiler-options -fPIC` produces
# `--compiler-options -Xcompiler -fPIC`, nvcc takes -Xcompiler as the option's
# argument, and then fails on a bare -fPIC -- the very error this wrapper
# exists to prevent, reintroduced by the wrapper itself.
prev=""
n=$#
while [ "$n" -gt 0 ]; do
  arg=$1
  shift

  case "$prev" in
    --compiler-options|-Xcompiler|--compiler-bindir|-ccbin|-Xlinker|--linker-options|-gencode|-arch|-code|-o|-I|-D|-L|-l)
      ## already spoken for by the preceding option
      set -- "$@" "$arg"
      prev=$arg
      n=$((n - 1))
      continue
      ;;
  esac

  case "$arg" in
    -fPIC|-fpic|-fPIE|-fpie|-shared|-fno-*|-fstack-protector*|-ffunction-sections|-fdata-sections|-pthread|-flto|-flto=*|-ffat-lto-objects)
      set -- "$@" -Xcompiler "$arg"
      ;;
    *)
      set -- "$@" "$arg"
      ;;
  esac

  prev=$arg
  n=$((n - 1))
done

exec "$nvcc" "$@"
