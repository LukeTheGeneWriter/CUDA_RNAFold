#
# CUDA backend for MFE prediction
#
# Adds --enable-cuda (default: off). Follows the same shape as the other
# optional external dependencies in ac_rna_features.m4 (MPFR, GSL): a feature
# rather than a package, since that is how this tree treats optional libraries.
#
# Sets, when enabled and usable:
#   NVCC_BIN        the nvcc binary
#   NVCC_HOST_CC    the host compiler nvcc should drive
#   NVCC_FLAGS      compilation flags for .cu translation units
#   CUDA_LIBS       what to link the final library against
#   VRNA_WITH_CUDA  config.h macro
#   VRNA_AM_SWITCH_CUDA   automake conditional
#
# Failure to find a usable nvcc turns the feature off with a warning rather than
# failing configure, so an --enable-cuda on a machine without a toolkit still
# produces a working CPU-only build.

AC_DEFUN([RNA_ENABLE_CUDA], [

  RNA_ADD_FEATURE([cuda],
                  [CUDA GPU backend for batched MFE prediction],
                  [no])

  ## let the user point at a toolkit that is not on PATH
  AC_ARG_WITH([cuda-prefix],
              [AS_HELP_STRING([--with-cuda-prefix=DIR],
                              [CUDA toolkit installation prefix])],
              [cuda_prefix="$withval"],
              [cuda_prefix=""])

  ## the compute capabilities to generate code for; overridable because the
  ## right answer is entirely a property of the machine this will run on
  AC_ARG_WITH([cuda-arch],
              [AS_HELP_STRING([--with-cuda-arch=LIST],
                              [CUDA compute capabilities, comma separated @<:@default: 60,70,75,80,86,89@:>@])],
              [cuda_arch="$withval"],
              [cuda_arch="60,70,75,80,86,89"])

  RNA_FEATURE_IF_ENABLED([cuda],[

    AS_IF([test "x$cuda_prefix" != "x"],
          [AC_PATH_PROG([NVCC_BIN], [nvcc], [no], [$cuda_prefix/bin$PATH_SEPARATOR$PATH])],
          [AC_PATH_PROG([NVCC_BIN], [nvcc], [no])])

    AS_IF([test "x$NVCC_BIN" = "xno"],[
      AC_MSG_WARN([
==========================
Could not find nvcc.

The CUDA backend needs the CUDA toolkit. Install it, or point at it with
--with-cuda-prefix=DIR. Continuing with the CUDA backend DISABLED.
==========================
      ])
      enable_cuda=no
    ],[
      ## nvcc drives a host compiler; it must be the one building the rest of
      ## the tree, or the objects will not link together
      AS_IF([test "x$NVCC_HOST_CC" = "x"], [NVCC_HOST_CC="$CC"])

      AC_MSG_CHECKING([whether $NVCC_BIN can compile a CUDA translation unit])

      cat > conftest.cu <<_ACEOF
#include <cuda_runtime.h>
__global__ void vrna_conftest_kernel(int *p) { *p = 1; }
int main(void) { int n = 0; return (cudaGetDeviceCount(&n) == cudaSuccess) ? 0 : 0; }
_ACEOF

      ## build the -gencode list from the requested architectures
      NVCC_ARCH_FLAGS=""
      for arch in `echo "$cuda_arch" | tr ',' ' '`; do
        NVCC_ARCH_FLAGS="$NVCC_ARCH_FLAGS -gencode arch=compute_${arch},code=sm_${arch}"
      done

      AS_IF([$NVCC_BIN -ccbin "$NVCC_HOST_CC" -c conftest.cu -o conftest.cu.o >/dev/null 2>&1],[
        AC_MSG_RESULT([yes])

        ## No -fPIC here: libtool appends the host compiler's PIC flags itself,
        ## and mfe/cuda/nvcc-libtool.sh forwards them with -Xcompiler. Setting
        ## it here as well produced `--compiler-options -Xcompiler -fPIC`, in
        ## which nvcc consumes -Xcompiler as the option's argument and then dies
        ## on the bare -fPIC.
        NVCC_FLAGS="-O3 $NVCC_ARCH_FLAGS"

        ## Locate libcudart rather than assuming the linker's default search
        ## path finds it. It usually does not: nvcc is frequently outside
        ## /usr (a conda prefix, /usr/local/cuda, a module), and the failure is
        ## a pile of undefined references to cudaGetDeviceCount at the FINAL
        ## link of something unrelated, long after configure said yes.
        cuda_libdir=""
        AS_IF([test "x$cuda_prefix" != "x"],
              [cuda_search="$cuda_prefix/lib64 $cuda_prefix/lib"],
              [cuda_bindir=`AS_DIRNAME(["$NVCC_BIN"])`
               cuda_root=`AS_DIRNAME(["$cuda_bindir"])`
               cuda_search="$cuda_root/lib64 $cuda_root/lib"])

        for d in $cuda_search; do
          AS_IF([test -f "$d/libcudart.so" || test -f "$d/libcudart.a"],
                [cuda_libdir="$d"; break])
        done

        AS_IF([test "x$cuda_libdir" != "x"],
              [CUDA_LIBS="-L$cuda_libdir -lcudart"],
              [CUDA_LIBS="-lcudart"
               AC_MSG_WARN([could not locate libcudart; relying on the default library search path])])

        AS_IF([test "x$cuda_prefix" != "x"],
              [NVCC_FLAGS="$NVCC_FLAGS -I$cuda_prefix/include"])

        AC_DEFINE([VRNA_WITH_CUDA], [1],
                  [Build the CUDA GPU backend for MFE prediction])
      ],[
        AC_MSG_RESULT([no])
        AC_MSG_WARN([
==========================
Found $NVCC_BIN but could not compile a trivial CUDA program with it.

Continuing with the CUDA backend DISABLED. See config.log for the failure.
==========================
        ])
        enable_cuda=no
      ])

      rm -f conftest.cu conftest.cu.o
    ])
  ])

  AC_SUBST(NVCC_BIN)
  AC_SUBST(NVCC_HOST_CC)
  AC_SUBST(NVCC_FLAGS)
  AC_SUBST(CUDA_LIBS)

  AM_CONDITIONAL(VRNA_AM_SWITCH_CUDA, test "x$enable_cuda" = "xyes")
])
