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
  ## right answer is entirely a property of the machine this will run on.
  ##
  ## NO FIXED DEFAULT. This used to be 60,70,75,80,86,89, which stopped building
  ## the day CUDA 13 removed sm_60 and sm_70 ("nvcc fatal: Unsupported gpu
  ## architecture 'compute_60'", Colab, 2026-10-02). Unset now means: ask the
  ## toolkit what it can emit, then build for the local device if one is visible,
  ## else a fat binary over what the toolkit supports (see below).
  AC_ARG_WITH([cuda-arch],
              [AS_HELP_STRING([--with-cuda-arch=LIST],
                              [CUDA compute capabilities, comma separated @<:@default: the local device, else everything the toolkit supports@:>@])],
              [cuda_arch="$withval"],
              [cuda_arch=""])

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

      ## Pick the architectures (the same rule Lukes_Flow_Batching uses since
      ## 04480802). Requested: as given. Otherwise ask the toolkit what it can
      ## emit, and keep the local device's capability if there is one -- the build
      ## host is often not the run host, so detection must not REQUIRE a GPU -- or
      ## else a fat binary over a broad list, intersected with what is supported.
      AS_IF([test "x$cuda_arch" != "x"],[
        cuda_arch_list=`echo "$cuda_arch" | tr ',' ' '`
      ],[
        cuda_arch_supported=`$NVCC_BIN --list-gpu-code 2>/dev/null | sed -e 's/^sm_//' | tr '\n' ' '`
        AS_IF([test "x$cuda_arch_supported" = "x"],[
          ## an nvcc too old for --list-gpu-code: probe, so the list cannot hold a
          ## capability the toolkit will later reject
          echo '__global__ void k(void){}' > conftest_arch.cu
          for a in 50 52 53 60 61 62 70 72 75 80 86 87 89 90 100 120; do
            AS_IF([$NVCC_BIN -gencode arch=compute_${a},code=sm_${a} -c conftest_arch.cu -o conftest_arch.cu.o >/dev/null 2>&1],
                  [cuda_arch_supported="$cuda_arch_supported $a"])
          done
          rm -f conftest_arch.cu conftest_arch.cu.o
        ])
        cuda_local_cc=`nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' .' | sort -u | tr '\n' ' '`
        cuda_arch_list=""
        for a in $cuda_local_cc; do
          for s in $cuda_arch_supported; do
            AS_IF([test "x$a" = "x$s"], [cuda_arch_list="$cuda_arch_list $a"])
          done
        done
        AS_IF([test "x$cuda_arch_list" = "x"],[
          for a in 60 70 75 80 86 89 90 100 120; do
            for s in $cuda_arch_supported; do
              AS_IF([test "x$a" = "x$s"], [cuda_arch_list="$cuda_arch_list $a"])
            done
          done
        ])
      ])

      ## build the -gencode list, plus PTX for the highest so a newer device than
      ## anything compiled for still runs by JIT
      NVCC_ARCH_FLAGS=""
      cuda_arch_highest=""
      for arch in $cuda_arch_list; do
        NVCC_ARCH_FLAGS="$NVCC_ARCH_FLAGS -gencode arch=compute_${arch},code=sm_${arch}"
        cuda_arch_highest="$arch"
      done
      AS_IF([test "x$cuda_arch_highest" != "x"],
            [NVCC_ARCH_FLAGS="$NVCC_ARCH_FLAGS -gencode arch=compute_${cuda_arch_highest},code=compute_${cuda_arch_highest}"])
      cuda_arch_report=`echo $cuda_arch_list | tr " " ","`

      ## The test compile USES the architecture flags. It used to compile without
      ## them, so an unsupported list passed configure and failed in make, minutes
      ## later -- exactly how the CUDA 13 break surfaced.
      AS_IF([test "x$cuda_arch_list" != "x" && $NVCC_BIN -ccbin "$NVCC_HOST_CC" $NVCC_ARCH_FLAGS -c conftest.cu -o conftest.cu.o >/dev/null 2>&1],[
        AC_MSG_RESULT([yes, for sm_$cuda_arch_report +PTX])

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

        ## AND libstdc++: nvcc output is C++, but libtool links with the C compiler.
        ## CUDA 13 emits thread-safe static guards (__cxa_guard_*) in kernel launch
        ## stubs, and without this the final RNAfold link fails (Colab, 2026-10-02;
        ## Lukes_Flow_Batching 2b96539b). Dropped by the linker where unneeded.
        AS_IF([test "x$cuda_libdir" != "x"],
              [CUDA_LIBS="-L$cuda_libdir -lcudart -lstdc++"],
              [CUDA_LIBS="-lcudart -lstdc++"
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
