#
# CUDA backend for MFE prediction -- found and configured automatically
#
# THE REQUIREMENT THIS MACRO EXISTS TO MEET (PORT_ZEROCONF_SCOPE.md section 11):
# upstream ViennaRNA documents `./configure && make && sudo make install` and
# nothing else. No accelerator flag appears anywhere in its instructions. So a bare
# ./configure MUST produce an accelerated binary wherever that is possible, and a
# working CPU-only binary everywhere else. Anything we require beyond upstream's
# documented path is a defect for the thousands of people already using ViennaRNA.
#
# Hence THREE states, not two:
#
#   (nothing given)   AUTO. Look for a toolkit. Found and usable -> build the
#                     backend. Not found -> say so once, in the summary, and build
#                     a CPU-only library exactly as stock ViennaRNA would.
#   --enable-cuda     DEMAND. The user asked for the accelerator, so a missing or
#                     broken toolkit is an ERROR rather than a silent downgrade.
#                     This is the whole point of the flag now that auto is the
#                     default: it turns a warning into a failure.
#   --disable-cuda    OFF. Never look, never build it.
#
# The distinction matters because of how this project has been burned before: a
# GPU-less build of this tree emits ZERO bytes of stderr and byte-identical
# stdout, so a silent downgrade is indistinguishable from success. Under AUTO the
# downgrade is legitimate and announced; under an explicit --enable-cuda it is a
# failure. See feedback_silent_fallback_needs_positive_evidence.
#
# Sets, when enabled and usable:
#   NVCC_BIN        the nvcc binary
#   NVCC_HOST_CC    the host compiler nvcc should drive
#   NVCC_FLAGS      compilation flags for .cu translation units
#   CUDA_LIBS       what to link the final library against
#   VRNA_WITH_CUDA  config.h macro
#   VRNA_AM_SWITCH_CUDA   automake conditional
#
# Reported through result_cuda / cuda_report_* in the configure summary.

AC_DEFUN([RNA_ENABLE_CUDA], [

  ## ------------------------------------------------------------------ intent
  ## Autoconf parses every --enable-* into enable_<name> in its command-line
  ## preamble, which runs BEFORE any AC_ARG_ENABLE shell code. So reading
  ## $enable_cuda here, before RNA_ADD_FEATURE defaults it, is what distinguishes
  ## "the user said --enable-cuda" from "the user said nothing". There is no other
  ## way to tell: once RNA_ADD_FEATURE has applied its default both look like
  ## enable_cuda=yes.
  cuda_user_choice="$enable_cuda"

  ## Default YES, so the documented opt-out is --disable-cuda and the bare
  ## ./configure path reaches the detection below.
  RNA_ADD_FEATURE([cuda],
                  [CUDA GPU backend for batched MFE prediction (default: enabled when a CUDA toolkit is found)],
                  [yes])

  ## let the user point at a toolkit that is nowhere we would look
  AC_ARG_WITH([cuda-prefix],
              [AS_HELP_STRING([--with-cuda-prefix=DIR],
                              [CUDA toolkit installation prefix @<:@default: autodetected@:>@])],
              [cuda_prefix="$withval"],
              [cuda_prefix=""])

  ## The compute capabilities to generate code for. Empty means "work it out",
  ## which is the documented default; a value here is taken verbatim and trusted.
  AC_ARG_WITH([cuda-arch],
              [AS_HELP_STRING([--with-cuda-arch=LIST],
                              [CUDA compute capabilities, comma separated, or `native' @<:@default: autodetected@:>@])],
              [cuda_arch="$withval"],
              [cuda_arch=""])

  cuda_report_nvcc=""
  cuda_report_arch=""
  cuda_report_host=""
  cuda_report_why=""

  RNA_FEATURE_IF_ENABLED([cuda],[

    ## ------------------------------------------------------- 1. find the toolkit
    ##
    ## PATH alone is not enough and never was. nvcc is routinely installed
    ## somewhere only a module file or a conda activation script knows about, and
    ## on those hosts a bare ./configure would find nothing while `nvcc --version`
    ## works fine for the user in an interactive shell. So look in the places a
    ## CUDA toolkit actually lands, newest versioned install first.
    ##
    ## Order of precedence, deliberately: an explicit --with-cuda-prefix wins over
    ## everything, then PATH (the user's expressed intent for this shell), then the
    ## conventional locations.
    cuda_search_path=""
    AS_IF([test "x$cuda_prefix" != "x"],
          [cuda_search_path="$cuda_prefix/bin$PATH_SEPARATOR"])

    cuda_search_path="$cuda_search_path$PATH"

    for cuda_hint in "$CUDA_HOME" "$CUDA_PATH" "$CUDAToolkit_ROOT" "$CONDA_PREFIX" \
                     /usr/local/cuda /opt/cuda /usr/lib/cuda; do
      AS_IF([test "x$cuda_hint" != "x" && test -x "$cuda_hint/bin/nvcc"],
            [cuda_search_path="$cuda_search_path$PATH_SEPARATOR$cuda_hint/bin"])
    done

    ## versioned side-by-side installs: prefer the newest. `sort -V` is not
    ## universal, so fall back to plain reverse sort, which is right for the
    ## common single-digit-major case and harmless otherwise.
    cuda_versioned=`ls -d /usr/local/cuda-* /opt/cuda-* 2>/dev/null | sort -Vr 2>/dev/null || ls -d /usr/local/cuda-* /opt/cuda-* 2>/dev/null | sort -r`
    for cuda_hint in $cuda_versioned; do
      AS_IF([test -x "$cuda_hint/bin/nvcc"],
            [cuda_search_path="$cuda_search_path$PATH_SEPARATOR$cuda_hint/bin"])
    done

    AC_PATH_PROG([NVCC_BIN], [nvcc], [no], [$cuda_search_path])

    AS_IF([test "x$NVCC_BIN" = "xno"],[
      cuda_report_why="no nvcc found on PATH, in CUDA_HOME/CUDA_PATH, or under /usr/local/cuda*"
      enable_cuda=no
    ],[
      cuda_bindir=`AS_DIRNAME(["$NVCC_BIN"])`
      cuda_root=`AS_DIRNAME(["$cuda_bindir"])`
      AS_IF([test "x$cuda_prefix" != "x"], [cuda_root="$cuda_prefix"])

      cuda_version=`$NVCC_BIN --version 2>/dev/null | sed -n 's/.*release \([[0-9.]]*\).*/\1/p'`
      cuda_report_nvcc="$NVCC_BIN${cuda_version:+ (release $cuda_version)}"

      ## --------------------------------------------- 2. pick the architectures
      ##
      ## The old hard-coded list 60,70,75,80,86,89 is a latent build failure: CUDA
      ## 13 DROPPED sm_60, so that list stops compiling on a new toolkit, and it
      ## also misses sm_90 and later entirely, so a current datacentre GPU gets
      ## nothing but JIT. Ask the toolkit what it supports instead of asserting it.
      ##
      ## Then narrow. Two cases, and the distinction is the one section 2.2 of the
      ## scope document insists on -- ARCHITECTURE DETECTION MUST NOT REQUIRE A
      ## LOCAL GPU, because the build host is frequently not the run host:
      ##
      ##   a device is visible  -> build for exactly its capability. Fast build,
      ##                           optimal code, and PTX is still emitted so
      ##                           another device can JIT.
      ##   no device visible    -> build a fat binary over everything this toolkit
      ##                           supports that we care about, plus PTX.
      cuda_arch_native=no
      AS_IF([test "x$cuda_arch" = "xnative"],
            [cuda_arch_native=yes; cuda_arch=""])

      AS_IF([test "x$cuda_arch" != "x"],[
        cuda_arch_list=`echo "$cuda_arch" | tr ',' ' '`
        cuda_arch_source="requested"
      ],[
        ## what can this nvcc actually emit?
        cuda_arch_supported=`$NVCC_BIN --list-gpu-code 2>/dev/null | sed -e 's/^sm_//' | tr '\n' ' '`

        AS_IF([test "x$cuda_arch_supported" = "x"],[
          ## An nvcc too old for --list-gpu-code. Probe instead of guessing: one
          ## trivial compile per candidate is slow but this path is rare, and it
          ## cannot produce a list the toolkit will later reject.
          echo '__global__ void k(void){}' > conftest_arch.cu
          cuda_arch_supported=""
          for a in 50 52 53 60 61 62 70 72 75 80 86 87 89 90 100 120; do
            AS_IF([$NVCC_BIN -gencode arch=compute_${a},code=sm_${a} -c conftest_arch.cu -o conftest_arch.cu.o >/dev/null 2>&1],
                  [cuda_arch_supported="$cuda_arch_supported $a"])
          done
          rm -f conftest_arch.cu conftest_arch.cu.o
        ])

        ## which capabilities are actually attached to this machine, if any
        cuda_local_cc=""
        for cuda_smi in "$cuda_root/bin/nvidia-smi" nvidia-smi; do
          AS_IF([test "x$cuda_local_cc" = "x"],[
            cuda_local_cc=`$cuda_smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' .' | sort -u | tr '\n' ' '`
          ])
        done

        cuda_arch_list=""
        AS_IF([test "x$cuda_local_cc" != "x"],[
          ## keep only capabilities this toolkit can target -- a brand new GPU on
          ## an old toolkit must not turn into a broken -gencode
          for a in $cuda_local_cc; do
            for s in $cuda_arch_supported; do
              AS_IF([test "x$a" = "x$s"], [cuda_arch_list="$cuda_arch_list $a"])
            done
          done
          cuda_arch_source="detected from the local device(s)"
        ])

        AS_IF([test "x$cuda_arch_list" = "x"],[
          ## No device, or none this toolkit can target. Build broadly: this is the
          ## cluster case, where the login node has no GPU and the compute nodes do.
          for a in 60 70 75 80 86 89 90 100 120; do
            for s in $cuda_arch_supported; do
              AS_IF([test "x$a" = "x$s"], [cuda_arch_list="$cuda_arch_list $a"])
            done
          done
          AS_IF([test "x$cuda_local_cc" = "x"],
                [cuda_arch_source="no local device, so a portable fat binary"],
                [cuda_arch_source="local device(s) $cuda_local_cc unsupported by this toolkit, so a portable fat binary"])
        ])
      ])

      AS_IF([test "x$cuda_arch_native" = "xyes"],[
        NVCC_ARCH_FLAGS="-arch=native"
        cuda_report_arch="native"
      ],[
        AS_IF([test "x$cuda_arch_list" = "x"],[
          ## Nothing to target at all. Do not emit a flagless nvcc invocation and
          ## hope: that silently produces code for whatever the toolkit's default
          ## is, which is the kind of unstated assumption this file exists to kill.
          cuda_report_why="$NVCC_BIN reports no usable compute capabilities"
          enable_cuda=no
        ],[
          NVCC_ARCH_FLAGS=""
          cuda_arch_highest=""
          for a in $cuda_arch_list; do
            NVCC_ARCH_FLAGS="$NVCC_ARCH_FLAGS -gencode arch=compute_${a},code=sm_${a}"
            cuda_arch_highest="$a"
          done
          ## PTX for the highest, so a device newer than anything we compiled for
          ## still runs by JIT instead of failing with "no kernel image available"
          NVCC_ARCH_FLAGS="$NVCC_ARCH_FLAGS -gencode arch=compute_${cuda_arch_highest},code=compute_${cuda_arch_highest}"
          cuda_report_arch=`echo $cuda_arch_list | tr ' ' ','`" +PTX ($cuda_arch_source)"
        ])
      ])
    ])
  ])

  ## ------------------------------------------------- 3. does it actually work
  RNA_FEATURE_IF_ENABLED([cuda],[

    cat > conftest.cu <<_ACEOF
#include <cuda_runtime.h>
__global__ void vrna_conftest_kernel(int *p) { *p = 1; }
int main(void) { int n = 0; return (cudaGetDeviceCount(&n) == cudaSuccess) ? 0 : 0; }
_ACEOF

    ## nvcc drives a host compiler. It must be ABI-compatible with the one
    ## building the rest of the tree, and nvcc REFUSES host compilers newer than
    ## it knows about -- which is the single most common reason a CUDA build fails
    ## on an up-to-date distribution. Rather than disabling the backend over it,
    ## fall back through older gcc majors and SAY SO, because a working build with
    ## a stated caveat beats a silent 1x.
    AS_IF([test "x$NVCC_HOST_CC" != "x"],
          [cuda_host_candidates="$NVCC_HOST_CC"],
          [cuda_host_candidates="$CC gcc-14 gcc-13 gcc-12 gcc-11 gcc-10 gcc-9 clang"])

    AC_MSG_CHECKING([whether $NVCC_BIN can compile a CUDA translation unit])

    cuda_host_ok=""
    for cuda_cc in $cuda_host_candidates; do
      AS_IF([test "x$cuda_host_ok" = "x"],[
        AS_IF([test "x$cuda_cc" = "x$CC"], [], [
          ## only consider an alternative that exists
          AS_IF([($cuda_cc --version) >/dev/null 2>&1], [], [continue])
        ])
        AS_IF([$NVCC_BIN -ccbin "$cuda_cc" $NVCC_ARCH_FLAGS -c conftest.cu -o conftest.cu.o >/dev/null 2>&1],
              [cuda_host_ok="$cuda_cc"])
      ])
    done

    AS_IF([test "x$cuda_host_ok" = "x"],[
      AC_MSG_RESULT([no])
      cuda_report_why="found $NVCC_BIN but it could not compile a trivial CUDA program with any available host compiler (see config.log)"
      enable_cuda=no
    ],[
      AC_MSG_RESULT([yes])
      NVCC_HOST_CC="$cuda_host_ok"
      cuda_report_host="$NVCC_HOST_CC"
      AS_IF([test "x$NVCC_HOST_CC" = "x$CC"], [],
            [cuda_report_host="$NVCC_HOST_CC (NOT $CC -- nvcc rejected it)"
             ac_rna_warning="$ac_rna_warning
The CUDA objects will be built with $NVCC_HOST_CC rather than $CC, because nvcc
refused $CC. This normally works, since the interface between them is plain C,
but if the final link fails try --disable-cuda or a matching host compiler."])

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
      for d in "$cuda_root/lib64" "$cuda_root/lib" \
               "$cuda_root/lib64/stubs" "$cuda_root/targets/x86_64-linux/lib"; do
        AS_IF([test -f "$d/libcudart.so" || test -f "$d/libcudart.a"],
              [cuda_libdir="$d"; break])
      done

      AS_IF([test "x$cuda_libdir" != "x"],
            [CUDA_LIBS="-L$cuda_libdir -lcudart"],
            [CUDA_LIBS="-lcudart"])

      AS_IF([test -d "$cuda_root/include"],
            [NVCC_FLAGS="$NVCC_FLAGS -I$cuda_root/include"])

      ## And LINK it, not just compile it. The old macro stopped at -c, so a
      ## libcudart that configure could not find became a wall of undefined
      ## references at the final link of the library -- minutes later, in a
      ## message that names neither CUDA nor configure. Catch it here instead.
      AC_MSG_CHECKING([whether a CUDA program links with $CUDA_LIBS])
      AS_IF([$NVCC_BIN -ccbin "$NVCC_HOST_CC" conftest.cu.o -o conftest.cu.exe $CUDA_LIBS >/dev/null 2>&1],
            [AC_MSG_RESULT([yes])
             AC_DEFINE([VRNA_WITH_CUDA], [1],
                       [Build the CUDA GPU backend for MFE prediction])],
            [AC_MSG_RESULT([no])
             cuda_report_why="compiled a CUDA program but could not link it ($CUDA_LIBS); libcudart was not found"
             enable_cuda=no])

      rm -f conftest.cu.exe
    ])

    rm -f conftest.cu conftest.cu.o
  ])

  ## ------------------------------------------------ 4. honour the user's intent
  ##
  ## AUTO may downgrade. An explicit --enable-cuda may NOT: somebody who typed the
  ## flag wants the accelerator, and handing them a CPU-only binary with a warning
  ## buried in 400 lines of configure output is how 50 minutes of A100 time got
  ## spent on a build that could not possibly have used the GPU.
  AS_IF([test "x$enable_cuda" = "xno" && test "x$cuda_user_choice" = "xyes"],[
    AC_MSG_FAILURE([
==========================
--enable-cuda was requested but the CUDA backend cannot be built:

  $cuda_report_why

Fix the toolkit, point at it with --with-cuda-prefix=DIR, or drop --enable-cuda
to let configure decide (it will then build a working CPU-only library).
==========================
    ])
  ])

  ## Under AUTO, a machine with no toolkit is the normal case and must not look
  ## like something went wrong. It is reported once, in the summary.
  AS_IF([test "x$enable_cuda" = "xno" && test "x$cuda_report_why" != "x"],
        [AC_MSG_NOTICE([CUDA backend not built: $cuda_report_why])])

  AC_SUBST(NVCC_BIN)
  AC_SUBST(NVCC_HOST_CC)
  AC_SUBST(NVCC_FLAGS)
  AC_SUBST(CUDA_LIBS)

  AM_CONDITIONAL(VRNA_AM_SWITCH_CUDA, test "x$enable_cuda" = "xyes")
])
