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

  ## What reaching the GPU costs, hot and cold (3b below). On by default where a device
  ## is visible: it adds ~15 s to configure, and nothing where there is no GPU.
  AC_ARG_WITH([gpu-wake-probe],
              [AS_HELP_STRING([--without-gpu-wake-probe],
                              [do not time how long reaching the GPU takes (hot, and after 12 s idle); RNAfold then budgets a default. The values can be given instead as vrna_cv_gpu_wake_cold=SECONDS (and _hot, and _host=NAME to tie them to one host)])],
              [cuda_wake_probe="$withval"],
              [cuda_wake_probe=yes])

  cuda_report_nvcc=""
  cuda_report_wake=""
  cuda_report_arch=""
  cuda_report_host=""
  cuda_report_why=""
  cuda_report_hint=""

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
      ## A GPU the user can SEE but no toolkit to build for it is the case worth a
      ## sentence: say how to get the toolkit, not only that it is missing. Under WSL
      ## the driver comes from Windows, so the advice must be "toolkit only" -- a Linux
      ## driver package installed inside WSL can shadow WSL's own libcuda.
      AS_IF([nvidia-smi -L >/dev/null 2>&1 || test -e /proc/driver/nvidia/version],[
        AS_IF([test -e /dev/dxg || test -d /usr/lib/wsl/lib],[
          cuda_report_hint="an NVIDIA GPU is visible (WSL) but no CUDA toolkit is installed: install the
                                TOOLKIT ONLY, never a Linux driver -- tools/setup_ubuntu.sh does it
                                (NVIDIA's cuda-toolkit from the wsl-ubuntu repository) -- then re-run ./configure"
        ],[
          cuda_report_hint="an NVIDIA GPU is visible but no CUDA toolkit is installed: install one
                                (Ubuntu: tools/setup_ubuntu.sh, or sudo apt install nvidia-cuda-toolkit)
                                and re-run ./configure"
        ])
      ])
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

      ## AND libstdc++. nvcc output is C++, but the binaries and libRNA are linked
      ## by libtool with the C compiler, which does not add it. Through CUDA 12.4
      ## nothing in our objects needed it; a newer nvcc on Colab (2026-10-02,
      ## the python3.13 image) emits thread-safe function-local-static guards in
      ## the kernel launch stubs, and every build died at the final RNAfold link
      ## on undefined __cxa_guard_acquire/release/abort -- in device.o, gquad.o and
      ## modular_decomposition.o, code that had not changed. The check below links
      ## with nvcc, which adds libstdc++ itself, so it could not see this.
      AS_IF([test "x$cuda_libdir" != "x"],
            [CUDA_LIBS="-L$cuda_libdir -lcudart -lstdc++"],
            [CUDA_LIBS="-lcudart -lstdc++"])

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

  ## ------------------------------------------ 3b. what reaching the GPU costs
  ##
  ## RNAfold decides per input whether the device can win (gpu_batch.c, S3), and the
  ## largest fixed term is reaching it at all. That is a property of the MACHINE, and of
  ## how long the device idled: on the RTX 3050 laptop 0.07 s straight after another
  ## process used it, 1.4 s once it has idled 10 s or more, flat out to 120 s
  ## (PORT_SHORT_SEQUENCES.md 9.2). So time both, here, where a device can be asked:
  ##
  ##   hot   the median of 5 back-to-back runs, after one that wakes it;
  ##   cold  one run after 12 s of idle.
  ##
  ## RNAfold budgets the COLD one, and only on the host measured here (a build host is
  ## often not the run host). The hot one is reported. Order matters: step 2's
  ## nvidia-smi query wakes the device fully, so the cold sample needs its own idle.
  ##
  ## The probe links cudart statically (nvcc's default), so a libcudart.so off the
  ## loader's path cannot turn "a device answered" into "no device".
  AS_IF([test "x$enable_cuda" = "xyes"],[
    cuda_wake_hot=""
    cuda_wake_cold=""
    cuda_wake_host=""
    cuda_wake_how=""

    AS_IF([test "x$vrna_cv_gpu_wake_cold" != "x"],[
      cuda_wake_cold="$vrna_cv_gpu_wake_cold"
      cuda_wake_hot="${vrna_cv_gpu_wake_hot:-$vrna_cv_gpu_wake_cold}"
      cuda_wake_host="$vrna_cv_gpu_wake_host"
      cuda_wake_how="given"
    ],[test "x$cuda_wake_probe" = "xno"],[
      cuda_report_wake="not measured (--without-gpu-wake-probe); RNAfold budgets its default"
    ],[test "x$cross_compiling" = "xyes"],[
      cuda_report_wake="not measured (cross-compiling); RNAfold budgets its default"
    ],[
      AC_MSG_CHECKING([what reaching the GPU costs, hot and after 12 s idle])

      cat > conftest_wake.cu <<_ACEOF
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
int main(void)
{
  struct timespec a, b;
  int             n = 0;

  clock_gettime(CLOCK_MONOTONIC, &a);
  if ((cudaGetDeviceCount(&n) != cudaSuccess) || (n < 1))
    return 1;

  clock_gettime(CLOCK_MONOTONIC, &b);
  printf("%.3f\n", (double)(b.tv_sec - a.tv_sec) + 1e-9 * (double)(b.tv_nsec - a.tv_nsec));
  return 0;
}
_ACEOF

      ## a driver that hangs must not hang configure
      cuda_wake_to=""
      AS_IF([(timeout 5 true) >/dev/null 2>&1], [cuda_wake_to="timeout 60"])

      AS_IF([$NVCC_BIN -ccbin "$NVCC_HOST_CC" $NVCC_FLAGS conftest_wake.cu -o conftest_wake.exe >/dev/null 2>&1 &&
             $cuda_wake_to ./conftest_wake.exe >/dev/null 2>&1],[
        cuda_wake_hots=""
        for cuda_i in 1 2 3 4 5; do
          cuda_wake_hots="$cuda_wake_hots `$cuda_wake_to ./conftest_wake.exe 2>/dev/null`"
        done
        cuda_wake_hot=`for cuda_v in $cuda_wake_hots; do echo "$cuda_v"; done | sort -n | sed -n 3p`
        sleep 12
        cuda_wake_cold=`$cuda_wake_to ./conftest_wake.exe 2>/dev/null`
        cuda_wake_host=`uname -n 2>/dev/null`
        cuda_wake_how="measured"
        AC_MSG_RESULT([${cuda_wake_hot:-?} s hot, ${cuda_wake_cold:-?} s cold])
      ],[
        AC_MSG_RESULT([no device answered])
        cuda_report_wake="not measured (no device answered at configure time); RNAfold budgets its default"
      ])

      rm -f conftest_wake.cu conftest_wake.exe
    ])

    ## Numbers or nothing: what reaches config.h is reprinted by awk, so a stray
    ## character in a given value cannot become a compile error in gpu_batch.c. A
    ## cold sample under the hot median (noise) is raised to it.
    AS_IF([test "x$cuda_wake_how" != "x"],[
      cuda_wake_ok=`awk -v c="$cuda_wake_cold" 'BEGIN { print ((c + 0 > 0) ? "yes" : "no") }'`
      AS_IF([test "x$cuda_wake_ok" = "xyes"],[
        cuda_wake_hot=`awk -v h="$cuda_wake_hot" 'BEGIN { printf "%.3f", ((h + 0 > 0) ? h + 0 : 0) }'`
        cuda_wake_cold=`awk -v c="$cuda_wake_cold" -v h="$cuda_wake_hot" 'BEGIN { printf "%.3f", ((c + 0 < h + 0) ? h + 0 : c + 0) }'`
        cuda_wake_host=`echo "$cuda_wake_host" | tr -d '"\\\\'`
        AC_DEFINE_UNQUOTED([VRNA_GPU_WAKE_HOT_S], [$cuda_wake_hot],
                           [Reaching the GPU, seconds, straight after another process used it (configure measured or was given it)])
        AC_DEFINE_UNQUOTED([VRNA_GPU_WAKE_COLD_S], [$cuda_wake_cold],
                           [Reaching the GPU, seconds, after 12 s idle; RNAfold budgets this one])
        AC_DEFINE_UNQUOTED([VRNA_GPU_WAKE_HOST], ["$cuda_wake_host"],
                           [The host VRNA_GPU_WAKE_COLD_S applies to; empty means every host])
        cuda_wake_where="on every host"
        AS_IF([test "x$cuda_wake_host" != "x"], [cuda_wake_where="on $cuda_wake_host only"])
        cuda_report_wake="$cuda_wake_hot s hot, $cuda_wake_cold s after 12 s idle ($cuda_wake_how); RNAfold budgets the cold one $cuda_wake_where"
      ],[
        cuda_report_wake="not measured (the probe gave '$cuda_wake_cold'); RNAfold budgets its default"
      ])
    ])
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
