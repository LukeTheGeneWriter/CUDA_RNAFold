# Build, SWIG and install audit (2026-10-18)

*Requested by Luke as part of the cleanup effort. The audit covers configure, the full
accelerated build, the CPU-only builds with and without Python, a forced verbose rebuild of
every `.cu` file, `-Wall -Wextra` over our C sources, `make install` into a scratch DESTDIR,
static and shared links against the installed library, and `import RNA` with Python warnings
fatal. Host: WSL2 Ubuntu, gcc 15.2 (nvcc's host side is gcc 13), CUDA 12.4, SWIG 4.4.0,
Python 3.14.4, autoconf 2.72, automake 1.18.1. Tree: LFB at S1+S2 (PORT_SHORT_SEQUENCES.md).
Nothing here changes behaviour; each finding is a proposed cleanup item.*

## Findings, by consequence

| # | finding | consequence | proposed fix | size |
|---|---|---|---|---|
| **B1** | The hand-written `.lo` rules for the `.cu` files (`src/ViennaRNA/Makefile.am`) list only our own headers. They omit `config.h`, the ViennaRNA headers the device code includes (`fold_compound.h`, `model.h`, `params/basic.h`, `datastructures/basic.h`, ...), and the flags. Observed: after a reconfigure, the accelerated rebuild (zeroconf m1) recompiled **no** `.cu` file. | A header or configure change can leave **stale device objects** linked against a changed host struct. This is the stale-binary trap, in the build system. It does not bite a fresh build or a user; it bites development trees. | Let nvcc write dependencies (`-MD -MF mfe/cuda/$*.d`, included by the Makefile), or at least add `$(top_builddir)/config.h` and the ViennaRNA headers to each rule. | small |
| **B2** | nvcc is given only `-O3 <arch> -I...`, not `-DHAVE_CONFIG_H` or the `-D` flags the C side gets (`VRNA_WITH_SVM`, `VRNA_WITH_NAVIEW_LAYOUT`, `USE_FLOAT_PF`, `VRNA_DISABLE_C11_FEATURES`). So the `.cu` files' `#ifdef HAVE_CONFIG_H` includes are dead. `vrna_fold_compound_t` also ends with `#ifdef VRNA_WITH_SVM vrna_zsc_dat_t zscore_data`, so the device TUs see a **shorter struct** than the host. | **Latent, not live.** Only the tail member differs, and no device-side code uses `sizeof` of the struct or stores it by value (checked). It becomes live the day someone reads a member past it, or a configure option changes a member in the middle, and nothing would warn. | Pass `$(DEFS) $(RNA_CPPFLAGS)`, or at least `-DHAVE_CONFIG_H`, to nvcc in `NVCC_ALL_FLAGS`. | small |
| **B3** | The installed `RNAlib2.pc` lacks `@CUDA_LIBS@` (`-lcudart -lstdc++`). A static program that reaches the accelerator (`vrna_cuda_devices()`, `vrna_mfe_batch()`) fails to link with the installed `pkg-config --static` flags: hundreds of undefined `cuda*` references. With `-lcudart` added it links and runs (`devices 1`). | Downstream static linkers of the accelerated library. Shared linking is fine (`libRNA.so` records cudart). So is a static program that calls only upstream API (`vrna_fold()` linked and ran), because the archive members holding CUDA are never pulled in. | `Libs.private: -lm @CUDA_LIBS@` in `RNAlib2.pc.in`. | one line |
| **B4** | ptxas: `Value of threads per SM for entry ly_carry3_lb_kernel<...> is out of range. .minnctapersm will be ignored` (6 instantiations). | The `__launch_bounds__` minimum-blocks hint on the Lyngsø load-balanced carry kernel is **ignored** on sm_86. That is the "LB knob only" kernel (A100 S9), so the hint may do something on sm_80 and nothing here. | Check the hint against sm_80 and sm_86 limits; drop it or derive it per architecture. Record which arch it was tuned on. | small |
| **B5** | The `.cu` files raise nvcc front-end warnings: set-but-unused `g_fm2_cells` (modular_decomposition.cu:1420) and `g_gq_entries/rowslots/nfiles/row_total` (gquad.cu:88–90, 308); unreferenced `mask` (int_loop.cu:445), `matrices` (1679) and function `E_int_loop` (1659). One gcc warning, `-Wformat-extra-args` at int_loop.cu:195, is `gpuAssert`'s `"CUDA error: on first kernel.\n"` being passed an unused `file`. | Dead state and dead code (cleanup C-series territory). The format one is harmless: the extra argument is ignored and the message is complete. | Delete the dead ones; drop the stray argument. | small |
| **B6** | gcc `-Wformat`: `mfe_cuda.c:1343` prints a `size_t` with `%u` (`RNA_CIRC_VERIFY` message). | Wrong number printed above 4 G cells; UB in principle. Debug path only. | `%zu`. | one line |
| **B7** | Ten `-Wdiscarded-qualifiers` in `mfe_cuda.c` / `fill_arrays.c`. `par_mfe()` takes `const vrna_fold_compound_t **` and then calls upstream functions that take non-const (`vrna_mfe`, `vrna_fold_compound_prepare`, `stat_cb`, `vrna_backtrack_from_intervals_bps`, ...). | The `const` is a promise the code does not keep: those calls write the compounds. Harmless at run time; misleading to a reviewer. | Drop `const` from that parameter chain (it is ours), or cast once at the boundary with a comment. | small |
| **B8** | `-Wall -Wextra` on the driver: `gpu_batch.c` has set-but-unused `VC` and `Str` and unused `chunk`; `RNAfold.c` has unused `verbose`; `mfe_cuda.c` has a missing initializer (`backtrack_pool_args_t.ret_id`); `mb_loop_fast.c` has an `unsigned *` assigned to `int *` and a set-but-unused `domains_up`. | Leftovers from the C6 moves and the fork. The signedness one deserves a look, the rest are dead. | Clean up with the C-series. | small |
| **B12** | `make` regenerates the SWIG wrapper and `interfaces/Python/RNA.py` when an interface file changes, but does not refresh the copy the build-tree package imports, `RNA/RNA.py` (its rule is reached only through the `.pyc` targets, not `all-local`). Found 2026-10-19: after adding `cuda.i`'s `cuda_device_name()`, `_RNA*.so` had it and `import RNA` did not (`RNA/RNA.py` still dated 10-13). | A build-tree `import RNA` can pair a new `_RNA` with an old proxy: a new function is missing, or a changed signature is called the old way. Install copies afresh, so it does not reach users. | Add `RNA/RNA.py` to `all-local`'s prerequisites. | one line |
| **B9** | `make install` ends with libtool's `remember to run 'libtool --finish /usr/local/lib'` and the same for `.../python3.14/dist-packages/RNA`. | Stock autotools/upstream behaviour on a DESTDIR install. Benign. | None (note it in the install docs if users ask). | — |
| **B10** | configure WARNINGs: `re-set ar/ranlib/nm to compiler specific wrappers` (LTO), `Can't find libgsl`, `Failed to find mpfr.h`, and the doc toolchain (sphinx, pdflatex, makeindex, bibtex). | All upstream and all optional. gsl/mpfr only change optional features. | None. | — |
| **B11** | Upstream's own warnings, identical in every configuration: `-Wdiscarded-qualifiers` in `plotting/eps.c`, `utils/string_utils.c`, `io/io_utils.c`, `RNAplex.c`, `RNAsnoop.c` (gcc 15's `strchr` returns const for const input), and `-Wformat-overflow` at `RNAsnoop.c:1373`. | Not ours. `string_utils.c` / `io_utils.c` are in RNAlib and would be a clean upstream PR. | Optional upstream PR, not part of our patch set. | — |

## What was checked and is fine

- **SWIG 4.4.0 / Python 3.14.** The wrapper generates with only upstream's suppressed warnings
  (`-w511,509`). `import RNA` succeeds with `-W error` from both the build tree and the
  installed tree. The module installs to `/usr/local/lib/python3.14/dist-packages/RNA`, on
  `sys.path`, thanks to the `debian-local-scheme` fix.
- **Installed headers.** `ViennaRNA/mfe/cuda/engine.h` installs and compiles stand-alone for
  a downstream user.
- **Shared link and run** against the installed `libRNA.so`: correct.
- **nvcc host compiler.** Ubuntu's nvcc 12.4 rejects gcc > 13 (`crt/host_config.h`), but its
  wrapper scripts (`/usr/lib/nvidia-cuda-toolkit/bin/gcc`) quietly select gcc 13, so
  `-ccbin gcc` works here. On NVIDIA's own toolkit with a newer gcc, `ac_rna_cuda.m4` already
  walks gcc-14 down to clang and says so in the summary. No action.
- **Configure summary.** The CPU-only case states the reason and that folding runs on the CPU
  "exactly as stock ViennaRNA does".

## Build time (the zeroconf question)

See PORT_SHORT_SEQUENCES.md section 7.

- Each configure rewrites `config.h`, so each zeroconf case is a full rebuild. Ironically that
  never includes the `.cu` files, which is B1.
- About half of each build is the LTO link of about 30 programs.
- The SWIG wrapper is one 4.4 MB C++ file: 58 s and 1.2 GB per compile, and it is compiled
  **twice** because static libraries are on.

Harness-only fixes (`--disable-lto --disable-static` in the zeroconf flags, skip the restore
rebuild on a throwaway copy) are expected to take 908 s to about 350–400 s.
