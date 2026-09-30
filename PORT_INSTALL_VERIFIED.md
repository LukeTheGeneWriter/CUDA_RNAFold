# Installing from upstream's own instructions: verified, and two findings

Run 2026-09-28. `PORT_ZEROCONF_SCOPE.md` §11 set the requirement — *a user who has never
heard of this fork should get acceleration by following upstream's instructions.* This is
that requirement executed, verbatim, on a fresh clone.

---

## 1. What was run

Exactly the ViennaRNA GitHub page's git-repository path, with no helper script of ours,
no `--without-*` flags and no environment variables:

```bash
git clone … ViennaRNA && cd ViennaRNA
cd src && tar -xzf libsvm-3.35.tar.gz && cd ..
cd src && tar -xjf dlib-20.0.tar.bz2  && cd ..
autoreconf -i
./configure --prefix=$HOME/ViennaRNA      # their documented no-root variant
make
make install
```

Prerequisites installed were **exactly** upstream's list for Debian: `build-essential`,
`autoconf`, `automake`, `libtool`, `pkg-config`, `gengetopt`, `help2man`, `bison`,
`flex`, `vim-common`, `swig`, `liblapacke`, `liblapack`, `gfortran`. Four of them were
genuinely absent on the test machine beforehand.

**The honest limit:** this is a fresh clone in a fresh prefix on an *existing* machine,
not a fresh distro. `nvcc`, `gcc` and `swig` were already present. A user on a CUDA
machine has `nvcc`, so that is realistic — but it means this run cannot speak to the
no-toolkit experience. `tests/zeroconf_configure.sh` covers that separately by hiding
`nvcc` behind a shim, and passes 16/16.

## 2. Result: it works

| criterion | result |
|---|---|
| 1 — bare `./configure` accelerates | **PASS** |
| 2 — no env var needed at run time | **PASS** |
| 3 — configure states its verdict | **PASS** |
| 4 — the git-clone path works | **PASS** |

`autoreconf -i` is sufficient. Our `autogen.sh` adds `-I config`, which is unnecessary:
`configure.ac` declares `AC_CONFIG_MACRO_DIR([m4])`, and `autoreconf -i` populates the
aux dir itself.

What a bare `./configure` prints:

```
GPU Acceleration
----------------
  * CUDA backend              : yes
      - nvcc                  : /usr/bin/nvcc (release 12.4)
      - host compiler         : gcc
      - compute capabilities  : 86 +PTX (detected from the local device(s))
```

The **installed** binary, with nothing set:

```
$ RNAfold --noPS -i seqs.fa
bin/RNAfold.c   GPU acceleration ON (1 device); reaching the device measured 1.61 s,
                so with 1 job a chunk needs 1.08e+06 matrix cells to beat the host
```

And Python:

```python
import RNA
RNA.cuda_devices()      # 1
RNA.cuda_fold(seqs)     # [(structure, energy), …]  -- verified against the CPU
```

**`make` succeeded**, so neither of the two clone-build blockers this project has
recorded is live on the documented path: `doc/man2rst.py`'s mode did not stop it, and
`yacc` was present because upstream's own list installs `bison`. An earlier `yacc:
command not found` in RNAforester was this machine missing a prerequisite upstream
explicitly tells you to install — **not** a defect in the tree, and it is withdrawn as
one.

## 3. Finding: the Python module is not under `$PREFIX/lib`

With `--prefix=$HOME/ViennaRNA` the module installs to

    $PREFIX/local/lib/python3.14/dist-packages/RNA/

Note the `local/` segment — Debian's `posix_local` install scheme. So the obvious
`PYTHONPATH=$PREFIX/lib/python3.x/site-packages` finds nothing, and it reads as "the
bindings did not install" when they did. Upstream's page does not mention it. Worth a
line in our README and worth raising with TBI, since it affects their no-root users on
any Debian-family system.

## 4. Finding: `gpu_bytes_per_file()` is conservative enough that the library never needs to chunk

This started as a suspected defect and ended as a null, and the sequence is worth
recording because the reasoning was wrong twice.

**The suspicion.** `vrna_mfe_batch()` hands the whole batch to the backend;
`cuda_batch_cb()` calls `par_mfe(n, …)` with every record. All the sizing — the VRAM
budget, the chunk loop — lives in `RNAfold.c`. So `RNA.cuda_fold(seqs)` from Python
appeared to have nothing protecting it, which is the same shape as the teardown defect
this binding found in 2026-09: *a rule living in the one caller, and a second caller that
does not follow it.*

**A fix was written before the problem was measured.** That was the error. The chunk loop
went into `cuda_batch_cb()` with a code comment asserting "a few hundred records at
production length is tens of gigabytes of device buffers" — a number never measured.

**What measurement showed.** The clean-room build above predates the fix, which made a
free A/B available:

| batch | no chunking | with chunking |
|---|---|---|
| 200 × 2000 | works | works |
| 400 × 1500 | works | works |
| 1000 × 1200 | works | works |

1000 × 1200 runs as **one sweep**, `init_gpu(1000, 1200)`, all 1000 records resident, on
a 4 GB card. And the chunked build reported **1 chunk** for every one of those shapes —
so the loop **never fired at all**. It fixes no observed failure and cannot, on this
hardware, be reached.

**And a timing claim that was an artefact.** A first A/B run gave 25.2 s unchunked against
39.0 s chunked at 200 × 2000, and that was written up as "chunking costs 55 %". It is
nothing of the kind. The runs were all-A-then-all-B on a laptop this project's own notes
record as power-capped, where local A/B is only usable above ~5 %. Re-run **ABAB**:

    rep 1:  nochunk 36.6 s   chunked 36.1 s
    rep 2:  nochunk 36.8 s   chunked 36.6 s

No difference, and both slower than the 25.2 s outlier that anchored the claim. The
change is performance-neutral.

**Disposition: reverted**, on the ground that it is never-exercised code relocating the
teardown on a correctness-critical path with no demonstrated benefit — not because it is
slow, which it is not.

**What to keep from it.** If a genuine OOM case is ever found, the guard should be rebuilt
on a *measured* failure. The useful datum is that `gpu_bytes_per_file()` × n stays inside
`compute_gpu_usable_bytes()` even at 1000 × 1200 on 4 GB, so the existing budget is
conservative rather than tight, and a guard built on it would need calibrating against
real allocations before it could be trusted to fire correctly.

## 5. Method notes

- **Write the test before the fix.** The chunking fix existed for an hour before anything
  measured whether the problem did. "The large batches passed" was then read as
  confirmation, when the unchunked build passed them too.
- **ABAB, not AABB.** Already a note in this project
  (`feedback_probes_that_could_not_reach`), and it still got skipped. On a power-capped
  machine an AABB layout attributes warm-up and throttling to the change under test.
- **A passing import can be the wrong module.** The first Python probe globbed for a
  directory named `RNA` under `$PREFIX/lib`, found the *Perl* binding's `auto/RNA`, and
  Python imported it as an empty **namespace package** — so `import RNA` succeeded and
  `RNA.cuda_devices` did not exist. Same class as the fallback failures elsewhere in this
  project: the probe reached something, just not its subject.

---

## 6. Re-checked 2026-09-30 against the live README — two more paths, two defects fixed

The README documents six paths. Clone and `--prefix` were §1–§2. Two more were testable:

**The release tarball — PASSES.** `tar -zxvf ViennaRNA-2.7.2.tar.gz && ./configure &&
make && make install`, with NO `autoreconf`, from a tarball built by `make distdir`: CUDA
backend yes, the installed `RNAfold` swept on the device and matched the CPU byte for byte.
All 30 tracked `mfe/cuda` files are distributed (their `EXTRA_DIST` is deliberately
unconditional, so a tarball made on a machine without CUDA still carries the kernels).
One caveat, and it is upstream's: `make dist` from a `--without-doc` tree fails in `doc/`
(it needs Sphinx to produce the distributed pages), so the test tarball skipped `doc/` —
TBI builds releases with the doc toolchain present.

**DEFECT 1 — on a machine with NO CUDA, `./configure && make` FAILED.** `interfaces/RNA.i`
includes `cuda.i` unconditionally, so the Python and Perl modules reference
`vrna_cuda_devices()` on every build; but `engine.c`, its only definition, was compiled
only inside the CUDA conditional. The module linked with the symbol undefined, `make`
byte-compiles the package — which imports it — and the build stopped with
`ImportError: … undefined symbol: vrna_cuda_devices`. That is every upstream user without
an NVIDIA toolkit who has Python and SWIG. **The CPU-only bar could not see it:
`tests/zeroconf_configure.sh` configures `--without-python`.** Fix: without CUDA,
`Makefile.am` builds `engine.c` alone (host compiler, its existing stubs). Verified from a
clean clone with `--disable-cuda`: `make` OK, 0 undefined `vrna_cuda` symbols,
`RNA.cuda_devices() == 0`, `RNA.cuda_fold()` equals `RNA.fold()`.

**DEFECT 2 — `python -m build` after `./configure` FAILED.** The generated `setup.py`
globs `src/ViennaRNA/**/*.c*`, which catches six `.cu` files (setuptools:
`UnknownFileType`) and `fill_arrays*.c` / `mb_loop_fast.c`, which are `#include`d or
unbuilt. Fix: `setup.py.in` keeps only `engine.c` from `mfe/cuda` and compiles it with
`VRNA_CUDA_HOST_ONLY`, which `engine.c` honours by un-defining `VRNA_WITH_CUDA` for itself.
**Not** by adding `VRNA_WITH_CUDA` to setup.py's `comment_lines()` list: that rewrites
`config.h` in place and permanently, so a later `make` in the same tree would build a CLI
that silently never accelerates. Verified in a CUDA-configured tree (the README's exact
case): the wheel builds, compiles only `engine.o` from `mfe/cuda`, imports, folds equal to
`RNA.fold()`, and `config.h` still defines `VRNA_WITH_CUDA` afterwards. The wheel is
CPU-only by design; the accelerated module is the one `make install` builds.

**Closed:** `tests/zeroconf_configure.sh` case 7 now builds the Python module with no toolkit
reachable and imports it (all 7 cases pass). conda / PyPI / binaries remain out of scope.
