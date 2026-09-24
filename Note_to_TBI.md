# CUDA_RNAFold — file manifest against ViennaRNA 2.7.2

*Prepared for the ViennaRNA maintainers (TBI). Branch `Finished_Port`, measured
against the `v2.7.2` tag in this repository — upstream release `1ffec79f`
(Ronny Lorenz, 2025-12-29), which is the exact merge-base and an ancestor of this
branch. Every count below is reproducible with `git diff v2.7.2..Finished_Port`;
nothing here is taken from a vendored copy that could drift.*

This document is a **manifest**: what each new file is, and what changed in each
file ViennaRNA already owns. It deliberately does not re-argue the proposal —
that is `PORT_UPSTREAM_PROPOSAL.md` — nor restate the measurements, which are in
`CUDA_RNAFold_History.md`.

---

## 1. How to read the diff

`git diff --shortstat v2.7.2..Finished_Port` reports **146 files, +36 249, −24**
(including this document), after the notebooks and result JSON were untracked.
Even that is more than the proposal: about half the remaining lines are the
project's own `PORT_*.md` scope documents and tooling, which are how the port was
argued rather than anything offered upstream. The notebooks and measurement JSON
that used to dominate this figure have been untracked entirely.

The split that matters:

| | files | lines | what it is |
|---|---|---|---|
| **A. Library files upstream owns** | **8** | **+477 / −7** | the part that needs defending |
| **B. The driver, `src/bin/RNAfold.c`** | 1 | +1 670 / −13 | ours in effect; not proposed |
| **C. Build system (modified)** | 7 | +196 / −4 | `--enable-cuda` wiring |
| **D. New CUDA backend** `src/ViennaRNA/mfe/cuda/` | 18 | +13 113 | a new subdirectory; take it or leave it |
| **E. New autoconf macros** | 2 | +193 | `m4/ac_rna_cuda.m4`, `ac_rna_asserts.m4` |
| **F. New tests and fixtures** | 35 | +4 023 | including standalone upstream reproducers |
| **G. Documents and tools** | 75 | +16 577 | **not code, not proposed** |

**All 24 deleted lines** are accounted for: 7 in the eight library files, 13 in
`RNAfold.c`, 4 list-continuations in `tests/Makefile.am`. There is no upstream
code removed anywhere else in the tree.

> **A correction to our own earlier numbers, now fixed at source.**
> `CUDA_RNAFold_History.md` §2 reported the modified upstream files as
> "9 files, +2 296 / −20". The line total there included the +150 of
> `Makefile.am` glue while the file count did not; the nine files (rows A + B
> above) are **+2 147 / −20**. Its headline had drifted too. Both are corrected
> in that document as of this commit, and every figure in both is now computed
> from the tree rather than carried forward.

### Verifying the claims in this document

```sh
git diff --numstat v2.7.2..Finished_Port      # every file, every count
tools/list_local_patches.sh                   # the live patch inventory
tools/list_local_patches.sh --check           # pairing check only; non-zero on error
```

`list_local_patches.sh` reads the **source**, not this document. It lists every
marked region, verifies each `BEGIN` pairs with its `END` in order, and — the
part that matters — lists every upstream file changed against `v2.7.2` and
**fails if one is not marked**. An unmarked edit is a failure of the honesty
check, not a silent omission.

---

## 2. Part A — the eight files ViennaRNA already owns

Every edit is bracketed in-source, so it can be found without reading thirteen
thousand lines of CUDA to locate three hundred:

```c
/* VRNA-PATCH-BEGIN(<id>, <CLASS>) -- PORT_LOCAL_PATCHES.md
 * why
 */
   ...the change...
/* VRNA-PATCH-END(<id>) */
```

**12 marked regions across 8 files.** The classes are separated because they have
very different odds and should be judged separately:

| class | meaning | standalone? | count |
|---|---|---|---|
| **DEFECT** | an upstream bug we fixed, with a reproducer | **yes** — stands whether or not any CUDA work is accepted | 1 |
| **SEAM** | an attachment point we need that upstream lacks; shaped to be useful on CPU with no GPU at all | yes, as a feature proposal | 7 |
| **REACH** | a capability upstream **already has** that its public API cannot reach | yes — usually the hardest to argue against | 4 |

### The inventory

| file | +/− | regions |
|---|---|---|
| `src/ViennaRNA/params/params.c` | +73 / −5 | `params-cache-race` (**DEFECT**) |
| `src/ViennaRNA/mfe/mfe.c` | +166 / −2 | `batch-backend-state`, `inside-engine-hook` (SEAM); `circular-postprocess`, `bps-backtrack` (REACH) |
| `src/ViennaRNA/mfe/global.h` | +85 / 0 | `batch-backend-api` (SEAM); `circular-postprocess` (REACH) |
| `src/ViennaRNA/grammar/mfe.h` | +72 / 0 | `inside-engine-api` (SEAM) |
| `src/ViennaRNA/grammar/gr_extension_mfe.c` | +35 / 0 | `inside-engine-bind` (SEAM) |
| `src/ViennaRNA/grammar/grammar.c` | +16 / 0 | `inside-engine-prepare` (SEAM) |
| `src/ViennaRNA/intern/grammar_dat.h` | +15 / 0 | `inside-engine-slots` (SEAM) |
| `src/ViennaRNA/backtrack/global.h` | +15 / 0 | `bps-backtrack` (REACH) |

### 2.1 The one DEFECT — `params-cache-race`

`SPEEDUP_PARAMS` is a process-wide `vrna_param_t` cache that `vrna_params()` and
`vrna_exp_params()` both **read and write** on every call — including on a cache
*hit*, where the caller's `window_size`, `min_loop_size` and `max_bp_span` are
written into the shared copy before the comparison. The nearby
`#pragma omp threadprivate` covers only `id`/`pf_id`, not the cache, and
`SPEEDUP_PARAMS` is `#define`d to 1 with no way to opt out.

This is a data race for **any** threaded caller, with or without a GPU. The
reproducer is `tests/upstream/params_race_probe.c` (and `tools/params_race.c`):
it reports **11 999 of 16 000 parameter tables wrong** when model details differ
between threads, and 0 of 16 000 when they are identical — which is why it has
gone unnoticed, since `RNAfold` passes one model for the whole run.

**This is the strongest standalone contribution here** and we would submit it on
its own regardless of what happens to the rest.

**It is not the only defect we found, only the only one we patched.**
`PORT_UPSTREAM_PROPOSAL.md` Part 1 reports five in 2.7.2 — three in the library
and two that stop a build from git — each with a reproducer in `tests/upstream/`
(§7). We report rather than patch them on purpose: they are independent of
CUDA, and fixing them inside a large feature branch would bury them.

### 2.2 The SEAM patches — one idea in four files plus a batch entry

Five of the seven SEAM regions implement a single thing: **a fold compound can
be handed an alternative implementation of the MFE inside (matrix-fill) step.**

- `intern/grammar_dat.h` adds four fields to `aux_grammar` (`engine`,
  `engine_data`, `engine_prepare_data`, `engine_free_data`);
- `grammar/mfe.h` declares the public setter and its callback type;
- `grammar/gr_extension_mfe.c` binds it;
- `grammar/grammar.c` gives it the same prepare/free lifecycle the other
  aux-grammar callbacks already have;
- `mfe/mfe.c` calls it where `fill_arrays()` would run.

It is **per fold compound**, it defaults to unset, and an engine that declines
leaves upstream's own recursion to run. With no engine registered, the code path
is upstream's exactly.

The remaining two SEAM regions add `vrna_mfe_batch()` and
`vrna_mfe_batch_backend_set()` (`mfe/global.h`, `mfe/mfe.c`): a process-wide,
at-most-one backend for folding *a batch* of compounds, because a batch is not a
property of any single fold compound. **A backend that declines falls through to
a plain loop over `vrna_mfe()`**, so the answer is the library's own either way.
That is what makes the accelerator optional rather than load-bearing.

### 2.3 The REACH patches — capabilities that exist but cannot be called

- **`bps-backtrack`** (`backtrack/global.h`, `mfe/mfe.c`) — exposes
  `vrna_backtrack_from_intervals_bps()`, which fills the caller's `vrna_bps_t`
  directly instead of down-converting to `vrna_bp_stack_t`. The legacy form
  discards a G-quadruplex's layout (`bp.L`, `bp.l[3]`), which `vrna_db_from_bps()`
  needs to render the box — so a caller backtracking pre-filled matrices
  **cannot produce a correct gquad structure through the legacy entry at all**.
- **`circular-postprocess`** (`mfe/global.h`, `mfe/mfe.c`) — `vrna_mfe()` runs
  `postprocess_circular()` unconditionally under `md.circ`, so a fold that goes
  through upstream gets it for free. A caller that fills the matrices itself has
  no way to invoke it, which is why `-c` was declined until this was exposed.

---

## 3. Part B — `src/bin/RNAfold.c` (+1 670 / −13)

Declared a local patch **whole-file** (`VRNA-PATCH-FILE(rnafold-driver, DRIVER)`)
rather than hunk by hunk: marking each hunk would be noise pretending to be
precision. It is a CUDA chunker wrapped around upstream's per-record loop —
accumulating records into batches, budgeting VRAM, building fold compounds on a
thread pool, dispatching a chunk to `vrna_mfe_batch()`, and falling back to the
per-record path below a size threshold.

**It is ours in effect and is not part of any proposal.** The 13 deleted lines
are the per-record body being moved into the chunk builder, not behaviour
removed.

Its CUDA-specific surface is small and guarded: one include, a handful of
`#ifdef VRNA_WITH_CUDA` blocks, and three calls —
`vrna_cuda_devices()`, `vrna_cuda_register_batch_backend()` and
`vrna_cuda_engine_allow()` (a test hook). Everything else — the chunk
accumulator, the VRAM budget, the fold-compound build pool, the fallback
threshold — is backend-agnostic and would serve any batch backend. Compiled
without `--enable-cuda` the file is upstream's driver plus chunking.

---

## 4. Part C — build system (modified, +196 / −4)

| file | + / − | what |
|---|---|---|
| `src/ViennaRNA/Makefile.am` | +144 / 0 | the `.cu` build rules, the CUDA sources list, `libRNA` link |
| `tests/Makefile.am` | +30 / −4 | registers the new `.ts` suites; the 4 deletions are list continuations |
| `.gitignore` | +8 / 0 | build artifacts |
| `src/bin/Makefile.am` | +6 / 0 | binaries link `libRNA_conv.la` directly, so they need `$(CUDA_LIBS)` themselves — it is not inherited from `libRNA.la` |
| `silent_rules.mk` | +5 / 0 | an `NVCC` line for `make V=0`, matching the existing style |
| `m4/ac_rna.m4` | +3 / 0 | calls `RNA_ENABLE_CUDA`, then `RNA_ENABLE_ASSERTS` **after** it (that macro appends `-DNDEBUG` to `NVCC_FLAGS`, which `RNA_ENABLE_CUDA` sets) |
| `doc/man2rst.py` | mode only | upstream ships it mode 644; the port sets 755, without which a fresh clone from git does not build |

**`configure.ac` is not modified.** The feature attaches through `m4/ac_rna.m4`,
which is where upstream already aggregates its `RNA_ENABLE_*` macros.

---

## 5. Part D — the new CUDA backend, `src/ViennaRNA/mfe/cuda/` (18 files, +13 113)

A new subdirectory. Nothing upstream owns is touched by it, so this part can be
taken or left independently of Part A.

### Attachment and orchestration

| file | lines | what it is |
|---|---|---|
| `engine.c` | 530 | The backend's attachment point and **routing guard**. Contains no CUDA. Decides *whether* a fold compound may go to the device — the decision that has to be conservative, since anything it wrongly admits is a wrong answer. |
| `engine.h` | 93 | The engine's public header. |
| `mfe_cuda.c` | 1 458 | The batch orchestrator: chunk lifecycle, per-record fetch and backtrack, the worker pool, phase accounting. |
| `stub2.h` | 1 039 | Shared declarations and the index helpers (`Indx()`, `Hoff()`), widened so arrays may exceed 2×10⁹ elements. Also the documented home of the `RNA_*` environment knobs. |

### Host-side row loop

| file | lines | what it is |
|---|---|---|
| `fill_arrays.c` | 472 | Host helpers for the row loop. |
| `fill_arrays_loop.c` | 540 | The row loop itself — the sweep that drives every device phase, row by row. |
| `mb_loop_fast.c` | 357 | Multibranch host code, split off from `multibranch_loops.c` to isolate the data dependence. |

### Device code

| file | lines | what it is |
|---|---|---|
| `device.cu` | 716 | Device discovery, streams, pinned transfers, the per-row offset tables. The first `.cu` translation unit; it exists partly to exercise the nvcc build rule. |
| `int_loop.cu` | 2 403 | The interior-loop kernel and the `d_S` / `d_my_c` device buffers. |
| `int_loop_kernel_body.inc` | 173 | The kernel body, included once per candidate `BLOCK_SIZE` so several instantiations stay available to a selection strategy. |
| `interior_loopx.h` | 213 | Device-callable interior-loop energy helpers. |
| `hp_mb_loop.cu` | 2 116 | Hairpin / multibranch / 3′-extension precompute. Replaces four full `nfiles × ijsize` **host** arrays that used to live in `fill_arrays.c`. |
| `modular_decomposition.cu` | 2 241 | The `fML` modular-decomposition kernel — the dominant phase at production sizes. |

### G-quadruplex transport

| file | lines | what it is |
|---|---|---|
| `gquad.c` | 162 | Host side: flattens the batch's `c_gq` matrices. Separate translation unit for a **build** reason — the ViennaRNA headers carry no `extern "C"`. |
| `gquad.cu` | 434 | Carries `c_gq` to the device and provides the lookup. |
| `gquad_dev.h` | 61 | The device-side `c_gq` lookup — one definition, because its two call sites must agree with upstream and with each other. |

### Build shim and a bit-trick

| file | lines | what it is |
|---|---|---|
| `nvcc-libtool.sh` | 69 | Lets libtool drive nvcc: libtool appends the host compiler's PIC flags to whatever compiler it is handed, which nvcc does not accept directly. |
| `nth.h` | 25 | `find_nth_set_bit`. **Third-party provenance** — see §8. |

---

## 6. Part E — new autoconf macros (2 files, +193)

| file | lines | what |
|---|---|---|
| `m4/ac_rna_cuda.m4` | 135 | Adds **`--enable-cuda` (default: off)**. Locates `nvcc`, pins the host compiler nvcc drives to the one building the rest of the tree, and sets `NVCC_FLAGS` / `CUDA_LIBS`. `--with-cuda-prefix=DIR` and `--with-cuda-arch=LIST` are available. **Failure to find a usable nvcc disables the feature with a warning rather than failing configure**, so `--enable-cuda` on a toolkit-less machine still produces a working CPU build. |
| `m4/ac_rna_asserts.m4` | 58 | Assertion control that also reaches `NVCC_FLAGS` — without it `-DNDEBUG` reached the C compiler and never nvcc. |

> **The flag is `--enable-cuda`, not `--with-cuda`.** Autoconf treats an unknown
> `--with-*` as a warning rather than an error, so the wrong spelling configures
> cleanly, builds cleanly, and produces a silently CPU-only binary. We lost an
> A100 session to exactly that.

---

## 7. Part F — new tests and fixtures (35 files, +4 023)

Two groups, and the second may be of more immediate interest than the first.

**`tests/mfe_cuda_*.ts` and `tests/mfe_engine.ts`** — suites in upstream's own
harness, one per option family: `guard`, `gquad`, `constraints`, `nsp`, `nolp`,
`dangles`, `span`, `noclosinggu`, `fm1`, `circ`, plus the engine seam itself.
With reference outputs under `tests/circ/`, `tests/gquad/`, `tests/salt/`,
`tests/noclosinggu/`. The standing bar throughout is that **output must be
byte-identical to the same binary with the GPU path off** — self-comparison
rather than an oracle.

**`tests/upstream/`** — standalone reproducers for the upstream behaviour we
report, runnable in about a minute via `tests/upstream/run_probes.sh`, and
**independent of any CUDA**:

| probe | what it demonstrates |
|---|---|
| `params_race_probe.c` | the `params.c` cache race — the one DEFECT we patched |
| `params_window_probe.c` | the same cache's **hit** path writing three caller fields into the shared static before comparing |
| `nolp_salt_probe.c` | whether `--noLP`'s stacking term misses the salt correction every other stack in the MFE recursion receives |
| `aux_index_probe.c` | whether the MFE aux-grammar inside callback receives the segment's 5′ delimiter `i` as documented, or the **rule index**, because of a shadowing `size_t i` at `mfe/mfe.c:502` |
| `circ_fm2_probe.c` | whether `fM2_real` is the same quantity the multibranch modular decomposition already computes for `fML` |
| `gquad_sparsity_probe.c` | how sparse `c_gq` is (under 5 % full) |
| `gquad_rowstats_probe.c` | `c_gq`'s row structure, which is what the device representation turns on |
| `scope_one.c` | one model detail per *process* — `par_mfe()` carries energy-parameter state across calls |
| `cuda_sweep_harness.c` | drives the batch sweep directly, so `RNA_ROW_VERIFY` has something to run |
| `thread_parity.sh` | threaded-versus-serial parity |

**Note that only the first of these became a patch.** The others are reported
rather than fixed, deliberately: they are upstream questions with nothing to do
with CUDA, and burying them inside a large feature branch would be the wrong
shape for review. `PORT_UPSTREAM_PROPOSAL.md` Part 1 is where they are argued.

---

## 8. Provenance, and one thing to confirm before publication

**The CUDA kernels are not new work from scratch.** Several files carry
`WBL … ViennaRNA-2.3.0` headers and CVS-style revision markers — they originate
in **William B. Langdon's** CUDA work against ViennaRNA 2.3.0 and have been
forward-ported to 2.7.2 here. `int_loop.cu`, `modular_decomposition.cu`,
`fill_arrays.c`, `fill_arrays_loop.c`, `mb_loop_fast.c`, `interior_loopx.h` and
`stub2.h` all carry that lineage; `hp_mb_loop.cu`, `engine.c`, `device.cu` and
the `gquad` files are new.

`nth.h` (25 lines) credits **njuffa**, from an NVIDIA developer-forum post
(`find_nth_clear_bit`, 2014-12-29).

**We flag both rather than resolve them:** attribution and licence compatibility
for the Langdon-derived files and for the forum snippet should be settled
explicitly before any upstream submission. This document records what the source
headers say; it does not make a licensing claim.

**Attribution line added 2026-09-24.** Every file in this tree that already
carried an author/date/change block now also carries

```
WBL & LAW added CUDA enabled arm of RNAFold 24/09/2026
```

— ten files: the eight CUDA-directory files with existing changelogs
(`fill_arrays.c`, `fill_arrays_loop.c`, `hp_mb_loop.cu`, `int_loop.cu`,
`interior_loopx.h`, `mb_loop_fast.c`, `modular_decomposition.cu`, `stub2.h`),
plus `mfe_cuda.c` (whose block is inherited from upstream's `mfe.c`, so the line
is a clearly separate trailing entry) and `src/bin/RNAfold.c` (in *our*
`VRNA-PATCH-FILE` block, not in upstream's author block below it).

**It was deliberately NOT added to `params.c` or `mfe.c`**, the only two files
upstream owns that carry author blocks. In `params.c` the line would be simply
untrue — our change there is the parameter-cache race fix, which has nothing to
do with CUDA. In `mfe.c` it would be accurate but would mean writing our names
into Ivo Hofacker's authorship header, which is a decision for the maintainers
to invite rather than for us to take. `nth.h` was also left alone: it is a
25-line credit block for njuffa's snippet and is not ours to annotate.

---

## 9. What is deliberately not offered

Group G of the table in §1 — **75 files and 16 577 lines** of `PORT_*.md` scope
documents, specifications and notebook-generator tooling. They are how the port
was argued and measured, not part of what is being proposed.

The Colab notebooks and the measurement JSON that used to sit alongside them are
**no longer tracked at all**, on this branch or on the development branch: none
of it exists in ViennaRNA and none of it ever would. That is why the headline in
§1 is 146 files rather than the 220 an earlier version of this document reported.

## 10. Where to read further

| question | document |
|---|---|
| what should upstream actually be asked for | `PORT_UPSTREAM_PROPOSAL.md` |
| what the patch classes mean, and each patch's rationale | `PORT_LOCAL_PATCHES.md` |
| what the tree is, what is accelerated, what is declined | `CUDA_RNAFold_History.md` |
| per-option status, and why each was accelerated or declined | `PORT_OPTION_STATUS.md` |
| earlier merge analysis (partly superseded — read its §0 first) | `MERGING.md` |
