# What this branch adds to ViennaRNA 2.7.2, and why

*Branch `port27`, based on the `v2.7.2` tag — which is upstream `master` exactly
(`1ffec79`), so this is a diff against current HEAD, not against a release you
have moved on from.*

The claim is narrow and worth stating first: **this is an accelerator for the
existing CPU implementation, not a second implementation.** Every answer it
produces is ViennaRNA's own. Anything the GPU path cannot reproduce exactly is
folded by your code, unchanged, in the same run.

---

## The shape of the diff

```
12 upstream files modified   868 insertions, 6 deletions
28 files added               (all new, none shadowing anything)
```

**Five of the six deleted lines are list continuations and a declaration.** The
entire substantive deletion in the library is one line:

```c
-    energy = fill_arrays(fc, ms_dat);
```

replaced by five lines that call an optional backend and fall through to that
same `fill_arrays()` when there is none.

| upstream file | + | what it is |
|---|---|---|
| `src/ViennaRNA/mfe/mfe.c` | 74 | the inside-engine dispatch, `vrna_mfe_batch()` |
| `src/ViennaRNA/mfe/global.h` | 64 | `vrna_mfe_batch()` API |
| `src/ViennaRNA/grammar/mfe.h` | 68 | `vrna_gr_set_inside_engine()` API |
| `src/ViennaRNA/grammar/gr_extension_mfe.c` | 30 | its implementation |
| `src/ViennaRNA/grammar/grammar.c` | 11 | prepare/release lifetime for engine data |
| `src/ViennaRNA/intern/grammar_dat.h` | 11 | four fields on the internal grammar struct |
| `src/bin/RNAfold.c` | 443 | the chunk dispatch path (see below) |
| `src/ViennaRNA/Makefile.am`, `src/bin/Makefile.am`, `tests/Makefile.am`, `m4/ac_rna.m4`, `silent_rules.mk` | 167 | `--enable-cuda`, `--enable-asserts`, nvcc rules |

Everything else is new files under `src/ViennaRNA/mfe/cuda/`.

---

## The two API additions

Both are useful to you with no accelerator present, which is the point.

### 1. An inside-engine seam

`vrna_mfe()` calls a `PRIVATE static fill_arrays()` unconditionally, so there is
no way to supply an alternative matrix fill without shadowing `vrna_mfe()` —
which is exactly the hack this project carried for years and has now deleted.

```c
handled = 0;
if ((fc->aux_grammar) && (fc->aux_grammar->engine))
  handled = fc->aux_grammar->engine(fc, &energy, fc->aux_grammar->engine_data);
if (!handled)
  energy = fill_arrays(fc, ms_dat);
```

Registered per fold compound through the existing `aux_grammar` machinery, so
there is no new public struct and no global state. **Declining is part of the
interface**: a backend returns 0 for anything it does not fully support and gets
your answer instead of quietly computing a different one.

What made this a five-line change rather than a redesign: `fill_arrays()` writes
only `fc->matrices` and returns `f5[length]`, and `backtrack()`'s only data
source is the same `fc`. So an alternative fill is a drop-in, and backtracking,
circular post-processing, constraints and output remain yours.

### 2. A batch entry point

```c
unsigned int vrna_mfe_batch(vrna_fold_compound_t **fcs, size_t n,
                            char **structures, float *energies);
unsigned int vrna_mfe_batch_backend_set(vrna_mfe_batch_f cb, void *data);
```

`vrna_mfe_batch()` means "call `vrna_mfe()` on each", and with no backend
registered that is literally its implementation. It exists because a batch is
the only thing some backends can exploit — a GPU's advantage is entirely in
width, and there is currently no way to tell the library "here are four hundred
sequences" rather than asking it four hundred separate questions.

It is also the natural home for the thread pool and `vrna_ostream_t` ordering
that presently live in the `RNAfold` binary, where every other caller has to
reimplement them.

---

## How parity is achieved

Not by implementing every feature on the GPU. By **routing**.

`RNAfold.c` decides once per run whether the device path can reproduce this
model exactly. If not, the run folds on your per-record path and the user sees
stock ViennaRNA behaviour. Verified across the option surface — same binary,
GPU path on versus off, byte-for-byte:

| accelerated | routed to CPU, identical output |
|---|---|
| default, `-T`, `--dangles=2` | `-g`, `-c`, `-d0`, `-d1`, `-d3` |
| `-p`, `-p0`, `--MEA`, `--bppmThreshold` | `--noLP`, `--noGU`, `--noClosingGU` |
| | `--logML`, `--salt`, constraints, SHAPE, modified bases |

The partition function keeps working **with the GPU active**, because
`process_record()` still performs it itself; the batch supplies only the MFE
energy and structure.

Two guard layers, deliberately: a run-level decision in the driver, and a
per-batch backstop in the sweep that errors if an unsupported model reaches it
anyway. The second should be unreachable, and exists so that "unreachable" is
enforced rather than assumed.

---

## Verification

- **Your test suite**: `make check` green with the backend built and with it
  absent (137 tests, including 7 new ones for the seam and routing guard).
- **Byte-identical output**: 239 records over six reference inputs, with the GPU
  path on and off, at VRAM budgets 4/8/16/32 MB.
- **Cell-for-cell**: 161 695 device cells compared against the host recursion,
  zero disagreement.
- Energy model unchanged: 239/239 records reproduced from stock 2.7.2 before any
  CUDA work began.

## Three defects found in 2.7.2 along the way

Reported separately with reproducers, independent of any of the above:
`mfe/mfe.c:502` passes the rule index where auxiliary grammar callbacks expect
`i`; the `SPEEDUP_PARAMS` cache is an unsynchronised race (97.4 % of parameter
sets wrong under contention in one configuration); and two build defects that
stop a build from a fresh clone.

## What is not here

G-quadruplexes, circular RNA, salt corrections, multistrand and modified bases
are **routed, not accelerated**. Designs and frozen test vectors exist for the
first three; none of them is a prerequisite for this diff.
