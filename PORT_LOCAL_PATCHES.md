# Every change this project makes to a file ViennaRNA already owns

*Generated from the source, not maintained by hand. Run
`tools/list_local_patches.sh` for the live inventory; this document explains
what the classes mean and why each patch exists.*

---

## Why this exists

The CUDA path lives in its own new subdirectory, `src/ViennaRNA/mfe/cuda/`, and
that part is easy to talk about: upstream can take it or leave it, and nothing
they own changes. What actually needs defending in a pull request is the handful
of edits to files upstream already owns — **10 marked regions, ~300 lines, in 8
files** — and before this convention they were discoverable only by diffing
against `v2.7.2` and reading twelve thousand lines of added CUDA to find the
three hundred that matter.

Every such edit is now bracketed in-source:

```c
/* VRNA-PATCH-BEGIN(<id>, <CLASS>) -- PORT_LOCAL_PATCHES.md
 * why
 */
   ...the change...
/* VRNA-PATCH-END(<id>) */
```

`tools/list_local_patches.sh` lists them, checks that every BEGIN pairs with its
END in order, and — the part that matters — **lists every upstream file changed
against `v2.7.2` and fails if one is not marked.** It is a bar, not a catalogue:
an unmarked edit is a build failure of the honesty check, not a silent omission.

> **Two things the tool had to learn the hard way.** The baseline is the
> **`v2.7.2` tag** (which is exactly the merge-base), not `upstream/master` —
> diffing against master drags in every upstream commit since the release and
> buried the nine real files under a hundred irrelevant ones. And it passes
> `--ignore-cr-at-eol`, because this repo is checked out on Windows with
> `core.autocrlf=true`, so the same command run from WSL called *every vendored
> file* modified. The tool gave two different answers depending on which shell
> ran it until both were fixed.

## The classes, and why they are separate

| class | meaning | can it be submitted alone? |
|---|---|---|
| **DEFECT** | An upstream bug we fixed, with a reproducer. | **Yes** — and it stands whether or not any CUDA work is ever accepted. |
| **SEAM** | An attachment point the accelerator needs and upstream does not have. Shaped to be useful on CPU even with no GPU backend. | Yes, as a feature proposal. |
| **REACH** | A capability upstream **already has** that its public API cannot reach. | Yes, and usually the hardest to argue against. |
| **DRIVER** | `src/bin/RNAfold.c`. Ours in effect; not part of any proposal. | N/A |

The split matters because the three kinds have very different odds. A DEFECT
with a failing test is nearly unarguable. A REACH says "you already do this, we
just cannot call it". A SEAM asks for new API and is the one that needs the
strongest case.

## The inventory

### DEFECT

**`params-cache-race`** — `src/ViennaRNA/params/params.c`

`SPEEDUP_PARAMS` is a process-wide `vrna_param_t` cache that `vrna_params()` and
`vrna_exp_params()` both read **and write** on every call — including on a cache
*hit*, which writes the caller's `window_size`, `min_loop_size` and `max_bp_span`
into the shared copy before comparing. The `#pragma omp threadprivate` nearby
covers only `id`/`pf_id`, not the cache, and `SPEEDUP_PARAMS` is `#define`d to 1
with no way to opt out.

Reproducer `tools/params_race.c`: **11 999 of 16 000 parameter tables wrong
without the lock, 0 of 16 000 with it.** Threads requesting *identical* model
details corrupt nothing — every racing `memcpy` writes the same bytes — which is
exactly why `RNAfold -j` never visibly broke and why the bug survived.

A mutex rather than thread-local storage, because `vrna_param_t` is ~250 KB and a
library cannot know how many threads its caller has. Without pthreads the cache
is **disabled** rather than left racy: slower, not wrong.

### REACH

**`bps-backtrack`** — `src/ViennaRNA/mfe/mfe.c`, `src/ViennaRNA/backtrack/global.h`

`vrna_backtrack_from_intervals()` is the only **public** entry that backtracks
pre-filled matrices. Internally it builds a modern `vrna_bps_t`, calls the
**private** `backtrack()` — which *does* produce a G-quadruplex's layer/linker
layout in `bp.L` / `bp.l[3]` — and then downconverts to the legacy
`vrna_bp_stack_t`, **discarding both**. Upstream's own comment on that loop reads
`/* copy bps elements to bp_stack?! */`.

The consequence is not cosmetic. `vrna_db_from_bps()` renders a quadruplex with
`vrna_db_insert_gq(structure, i, bp.L, bp.l, length)`; with the layout gone,
`vrna_db_from_bp_stack()` can only write a single `+` where the box belongs.
**Measured: a fold whose energy was byte-exact against upstream rendered 2 `+`
characters where upstream rendered 14.** Nor is it recoverable afterwards —
`vrna_bt_gquad()` is public but needs the `(i,j)` span, and the marker sets
`i == j`.

`vrna_backtrack_from_intervals_bps()` adds the bps form. It is an **addition**;
every existing caller is untouched.

### SEAM

Five regions, one idea: let something else fill the matrices.

| id | file | what |
|---|---|---|
| `inside-engine-api` | `grammar/mfe.h` | the callback type and binder declarations |
| `inside-engine-bind` | `grammar/gr_extension_mfe.c` | `vrna_gr_set_inside_engine()` |
| `inside-engine-slots` | `intern/grammar_dat.h` | four fields on `aux_grammar` |
| `inside-engine-prepare` | `grammar/grammar.c` | the same prepare/free lifecycle the other aux-grammar callbacks have |
| `inside-engine-hook` | `mfe/mfe.c` | **the seam itself** |

`inside-engine-hook` is the whole ask, and it is small enough to quote:

```c
handled = 0;
if ((fc->aux_grammar) && (fc->aux_grammar->engine))
  handled = fc->aux_grammar->engine(fc, &energy, fc->aux_grammar->engine_data);
if (!handled)
  energy = fill_arrays(fc, ms_dat);
```

Everything after that point — backtracking, circular post-processing, output — is
unchanged. An engine that declines leaves upstream's own path untouched.

Plus two for many-at-once folding: `batch-backend-api` (`mfe/global.h`) and
`batch-backend-state` (`mfe/mfe.c`). `vrna_mfe_batch()` is useful to upstream on
CPU on its own — it is a place to put *any* many-at-once folder, threaded or not.

### DRIVER

**`rnafold-driver`** — `src/bin/RNAfold.c`, +1333 lines, declared once at the top
rather than bracketed hunk by hunk. It adds chunked GPU batch folding around
upstream's own per-record processing; everything that decides an **answer** still
comes from RNAlib. **Not part of any upstream proposal** — a driver is the
caller's business.

## What this changes about the merge story

`MERGING.md` records the blast radius as "nine files, +996, −7". That still holds
for *upstream's own code*; the marker comments add ~120 lines of explanation on
top, which is a price worth paying for being able to answer "what did you change
and why" with a command instead of a diff read.

## Verification

```
tools/list_local_patches.sh          # inventory + pairing + unmarked-file check
tools/list_local_patches.sh --check  # exit non-zero only
```

The markers are comments in eight C files and headers, so the real bar is that
they compile: **`make check` 152/152** with every marker in place. Two of them
were initially inserted *inside* an existing doc comment, which terminated it and
turned the rest of the block into code — caught by the build, fixed by moving the
marker above the comment it documents.
