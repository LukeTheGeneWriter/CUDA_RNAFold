# Port Phase 1 — what to put to the ViennaRNA maintainers

*Written 2026-09-05 against pristine ViennaRNA 2.7.2 — the release tarball at
`~/vrna27/ViennaRNA-2.7.2` (`make check` 126/126, per `PORT_PHASE0.md`) for
Part 1, and a clone of upstream's git at the `v2.7.2` tag for Part 1b. Those are
the same source: the tag is upstream `master` exactly (`1ffec79`). Every line
number below is from that tree. Every claim about behaviour was **run**, not
read: the probes are frozen in `tests/upstream/` and reproduce in about a minute
with `tests/upstream/run_probes.sh`.*

> **STATUS 2026-09-06 — the seam in Part 3 is no longer a proposal. It is built,
> it runs, and it has been measured.** RNAfold folds on the GPU on ViennaRNA
> 2.7.2 through exactly the two additions described below, byte-identical to
> upstream across a 239-record reference set, with `make check` at 142/142. The
> diff against `v2.7.2` is **12 upstream files modified, 59 added, and 6 lines
> deleted** — two in `mfe/mfe.c` and four list continuations in a `Makefile.am`.
> Measured on an NVIDIA L4 against upstream 2.7.2 itself: **8.06x** at
> 120 x 2000 nt, **5.65x** at 60 x 5601 nt, **7.88x** on a shuffled mixed-length
> workload. Section 3.3 is new, and is the one thing we have found that the seam
> does not by itself make possible.

This is the phase that decides whether the CUDA path is upstreamable. It is a
conversation with Lorenz, Stadler and Langdon — the defects in Part 1 are
submittable today, and Part 3 is now a description of working code rather than
a sketch.

Three things go in the message, in this order:

1. **Five defects in 2.7.2**, each with a reproducer and a measured effect —
   three in the library (Part 1) and two that stop a build from git (Part 1b).
   They are independent of CUDA, submittable today, and they establish that we
   have read the tree carefully before asking for anything.
2. **The problem we cannot solve without them**: there is no seam in `vrna_mfe()`
   for an alternative engine, and the grammar API cannot serve as one.
3. **A concrete, minimal proposal** for that seam plus a batch entry point,
   shaped to be useful to upstream on CPU even if no GPU backend ever exists.

The two build defects are last in severity and first in contact: they are what
the next person who clones the repository will hit, before reading a line of
the library.

---

# Part 1 — Three library defects in 2.7.2, with reproducers

## Defect A — auxiliary grammar rules are called with the wrong `i`

**`src/ViennaRNA/mfe/mfe.c:500-504`**, inside `fill_arrays()`:

```c
      if (fc->aux_grammar)
        /* call auxiliary grammar rules */
        for (size_t i = 0; i < vrna_array_size(fc->aux_grammar->aux); i++)
          if (fc->aux_grammar->aux[i].cb)
            (void)fc->aux_grammar->aux[i].cb(fc, i, j, fc->aux_grammar->aux[i].data);
```

The `size_t i` of the inner loop **shadows** the outer row index `i` of
`fill_arrays()`. The callback therefore receives the *rule index* where the
documented contract (`grammar/mfe.h:37`) says it receives *"the 5' delimiter of
the sequence segment"*. With one rule registered, every call sees `i == 0`; with
two, rule 0 always sees `0` and rule 1 always sees `1`.

The partition-function twin does it correctly, which settles what was intended —
**`partfunc/partfunc.c:410`** names the rule index `c` and passes the real `i`:

```c
        for (size_t c = 0; c < vrna_array_size(fc->aux_grammar->exp_aux); c++) {
          if (fc->aux_grammar->exp_aux[c].cb)
            (void)fc->aux_grammar->exp_aux[c].cb(fc, i, j, fc->aux_grammar->exp_aux[c].data);
```

**Measured** (`tests/upstream/aux_index_probe.c`) — two rules registered on a
30 nt sequence, both returning `INF` so the MFE cannot move:

```
rule 0: 435 calls, i in [0, 0], j in [2, 30]
rule 1: 435 calls, i in [1, 1], j in [2, 30]
VERDICT: BUG CONFIRMED -- callbacks receive the RULE INDEX as i
```

435 = 30·29/2, so the *call count* is right and only the argument is wrong —
which is exactly why this is invisible to the test suite: nothing crashes, and
any rule whose contribution does not depend on `i` still gives the right answer.

**Fix:** rename the inner variable, as the PF path already does. One line.

**Blast radius:** every `vrna_gr_add_aux()` rule in the MFE recursion. The five
typed registrations (`_f`, `_c`, `_m`, `_m1`, `_m2`) are unaffected — they are
called from the loop-energy functions with the real indices; only the generic
`aux` bucket is wrong.

## Defect B — the `SPEEDUP_PARAMS` cache is an unsynchronised data race

**`src/ViennaRNA/params/params.c`.** `SPEEDUP_PARAMS` is `#define`d to `1` at
**line 23** unconditionally — there is no configure switch and no way to opt out
short of editing the source. It introduces four file-scope statics
(**lines 100-106**) — `p_pre`, `p_pre_init`, `pf_pre`, `pf_pre_init` — with **no
lock, no `omp critical`, and no atomic anywhere in the file**. The
`#pragma omp threadprivate` at **line 111** covers only `id` and `pf_id`, *not*
the cache.

`vrna_params()` (**lines 144-178**) both reads and writes it:

```c
  if (p_pre_init) {
    p_pre.model_details.window_size   = md_p->window_size;   /* 158: shared WRITE */
    p_pre.model_details.min_loop_size = md_p->min_loop_size; /* 159: shared WRITE */
    p_pre.model_details.max_bp_span   = md_p->max_bp_span;   /* 160: shared WRITE */

    if (memcmp(md_p, &(p_pre.model_details), sizeof(vrna_md_t)) == 0)
      return vrna_params_copy(&p_pre);                       /* 163: shared READ  */
  }

  cp = get_scaled_params(md_p);
  memcpy(&p_pre, cp, sizeof(vrna_param_t));                  /* 170: shared WRITE */
```

`vrna_exp_params()` (**lines 181-**) repeats the pattern for the partition
function, so both prediction paths are affected. `vrna_fold_compound()` reaches
this through `vrna_params()` (`fold_compound.c:622`), so any program that builds
fold compounds from more than one thread is on this path.

### What it actually costs — measured, and narrower than it first looks

Two scenarios, `tests/upstream/params_race_probe.c`, 8 threads × 20 000
`vrna_params()` calls, every returned table compared byte-for-byte against a
single-threaded reference (skipping only the `id` field, which is meant to vary):

| scenario | tables checked | wrong |
|---|---|---|
| threads request **different** model details (T = 20/30/37/50 °C) | 160 000 | **25 140 (15.7 %)** |
| threads request **identical** model details | 160 000 | **0 (0.0 %)** |

**The second row is the one that keeps this report honest.** Upstream's own
`RNAfold -j` gives every record the same model details, so it reaches the racy
code on every record and is nonetheless *not* observably wrong: once the cache
holds the right table, every racing `memcpy` writes identical bytes over
identical bytes. An earlier note of ours claimed threaded `RNAfold` produces
silently wrong energies; that overstates it, and the corrected statement is:
**the race is real and it corrupts results only when concurrent callers differ
in their model details.** It is still undefined behaviour in every case.

Who differs in their model details, in practice: a temperature or salt sweep
run in threads, an application folding with several parameter sets at once, and
anything driving RNAlib from Python threads — all supported uses of a library
that documents itself as thread-safe when built accordingly.

### Defect C — the cache-hit path corrupts window settings *at the same energy model*

This is the same code and a distinct, much likelier failure. Lines 158-160 write
the caller's `window_size` / `min_loop_size` / `max_bp_span` **into the shared
static** before comparing, precisely so those three fields are ignored by the
`memcmp`. Single-threaded that is correct and returns the caller's own values.
Concurrently, threads overwrite each other between the write and the
`vrna_params_copy()` two lines later.

`tests/upstream/params_window_probe.c` — 8 threads, same energy model, each
asking for its own `window_size` and `max_bp_span`:

| threads | parameter sets | carrying another thread's window |
|---|---|---|
| 8 | 160 000 | **155 894 (97.4 %)** |
| 1 (control) | 20 000 | **0 (0.0 %)** |

97 % under contention, 0 % single-threaded: purely a concurrency defect, and it
bites callers who all want the *same* energy model — the RNALfold / RNAplfold
shape, or any process mixing windowed and global folding.

### Fix options, in the order we would suggest them

1. Make the cache `#pragma omp threadprivate` like `id` — matches the existing
   style, but only helps OpenMP builds and multiplies the cache by thread count.
2. Guard the whole read-compare-copy with a lock. Correct everywhere, costs the
   thing the cache was added to save.
3. Drop the three-field write entirely: compare `md_p` against the cached
   `model_details` *field by field, skipping* those three, instead of mutating
   shared state to make `memcmp` succeed. This removes Defect C outright and
   makes the remaining race benign-but-still-UB.
4. Add a configure switch so the cache can be turned off. Useful regardless,
   since today there is none.

Related, and already noted in `MERGING.md`: `params->id = ++id` is now
`threadprivate`, which fixes the race under OpenMP but silently drops global
uniqueness of the id — two threads can hand out the same id. Our fork uses
`__sync_add_and_fetch(&id, 1)`, which keeps uniqueness and needs no OpenMP.

---

# Part 1b — Two defects that stop a build from git

*Found 2026-09-05 while standing up a build of `master` from the repository
rather than from a release tarball, which is what this project now develops
against. Both are one-line fixes and neither affects a tarball build at all.*

Building from git needs a larger toolchain than building the tarball, which is
expected and fine: `autoconf`/`automake`/`libtool` for `autogen.sh`, `gengetopt`
for the 25 `.ggo` files whose generated `src/bin/*_cmdl.[ch]` are not tracked,
`makeinfo` for `man/RNAlib.info`, and the `src/dlib-*.tar.bz2` /
`src/libsvm-*.tar.gz` bundles unpacked into `src/` as `configure` instructs.

The two below are different: they are not real dependencies, they are places
where a rule survives into a configuration that cannot satisfy it. Each one
stops `make` dead, and each is invisible from a tarball because the tarball
ships the artefact the rule would have produced.

## Build defect 1 — a git build requires doxygen even with the manual disabled

`doc/doxygen/Makefile.am` puts the consumer of the XML **outside** the guard
that owns the only rule able to produce it:

```make
 6  if WITH_REFERENCE_MANUAL
 8    REFERENCE_MANUAL_FILES_XML = xml/*
14    if WITH_REFERENCE_MANUAL_BUILD
24      $(REFERENCE_MANUAL_FILES_XML): doxygen-xml     <- the only rule
26    endif WITH_REFERENCE_MANUAL_BUILD
28  endif WITH_REFERENCE_MANUAL
30  noinst_DATA = $(REFERENCE_MANUAL_FILES_XML)         <- outside both
```

`WITH_REFERENCE_MANUAL_BUILD` is `test "x$doxygen" != xno`
(`m4/ac_rna_refman.m4:215`). So with doxygen absent, `noinst_DATA` still demands
`xml/*` and nothing can build it:

```
make[2]: *** No rule to make target 'xml/*', needed by 'all-am'.  Stop.
```

**`--without-doc` does not avoid it.** Configuring from a pristine tree with
doxygen forced absent (`ac_cv_path_doxygen=no`) and reading the two conditionals
back out of `config.status`:

| configure flags | `WITH_REFERENCE_MANUAL` | `..._BUILD` | result |
|---|---|---|---|
| *(none)* | **true** | false | **broken** |
| `--without-doc` | **true** | false | **broken** |
| `--without-doc-pdf --without-doc-html` | **true** | false | **broken** |
| all three together | false | false | ok |

So there *is* a workaround — disable all three — but the flag that reads as if
it should be sufficient is not, and the failure it produces names neither doc
nor doxygen. We did not trace *why* `--without-doc` alone leaves the conditional
true; the four rows above are reproducible and the diagnosis is yours to make.
(For the last row we confirmed the conditional clears, not a full build.)

**Fix:** move `noinst_DATA` inside `if WITH_REFERENCE_MANUAL_BUILD`, so the
files are demanded only where they can be produced.

## Build defect 2 — `--without-python` breaks the man page build

`doc/source/man/Makefile.am:38`:

```make
	$(man2rst_verbose)$(PYTHON3) ../../man2rst.py -i $< -o $@
```

With `--without-python`, `$(PYTHON3)` expands to **nothing**, so `make` tries to
execute `man2rst.py` directly. It is tracked mode `100644`, so every one of the
25 pages fails identically:

```
/bin/bash: line 1: ../../man2rst.py: Permission denied
make[3]: *** [Makefile:743: RNA2Dfold.rst] Error 126
```

Invisible from a tarball, which ships the generated `.rst` files.

**Fix:** either fall back to a plain `python3` when the bindings are disabled,
or give `man2rst.py` a shebang and mode `755` so an empty `$(PYTHON3)` is
harmless. We checked whether this is a pattern rather than a one-off: of 105
tracked non-executable scripts, `man2rst.py` is the only one a `Makefile.am`
invokes this way.

Our own workaround needs no upstream change — pass `PYTHON3=$(command -v python3)`
to `configure` — so this is reported rather than worked around in our tree.

---

# Part 2 — The problem we cannot solve on our own

**`vrna_mfe()` has no seam.** `src/ViennaRNA/mfe/mfe.c:248`:

```c
    energy = fill_arrays(fc, ms_dat);
```

`fill_arrays()` is `PRIVATE` (**declared at mfe.c:99, defined at mfe.c:420**) and
the call is unconditional. There is no hook, no dispatch, no build-time
alternative. To supply a different MFE engine today, the only mechanism available
is to **define a second `vrna_mfe()` in another translation unit and let link
order decide** — which is exactly what our fork does, and exactly what we want to
delete. It is also why the fork carries a `vrna_mfe_cpu()`: a differently-named
door back to the real implementation for the CPU worker threads.

**The grammar API is not a substitute**, and we checked this before asking.
Auxiliary rules are combined *into* the existing recursion per `(i, j)` inside
`fill_arrays()` (mfe.c:500-504) — they are additive extensions evaluated cell by
cell, in upstream's own loop order. They cannot replace the recursion, and they
cannot batch: a GPU's entire advantage is folding many sequences concurrently,
and a per-cell callback is the worst possible granularity for that.

**The good news, and it is what makes a small seam sufficient:** `fill_arrays()`
writes **only** `fc->matrices` (`f5`, `c`, `fML`, `fM1`, `fM2_real`, taken from
`fc->matrices` at mfe.c:438-442), plus locals it frees before returning, and it
returns `f5[length]` (mfe.c:520). `backtrack()` is a separate function whose only
data source is the same `fc`. So an alternative engine that fills those matrices
and returns `f5[length]` is a **drop-in at one line**, and upstream's
backtracking, circular post-processing, constraint handling and output path stay
untouched and stay authoritative.

That is exactly what our fork's GPU path produces — and we can already prove it
cell-for-cell: `RNA_ROW_VERIFY` compares every swept device cell against the host
recursion, and the byte-identical bar (239 records, 20-8001 nt) is reproduced by
pristine 2.7.2 as of Phase 0.

## The second gap, found by building the first — `postprocess_circular()`

Implementing the seam exposed a companion problem we did not anticipate, and it
is the same shape as the first.

An engine that replaces the matrix fill must afterwards run the steps that
*consume* those matrices. Upstream has already made most of them reachable:

| post-fill step | status in 2.7.2 |
|---|---|
| `vrna_mfe_exterior_f5()` | **public** — `mfe/exterior.h` |
| `vrna_backtrack_from_intervals()` | **public** — `backtrack/global.h` |
| `vrna_mfe_multibranch_m2_fast()` | **public** — `mfe/multibranch.h` |
| `vrna_mfe_multibranch_m1()` | **public** — `mfe/multibranch.h` |
| `postprocess_circular()` | **`PRIVATE`, no header** — `mfe/mfe.c:103` |

Four of the five are public. The fifth is the one that turns filled matrices
into a circular energy and backtrack, and it cannot be called from outside
`mfe.c`.

This does not affect correctness for us: circular folds are declined by our
routing guard and handled by upstream's own per-record path, so `RNAfold -c`
gives upstream's answer today. It is purely a question of whether circular RNA
can ever be *accelerated* by anyone using the seam — and the answer is currently
no, for a reason unrelated to the recursion.

We think this is a gap in upstream's own pattern rather than a request specific
to us: it follows from the same premise as §3.1. If a caller may replace the
inside fill, the steps that consume the result have to be callable.

---

# Part 3 — What we propose

Two additions. Neither changes any default, and both are useful to upstream
without a GPU in the room.

## 3.1 An opt-in inside-engine seam, per fold compound

Registered like the existing extensions — no global state, no new struct in a
public ABI-sensitive position, and it rides the `aux_grammar` machinery that
already carries per-`fc` callbacks with `prepare` / `release` lifetimes:

```c
/* returns 0 if an engine is already registered on this fc */
unsigned int
vrna_gr_set_inside_engine(vrna_fold_compound_t   *fc,
                          vrna_gr_engine_f        cb,
                          void                   *data,
                          vrna_auxdata_prepare_f  prepare,
                          vrna_auxdata_free_f     release);

/* fills fc->matrices and returns f5[length], or VRNA_ENGINE_DECLINED */
typedef int (*vrna_gr_engine_f)(vrna_fold_compound_t *fc, void *data);
```

and `mfe.c:248` becomes:

```c
    energy = INF;

    if ((fc->aux_grammar) && (fc->aux_grammar->engine))
      energy = fc->aux_grammar->engine(fc, fc->aux_grammar->engine_data);

    if (energy == VRNA_ENGINE_DECLINED)
      energy = fill_arrays(fc, ms_dat);
```

**The `DECLINED` return is the load-bearing part of the design, not a
convenience.** A backend must be able to look at an `fc` and hand it back —
soft constraints it cannot evaluate, G-quadruplexes, multiple strands, unusual
dangle models — and get upstream's own answer, rather than silently computing a
different one. Our fork already lives by this rule (constraints, SHAPE and
ligand motifs disable the GPU path today, and `-c`, `-g` and `--noClosingGU` now
refuse outright rather than answer wrongly), and making it part of the *interface*
means no backend can skip it.

Cost to upstream: one struct field, one setter, five lines in `vrna_mfe()`, and
no change to any existing behaviour when no engine is registered.

## 3.2 A batch entry point

```c
int
vrna_mfe_batch(vrna_fold_compound_t **fcs,
               size_t                 n,
               char                 **structures,   /* may be NULL */
               float                 *mfes);
```

with the **default implementation being a plain loop over `vrna_mfe()`**, so the
API is complete, testable and useful the day it lands, with no backend at all.

This is the API that does not exist upstream at any level, and its absence is
the real obstacle: a GPU's advantage is entirely in batch width, and there is
currently no way to express "here are 400 sequences" to RNAlib. We would pair it
with a batch-capable engine hook so a backend can fill *N* compounds' matrices in
one sweep, after which each record is backtracked independently — which
parallelises with the thread pool `RNAfold.c` already has.

It is worth saying that this is not only a GPU concern. A batch entry point is
the natural home for upstream's own thread pool and for `vrna_ostream_t` ordered
output, both of which currently live in the `RNAfold` binary rather than in the
library, and both of which every other caller has to reimplement.

## 3.3 Expose the circular post-processing

The minimal form, mirroring the four functions that are already public:

```c
/* mfe/global.h or a circular-specific header */
int vrna_mfe_circular_postprocess(vrna_fold_compound_t *fc,
                                  vrna_bts_t            bt_stack);
```

i.e. `postprocess_circular()` (`mfe/mfe.c:598`) given external linkage and a
declaration, with no change to its body or to any call site. `vrna_mfe()` keeps
calling it exactly as it does now (`mfe.c:323`).

We have deliberately **not** written a copy of it in our fork. It is roughly 150
lines of upstream logic that would drift on the next release, and duplicating it
would contradict the principle the rest of this port is built on: anything the
device cannot reproduce is folded by upstream's own code, so upstream stays the
authority on what the answer is.

## 3.4 The question we would like decided

**Does the CUDA backend live in-tree behind `--with-cuda`, or as a separate
linkable package that registers itself through the seam above?**

We can build either, and the seam is the same. What differs is who carries the
`nvcc` build rules and the CI cost. Our reading is that the seam is worth having
upstream *regardless* of that answer, which is why it is proposed on its own.

---

# Status and what happens next

- **Part 3's seam is built and measured** (see the banner at the top). The two
  additions are `vrna_gr_set_inside_engine()` on the existing `aux_grammar`, and
  `vrna_mfe_batch()` / `vrna_mfe_batch_backend_set()`. The batch function's
  no-backend implementation is literally a loop over `vrna_mfe()`, so it is
  useful with no accelerator present, and **the driver contains exactly one
  CUDA-specific line** — the registration call.
- **§3.3 is the one open ask.** It blocks accelerating circular RNA and nothing
  else; circular folding is correct today via routing.
- Part 1's three library defects and Part 1b's two build defects are ready to
  send today; they do not depend on anything else in this project.
- Part 2 and 3 are the design conversation. Phase 2 (the mechanical rebase of
  version-independent infrastructure onto 2.7.2) does **not** block on the
  answer — the offset tables, chunker, sweep driver and scratch pool are
  version-independent, and the salt work argues for pulling that rebase forward
  (`PORT_SALT_SPEC.md`).
- What *does* block on the answer is de-shadowing `vrna_mfe()`, which is the
  difference between a reviewable pull request and an unreviewable one
  (`MERGING.md` §3).

## Reproducing everything here

```sh
tests/upstream/run_probes.sh [path-to-ViennaRNA-2.7.2] [path-to-gcc]
```

Builds five binaries against the pristine tree's `libRNA.a` and prints the five
verdicts above, including the two controls. The probes use only the public
RNAlib API and contain no CUDA_RNAFold code, so they can be attached to an
upstream issue as they are.

The two build defects of Part 1b need no probe — they reproduce from a clone:

```sh
git clone https://github.com/ViennaRNA/ViennaRNA.git && cd ViennaRNA
tar -xjf src/dlib-*.tar.bz2 -C src/ && tar -xzf src/libsvm-*.tar.gz -C src/
./autogen.sh
./configure --without-doc --without-python   # plus whatever else you exclude
make
```

(with `gengetopt` and `makeinfo` installed, or it stops earlier on those — those
two are genuine build-from-git requirements rather than defects). That sequence
stops at `No rule to make target 'xml/*'` when doxygen is absent (defect 1), and
at `../../man2rst.py: Permission denied` once past it (defect 2). Our own standup
of that build, carrying the workaround for each, is `tools/standup_git_build.sh`
on our `port27` branch.
