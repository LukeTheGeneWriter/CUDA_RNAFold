# Merging the CUDA path back into ViennaRNA

*Working notes, written 2026-09-02 from `Continuous_Flow_Batching` at `5721a53`
plus the uncommitted phase C4 change. This is a map of what was changed and why,
aimed at whoever eventually prepares an upstream pull request — not a proposal
that upstream accept it as-is.*

---

## 1. Lineage: what this is a fork of

Three layers, and it matters which one a given line of code belongs to:

| Layer | What | Provenance |
|---|---|---|
| **0** | ViennaRNA **2.3.0** | upstream; `configure.ac` declares `2.3.0cuda` |
| **1** | A CUDA port of `mfe.c` and its inner loops | W. B. Langdon, 2017–2018 (see the `WBL` changelog atop `mfe_cuda.c`); arrived here as a bulk import, commits `66e91ad`/`a6027d4`, 2026-07-10 |
| **2** | This project | 63 commits, 2026-07-19 → present |

**The single largest merge obstacle is layer 0, not layers 1–2.** The base is
ViennaRNA 2.3.0; upstream is now on the 2.7 series. Everything below assumes a
merge is a *port* onto a current upstream tree, not a rebase.

---

## 2. Blast radius on upstream library code: three files

This is the good news, and it belongs early because the diff *looks* enormous. Of
the ~180 files in `src/ViennaRNA/`, this project's 63 commits touched exactly
**three** that are genuine upstream ViennaRNA sources — all three in one commit
(`021aa64`).

### `params.c` — an upstream bug, fixable upstream today

```c
-  params->id = ++id;
+  params->id = __sync_add_and_fetch(&id, 1);
```

`get_scaled_params()` did an unsynchronised read-modify-write on a file-scope
`static int id`. Any caller building fold compounds from more than one thread
races here. **This has nothing to do with CUDA** and is the one change in the
whole fork submittable upstream on its own, today, as a two-line diff. The
builtin was chosen over `<stdatomic.h>` deliberately so the fix does not touch
this shared file's build flags.

### `mfe.c` / `mfe.h` — `vrna_mfe_cpu()`, and why it exists

An 18-line addition exposing `vrna_mfe()`'s body under a second name. It is not a
feature; it is a **workaround for a symbol collision**, and it is the ugliest
thing in the fork from a reviewer's point of view. See §3.

Everything else this project changed lives in files that do not exist upstream at
all, plus `src/bin/RNAfold.c` and the build system.

---

## 3. The symbol collision — fix this before writing a PR

`mfe_cuda.c` is a **fork of upstream `mfe.c`** that defines its own `vrna_mfe()`.
Both objects link into the same binary; `mfe.c`'s copy still exists and is still
built, because the other tools (`RNAsubopt`, `RNAplfold`, …) need it. Which
`vrna_mfe()` a call site gets is decided by link order, and the CUDA one is
explicitly *not* safe for `VRNA_FC_TYPE_SINGLE` fold compounds — hence
`vrna_mfe_cpu()`, a differently-named door to the real implementation for the CPU
worker pool.

This works, and it is load-bearing, but no upstream maintainer will take it. A
merge needs the CUDA path to be **additive** — a new entry point
(`vrna_mfe_batch()` or similar) that never shadows an existing symbol — rather
than a shadowing redefinition. Doing that also deletes the need for
`vrna_mfe_cpu()` entirely, which removes two of the three upstream-file
modifications above.

**Treat this as prerequisite work, not merge-time cleanup.** It is the difference
between a reviewable PR and an unreviewable one.

---

## 4. New files

None of these exist upstream. This is where essentially all the work lives.

| File | Size | What it is |
|---|---|---|
| `src/ViennaRNA/mfe_cuda.c` | 1954 ln | Fork of `mfe.c`. Owns `par_mfe()` — the batch entry point — plus chunk setup, slot scheduling, backtracking, thread pools |
| `src/ViennaRNA/stub2.h` | 786 ln | The whole host↔device interface: ~45 functions, the `Indx`/`Hoff`/flatten helpers, the env-var accessors |
| `src/ViennaRNA/modular_decomposition.cu` | — | Multiloop decomposition sweep. **73% of wall time**, at its DRAM bandwidth floor |
| `src/ViennaRNA/int_loop.cu` | — | Interior loops. `int_loop_kernel_body.inc` is its templated body, instantiated at four block sizes |
| `src/ViennaRNA/hp_mb_loop.cu` | — | Hairpins and multiloop 3′ closure |
| `src/ViennaRNA/fill_arrays.c`, `fill_arrays_loop.c` | 319 + 486 ln | The row sweep driver, split out of `mfe.c`'s `fill_arrays()` |
| `src/bin/RNAfold_cpu_queue.{c,h}` | 155 + 49 ln | CPU worker pool for short/off-batch sequences (this project's, `021aa64`) |

`src/bin/RNAfold.c` is an upstream file but heavily modified: it owns the
streaming read loop, the VRAM-budgeted chunker, and dispatch between the GPU batch
and the CPU queue.

---

## 5. How a fold actually reaches the GPU

Upstream RNAfold folds one record at a time. This fork folds a **batch**, and the
batching model is what a reviewer most needs to understand.

1. **Stream and chunk** (`RNAfold.c`). Records are read one at a time and
   accumulated. A chunk closes when the next record would exceed the VRAM budget
   (`cudaMemGetInfo` × 0.85, capped by `RNA_GPU_VRAM_BUDGET_MB`). Cost comes from
   `gpu_bytes_per_file(length)`, whose dominant term is the `O(length²)` `my_c`
   triangle — so a record twice as long costs roughly four times as much, and no
   single representative length can stand in for a mixed chunk.
2. **Slots** (`par_mfe`). A chunk is dealt into *slots*: by default one per
   record; under `RNA_SLOT_FLOW=k`, `k` records share a slot and run its queue
   back to back. Records are dealt round-robin in descending length order, so a
   slot's capacity is its first occupant's length and every later occupant fits by
   construction — no allocator, nothing to defragment.
3. **Flattened layout.** Every device buffer is addressed through per-slot offset
   tables (`row_off_H`, `tri_off_H`, `size_off_H`, `seq_off_H`, `hc_off_H`), and
   every kernel recovers its own record index with `flatten_index_to_H()`. This is
   what allows mixed lengths in a single batch.
4. **The sweep.** One row per iteration for every active record at once. Records
   join at their own top row and retire at their own last one; a record that has
   not joined, or has retired, has width 0 and contributes no cells.
5. **Backtrack.** Triangles are fetched per record and backtracked on a CPU thread
   pool, overlapping with the GPU.

`Indx(i,j) = j*(j-1)/2 + i` does not depend on a record's length, so a short
record in an over-sized slot occupies exactly that slot's prefix. That one fact is
what makes slot recycling tractable at all.

---

## 6. Configuration surface: 22 environment variables

The second thing upstream will object to, and fairly. Everything is configured by
`getenv()`. A merge should route these through `vrna_md_t` or a dedicated options
struct; they are listed here so the mapping can be done deliberately rather than
discovered.

**Default ON** — an out-of-the-box run is the fast one, since `08081e6`:

| Variable | Default | Notes |
|---|---|---|
| `RNA_GPU_SWEEP` | 1 | GPU-resident sweep; `0` restores the host round trip |
| `RNA_CUDA_GRAPH` | 1 | Graph capture for the three sweep kernels |
| `RNA_MD_TILE` | 32 | Power of two in [1,32] |
| `RNA_BACKTRACK_THREADS` | auto | `min(work, hw_concurrency − cpu_queue)` |
| `RNA_BUILD_THREADS` | auto | Parallel fold-compound construction |
| `RNA_FML_SCAN_THREADS` | 256 | Measured optimal; all six widths within 0.9% |

**Default OFF, opt-in:** `RNA_CPU_THREADS`, `RNA_CPU_THRESHOLD` (200),
`RNA_GPU_VRAM_BUDGET_MB`, `RNA_CONTINUOUS_FLOW`, `RNA_SLOT_FLOW`.

**Diagnostic / self-check, not for production:** `RNA_ROW_VERIFY`, `RNA_HC_VERIFY`,
`RNA_SLOT_TURNOVER`, `RNA_SLOT_CAPACITY=max`.

**Auto-tuned block sizes, each with an override:** `RNA_INT_LOOP_BLOCK_SIZE`,
`RNA_MD_BLOCK_SIZE`, `RNA_HP_MB_BLOCK_SIZE`, `RNA_LOAD_MY_C_BLOCK_SIZE`,
`RNA_NEW_C_BLOCK_SIZE`, `RNA_FMLI_BLOCK_SIZE`, `RNA_FML_PREV_BLOCK_SIZE`. Chosen at
runtime via `cudaOccupancyMaxPotentialBlockSize` (and, for `int_loop`, a timed
microbenchmark), so they need no configuration on a new device.

---

## 7. Build integration

CUDA detection already lives in `m4/ac_rna_features.m4` (layer 1): `NVCC_PATH`,
`NVCC_HOST_CC`, `NVCC_GENCODE`, `NVCC_FLAGS`, `NVCC_SAMPLES`, with `CUDA_SMS` for
compute capabilities. The three `.cu` files have explicit rules in
`src/ViennaRNA/Makefile.am`; `mfe_cuda.c` is compiled with `-DSTUB`.

**Convention that must survive a merge: do not run `autoreconf`.** The generated
`Makefile.in`/`configure` are committed and hand-maintained here.

---

## 8. Scope limits — what the GPU path does not do

A reviewer will ask, and the honest answer is that this accelerates one specific
case rather than replacing the general one.

- **MFE only.** No partition function, no suboptimal folding, no comparative
  (`VRNA_FC_TYPE_COMPARATIVE`) folding on the GPU.
- **Single sequences, non-circular.** `fill_arrays_circ()` and the comparative
  paths stay host-side.
- **The CPU queue (v1) replicates plain MFE folding only.** It does not apply
  constraints, SHAPE data or ligand motifs, and does not replicate `outfile`'s
  per-record output selection — so RNAfold **disables the queue entirely** whenever
  a run uses any of those, falling back to GPU-only. A deliberate refusal rather
  than a silent semantics drop.
- **No Python bindings.** Users reach ViennaRNA through `import RNA`, not the CLI;
  until the CUDA path is reachable there, adoption of a merged version would be
  near zero. CPU-fallback bones exist; the binding does not.

---

## 9. Correctness: what is established, and what is not

The standing bar for every change is **byte-identical output** across: 6 reference
inputs; VRAM budgets 4/8/16/32 MB; `RNA_MD_TILE=1`; `RNA_CUDA_GRAPH=0/1`;
`RNA_GPU_SWEEP=0`; CPU queue 0/4/8 threads (compared as a sorted multiset, since
the queue reorders output); ascending and descending input order; a `-C` run; plus
`RNA_ROW_VERIFY` with a non-zero cell count — a per-record device-vs-host
comparison of every sweep cell.

Two caveats a merge must not paper over:

1. **The bar is self-comparison, not an upstream oracle.** It proves the GPU path
   agrees with *this codebase's* host path. That is the right tool for proving a
   change behaviour-neutral, and it is deliberately how these phases are gated —
   but it does not prove agreement with upstream ViennaRNA.
2. **One open discrepancy — narrowed 2026-09-03, still open.** Against
   ViennaRNA 2.6.4, one record in five at 5601 nt disagreed: `-1717.70` vs
   `-1718.40`, the other four matching to the cent.

   The "genuine 2.3.0-vs-2.6.4 model difference" hypothesis is now **ruled out**.
   Folding all six reference inputs under the 2.6.4 wheel reproduces this
   codebase's own reference energies on **239 / 239 records, lengths 20–8001 nt,
   zero differences**. The energy model is unchanged across versions (see §11),
   so whatever that one record was, it was not version drift. It must be
   reproduced on its own terms — same sequence, same conditions — before the
   port, so it cannot become unattributable in a restructured tree.

---

## 10. A staged merge plan

Ordered by increasing difficulty and decreasing independence:

1. **`params.c` data-race fix.** Two lines, no CUDA dependency, upstreamable now.
2. **Port onto current upstream.** 2.3.0 → 2.7.x. The bulk of the work; nothing
   else should start until it is done.
3. **Remove the `vrna_mfe()` shadowing** (§3). Make the CUDA path additive. Deletes
   `vrna_mfe_cpu()` and reduces the upstream-file diff to one file.
4. **Route configuration through `vrna_md_t`** instead of `getenv()` (§6), keeping
   the env vars as an optional debug override if upstream will have them.
5. **Land the batch entry point** as an opt-in build (`--with-cuda`), defaulting
   off, with the host path untouched when CUDA is absent.
6. **Python bindings**, without which a merged CUDA path is unreachable for most
   users (§8).

---

## 10b. The 2.7.2 target, measured — findings from 2026-09-03

The port target is **ViennaRNA 2.7.2**. A pristine tree was downloaded and
compared against this fork's 2.3.0 base. Full phased plan:
`~/.claude/plans/serene-dazzling-gem.md`. The load-bearing findings:

### The energy model did NOT change — this is the big de-risking fact

Compiled-in Turner-2004 tables differ only by three *new* G-quadruplex constants.
Default `vrna_md_t` values are numerically identical. Salt corrections are new
but gated off at the default (`VRNA_MODEL_DEFAULT_SALT = 1.021`); modified bases
never touch the MFE recursion. **Confirmed empirically: 239 / 239 reference
records identical under 2.6.4, lengths 20–8001 nt.**

So the byte-identical verification bar **survives the port** for default folds
and stays the primary instrument. Upstream additionally ships **204 gold
reference files** for RNAfold plus a unit-test suite — an external oracle this
project has never had.

### What genuinely changed and hits us

| Change | Impact |
|---|---|
| `hc->matrix` (triangular `char`, `jindx[j]+i`) → **`hc->mx` (dense row-major `unsigned char`, `n*i+j`)** | Hits the GPU bitmask derivation (`24b3ff0`) directly; the buffer changes size class. Highest-value finding. |
| `Indx(i,j) = j*(j-1)/2 + i` **unchanged** for `c`/`fML`/`fM1`/`f5` | The flatten-and-offset architecture survives intact. |
| `turn` band moved from loop bounds into hard-constraint values | The fork's skip stays correct and is now an optimisation upstream no longer performs. |
| `fill_arrays()` is `PRIVATE static`, called unconditionally by `vrna_mfe()` | **No seam exists** for an alternative engine — see below. |
| The `grammar/` API is **additive only** (`MIN2`-combined per `(i,j)`) | Rules out "just use the extension API". It cannot batch rows. |
| Upstream `RNAfold.c` gained a thread pool and **`vrna_ostream_t` ordered output** | `RNAfold_cpu_queue.c` is largely redundant; adopting `vrna_ostream_t` retires the sorted-multiset compromise from the bar. |
| **New unguarded race in 2.7.2**: `SPEEDUP_PARAMS` cache in `params/params.c` | Worse than the `++id` race this fork already fixed, and upstream's own threaded RNAfold hits it. A concrete bug to report. |
| `Makefile.am` 279 → 821 lines, ~24 convenience libraries; no upstream accelerator hook anywhere | CUDA build glue is a rewrite. |
| `src/ViennaRNA` 174 files/138k lines → 444 files/235k lines | Old top-level headers survive as deprecated shims — migrate off them. |

### This sharpens §3, it does not change it

The de-shadowing problem is now precisely characterised: there is **no engine
seam in 2.7.2 at all**. `vrna_mfe()` calls a private static `fill_arrays()`
unconditionally, and the grammar hooks fire per-cell inside that CPU loop. So
"make the CUDA path additive" is not a refactor we can do unilaterally — it needs
a new upstream-agreed extension point, plus a genuinely *batch* entry point,
since a GPU's advantage is entirely in batch width and no such API exists at any
level upstream. That is a design conversation with the maintainers, and it is now
the first thing to bring them.

---

## 11. What to port next

Everything above is about landing the MFE batch path. This section is the
opposite question: given that machinery exists, where else in ViennaRNA would a
GPU option pay? Ranked by *reuse of what is already built*, not by raw speedup.

### 11.1 Partition function (`part_func.c`) — the best target by a wide margin

`part_func.c:407` is:

```c
for (j = turn + 2; j <= n; j++) {
  for (i = j - turn - 1; i >= 1; i--) {
    ...
    if (hc_decompose) {
      qbt1 += vrna_exp_E_hp_loop(vc, i, j);
      qbt1 += vrna_exp_E_int_loop(vc, i, j);
      qbt1 += vrna_exp_E_mb_loop_fast(vc, i, j, qqm1);
    }
```

Compare `mfe.c:255`, which this fork already ported: the same triangular
dependency, the same three-way hairpin / interior / multibranch decomposition,
the same `hard_constraints[jindx[j] + i]` gate, the same `my_iindx`/`jindx`
indexing. The differences are `+=` of Boltzmann factors where MFE does `MIN2`,
and a float type where MFE uses `int`.

**Everything transfers**: the flattened per-slot offset tables, the VRAM-budgeted
chunker, GPU-derived hard-constraint bitmasks, the CUDA-graph row sweep, the slot
scheduler. This is the same kernel shape with a different reduction operator.

Two obstacles that are easy to miss and expensive to discover late:

- **The verification methodology does not transfer.** Every phase of this project
  is gated on *byte-identical* output, which works because integer `min` is exact
  and order-independent. A floating-point sum is neither — reassociating it across
  threads changes the low bits. A partition-function port needs a tolerance-based
  bar, and that is a materially weaker instrument than the one used everywhere in
  §9. Decide what "correct" means *before* writing the kernel.
- **`FLT_OR_DBL` defaults to `double`** (`data_structures.h:46`; `float` only under
  `USE_FLOAT_PF`). That is 8 bytes per triangle cell against MFE's 4, which halves
  batch width — and batch width is where the whole advantage comes from.
  `USE_FLOAT_PF` is the obvious lever, and it points the same direction as the
  int16 stream idea for the MFE path.

### 11.2 Base-pair probabilities / the outside algorithm

Same triangle, same indexing, runs after the inside pass. The natural follow-on
to 11.1 and not independent of it — it consumes the partition function's matrices.

### 11.3 Iterated-folding tools — the cheapest real win, and zero new kernels

`RNAinverse` and `RNApvmin` call the folder thousands of times over a population
of candidate sequences. That is *already* a batch of independent records, which is
exactly what `par_mfe()` exists to fold. No new CUDA is required — only routing
those tools through the batch path instead of a per-sequence loop.

Worth noting what this does and does not want: design candidates are
uniform-length, so continuous flow (phases A–C) buys nothing there — the
row-count prize is structurally zero when every length is equal. Batch width
buys everything. These tools are the strongest argument for the batch entry
point being a *library* API rather than something buried in `RNAfold.c`.

### 11.4 Boltzmann sampling (`boltzmann_sampling.c`)

Many independent stochastic tracebacks from one set of matrices. Embarrassingly
parallel, and on a **different axis** from anything built here — across samples
within one record, not across records. Depends on 11.1.

### 11.5 `RNAplfold` / `LPfold.c`

Sliding windows bounded by `winSize` (`LPfold.c:301`) make many small independent
subproblems. A good GPU fit in principle, but per-window work is low and the
window shape wants a different kernel from the one this fork has.

### Poor fits, recorded so nobody re-derives them

`RNAsubopt` (Wuchty enumeration — branchy, unbounded state, output-size-dependent
work) and the tree-edit distance tools. Not worth GPU effort.

---

## Appendix: where the performance comes from

Context for a reviewer asking whether the complexity is justified. Measured on
Colab L4, 400 × 5601 nt, against the pre-existing CUDA reference branch — which
already had CUDA graphs, GPU energy precompute and heterogeneous compute:

| Change | Effect |
|---|---|
| GPU-derived hard-constraint bitmasks (`24b3ff0`) | −193.0 s |
| GPU-resident sweep (`1834292` + `f923c16`) | −131.8 s; host-combine 109.1 → 0.0 |
| Deferred + threaded fold-compound build (`20acbb3`) | −67.0 s, **and** −2.2 GB RSS |
| Scratch matrix pool (`c0dbbda`) | −47.6 s, **and** RSS 18.5 → 7.8 GB |
| Backtrack thread pool | −40.7 s |

Net **1585 → 333.0 s (4.76×)** against that reference, GPU 93.4% busy — and, the
number worth leading with, **peak RSS 18.5 → 5.9 GB**, because host memory rather
than the GPU was what limited batch width, and batch width is where the advantage
comes from.

Against single-core CPU RNAfold the same workload is roughly **75–125× per
record**; against a saturated 12-core box, about **8×**. The 8× is the honest
number for a sceptical reviewer.
