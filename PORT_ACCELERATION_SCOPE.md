# Scoping: accelerating the options currently routed to the CPU

*Written 2026-09-05 against `port27`. Companion to `PORT_UPSTREAM_DIFF.md`: that
document says what is routed; this one says what each would cost to accelerate.*

Parity is already achieved — every option produces ViennaRNA 2.7.2's answer
today, because anything the device cannot reproduce is folded by the CPU path.
So nothing here is a correctness item. Each is purely "should this option go
fast too, and what would that cost".

The order below is by **measured or structural cost**, not by how important the
feature sounds. Two of the entries may cost nothing at all, and the first job is
to find out which.

---

## Tier 0 — may already work; needs TESTING, not building

The routing guard rejects these, and it is not obvious that it must. Two facts
make them suspect:

- **`noLP` is baked into `ptype`** by `vrna_ptypes()` (`sequences/alphabet.c:271`,
  `if (md->noLP && (!otype) && (!ntype)) type = 0;`), and the fork **uploads
  ptype**. Nothing in the kernels needs to know about `noLP` for it to take
  effect — the pairs simply are not there.
- **Hard-constraint bitmasks are packed from the host's `hc->mx`** whenever
  `g_hc_seq_derived == 0`, which is the default (`mfe_cuda.c:283`). `hc->mx` is
  upstream's, already carrying `noGUclosure`, `noGU` and any structure
  constraints. `hp_mb_loop.cu:146` says as much: "works under constraints
  as-is".

### MEASURED, 2026-09-05

Each model detail folded through `par_mfe()` in its **own process** and compared
against `vrna_mfe()`, 12 records × 160 nt:

| model detail | result | verdict |
|---|---|---|
| baseline (control) | 12/12 | — |
| temperature 25 | 12/12 | already accelerated anyway |
| **`noGU`** | **12/12** | **already works — drop the rejection** |
| **`uniq_ML`** | **12/12** | **already works — drop the rejection** |
| **`logML`** | **12/12** | **already works on this sample** (see caveat) |
| `noLP` | **9/12** | does NOT already work; needs real work |
| `noGUclosure` | refused by the sweep | needs the internal-loop kernel |
| `dangles` 0 / 1 / 3 | refused by the sweep | genuinely simplified away |

Three rejections can go for the price of a test. `uniq_ML` makes sense in
hindsight: it only causes `fM1` to be allocated, and MFE never reads it — it
matters for subopt and the partition function, not here.

**Caveat on `logML` and `noGU`:** 12 random sequences at 160 nt is a sample, not
a proof. Both should get a proper bar before the guard is relaxed — the
difference these options make is data-dependent, and a sample this size can miss
a rare divergence. `noLP` is the cautionary example: `ptype` encodes it
(`alphabet.c:271`) and it still fails 3 of 12, so "the data is uploaded" is not
the same as "the recursion honours it".

**`noClosingGU` is more interesting than a plain rejection.** The sweep's own
guard explains why: the hairpin/multibranch kernel applies it but the live
internal-loop kernel does not, so `c` comes out internally inconsistent and
backtracking fails. It is *half implemented*, and finishing it is a bounded job
in one kernel rather than a new feature.

That guard's message is also now **stale**: it says there is no CPU-queue
workaround because "the CPU path is stock 2.3.0 and itself disagrees with
ViennaRNA 2.7.2". On `port27` the CPU path *is* 2.7.2, so routing is exactly the
workaround it says does not exist. The message should be corrected.

---

## Tier 1 — cheap, well understood, designs already written

### Salt corrections
Kernels need **nothing**: the multibranch contribution is baked into
`MLbase`/`MLclosing`/`MLintern` at parameter-init time. Hairpin and internal each
need one added term, and the tables are 33 ints — a textbook `__constant__`
candidate, which would also be this codebase's first use of it.
Full design and frozen vectors: `PORT_SALT_SPEC.md`.
**Cost: ~50 device lines.**

### Circular RNA
No new arithmetic. `fM2_real` is exactly the ML_ML_ML decomposition the sweep
already computes and discards — verified cell-for-cell, 8853 cells, zero
mismatches. The cost is **memory**: one more dense triangular matrix per record,
so `gpu_bytes_per_file()` grows and the chunk width shrinks. Post-processing
stays on the host, on the existing backtrack pool.
Full design and frozen vectors: `PORT_CIRC_SPEC.md`.
**Cost: ~100 device lines plus a chunk width.**

### `uniq_ML`
Structurally identical to circular: it wants `fM1`, another triangular matrix
the recursion already has the pieces for. Same memory trade, no new arithmetic.
**Cost: like circular, minus the backtracking subtlety.**

---

## Tier 2 — real kernel work, bounded

### G-quadruplexes
**Not a recursion port.** `c_gq` is built once up front
(`dp_matrices.c:547`), independent of the DP; all 49 MFE-path uses are reads. So
`mfe_gquad.c`'s 1262 lines never move — the host builds the table, the device
reads it, `bt_gquad.c` backtracks on the host.

Measured sparsity decides the representation: fill is **0.63–4.30 % and falls
with n**, entries grow roughly linearly (18 156 at n = 2400 against 2.88 M
cells). Dense upload would spend a whole extra triangular matrix to carry ~1 %
useful data. **Upload sparse.**

The cost is the lookups: CSR `get` is a binary search, landing in
`modular_decomposition`'s innermost loops which are already at their DRAM floor.
Three mitigations to measure rather than choose on taste — per-row skip,
row-local dense expansion, length-dependent density.
Full design: `PORT_GQUAD_SPEC.md`.
**Cost: moderate device work, concentrated in the hottest kernel.**

### Dangle models 0, 1, 3
The kernels do not merely assume `dangles == 2`, they were **simplified under
that assumption** — `fill_arrays.c:317` and `fill_arrays_loop.c:325` record
whole energy branches deleted outright because d2 makes them unreachable, and
`hp_mb_loop.cu` carries the collapsed forms.

So this is not "add a branch": it is restoring the general form of the
hairpin/multibranch energy evaluation and then re-verifying that d2 still
produces byte-identical output. `d1` is the awkward one — upstream itself
carries special `fM_d3`/`fM_d5` arrays for it in the circular path.

Worth noting the trade honestly: restoring generality may cost the d2 fast path
some of its speed, which is the common case. A templated or duplicated kernel
avoids that at the cost of code size.
**Cost: substantial, and it touches the fast path everyone uses.**

### `logML`
Changes the multibranch contribution from linear to logarithmic — arithmetic in
the hot kernel, not structure. Small in lines, but every multibranch cell pays
it, and it needs its own numerical agreement check.
**Cost: small in code, needs careful verification.**

---

## Tier 3 — invasive; decide before starting

### Soft constraints, SHAPE/probing data, modified bases
These reach the recursion as **arbitrary host callbacks** (`vrna_sc_t.f`).
Function pointers cannot run in a kernel, so there are only two honest options:

1. keep routing them to the CPU — what happens today, and correct; or
2. precompute the shipped, JSON-parameterised cases into a per-`(i,j)` table the
   kernels read, accepting that a user-supplied callback still routes.

Option 2 covers modified bases (which ship as parameter sets) but not general
soft constraints. **This needs an explicit decision rather than an attempt.**

### Multistrand
Genuinely inside the recursion in 2.7.2: `pair_multi_strand()`, new `fms5`/`fms3`
matrices, per-nucleotide strand bookkeeping — 123 references in `mfe/mfe.c`
alone. New matrices, new decompositions, and the sweep's single-sequence layout
assumptions to revisit.
**Cost: a rewrite of the multibranch/exterior kernels, not an extension.**

### Comparative folding (RNAalifold-style)
A different recursion with an `n_seq` dimension throughout, 85 references in
`mfe/mfe.c`. The flatten-and-offset layout is built for one sequence per record.
**Cost: effectively a second backend.**

### Sliding window (`RNALfold`/`RNAplfold`)
Different matrix layout entirely (`fML_local`, `hc->matrix_local`) and a moving
window rather than a triangular sweep. Shares almost nothing with this code
path.
**Cost: a separate project.**

---

---

## A defect found while scoping: `par_mfe()` caches energy parameters

Not a feature item, but it belongs here because it was found doing this work and
it affects anything that folds more than one batch.

`par_mfe()` carries **energy-parameter state across calls**, and
`teardown_gpu/2/3()` does not reset it:

```
temperature 25, as the first and only par_mfe() call   12/12 match
temperature 25, after one batch at 37 C                 0/12 match
```

The second batch is folded with the first batch's parameters. Silently — no
error, no assert, plausible structures throughout.

**Invisible in RNAfold today**, because model details are constant for a whole
run: every chunk uses the same `md`. It becomes reachable the moment anything
folds two batches with different models in one process, which is exactly what a
`vrna_mfe_batch()` caller or a Python binding would do. It also means every
measurement in this document had to be taken one model per process; a probe that
looped over models in one process measured the caching, not the models, and
initially reported `temperature 25` as broken and `noLP` as worse than it is.

Fix before `vrna_mfe_batch()` is offered to anyone: upload the parameter tables
per call, or key the cached upload on the model details and refresh when they
change.

---

## Suggested order

0. **Fix the parameter caching in `par_mfe()`** before anything else. It is a
   silent-wrong-answer bug on a path we are about to advertise, and it makes
   every other measurement in this area unreliable until it is gone.
1. **Bar and relax the three Tier 0 rejections** — `noGU`, `uniq_ML`, `logML`
   already match; give each a proper test set and drop it from the guard. Then
   test structure constraints (`-C`), which use the same `hc->mx` upload path
   and were never measured here.
2. **Salt**, then **`uniq_ML`**, then **circular** — all three are memory or
   parameter work with no new hot-path arithmetic, and two have frozen bars.
3. **G-quadruplexes** — the first genuine kernel work, and the one with a real
   performance question attached.
4. Everything in Tier 3 only with a decision made first.

Throughout: each feature keeps its CPU route as the fallback, so an
acceleration that is wrong is caught by comparing against the route it replaces
— the same self-comparison that has caught every error in this port.
