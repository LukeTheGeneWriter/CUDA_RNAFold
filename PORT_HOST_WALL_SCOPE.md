# `build` and `output` — 38 % of wall, and it is the same work twice

*Scoped 2026-09-09 against `0a198d57`. Measured on an RTX 3050 with
`tools/build_split.c` and a poisoning probe; the shares come from the 400 × 5601
stress runs (`stress272.json`, `stress272_t4.json`).*

With `gpuinit` closed, these are the two largest items left:

| | L4 | T4 | share |
|---|---|---|---|
| `build` | 121.96 s | 121.9 s | ~19 % |
| `output` | 124.75 s | 120.5 s | ~19 % |

**They are within 3 % of each other in all twelve arms across two machines,
because they are the same call.**

---

## 1. `build` — `vrna_fold_compound()`, and what is in it

`tools/build_split.c`, n = 5601, 5 reps:

| | s / record | share |
|---|---|---|
| `vrna_fold_compound(DEFAULT)` | **0.144** | 100 % |
| ... of which `vrna_ptypes()` | 0.038 | 26 % |
| ... the rest, dominated by `vrna_hc_init()` | ~0.105 | ~73 % |
| `vrna_fold_compound_prepare(MFE)` | 0.000 | — |
| `vrna_fold_compound_free()` | 0.001 | — |

Both halves are **O(n²) per record**: `default_hc_bp()` calls
`default_pair_constraint()` for every (i, j) and writes **both** triangles of a
dense `(n+1)²` byte matrix — 31.4 MB per record at 5601 nt — and `vrna_ptypes()`
walks the triangle again.

**`prepare` adds nothing**, which corrects a natural guess: `ptype` is already
built by `vrna_fold_compound(DEFAULT)`, not by the later
`vrna_fold_compound_prepare(fc, VRNA_OPTION_MFE)`. The probe prints what is
allocated rather than inferring it, after an earlier version of it inferred wrong
twice and announced the second error as a **negative** percentage.

### The GPU already computes this

`pack_hc_kernel` derives exactly these predicates — `rnafold_hc_cell()` is
`default_pair_constraint()` and `rnafold_ptype()` is `vrna_ptypes()` — for the
**whole batch in ~1.4 s** (§13). The host then spends ~122 s computing the same
information again into upstream's layout, **solely so upstream's backtrack can
index it**.

### Threading it is blocked by a defect we have already reported

The loop is embarrassingly parallel across records, and the 2.3.0 branch had
`RNA_BUILD_THREADS` for it (never ported). But `vrna_fold_compound()` reaches
`vrna_params()`, which is **Defect B** in `PORT_UPSTREAM_PROPOSAL.md`: the
`SPEEDUP_PARAMS` cache at `params.c:100-106` is four file-scope statics with no
lock, no `omp critical` and no atomic, read and written on every call.

**This upgrades Defect B from a theoretical race to a measured blocker on ~19 %
of wall**, which is a materially stronger thing to send upstream than what §Defect
B currently says. It should be re-written with this number in it.

*(A narrow escape hatch exists and is NOT recommended without more thought: every
record in a batch is built from the same `opt->md`, so all threads would write
identical bytes to the cache. Identical-value races are still races, and this
project does not ship "benign in practice".)*

---

## 2. `output` — a SECOND `vrna_fold_compound()` per record

`process_record()` (`RNAfold.c:1602`) calls

```c
vc = vrna_fold_compound(rec_sequence, &(opt->md), VRNA_OPTION_DEFAULT);
```

— the identical call the chunk loop already made and freed. That is the whole of
`output`.

**Measured by poisoning it.** Setting `vc = NULL` on the accelerated path
(prefolded, `--noPS`, no `-p`/`--MEA`/constraints/SHAPE/motifs/commands/
mod-bases/`--ImFeelingLucky`/`--benchmark`) and rebuilding:

| | `output` |
|---|---|
| baseline | 0.130 s |
| compound not built | **0.000 s** |

**Nothing dereferenced it** — no crash. The stage *is* the constructor.

The run printed nothing, though, because `process_record()` already has an
`if (!vc) { warn; return; }` guard immediately after. That is not a
counter-example; it is the shape the real fix has to take.

### Why it is genuinely dead on that path

With `record->prefolded` set, `vrna_mfe(vc, ...)` is **skipped** — the structure
and energy come from the batch. The remaining consumers of `vc` are all gated
off: `postscript_layout` by `--noPS`, everything from `vrna_exp_params_rescale`
onward by `-p`, plus MEA / ref-structure / ligand-motif / unstructured-domain
blocks. What is left is `vc->length`, and `vrna_mx_mfe_free(vc)` freeing matrices
that were never allocated.

### The fix, and its one hazard

A prefolded fast path that formats and prints without constructing a compound.
It needs `length`, and **that is the hazard**: the step-2b notes record
`strlen(seq) != vc->length` when the input carries whitespace. For the prefolded
path specifically it is safe — the chunk loop sizes `Str[i]` as
`strlen(chunk[i]->sequence) + 1` and hands that back as `prefolded_structure`, so
the two already agree by construction — but that argument has to be **asserted in
the code**, not left in a document.

`RNAfold.c` is an upstream file, so this wants to be an **additive early-out
block**, not a refactor of `process_record()` into lazy accessors. The bar is a
byte-identical output comparison with and without the fast path, over mixed
lengths, with and without each gating option.

---

## 3. What this is worth, and in what order

| | share | route | risk |
|---|---|---|---|
| **`output`** | ~19 % | delete a duplicate call | **low** — provably unused on the fast path, and the bar is byte-identical output |
| `build` | ~19 % | thread it | **blocked** on upstream Defect B |
| `build` | ~19 % | reuse the GPU's derivation | medium — needs `hc->mx` in upstream's dense layout D2H'd, ~31 MB/record |

**Do `output` first.** It is the only ~19 % in this project available by
*removing* work rather than adding a mechanism, and unlike `build` it needs
nothing from upstream.

Then re-send Defect B with the measurement attached. `build` stays blocked until
either upstream fixes the cache or we take the D2H route — and the D2H route only
makes sense after `output` is gone, because the two are the same 122 s and fixing
the constructor twice is wasted effort.
