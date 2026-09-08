# Circular RNA (`-c`) on the GPU path — design and frozen bar

*Written 2026-09-05 against ViennaRNA 2.7.2. Same shape as `PORT_SALT_SPEC.md`:
establish what the kernels actually need, cost it honestly, and freeze a
reference bar now so the implementation is mechanical when the rebase lands.*

**Current status:** `-c` is **guarded and refuses** on the GPU path as of
`a6bcc3b`. Before that it was accepted and silently returned the *linear*
answer (`PORT_FEATURE_AUDIT.md`) — the worst failure mode there is.

---

## The headline: the hot kernel already computes the missing matrix

Circular folding needs one thing from the matrix fill that linear folding does
not: **`fM2_real`**, a full triangular matrix. `md->circ` sets `ALLOC_CIRC`
(`datastructures/dp_matrices.c:579-580`), which allocates it at `size` ints
(`:1326`) — and, notably, *suppresses* `fM1` (`:1323-1324`), so circular folding
does not need `fM1` at all.

`fill_arrays()` fills it per cell (`mfe/mfe.c:500-504`) with
`vrna_mfe_multibranch_m2_fast()`, whose body (`mfe/mfe_multibranch.c:1143`) is
the ML_ML_ML modular decomposition:

```
fM2_real[i,j]  =  min over k of ( fML[i,k] + fML[k+1,j] )
```

That is precisely what `get_aux_arrays()` documents `DMLi[j]` as holding while
`fML` itself is being computed — so it is a value the multibranch sweep already
produces and currently throws away.

**Measured, not argued** (`tests/upstream/circ_fm2_probe.c`): fold with
`md.circ = 1`, then compare `fM2_real[i,j]` against `min_k(fML[i,k]+fML[k+1,j])`
recomputed from the filled `fML`, for every cell:

| sequence | cells checked | mismatches |
|---|---|---|
| 67 nt | 2 211 | **0** |
| 82 nt | 3 321 | **0** |
| 82 nt | 3 321 | **0** |

8 853 cells, zero differences. `fM2_real` **is** the modular decomposition.

**And the fork's kernel already computes it on the device.**
`modular_decomposition.cu:349` declares `int* d_dml; //DMLi` — the device twin of
the host's `DMLi`, produced one row at a time by `modular_decomposition_kernel`
and consumed by `new_c_kernel` as `DMLi1[j-1]`. Supporting circular RNA means
**persisting that row rather than recomputing anything**.

## What this actually costs

The arithmetic is free. The memory is not.

| item | cost |
|---|---|
| new arithmetic in the hot kernel | **none** — `d_dml` is already computed per row |
| new device memory | **one triangular `int` matrix per record**, the same size as the existing `fML` device buffer |
| `gpu_bytes_per_file()` | must be updated; VRAM per record rises, so the budgeted chunk width falls |
| `fM1_new` | **nothing** — filled by the post-processing itself (`mfe/mfe.c:655, 687, 712`), never by the sweep |
| host post-processing | `postprocess_circular()` runs per record over the filled matrices. Embarrassingly parallel across records; it belongs on the existing `RNA_BACKTRACK_THREADS` pool, not on the GPU. |

So the honest summary is: **circular RNA is cheap in compute and costs a chunk
width in VRAM.** That trade is the one to state up front, because on a
VRAM-bound workload a third matrix means proportionally more chunks, and chunk
count is what the whole flatten-and-offset architecture exists to minimise.

Worth noting the compensation: under `md->circ`, `fM1` is *not* allocated, so a
build that only ever folds circular RNA does not pay for both.

## Implementation sketch, in dependency order

1. **Sweep**: allocate the triangular `fM2_real` device buffer alongside `fML`;
   have `modular_decomposition_kernel` write `d_dml` into row `i` of it as well
   as into the rotating row buffer. Gate the allocation on `md->circ` so linear
   folds pay nothing.
2. **VRAM model**: add the matrix to `gpu_bytes_per_file()` under the same gate.
3. **Download**: `fM2_real` joins `c`/`fML` in the per-record readback.
4. **Post-processing**: call upstream's `postprocess_circular()` per record on
   the backtrack thread pool. No reimplementation — it is host code operating on
   host matrices, and reusing it verbatim is what keeps the bar meaningful.
5. **Guard**: retire the `-c` refusal from `a6bcc3b` only when the bar below is
   green, and keep it for any circular fold the sweep declines.

Backtracking needs a look in step 4: `postprocess_circular()` pushes onto
`bt_stack` itself, so the fork's backtrack entry has to accept a pre-seeded
stack rather than starting from `f5[n]`. That is the one part of this which is
not obviously mechanical.

## The bar, frozen now

`tests/circ/`:

- `circ_test.fa` — 6 records at 60/120/200/300/450/600 nt, seed 20260905,
  built from alternating GC-rich and AU-rich blocks so every record has real
  multibranch structure.
- `circ_reference.txt` — `RNAfold --noPS -c` from pristine 2.7.2.
- `linear_reference.txt` — the same records **without** `-c`.

The second reference is the point. The failure this feature actually had was
returning the linear answer, and a bar that only checks "does it match the
circular reference" catches that only by luck of formatting. Every record
differs, so the bar is sensitive to exactly that bug:

| record | circular | linear | Δ |
|---|---|---|---|
| 60 nt | −10.50 | −18.90 | 8.40 |
| 120 nt | −34.40 | −39.80 | 5.40 |
| 200 nt | −59.00 | −65.40 | 6.40 |
| 300 nt | −147.50 | −153.60 | 6.10 |
| 450 nt | −191.30 | −196.30 | 5.00 |
| 600 nt | −303.30 | −311.00 | 7.70 |

6 of 6 records show a circular effect; none is a weak vector.

## Status 2026-09-08 — correct today, and the CPU route is a floor not a plan

Three things re-measured or re-read today, in the order they change the picture.

**1. Circular already produces the RIGHT answer, at both levels.** This is not
future work; it is the status quo, and it is worth saying because "guarded and
refuses" above overstates it:

- `RNAfold -c` — `gpu_path_usable()` (`RNAfold.c`) turns the accelerator off for
  the **whole invocation**. `-c` is one model detail for every record, so this is
  equivalent to declining each; `verify_option_parity.sh` confirms `circ` is
  byte-identical via an asserted CPU route.
- `vrna_mfe_batch()` — the library guard declines circular fold compounds one by
  one, so the fallback loop calls `vrna_mfe()` per record, which reaches
  `mfe.c:322` and runs `postprocess_circular()`. A *mixed* batch therefore folds
  its circular records correctly on the CPU while everything else is accelerated.

**2. The recorded blocker is narrower than this document says.**
`postprocess_circular()` being `PRIVATE` does not block circular. It blocks
circular *inside the batch backend*, which is the only path that never reaches
`mfe.c:322`. See `PORT_OPTION_STATUS.md` §5.

**3. So the open work is speed, not correctness** — and there are three tiers,
which is Luke's framing (2026-09-08):

| tier | what | cost |
|---|---|---|
| **0 — today** | decline, fold on the CPU | correct, and *k*× slower on circular input |
| **1 — the design above** | GPU fills (incl. `fM2`), host post-processes | one chunk width of VRAM |
| **2** | post-process on the device too | almost certainly not worth it |

**Tier 1 is the one to build, and the PCIe cost is close to zero** — which is the
part that is easy to miss. The matrices *already* come back over PCIe for
backtracking, so persisting `fM2` adds it to a transfer that is happening anyway
rather than creating a new one. And the kernel already computes the values:
`fM2_real` **is** the `DMLi` the sweep currently discards (8853 cells verified,
§"the hot kernel already computes the missing matrix"). Nothing new is computed
and nothing new is transferred; a row is kept instead of dropped.

The real price stays what this document said from the start: **a third matrix
costs a chunk width**, and chunk count is what the whole flatten-and-offset
architecture exists to minimise — measured at roughly *k*× wall for *k* chunks,
because the loss is batch **width**, not per-chunk overhead
(`project_chunking_costs_batch_width`). On a VRAM-bound workload that trade could
plausibly cost more than circular folding gains, so tier 1 needs the chunk-count
effect measured on a real circular workload before it is worth landing.

**Tier 2 is recorded to be dismissed.** `postprocess_circular()` is host code
that is embarrassingly parallel *across records*, so it belongs on the existing
`RNA_BACKTRACK_THREADS` pool, not in a kernel. Porting it would mean
reimplementing upstream logic that would then drift — the exact thing reusing it
verbatim is meant to prevent.

## What is NOT covered here

- **`-c` with `-g`** is refused by upstream itself
  (`"G-Quadruplex support is currently not available for circular RNA"`), so the
  combination needs no work beyond keeping that refusal intact.
- **Circular multistrand** (`fms5`/`fms3`) is out of scope with the rest of
  multistrand.
