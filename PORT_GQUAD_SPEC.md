# G-quadruplexes (`-g`) on the GPU path — design and frozen bar

*Written 2026-09-05 against ViennaRNA 2.7.2, same shape as `PORT_SALT_SPEC.md`
and `PORT_CIRC_SPEC.md`.*

> **RE-SCOPED 2026-09-10 against the port — read the last section first.** The
> "lookup table, not a recursion" thesis holds and is stronger than this
> document knew: **the device work is TWO sites, not 49**, because most of
> `mfe.c`'s gquad code is inside `postprocess_circular()`, and three of the
> four stages are already free. The lookup-design recommendation below is also
> reordered by measurement.

**Current status:** `-g` is **guarded and refuses** on the GPU path as of
`a6bcc3b`. Before that it produced structures that were internally
self-consistent — 2.7.2's own `RNAeval -g` confirmed every energy to the cent —
but **suboptimal on 8 of 8 records, by 15 to 31 kcal** (`PORT_FEATURE_AUDIT.md`).
The GPU simply never considered the quadruplex alternatives.

---

## The headline: this is a lookup table, not a recursion

The obvious reading of "G-quadruplex support" is 1262 lines of `mfe/mfe_gquad.c`
to port into kernels. That is not what is required.

`c_gq` is built **once, up front, at matrix-allocation time**:

```c
/* datastructures/dp_matrices.c:547 */
vc->matrices->c_gq = vrna_mfe_gquad_mx(vc);
```

independently of the MFE recursion. Every one of its uses in the MFE path is a
**read** — `vrna_smx_csr_int_get(c_gq, i, j, INF)`. So the structure is:

| stage | where it runs |
|---|---|
| find quadruplexes, score them, build `c_gq` | **host, once per record, before the sweep** — reuse `vrna_mfe_gquad_mx()` verbatim |
| consult `c_gq` while filling `c`/`fML`/`f5` | **device**, read-only lookups |
| backtrack through a quadruplex | **host** — `backtrack/bt_gquad.c`, per record |

Only the middle row is kernel work. `mfe_gquad.c` itself never needs porting.

## How much data, and in what form — measured

`c_gq` is a sparse CSR matrix (`vrna_smx_csr(int)`), which is structurally alien
to the dense triangular flatten-and-offset layout everything else in this
project rests on. So the real question is whether to densify it on upload.

`tests/upstream/gquad_sparsity_probe.c`, G-rich sequence, `md.gquad = 1`:

| n | entries | cells `n(n+1)/2` | fill | dense cost/record |
|---|---|---|---|---|
| 100 | 217 | 5 050 | 4.30 % | 19.7 KiB |
| 300 | 314 | 45 150 | 0.70 % | 176.4 KiB |
| 600 | 4 330 | 180 300 | 2.40 % | 704.3 KiB |
| 1 200 | 11 661 | 720 600 | 1.62 % | 2 814.8 KiB |
| 2 400 | 18 156 | 2 881 200 | 0.63 % | 11 254.7 KiB |

Two things fall out, and they decide the design:

1. **Fill is under 5 % everywhere and falls as n grows.**
2. **Entry count grows roughly linearly in n, not quadratically** — 18 156
   entries at n = 2400 where the dense matrix has 2.88 M cells.

So a dense upload would cost **another whole triangular matrix per record**,
which is precisely the memory the chunker is short of, to carry ~1 % useful
data. At n = 2400 that is 11.25 MiB against roughly 145 KiB for the sparse form
— a factor of ~77. **Upload sparse.**

This is the opposite conclusion from circular RNA (`PORT_CIRC_SPEC.md`), where
the extra matrix is dense, unavoidable, and costs a chunk width. Worth stating
plainly because the two features look similar from the outside and their memory
stories are nothing alike.

## The kernel work, honestly

This is the expensive one of the three parity features specified so far, and the
cost is not in the table upload — it is in the lookups.

**Edit sites.** `with_gquad` branches in the MFE path: `mfe/mfe.c` 28,
`mfe/mfe_exterior.c` 11, `mfe/mfe_internal.c` 6, `mfe/mfe_multibranch.c` 4.
Each is a `MIN2` against a `c_gq` lookup, in the four decomposition families the
fork already mirrors in kernels.

**The lookup itself is the risk.** `vrna_smx_csr_int_get()` is a binary search
over a row's sorted column indices. In a kernel, run per cell, that is
divergent, unpredictable in latency, and lands in the innermost loops of exactly
the kernel (`modular_decomposition`) already measured to be at its DRAM floor.
Three ways to avoid paying it per cell, in the order worth trying:

1. **Per-row skip.** Most rows contain no quadruplex start at all. A per-row
   entry count (or a bitmask over `i`) lets the common case exit before any
   search, so the cost lands only on the small fraction of rows that need it.
2. **Row-local dense expansion.** Only a handful of `j` per `i` have entries;
   expanding one row into shared memory at row start makes each lookup an index
   rather than a search.
3. **Dense only for short records.** The dense form is 19.7 KiB at n = 100 and
   only becomes untenable at length. A length-dependent representation is
   legitimate here, and the chunker already reasons per record.

None of this can be measured until the kernels exist on 2.7.2 — it is Phase 3
work, and it should be measured rather than chosen from this list on taste.

**One parameter difference to carry across.** 2.7.2 adds three compiled-in
constants that 2.3.0 does not have: `GQuadLayerMismatch37`, `GQuadLayerMismatchH`
and `GQuadLayerMismatchMax`. They were the *only* energy-table difference found
between the two versions (`PORT_PHASE0.md`), and they are live for multi-layer
quadruplexes. A 2.3.0-based backport of this feature would therefore diverge from
the 2.7.2 oracle on exactly the records it is meant to fix — another reason this
waits for the rebase rather than being attempted on the current base.

## The bar, frozen now

`tests/gquad/`:

- `gquad_test.fa` — 6 records at 80/150/300/500/800/1200 nt, seed 20260905,
  G-runs with short linkers interleaved with ordinary structured RNA.
- `gquad_reference.txt` — `RNAfold --noPS -g` from pristine 2.7.2.
- `nogquad_reference.txt` — the same records **without** `-g`.

The second reference is what makes the bar sensitive to the fork's actual bug.
A GPU path that ignores quadruplexes returns a valid, self-consistent, *worse*
answer, so a bar that only compares against the `-g` reference tells you
"different" without telling you "worse in the specific way we already saw".

| record | `-g` | plain | G-quad gain |
|---|---|---|---|
| 80 nt | −34.51 | −9.20 | 25.31 |
| 150 nt | −77.99 | −68.90 | 9.09 |
| 300 nt | −199.18 | −73.70 | 125.48 |
| 500 nt | −383.73 | −125.40 | 258.33 |
| 800 nt | −693.78 | −250.15 | 443.63 |
| 1200 nt | −1046.28 | −257.27 | 789.01 |

6 of 6 records contain a quadruplex (`+`) in the `-g` structure, and every one
shows a gain — no weak vectors. The smallest gain, 9.09 kcal at 150 nt, is still
far outside anything a rounding difference could produce.

## Sequencing among the parity features

On measured cost rather than apparent size:

1. **Salt** — kernels need nothing; the multibranch contribution is baked into
   `MLbase`/`MLclosing`/`MLintern` at parameter-init time (`PORT_SALT_SPEC.md`).
2. **Circular** — no new arithmetic, one dense matrix, costs a chunk width
   (`PORT_CIRC_SPEC.md`).
3. **G-quadruplex** — this document: cheap data, real kernel work in the hot
   loops.
4. **Multistrand** — not specified yet; genuinely inside the recursion
   (`pair_multi_strand()`, `fms5`/`fms3`), and the only one that is a rewrite.
5. **Modified bases** — arbitrary user soft-constraint callbacks. Host function
   pointers cannot run in a kernel; this one may not be achievable at all and
   needs an explicit decision rather than an attempt.

---

# SCOPED AGAINST THE PORT, 2026-09-10

*The document above was written 2026-09-05, before the kernels existed on 2.7.2.
Its central claim — "this is a lookup table, not a recursion" — **holds and is
stronger than it knew**. Its edit-site inventory does not: it overcounted the
work by roughly an order of magnitude. Re-derived here against `port27` at
`c4adfae3`, with the row-structure measurements the design actually turns on.*

## 1. The inventory was ~10× too big, because most of it is CIRCULAR

The spec counted `with_gquad` branches as "`mfe/mfe.c` 28, `mfe/mfe_exterior.c`
11, `mfe/mfe_internal.c` 6, `mfe/mfe_multibranch.c` 4" — 49 sites.

**`mfe.c:597–2280` is `postprocess_circular()`.** Every one of `mfe.c`'s `c_gq`
uses (lines 759, 800, 890, 1095, 1194, 1273, 1293, 1378, 2138) falls inside it,
including all the "case 2.1 / 2.2 / case 3" quadruplex-in-a-loop machinery that
makes the file look expensive. **Circular is already declined**, and `-c -g` is
refused outright by RNAfold itself (`RNAfold.c:767`). `mfe_window.c` and
`mfe_exterior_window.c` are the sliding-window path — also declined.

The only remaining `mfe.c` site is `:4234`, `vrna_bt_gquad()`, inside
`backtrack()` — host, and already ours (see §2).

**For the linear global MFE fill the real inventory is:**

| site | what it is | where it runs |
|---|---|---|
| `mfe_exterior.c` `add_f5_gquad()`, 3 call sites | f5 exterior loop | **host — free** |
| `mfe_internal.c:637` → `vrna_mfe_gquad_internal_loop()` | 3 bounded (p,q) sweeps | **device** |
| `mfe_multibranch.c:986` in `extend_fm_3p()` | one `MIN2` against `c_gq(i,j)` | **device** |

**Two device sites.** Not 49.

## 2. Three of the four stages are already free, and that is not luck

The port calls upstream's own host code at each of them, so `-g` support falls
out of code that already runs:

| stage | port call site | gives us |
|---|---|---|
| build `c_gq` | `vrna_fold_compound_prepare(VC[i], VRNA_OPTION_MFE)`, `mfe_cuda.c:1090` → `dp_matrices.c:547` → `vrna_mfe_gquad_mx()` | the whole matrix, per record |
| exterior f5 | `vrna_mfe_exterior_f5(vc)`, `mfe_cuda.c:725` | `add_f5_gquad()` |
| backtrack | `vrna_backtrack_from_intervals()` (upstream PUBLIC) | `vrna_bt_gquad()` |
| fM1 under `uniq_ML` | `vrna_mfe_multibranch_m1()` | its own gquad handling |

`c_gq` is therefore **already being built today** whenever `md.gquad` is set —
the guard declines the compound before the sweep, but the matrix is constructed.
Nothing has to be added to get the data onto the host side.

> **A trap for whoever probes this.** `VRNA_OPTION_DEFAULT` is literally `0`
> (`fold_compound.h:398`), so `vrna_fold_compound(seq, &md, VRNA_OPTION_DEFAULT)`
> adds **no matrices at all** and `c_gq` comes back `NULL` — which reads exactly
> like "gquad is off". The matrices arrive at *prepare* time, not at
> fold-compound time. The first run of `gquad_rowstats_probe.c` reported "NO
> c_gq" for every record and was one edit away from being written up as a
> blocker. `RNAfold.c:1369` builds compounds with `VRNA_OPTION_DEFAULT` too, and
> is correct to: `par_mfe()` prepares them.

## 3. New measurement: the row structure, which decides the lookup design

`tests/upstream/gquad_rowstats_probe.c` (new). Entry counts reproduce the
2026-09-05 sparsity probe **exactly** (217 / 314 / 4330 / 11661 / 18156), which
cross-validates both.

On the frozen bar's own six records:

| record | entries | fill | rows with any entry | max entries in a row | CSR | dense |
|---|---|---|---|---|---|---|
| 80 | 291 | 8.98 % | 26/80 (32.5 %) | 19 | 2.6 KiB | 12.7 KiB |
| 150 | 23 | 0.20 % | 8/150 (5.3 %) | 5 | 0.8 KiB | 44.2 KiB |
| 300 | 868 | 1.92 % | 74/300 (24.7 %) | 27 | 8.0 KiB | 176.4 KiB |
| 500 | 1 586 | 1.27 % | 130/500 (26.0 %) | 33 | 14.4 KiB | 489.3 KiB |
| 800 | 2 909 | 0.91 % | 220/800 (27.5 %) | 32 | 25.9 KiB | 1 251.6 KiB |
| 1 200 | 7 196 | 1.00 % | 476/1200 (39.7 %) | 35 | 60.9 KiB | 2 814.8 KiB |

**Two findings, and the first reverses the spec's recommended order.**

1. **Per-row skip is the weakest of the three options, not the first.** The spec
   assumed "most rows contain no quadruplex start at all". They contain one
   **5–48 %** of the time, and **the fraction RISES with n** (27 → 15 → 37 → 48
   → 46 % on the synthetic series). At the sizes this project cares about, a
   per-row skip rejects roughly half the rows, not almost all.
2. **`max_row` is small and roughly CONSTANT in n** — 12–45 across n = 80…2400.
   So a row's entries always fit in a trivial amount of storage, whatever the
   sequence length. That is what makes the row-expansion designs viable.

Upload sparse is confirmed and is not close: **5–74× smaller**, growing with n.

## 4. Recommended design, per site

### The multibranch site — the port already has the machinery

`extend_fm_3p()` needs `c_gq(i, j)` for the very cell it is computing. The fork's
kernels are organised as *"row `i`, all records `H`, all `j`"* and already carry
per-row scratch of shape `nfiles*(length+1)` — `energy_hp_row`, `energy_mb_row`,
`energy_3p00_row`, `gate_row`, all filled once per row and indexed
`row_off_H[H]+j`.

**Add `gq_row` to exactly that pattern**: one cheap kernel per row expands each
record's `c_gq` row into it, and the lookup in the multibranch kernel becomes
`gq_row[row_off_H[H]+j]` — **an index, not a search**. This is the design the
spec's option 2 was reaching for, in the shape the codebase already uses. No
binary search on the device at all.

### The internal-loop site — iterate the ENTRIES, not the window

`vrna_mfe_gquad_internal_loop()` needs `c_gq(p, q)` for `p` in
`[i+1, i+MAXLOOP+1]` — **up to 31 different rows**, so the current-row buffer
does not serve it.

The window is bounded on both axes: `MAXLOOP` = 30, and the box size is
`VRNA_GQUAD_MIN_BOX_SIZE` = 11 to `VRNA_GQUAD_MAX_BOX_SIZE` = **73**
(`params/basic.h:41`), so `q − p < 63`. It is also doubly filtered — upstream
skips unless `S1[p] == 3` *and* `S1[q] == 3` (both must be G).

Given `max_row ≤ ~45` and ~half of rows empty, **scanning a row's sparse entries
and filtering by the `q` range beats searching the dense window**: it is ≤45
iterations for a populated row and a single bounds check for an empty one,
against ~63 lookups per `p` the other way.

## 5. The performance risk is smaller than the spec feared

The spec's main worry was that per-cell `c_gq` lookups would land "in the
innermost loops of exactly the kernel (`modular_decomposition`) already measured
to be at its DRAM floor".

**Everything here is gated on `with_gquad`.** With `-g` off — which is every run
this project has ever benchmarked — the added code is a predicted-not-taken
branch on a batch-constant flag. **The default path's performance is not at
risk.** The cost question is confined to `-g` runs, which have no performance
baseline to regress against, and which today produce *wrong answers* or a
refusal.

That reordering matters for sequencing: this feature cannot make the fast path
slower, so it does not have to wait behind the performance work.

## 6. What is still unmeasured

1. **The host cost of `vrna_mfe_gquad_mx()` per record**, and how it scales.
   `prepare` is 0.007–0.5 s for 400 records **without** gquad; with `-g` it also
   builds `c_gq`, and that is serial per record in the same stage that the build
   pipeline exists to hide. It may be free; it has not been looked at.
2. **`GQuadLayerMismatch37` / `H` / `Max`.** 2.7.2 adds these three constants and
   2.3.0 lacks them (`PORT_PHASE0.md` — the *only* energy-table difference found
   between the versions). They are live for multi-layer quadruplexes and reach
   the device only if the kernels need them; whether the two device sites do is
   not yet established.
3. **Whether `-c -g` stays refused.** RNAfold refuses it outright
   (`RNAfold.c:767`), so circular×gquad is out of scope by upstream's own
   decision, not ours. Worth stating in the guard rather than leaving implicit.
4. **Multi-record batching of `c_gq`.** The CSR is per record; the device needs
   the batch's worth flattened with a per-record offset table, exactly like
   `tri_off_H`. Cheap, but it is real plumbing and nobody has written it.

## 7. Staging

| phase | work | bar |
|---|---|---|
| **G0** | Flatten the batch's `c_gq` into device buffers (values + column indices + per-row offsets + per-record offsets), upload, and read it back unchanged | a device-vs-host `vrna_smx_csr_int_get()` comparison over every (i,j) |
| **G1** | The multibranch site: `gq_row` expansion + the `MIN2` in the fML kernel | `-g` output vs the frozen `gquad_reference.txt`, expected still wrong (the internal-loop term is missing) but *closer* |
| **G2** | The internal-loop site: three bounded sweeps over sparse entries | **byte-identical to `gquad_reference.txt` on all 6 records** |
| **G3** | Lift the `-g` guard (`engine.c:90`) and add a `tests/mfe_cuda_gquad.ts` in the shape of `mfe_cuda_nsp.ts` | RED-team it: revert G2 and confirm the test fails |

**G1 is deliberately expected to fail its own bar.** A stage whose bar is
"different, but in the predicted direction" is worth having here, because the two
device sites are independent and landing them together would make a wrong answer
impossible to attribute.

## 8. The frozen bar is intact and still the right bar

`tests/gquad/` still holds `gquad_test.fa`, `gquad_reference.txt` and
`nogquad_reference.txt`. The second is what makes it sensitive to the *actual*
historical failure — a GPU path that ignores quadruplexes returns a valid,
self-consistent, **worse** answer, and comparing only against the `-g` reference
would say "different" without saying "worse in the way we already saw". Every one
of the 6 records gains between 9.09 and 789.01 kcal/mol from `-g`, so there are
no weak vectors in it.
