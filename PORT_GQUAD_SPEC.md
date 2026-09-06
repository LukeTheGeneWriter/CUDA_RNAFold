# G-quadruplexes (`-g`) on the GPU path — design and frozen bar

*Written 2026-09-05 against ViennaRNA 2.7.2, same shape as `PORT_SALT_SPEC.md`
and `PORT_CIRC_SPEC.md`.*

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
