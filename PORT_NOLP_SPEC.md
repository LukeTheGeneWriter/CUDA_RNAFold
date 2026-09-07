# `--noLP` on the GPU path — diagnosis, design and frozen bar

*Written 2026-09-07, same shape as `PORT_SALT_SPEC.md`, `PORT_CIRC_SPEC.md` and
`PORT_GQUAD_SPEC.md`.*

**Current status:** `--noLP` is **guarded and refuses**, at the library
(`mfe/cuda/engine.c:96`) and at the driver (`src/bin/RNAfold.c:991`).

---

## 1. The record was wrong, and the way it was wrong is the point

The guard's own comment said `noLP` "still disagreed on 3 of 12 records", which
reads like a near miss — a tie-break, a rounding edge, something small.

It is not. That measurement **compared energies**, and energies agree on 45 of
60 records because the matrix's `f5` value comes out roughly right. Re-evaluating
each returned structure with `RNAeval` — the self-consistency check that needs no
oracle, and the one that caught the hard-constraint bug — gives:

| record | reported | structure re-evaluates to | pairs | |
|---|---|---|---|---|
| CPU 0 | −295.60 | −295.60 | 274 | consistent |
| GPU 0 | −295.60 | **−77.18** | **90** | off by 218.42 |
| GPU 1 | −294.94 | **+5.54** | **6** | off by 300.48 |
| GPU 2 | −275.20 | **−21.60** | **22** | off by 253.60 |

**8 of 8 sampled records inconsistent, by 87 to 300 kcal.** The matrix fill and
the backtrack do not agree *with each other*. This is the same failure shape as
the hard-constraint defect, and an energy-only comparison is structurally blind
to it: the number it checks is the one that happens to be nearly right.

Measured by lifting both guards on a throwaway build, folding 60 × 900 nt, and
comparing four ways (upstream `--noLP`, upstream plain, GPU `--noLP`, GPU plain):

- **C == D (noLP ignored entirely): 0 / 60.** The sweep is *not* simply ignoring
  the flag — `ptype` really is filtered (`sequences/alphabet.c:271`).
- **C strictly better than the oracle: 15 / 60. Worse: 0 / 60.** A perfectly
  one-sided error, which is the signature of a *missing constraint* rather than a
  miscomputation.
- **Lonely pairs**: upstream 0 everywhere; GPU plain 2–8 per record; GPU `--noLP`
  0 on 56 records and **1, 1, 1 and 3 on four of them**. `ptype` removes pairs
  that could never be in a helix; it cannot remove a pair that *could* be stacked
  but was chosen alone, because that is the recursion's decision.

## 2. `noLP` is a recursion change, not a data filter

`mfe/mfe.c:4413`:

```c
/* remember stack energy for --noLP option */
if (noLP) {
  stackEnergy = vrna_eval_stack(fc, i, j, VRNA_EVAL_LOOP_DEFAULT);
  new_c       = MIN2(new_c, cc1[j - 1] + stackEnergy);
  cc[j]       = new_c;
  e = cc1[j - 1] + stackEnergy;     /* <-- this is what lands in c[ij] */
} else {
  e = new_c;
}
```

Three facts, and each one matters:

1. **`c[i][j]` does not receive `new_c`.** It receives `cc1[j-1] + stackEnergy` —
   the best structure closed by (i,j) *given that (i+1,j-1) is also paired*. That
   is the whole of "no lonely pairs": a pair is only admitted into `c` if it
   stacks.
2. **The unconstrained value survives sideways**, in `cc[j]`, for the next row.
   Without it the recursion could never build a helix at all.
3. **`cc` is refilled with INF on every rotation** (`mfe.c:4472`), so a pair that
   is not evaluated leaves INF behind rather than a stale value.

`cc`/`cc1` rotate once per row alongside the multibranch helpers
(`rotate_aux_arrays`, `mfe.c:4460`) — the same point where this fork already
rotates `DMLi`/`DMLi1`/`DMLi2`.

## 3. The machinery was deleted because coverage said so

`mfe/cuda/fill_arrays_loop.c:215`:

```c
/* gcov says not used  remember stack energy for --noLP option *
   if(noLP) vrna_E_stack(vc, i, j) cc[j] = new_c */
```

It was dead because `noLP` was never exercised. Coverage measures the tests you
ran, not the options you support, and deleting a branch on that evidence removes
the feature while leaving the flag accepted. This is the ninth documented
instrument in this project that reported the truth about the wrong question.

**A sweep for the same pattern found four more**, and the reassuring result is
that every one is an option the guard already declines — the dead-code comments
and the guard list are the same list seen from two sides:

| declined option | deleted machinery |
|---|---|
| `noLP` | `fill_arrays_loop.c:215` — this document |
| `dangles ≠ 2` | `fill_arrays_loop.c:213` — coaxial stacking, `E_mb_loop_stack()` |
| `noGUclosure` | `int_loop.cu:1340` (`no_close`), `:1443` (the `no_close` continue) |
| `gquad` | `int_loop.cu:1244-1268`, `:1478` (`E_GQuad_IntLoop`) |
| domains | `int_loop.cu:1351` — the `domains_up->energy_cb` loop |

No *unguarded* deletion was found, so there is no second `-C`-class live bug in
this class. One case is unclassified: `int_loop.cu:1347` (`if (type == 0) type =
7;`) is not option-conditioned on its face; `type == 0` should be reachable only
under `energy_set != 0` or a constraint forcing a non-canonical pair, both
declined, but that has not been proven. **Caveat on the sweep itself:** it finds
deletions that left a comment behind. A silent deletion would not appear.

## 4. Design

**The backtrack needs nothing.** `vrna_backtrack_from_intervals()` (public,
`backtrack/global.h`) wraps upstream's own `PRIVATE backtrack()`, which already
branches on `noLP` at `mfe/mfe.c:4289` and calls `vrna_bt_stacked_pairs()`. It is
currently mis-walking our matrix precisely *because* that matrix was built under
the wrong convention. Fix the fill and the backtrack becomes correct for free.

**The edit site is one kernel.** In device mode the host `new_c` loop is skipped
(`fill_arrays_loop.c:165`, gated on `rnafold_gpu_sweep()`); `new_c_kernel`
(`hp_mb_loop.cu:1215`) computes `new_c` on the device. That kernel is small, is
already per-(H,j) row-shaped, and already receives the `gate_row` bitmask it
needs.

```
                      today                    with noLP
  new_e[o+j]   =   new_c                    =  cc1[o+j-1] + stackE
  (nothing)                                    cc[o+j] = MIN2(new_c, that)
```

**What has to be added:**

| piece | shape | precedent in this codebase |
|---|---|---|
| `d_cc`, `d_cc1` | row-shaped, `row_off_H` total | exactly `d_dml` / `d_dml1` |
| INF refill of `cc` each row | one kernel, row-shaped | `init_fML_kernel` over `hsize` |
| rotation per row | pointer swap | the `DMLi/DMLi1/DMLi2` rotate |
| `stackEnergy(i,j)` | per cell | the `ns=nl=0` case `IntLoop_X` already computes |

**`stackEnergy` is not new arithmetic.** `vrna_eval_stack()`
(`eval/eval_internal.c:228`) is a hard-constraint check —
`hc->eval_int(i, j, i+1, j-1, hc)` — followed by `eval_stack()`, which is the
0 × 0 interior loop: `P->stack[type][type_2]`, plus soft-constraint terms that
are declined anyway. The interior-loop kernel already evaluates exactly this case
(it is the one the salt work fed `SaltStack` into), and `ptype`/`gate_row` already
carry what the hard-constraint half needs.

**Cost.** One extra row-buffer pair per record (noise beside the triangle, same
scale as `d_dml`), one extra row-shaped kernel per row for the INF refill, and a
handful of instructions per cell — all of it *only* when `noLP` is set. The
multibranch kernel, which is 73 % of wall and at its DRAM floor, is untouched.

**Gate it.** `noLP` support lands behind the existing guard: the guard is lifted
only when the bar below is green, so one binary can be compared against itself
and against upstream in the meantime.

## 5. The bar, frozen now

The bar is **not** an energy comparison. That is what hid this defect for a
session, and a fix verified by energies could reproduce it exactly.

1. **Self-consistency, per record, no oracle needed.** Every returned structure
   re-evaluated with `RNAeval` must match the energy reported beside it. This
   alone would have caught the current defect, and it catches any future
   matrix/backtrack disagreement.
2. **Lonely-pair count must be 0** in every `--noLP` structure. Direct, and it
   fails on exactly the four records where `ptype` filtering was not enough.
3. **Byte-identical against upstream** `RNAfold --noPS --noLP` over the reference
   set — the project's standing bar, applied last rather than first.
4. **Not equal to the plain fold.** A `--noLP` run that matches the unconstrained
   answer has not applied the option; on 60 × 900 nt upstream differs from its own
   plain fold on 59 of 60 records, so this is a live discriminator.
5. **Multi-chunk identical to single-chunk**, since `cc`/`cc1` are per-record row
   buffers and a chunk boundary must not move an answer.

`tests/nolp/` mirrors `tests/salt/` and `tests/gquad/`: the 60 × 900 nt workload,
upstream's `--noLP` output as the reference, and its plain output as the
second reference that makes the bar sensitive to the failure actually seen.

## 6. Where this sits among the parity features

Revised from `PORT_GQUAD_SPEC.md`'s ordering, on measured cost:

1. **Salt** — done. Kernels needed nothing.
2. **`uniq_ML`** — done. Pure host post-pass.
3. **`noLP`** — this document. One small kernel, two row buffers, no new
   arithmetic, backtrack free. **Cheaper than circular**, and unlike circular it
   is not blocked on an upstream API decision.
4. **Circular** — no new arithmetic, but blocked on `postprocess_circular()`
   being `PRIVATE`, and it costs a chunk width.
5. **G-quadruplex** — cheap data, real work in the hot loops, per-cell sparse
   lookup in the kernel already at its DRAM floor.
6. **Multistrand** — genuinely inside the recursion; a rewrite.
7. **Modified bases** — host callbacks in a kernel; may not be achievable.
