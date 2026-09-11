# `--noClosingGU` on the GPU path — what was missing, and the defect it uncovered

*Written 2026-09-11 against ViennaRNA 2.7.2, after the fact rather than before
it: unlike `PORT_CIRC_SPEC.md` and `PORT_GQUAD_SPEC.md` this option was never
scoped in advance, because "half implemented" made it look like finishing work
rather than design work. It was — but the half that was missing dragged a
year-old latent defect into the light with it.*

**Status: ACCELERATED 2026-09-11.** All three gates lifted. `--noClosingGU` was
the last option behind all three, so `fill_arrays.c` now carries **no backstop at
all**.

---

## What the option is

`md->noGUclosure` forbids a GU or UG pair from *closing* a loop. It does **not**
forbid GU pairs (that is `--noGU`) and it does not stop a GU pair being a
*branch* of a multiloop or an external stem.

Upstream implements it in two unrelated places, and that split is the whole
story:

| where | which loops | mechanism |
|---|---|---|
| `constraints/hard.c:786-791` | hairpin, multiloop | the pair loses `HP_LOOP` and `MB_LOOP` from its hard-constraint context |
| `mfe/mfe_internal.c:305, 339, 415, 579` | bulge, interior | explicit `noclose` / `type2` skips in the recursion |

The second is not expressible as a hard constraint, which is why it is not one:
the rule there is **conditional on the loop's size**. A GU pair may still close a
*stack*. `mfe_stacks()` carries no test at all, and `vrna_E_internal()`
(`eval/eval_internal.c:104-108`) returns the stack energy **before** it consults
`no_close`. `INT_LOOP` therefore has to stay in the pair's context, and the
finer rule lives in the recursion.

---

## What the fork had, and what it did not

**The mask half already worked, and was already provable.** `rnafold_hc_opt()`
(`mfe/cuda/stub2.h`) is a device replica of `hc_reset_to_default()` and carries
the `noGUclosure` branch verbatim, including the `noGU` precedence above it.
`RNA_HC_VERIFY=1` rebuilds the masks the host way — straight out of
`VC[H]->hc->mx` — and compares them word for word. Under `--noClosingGU` on the
frozen fixture: **46 429 words × 4 masks, 0 mismatching**. That check needs no
oracle and no second fold.

Downstream, `new_c_kernel` (`hp_mb_loop.cu:1450`) skips both the hairpin and the
multibranch term when gate bit 1 (the pair is GU/UG) is set and `noGUclosure` is
on.

**The interior-loop half did nothing whatsoever.** `Energy()` evaluated every
candidate `(p,q)` regardless of either pair's type.

That is worse than it sounds, and it is why three gates rather than one. A
*suboptimal* matrix is still the minimum of *some* model. A matrix where the
hairpin rule is applied and the interior rule is not is the minimum of **no**
model — `c` is internally inconsistent, and the backtrack walks it assuming a
consistency that is not there.

---

## The fix

Two lines in `Energy()` (`int_loop.cu`), placed where upstream places its own
`continue`s:

```c
if (P->noGUclosure && (u1 || u2) &&
    ((type == 3) || (type == 4) || (type_2 == 3) || (type_2 == 4)))
  return INF;
```

Three things about it are deliberate.

**`(u1 || u2)` is the stack exemption**, and it is exact: `u1 == 0 && u2 == 0` is
precisely the `(i+1, j-1)` candidate that `mfe_stacks()` handles.

**`type` and `type_2` are the transformed values, matching upstream literally.**
`type` is the promoted enclosing ptype (`0 → 7`), `type_2` is `rtype[]` of the
promoted inner one. Neither transform can move a value into or out of `{3,4}` —
`0 → 7` never lands on 3 or 4, and `rtype[]` only swaps `3 ↔ 4` — so the test is
insensitive to which convention is used. Written to match upstream anyway,
because *this* function is where the `--nsp` defect lived for a year precisely
because it had drifted from upstream's spelling.

**The skip is in the CALLER, not in `IntLoop_X()`**, even though upstream's own
guard lives inside `vrna_E_internal()`. `Energy()` does
`energy = my_c[pq]; energy += IntLoop_X(...)`. An `INF` returned from the callee
would be **added** to a real negative `c[pq]` and land just *below* `INF` —
winning the `min` as a decomposition that is forbidden. Upstream avoids this the
same way, with explicit skips before the call; its in-callee test is unreachable
from that path.

The G-quadruplex interior term needed the same thing:
`vrna_mfe_gquad_internal_loop()` is called from **inside**
`mfe_internal.c`'s `if (!noclose)` block (`:637`), so `gq_internal_kernel`
returns early for a GU/UG closing pair. There is no enclosed-pair half there —
what is enclosed is a quadruplex, not a pair.

`noGUclosure` rides in `cuda_param_t`, appended last, for the same reason
`rtype[]` and `max_ninio` do: both readers already take a `cuda_param_t *` and
nothing else, and `sanity()` asserts every record in a batch agrees on it.

---

## The defect this uncovered: one-sided INF guards in `fml_scan_kernel`

**This is the part worth reading even if you never use `--noClosingGU`.**

`fml_scan_kernel` (`hp_mb_loop.cu`) composed its two fML terms like this:

```c
const int c_term = (e3p00[o+j] != INF) ? new_e[o+j] + e3p00[o+j] : INF;
const int fp     = fml_prev[o+j];
const int e3     = (fp != INF) ? fp + en_i : INF;   // hazard 1: en_i NOT guarded
```

Each line guards **exactly one side of a sum whose other side can be `INF`** —
and the second line's own comment said so.

`INF` is 10000000: a sentinel, not a number. `INF` plus a real negative energy
lands just *below* `INF`, so the result is no longer recognisable as "no such
decomposition" while still being far too large to win any `min` against a real
energy. It survives in fML, invisibly. Upstream guards the operand this code did
not — `extend_fm_3p()` (`mfe/mfe_multibranch.c:949-950`) reads `en = c[ij]` and
tests `if (en != INF)` before adding the stem energy. The port's `e3p00` test
stands in for the *enclosing* `evaluate(..., VRNA_DECOMP_ML_STEM)` instead, so
the `c[ij]` side was never tested at all.

**Why it had never bitten.** On the int32 path a ~1e7 value never wins a `min`
against a real energy, so the fold is unaffected. This is why it survived every
byte-identical run this project has, including 400 × 5601 across 22 arms.

**Why `--noClosingGU` made it fatal.** A GU/UG pair keeps `MB_LOOP_ENC` but loses
`HP_LOOP` and `MB_LOOP`, so `c[i][j]` is `INF` for far more `(i,j)` that still
reach this sum. Combined with `RNA_FML_INT16`, fML then carried **9999890**
(`INF` minus an `E_MLstem` of 110). `pack_fml_kernel` tests `v == INF` to choose
the `FML_INF16` sentinel; 9999890 is not `INF`, so it became a block **baseline**,
and the next real value in that block (180) was 9999710 away from it.
`__trap()` — correctly, and loudly.

Both lines are now `fml_tadd()`, the `INF`-safe add already defined for exactly
this purpose a few lines above. Fixed at the source rather than clamped at the
consumer, because the same fML feeds `DMLi`, and `DMLi` carrying near-`INF`
sentinels is `PORT_INVESTIGATIONS.md` item 3. (It does **not** close that item:
the reduction there adds two fML values without a guard of its own, which is a
separate instance of the same family.)

**Under WDDM a `__trap()` looks like a hang.** The process died; its zombie was
not reaped because a CUDA driver thread persisted, so `timeout 20` never
returned and the first diagnosis was "int16 + `--noClosingGU` hangs". The trap's
own `printf` goes to the process **stdout**, i.e. into the fold output, which is
where it was eventually found. Worth remembering: on this platform, *look in
stdout before concluding "hang"*.

---

## The bars

`tests/noclosinggu/` freezes the fixture and both references.

**The fixture is G/U-weighted by construction.** `--noClosingGU` can only bite
where a GU/UG pair would otherwise close something, and a uniform ACGU draw
produces a fixture that passes whether or not the feature works. On this one,
**19 of 20** records change structure under the flag.

| bar | what it proves | result |
|---|---|---|
| frozen `tests/noclosinggu/`, GPU vs pristine 2.7.2 | the answer | **0 differing lines**, `sweeps=1` asserted |
| `RNA_HC_VERIFY=1` under the flag | the mask half, **no oracle** | 46 429 × 4 words, **0 mismatching** |
| `--noLP` / `--noGU` / `-c` / `-g` / `-T 25` / `-4` / `--salt 0.5` | pairwise | all 0 |
| chunked, int16, chunked+int16 | the encoding | all 0, **no trap** |
| the DEFAULT path, 3 arms | that the `INF`-guard fix changed nothing | all 0 |
| `tests/mfe_cuda_noclosinggu.ts` | the regression bar | asserts GPU == upstream **and** `!=` the default fold |

**Red-teamed both ways.** Removing the `Energy()` skip makes the byte bar fail
with **72 differing lines**. Restoring the one-sided `INF` guard brings the
int16 trap back.

---

## What is NOT covered here

- **The partition function.** `exp_E_IntLoop()` bakes the same rule in
  (`eval/internal.h:736`) and is untouched by the sweep; `-p` still runs on the
  CPU path in `process_record()`.
- **Comparative / multistrand.** Both branches exist in the upstream sites above
  and neither is constructible from `RNAfold`.
- **Soft constraints.** Still declined at gate 2, independently of this option.
