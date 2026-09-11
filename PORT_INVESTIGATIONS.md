# Flagged for investigation

*Things noticed while doing something else, that look like they could bite and
have not been chased. Each entry says what was actually checked, so the next
person starts from evidence rather than from the suspicion.*

---

## 1. Pair-type resolution: the 0 → 7 promotion, and the ORDER it composes with

**Noticed while writing `gq_internal_kernel` (G2), 2026-09-10.**

`vrna_get_ptype_md()` (`sequences/alphabet.c:471-478`) is not a plain table read:

```c
unsigned int tt = (unsigned int)md->pair[i][j];
return (tt == 0) ? 7 : tt;
```

It **promotes a zero ptype to 7**. So does `vrna_get_ptype()`. The device's own
`Ptype()` / `Ptype2()` return the raw `pair[]` value and do **not**.

**This has already been a live wrong answer once.** `Energy()` in `int_loop.cu`
omitted the promotion (and swapped indices instead of applying `rtype[]`), which
made `--nsp` disagree with upstream on 28 of 30 records. Fixed 2026-09-10; the
`--nsp` guard came off as a result.

### What was checked

Every pair-type resolution site in the CUDA path, and all of them now promote:

| site | file | note |
|---|---|---|
| `Energy()` | `int_loop.cu:1073-1077` | fixed 2026-09-10 |
| `gq_internal_kernel` | `int_loop.cu:1229` | written with it (G2) |
| `hp_mb_3p_kernel` | `hp_mb_loop.cu:883` | promotes |
| multibranch closing | `hp_mb_loop.cu:900-901` | `rtype[raw_type]` **then** promote |
| `stack_row_kernel` | `hp_mb_loop.cu:1370-1371` | promote, matching `vrna_get_ptype()` |

### Why it is still worth an investigation

**The ORDER differs between sites, deliberately, and the source says so.**
`hp_mb_loop.cu:1323-1325` records that the stack site applies the promotion in
"the OPPOSITE order from the multibranch site above, which deliberately indexes
`rtype[]` with the raw value and only fixes up the *result*."

Both may well be right — they mirror different upstream expressions — but "two
sites do the same thing in opposite orders, on purpose" is exactly the shape that
produced the `--nsp` defect, and the correctness of each rests on a comment
rather than on a test.

**What to do:** a bar that exercises each site with a `md->pair` table containing
a genuine 0 where a pair is otherwise allowed, and with an asymmetric `--nsp`
spec, comparing against upstream per site rather than end-to-end. End-to-end
already passes; the question is whether it passes *for the right reason at every
site*.

---

## 2. `#define turn 3` shadows any identifier called `turn`

**Noticed while writing `gq_internal_kernel` (G2), 2026-09-10.**

`int_loop.cu:71` does `#define turn 3`, `#undef`ed only at `:1464`. For 1 393
lines, **any declaration named `turn` silently becomes `const int 3`** — which is
a syntax error where it is a parameter, and something stranger where it is not.

That is how it was found: a new kernel taking `const int turn` failed to compile
with `expected a ")"` at the parameter list, nowhere near the `#define`.

### What was checked

- The macro's live range is `int_loop.cu:71`–`:1464` only; no other CUDA file
  defines it.
- **Two existing kernel signatures carry `/*const int turn,*/` commented out**
  (`:773` `load_my_c_kernel`, `:1337`) — someone hit this before and worked
  around it by deleting the parameter rather than renaming the macro.
- `:1562` has a real variable `turn = md->min_loop_size;`, which is fine because
  it is past the `#undef` — but it means **the same identifier is a macro in the
  first half of the file and a variable in the second**.
- No live shadowing today. The hazard is latent, not active.

### Why it is worth fixing rather than remembering

`turn` is `md->min_loop_size` everywhere else in RNAlib, and pinning it to a
literal 3 inside one translation unit is a correctness assumption as well as a
naming hazard: a caller who sets `min_loop_size` to anything else would be
silently ignored by every kernel in this file. That assumption may be guarded
elsewhere — **it has not been verified** — and the guard, if any, is not next to
the `#define`.

**What to do:** rename the macro to `TURN_CONST` (or take it from
`md->min_loop_size` like everything else), and check whether a non-default
`min_loop_size` is refused by the routing guard. If it is not, that is a third
silent wrong answer of the same family as `--nsp` and `MAX_NINIO`.

---

## 3. DMLi carries near-INF sentinels that upstream's fM2_real never would

**Found by `RNA_CIRC_VERIFY` while landing circular support, 2026-09-11.**

`modular_decomposition_kernel`'s reduction is

```c
value = MIN2(fml_i[row_off_H[H]+y] + fml_j[tri_off_H[H]+yij], value);
```

with **no INF guard on either operand**. When one is `INF` (10000000) and the
other is a real negative energy, the sum lands just *below* INF and wins the
`min`. Measured: `fM2_real[58][70] = 9999750` where upstream has exactly
`10000000`.

Upstream's `mfe_multibranch_m2_fast()` guards each operand, so it never produces
such a value.

### What was checked

- It is **real and reproducible** — one cell on a 75 nt record, found the first
  time the verifier ran.
- It **matters for circular**, because `postprocess_circular()` tests
  `fM2_real[...] != INF` and would treat 9999750 as a two-branch decomposition
  that does not exist. Fixed by clamping at the fM2 store (`> INF/2 -> INF`),
  which is circular-only.
- **It was NOT fixed in the reduction itself**, deliberately. That value also
  feeds `DMLi`, consumed by `new_c_kernel` on the linear path, which is
  byte-identical to upstream across every test this project has — including
  400 x 5601 with `sha 7c0b3d633281` across 22 arms. Adding two INF tests to the
  inner loop of the largest GPU phase to fix a value the linear path evidently
  tolerates would trade a measured-good hot path for a theoretical one.

### A SIBLING OF THIS WAS FOUND AND FIXED, 2026-09-11 -- it is NOT this one

`fml_scan_kernel` (`hp_mb_loop.cu`) composed its two fML terms with the guard on
**one side only**:

```c
const int c_term = (e3p00[o+j] != INF) ? new_e[o+j] + e3p00[o+j] : INF;
const int e3     = (fp != INF) ? fp + en_i : INF;   // its own comment said
                                                    // "hazard 1: en_i NOT guarded"
```

so fML could carry `INF` minus a real energy. Invisible on int32 for the same
reason as below; **fatal** under `RNA_FML_INT16` + `--noClosingGU`, where
9999890 became a block baseline and tripped `__trap()`. Both are `fml_tadd()`
now. See `PORT_NOCLOSINGGU_SPEC.md`.

**That does not close this item.** The reduction below adds two fML values with
no guard of its own, which is a separate instance of the same family -- and the
measured 9999750 came from the reduction, not from fML. What the fix does change
is the *input*: the reduction's operands are now clean, so the only remaining
source of a near-INF `DMLi` is the reduction itself.

### The open question

**Can a near-INF `DMLi` bite `new_c_kernel`?** It evidently does not on any
input tested so far, but "evidently does not" is the same standing the `--nsp`
identities had for a year before an asymmetric spec broke them. The shape to
look for: a cell where `new_c` compares `DMLi1[j-1]` against a threshold or
adds to it, such that INF-minus-a-bit behaves differently from INF.

**What to do:** construct an input where an INF `fML` sits adjacent to a large
negative one in the reduction's range, and diff the linear fold against
upstream. If it holds, record *why* it cannot bite rather than that it has not.

---

## How to use this file

Add an entry when you notice something that (a) could change an answer, (b) you
are not going to chase now, and (c) would otherwise survive only in a commit
message. Record **what was actually checked**, not just the worry — an entry that
says "this looks dangerous" is worth much less than one that says "these five
sites were checked, they agree, and here is the one thing that would break them".
