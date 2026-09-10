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

## How to use this file

Add an entry when you notice something that (a) could change an answer, (b) you
are not going to chase now, and (c) would otherwise survive only in a commit
message. Record **what was actually checked**, not just the worry — an entry that
says "this looks dangerous" is worth much less than one that says "these five
sites were checked, they agree, and here is the one thing that would break them".
