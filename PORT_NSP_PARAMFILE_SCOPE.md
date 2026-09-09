# `--nsp` and `-P/--paramFile` — scope

*Scoped 2026-09-09 by code reading against `port27` @ `82fdbb07`. Nothing here
is measured yet; every claim below names the line it came from so the
discriminating run is cheap. Opened because `PORT_OPTION_STATUS.md` §4 lists
both as ungoverned, `--nsp` as a live wrong answer and `-P` as untested.*

**They are related — both reach the device only through the energy tables and
the pair table — and the conclusion for each is different.** `--nsp` has a
located mechanism and wants a guard. `-P` has no guard to write, because the
option is invisible at the guard; it wants assertions at upload time.

---

## Part 1 — `--nsp`

### 1.1 What is NOT wrong (ruled out by reading)

The recorded note "`nonstandards` appears zero times in `engine.c`" is about the
**guard**, and it invited the wrong hypothesis — that the device never sees the
non-standard pairs at all. It does. All of the following were checked and are
correct:

| assumption | evidence |
|---|---|
| `vrna_md_set_nonstandards()` calls `vrna_md_update()` | `model.c:336` — so `md->pair` really does carry the 7s before any fold compound is built |
| the pair table reaches the device | `int_loop.cu:368`, `hp_mb_loop.cu:341` — the whole `[8][8]` corner of `md->pair`, uploaded per batch; 7 passes the `< 8` assert |
| every energy table is uploaded at full `[NBPAIRS+1]` extent | `load_param()` `int_loop.cu:254`, `load_param2()` `hp_mb_loop.cu:220` — `stack`, `mismatchI/1nI/23I`, `int11/21/22`, `mismatchH`, `mismatchM`, `MLintern` all include row/column 7 |
| `rtype` reaches the device | `hp_mb_loop.cu:232`, all 8 entries, so `rtype[7] = 7` is present |
| the device's `ptype` replica | `stub2.h:770 rnafold_ptype()` == `vrna_ptypes()` (`alphabet.c:240`) with noLP off: both are just `md->pair[S[i]][S[j]]` |
| the device's hard-constraint replica | `stub2.h rnafold_hc_cell()` == `default_pair_constraint()` (`hard.c:761`) case for case, including the `type==3 or 4` GU branch — type 7 falls to `default:` on **both** sides |
| the noLP stack path | `stack_row_kernel()` `hp_mb_loop.cu:1318-1325` does the 0-to-7 fixup *and* `P->rtype[]`, in the correct order, with a comment explaining why |

So the bitmask is right and the tables are right. That is worth stating plainly,
because it moves the search to one function.

### 1.2 The mechanism — `Energy()` in `int_loop.cu`

`int_loop.cu:993-996`, the live interior-loop device path:

```c
const unsigned char type   = Ptype(S,pair_,H,nfiles,i,j);   /* raw           */
const unsigned char type_2 = Ptype(S,pair_,H,nfiles,q,p);   /* index-swapped */
```

Upstream's interior loop computes the same two values as:

```c
type   = vrna_get_ptype(ij, ptype);          /* tt == 0 ? 7 : tt          */
type_2 = rtype[vrna_get_ptype(pq, ptype)];   /* promote FIRST, then rtype */
```

**Two divergences, both invisible at default settings:**

1. **No 0-to-7 promotion.** `vrna_get_ptype()` (`alphabet.c:482`) maps a zero
   ptype to 7; the device uses the raw value. The commented-out line directly
   above (`int_loop.cu:994`) shows the original shape — `rtype[ptype[pq]]` — so
   the promotion was dropped when the array read became a recompute.
2. **`type_2` is obtained by swapping the index order instead of applying
   `rtype[]`.** `pair[S[q]][S[p]]` in place of `rtype[pair[S[p]][S[q]]]`.

At default settings these are *identities*, which is exactly why the port has
been byte-identical for a year. `md->rtype[]` is **built from** `md->pair[]`
(`model.c:1098-1100`, `rtype[pair[i][j]] = pair[j][i]`), so
`rtype[pair[p][q]] == pair[q][p]` holds by construction; and a cell with ptype 0
is refused by the hard-constraint mask before `Energy()` ever runs, so the
promotion never fires.

**`--nsp` breaks the first identity**, at `model.c:1103-1104`:

```c
  /* handle special cases separately */
  md->rtype[0]  = 0;
  md->rtype[7]  = 7;
```

`rtype[7]` is **forced** to 7, overriding whatever the derivation loop wrote.
With an asymmetric spec — `--nsp=GA`, no leading `-`, which is what the audit
ran — `md->pair[G][A] = 7` while `md->pair[A][G]` stays **0**. So:

| | upstream | device |
|---|---|---|
| `type_2` for that enclosed pair | `rtype[7]` = **7** | `pair[A][G]` = **0** |

Row 7 of `stack`/`int11`/`int21`/`int22`/`mismatchI` is the non-standard row
(`NST`/`NSM`, `default.c:53-56`); row 0 is not. The device indexes a different
row of five energy tables. **That is a different energy model, partially
applied** — which is the reported signature exactly: worse on 27, better on 1,
deterministic.

`--nsp="-GA"` (symmetric) sets both directions to 7 and would mask divergence
(2) while leaving (1); a symmetric-vs-asymmetric A/B is therefore the cheapest
confirmation of this diagnosis.

**Corroboration inside the fork:** `stack_row_kernel()` does the promotion and
uses `rtype[]`; `hp_mb_3p_kernel()` uses `rtype[]` with a documented raw-index
convention; only `Energy()` does neither. Under `--nsp` the fork's own three
type-resolution sites disagree with each other.

**Status: a hypothesis from reading, not a measurement.** It names lines and it
predicts a specific asymmetry, so it is falsifiable in one run.

### 1.3 The decision does not wait on the diagnosis

Guard it now. `-c`/`-g`/`--noClosingGU` precedent, one `DECLINE` in
`vrna_cuda_engine_supports()`.

**Test the effect, not the field.** The `-C` lesson was "test `hc->depot`, not
`hc->mx`" — here it is the mirror: test `md->pair`, not `md->nonstandards`.
Default `BP_pair` (`pair_mat.h:21-30`) contains only 0..6, so:

```c
  /* A 7 anywhere in the pair table means non-standard pairs are enabled,
   * however they were requested -- md->nonstandards, the deprecated global,
   * or a hand-built md. See PORT_NSP_PARAMFILE_SCOPE.md. */
  for (i = 0; i <= MAXALPHA; i++)
    for (j = 0; j <= MAXALPHA; j++)
      if (md->pair[i][j] == 7)
        DECLINE("non-standard base pairs (--nsp)");
```

catches every route into the feature, including the ones a library caller can
take that never touch `md->nonstandards[]`. Checking the field alone would be a
guard that a `vrna_md_update()`-then-copy sequence walks straight past.

**Then fix `Energy()` anyway** — the two divergences are latent defects
independent of `--nsp`, and fixing them is ~4 lines. Guard first (it is the
correctness bar), lift the guard only against a measured byte-identical run.

### 1.4 Bars

- `tools/verify_option_parity.sh --nsp=GA` with the **CPU route asserted** —
  this is what the guard makes true.
- Before/after the `Energy()` fix: `--nsp=GA` *and* `--nsp="-GA"` against
  upstream, 30+ records, **mixed lengths** (the 2026-09-08 lesson: uniform-length
  fixtures cannot see state that crosses records).
- Add `--nsp` to `tools/verify_option_matrix.sh` once it is accepted, not
  before — a declined option contributes nothing to a pairwise matrix.

---

## Part 2 — `-P` / `--paramFile`

### 2.1 Why this is not a guard problem

`ggo_get_read_paramFile()` (`gengetopt_helpers.h:137`) calls `vrna_params_load()`,
which **mutates library globals**. By the time a fold compound exists, `-P` has
left no flag behind: `vrna_params()` has already baked the file's values into
`P`, and `md` looks ordinary. **The guard cannot see `-P`,** and a `DECLINE` for
it cannot be written.

That is the whole difference from `--nsp`. The right shape is **assertions at
upload time against `P`**, in `load_param()`/`load_param2()`, where the device's
assumptions actually live.

### 2.2 The defect: `MAX_NINIO` is hardcoded, and its check cannot fail

`MAX_NINIO` is **not a constant.** It is a public, writable library global
(`default.c:70`, declared `extern` at `default.h:70`) and a parameter file
overwrites it (`io.c:671`, the `NINIO` block's third value; `io.c:1601` writes it
back out).

The device hardcodes it:

```
src/ViennaRNA/mfe/cuda/int_loop.cu:42    #define MAX_NINIO 300
src/ViennaRNA/mfe/cuda/int_loop.cu:330   assert(MAX_NINIO == 300); //ViennaRNA/energy_par.c
```

The `#define` shadows nothing — `default.h` is not included here — so line 330
expands to `assert(300 == 300)`. **It is the next entry on the "check that could
not fail" list**, and it is precisely the kind this project keeps finding: it
reads as a guard against the exact hazard it does not test.

A parameter file with a non-default NINIO maximum is therefore a **silent wrong
answer on the interior-loop path** — the same family as `--nsp` and `-C`, reached
by a far more commonly used flag.

**Fix:** include `ViennaRNA/params/default.h` in `int_loop.cu`, drop the
`#define`, and pass the live global through `cuda_param_t` alongside `ninio2`
(one `int`, and the struct already carries padding). If passing it is judged out
of scope, the *minimum* is to make the check real — refuse in `load_param()` when
`MAX_NINIO != 300`. That is an honest stopgap, and it fires rather than lies.

### 2.3 The rest of the `-P` surface, in risk order

| # | assumption | where | what `-P` does to it |
|---|---|---|---|
| 1 | `MAX_NINIO == 300` | `int_loop.cu:42,330` | **overwritten by the file — §2.2** |
| 2 | int16's offset bound is `B/2 x 340` | `INT16_FML_SCOPE.md:303-313` | 340 is the most negative **default** `stack37` entry. `-P` replaces `stack37`, so the bound is **unproven**. Already flagged at `INT16_FML_SCOPE.md:446`. The fix is cheap and better than a refusal: compute the bound from `P->stack` at `load_param()` (a min over 64 ints) instead of a literal. Until then `RNA_FML_INT16` plus a non-default parameter file must be refused. |
| 3 | `lxc` fits a float | `int_loop.cu:260`, `hp_mb_loop.cu:231` | narrowed to `float` on both uploads while the host keeps `double` — and `hp_mb_loop.cu:762` deliberately calls `double log()` "to match host precision". The default 107.856 round-trips; an arbitrary file value need not. Cheap check: assert `(double)(float)P->lxc == P->lxc`, or widen the field. |
| 4 | `mismatchExt`, `dangle5`, `dangle3` are dead | `int_loop.cu:79,88-89`; `hp_mb_loop.cu:808` | argued dead for `dangles=2` with `cp==-1`, and `-P` does not change that argument. But the argument was made against the default table and should be **re-asserted**, not assumed, in the same pass. |
| 5 | special-hairpin strings fit fixed extents | `load_param2()` `hp_mb_loop.cu:224-229` | `Tetraloops[1401]`/`Triloops[241]`/`Hexaloops[1801]` are fixed in `vrna_param_t` too, so the copy is sound. **But the device scans them with hardcoded strides and terminators** (`hp_mb_loop.cu:780-820`, `off+=9` / `off+=6` / `off+=7`). A file with a different *count* of special loops is fine; the stride assumption is the thing to assert. |
| 6 | one parameter set per batch | `sanity()` `int_loop.cu:225-251` | checks `MLbase`, `TerminalAU`, `ninio[2]`, `lxc`, `md->pair` only. Not reachable from RNAfold (one file per run) but **is** reachable from `vrna_mfe_batch()`, which is the API this port exists to accelerate. Worth widening to a `memcmp` of the uploaded structs. |
| 7 | `-P DNA` is just another table | `gengetopt_helpers.h:140-143` | it also calls `set_salt_DNA(md)`, which moves `salt`, `saltDPXInit`, **`helical_rise` and `backbone_length`**. The last two sit in `PORT_OPTION_STATUS.md` §4 as "accelerated and identical, but did not bite". **`-P DNA` is the input that makes them bite** — so it is one test that closes two open entries. |

### 2.4 Bars

1. **`-P` with the stock file re-read from disk** (`misc/rna_turner2004.par`).
   Must be byte-identical to no `-P` at all — this is the self-comparison shape,
   and it isolates the *loading path* from the *values*.
2. **`-P DNA`** — byte-identical vs upstream, and it exercises the salt and
   geometry couplings (§2.3 #7).
3. **A deliberately perturbed file**: take the stock `.par`, change **only** the
   NINIO maximum, and fold. It must disagree with the current binary
   (demonstrating §2.2 is live) and agree after the fix. *Confirm the bar is RED
   on the unfixed build before accepting it* — the 2026-09-08 rule.
4. Same file, perturbed `stack` row instead, with `RNA_FML_INT16=1`, to exercise
   the bound in §2.3 #2.

---

## What to do, in order

1. ~~**Guard `--nsp` on `md->pair[i][j] == 7`.**~~ **DONE 2026-09-09,
   UNBUILT** — `engine.c` after the `energy_set` check, plus
   `tests/mfe_cuda_guard.ts::test_guard_declines_nonstandard_pairs` covering
   both the `vrna_md_set_nonstandards()` route and a direct pair-table poke that
   leaves `md->nonstandards` empty. The second case is the one that fails if the
   guard is ever "simplified" to test the field.
2. ~~**Make `MAX_NINIO` real.**~~ **DONE 2026-09-09, UNBUILT** — the `#define`
   is gone, `params/default.h` is included explicitly, and the live global is
   carried per batch as `cuda_param_t::max_ninio` (appended last, so no offset
   above it moves) and passed to `IntLoop_X()` in place of the macro. The
   `assert(300 == 300)` in `init_gpu2()` is deleted rather than repaired: with
   the value now flowing from the global there is nothing left for it to assert.
3. ~~**Run the four `-P` bars** (§2.4).~~ **Bars 1 and 3 DONE 2026-09-09** —
   both pass, and bar 3 is RED on the pre-fix build, so item 2 was a live defect
   and not a tidy-up. **Bars 2 (`-P DNA`) and 4 (perturbed `stack` row under
   `RNA_FML_INT16`) are still open**, and bar 2 is the one that also closes
   `--helical-rise` / `--backbone-length`.
4. **Fix `Energy()`'s two type divergences** (0-to-7 promotion, `rtype[]` instead
   of the index swap), then re-measure `--nsp` and lift the guard only on a
   byte-identical result over mixed lengths.
5. **Derive the int16 bound from `P->stack`** rather than the literal 340.

### Measured 2026-09-09 — both fixes verified on hardware

Built and run in WSL on an **RTX 3050 Laptop GPU** (nvcc 12.4, `sm_86` in the
gencode list), `~/port27head` at `8d2b174e`, 12 records of **mixed** length
62-401 nt. Reference is the same binary with the accelerator off, which isolates
it as the only variable.

**`make check`: 146/146, 0 fail, 0 error** (145/145 before, plus the new guard
case).

**The `--nsp` guard, with the route asserted:**

| | route | vs CPU |
|---|---|---|
| default | **GPU** (1 sweep) | identical |
| `--nsp=GA` | **CPU** | identical |
| `--nsp="-GA"` | **CPU** | identical |

`--nsp` bites on this input (8 lines differ from an unflagged fold), so the
comparison is not vacuous, and `default` still routes to the GPU, so the guard
did not over-tighten.

**And the bar goes RED without the guard.** With the `DECLINE` condition
neutered and the library rebuilt, `--nsp=GA` takes the **GPU** route and
**differs from the CPU on 8 lines**, while `default` stays identical. That both
reproduces the reported defect on this hardware and proves the check can fail —
the 2026-09-08 rule, after the noLP regression test that passed against a
deliberately broken binary.

**The `MAX_NINIO` fix — §2.4 bars 1 and 3, and the defect was live:**

| binary | `-P` file | route | vs CPU |
|---|---|---|---|
| fixed | *(none)* | GPU | identical |
| fixed | stock `rna_turner2004.par` | GPU | identical |
| fixed | NINIO max 300 → **80** | GPU | identical |
| **old hardcoded 300** | stock | GPU | identical |
| **old hardcoded 300** | NINIO max **80** | GPU | **DIFFERS on 9 lines** |

Bar 1 passes, so the loading path itself was never the problem. The perturbation
bites (9 lines differ on the CPU side between stock and NINIO-80). And the
pre-fix binary is wrong **only** on the perturbed file — which is the signature
of a hardcoded constant, and confirms §2.2 was a live silent wrong answer for
`-P`, not a hypothetical.

Both experiments patch, rebuild, measure, then restore and rebuild; `git diff`
on `engine.c` and `int_loop.cu` is empty afterwards. Scripts are in the session
scratchpad, not the tree — they should be folded into
`tools/verify_option_parity.sh` (an `--nsp` CPU-route row) and a new
`tools/verify_paramfile_parity.sh`.

**One trap worth recording.** The first `make check` reported **147 total, 1
FAIL, 1 ERROR** on `test_guard_declines_nonstandard_pairs` — and the guard was
fine. The red-team script had patched `engine.c` and rebuilt the library *while
that `make check` was still running*, so the test linked against the neutered
build. A new shape of the stale-binary trap this project keeps meeting: not an
old binary, but a **concurrently mutated** one. Never run a red-team rebuild
alongside a test suite in the same tree.
