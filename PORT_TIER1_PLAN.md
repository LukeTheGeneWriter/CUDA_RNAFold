# Plan: GPU support for salt, `uniq_ML`, and circular RNA

*Written 2026-09-05. These are the three cheapest accelerations on the board —
all memory or parameter work, none of them adding arithmetic to the hot kernel.
Designs and frozen vectors already exist; this is the sequencing and the
verification each needs.*

> **STATUS 2026-09-06 — see `PORT_SESSION_2026-09-06.md`.**
> `uniq_ML` **DONE** (fM1 rebuilt on the host from the fetched `c` triangle;
> `tests/mfe_cuda_fm1.ts` compares it to upstream cell for cell). Salt **DONE**
> (28 frozen vectors reproduced, GPU == CPU at 7 concentrations;
> `tools/verify_salt_parity.sh`). Circular **BLOCKED, no code written**:
> `postprocess_circular()` is `PRIVATE` in `mfe/mfe.c:103` and the batch path
> never enters `vrna_mfe()`, so it cannot be reached. Four of the five post-fill
> steps circular needs are already public; this is the fifth, and it is an
> upstream ask rather than an implementation task.
>
> One correction to the sequencing below: it says circular "depends on
> `uniq_ML`'s machinery". It does not — `md->circ` *suppresses* `fM1`
> (`dp_matrices.c:1323-1324`), so the two features never coexist.

**Order: `uniq_ML` → salt → circular.** Justified below; it is not the order of
value, it is the order of risk. `uniq_ML` is the smallest change that exercises
the pattern the other two need (produce one more matrix, prove it), salt touches
no matrix at all, and circular depends on `uniq_ML`'s machinery plus a
backtracking change that is the only genuinely novel part of the three.

---

## 0. First, a correction that changes what `uniq_ML` means

`uniq_ML` currently **passes through the driver** and produces correct MFE — the
recursion never reads `fM1`. But the sweep only initialises `fM1` to INF and
never fills it (`fill_arrays.c:271`, *"no kernel computes it and nothing fetches
it"*).

That is safe for RNAfold, which never reads `fM1`, and unsafe for
`vrna_mfe_batch()`, whose caller may go on to `vrna_subopt()`. The library guard
now declines `uniq_ML` for exactly that reason while the driver still allows it.

So "GPU support for `uniq_ML`" means: **fill `fM1`**, then lift the library
guard. That is real work, not a flag change.

---

## 1. `uniq_ML` — fill `fM1`

**What it is.** `fM1[i,j]` is the multibranch region with exactly one component,
computed by `vrna_mfe_multibranch_m1()` → `extend_fm_3p()`. Upstream fills it in
the same cell loop as `fML` (`mfe/mfe.c:497`).

**Why it is the right thing to do first.** It is the smallest instance of the
pattern circular also needs — *produce one additional triangular matrix from the
sweep and prove it cell-for-cell* — with no post-processing, no backtracking
change, and no new energy terms. Whatever goes wrong here goes wrong in the
cheapest possible setting.

**Work**
1. Allocate the `fM1` device buffer alongside `fML`, gated on `uniq_ML` so a
   default fold pays nothing.
2. Extend the multibranch kernel to emit `fM1` per cell. `extend_fm_3p()` is a
   3'-extension of `fML` — the same data the kernel already has in registers.
3. Fetch it in the per-record readback next to `c`/`fML`.
4. Update `gpu_bytes_per_file()` under the same gate.
5. Lift the `uniq_ML` decline in `mfe/cuda/engine.c`.

**Verification.** `RNA_ROW_VERIFY` extended to cover `fM1` — this is exactly
what that instrument is for, and it is already the check that proved the
`hc->mx` conversion over 161 695 cells. Then a `vrna_subopt()` comparison after
`vrna_mfe_batch()` with `uniq_ML` set, since that is the caller the guard was
protecting.

**Estimate:** one focused session. **Risk: low.**

---

## 2. Salt corrections

**What it is.** Six new `vrna_md_t` fields and five `vrna_param_t` fields, all
present in 2.7.2 already. Salt shifts hairpin, internal and stack energies; the
multibranch contribution is **already baked into `MLbase`/`MLclosing`/`MLintern`
at parameter-init time**, so the hot multibranch kernel needs *nothing*.

**Why second.** It touches no matrix and adds no memory — the cheapest of the
three in every dimension except that it is the first to move *energy* data
rather than structure. Doing it after `uniq_ML` means the upload plumbing has
just been exercised.

**Work**
1. Add `SaltStack` (1 int) and `SaltLoop[MAXLOOP+2]` (32 ints) to the uploaded
   parameter struct. **33 integers** — a textbook `__constant__` candidate, and
   this codebase's first use of it.
2. One added term in the hairpin path, one in the internal-loop path, one for
   the stack case.
3. Hairpins can exceed `MAXLOOP`, where upstream falls back to the closed form
   `vrna_salt_loop_int()` needing `exp`/`log`. **Do not port a transcendental
   into a kernel** — precompute the table to the batch's longest sequence
   (`n_max` ints, still kilobytes) and index it. The kernel then has no branch
   and no math, just an add.
4. Lift the `salt` decline in both guards.

**Verification.** `tests/salt/` is already frozen: 4 records at 60/120/200/300 nt
against pristine 2.7.2 at 0.05/0.1/0.2/0.5/1.021/2.0/5.0 M. Plus the invariant
that matters most and is already verified — **`--salt 1.021` is byte-identical
to no `--salt` at all**, so the feature cannot move the standing bar.

**Watch for:** the parameter-caching bug fixed today. Salt changes
`vrna_param_t`, so a batch at one salt followed by a batch at another is exactly
the shape that was silently wrong until this morning. Add a two-salt,
one-process case to the bar.

**Estimate:** ~50 device lines plus the table. **Risk: low**, given the frozen
vectors.

---

## 3. Circular RNA

**What it is.** `md->circ` allocates `fM2_real` (and suppresses `fM1`). The
sweep must produce `fM2_real`; upstream's `postprocess_circular()` then does the
rest on the host.

**The finding that makes it cheap, already verified:** `fM2_real[i,j]` is
*exactly* `min_k(fML[i,k] + fML[k+1,j])` — the ML_ML_ML decomposition the sweep
already computes as `DMLi` and discards. Measured over 8853 cells, zero
mismatches (`tests/upstream/circ_fm2_probe.c`). And
`modular_decomposition.cu:349` already declares `d_dml //DMLi`: the kernel
produces these values per row today.

**Why last.** Two reasons, neither about difficulty of the fill:
- it needs `uniq_ML`'s "extra matrix" machinery, so it is cheaper after step 1;
- **backtracking is the only genuinely novel part of the three.**
  `postprocess_circular()` seeds `bt_stack` itself rather than starting from
  `f5[n]`, so the backtrack entry has to accept a pre-seeded stack. Everything
  else here is plumbing.

**Work**
1. Persist `d_dml` into a triangular `fM2_real` buffer per row, gated on
   `md->circ`. **No new arithmetic** — the value is already computed.
2. `gpu_bytes_per_file()` grows by one triangular matrix; the chunk width
   shrinks correspondingly. This is the real cost.
3. Readback alongside `c`/`fML`.
4. Call upstream's `postprocess_circular()` per record on the existing backtrack
   thread pool. **Do not reimplement it** — reusing it verbatim is what keeps
   the bar meaningful.
5. Resolve the pre-seeded `bt_stack` question against
   `vrna_backtrack_from_intervals()`.
6. Lift the `circ` decline in both guards and retire the backstop for it.

**Verification.** `tests/circ/` is frozen, and carries a **second reference with
the feature off** — every record differs from its linear answer by 5.0–8.4
kcal/mol, so a path that silently returns the linear answer (the fork's actual
historical bug) is detectable rather than merely "different".

**Estimate:** two sessions, most of it in step 5. **Risk: medium**, concentrated
entirely in backtracking.

---

## What is shared, and worth building once

All three need the same thing: **an additional triangular matrix produced by the
sweep, gated on a model detail, budgeted, and read back.** Doing `uniq_ML` first
is partly to build that path properly — a gated extra-matrix mechanism that
circular then reuses rather than reinvents.

Two rules to carry through all three, both learned the hard way in this port:

- **`RNA_ROW_VERIFY` must cover the new matrix.** It is the only instrument that
  checks device against host per cell, and end-to-end output is strictly weaker
  — it proves values right *where they are read*, not everywhere.
- **Guard-lifting is the last step, never the first**, and the library guard is
  lifted only when everything a caller might read is filled — not when the
  answer we happened to check came out right.

## What this does not include

G-quadruplexes, dangle models other than 2, `noLP`, multistrand, comparative
folding and the sliding-window path. Those are Tier 2 and 3 in
`PORT_ACCELERATION_SCOPE.md`; none is a prerequisite for these three, and all
three of these keep their CPU route as the fallback throughout.
