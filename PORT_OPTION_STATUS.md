# RNAfold option surface — what is accelerated, updated 2026-09-10

*All 60 CLI options in `src/bin/RNAfold.ggo`, classified against the library
guard (`mfe/cuda/engine.c`) and, where stated, against a measured run on
30 × 80–1240 nt. "Verified" means byte-identical GPU vs CPU on that input with
the route asserted; "assumed" means the guard covers it but nothing has run.*

---

## `--nsp` — GUARD LIFTED 2026-09-10, because the CAUSE was fixed

**`--nsp` is now ACCELERATED and byte-identical.** It was declined from
2026-09-09 to 2026-09-10 while a live wrong answer (28 of 30 records, worse on
27, better on 1) was diagnosed. The guard is gone because `int_loop.cu`'s
`Energy()` was fixed, not because the risk was re-assessed.

**The fix is four lines.** `Energy()` resolved the two interior-loop pair types
as `type = Ptype(i,j)` raw and `type_2 = Ptype(q,p)` — no `0 → 7` promotion, and
an index swap in place of `rtype[]`. Both are exact identities at default
settings (`rtype[]` is *built* as `rtype[pair[i][j]] = pair[j][i]`, and a ptype-0
cell is refused by the hard-constraint mask before `Energy()` runs), which is why
the port was byte-identical for a year without them. `--nsp` breaks the second at
`model.c:1104`, where `rtype[7]` is **forced** to 7. It now promotes and applies
`rtype[]`, so all three of the fork's type-resolution sites agree with upstream
and with each other — under `--nsp` the fork previously disagreed with *itself*.

`rtype[8]` is carried per batch in `cuda_param_t`, appended last so no offset
above it moves — the same shape as the `MAX_NINIO` and salt fields.

**Measured, RTX 3050, 30 records of 45–1200 nt**, GPU vs the same binary with the
accelerator off: byte-identical for `--nsp=GA`, `--nsp=-GA` and `--nsp=-AC,GA`,
and in combination with int16, `--noLP`, `--noGU`, `-4`, `--salt` and `-T`.
**Nine arms, every one of which BITES** (54–60 lines against the unflagged fold).

**RED without the fix, on the same binary:** `--nsp=GA` differs on 20 lines,
`--nsp=AC` on 22.

### The symmetric form is a FALSE GREEN — this is the part to remember

`--nsp="-GA"` sets **both** directions to 7, which restores
`rtype[pair[p][q]] == pair[q][p]` and masks the divergence completely. On the
broken binary it gave **0 differing lines** while the asymmetric specs gave 20
and 22. A symmetric-only test passes against the bug.

`tests/mfe_cuda_nsp.ts` (3 cases, replacing the guard test) therefore leads with
two **asymmetric** specs and asserts the pair table really is asymmetric rather
than trusting the spec string. Confirmed RED before the green was believed: with
`Energy()` reverted, both asymmetric cases fail and the symmetric one passes.

**Its fixture is not hand-picked either.** The first version used five 80–92 nt
sequences chosen by eye and **passed against the broken binary** — because
`--nsp` changing the answer and the GPU/CPU divergence being *exercised* are two
different things, and `bites > 0` only establishes the first. The four sequences
now used were selected by folding 30 random 45–1200 nt sequences on a
deliberately broken build and keeping the shortest that disagreed.

---

## 1. Accelerated — verified byte-identical

| option | note |
|---|---|
| *(default)* | the reference bar |
| `-T` / `--temp` | affine on the parameters at init; 37 °C is the identity |
| `--salt` | hot multibranch kernel needed nothing (`PORT_SALT_SPEC.md`) |
| `--noLP` | accepted `9d3f63cc`; `+RNA_SLOT_FLOW` fixed `364f92f6` |
| `--noGU` | |
| `-4` / `--noTetra` | **verified today** — identical, and it bites |
| `-d2` / `--dangles=2` | the only accepted dangle model |
| `-p`, `--MEA`, `--bppmThreshold`, `--betaScale`, `-S` | the **MFE fill** is accelerated; the partition function itself runs on the CPU afterwards |

Also accepted by the guard but **unreachable from the CLI**: `uniq_ML` (only
`--ImFeelingLucky` sets it, and that also enables stochastic backtracking, so no
byte-comparable bar exists). Covered by `tests/mfe_cuda_fm1.ts` instead.

## 2. Declined — correctly routed to the CPU, same answer

Verified as *CPU route asserted*, not merely observed (`verify_option_parity.sh`,
18/18):

| option | guard reason |
|---|---|
| `-c` / `--circ` | circular RNA — see §5, the story has changed |
| `-g` / `--gquad` | G-quadruplexes (`PORT_GQUAD_SPEC.md`) |
| `-d0`, `-d1`, `-d3` | dangle model other than 2 |
| `--noClosingGU` | |
| `-C`, `--enforceConstraint`, `--canonicalBPonly`, `--batch` | hard structure constraints — closed a live defect |
| `--shape`, `--shapeMethod`, `--shapeConversion` | soft constraints |
| `--sp-data`, `--sp-strategy`, `--sp-preprocess` | soft constraints |
| `--motif` | unstructured domains (ligand motifs) |
| `--commands` | auxiliary grammar rules |
| `-m` / `--modifications`, `--mod-file` | host callbacks in a kernel |
| `--energyModel` | non-default energy set |
| `--maxBPspan` *(when < length)* | restricted base pair span |
| *(multi-strand input)* | `fms5`/`fms3` are a rewrite |
| *(comparative / alignment)* | different recursion entirely |

## 3. Neutral — never reach the recursion (25)

`-v`, `-i`, `-o`, `-j`, `--unordered`, `--noconv`, `--auto-id`, `--id-prefix`,
`--id-delim`, `--id-digits`, `--id-start`, `--filename-delim`,
`--filename-full`, `--log-level`, `--log-file`, `--log-time`, `--log-call`,
`--benchmark`, `--bm-output`, `--bm-output-append`, `--bm-rm-pk`, `--bm-rm-nc`,
`--noPS`, `--noDP`, `-t` / `--layout-type`.

I/O, identifiers, logging, plotting and post-hoc accuracy scoring. They compose
with the GPU path because they never touch it.

## 4. Ungoverned — no guard decision, and only partly tested

**This is the category that matters, and it is where `--nsp` was found.**

| option | status |
|---|---|
| `--nsp` | **ACCELERATED 2026-09-10 — see the top of this file.** No longer ungoverned, and no longer declined: the `Energy()` divergence behind it is fixed. |
| `-P` / `--paramFile` | **ALL FOUR BARS NOW RUN, 2026-09-10 — and bar 4 found a second live silent wrong answer.** `RNA_FML_INT16=1` plus a parameter file with large `stack` magnitudes folded **wrong on 8 of 12 records, by up to 31.8 kcal/mol**: the int16 offset bound is derived from the DEFAULT table's −340, the pack kernel's `assert(0)` guard was a **no-op under `-DNDEBUG`**, and its device `printf` corrupted stdout with 48 781 lines. Fixed by vetting the loaded table in `par_mfe()` **before `init_gpu()` commits the mode**, declining to int32. Bars 1–4 + both over-tightening checks: **13/13**. **Still cannot be guarded** — `vrna_params_load()` mutates library globals, so by fold-compound time `-P` has left no flag for the guard to see. Its assumptions belong in `load_param()` instead. **One was a LIVE wrong answer and is fixed:** `MAX_NINIO` was a `#define` of 300 on the device, "checked" by an `assert(300 == 300)`, while the real `MAX_NINIO` is a writable global a parameter file overwrites (`params/io.c:671`). Measured 2026-09-09 — with the stock file GPU == CPU, but with only the NINIO maximum moved 300 → 80 the **pre-fix binary differs on 9 of 12 records**; the fixed one is identical. Six more assumptions were ranked in `PORT_NSP_PARAMFILE_SCOPE.md` §2.3; **four of them (`lxc` narrowed to float, the dead mismatch/dangle tables, special-hairpin strides, one-parameter-set-per-batch) are still unchecked** against a non-default table. |
| `--ImFeelingLucky` | GPU differs from CPU — **but that is noise, not a defect.** Two CPU runs of the same flag also differ, so the backtracking really is stochastic and **a byte-identical bar cannot judge this option at all.** It needs a distributional bar, or none. Checked before reporting, because "GPU ≠ CPU" looked exactly like `--nsp` until the CPU was compared against itself. |
| `--batch` | **not reachable by this harness** — both sides produced *no output*, because `--batch` changes input parsing and the plain FASTA gave it nothing to do. Two empty outputs are not a match; scored as untested rather than passing. |
| `--backbone-length` | **CLOSED 2026-09-10.** Bites at the DNA value 6.76 **under `--salt`** (24 lines) and the GPU matches the CPU. The earlier "did not bite" was a missing `--salt`: the geometry feeds the salt model only. Bar: `tools/verify_paramfile_bars.sh`. |
| `--helical-rise` | **PARTLY CLOSED 2026-09-10.** It is wired and parity holds where it bites — but it does **not** bite at the DNA value 3.4 (2.8 → 3.4 moves no integer energy at 62–401 nt); it bites at 10 and 100. So `-P DNA` closes `--backbone-length` and **not** this. "Did not bite" was a property of the value, not of the plumbing. |
| `--maxBPspan` *(== length)* | identical, did not bite; the restricted case is declined |

## 5. Circular (`-c`) — the blocker is narrower than recorded

`PORT_CIRC_SPEC.md` and `PORT_UPSTREAM_PROPOSAL.md` §3.3 record circular as
blocked on `postprocess_circular()` being `PRIVATE` (`mfe/mfe.c:103`). Re-reading
`mfe.c` today, that is **only true of the batch path**:

```c
    if (!handled)
      energy = fill_arrays(fc, ms_dat);

    if (fc->params->model_details.circ)
      energy = postprocess_circular(fc, bt_stack);      /* mfe.c:322 */
```

`postprocess_circular()` runs **after** the engine branch, on whatever matrices
the engine filled, unconditionally on `md.circ`. The seam's own comment says so:
*"Everything after this point — backtracking, circular post-processing, output —
is unchanged either way."*

So for the **single-fold path** (`vrna_mfe()` with the engine attached), circular
post-processing is already done for us by upstream, with no new API. What
genuinely bypasses it is **`vrna_mfe_batch()`**, which hands the whole batch to
the backend and never reaches `mfe.c:322`.

**Two consequences.**

1. The upstream ask should be re-scoped. It is not "please make
   `postprocess_circular()` public so we can do circular RNA" — it is "our batch
   entry point cannot reach the post-processing that the per-fold path gets for
   free". Narrower, more defensible, and it invites the obvious counter-question:
   should `vrna_mfe_batch()`'s fallback loop be the answer instead?
2. **There may be a route with no upstream change at all**: let the batch
   backend *decline* circular fold compounds, so `vrna_mfe_batch()`'s own
   per-record loop calls `vrna_mfe()` — which does the post-processing. Circular
   inputs would fold correctly at CPU speed while everything else is accelerated,
   instead of being refused outright.

Neither is verified. What is verified is that the recorded blocker does not
apply to the path most library callers use, and `PORT_CIRC_SPEC.md`'s §"cost a
chunk width" analysis (fM2 already computed by the hot kernel and discarded) is
unaffected either way.

---

## Summary

| | count |
|---|---|
| accelerated, verified | **10** CLI options (+ `uniq_ML`, no CLI flag) — `--nsp` and `--backbone-length` added 2026-09-10 |
| declined, route asserted | **20** — `--nsp` moved OUT on 2026-09-10 when its cause was fixed |
| neutral | 25 |
| **ungoverned** | **3 — `-P` (unguardable by construction, but all four bars now run), `--ImFeelingLucky` (no byte bar can judge it) and `--batch` (unreachable by the harness)** |

**No live silent wrong answer is currently known on the option surface.** That
is the first time this file has been able to say so. It is a statement about
what has been *looked at*, and that set grew on 2026-09-10: all four `-P` bars
now run (13/13), `--backbone-length` is closed, and `--helical-rise` is closed
wherever it bites. **`-P` found a second live silent wrong answer on the way**
(int16 + a large-`stack` file, 8 of 12 records, up to 31.8 kcal/mol) — now fixed
and barred. What remains unlooked-at: four of `-P`'s seven ranked assumptions,
`--batch` (unreachable by the harness), and `--ImFeelingLucky` (no byte bar can
judge it).
