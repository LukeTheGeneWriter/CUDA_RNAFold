# RNAfold option surface — what is accelerated, 2026-09-08

*All 60 CLI options in `src/bin/RNAfold.ggo`, classified against the library
guard (`mfe/cuda/engine.c`) and, where stated, against a measured run on
30 × 80–1240 nt. "Verified" means byte-identical GPU vs CPU on that input with
the route asserted; "assumed" means the guard covers it but nothing has run.*

---

## `--nsp` — GUARDED 2026-09-09 (was a live wrong answer)

**`--nsp=GA` was accelerated, changed the answer, and the GPU disagreed with the
CPU on 28 of 30 records — deterministically, worse on 27 and better on 1.** Not
ignored, *partially* applied, which is the worst of the three possibilities
because it looks like it is working. Same family as the `-C` defect.

**The guard now declines it** (`engine.c`, after the `energy_set` check). It
tests `md->pair[i][j] == 7`, **not** `md->nonstandards[0]` — the pair table is
where every route into the feature ends up, and the field is the one a copied or
hand-built `md` walks straight past. Default `BP_pair` (`pair_mat.h:21-30`) holds
only 0..6, so a 7 can mean nothing else. Bar:
`tests/mfe_cuda_guard.ts::test_guard_declines_nonstandard_pairs`, which covers
both routes and asserts the poke survives into the compound.

**The mechanism is now located**, and it is not in `engine.c` at all — it is
`int_loop.cu`'s `Energy()`, which resolves the two interior-loop pair types with
two shortcuts that are exact identities at default settings and stop being
identities under `--nsp`: no `0 → 7` promotion, and an index swap in place of
`rtype[]`. `rtype[7]` is *forced* to 7 at `model.c:1104`, which is what breaks
the second one. Full derivation, and what lifting the guard would take, in
`PORT_NSP_PARAMFILE_SCOPE.md` §1.

**Status: guarded and unbuilt.** No local toolchain was available on 2026-09-09,
so neither the guard nor its test has been compiled or run.

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
| `-c` / `--circ` | circular RNA — see §4, the story has changed |
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
| `--nsp` | **DECLINED 2026-09-09 — see the top of this file.** No longer ungoverned. |
| `-P` / `--paramFile` | **UNTESTED, and it cannot be guarded** — `vrna_params_load()` mutates library globals, so by fold-compound time `-P` has left no flag for the guard to see. Its assumptions belong in `load_param()` instead. **One is already fixed:** `MAX_NINIO` was a `#define` of 300 on the device, "checked" by an `assert(300 == 300)`, while the real `MAX_NINIO` is a writable global a parameter file overwrites (`params/io.c:671`) — now carried per-batch in `cuda_param_t`. Six more assumptions ranked in `PORT_NSP_PARAMFILE_SCOPE.md` §2.3; the four bars are §2.4. |
| `--ImFeelingLucky` | GPU differs from CPU — **but that is noise, not a defect.** Two CPU runs of the same flag also differ, so the backtracking really is stochastic and **a byte-identical bar cannot judge this option at all.** It needs a distributional bar, or none. Checked before reporting, because "GPU ≠ CPU" looked exactly like `--nsp` until the CPU was compared against itself. |
| `--batch` | **not reachable by this harness** — both sides produced *no output*, because `--batch` changes input parsing and the plain FASTA gave it nothing to do. Two empty outputs are not a match; scored as untested rather than passing. |
| `--helical-rise`, `--backbone-length` | accelerated and identical, **but did not bite** on the test input — so the comparison proved nothing. Needs an input where they change the answer. |
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
| accelerated, verified | 8 CLI options (+ `uniq_ML`, no CLI flag) |
| declined, route asserted | 20 + `--nsp` **(new 2026-09-09, route not yet asserted — unbuilt)** |
| neutral | 25 |
| **ungoverned** | **5, of which 1 is untested and high-risk (`-P`, and it is unguardable by construction) and 1 cannot be judged by a byte bar at all (`--ImFeelingLucky`)** |

**No live silent wrong answer is currently known on the option surface.** That
is the first time this file has been able to say so. It is a statement about
what has been *looked at*: `-P` is untested, `--helical-rise` /
`--backbone-length` have never been given an input where they bite, and
`--batch` is unreachable by the harness.
