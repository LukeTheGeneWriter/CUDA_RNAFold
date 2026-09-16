# RNAfold option surface: acceleration state and guard state

*All **60** options in `src/bin/RNAfold.ggo`, audited 2026-09-11.*

***Gate 3 has ONE entry, and it is a different kind of entry.*** *The three
options that used to be backstopped -- `-g`, `-c`, `--noClosingGU` -- were all
retired by implementation on 2026-09-10/11. `fill_arrays.c` was empty for a few
hours, then gained a backstop on **dangle models 1 and 3** when 0 was
implemented (2026-09-11). Everything that left the list was missing arithmetic;
what is on it now needs missing DP **state**, which is why it is not simply the
next thing to write.*

*Most rows are **measured**, not argued: `tools/verify_option_parity.sh` runs 40
checks over 30 mixed-length records and asserts, for each, that the answer is
byte-identical to the same binary with the accelerator off **and** that the run
took the route it claims. Re-run 2026-09-11 against `7fa85d4e`: **all 40 green**,
`the CUDA build matches the CPU build across the option surface`, with
`noClosingGU` now in the ACCELERATED block at `1 sweeps, 30/30 records`. Rows
marked **(asserted)** were read from the guard rather than run — they are named
so the distinction stays visible.*

---

## There are THREE gates, and they are not the same gate

This was not written down before, and it changes how to read every "declined"
row.

| # | gate | where | what it is for |
|---|---|---|---|
| **1** | `gpu_path_usable()` | `src/bin/RNAfold.c` | **an optimisation.** Decides whether the driver registers the backend at all. |
| **2** | `vrna_cuda_engine_supports()` | `src/ViennaRNA/mfe/cuda/engine.c` | **the authority.** Consulted per fold compound; a decline falls back to upstream's own `vrna_mfe()`. |
| **3** | `VRNA_CUDA_BACKSTOP` | `src/ViennaRNA/mfe/cuda/fill_arrays.c` | **a tripwire.** `exit()`s if an unsupported model reaches the sweep — i.e. if 1 and 2 both have a hole. |

**Gate 1 is a strict subset of gate 2, and that is correct.** `RNAfold.c`'s
comment claims it "mirrors" gate 2; it does not, and it does not need to. Three
things gate 2 declines that gate 1 never checks:

| declined by gate 2 only | consequence |
|---|---|
| `logML` | no CLI flag exists, so unreachable from RNAfold |
| multistrand / comparative compounds | not constructible from RNAfold's input |

**A hole in gate 1 costs performance. A hole in gate 2 costs correctness.**
**Nothing is covered by all three any more** -- `-c` and `--noClosingGU` were the
last two and both shipped on 2026-09-11. Gate 3 is not obsolete; it is the reason
"unreachable" was enforced rather than assumed for a year. It simply has nothing
left to watch.

---

## The table

**ACCEL** — folds on the GPU, byte-identical to the CPU path ·
**DECLINED** — routes to the CPU, byte-identical ·
**NEUTRAL** — never reaches the recursion; the fold is still accelerated ·
**UNREACHABLE** — cannot be exercised from this CLI

| option | acceleration | guard | evidence |
|---|---|---|---|
| *(default)* | **ACCEL** | — | measured |
| `-T` / `--temp` | **ACCEL** | — | measured, 37 and 25 °C |
| `-d2` / `--dangles=2` | **ACCEL** | — | measured |
| **`-d0` / `--dangles=0`** | **ACCEL** *(new, 2026-09-11)* | — | measured, incl. int16/chunked/`-g`/`-c`/`-p`; `tests/mfe_cuda_dangles.ts` |
| `-d1`, `-d3` | DECLINED | **1, 2, 3** *(the only backstopped option)* | **measured 2026-09-16**: gate 3 fires, see below |
| `-p` / `--partfunc` | **ACCEL** *(MFE fill)* | — | measured, `-p` and `-p0` |
| `--MEA` | **ACCEL** *(MFE fill)* | — | measured |
| `--bppmThreshold` | **ACCEL** *(MFE fill)* | — | measured |
| `--betaScale` | **ACCEL** *(MFE fill)* | — | measured |
| `--pfScale` | **ACCEL** *(MFE fill)* | — | measured |
| `--noLP` | **ACCEL** | — | measured; `tests/mfe_cuda_nolp.ts` |
| `--noGU` | **ACCEL** | — | measured |
| `-4` / `--noTetra` | **ACCEL** | — | measured |
| `--salt` | **ACCEL** | — | measured, 0.2 and 1.5 M |
| **`-g` / `--gquad`** | **ACCEL** *(new, 2026-09-10)* | — | measured, incl. int16; `tests/mfe_cuda_gquad.ts` |
| **`--nsp`** | **ACCEL** *(new, 2026-09-10)* | — | measured, symmetric **and asymmetric**; `tests/mfe_cuda_nsp.ts` |
| `-P` / `--paramFile` | **ACCEL** | **none possible** | measured; 4 bars in `verify_paramfile_bars.sh` |
| `--helical-rise` | **ACCEL** | — | measured *(only bites under `--salt`)* |
| `--backbone-length` | **ACCEL** | — | measured *(only bites under `--salt`)* |
| `--maxBPspan` == length | **ACCEL** | — | measured |
| **`--maxBPspan` < length** | **ACCEL** *(new, 2026-09-11)* | — | measured at spans 3–400; `RNA_HC_VERIFY` 0/185 716 words; `tests/mfe_cuda_span.ts` |
| `-j` / `--jobs` | **ACCEL** | — | measured |
| `--unordered` | **ACCEL** | — | measured **sorted** — see note |
| `--ImFeelingLucky` | **ACCEL** | — | route measured; **no byte bar possible** — see note |
| **`-c` / `--circ`** | **ACCEL** *(new, 2026-09-11)* | — | measured, incl. int16/`--noLP`/`--salt`; `tests/mfe_cuda_circ.ts` |
| **`--noClosingGU`** | **ACCEL** *(new, 2026-09-11)* | — | measured, incl. int16/chunked/`-c`/`-g`; `tests/mfe_cuda_noclosinggu.ts` |
| `--energyModel` | DECLINED | 1, 2 | **measured 2026-09-16**: silently ignored, see below |
| **`-C` / `--constraint`** | **ACCEL** *(new, 2026-09-16)* | — | measured; `verify_constraint_parity.sh --expect-accelerated`, **5 shapes / 30 records, byte-identical** |
| **`--canonicalBPonly`** | **ACCEL** *(new, 2026-09-16)* | — | measured *(rides `-C`: it only changes what the depot contains)* |
| **`--enforceConstraint`** | **ACCEL** *(new, 2026-09-16)* | — | measured; it is the option both `pipe` shapes are built on |
| `--shape` | DECLINED | 1, 2 | **measured 2026-09-16**: silently ignored, see below |
| `--shapeMethod` | DECLINED | 1, 2 | asserted *(same path as `--shape`)* |
| `--shapeConversion` | DECLINED | 1, 2 | asserted |
| `--sp-data` | DECLINED | 1, 2 | asserted |
| `--sp-strategy` | DECLINED | 1, 2 | asserted |
| `--sp-preprocess` | DECLINED | 1, 2 | asserted |
| `--motif` | DECLINED | **2** *(gate 1 keeps it as an optimisation)* | **measured 2026-09-16, both halves**: a motif derived from the fold's own interior loop bites (−21.80 → −29.80), and gate 2 declines it on `fc->sc` with the answer matching |
| **`--commands`** | **ACCEL when it queues only hard constraints**, DECLINED otherwise *(new, 2026-09-16)* | 2 | measured both ways: an HC-only file sweeps and matches; an `E` (soft) file takes the CPU route and matches |
| `-m` / `--modifications` | DECLINED | 1 | measured |
| `--mod-file` | DECLINED | 1 | asserted *(requires `--modifications`)* |
| `--batch` | **UNREACHABLE** | — | measured: exits 1 on FASTA input |
| `-v`, `-i`, `-o`, `--noconv` | NEUTRAL | — | measured |
| `--auto-id`, `--id-prefix` | NEUTRAL | — | measured |
| `--id-delim`, `--id-digits`, `--id-start` | NEUTRAL | — | asserted |
| `--filename-delim`, `--filename-full` | NEUTRAL | — | asserted |
| `--log-level` | NEUTRAL | — | measured |
| `--log-file`, `--log-time`, `--log-call` | NEUTRAL | — | asserted |
| `--noPS` | NEUTRAL | — | measured *(used by every arm)* |
| `--noDP` | NEUTRAL | — | measured *(requires `-p`)* |
| `-t` / `--layout-type` | NEUTRAL | — | asserted |
| `--benchmark`, `--bm-*` | NEUTRAL | — | asserted *(post-hoc scoring)* |

### Two options a byte bar cannot judge, for different reasons

**`--unordered`** emits records in *completion* order by design, so a
byte-identity check reports a difference that is **the option working
correctly**. `check_sorted()` exists for it: same records, same answers, any
order. Measured identical when sorted.

**`--ImFeelingLucky`** uses stochastic backtracking, so **two CPU runs of the
same flag also differ.** It is accelerated and its route is asserted, but no
deterministic bar can judge its output. The reason this is written down rather
than just noted: "GPU ≠ CPU" here looked exactly like the `--nsp` defect until
the CPU was compared against itself.

---

## Summary

| | count |
|---|---|
| **ACCELERATED, byte-identical** | **31** *(`--commands` conditionally)* |
| DECLINED, CPU route asserted | 9 |
| NEUTRAL | 19 |
| UNREACHABLE from this CLI | 1 |

**No silent wrong answer is known anywhere on the option surface** — and unlike
earlier versions of that claim, it now rests mostly on measurements rather than
on guard reading.

### What changed on 2026-09-10/11

- **`-g` moved DECLINED → ACCELERATED.** Three gates lifted; the last blocker was
  the *backtrack*, not the recursion (`VRNA-PATCH(bps-backtrack)`).
- **`--nsp` moved DECLINED → ACCELERATED.** `Energy()`'s missing 0 → 7 promotion
  and its index-swap-for-`rtype[]` were fixed.
- **`--backbone-length` and `--helical-rise` closed.** They only bite under
  `--salt`; the earlier "did not bite" was a missing `--salt`.
- **`-P`'s four bars all run.** Bar 4 found a live silent wrong answer (int16
  plus a large-`stack` file), now fixed.
- The harness went from 18 checks to 40 and gained `check_sorted()`.
- **`-c` moved DECLINED → ACCELERATED (2026-09-11).** It was never new
  arithmetic: `DMLi` **is** `fM2_real`, and the sweep discarded it one row
  later. It costs a **chunk width** of VRAM, counted in
  `modular_decomposition_bytes_per_file()`.
- **`--maxBPspan` moved DECLINED → ACCELERATED (2026-09-11).** The span is a
  pure hard constraint (`hard.c:778`, `(j - i) < md->max_bp_span`) and the device
  replica already had the test; what was missing is that the span arrived as
  **one scalar for a batch in which it is per record** — `vrna_fold_compound()`
  gives every compound its own length as a default span, so even the default
  model differs per record in a mixed-length batch. `d_span_H` is now a table
  beside `d_len_H`. A second defect went with it: the device clamped `span < 5`
  up to the record length, which would have turned `--maxBPspan=3` into an
  **unrestricted** fold.
  **Neither defect could have changed an answer**, and the reason is structural:
  the span is **nested-monotone**, so an over-permissive mask can only admit
  extra *outermost* pairs and `vrna_mfe_exterior_f5()` — host-side, upstream's
  own — filters exactly those. Measured both ways; see
  `tests/mfe_cuda_span.ts`.
- **`-d0` moved DECLINED → ACCELERATED (2026-09-11), for ONE line.** d0 and d2
  share the recursion (`mfe_multibranch.c:686` dispatches `ml_pair_d0` or
  `ml_pair_d2`, both reading only `dmli1`), and — the part I got wrong first —
  upstream **zeroes `P->mismatchM` at d0** (`params.c:644`), so both multibranch
  sites were already correct. Measured: 0 nonzero `mismatchM` entries at d0
  against 175 at d2; red-teaming those two sites changes nothing. The single
  real gap was `vrna_mfe_gquad_internal_loop()`'s `mismatchI`, which is **not**
  zeroed (158 nonzero at both) — so `-d0` alone was already right and merely
  refused, and `-d0 -g` was the one combination that answered wrongly.
  **d1 and d3 stay declined**: `ml_pair_d1()` reads `dmli2` as well as `dmli1`,
  a second `DMLi` generation the sweep does not carry, and d3 adds coaxial
  stacking. That is new DP state, not new arithmetic — and it is why gate 3 is
  not empty any more.
- **`--noClosingGU` moved DECLINED → ACCELERATED (2026-09-11).** It was HALF implemented, which is worse than unimplemented: the
  hairpin/multibranch half was applied and the interior-loop half was not, so `c`
  was the minimum of *no* model rather than of a different one. The missing half
  is upstream's own `mfe_internal.c` skips, stack case exempted.
  **It also uncovered a year-old latent defect**: `fml_scan_kernel` guarded only
  ONE side of each of two `INF`-capable sums, so fML could carry `INF` minus a
  real energy. Invisible on int32 (such a value never wins a `min`), fatal under
  int16, where it became a block baseline and tripped `__trap()`. Both sums now
  use `fml_tadd()`. See `PORT_NOCLOSINGGU_SPEC.md`.

### Still honestly open

1. **`-P` cannot be guarded at all**, by construction: `vrna_params_load()`
   mutates library globals and leaves no flag for a guard to test. Its
   assumptions live in `load_param()` instead, and **four of seven ranked ones
   are still unchecked** against a non-default table (`lxc` narrowed to `float`,
   the dead mismatch/dangle tables, special-hairpin strides,
   one-parameter-set-per-batch).
2. **`--batch` is unreachable by this harness**, so it is untested — not passing.
3. **`--ImFeelingLucky` has no bar** and cannot have one of this kind.
4. **11 rows are asserted, not measured.** Each is either a pure I/O option or
   shares a code path with a measured sibling — but that is an argument, not a
   run.
5. **`-c` shipped on 2026-09-11**, so the list of genuinely missing capabilities
   is now empty for single-sequence linear-or-circular folding. What remains
   unsupported is *multistrand* and *comparative*, neither reachable from
   RNAfold's input, and the constraint family (`-C`, `--shape`, `--motif`,
   `--commands`, `-m`), which is declined by design rather than by gap.

### Re-running this audit

```
tools/verify_option_parity.sh <build-tree> <mixed-length.fa>
```

Mixed lengths matter: the `--noLP` + `RNA_SLOT_FLOW` defect was invisible to
uniform-length fixtures.

---

## Every DECLINED row is now measured, and none of them already works

*2026-09-16, `tools/probe_declined_options.sh` against the local CUDA build.*

Five options moved DECLINED → ACCELERATED in this project because somebody
lifted the check and compared — `noGU`, `uniq_ML`, `--nsp`, `-g`, `-d0`. Each
time the measurement was made in a scratch build, which is slow and is exactly
the shape of the stale-binary traps recorded in `PORT_INVESTIGATIONS.md`. The
`RNA_ENGINE_ALLOW` test hook (`mfe/cuda/engine.c`, announced on stderr, lifts a
named check in **both** gate 1 and gate 2) makes the same measurement from a
shipped binary, and this is the first sweep with it.

| option | verdict | what the device actually does |
|---|---|---|
| **`-C`** hard constraints | **DIFFERS**, 7 of 8, −59.70 kcal | not ignored — **fill and backtrack disagree**: all 8 structures re-evaluate to a different energy than the run reports. Reproduces the 2026-09-06 signature exactly |
| **`--energyModel 1`** | **DIFFERS**, 8 of 8, +547.50 kcal | **silently ignores it** — byte-identical to the fold with no option |
| **`--energyModel 2`** | **DIFFERS**, 3 of 8 | silently ignores it |
| **`--shape`** (soft) | **DIFFERS**, −8.75 kcal | silently ignores it |
| **`--commands`** | **DIFFERS**, −2.30 kcal | silently ignores it |
| **`-d1`, `-d3`** | **TRAPPED** | gate 3 fires: *"this CUDA build implements dangle models 0 and 2 (got 1); … `vrna_cuda_engine_supports()` should have declined this"* |
| `--motif` (ligand) | **not measured** | the probe cannot make a motif bind on a synthetic sequence, so nothing is claimed — see below |

**Candidates for acceleration: zero.** Every guard tested is earning its keep,
and four of them are standing between the user and a *silently* wrong answer
rather than a loud one.

### Two traps this sweep walked into first, both worth keeping

**1. The fixture was in the wrong alphabet, and it inverted the verdict.**
`--energyModel 1/2` means *A pairs B, C pairs D*. On the ACGU fixture every
other case uses, **no pair is legal at all**: both routes return all-dots at
0.00, and the first run of this probe reported `AGREES 8/8 records identical,
swept` — a perfect agreement between two empty answers. On the ABCD alphabet the
same binary has the CPU finding −45.10 and the device returning 0.00. Same
option, same build, opposite verdicts, decided entirely by the fixture's
alphabet. The probe now carries its own ABCD fixture and refuses any case whose
CPU reference folds nothing.

**2. The ligand motif did not bite.** The theophylline aptamer from upstream's
own documentation was reported `AGREES` on a sequence that does not contain it;
after embedding both segments it is *well-formed* but still does not bind, even
at −30 kcal, so the CPU answer never moves. The probe reports **SKIPPED**, and
the row above says *not measured* rather than claiming anything. **A motif
fixture that actually binds is the one piece of this sweep still owed.**

Both traps are the same shape as the eleven in `project_port27_checks_that_lied`:
the check could not reach what it claimed to test, and reported success for that
reason. The probe now makes three assertions before it will report a comparison
at all — the CPU reference must fold *something*, the option must *change* the
CPU answer, and the forced run must *actually sweep*.

---

## What each DECLINED option would need

The table above says *whether* an option is accelerated. This says **what it
would take**, which is the question that decides what to build next. Sizes are
judgements; the mechanisms are read from the code or measured on 2026-09-16 with
`tools/probe_declined_options.sh`.

| option(s) | what is actually missing | kind of work | size |
|---|---|---|---|
| **`-C`**, `--canonicalBPonly`, `--enforceConstraint` | **An ORDERING bug, not missing arithmetic.** `init_gpu3()` packs the hard-constraint masks at `mfe_cuda.c:1238`; `vrna_fold_compound_prepare()` — which is what materialises `hc->depot` into `hc->mx` — runs at line 1243, *after* it. And `g_hc_seq_derived` (`:1226`) lets the device derive the masks from the sequence on a predicate that reads `noLP` and never asks whether a depot exists | move the pack after the prepare; make the derive predicate consider `hc->depot` — **both done, and they land 3 of 5 constraint shapes**. What remains is `hc->up_hp` / `up_int`: the device carries only `up_ml`, so a base forced to PAIR does not stop the sweep leaving it unpaired inside a hairpin or interior loop | **small, then two O(n) uploads and one comparison in each of two kernels** |
| **`--shape`**, `--shapeMethod`, `--shapeConversion`, `--sp-*` | soft-constraint terms the kernels never add. Deigan is a per-nucleotide stacking bonus (`sc->energy_stack`, O(n)); Zarringhalam a per-position unpaired term (`sc->energy_up`). A generic `sc->f` callback cannot run in a kernel at all | upload two O(n) arrays and add a term at the hairpin, interior and multibranch sites — for the **table-based** methods only | medium; callbacks stay declined permanently |
| **`--commands`** | nothing of its own: a command file queues hard and/or soft constraints | inherits `-C` and soft constraints | follows the two above |
| **`--motif`** (`domains_up`) | unstructured-domain energies come from a callback evaluated per (i, j, loop context) | either evaluate the domain on the host into a table the kernels read, or port the callback | medium–large, and **the effect is still unmeasured** — no fixture yet makes a motif bind |
| **`-m` / `--modifications`**, `--mod-file` | modified bases change both the pair rules and the loop energies, per modification | new tables plus `Energy()` changes | large |
| **`--energyModel` 1/2** | the device **re-derives `ptype` from the sequence using the standard alphabet**, while `energy_set > 0` changes the encoding (`A`=1, `B`=2, …) *and* the pair table. Measured: the device returns the plain fold, byte-identical to no option | upload the host's `ptype`/pair table instead of deriving, and range-check every table index against codes > 4 | medium |
| **`-d1`, `-d3`** | `ml_pair_d1()` reads `dmli2` as well as `dmli1` — a **second `DMLi` generation** the sweep does not carry; `-d3` adds coaxial stacking on top | new DP state carried through the sweep and its chunking | large |
| `logML` | a log-scaled multibranch term in the recursion — and **no CLI flag exists**, so it cannot be barred from RNAfold at all | kernel work plus a way to test it | not reachable from this CLI |
| sliding-window hard constraints | RNAplfold's layout, not a global fold | out of scope for this binary | — |
| multistrand, comparative (`VRNA_FC_TYPE_*`) | a different recursion entirely | — | — |

### The `-C` diagnosis, because it moved a whole tier

`-C` was filed with "needs real work". It needs an ordering fix, and the
measurement that shows it is worth stating in full:

1. With the guard lifted, the device's answers are **better than legal** on all
   8 records (−14.10 → −17.90 and so on). Better-than-legal means the fill did
   not see the constraint.
2. The returned structures **violate no constraint** and re-evaluate to a
   different energy than reported. The backtrack is walking a matrix that was
   filled for a different problem.
3. **`RNA_HC_VERIFY=1` reports `436 words x 4 masks, 0 mismatching`** on exactly
   those folds.

(3) is the important one, and it is this project's oldest failure mode wearing a
new hat. The verifier rebuilds the masks from `VC[H]->hc->mx` and compares — but
it runs inside `init_gpu3()`, *before* prepare, when `hc->mx` is still the
unconstrained default. **It compares two unconstrained matrices and agrees.**
`RNA_ROW_VERIFY` was blind to the same defect in 2026-09-06 for a different
reason (it compared the fork against itself). A verifier only means something
after the thing it verifies exists.

Nothing here is a hot-path change, which is why `-C` is now the cheapest
capability left on the declined list rather than the most expensive.

### `-C`, continued: the ordering fix lands three of five shapes

The fix was made (`mfe_cuda.c`: the prepare loop hoisted above `init_gpu*`, and
`g_hc_seq_derived` now asks per record whether a depot or a soft constraint
exists instead of reading `noLP` alone). With it, and the guard lifted for the
measurement only:

| shape | constraint | result |
|---|---|---|
| `dots` | control, no constraint | identical, 1 sweep |
| `xpaired` | force UNPAIRED every base the free MFE pairs | **identical**, 1 sweep, self-consistent |
| `farpair` | force a pair the free fold does not contain | **identical**, 1 sweep, self-consistent |
| `pipeblock` | `--enforceConstraint`, force a block PAIRED | **DIFFERS** — device better than legal (−18.40 vs −16.00) |
| `pipehairpin` | `--enforceConstraint`, force a base inside a hairpin loop PAIRED | **DIFFERS** |

**So the guard stays**, and what remains is now named precisely. `hc->mx` holds
the legality of a **pair**; the legality of leaving a base **unpaired** lives
entirely in `hc->up_hp`, `up_int`, `up_ml` and `up_ext` — and the device carries
**only `up_ml`** (`hp_mb_loop.cu:187`, one byte per position). Forcing a base to
pair does not change any pair's legality, so nothing in the four masks moves,
and the sweep goes on allowing hairpins and interior loops whose unpaired span
covers a base that upstream requires to be paired.

**What is left for `-C`:** upload `up_hp` and `up_int` beside `up_ml`, and test
them where upstream does — `up_hp[i+1] >= j-i-1` in the hairpin kernel,
`up_int[i+1] >= u1` and `up_int[q+1] >= u2` in the interior-loop kernel. Two
O(n) arrays and one comparison in each of two kernels. `up_ext` needs nothing:
the exterior loop is upstream's own host code.

**And one upstream semantic, measured, that the first version of these shapes
got wrong:** `|` ("paired with something") is **not enforced without
`--enforceConstraint`**. The same constraint leaves the MFE at the free answer
(−21.60) under plain `-C` and moves it to −14.00 with the flag. A shape that
does not bite reports agreement for the wrong reason, so both pipe shapes carry
the flag — which also puts `--enforceConstraint` itself under test, since it is
the option that makes `hc->up_*` restrictive in the first place.

---

## `-C` SHIPPED, 2026-09-16 — and it needed both halves of what a constraint is

The "what it would need" row above said *small, then two O(n) uploads and one
comparison in each of two kernels*. That is what it took, and the order in which
the three pieces were found is the useful part:

**1. An ordering defect (committed separately).** `init_gpu3()` packed the four
hard-constraint masks five lines before `vrna_fold_compound_prepare()`
materialised `hc->depot` into `hc->mx`. Invisible for every model the guard
admitted, because their pre-prepare matrix already equals the final one.
`RNA_HC_VERIFY` reported **0 mismatching words** on exactly the folds that came
out wrong — it rebuilds from `hc->mx` at pack time, so it was comparing two
unconstrained matrices.

**2. `up_hp`, and it does not belong in a kernel that seemed obvious.** The
first attempt put the hairpin span check in `fill_arrays_loop.c`'s row combine —
which is **dead code under `RNA_GPU_SWEEP`**, the shipped default. The live site
is `new_c_kernel`, and the gate there is upstream's own rule verbatim:
`up_hp[i+1] >= j-i-1` (`wrap_hairpin_hc.inc:42-52`). `up_hp` is a **count**, not
a boolean like `up_ml`: a multibranch loop extends one base at a time, a hairpin
is admitted as a whole span.

**3. `up_int`, restoring a line that was deleted years ago.** `Energy()` in
`int_loop.cu` carried a commented-out `if(hc_up[q+1] < j_q) return INF;` marked
*"this should not be needed as using Hc"* — true exactly while nothing could
force a base to pair, which is what the routing guard was for. Both runs are
checked now, `u1 = p-i-1` and `u2 = j-q-1`, per `wrap_internal_hc.inc:57-67`.

**Both arrays are NULL when no record carries a depot**, so an unconstrained
fold reads neither and the hottest loop in the project gains no loads. `up_ext`
needed nothing: the exterior loop is `vrna_mfe_exterior_f5()`, upstream's own.

**What it measured, before and after** (8 records, 80–240 nt, forced-paired
blocks under `--enforceConstraint`):

| | CPU | GPU before | GPU after |
|---|---|---|---|
| worst record | −85.20 | **−102.60** (better than legal) | **−85.20** |
| records matching | — | 0 of 6 | **6 of 6** |

And the bar, with no env hook and the guard genuinely lifted: **5 shapes, 30
records, 80–1240 nt, all byte-identical to the CPU route with the sweep running,
every energy self-consistent against RNAeval.**

**Gate 3 is empty again** — `-d1`/`-d3` are the only backstopped options, and
they remain so.

---

## `--commands`: the guard was refusing the option because the driver never applied it

**2026-09-16.** A command file can queue **three different kinds** of thing
(`io/commands.h`): hard constraints (`VRNA_CMD_PARSE_HC`), soft constraints
(`_SC`) and unstructured domains (`_UD`). RNAfold parses with `_DEFAULTS`, which
is all three. So `--commands` is not one option — it is whichever of the three
the file happens to contain, and that cannot be known from the flag.

**Why it was silently ignored rather than declined.** `build_one()` — the chunk
path's own compound builder — applied `-C` constraints and *not* command files.
So the routing guard inspected a compound that had no `sc`, no `domains_up` and
no `hc->depot`, found nothing to refuse, and the batch folded unconstrained. The
gate-1 check on `opt->cmds` was the only thing standing between the user and a
wrong answer, which is why removing it alone would have been a defect rather
than a feature.

**The fix is to apply the file and let gate 2 decide**, which it already knows
how to do:

| the file queues | where it lands | gate 2 |
|---|---|---|
| hard constraints (`P`, `F`, `A`, `C`) | `fc->hc->depot` | **accepted** — supported since `-C` shipped |
| soft constraints (`E`) | `fc->sc` | declined, `"soft constraints"` |
| unstructured domains | `fc->domains_up` | declined, `"unstructured domains (ligand motifs)"` |

**Measured, 40 ragged records, 200–900 nt:**

| file | bites? | sweeps | vs CPU route |
|---|---|---|---|
| `P 10 0 8` + `P 30 0 6` (hard only) | yes | **1** | **byte-identical** |
| `E 12 0 5 -2.0` (soft) | yes | **0** | byte-identical |

The first is accelerated, the second routes to upstream, and neither needed a
new check: gate 2's existing `fc->sc` and `fc->domains_up` tests do the work
once the compound actually carries what the file asked for.

**The transferable part:** a guard can only refuse what it can see. When the
driver builds its own compound, *everything* `process_record()` would apply to
its own has to be applied there too — or the guard is inspecting a different
object than the one that gets folded.

---

## The five that are left, and why each is where it is

*After `--commands` (2026-09-16), the declined list is five entries and none of
them is a small job. Written down so the next session starts from a diagnosis
rather than a re-derivation.*

### `--energyModel` — diagnosed, and the recommendation is to leave it declined

**The device derives pair types from the ALIASED encoding; upstream uses the
RAW one.** `d_S2` is `VC[H]->sequence_encoding`, and `vrna_seq_encode()` applies
`md->alias[]` (`alphabet.c:299`). Upstream's `ptype` comes from
`sequence_encoding2`, the raw codes.

At `energy_set = 0` the alias is the identity for ACGU, so the two agree — which
is why this never mattered in four years. At `energy_set = 1` on the intended
ABCD alphabet, A→3 and B→2, so the device asks `pair[3][2]` where upstream asks
`pair[1][2]`. `model.c:1042` fills `pair[i][i+1] = 2` on **raw** indices, so the
device's lookup is 0 and **nothing pairs** — exactly the all-dots 0.00 measured.

**Cost to fix, revised 2026-09-16 after reading the packing.** Smaller than it
first looked, and bounded by one hard limit:

* `Ptype()` reads `int_loop`'s **3-bit packed** sequence (`unpack()`, 10 codes
  per 32-bit word, `assert(si < 8)`) and indexes an **8x8** pair table. Raw
  codes for the intended A/B/C/D alphabet are **1-4**, so they fit — but the
  alphabet runs to `MAXALPHA = 20`, and codes above 7 **cannot be represented
  at all** without rewidening the packing.
* So the feature is inherently **conditional**, the same shape as `--maxBPspan`
  and `--commands`: accept `energy_set` when every code fits in three bits
  (letters A-G), decline otherwise. That is a checkable predicate, not a
  caveat.
* The work is then: pack the RAW encoding into a second array beside the
  aliased one, read it in `Ptype()` and `Ptype2()` behind a uniform flag, and
  add the guard predicate. **The default path pays nothing** -- `cell_invariants`
  derives the type ONCE PER CELL (H1 hoisted it there), so even when enabled the
  cost is one extra packed load per cell, not per candidate.

**Value:** an option whose intended use is a synthetic alphabet, for theory and
design work rather than real RNA. The CPU route is correct and always will be.

**Recommendation: build it only if there is a user.** Without one, the 3-bit
packing gives a *principled* permanent decline -- "this backend encodes a
four-letter alphabet in three bits, and `energy_set` is defined over twenty
letters" -- which is a better answer than an implementation gap.

### `--shape` / `--shapeMethod` / `--shapeConversion` / `--sp-*` — the one worth doing, and it needs a decision first

SHAPE data does not arrive as one addend. It reaches the recursion through the
**soft-constraint wrapper layer** — `internal_sc.inc`, `multibranch_sc.inc`,
`hairpin_sc.inc`, `exterior_sc.inc` — a set of function pointers woven through
every loop type, carrying `energy_up`, `energy_bp`, `energy_stack` and arbitrary
callbacks.

Deigan's own term is simple in isolation (a per-nucleotide stacking bonus,
O(n)), and so is Zarringhalam's (a per-position unpaired term). The **decision**
is how much of the surface to support:

| | guard | device work |
|---|---|---|
| Deigan only | narrow: accept iff `sc->energy_stack` is the only thing set | one O(n) upload, a term at three sites |
| Deigan + Zarringhalam | as above plus `energy_up` | two uploads, terms at four sites |
| general soft constraints | impossible for `sc->f` callbacks | — |

**This is the highest-value declined option** — SHAPE data is common in real use,
unlike the other four — and the narrow version is tractable. It is flagged for a
session with a decision made up front rather than discovered mid-implementation.

### `--motif` — MEASURED (2026-09-16), and the fixture was the whole difficulty

A ligand motif binds a **sequence _and_ a structure**: that is the point of the
option — a protein binds a known site and stabilises it. So a motif taken from
the documentation cannot bind a random sequence, and three attempts to force one
failed, including a synthetic four-pair motif at −30 kcal/mol.

**The construction that works is to read both halves off a real fold.** Find an
interior loop in the free MFE — a closing pair `(i,j)` with an enclosed pair
`(p,q)` and unpaired bases on both sides — and emit exactly that sequence and
that dot-bracket. It is then present and formable by construction, and the bonus
shows up as an energy shift: **−21.80 → −29.80** for a −8.0 motif, same
structure, ligand bound. `tools/probe_declined_options.sh` now derives it that
way, so the row stays measured.

**And the port side is now structural rather than incidental.** `build_one()`
applies the motif to the chunk's own compound, so `vrna_sc_add_hi_motif()`'s
soft constraint is on the object the guard inspects: with gate 1 forced open,
gate 2 declines on `fc->sc` and the answer is byte-identical to the CPU route —
**0 sweeps, and for the right reason.** Gate 1 still refuses `--motif` first,
deliberately: a motif always lands in `sc` and is always declined, so building a
batch for it would be pure waste.

### `-m` / `--mod-file` — inherits the SHAPE decision

Modified bases install soft constraints (`vrna_sc_mod_*`,
`constraints/sc_cb_mod*.c`), so they sit behind whatever is decided for
`--shape`, plus per-modification energy tables of their own.

### `-d1` / `-d3` — new DP state

`ml_pair_d1()` reads `dmli2` as well as `dmli1` — a second `DMLi` generation the
sweep does not carry — and `-d3` adds coaxial stacking on top. The only entry
left on gate 3, and the only one where the missing thing is state rather than
arithmetic.

---

## `--shape`: ON HOLD pending Dr. Lorenz (2026-09-16)

Not blocked on the port. The scoping decision — Deigan only, Deigan plus
Zarringhalam, or the general soft-constraint surface — is being taken with the
author of the method rather than inferred from the source.

**Two things to carry into that conversation:**

1. **SHAPE is a soft constraint by design**: it does not forbid anything, it
   re-weights, nudging the prediction toward what the probing data supports.
   That is why it reaches the recursion through the wrapper layer
   (`internal_sc.inc`, `multibranch_sc.inc`, `hairpin_sc.inc`,
   `exterior_sc.inc`) rather than through `hc->mx` — and why the device work is
   "add a term at N sites", not "mask a cell".
2. **It is known to handle pseudoknotted data poorly** (Luke, from Lorenz).
   That is a property of the *model*, not of this port: the recursion this
   backend accelerates is strictly nested, so a pseudoknot cannot be
   represented at all, with or without SHAPE. **Flagged for the day
   pseudoknots come up**, which is a much larger question than an option —
   ViennaRNA has separate machinery for them (`RNAPKplex` and friends) and none
   of it shares the recursion this backend fills.

**When SHAPE does land, the pseudoknot limitation should be stated in the same
place as the feature**, so a user reading "SHAPE is supported" also reads what
SHAPE cannot do — the honest version of the claim, and the one a probing
experiment actually needs.
