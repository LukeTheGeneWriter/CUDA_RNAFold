# Feature audit: CUDA_RNAFold (ViennaRNA 2.3.0 base) vs ViennaRNA 2.7.2

*Run 2026-09-04, empirically, on both built binaries — not argued from source.
Input: 8 × 300 nt, G-run rich, GPU batch path engaged, `RNA_CPU_THREADS=0`
(the default). Harness: `~/audit.py`; raw verdicts `~/phase5/audit.json`.*

The question this answers is **not** "which features are implemented". It is
**"which options can silently change the answer"** — because an unsupported
feature that errors out is a gap, while an unsupported feature that returns a
plausible number is a defect.

Verdicts: **NO DOOR** (option rejected outright — safe) · **REFUSED** (accepted,
then errors — safe) · **PARITY** (agrees with 2.7.2 byte for byte) ·
**IGNORED** / **DIVERGES** (accepted, answer differs — *unsafe*).

---

## The headline: two options lie, and G-quadruplex is one of them

### `-g` / `--gquad` — **produces wrong answers, silently**

The fork emits G-quadruplex structures (all 8 test records contain `+`), and its
energies are **internally self-consistent** — 2.7.2's own `RNAeval -g` confirms
every number to the cent. But every one is **suboptimal**:

```
rec   fork -g    true optimum (2.7.2)   short by
0     -364.00    -395.10                31.10
1     -396.00    -426.40                30.40
2     -400.82    -420.50                19.68
3     -399.70    -415.30                15.60
4     -361.80    -392.20                30.40
5     -401.20    -426.70                25.50
6     -361.80    -382.40                20.60
7     -432.00    -450.00                18.00
```

8 of 8 wrong, by 15–31 kcal/mol. **This is the worst failure shape available**: a
well-formed structure, a self-consistent energy, no warning, no non-zero exit.

Cause: the GPU kernels have G-quad **commented out** — `//*ggg`,
`//with_gquad` at `src/ViennaRNA/int_loop.cu:1244-1268` — so the sweep never
scores a G-quad contribution into `c`/`fML`, while the 2.3.0 host backtrack still
knows how to *emit* G-quads. The search is therefore blind to exactly the
structures the flag was set to find.

Worth noting for the port: 2.7.2 adds three G-quad constants that 2.3.0 lacks
(`GQuadLayerMismatch37/H/Max`). They did **not** affect this comparison —
2.7.2's `RNAeval` scored the fork's structures identically — but they are a live
difference for multi-layer quadruplexes and must not be assumed inert.

### `-c` / `--circ` — **accepted and ignored**

On the default GPU path the fork's `-c` output is **byte-identical to its own
non-circular output**, while 2.7.2's genuinely differs. It returns the linear
answer for a circular fold.

With the CPU queue on and every record routed to it
(`RNA_CPU_THREADS=4 RNA_CPU_THRESHOLD=100000`) the fork matches 2.7.2 **exactly**.
So the circular code is correct — the GPU path drops it. **This corrects the
plan's note that circular RNA is "currently routed to CPU": it is not routed,
and nothing guards it.** Which answer you get depends on the queue threshold.

---

## Full option surface

### Silently wrong — fix before anything else (2)

| option | verdict | detail |
|---|---|---|
| `-g` `--gquad` | DIVERGES | valid but suboptimal, 8/8 records, up to 31 kcal |
| `-c` `--circ` | IGNORED | returns the linear answer; correct only via the CPU queue |

### Refused — safe, but a parity gap (13)

| option | how it fails |
|---|---|
| `-d0` `-d1` `-d3` | **clean guard**: `"this CUDA build requires --dangles=2 (got 0)"` — the model the others should follow |
| `-c -g` | clean: `"G-Quadruplex support is currently not available for circular RNA"` (inherited from 2.3.0) |
| `-p` `-p0` `--MEA` `--betaScale` `--bppmThreshold` `--canonicalBPonly` | partition function: dies `rc=1` **after emitting 1 record** |
| `--noClosingGU` | `ERROR: backtracking failed in repeat` — internal failure, not a guard |
| `--shape` | `ERROR: backtracking failed in repeat` — same; soft constraints never reach the kernels |

`--noClosingGU` and `--shape` fail *inside* the algorithm rather than at a guard.
`g_hc_seq_derived` (`src/bin/RNAfold.c:1035`) excludes constraints/SHAPE/motifs/
commands and `noLP` — but **not `noClosingGU`**, so the GPU derives bitmasks
inconsistent with the host backtrack and the walk fails.

### No door at all — 2.7.2 features the fork cannot express (7 tested, 23 total)

`--salt` · `--helical-rise` · `--backbone-length` · `--modifications` ·
`--mod-file` · `--sp-data` / `--sp-preprocess` / `--sp-strategy` · `--jobs` ·
`--unordered` · `--noDP` · `--log-*` · `--benchmark` / `--bm-*` ·
`--filename-*` · `--id-delim` · `--energyModel` (documented but see below)

These are safe (the CLI rejects them) and they are the honest measure of the
parity gap: **23 of 2.7.2's ~60 options do not exist here.**

### Parity — verified byte-identical (14)

`--noLP` · `--noGU` · `--noTetra` · `--maxBPspan` · `-T` · `--nsp` · `--noconv` ·
`-C --batch` · `-C --batch --enforceConstraint` · `--commands` · `--auto-id` ·
`--id-prefix` · `-v` · `-P <parameter file>`

Hard constraints work. **Soft constraints do not.** That split is the useful
generalisation: what the GPU can express as a bitmask survives; what needs a
per-cell energy term does not.

`-P` deserves a note — an earlier run showed it diverging, which was my error:
I fed a 2.7.2 `.par` to the 2.3.0 binary. Each binary with **its own** copy of
`rna_turner1999.par` agrees on all 8 records. The parameter *files* differ
between versions; the ported tree must ship 2.7.2's.

### A crash, unrelated to the port (1)

`--energyModel 1` **segfaults nondeterministically** — 1 of 3 runs with the queue
off, 1 of 3 with it on, rc=139. 2.7.2 handles the same invocation fine. The
option is in the fork's `.ggo` but effectively undocumented. This is a live
defect in the current tree, independent of the version bump.

---

## What "match all features up to 2.7.2" actually costs

The gap is not one feature; it is four independent bodies of work, and they
differ by more than an order of magnitude in cost:

| # | feature | cost | why |
|---|---|---|---|
| 1 | **Routing guards** | hours | Turn `-c` and `-g` from wrong into refused. Does not implement anything — it makes the tree *honest*. |
| 2 | **Salt corrections** | small | Default-off, three call sites; the multibranch term is already baked into `MLbase`/`MLclosing`/`MLintern` at init, so those kernels need nothing. |
| 3 | **Circular RNA** | moderate | Post-processing outside the triangular sweep; the CPU code already works, so this is wiring, not derivation. |
| 4 | **G-quadruplex** | substantial | `mfe_gquad.c` is 1262 lines. Needs the `ggg` matrix on the device and a G-quad term in the `c`/`fML` recursions — i.e. real kernel work in the hot path. |
| 5 | **Multistrand** | substantial | Genuinely inside the core recursion (`fms5`/`fms3`/`fM2`, per-nucleotide strand bookkeeping). Not a bolt-on. |
| 6 | **Modified bases / SHAPE** | needs a decision | Implemented upstream as **arbitrary user callbacks** (`vrna_sc_t.f`). Host function pointers cannot run in a kernel. Either route soft-constrained folds to the CPU, or precompute the shipped JSON modifications into a per-`(i,j)` table. **May not be fully achievable.** |
| 7 | **Partition function** | separate project | `-p` is a whole second algorithm, not a flag. |

Item 1 is the only one that is urgent, and it is cheap. Items 2–7 are Phase 5 of
the port plan and are sequenced *after* the rebase, deliberately: implementing
G-quad kernels against the 2.3.0 tree means deriving them twice, and it would
muddy the byte-identical bar the port depends on.

---

## OPEN: a COMBINATION audit is still owed (Luke, 2026-09-07)

Everything above audits options **one at a time**. That is not sufficient, and
two findings on 2026-09-07 are why:

- **`--noLP` + `RNA_FML_INT16` do not compose.** Each is individually correct
  and verified; together the int16 encoding's premise fails, because `noLP` puts
  near-INF *finite* values into `fML` that a per-block 16-bit offset cannot
  represent. Refused at init. Nothing in a per-option audit could have predicted
  this — it took running the pair.
- **`--noLP` + `--salt` is an UPSTREAM inconsistency** (Defect D in
  `PORT_UPSTREAM_PROPOSAL.md`): the `noLP` stacking term is uncorrected for salt
  while the interior-loop path is corrected, and it is silent at default salt
  because `SaltStack` truncates to 0 above 0.5 M.

Both are *pairwise* properties. The accepted set is now default, temperature,
`noGU`, `uniq_ML`, salt and `noLP`, plus the orthogonal switches
`RNA_FML_INT16`, `RNA_SLOT_FLOW`, `RNA_CONTINUOUS_FLOW`, `RNA_GPU_CHUNK` and
`RNA_MIN_GPU_BATCH` — which is far more pairs than have ever been run together.

Known refusals of a PAIR, as opposed to an option:

| pair | why |
|---|---|
| `RNA_FML_INT16` + `RNA_SLOT_FLOW` | a slot handover leaves stale baselines |
| `RNA_FML_INT16` + `--noLP` | near-INF finite values exceed the 16-bit offset |

The audit to run before release should be a matrix, not a list, and it should
assert the three things a per-option check cannot: that the pair produces the
same answer as the CPU route, that it produces the same answer as the *other*
order of enabling, and that where a pair is refused, it is refused **at init**
rather than folding something plausible. `tools/verify_option_parity.sh` is the
natural home; today it walks options singly.

---

## The audit RAN, 2026-09-08: `tools/verify_option_matrix.sh`

**35 of 36 pairs sound. One live wrong-answer defect, and it is a THIRD
pairwise refusal that no per-option check could have predicted.**

It lives in its own file rather than in `verify_option_parity.sh` as suggested
above: the matrix is ~90 invocations, and folding it into the fast single-option
bar would have made that bar too slow to run casually. `verify_option_parity.sh`
keeps the option surface; this keeps the pairs.

Dimensions: the four CLI-reachable accepted options (`-T`, `--noGU`, `--salt`,
`--noLP`) against each other in **both enabling orders**, each against the five
switches (`RNA_FML_INT16`, `RNA_SLOT_FLOW`, `RNA_CONTINUOUS_FLOW`,
`RNA_MIN_GPU_BATCH`, `RNA_GPU_CHUNK`), and the switches against each other.

### The finding: `--noLP` + `RNA_SLOT_FLOW` is a silent wrong answer

| | |
|---|---|
| `--noLP` alone | GPU == CPU |
| `RNA_SLOT_FLOW=2` alone | GPU == CPU |
| **together** | **17 of 30 records disagree, deterministically** |

Diagnosed with `PORT_NOLP_SPEC.md`'s own bar rather than an energy comparison,
which matters because the two possible causes look identical in the energy:

- **structures are SELF-CONSISTENT** — every returned structure re-evaluates
  under `RNAeval` to exactly the energy printed beside it, 0 disagreements. So
  this is *not* the matrix/backtrack disagreement that the original `noLP`
  defect was;
- **worse on 17, better on 0.** A perfectly one-sided error, and the direction
  is the diagnosis: the GPU never finds anything *better*, only worse. That is
  lost state, not miscomputation.

**Mechanism, confirmed in the source.** `reset_slot_md()`
(`modular_decomposition.cu:972`) resets `d_fml_j`, `d_dml`, `d_dml1` and
`d_fml_prev` — its comment says "exactly the buffers `init_fML()` fills". `noLP`
added `d_cc`/`d_cc1` (`hp_mb_loop.cu:134`), row-shaped `SLOT_ALLOC` buffers
initialised at chunk start and rotated per row, and **nothing added them to the
reset**. A slot handover mid-sweep therefore leaves the new occupant holding the
previous record's `cc1`; since `c[i][j]` receives `cc1[j-1] + stackEnergy`, a
stale `cc1` *forbids* pairs rather than mispricing them — strictly fewer options,
hence worse-never-better.

This is the **same failure shape** as the already-refused
`RNA_FML_INT16 + RNA_SLOT_FLOW`, from the same cause: a feature added per-slot
state and the handover was not extended.

**Severity.** `RNA_SLOT_FLOW` is off by default and was measured as a 1.26-1.81x
regression, so exposure is low — but `--noLP` is accepted and the switch is live,
so the pair returns a plausible suboptimal structure today.

**FIXED 2026-09-08.** The first diagnosis above was the right locus and the
wrong buffer, and the correction matters more than the fix:

`reset_slot_md()` not resetting `d_cc`/`d_cc1` is real, and resetting them
per slot changed **nothing** -- byte-identical 17/30. The reason is the actual
cause. `refill_gpu3()` runs at EVERY handover, takes no slot argument, and
calls `init_gpu3()`, whose `SLOT_ALLOC` skips the malloc on a refill but
whose KERNEL LAUNCHES still run. So each handover INF-filled `cc`/`cc1` for
the WHOLE batch, wiping the mid-recursion `cc1` of every record still running
in every other slot -- which is why the per-slot reset was a no-op, and why
**eleven of the corrupted records had already retired before their own slot was
ever touched**. `cc`/`cc1` are the only sweep state prefilled in
`init_gpu3()`, which is exactly why no other option ever noticed.

The fix is both halves: guard that prefill with `!g_refill3` so a refill
cannot wipe live slots, and reset the incoming slot's own rows with the new
`reset_slot_nolp()`. **36/36 pairs, `make check` 145/145, noLP bar green,
and correct at SLOT_FLOW k=1,2,3,4,7, with chunking, and on 600/1600-record and
mixed-length inputs.**

### The regression test could not fail, and that is the lesson

The first version of the new `verify_nolp_parity.sh` arm passed against a
binary with the fix **deliberately removed**. Its fixtures (`u900`, `u2000`)
are UNIFORM-LENGTH, so every slot retires on the same iteration and the global
wipe lands when all neighbours are themselves starting fresh -- harmless.
Measured on the broken binary: **0 of 60 records wrong at one distinct length,
17 of 30 at thirty distinct lengths.** The arm now builds its own mixed-length
fixture and asserts both preconditions it depends on (the fixture is mixed, and
slots were actually shared), then was confirmed RED on the reverted build and
green on the fixed one.

### Known refusals: both still refuse, and refuse correctly

`int16+noLP` and `int16+slotflow` each exit non-zero with **no output at all** —
refused at init, not after emitting a plausible answer.

### What the audit found about its OWN instruments

Three, and each would have produced a confident wrong report:

1. **`bar_preflight.sh` could not see a tree checked out at an old commit.** Its
   test is "binary newer than the sources *in this tree*", and an old checkout
   has old sources. `~/port27cuda` was three commits back, so the first run was
   about to judge a binary with `noLP` still declined. Fixed: it now compares the
   build tree's commit against the tree the bar itself came from.
2. **`bar_preflight.sh` also cried wolf.** It scanned generated `*_cmdl.[ch]`,
   which gengetopt rewrites in arbitrary order during a build, so a perfectly
   current binary failed. A bypassed bar is how you get back to a false pass.
3. **`peak/iteration N records` is a PEAK, not a total.** Under `RNA_SLOT_FLOW=2`
   it reads 15 for 30 records, so summing it claims a 50% CPU fallback about a
   run whose row and cell totals are *identical* to the non-flow run. **This
   matters beyond the audit: benchmark v5 sums the same field for its
   `gpu_records` gate, and is correct only because no v5 arm enables flow.** The
   matrix uses the sweep's cell total instead, which is invariant across options
   and flow.

### Still not reached

`uniq_ML` (accepted, but no byte-comparable RNAfold flag), `logML` (declined, no
flag at all), and **triples** — both known refusals were pairwise, but nothing
here proves a third option cannot interact.
