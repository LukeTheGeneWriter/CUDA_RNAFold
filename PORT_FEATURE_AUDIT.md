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
