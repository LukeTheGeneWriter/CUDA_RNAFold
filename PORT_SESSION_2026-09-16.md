# Session 2026-09-16 — handoff

**Where to start tomorrow:** you are on **`Lukes_Flow_Batching`**, forked from
**`Finished_Port`** (`30f5329d`). Read `CUDA_RNAFold_History.md` for the
position, then `PORT_LUKES_FLOW_BATCHING.md` for the plan. One thing needs an
A100 and nothing else does.

---

## 1. What landed

| | what | bar |
|---|---|---|
| **`-C` hard constraints ACCELERATED** | ordering fix + `up_hp` + `up_int`; guard lifted in both gates | `verify_constraint_parity.sh` 5 shapes × 30 records, `tests/mfe_cuda_constraints.ts` 3/3 |
| **`--commands` conditionally accelerated** | accepted when the file queues only hard constraints, declined otherwise | measured both ways |
| **`--motif` measured** | fixture derived from the fold's own interior loop; gate 2 declines on `fc->sc` | 0 sweeps, byte-identical |
| **`--energyModel` REJECTED** | decision, not a deferral | — |
| **`RNA_BUILD_PIPELINE` → memory-gated AUTO** | on when the next chunk fits in half of `MemAvailable` | measured both sides of the bar on an A100 |
| **The AUTO estimator re-based** | `0.727·L² + 72.5·L` measured; it now charges `L² + 128·L` instead of `2·(L+1)²` | six shapes, 1200–12000 nt |
| **`RNA_STREAM_OVERLAP` 1 and 2** | level 1 correct and −0.9 %; **level 2 raced, fixed, unverified at scale** | §G, default 0 |
| **`RNA_ENGINE_ALLOW` + `probe_declined_options.sh`** | every declined option forced onto the device and recorded | zero candidates: all ten guards earn their keep |
| **`option_status_census.py`** | the option counts come from a tool now | 60 options, 0 unclassified |
| **`Finished_Port` + `Lukes_Flow_Batching`** | base tip and development fork, tree cleaned | `git diff v2.7.2..Finished_Port` |

Every one of these is pushed. `verify_option_parity.sh` was re-run **45/45
identical** after each change that touched `RNAfold.c`.

---

## 2. What the A100 runs overturned

Three of the day's bigger claims were our own, and the measurements took them
away. That is the useful part of the record.

**§32 — §31.1's inflation warning was itself wrong.** Phase-synced wall 90.64 s
vs production 90.46 s: **0.2 %**. Every absolute figure in §30 stands. The
mechanism is the interesting half: under forced syncs the parts sum to the whole
*and production has the same wall*, so **production overlaps essentially nothing
today**.

**§34 — the roofline position is a property of the GRID, not the kernel.** §33.1
measured `modular_decomposition` at 7.9 % of DRAM peak on a 142-block fixture
and retired the L4's "move fewer bytes" conclusion. At the production shape
(grid 16 683, 77 waves/SM) the same kernel reads **82.8 %**. It is
**bandwidth-bound in production**. `imc_miss` explains itself away on the same
axis (3.25 → 0.02). **Quote the grid with every roofline number.**

**§35 — the occupancy hypothesis is refuted.** Raising warps per block lifts the
ceiling (50 → 56.25 %) and **lowers achieved occupancy** (42.5 → 36.0 %), with
`int_loop` +0.0/+1.6/+7.4 %. The ceiling was never the binding constraint: at
c=1 the kernel achieves **85 % of it** with 151.8 waves/SM. It is
**retirement-limited**, and binning cells by candidate width is the only lever
left. `__launch_bounds__` work is off the list.

---

## 3. Everything that lied today, and why

The session's real yield. Each of these reported success or silence while
failing to reach its target.

1. **`RNA_HC_VERIFY` compared two unconstrained matrices** — it rebuilds from
   `hc->mx` *at pack time*, and the masks were packed before
   `vrna_fold_compound_prepare()` materialised the depot. It reported
   `0 mismatching` on exactly the folds that came out wrong.
2. **The hairpin gate went into dead code.** `fill_arrays_loop.c` *looks* like
   where the hairpin term is combined; that loop sits under
   `if(!rnafold_gpu_sweep())` and the default is the GPU-resident sweep. The bar
   failed again, byte-for-byte unchanged, which is what pointed at
   `new_c_kernel`.
3. **A device-side wait does not bound the HOST.** Stream-overlap level 2 queued
   `hp_mb(i−2)` while `fml_scan(i)` still read its buffer — two-deep parity
   buffers were not enough because the host runs ahead freely. It returned **two
   different wrong answers from two runs of the same arm**, at 400 × 5601 and
   never at 60 × 1500.
4. **Scaling §A measured nothing.** All six fixtures were **one chunk**, and a
   one-deep pipeline holds a second chunk only when a second exists. It reported
   the pipeline costing 0.00 GB at every shape. Fixed by capping records per
   chunk at `n//2` and asserting `chunks >= 2`.
5. **An occupancy ladder written for a register count the build did not have.**
   `ncu` reported **53 regs/thread**, not the 48 the scope argued from — the
   `up_int` parameter added that morning for `-C` cost 5. **Register counts move
   with the source, not just the toolkit.**
6. **A fixture in the wrong alphabet inverted a verdict.** `--energyModel` on
   ACGU makes *no pair legal*: both routes return all-dots at 0.00 and the probe
   reported `AGREES 8/8`. On ABCD the same binary has the CPU at −45.10 and the
   device at 0.00.
7. **A motif that could not bind.** Three attempts, including a synthetic
   four-pair motif at −30 kcal/mol. A ligand motif matches a **sequence and a
   structure**, so the fixture has to be *derived from a real fold*.
8. **Hand-maintained counts drifted.** The option summary said 31/9/19/1 while
   the rows said something else. It is produced by a tool now.

The through-line, and it is worth saying once: **derive the test input from the
program's own output, and assert that the thing under test was actually
reached.** `probe_declined_options.sh` now makes three such assertions before it
will report a comparison at all.

---

## 4. Open

**Needs an A100 — the only thing that does:**

* **`CUDA_RNAFold_Scaling.ipynb` §G, level 2.** The parity-gate fix is correct by
  construction and **unverified at 400 × 5601**, which is the only scale where
  the race fired. The section is ABBA-ordered and refuses to interpret timings if
  any sha moved. Level 1 (−0.9 %) is already clean.

**Needs you:**

* **`--shape`** — deferred on a test bench: real reactivities, a case that
  **changes the CPU answer**, and at least one known pseudoknot. Scope is
  settled (Deigan only — one carrier, and the best average performer in Lorenz
  et al. 2016).

**Needs neither — the plan:**

* **`PORT_LUKES_FLOW_BATCHING.md`**, in its own order: instrument the traffic
  claim, then **T2** row-granular stream-out (cheapest, no shared state, and a
  12.9 s tail matters *more* once the sweep is faster), then **T1** column
  tiling (`md` 37.8 → 12–15 s, the largest lever in the project), then re-test
  int16 **as a check on T1**, then T3.

---

## 5. Position

400 × 5601 nt on an A100-SXM4-40GB, shipped defaults: **85.9 s**, sustained
across eight consecutive folds (0.7 % spread, `0x0` throttling, one sha).
**32.4×** upstream's CPU path on the same workload.

**25 options accelerated, 10 declined with route *and* effect measured, 0
unclassified.** No silent wrong answer is known anywhere on the option surface,
and that now rests on having forced every declined option onto the device rather
than on reading the guard.

---

## 6. Later the same day: the Python binding, and the defect it found

`import RNA` is how ViennaRNA is actually used. The accelerator was reachable
only from `RNAfold`, so its adoption was whatever the command line got — which
for a backend whose whole advantage is folding a **batch** at once is close to
nothing. `interfaces/cuda.i` closes that:

```python
import RNA
RNA.cuda_devices()          # usable CUDA devices; 0 is not an error
RNA.cuda_enable()           # register the batch backend (idempotent)
RNA.cuda_fold(seqs)         # -> [(structure, energy), ...]
RNA.cuda_fold(seqs, md)     # ... under a model of your own
```

`cuda_fold()` is correct with or without a GPU: with no device registered it
folds through upstream's own `vrna_mfe()`. A script should not have to branch on
hardware to be right.

Three helpers rather than a wrapped `vrna_mfe_batch()`, because `RNA.i:243`
ignores every `^vrna_` function and the C signature asks the caller for a
pointer-to-array-of-pointers plus two output buffers. `my_*` + `%rename` is the
style this interface already uses for exactly that reason.

### What it found on its second call

**A second `vrna_mfe_batch()` in one process reused the FIRST batch's device
state.** `init_gpu/2/3` each open with `if(!first) return;`, and the teardown
that resets those flags was called by **`RNAfold.c` only** — "release the device
state this chunk sized, before the next chunk sizes its own". The backend's own
callback never did it, so for any caller that was not the driver:

| second batch | what happened |
|---|---|
| identical to the first | correct |
| **different model** | **wrong answers, no error** — the per-batch parameter upload sits *below* that early return, so a batch at 25 °C was folded with the 37 °C tables |
| **longer records** | CUDA error 719, or an invalid-argument copy out of an under-sized buffer |

Invisible from `RNAfold` by construction: one model per run, records sorted
descending, so no chunk ever outgrows the first. Reached by the Python binding
immediately, because that is what a script does — fold a batch, fold another.

**Fix:** the teardown moved into `cuda_batch_cb()` (`mfe/cuda/engine.c`), where
it is the last step of folding a batch rather than the caller's housekeeping.
The driver's own calls stay and become no-ops — all three teardowns are
idempotent. Cost is one free/re-allocate round per batch, which the driver was
already paying per chunk; `stage_teardown_s` now reads 0.000 because the work is
charged inside the batch instead.

**Bars.** `tests/python/test_RNA-cuda.py`, seven checks, every one a parity
check against `RNA.fold_compound().mfe()` so the file is meaningful on a machine
with no CUDA at all. Red-teamed the only way that means anything: with the three
teardown calls removed and the tree rebuilt, checks 5, 6 and 7 **fail** — wrong
structures, not errors. CLI parity is unchanged: 24 records, CPU vs 1 chunk vs
4 chunks, one sha across all three; `mfe_cuda_guard` 15/15.

`tests/mfe_cuda_guard.ts` check 11 inverted with it —
`test_guard_declines_hard_structure_constraints` became
`test_guard_accepts_...`. It had been stale since `-C` shipped this morning;
it asserted the decline the constraint work removed.

### Build glue

The extension links `libRNA_conv.la`, a **convenience** library, which carries
no link dependencies of its own — so `$(CUDA_LIBS)` on `libRNA_la` does not
reach it. Without it the module builds cleanly and then fails at import with
`undefined symbol: cudaMemcpyAsync`. Added under `if VRNA_AM_SWITCH_CUDA` to
both `interfaces/Python/Makefile.am` and `interfaces/Perl/Makefile.am`; empty
when CUDA is off.

The other import-time failure, `undefined symbol: _Z17vrna_cuda_devicesv`, is
C++ mangling: the wrapper is compiled as C++ and no ViennaRNA header carries a
linkage guard, so `cuda.i`'s `%{ %}` block wraps its includes in `extern "C"`
exactly as `RNA.i:16` does for every other header.

### Known, not ours

`python/test_RNA-utils.py` check 8 ("Slice pair table") fails on this box:
`SWIGPY_SLICEOBJECT` does not resolve under swig 4.4.0 + Python 3.14, so the
slice overload is never generated. Nothing to do with CUDA — it is upstream's
`var_arrays.i` against a newer toolchain than it was written for.
