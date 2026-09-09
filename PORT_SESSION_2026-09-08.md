# Session 2026-09-08 — the audit, the fix, and where the wall clock actually goes

*16 commits, `9d3f63cc` → `486d277a`, all pushed to `origin/port27`.
Previous handoffs: `PORT_SESSION_2026-09-06.md`, `PORT_SESSION_2026-09-05.md`.*

**Read this first, then `PORT_OPTION_STATUS.md` (one live defect is open),
then `PROFILE272_DEEP_RESULTS.md` (the performance picture changed).**

---

## 0. The three things that matter most

1. **`--nsp` is a live silent wrong answer and is NOT fixed.** Accelerated,
   ungoverned, 28 of 30 records disagree with the CPU deterministically. It needs
   a guard. `PORT_OPTION_STATUS.md`.
2. **The performance target moved.** `modular_decomposition` is **31 %** of wall
   at 120 × 5601, not the ~73 % everything assumed. Amdahl now predicts
   end-to-end to 1.5 %, and **int16 is finished as an optimisation target**.
3. **`--noLP` + `RNA_SLOT_FLOW` was a live wrong answer and is fixed** — found by
   the combination audit that had been owed since 2026-09-07.

---

## 1. The pairwise audit, and the defect it found

`tools/verify_option_matrix.sh` (new). 36 pairs: the four CLI-reachable accepted
options against each other **in both enabling orders**, each against the five env
switches, and the switches against each other. Asserts the three properties
`PORT_FEATURE_AUDIT.md` asked for, plus a fourth — that the run took the **route**
it claims.

### `--noLP` + `RNA_SLOT_FLOW`: 17 of 30 records wrong

| | |
|---|---|
| `--noLP` alone | GPU == CPU |
| `RNA_SLOT_FLOW=2` alone | GPU == CPU |
| **together** | **17 of 30 disagree, deterministically** |

Diagnosed with `PORT_NOLP_SPEC.md`'s bar, not an energy comparison — which
mattered, because **every returned structure was self-consistent** (each
re-evaluated under `RNAeval` to its own printed energy) and merely *suboptimal*:
worse on 17, better on 0. An energy check *and* a self-consistency check would
both have passed. Only comparison against upstream catches it.

**The obvious cause was not the cause.** `reset_slot_md()` never covered noLP's
`d_cc`/`d_cc1`; fixing that changed **nothing**, byte-identical 17/30. The real
cause: **`refill_gpu3()` runs at every handover and takes no slot argument.** It
calls `init_gpu3()`, whose `SLOT_ALLOC` skips the malloc on a refill but whose
*kernel launches still run* — so every handover INF-filled `cc`/`cc1` for the
**whole batch**, wiping the mid-recursion state of every record still running
elsewhere.

**The tell was in the counts: 15 handovers but 17 wrong records.** The wrong set
could never have been just the incoming occupants. `RNA_SLOT_FLOW=1` (flow on,
zero handovers) was correct throughout, which located it.

Fixed in `364f92f6` with both halves: `!g_refill3` on the prefill, plus a
per-slot `reset_slot_nolp()`.

### Triples: 24 focused, all green (`45e02946`)

Every triple contains a flow dimension, because flow is where per-slot state
lives. 24/24 identical. **The matrix deliberately stays pairwise on that
evidence** rather than being generalised to N-ary speculatively. What it does not
prove is recorded: 24 of 84, `RNA_FML_INT16` excluded entirely.

---

## 2. Five instruments that lied

This is the session's through-line, and it is worth reading as a list.

1. **`bar_preflight.sh` could not see a build tree at an old commit.** Its test is
   "binary newer than the sources *in this tree*", and an old checkout has old
   sources. `~/port27cuda` sat three commits back and the first matrix run was
   about to judge a binary with `noLP` still declined. It **printed** the tree's
   commit throughout — printing a fact is not checking it. Now compares against
   the tree the bar itself came from.
2. **The same file then cried wolf**, scanning generated `*_cmdl.[ch]` that
   gengetopt rewrites mid-build, failing a perfectly current binary. The mirror
   of (1) and just as bad: a bypassed bar is a false pass.
3. **`verify_option_parity.sh` printed the route and never asserted it.** An
   option that quietly stopped being accelerated still scored `identical [CPU
   route]`. Now takes an expected route; 18/18 with the route asserted.
4. **`peak/iteration N records` is a PEAK, not a total.** Under `RNA_SLOT_FLOW=2`
   it reads 15 for 30 records, so summing it claims a 50 % CPU fallback about a
   run whose row and cell totals are *identical* to the non-flow run.
   **Benchmark v5 sums this same field** and is correct only because no v5 arm
   enables flow.
5. **`RNA_ROW_VERIFY` + `--noLP` lied in both directions.** 922 897 of 8 262 880
   cells reported "mismatching" — every one a false alarm against a device result
   byte-identical to upstream — *and* it corrupted the answer, because in verify
   mode `load_my_c` uploads the host's non-noLP `new_C` over the device's. The
   returned fold matched neither upstream's `--noLP` answer nor the plain fold.
   Refused at init in `ec7c4f7d`.

**And two in my own work**, both caught by results that contradicted themselves:
`RNA_GPU_CHUNK=7` is *below* `MIN_GPU_BATCH=10` so every chunk fell to the CPU;
and a regex matching `[0-9]+ cells;` summed the sweep total *and* the peak,
producing the self-refuting line "0 % of cells folded on the CPU".

### The regression test that could not fail

The first version of the new `verify_nolp_parity.sh` slot-flow arm **passed
against a binary with the fix deliberately removed.** Its fixtures are
uniform-length, so every slot retires on the same iteration and the global wipe
lands when all neighbours are also starting fresh — harmless. Measured on the
broken binary: **0 of 60 records wrong at one distinct length, 17 of 30 at
thirty.**

**Any future slot-flow or continuous-flow bar must use MIXED lengths and assert
that it does.** The arm now builds its own mixed fixture, asserts both
preconditions, and was confirmed RED on the reverted build before being accepted.

---

## 3. Benchmark v5 — 32.4× and a cheaper baseline

`5c1957eb`, results `1ebdd56c`, doc `BENCH272_V5_RESULTS.md`, raw
`bench272_v5.json`. **Tesla T4**, 400 × 5601.

| arm | wall | vs upstream |
|---|---|---|
| A upstream 2.7.2 | 25 633 s = **7.12 h** *(extrapolated)* | 1.00× |
| C int32, 5 chunks | **791.9 s** | **32.37×** |
| E int16, 4 chunks | 784.8 s | 32.66× |

- **32.4×** at this size, against 8.06/5.65/7.88× at the sizes v1–v3 could
  afford — the batch machinery only has room when there are enough records.
- **A/B = 0.9902**: the port costs upstream's own CPU path nothing, at ten times
  the workload of the first timing. *This is the row to lead with in a PR.*
- v5's method: **94 CPU folds against v4's 230** for the same 10× reach, because
  `vrna_cstr_fflush()` flushes per record so one streamed run yields `t(1)…t(40)`.
  The substitution was *checked*: a stream-built model predicted standalone runs
  within 0.96 %/1.05 %.
- `valid: False` is an **artifact** — Colab cloned a stale `origin/port27`
  (`10acb583`) where `RNA_MIN_GPU_BATCH` did not exist. `C/Cm = 0.995`
  corroborates. **Push before running.**

---

## 4. The performance picture changed completely

### int16, profiled (`54386f62`, `PROFILE272_RESULTS.md`)

L4 at full clocks. **The kernel is 1.46× faster under int16.** DRAM reads fall to
0.624, which pins `fml_j` at **~75 % of the kernel's read traffic**
(`1 − f/2 = 0.624`). It **never leaves the DRAM roof** (89.3 → 80.9 % of peak), so
there is no second lever behind int16. SM doubles 23.5 → 46.1 % — that is the
decode cost, and it hides under the memory system here, which *is* the
machine-dependence mechanism.

**`lts__t_sector_hit_rate` is 6.1 %** — L2 does **not** catch the per-row `fML`
column re-reads, so the shared-memory staging idea would not duplicate the
hardware. **That is a live lever.**

### The deep profile (`e3763d7d`, `PROFILE272_DEEP_RESULTS.md`)

| size | wall | `modular_decomp` | share |
|---|---|---|---|
| 16 × 2000 | 2.6 s | 0.4 s | 16.7 % |
| 40 × 5601 | 68.5 s | 20.7 s | 30.2 % |
| **120 × 5601** | **201.4 s** | **62.4 s** | **31.0 %** |

**The ~73 % share does not survive scale.** And Amdahl now predicts end-to-end to
~1.5 % (share 31.0 %, kernel 1.610× → 1.133× predicted, 1.117× measured), so we
understand the wall clock.

**int16 is finished as an optimisation target.** An infinitely fast
`modular_decomposition` gives 1.45× at this size and 1.61× is already banked.
This also retires the T4 question: v5's 1.009× needed no appeal to clocks.

**`int_loop` is 0.29 s — 0.1 % of wall.** The kernel that once led and was
"work-bound" is now noise. The batch-width work succeeded; the old mental model
is stale.

### The instrumentation was already there, and half of it was dead (`b799a820`)

`print_stage_timing_stats()` has always reported build/prepare/prefill/backtrack/
output/gpuinit/teardown/free via `atexit` — **the deep notebook parsed only the
phase line and threw the stage line away.** Four of the eight were **dead
counters**: declared, printed, never incremented. They read `0.000` for the life
of the project, which says "this costs nothing" and means "nobody measured it".

| 24 × 2000 | build | output | teardown | free | non-sweep |
|---|---|---|---|---|---|
| before | 0.000 | 0.000 | 0.000 | 0.000 | 0.294 s |
| **after** | **0.379** | **0.303** | 0.005 | 0.009 | **0.954 s** |

Residual at that size fell from ~61 % to **28.8 %**. `build`
(`vrna_fold_compound()` building O(n²) ptype and hard-constraint tables) is the
largest single recovered piece.

**Caveat in the code:** `RUN_IN_PARALLEL` is `thpool_add_work()` when
`max_threads > 1`, so `stage_output_s` measures *dispatch* under `-j`. Read it
only from a run without `-j`.

---

## 5. Upstreaming: the merge map is re-based (`c2e77b56`, `c674a105`)

Re-measured `v2.7.2..port27` rather than restating 2026-09-02's figures:

**Nine upstream files, 996 lines added, SEVEN deleted** — and two of the seven are
in `mfe.c` and are *moved*, not removed. The fork is essentially additive; ~9 200
lines live under `src/ViennaRNA/mfe/cuda/`. The reviewer's question is not "what
did you change in ViennaRNA" but "will you carry a new subdirectory".

Three of MERGING.md's central claims were obsolete, all in our favour: the
2.3.0→2.7.x port is **done**; `params.c` is **not** modified (the race is
*reported* as Defect B, which is the right shape for a PR); and **the symbol
collision is gone** — the seam replaced it, and `vrna_mfe_cpu()` survives only as
a stale comment at `mfe_cuda.c:1107`.

**The critical path is no longer code.** Steps 4 and 6 are ours whenever we
choose; step 5 and circular both wait on a conversation with the maintainers that
has not been started.

### Step 4 scoped, and its mechanism rejected (`PORT_CONFIG_SCOPE.md`)

Keep the goal, reject `vrna_md_t`: wrong struct (energy model vs scheduling), an
ABI break for every consumer, and it promises per-compound scope for **six knobs
that are process-wide caches**. Instead: per-compound settings ride the seam's
existing `void *data` (**zero upstream lines**); process-wide gets an honest
setter; the 11 autotuning knobs and 2 verifiers stay environment-only.

Two couplings recorded: **every refusal must move with the configuration** (three
pairings are refused by reading `getenv`; an API route would bypass them — the
`-C` hole's shape), and the combination audit is env-shaped, so a second route
doubles what it must cover.

---

## 6. Circular: correct today, and the blocker is narrower (`08a71e6d`)

**Circular already produces the right answer**, at both levels: `RNAfold -c`
disables the accelerator for the whole invocation, and `vrna_mfe_batch()` declines
circular compounds so its fallback loop calls `vrna_mfe()` — which reaches
`mfe.c:322` and runs `postprocess_circular()`. A *mixed* batch folds its circular
records correctly on the CPU while everything else is accelerated.

So the `PRIVATE` symbol does **not** block circular. It blocks circular *inside
the batch backend*, the one path that never reaches that line.

| tier | | cost |
|---|---|---|
| 0 (today) | decline, fold on CPU | correct, *k*× slower |
| **1** | **GPU fills incl. `fM2`, host post-processes** | **one chunk width of VRAM** |
| 2 | post-process on device too | not worth it |

**Tier 1's PCIe cost is ~zero** — the matrices already come back for
backtracking, and `fM2_real` **is** the `DMLi` the sweep discards. Nothing new is
computed or transferred; a row is kept instead of dropped. **The price is a chunk
width**, so measure the chunk-count effect on a real circular workload first.

Tier 2 is dismissed: `postprocess_circular()` is embarrassingly parallel *across
records* and belongs on the existing backtrack pool.

**On the ensemble idea** (Luke): returning local minima is a real scientific want,
and suboptimal structures walk the *same* MFE matrices the GPU already produces,
so it is closer than it sounds. But **Boltzmann sampling cannot ride on this** —
it needs the partition function, which we do not compute at all (`MERGING.md`
§11.1 scopes that as the best next port). And if backtracking turns out to be a
large part of the residual, device-side backtracking is attacking the *dominant*
cost rather than adding a feature.

---

## 7. What is open, in priority order

1. **Guard `--nsp`.** Live silent wrong answer. `-c`/`-g` precedent applies.
2. **Test `-P` / `--paramFile`.** Untested, replaces the entire parameter set; if
   the upload path misses a table it is another `--nsp`, and it is a far more
   commonly used flag.
3. **Run `CUDA_RNAFold_Stress272.ipynb`** (Colab is loaded). 400 × 5601,
   multi-chunk, VRAM-budget sweep, full accounting. It answers: does the residual
   survive multi-chunk, does the kernel share recover at scale, and does the
   ~*k*× chunking rule hold at 5601 nt?
4. **Chase the remaining residual** — input parsing and the chunk-accumulation
   loop are what is left after the four counters.
5. **Restore the noLP host branch** (`fill_arrays_loop.c:215`). Turns the
   `RNA_ROW_VERIFY` refusal from an honest stopgap into a real fix. It is the
   **sixth** entry on the "gcov says not used" list.
6. **Send Part 1 of the upstream proposal** — four defects plus two build
   defects, all independent of whether upstream wants the accelerator.
7. **Circular tier 1**, after measuring the chunk-count cost.
8. Inputs where `--helical-rise` / `--backbone-length` actually bite; a
   distributional bar for `--ImFeelingLucky`; `--batch` is unreachable by the
   current harness.

## 8. Operational notes

- **`origin/port27` is now current at `486d277a`. Push before any Colab run** —
  v5 benchmarked `10acb583` and two arms failed for that reason alone.
- **A git-tree build needs two workarounds** that `standup_git_build.sh` knew and
  the notebook generators did not: unpack `src/dlib-*.tar.bz2` and
  `src/libsvm-*.tar.gz`, and pass `PYTHON3` to `configure` (without it
  `--without-python` leaves `$(PYTHON3)` empty and make tries to *execute* a
  mode-644 `man2rst.py`). Both are now in all four generators.
- **`~/port27cuda` is stale and CRLF-polluted**; `~/port27head` is the current
  build tree. Seven bars still default to the stale one — `bar_preflight` now
  fails loudly on it, but the default is unchanged.
- `ncu --csv` writes to **stdout, and so does RNAfold**. Always `--log-file`, and
  gate on the log, because `No kernels were profiled` is on stdout too.
