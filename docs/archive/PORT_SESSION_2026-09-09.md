# Session 2026-09-09 — two silent wrong answers closed, and the wall moved 641.8 → 424.5 s

*29 commits, `82fdbb07` → `16d77147`, all pushed to `origin/port27`.
27 files, +4397 / −135. Previous handoff: `PORT_SESSION_2026-09-08.md`.*

**Read this, then `PORT_HETEROGENEOUS_SCOPE.md` (the plan), then
`STRESS272_RESULTS.md` §14 (the numbers). One Colab run is IN FLIGHT — see §7.**

---

## 0. The four things that matter most

1. **Both known silent wrong answers are closed and measured.** `--nsp` is
   guarded, `MAX_NINIO` is fixed, and **both bars were confirmed RED without the
   fix**. `PORT_OPTION_STATUS.md` can say, for the first time, that no live
   silent wrong answer is known on the option surface.
2. **The wall went 641.8 s → 424.5 s at 400 × 5601** (T4, same card), from three
   changes: `gpuinit` 153 → 1.6 s, `output` 120.5 → 0.1 s, and the build/fold
   pipeline at −21.4 %. **Output hash unchanged throughout** — `7c0b3d633281`
   across every arm of four stress runs.
3. **Defect B is fixed, not just reported.** `params.c` now locks the
   `SPEEDUP_PARAMS` cache, with a failing test (`tools/params_race.c`:
   0/16 000 wrong locked, 11 999/16 000 unlocked). That should be **re-sent
   upstream as a patch rather than a report**.
4. **The remaining wall is `modular_decomp` 54 %, `hp_mb` 25 %, `backtrack`
   11 %.** Transfers are 4.5 % and closed. `build` is now mostly hidden.

---

## 1. Correctness: two live defects closed

### `--nsp` (`a2eaf80e`, mechanism in `PORT_NSP_PARAMFILE_SCOPE.md`)

Was accelerated, ungoverned, wrong on 28 of 30 records. **Now declined.** The
guard tests `md->pair[i][j] == 7` — the *effect* — not `md->nonstandards`, the
request, because the field is one of several routes into the feature and the pair
table is where all of them land.

**The mechanism was located and is NOT in `engine.c`:** `int_loop.cu`'s `Energy()`
resolves interior-loop pair types with two shortcuts that are exact identities at
default settings — no `0→7` promotion, and an index swap in place of `rtype[]`.
`--nsp` breaks the second at `model.c:1104`, where `rtype[7]` is *forced* to 7.
Both are latent defects independent of `--nsp`; fixing them is ~4 lines and would
let the guard lift.

### `MAX_NINIO` (`8d2b174e`)

A `#define` of 300 on the device, "checked" by `assert(MAX_NINIO == 300)` — which
expanded to `assert(300 == 300)`. The real `MAX_NINIO` is a **writable library
global a parameter file overwrites** (`params/io.c:671`). Measured: with only the
NINIO maximum moved 300 → 80, the pre-fix binary is wrong on **9 of 12 records**.

`-P/--paramFile` **cannot be guarded** — `vrna_params_load()` mutates globals and
leaves no flag — so its assumptions belong in `load_param()`. Six more are ranked
in `PORT_NSP_PARAMFILE_SCOPE.md` §2.3; **bars 2 (`-P DNA`) and 4 (int16 + `-P`)
are still unrun.**

---

## 2. Performance: 641.8 → 424.5 s, in three steps

| step | commit | effect |
|---|---|---|
| `gpuinit` derived on device | `a2d19bd2` | 153.2 → **1.6 s** |
| `output` fast path | `21c6f644` | 120.5 → **0.1 s** |
| build/fold pipeline | `5345f59f` | **−16.8 / −19.2 / −21.4 %** |

### `gpuinit` — and the trap inside it

87 % of `gpuinit` was host bitmask packing. The GPU replacement already existed,
gated on `g_hc_seq_derived`, whose declaration said *"Set once by RNAfold.c"* —
**and nothing set it.** The 2.3.0 branch did; the port lost the line.

**Restoring it returns a WRONG ANSWER on 9 of 10 records.** The missing line was
load-bearing. Two device divergences had to be fixed first: the `i==j` diagonal
open-coded `CLOSING_LOOPS` where upstream writes `ALL_LOOPS`, and `max_bp_span`
was passed as a batch scalar when it is **per record**. Both fixed; the flag is
now *derived* in `par_fill_arrays()` rather than set by a caller, because a lost
assignment to a default-0 flag is invisible — it only makes you slower.

### `output` — the same call, twice

`process_record()` rebuilt the fold compound the chunk loop had already built and
freed. Proved by poisoning: `vc = NULL` on the accelerated path took `output`
0.130 → 0.000 s with **nothing dereferencing it**. Now conditional, with one
shared predicate (`output_needs_compound()`) used by both the fast path and the
pipeline so they cannot drift.

### The pipeline (option B) — validated at scale

| budget | chunks | i32 | i32pipe | delta | hidden |
|---|---|---|---|---|---|
| natural | 5 | 527.5 | 438.7 | −16.8 % | 82 % |
| half | 7 | 530.4 | 428.5 | −19.2 % | 89 % |
| quarter | 14 | 539.8 | **424.5** | **−21.4 %** | **95 %** |

Hidden fraction tracks `(chunks−1)/chunks` exactly. **Costs ~2× host RSS**
(4.86 → 9.40 GB); the residual goes negative in pipelined arms, which is the
correct signal that two timers count the same seconds.

---

## 3. Defect B fixed, and option A

`params.c`'s `SPEEDUP_PARAMS` cache is now mutex-guarded in both `vrna_params()`
and `vrna_exp_params()`, using the `VRNA_WITH_PTHREADS` the library already has.
`get_scaled_params()` stays outside the lock so a miss does not serialise
everyone. Without pthreads the cache is disabled rather than left racy.

**`tools/params_race.c` is the bar and it goes red:** 0 of 16 000 wrong with the
lock, **11 999 of 16 000 (75 %) without**.

On top of it, **option A** (`RNA_BUILD_THREADS`, default off): build 0.471 →
0.105 s on 12 cores, **4.5×**, all arms byte-identical. A and B compose — with A
on, 69 % of the shortened build is still hidden by B.

**Consequence for the PR:** `params.c` is no longer an unmodified file. That
costs a line in `MERGING.md` and buys a real bug fix. **Defect B should be
re-sent as a patch with a failing test.**

---

## 4. Option C — built, correct, and honestly unproven

`RNA_CPU_SLICE`, default off. Holds `m` records back from each chunk onto the
`-j` pool. No new folding code; `m = 0` without `-j`, and the run says so.
Byte-identical in every arm.

**The wall effect is unresolved and this machine cannot resolve it.** Under
sustained load the laptop GPU fell 1057 → 712 MHz; control arms drifted +47 % and
+55 % across single sweeps. Deltas ranged −2.9 % to +12.4 % with no ordering by
cap or pool size.

**Where C stands, with a mechanism rather than an estimate:** the accelerated
path uses **one core of twelve**, and `nsys` shows the busy thread is *executing*
(state R 95 %), not parked. So on ≤4 cores there is no idle capacity and a folder
takes time **directly from the thread feeding the GPU**. `SCHED_IDLE` makes that
safe but worthless there. On ≥8 cores the arithmetic holds (~27–44 %) and
`SCHED_IDLE` should be used so the guarantee is structural.

---

## 5. The transfer path — closed, after two wrong turns

`nsys` put `cudaMemcpy` at **70.4 %** of host time inside CUDA calls. Pinning the
per-row upload staging bought **−17 %** on the average call (351 → 291 µs) and
moved the share only 70.4 → 67.4 %.

**Then I found the premise for "pin the bulk buffers next" was false.** `new_e`
and `energy_min` sit behind `if(!rnafold_gpu_sweep())`, and the GPU-resident
sweep has been default since 2026-08-30 — **they do not execute.** The remaining
calls are the small uploads, and their cost is WSL2/WDDM per-call submission
latency, not staging.

**On the T4, transfers are 3.6 % of wall.** The 70 % was API-time share on WSL2,
a local artifact. The transfer path is not worth more work; cutting it further
means cutting the *number* of calls, which is a much bigger change.

---

## 6. Method — six probes that could not reach what they tested

Recorded in `feedback_probes_that_could_not_reach`. The ones that cost real time:

1. **8 records < `MIN_GPU_BATCH`** → an "accelerated" profile described a **pure
   CPU fold**. Always assert `sweep shape:` is present.
2. **`cudaSetDeviceFlags` after `cudaGetDeviceCount`** fails silently — the probe
   creates the context. A knob that quietly does nothing is worse than a missing
   one.
3. **"the importer is nowhere on the system"** — concluded and committed before
   the `find` returned. It was there.
4. **Summed stage timers and called the remainder "spin"**, omitting the phase
   timers that accounted for it.
5. **A/B/A/B is biased** under a monotonic trend: median said −1.0 % and
   min-vs-min said +13.2 % from the same ten runs. **Use ABBA.**
6. **A rebuild during `make check`** produced a false FAIL (trap 8, one mutation
   per tree at a time).

Three red-team bars were confirmed RED before their green was believed: the
`--nsp` guard, the `MAX_NINIO` fix, and the params.c race.

---

## 7. IN FLIGHT and what to do first tomorrow

**A Colab run is executing** the notebook at `16d77147`: four arms per budget
(i32, i16, i32pipe, **i16pipe**), budgets ordered quarter → half → natural, with
a host-RAM guard that skips a pipelined arm whose projected RSS will not fit.

**i16 + pipeline has never been measured** and is the largest untested number:
int16 alone is −3.6 %, the pipeline alone −21.4 %, and `modular_decomp` is 54 %
of the post-pipeline wall. Expect i16pipe/quarter near 360–370 s if they compose;
**if it comes in flat, that says the `hp_mb` int16 regression eats the gain once
build is hidden** — which is its own finding.

Expect `i16pipe/natural` to be **skipped**: it projects to ~12.4 GB against a T4
instance's ~12.7 GB. That is the guard working.

### Then, in order

1. **Read the run.** If i16pipe composes, int16's default-off status is worth
   revisiting on many-chunk workloads.
2. **`backtrack` is 11 % of wall** and the pipeline does not hide it — it runs
   inside the fold. Already threaded on `auto`; the question is core starvation
   versus a real serial section. **Largest un-attacked host item.**
3. **Re-send Defect B upstream as a patch** with `tools/params_race.c`.
4. **`-P DNA` and int16 + `-P`** — the two unrun bars, and `-P DNA` also closes
   `--helical-rise` / `--backbone-length`.
5. **Fix `Energy()`'s two type divergences** and re-measure `--nsp`; the guard
   hides them but they are latent.
6. **`MIN_GPU_BATCH` should be derived from record count and length** (Luke).
   Every GPU-path speedup moves the break-even, so a constant drifts out of date
   with each improvement — and it is the honest way to report how many jobs are
   not worth the GPU. It has already caused two measurement bugs this session.

## 8. Operational

- **`origin/port27` is at `16d77147`. Push before any Colab run.**
- **The WSL box was rebuilt** and its toolchain reinstalled this session:
  `wsl -u root` needs **no password**; gcc, nvcc 12.4, doxygen, gengetopt, check.
  **Install doxygen BEFORE `./configure`.** `~/port27head` is the live CUDA tree.
- **`nsys` works but does not auto-import** — run
  `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter` by hand.
- **This laptop cannot hold a GPU clock** under sustained load (1057 → 712 MHz).
  Any local wall measurement below ~15 % is noise; use ABBA and medians, or run
  it on Colab.
- New knobs, **all default off**: `RNA_BUILD_PIPELINE`, `RNA_BUILD_THREADS`,
  `RNA_CPU_SLICE` (+`RNA_CPU_SLICE_CAP`), `RNA_GPU_BLOCKING_SYNC`.
  The `params.c` fix is **not** gated — it is correctness.
