# Resource budgets (RAM, VRAM, threads) with backtrack pipelining P3/P4: plan

Signed off 2026-10-13 (section 7). Started after the cleanup (`post-cleanup`), as queued in
PORT_CLEANUP_PLAN.md section 9, and run together with the rest of N5a
(PORT_BACKTRACK_PIPELINE.md) because the two decide the same things: how many chunks may
be resident, how much host memory a trace holds, and how many threads compete for the
cores.

## 1. Why

Students run this on laptops with a hundred browser tabs open. Today every run takes as
much as it can see: 85 % of free VRAM, a backtrack worker per core, and host memory up to
half of what is available for the build pipeline. A run that is correct but freezes the
machine is a failure for that user.

## 2. Decisions already made (Luke, 2026-10-12; not re-asked)

| | |
|---|---|
| Units | a number (MB for RAM and VRAM, a count for threads) or `quarter` / `half` / `threequarter` / `full` |
| Fractions | of the machine's TOTAL, as an upper limit that still applies today's soft limits, so the run keeps adapting to what is free right now |
| Defaults | Python `half`; the CLI `full` (today's behaviour) |
| A record bigger than the budget | warn, and fold it anyway |
| Threads | join the budget: build, backtrack and trace threads |

## 3. Measured first (M0, laptop, 2026-10-13)

RTX 3050 4 GB, 12 threads, WSL with 6.6 GB. Peak RSS (`/usr/bin/time -v`), one consumer toggled
per arm; output identical in every GPU arm.

| arm | A_longseq (60 × ≤2800, 1 chunk) | C400_lite (12 × 5601, 12 chunks at 256 MB) |
|---|---|---|
| default | 1053 MB, 4.60 s | 353 MB, 14.5 s |
| build pipeline off | 1053 | 308 (−45: the next chunk's compound) |
| build threads 1 | 1053 | 353 |
| **backtrack threads 1** | **537** (−516: 11 workers × 47 MB), 4.35 s | 353 (one record per chunk uses one worker anyway) |
| **backtrack pipeline on** | 1052 | **714** (+361: the retained chunk's trace beside the next sweep) |
| scratch pinned, no stage | 981 | 345 |
| CPU only, `-j1` / `-j4` | 54 / 179 MB | 178 / 673 MB (≈ 165 MB per concurrent 5601 fold) |
| GPU `-j4` + CPU slice | 1030 | 490 |
| Python `RNA.fold(list)`, GPU / `RNA_GPU=0` | 1073 / **1410** | — |

What it says:
1. **Backtrack workers are the largest host consumer and buy nothing here.** Each holds a c/fML
   pair sized to the chunk's longest record (2 × 4 × L²/2 bytes, measured ≈ 1.5× that with its
   stack); the default is one per core. Twelve to one cut peak RSS in half and the wall did not
   rise (4.60 → 4.35 s).
2. **The backtrack pipeline doubles the host peak at 5601 nt.** Its AUTO gate (N5a §3.5) prices
   VRAM and the build pipeline's gate, not the trace's host scratch.
3. **A gap that can OOM today:** Python on the CPU path peaks 1.3× above the GPU path. The window
   model in `interfaces/cuda.i` charges each compound 1.5 L² bytes, unfolded; folded on the host,
   a compound keeps its full matrices (≈ 5.4 L² bytes) until the window is freed. On a GPU-less
   laptop, the case the budgets are for, the model is about 3.5× short.
4. CPU folds cost ≈ 5.4 L² bytes each, so `-j` and the CPU slice scale the peak with the pool.

**M1 (laptop, partial):** with our context alive the backend sees 3304 MB free of 4096 (usable 2808 = 0.85 × free), so 792 MB is held by the context plus whatever Windows keeps on the dGPU. Under WSL `nvidia-smi` cannot split the two (it reports 0 MB idle and 81 MB after a fold), so the context alone is measured on the A100 in P4, where the driver is native. The device peak per chunk is still to measure.
Still to measure: **M2**, the host model's constants across lengths 600 / 1200 /
2800 / 5601 and 1 / 4 / 12 workers, so the prediction is fitted, not asserted.

## 4. Design

**4.1 One resolver.** `rnafold_budget_resolve()` in the backend, used by RNAfold, the library
path (`engine.c`) and Python. Inputs: `RNA_RAM_BUDGET`, `RNA_VRAM_BUDGET`, `RNA_THREAD_BUDGET`
(and `RNA_GPU_VRAM_BUDGET_MB` as an alias), or Python's `RNA.set_budget(ram=, vram=, threads=)`
and per-call keywords on `RNA.fold` and `fold_compound.mfe`. Output, per run and announced once:

```
budget: RAM 3389 MB (half of 6776; 6103 available) | VRAM 1638 MB (half of 4096, minus the context, measured in M1) | threads 6 (half of 12)
```

- RAM = min(fraction × MemTotal, today's soft limit on MemAvailable), re-read at each chunk.
- VRAM = min(fraction × total, 0.85 × free) − the context, re-read after each teardown, as now.
- Threads = max(1, fraction × hardware threads).

**4.2 A true host prediction.** One function prices a chunk's host peak:
`compounds(chunk) [+ compounds(next) if the build pipeline runs] + workers × scratch(L_max)
[+ the retained chunk's trace scratch if the backtrack pipeline runs] + concurrent CPU folds ×
5.4 L² + stages + base`, with M2's constants. Admission uses it. When it does not fit, shed in
this order, cheapest first (M0 says workers cost nothing to shed): backtrack workers, then the
backtrack pipeline, then the build pipeline, then the chunk width. Python's window uses the same
function, and charges a host-folded compound its matrices.

**4.3 VRAM.** `compute_gpu_usable_bytes()` takes the resolved budget; the context counts against
it; while a chunk is retained for the backtrack pipeline, the next chunk is sized against what is
left (N5a's "two chunks fit" gate becomes this one computation).

**4.4 Threads.** Build, backtrack and trace threads draw from the budget (the trace pool and the
builder share it, N5a §3.6). Default `full` in the CLI changes nothing.

**4.5 P3.** `engine.c` already splits a list into VRAM-sized sub-chunks; it may hand sub-chunk k's
trace to a trace thread while k+1 sweeps, and joins all of them before `vrna_mfe_batch()` returns,
so the library call stays synchronous. The retained chunk is charged by 4.2 and 4.3.

**4.6 Evidence and refusals.** A record over budget: one warning naming the record, its need and
the budget, then folded. Every decision that shed something says what it shed.

## 5. Stages and bars

| stage | content | bar |
|---|---|---|
| M1 | context and per-chunk device peak (laptop; A100 in P4) | numbers recorded |
| M2 | host model constants; predicted vs measured | prediction within 10 % of peak RSS on every M0 arm |
| B1 | resolver, env, Python API, the report line | parser unit test; the line appears; `full` changes no byte and no chunk count |
| B2 | VRAM: context counted, retained chunk charged | device peak ≤ budget at quarter / half / threequarter (sampled), output identical |
| B3 | RAM: model in admission, shedding order, Python window incl. the CPU path | peak RSS ≤ budget at each level on the M0 fixtures, CLI and Python, GPU and `RNA_GPU=0`; output identical; stderr shows what was shed |
| B4 | threads | observed threads ≤ budget; output identical |
| P3 | `engine.c` overlap | binding suite; `RNA.fold(list)` multi-chunk, identical to off and to CPU; positive evidence of overlap |
| P4 | A100: N5a arms plus budget arms (half and full at 400 × 5601) | by eye, as N5a 9.5 |

Every stage also runs the standing bars (`tools/coverage/bars.sh`) and the zeroconf bar on a copy.

## 6. Synergies and tradeoffs

| pair | relation |
|---|---|
| budgets + backtrack pipeline | SAME COMPUTATION: the pipeline's AUTO gate is "two chunks fit the budget" |
| backtrack workers vs wall | M0: shedding them is free on the laptop; P4 checks the A100, where the host pools were measured to pay (build + backtrack auto vs serial: 1.23×, Colab 400 × 5601, 2026-08-30) |
| RAM budget vs chunk width | TRADEOFF: narrower chunks cost about 2 s per chunk on the A100 (`project_chunking_costs`); shed last |
| threads budget vs `-j` | see decision 1 |

## 7. Decisions for sign-off

1. **Explicit settings vs the budget.** If a user sets `-j 8` or `RNA_BACKTRACK_THREADS=8` and also a
   thread budget of 4: the explicit setting wins, with a warning that it exceeds the budget.
   *Recommended* (the budget governs defaults; it never silently overrides a number the user typed).
2. **Shedding order** under a RAM budget: backtrack workers → backtrack pipeline → build pipeline →
   chunk width (4.2). *Recommended*, per M0.
3. **Fix the Python CPU-path window now**, as B3's first item, since it can OOM a GPU-less machine
   today. *Recommended.*

**Signed off by Luke, 2026-10-13: "Yes to all three, go ahead."** Explicit settings win over the
budget, with a warning; the shedding order is backtrack workers → backtrack pipeline → build
pipeline → chunk width; the Python CPU-path window is fixed first.

## 8. Results

### 8.1 Decision 3: the Python CPU-path window (2026-10-13)

**M2a, host bytes per MFE compound** (RSS per compound, built / folded by upstream's
`vrna_mfe()`): 2.24 / 6.30 L² at 600 nt, 1.28 / 4.83 at 1200, 1.13 / 5.01 at 2800, 1.28 / 4.93
at 5601. The window charge stays 1.5 L² + 128 L for a compound the device folds, and is
5 L² + 1024 L (bounds every row) for one the host folds. Whether the host folds is decided once
per call: the backend is unregistered, `RNA_GPU=0`, no device, or the engine declines the first
compound (every record shares one model and no constraints).

`RNA.fold(list)` on A_longseq (60 × ≤ 2800 nt), peak RSS, same answer in every arm:

| arm | before | after |
|---|---|---|
| CPU path, `RNA_HOST_AVAIL_MB=600` (300 MB windows) | **1083 MB** | **345 MB** |
| GPU path, same | 905 | 906 |
| CPU path, real MemAvailable (all 60 fit half of it) | 1415 | 1415 |
| GPU path | 1078 | 1078 |

Bar: `tools/coverage/py_window_check.sh` (CPU arm ≤ 420 MB, one sha); it fails on the old
code. Binding suite 20/20. Note the GPU arm: 906 MB under a 300 MB window, because the window
bounds only the compounds, not the 12 backtrack workers -- that is B3.

### 8.2 B1: the resolver, the API and the report (2026-10-13)

`src/ViennaRNA/mfe/cuda/budget.c`, built into the CUDA and the CPU-only library. `engine.h`:
`vrna_budget_set/default/ram/vram/threads/is_full/report`, `vrna_cuda_device_memory()`.
Environment `RNA_RAM_BUDGET`, `RNA_VRAM_BUDGET`, `RNA_THREAD_BUDGET`; Python `RNA.set_budget()`
(all or nothing; `""` refused -- `reset=True` is the way back), `RNA.budget()`, and
`ram=` / `vram=` / `threads=` on `RNA.fold` and `fc.mfe`, restored after the call. Python sets
`half` at import. Nothing obeys a budget yet (B2-B4).

Bars: binding suite 27/27 (7 new: Python default half, words and numbers, junk refused and
nothing left behind, precedence set > environment > default, per-call restore, one report
line, silence at full and when quiet); refcheck C_mixed, extreme, F_lite @ 64 and C400_lite @
256 identical with the same chunk counts (the CLI default `full` changes nothing); the CLI
prints `budget: RAM 2048 MB (set) | VRAM full | threads 6 (half of 12)` when one is set,
nothing by default, and names an unparseable value. **Zeroconf owed**: the run was stopped for
low host memory before it reached it.

Build-tree note: after a rebuild the binding imports `interfaces/Python/RNA/RNA.py`, a copy of
the regenerated `RNA.py` that make may skip when Windows and WSL clocks disagree; copy it, or
a fixed test keeps failing against the old module.

### 8.3 B2: the VRAM budget (2026-10-13)

`compute_gpu_usable_bytes()` takes `RNA_VRAM_BUDGET` (whole process) on top of the soft limit;
`RNA_GPU_VRAM_BUDGET_MB` stays the buffers-only test cap. What the process holds is MEASURED:
the context and driver reserve as the total minus what is free at the first query, its buffers
as what has left the free pool since -- so a chunk retained for the backtrack pipeline is
charged with no ledger. **M1 settled under WSL:** every process sees 3304 of 4096 MB free
whether or not another holds a context, so the 792 MB is this process's own reserve. The room
left under the budget gets the soft limit's own 0.85 margin: without it two quarter-budget
arms ended 8 and 14 MB over (the chunk model misses allocator granularity and side buffers).
A record over the budget but within the soft limit is warned about (five named, then a count)
and folded ALONE on the device; only one over the soft limit goes to the cores, as before.
Evidence: par_mfe() samples free VRAM after each chunk's allocations and at its sweep's end,
and the exit line reports the peak held.

| fixture | full | threequarter | half | quarter | half + bt pipeline | 900 MB |
|---|---|---|---|---|---|---|
| C400_lite (12 × 5601) | 1 sweep | 1, 2420 / 3072 MB | 2, 1746 / 2048 | 12, 938 / 1024 | 3, 1848 / 2048 | 12, 6 warnings, 938 (over, as warned) |
| A_longseq (60 × ≤ 2800) | 1 | 1, 1958 / 3072 | 2, 1864 / 2048 | 8, 1004 / 1024 | 2, 1864 / 2048 | 17, 892 / 900 |
| F_lite (57) | 1 | 1, 1136 / 3072 | 1, 1136 / 2048 | 2, 996 / 1024 | 1, 1136 / 2048 | 5, 898 / 900 |

Output identical to the frozen reference in all 18 arms; `full` prints nothing and changes
nothing (refcheck 9/9, same chunk counts); binding 27/27 (the report test now also asserts
the peak line says "within it").

Zeroconf (owed since decision 3 and B1, run at the end of B2 at JOBS=1): cases 1-6 pass (15
checks: the CPU-only build compiles budget.c, folds, is silent and byte-identical to the
accelerated build). **Case 7, the CPU-only build WITH the Python module, is owed**: the run was
stopped for low host memory while compiling RNA_wrap.cpp, for the third time this session. It
passed at C9 (JOBS=2) when the laptop had more free memory.

**Zeroconf case 7 closed (2026-10-13):** rerun alone after Luke closed his browser: the CPU-only
build with the Python module builds, has no undefined `vrna_cuda` / `vrna_budget` symbols,
imports with 0 devices, folds on the host with the same answers, and resolves the budgets
(Python default half; VRAM "no ceiling" with no device); `set_budget()` works. All 7 checks
pass, so the zero-config contract holds for decision 3, B1 and B2. Memory, sampled every 20 s:
the SWIG wrapper's compile (cc1plus) peaked at 1.15 GB and WSL never had less than 4.9 GB
available -- the three earlier stops were pressure on the Windows side (13.7 GB, the browser),
not inside WSL.

### 8.4 M2b: the host model, fitted (2026-10-13)

Uniform lengths, one chunk each; backtrack workers 1 / 4 / 12 and half the records; then
2800 × 40 in four chunks with each pipeline toggled. Peak RSS:

| L | n | w1 | w4 | w12 | w1, n/2 | per worker | per compound |
|---|---|---|---|---|---|---|---|
| 600 | 200 | 314 | 343 | 419 | 213 | 9.5 MB | 1.01 MB |
| 1200 | 100 | 349 | 394 | 514 | 232 | 15.0 | 2.34 |
| 2800 | 40 | 609 | 746 | 1113 | 378 | 45.8 | 11.6 |
| 5601 | 12 | 804 | 1280 | **2550** | 532 | 158.7 | 45.3 |

2800 × 40 in 4 chunks, 4 workers: neither pipeline 498 MB; build pipeline 578; backtrack
pipeline 958; both 950. CPU path at 2800: 54 MB at `-j1`, 179 at `-j4`.

**The model** (bounds every arm, never under; 0-13 % over):
- per backtrack worker: 8 MiB + 5 L² (7.5 L² with `-c`: fM2 is a third triangle)
- per compound on the GPU path: 1.5 L² + 192 L + 0.5 MiB (RNAfold's old 1.0 L² + 128 L is short)
- process base on the GPU path: 100 MiB (the CUDA runtime on the host); 16 MiB on the CPU path
- after the first chunk: + one chunk's compounds (freed memory the allocator keeps)
- build pipeline: + the next chunk's compounds
- backtrack pipeline: + two chunks' compounds + 1.2 × the workers' scratch
- a host fold: 5 L² + 1024 L each (8.1)

At 5601 nt twelve workers cost 2.55 GB against 0.80 GB with one, at the same wall (8.9 vs 9.2 s).

### 8.5 B3: the RAM budget (2026-10-13)

Four gates, in decision 2's order, each announced when it acts:
1. **backtrack workers** (`vrna_budget_workers()`, at par_mfe's backtrack): as many as fit beside
   what the process holds, at least one; an explicit `RNA_BACKTRACK_THREADS` wins, with a warning.
   Under a budget the grow-only scratch pool is trimmed to the workers in use and released at the
   end of a batch, so a library caller keeps nothing between calls.
2. the backtrack pipeline: opt-in today, so an explicit `RNA_BT_PIPELINE=1` wins (decision 1);
   its AUTO gate (after P4) is this same computation.
3. **the build pipeline** (RNAfold): off when the next chunk will not fit beside the pending one.
4. **chunk width** (RNAfold admission; the Python window): a record joins while the chunk fits at
   one worker and no pipeline; one record alone over the budget is warned about and folded.

**What the process holds is MEASURED (its RSS), not modelled.** The first version used the fitted
base (8.4: 100 MiB) and was right for RNAfold and 30 MB short for Python (interpreter + a 56 MB
module); RSS is exact for both and already includes memory the allocator keeps after a free. The
fitted model prices only what is ABOUT to be allocated: compounds not yet built, workers' scratch.
Margins, as for the VRAM budget's 0.85: the worker gate leaves 10 % of the room (without it a
Python fold at 700 MB peaked at 718), the Python window 5 % (without it the host path ended 1 MB
over).

| run | budget | peak RSS | what was shed |
|---|---|---|---|
| 5601 × 12, one chunk | full | 2550 MB | -- |
| | 1500 MB | 1281 | workers 12 → 4 (same wall, 9.0 vs 9.4 s) |
| | 1000 MB | 963 | workers 12 → 2 |
| | 700 MB | 714 (over, as warned) | build pipeline off; chunks narrowed; workers 9 → 1; three records warned (one alone needs 709-800 MB) |
| 2800 × 40, 4 chunks | full | 947 | -- |
| | 800 / 600 MB | 613 / 500 | workers per chunk (1-6 / 1-3) |
| A_longseq (60 × ≤ 2800) | 700 MB | 678 | workers 12 → 4 |
| Python `RNA.fold(list)`, GPU / host | `ram=700` | 675 / 649 | workers 12 → 3 → 2 / the window |

Output identical to the reference in every arm; refcheck 9/9 at `full` with the same chunk counts;
binding 27/27; `py_window_check.sh` passes.

### 8.6 B4: the thread budget (2026-10-14)

Two steps, both in `budget.c`:
1. **Cap.** `vrna_budget_cap_threads()` caps each pool's auto size at the budget, announced
   once per pool: build threads, backtrack workers (before B3's RAM gate, which may shed more),
   and `-j` / `-j 0`. An explicit `RNA_BUILD_THREADS`, `RNA_BACKTRACK_THREADS` or `-j N` wins,
   with a warning (decision 1).
2. **Draw.** Pools overlap (a build pipeline's builders beside this chunk's backtrack; with
   `RNA_BT_PIPELINE=1` a trace beside the next sweep and the next build), so capping each pool
   alone allows twice the budget. Every pool therefore draws its workers from one shared count:
   an auto pool waits while the budget is spent, then takes what is free (at least one); an
   explicit pool never waits and is counted. A pool of n counts n whether it spawns them or runs
   serially on its caller; a caller that only joins is not counted. No holder waits on anything
   while it holds workers, so the wait cannot deadlock. At `full` nothing is counted and nothing
   waits. The `-j` pool is capped, not drawn: its threads live for the whole run.

Positive evidence: an exit line, `thread budget: peak N concurrent workers of a budget of B
(W draws waited, S took fewer than asked)`; and, independently, a sampler that counts the
process's RUNNING threads (state R in `/proc/<pid>/task/*/stat`, every 10 ms). maxR also counts
the main thread driving the GPU and CUDA's own threads, which no pool owns; it moves by
one between repeat runs (budget 1: maxR 1, then 2). The internal peak never exceeded the budget.

Laptop, 2800 × 40 (`~/m2b`), 12 hardware threads:

| arm | wall | maxR | peak (budget) | output |
|---|---|---|---|---|
| full | 5.6 s | 12 | -- (no line) | reference |
| `RNA_THREAD_BUDGET=4` / 2 / 1 | 4.8 / 5.2 / 6.3 s | 4 / 2 / 1 | 4 / 2 / 1 | identical |
| 2 + `RNA_BACKTRACK_THREADS=8` | 5.1 s | 8 | 8 (2), warned | identical |
| 2 + `RNA_BUILD_THREADS=8` | 4.9 s | 8 | 8 (2), warned | identical |
| 3 + `-j` / `-j6` | 5.0 / 5.0 s | 3 / 3 | 3 (3) / warned for `-j6` | identical |
| full + `-j` | 4.8 s | 12 | -- | identical |
| **contention:** `RNA_BT_PIPELINE=1`, `RNA_VRAM_BUDGET=1100` (8 chunks), budget 2 / 3 | 10.1 / 10.2 s | 4 / 4 | 2 / 3, **5 draws waited, 1 took fewer** | identical |

Without contention (one chunk, or chunks of one record) no draw waits: the builders finish
inside the sweep before the backtrack starts. The wait path was reached only with the backtrack
pipeline and multi-record chunks; it is exercised there, in both budgets, with no hang. Shedding
threads cost nothing on the laptop (the fold is GPU-bound); P4 prices it on the A100, where the
host pools were measured to pay (1.23×, 2026-08-30).

Python: `test_thread_budget_binds_backtrack_workers` (threads=2: the cap acts, peak ≤ 2,
answers equal the CPU's). Binding 28/28; refcheck 9/9 at `full`; `bt_pipeline_check.sh` 9/9
(negative controls bite). Zeroconf for B3 + B4 on a copy (`~/zcb4`, 2026-10-14): all 7 cases pass,
including case 7 (CPU-only + Python).

## 9. P3 scope: the backtrack pipeline inside `vrna_mfe_batch()` (2026-10-14)

### 9.1 Where it stands

| piece | today |
|---|---|
| P1 retire → fetch → trace | every caller, with `RNA_BT_PIPELINE=1` |
| P2 trace beside the next sweep | **RNAfold only**: `cuda_batch_cb()` lets only the LAST sub-chunk of a call hand off (`rnafold_bt_async_allow(end == n)`), and only to a driver that attached a callback |
| library lists (`RNA.fold(list)`, `fc.mfe` batches, C callers) | `cuda_batch_cb()` splits the list into VRAM-sized sub-chunks and traces each in place: **nothing overlaps** |
| the chunk-cells cap (`RNA_GPU_CHUNK_CELLS`, 7.53e8) | **RNAfold only** (`rnafold_set_chunk_cells_cap()` from `gpu_batch.c`); library sub-chunks are as wide as VRAM allows |
| budgets under the pipeline | VRAM: a retained set lowers measured free memory, so `gpu_budget_allowed()` already charges it. RAM: trace workers pass B3's worker gate. Threads: they draw from B4's count. The scratch trim skips while a trace is live (`bt_pool_trim`). |

### 9.2 Design

Inside `cuda_batch_cb()`, sub-chunk k (not the last) hands its trace to the trace thread; the loop
tears down, sizes, sweeps sub-chunk k+1; that `par_mfe()` joins k after its sweep (the existing
one-at-a-time rule) and so on. The LAST sub-chunk is traced in place, unless a driver attached
(RNAfold: P2's hand-off, unchanged). Before returning, the call joins any internal trace still
pending (e.g. when the list ends with a host-folded oversized record), so **`vrna_mfe_batch()`
stays synchronous** (decision 9.1 of N5a: no new public API).

Changes, all internal:
1. `bt_async_t` carries its own callback: an internal hand-off has none (the trace writes straight
   into the caller's `structures[]` / `energies[]`, which outlive the call); a driver hand-off keeps
   RNAfold's. Today the join calls the driver's callback for any trace.
2. `rnafold_bt_async_allow()` gains a mode: `INTERNAL` for k < last, `DRIVER` for the last (today's).
3. `cuda_batch_cb()`: the allow per sub-chunk, a drain of an internal trace before
   `rnafold_bt_pool_release_under_budget()` and before return.
4. Evidence: the existing `backtrack pipeline: ... traced beside the next sweep ... OVERLAPPED` line
   (exit, `RNA_GPU_VERBOSE` in Python), with internal hand-offs counted.

### 9.3 The tradeoff: a retained set narrows the next sub-chunk

While k is retained, k+1 is sized against VRAM minus k's triangles. Library sub-chunks are VRAM-wide
(9.1), so on any VRAM-bound list **every sub-chunk after the first gets narrower, and there are more
of them**. On the A100 a chunk costs about 2 s (`project_chunking_costs`, A100), against about 1.3 s of
backtrack hidden per chunk: the pipeline could lose. RNAfold avoids it because the cap, not VRAM,
sets its width (N5a §7: "the cap keeps chunks small, so two fit").

Two rules fix it; both are decisions below:
- **(a) Apply the chunk cap in `cuda_batch_cb()`** with RNAfold's default. Library chunks become the
  width that won on the A100 (capped 61.0 vs uncapped 74.8 s at 400 × 5601), and two of them fit.
  Output is identical by construction; chunk counts of library calls change.
- **(b) Never narrow for the pipeline.** When the next sub-chunk sized beside the retained set holds
  fewer records than it would without it, join the pending trace first, then size it again. That is
  exactly the sequential order for that step, so it can cost no more than `RNA_BT_PIPELINE=0`, and
  chunk counts equal the sequential run's. It also applies to RNAfold's own in-engine splits. A hint that RNAfold P2 splits today: the B4
  contention arm (`RNA_VRAM_BUDGET=1100`) retired 11 chunks where the build pipeline reported 8;
  to confirm with the chunk trace before (b) is written.

### 9.4 Bars

1. Byte-identical: `RNA.fold(list)` multi-chunk with `RNA_BT_PIPELINE=1` vs `=0` vs `cpu_only`
   (`RNA_GPU_VRAM_BUDGET_MB` small, as `CHUNK_PROBE`), circ, an oversized-record list, and one
   ending in a host-folded record (the drain).
2. Positive evidence: internal hand-offs counted > 0 and OVERLAPPED > 0 in a subprocess test; chunk
   counts equal to `=0` under (b).
3. Negative controls (`RNA_BT_NEGCTL=1/2`) bite in the library path too.
4. Budgets: VRAM peak within budget with a retained set; thread peak ≤ budget; RAM within.
5. Standing: binding suite, refcheck 9/9, `bt_pipeline_check.sh` (RNAfold P2 unchanged except (b)),
   zeroconf on a copy.
6. Laptop: correctness only; wall in P4.

### 9.5 Synergies and tradeoffs

| pair | relation |
|---|---|
| P3 + chunk cap in the library (a) | SYNERGY: two capped chunks fit, so the overlap needs no narrowing; the cap is also the faster width on the A100 by itself |
| P3 vs wide library chunks | TRADEOFF without (a)/(b): more chunks at ~2 s each against ~1.3 s hidden |
| P3 + Python RAM window (B3) | P3 overlaps within ONE call; a list the window splits into several calls overlaps only inside each window. Overlap across calls would make the call asynchronous: out of scope |
| P3 + thread budget (B4) | the trace's workers draw from the same count as everything else; Python's default `half` halves the trace pool |
| P3 + N5 (device backtrack) | EITHER/OR, as for RNAfold |

### 9.6 Decisions for sign-off

1. **Apply RNAfold's chunk cap in the library path** (`RNA_GPU_CHUNK_CELLS`, same default and
   override). *Recommended*: it is the measured faster width and what makes two chunks fit; output
   identical; changes library chunk counts, so it lands as its own commit with its own bar.
2. **Never narrow for the pipeline (rule b), in the engine, for both the library and RNAfold's
   in-engine splits.** *Recommended*: P3 can then cost nothing in chunk count, and it removes the
   narrowing RNAfold's P2 appears to have today (9.3, to confirm).
3. **Keep opt-in (`RNA_BT_PIPELINE=1`)** for the library too until P4 decides AUTO for both.
   *Recommended* (N5a decision 9.2).

**Signed off by Luke, 2026-10-14: "yes to all three, go ahead."** Order: decision 1 (the cap in the
library) lands first as its own commit and bar, then P3 with rule (b).

### 9.7 Results: decision 1 and P3 (laptop, 2026-10-14)

**Decision 1** (96a98bc4): the library applies the cell cap. Binding test: cap 170 000 → 3 device
chunks for 6 × 400, 0 → 1. Python hashes unchanged at 40 × 2800 and 60 × 1200; int16 chosen
before and after. On the laptop Python's half-VRAM budget binds before the cap, so the A100 (P4)
is where library chunk counts change.

**P3.** `RNA.fold(list)`, `RNA_BT_PIPELINE=0` vs `1`, answers equal to the CPU in every arm:

| list | chunks off / on | inside one call: traced beside the next / joined early | fold off → on |
|---|---|---|---|
| 8 × 400, cap 170 000 | 4 / 4 | 3 / 0 | 1.20 → 0.53 s |
| 24 × 1200, cap 2.2e6 | 8 / 8 | 7 / 0 | 1.76 → 1.75 s |
| 40 × 2800 (VRAM-bound, 2 chunks) | 2 / 2 | 1 / 0 | 6.77 → 6.44 s |
| 100 × 2800 (VRAM-bound, `RNA_GPU_CHUNK_CELLS=0`) | 4 / 4 | 3 / **2** | 15.41 → 15.31 s |

Rule (b) fired where it must (100 × 2800: two sub-chunks would have been narrowed) and the chunk
count stayed at 4. The binding suite's `cuda_pipelineTest` adds: a cap-bound list (overlap, no early
join, same chunks), a VRAM-bound list ending in host folds (the end-of-call drain), and
`NEGCTL=2`, which bites on internal hand-offs. Binding 32/32; refcheck 9/9; `bt_pipeline_check.sh`
9/9 (`NEGCTL=2` now ends in rc 134/139 on circ/noLP where it was rc 1: still a bite, the freed set
is simply reused differently).

**Found: RNAfold's own admission narrows under P2.** 2800 × 40 at `RNA_VRAM_BUDGET=1100`: 6 chunks
with the pipeline off, **9 with it on**, same output. The driver (`gpu_batch.c`) sizes each chunk
with `compute_gpu_usable_bytes()` while the previous chunk's retained set is still on the device;
rule (b) lives in the engine and cannot see it (it fired twice there, on in-engine splits). On the
A100 the cap, not VRAM, usually sets RNAfold's width, so this bites only VRAM-bound runs; it would
still cost about 2 s a chunk there. Fix: the same rule at the driver's admission.


**Fixed: rule (b) at the driver** (Luke, 2026-10-14). `gpu_batch.c` sizes a new chunk as if a pending retired
set were gone (+ 0.85 × its bytes, the margin both VRAM limits leave); if the chunk then does not fit beside
it, the engine's rule (b) joins the trace before the sweep. 2800 × 40, off vs on, one sha throughout:

| `RNA_VRAM_BUDGET` | sweeps off / on (before the fix: on) | traced beside the next sweep | joined early |
|---|---|---|---|
| 1100 | 6 / 6 (9) | 6 | 5 |
| 1600 | 3 / 3 | 3 | 1 |
| full | 1 / 1 | 1 | 0 |

Where VRAM sets the width the overlap is given up rather than a chunk added (decision 2's tradeoff); where
the cap sets it (the A100's usual case) nothing joins early. Binding 32/32, refcheck 9/9,
`bt_pipeline_check.sh` 9/9.

## 10. P4 scope: the A100 session (2026-10-14)

### 10.1 What exists

`tools/make_nb_n5a.py` writes the N5a P4 notebook (generated 10-10 for P2, **never run**; the `.ipynb` is
gitignored, the generator is the source). Sections: **G** exactness (12 cases × CPU / off / on, NEGCTL 1 and 2,
the `--unordered` refusal), **T** off vs on at 400 × 5601, 3000 × 1200, 1200 × 1200 (ABBA × 3) and the 2400 soak,
**H** the trace-thread split under full heterogeneous load (4 / 8 / 12 / auto / decision 9.4's formula), **V**
report. Pre-registered: Q0 (G passes), Q1 (AUTO becomes the default if on beats off beyond spread at 400 × 5601
and 3000 × 1200 and loses nowhere), Q2 (a thread arm replaces auto only if it beats it beyond spread). About
75 min. It builds `--without-python`, so it cannot see the library.

Since then: budgets B1-B4, decision 1 (the cap in the library), P3, the driver's rule (b).

### 10.2 Plan: extend the N5a notebook into one session

1. **Head checks.** LFB must carry P3 and the budgets (`RNAFOLD_BT_ASYNC_INTERNAL`, `vrna_budget_threads_acquire`,
   `rnafold_bt_pending_device_bytes` in `gpu_batch.c`), or it stops before building.
2. **G additions.** Per multi-chunk case, RNAfold's chunk count with the pipeline on must equal off (the driver's
   rule (b) at A100 scale); record the early-join count.
3. **T and H unchanged**, plus one H arm: **`RNA_THREAD_BUDGET=NPROC`** under the same load. A number (not
   `full`) turns on B4's shared count, so the builders and the trace pool draw from one count instead of each
   taking every core (the CPU slice is not drawn, 10.4). It is the mechanism H was asked to find by hand.
4. **B, budgets at scale (new).** 400 × 5601, CLI, `full` vs `half` (RAM, VRAM, threads together) and each
   resource alone at `half`, ABBA × 3. Per arm: wall, chunk count, peak RSS (`/usr/bin/time -v`), the VRAM peak
   line (B2's measured context on an A100, owed since M1), the thread peak line, what was shed.
5. **L, the library (new).** A second build `--with-python` (about 8 min). `RNA.fold(list)` on 400 × 5601 and
   3000 × 1200, Python's default budget (`half`) and `full`:
   - `RNA_GPU_CHUNK_CELLS` default vs `0`: decision 1's effect where the cap, not VRAM, binds;
   - `RNA_BT_PIPELINE` 0 vs 1: P3, with the internal hand-off and early-join counts;
   - answers hashed against the CLI's for the same fixture (one sha per fixture across CLI and library).
6. **V** reports all of it with the rules below.

About 75 + 10 (B) + 25 (L, incl. the build) ≈ **110 min** on an A100-80GB. Sections save as they go and resume.

### 10.3 Pre-registered rules (proposed)

- **Q0, Q1, Q2 as signed for N5a**, with Q1 judged on the CLI arms (T) **and** the library arms (L): AUTO for
  `RNA_BT_PIPELINE` becomes the default for both only if neither loses beyond spread anywhere.
- **Q3, budgets hold at scale (gating):** at `half`, peak RSS ≤ the RAM budget, the VRAM peak line says "within
  it", the thread peak ≤ the thread budget; output identical to `full`. A failure is a bug to fix, not a tradeoff.
- **Q4, decision 1 (gating for keeping it):** library wall with the cap ≤ without it, beyond spread or within it.
  If the cap loses in the library, it is reverted there (RNAfold keeps its own, measured 10-06).
- **Reported, not gating:** what `half` costs in wall on the A100 (CLI and library), what each resource alone
  costs, early joins, and whether the thread-budget H arm beats auto.

### 10.4 Notes and limits

- The CPU slice (`RNA_CPU_SLICE` + `-j`) is capped by the thread budget only through `-j`'s auto size, not drawn
  from the shared count (its threads live for the run, B4). The H arm measures the count as built.
- Colab's A100 VMs carry far more host RAM than the laptop (the notebook records MemTotal; I recall about 83 GB on
  the high-RAM tier, unverified), so Q3's RAM half may not bind at 400 × 5601. The VRAM half (40 GB) binds. Report which bound.
- The AUTO gate for the backtrack pipeline, if Q1 passes, is B3's computation ("two chunks fit the budget"),
  implemented after this run as its own stage with its own bar.

### 10.5 Decisions for sign-off

1. **One session: extend the N5a notebook** (shared build, fixtures and harness) rather than a second notebook.
   *Recommended:* one A100 session, and T, H, B and L see the same binary.
2. **Q3 and Q4 as written** (budgets gating on correctness at scale; the library cap kept only if it does not
   lose). *Recommended.*
3. **Q1 judged on the library too** (AUTO for both or for neither). *Recommended:* one schedule, as N5a §3.7 says.
4. **The thread-budget arm in H.** *Recommended:* it costs one arm and tests the mechanism B4 built.

**Signed off by Luke, 2026-10-14: "yes to all four, go ahead."**
