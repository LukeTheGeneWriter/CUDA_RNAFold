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
