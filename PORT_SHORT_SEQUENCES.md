# Short sequences: Bill's report, what it measured, and the plan

*2026-10-18. Status: SIGNED OFF (Luke: S1 → S2 → S3 with the recommendations; D1 never offer below
the minimum length; D2 yes). S1 + S2 built, section 8. Laptop (RTX 3050 4 GB, 12 logical cores, WSL2),
LFB cfc2b408 (G0), build tree ~/pyqc. Scripts in the session scratchpad: tty_repro.py, wake.sh,
sp_short.sh, sp_speed.sh.*

## 1. The report (W. B. Langdon, 2026-10-18)

1. **Interactive input.** `RNAfold --noPS`, then type `ggggggggggaaaacccccccccc` at the prompt.
   Stock 2.7.2 answers immediately. LFB prints `fml_j AUTO: ...` and the next prompt, and the answer
   only appears after Ctrl-D.
2. **Startup cost.** For a short sequence, waking the GPU (20–40 ms on Bill's machine) takes longer
   than the CPU needs for the whole fold.

## 2. Both reproduced, and why they happen

**Interactive input (a defect).** Under a pty (`tty_repro.py`), with `RNA_GPU=0` the answer appears
when the line is typed; by default, only after Ctrl-D. Three things cause it:

- `process_input()` hands every record to `gpu_batch_offer()`, which holds it in the chunk until the
  chunk fills or `gpu_batch_end()` runs at EOF.
- Nothing checks for a terminal.
- The 24-nt record is 300 cells, far under the floor (5.7e4), so at EOF it goes to the CPU anyway.
  It was never going to use the device; it only waited.

**Startup cost.** One short record per invocation, medians of 15 (`wake.sh`):

| arm | wall per invocation |
|---|---|
| `RNA_GPU=0` | ~100 ms |
| default | ~290–360 ms |

That is about +200 ms here (WSL; Bill sees 20–40 ms on native Linux), for a record that then folds
on the CPU. Where the device gets touched before anything earns it:

- `gpu_batch_begin()` calls `vrna_cuda_devices()` unconditionally. That creates the CUDA context,
  and it does so even under `RNA_GPU=0`, because the probe runs before the switch is read.
- The first `gpu_batch_offer()` runs `compute_gpu_usable_bytes()`, which prints `fml_j AUTO`.
- Teardown at exit.

**The probe's own 0.07–0.09 s understates the device's fixed cost about fourfold.** The 16-record
arms below put it near **0.3 s**. That probe time is the F the admission floor uses.

## 3. Sparse md at short lengths: exact, and no limiter is warranted

**Exactness** (`sp_short.sh`, device forced on with `RNA_GPU_WORK_FLOOR=0`; every record's route
from `RNA_RECORD_TRACE`):

- Inputs: every length 1–64, then 70–300 in steps of 10, 48 records each; low-complexity repeats
  (poly-A, GC, AU, GGGA, ...); and a 400-record mixed file.
- Options: default, `-d0`, `--noLP`.
- **273/273 device runs were byte-identical to the CPU.** The cell-for-cell selftest compared
  **34.4 M cells, 0 mismatching.**
- The 15 non-passing lines are expected:
  - Lengths 1–3 cannot pair, so there is no sweep.
  - `-d3` is declined to the CPU by design.
  - All 15 are byte-identical anyway.
- Low-complexity repeats overflow the candidate lists: about 9 % of md cells took the dense
  fallback. Exact, and the fallback count is printed.

**Speed** (`sp_speed.sh`, -j12, wall in seconds, mean of two ABBA reps). sp = device forced, sparse
md; dn = device forced, dense md; auto = today's admission rule.

| L \ N | arm | 16 | 128 | 1024 | 4096 |
|---|---|---|---|---|---|
| 30 | cpu | 0.09 | 0.10 | 0.14 | 0.56 |
|    | auto | 0.26 | 0.28 | 0.32 | **2.99 (gpu)** |
|    | sp / dn | 0.39 / 0.38 | 0.49 / 0.45 | 1.00 / 1.01 | 2.91 / 2.71 |
| 50 | cpu | 0.11 | 0.11 | 0.42 | 1.51 |
|    | auto | 0.32 | 0.28 | **1.20 (gpu)** | **3.59 (gpu)** |
|    | sp / dn | 0.44 / 0.42 | 0.47 / 0.59 | 1.13 / 1.19 | 3.52 / 3.24 |
| 75 | cpu | 0.19 | 0.16 | 1.35 | 3.22 |
|    | auto | 0.28 | 0.34 | 1.32 | **4.16 (gpu)** |
|    | sp / dn | 0.44 / 0.42 | 0.54 / 0.53 | 1.32 / 1.28 | 4.32 / 3.84 |
| 100 | cpu | 0.11 | 0.39 | 1.99 | 5.89 |
|     | auto | 0.29 | **0.56 (gpu)** | 1.53 | 5.08 |
|     | sp / dn | 0.44 / 0.60 | 0.53 / 0.53 | 1.50 / 1.49 | 4.77 / 4.89 |
| 150 | cpu | 0.14 | 0.65–1.23 | 4.08 | 14.85 |
|     | sp / dn | 0.47 / 0.43 | 0.66 / 0.59 | 2.00 / 1.99 | 7.11 / 7.06 |
| 200 | cpu | 0.19 | 1.52 | 7.78 | — |
|     | sp / dn | 0.48 / 0.46 | 0.68 / 0.67 | 2.79 / 2.71 | — |
| 300 | cpu | 0.49 | 2.72 | 17.03 | — |
|     | sp / dn | 0.50 / 0.53 | 0.97 / 0.98 | 4.82 / 4.97 | — |
| 400 | cpu | 1.26 | 4.38 | 30.24 | — |
|     | sp / dn | 0.58 / 0.59 | 1.34 / 1.34 | **7.87 / 8.31** | — |
| 600 | cpu | 2.02 | 10.11 | 69.97 | — |
|     | sp / dn | 0.71 / 0.75 | 2.41 / 2.61 | **16.87 / 18.85** | — |

**Sparse md is neutral below 300 nt** (sp and dn within each other's noise) and starts paying at
400 nt (−5 % at 400×1024, −10 % at 600×1024), as on the A100. By the whole-run-wall rule (Q7) there
is nothing for a length limiter to buy, and it would add a second code path to keep exact. **The
recommendation is no sparse limiter.**

**The admission rule is what's wrong at short lengths, in both directions:**

- **It admits short records the device loses on.** At L ≤ 75 the device never wins at any N:
  - 30 nt × 4096: 2.99 s vs 0.56 s on the CPU, **5.4× slower**.
  - 50 nt × 4096: 2.4× slower.
  - 75 nt × 4096: 1.3× slower.

  The floor admits these because it is a total-cells threshold with no per-record term: enough
  short records always add up to the floor.
- **It charges the wake cost to records it then sends to the CPU.** At small N, auto routes to the
  CPU but still pays about +170 ms, the eager probe.

## 4. The cost model the data supports (this is the L² minimum batch)

Per-record marginal costs at -j12, from the N-to-N differences above (cells = L(L+1)/2):

- host: about **3.5e-7 s × cells**. Constant per cell to within about ±25 % from 30 to 600 nt,
  because the interior-loop window is capped. So L² is the right unit for the host.
- device path: about **0.55 ms + 1.0e-7 s × cells per record**, plus a fixed **~0.3 s** per run.
  - The 0.55 ms is host-side, per-record work on the device path: build, matrix fetch and trace.
    The matrix fetch (`fetch_mx`) is the largest part: 0.40 s of 1.5 s at 100×1024.
  - The total-cells floor has no term for it, which is why short records are mis-admitted.

A uniform chunk of N records at length L earns the device when

    N · t_host(L) / jobs  >  F_dev + N · t_dev(L)

That gives two limits, and together they are Luke's minimum GPU batch:

- **A minimum length.** Below it, t_host/jobs ≤ t_dev, the inequality fails for every N, and no
  batch of that length should go to the device. Here that is about 65–95 nt at -j12. With fewer
  jobs it moves shorter.
- **A minimum batch above that length:**

      n_min(L) = ceil( F_dev / (t_host(L)/jobs − t_dev(L)) )

  It falls steeply with L², which the data shows:

  | L | predicted n_min | measured: CPU wins at | device wins at |
  |---|---|---|---|
  | 100 nt | ~420 | 128 | 1024 |
  | 200 nt | ~70 | 16 | 128 |
  | 300 nt | ~30 | 16 is a tie | — |

  Small N also needs the host's wave quantisation, `ceil(N/jobs)` (16 records on 12 jobs is two
  waves). Without it the model under-calls the device at 400 nt × 16, where the device wins 2.2×.

## 5. Plan (in order; each step committed only with its bars green)

**S1. Interactive input folds before the next prompt (the defect).**
When stdin is a terminal, a record is never held for a later chunk:

- If the record alone earns the device (S3), it is folded as a one-record chunk at once: flush,
  pipeline drain, backtrack drain.
- Otherwise it goes to the per-record path, as in stock ViennaRNA.

Bars:
- The pty harness (`tty_repro.py`, moved into tests/) asserts the answer appears **before EOF**,
  for Bill's 24 nt and for a record long enough to earn the device (with positive sweep evidence
  for that one).
- Output byte-identical to `RNA_GPU=0`.
- The standing bars.

**S2. No CUDA until the input could earn it (lazy device).**

- `gpu_batch_begin()` stops probing.
- Records are held host-side until their estimated host time could exceed a lower bound of the
  device's fixed cost. Only then does the probe run, measuring F.
- An input that ends first is folded on the CPU path without ever creating a context: no wake
  cost, no device chatter.
- `RNA_GPU=0` never touches CUDA.
- Under S1 (a terminal), a record that does not earn the device on its own is not held at all.

Bars:
- One-shot wall for Bill's record within 10 ms of `RNA_GPU=0` (medians).
- No `GPU acceleration ON` line and no context: positive evidence via the trace.
- Zeroconf case 2 still sees the device engage on its 12 × 600–1200 fixture.

**S3. Admission by the per-record model (replaces the total-cells floor).**

- The section 4 inequality, using:
  - F_dev = the measured probe plus the init it predicts. Calibrated here as ~0.3 s; the prior
    applies until measured.
  - t_host(L) = cells / R_host per job, with wave quantisation.
  - t_dev(L) = h + cells / R_dev.
- R_host, h and R_dev are compiled-in constants from this matrix. They stay overridable, keeping
  `RNA_GPU_WORK_FLOOR=0` as "always the device".
- The run reports how many records it declined as too small, and where the minimum length fell.
- These constants are machine-dependent. The tuner in PORT_ZEROCONF_SCOPE.md §3.1 is where they
  get measured instead.

Bar: re-run `sp_speed.sh`. **In every cell, auto must be within 10 % of min(cpu, sp)**; today it
fails by up to 5.4×. The existing crossover fixtures (the 3×300 / 8×200 / 1×1600 table in
`gpu_batch.c`) must route the same way or better.

**S4. Sparse md limiter: not built.** Section 3 is the evidence. Recorded in PORT_SPARSE_MD.md.

**S5. Default-path chatter (decision for Luke, D2 below).**

## 6. Decisions for Luke

- **D1.** Minimum-length behaviour: below the measured minimum length, never offer to the device
  (recommended), or still allow very large batches?
- **D2.** Chatter on the default path. Bill saw `fml_j AUTO`; the default also prints
  `build pipeline AUTO` and `build threads`. Recommended:
  - Those three move to `--verbose`.
  - `GPU acceleration ON` stays, printed once at the probe (now lazy), as the positive evidence
    harnesses need.
  - Nothing is printed when the device was never needed.

  Only zeroconf and the local notebook generators grep these lines.
- **D3.** Should S3's constants be calibrated on the A100 too (one notebook cell re-running
  `sp_speed.sh`) before it becomes the default? The minimum length depends on the host/device
  ratio.

## 7. Zeroconf: where its 15 minutes go (measured 2026-10-18, rc 0, JOBS=4)

| phase | s |
|---|---|
| copy + 6 configures | ~90 |
| accelerated build (m1) | 152 |
| CPU-only build (m5) | 150 |
| CPU-only + Python build (m8) | 340 |
| restore rebuild (m7) | 169 |
| **total** | **908** |

- Each configure rewrites `config.h`, so every case is a full rebuild.
- About half of each libRNA build is **LTO link of the ~30 programs** (`lto1 ... ltrans` for
  RNAheat, RNAplex, RNAup, ... fills the sampler).
- m8 adds the SWIG wrapper:
  - One `RNA_wrap.cpp` compile takes 58 s at 1.2 GB.
  - libtool compiles it twice, because static libraries are on (`build_old_libs=yes`).
  - Then an LTO link of the Python module, about 50 s.
- The three earlier section-7 deaths happened inside that wrapper compile and coincided with the
  WSL VM going down. It was not hung.

Proposed harness-only speedups (no product change):

- `--disable-lto --disable-static` in COMMON / PYCOMMON. These are upstream's own switches, and the
  cases test CUDA detection, not LTO.
- `ZC_NO_RESTORE=1` skips the restore rebuild when the tree is a throwaway copy, as in the bars.

Expected: 908 s to roughly 350–400 s.

## 8. S1 + S2, as built (2026-10-18)

**What changed** (`src/bin/gpu_batch.c`, `RNAfold.c`, `RNAfold_gpu.h`; `engine.c/.h` and
`mfe_cuda.c` for D2):

- **S1.** `gpu_batch_begin(opt, interactive)` takes `isatty(stdin)`. At a terminal a record is
  never held: if it earns the device on its own it is folded at once, alone, with both pipelines
  drained (`flush_chunk_now()`); otherwise it goes to the per-record path.
- **S2.** `gpu_batch_begin()` no longer probes. It settles one of four modes:
  - OFF: `RNA_GPU=0`, still announced, and now without touching CUDA.
  - PENDING.
  - DECLINED: options the sweep cannot run.
  - ON, reached in `gpu_activate()`.

  While PENDING, records are held host side until `work_would_earn_gpu()` says the held work
  could pay for reaching the device (prior `VRNA_GPU_WAKE_COST_PRIOR` = 0.3 s). Then the device is
  probed and the held records join chunks in input order. An input that ends first is
  dispatched to the per-record path with no context ever created. A DECLINED run reaches the
  device only once its input would have mattered, so the "GPU acceleration OFF: <option>" line
  appears exactly when it is worth reading.
- **The budget report** no longer asks the card's size unless a VRAM budget is set. That
  argument was a second probe on every run.
- **D2.** `fml_j AUTO`, `build pipeline AUTO` and the default `build threads` line are said only
  under `--verbose` (`vrna_cuda_set_verbose()`). An explicit `RNA_BUILD_THREADS` / `RNA_FML_INT16`
  still announces itself.

**Bar: `tests/short_input_check.sh` (+ `tests/tty_answer.py`, a pty driver). All green; the
negative control bites.**

| case | result |
|---|---|
| Bill's 24 nt at the prompt | answered before Ctrl-D, 0.001 s, −25.20 as stock |
| 1500 nt at the prompt | answered before Ctrl-D, 2.0 s, ON the device (sweep shape) |
| one 24 nt record, piped | **libcuda never loaded**; stderr empty; = `RNA_GPU=0` |
| `RNA_GPU=0` | libcuda never loaded; still announced |
| 3 × 100 nt; `-d3` small | libcuda never loaded; no decline line |
| 12 × 600–1200 nt | device reached, all 12 routed to it, = CPU |
| 30 × 24 nt held, then 12 long | all 42 written in input order, = CPU, swept |
| `-d3` 12 × 1200 nt | "GPU acceleration OFF: dangle model 1 or 3 ..." |
| `RNA_GPU_WORK_FLOOR=0` | still forces the device |
| one-shot wall, median of 9 | **8 ms default, 8 ms `RNA_GPU=0`** (was ~290–360 ms vs ~100 ms) |
| negctl: the G0 binary | loads libcuda 8 times on one 24 nt record: the check can fail |

"Never touched the device" is shown, not assumed. libcudart is linked at load time, but it
`dlopen()`s the driver (`libcuda.so.1`) only at the first CUDA call. So `LD_DEBUG=files` naming
libcuda is evidence that a context was attempted. The check needs no strace, and it ports to
Colab.

### 8.1 Found on the way: reaching the device is BIMODAL, and it contaminated section 3

E-A (variable batch size, 1–4096 records at 60–300 nt, -j1/-j4/-j12) showed a step of about
+0.7 s in the **CPU** arm once a run lasted more than about 0.4 s, at every length and job count.

- **It was a wait, not work.** At 200 nt × 16, -j1: wall 1.17 s, user 0.38 s.
- **The cause is the probe.** On this host `cudaGetDeviceCount()` costs either about **0.07 s or
  about 0.75 s**, depending on whether the WSL driver is warm. A standalone program measured
  0.71 s on every call; RNAfold logged 0.80 s and 0.07 s on back-to-back runs. This is the same
  warm/cold pair the floor's comment in `gpu_batch.c` records (0.09 / 0.75 s).
- **The G0 binary probes under `RNA_GPU=0` too.** So every "cpu" arm in section 3 and in E-A
  carried a random +0.7 s, and every crossover there leans toward the device. The S2 binary's
  `RNA_GPU=0` never loads libcuda: its wall equals its user time (0.37/0.37 s at 200 × 16).

**E-A is being re-run on the S2 binary as E-A2.** The device rows log the measured F, so the
fit can separate "reaching the device" (bimodal, measured per run) from the per-chunk and
per-record costs. S3's constants come from E-A2, not section 3.

### 8.2 E-A2: variable batch size, clean (S2 binary), and the model S3 builds

1–4096 uniform records at 60–300 nt, -j12, plus -j1 and -j4 at 100 and 200 nt. ABBA × 2.
Every device row logs its measured reach cost F, and the fit subtracts it. Every arm's output
was byte-identical to the CPU's.

| term | value |
|---|---|
| reaching the device, F | median **0.08 s**; cold (0.75–1.43 s) in **77 of 264** runs |
| device fixed cost beyond F | **0.30–0.36 s per run**, flat in L and -j (the first real CUDA work: context, modules) |
| device per record | **0.68 ms + 8.1e-8 s/cell** |
| host per record, one job | ~**1e-6 s/cell**, rising gently with L (≈ cells^1.15) |
| host speed-up with -j | **-j4 2.7×, -j12 3.7–3.8×** (12 logical cores, power-capped laptop) |

| L | fitted n* (-j12) | measured crossover (-j12) | n* (-j1) | crossover (-j1) |
|---|---|---|---|---|
| 60, 80 nt | ∞ | none up to 4096 | | |
| 100 nt | 1778 | 4096 | 89 | 128 |
| 120 nt | 532 | 1024 | | |
| 150 nt | 214 | 256 | | |
| 200 nt | 92 | 128 | 17 | 32 |
| 300 nt | 33 | 64 | | |

**The old floor was wrong in three ways:**

- It had no per-record device term. Below about 85 nt the device loses at every N.
- It assumed the host scales linearly with -j. Here -j12 is 3.75× one job, not 12×.
- It charged the probe alone as F, when reaching the device costs F plus another ~0.34 s.

**The model S3 implements.** For a chunk of records i:

    host wall  = Σ t_h(c_i) / S(j)              t_h(c) = K·c^1.15 per record, one job
    device     = [F + C0 if not yet reached] + Σ (h + k·c_i)
    S(j)       = j / (1 + β(j − 1))             β = 0.18: S(4) = 2.6, S(12) = 4.0

A record goes to the device's chunk only if its cells reach c*. That is the minimum length,
where t_h(c)/S(j) = h + k·c; below it the device loses at any N (D1). A chunk is folded on the
device only if the host wall exceeds the device's.

- **Before the probe,** F is the prior (0.08 s, the measured median) plus C0 (0.34 s).
- **After the probe,** F is the measured value.
- **After the first chunk,** only the per-chunk C (~0.04 s) remains.

On these constants the model reproduces the crossovers above to within one doubling of N.

**Machine dependence.** K, h, k, C0 and β are this laptop's numbers. β especially:
- A server CPU scales far closer to linearly, which makes the host stronger and moves every
  crossover toward the CPU.
- All five stay overridable.
- D3, calibration on the A100 host, is where the next numbers come from.

## 9. S3, as built (2026-10-18)

**What changed** (`src/bin/gpu_batch.c` only): the total-cells floor and S2's 0.3 s wake prior
are replaced by the section 8.2 model, evaluated per record and per chunk.

- `gpu_model_begin()` solves the **minimum length** once per input stream, for this -j:
  **45 nt at -j1, 73 nt at -j4, 95 nt at -j12** on the compiled-in constants.
  A piped record under it is never offered to the device: not held, not counted toward waking it
  (D1). `--verbose` counts them at the end.
- A held input (PENDING) wakes the device once `host wall > prior + C0 + Σ device`; a chunk goes
  to the device once `host wall > C0 (first chunk) or 0.04 s (later) + Σ device`. At a terminal
  the record is asked alone, at one job, so a single record earns the device from about 700 nt.
- The ON line now says where the minimum length fell
  ("with 12 jobs, records under 95 nt fold on the host").
- `RNA_GPU_COST_MODEL` overrides any of K, h, k, C0, beta, and announces itself. A forced
  `RNA_GPU_WORK_FLOOR` is the pre-S3 test exactly, with no minimum length, so every harness that
  forces the path is unchanged.

**Routing, from the binary's own evidence** (RNA_RECORD_TRACE routes, ON line; every output
byte-identical to `RNA_GPU=0`):

| shape | -j1 | -j12 |
|---|---|---|
| 3 × 300, 8 × 200, 30 × 100 | host | host |
| 8 × 400 | device | host |
| 256 × 80, 300 × 120, 4096 × 60 | device | host (80, 60 nt: under the minimum length) |
| 1 × 1600, 20 × 600, 64 × 300 | device | device |

Bill's 24 nt, piped: stderr empty, libcuda never loaded. Mixed 30 × 24 + 64 × 300 at -j12:
the 30 short records fold on the host, the 64 on the device.

**Against E-A2, offline** (each cell charged the measured median wall of the arm the model
picks): **128.3 s over the whole matrix against the oracle's 123.1 s (+4 %); the old floor 165.9 s
(+35 %)**. Worst cell 2.0× (model) vs 11.9× (old: 512 × 60 nt at -j12). Every remaining miss is
the device chosen one doubling of N early.

### 9.1 Found on the way: the cold probe is GPU idle time

The misses all sit just above a step of about +0.6 s in the device walls, at every length. No
code path switches there (the `--verbose` lines at 300 nt × 8 and × 16 are identical). The step
is the probe:

| E-A2 device runs | cold (F > 0.3 s) |
|---|---|
| straight after another device run | **0 of 132** |
| after a CPU arm | **58 % of 132** |

The cold runs start once the CPU arms before them last about 0.2 s or more. That is the
laptop dGPU dropping to a deeper power state when idle (presumably Optimus runtime power
management; WSL exposes no sysfs power state to confirm it). ABBA put one warm device run in
every pair, so E-A2's median F of 0.08 s is the warm case. **A user's one-shot run usually starts
with the GPU idle, so on this laptop F is about 0.7 s.**
