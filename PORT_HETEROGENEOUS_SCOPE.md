# Heterogeneous execution and `-j` — options

*Scoped 2026-09-09 against `6d3bdc19`. Numbers from `stress272_t4.json`
(400 × 5601, T4, commit `851d1b04`) with the `output` fast path applied
arithmetically, since that run predates it. **No new measurement here** — this is
a design note over existing data, and every option ends with what would settle it.*

---

## 0. Where the machine actually idles

**MEASURED 2026-09-09** on the run with both fixes in (`6d3bdc19`, T4, 523.7 s,
`stress272_t4_fastpath.json`) — this section was arithmetic over a pre-fix run
until then, and the arithmetic held to within 2 %:

| | s | share of the 523.7 s wall |
|---|---|---|
| GPU kernels | 325.8 | 62.2 % |
| transfers | 18.9 | 3.6 % |
| **host-only (GPU idle)** | **177.6** | **33.9 %** |
| — of which `build` | 128.1 | **72 % of the idle** |
| — of which `backtrack` | 47.1 | 27 % |

Stable across all six arms (33.6–36.0 % idle).

**`backtrack` is already threaded** — `RNA_BACKTRACK_THREADS` defaults to `auto`
= `nproc − cpu_queue_threads` (`mfe_cuda.c:541`) — and is still 9 % of wall. That
is not a missing optimisation, it is a **core-starved host**, and it is direct
evidence for §C: Colab's T4 instances are the worst case for any CPU-side scheme.

**So the prize is ~32 % of wall, and three quarters of it is one serial loop.**

### Why this is not the pipeline idea that was already retired

`project_step2_validated_458s` retired "the producer/consumer chunk pipeline and
every *hide host setup behind GPU work* variant" **by measurement** — GPU idle had
collapsed from 18.6 % to **2.4 %**, leaving ~2 % on the table.

That measurement was taken on the 2.3.0 branch **with `RNA_BUILD_THREADS`
enabled** (step 2b). **The 2.7.2 port never carried it** — `RNA_BUILD_THREADS`
appears nowhere in this tree, while `RNA_BACKTRACK_THREADS` did survive
(`mfe_cuda.c:535`). The idle collapsed *because* build was threaded; the port
lost the threading and the idle came back.

**The retirement is therefore not binding here** — but note what it implies: the
2.3.0 tree got to 2.4 % idle by threading build, *not* by pipelining. That is
evidence about which option to reach for.

---

## 1. What `-j` does today

```
main: serial reader
  ├ eligible record  → append to gpu_chunk[]; flush_gpu_chunk() when full
  │                     (flush is SYNCHRONOUS — the reader blocks)
  └ ineligible       → RUN_IN_PARALLEL(process_record, record)   ← -j pool
```

`RUN_IN_PARALLEL` is `thpool_add_work()` when `jobs > 1`, else an inline call.

So there **is** already record-level heterogeneity — records below
`MIN_GPU_BATCH`, or with unsupported options, fold on the CPU pool while the main
loop continues — but there is **no concurrent CPU+GPU folding of a batch**,
because `flush_gpu_chunk()` blocks the reader for its whole duration. The 2.3.0
fork's `RNAfold_cpu_queue.c` was **retired rather than ported**
(`RNAfold.c:1183`).

Three constraints any option must respect:

1. **Output order.** Records hold `vrna_ostream_t` slots; anything that moves a
   record between paths must keep its slot, or `--unordered` becomes the only
   correct mode.
2. **Defect B.** `vrna_fold_compound()` → `vrna_params()` → the unsynchronised
   `SPEEDUP_PARAMS` cache. Measured: **0 / 160 000 wrong when callers share model
   details, 25 140 / 160 000 (15.7 %) when they differ.** RNAfold gives every
   record the same `md`, so it is UB that this caller cannot observe — but it is
   UB.
3. **Host RAM.** A chunk already holds ~47 MB of `ptype` + `hc->mx` per record;
   RSS was 4.9–11.5 GB at 400 × 5601. Anything holding two chunks doubles it.

---

## 2. The options

### A — Thread the build loop (restore `RNA_BUILD_THREADS`)

Attacks 74 % of the idle directly. `flush_gpu_chunk()`'s build loop is
embarrassingly parallel across records.

- **Prize:** `build` 121.9 → ~121.9/c. At 8 cores, ~107 s of a 521 s wall ≈ **20 %**.
- **Cost:** it is Defect B. Every builder thread hits the racy cache. Measured
  non-wrong for identical `md`, which RNAfold guarantees — but the library entry
  point `vrna_mfe_batch()` does **not**, and this code sits under it.
- **`-j` interaction:** competes for the same cores as `process_record`. Under
  the fast path `output` is ~0, so on an accelerated run there is little else for
  the pool to do — they mostly do not collide. On a mixed run (some records CPU-folded)
  they do.
- **Settles it:** whether we ship UB. Not a measurement question.

### B — Pipeline the build behind the previous chunk's GPU work

One builder thread constructs chunk *N+1* while the GPU folds chunk *N*.

- **Prize:** hides up to `min(build_per_chunk, gpu_per_chunk)`. At 5 chunks that is
  ~24 s hidden on each of 4 chunks ≈ **18 %** — comparable to A.
- **Cost:** two chunks of compounds alive at once. At the natural budget that is
  ~+4.9 GB RSS, which is the single largest objection.
- **Decisive advantage over A: it is RACE-FREE.** One builder thread means
  `vrna_params()` is never called concurrently. Defect B does not apply.
- **`-j` interaction:** adds one thread beside the pool rather than contending
  inside it; the reader stops blocking, so the pool keeps getting work.
- **Settles it:** an RSS measurement at the natural and half budgets.

### C — Heterogeneous folding: give the `-j` pool a slice of each chunk

The literal reading of "run the accelerator with the CPU". **Measured
2026-09-09** with `tools/cpu_gpu_ratio.sh`, both sides on the same machine at the
same time — RTX 3050 laptop, 12 cores:

| length | CPU s/rec (1 core) | GPU s/rec | ratio per core | CPU could add, 12 cores |
|---|---|---|---|---|
| 300 | 0.107 | 0.0033 | 32.9× | 36.5 % |
| 600 | 0.394 | 0.0092 | 42.9× | 27.9 % |
| 1200 | 1.420 | 0.0356 | 39.8× | 30.1 % |
| 2400 | 7.046 | 0.1654 | 42.6× | 28.2 % |

**Two corrections to what this document said before.**

1. **The ratio is ~33–43×, not 76×, and it is roughly FLAT with length.** The
   earlier 76× came from comparing upstream's CPU seconds in one Colab session
   against GPU seconds from a *differently throttled* one — a cross-machine
   comparison presented as a ratio. And the guess that "C gets much better at
   short lengths" is **not supported**: 300 nt is only mildly better, and even
   that is understated for the GPU, because 64 records at 300 nt sits right at
   `MIN_GPU_BATCH`'s break-even and the device is not well used there.
2. **C does NOT compete with A for cores.** They occupy *different phases*:
   threaded build burns cores during the build, CPU folding burns them during the
   GPU fill, and those are sequential. Only B overlaps the GPU window, and it
   needs exactly one core. **C is largely additive to A, and costs B one core.**

**The clock caveat, and it cuts against C.** This card was at 1057 / 2100 MHz —
half clocks. A full-clock datacenter GPU roughly doubles the ratio to ~65–85×
and halves the CPU's share.

**So the honest figure is `c / ratio`, and the argument for C is core count:**

| host | ratio ~70× (full-clock GPU) |
|---|---|
| Colab T4, 2 vCPU | ~3 % |
| Colab L4, 8–12 vCPU | ~11–17 % |
| 32-core workstation | ~46 % |
| 64-core server | ~90 % |

**Verdict: worth doing, and the earlier "declined" was wrong.** It is small on
the machines we happen to benchmark on and large on the machines this tool would
actually be deployed on. It is also the only option here whose value *grows* with
the host rather than being capped by a fixed idle window.

**Design it as a shared work queue, not a fixed split.** A predicted split needs
the ratio, the core count and the GPU's clock state to be known in advance; a
queue where the GPU takes batches and CPU workers take singles is self-balancing
across all three, and degrades correctly when the GPU is absent, busy or slow.
Records keep their `vrna_ostream_t` slots, so output order is unaffected — this
is what `RNAfold_cpu_queue.c` did before it was retired, and reviving that idea
is now justified where in §2 of this document it was not.

### D — Thread `build` *and* pipeline it (A + B)

They compose: the pipeline hides the build, threading shortens what has to be
hidden. Only worth designing once one of them is measured.

---

## 3. On making CUDA the default

Separate from the above, and worth separating in the PR too.

Today: `gpu_enabled = (vrna_cuda_devices() > 0) && getenv("RNA_GPU_CHUNK")`.
The device probe already exists; what is missing is a CLI flag and a default.

- Flipping the default changes behaviour for every existing user of a
  CUDA-enabled build, so it wants its own flag (`--cuda` / `--no-cuda`) and its
  own note in the PR, not a silent policy change inside a performance commit.
- `gpu_path_usable()` already refuses the unsupported option surface and falls
  back silently, so the failure mode of "on by default" is *slow*, not *wrong* —
  which is the right shape.
- The one real risk is a machine with a CUDA device that is busy, tiny, or
  shared. `MIN_GPU_BATCH` and the VRAM budget already guard the small cases.

---

## 4. Recommendation

**B first, then C, then A.**

- **B** is race-free, worth ~18 %, and its only objection (RSS) is a measurement
  we can take. It also unblocks C by construction: once the reader stops blocking
  on `flush_gpu_chunk()`, there is a place to put CPU work.
- **C** is worth `cores / ~70`, so ~3 % on a 2-vCPU Colab T4 and **~46 % on a
  32-core workstation**. It is the only option whose value grows with the host.
  Build it as a work queue, not a split.
- **A** is the largest single number (~20 %) and the simplest code, but it needs
  either shipping UB or Defect B fixed upstream, so it goes last despite being
  the most obvious.

**C was recorded as declined in the first draft of this document and that was
wrong** — on a bad cross-machine ratio and on a "competes for cores" claim that
does not survive looking at which phase each option occupies.

~~Before any of it: re-measure the wall on the current tree.~~ **DONE** — §0 is
now measured rather than arithmetic (`6d3bdc19`, 523.7 s), and the predicted
31.6 % idle came back as **33.9 %**. The design stands on real numbers.

The one number that moved enough to matter: `backtrack` is **9 % of wall**, not
the 1 % the L4 run suggested, and it is *already* using every core the instance
has. On a core-rich host it shrinks and `build` becomes ~80 % of the idle; on a
core-poor one it stays and caps what C can return. Both effects point the same
way — **C's value is a property of the host**, which is the case for building it
as a self-balancing queue rather than a tuned split.
