# Heterogeneous execution and `-j` — options

*Scoped 2026-09-09 against `6d3bdc19`. Numbers from `stress272_t4.json`
(400 × 5601, T4, commit `851d1b04`) with the `output` fast path applied
arithmetically, since that run predates it. **No new measurement here** — this is
a design note over existing data, and every option ends with what would settle it.*

---

## 0. Where the machine actually idles

T4, i32/natural, after the `output` fast path:

| | s | share of the ~521 s wall |
|---|---|---|
| GPU kernels | 335.5 | 64 % |
| transfers | 19.7 | 4 % |
| **host-only (GPU idle)** | **164.6** | **31.6 %** |
| — of which `build` | 121.9 | **74 % of the idle** |
| — of which `backtrack` | 40.6 | 25 % |

On the L4, applying both fixes arithmetically, the same figure is **~33 % idle
with `build` at ~31 % of the new wall.** The two cards agree on the shape.

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

The literal reading of "run the accelerator with the CPU".

- **Prize, and it is small.** Upstream folds 400 × 5601 in 25 633 s = **64 s per
  record per core** (bench v5 arm A). The GPU does 335 s / 400 = **0.84 s per
  record**. That is **~76× per core**; with 8 cores the CPU can add **~10 %**
  throughput, and only while perfectly load-balanced.
- **Cost:** those are the same cores A and B need, and building is worth more per
  core than folding — build is on the critical path for *every* record, whereas
  CPU folding substitutes for a device that is 76× faster at it.
- **Verdict: do not do this before A or B.** It competes for the scarce resource
  and pays the worst rate for it. It becomes interesting only once the GPU is
  saturated and the cores are otherwise idle.

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

**B first, then A.** B is race-free, comparable in prize, and its only objection
(RSS) is a measurement we can take. A is larger and simpler but requires
shipping UB, or getting Defect B fixed upstream first.

**C is not worth doing** on these numbers and should be recorded as declined with
the 76× ratio, so it does not get re-proposed.

Before any of it: **re-measure the wall on the current tree.** Both large host
stages have changed since the numbers above were taken, and every figure in §0 is
arithmetic on a run that predates the `output` fix. Designing a scheduler against
a stale decomposition is exactly the mistake the ~73 % `modular_decomp` share
caused earlier.
