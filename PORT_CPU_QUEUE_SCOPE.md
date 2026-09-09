# Option C — folding on the CPU while the GPU folds

*Scoped 2026-09-09 against `34dc6d9d`, after A and B landed.
Measured on the local box (12 cores, throttled RTX 3050).*

---

## 1. The number that matters

`/usr/bin/time -v`, 64 × 1500 nt, three chunks, accelerated path:

| configuration | CPU used (of 1200 %) |
|---|---|
| baseline | **98 %** |
| `RNA_BUILD_THREADS=auto` | 106 % |
| `RNA_BUILD_PIPELINE=1` | 105 % |
| **A + B together** | **107 %** |

**The accelerated path uses about ONE core out of twelve, and A and B barely
move that** — they shorten a stage that is only ~5 % of wall here. Eleven cores
are idle for the whole run.

This is the same shape as the `1.00 cores of 12` finding that originally made
thread pools worth doing, and it is the entire case for C.

## 2. What it is worth

Extra throughput ≈ `free_cores / ratio`, where `ratio` is CPU-seconds per record
on one core over GPU-seconds per record (`tools/cpu_gpu_ratio.sh`):

| host | ratio | free cores | C adds |
|---|---|---|---|
| Colab T4, 2 vCPU | ~70 | ~1 | **~1.5 %** |
| Colab L4, 8–12 vCPU | ~70 | ~10 | ~14 % |
| **this laptop, 12 cores** | ~40 (half-clock GPU) | ~11 | **~27 %** |
| 32-core workstation | ~70 | ~31 | ~44 % |
| 64-core server | ~70 | ~63 | ~90 % |

**Note the inversion: C is the first option whose bar belongs HERE rather than on
Colab.** A and B are worth ~0 locally and ~20 % on Colab; C is worth ~1.5 % on
Colab's T4 instances and ~27 % on this laptop. Measuring it on Colab would
conclude it is worthless.

## 3. The design is much smaller than §2 of PORT_HETEROGENEOUS_SCOPE.md assumed

That document said C needed the fold to become asynchronous so the reader could
keep feeding CPU workers. **It does not.** The pieces already exist:

- `RUN_IN_PARALLEL(process_record, rec)` folds a record on the CPU *today* — that
  is the `MIN_GPU_BATCH` fallback path — and with `-j` it returns immediately.
- A record that never entered a GPU chunk has `prefolded == 0`, so
  `process_record()` folds it itself. No new folding code.
- Output order is by `vrna_ostream_t` slot, requested at read time, so routing
  does not affect the file.
- **Defect B is fixed**, so several threads may now call `vrna_fold_compound()`
  at once. Before `34dc6d9d` this design was unavailable on correctness grounds.

So **C v1 is: hold back `m` records from each chunk and dispatch them to the `-j`
pool immediately before folding the GPU part.** The pool works through them while
the GPU works through the rest. That is a change to one function.

```
flush point for chunk of n records:
    m = cpu_slice(n)                    /* 0 today */
    for i in [n-m, n):  RUN_IN_PARALLEL(process_record, chunk[i])
    pipeline_flush(chunk, n-m, opt)     /* unchanged */
```

**C v1 requires `-j`.** Without a pool `RUN_IN_PARALLEL` runs inline and the
"slice" is folded serially before the GPU starts — strictly worse. So `m` must be
0 whenever `opt->jobs <= 1`. That is a real dependency and should be documented,
not hidden.

## 4. Choosing `m`

`m = free_cores × t_gpu_chunk / t_cpu_record`, and both terms are observable at
runtime — the previous chunk's fold time is known, and CPU per-record cost can be
sampled.

Two ways to get there:

**(a) Adaptive across chunks.** Start at `m = 0`, and after each chunk compare
when the slice finished against when the GPU part finished, moving `m` toward
balance. Converges within a few chunks on a long run and needs no model. Requires
a completion counter from the pool, which `thpool` does not expose — an atomic
incremented at the end of `process_record` would do.

**(b) Calibrated once.** Fold a couple of records on one core at startup,
measure, compute `m`. Simpler, but pays a serial calibration and is wrong the
moment the GPU clock changes — which this session has watched happen repeatedly.

**(a) is the right answer** for the same reason the work-queue framing was: the
ratio depends on the card's clock state, which moves under you.

## 5. What could go wrong

| risk | severity | mitigation |
|---|---|---|
| **Tail imbalance** — the CPU slice is still folding after the GPU has finished everything, so wall is set by the slowest core | **highest** | keep `m` conservative; never put the *longest* records in the slice (CPU cost is ~O(n³)); consider draining the slice before the final chunk |
| Oversubscription — build (A), builder (B), backtrack, output and now folders | low **today** (1 core of 12 used) but real on a busy host | one core budget shared by all pools, rather than each calling `nproc` independently |
| Host RAM — each CPU folder needs its own O(n²) matrices, ~125 MB at 5601 nt | moderate | 11 workers ≈ 1.4 GB on top of the chunk; count it, and cap workers by RAM as well as cores |
| Routing is non-deterministic | **none, and this is already proven** | which route a record takes does not change its answer — 18 stress arms across three runs share one output hash, and `verify_option_parity.sh` asserts route-vs-answer independence |

The tail case is the one that can make C a *regression* rather than a
disappointment, and it is why `m` must be conservative and length-aware.

## 6. Bars

1. **Byte-identical output** with `m > 0` against `m = 0`, mixed lengths, several
   chunk counts. Cheap and decisive, and the routing-independence work means a
   failure here would be a real defect rather than an expected difference.
2. **Wall on this box**, 12 cores: the prediction is ~27 %, and it is the first
   option where a local measurement is the *primary* evidence.
3. **`m = 0` when `-j` is absent** — assert it, because the failure mode is a
   silent slowdown rather than an error.
4. **The tail**: a run whose record count is a poor fit for the slice, checked
   for a wall *regression* rather than only for correctness.
5. `make check`, and the option matrix, since C changes routing.

## 7. Recommendation

**Do C v1 (the static slice) with adaptive `m`, gated behind `RNA_CPU_SLICE`,
default off.** It is one function, it reuses the existing CPU fold path entirely,
and it is worth more on a real workstation than A and B combined.

**Do not build the general work queue first.** The queue is the right end state if
records ever need to migrate between routes mid-flight, but it needs a shared
structure the streaming reader does not currently have, and v1 answers the
question the queue would answer — *how much is the CPU actually worth here* —
with a fraction of the risk.
