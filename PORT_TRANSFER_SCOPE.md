# The transfer path — pinned staging, and what is actually left

*2026-09-09, against `946e57d8`. Local RTX 3050 under WSL2.*

## Why

`nsys` put **`cudaMemcpy` at 70.4 % of all host time inside CUDA calls** — 12 042
calls, 351 µs average, against `cudaLaunchKernel` at 11.8 µs and
`cudaDeviceSynchronize` at 2.4 % of the total. That is where the one busy core
goes, and it is why `RNA_GPU_BLOCKING_SYNC` could never have helped.

## What landed

The per-row uploads (`upload_i_H`, `upload_size_off_H`, in both kernel drivers)
now stage through **pinned** host memory. Each already kept a shadow copy for
content-deduplication; the shadow is now allocated with `cudaHostAlloc` and is
also the copy source, so a synchronous copy out of pageable memory — which makes
the runtime stage through an internal pinned buffer with the CPU participating —
becomes a copy out of memory that is already pinned.

Deliberately **synchronous**. `cudaMemcpyAsync` would let the host run ahead and
overwrite the staging buffer while the previous copy was still in flight; doing
that correctly needs a ring of buffers plus events, which is a bigger change to
make only if this one is not enough. `rnafold_pinned_alloc()` falls back to
`malloc` if pinning fails — slow rather than broken.

## Result: real, and much smaller than hoped

| | before | after |
|---|---|---|
| `cudaMemcpy` share of CUDA API time | 70.4 % | **67.4 %** |
| average call | 351 µs | **291 µs** (−17 %) |
| call count | 12 042 | **12 042** (unchanged, as expected) |

Correctness is settled: **byte-identical to the pre-change binary and to the CPU
route**, `make check` 146/146, and no new compiler warnings (the 16 in the build
log are pre-existing — a `BLOCK_SIZE` redefinition and a `-Wformat-extra-args`
bug that predate this work).

**Wall clock is not quotable here.** Between the two profiles the card recovered
from thermal throttling — `hp_mb` went 7.42 s → 1.64 s on identical input — so
any wall comparison across them measures temperature, not code.

## What is actually left, and it is not the index uploads

The pinning touched the *small* per-row copies: a `nfiles`-element row index and
a `nfiles+1`-element offset table, ~96 and ~200 bytes. Making those free could
never have removed 70 %.

The distribution says where the rest is: **median 54.6 µs, mean 291 µs, max
167 ms.** That tail is the bulk per-row traffic — `new_e` and `energy_min`, each
`g_row_total` ints (~192 KB at 24 × 2000 nt) — not the index uploads. Those are
the "six blocking pageable `cudaMemcpy` per row" the row-loop notes recorded, and
they are the real target.

Next, in order:

1. **Pin the bulk row buffers too**, which is the same change against much bigger
   payloads and should carry most of the remaining cost.
2. **Then** consider async + a ring + events, which only pays once the copies are
   pinned.
3. **Re-measure on a native-Linux datacenter card before quoting any of this.**
   WSL2's WDDM submission path has unusually high per-call latency, so the 70 %
   is a local upper bound. The fix is right on any host; the magnitude may not
   transfer.

## Future work: MIN_GPU_BATCH should be a function, not a constant

`VRNA_MIN_GPU_BATCH` is a fixed 10. It was already known to be wrong: break-even
is length-dependent — ~65 records at 300 nt, ~10 at 600 nt, **1 at ≥1200 nt** —
so a single constant is right only near 600–700 nt and too small below it.

It should be **derived from record count and record length** at the flush point,
where both are known. Two reasons this matters more now than it did:

- Every optimisation that makes the GPU path faster **moves the break-even**, so
  a constant drifts further out of date with each improvement. The threshold is a
  property of the current implementation, not of the problem.
- It is the honest way to answer *"how many jobs are genuinely not worth the
  GPU?"* — a question worth being able to answer directly, and one that gets more
  interesting as the port gets faster. A run could report it: records folded on
  the device, records declined as too small, and what the crossover was.

The pieces already exist — `gpu_bytes_per_file()` knows the size model,
`cpu_gpu_ratio.sh` measures the crossover — so this is a calibration and a
formula, not new machinery.

---

## CORRECTION — "pin the bulk row buffers next" was based on a false premise

The section above named `new_e` and `energy_min` as the remaining target, inferred
from the `max 167 ms` outlier in the call distribution. **Both are gated off in
the default configuration.**

```c
if(!rnafold_gpu_sweep())
  cudaMemcpy(d_new_e, new_e, g_row_total*sizeof(int), H2D);   /* int_loop.cu */
if(!rnafold_gpu_sweep()) {
  cudaMemcpy(energy_min, d_energy_min2, g_row_total*sizeof(int), D2H);
```

`RNA_GPU_SWEEP` **defaults to 1** (`mfe_cuda.c:126`) — the GPU-resident sweep has
been the default since 2026-08-30 — so neither copy executes. They are the host
sweep's traffic, and the host sweep is off.

I named them from the shape of the distribution without checking whether they
run. That is the same error as attributing a remainder without enumerating the
buckets, one section earlier in this same document.

### So what are the 12 042 calls?

They are the small per-row uploads — the ones already pinned. ~6 per row × ~2000
rows. Their cost is **not** pageable staging (pinning them bought only 17 %); it
is **per-call submission latency**, ~291 µs to move 96 bytes. That is a WSL2/WDDM
property, not an algorithmic one.

The single large outlier is `fetch_mx` — the `c` triangle read back per record
for backtracking, 24 calls, unavoidable and already only 1.4 % of wall on the T4.

### The revised verdict: the transfer path is done for now

| | |
|---|---|
| transfers on the T4 (`load_my_c` + `fetch_mx`) | **3.6 % of wall** |
| upper bound on any further transfer work there | **≤ 3.6 %** |
| the 70 % figure | CUDA-**API-time** share under WSL2, not wall, and a local artifact |

Reducing it further means reducing the *number* of calls, not their cost —
`upload_i_H` and `upload_size_off_H` push a per-record row index and an offset
table that are largely derivable on the device from the row number. That is a
real optimisation and a much bigger change than pinning, for ≤3.6 % on the
hardware that matters.

**Not worth doing next.** After the pipeline landed (424 s), the wall is
`modular_decomp` 54 %, `hp_mb` 25 %, **`backtrack` 11 %**, transfers 4.5 %.

Two things outrank it:

1. **int16 + pipeline has never been run.** Every stress notebook arm is i32-only
   for the pipeline. int16 alone is −3.6 %; combined is the obvious untested
   number and `modular_decomp` is 54 % of the remaining wall.
2. **`backtrack` is 11 % and is not hidden by the pipeline** — it runs inside the
   fold. It is already threaded (`auto`), so the question is whether it is
   core-starved or has a real serial section.
