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
