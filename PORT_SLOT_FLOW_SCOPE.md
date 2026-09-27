# Slot flow: why it measures badly, and what was never actually tested

Investigation 2026-09-27, prompted by Luke: *"I'm suspicious that we may be shorting the
potential of slot flow... I hypothesize that something structural from building upon the
other design is hampering our attempts to show that slot flow would pay."*

That hypothesis is **confirmed, with a named mechanism**. Slot flow's measured cost is
an O(n²) re-initialisation inherited from the pre-slot design, and its GPU work is
unchanged.

> **READ SECTION 7 BEFORE SECTIONS 3 AND 4.** They argue that the cache thesis was
> untested and that a phase skew was worth building. It has since been tested on an
> A100 and **it is falsified**: 22× more live data produces the same L2 hit-rate curve,
> md's time rises monotonically as residency falls, and the residency the device needs
> is 36× the residency L2 can hold. The phase skew is dead with it. Sections 3 and 4 are
> kept for the reasoning and the arithmetic, not the conclusion.

---

## 1. Slot flow does the same work, and the GPU does not notice it

`fill_arrays_loop.c` already prints the sweep shape. At two shapes, sweeping
`RNA_SLOT_FLOW=k`, everything on the device is flat:

| 600×300 | wall | iters | record-rows | cells | GPU+transfer | non-sweep |
|---|---|---|---|---|---|---|
| flow only | 5.84 | 296 | 177 600 | 26 373 600 | 4.185 | **0.501** |
| k=2 | 13.04 | 592 | 177 600 | 26 373 600 | 4.157 | **12.386** |
| k=4 | 11.33 | 1 184 | 177 600 | 26 373 600 | 4.244 | 10.845 |
| k=6 | 10.29 | 1 776 | 177 600 | 26 373 600 | 4.323 | 9.712 |

| 100×1200 | wall | iters | record-rows | cells | GPU+transfer | non-sweep |
|---|---|---|---|---|---|---|
| flow only | 13.78 | 1 196 | 119 600 | 71 580 600 | 12.205 | **0.466** |
| k=2 | 16.38 | 2 392 | 119 600 | 71 580 600 | 11.749 | **16.186** |
| k=6 | 17.78 | 7 176 | 119 600 | 71 580 600 | 13.351 | 17.415 |

**`cells` and `record-rows` are byte-identical across every arm.** Slot flow does not do
more work. **GPU+transfer time is flat** — within 4 % at 600×300, and the small rise at
100×1200 (12.2 → 13.4) is the residency effect, not the penalty. The whole penalty is
host-side, outside the sweep: **0.50 s → 12.4 s, a 25× increase**.

`iters` scales exactly as k (296 → 592 → 1184 → 1776) while GPU time does not move, so
an extra iteration costs nothing on the device. Iteration count is not the problem
either.

## 2. The mechanism: `refill_gpu3` was never scoped to a slot

`fill_arrays_loop.c` calls two things at a handover:

```c
refill_slot2(nfiles, VC, turn, length, 512, tri_off_H, row_off_H, cap_H, s);
refill_gpu3 (nfiles, VC, turn, length, 512, row_off_H, cap_H);
```

`refill_slot2` is **slot-scoped** — it sets `g_slot_only`/`g_slot_index` so
`init_gpu2`'s uploads and fills touch only slot `s`. `refill_gpu3` takes **no slot
argument** and calls `init_gpu3(nfiles, ...)` for the whole batch. `hp_mb_loop.cu` says
so in as many words:

> `refill_gpu3()` is called at EVERY slot handover and takes no slot argument, so
> without this guard each handover INF-filled cc/cc1 for the WHOLE batch

So every handover re-runs the batch-wide hard-constraint pack. Cost is
**O(refills × slots)**, and `refills = n − slots`, `slots = ⌈n/k⌉`, so this is
**O(n²(1−1/k)/k) — quadratic in the record count.**

The `pack` counter inside the `gpuinit` breakdown accumulates across refills, so it
measures this directly:

| shape | k | slots | refills | refills×slots | pack |
|---|---|---|---|---|---|
| 600×300 | 2 | 300 | 300 | 9.00e4 | 7.326 s |
| 600×300 | 6 | 100 | 500 | 5.00e4 | 4.124 s |
| 100×1200 | 2 | 50 | 50 | 2.50e3 | 3.272 s |
| 100×1200 | 6 | 17 | 83 | 1.41e3 | 1.847 s |

Predicted vs measured ratio: **1.800 vs 1.776** at 600×300, **1.772 vs 1.772** at
100×1200. Two shapes, four points, an exact fit.

It also explains the A100 result that started this. At 10000×500 with k=2 the product is
**2.5e7 units — 278× the 600×300 probe** — which is why that shape lost 4.2× (58.5 s vs
13.8 s) while 400×5601 (4.0e4 units) lost only 1.2×.

**A HYPOTHESIS THIS KILLED, RECORDED.** The first guess was a per-refill fixed cost: a
`cudaDeviceSynchronize()` plus a slot upload plus an O(L²) `init_my_c_kernel`, all of
which `refill_slot2` really does do. It is wrong. At 600×300 the penalty **falls** as
the refill count **rises** (7.93 s at 300 refills → 4.81 s at 500), because the penalty
tracks `refills × slots` and `slots` falls faster than `refills` rises. Per-refill cost
is not what is being paid.

**And note how this defect was previously met.** The same missing slot argument caused a
wrong answer under `--noLP` + slot flow (2026-09-08), and it was fixed by adding a
`!g_refill3` guard around the cc/cc1 fill — patching the symptom while leaving the
whole-batch re-initialisation in place. That is exactly the structural inheritance Luke
suspected: the refill path was grafted onto a batch-shaped initialiser, one of the two
calls got scoped, and the other did not.

**The fix is scoping, not redesign**: give `init_gpu3` the same `g_slot_only` treatment
`init_gpu2` already has, so a handover repacks one slot. That turns O(n²) into O(n) and
should remove essentially all of the measured penalty, since the GPU work is already
flat.

## 3. The synchronised late-stage peak is real, and worse than stated

Luke: *"because smaller records joined late in the previous design, all of the records
were hitting their late stage traffic peak at the same time."*

Correct, and the mechanism is structural rather than incidental. The sweep is a single
global `i`-loop descending from the longest record. Every record's rows run from its own
top row down to 1 — so **all resident records reach i = 1 together**, by construction.
A record's live set is `(n−i)²/2` cells, so every resident record peaks at the same
instant.

Continuous flow's contribution is to let each record **start** at its own top row. But
the traffic distribution is:

- first 10 % of rows: **0.10 %** of reads
- last 10 % of rows: **27 %** of reads

**So continuous flow staggers the phase that carries a thousandth of the traffic and
leaves the phase that carries a quarter of it perfectly synchronised.** The stagger is
applied where it cannot help.

Slot flow reduces the number of simultaneous peaks from `nfiles` to `slots`, which is
real, but it does not destagger them: all first occupants still march in lockstep.

### What a phase skew would buy, exactly

Offset slot `s` so it begins its first record `s·L/K` iterations in. With K residents at
length L, live cells for a record with fraction f of its rows remaining is `(fL)²/2`:

- synchronised (all f = 1): total **K·L²/2**
- uniformly skewed (f_j = j/K): total **K·L²/2 · mean(f²) = K·L²/6**

**Exactly 3× smaller at the worst moment, and constant in time instead of ramping.**

Residents whose combined footprint stays inside a 40 MB L2 at int16:

| n | synchronised | skewed |
|---|---|---|
| 1200 | 29.1 | 87.3 |
| 2400 | 7.3 | 21.8 |
| 5601 | **1.3** | **4.0** |
| 8000 | 0.7 | 2.0 |

At production length that is the difference between one resident and four — a 3×
parallelism gain at equal cache pressure. It needs **no SM affinity and no new kernel**:
only different initial values for `i_H[s]`, the per-slot row pointer
`fill_arrays_loop.c` already maintains.

## 4. Why every experiment so far was incapable of showing this

**The laptop cannot reach the regime.** Measured with `cudaGetDeviceProperties`: the RTX
3050 Laptop has **L2 = 1.50 MB** and 16 SMs. One record's triangle at int16:

| n | int16 | records fitting a 1.50 MB L2 | records fitting a 40 MB L2 |
|---|---|---|---|
| 300 | 0.09 MB | 17.4 | 464 |
| 1200 | 1.37 MB | 1.1 | 29.1 |
| 2400 | 5.50 MB | 0.27 | 7.3 |
| 5601 | 29.92 MB | **0.05** | **1.34** |
| 8000 | 61.04 MB | 0.02 | 0.66 |

A 5601 nt record at int16 is **20× the laptop's entire L2**. The k-sweep above ran at
300 and 1200 nt, where the footprint went from 69× over L2 to 12× over L2 — never
resident at any k. **Flat GPU time there is exactly what the thesis predicts too, so it
is not evidence against it.** It is an untested case.

**The A100 at 5601 nt is the only place the thesis is decidable**, and it is decidable
precisely because **one record fits (1.34) and two do not.** That is a sharp boundary,
not a gradient.

It also has a ceiling worth knowing: at 8000 nt even one record does not fit (0.66), so
single-record L2 residency is a property of lengths up to roughly 6500 nt at int16.
`(n(n+1)/2)·2 ≤ 40 MB` gives n ≤ 6478.

**And the residency evidence we thought we had is confounded.** Flow3 §B varied
residency by varying *chunk width* (4, 24, 48, 96 residents at 96×5601) and found 48
best, 4 worst at 40.60 s. But chunk width also sets chunk *count*, and a chunk costs
~2 s: 4 residents means 24 chunks ≈ 48 s of per-chunk cost alone. That experiment could
not separate residency from chunk count. **Slot flow is the clean version** — it holds
the chunk at `nfiles` and varies only how many records are resident — which makes
fixing §2 a prerequisite for measuring §3 at all.

## 5. What to do, in order

1. **Scope `init_gpu3` on refill** (§2). Mechanical, O(n²) → O(n), no design change.
   Bar: `tests/` parity plus byte-identical output against slot flow off, and `pack`
   falling to O(n).
2. **Re-measure the k sweep on the A100 at 5601 nt**, int16, watching md's **L2 hit
   rate** rather than the wall. This is the first honest test of the cache thesis. The
   prediction that makes it falsifiable: md's L2 hit rate should rise sharply as
   residency crosses from 2 to 1, and not otherwise.
3. **Only then prototype the phase skew** (§3), which is an offset on `i_H[s]` and
   nothing else. It is worth 3× on footprint, which is what buys back the parallelism
   that single-record residency costs.

Step 1 is worth doing regardless of whether the thesis holds, because it is a quadratic
cost in a shipped code path. Steps 2 and 3 are the part that is currently unknown rather
than disproven — and the reason they look disproven is that they were measured on a
machine with 1.5 MB of L2.

## 6. What is genuinely closed, so it is not re-litigated

- **A dedicated SM cannot persist a fold.** 192 KB of L1/shared per A100 SM against a
  24.2 MB late-stage working set at 5601 nt int16 is **129×** short. What fits is ~17
  rows, or a 32-column band — and staging `fml_i` in shared measured **15 % slower**,
  while the band/corner lookback was worse at **every** K including 100 % coverage.
- **Block-owns-columns is measured**: the megakernel is 30–109 % slower, equal-width
  column ownership alone costs 3.4×, and 80 registers cap occupancy at 12.5 % against
  md's 33.8 %. Pinning late-stage folds to SMs is that design, applied to the phase that
  carries 27 % of the traffic.

---

# 7. THE CACHE THESIS IS FALSIFIED (A100, 2026-09-27)

Run at commit `420e4025` (pre-fix, which does not matter: at 48 records the O(n^2)
repack is 576 units, 0.64 % of the 600x300 probe, and NCU measures
`modular_decomposition_kernel` while the repack is a different kernel plus host time).

## 7.1 The control moved exactly as much as the test

48 records, `RNA_SLOT_FLOW=k`, footprints differing by **22x** at every k:

| k | residents | fp @1200 | fp @5601 | L2 hit % @1200 | L2 hit % @5601 |
|---|---|---|---|---|---|
| off | 48 | 66.0 MB | 1436.3 MB | 23.89 | 24.42 |
| 2 | 24 | 33.0 MB | 718.2 MB | 18.46 | 18.54 |
| 8 | 6 | 8.2 MB | 179.5 MB | 26.48 | 26.97 |
| 24 | 2 | 2.7 MB | 59.8 MB | 36.27 | 36.90 |
| 48 | 1 | 1.4 MB | 29.9 MB | 48.11 | 46.61 |

**Twenty-two times the live data, and the same curve to within half a point.** md's L2
hit rate is set by RESIDENCY, not by footprint. That was the stated falsification
condition for section B, and it fired.

## 7.2 The hit-rate rise is starvation, not a win

| | SM % | md_s | DRAM GB/s | DRAM GB total |
|---|---|---|---|---|
| 5601, off (48 res) | 38.3 | 4.20 | 253.8 | 1066 |
| 5601, k=2 (24) | 23.1 | 5.19 | 331.0 | 1718 |
| 5601, k=8 (6) | 10.1 | 6.82 | 154.8 | 1056 |
| 5601, k=24 (2) | 3.7 | 10.79 | 63.8 | 688 |
| 5601, k=48 (1) | 1.9 | 15.29 | 36.3 | 555 |

`md_s` rises **monotonically** and SM occupancy collapses **38.3 % -> 1.9 %**. DRAM GB/s
falls *with* occupancy rather than against it, so the kernel is doing less per unit time
— not fewer reads per unit of work. At 1200 nt total DRAM bytes actually go UP
(32 -> 69 GB) as residency falls.

Fewer residents is strictly worse at every step, at 48 records where the repack is
negligible:

| | k=2 | k=8 | k=24 | k=48 |
|---|---|---|---|---|
| md at 5601 | +23.6 % | +62.4 % | +156.9 % | +264.0 % |
| md at 1200 | +16.7 % | +175.0 % | +608.3 % | +1242 % |

## 7.3 The number that closes it

- residents for usable SM fill at 5601 nt: **~48** (measured 38.3 %)
- residents whose triangles fit 40 MB of L2: **1.34**
- **gap: 36x.** With the 3x phase skew of section 3: still **12x** short.

The parallelism an A100 needs and the footprint L2 can hold are two orders of magnitude
apart. They cannot both be satisfied, **so the phase skew is dead too** — it addresses
footprint, and footprint is not the binding constraint. Do not build it.

## 7.4 What survives

- **The repack was a real O(n^2) defect** and is fixed (section 2). It takes slot flow
  from a 2.2x loss to roughly break-even at k=2, degrading as k rises. It is a defect
  fix with no performance case behind it.
- **"48 residents is optimal" is dead as folklore.** The curve is monotone: more
  residents is always better, at both lengths. Flow3 section B's optimum at 48 was
  chunk count, exactly as section 4 suspected.
- **Residency is not a cache lever at all.** Any future scheduling idea that reduces the
  number of concurrently-swept records starts 36x underwater on SM fill, whatever it does
  to footprint.

## 7.5 Method notes from this run

- Section A's "one sha across every arm: NO (2 distinct)" was **the check being wrong**,
  not a defect: it compared shas across two different fixtures. Within each shape the sha
  is constant across every k. Slot flow is answer-neutral.
- An earlier claim in this session that the fix made slot flow "win by 20 %" was measured
  on the BROKEN intermediate binary that produced wrong answers, so those walls were
  meaningless. Corrected numbers are in section 2.
- The 1200 nt control is the cheapest cell in the notebook and it is the one that ended
  the thesis. Build the control first.

