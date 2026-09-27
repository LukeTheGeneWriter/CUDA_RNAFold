# Blocked Zuker: are we in a local optimum, and can we get out?

Prompted 2026-09-27 by Luke: *"Do not use these results to suggest a small pivot, use
them to consider whether or not we'd even achieve the theoretical optimal design. If we
keep making small greedy steps forwards, would we just find ourselves in a local optimum
instead of the theoretical optimum?"*

**The answer is yes, we are in one, and the evidence is this project's own history.** A
prototype on the laptop measures **6.30× on md's inner loop, bit-exact**, from a change
no sequence of small steps would ever have found.

---

## 1. Why every previous optimisation was confined to a level set

`modular_decomposition` computes

    fM2[i][j] = min over k in [i+TURN+1, j-TURN-2] of ( fML[i][k] + fML[k+1][j] )

The production kernel evaluates one output cell per warp, with lanes striding the
reduction down a contiguous run of column `j`. Read from the source rather than the
comments, two facts follow:

- `fML[i][k]` comes from a contiguous **row buffer**, re-read by every `j`, so it is
  cached by anything at all;
- `fML[k+1][j]` is **pure streaming with no intra-row reuse**. Per row, the union of
  those reads is the whole sub-triangle, **each element read exactly once**. The reuse is
  entirely *across* rows, at a reuse distance of one full row.

So the loop is 2 ops per 2 bytes at int16 = **1.0 ops/byte**, against a machine balance
of ~13 on an A100 and ~20 on this laptop. **md at 82.8 % of DRAM peak is not a tuning
failure — it is what this loop is.**

And that is the level set. Every optimisation this project has tried lives inside it:

| change | what it moved | intensity after |
|---|---|---|
| int16 fML | bytes per element, 4 → 2 | 1.0 ops/byte |
| warp-per-cell scan, block sizes, flat grid, hoisting, log32 lookup | instructions per cell | 1.0 |
| diagonal band, shared corner, `fml_i` in shared, L2 persisting window | *where* the stream is read from | 1.0 |
| (min,+) pruning | number of elements read | 1.0 |
| continuous flow, slot flow, residency caps, admission | which records are resident | 1.0 |
| the megakernel | launches and row buffers (calls `md_cell()` **verbatim**) | 1.0 |

**No local move can change a fixed arithmetic intensity.** That is why the roofline
result has been stable across dozens of experiments — the experiments were all inside
the level set, and the level set has a ceiling.

It also explains the *pattern* of failures, which is more useful than the individual
results. The band, the corner, the shared stage and the SM-resident window all tried to
cache a stream whose reuse distance is one full row. They could not have worked, at any
size, in any arrangement. That is not five unlucky experiments; it is one structural
mistake made five times, and a greedy search will keep making it because each attempt
looks locally reasonable.

## 2. What is outside the level set

`min over k of (fML[i][k] + fML[k+1][j])` is a **(min,+) inner product**, and over a tile
of the output it is a **(min,+) matrix product**. A matrix product is the textbook case
where blocking converts bandwidth into on-chip reuse: two `b × b` blocks in shared give
`b³` work for `2b²` loads, so intensity rises by `b/2`.

**On legality**, because it is the first objection. In the real sweep
`md(i) → c(i-1) → fML(i-1) → md(i-1)`, so md's *rows* are strictly ordered and the
production schedule (a whole row at a time) is forced by that. **Tiles are not.** For
output tile `(I,J)`, every `k`-block strictly between `I` and `J` refers to fML blocks at
block-distance `< J-I`, so ordering tiles by anti-diagonal `d = J-I` makes them all
complete. Only the `k` values inside blocks `I` and `J` need the sequential treatment,
and for `d ≫ 1` that is a vanishing share of the work.

**Blocked Zuker is legal. The current row-at-a-time order is simply the most
traffic-heavy legal schedule.**

## 3. Measured, on this laptop, bit-exact

`tools/proto_blocked_md.cu` builds the same computation three ways on identical synthetic
fML, and checks every cell of the output for bit-equality. RTX 3050 Laptop (1.50 MB L2,
16 SMs, 100 KB shared/SM), n = 4096, a 16 MB triangle — 10× the L2, so the streaming
version genuinely misses.

| kernel | ms | vs 1-launch | exact |
|---|---|---|---|
| streaming, one launch **per row** (production's schedule) | 150.96–189.64 | — | — |
| streaming, one launch total (isolates the loop from launches) | **150.96** | 1.00× | — |
| blocked, b=16, 1 result/thread | 82.54 | 1.83× | yes |
| blocked, b=64, 4 results/thread | 73.80 | 2.05× | yes |
| blocked + register tile, b=32, 2×2 | 45.42 | 3.32× | yes |
| blocked + register tile, b=64, 2×2 | 44.23 | 3.41× | yes |
| **blocked + register tile, b=64, 4×4** | **23.95** | **6.30×** | **yes** |
| blocked + register tile, b=96, 4×4 | 27.99 | 5.39× | yes |
| + packed int16 SIMD, b=64, 4×4 | 44.12 | 3.39× | yes (but slower — see below) |

Stable across size: **5.82× at n=2048, 6.30× at n=4096, 6.08× at n=8192.**

(b=128 with an 8×8 tile needs 66 KB of shared against a 48 KB per-block limit, so that
row is a failed launch, not a result.)

### The two-step nature of the result is the interesting part

Naive blocking alone gave only 2.05×, and NCU says exactly why:

| | streaming | blocked (b=16) | blocked + 4×4 reg |
|---|---|---|---|
| DRAM bytes | 22.77 GB | 1.49 GB | **0.476 GB** |
| L2 hit | 4.67 % | 44.25 % | 71.52 % |
| shared wavefronts | 5.0e7 | 7.8e8 | 2.0e8 |
| instructions | 4.55e9 | 4.69e9 | **1.32e9** |
| warps active | 89.5 % | 99.9 % | 66.5 % |

Plain blocking **converted a DRAM bottleneck into a shared-memory and issue bottleneck
almost exactly 1:1** — 15.3× less DRAM, 15.5× more shared traffic, instruction count
*unchanged*. That is why it only bought 2×.

The fix is arithmetic, not tuning: one (min,+) step costs two shared loads, an add and a
min — 2 useful ops for 4 instructions. A thread owning an `RM × RN` sub-tile loads
`RM + RN` values and does `RM × RN` steps, so shared accesses per step fall from 2.00 to
`(RM+RN)/(RM·RN)` = 0.50 at 4×4. Instructions fell 3.4× and DRAM fell a further 3×.

**A second trick, free:** the `k` range `[i+TURN+1, j-TURN-2]` is masked into the
*staging* rather than tested in the inner loop. X's half of the condition depends only on
`(r,t)` and Y's only on `(c,t)`, so both apply while writing shared memory. The inner loop
then has no branch at all and unrolls fully. It stays exact because a masked entry
contributes `INF16` plus a real value, ≥ 12128, while every legitimate sum is ≤ 4000.

### Headroom left in the prototype

- **Registers cap occupancy at 67 %** (56 regs × 256 threads → 4 blocks/SM). `warps
  active` = 66.5 % matches exactly. Trimming the accumulator block or the index
  arithmetic should recover some of it.
- **Packed int16 SIMD: TRIED, and it is a NULL on Ampere.** `__vaddss2` + `__vmins2`
  pack two reduction steps per instruction and are bit-exact here (values stay inside
  ±32256, so saturation never triggers). Measured at n=4096: **44.12 ms against 23.38 ms
  for the scalar 4×4 — 1.9× SLOWER**, with **2.03× MORE instructions** (351M vs 173M).
  The SASS says why: `__vmins2` is synthesised from `IMNMX` + `IADD3` + `PRMT` + `SHF`
  + `LOP3`. **These intrinsics are emulated on sm_86, not native**, so packing costs
  more than it saves.

  **But the instruction exists on Hopper.** DPX (`__viaddmin_s16x2`, `__vimin3_s16x2`)
  is a native *fused add-then-min on two packed int16 lanes*, which is precisely this
  recurrence's primitive, and it is sm_90+. So the blocked kernel's remaining
  instruction-bound headroom is unlockable on an H100 and not on an A100 — worth knowing
  for hardware planning, and worth a `#if __CUDA_ARCH__ >= 900` arm whenever this is
  built for real.
- It is no longer memory-bound at all: 0.476 GB in 23.95 ms is 20 GB/s on a 192 GB/s bus.

## 4. What it is worth end to end, stated so it cannot be oversold

md is **55 %** of GPU time at 48 × 5601 on the A100 (4.20 s of 7.70 s). Amdahl therefore
caps this hard:

| scenario | gpu_total | speedup |
|---|---|---|
| as built | 7.70 | 1.00× |
| md 6.3× (this prototype) | 4.17 | 1.85× |
| md at machine balance (~13×) | 3.82 | 2.01× |
| + int_loop window tiled 2× | 3.07 | 2.51× |
| + int_loop window tiled 4× | 2.70 | 2.85× |

So **~2× on GPU time, not 6×** — because md is only half of it. That is still larger
than every scheduling lever measured on this branch put together, and unlike them it
moves the roofline rather than the schedule.

`int_loop` is the natural second target and a *different* shape: one block per `(H,j)`
cell scanning a fixed 30×30 window of `my_c`. Neighbouring cells share nearly the whole
window, so a block owning 32 consecutive `j` reads `(30+32)×30` instead of `32×900` —
**15.5× fewer loads**, in 7.3 KB of shared. And it is latency-bound with
`long_scoreboard` dominant, so removing loads attacks the actual stall rather than bytes.

## 5. What integration actually costs — the honest part

The prototype is **not** a port. It takes fML as a given matrix, which is exactly what
isolates the `n³/6` term from the recurrence that produces it. Integrating means:

1. **A tile-wavefront driver.** The sweep becomes an outer loop over block anti-diagonals
   `d`, with all tiles at one `d` independent. That replaces the per-row `i`-loop in
   `fill_arrays_loop.c`, which is the spine every other phase hangs off — so int_loop,
   hp_mb, new_c and the fML scan must all be re-expressed on the same tile schedule, or
   run on the old one with a synchronisation between them.
2. **Diagonal tiles need the sequential recurrence.** Off-diagonal tiles are pure
   products of finished blocks; tiles on the block diagonal contain their own dependency
   and need the within-tile order. That is a second kernel, or a branch in the same one.
3. **A row-major mirror of fML.** The prototype reads `fML[i][k]` from a row-major copy
   because the triangle makes that access strided. Production already maintains a row
   buffer for the current row; tiling needs `b` rows at once.
4. **The correctness bar is the easy part.** Byte-identical fold output against
   `RNA_GPU=0` on mixed-length fixtures, plus option parity — the harnesses already do
   this, and the prototype's own cell-for-cell check shows the arithmetic can be made
   exact.

This is a **week-scale change to the spine of the sweep**, not an afternoon's kernel
edit. Which is the whole point of the question Luke asked: a greedy search never takes it,
because every intermediate state is *worse* than where we are now — you pay the
restructuring before you get the reuse. The only way to reach it is to decide to, on the
strength of an isolated measurement made before committing. That measurement now exists
and says 6.30×.

## 6. What this says about the megakernel, and about slot flow

The megakernel calls `md_cell()` verbatim, so it cannot move md's 1.0 ops/byte. Fusion
removes launches and keeps row buffers on chip — both real, both small — but **a fused
kernel at the DRAM roofline is still at the DRAM roofline.** The recorded split (26 %
faster than the per-phase path on the laptop at G≥8, 30–109 % slower on the A100) is
consistent with that: trading DRAM for on-chip locality pays on a 192 GB/s laptop and
much less at 1555 GB/s. The A100 needs the intensity fix, not the fusion.

Slot flow is now cheap (the O(n²) repack is scoped, `pack` 7.326 → 0.134 s) and
answer-neutral, so it remains a usable scheduling option. But residency is not a cache
lever: md's own numbers say one resident costs **3.63×** for identical work, against an
absolute ceiling of ~4.7× from a perfect DRAM→L2 conversion — and the reason is width,
not cache. One record's row averages 2800 cells ≈ **0.41 of one A100 wave**, and a
DRAM-bound kernel needs many waves in flight. Blocking is the opposite trade: it gives
each *thread* more work, so it needs fewer waves rather than more residents.

## 7. So: local optimum, and how to tell next time

The tell is that **the limiter never moved.** Twelve classes of optimisation, and md
stayed at ~82 % of DRAM peak throughout — because every one of them changed the schedule,
the placement or the instruction count while leaving arithmetic intensity at 1.0 ops/byte.

The cheap diagnostic, usable before any experiment: **compute the loop's ops per byte and
compare it to the machine balance.** If the ratio is far from 1, no amount of reordering
or caching will help, and the only useful changes are the ones that alter the ratio. That
test costs nothing and would have retired the band, the corner, the shared stage, the L2
window and the residency thesis before any of them were built.
