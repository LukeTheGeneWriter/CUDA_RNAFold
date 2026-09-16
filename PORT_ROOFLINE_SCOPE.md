# Where we actually are: throughput, and what the A100 is capable of

*Written 2026-09-16 against commit `c4f7ddaa`. Everything measured here comes
from the A100-SXM4-40GB run in `lookup_a100_run2.json` (STRESS272 §32) except
where a different machine is named — and naming the machine turns out to be the
whole point of this document.*

---

## 1. The wall, at the fastest configuration we have

400 × 5601 nt, two chunks, build pipeline on: **85.26 s**.

| | s | share of wall | what it is |
|---|---|---|---|
| `modular_decomp` | 37.84 | **44 %** | the O(n³) multiloop scan — the algorithm's main term |
| `int_loop` | 16.54 | 19 % | interior loops, MAXLOOP-bounded |
| `fetch_mx` | 7.68 | 9 % | D2H copy of `c` and `fML` for backtracking |
| `hp_mb` + `load_my_c` | 3.28 | 4 % | hairpin/multibranch, matrix staging |
| **GPU total** | **65.3** | **77 %** | |
| `build` | 18.1 | 21 % | host: fold compounds — 6.5 s now hidden by the pipeline |
| `backtrack` | 5.2 | 6 % | host |
| everything else (gpuinit, output, free, teardown) | 1.3 | 2 % | host |
| **host total** | **24.6** | **29 %** | of which ~18 s is exposed |

GPU phase times are the phase-synced ones, which §32.1 showed are the true GPU
times *and* add only 0.2 % to the wall — so this table is production, not an
instrument artefact.

## 2. Throughput, in the units the problem is actually in

| | |
|---|---|
| DP cells computed | **6.275 × 10⁹** (400 × the 5601 nt triangle) |
| end-to-end | **73.5 M cells/s** — 4.7 records/s, 2.63 M nt/s |
| while the GPU has the floor | **96.1 M cells/s** |
| inner-loop iterations in `modular_decomp` | **1.17 × 10¹³** (≈ N·n³/6) |
| achieved rate | **3.10 × 10¹¹ lane-iterations/s** |

## 3. What the card can do, and what fraction of it we are using

A100-SXM4-40GB at the measured 1410 MHz: 108 SMs, **1.95 × 10¹³**
thread-instructions/s at full issue, **9.75 × 10¹² INT32 ops/s**, **1.55 TB/s**
DRAM, 400 W (we draw 245).

| measure | achieved | ceiling | fraction |
|---|---|---|---|
| `modular_decomp` instruction issue (4–8 instr per iteration) | 1.2–2.5 × 10¹² instr/s | 1.95 × 10¹³ | **6–13 %** |
| `modular_decomp` DRAM (NCU, §23.2; **re-measured 7.9 % in §33.1**) | 53–123 GB/s | 1 555 GB/s | **3.4–7.9 %** |
| `int_loop` DRAM (NCU, §32 run) | ~25 GB/s | 1 555 GB/s | **1.6 %** |
| `int_loop` occupancy | 18.6 % of peak warps | 100 % (50 % at one warp/block) | **19 %** |
| `fetch_mx` PCIe | 6.5 GB/s over 50.2 GB | ~20–25 GB/s pinned | **~30 %** |
| board power | 245 W | 400 W | 61 % |

**Nothing here is near a roof.** Not bandwidth (1.6–3.4 %), not arithmetic
(6–13 %), not the link (30 %). The card is clocked to the top, thermally
unbothered (§30 D1), and mostly waiting.

### What it is waiting for

`int_loop`'s stall mix on this A100 says it in one line: **`long_scoreboard`
5.89 stalls per issue-active cycle** — global-memory latency — against
`barrier` 0.00 and `mio_throttle` 0.00. L1 hits 74.6 %, L2 83.3 %, so the data
is *already in cache*; what is missing is enough independent work in flight to
cover the latency of getting it from there.

Two structural reasons, both visible in the launch metrics:

- **One warp per block** (`launch__block_size 32`) makes
  `launch__occupancy_limit_blocks = 32` the binding limit: 32 warps of a
  possible 64 per SM, so **50 % occupancy is the hardware ceiling before any
  code runs**, and we achieve 18.6 %.
- **The grid is one wave.** 3 408 blocks over 108 SMs is **31.6 blocks per SM**
  against that cap of 32. There is no second wave to fill the tail, so the
  kernel's duration is set by its slowest block, not its average one.

## 4. Why the returns have been single-digit

Because every lever we have pulled recently was an *instruction-count* lever on
a *latency-bound* kernel. §26 recorded the shape and we kept re-discovering it:
H1 removed 14 % of the loads and bought 3.7 %; H6 deleted nine dependent loads
from the prologue and bought 5.2 % of `int_loop` (**0.8 % of wall**); H7 bought
2.9 % of it (0.35 % of wall). Those are honest wins and they are *supposed* to
be small — removing work from a unit that is not the bottleneck moves the
bottleneck not at all.

**And the model we have been optimising under came from the wrong machine.**
`PROFILE272_RESULTS.md` measured `modular_decomposition_kernel` at **89.3 % of
DRAM peak** and concluded, correctly for that box: *"the way to go faster is to
move fewer bytes — not to restructure the compute."* That was an **L4**. The
same kernel on the A100 is at **3.4 % of DRAM peak** (§23.2), and int16 — the
whole "move fewer bytes" programme — measured **+0.0 %** on this card with zero
throttling. We have been aiming at a bottleneck this hardware does not have.

## 5. The measurement that is missing, and it is the big one

`modular_decomp` is **44 % of wall** and **we have never profiled it on an
A100**. We have its DRAM share (3.4 %) and nothing else: no stall mix, no
occupancy, no launch geometry, no wave count. Every one of those is a number we
*do* have for `int_loop`, which is less than half its size.

**ANSWERED 2026-09-16 — see §7 below.** What follows is the scope as written
before the run; the prediction in it is what the measurement was judged against.

So the next measurement is not a speed A/B at all:

```
ncu --section WarpStateStats --section SchedulerStats --section Occupancy \
    --section LaunchStats -k modular_decomposition_kernel ...
```

**This is now notebook §G** (`CUDA_RNAFold_Lookup.ipynb`, four arms: shipped,
`RNA_MD_TILE=8`, `=1`, and int16), with `launch__waves_per_multiprocessor` added
so the one-wave claim is measured rather than derived. If it comes back looking like
`int_loop` — `long_scoreboard` dominant, occupancy under 25 %, one wave — then
the kernel is latency-bound too, the L4 conclusion is formally retired, and the
work is *more parallelism*, not *fewer bytes*: cells per block, blocks per wave,
independent loads per thread.

## 6. An honest ceiling

Arithmetic from stated assumptions, **not** a measurement or a promise:

| phase | now | if | then |
|---|---|---|---|
| `modular_decomp` | 37.84 | issue efficiency 6–13 % → ~30 % | ~13 s |
| `int_loop` | 16.54 | same | ~6 s |
| `fetch_mx` | 7.68 | pinned + async instead of pageable | ~2.5 s |
| `hp_mb` + `load_my_c` | 3.28 | same as above | ~1.5 s |
| exposed host | ~18 | deeper pipeline / the CPU work queue | ~6 s |
| **wall** | **85.26** | | **~29 s** |

That is **~3×**, and every row of it is a restructuring rather than a tweak.
The two rows with the least uncertainty are the transfer one (we know the bytes,
we know the rate, we know pinned memory is ~3× pageable) and the host one (we
know the pipeline hides `(chunks−1)/chunks` of `build` and that two chunks is
the fastest chunk count — a *deeper* pipeline or a heterogeneous queue is the
only way to hide more of it at that width).

**The thing not to do is another 3 % kernel tweak.** We know what those are
worth now: §32.5 ranks every open lever, and the top of that list — chunk width
at −26.4 % — was not a kernel change at all.

---

## 7. The missing profile is no longer missing (2026-09-16, §33.1)

§5 called `modular_decomposition_kernel` the highest-value measurement left.
It has been made, and the prediction written before the run holds:

| | predicted | measured |
|---|---|---|
| dominant stall | `long_scoreboard` | **11.67 per issue, 48 % of all stall cycles** |
| DRAM | low single digits | **7.9 % of peak** |
| occupancy | well under 50 % | 46.1 % |
| waves/SM | near 1 | 0.66 *(at the profiling fixture — see the caveat)* |

**So both of the kernels that matter are latency-bound, and §4's diagnosis of
why the returns are single-digit stands for 63 % of the wall rather than 19 %.**

Three things the tile sweep added that the prediction did not contain:

1. **The 32-lane tile is buying COALESCING, not just parallelism.** Narrowing it
   costs 21 % at 8 lanes and 247 % at 1, and sectors per request go 2.04 → 10.99
   with `lg_throttle` 0.06 → 6.05. Any restructuring that breaks lane-striding
   along a row will lose more than it gains.
2. **int16's mechanism on this card, measured directly**: 0.79× the DRAM bytes
   and **6.8 % longer**, with SM throughput rising 17.3 → 19.5 %. Fewer bytes,
   more ALU, on a machine with bandwidth to spare.
3. **`imc_miss` is 3.25 per issue, 10× `int_loop`'s.** Nobody has an explanation.
   It is a `__constant__`-bank quantity, and `__constant__` memory is the one
   never-touched item in the fine-tuning list.

### What this makes the ordered work

1. **Re-measure the wave count at the production shape.** 0.66 waves/SM comes
   from 60 × 1800 nt at chunk 24; at 400 × 5601 the grid is far larger. Optimising
   for a fixture's launch geometry is how §27 misread registers.
2. **More independent loads per lane** — unroll the `y` scan so several loads are
   in flight per iteration. This attacks `long_scoreboard` directly, it is the
   same lever H1 used on `int_loop` for −3.7 %, and it does not disturb the
   lane-striding that (1) above says is load-bearing.
3. **Explain `imc_miss`.** Cheap to look at, 13 % of the stall cycles, and it has
   a named suspect.
4. **Leave int16 off on this class of card** and quote the mechanism, not just
   the null.
