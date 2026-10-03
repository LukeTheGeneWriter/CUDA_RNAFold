# Integrating blocked Zuker into md: the plan

Follows `PORT_MD_BLOCKING_SCOPE.md`, which measures **7.66× bit-exact** on the isolated
(min,+) product. This document is the shape of the integration, the dependency proof it
rests on, and the staging — written before the code, because this touches the spine of
the sweep and a half-finished restructure is worse than none.

---

## 1. The dependency structure, read from the source

Three facts, each checked in the code rather than inferred:

**(a) `fML` does not depend on `fM2`.** `fml_scan_block.inc` computes `fML[i][j]` from
`new_e` (the row's `c`), `e3p00`, `gq_row`, `fml_prev` (row `i+1`) and `up_ml_ok`. DMLi
is not an input. So

    fML[i][j]  <-  c[i][j],  fML[i+1][j],  fML[i][j-1]

and the within-row part is a left-to-right affine (min,+) scan, which is why it is a
Hillis-Steele scan today.

**(b) `md` consumes `fML[i][*]` and produces `fM2[i][*]`.** From `md_cell.inc`:

    fM2[i][j] = min over k in [i+TURN+1, j-TURN-2] of ( fML[i][k] + fML[k+1][j] )

with `A[y] = fML[i][i+TURN+1+y]` from the row buffer and `B[y] = fML[i+TURN+2+y][j]`
walking down column `j`.

**(c) `c` consumes `fM2` from the row below.** `c[i][j]` takes the multiloop closing term
from `DMLi1[j-1] = fM2[i+1][j-1]`.

Composing: **`md(i) → c(i-1) → fML(i-1) → md(i-1)`.** md's rows are strictly ordered, and
the production schedule — a whole row per iteration — is forced by that. It is *not*
forced to be a whole row at a time in the column direction, and that is the opening.

## 2. Why row-at-a-time cannot be blocked, and tiles can

For a single row, `out[j] = min over k of (A[k] + B[k][j])` is a **matrix–vector**
product: every element of `B` is read exactly once. A matvec has arithmetic intensity ~1
by construction and there is nothing to block. That is the real reason every caching
attempt on md failed — not the cache, the schedule.

Blocking needs `RB > 1` rows sharing one `B` block. The obstruction is (c): row `i-1`'s
`fML` needs `c[i-1][*]` which needs `fM2[i][*]`. So rows cannot be batched *as whole
rows*. They can be batched **within a column block**, because the dependency is per-cell,
not per-row:

For tile `(I, J)` — rows in block `I`, columns in block `J` — split the `k` range:

| `k` lies in | `fML[i][k]` from | `fML[k+1][j]` from | status |
|---|---|---|---|
| blocks strictly between `I` and `J` | tile `(I,K)`, `K<J` | tile `(K+1,J)`, earlier block-row | **complete** |
| block `I` | tile `(I,I)` | tile `(I,J)`, row `>i` | within tile, earlier in `i`-descending order |
| block `J` | tile `(I,J)`, col `<j` | earlier block-row | within tile, earlier in `j`-ascending order |

So a tile splits into **a bulk that is a pure (min,+) product of finished blocks** and **a
tail that is sequential inside the tile**. For `J-I = d`, the bulk is `(d-1)/d` of the
work, so the tail is negligible as soon as `d` is more than a few.

## 3. The target schedule: a block-row sweep, COLUMN BLOCK MAJOR

This is the part that makes integration tractable. A full anti-diagonal wavefront over
tiles would replace the sweep entirely. It is not needed:

```
for I = last block-row .. 0            # descending, like today's i-loop but RB rows wide
    for J = I .. last block            # ASCENDING columns -- and this order is forced
        BULK(I,J)                      # ONE launch, all RB x CB cells, blocked
        for i in block I, DESCENDING   # the existing row-shaped kernels, restricted
            physics(i, columns of J)   #   to this column block
            fM2[i][j] = min( BULK[i][j], cor1, cor2 )   for j in block J
```

**The order is not a preference.** Rows inside a block are serially dependent —
`md[i][j]` needs `md[i+1][j-TURN-3]` through `c` — so the rows of a block cannot run
concurrently, and *that* is why `RB > 1` buys nothing unless md's `k` range is split.
Splitting it at `imax` (the first row of the block, since `i` descends) and at `j0`
gives three pieces, and only the first is available for every row of the block up
front:

| piece | `k` range | `A = fML[i][k]` from | `B = fML[k+1][j]` from | ready |
|---|---|---|---|---|
| **bulk** | `[imax, j0-1]` | row `i`, columns `< j0` → earlier **column** blocks | rows `k+1 > imax` → earlier **block-rows** | **before the tile runs, for all RB rows** |
| cor1 | `[i+TURN+1, imax-1]` | row `i`, near-diagonal, columns `< j0` | rows `k+1 ∈ [i+TURN+2, imax]` → **inside** the block | only after the rows above `i` |
| cor2 | `[j0, j-TURN-2]` | row `i`, **this** column block | rows `k+1 > j0 > imax` → earlier block-rows | after row `i`'s own fML scan |

The three ranges are contiguous and cover `[i+TURN+1, j-TURN-2]` exactly, so `min` of
the three **is** md. The bulk is `(j-i) - (RB+CB)` of the `(j-i)` terms, so the corners
are a few percent while `RB+CB` stays small — and they are the reason the column blocks
must ascend (cor1 and cor2 both need data from earlier column blocks of the same
block-row).

**What changes:** one iteration advances `RB` rows instead of one; the four per-cell
phases (`int_loop`, `hp_mb`, `new_c`, the fML scan) run on a column *range* instead of a
whole row; the fML scan needs a carry-in from column `j0-1`, which is legal because it
is an affine (min,+) scan and therefore associative.

**What that costs — and this is the knob that matters.** The physics is relaunched once
per `(row, column block)` instead of once per row, so the launch count multiplies by
`n/(2·CB)`. Against that, the corner work grows as `RB+CB`. Large `CB` for few launches,
small `RB+CB` for a cheap corner: §3.6 measures where that lands.

`fill_arrays_loop.c`'s structure, the row tables, continuous flow's per-record row
pointers and the retire pool all survive, because the outer loop keeps its shape and
direction.

## 3.6 The decomposition, measured: `tools/proto_blocked_md3.cu`

**It is exact.** Every configuration tried reproduces the streaming reference cell for
cell — 7 corner shapes × 10 block geometries, three runs, n = 4096 with 15 % INF.

And it is worth what the blocking thesis claimed, **once the corner is not shaped like
`md_cell`**. RTX 3050 at 1740 MHz, against the **per-row** reference (one launch per
row, which is what production actually does):

| corner lanes/cell | corner ms | total ms | vs per-row |
|---|---|---|---|
| 32 — `md_cell`'s own shape | 29.50 | 52.53 | 3.52× |
| 8 | 13.39 | 37.34 | 5.33× |
| **4** | **8.24** | **32.97** | **5.69×** |
| 2 | 7.54 | 32.46 | **5.91×** |
| 1 thread, no reduction | 10.39 | 34.86 | 5.50× |

**A 32-lane shuffle reduction over a ~64-element range is nearly all reduction.** The
corner went from 53 % of the time to 25 % by giving each cell 4 lanes instead of 32, and
that single change is worth more than every block-geometry choice below. md's warp-per-
cell shape is right for a 1400-element column and wrong for a 64-element corner; the
same kernel cannot have both.

| geometry | bulk | corner | total | vs per-row |
|---|---|---|---|---|
| RB32 CB32 KB32 4×4 | 41.84 | 8.27 | 50.11 | 3.80× |
| RB64 CB32 KB32 4×4 | 32.36 | 7.42 | 39.79 | 4.68× |
| RB64 CB64 KB32 4×4 | 24.74 | 8.24 | 32.97 | 5.69× |
| RB64 CB128 KB32 4×4 | 20.63 | 11.34 | 31.97 | 5.82× |
| **RB128 CB64 KB32 4×4** | **20.96** | 10.22 | **31.19** | **6.23×** |
| RB128 CB128 KB32 4×4 | 18.15 | 12.47 | 30.62 | 6.13× |
| RB128 CB256 KB16 8×8 | 14.99 | 16.94 | 31.93 | 5.90× |

The bulk falls monotonically with `RB` (reuse **is** `RB`: 41.8 → 20.9 ms from RB32 to
RB128) and the corner rises with `RB+CB`, and the total is flat at 30–32 ms across the
whole RB64–128 / CB64–128 region. **So `CB` is nearly free to choose, and it should be
chosen to minimise launches, not kernel time** — which points at the largest `CB` the
corner tolerates, around 128.

**`KB` had to be decoupled from `CB`.** Staging `KB` deep costs
`4·(KB·(RB+1) + KB·(CB+1))` bytes; tied to `CB`, `CB=128` needs **99 KB** and cannot
launch at all. Decoupled at `KB=32` it needs 25 KB, which is what makes the `CB` sweep —
the one that prices the launch multiple — possible at all.

### Two things this measurement corrected in the previous one

**§3.5's 7.66× is an upper bound no schedule can reach, for two separate reasons.**
First, `proto_blocked_md2.cu`'s tile blocks the whole `k` range including blocks whose
`B` operand lies *inside* the row block — values that do not exist yet. It gets away
with it because it is handed a complete synthetic `fML`. Second, its reference was timed
**once, at the top of the run, on a cold device**: re-run in-session it reports 5.86×,
not 7.66×. Both prototypes' absolute numbers move by 4× with this laptop's clock
(1057 MHz against a 2100 MHz maximum, at 16 W), so `proto_blocked_md3.cu` times **each
arm between two reference passes** and uses their mean. That is the only reason its
numbers are stable to 5 % run to run.

**A failed launch cost 0.00 ms and looked like a 33× win.** `RB256 CB256` with an 8×8
register tile needs 1024 threads × 64+ registers, over the 65536-per-block limit. The
first version of the harness printed it as a fast MISMATCH rather than as a failure.
Every launch is now checked — the same lesson as `nvcc … | head; echo rc=$?`.

## 4. Staging, with a bar for each

Each stage is independently landable and independently verifiable. No stage leaves the
tree slower or less correct than it found it.

### Stage 1 — the primitive, in-tree, exercised at `RB = 1` — **LANDED 2026-09-28**

It found three bugs and one wrong fix; `md_block.inc`'s header has them all. The one that
matters for the rest of this plan: **the range must be MASKED, not BOUNDED.** Masking it
to INF is wrong (a masked entry contributes `INF + negative`, lands below INF and wins the
min), and per-cell `t` bounds are exact but cost **3.6×** because the range differs for
every cell of a register tile. A mask **larger** than INF with the accumulator initialised
to INF is exact and branchless. **Stage 2 therefore starts from a branchless primitive**,
which matters because the bounded form would have eaten most of the win before Stage 2
began.

`md_block.inc`: the blocked tile product as a device function shaped for production —
per-record `tri_off_H`/`row_off_H` offsets, the int16 decode with per-64 baselines, the
same `INF` semantics. Selected by `RNA_MD_BLOCK_SELFTEST=1`, default off, which recomputes every row through the primitive and compares cell for cell against live md output.

At `RB = 1` the tile is `1 × CB`, which is still a matvec, so **this stage buys no
speed**. It buys the thing that must be right before `RB > 1` is worth attempting: the
index arithmetic, the decode, and the masking, validated in situ.

**Bar:** byte-identical fold output against `RNA_MD_BLOCK` unset, on mixed-length
fixtures, plus full option parity. Plus `RNA_MD_BLOCK` announced on stderr so a run that
did not take the path cannot pass for one that did.

### Stage 2 — `RB > 1`: the bulk kernel and the two corners

Raise the row block. The bulk becomes a real (min,+) product with reuse; the corners stay
per-row and use the existing kernels restricted to a column range. §3.6 measures the
whole decomposition at **6.23×** on an isolated `fML`, bit-exact, so the arithmetic is
settled before any driver work starts. What stage 2 adds in-tree:

1. `md_block.inc` gains the bulk tile (`RB`, `CB`, `KB` independent) and a corner entry
   point with a **tunable lane count** — 4 lanes, not md's 32. That is worth 1.6× on its
   own and is the single most important number in §3.6.
2. A `RB × CB` accumulator per record, `RB·CB·4` bytes × records — 2.9 MB at
   RB=32, CB=512, 44 records. Small, but new state to keep coherent.
3. A column-range argument on `int_loop`, `hp_mb`, `new_c` and the fML scan, plus a
   carry-in for the scan.

**Bar, and it is NOT the one this document originally set.** The original bar was
*"`dram__bytes.sum` for md down by roughly `CB/2`"*. That bar is wrong on an A100 at
production scale, for a reason worth keeping: md reads **2.39 bytes per B-element**
(§`project_a100_stress_results`), i.e. essentially all of its B stream already comes from
DRAM exactly once. Blocking cannot remove bytes that are only read once — it removes
*re-reads*, and md has none. What blocking actually removes is **load instructions and
the latency behind them**: the register tile does `RM+RN` loads for `RM·RN` steps, 0.50
against 2.00.

So the stage-2 bars are:

- **byte-identical fold output**, on mixed-length fixtures, with full option parity;
- **md wall time down**, which is the only claim that matters end to end;
- `smsp__inst_executed` and `l1tex__t_requests` for md **down by ≈ 4×**, which is the
  mechanism. If instructions fall and time does not, the bulk is bound by something else
  and the stage stops.

## 3.7 Stage 3's launch multiple, PRICED — and it rules out the per-row tail

§3 says the physics relaunches once per `(row, column block)`, so the launch count
multiplies by `n/(2·CB)`, and notes that this is "the knob that matters". It is worse
than that: **at `CB ≤ 128` it costs more than md can save**, so the schedule as scoped
cannot win by tuning. This was measured before the driver was written, with
`RNA_MD3_LAUNCH_PROBE=k` (device.cu): `k` extra no-op launches per sweep row, same
stream, fold unchanged.

RTX 3050, 6 records of 2400–3000 nt, warm-up discarded, 3 reps:

| k | mean wall | extra launches | µs per launch |
|---|---|---|---|
| 0 | 2.00 s | — | — |
| 32 | 3.05 s | 96 000 | **10.90** |
| 128 | 6.05 s | 384 000 | **10.53** |

Linear in `k` to 3 %, which is what makes it a per-launch cost rather than a
coincidence. (The first attempt at this used a 1.5 s fold where the cold k=0 run came
out at 1.97 s against 1.02 s warm — a 2× artefact bigger than the whole signal.)

Applied to the A100 at 400 × 5601 — 50 373 sweep iterations (= `int_loop_kernel`'s
launch count, `RNA_LAUNCH_STATS`), ~8 launches each, md 33.74 s of 59.68 s GPU:

| CB | launch multiple | extra launches | cost @4 µs | cost @10.5 µs | md after | net @4 µs |
|---|---|---|---|---|---|---|
| 128 | 21.9× | 8.4e6 | 33.7 s | 88.3 s | 9.0 s | **−8.9 s** |
| 256 | 10.9× | 4.0e6 | 16.0 s | 42.1 s | 10.5 s | +7.2 s |
| 512 | 5.5× | 1.8e6 | 7.2 s | 18.9 s | 13.1 s | +13.5 s |
| 1024 | 2.7× | 7.0e5 | 2.8 s | 7.3 s | 16.7 s | +14.3 s |

Two things fall out, and they point the same way.

**The geometry the kernel wants and the geometry the schedule wants are in conflict.**
§3.6 found the kernel flat across `CB` 64–128 and the corner rising with `RB+CB`; this
table wants `CB ≥ 512`, where the corner is ~24 % of the work and caps md at ~2.6×
however good the bulk is. The net never exceeds ~14 s of a 72 s wall — **about 1.24×** —
and that is with the optimistic 4 µs. At the 10.5 µs actually measured here, `CB=512`
breaks even.

**So the tail must stop being per-row.** The rows of a block are serially dependent only
through a 6-column skew, so one kernel can own a `RB × CB` tile and walk its rows with
`__syncthreads()` instead of returning to the host `RB` times for each of 4 phases. At
`RB=128, CB=512` that is **11 fused launches per block-row against 1024 today** — stage 3
stops paying a launch multiple and becomes a launch *reduction*, which also removes the
per-row overhead the megakernel was built to attack.

**This is not the megakernel again, and the difference is the part that killed it.**
`project_megakernel_a100_verdict`: 30–109 % slower, from 80 registers capping occupancy
at 12.5 % against md's 33.8 %, and equal-width column ownership costing 3.4×. Here md's
product stays a **separate, well-shaped kernel** — it is not inside the fused thing at
all — so the fused kernel holds only the four cheap per-cell phases, and the columns it
owns are one tile's `CB`, not a static slice of the row. Whether its register footprint
behaves is the open question, and it is the first thing to measure.

**Revised order (this becomes the new stage 3, ahead of the tile-shaped phases below):** (a) fuse the four physics phases over a tile, on today's
row-at-a-time schedule, with `RB=1, CB=`row — byte-identical, and it should already be
faster because it deletes launches; (b) then raise `RB` and add the bulk. Step (a) is
independently useful and independently verifiable, which is the property every stage in
this document is supposed to have and the per-row tail did not.

**Run `RNA_MD3_LAUNCH_PROBE` on the A100 before committing to any of this.** Every number
in the table above scales with one laptop-measured constant, and the A100's per-launch
cost is the single input that decides between "+14 s" and "break even".

## 3.8 Measured on the A100 — and §3.7's rejection is OVERTURNED

Run 2026-09-29, commit `9f7be2ca`. **A launch costs 2.65 µs on an A100, not 10.5 µs.**
`RNA_MD3_LAUNCH_PROBE` at 6 × 3000 nt, two reps, one sha throughout:

| k | mean wall | extra launches | µs/launch |
|---|---|---|---|
| 0 | 1.17 s | — | — |
| 32 | 1.42 s | 95 872 | **2.61** |
| 128 | 2.19 s | 383 488 | **2.68** |

Linear to 3 %. The laptop's 10.5 µs was 4× too high **for the device that matters**, and
§3.7 rejected the per-row tail on it. Re-priced:

| CB | multiple | extra launches | cost @2.65 µs | md after | **net** | (net @4 µs) |
|---|---|---|---|---|---|---|
| 128 | 21.9× | 8.41e6 | 22.3 s | 9.0 s | **+2.5 s** | −8.9 s |
| 256 | 10.9× | 4.01e6 | 10.6 s | 10.5 s | **+12.6 s** | +7.2 s |
| **512** | 5.5× | 1.80e6 | 4.8 s | 13.1 s | **+15.9 s** | +13.5 s |
| 1024 | 2.7× | 6.99e5 | 1.9 s | 16.7 s | **+15.2 s** | +14.3 s |

**So the per-row tail is viable after all** — positive at every `CB`, best near 512 at
about **+15.9 s of a 72.21 s wall, 1.28×**. §3.7's "the per-row tail LOSES" holds only at
`CB ≤ 128` and only on laptop numbers. The fused tail is still the better structure, but
it is no longer a precondition: the simple schedule pays, which makes it the right thing
to build first.

**The lesson to keep is not "the laptop was wrong".** It is that the rejection rested on
one constant measured on the wrong device, and the probe that produced it costs three
minutes. Any future argument of the form "this schedule multiplies launches, so it
cannot pay" is void until `RNA_MD3_LAUNCH_PROBE` has run on the target.

### And step (a) — the row fusion — WINS at production

The 2×2, 400 × 5601, two reps, **one sha across all four arms**:

| arm | wall | vs default | gpu_total |
|---|---|---|---|
| `ov1_fuse0` (today's default) | 72.21 | — | 59.28 |
| `ov0_fuse0` (single-stream control) | 73.16 | +1.3 % | 60.33 |
| **`ov0_fuse1` (the fusion)** | **71.25** | **−1.3 %** | **58.64** |
| `ov1_fuse1` | 71.82 | −0.5 % | 58.99 (REFUSED, ≈ `ov1_fuse0`) |

Stream overlap level 1 is worth **1.3 %** here (not the −0.4 % on record), and the fusion
beats the default **while giving that overlap up** — so its own contribution is nearer
2.6 % of GPU-side work. The pre-registered criterion in §F of the run (*"if `ov0_fuse1`
beats `ov1_fuse0`, integrate it with the stream protocol and make it the default"*) is
**met**, so that integration is now justified by measurement rather than by a laptop.

**One number in that table is an artefact and must not be read as a regression:** md
shows **51.75 s** in the fused arm against 33.6 s elsewhere, while `gpu_total` went
*down* (58.64 against 59.28) and the wall went down. The phase timers do not synchronise
unless `RNA_PHASE_SYNC` is set, so they attribute rather than measure; collapsing three
launches into one moved where the queue drains, and md's timer absorbed it. This is
exactly `feedback_phase_timers_charge_the_drain`, and `gpu_total` is the number to read.

### md's intensity, confirmed on the A100 with the denominator fixed

`init_gpu()` reports **47 records per launch (9 chunks)**, and at f = 0.50 that gives
**2.245 bytes per B-element and 0.89 ops/byte**, against a machine balance near 13. The
same data divided by all 400 records — the previous notebook's bug — reads **0.264 bytes**
and would again have said the thesis was refuted. The correction is now validated on the
device it was wrong about.

> **2026-10-03: the driver is planned in `PORT_MD_BLOCKING_DRIVER.md`** — the schedule
> with a read-by-read readiness table, a right-looking bulk update, the buffer audit,
> v1's refusals, stages 3a–3e with bars, and the decisions awaiting sign-off. Stage 2's
> negative control (`RNA_MD_BLOCK_SELFTEST=3`) has now run and bites.

### Stage 3 — tile-shaped `int_loop` / `hp_mb` / `new_c` / fML scan

The tail's four phases become tile-shaped. This is the largest stage and the one that
actually removes the per-row launch count. `int_loop` gains its own lever here for free:
its 30×30 `my_c` window is shared by neighbouring `j`, so a tile of `CB` columns reads
`(30+CB)×30` instead of `CB×900` — **15.5× fewer loads at `CB = 32`**, in 7.3 KB of
shared, aimed at a kernel whose dominant stall is `long_scoreboard`.

**Bar:** byte-identical, and md + int_loop time down. This is where the ~2–2.5× end-to-end
should appear.

### Stage 5 — tuning, and the Hopper arm

`CB`, `RB`, `RM × RN` and the shared budget are all machine-dependent; `b = 64` with a
4×4 register tile won on an RTX 3050 and an A100 may differ. And on `sm_90+`, DPX
(`__viaddmin_s16x2`) is a native fused add-min on packed int16 — exactly this primitive —
so a `#if __CUDA_ARCH__ >= 900` arm belongs here. Packed int16 is a measured **null on
Ampere** (emulated: 1.9× slower, 2.03× more instructions), so it must stay behind the
arch test.

## 5. What could go wrong, written down now

- **The tail is bigger than the estimate.** The bulk is `(d-1)/d` of the work *per tile*,
  but tiles near the diagonal have small `d`, and the diagonal band is where `int_loop`'s
  MAXLOOP window lives. If the near-diagonal tiles dominate, blocking helps less than
  the isolated prototype suggests. **Measurable before Stage 2**: sum the work over tiles
  weighted by `1/d`.
- **`fML` needs a row-major mirror.** The prototype reads `fML[i][k]` row-major because
  the triangle makes that strided. Production keeps one row buffer; a tile needs `RB`
  rows. That is `RB × n` int16 per record — small, but it is a new buffer and a new
  thing to keep coherent.
- **int16 baselines are per-64 along a column.** The decode in `md_cell.inc` assumes the
  reduction walks *down a column*, so the baseline slot advances with `y`. A blocked
  stage reads a `CB × CB` patch, and the baseline arithmetic has to survive that. This is
  the single most likely place for a silent wrong answer, and it is why Stage 1 exists.
- **Occupancy.** The prototype's 4×4 register tile used 56 registers and capped occupancy
  at 67 %. md today gets 33.8 %, so this is not a regression, but shared memory plus
  registers plus `CB` interact and the plan must not assume the prototype's numbers
  transfer.
- **Continuous flow and slot flow both index by per-record row.** `i_H[H]` is per record,
  so with `RB > 1` a record's *block* of rows must be tracked instead. Records at
  different phases is the normal case, so this is a real interaction, not a corner.

## 6. Why this is worth a week when the levers on this branch were worth percent

md is 55 % of GPU time and its loop runs at **1.0 ops/byte against a machine balance of
~13**. Everything tried so far — int16, warp-per-cell, block sizes, the band, the corner,
shared staging, the L2 window, pruning, continuous flow, slot flow, residency, the
megakernel — left that number unchanged, which is why the limiter never moved. Blocking
is the only change on the table that alters it.

The honest ceiling is **~1.90×** on GPU time from md alone (at the corrected 7.66×) and **~2.5×** with int_loop's
lever, capped by Amdahl rather than by the kernel. Not 6×. But it is larger than every
scheduling result on this branch put together, and it is the only one that moves the
roofline instead of the schedule.
