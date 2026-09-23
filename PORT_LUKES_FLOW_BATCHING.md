# Luke's Flow Batching

*A plan, written 2026-09-16 against `3bf72c46`. STARTED 2026-09-17 -- see section 8
for what has landed and the order Luke set.*

Three changes, in one plan because they share a thesis — **the device should be
folding, not waiting for or shipping bytes** — and because the order they land
in matters more than any one of them.

| | what | target | evidence it is there |
|---|---|---|---|
| **T1** | **column tiling in `modular_decomposition`** | **md 37.8 → 12–15 s** | 82.8 % of DRAM peak, ~47 TB of traffic, 4 B per lane-iteration (§34.1, §35) |
| **T2** | **row-granular stream-out** | `fetch_mx` + `backtrack` 12.9 s → ≤ 2 s exposed | pageable 6.5 GB/s against ~25 pinned; 0.77 GB/s average demand = 0.05 % of DRAM |
| **T3** | **continuous flow, for retirement** | chunk count ↓ on ragged input | C4 measured up to **3× fewer chunks**; a chunk transition costs 1.99 s |

**Combined, honestly: 85.9 s → ~55 s at 400 × 5601, and T1 is most of it.**
T2 and T3 are worth ~14 % between them and are much cheaper to build.

---

## 1. The logic, written down so it can be argued with

### 1.1 How far back a row looks — the two kernels differ completely

| | lookback | why |
|---|---|---|
| `c` / `int_loop` | **≤ 30 rows** | the interior loop admits `(p,q)` only when `(p−i−1)+(j−q−1) ≤ MAXLOOP = 30`, so `p ≤ i+31` |
| `fML` / `modular_decomposition` | **the whole column** | `min over y of (fML[i][y] + fML[y+1][j])` walks column `j` from row `i` down to `j` |

**The cache measurements already say this.** §34.2: `int_loop`'s L1 *rises*
72.9 → 78.8 % and L2 stays flat near 82 % from 600 to 8000 nt — a bounded
working set that does not care about length. `modular_decomp`'s L2 collapses
**42.6 → 15.2 %**, because its working set *is* the column and the column grows
with the record.

It is **not** a layout problem: `Indx(i,j) = j(j−1)/2 + i`, so consecutive `i` at
fixed `j` are contiguous and the column walk is already coalesced (1.74–3.46
sectors per request). It is a **volume** problem.

### 1.2 The volume, and what tiling does to it

`md` performs **1.17 × 10¹³ lane-iterations** (`N·L³/6`), each consuming one
fresh 4-byte column element: **~47 TB**. At 82.8 % of 1 555 GB/s over 37.8 s,
that is exactly what was measured — the model and the meter agree.

Tile the *i*-loop into blocks of **B** rows and keep column `j`'s live segment in
shared memory or registers for the block. **Each row extends the column by
exactly one element**, so B rows cost `column + B` loads instead of
`B × column`:

| B | md DRAM demand | what would bind instead |
|---|---|---|
| 1 (today) | **82.8 %** | bandwidth |
| 4 | ~21 % | latency |
| 8 | ~10 % | latency |

### 1.3 What tiling costs

- **Shared memory.** A full column at 5601 nt is 22 KB against 164 KB per SM, so
  a tile holds few cells and occupancy falls. **That is the correct trade only
  because DRAM is what binds** — and §35.1 is the warning attached: occupancy
  was *not* what bound `int_loop`, and the same assumption must not be imported
  here without measuring.
- **A skewed dependency inside the tile.** Cell `(i−1,j)` needs `(i,j)`, so the
  rows of a tile cannot run in parallel; they need a wavefront order within the
  block.
  **CORRECTED 2026-09-17 — this understates it, and the understatement is fatal
  to T1 as written.** `md(i−1)` reads `fml_i` = row `i−1` of fML, which exists
  only after `md(i) → new_c(i−1) → load_my_c(i−1) → fml_scan(i−1) →
  load_fML(i−1)`, with `int_loop(i−1)` and `hp_mb_3p(i−1)` feeding `new_c`. **The
  whole six-kernel row chain sits between two md rows**, so no md-only tile can
  span two of them. Column reuse across rows requires fusing the row chain —
  see `PORT_MEGAKERNEL_SCOPE.md`. The cheap half of the same win, per-record L2
  residency at int16 (a 31.4 MB triangle against 40 MB of L2), needs no fusion
  and should be measured first.
- **Lane-striding must survive.** §33.1: `RNA_MD_TILE=1` cost **247 %** and took
  sectors/request 2.04 → 10.99. Any tiling that breaks 32 lanes striding one row
  together will lose more than it gains.

### 1.4 Why the product can leave during the fold

`c[i][*]` is written **once**, by `load_my_c` at row `i`, and never rewritten;
later rows only read it. Same for `fML` row `i`. So **every row is final the
instant it is computed** while remaining resident for later rows.

At 400 × 5601 that is ~4.5 MB per row, so a **~10 MB pinned double buffer** on a
copy stream streams the whole 50 GB out under the sweep. Transfers execute on
**copy engines, not SMs** — they consume PCIe and DRAM read bandwidth, and
0.77 GB/s average is **0.05 % of peak**. The exit path is expensive today
because it is *serialised after the sweep and pageable*, not because it is
heavy.

### 1.5 Why retirement needs continuous flow

With every record marching the same `i`, a short record is idle at the **start**
and all records finish together at row 1 — **no matrix completes early, so there
is nothing to retire**. Only per-record rows change that, which is what
continuous flow's phases B/C were.

**And the prize must be sized honestly:** at 400 × 5601 the VRAM budget already
picks **2 chunks**, and a transition costs 1.99 s, so slot reuse can win ~2 % of
wall there. Its case is ragged, many-chunk input — H5's 1 500 mixed records took
3 chunks; a 2048 MB budget took 27.

---

## 2. Order of operations, and why this order

**The rule: every step must be measurable on its own, and no step may block the
next.**

### Step 0 — instrument the claim before building against it
Add per-kernel DRAM traffic (`dram__bytes_read.sum` is already in the §B metric
set) and a bytes-per-lane-iteration figure to the Scaling notebook, at the
**production** shape. T1's entire case is "47 TB, 4 B per iteration"; if that
number is not reproducible at scale, the plan changes before a line is written.

### Step 1 — T2, the stream-out *(independent of everything else)*
Pinned double buffer, copy stream, row-granular D2H issued after `load_my_c(i)`.
First because it is the cheapest, has no shared device state with the compute
path, and **de-risks the exit path before the kernel rewrite makes the sweep
shorter** — a 12.9 s tail matters more, not less, once the sweep is 25 % faster.

### Step 2 — T1, the column tiling *(the big one)*
Only after Step 0 has confirmed the traffic model and Step 1 has taken the exit
path off the critical path. Build it behind `RNA_MD_ROWTILE=B`, default 1, so
today's kernel is the control and the A/B is in-process.

### Step 3 — re-test int16 *(cheap, and it is a check on Step 2)*
int16 halves the column: the **same axis** as tiling. Today it measures +0.0 %
end-to-end on an A100 (§23) and 6.8 % *slower* on a small fixture (§33.1) — the
second because bandwidth was free there. After T1, if the tiling worked, int16's
value should fall towards zero *for the same reason*. **That prediction is the
test**: if int16 still pays after tiling, the tiling did not remove the
bandwidth bound.

### Step 4 — T3, retirement and slot reuse
Ragged workloads only, and measured on chunk count first and wall second.
Continuous flow's phases A–C4 already exist on their branch; this is a rebase
and a re-measure, not a new build.

### Step 5 — re-derive the roofline
§34.1's lesson is that the roofline position is a property of the grid and the
workload, not the kernel. After T1 and T2, every share in
`PORT_ROOFLINE_SCOPE.md` moves and the ranking has to be recomputed rather than
assumed.

---

## 3. Tools already in hand

**Do not rebuild these.**

| tool | what it gives this plan |
|---|---|
| `RNA_STREAM_OVERLAP` (`device.cu`) | streams, events, the parity-gate pattern, and the teardown discipline. **T2 rides on this machinery** rather than adding its own |
| `rnafold_pinned_alloc/free` (`stub2.h`) | the pinned host buffers T2 needs, with the existing fallback when pinning fails |
| the build pipeline (`RNA_BUILD_PIPELINE`, AUTO) | the **template** for T2: a host-side stage overlapped with a GPU phase, with a memory gate and a measured estimator |
| `rnafold_compound_bytes()` + `projected_chunk_bytes()` | the VRAM/RSS accounting any new buffer must be added to |
| `Continuous_Flow_Batching` A–C4 | schedule, slot reuse, `projected_chunk_bytes()` from the schedule. **T3 is a rebase** |
| `RNA_MD_TILE`, `RNA_MD_BLOCK_SIZE` | the existing knobs on the kernel T1 rewrites; `RNA_MD_TILE=1` is the coalescing control |
| `RNA_HC_VERIFY`, `RNA_ROW_VERIFY` | word-for-word mask and row checks — but read §32's lesson first: **a verifier means nothing until the thing it verifies exists** |
| `RNA_PHASE_SYNC`, `RNA_LAUNCH_STATS` | true per-phase GPU time (costs 0.2 % of wall, §32.1) and per-launch device time |
| `CUDA_RNAFold_Scaling.ipynb` §A–§G | the runner, the ncu probe with limiters/waves/lanes, ABBA ordering, and the sha bar that caught the level-2 race |
| `verify_option_parity.sh` (45), `verify_constraint_parity.sh` (5 shapes), `probe_declined_options.sh` | the regression bars every step must pass unchanged |

---

## 4. Targets, stated so they can fail

| | measure | today | target | falsified if |
|---|---|---|---|---|
| T1 | `md` phase | 37.84 s | **12–15 s** | `md`'s DRAM % does not fall as B rises — then the traffic model is wrong |
| T1 | `md` DRAM | 82.8 % | < 25 % at B=4 | as above |
| T2 | `fetch_mx` exposed | 7.68 s | **≤ 1 s** | wall does not move when `fetch_mx` → 0, i.e. the exit path was not on the critical path |
| T2 | D2H rate | 6.5 GB/s | ~20 GB/s | pinning does not change the rate — then the bottleneck was never the staging copy |
| T3 | chunks, ragged | 3 (H5) | 1–2 | chunk count does not fall — slot reuse is not reclaiming |
| all | **sha** | one value | **one value** | **any arm moves. Stop.** |

**Wall, combined: 85.9 s → ~55 s.** Stated as arithmetic from the three rows
above, not as a promise.

---

## 5. What must not break, from things that already bit us

1. **Lane-striding** — 247 % if broken (§33.1).
2. **Host run-ahead is not bounded by a device-side wait.** The level-2 race
   (§35.2) came from exactly this: the host queued `hp_mb(i−2)` while
   `fml_scan(i)` still read its buffer. Any new stream needs a gate *per reused
   buffer*, not per row.
3. **Graph capture wants stable pointers.** A buffer whose address changes per
   row forces a re-instantiate per row.
4. **Register counts move with the source**, not just the toolkit — `up_int`
   cost 5 (§35.1). Any occupancy arithmetic must be re-derived after a kernel
   change, not carried forward.
5. **VRAM accounting** — every new device buffer goes into the budget model, or
   the chunker's estimate silently drifts from the truth.
6. **A verifier proves nothing about a path it does not run** (§32, `RNA_HC_VERIFY`
   comparing two unconstrained matrices).

---

## 6. Explicitly out of scope

- **`__launch_bounds__` / occupancy work on `int_loop`** — refuted in §35.1:
  raising the ceiling *lowered* achieved occupancy, and the kernel is
  retirement-limited, not occupancy-limited.
- **Cross-cell compaction to fill lanes** — §34.4 measured warps 83–94 % full.
  (Binning cells by *width* to decouple **retirement** is a different idea and
  stays live, but it belongs to `int_loop`, not to this plan.)
- **Device-side backtracking** — it would make T2 almost unnecessary by shrinking
  the product from 126 MB to a few hundred bytes per record. It is a much larger
  change than anything here and is noted as the natural successor, not a step.

---

## 7. Shelf note

Parked deliberately. The feature-integration list closes first, so that this
starts from a tree with no open correctness questions — which is also the only
state in which a byte-identity bar means anything.

---

## 8. Started 2026-09-17: what the first A100 run changed, and the new order

**Luke's summary of what landed:** *pinned memory buffers and multi-worker fetch
to create a faster exit path D2H. Fetch and backtrack 60 % faster. Row tables
built once before sweep.*

### 8.1 Step 0 reproduced; level 2 did not survive its fix

`CUDA_RNAFold_Scaling.ipynb` at `225fff11` (A100-SXM4-40GB, 1410 MHz):

- **Step 0 holds.** `modular_decomp` at 200 × 5601: **82.79 % of DRAM peak**,
  77.24 waves/SM, 30.1 of 32 lanes per instruction. T1's traffic case stands.
- **Stream overlap §G:** level 1 is **−1.0 %, sha identical**. Level 2 is −1.7 %
  and returned **two different wrong shas in two runs** (`f0740997`,
  `221bce6f` against `49ad5c81`), with the parity-event fix (`3bf72c46`) built
  in. The default is 0, so nothing shipped is affected.

**The cause, read from the code:** level 2 removed the per-row md-stream sync,
which was the only thing bounding the host. `upload_size_off_H`/`upload_i_H`
then overwrote shared device tables with **blocking copies on the default
stream**, and a blocking copy waits for the default stream only — not for the
cell/hp/md kernels already queued to read those tables. `upload_md_tables`
likewise rewrote a pinned parity shadow that an earlier async copy could still
be reading. And `gq_row_kernel` filled `fml_scan`'s input on the cell stream,
behind an event recorded before it was issued.

### 8.2 T2 as written has nowhere to put the rows

Backtracking cannot start before row 1 in lock-step, and it needs random access
to the whole `c` and `fML` triangles. The host holds only one scratch pair per
backtrack worker today (~1.5 GB). Streaming rows out during the sweep needs
**every** record's triangles host-side at once — ~50 GB at 400 × 5601, 25 GB for
`c` alone — to move ~7 s of copy under the sweep. So T2 splits:

- **T2a, the exit path itself.** The workers each issue blocking pageable copies
  on the default stream, so they serialise at 6.5 GB/s. Per-worker pinned
  scratch plus a per-worker copy stream parallelises them at the pinned rate for
  ~1.5 GB of pinned host memory and no RSS growth.
- **T2b, `c` streaming with a 32-row ring.** On the device only `int_loop` reads
  `c`, and never more than 31 rows back (MAXLOOP). **The point is cache
  occupancy, not VRAM** (Luke): keep those 32 rows in the fastest memory the SMs
  have, and let `c` leave the device as each row completes, which also removes
  that traffic from the end of the task.

### 8.3 The order, set by Luke on 2026-09-17

1. **Fix 3 — the chunk's row tables** (below). Removes every per-row table
   upload and, with it, the class of race that broke level 2.
2. **T2a** — the exit path.
3. **T2b** — `c` in a 32-row ring, streamed out.
4. **Flagged for later:** device-side backtracking.

**The architecture this is heading toward** (Luke): each *sequence* runs as its
own kernel launch, with intra-row parallelism inside it and `c` streaming. It is
scoped as part of T2b, where the per-record ring makes the per-record launch
natural.

### 8.4 Fix 3 — the chunk's row tables

`size_off_H`, `side_off_H` and `i_H` get **one slot per sweep row**, indexed by
the iteration `i`, in `device.cu` (`rnafold_rowtab_*`). A slot is written once
per chunk.

| path | how slots are filled | per-row table traffic |
|---|---|---|
| lock-step (default) | all rows on the host, **one pinned upload before row 1** | **none** |
| continuous flow / schedule | each slot as its row is built (slots turn over mid-sweep) | one small blocking copy into a **fresh** slot |

- **Kernels are unchanged.** Each file binds its table pointers to row `i`'s slot
  where it used to upload (`bind_row_tables(i)`). The md graph now captures
  kernels only — the two H2Ds it used to capture from host stack tables are gone.
  The pointers move every row; `graph_forced_reinstantiate_count` stays at the one
  expected boundary, so the update path absorbs it.
- **The host reads the same bytes the device does**: `fill_arrays_loop.c` takes
  `size_off_H`/`side_off_H` from the slot, so the two cannot drift.
- `gq_row_kernel` moved to the md stream, where its only reader lives.
- Budgeted in `hp_mb_loop_bytes_per_file()`: `(L+1)·(3·8+4)` bytes per record,
  ~160 KB at 5601 nt. ~22 MB for the whole table at 200 × 5601.
- Removed: three files' per-row upload functions and content shadows, level 2's
  private md tables and their pinned parity shadows.

**Memory tiers, which is the question this plan asks of every change:** the
tables move from 16 803 small blocking H2Ds (host run-ahead stalled at each) to
one pinned PCIe burst per chunk into GDDR. The kernels read them from GDDR as
before; nothing about SM-side traffic changes.

**Bars:** see 8.5.

### 8.5 Fix 3's bars, and a second defect the new bar found

Local (RTX 3050, 1057/2100 MHz), control = the same branch at `aa9fce39` built
from a clean clone:

| bar | result |
|---|---|
| `verify_option_parity.sh` on `asc.fa` | **45/45 identical** (GPU arms on the GPU, declined arms on the CPU) |
| `verify_constraint_parity.sh --expect-accelerated` | **5/5 shapes** |
| overlap bar (`ov_bar`, 9 cases × 9 arms) | **81/81** match the control's level-0 sha |
| md graph | 1 forced re-instantiate per chunk, as before: moving table pointers cost nothing |

The overlap bar's cases: `C_mixed` with default, `--noLP`, `-g`, `-c`, `-d0`;
`desc`/`asc`/`C_mixed` capped to **2–4 chunks**. Its arms: levels 0, 1, 2 (×3),
graph-off at levels 1 and 2, continuous flow at levels 0 and 2.

**The second defect: `RNA_CUDA_GRAPH=0` under any overlap level was wrong — on
the control binary too, with a different wrong sha each run.** The graph-off
path issued the md chain onto `graph_stream`, which is ordered against the
default stream only, while the row around it runs on the cell/md streams. The
graph path never had this because it *launches* on the md stream. Fixed: the
graph-off path now issues on the md stream when overlap is on (`ISSUE_STREAM`).
A diagnostic path, but the only way to A/B the graph, so it has to be right.

**What the local bars cannot show:** the control's *graph-on* level 2 never
raced here — not on `C_mixed`, not in three runs of `D_long` (30 × 3000–5900).
It fired only at 400 × 5601. So locally the fix rests on the construction (no
slot is ever rewritten) and on the graph-off race, which did fire and is gone.
**The claim is settled only by Scaling §G on an A100**, which now builds this
branch (`tools/make_nb_scaling.py`, `BRANCH`).

### 8.6 T2a — the exit path, one copy stream per worker

**Before:** after the sweep, each backtrack worker copied its record's `c` and
`fML` triangles with a **blocking pageable `cudaMemcpy` on the default stream**.
Twelve workers, one stream, one driver staging buffer: they queued behind each
other, ~6.5 GB/s at 400 × 5601, 7.2 s of wall.

**After:**

| system | before | after |
|---|---|---|
| host threads | 12 workers, serialised on one stream | 12 workers, **concurrent** |
| device streams | default stream only | **one copy stream per worker** (`rnafold_xfer_*`, device.cu) |
| pinned host memory | none (driver-staged) | **one 8 MB stage per worker** (`RNA_XFER_STAGE_MB`), pinned once per process |
| PCIe | one pageable copy at a time | concurrent pinned slices, up to the card's copy engines |
| host DRAM | one calloc'd scratch pair per worker, **per chunk** | same pair, malloc'd, **kept across chunks**; each worker memcpys its own slices |

**What did not work first, and why the stage exists.** The first version
pinned each worker's whole scratch pair. The fetch fell ~10×, but pinning 1.7 GB
cost **35 worker-seconds (~3 s of wall)** under WSL, more than the copy it saved
on a one-chunk fold, and it scales with length × workers. A fixed stage makes
the pin cost `workers × 8 MB`, once.

**Measured locally** (RTX 3050 / WSL, `D_long` 30 × 3000–5900, ABBA ×2; the GPU
clock wandered 210–712 MHz, so walls are **not** usable, but this phase is host
and PCIe work):

| | fetch_mx + backtrack, s (four runs) | mean |
|---|---|---|
| fix 3 (A) | 2.42, 3.33, 3.77, 2.64 | **3.04** |
| T2a (B) | 1.21, 1.14, 1.42, 1.06 | **1.21 (−60 %)** |

One sha across all eight runs. Peak RSS **+0.3 GB** (the stages plus a pool that
is no longer freed between chunks). Stage size, one run each: 4 MB 1.10 s,
16 MB 1.66 s, 64 MB 2.62 s, 256 MB 9.49 s — pinning dominates on this host, so
**re-sweep 4/8/32 on the A100** before trusting the default.

**Also fixed:** under continuous flow, a retiring record's fetch now drains the
row streams first (`on_retire_cb`). At `RNA_STREAM_OVERLAP` ≥ 1 nothing at the end
of an iteration waits for them, and the fetch runs on its own stream.

**Bars:** 13-arm A/B against the control (`aa9fce39`) — default, backtrack threads
0 and 3, `-c`, int16, continuous flow, slot flow with and without level 2, level
2, `--noLP -p`, 3 chunks, 3 chunks + int16, `F_extreme` — **13/13 identical**.

**Target, from section 4:** `fetch_mx` exposed ≤ 1 s at 400 × 5601 (from 7.68 s).
Falsified if the wall does not move when `fetch_mx` falls.

### 8.7 T2b is scoped: `PORT_T2B_SCOPE.md`

Written 2026-09-17, nothing built. The number it turns on: `int_loop` is the only
device reader of `c`, and it never looks further back than MAXLOOP, so **the live
working set is 31 × L per record — 0.7 MB at 5601 nt against a 62.8 MB
triangle, 1.1 %.** Three separable pieces: the 32-row ring (device bytes per
record 172 → ~110 MB, and the live set becomes small *and contiguous*), streaming
`c` out as each row is written (host destination confirmed: backtracking indexes
`fc->matrices->c` randomly, so 25 GB host-side behind a memory gate until
device-side backtracking lands), and one launch per sequence — which needs a
graph, because it takes 33.6 k launches to 13.4 M. Two probes decide it before a
line is written: does the layout flip keep lane-striding, and what does a launch
actually cost at production and at tail widths.

**Revised the same day (Luke): make the window ~100 rows, not 32** — 32 behind
plus ~68 ahead, filled row after row without leaving the window. It buys more
than the extra 1.5 MB per record costs: a 32-row ring must retire a row about
every row (2.24 M copies of 22 KB per chunk, ~11 s of issue overhead to save a
7 s exit path), while a 100-row window flushes 68 rows at a time (~33 k copies
of 1.5 MB, ~0.2 s). It also makes residency a parameter — 100 rows is 160 KB at
400 nt (one SM's shared memory) and 2.24 MB at 5601 (L2 holds ~17 records'
worth), which is why the records in flight have to be a **subset**, tying the
window to per-sequence launches. What it does not do is cut launches: the row
chain is serial, so that needs a persistent megakernel, flagged and not
scheduled. And none of it helps `fML`, whose column walk is T1's problem.

### 8.8 The megakernel is scoped: `PORT_MEGAKERNEL_SCOPE.md`

Asked for by Luke while scoping the per-record launches: with one kernel per
sequence, can the fML **columns** be cached too? Scoping it turned up the
correction in §1.3 — **T1's row tiling is not legal inside `modular_decomposition`
alone**, because the whole row chain sits between two md rows. So the megakernel
is not a launch-overhead play (that is worth ~2 s); it is the only structure in
which T1's column reuse can happen at all.

Three tiers answer "cache the columns", and they are not equally expensive:
**(a)** per-record **L2 residency** — an int16 triangle is 31.4 MB against 40 MB
of L2, with ≤ 30 MB pinnable as persisting, so md's re-reads come from L2 at
~4–5 TB/s instead of DRAM at 1.55. **This needs no megakernel**, only the record
subsets T2b-3 already implies, and should be measured first. **(b)** on-chip
column state inside a fused row chain: 17.7 MB of shared memory across 108 SMs is
~56 % of that triangle, so md's DRAM demand roughly halves. **(c)** both, plus
int16 — which **inverts** the plan's Step 3 prediction: under residency int16 is
what makes the triangle *fit*, so its value rises rather than falling.

Costs that are not negotiable: one block size for every phase, registers set by
the worst phase, ~39 k grid barriers per chunk (**probe that first**), and the
loss of per-phase timers, ncu attribution, `RNA_ROW_VERIFY` and mid-sweep
retirement — each of which has to be re-provided or deliberately given up. A
multi-second kernel also cannot run under WDDM, so this one is Colab-only.

### 8.9 The A100 run that settles fix 3 and T2a (2026-09-17, `1d5eb219`)

**§G: the level-2 race is gone.** One sha across all six arms
(`49ad5c81b5fa`), where the same section at `225fff11` returned **three**
(`49ad5c81`, `f0740997`, `221bce6f`). The per-chunk row tables removed the class.

| level | wall a | wall b | vs level 0 | sha |
|---|---|---|---|---|
| 0 | 87.41 | 87.15 | — | one |
| 1 | 86.96 | 86.90 | **−0.4 %** | same |
| 2 | 86.68 | 86.52 | **−0.8 %** | same |

Level 2 is correct and worth −0.8 % against a scoped ~21 %: `modular_decomp` is
at 82.8 % of DRAM peak, so the two chains compete for the memory system the
moment they overlap. The co-tenancy argument was half right, and now it is
measured rather than argued.

**The exit path, phase-synced (`F_c1`, the arm where a phase timer means what it
says):**

| | before (`225fff11`) | after (`1d5eb219`) | delta |
|---|---|---|---|
| `fetch_mx` | 7.24 | **1.85** | **−5.39** |
| `backtrack` | 5.70 | 7.63 | +1.93 |
| `int_loop` | 17.43 | 17.11 | −0.33 |
| `hp_mb` | 1.89 | 1.66 | −0.24 |
| `modular_decomp` | 37.83 | 37.55 | −0.28 |
| **wall** | **91.82** | **87.36** | **−4.46 (−4.9 %)** |

So **T2a is −3.46 s net** (the exit path went 12.94 → 9.48 s) and **fix 3 is
−0.85 s** across the three kernels whose rows no longer wait on a blocking
upload. They sum to −4.31 against a measured −4.46.

**The wall moved with `fetch_mx`, which was the falsification test in §4** — had
it not, the exit path would have been off the critical path and T2a worthless.

**The serial control says the win is CONCURRENCY, not pinning.**
`RNA_BACKTRACK_THREADS=0`: `fetch_mx` **13.10 s**, `backtrack` 54.43, wall
144.87. One worker through one stage is worse than twelve workers through one
pageable stream; twelve through twelve stages is 1.85.

**Stage size barely matters, as predicted** (< 15 % was the written prediction,
18 % was measured, and not monotone — so it is mostly noise): 4 MB → 1.730,
8 MB → 1.964, 32 MB → 1.624. **Default 8 stands**; 32 MB also costs +0.3 GB RSS.

**What is left in the exit path, and it is now the obvious next cut.**
`backtrack` rose 1.93 s: that is the memcpy out of the stage, which the workers
now do themselves. On WSL, pinning the scratch outright cost 35 worker-seconds
and was refused for that reason — **but this host is not WSL**. Pinning 1.5 GB
here should cost ~0.2–0.5 s once, against the 1.9 s of memcpy it would delete.
That is a one-knob experiment (`RNA_XFER_STAGE_MB=0` meaning "pin the scratch")
and it is worth ~2 % of wall.


### 8.10 Defaults changed, and one design question answered with a null

**`RNA_STREAM_OVERLAP` now defaults to 1** (Luke, on §8.9's measurement: one sha
across six arms, level 1 −0.4 %). Level 0 is still the control every sha
comparison is made against; level 2 stays opt-in until the §H stress soak.

**The backtrack scratch is pinned when pinning is cheap — decided by
measurement, not by a constant.** `RNA_XFER_STAGE_MB` unset is now AUTO: page-lock
16 MB, time it, and pin the scratch outright if the host does it faster than
0.25 s/GB; otherwise keep the 8 MB stage. `=0` forces pinning, `=N` forces a
stage. The reason is that the two hosts disagree violently — the A100 wants
pinning (it would delete the +1.93 s of stage memcpy), while WSL measured the
exit path at **3.09 s pinned against 0.91 s staged**.

Two things the probe taught us on the way:

- **Page-locking does not scale across threads.** The single-threaded probe says
  0.83 s/GB; twelve workers pinning their own scratch cost **29 worker-seconds**
  for 1.7 GB. It serialises in the kernel's memory-map lock, so concurrency makes
  it worse. The AUTO threshold carries margin for exactly this.
- **So the pool is now allocated serially, up front**, on one thread, sized to
  the chunk's longest record: **29.2 → 2.1 worker-seconds**, a 14× cut, and the
  allocation leaves the backtrack phase's timing where it never belonged.

**The INF-sparsity idea is dead, measured** — `RNA_MD_INF_STATS`, see
`PORT_MEGAKERNEL_SCOPE.md` §2(d). 0.1 % of md's column stream is INF at 4 800 nt
and the fraction falls with length, so there is no compression lever. The probe
cost one build and four folds, which is what a cheap "no" should cost.

---

## 9. Phase 4b — retiring a record off the sweep thread (built 2026-09-23)

Until 4b, `on_retire_cb()` ran on the sweep's own thread: every slot handover
stopped the sweep for one record's **fetch and backtrack**, with a pool of
exactly one scratch. The post-sweep equivalents at 400 × 5601 are fetch 1.85 s
and backtrack 7.63 s, so the inline half was the expensive one.

**The split is set by one ordering constraint, not by taste.**
`refill_slot2()` overwrites exactly the retiring record's device cells later in
the same loop iteration, so the D2H must be ordered before it. Everything after
the copy — `RNA_CIRC_VERIFY`, fM1, `vrna_mfe_exterior_f5()`, the MFE read, the
backtrack — is **pure host work that makes no CUDA call at all**, which is why
it can move and the fetch cannot:

| half | where | why |
|---|---|---|
| `backtrack_fetch_slot()` | the sweep's thread | must precede `refill_slot2()`; already waits on one copy stream, not the device (`rnafold_d2h_w`) |
| `backtrack_finish_slot()` | a worker | touches no device memory; the sweep has no reason to wait |

**The bounded resource is the scratch pair, not the thread.** A worker holds one
record's two (or three) triangles until it is done, so the pool is W scratches
and W workers and the sweep BLOCKS when all W are out. That backpressure holds
peak host matrix memory at W × one record's worth — the bound `backtrack_all()`
has always had, under the same `RNA_BACKTRACK_THREADS`. `=0` gives W=1 and the
old serial path exactly, which is the control every sha is compared against.

**Two defects found by review rather than by a run:**

- the cleanup keyed on `started`, so a pool whose **first** `pthread_create()`
  failed leaked its whole queue and freelist. `r->q` is the allocation witness
  now, not the thread count;
- the bar's own worker-count parse, `s/.*\([0-9]\+\) retire workers.*/\1/`, has
  a greedy `.*` that eats the leading digit — "12 retire workers" read as **2**,
  and the `> 1` assertion passed for the wrong reason. A check quietly reading
  the wrong number is worse than no check.

### 9.1 The bar needs TWO fixtures, because neither reaches both properties

`tools/verify_retire_pool.sh`:

- **mixed** (30 records, 200–1360 nt) for **correctness**. Uniform lengths retire
  every slot on the same iteration, so a handover defect lands while every
  neighbour is itself starting fresh and is invisible — the --noLP slot-flow
  defect hid behind exactly that.
- **u2000** (24 records, one length) for **concurrency reachability**. The
  opposite property: every slot retires together, so records really are in
  flight at once. The mixed arm measures **peak 1 in flight** — it hands off and
  the worker finishes before the next handover arrives — so a green mixed run
  says *nothing* about whether two records were ever overlapped. u2000 measures
  peak 3 and the bar asserts > 1 there and nowhere else.

The counters (`peak N in flight, M stalls`) exist for that assertion, not for
tuning.

**Negative control, run by hand:** with the fetch moved into the worker — the
exact ordering violation above — **both** arms went DIFF against upstream. The
bar can reach its own subject.

**Local result (RTX 3050, WSL):** option parity 45/45, constraint parity 5/5
(`--expect-accelerated`), `verify_retire_pool` all green, byte-identical to
upstream and to the serial control at `RNA_SLOT_FLOW` ∈ {2,4}. **Unmeasured for
speed** — the laptop's handovers are microseconds and its clock wanders. §A of
`CUDA_RNAFold_Flow.ipynb` is the measurement.

### 9.2 `RNA_MD_BAND` is rounded up to a warp

K is the band's **per-column stride** (`band[H*stride + j*K + off]`,
`stride = (len+2)*K`), so it sets the alignment of every column's segment, not
just the width. At K=32 ints = 128 B each column starts on a transaction
boundary; at K=48 every second column straddles two 128 B sectors — an extra
sector per warp on a kernel already at 82.8 % of DRAM peak. That is a
**coalescing** cost, not a divergence one: the band is read uniformly across the
warp. Both the requested and the effective K are printed, because a sweep that
records the requested one silently samples the wrong points.

### 9.3 The axis no A100 run has swept: records in flight

Every A100 run so far has measured **intra-sequence** parallelism — rows in
flight for one batch — and the card has been 81 % idle on memory throughout
(940 MB at 24 × 2400, 7.6 GB at 384 × 2400). That is not what this branch is
for.

**L2 is device-wide, 40 MB, and cannot be partitioned per record.** Containment
makes a record's working set contiguous enough to be resident; it does not give
each record its own cache. So concurrency and residency pull against each other
and there is a **knee**:

| length | fML int32 | int16 | records in 40 MB (int16) |
|---|---|---|---|
| 1200 nt | 2.9 MB | 1.4 MB | ~27 |
| 2400 nt | 11.5 MB | 5.8 MB | **~7** |
| 5601 nt | 62.8 MB | 31.4 MB | **~1** |

`CUDA_RNAFold_Flow.ipynb` (`tools/make_nb_flow.py`) sweeps it. The knee must
**move with length** and **move with int16**, or whatever it is, it is not
residency — §C measures the L2 sector hit rate directly so the shape in §B is
attributed rather than assumed. **C gates B**: a knee without a mechanism is a
number to be explained, not a result to build on.

**And it is stated so it can fail:** monotone-down means the card is still
underfilled at that length and containment is not yet binding; flat means the
working set was never the constraint at this scale. Both are recorded here in
advance so neither can be rediscovered as a success.

### 9.4 What is NOT the mechanism (asked 2026-09-23)

- **L2 cannot be staged into.** It is transparent and physically indexed; there
  is no allocator and no DMA destination. The only control is
  `cudaAccessPolicyWindow` marking a range `Persisting` within a ~30 MB carve-out
  of the 40 — an **eviction-priority hint on a range you already touch**.
- **Copy engines cannot feed it.** Their destination is always a VRAM address. A
  D2D `cudaMemcpyAsync` would pull data through L2 but writes a second copy and
  evicts what was there — more traffic, on a kernel already at 82.8 % of peak.
- **What does exist:** `prefetch.global.L2 [ptr]` (inline PTX, issued by the SM,
  costs issue slots not registers) and — the real version of "a feed that does
  not take SM capacity" — **`cp.async`** (Ampere global→shared, bypasses the
  register file, `.cg` bypasses L1). It lands in **shared memory**, not L2.
  Hopper's TMA is the dedicated engine; sm_80 has no TMA.
- **"L1" is shared memory** (same SRAM since Volta; 192 KB/SM on A100, 164 KB
  addressable as shared). A record's triangle is 31.4 MB at int16 against
  164 KB per SM — it never fits, which is why the lever is a *window* (the band,
  T2b's row window) and not a staged copy of the matrix.
- **And the standing warning:** shared-memory staging for md measured **15 %
  slower**, because staging only pays where there is REUSE. Any new tier has to
  be justified by reuse, not by being closer to the SM.

### 9.5 `RNA_MK_CORNER_K` at 32·x — and the K-versus-W trade it exposed

Rounded up to a warp at the read site, so the shared-memory budget below it is
computed from the value the kernel will actually use. Rounding *after* the
budget would under-reserve.

**The degradation ladder now steps through warp multiples only.** It used to
halve while `K > 8`, landing on 16 and 8 — widths that start every other column
mid-line, which is precisely what rounding the request was meant to prevent. It
now goes 128 → 64 → 32 → **0**.

**And that exposed the real constraint, which is not K.** The corner needs
`K × W × 4` bytes, where `W` is columns per block. At 5601 nt the megakernel
takes W ≈ 452, so even K=32 wants 57 KB and the ladder runs to 0 — a
warp-aligned corner **does not fit at production width at all**. The old ladder
hid this by silently running K=16: a sweep of 32/64/96/128 would have reported
four distinct arms that were all the same kernel.

So **K and W trade off directly**, and a corner sweep is only meaningful with
`RNA_MK_SMEM_KB` raised or more blocks. `§F` of `CUDA_RNAFold_Flow.ipynb` parses
the *effective* K out of the on-chip line rather than the request, and says so
loudly when every arm collapses to one point.

Byte-identical after the change: per-phase, megakernel, and megakernel+corner
all match upstream on the mixed fixture.

### 9.6 Flagged, not tested — a fixed K is the wrong shape

Luke, 2026-09-23. The live set changes shape across the sweep: **at low k many
relevant columns each holding few elements; at high k few columns (the corner)
each long.** A constant K therefore over-allocates at one end and
under-allocates at the other for the same byte budget, and a K that varies with
position — wide and shallow early, narrow and deep late — fits the same bytes to
the live set as it actually is.

Not cheap to just try: K is the **stride** in both layouts, so a varying K turns
the addressing from a multiply into a prefix sum (an offset table per column,
affordable — `d_colb_off` is already exactly that — but a layout change, not a
knob).

Note this is *not* pre-empted by the ownership proof. That proof is about
partitioning COLUMNS among blocks and shows equal width is the optimal **fixed**
partition; it says nothing about varying the cache **depth** along the sweep.

**Order: establish that the window pays at all first** (§F: band against corner
at matched K, so the caching idea is separated from the ownership tax). Only if
the band improves with K is the variable-K refinement worth building.

### 9.7 Notebooks and results are local only

This tree is destined for upstream review and no notebook or result file exists
in ViennaRNA, nor will it. `.gitignore` now excludes `/CUDA_RNAFold_*.ipynb` and
root-level `/*.json` (root-anchored, so `doc/breathe/.../schema.json` is
untouched). Generate with `tools/make_nb_*.py` and upload to Colab by hand — the
notebook clones the branch for its **source**, so it never needs to be in it.

**Done 2026-09-23 (Luke): 51 files untracked** with `git rm --cached` -- ten
`CUDA_RNAFold_*.ipynb`, 27 root-level result `.json`, and the 14
`tools/make_nb_*.py` generators, on the principle that none of it exists in
ViennaRNA. Every one is still on disk; only the tracking is gone.

**Deliberately NOT removed:** `misc/rna_mod_*.json` (upstream's own modified-base
parameter files) and `doc/breathe/.../schema.json`. The `.gitignore` patterns are
root-anchored for exactly this reason, and the untrack list was filtered to
root-level paths before it ran.

Nothing in `Makefile.am` or `configure.ac` referenced any of it, so `make dist`
is unaffected -- checked, not assumed.

**They remain in git HISTORY.** The tree a reviewer clones is clean from here on,
but a genuinely clean history needs either a rewrite or a fresh branch built from
the source files alone. That is still open.
