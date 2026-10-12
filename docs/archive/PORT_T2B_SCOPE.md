# T2b — `c` in a ~100-row window, streamed out, and one launch per sequence

*Scope, written 2026-09-17 against `f9d6735d`, revised the same day with Luke's
100-row window. Nothing built. His framing: the problem is **cache occupancy,
not VRAM** — keep the rows the SMs are actually using in the fastest memory we
can, fill row after row inside that window without going to another tier, let
`c` leave the device once a row is more than 32 behind, and run each sequence as
its own kernel launch with intra-row parallelism inside it.*

---

## 1. What `int_loop` actually reads, from the code

`int_loop_kernel_body.inc`: one **block per cell (H, j)**, threads iterating the
cell's interior-loop candidates. For each candidate `Energy()` reads
`my_c[tri_off_H[H] + Indx(p,q)]` with

```
p = i+1+row,  q = q0+column,  row,column >= 0,  (p-i-1) + (j-q-1) <= MAXLOOP = 30
```

so one cell reads a **triangular window of at most 31 × 31** cells, anchored at
`(i+1, j-1)`. It is the only reader of `c` on the device.

| | cells | bytes (int32) |
|---|---|---|
| one cell's window | ≤ 496 (triangle of 31) | ≤ 2.0 KB |
| **one row of one record** (all `j`) | 31 × L | **~0.7 MB at L = 5601** |
| one record's whole triangle | (L+1)(L+2)/2 | **62.8 MB at L = 5601** |
| **a 200-record chunk's triangles** | — | **12.6 GB** |

**So the live working set is 1.1 % of what is allocated.** Everything below
follows from that one ratio.

`Indx(i,j) = j(j−1)/2 + i` is column-major, so a window is 31 short runs of 31
consecutive ints, one per column — already coalesced-ish, and reflected in the
measured hit rates (§34.2: `int_loop` L1 72.9 → 80.9 %, L2 flat ~82 % from 600
to 8000 nt; the MAXLOOP window is why length does not hurt it).

---

## 2. What changes, in three separable pieces

**They are separable, and the order matters: each is measurable alone.**

### T2b-1 — the window: `c` stops being a triangle on the device

**Luke, 2026-09-17: make the window ~100 rows, not 32** — 32 *behind* (what
`int_loop` reads) plus ~68 *ahead*, pre-allocated space for answers not yet
computed, so the sweep fills row after row inside the window and only touches
another tier when the window is full. The sweep runs `i` downward, so "behind"
is higher `i` (already computed) and "ahead" is lower `i`.

Indexed `win[(i mod W) * (L+1) + j]`, `W = 100`. `R = MAXLOOP + 2 = 32` is the
floor — the part correctness needs; everything above it is there to make the
flush and the residency better, and is a free parameter (`RNA_C_WINDOW`).

| | today | W = 32 | **W = 100** |
|---|---|---|---|
| device bytes per record (c) | 62.8 MB | 0.72 MB | **2.24 MB** (−96.4 %) |
| device bytes per record (total) | ~172 MB | ~110 MB | **~111 MB** |
| records per 34 GB chunk | 200 | ~310 | **~306** |
| what a row's reads walk | a 62.8 MB triangle | 0.7 MB | **2.24 MB** |
| D2H copies per record | 1 of 62.8 MB, after the sweep | 5 601 of 22 KB | **82 of 1.5 MB** |

**The copy count is the argument for the bigger window.** A 32-row ring has to
retire a row about every row: 5 601 copies per record, **2.24 M for a
400-record chunk**, each only 22 KB. At even 5 µs of issue overhead that is
~11 s of overhead to save a 7 s exit path — the ring alone makes T2b-2
marginal. Flushing 68 rows at a time is 1.5 MB per copy and **~33 k copies for
the chunk**, which is a size PCIe is happy with and an overhead of ~0.2 s.

**And residency becomes a parameter rather than an accident:**

| record length | 100 rows | fits |
|---|---|---|
| 400 nt | 160 KB | **one SM's shared memory** (164 KB) — the whole window |
| 1 800 nt | 720 KB | ~55 records' windows in 40 MB of L2 |
| 5 601 nt | 2.24 MB | **~17 records' windows in L2** |

That last row is the catch, and it is what ties this piece to T2b-3: with 200
records marching in lock-step their windows sum to **448 MB and L2 holds none of
it**. The window is only resident if **fewer records are in flight at once** —
which costs nothing, because one 5 601-nt record's row is up to 5 601 cells and
already fills 108 SMs by itself. ~16 records while rows are wide, more at the
tail, is the scheduler this implies.

Today the same reads are scattered through a 12.6 GB allocation and hit in
L1/L2 only because the *address* window is small. In the window they are small
**and contiguous**, which cuts TLB pressure and makes a per-block shared-memory
stage of the cell's own 31 × 31 (≤ 2 KB) trivially addressable — a second tier
under the window, if the probe says the inner loop wants one.

**Two things the window must not get wrong.** The ahead rows are **INF-filled
before use** — `load_my_c` asserts each cell is INF before writing, and that
assert is load-bearing. And a slot may not be refilled until its flush has
completed: **one event per flush, waited on by the host**, because a device-side
wait does not bound the host (the level-2 lesson, §8.1).

**The layout flip is the risk.** Today a window is column-major runs; in the
window, row-major means a cell read becomes 31 runs of 31 along `q`, which is the
*same shape*, so lane-striding is preserved — but this must be checked against
§33.1's warning (breaking lane-striding cost **247 %**) before anything else is
measured.

### T2b-2 — stream `c` out as it is written

`load_my_c(i)` writes row `i` and nothing on the device ever rewrites it. So
row `i` is final the instant that kernel completes, and can leave immediately.

- **Where it must land: host memory.** `bt_exterior_f5.c`, `bt_internal.c`,
  `bt_multibranch.c`, `subopt.c` and the `fM1` reconstruction all index
  `fc->matrices->c` randomly. Until backtracking moves onto the device (flagged
  for later), the host holds the triangle: **25 GB at 400 × 5601**, so this needs
  a memory gate exactly like `RNA_BUILD_PIPELINE`'s.
- **How:** T2a's machinery, reused rather than rebuilt — `rnafold_d2h_w()`'s
  pinned stage and per-stream copies, issued on a copy stream after
  `load_my_c(i)`, with one event per stage slot so the host never rewrites a
  slot still in flight (the level-2 lesson: **a device-side wait does not bound
  the host**).
- **Traffic:** the same bytes as today's exit path (~25 GB), but spread across
  the sweep instead of serialised after it. At 400 × 5601 the sweep is ~60 s, so
  the demand is **~0.4 GB/s — about 2 % of a pinned PCIe link** and 0.03 % of
  DRAM. It rides copy engines, which use no SMs.

### T2b-3 — one launch per sequence

Today every kernel is launched once per row across the whole chunk, and each
block finds its record with `flatten_index_to_H()` (a binary search over
`size_off_H`). Per-sequence launches would give: the record's window base as a
kernel argument (no search, no per-record offset tables), a natural home for
per-record streams, and per-record retirement.

**The 100-row window does NOT reduce launches by itself, and I hoped it would.**
Within a row the chain is `int_loop → new_c → load_my_c → fml_scan → md →
fml_prev → snapshot`, and row `i` cannot start until row `i+1`'s `c` exists,
which needs the multibranch half of the row too. So fusing rows into one launch
means fusing **all six kernels into a persistent megakernel** with a grid-wide
sync per row — and a cooperative grid holds only ~3 456 single-warp blocks
against a 5 601-cell row, so blocks would have to loop over cells. That is a
real design and the natural end of this road, but it is much bigger than the
window and is **flagged, not folded in**.

**The launch arithmetic has to be respected.** At 400 × 5601: 6 kernels/row ×
5601 rows = **33.6 k launches today**; per-sequence that becomes **13.4 M**. At
even 2 µs of launch overhead that is 27 s — a third of the current wall. So
per-sequence launches only pay **inside a CUDA graph**: one graph per row holding
the records' nodes, replayed. A graph node still costs ~1 µs, so 400 nodes/row ×
5601 rows ≈ **2.2 s**, against a saved binary search and better tail behaviour.

**And the tail cuts the other way.** Near the end of the sweep a record's row is
a few cells wide; 400 tiny grids fill the machine worse than one merged grid of
the same total width. **The honest shape is probably hybrid** — per-sequence
while rows are wide, merged once they are narrow — and that is a decision to
make on a measurement, not in this document.

---

### T2b-0 — what the window cannot help: `fML`

**`modular_decomposition` walks a whole column**, from row `i` down to `j`, so
its working set *is* the triangle and grows with length — the 82.8 %-of-DRAM
finding T1 exists to fix. No `c` window changes that, and the `fML` triangle
stays where it is (62.8 MB per record, 31 MB under int16). **"Iterate in fast
memory without touching another tier" is true of `c` and `int_loop`, and false
of the multibranch half.** Saying so here so that a later measurement showing
DRAM traffic barely moved is read as expected rather than as a failure.

---

## 3. What has to be decided before building

1. **Does the window's layout keep lane-striding?** Probe first: `RNA_MD_TILE=1`
   cost 247 % by breaking it. Measure `sectors/request` and `lg_throttle` for
   `int_loop` before and after the flip, on the production shape.
2. **What is a launch actually worth here?** A probe that issues the same row's
   work as N per-record launches vs 1 merged launch, graph and no graph, at
   400 × 5601 and at the tail widths. **This decides T2b-3 by itself** and is
   cheap — no correctness surface.
3. **Does the host have 25 GB?** If not, the window is still worth it for VRAM and
   cache, but the triangle has to stay on the device, and T2b-2 is off. The gate
   must be a measurement (`MemAvailable`), like the pipeline's.
4. **Does `c` have any other device reader?** Today no: `gquad.cu` does not touch
   it, `fml_scan` reads `d_new_e` (the row), not the triangle. **Re-check before
   shrinking it** — a reader that walks further back than MAXLOOP would read a
   recycled row and silently return a better-than-legal answer.

---

## 4. Order, with a bar at every step

| step | what | bar |
|---|---|---|
| 0 | probe 2 (launch cost) and probe 1 (striding) | numbers only, no sha risk |
| 1 | window behind `RNA_C_WINDOW=W` (0 = today's triangle; 32 is the correctness floor, 100 the proposal), triangle still filled | **sha identical at every W**, `RNA_ROW_VERIFY` clean |
| 2 | stop filling the triangle when the window is on; stream rows out to the host | sha identical; host RSS gated; `fetch_mx` → ~0 |
| 3 | chunk width: let the budget see the smaller per-record cost | chunk count falls at a fixed budget; wall improves by ~2 s per chunk removed |
| 3b | sweep W (32/64/100/200) for copy count vs residency | a W that wins on both is the default; if they disagree, copy count decides — it is the bigger number |
| 4 | per-sequence launches over a SUBSET of records, behind their own knob, graph-captured | sha identical; launch count, L2 hit rate and wall reported |
| 5 | *(flagged, not scheduled)* the persistent megakernel that fuses the row chain | — |

**Every step keeps one sha across every option arm** (`verify_option_parity.sh`,
`verify_constraint_parity.sh`, the overlap bar). A wrong answer here is invisible
without them: a recycled `c` row produces a *plausible* structure.

---

## 5. What this is worth, stated so it can fail

| | claim | falsified if |
|---|---|---|
| VRAM | per-record device bytes 172 → ~110 MB | the budget model does not move the chunk count |
| chunks | 400 × 5601 goes 2 chunks → 1 | it does not, or a wider chunk is slower per record |
| wall | ~2 s per chunk removed (measured: 1.99 s) | removing a chunk does not move the wall |
| exit | `fetch_mx` → ~0 (the bytes left during the sweep) | wall does not move when it does — then the exit path was never on the critical path |
| cache | `int_loop` L2 hit rate up, sectors/request flat | L2 falls, or sectors/request rises — the layout flip hurt |
| **int_loop time** | **unchanged or slightly better** | it gets *worse*: the window's addressing cost more than the locality bought |

**What T2b is NOT claiming:** it does not make `int_loop` fundamentally faster.
`int_loop` is retirement-limited and 26.5 of 32 lanes active (§35, §34.4); its
reads already hit L1 at ~80 %. The prize here is **VRAM, chunk width, and an exit
path that costs nothing**, plus the groundwork for per-sequence launches and,
after those, device-side backtracking — which would delete T2b-2 entirely by
never sending `c` to the host at all.
