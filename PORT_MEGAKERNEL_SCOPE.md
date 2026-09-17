# The megakernel — fusing the row chain, and what it unlocks for `modular_decomposition`

*Scope, written 2026-09-17 against `230056a2`. Nothing built. Asked for by Luke
while scoping T2b's per-record launches: if each sequence gets its own kernel,
can the fML **columns** be cached too?*

---

## 1. The finding that reorders the plan: T1 as written is not legal

`PORT_LUKES_FLOW_BATCHING.md` T1 proposes tiling the *i*-loop into blocks of B
rows **inside `modular_decomposition`**, keeping column `j`'s live segment in
shared memory or registers across those rows. It priced the tile's own skew
("cell `(i−1,j)` needs `(i,j)`") — but that is not the binding dependency.

Read the kernel: the inner loop is

```
value = MIN2(fml_i[row_off_H[H]+y] + fml_j[tri_off_H[H] + Indx(i,j)+turn+2+y], value)
```

so it needs **`fml_i` = row `i` of fML**. And row `i−1` of fML exists only after
the whole row chain has run:

```
md(i) -> DMLi -> new_c(i-1) -> load_my_c(i-1) -> fml_scan(i-1) -> load_fML(i-1) -> fmli(i-1) -> md(i-1)
         ^ int_loop(i-1) and hp_mb_3p(i-1) feed new_c(i-1) too
```

**So two consecutive md rows cannot be fused without fusing everything between
them.** Column reuse across rows — the entire basis of T1's 82.8 % → ~21 % —
requires the megakernel, or at least a fused row-chain kernel. That is the real
reason to build one, and it is a bigger claim than "save launch overhead".

---

## 2. What the columns actually look like, and which tier can hold them

For cell `(i,j)`, `Indx(i,j) = j(j−1)/2 + i`, so `+y` walks **down column j**:
a contiguous run of `j − i − 2·turn − 3` ints. At row `i−1` it is the same run
plus one element at the top. **Every element is re-read on every row** — there
is no hot sub-range to prefer, so caching pays in proportion to the *fraction*
of the triangle a tier can hold.

| what | bytes at L = 5601 | tier that could hold it |
|---|---|---|
| `fml_i`, one row (every cell reads it) | 22 KB | already L1/L2-resident; not the problem |
| one block's `c` window (32 owned columns × 31 rows) | **7.8 KB** | **shared memory, comfortably** |
| one record's fML triangle, int32 | 62.8 MB | DRAM only |
| **one record's fML triangle, int16** | **31.4 MB** | **≈ L2 (40 MB; ≤ 30 MB can be pinned as persisting)** |
| all on-chip shared memory (108 SMs × 164 KB) | 17.7 MB | ~56 % of the int16 triangle |
| all on-chip registers (108 SMs × 256 KB) | 27.6 MB | — |

Three answers to "can we cache the columns", in increasing order of ambition:

**(a) L2 residency, per record — available as soon as records are scheduled in
subsets, with no megakernel at all.** At int16 a 5601-nt triangle is 31.4 MB
against 40 MB of L2, and `cudaAccessPolicyWindow` can mark up to ~30 MB of it
*persisting*. Then md's re-reads come from L2 at ~4–5 TB/s instead of DRAM at
1.55 TB/s. **This is the cheapest large win available and it is a T2b-3
by-product**, not a megakernel feature. It is also length-gated: int32 fits only
below ~3 900 nt, int16 below ~7 000.

**(b) On-chip column state inside a fused row chain — the megakernel.** With
blocks owning fixed column ranges for the whole sweep, the part of each column
that fits in that block's shared memory is read from DRAM **once**, not once per
row. Traffic falls by the fraction held: 17.7 MB of a 31.4 MB int16 triangle is
~56 %, so md's DRAM demand roughly halves even before any tiling.

**(c) Both, plus int16 — and note this inverts an earlier prediction.** The plan
says int16 should stop paying once bandwidth is no longer the bound. Under
residency the opposite holds: **int16 is what makes the triangle fit**, so its
value *rises*. Whichever way the measurement goes, it is informative.

**(d) Compressing the column, on the grounds that an INF operand can never win a
min — MEASURED AND DEAD (2026-09-17).** `RNA_MD_INF_STATS=k` samples every k-th
row and counts exactly what md reads, including how many *aligned 32-wide tiles*
(the granularity a load could actually be skipped at) are entirely INF:

| L | elements sampled | INF | tiles all-INF |
|---|---|---|---|
| 600 | 1.5 M | 0.6 % | 0.6 % |
| 1 200 | 6.2 M | 0.3 % | 0.3 % |
| 2 400 | 24.9 M | 0.1 % | 0.1 % |
| 4 800 | 100.3 M | **0.1 %** | **0.1 %** |

**A bitmap would skip 0.1 % of the traffic, and the fraction FALLS with length.**
The INF entries are the near-diagonal band where a span is too short to hold a
loop; that band has a fixed width, so its share shrinks as `L` grows. The column
stream is dense, and compression is not a lever. Do not re-propose it.

---

## 3. What the megakernel is, concretely

One **cooperative** launch per chunk (or per record group), holding a persistent
grid that loops over rows internally:

```
cooperative_groups::grid_group g = this_grid();
for (i = top; i >= 1; i--) {
    int_loop_phase(i);        g.sync();     // blocks stride over this row's cells
    hp_mb_phase(i);           g.sync();     // (independent of int_loop: could share a sync)
    new_c_phase(i);           g.sync();
    load_my_c_phase(i);       g.sync();     // writes the c window slot for row i
    fml_scan_phase(i);        g.sync();     // one block per record, others idle at the barrier
    load_fML/md/load_min(i);  g.sync();     // md reads its owned columns from shared where held
    fml_prev + snapshot(i);   g.sync();
}
```

- **Block → work mapping is persistent**: block `b` owns a fixed range of columns
  `j` for its record, for the whole sweep. That is what makes on-chip column
  state and a resident `c` window possible; it is also what makes the megakernel
  different from "the same kernels, launched less often".
- **The `c` window (T2b) lives beside it**: a block's own 31 × (C+31) window is
  7.8 KB of shared memory at C = 32, updated by one row per iteration. Luke's
  "keep iterating in fast memory" is literally this line.
- **Records in subsets** so the L2 story of §2(a) holds: ~16 at 5601 nt.

---

## 4. What it costs, and the parts that are not negotiable

| constraint | number on an A100 | consequence |
|---|---|---|
| cooperative grid must be **resident** | at 256 threads/block and ~53 regs: 4 blocks/SM → **432 blocks, 110 k threads** | a 2.24 M-cell row is walked by stride loops, ~650 cells per warp per row |
| **one block size for every phase** | int_loop wants 1 warp/cell; md ran best at 512–768; fml_scan carries a scan along `j` | every phase must be rewritten against one shape (e.g. 256 = 8 warps, 8 cells per block for int_loop) |
| **registers = max over phases** | int_loop 53, md 40 | occupancy for the *whole* kernel is set by the worst phase |
| **grid.sync per phase** | ~7 per row × 5 601 rows = **~39 k barriers** | a barrier costs microseconds; **probe this first** — at 5 µs it is 0.2 s, at 50 µs it is 2 s |
| no CUDA graphs, no per-kernel launch | — | the graph machinery and its re-instantiate counters retire |

**What breaks and must be re-provided:**

- **Per-phase timers** (`RNA_PHASE_SYNC`) and **ncu per-kernel attribution** both
  collapse to one kernel. Replacement: device-side clock64() accumulators per
  phase, written to a small buffer. Without this the megakernel is unmeasurable,
  and an unmeasurable fast path is how this project got a 1.72× shipped default.
- **`RNA_ROW_VERIFY`**, which compares device rows against host loops each row.
  Inside a megakernel there is no host between rows. Keep the old path as the
  verification mode and gate the megakernel off while verifying.
- **Continuous-flow retirement** calls back to the host mid-sweep. Either defer
  retirement to the end (losing T3) or have the kernel post completions to mapped
  host memory for a polling host thread.
- **Device asserts and error attribution**: a fault surfaces at the end of a
  30-second kernel with no node to blame. The existing `gpuErrchk` per launch is
  the thing being deleted.
- **Watchdog/timeout**: a multi-second kernel is fine on a headless A100, fatal on
  a display GPU (the RTX 3050 under WDDM). The local box may not be able to run
  it at all — plan on Colab for this one.

---

## 5. Order, smallest testable step first

Each step is independently measurable and keeps the sha.

| step | what | why it is first |
|---|---|---|
| 0 | **probe grid.sync cost** (empty cooperative kernel, 39 k barriers, real grid) and **probe per-record L2 residency** (int16 + `cudaAccessPolicyWindow`, md phase only) | both are cheap, and either can kill or re-rank the design before a line of the megakernel is written |
| 1 | **fuse the independent pair**: `int_loop` + `hp_mb_3p` into one kernel with a block-role split (they share no data; today they are two launches) | smallest real fusion, no barrier needed inside, directly tests the "one block size for two phases" problem |
| 2 | **fuse the cell chain**: `new_c` + `load_my_c` (+ the `c` window write) | the pair with the tightest producer/consumer coupling |
| 3 | **one row, one kernel**: all seven phases, grid.sync between, still launched per row | isolates the fusion cost from the persistence win; wall should be *no worse* |
| 4 | **persistent across rows**, blocks owning columns, `c` window and column state on chip | the actual prize: T1's reuse, now legal |
| 5 | re-test int16 **as a residency check**, not a bandwidth check | see §2(c) |

**Bars at every step:** one sha across the option matrix
(`verify_option_parity.sh`, `verify_constraint_parity.sh`, the overlap bar), and
the megakernel behind its own knob (`RNA_MEGAKERNEL`, default off) so the
existing path stays the control.

---

## 6. What it is worth, stated so it can fail

| | claim | falsified if |
|---|---|---|
| launches | 33.6 k → 1 per chunk | — (bookkeeping, not a prize on its own) |
| barriers | ~39 k grid syncs replace ~33.6 k launches + per-row host syncs | the barrier probe comes back above ~20 µs, in which case fusion must stop at step 3 |
| **md DRAM traffic** | **roughly halves** from on-chip column state (17.7 MB of a 31.4 MB int16 triangle) | traffic does not move — then the ownership mapping is not holding what it claims |
| md phase | 37.8 s → **12–20 s** combined with L2 residency | md stops being bandwidth-bound and becomes latency- or issue-bound at a similar wall |
| occupancy | unchanged or better | the fused register count drops occupancy below what the phases had separately — the likeliest way this fails |
| **L2 residency alone** (§2a, no megakernel) | md phase −20 % or better at ≤ 5 600 nt, int16 | no change — then the triangle was never L2-resident and the carve-out did not take |

**The honest summary:** the megakernel is not primarily a launch-overhead play —
that is worth ~2 s. It is the only structure in which **T1's column reuse is
legal**, and T1 is the largest lever in the project. The cheap half of the same
idea — per-record L2 residency at int16 — needs no megakernel at all and should
be measured first, because if it delivers most of the traffic win, the megakernel
has to justify itself on what is left.
