# Blocked md, stage 3: the driver — plan for sign-off

Follows `PORT_MD_BLOCKING_INTEGRATION.md` (§3 the schedule, §3.6 the measurement, §3.8 the
A100 launch price). Stages 1 and 2 put the arithmetic in the tree. **This stage changes
the order of the sweep**, so nothing is written until this plan is agreed.

---

## 1. Where it stands

| | status |
|---|---|
| Stage 1, the tile primitive at `RB = 1` | landed; `RNA_MD_BLOCK_SELFTEST=1` |
| Stage 2, bulk + two corners at `RB > 1` | landed; `RNA_MD_BLOCK_SELFTEST=2`, **795 024 832 cells, 0 mismatching**, int16 and int32 (laptop, 24 mixed records 300–2300 nt, 2026-10-03) |
| Stage 2's negative control | **ran for the first time 2026-10-03**: `RNA_MD_BLOCK_SELFTEST=3` drops the one term `k = imax` from the decomposition, and the same check reports **1 061 795 mismatching** on both encodings. So the check sees a one-term range error |
| The kernel, isolated | 6.23× bit-exact at RB128 CB64 against the per-row reference (§3.6) |
| A launch on the A100 | 2.65 µs, so the per-row tail is net positive at every `CB`, best near 512 (§3.8) |
| **The schedule** | **unproven.** Stage 2 reads a finished triangle; it would pass with the order wrong. This stage is the first that can fail on order |

**What it is worth, honestly.** md is ~55 % of GPU time. §3.8 prices the simple schedule
at about **1.28× end to end** at `CB = 512`. That figure counts launch *overhead* only. It
does not count the second cost of the same choice: every per-row physics launch now covers
`CB × records` cells instead of a whole row (at 400 × 5601, 47 records per chunk: ~24 k
cells against ~130 k), and narrow grids run each cell less efficiently. **The A100 run's
section E measures exactly that curve** (µs per row against cells per row, for every
row-phase kernel). The plan's numbers are re-priced from it before any code is written.

---

## 2. The schedule, precisely

Rows descend as today, `RB` at a time. Within a block-row `I = [imin, imax]`:

```
columns of block-row I:
   Jd = (·, imax]                 the diagonal triangle: every row's own near-diagonal part
   J1 = [imax+1, imax+CB], J2, …  ascending, CB wide, per record up to its own length

for J in Jd, J1, J2, … (ascending):
    for i = imax down to imin:
        physics(i, J)              int_loop, hp_mb, new_c, c store, fML scan — columns of J only
        md(i, J) = min( ACC[i][j], cor1(i,j), cor2(i,j) )   for j in J
        pack row i's fML for the columns of J
    UPDATE(J): for every row i of the block and EVERY later column j > max(J):
        ACC[i][j] = min( ACC[i][j], min over k in J of fML[i][k] + fML[k+1][j] )
```

**The bulk is computed right-looking, not left-looking.** §3 of the integration plan
computes `BULK(I,J)` as one product just before `J` runs. At production that is one tile
per record per launch, 47 blocks on a 108-SM device. Once column block `J` is finished for
all `RB` rows, its contribution to every later column is ready, because `B = fML[k+1][j]`
has `k+1 > imax` and lies in earlier block-rows. One wide launch then covers every later `J`
of every record. The total work is the same, but the grids are wide. The cost is an
accumulator `ACC` of `RB × row width × records` ints: 128 × 5601 × 47 × 4 B = 135 MB at
production. That is affordable, and it enters the VRAM budget.

For `Jd` only the term `k = imax` falls in the bulk range, so its `UPDATE` is rank-1.

### Every read, and why it is ready

| step | reads | lives in | ready because |
|---|---|---|---|
| int_loop(i, j) | `c[p][q]`, `p ∈ (i, i+31]`, `q < j` | rows `p ≤ imax`: this block-row, `J` and earlier; `p > imax`: earlier block-rows | within `J` rows descend, so `p > i` is done; earlier `J` are done |
| hp_mb(i, j) | `fM2[i+1][j-1]` | row `i+1`, column `j-1` ∈ `J` or earlier | row `i+1` ran before row `i` in this `J`; `i = imax` reads the previous block-row |
| fML scan(i, J) | `c[i][j]`, `fML[i+1][j]`, `fML[i][j0-1]` | the carry-in is row `i`'s last column of the previous `J` | `J` ascends; the scan is an affine (min,+) scan, so a carry-in is exact |
| md cor2(i, j) | `fML[i][k]`, `k ∈ [j0, j-turn-2]`; `fML[k+1][j]`, `k+1 > imax` | this row's scan of `J`; earlier block-rows | the scan ran first in this step |
| md cor1(i, j) | `fML[i][k]`, `k < imax`; `fML[k+1][j]`, `k+1 ∈ (i+turn+1, imax]` | `Jd`; rows of this block at column `j` ∈ `J` | `Jd` ran first; those rows are above `i`, done earlier in `J` |
| `ACC[i][j]` | every `UPDATE(K)`, `K < J` | | each update ran when its `K` finished |
| int16 decode | one baseline per (column, 64-row block), set by the first row that reaches it | | **each column's rows are still visited in descending order**, so the first row to reach a baseline block is the same row as today |

The last row of that table is the int16 encoding's whole correctness condition, and it
survives because column-block-major reorders *columns*, never rows within a column.

### The diagonal triangle `Jd`

Its cells have `j ≤ imax`, so their whole `k` range lies inside the block and md is
`cor1` alone. The corner function already handles that range: call it with `j0 > khi` and
pass 2 is empty. No bulk and no new code path. `Jd` is also where the near-diagonal
work concentrates, which is the integration plan's §5 risk 1. Its share is reported per
run (cells in `Jd` against all cells), because it bounds what blocking can buy.

---

## 3. What the data structures need

**Row-shaped buffers fall into two kinds**, and only one needs to grow:

| kind | examples | after the change |
|---|---|---|
| **scratch within one `(i, J)` step**: written and consumed in the same step | `energy_hp_row`, `energy_mb_row`, `energy_3p00_row`, `gate_row`, `energy_min2`, `new_e`, `buf`, `stack_row` | **unchanged.** Same layout, indexed by column, written only on `J`'s columns |
| **carriers across steps**: written in step `(i, J)`, read in a later step with other rows in between | `dml` (fM2 row, read by row `i-1`'s hp_mb), `dml1`, `fml_prev`, `energy_min`/`fml_i` (row `i`'s fML, read by its own later `J` and by `UPDATE`) | **`RB + 1` rows deep**, a ring indexed `i mod (RB+1)`. The same shape stage 2's `md_blk_ring` already uses |

The classification is the first deliverable of stage 3a. It is done from the source,
buffer by buffer, as `feedback_hand_rolled_barrier_needs_fences` demands of any change
that overlaps steps, and committed as a table in this file — **§3.1, done 2026-10-03.**

### 3.1 The audit, from every read in the source (stage 3a, deliverable 1)

The table above draws the line by *row*. Reading every index expression shows the line
that matters is the **column offset** of each read. Column-block-major order changes which
columns run when, but **every single column still sees its rows strictly descending**. So a
buffer that row `i` reads at the *same column* `j` it was written by row `i+1` still holds
row `i+1` there, with one row of storage. Only a read that reaches **left** of `j` can
land on a column that a later row of the block has already overwritten in an earlier
column block.

| buffer | written at | read at | by | tile order needs |
|---|---|---|---|---|
| `d_dml` → `d_dml1` (fM2 row; swapped per row under `RNA_MD_TAIL`, copied otherwise) | (i, j) md (`md_cell.inc:217`) | (i, j) md_close; **(i−1, j+1)**, i.e. row `i+1` at **`j−1`** from row `i`, in new_c (`hp_mb_cells.inc:182`) | new_c | **ring of RB+1 rows**. At RB = 1 that is exactly today's pair |
| `d_cc` → `d_cc1` (`--noLP`; swapped per row, `hp_mb_loop.cu:1493`) | (i, j) new_c (`hp_mb_cells.inc:202`) | row `i+1` at **`j−1`** (`hp_mb_cells.inc:197`) | new_c | **ring of RB+1**, the same as dml |
| `d_energy_min` (row `i`'s fML before the DMLi min; md's `fml_i` under the collapsed tail) | (i, j) fML scan (`fml_scan_block.inc:127`) | (i, j) md_close; **(i, k)** for every `k ∈ [i+turn+1, j−turn−2]`, md's A operand (`md_cell.inc:121/201`) | md | **ring of RB+1**: md(i, J) reads row `i` back into earlier column blocks |
| `d_fml_prev` (row `i`'s final fML) | (i, j) md_close (`md_chain_cells.inc:128`) | row `i+1` at the **same** `j` (`fml_scan_block.inc:85/214`) | fML scan | **one row, unchanged** |
| `d_new_e`, `d_energy_min2`, `d_gate_row`, `d_energy_hp_row`, `d_energy_mb_row`, `d_energy_3p00_row`, `d_energy_stack_row`, `d_gq_row` | (i, j) | (i, j), same step | | **unchanged scratch** |
| the c ring (`d_c_ring`, slot `p & 31`) | row `p` | rows `p ∈ (i, i+31]`, columns `q < j` | int_loop | **RB + 32 slots** (column-indexed, so depth is its only problem) |
| the triangles `d_my_c`, `d_fml_j` | per (i, j) | anywhere below | int_loop, md | unchanged: written once per cell, read after |
| `d_fml_stage` (`RNA_ROW_BATCH`) | | | | refused on the tile path (§4) |

So **three rotating row buffers become rings, `fml_prev` stays, and the scratch is
untouched**, which is less than §3's table first assumed. Two reads (`dml1`, `cc1`) cross a
column-block boundary only at its *first* column, `j0−1`. Keeping one edge column per row is
a possible later saving against a full ring. It is not worth it for v1, because at RB = 64
a ring of three such buffers is 3 × 65 × row width × 4 B ≈ 4 MB per record at 5601 nt.

**First code step (3a.1), landable on its own:** replace the per-row pointer swaps of
`dml`/`dml1` and `cc`/`cc1`, and the single `energy_min`, by row-indexed rings with depth
`RB + 1`, default 2. At depth 2 the ring *is* today's swap, so this is byte-identical by
construction, and the bars say so. Column ranges (3a.2) come after, on top of it.

### 3.2 How a kernel addresses a tile (stage 3a.2 — built 2026-10-03)

**Revised from the first draft**, which proposed a 2-D grid per tile. Reading the
kernels showed something smaller. Every row kernel already turns its flat index into a
column as `j = (m − off[H]) + i + turn + 1`, from the row's offset table, and in
lock-step every record is on the same row. So a column range changes that formula by
**one scalar per launch**, `max(0, jlo − (i+turn+1))` (and `max(0, jlo − (i+2·turn+3))`
for md's `side` range). That scalar is carried in **one extra slot of the table it
already reads**, `off[nfiles+1]`:

- **The row tables' stride becomes `nfiles + 2`** (`RT_STRIDE`, `device.cu`). A row's own
  slots hold 0 there (the existing `memset`), so whole rows are byte-identical by
  construction. The megakernel walks the same tables on the device and now takes the
  stride from `rnafold_rowtab_stride()` instead of repeating `nfiles + 1`.
- **21 kernel sites** add the slot when turning the index into a column, and both fML
  scan variants seed their carry from `energy_min[o + j0 − 1]` when it is non-zero
  (`INF`, as before, at 0).
- **A tile's tables are built by the driver**: each record's width clipped to
  `[jlo, jhi]`, the shift in the extra slot. `rnafold_tile_begin(i)` uploads them and
  points row `i`'s lookups at them until `rnafold_tile_end()`. Every phase binds its
  table per call and sizes its grid from the host table's total, so no phase's
  signature changed. No per-tile tables are precomputed and no 2-D variants were
  written (`int_loop` alone has the cells-per-warp, GRIDY, ring and U2 variants).
- **`RNA_MD_TILE_CB=N`** runs 3a′ (`RB = 1`, every row as `N`-column blocks). It refuses
  with the reason, and the row path runs, unless overlap 0, graphs off, the c ring off
  and the row batch off. Those four hold or flush whole rows.

**THE BUG THE TILE BAR CAUGHT, and the rule it leaves.** The first version added the
shift to `j` only. But several kernels use their local index for more than `j`:
md's reduction runs `for (y …; y <= x; …)` with `x` the cell's position, `md_close`
chooses the triangle's rule by `mj >= turn+1`, and two kernels key "the row's first
cell" on `mj == 0`. With `x` local to the *tile*, md's `k` loop stopped early inside
any tile that started past md's first column. One 120-nt record came out −36.90
against the CPU's −37.70, correct at widths ≥ 60 and wrong at 40 and 10. Shift 0 hid
it completely: every default-path bar passed. The fix puts the shift into the
**row-local index** (`x`, `mj`) wherever it is used beyond `j`. The rule for 3b and 3c:
**a tile-local index is never a row position**, and any formula keyed on "where in the
row" must see the row position.

**Packing.** `pack_fml_cell` closes row `i` at the end of iteration `i`. In tile order it
packs row `i`'s columns of `J` at the end of step `(i, J)`, because `cor1` reads rows of the
same block from the triangle. `RNA_ROW_BATCH` defers triangle writes by up to 5 rows, which
breaks that. **v1 refuses it on the tile path.** Folding the stage into the tile is a 3d
question.

**The c ring.** It holds the last 32 rows and is indexed `row mod 32`. In tile order, after
`J` the ring holds rows `imin…imax` of `J`'s columns, while row `imax` at `J+1` still needs
rows `imax+1…imax+31`. For `RB ≥ 32` those slots are overwritten. **v1 makes the ring
`RB + 32` deep** (it is column-indexed, so depth is its only problem), or refuses the ring
if that does not fit. The A100 run's section E decides whether the ring is still worth
having at tile widths.

---

## 4. What v1 refuses

Every refusal prints a one-line banner and runs today's row path, so the answer is
byte-identical whatever the reason. All of them were found by reading the gates that
already exist (`c_ring_refuse`, `row_fuse`, `md_tail`).

| feature | v1 | why |
|---|---|---|
| continuous flow, slot flow | **refuse** | records on different rows; a block-row needs every record on the same `RB` rows |
| `RNA_ROW_BATCH` | refuse on the tile path | deferred packing (§3) |
| `RNA_ROW_FUSE`, the megakernel | refuse | each owns a whole row |
| stream overlap ≥ 1 | **v1 runs single-stream** | the overlap protocol publishes per row; per step is 3d |
| CUDA graphs | off on the tile path in v1 | capture is per row; a graph per block-row is 3d, and it is how the launch multiple gets paid down |
| `-g` | **included**: `gq_row_kernel` takes the column range in 3a.2 | §3.1: `gq_row` is same-step scratch, read at (i, j) only |
| `--circ`, `--noLP`, `-C` | **included**, each with its own parity case | their extra phases (`fM2_real`, `stack_row`, the hc depot) are per cell |

---

## 5. The stages, each landable and verifiable

| stage | change | bar | can fail on |
|---|---|---|---|
| **3a** | column ranges and scan carry-in on every row phase; carriers become `RB+1` rings; `RB = 1, CB = row` | **byte-identical** to today on the full bars (§6), because it is the same order | index arithmetic |
| **3a′** | same, `RB = 1`, `CB < row` | byte-identical; slower (launch multiple), and that cost is recorded | the carry-in, per-`J` packing |
| **3b** | `RB > 1`, column-block-major, **md still `md_cell` over its full range** | byte-identical, plus **a schedule negative control**: run `J` descending, which must break | **the order.** First stage that can |
| **3c** | md becomes `min(ACC, cor1, cor2)` with right-looking `UPDATE` and 4-lane corners; `RNA_MD_BLOCK=1`, default off | byte-identical; md + update GPU time down; `smsp__inst_executed` for md down ≈ 4× | the decomposition in the live order |
| **3d** | A100 tuning: `RB`, `CB`, `KB`, corner lanes; graph per block-row; the ring and row batch folded back in | wall at 400 × 5601 and 3000 × 1200 | |
| **3e** | fewer launches: fuse the tile's physics, or run `(i, J)` and `(i+1, J+1)` together as a wavefront for width | wall | |

**Default-on rule, written before the measurement:** `RNA_MD_BLOCK=1` becomes the default
only if, on the A100, it is faster than the default by more than the larger arm spread at
400 × 5601, is no more than 1 % slower at 3000 × 1200, and passes every bar in §6. A
length-dependent switch (blocked only above some length) is an allowed outcome.

---

## 6. The bars

On every stage, the **CPU column**, never agreement between GPU arms:

- `tools/verify_option_parity.sh`: 45 options, GPU against `RNA_GPU=0`;
- `tools/verify_constraint_parity.sh`: 5 shapes;
- `RNA_TRI_CHECKSUM=1`: c and fML triangles identical to the row path, int16 and int32;
- mixed-length fixtures (records of different lengths end at different `J`), G-rich `-g`,
  `--circ`, `--noLP`;
- `tests/python/test_RNA-cuda.py`, which has a second call in one process. A new
  `ACC` buffer is exactly the kind of state the binding has caught leaking between
  batches twice before;
- positive evidence on every arm: the tile banner printed, with `RB`, `CB`, `Jd` share.

---

## 7. What could go wrong

- **Narrow grids cost more than launches.** This is the main unknown. Section E of the
  A100 run measures it before any code exists. If the per-cell cost at tile width is
  more than ~1.5× the full-row cost, the per-row tail does not pay and 3e (fusion or
  wavefront) moves ahead of 3c.
- **`Jd` is large.** At `RB = 128` the diagonal triangle is 8 k cells per record per
  block-row, all handled by `cor1`. Measured per run, not assumed.
- **The `ACC` buffer's VRAM** shrinks chunks: 135 MB at production is about 1 record of
  5601 nt. The budget has to include it, or `desc`-order OOB (fixed once) comes back.
- **Carriers I miss.** A row buffer read across steps but classified as scratch gives a
  wrong answer only when rows interleave, which `RB = 1` cannot show. That is why 3b
  exists before 3c, and why its negative control (reverse `J`) must go red first.

---

## 8. Decisions (asked 2026-10-03, answered below)

1. **Right-looking `UPDATE`** (wide grids, a 135 MB accumulator) **or left-looking
   `BULK(I,J)`** as §3 first wrote it (47-block launches, no accumulator)? I recommend
   right-looking. Left-looking underfills the device at production by construction.
2. **The v1 refusal list (§4)**, in particular single-stream and graphs off on the tile
   path, accepting that v1 measures the decomposition rather than a tuned schedule.
3. **Starting geometry**: `RB = 64`, `CB = 512`, `KB = 32`, 4-lane corners, then sweep in
   3d. `RB = 64` rather than 128 keeps `Jd` and the rings half the size while the kernel
   is flat across 64–128 (§3.6).
4. **Whether stage 3 waits for the A100 run's section E.** I recommend 3a starts now (it
   is byte-identical and needed whatever E says), and 3b/3c wait for E.

**Decided 2026-10-03 (Luke): all four as recommended.** Right-looking `UPDATE`; v1's
refusal list; `RB = 64, CB = 512, KB = 32`, 4-lane corners; 3a starts now, 3b/3c wait for
the A100 run's section E.

---

## 9. Kept for later: the left-looking bulk

Not chosen, but not discarded. Luke asked for it to stay on record in case experiment
favours it. As §3 first wrote it, the bulk for column block `J` is one product computed
just before `J` runs:

```
for J ascending:
    BULK(I,J) = min over k in [imax, j0-1] of fML[i][k] + fML[k+1][j]   (all RB rows, J's columns)
    for i descending: physics(i,J); md(i,J) = min(BULK[i][j], cor1, cor2)
```

**What it has over right-looking.** No accumulator: the bulk lives in registers and
shared memory for one tile and is consumed at once, so there is no `RB × width × records`
buffer (135 MB at RB = 128 at production) and no VRAM taken from chunk width. Each output
cell is written once rather than min-updated once per earlier column block, so global
read-modify-write traffic is lower. And it is exactly what `md_block2_selftest_kernel`
already computes, so the kernel exists.

**Why it lost the first round.** One launch per (block-row, `J`) covers `RB × CB` cells
per record, so at 47 records the grid is 47 blocks on a 108-SM A100. It underfills the
device by construction, while right-looking's `UPDATE(J)` covers every later column of
every record in one launch.

**When it could win**, and so what would reopen it:

- **Wide chunks.** With hundreds of records per chunk (short sequences, a larger device),
  47 blocks becomes hundreds and the underfill disappears.
- **Several `J` per launch.** If 3e's wavefront runs `(i, J)` beside `(i+1, J+1)`, the
  bulks of several column blocks are ready at once and can share a launch.
- **VRAM-bound chunks.** If the accumulator's 1 to 2 % of VRAM measurably shrinks chunk width
  at production, left-looking's zero footprint is worth more than its grid.
- **Measured**, not argued. 3d can carry both behind a switch (`RNA_MD_BLOCK_BULK=right|left`):
  the corners, the schedule and the bars are shared, and only the bulk's placement differs.

---

## 10. What the A100 Queue run said (2026-10-03, `ffdba8c5`)

**Section E1 priced the tile width, and the ratio is 1.59**, above §7's threshold of 1.5.
It compares the per-cell cost of the physics kernels at ~24 k cells (one `CB = 512` tile
of 47 records) with their cost at ~130 k (a production row). It is not uniform:

| kernel | ns/cell at 24 k | at 130 k | ratio |
|---|---|---|---|
| `int_loop_warp_kernel` | 3.97 | 3.22 | 1.23 |
| `hp_mb_3p_kernel` | 0.78 | 0.32 | 2.48 |
| `md_close_row_kernel` | 0.24 | 0.07 | 3.30 |
| `new_c_kernel` | 0.28 | 0.08 | 3.67 |
| `c_ring_store_kernel` | 0.22 | 0.05 | 4.02 |
| `fml_scan_kernel` | 0.69 | 0.14 | 4.97 |
| **summed** | **6.17** | **3.88** | **1.59** |

`int_loop`, the expensive one, barely minds the narrower grid. The ratio comes from the
four small kernels, which take 4–19 µs per launch whatever their width (E1's µs/row barely
moves from 2 to 128 records). They are bound by **per-launch latency**, not work. So the
per-tile tail does not pay as separate launches, and as §7 pre-registered, **fusing the
tile's small kernels (3e) moves ahead of 3c.** One kernel per tile for `new_c` + c store +
fML scan + `md_close` (+ `hp_mb_3p`) turns four or five latency-bound launches into one.

**The new order:** 3a (column ranges, carry-in; byte-identical) → 3a′ (`CB < row`; records the
unfused cost) → 3b (the schedule and its negative control) → **3e-fuse** (the tile's small
kernels as one) → 3c (the blocked md) → 3d (tuning). 3a and 3b are unchanged by this: fusion
needs the column ranges and the order proven first.

---

## 11. Stage 3b, built 2026-10-03

**The sweep is now a sequence of steps** (`fill_arrays_loop.c`), each one (row, column
block). Without tiles it is one step per row, rows descending: the old loop, the same
calls in the same order (36/36 configurations byte-identical against the CPU, 45/45 options).
With `RNA_MD_TILE_CB=N RNA_MD_TILE_RB=RB` it is block-rows of `RB` rows, and for each
column block, left to right, the block-row's rows top down: column-block major. A row's
first step does its per-row head and its last step its tail. The rotating rows are
**selected by row** at every step (`rnafold_md_ring_select`, `rnafold_cc_ring_select`:
slot = row mod (RB+1)) instead of advanced, because rows interleave. md still runs over
its whole `k` range, so this stage tests **the order** and nothing else.

**The order bar caught a real order error.** Every geometry with `CB < RB` failed
(RB33 CB32 failed, RB33 CB33 passed). The cause was the diagonal-band cell. `md_close`
wrote `fml_prev[i + turn] = INF` from row `i`'s *first* step, for row `i−1` to read.
When `CB < RB`, an upper row's first step comes in a later column block than the lower
rows' work on that column, so the late write clobbered the lowest row's real fML there,
and the next block-row read INF. That cell is INF by definition, so the fix makes the
scan treat `(i+1, i+turn+1)` as INF itself, and the two band writes are gone. That is
safe in row order too, where lower rows overwrote that column anyway. §3.1's audit
missed this read because it listed the *regular* reads. The rule: **audit the
special-case writes too** (band cells, sentinels, first and last cells), because those
are the ones whose timing a reordering changes.

**Negative control:** column blocks right to left (`RNA_MD_TILE_REVERSE=1`) changes both
the structures and the triangles at RB = 1 and RB = 5. The order bar can see an order
error, and the right-to-left order is the error it sees.

---

## 12. Stage 3e-fuse, built 2026-10-04

**One launch per tile step for the c chain and the fML scan.** `tile_front_kernel`
(`hp_mb_loop.cu`) runs one block per record over the step's column block. Each thread
walks its cells through `hp_mb_3p_cell`, `stack_row_cell` (noLP), `new_c_cell` and
`load_my_c_cell`, which are the standalone kernels' own bodies. Then a `__syncthreads()`
and `fml_scan_block` with the carry-in. That is five launches (hp_mb_3p, stack row,
new_c, load_my_c, fml_scan) become one. It is on whenever tiles are (`RNA_MD_TILE_FUSE=1`
is printed), and `RNA_MD_TILE_FUSE=0` is the separate-kernel arm. The `-g` row expansion
reads only `c_gq`, so it stays a launch of its own, just before.

**Why the cells need no barrier and the scan needs one.** This is row_cells_kernel's
argument. Every cell reads its own `j`, or the previous row at `j−1` (an earlier launch).
The scan reads every `j` of the block, written by other threads, so one block per record
turns that dependency into a `__syncthreads()`. `new_e` and `energy_3p00_row` are written
and read in the same launch, so they are not `__restrict__` in this kernel. The SASS
confirms that the scans read them with coherent `LDG.E`, not `LDG.E.CONSTANT`.

**`md_close` is not fused here, deliberately.** It reads md's output for its own cell,
so it belongs in md's epilogue, and 3c replaces today's md kernel with the corner
kernel. It is fused there, in 3c, not into a kernel 3c deletes.

**Bars (laptop, 2026-10-04).**
- **The 3e matrix: 54/54** (plain RB 1/2/5/33 × CB 7/64/333; `-g`, noLP, circ; int16 and
  int32). Each arm is checked against the CPU's structures and the row path's c and fML
  triangles, and each prints `RNA_MD_TILE_FUSE=1`.
- **`RNA_MD_TILE_FUSE=0`** still runs and gives the same answer.
- **Negative controls**, all of which must and do change the structures and the
  triangles:
  - `RNA_MD_TILE_FUSE_NEGCTL=1` runs the scan before the cells, at RB 1 and RB 5;
  - reversed column blocks under fusion.

**What it buys on the laptop** (47 × 2000 nt, RB 64, CB 512, single stream, ABBA). The
clock drifted 60 % through the run, so each arm is normalised by its own int_loop time,
which fusion does not touch. (c chain + scan + md) / int_loop is:

| arm | ratio |
|---|---|
| row path | 0.80–0.82 |
| tile, unfused | 0.89–0.93 |
| tile, fused | 0.79–0.87 |

The tile overhead of these phases is gone. What remains of the tile overhead is
int_loop's own: +12 % here, 1.23× on the A100 (§10). The A100 decides, with §10's E1
shape: price the fused kernel per cell at 24 k against 130 k.

**Next: 3c.** md becomes `min(ACC, cor1, cor2)` with the right-looking `UPDATE`, and
`md_close` moves into the corner kernel's epilogue.

---

## 13. Stage 3c, built 2026-10-04: correct, and slower than md per cell at CB 512

**What it is.** `RNA_MD_BLOCK=1` (tile path only; it refuses without `RNA_MD_TILE_CB`) makes
md at cell (i, j) of column block [jlo, jhi] in block-row [bot, top] equal to
`min(ACC[i][j], cor1, cor2)`:
- **The UPDATE** (`md_blk_update_kernel`) is right-looking. After column block J's last
  row (`bot`), one launch folds k ∈ [max(top, jlo), jhi] into ACC for every later column
  of every record. Its tiles are RT 32 × CT 64 × KB 32, with 4 × 4 outputs per thread.
- **The corners** (`md_blk_corner_kernel`) use 4 lanes per cell over [i+turn+1, top−1] and
  [jlo, j−turn−2], then take the min with ACC. The same kernel does md's epilogue (`d_dml`,
  and the clamped fM2 under `--circ`).
- **ACC** is RB rows of the row-buffer layout, reset at each block-row's first step,
  allocated per chunk, freed by the pointer, and charged in the byte model.

The operands are md_cell's own. A is row i's E from its energy_min ring slot. B is the
triangle, decoded the same way. The sums are the same unguarded A + B, so the answer is the
min over the same set: byte-identical by construction.

**A bug the first run caught.** `md_block_product<CB,…>` loops over CB because it assumes
the staging depth *is* the column count. Stage 2 only ever ran CB = KB = 32, so this never
showed. At CT 64 / KB 32 it read past the staged rows (an illegal memory access). The
UPDATE now has its own KB-deep loop. **A shape-coincident test (CB == KB) cannot see a
depth/width mix-up.**

**Bars (laptop, 2026-10-04).**
- **The 3c matrix: 54/54** (plain RB 1/2/5/33 × CB 7/64/333; `-g`, noLP, circ; int16 and
  int32). Each arm is checked against the CPU's structures and the row path's c and fML
  triangles, prints `RNA_MD_BLOCK=1 ACTIVE`, and runs at least one UPDATE.
- **Refusal:** without tiles it refuses.
- **Negative controls:** `RNA_MD_BLOCK_NEGCTL=1` drops k = kmin from each UPDATE. The
  triangles DIFFER at RB 1, 5 and 33; the structures differ only at RB 5, so the triangle
  bar is the one that sees it. Reversed column blocks under blocking also DIFFER.
- **Default path:** the default matrix is 28/28, option parity 45/45, and the binding
  suite passes.

**What it costs on the laptop** (47 × 2000 nt, RB 64, single stream, fused front; md timer,
which includes UPDATE, corners and md_close, divided by the same run's int_loop):

| CB | row path | tile, md per cell | tile, blocked md |
|---|---|---|---|
| 512 | 0.75–0.79 | 0.74–0.77 | **0.85–0.92** |
| 64  | 0.67–0.78 | 0.77–0.89 | **0.66–0.76** |

**Why CB 512 loses: cor2 scales with CB.** cor2's range is the current column block, so
at CB 512 each cell walks ~CB/2 ≈ 256 terms on 4 lanes, with no reuse. The isolated 6.23×
was measured at CB 64. At CB 64 blocked md beats per-cell md on the same tiles and roughly
matches the whole row. But CB 64 multiplies the per-step launches (int_loop and the c
chain), and the wall is far worse (10–12 s against 7 s).

**So CB plays two roles that pull opposite ways:** the physics wants wide tiles, and md's
corners want narrow ones. That is the first thing 3d has to settle. Two candidates:
- **Block cor2 inside J.** Split J into CB_md-wide sub-blocks, and run an UPDATE between
  them within the step sequence's own row order. That needs rows re-ordered per sub-block,
  so it is a schedule change.
- **Make cor2 a blocked product too.** Its B rows k+1 > jlo are mostly earlier block-rows
  (all of them when jlo > top), so for J right of the diagonal, cor2 is a (min,+) product
  of row i's E on J with a finished B patch. It could be staged and shared across the
  block-row's RB rows like the bulk. Rows inside the block-row (k+1 ≤ top) remain per cell.

Until then RNA_MD_BLOCK stays **off by default** (§5's default-on rule needs an A100 win).
`md_close` stays its own launch; folding it into the corner kernel's epilogue waits for the
corner design to settle.
