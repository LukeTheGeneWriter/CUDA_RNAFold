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
