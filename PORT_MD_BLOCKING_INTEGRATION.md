# Integrating blocked Zuker into md: the plan

Follows `PORT_MD_BLOCKING_SCOPE.md`, which measured **6.30× bit-exact** on the isolated
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

## 3. The target schedule: a block-row sweep

This is the part that makes integration tractable. A full anti-diagonal wavefront over
tiles would replace the sweep entirely. It is not needed:

```
for I = last block-row .. 0            # descending, like today's i-loop but RB rows wide
    for J = I .. last block            # ascending columns
        # BULK: parallel over the whole RB x CB tile
        acc[r][c] = min over K in (I,J) of  (min,+) product of fML(I,K) and fML(K+1,J)
        # TAIL: sequential inside the tile, i descending, j ascending
        for i in block I, descending:
            for j in block J, ascending:
                fM2[i][j] = min( acc[r][c], k in block I, k in block J )
                c  [i][j] = ...  uses fM2[i+1][j-1]
                fML[i][j] = min( fML[i+1][j]+b, fML[i][j-1]+b, c[i][j]+stem )
```

Everything the bulk reads is complete: `fML(I,K)` was produced at an earlier `J` in this
same block-row, and `fML(K+1,J)` at an earlier block-row. The outer loop keeps its
present shape and direction, which means `fill_arrays_loop.c`'s structure, the row
tables, continuous flow's per-record row pointers and the retire pool all survive.

**What changes:** one iteration advances `RB` rows instead of one, and within an
iteration the per-cell physics (`int_loop`, `hp_mb`, `new_c`, the fML scan) runs on a
`RB × CB` tile instead of a full row.

**What that costs:** those four phases are currently row-shaped kernels with row-shaped
buffers. Running them tile-shaped is the bulk of the work, and it is why this is
week-scale rather than a kernel swap.

## 4. Staging, with a bar for each

Each stage is independently landable and independently verifiable. No stage leaves the
tree slower or less correct than it found it.

### Stage 1 — the primitive, in-tree, exercised at `RB = 1`

`md_block.inc`: the blocked tile product as a device function shaped for production —
per-record `tri_off_H`/`row_off_H` offsets, the int16 decode with per-64 baselines, the
same `INF` semantics. Selected by `RNA_MD_BLOCK=CB`, default off.

At `RB = 1` the tile is `1 × CB`, which is still a matvec, so **this stage buys no
speed**. It buys the thing that must be right before `RB > 1` is worth attempting: the
index arithmetic, the decode, and the masking, validated in situ.

**Bar:** byte-identical fold output against `RNA_MD_BLOCK` unset, on mixed-length
fixtures, plus full option parity. Plus `RNA_MD_BLOCK` announced on stderr so a run that
did not take the path cannot pass for one that did.

### Stage 2 — `RB > 1`, tail on the host or in a second kernel

Raise the row block. The bulk becomes a real (min,+) product with reuse, which is where
the 6.30× lives. The tail is the sequential `RB × CB` corner; the first cut can run it
one row at a time with the existing kernels, paying `RB` times the per-row launch cost on
a `1/CB` slice of the work.

**Bar:** byte-identical again, and `dram__bytes.sum` for md down by roughly `CB/2`. If the
bytes do not move, the bulk is not doing what this document claims and the stage stops.

### Stage 3 — tile-shaped `int_loop` / `hp_mb` / `new_c` / fML scan

The tail's four phases become tile-shaped. This is the largest stage and the one that
actually removes the per-row launch count. `int_loop` gains its own lever here for free:
its 30×30 `my_c` window is shared by neighbouring `j`, so a tile of `CB` columns reads
`(30+CB)×30` instead of `CB×900` — **15.5× fewer loads at `CB = 32`**, in 7.3 KB of
shared, aimed at a kernel whose dominant stall is `long_scoreboard`.

**Bar:** byte-identical, and md + int_loop time down. This is where the ~2–2.5× end-to-end
should appear.

### Stage 4 — tuning, and the Hopper arm

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

The honest ceiling is **~1.85×** on GPU time from md alone and **~2.5×** with int_loop's
lever, capped by Amdahl rather than by the kernel. Not 6×. But it is larger than every
scheduling result on this branch put together, and it is the only one that moves the
roofline instead of the schedule.
