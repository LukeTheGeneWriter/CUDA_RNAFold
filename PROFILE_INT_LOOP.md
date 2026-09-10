# `int_loop_kernel`: block size, int16, and a coincidence worth chasing

*Written 2026-09-10 after the phase-synced T4 run (`cd2b000b`) made `int_loop`
the second-largest GPU phase. This is a profiling PLAN with the analysis that
motivates it, not a results doc — the measurements it asks for have not been
made. Read `PORT_INT_LOOP_SCOPE.md` first for why this kernel is suddenly
interesting and why the obvious lever is already closed.*

---

## 0. The one number that prompted this

Both phase-synced arms at 400 × 5601 do **identical work** — 6 266 401 200 cells,
400 records — and differ only in how that work is packed into chunks.

| | i32 | i16 | ratio |
|---|---|---|---|
| chunks | 14 | 11 | |
| **records per chunk** | **28.6** | **36.4** | **1.273** |
| `int_loop`, total | 107.26 s | 137.70 s | 1.284 |
| **`int_loop` per cell** | **17.12 ns** | **21.98 ns** | **1.284** |

**The per-cell slowdown (1.284) matches the records-per-chunk ratio (1.273) to
within 0.9 %.**

That is one pair of points and it could be a coincidence. But it has a mechanism
behind it, and the mechanism predicts things that are cheap to test.

## 1. Why records-per-chunk could plausibly set the cost

`int_loop_kernel` launches **one block per `(H,j)` cell** and each block reads
`my_c[tri_off_H[H] + Indx(p,q)]` over a `(p,q)` window bounded by `MAXLOOP = 30`.

**`d_my_c` is `int*` unconditionally — int16 does not touch it.** What int16 does
is halve the *fML* triangle, which frees VRAM, which lets the chunker fit **1.27×
more records per chunk**. Every one of those records brings its own full-width
int32 `my_c` triangle into the working set that a single sweep row strides over.

So the candidate story is: `int_loop` is limited by locality on `my_c`, and
widening a chunk scatters its reads further apart. If true:

- the cost has **nothing to do with the encoding** and everything to do with
  chunk width;
- it attaches to **anything** that widens a chunk — including the build
  pipeline, and including any future VRAM saving;
- and int16's real benefit is *better* than §19 measured, because part of what it
  gives back is a chunking side-effect that could be tuned away independently.

## 2. This is in tension with what the source already concludes

`int_loop.cu:1168-1171`, from the 2026-08-20 NCU sweep, ends:

> *"Real occupancy gains for this kernel come from more **blocks** in flight
> (bigger batches, more concurrent (H,j) cells — exactly what
> staggering/mixed-length batching is for), not from bigger blocks."*

**More records per chunk is precisely "more blocks in flight".** That comment
says it should help; §0 says the per-cell cost rose 28 % when it happened.

Both can be true — more blocks improves *occupancy* while a larger working set
worsens *locality*, and which wins depends on where the kernel actually binds.
But nobody has measured which, because **this kernel has never been profiled at
all**: `tools/make_nb_profile272_deep.py` ran `ncu -k int_loop_kernel` against
symbols actually named `int_loop_kernel_32/_64/_128/_256`, matched nothing, and
recorded `null` (fixed in `30a041ce`).

**That tension is the thing to resolve, and it is the reason to re-open block
size rather than take the 2026-08-20 answer as final** — that sweep was run on
int32, on a laptop 3050, at grids up to ~54 000 blocks. A 400 × 5601 row launches
~2.2 M, and int16 changes the working set underneath it.

## 3. The experiment

All arms at 400 × 5601 on a T4, **`RNA_PHASE_SYNC=1`**, comparing
**`int_loop`'s synced phase time**. Never a synced wall; never synced against
unsynced. Every arm must report `sha 7c0b3d633281` and a `sweep shape:` line.

### 3a. The decisive one: separate chunk width from datatype

`RNA_GPU_VRAM_BUDGET_MB` chosen so each datatype runs at **both** chunk counts.

| arm | datatype | chunks | rec/chunk | what it isolates |
|---|---|---|---|---|
| A | i32 | 14 | 28.6 | baseline (have it: 107.26 s) |
| B | i16 | 11 | 36.4 | baseline (have it: 137.70 s) |
| **C** | **i16** | **14** | **28.6** | **int16 at i32's chunk width** |
| **D** | **i32** | **11** | **36.4** | **int32 at i16's chunk width** |

**This is the whole experiment.** Everything else is secondary.

- If **C ≈ A** and **D ≈ B** → the cost is **chunk width**, not the encoding.
  int16's `int_loop` penalty is an artifact of it enabling wider chunks, and the
  §19 give-back should be re-stated.
- If **C ≈ B** and **D ≈ A** → it really is the encoding, and the mechanism is
  something the kernel does differently that inspection has not found — since
  `int_loop.cu` contains no int16 code at all, that would be a genuinely
  surprising result and worth a lot.
- Anything in between → both matter, and the split tells you how much.

**Prediction, stated before the run:** C lands near 107 s and D near 137 s. Write
it down so the prediction can be wrong.

### 3b. Block size, re-opened — but only on the axes that changed

The 2026-08-20 sweep already answered `32 vs 256` on int32 at grids ≤ 54 000. Do
not repeat it. Test only what is new:

| axis | old coverage | new |
|---|---|---|
| grid size | ≤ 54 000 blocks | **~2.2 M** (40× larger; the gap was *narrowing* monotonically) |
| datatype | int32 only | **int16 too** |
| sizes | 32, 256 | **32, 64** (64 is the one never tried, and the cheapest step off one warp) |

Four arms: `{32, 64} × {i32, i16}`, phase-synced, at the real grid.

**Why 64 specifically.** The recorded objection to bigger blocks is that each
`(i,j)` search is bounded by `MAXLOOP = 30`, so only one warp of work exists per
cell and extra threads idle at a `__syncthreads()`. That argument is strongest
against 256 and weakest against 64, which adds exactly one warp and — on sm_75,
where 16 blocks/SM and 32 warps/SM are the limits — is the step that lifts the
occupancy ceiling from 50 % to 100 %. It is the only untested size where the
recorded reasoning and the occupancy arithmetic disagree.

### 3c. NCU, now that it can actually match the kernel

With the `-k` pattern fixed, collect for i32 and i16 at the same chunk width:

- `achieved_occupancy` — is 50 % actually the ceiling in practice?
- `dram__throughput` and `lts__t_sector_hit_rate` — is it at the DRAM roof like
  `modular_decomposition_kernel`, or is L2 the story?
- warp stall reasons — long-scoreboard (memory) vs barrier (the `__syncthreads()`
  the block-size argument turns on).

**The single most informative counter is L2 hit rate as a function of chunk
width.** If §1's story is right, widening the chunk should visibly cost L2 hits
on `my_c` while DRAM throughput stays flat.

## 4. What would change if §1 holds

1. **`MIN_GPU_BATCH` and the VRAM budget stop being purely "fit more, go
   faster".** There would be a locality cost to chunk width that no current model
   accounts for, and the quarter-budget result (fastest *and* smallest) would
   have a second reason behind it beyond the pipeline's `(k−1)/k`.
2. **int16's ledger improves.** Its −60 s in `modular_decomp` would stand while
   part of its +30 s in `int_loop` would be reassigned to chunking.
3. **The build pipeline inherits the question**, since it too changes how much is
   resident at once.

None of that is actionable until 3a runs. It is four arms.

## 5. Caveats, stated up front

- **§0 is two data points.** The 1.273 / 1.284 correspondence is suggestive, not
  evidence. It is the reason for the experiment, not a result of it.
- Chunk count and records-per-chunk are not independent of anything else the
  chunker does; forcing a budget changes VRAM headroom too, which is why 3a runs
  *both* datatypes at *both* widths rather than moving one arm.
- Everything here is phase-synced, and phase-sync at this scale was measured to
  cost ~0 % (§19.4) — but that was at one chunk width. If a synced run at a
  different width costs something, say so rather than comparing across.
