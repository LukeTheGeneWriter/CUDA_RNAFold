# `int_loop_kernel` — scoped 2026-09-10, and the obvious lever is already closed

> **ANSWERED 2026-09-11. The lever was NOT closed: block size 64 is now the
> default, measured on two architectures. This document was wrong twice and the
> scoring is at the end — jump there, then read the body for how it got there.**

*Written after the phase-synced T4 run (`cd2b000b`, `STRESS272_RESULTS.md` §19)
made this the second-largest GPU phase. Read §19 first.*

---

## Why this is suddenly the target

| phase | async profile said | **truth (phase-synced)** | int16 delta |
|---|---|---|---|
| `modular_decomp` | 63.1 % | 60.6 % | −60.11 s |
| **`int_loop`** | **0.8 %** | **30.0 %** | **+30.44 s (+28.4 %)** |
| `hp_mb` | 29.8 % | 4.1 % | +0.27 s |

**107.3 s of a 535 s wall**, recorded as ~0.8 % for the life of the project
because its host timer only ever measured a kernel *launch*.

## The obvious lever is CLOSED, and was closed before I looked

`int_loop_kernel` launches **one block per (H,j) cell**
(`int_loop_kernel_body.inc:44`) at `BLOCK_SIZE == 32` — one warp per block —
described in the source as "the STOPGAP default" (`int_loop.cu:1206`). Shared
memory is ~256 B/block, so on sm_75 the 16-blocks-per-SM limit alone caps
occupancy near **50 %**. Raising the block size looks like free money.

**It is not, and `int_loop.cu:1152-1173` already records the measurement.**
Re-measured 2026-08-20 on an RTX 3050 via `ncu --set basic`,
`RNA_INT_LOOP_BLOCK_SIZE=32` vs `256`, grids from ~40 to ~54 000 blocks:

- occupancy at 256 **is** dramatically higher — 36–62 % against 5–33 %;
- and it was **slower at every grid size tried**, from 1.07× at small grids to
  2.35× at grid ~500, still ~1.09× slower at grid ~54 000. **Never faster.**

The recorded cause: each `(i,j)` interior-loop search is bounded by
`MAXLOOP = 30`, so the cooperative decode/prefix-sum/lookup **only ever has one
warp's worth of work per cell**. A bigger block adds `__syncthreads()` and the
cross-warp `warp_min` combine for threads with nothing to do. The comment's own
conclusion: *"Real occupancy gains for this kernel come from more **blocks** in
flight … not from bigger blocks."*

> **This is the lesson, not a footnote.** I derived the occupancy ceiling
> correctly and drew the wrong conclusion from it, because I read the STOPGAP
> comment and not the thirty lines above it that retired the idea. The knob
> `RNA_INT_LOOP_BLOCK_SIZE` exists *because* someone already went down this road.
> **Read the block-size history before proposing a block size.**

The one thing genuinely untested is scale: the largest grid measured was ~54 000
blocks and a 400 × 5601 row launches **~2.2 M**, 40× beyond it, with the gap
narrowing monotonically as grids grew. That is worth *one confirmation arm*, not
a work programme, and it is not where the 30 s is.

## The real gap: this kernel has never been profiled, and the reason is a typo

`tools/make_nb_profile272_deep.py` has been running `ncu -k int_loop_kernel`
since it was written. **There is no symbol by that name.**
`int_loop_kernel_body.inc` is `#include`d once per candidate block size and
concatenates the size on, so the real symbols are `int_loop_kernel_32`, `_64`,
`_128`, `_256` — confirmed with `cuobjdump -symbols`.

So `ncu` matched nothing, and `profile272_deep.json` faithfully records:

```json
"int_loop_kernel/i32": null,
"int_loop_kernel/i16": null,
```

beside real data for `modular_decomposition_kernel`. **The probe reported its own
failure honestly and nobody chased the null** — for three weeks, on what turned
out to be 30 % of GPU time. Same family as every entry in
`project_port27_checks_that_lied`: the probe could not reach what it claimed to
measure.

**Fixed** (this commit): the kernel list carries an explicit `ncu -k` pattern,
`regex:^int_loop_kernel_[0-9]+$`, so it survives `RNA_INT_LOOP_BLOCK_SIZE` too;
and a null is now reported as **a broken probe, not a result**.

## The int16 question: the kernel has no int16 in it

Confirmed by inspection: `int_loop.cu` and `int_loop_kernel_body.inc` contain no
reference to `d_fml_j16`, `fml_b` or `fml_decode`, and `d_my_c` is `int*`
unconditionally. The only int16 mention in the file is host-side —
`load_param()` calling `rnafold_fml_int16_vet_params()`, which validates a `-P`
table and changes nothing the kernel computes.

**So the +30.44 s cannot be arithmetic.** Both synced arms do *identical* work:
6 266 401 200 cells, 400 records. They differ only in chunk count — **i32 14,
i16 11**.

**Leading hypothesis: it tracks records-per-chunk, not the encoding.** int16
halves the fML triangle so more records fit per chunk (~36 vs ~29), but `my_c`
stays int32, so the `d_my_c` working set the kernel reads is **~24 % larger per
chunk** in the i16 arm. If `int_loop` is limited by locality on `my_c`, that is
the mechanism — and it would attach to *any* change that widens a chunk, the
build pipeline included.

This is the same shape as the `hp_mb` finding it replaced: a cost attributed to
int16 that int16 only *enabled*.

## What to do, in order

1. **Re-run the deep profile with the fixed `-k` pattern.** First counter data
   ever for this kernel: occupancy, DRAM/L2 throughput, stall reasons, i32 vs
   i16. Everything below is speculation until this exists.
2. **Test the chunk-width hypothesis** — no code needed. Run i16 at the quarter
   budget with the chunk count forced to i32's 14 (`RNA_GPU_CHUNK`, or a VRAM
   budget that yields 14), phase-synced, and compare `int_loop`. If the +30 s
   largely goes, the finding is about chunking, not int16.
3. **One block-size confirmation arm at real scale** (32 vs 64 at ~2.2 M blocks,
   phase-synced, `sha` asserted) — only to close the extrapolation gap above.
4. **Then**, and only then, consider the kernel body. Do not pre-commit: this is
   the kernel whose comment records **two** prior regressions from tuning without
   data, and it has now produced a third near-miss from reasoning without
   reading.

## Bars

Every arm must report `sha 7c0b3d633281` and a `sweep shape:` line. Compare
**synced** `int_loop` between arms; never a synced wall, never synced against
unsynced.

---

# ANSWERED 2026-09-11 — and this document was wrong twice

The run is `CUDA_RNAFold_IntLoop.ipynb`, 12 arms on a T4 at 400 × 5601, plus four
NCU CSVs. Full write-up: `STRESS272_RESULTS.md` §20. Short version, scored
against what this document predicted:

| this document said | what happened |
|---|---|
| "the obvious lever is CLOSED" — 32 stays | **WRONG. 64 wins by 10.8% on the T4 and 13–18% on an RTX 3050.** The closed sweep compared 32 against 256 and never tried 64. |
| 50% occupancy ceiling from 16 blocks/SM | **RIGHT, and now measured: NCU says 44–48%.** |
| the int16 penalty tracks records-per-chunk | **WRONG, and refuted by the arms built to test it.** Chunk width costs ~3.5 s; the datatype costs ~25 s at either width. |
| "this kernel has never been profiled" | fixed — and it is **not bandwidth-bound**: 4.2–5.2% of DRAM peak. |

## The one that matters most

**NCU says `int_loop_kernel` is identical under int16 and int32 — 717 676.8 ns
against 717 683.2 ns, and every other counter matches too.** So the +25 s in the
`int_loop` *phase* is not this kernel doing more work, and no amount of kernel
tuning will recover it. At scale every phase except `modular_decomp` is slower
under int16, including two (`hp_mb`, `load_my_c`) that touch no int16 data at
all. That is a device-level signature and it needs the probes named in §20.4,
not more work here.

## The lesson this document already tried to teach itself, restated

Its own §"The obvious lever is CLOSED" opened by correcting a wrong headline —
I had proposed 32 → 64 as a free occupancy win without reading the 30 lines
above the STOPGAP comment, which recorded an NCU sweep. Reading them closed the
question.

**They should not have closed it.** That sweep tested 32 and 256 and generalised
from the two ends of the range: *"real occupancy gains for this kernel come from
more blocks in flight, not from bigger blocks."* The generalisation is false in
the middle, and 64 — the one value that lifts the block-count ceiling without
outrunning a MAXLOOP-bounded search space — was never run.

So the sequence was: a wrong headline, corrected by reading; the correction
believed because it cited a measurement; and **the measurement did not cover the
case.** The fix is not "read more carefully" — that step worked. It is to check
what a cited sweep actually *varied* before treating it as closing a range.

## What is left here

Nothing. Block size is promoted (`INT_LOOP_DEFAULT_BLOCK_SIZE`, 64, with the
history rewritten at the instantiation site). The remaining `int_loop` questions
are in `STRESS272_RESULTS.md` §20.6, and the live one is the device-level int16
slowdown, which is not an `int_loop` problem at all.

---

## FULLY CLOSED 2026-09-11 (STRESS272_RESULTS.md 22)

The one question left after the first run -- why the phase is 25 s slower under
int16 while NCU says the kernel is identical -- is answered: **a POWER CAP.**
Both arms sit at 66 W with throttle reason 0x4; the int16 arm runs at 781 MHz
against 1044, and 97.5% of the slowdown is that clock ratio. Per-launch device
time (84.98 s vs 110.74 s) says it is the kernel and not launch overhead
(1.67 s vs 1.73 s).

**NCU locks clocks**, so it could never have seen this. Both of its numbers were
right and neither was the answer.

Scored against what this document predicted, the final tally is 1 of 4:

| this document said | outcome |
|---|---|
| the obvious lever is CLOSED -- 32 stays | **WRONG.** 64, measured on two architectures. |
| 50% occupancy ceiling from 16 blocks/SM | **RIGHT**, and now measured: 39.6% achieved at bs32, 69.8% at bs64. |
| the int16 penalty tracks records-per-chunk | **WRONG**, refuted by the arms built to test it -- and the chunk-width penalty turned out to be the SAME clock effect, not locality. |
| the kernel has never been profiled | fixed, twice: traffic, then stalls. |

And the three levers proposed afterwards are all closed with data: shared-memory
staging (premise refuted at ratio 0.55, lever neutral), coalescing (2.2
sectors/request, better than ideal), int16 `my_c` (no speed ceiling,
negative capacity benefit).

**The measured next target** is not memory at all: `wait` is 23-24% of
stall cycles, `barrier` + `short_scoreboard` 9.5% rising to 29.6% at
bs128, and `long_scoreboard` only 12-16%. A warp-synchronous scan attacks
the first two together. See STRESS272_RESULTS.md 22.5.

