# The 50 % occupancy ceiling: where it comes from and what stands in the way

*Written 2026-09-16 against `3e1d9b0d`, from the §34 Scaling run
(`scaling_a100_ncu.json`) and the register census in `registers(4).json`
(CUDA 12.8, Colab).*

---

## 1. Which kernel this is about

Only one. The two hot kernels are in completely different regimes at the
production shape (200 × 5601):

| | occupancy | its own ceiling | DRAM | verdict |
|---|---|---|---|---|
| `int_loop_warp_kernel` | **45.6 %** | **50 %** | 4.7 % | **ceiling-bound, latency-bound: this document** |
| `modular_decomposition_kernel` | 68.9 % | 75 % | **82.8 %** | bandwidth-bound — occupancy is not its lever |

`int_loop` is at **91 % of a ceiling it cannot cross**, with `long_scoreboard`
at 4.31 and DRAM at 4.7 % of peak: a kernel waiting on memory latency with no
more warps available to hide it. `modular_decomposition` is within ~20 % of the
memory roof (§34.1), so filling its remaining lanes would only ask for bandwidth
it does not have. **Everything below is about `int_loop`.**

## 2. Where the 50 % comes from — and it is not registers

sm_80 gives an SM **65 536 registers, 64 warps, and 32 blocks**. The warp kernel
runs **one warp per block** (`INT_LOOP_WARP_DEFAULT_BLOCK_SIZE = 32`, one cell
per warp), so:

> **32 blocks/SM × 1 warp = 32 warps of a possible 64 = 50 %, before a single
> register is allocated.**

`ncu` says exactly this: for the `bs32` arm the four limiters read
`blocks 32 / registers 40 / shared_mem 32 / warps 64` — **blocks** is the
smallest, and it is a hardware constant. There is no shared memory in this
kernel at all (the warp-scan design carries its state in registers and
shuffles), so that limiter never binds.

**The consequence that decides the whole question:**

| regs/thread | warps/SM allowed by registers | **c = 1** | c = 2 | c = 4 | c = 8 |
|---|---|---|---|---|---|
| 58 | 32 | **50.0 %** | 50.0 % | 50.0 % | 50.0 % |
| **48 (today)** | 42 | **50.0 %** | **65.6 %** | 62.5 % | 62.5 % |
| 40 | 51 | **50.0 %** | **78.1 %** | 75.0 % | 75.0 % |
| 32 | 64 | **50.0 %** | **100 %** | 100 % | 100 % |

**At one warp per block the answer is 50 % for every register count.** Cutting
registers alone buys nothing. The two levers only work together, and the first
one has to be more warps per block.

## 3. The measurement that closed this question was taken in a regime where the
prize did not exist

`int_loop.cu:1627-1651` records the sweep that made one cell per block the
default, on an A100 at 400 × 5601:

```
 32 threads = 1 cell   20.65 s   <-- best
 64         = 2        20.86 s   +1.0%
128         = 4        21.35 s   +3.4%
256         = 8        22.88 s  +10.8%
```

and explains it correctly: *"the BLOCK is still the allocation and retirement
unit: a block holds its registers until its LAST warp finishes, and cells have
wildly different candidate counts."* Then it adds, in parentheses:

> *"(The occupancy ceiling is identical either way — at **58 regs/thread** sm_80
> allows 32 blocks × 1 warp or 16 blocks × 2 warps, 32 warps both times.)"*

**That parenthetical is the whole finding, and it is no longer true.** At 58
registers every value of *c* yields 32 warps, so the sweep could only ever
measure the **cost** of wider blocks and never their **benefit** — the row of
the table above where all four columns read 50 %.

**Today's toolkit reports 48 registers for every instantiation**
(`registers(4).json`, CUDA 12.8, all twelve `int_loop_warp_kernel<c,G,W>`
variants). At 48, `c = 2` is **42 warps against 32 — a 31 % increase in resident
warps** — against a coupling cost the same sweep measured at **+1.0 %**.

**The sweep's conclusion should not be carried forward without re-running it.**
It is not wrong; it answered a different question than the one now being asked.

## 4. The obstacles, in the order they will bite

**(a) Retirement coupling — the real one, and it is measured.** A block keeps
its registers until its last warp exits. Cells in a row differ wildly in
candidate count, so a block of *c* cells runs at the speed of its slowest.
Measured cost: +1.0 % at *c* = 2, +3.4 % at 4, **+10.8 % at 8** — monotone, and
in a regime where nothing was gained in exchange.

*What would fix it:* put cells of **similar width** in the same block. That is
the compaction idea from `PORT_SCOUT_COMPACTION_SCOPE.md` arriving from the
other direction — not to fill idle lanes (§34.4 measured those at 83 % full)
but to **decouple retirement**. Binning by candidate count is the same machinery
and a much better motivated use of it.

**(b) Registers, but only as the *second* limiter.** At *c* ≥ 2 the binding
resource becomes 48 regs/thread. Shedding 8 (to 40) takes the ceiling to 78 %;
shedding 16 (to 32) reaches 100 %. `__launch_bounds__` will force either, and
**a spill in this kernel is worse than the occupancy it buys** — it is
latency-bound, and a spilled operand becomes another dependent local-memory
load. Any attempt must watch `local_load`/`local_store` and
`l1tex__t_sectors_pipe_lsu_mem_local_op_ld`.

**(c) The register count is a property of the TOOLKIT, not the source.** 58
under the toolkit that ran the closing sweep, **48** under CUDA 12.8, 48 locally
under 12.4. Every number in §2 moves with it, so a change tuned to 48 may be
mis-tuned elsewhere, and the ceiling arithmetic must always be quoted with the
toolkit that produced it.

**(d) The waste guard gets harder to satisfy as *c* grows.** H6's 2-D grid
launches `nfiles × ceil(maxw / c)` blocks against `sum(w_H) / c` cells of real
work, and the guard declines it above 25 % waste
(`INT_LOOP_GRIDY_WASTE_NUM/DEN`). Raising *c* raises the waste for the same
raggedness, so wider blocks make the 2-D grid unavailable more often — and H6 is
worth −5.5 % of `int_loop`. **These two changes interact and must be measured
together, never one at a time.**

**(e) And the prize is bounded by the phase.** `int_loop` is **19 % of wall**.
If +31 % resident warps converted perfectly into latency hiding — it will not —
the kernel's `long_scoreboard` share is 4.31 of ~13 total stall cycles, so a
generous reading is 10–20 % of the kernel, i.e. **2–4 % of wall**. This is worth
doing because it is cheap to test, not because it is large.

## 5. What to run, in order

**E1 — re-run the *c* sweep at production scale on the current toolkit, with
`ncu` confirming achieved occupancy per arm.** Four arms
(`RNA_INT_LOOP_BLOCK_SIZE` = 32/64/128/256) at 400 × 5601, reporting wall,
`int_loop`, achieved occupancy, and the limiter table. This is the decisive
experiment and it needs no code: it asks whether 42 warps at +1.0 % coupling
beats 32 warps at zero. **Assert the register count in the same run** — the
whole question turns on it.

**E2 — if *c* = 2 wins or ties, push registers to 40 with `__launch_bounds__`**
(ceiling 78 %), and check for spills in the same profile. If spills appear,
stop: the ladder in §2 is an upper bound on what occupancy can buy, and a spill
is a certain cost against an uncertain gain.

**E3 — only if coupling still dominates: bin cells by candidate width.** This is
the expensive one, it needs the compaction machinery, and it is the only thing
that makes *c* ≥ 4 attractive. It also has to keep lane-striding intact
(§33.1: breaking it cost 247 %).

**E4 — re-check the waste guard at whatever *c* wins.** If the 2-D grid starts
being declined on real chunks, some of E1's gain is being paid back out of H6.

**Not on this list:** anything aimed at `modular_decomposition_kernel`'s
occupancy. It is at 68.9 % of a 75 % ceiling and **82.8 % of DRAM peak** — more
warps there would queue for bandwidth that is already spoken for.
