# Scout-ahead compaction: group the cells that need work, before the warp does

*Design question recorded 2026-09-16 (Luke). Not scheduled, not costed, not
refuted — written down so it is not re-derived from scratch later.*

---

## 1. The question, as asked

> While the main process is sweeping rows at increasing `k`, does it make sense
> to send a scouting process ahead at `k+1` that performs non-serial
> calculations? The logic: for each `ij` in the matrix, the algorithm evaluates
> the bitmask (is this a legal pairing), making a warp highly prone to
> divergence — one `ij` in a `k` row may require a sprawling calculation while
> another is trivial because the pairing is illegal. Grouping the cells that
> need real work into a warp would keep everything running smoothly and free up
> resources that would otherwise sit idle taking up a warp lane on a trivial
> calculation.

## 2. Why this is pointed at the right thing

§33.1 measured both hot kernels as **latency-bound**, not bandwidth-bound:
`long_scoreboard` is 11.67 cycles per issue-active in
`modular_decomposition_kernel` and 5.89 in `int_loop`, against 7.9 % and 1.6 %
of DRAM peak. A latency-bound kernel is one that does not have enough
*independent work in flight*, and a lane idling behind a lane that is doing a
sprawling interior-loop search is precisely work that is not in flight.

Two measurements already say the shape of the workload is ragged in exactly the
way the question assumes:

- **The waste guard exists because of it.** H6's 2-D grid is *declined* on
  ragged input (§B): `nfiles 29, maxw 1, blocks 29 vs flat 1 — waste 29.00×`.
  That guard is a measurement of how much a rectangular launch over a ragged
  row wastes.
- **`int_loop` already compacts, one cell at a time.** The kernel builds a
  per-column bitmask of legal `(p,q)` candidates, prefix-sums it, and hands
  thread *t* the *t*-th set bit (`find_nth_set_bit`). So compaction *within* a
  cell is already the design. The question asks for compaction *across* cells,
  which is the level nothing does today.

## 3. What is actually known about the divergence

Nothing direct, and that is the first thing to fix. What §33.1 gives:

| | `int_loop` (`base`) | `modular_decomp` (`md_t32`) |
|---|---|---|
| `not_selected` | 0.373 | **2.207** |
| `no_instruction` | 0.283 | 0.767 |
| `branch_resolving` | 0.660 | 0.380 |
| `long_scoreboard` | 6.003 | **11.667** |

`not_selected` at 2.2 in `md` means warps were *ready and not chosen* — a
scheduler-contention signature, not an idle-lane one. **None of these is a
divergence metric.** The metric that would answer it directly is
`smsp__thread_inst_executed_per_inst_executed.ratio` (average active lanes per
issued instruction, 32 = perfect). That is a one-line addition to the notebook's
§G probe and it should be measured **before** any of this is designed further.

## 4. The two shapes this could take

**(a) A scouting kernel at `k+1`.** A cheap pass over row `k+1` that evaluates
only the bitmask and writes a compacted list of the `ij` that need real work,
while row `k` is still folding. The main kernel then reads that list instead of
a range.

- The dependency question is the whole of it: the legality bitmask for `(i,j)`
  is a function of the *sequence* and the *hard constraints*, not of any DP
  value, so a scout **can** run arbitrarily far ahead. That is a real asymmetry
  and it is why the idea is not obviously blocked: what is serial in this
  algorithm is the energies, not the legality.
- It costs a second kernel launch per row and a buffer, against rows that are
  already short at the end of the sweep.

**(b) No scout at all — compact in the launch.** If the legality bitmask is
known without any DP state, the *host* (or a single prologue kernel per chunk)
can build the compacted index for every row up front, once, and every row launch
becomes a dense grid over exactly the cells that need work. This is the same
trick as H6's `blockIdx.y` and the same reason it worked: a question the host
can answer should not be asked per warp.

**(b) looks strictly better than (a) unless the compaction itself is expensive**,
and it has no concurrency hazard at all. The scouting framing is worth keeping
because it is what makes (b) visible — the observation that legality is not
serial is the load-bearing part, not the pipelining.

## 5. What would have to be true for it to pay

1. **Lanes are actually idling.** Measure active-lanes-per-instruction first
   (§3). If `int_loop` is already near 32, compaction across cells buys nothing
   and the ragged work is inside the candidate search, which already compacts.
2. **The compacted grid does not break coalescing.** §33.1's tile sweep is the
   warning: `RNA_MD_TILE=1` costs **247 %** and takes sectors per request from
   2.04 to 10.99, because 32 lanes striding one row together is what makes the
   loads coalesce. A compaction that gathers cells from scattered `j` would
   scatter their loads too. **Any design here must keep lanes on adjacent `j`.**
3. **The saving survives the guard.** The waste guard already declines the 2-D
   grid when a row is ragged, so some of this prize is being collected today by
   simply not launching the empty cells.

## 6. Where it sits against everything else

`modular_decomp` is 44 % of wall and `int_loop` 19 %, and both are
latency-bound, so the ceiling on "keep the lanes busy" work is large. But
`PORT_ROOFLINE_SCOPE.md` §7 already lists two cheaper things with the same
target — measure the wave count at production shape, then unroll the `y` scan
for independent loads — and both are strictly smaller than a compaction
rewrite. **This is the third item on that list, not the first.**

**Do first:** add `smsp__thread_inst_executed_per_inst_executed.ratio` to the
§G probe. It is one metric, it costs nothing, and it decides whether any of
this is worth designing.
