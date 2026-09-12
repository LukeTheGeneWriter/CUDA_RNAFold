# Optimising the warp-synchronous scan — hypotheses, ranked, each with its test

*Written 2026-09-12, after `int_loop_warp_kernel` landed at −21.3 % on sm_86
(`STRESS272_RESULTS.md` §24). Nothing here is implemented. Every item says what
would be measured and how confident the premise is, because the last three
optimisation ideas in this kernel's history were all plausible and two were
wrong.*

---

## 0. THE FIRST STEP IS NOT AN OPTIMISATION

**Profile the new kernel before proposing anything for it.**

Every hypothesis below is a claim about *which stall dominates now*, and the
stall mix has certainly changed — the whole design deleted `__syncthreads()`
and both shared arrays, so `barrier` and `short_scoreboard` should be near zero.
The measured mix that motivated the rewrite was:

| stall | T4 (twin) | A100 (twin) |
|---|---|---|
| `wait` | 23.3 – 24.4 % | 19.6 – 20.1 % |
| `barrier` | 0.1 → 20.7 % | 0.0 → 19.4 % |
| `long_scoreboard` | 12.0 – 15.8 % | 22.7 – 27.5 % |
| `short_scoreboard` | 9.4 – 8.9 % | 7.6 – 6.7 % |

Reasoning from *that* table about the *new* kernel is exactly the mistake made
with the energy-table hypothesis in §20.3, which `long_scoreboard` at 12–16 %
then refuted. `tools/intloop2_addendum.py` already does this profile; it needs
one line changed to point at `int_loop_warp_kernel` and `RNA_INT_LOOP_WARP=1`.

**Two things that profile must answer before anything else:**

1. **Did `barrier` and `short_scoreboard` actually go to zero?** If not, the
   −21.3 % came from somewhere other than the intended mechanism and the design
   rationale is wrong even though the number is right.
2. **What happened to registers and occupancy?** The design *moved* `col_mask`
   and `prefix` from shared memory into registers. That is 2 more live registers
   per lane on a kernel already at 50–54 regs/thread. **Crossing 64 costs
   occupancy**, and that would silently eat part of the win — a risk, not an
   opportunity, and the kind that hides inside a net gain.

---

## H1. Hoist the cell-invariant work out of `Energy()`

**The observation.** `Energy()` is called once per candidate and recomputes
three quantities that depend only on `(i, j)` — i.e. are constant for the whole
cell, which is now the whole warp:

```c
const unsigned char type_raw = Ptype(S,pair_,H,nfiles,i,j);   // (i,j) only
const unsigned char type     = (type_raw == 0) ? 7 : type_raw;
...
const int si1 = unpack(S,H,nfiles,i+1);                        // i only
const int sj1 = unpack(S,H,nfiles,j-1);                        // j only
```

`Ptype` is two `unpack`s plus a `pair_` load; each `unpack` is an index
computation (`(H + nfiles*i)/10`), a global load and a shift/mask — roughly six
to eight instructions. So **about four redundant loads and thirty redundant
instructions per candidate**, against an `IntLoop_X` body of perhaps fifty to a
hundred. With tens to hundreds of candidates per cell that is a large fraction
of the kernel's instruction count.

**Why I am NOT confident it is a real opportunity.** Everything involved is
loop-invariant and every pointer is `__restrict__`, so `nvcc`'s loop-invariant
code motion can legally hoist all of it, and probably does. The counter-argument
is register pressure: a compiler that is short of registers will happily
*rematerialise* an invariant rather than keep it live across a long loop body,
and this loop body is long.

**The test, in order of cost.** Dump SASS for the kernel and count `LDG`s inside
the work loop (`cuobjdump -sass`); if the invariant loads are already outside,
H1 is closed for free. If they are not, hoist them by hand — pass `type`, `si1`,
`sj1` in as parameters — and A/B. Byte-identity is by construction.

**Expected value if real: HIGH. Confidence it is real: LOW-MEDIUM.** Cheap to
settle, which is why it is first.

### H1b — the `--noClosingGU` corollary

The interior-loop guard is

```c
if (P->noGUclosure && (u1 || u2) &&
    ((type == 3) || (type == 4) || (type_2 == 3) || (type_2 == 4)))
  return INF;
```

The `type` half is **cell-invariant**. When `--noClosingGU` is on and the
closing pair is GU/UG, *every* non-stack candidate in the cell returns `INF` —
so the entire enumeration could be skipped and replaced by the single stack
candidate. That is an algorithmic short-circuit, not a micro-optimisation.

It only helps one option, so it is worth doing only if H1 is done anyway (the
hoist is what makes the test cell-level). **Value: LOW in general, HIGH under
that flag.**

---

## H2. Replace `find_nth_set_bit()` with the `fns` instruction

**The observation.** `find_nth_set_bit()` (`nth.h`) is a hand-written
bit-population search: five masking/shift stages building `c2/c4/c8/c16/c32`,
then four dependent conditional accumulations. Roughly fifteen ALU operations
with a **dependency chain about six deep**, executed once per candidate.

sm_70 and later have a single instruction for exactly this — PTX `fns.b32`,
"find the n-th set bit" — reachable with inline PTX.

**Why this is attractive.** It is the purest possible attack on `wait`, which is
a *fixed-latency dependency* stall: it replaces a six-deep chain with one
instruction. It is small, local, and its correctness is testable exhaustively
offline rather than by folding — enumerate all 2³² masks is too many, but all
masks of ≤ 31 bits with every valid `n` is a few million cases and can be
brute-forced against the existing implementation on the host.

**The risks.** `fns` has subtle semantics (base/offset, and the "not found"
return of `0xFFFFFFFF`), and the existing function also returns the population
count through `c32`, which the warp kernel currently discards but the twin may
not. Both are checkable.

**Expected value: MEDIUM. Confidence: HIGH that it is strictly fewer
instructions; MEDIUM that it moves the wall**, since it is one chain among
several.

---

## H3. Stop the rarely-used energy tables from evicting the hot ones

**The observation.** `cuda_param_t` is about 256 KB, and it is dominated by
three tables that the code's own comment says are **seldom used**:

| table | size | when used |
|---|---|---|
| `int22` | ~202 KB | 2×2 interior loops only |
| `int21` | ~40 KB | 2×1 only |
| `int11` | ~8 KB | 1×1 only |
| `internal_loop`, `bulge` | ~124 B each | **every** loop |
| `mismatchI`, `mismatch1nI`, `mismatch23I` | ~800 B each | most loops |
| `stack` | 256 B | stacks and 1-bulges |

So the **hot working set is three to four kilobytes** and the cold set is two
hundred. Every rare `int22` access drags a 128-byte line into a 64 KB L1 that is
otherwise holding the hot set — and measured L1 hit rate is 72–74 %, lower than
a 3 KB working set has any right to be.

**The fix is not to stage the hot tables.** That is the move that just lost in
`modular_decomposition_kernel`, and for a reason that applies here too: the hot
tables have enormous reuse and should already be resident. The fix is to make
the *cold* accesses non-polluting — `__ldcs` (streaming, evict-first) or
`__ldg` with a non-temporal hint on `int11`/`int21`/`int22` only.

**Why not `__constant__` memory** (the idea sitting in the fine-tuning notes):
constant memory broadcasts *one address per cycle* and **serialises divergent
access**. These lookups are indexed by `type`, `type_2` and four nucleotides, so
they are maximally divergent across a warp. Constant memory would be *worse*
than L1 here. This is worth writing down because "small hot table → constant
memory" is the obvious wrong answer.

**The test.** NCU `l1tex__t_sector_hit_rate.pct` before and after, plus the wall
A/B. If L1 hit rises and `long_scoreboard` falls, the mechanism is confirmed.

**Expected value: MEDIUM-HIGH on the A100** (where `long_scoreboard` is the top
stall at 27 %), **LOWER on the T4** (12–16 %). Confidence: MEDIUM.

---

## H4. Make the column lookup cheaper than five shuffles

**The observation.** Each work item costs a five-step shuffle binary search.
But the column is **monotone in `w`**, and lane `l` processes
`w = l, l+32, l+64, …` — so across a lane's ≤ 16 iterations its column advances
by at most `maxcol ≤ 30` *in total*. An incremental walk is therefore O(1)
amortised per item, against O(5) for the search: about **80 shuffles per lane
replaced by at most 30 advances**.

**Why it is not trivial.** `__shfl_sync` must be reached by every lane in the
mask, so a `while (advance needed)` loop with a shuffle inside is undefined —
which is precisely why the search is a fixed five iterations today. Three ways
out:

1. **Keep `incl` in shared memory, per warp** (32 ints × cells-per-block — a few
   hundred bytes) and walk it with ordinary loads, where divergence is legal.
   Trades the shuffle chain for `short_scoreboard` that the rewrite just
   removed. Might still win: a monotone walk touches far fewer values than a
   restarted binary search.
2. **Fixed-bound advance with a fallback.** Advance a constant number of steps
   uniformly, detect non-convergence with `__ballot_sync`, and fall back to the
   full search only when some lane needs it. Correct but fiddly.
3. **Leave it.** If the profile says the search is not hot, this is noise.

**Expected value: UNKNOWN until §0 is done.** Explicitly gated on the profile.

---

## H5. Things deliberately NOT proposed, and why

**Bigger blocks / more cells per warp-block.** Already a free parameter
(`RNA_INT_LOOP_BLOCK_SIZE` now selects *cells per block*). Needs a sweep, not a
design — and note it is a **different quantity** from the twin's threads-per-cell,
so 64 being right there says nothing here.

**Lane-per-column instead of the flat work index.** Rejected on arithmetic:
column `c` has only `c + 1` candidate rows, so a lane-per-column split has a
worst-case imbalance of `max popcount / (total/32)` ≈ 2× even when dense, and
much worse when sparse. The flat index is why the work is balanced.

**Broadcasting `H` from lane 0 instead of 32 redundant `flatten_index_to_H`
searches.** All 32 lanes execute in lockstep and hit the *same* address, which
is a broadcast, so it costs instruction issue but no extra latency or traffic.
No gain.

**Anything about bandwidth.** `int_loop_kernel` runs at **4.0 % of DRAM peak on
a T4 and 0.8 % on an A100**, with 90–95 % of requests cache-served. Coalescing
is already 2.23 sectors/request against an ideal of 4.0. This kernel is not
memory-*bandwidth* bound on any card measured, and proposals of that shape
should be refused by default.

---

## 6. And the honest context

`int_loop` is **12 % of the A100 wall** (`STRESS272_RESULTS.md` §23.5). The
−21.3 % already banked is 2.6 % of wall. Everything above is a fraction of a
fraction: H1 at, say, 15 % of the kernel would be another 1.8 % of wall.

`build` was **54 %** of that wall and is now threaded. This document exists so
the kernel work can be resumed deliberately rather than by reflex — and §0 is
the only part of it that should happen without a reason to expect a win.
