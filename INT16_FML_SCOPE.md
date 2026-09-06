# int16 `fml_j`: measured, viable, and bit-exact

*Scoped 2026-09-06, after the 2.7.2 port landed and the first benchmark against
upstream. Supersedes the "not started, not scoped" status of this idea.*

**Verdict: viable, at B=64.** A per-block offset encoding is bit-exact with
~5.1x headroom at B=64 across every length, temperature and salt tested, and
— the property that decides it — the required range does **not grow with
sequence length**. It DOES grow as temperature falls, which the first pass
missed by measuring only the 37 °C default; see "Two holes" below.

---

## Why this is the target, restated with the new numbers

The v1 benchmark against ViennaRNA 2.7.2 found something counter-intuitive:

| workload | speedup over upstream |
|---|---|
| 120 x 2000 nt | **7.98x** |
| 60 x 5601 nt | **5.63x** |

The accelerator gets *worse* as records get longer, which is the opposite of the
intuition that long records favour a GPU. The explanation is that
`modular_decomposition_kernel` is at its DRAM floor (88.3% throughput, measured
by NCU), so at long lengths the device saturates bandwidth while the CPU keeps
scaling with compute.

That makes this lever specific rather than general: **it targets exactly the
regime where the accelerator is currently weakest.**

The traffic, measured earlier: 93.3 TB of loads for 1.17e13 `MIN2` ops. `fml_i`
is ~46.6 TB but served from L1 (a block's window is ~22 KB against Ada's 128 KB
L1) — a shared-memory attempt on it was tried and failed for that reason. The
irreducible DRAM stream is **`fml_j`, ~46.7 TB**: every cell reads its own
column slice and there is no reuse to exploit.

## The access pattern, from the code

`modular_decomposition_ij` (the host reference in `modular_decomposition.cu`)
walks

```c
int ij  = indx[j] + i;
int k   = i + turn + 1;
int k1j = ij + turn + 2;
for (; k <= j - 2 - turn; k++, k1j++)
    ... my_fML[k1j] ...
```

`k1j` increments while `indx[j]` stays fixed, so the stream runs **contiguously
down column j** of the triangular matrix — entries `(m, j)` for ascending `m`.

## The obstacle, and why a naive int16 fails

`INF` is 10000000 and a 5601 nt fold reaches about -186870 in units of
0.01 kcal/mol, against an int16 range of ±32767. Both the sentinel and the real
values overflow. The only bit-exact route is an offset encoding.

## Measurement 1: a per-COLUMN baseline does NOT work

Worst-case spread of finite values within one column (random sequences, MFE
folds, host-side pass over the filled `fML` — no kernel change needed):

| n | worst column spread | fits uint16 (65534)? |
|---|---|---|
| 500 | 15 510 | yes |
| 1000 | 32 940 | yes |
| 2000 | 63 930 | barely |
| 3000 | 95 480 | **no** |
| 5601 | 187 270 | **no, by 3x** |

It scales linearly with n, and at 5601 nt a single column's spread (187 270) is
essentially the whole global range (187 390). That is obvious in hindsight:
column `j` holds `fML[i][j]` for every `i`, i.e. segments `[i..j]` of every
length, so it necessarily spans from near-zero to the most negative energy in
the fold.

## Measurement 2: a per-BLOCK baseline works, with room to spare

The kernel reads the column *contiguously*, and consecutive entries differ only
by the cost of extending a segment by one base. So the number that actually
decides the idea is the spread within a block of B consecutive entries:

| n | B=32 | B=64 | **B=128** | B=256 | B=1024 |
|---|---|---|---|---|---|
| 1000 | 2 600 | 3 440 | **5 330** | 8 900 | 30 090 |
| 2000 | 2 280 | 3 560 | **6 020** | 9 460 | 32 490 |
| 3000 | 2 530 | 3 960 | **6 070** | 9 760 | 33 840 |
| 5601 | 3 410 | 5 670 | **7 300** | 10 170 | 34 750 |
| 8001 | 3 040 | 4 300 | **7 510** | 11 260 | 38 620 |

**The spread is flat in n.** 5 330 to 7 510 across an eightfold length range,
where the per-column spread grew linearly. That is what makes the encoding safe
at any length rather than up to some cliff.

At B=128 the worst case on the DEFAULT model is 7 510 against a signed int16
range of ±32767. That is the number the first pass stopped at; it is not the
design margin, because the default model is not the worst model. See the
worst-case table below, which selects B=64.

## The ordering subtlety, and why it is not a problem

`fML` fills by descending row, so entry `(m, j)` is written at row `m` and read
only at rows `i < m`. A block spanning `[base, base+B-1]` is therefore only
*partially written* while its upper part is already being read, and a baseline
defined as the block minimum would change as the block fills — invalidating
offsets already written.

The resolution is that **the baseline does not have to be the minimum.** Any
member of the block works, because what bounds the offset is the block's
*spread*, not the choice of origin. Taking the baseline to be the
first-written entry (the highest index, written at the earliest row) makes it
final the moment the block is created, and every later entry is a signed delta
from it, bounded by the spreads above.

## Two holes in the above, found by re-reading the existing width analysis

`hp_mb_loop.cu:873-903` already contains a value-range analysis, written when
the fml_scan tile width was made tunable. Re-reading it against this proposal
exposed two assumptions in the measurements above that were not tested. Both
were checked rather than argued; both hold, but the second one moved a number.

### Hole 1: "non-finite" might not be a single value — RESOLVED

That comment records that the host's two INF guards are **not symmetric**: when
`fml_prev[j]` is a real energy and `en_i` is INF, the host computes
`fml_prev[j] + 10000000` — *"a large positive number, NOT INF"* — and the INF
test is `== INF`, never `>= INF`, precisely so such values survive as distinct.

Measurement 1 lumped everything `>= INF/2` into "INF" and discarded it, then
concluded that one sentinel would do. If `fML` carried a *spread* of near-INF
values, one sentinel would be wrong.

**Measured: it does not.** Across default, 10 °C, 60 °C and salt 0.05/0.2/5.0 at
3000 nt, every non-finite cell in `fML` is *exactly* 10000000 — near-INF cells:
**0** in every model. The asymmetric-guard values are real, but they live in the
**row buffers** (`fml_prev`, the scan's `A[j]`), not in the `fML` triangle. Since
`d_fml_j` *is* `my_fML`, a single sentinel is sufficient for the buffer actually
being narrowed. The hazard is genuine; it is simply not in this stream.

### Hole 2: the spread was measured on the default model only — MATTERS

The same comment bounds the tropical sums by `length*|MLbase|` *"with a
non-default parameter file"*, i.e. it already identifies non-default parameters
as the case that stretches ranges. Everything in Measurement 2 used
`vrna_md_set_default()`.

**Measured at 3000 nt, B=128:**

| model | worst blocked spread |
|---|---|
| default (37 °C, 1.021 M) | 6 060 |
| 60 °C | 3 640 |
| salt 0.05 M | 5 401 |
| salt 5.0 M | 6 414 |
| **10 °C** | **8 837** |
| 10 °C + salt 0.05 M | 8 040 |

Temperature dominates, and in the direction that costs: **lower temperature
means stronger pairing, larger energy magnitudes, wider spreads**. 10 °C is 46%
worse than the default. Salt is comparatively mild — which is consistent with
salt entering as an additive per-loop correction rather than rescaling the
stack energies.

So the honest headroom is not the 4x quoted from the default model. See the
worst-case table below, which is what the design should be sized against.

### The worst case, which is what the design must be sized against

Length x temperature x salt, B=128, worst blocked spread:

| n | 37 °C | 20 °C | 10 °C | 0 °C |
|---|---|---|---|---|
| 2000 | 5 990 | 7 798 | 8 742 | 9 876 |
| 5601 | 6 400 | 8 117 | 9 602 | **11 041** |
| 8001 | 6 860 | 8 437 | 9 602 | **11 041** |

(at 1.021 M; salt 5.0 M adds ~1%, salt 0.05 M subtracts ~10%)

**Worst over everything tested: 11 148**, at 0 °C — **2.9x** headroom on a
signed int16, not the 4x the default model suggested. Temperature is the
dominant axis and it is monotonic: colder is always worse. Length matters much
less, and above 5601 nt barely at all.

*One oddity worth recording rather than hiding:* 5601 and 8001 give byte-identical
worst spreads at 10 °C and 0 °C but differ at 37 °C and 20 °C. The probe seeds
its RNG identically, so the 8001 sequence contains the 5601 one as a prefix;
the likeliest explanation is that at low temperature the worst-case block is a
locally-determined motif inside that shared prefix. It does not affect the
bound, but it means these two rows are not independent samples.

### Choose B=64, not B=128 — and a correction to my own arithmetic

I previously wrote the net stream as "0.50 + 0.03 = 0.53x". That is wrong: it
added the baseline overhead as a fraction of the *original* stream rather than
of the *narrowed* one. The stream is `2 + 4/B` bytes per entry against 4:

| B | bytes/entry | net stream | worst spread | headroom |
|---|---|---|---|---|
| 64 | 2.063 | **0.516x** | ~6 400 | **~5.1x** |
| 128 | 2.031 | **0.508x** | 11 148 | 2.9x |

So the real figure is ~0.51x either way — closer to the theoretical 0.50x than
I claimed, which makes the case slightly *better*. And it makes the choice
obvious: **B=64 costs 1.6% more stream than B=128 and buys roughly double the
safety margin.** Given that this codebase's integer history is a list of bounds
that held until they did not, that is the right trade. B=128 is not wrong today;
it is simply thinner than it needs to be for nothing gained.

### The residual unknown, and its mitigation

A user-supplied parameter file (`-P`) is the case the original comment names and
the one case that cannot be enumerated by sampling. The mitigation is not a
wider bound but a **runtime trap**: the narrowing in `load_fML_kernel` must
range-check and abort rather than wrap. A silent wrap here would produce a
plausible, self-consistent, wrong answer — the exact failure mode this project
keeps having to guard against, and the reason the earlier integer fixes had to
*widen* rather than assume.

## Knock-on work

- `load_fML_kernel` writes `fml_j` from `energy_min` (int32). Needs the
  narrowing plus a range check that **traps rather than wraps** — silent wrap is
  the one failure mode that would look like a plausible wrong answer.
- INF needs a reserved sentinel. 99.8% of cells are finite at 5601 nt, so this
  costs one value, not a branch-heavy path.
- `fetch_fML()` currently copies the device triangle straight into
  `VC[H]->matrices->fML` (int32), relying on the layouts matching. Narrowing the
  device side breaks that memcpy-shaped fetch and needs a widening conversion,
  in a phase already measured at ~9.4 s (`fetch_mx`).
- **VRAM halves for the largest buffer**, so chunks widen. That is a second-order
  win and it is now measurable: the v2 benchmark's multi-chunk arm reports the
  per-chunk penalty directly, so the value of fitting more records per chunk
  stops being a guess.
- `fml_i` should be left as int32. It is served from L1, so narrowing it buys
  nothing and would cost a widen in the hottest loop.

## What it should be worth

If the kernel stays bandwidth-bound, a 0.52x stream takes it to ~0.52x of its
time. On the 2.7.2 numbers, the 5601 nt workload's 102.96 s is roughly 73%
`modular_decomposition`; 0.52x on that share predicts **~65-70 s**, i.e. A/C
rising from 5.63x to about **8.3x** — which would close the gap with the 2000 nt
case almost exactly. That symmetry is a useful prediction to test against,
because it falls out of the model rather than being fitted to it.


## Two alternatives, closed with data

Both were proposed as ways to avoid the offset encoding entirely: represent the
energies as small discrete symbols and recover full precision from a table.
Measured rather than argued (`tools/int16_quantum.c`), because the idea is good
enough that it will be proposed again otherwise.

### A common divisor — there isn't one

Every entry in `stack37` is a multiple of 10, so it is reasonable to expect
`fML` to carry a common factor worth 3.3 free bits. **It does not: `gcd == 1` at
every temperature, including 37 °C.** The `lxc * log(size/30)` extrapolation is a
truncated float and the Ninio terms are not round, and either alone is enough to
break it.

### A global dictionary — fits at 37 °C, fails on two independent axes

| model | n | distinct values in fML | ≤ 65536? |
|---|---|---|---|
| default 37 °C | 2000 | 6 407 | yes |
| default 37 °C | 5601 | 18 549 | yes |
| **10 °C** | **2000** | **77 852** | **no** |
| 23.5 °C | 2000 | 62 529 | barely |
| 10 °C + salt 0.05 | 2000 | 79 078 | no |

- **Temperature.** Off 37 °C every parameter table picks up its own truncation
  residue, sums go dense, and the count blows past 65536 at n=2000 — a *shorter*
  sequence than the default model handles comfortably.
- **Length.** Even at 37 °C the count is roughly linear in n, so the default
  model would exceed 65536 somewhere near 20 000 nt.

A 16-bit symbol space whose validity depends on both temperature and sequence
length is a coincidence, not a representation.

### Deferring evaluation to the end cannot work at all

Independent of the counts. This is a **min-plus** DP: choosing
`fML[i][j] = min(...)` requires a *numeric comparison at every cell*. "Evaluate
once we know what the equation looks like" inverts the dependency — the equation
is precisely what the comparisons select, so deferring means carrying the whole
search space.

Two further blockers even where the vocabulary fits: symbols do not add
(`sym(a) + sym(b) != sym(a+b)`, so the hot loop would need a dependent lookup
into a 256 KB table, in a kernel that is *already* DRAM-bound), and the
dictionary would have to exist before the sweep that produces the values.

**What the idea was right about:** the sparsity is real — 6 407 distinct values
across a 60 270 span at 37 °C, about 10% density. That is exactly *why* the
blocked spread comes out near 6 000 instead of near 60 000. The structure is
there; it is simply not a small enough alphabet to index.

## A provable bound, and a better mitigation than a runtime trap

The most negative `stack37` entry is **−340** (CG/GC, 0.01 kcal units). A window
of B consecutive positions admits at most B/2 stacked pairs, so the offset is
bounded by `B/2 × 340` — **an expression with no `n` in it**:

| B | provable bound | measured worst | int16 ceiling |
|---|---|---|---|
| 64 | **10 880** | ~6 400 | 32 766 |
| 128 | **21 760** | 11 148 | 32 766 |

So the encoding is length-independent **by construction**, not by extrapolation
from an 8001 nt sample. The bound counts stacking only — tetraloop bonuses and
dangles also contribute — so it is the right shape of argument rather than a
finished proof.

That suggests a better mitigation than the runtime trap proposed above.
Temperature and a `-P` file are both fully baked into the parameter tables by the
time `vrna_params()` returns — temperature is applied to the ~50 tables once at
init and **never reaches the DP** (`grep temperature` in `mfe/mfe.c` finds
nothing), and it is affine, so it commutes with summation. Therefore the worst
per-base negative contribution is knowable *for the loaded table*, before any
kernel launches:

> **assert `B/2 × worst_negative_contribution < 32766` once, at startup.**

That converts the one case sampling cannot cover into a refusal up front rather
than a wrap caught mid-fold — which matters, because a silent wrap here produces
a plausible, self-consistent, wrong answer.

## Modified bases do not interact with this

`constraints/soft_special.h` — *"specialized implementations that utilize the
soft constraint callback mechanism"*. Modified bases are **soft constraints, not
an enlarged alphabet**: the sequence stays 4-letter with a declared "fallback
base". So `ptype`, `S` and the pair tables are untouched, and the encoding — which
is over energies, not symbols — cannot see them. They are already declined by
`fc->sc != NULL` in the library guard and `opt->mod_params` in the driver.

They could only ever stretch the range if they were brought onto the device,
which is a Tier 3 decision in its own right, and at that point they fall under
the same load-time check as `-P`.

## The bar

Unchanged and non-negotiable: **byte-identical output** across the standing
matrix. This encoding can meet it; coarser energy units (0.1 kcal/mol) cannot,
which is why that route stays rejected — it is a model change wearing a
data-layout costume.

*Probes: `int16_range.c`, `int16_blocked.c` (host-side, no GPU, no kernel
change).*
