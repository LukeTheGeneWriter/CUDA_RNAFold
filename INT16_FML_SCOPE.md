# int16 `fml_j`: measured, viable, and bit-exact

*Scoped 2026-09-06, after the 2.7.2 port landed and the first benchmark against
upstream. Supersedes the "not started, not scoped" status of this idea.*

**Verdict: viable.** A per-block offset encoding is bit-exact with roughly 4x
headroom at 8001 nt, and — the property that decides it — the required range
does **not grow with sequence length**.

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

At B=128 the worst case is 7 510 against a *signed* int16 range of ±32767 —
about 4x headroom — while the baseline costs one int32 per 128 entries, i.e.
3.1%. Net stream **0.53x** rather than a theoretical 0.50x.

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

If the kernel stays bandwidth-bound, a 0.53x stream takes it to ~0.53x of its
time. On the 2.7.2 numbers, the 5601 nt workload's 102.96 s is roughly 73%
`modular_decomposition`; 0.53x on that share predicts **~65-70 s**, i.e. A/C
rising from 5.63x to about **8.3x** — which would close the gap with the 2000 nt
case almost exactly. That symmetry is a useful prediction to test against,
because it falls out of the model rather than being fitted to it.

## The bar

Unchanged and non-negotiable: **byte-identical output** across the standing
matrix. This encoding can meet it; coarser energy units (0.1 kcal/mol) cannot,
which is why that route stays rejected — it is a model change wearing a
data-layout costume.

*Probes: `int16_range.c`, `int16_blocked.c` (host-side, no GPU, no kernel
change).*
