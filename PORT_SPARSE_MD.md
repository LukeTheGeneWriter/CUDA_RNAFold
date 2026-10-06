# Sparse multiloop decomposition (C1): plan for sign-off

Written 2026-10-05 on `Lukes_Flow_Batching` (`ae698bcf`), campaign C1 of the LFB roadmap.
Nothing here is built yet. Section 9 lists the decisions that need Luke.

## 1. Why

`modular_decomposition` (md) costs **31.2 s of LFB's 66 s** at 400 × 5601 on the A100. It is
**bandwidth-bound**: 82.8 % of DRAM peak, 0.84 ops/byte. Every lever tried so far streams the same dense terms
more cheaply: int16 storage (−20.7 %, shipped), the corner cache, pruning, and the blocked/tiled driver
(md −12 %, but the tile path's per-step floor lost the wall; paused). Sparse folding (Wexler et al. 2007;
Backofen et al. 2011; Will & Jabbari, *SparseMFEFold*, AMB 2016) **skips the terms that cannot win**.
SparseMFEFold's predictions are identical to `RNAfold -d0`.

## 2. The rule, adapted to our md

Our md does not use the textbook form. For row `i` and column `j`:

    DMLi(i,j) = min over k of  A(i,k-1) + fML(k,j)

`A` is row i's fML **without the split term** (`energy_min` after the fML scan, uploaded by `load_fML`).
`fML(k,j)`, `k > i`, comes from finished rows. Wexler's candidates need the *full* left operand, which would
serialise the row, so they do not apply as written. The variant below needs only one property of `A`:
`A(i,k) <= A(i,k-1) + MLbase` wherever `k` may be unpaired in a multiloop. The fML scan builds `A` with
exactly that term.

**Candidate.** Split point `k` of column `j` is a candidate unless either holds:
- `up_ml(k)` and `fML(k,j) >= fML(k+1,j) + MLbase`. Split `k` is then dominated by split `k+1`, since
  `A(i,k-1) + MLbase + fML(k+1,j) >= A(i,k) + fML(k+1,j)`.
- `up_ml(j)` and `fML(k,j) >= fML(k,j-1) + MLbase`. The term is then dominated by the scan term below,
  since `A(i,k-1) + fML(k,j-1) + MLbase >= DMLi(i,j-1) + MLbase`.

**Recurrence.**

    DMLi(i,j) = min( up_ml(j) ? DMLi(i,j-1) + MLbase : INF ,
                     min over candidates k of A(i,k-1) + fML(k,j) )

The scan term is never below the true value: `fML(k,j) <= fML(k,j-1) + MLbase` whenever `j` may be unpaired.
So the recurrence is **value-identical** cell by cell, not merely the same structure. Turner 2004 has
`MLbase = 0`, which makes the scan a plain running minimum along the row.

The `up_ml` conditions are what make this safe under `-C`. Without them, a column whose `j` is forced to pair
would drop a non-candidate whose dominating term does not exist.

## 3. Evidence so far

Measured on upstream 2.7.2's own matrices, from random ACGU sequences with defaults (Turner 2004, d2).
Tools are in the session scratchpad `lit/` and move into `tools/` at S0.

| length | candidate share of finite `fML(k,j)` | per column | md right-operand terms |
|---|---|---|---|
| 500 | 6.86 % | 16.8 | 14.9× fewer |
| 2000 | 7.10 % | 70.7 | 14.8× fewer |
| 5601 | 7.11 % | 198.7 | 14.3× fewer |

**Exactness.** Dense against sparse on every cell of 2,335,800 (300 nt × 20, 1200 nt × 2): **0 differ**.
**Negative control:** with the scan term dropped, **591,944 of 897,000 differ**. This was checked with the
full left operand. The GPU's nosplit operand relies on the same inequality, and S0 re-checks it in that form.

**S0 done (2026-10-05): `tools/md_sparse_equiv.sh BUILD_DIR`.** It runs 11 option cases, each with
upstream's full fML row as the left operand and with a synthetic row that has the chain property and nothing
else. That second form is what speaks for the GPU's nosplit operand. Cases:
- plain, `--noLP`, `--circ`, `-d0`, salt 0.2, `-T 25`, `--maxBPspan 150`;
- `-g` on G-rich input;
- `-C` with `|` positions, with and without `--enforce`;
- `-C --enforce` with an outer pair and `x` stretches.

**All 22 hold, 0 cells differ,** against both stock 2.7.2 and LFB's own build. The negative controls bite:
- scan term dropped: **717,263 of 1,022,175** cells wrong;
- `up_ml` ignored: **168,086 / 177,190 of 279,400** wrong (full / chain operand).

Trap found on the way: **without `--enforceConstraint`, upstream does not apply `|` at all** (`up_ml` stays
full), so a `-C` fixture that omits `--enforce` cannot reach the hc clause. Forced pairs make md denser:
6.8× fewer terms instead of 15–18×.

## 4. GPU design

**4.1 Candidate lists.** There is one append-only list per (record, column `j`) of `(k, fML(k,j))` entries.
Rows run in descending `i`, so row `k` appends at most one entry to each column, after its fML values are
final. A list is therefore in descending `k`, and the newest entries are the smallest `k`. A row-`i` cell
reads the list from the start and stops at `k < i+turn+2`, which skips at most `turn+1` trailing entries.
Each column gets one append per row, so the append needs **no atomics**: the thread that owns `(k,j)` writes
slot `len[j]` and bumps it.

**4.2 The append kernel** runs once per row, after `fml_prev` holds the row. The test needs only **row-shaped
data**: `fML(k,j)` and `fML(k,j-1)` from the current row, `fML(k+1,j)` from the previous one. So the lists
never read the triangle. The kernel is elementwise over the row's cells and coalesced.

**4.3 Sparse md cell.** The cell walks its column's list, with `TILE` lanes striding entries, so the reads are
coalesced within a list. It gathers `A(i,k-1)` from the row buffer `fml_i`: 22 KB per record at 5601 nt, and
all 400 records' rows (8.8 MB) stay L2-resident on the A100. A min-reduce produces the sparse term `S(i,j)`.
Sums are INF-guarded, so the result matches upstream rather than the dense path's near-INF artefacts
(`md_cell.inc`'s clamp note).

**4.4 Row scan.** `DMLi(i,·)` is a prefix scan of `(min, +MLbase)` along the row, reset where `!up_ml(j)`. It
reuses the associative composition the fML scan already uses (`fml_scan_block.inc`). It then writes `dml` and,
for circular folds, `fm2`, as today.

**4.5 Capacity and fallback.** Each column gets capacity `min(j, ceil(rho*j) + pad)` from an offsets table
(like `d_colb_off`). The default `rho = 0.25` is 3.5× the measured density. A column that overflows is
**flagged dense**, and its cells take today's dense path, which reads the triangle. The banner prints the
number of overflowed columns, so the fallback has positive evidence that it ran.
`gpu_bytes_per_file()` charges the lists so that chunking stays honest.

**4.6 Entry format: both, measured** (section 9.1). `int2` (`k`, value), 8 B, is 0.56 B per dense term
against int16's 2 B, about 3.6× fewer bytes. Packed, 4 B (`uint16 k` + int16 delta on the existing
per-64-row baselines), is about 7× fewer. Selected by `RNA_MD_SPARSE_ENTRY=8|4`. The 4-B form also needs
`L < 65536`; longer records use 8 B.

**As built (S3), the 4-B form differs from that sketch:**
- **Its own baselines, not int16 storage's `d_fml_b`.** Those are written when the triangle is packed, and
  `RNA_ROW_BATCH` delays that past the append.
- **One baseline per 32 slots of a list,** which is one warp step of the readers.
- **Each entry stores its step from the previous entry,** and the reader decodes with a warp prefix sum.
  Deltas from the baseline did not fit: 32 candidates span hundreds of rows of `k`, and plain 1700-nt input
  overflowed int16 in about 300 columns.
- **A step that still does not fit closes the segment early.** A column's first candidates sit next to `j`
  (fML about +300), and the next can be about −35,000. The rest of the segment is padded with `k = 0`, which
  is never a split point.
- **Capacity is counted in slots.** A column gets its capacity rounded up to 32, padding included. It
  overflows when the slots run out.
- **`k > 65535` overflows only that column,** not the whole chunk, so the byte model stays per record.

**4.7 What stays.** The fML triangle stays in v1, because backtracking (`fetch_mx`) and the dense fallback
need it. Since md would no longer read it, a later stage could stream fML rows to the host instead. That frees
VRAM for wider chunks and synergises with C3 (device backtracking).

**4.8 v1 refusals,** each with a printed reason, falling back to the dense md:
- the tile/blocked path (`RNA_MD_BLOCK`, `RNA_MD_TILE*`);
- the megakernel;
- slot flow and continuous flow;
- `RNA_MD_PRUNE`;
- soft constraints (already declined by the engine).

Compatible: int16 storage (no longer read by md), `ROW_BATCH`, the c ring, stream overlap 1/2, graph capture
(the append and the scan are two more fixed-shape launches per row).

## 5. Bars

1. **S0 tool:** `tools/md_sparse_equiv.c`, the theorem on upstream's matrices. It uses the nosplit left
   operand and covers plain, `--noLP`, `-g`, `--circ`, salt, `-d0`, `-C --enforceConstraint` and
   `--maxBPspan`, plus the negative control (scan dropped → red).
2. **Device selftest** `RNA_MD_SPARSE_SELFTEST=1`: dense and sparse computed side by side per cell, mismatches
   counted. `RNA_MD_SPARSE_NEGCTL=1` drops the scan term and must go red **in the triangles**, because the 3c
   lesson was that structures can stay the same when a term is dropped.
3. `RNA_TRI_CHECKSUM` c/fML triangles against `RNA_GPU=0` on the parity cases.
4. Then the usual bars: 45-option parity, the default matrix, the binding suite, and a mixed-length soak.
5. **Forced fallback:** `RNA_MD_SPARSE_RHO=0` (every column dense) and a tiny `rho` (a mix) must be
   byte-identical, with the overflow banner as evidence.
6. **Adversarial density:** homopolymers, `(GC)n`, `(AU)n`, designed hairpin arrays. Report the density;
   the fallback must engage, never the answer change.
7. **Natural sequences:** the density on a real-RNA fixture. Literature says sparser than random, but our
   workload's matters.

## 6. Measurement

- **Laptop:** ncu md bytes per launch and md time, sparse against row path, at 2000 and 5601 nt over several
  record counts. Per-launch denominators.
- **A100 notebook:** 400 × 5601, 3000 × 1200, 1200 × 1200 and the soak, ABBA × 3, with a P arm for phase
  attribution.
- **Matrix:** entry 8 / 4 B × `rho` 0.10 / 0.25 / 0.50 at several lengths (e.g. 600, 1200, 2400, 5601 nt),
  plus the row path and the best blocked arm (CB 1024) for the retirement test in section 9.
- **No pre-registered threshold** (section 9.5). The md phase, wall, spread and overflow counts are reported
  and judged by eye. Every correctness bar must be green regardless.

## 7. Stages

| stage | content | bar |
|---|---|---|
| S0 | tool in tree, hc/option coverage, nosplit form | 0 mismatches + negctl red |
| S1 | lists + append kernel; sparse computed BESIDE dense (selftest only) | selftest 0, negctl red |
| S2 | sparse md + row scan replace dense under `RNA_MD_SPARSE=1` | triangles + parity |
| S3 | capacity (`RHO`), overflow fallback, VRAM accounting, 4-B packed entries (`ENTRY=4`) | forced-fallback bars, both formats |
| S4 | laptop matrix: entry × rho × length | ncu bytes per launch, md time |
| S5 | A100 notebook: the matrix + the blocked-driver retirement test → default decision | section 6, by eye |
| S6 (C3) | md stops needing the device fML triangle; stream rows to the host | wider chunks |

**S3 done (2026-10-06),** laptop, 8 records of 300–1700 nt (plus G-rich and enforced `|` fixtures):

- **Both formats are exact everywhere.** Output equals the dense md and `RNA_GPU=0`, and the c/fML triangle
  hashes equal the dense md's. The cases are plain, noLP, circ, d0, salt 0.2, T25, maxBPspan 150, `-g`,
  int32, `ROW_BATCH`, overlap 1, graphs, and rho 0 / 0.02 / 0.50.
  - Under `-C --enforceConstraint` the output is exact. The triangle hashes still differ, from the dense
    md's near-INF artefacts, as in S2.
- **Forced fallbacks are byte-identical.** At rho 0, 4,556 (8 B) and 2,723 (4 B) columns overflow. With
  `RNA_MD_SPARSE_DELTA_MAX=50` and `=0` (test only), 5,506 and 5,619 columns overflow.
- **Selftests:** 4,772,288 cells, 0 mismatching in either format. NEGCTL gives 3,421,290 / 3,421,310
  mismatching, and output and triangles change.
- **Overflowed columns at rho 0.25:** 39 with 8-B entries against 13 with 4-B entries, plus 828 padded slots.
  Rounding to whole segments gives the 4-B form spare slots.
- **Lists at 8 records, 1700 nt max:** 9.8 MB with 8-B entries, 5.6 MB with 4-B entries.
- **Admission:** `gpu_bytes_per_file()` charges the lists. Every chunk compares its allocation with the
  charge and warns on a shortfall. Allocation is at most the charge for one chunk, budget-forced chunks and
  single-record chunks, in both formats. A halved charge fires the warning.
- **Binding suite:** 20/20 with the dense md and with each format. The binding reaches the sparse md
  (banner seen, equal to `fold_compound`).

**S4 done (2026-10-06),** laptop RTX 3050 (4 GB, power-capped). Random sequences, one chunk per run,
`RNA_PHASE_SYNC=1`. The headline trio ran in ABCCBA order (dense, 8 B, 4 B at rho 0.25) and is reported as
pair means. **Every arm's output equals the dense md's** at every length.

| L × records | md dense | md 8 B | md 4 B | wall dense | wall 8 B | wall 4 B |
|---|---|---|---|---|---|---|
| 600 × 400 | 0.635 s | 0.440 s (1.44×) | 0.477 s (1.33×) | 2.98 s | 2.71 s (−9 %) | 2.66 s (−11 %) |
| 1200 × 120 | 0.975 s | 0.569 s (1.71×) | 0.614 s (1.59×) | 3.65 s | 3.17 s (−13 %) | 3.21 s (−12 %) |
| 2400 × 32 | 1.589 s | 0.741 s (2.14×) | 0.813 s (1.95×) | 4.82 s | 3.86 s (−20 %) | 3.92 s (−19 %) |
| 5601 × 8 | 4.47 s | 1.634 s (2.74×) | 1.848 s (2.42×) | 10.16 s | 6.76 s (−33 %) | 6.95 s (−32 %) |

- **5601 drifted:** the SM clock fell from 1732 to 1357 MHz, and the second dense run's `int_loop` rose from
  2.98 to 3.57 s. Normalised by `int_loop`, md is **2.28× (8 B) and 2.01× (4 B)** faster. The true figure
  lies between that and the raw ratio.
- **`int_loop` is not a perfectly independent control here.** It is 4–8 % lower in every sparse arm at
  600–2400 nt, where nothing drifted, plausibly because the dense md no longer evicts its data from L2.

**ncu per launch,** 5601 × 8, row i ≈ 2800, three launches each, means:

| | DRAM read | time | effective rate |
|---|---|---|---|
| dense `modular_decomposition_kernel` | 66.5 MB | 0.815 ms | 82 GB/s |
| 8 B: cell + scan + append | 19.0 + 0.12 + 0.36 = 19.5 MB (**3.4× fewer**) | 0.215 + 0.016 + 0.026 = **0.257 ms (3.2×)** | cell 88 GB/s |
| 4 B: cell + scan + append | 10.4 + 0.12 + 0.39 = 10.9 MB (**6.1× fewer**) | 0.313 + 0.016 + 0.027 = **0.356 ms (2.3×)** | cell 33 GB/s |

The byte ratios are what section 4.6 predicted (3.6× and ~7×). The DRAM bytes check out against the list
lengths: about 0.07 × 1400 entries × 8 B × 22.4 K cells ≈ 17.5 MB.

**Findings:**
1. **Sparse md wins at every length, and the win grows with length.** md is 1.44× faster at 600 nt and
   2.3–2.7× faster at 5601 nt. Wall drops 9 % to 33 %.
2. **8 B is faster than 4 B everywhere** on this card, by 7–12 % in md. Once sparse, the cell kernel is no
   longer bandwidth-bound, so the 4-B form's warp decode costs more than the bytes it saves. The 4-B form's
   value is VRAM: its lists are 47 % smaller (128 MB against 243 MB at 5601 × 8), which buys batch width.
   S5 decides which matters more on the A100.
3. **rho:**
   - 0.10 is slower: 2.19 s against 1.63 s at 5601 (8 B), because 11,794 columns overflow to the dense
     path.
   - 0.50 is no faster than 0.25 and doubles the lists (482 MB against 243 MB).
   - **0.25 stays the default.** Density is 6.5–7.2 % at every length.
4. **The gap to the 14× term count** comes from latency, not bandwidth:
   - one warp per cell, with short lists near the diagonal that leave lanes idle;
   - three launches per row instead of one;
   - the per-record scan (16 µs per row).

   These are S5+ candidates if the A100 agrees.

**S5 notebook written (2026-10-06): `CUDA_RNAFold_SparseMD.ipynb`** (repo root, gitignored like the others;
generator `make_nb_sparse.py` in the session scratchpad). Arms: dflt, S8, S4, and B1024 (the tile path with
CB 1024 × RB 64, `RNA_MD_BLOCK=1`, 16 lanes).

| section | content |
|---|---|
| A | 14 option and fallback cases against `RNA_GPU=0`; selftest and NEGCTL at 8 × 5601 |
| B | the entry × rho matrix at 600 × 3000, 1200 × 1200, 2400 × 300 and 5601 × 80, phase-synced |
| C | retirement test at 400 × 5601 and 3000 × 1200: 3 alternating reps plus a phase-synced pass per arm |
| D | 1200 × 1200 and the re-baseline's 2400-record soak |
| E | ncu per launch at 44 × 5601, row 2800 |
| V | gates, the retirement comparison, entry format against chunk count |

`run()` asserts every arm's banners and the absence of refusals and admission warnings. Every fixture must
give one output across all arms. A full local smoke run passed (shrunk fixtures).

**S5 done (2026-10-06): A100-SXM4-80GB, LFB `7555a713`.**

**Correctness, all gates MET:**
- **Section A:** 14 cases, in both formats, equal to `RNA_GPU=0`.
- **Selftest at 8 × 5601:** 125,104,224 cells, 0 mismatching in each format. NEGCTL gives 88.9 M
  mismatching.
- **One output per fixture across every arm,** in B, C and D (all 4 arms in C).
- **No refusal and no admission warning anywhere.**

**Retirement test, C** (3 alternating reps without phase sync, plus one P pass with phase sync):

| | 400 × 5601 wall (spread) | md (P) | 3000 × 1200 wall (spread) | md (P) |
|---|---|---|---|---|
| dflt | 60.99 s (0.1 %) | 30.92 s | 18.85 s (1.2 %) | 3.98 s |
| **S8** | **43.09 s (0.3 %), −29.3 %** | **12.61 s, 2.45×** | **17.43 s (1.1 %), −7.5 %** | 2.60 s, 1.53× |
| S4 | 46.48 s (0.3 %), −23.8 % | 15.94 s, 1.94× | 17.68 s (2.2 %), −6.2 % | 2.82 s, 1.41× |
| B1024 | 72.61 s (1.2 %), +19.1 % | 29.69 s, 1.04× | 20.29 s (0.3 %), +7.6 % | 4.78 s, 0.83× |

**D:** 1200 × 1200 is −5.6 % (S8) and −5.1 % (S4). The 2400-record soak goes 116.9 → **94.9 s (−18.8 %,
S8)** and 97.8 s (−16.4 %, S4). The soak has 18 chunks in every arm.

**B, md (P) speedup by length,** S8 / S4 at rho 0.25:

| | 600 × 3000 | 1200 × 1200 | 2400 × 300 | 5601 × 80 |
|---|---|---|---|---|
| S8 | 1.29× | 1.51× | 1.86× | 2.42× |
| S4 | 1.24× | 1.40× | 1.63× | 1.90× |

- rho 0.50 equals rho 0.25 within noise.
- rho 0.10 is slower: hundreds of thousands of columns overflow to the dense path.

**E, ncu per launch,** 44 × 5601 (one production chunk), row 2800:

| | time per row | DRAM read per row | rates |
|---|---|---|---|
| dense md | 517 µs | 372 MB | 720 GB/s, DRAM 36.7 %, SM 53 % |
| S8 | 210 µs (2.46×) | 114.5 MB (3.25× fewer) | cell kernel 650 GB/s, DRAM 36.9 %, SM 42 %, L2 hit 42 % |
| S4 | 273 µs (1.89×) | 65.5 MB (5.68× fewer) | cell kernel 267 GB/s |

The rate cross-check holds:
- **Dense:** 517 µs × 5601 rows × 9 chunks = 26 s, against md(P) 30.9 s.
- **S8:** 210 µs gives 10.6 s against md(P) 12.6 s. The gap, about 2 s, is launch gaps across 3 launches
  per row.

**Decisions this settles:**
1. **9.4, the blocked md driver: RETIRED.** Sparse md beats B1024 beyond spread at both scales: −40.7 % at
   400 × 5601 and −14.1 % at 3000 × 1200. Per the rule, the driver stays in the tree, gated off
   (`RNA_MD_BLOCK`, `RNA_MD_TILE_*`). The variable-K window and `RNA_MD_PRUNE` follow it.
2. **9.1, entry format: 8 B.**
   - It is faster everywhere: md is 1.26× faster than 4 B at 400 × 5601, and wall is 3.4 s lower.
   - The 4-B form's 47 % smaller lists bought **no chunks** at any scale: 9, 3, 2 and 18 chunks in both
     formats. Chunk count is set by the residency/cells cap, not by VRAM.
   - 4 B stays selectable. It would matter only if VRAM ever became the binding limit.
3. **The latency cut is justified on the A100.** The cell kernel runs at 37 % DRAM and 42 % SM, with 2 s of
   launch gaps per 12.6 s. Neither bandwidth nor compute binds.

**Still needed before `RNA_MD_SPARSE=1` becomes the default:**
- bars 6 and 7 (adversarial density and natural sequences);
- the full default-path bars (45-option parity, the default matrix, the binding suite).

**N1: DEFAULT ON (2026-10-06, Luke: "Start N1 now").** `RNA_MD_SPARSE` unset now means sparse md, with 8-B
entries and rho 0.25.
- `RNA_MD_SPARSE=0` restores the dense md.
- `RNA_MD_SPARSE_SELFTEST=1` without an explicit `RNA_MD_SPARSE` still means the selftest.
- Refusals print only when the user asked for sparse md; by default those configurations simply keep the dense
  md, like the other default features' refusals.
- The ACTIVE banner and the exit report are silent under `vrna_cuda_quiet()`, the Python binding.

**Bars,** laptop, on the default-on build:

- **Positive evidence:**
  - the default prints the banner and the report;
  - `=0` prints neither;
  - both equal `RNA_GPU=0`;
  - the Python fold prints 0 md_sparse lines.
- **Bar 6, adversarial input** (polyA, `(GC)n`, `(AU)n`, `(GU)n`, `(CAG)n`, `(CUG)n`, two hairpin arrays,
  `CG`×1500, `(GGCC)n` with and without `-g`, GC-only random, G-rich with and without `-g`):
  - **every case equals both the CPU and the dense md;**
  - density runs from 0 % (polyA) and 0.05 % (G-rich) up to **26.3 % for `(GC)n`/`(AU)n`**, 25.6 % for
    `CG`×1500 and 22.4 % for GC-only random;
  - there the overflow fallback takes over: 529 K cells in `(GC)n` and 2.18 M in `CG`×1500;
  - the cost of those cases at scale is N2 section W.
- **Bar 7, natural input:** 10,082 de-gapped Rfam seeds plus the 16S/riboswitch sequences from upstream's
  test data. All are equal to the CPU. Density is **7.78 %**, and 8.24 % on the two ≥ 1000-nt rRNAs, close to
  random's 7 %.
- **Default-path bars, all green:**
  - `verify_option_parity`: the option surface matches the CPU;
  - `verify_option_matrix`: 36/36 pairs;
  - `verify_gpu_cli` at budgets 4/8/16/32 MB: frozen references byte for byte;
  - the Python binding suite: 20/20.

**Trap found while flipping the default:** every harness whose "dense" reference arm left `RNA_MD_SPARSE`
unset now runs sparse against sparse. The S3 bar's arm is now pinned to `=0`; S5's `dflt` arm would need the
same. Pin the reference, never inherit the default.

**`RNA_MD_SPARSE_LANES=8|16|32` (default 32)** puts 4 or 2 cells in a warp, for the short lists near the
diagonal; it is the first latency-cut knob. 4-B entries refuse it, because their decode is a whole-warp scan.
The kernel is templated on L. Dead groups clamp to a real cell and write nothing, so the full-warp shuffles
stay legal. N2 section L measures it.

**LATER: the latency cut (Luke, 2026-10-06: "note the latency cut possibility for later").** Not started.
It waits for S5's A100 numbers, which say whether sparse md is latency-bound there too. Candidates, in rough
order of expected yield:
- **Pack cells by list length,** instead of one warp per cell. Short lists near the diagonal leave most of
  a warp idle. Options: a warp splits across several short-list cells, or a sub-warp group (8 or 16 lanes)
  is sized to the list.
- **Fuse the three launches per row** (cell, scan, append) where the order allows. The scan needs every
  cell's best, so it is a grid-wide dependency. The append can join the next row's chain, and graph
  capture already hides some launch cost.
- **Scan with more parallelism per record:** a warp-level scan over the row in place of 256 threads × a
  serial segment. The scan is 16 µs per row at 5601 nt.
- **Cheaper 4-B decode,** if the 4-B form wins S5 on chunk width: the 5-step shuffle scan per warp step is
  what made it slower than 8 B.

## 8. Risks

- **Density on the real workload.** Random sequences give 7 %. Structured or repetitive input could be far
  denser; the cap and fallback bound the damage.
- **Scan and append launches:** two per row, about 2 × 2.65 µs × 5601 rows per chunk, roughly 30 ms. Small,
  but launch tax has surprised us before.
- **Gather locality:** if `A` gathers miss L2 at high record counts, the byte win shrinks. ncu will show it.
- **Near-INF semantics:** the guarded sums change near-INF artefacts the dense path tolerated. The triangle bar
  against the CPU decides which one is right.

## 9. Decisions (Luke, 2026-10-05)

1. **Build both entry formats and measure them.** `RNA_MD_SPARSE_ENTRY=8|4`: 8 B `int2`, and 4 B packed
   (`uint16 k` + int16 delta on the per-64-row baselines). The 4-B form moves from S4 into S3.
2. **Capacity is measured, not chosen:** `RNA_MD_SPARSE_RHO` at 0.10, 0.25 and 0.50, across several lengths.
   0.25 is only the starting default.
3. **v1 refusal list: Claude's call, recorded here.** These refuse, with a printed reason, and fall back to
   the dense md:

   | refused | why |
   |---|---|
   | tile/blocked path (`RNA_MD_BLOCK`, `RNA_MD_TILE*`) | different schedule: md runs per tile with rings of carriers. Sparse needs whole rows (the scan runs along a row) |
   | megakernel | its own fused md; one implementation at a time |
   | slot flow, continuous flow | records sit on different rows and slots are handed over. Lists are per record, so it could work, but list lifetime across handovers is untested |
   | `RNA_MD_PRUNE` | a different dense-skipping scheme with its own summaries; pointless with sparse |

   Soft constraints are already declined by the engine, so they never reach md. Everything on today's default
   path stays compatible: int16 storage, `ROW_BATCH`, the c ring, stream overlap 1/2, graph capture. Each
   refusal can be revisited once v1 is measured.
4. **The blocked driver is retired only after a confirmed test at scale.** An A100 run at 400 × 5601 and
   3000 × 1200 puts sparse md against the best blocked configuration from §14 (CB 1024) and the row path. If
   sparse wins, the driver is retired but **kept in the tree, gated off** (`RNA_MD_BLOCK` stays). The
   variable-K window and `RNA_MD_PRUNE` follow the same rule.
5. **No pre-registered threshold.** Section 6's measurements are reported and judged by eye.
