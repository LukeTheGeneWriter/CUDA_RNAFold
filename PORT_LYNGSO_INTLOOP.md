# Lyngsø interior loops (N3): plan for sign-off

Roadmap N3 (was C2). Literature: Lyngsø, Zuker & Pedersen 1999 ("Fast evaluation of internal loops in RNA
secondary structure prediction"). **Status: S0 done (exactness proven on the CPU); the GPU stages below await
Luke's sign-off.**

## 1. Why

With sparse md on by default, `int_loop` is the largest GPU phase at 400 × 5601 on the A100: **16.5 s, 34 %
of the phase-synced wall** (N2). N2's ncu, per launch at row 2800 of a production chunk (44 × 5601):

| kernel | time | SM | DRAM | occupancy | active lanes | registers |
|---|---|---|---|---|---|---|
| `int_loop_warp_kernel` | 363 µs | 62 % | 5 % | 38 % | 25.8 / 32 | 61 |

**It is issue-bound, not memory-bound.** The work is the candidate count, so a method that evaluates fewer
candidates should mostly turn into time. That is not true of md, where fewer bytes mattered.

**The method.** Today every live cell (i,j) evaluates every allowed inner pair (p,q) with u1+u2 ≤ 30: up
to ~465 positions, 5–7 warp passes. Lyngsø observes that a *generic* interior loop's energy separates:

```
E(i,j,p,q) = internal_loop[u] + mismatchI(outer: i,j) + [ mismatchI(inner: p,q) + MIN2(MAX_NINIO, |u1-u2| ninio) ]
             with u = u1 + u2, plus salt(u) when salt is set
```

The bracket plus c(p,q) does not depend on (i,j), so a table G(i,j,u) of the best bracket per total size
**carries** from (i+1,j-1) to (i,j). Each cell then needs only:
- the **fresh** loops whose short side is exactly 2, plus 3×3 / 3×4 / 4×3;
- the **direct** loops: stacks, bulges, 1×n and the special 2×2 / 2×3 / 3×2 tables;
- 27 **uses** of G.

## 2. Exactness: S0, done

`tools/lyngso_equiv.sh BUILD_DIR` (commit `8d3deef3`): on upstream's own finished c matrix,
`min(direct, carry, gquad term)` equals `vrna_mfe_internal()` for every (i,j).

- **0 cells differ** under plain, `-C --enforceConstraint` ('x', '|', enforced pairs), `--noClosingGU`, `-g`,
  `--noLP` and salt 0.2, at 400 × 6 and 1200 × 2, against both stock 2.7.2 and LFB's build.
- **Negative controls:**
  - hard-constraint mask ignored: 5,527 cells differ;
  - carry dropped: 13,456 cells (plain) and 5,004 (`-C`).
- **Evaluations: 1.85× fewer than dense** (10-05 CPU count). That includes maintaining G for every cell.

Rules S0 established, which the GPU must reproduce:

| rule | detail |
|---|---|
| hard-constraint mask on the carry | a carried loop gains unpaired i+1 and j-1. `up_int` counts consecutive unpairable positions, so the carry is valid **iff** `up_int[i+1] ≥ 1 && up_int[j-1] ≥ 1`. Fresh entries need their own runs. |
| inner pair | allowed as an enclosed pair (`INT_LOOP_ENC`); not GU under `--noClosingGU` (stacks exempt); checked when the entry is made |
| outer pair | `INT_LOOP` context and `--noClosingGU`, checked once at use |
| **pair type** | ptype 0 on a pair the constraints allow is **7** (non-standard), and `rtype` applies after the promotion. The GPU's `Energy()` already does this (`cell_invariants()`, the `--nsp` fix). The new kernels must share that code, not copy it. |
| G-quads inside interior loops | the existing `gq_internal_kernel`, unchanged |

## 3. GPU design

**3.1 Two kernels where there is one.** Lyngsø's G must be maintained for **every** cell, including the 62.6 %
that cannot pair (N2's dead-warp count). Dead-warp packing skips those cells, so a single kernel would force a
choice between Lyngsø and packing. **Splitting resolves it:**

- **`il_carry_kernel` (all cells, elementwise).** It writes G(i,·,u) for row i:
  - the carry from row i+1, column j-1, masked by `up_int`;
  - fresh entries: about 55 cheap evaluations, each c(k,l) + mismatchI(inner) + ninio. No (i,j) energy, no
    reduction.

  The fresh entries' inner pairs lie at rows i+3 … i+29, columns j-3 … j-5, all inside the c ring's 32
  rows. One thread per (cell, u-group), coalesced along j.
- **`il_eval_kernel` (live cells only, dead-warp packed as today).** It computes the direct loops plus the
  27 uses: `min_u G(i,j,u) + internal_loop[u] + mismatchI(outer) + salt(u)`. Same warp-per-cell shape,
  same hc column masks, but about 120 direct positions where today there are ~465.

**3.2 G storage.** Two row-shaped buffers (row i+1 and row i), laid out `[u][row_off_H + j]`. Each u-plane is
row-shaped, so lanes walking j are coalesced.
- **Size:** 27 u-values × (L+1) × 4 B × 2 rows ≈ **1.2 MB per 5601-nt record**, about 53 MB for a
  44-record chunk.
- **Admission:** charged in `gpu_bytes_per_file()`, like the sparse lists, with the same allocation ≤ charge
  check and warning.

**3.3 Energy terms.** Direct loops call the existing `Energy()`, unchanged. Fresh and use terms are new
`__device__` helpers built from the same `cuda_param_t` tables (`internal_loop`, `mismatchI`, `ninio`, the
device salt table `rnafold_build_salt_table()`), with the type rules of §2. Sums are INF-guarded, as in sparse
md.

**3.4 Knobs and refusals.** `RNA_INT_LOOP_LYNGSO=0|1`, off until the bars pass. Then default-on is Luke's
call, as with sparse md.
- `RNA_INT_LOOP_LYNGSO_SELFTEST=1` computes both and compares cell by cell.
- `_NEGCTL=1` drops the carry, and must go red.
- **v1 refusals (dense kernel runs, with a reason):** slot flow and continuous flow (records on different
  rows; the carry is per record per row), the megakernel, `RNA_INT_LOOP_UNROLL > 1`.
- **Compatible:** the c ring (it reads the same rows), dead-warp packing (eval kernel), graphs, int16/int32,
  sparse md.

## 4. Bars

1. **S0 bar:** `tools/lyngso_equiv.sh`, done.
2. **Device selftest:** Lyngsø vs dense, every cell of every row, both computed. 0 mismatches. NEGCTL red in
   the triangles, not only in the output: the 3c lesson is that structures can survive a dropped term.
3. **Triangles:** `RNA_TRI_CHECKSUM` c equal to the dense kernel's and to `RNA_GPU=0` on the option matrix
   (plain, noLP, circ, d0, salt, T25, maxBPspan, `-g`, `-C --enforce`, `--noClosingGU`, `--nsp`, a `-P`
   parameter file).
4. **Default-path bars:** `verify_option_parity`, `verify_option_matrix`, `verify_gpu_cli` budgets, the
   binding suite, the soak, plus N1's adversarial and natural sets.
5. **The ring and packing interplay:** c ring on and off, G (cells per warp) auto/1/8, overlap 0/1/2, graphs
   on and off. One output for all.

## 5. Stages

| stage | content | bar |
|---|---|---|
| S0 | CPU exactness incl. hc/noGU/-g/noLP/salt | **done**: 0 differ, both negctls red |
| S1 | `il_carry_kernel` + G buffers; Lyngsø computed BESIDE dense (selftest only) | selftest 0, NEGCTL red |
| S2 | `il_eval_kernel` replaces the dense kernel under `RNA_INT_LOOP_LYNGSO=1` | triangles + parity |
| S3 | ring, packing, overlap, graphs interplay; admission charge; refusals | §4 bars 4–5 |
| S4 | laptop ncu: per-launch time and instruction counts, both kernels against dense, at 600–5601 | per-launch numbers |
| S5 | A100 notebook: the matrix + **the flagged c-ring K-over-row-width test, on the new kernel** + the `RNA_INT_LOOP_UNROLL=2` re-test → default decision | by eye |

**S1 done (2026-10-06, laptop RTX 3050).** `ly_selftest.inc`, under `RNA_INT_LOOP_LYNGSO_SELFTEST=1`:
- **`ly_carry_kernel`** maintains G over every column of every row, ping-ponging two `[u][row_total]`
  buffers (u = 6..30, 25 planes), so no stale value survives.
- **`ly_check_kernel`** recomputes each cell the dense kernel computes, as the direct loops through `Energy()`
  itself plus the 25 G uses, reading the same c (ring or triangle) through the same reader types. It compares
  with the dense `energy_min2`.

Results on 8 records of 300–1700 nt (plus G-rich and enforced `|`), 4.8 M cells, **0 MISMATCHING** on all 12
configurations:
- **options:** plain, `--noLP`, `--circ`, `-d0`, salt 0.2, `-T 25`, `--maxBPspan 150`, `-g`,
  `-C --enforceConstraint`, `--noClosingGU`;
- **settings:** c ring off, int32.
- **NEGCTL** (carry dropped): **129,208 mismatching**.
- **Default path untouched:** output equal to `RNA_GPU=0`, no Lyngsø output at all. Refused under slot and
  continuous flow, with a reason.

**Work on the plain fixture:**

| | dense | Lyngsø |
|---|---|---|
| full interior-loop evaluations (`Energy()`) | 319.0 M | **78.3 M (4.1× fewer)** |
| fresh entries (a c read plus two lookups each) | | 91.9 M |
| carried copies | | 103.8 M |
| uses (25 adds per live cell, 1.79 M live cells) | | 44.8 M |

The expensive evaluations are what shrinks. S2 decides whether the cheap ones are cheap enough on the device.

**S2 done (2026-10-07, laptop).** `RNA_INT_LOOP_LYNGSO=1` (off by default).

**As built, simpler than §3.1's separate eval kernel:**
- **`int_loop_lyngso.inc`'s carry kernel** runs first each row, on the cell stream.
- **The existing warp kernel's `LY` variant** (`int_loop_cell.inc`) is the evaluation. Each lane ANDs its
  column's hard-constraint mask with the direct rows for that column's u2:
  - u2 ≤ 1: all u1;
  - u2 = 2: u1 ≤ 3;
  - u2 = 3: u1 ≤ 2;
  - otherwise: u1 ≤ 1.

  So the warp scan enumerates only stacks, bulges, 1×n and the specials. Lanes 0–24 then add one G use each
  before the warp reduction.
- **What comes for free:** dead-warp packing, the c ring reader, the 2-D grid and the warp search, all
  unchanged.
- **Unread slots are not written:** the carry kernel skips slots j ≤ i, which no reader touches.
- **Admission:** the G buffers are charged in `int_loop_bytes_per_file()`.
- **Refused, with a reason:** `RNA_INT_LOOP_UNROLL > 1`, the block-per-cell kernel, slot and continuous flow,
  and the megakernel.

**Bar (`ly_s2.sh`): output AND c/fML triangle checksums equal the dense kernel's and `RNA_GPU=0`'s on 19
cases:**
- **options:** plain, noLP, circ, d0, salt 0.2, T25, maxBPspan 150, `-g`, `-C --enforceConstraint`,
  `--noClosingGU`;
- **settings:**
  - c ring off;
  - cells per warp 1 and 8;
  - overlap 0 and 1;
  - graphs off;
  - int32;
  - dense md;
  - a 16 MB budget (multi-chunk).

Both refusals print, and the production NEGCTL (carry dropped) changes the output (8 lines) and the triangles.

**Laptop timing,** ABBA, `RNA_PHASE_SYNC=1`, same sha in every pair. The GPU sat in a ~12 W power cap at
1057 MHz, so the absolute times are ~4× this morning's, but both arms ran under the same cap. Graphs on/off
were checked and are not the cause (12.2 / 11.9 / 12.1 s).

| L × records | `int_loop` dense → Lyngsø | wall |
|---|---|---|
| 1200 × 120 | 8.10 → 6.02 s (**−25.7 %**) | −17.7 % |
| 2400 × 32 | 8.92 → 6.87 s (**−23.0 %**) | −15.3 % |
| 5601 × 8 | 12.54 → 10.05 s (**−19.9 %**) | −10.6 % |

The carry kernel is inside the `int_loop` phase, so these are net. The A100 (S5) decides the default.

**S3 partial (2026-10-07, laptop, `ly_s3.sh`, `RNA_INT_LOOP_LYNGSO=1` exported for every arm).** Positive
evidence: the ACTIVE banner printed. All green so far:
- `verify_option_parity`: the CUDA build matches the CPU build across the option surface;
- `verify_option_matrix`: 36/36 pairs agree, on the route they claim;
- `verify_gpu_cli` budgets 4/8/16/32 MB: byte for byte against the frozen references;
- Python binding suite: 20 pass, 0 fail;
- bar 6 (adversarial density): all 14 cases (12 fixtures + 2 under `-g`), `vsCPU=SAME vsDense=SAME swept=1`.

**Not run:** Claude Code reaped the run for host memory pressure after bar 6, so bar 7 (the natural set) is
still open on the laptop. The S5 notebook's G section runs the natural set with Lyngsø on.

**S5 notebook written (2026-10-07): `CUDA_RNAFold_LyngsoS5.ipynb`** (gitignored; generator `make_nb_s5.py`).
Sections:
- **G:** exactness, which stops the run on any failure.
- **T:** dense vs Lyngsø at 400 × 5601, 800 × 2400, 3000 × 1200 and 3000 × 600, plus the soak, judged by the
  pre-registered rule D0–D2.
- **U:** the UNROLL=2 re-test, with a ring-off control.
- **I:** interplay.
- **R:** the flagged c-ring K over row widths, on the Lyngsø kernel.
- **E:** ncu per launch, plus the per-row ring/carry cost.
- **V:** the report.

The laptop smoke passed end to end. Two harness lessons from it:
- **The triangle checksums are per chunk.** The G buffers are charged to admission, so under a 16 MB budget
  Lyngsø splits into 9 chunks against dense's 6. The multi-chunk case now gives the dense arm
  `RNA_INT_LOOP_LYNGSO_SELFTEST=1` (same charge, so the same partition, and 0 MISMATCHING required) and
  requires an identical partition before comparing triangles.
- **The natural set includes the 16S/riboswitch files** (2 records ≥ 1000 nt).

**Smoke hint, to be confirmed on the A100:** `ly_carry_kernel` is not free on narrow rows. On the laptop
(power-capped, tiny fixtures):

| fixture | `ly_carry_kernel` | warp kernel saving | occupancy |
|---|---|---|---|
| 4 × 1500 | 106 µs per launch | 52 µs (118 → 66 µs) | 13 % |
| 60 × 200 | 123 µs | 110 µs | |

At 5 records × 400 nt, the Lyngsø `int_loop` phase was +54 % against dense. The carry's cost is per row,
almost regardless of width, so few-record chunks lose. R and E measure this on the A100. If it holds, the
default wants a **row-width gate** (cells per row), not a length gate, and the carry kernel wants more
parallelism (one thread per (u, cell) instead of a loop over u).

### S5 results (2026-10-07, A100-SXM4-80GB, LFB 5f4ece9a)

**Pre-registered outcome: DEFAULT ON** (D0, D1, D2 all met). One output per fixture in every arm.

**G (exactness) PASS in full:**
- the 20-case matrix: output = dense = `RNA_GPU=0`; triangles equal under the same partition, including
  9 chunks at 16 MB;
- 14 adversarial cases;
- 10,082 natural records, including the long ones;
- device selftest: 0 of 10.57 M cells mismatching; its NEGCTL gives 278,079;
- production NEGCTL bites in both the output and the triangles;
- both refusals print.

**T (wall: median of 3, ABBA; int_loop: phase-synced pass):**

| fixture | dense | Lyngsø | wall | `int_loop` (P) |
|---|---|---|---|---|
| 400 × 5601 | 41.32 s | 39.40 s | **−4.6 %** (beyond spread) | 16.54 → 16.77 s (+1.4 %) |
| 800 × 2400 | 14.93 | 14.38 | **−3.7 %** (beyond) | −7.1 % |
| 3000 × 1200 | 16.54 | 15.83 | **−4.3 %** (beyond) | −10.3 % |
| 3000 × 600 | 5.44 | 5.28 | −2.8 % (within) | −8.0 % |
| soak 2400 (1 rep) | 90.78 | 92.40 | **+1.8 %** | |

**The soak is the caveat.** It is one rep each, so there is no spread. D2's soak tolerance was 2 %,
fixed in the notebook before the run, so +1.8 % passes it narrowly.

**What the numbers say:**

1. **At 5601 the work saving is ~0, and the wall gain is borne by overlap.** ncu at 44 × 5601:

   | kernel | time | instructions |
   |---|---|---|
   | dense `int_loop` | 365 µs | 97 M |
   | Lyngsø eval | 175 µs | 42 M |
   | `ly_carry_kernel` | **191 µs** | 44 M |

   The carry costs what the eval saves. At 3000 × 600 it is 2361 µs dense against 1159 + 991 µs (−9 %). The
   phase-synced runs agree: synced wall at 400 × 5601 is 47.78 dense against 48.18 Lyngsø. Yet production
   (overlap 2) is −4.6 %. The carry overlaps md/hp_mb on the other streams, which shortens the dependent
   chain. I (below) confirms it: with Lyngsø, overlap 0 and 1 cost +11.5 % and +10.2 %, against +4.6 % and
   +3.5 % dense in N2.
2. **Narrow rows lose, as the smoke hinted** (R, phase-synced `int_loop`, Lyngsø K16 against dense K16):

   | width | `int_loop` | wall |
   |---|---|---|
   | 25 × 1200 | +59 % | +3.4 % |
   | 100 × 1200 | +17 % | |
   | 400 × 1200 | −4.5 % | |
   | 1500 × 1200 | −7.5 % | |
   | 3000 × 1200 | −7.9 % | |
   | 8 × 5601 | +42 % | +7.5 % |
   | 48 × 5601 | +11 % | |
   | 200 × 5601 | +2.9 % | |

   The carry is a per-row cost over every cell, dead ones included (§8 risk 1). A mixed-length chunk has
   narrow rows at the top of its sweep, where only the long records are active. That is the likely soak
   penalty.

**U (`RNA_INT_LOOP_UNROLL=2`): retire it.**
- 400 × 5601: +3.8 %; 3000 × 1200: +3.0 %.
- That is worse than ring-off alone (+2.4 % and +2.3 %), so the unroll itself loses too.

**I (Lyngsø on, 400 × 5601, 1 rep each), relative to the Lyngsø default:**

| arm | wall |
|---|---|
| overlap 1 | +10.2 % |
| overlap 0 | +11.5 % |
| graphs off | 0.0 % |
| cells per warp 1 | +6.5 % |
| cells per warp 8 | −0.6 % |
| int32 | −0.1 % |
| int16 | +0.9 % |

The defaults stand.

**R (the flagged c-ring K over row widths): CLOSED, K16 stays.**
- K is flat at every width: ≤ 0.6 %, except K8 at 3000 × 1200 (−2.2 %, within its 4.3 % spread).
- E's per-row sums differ by ≤ 1 % across K, early and late in the sweep, narrow and wide.
- The ring itself pays everywhere: ring-off costs +4.4 to +18.8 % of `int_loop`.

**Next levers (proposals, need sign-off):**
- **S6, carry v2:**
  - Precompute the fresh-entry value once per (k, l): `e(k,l) = c(k,l) + mismatchI[...]`. It does not depend
    on the outer pair, and the ninio term depends on u alone.
  - The carry then reads e from contiguous row and column segments that neighbouring cells share 24 of 25 of
    (tile through shared memory), instead of ~55 c reads plus lookups per cell.
  - Target: make the work saving real at 5601, where it is ~0 today.
- **S7, a mid-sweep switch:**
  - Run dense while a row is narrow.
  - At the first row whose active cells per row cross a threshold, initialise G once with one dense-style
    generic-loop row, then carry.
  - This aims straight at the soak's narrow rows. A per-row on/off is not possible, because G must be
    carried through every row once started.

## 6. Expected size

CPU counting says 1.85× fewer evaluations. If the kernel stays issue-bound, `int_loop` drops from 16.5 s
towards about 9 s, plus the carry kernel, which is elementwise and should cost ~1–2 s.

**Estimate: −5 to −7 s at 400 × 5601**, about −12 to −16 % of today's 43.6 s. That is before N5a. The two are
additive (N5a hides host work, N3 shrinks GPU work).

## 7. Synergies and tradeoffs

| pair | relation |
|---|---|
| Lyngsø + dead-warp packing | **compatible by the two-kernel split** (§3.1); otherwise either/or |
| Lyngsø + c ring | SYNERGY: fresh entries read rows i+3 … i+29, inside the ring. The ring-K test must run on the new access pattern (ORDERING: after S2). |
| Lyngsø + sparse md | ADDITIVE: different kernels, nothing shared |
| Lyngsø + N5a backtrack pipelining | ADDITIVE. A shorter sweep per chunk still covers ~1.3 s of backtrack until the sweep drops below ~1.5 s per chunk. |
| Lyngsø + `RNA_INT_LOOP_UNROLL` | to re-measure (S5): fewer candidates per cell may change the latency balance U=2 was for |
| Lyngsø + the megakernel | refused (dead branch, register-bound) |
| G buffers vs chunk width | negligible VRAM (~1.2 MB per record); the cap binds anyway |

## 8. Risks

1. **The carry kernel's cost.** It touches every cell, the dead ones included, every row. Elementwise and
   coalesced, but it adds a launch per row (graphs help) and ~55 c reads per cell. S4 measures it. If it
   outweighs the saving at short lengths, Lyngsø can be length-gated.
2. **Register pressure.** The eval kernel adds the 27-term use loop to `Energy()`'s footprint (61 registers
   today; registers already bind occupancy at 38 %).
3. **Salt beyond MAXLOOP+1.** `vrna_salt_loop_int` is computed on the host for u+2 > 31. S0 found this, and
   the device table must cover the same range.
4. **`--noLP` on the GPU** has its own stack path (`stack_row_kernel`). S0 proves the interior-loop term is
   exact under noLP's c, but the GPU's noLP handling of which loops are legal must be read, not assumed. S1's
   selftest covers it.
5. **Non-standard types (§2):** the new helpers must call the same type code as `Energy()`.

## 9. Decisions for Luke

**SIGNED OFF 2026-10-07** (Luke: "I sign off on the N3 plan"). The two-kernel design, the v1 refusals, the
S5 ring-K grid and N3-before-N5a stand as written below.


1. **Sign-off on the two-kernel design** (§3.1), which keeps dead-warp packing.
2. **v1 refusals** (§3.4): slot and continuous flow, the megakernel, UNROLL > 1. OK?
3. **The A100 S5 notebook** also runs the flagged c-ring K-over-row-width test, on the new kernel. Confirm
   the K grid: {4, 8, 16, 24, 32} and ring off; records 25 … 3000 at fixed L.
4. **Order relative to N5a:** both plans are ready. N5a is host-side and touches the library API (its §9.1).
   N3 is device-side and self-contained. My suggestion: **build N3 S1–S3 first**. It needs no API decision,
   and its laptop bars are well-trodden. N5a P1 follows once its seam question (9.1 there) is decided.
