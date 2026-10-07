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

## 10. S6 and S7: making the carry pay (plan for sign-off, 2026-10-07)

**State.** Lyngsø has been the default since S5. The pre-registered rule said so, and Luke agreed. At 5601
nt it saves almost no GPU work: the carry kernel (191 µs per row) costs what the eval kernel saves (365 →
175 µs). The −4.6 % wall comes from overlap. Narrow rows lose: +59 % `int_loop` at 25 × 1200, +42 % at
8 × 5601. The soak is +1.8 %.

### 10.1 Why the carry costs what it does

`ly_carry_kernel` runs one thread per row slot, and each thread walks u = 6…30 serially. Per u, it computes
up to 4 fresh entries, and each entry is a chain of dependent loads:
- `Hc`;
- `c(k,l)`;
- `Ptype` (two loads from S);
- two `unpack`s;
- `mismatchI`.

That is roughly 25 × (2–4) × 5 loads on one thread's critical path. **On a narrow row there are too few
threads to hide that latency, so the kernel's time is one thread's chain.** That is the fixed per-row cost
the R sweep shows (~35 µs per row at 25 × 1200). On a wide row the same chain is work: 44 M instructions
per launch at 44 × 5601, as much as the eval kernel.

Two facts make both costs avoidable:
1. **The 25 u planes are independent.** `G(i,j,u)` reads `G(i+1,j-1,u-2)` and fresh entries of size u only.
   Nothing couples u to u′ inside a row.
2. **A fresh entry's expensive part does not depend on the outer pair.**

   `e(k,l) = c(k,l) + mismatchI[rtype(type(k,l))][S(l+1)][S(k-1)]`

   It is INF when `Hc(k,l)` forbids the pair, `c ≥ INF`, or noClosingGU rejects the type. The ninio term
   depends on u alone, and the up_int gates on (i, u1) and (l, u2). So every outer pair that uses (k,l)
   recomputes the same e: up to 25 × 2 times.

### 10.2 S6a: one thread per (u, slot)

- **Grid:** 25 × row_total threads, u-major, so a warp is 32 consecutive slots of one u plane. The reads of
  `Gp`, the writes of `Gc` and the c reads along a row stay coalesced, as today.
- **Chain:** each thread does one carry load plus at most 4 fresh entries, so the critical path shrinks
  ~25×.
- **Exactness:** exact by construction, since this is the same arithmetic split across threads.
- **Expected:** most of the narrow-row penalty goes; wide rows unchanged or slightly better (more memory
  parallelism, same instructions).
- **Cost:** about 40 lines; the kernel signature is unchanged.

### 10.3 S6b: an e ring (precomputed entries)

- **What it is:** a 32-row ring of `e(k,l)`, laid out like the c ring (row_off_H + l, slot k & 31), held
  whether the c ring is on or off.
- **Who writes it:** `carry(i)` writes `e(i+1, ·)` for its columns as a side job.
- **Why that is safe:**
  - Row i+1's c is final by then. The carry is on the cell stream, launched after the same waits as
    `int_loop(i)`, whose stacks read `c(i+1,·)`. **To verify in code before building:** the event waits
    must be enqueued before `ly_eval_row()`.
  - `carry(i)` reads e rows i+3 … i+29 only, all written by earlier carries, so the kernel never reads
    what it writes.
- **The fresh entry becomes:**

  `e(k,l) + MIN2(max_ninio, |u1-u2|·ninio2)`, gated by `up[i+1] ≥ u1 && up[l+1] ≥ u2`

  That is one load (plus up_int, which is cached) instead of about 6 dependent ones.
- **Admission:** the e ring is charged like the c ring: 32 × row_total × 4 B, the same size as the c ring. That
  is ~34 MB at a 48 × 5601 chunk and ~150 MB at a 1000 × 1200 chunk, a few % of the chunk. It costs records per chunk only on small GPUs.
- **Expected:** the carry's instructions fall from 44 M to roughly 8–12 M per launch at 44 × 5601, so
  Lyngsø's work saving becomes real on long records: `int_loop` per row ~365 → ~230 µs, against ~366 µs
  today.

### 10.4 S7: switch on mid-sweep (only if S6 leaves narrow rows losing)

**Mechanism:**
- Run dense while a row is narrow.
- At the first row i0 (sweeping down) whose active cells per row, Σ_H max(0, len_H − i), cross a
  threshold T, run a one-off `ly_init_kernel`. It computes `G(i0, j, u)` directly: the min over every
  generic loop of size u with outer (i0, j), the dense enumeration restricted to generic loops, for one
  row.
- Carry from i0 − 1 onward.

**Why per row, not per chunk:** G must be carried through every row once started, but it can start late.
A chunk's narrow rows are at the top of its sweep, which is where the soak loses.

**Exactness:** `G(i0)` from the init must equal what the carry would hold. That is S0's equivalence
(lyngso_equiv), which the selftest checks on the device.

**Bars:**
- a forced switch row `RNA_INT_LOOP_LYNGSO_FROM=<row>` at several rows: output and triangles equal dense;
- the selftest at the switch row;
- a NEGCTL that skips the init: it must go red.

T comes from the laptop R sweep and is confirmed on the A100.

**Skip S7 if S6a+b brings 25 × 1200 and 8 × 5601 within spread of dense.** Most of the narrow-row cost is
latency, which S6a attacks directly.

### 10.5 Bars for S6 (pre-registered)

1. **Exactness:**
   - `ly_s2.sh` (the 19 cases) plus the multi-chunk case;
   - the device selftest at 0 MISMATCHING with its NEGCTL red;
   - the production NEGCTL red;
   - S6b: a NEGCTL that drops the e ring write must go red (stale e).
2. **Laptop, ABBA, phase-synced `int_loop`:** an R-style width sweep at 1200 (25, 100, 400 records) and 5601
   (8, 48), S6 against S5 Lyngsø against dense, plus ncu per launch for the carry at 44 × 5601 and 4 × 1500.
3. **A100 notebook S8:**
   - T again, with the soak repeated 3 times;
   - the R widths;
   - the decision on S7.
4. **Default-path bars** after each stage: parity, matrix, gpu_cli budgets, binding suite.

### 10.6 Decisions for Luke

1. **Order:** S6a, measure, then S6b, measure, then decide S7. Each lands behind `RNA_INT_LOOP_LYNGSO_V2=1`
   until its bars pass, then becomes the Lyngsø path.
2. **The e ring's memory:** charged to admission, the same size as the c ring (~34 MB at 48 × 5601, ~150 MB at 1000 × 1200). OK?
3. **One A100 notebook after S6b** (not after each stage), to save A100 hours.

### 10.7 S6a results (2026-10-07, laptop RTX 3050 at full clocks, 25 W / 1725 MHz)

`RNA_INT_LOOP_LYNGSO_V2=1` (off by default). One `ly_carry_u()` serves both kernels, so the arithmetic is
shared by construction.

**Exact:**
- The device selftest gives 0 MISMATCHING with **identical entry counts to v1** (fresh, carried and direct,
  to the unit) on mix, G-rich, `--noLP`, `-C --enforceConstraint` and `--circ`.
- V2 + NEGCTL bites: 129,208 and 24,127 mismatching.
- `ly_s2.sh` with V2 exported: all 19 cases equal dense and the CPU in output and triangles.
- Refusals print; the production NEGCTL changes 8 lines and the triangles.

**Slower on the laptop.**

Phase-synced `int_loop` (s), 2 runs each in ABBA order, one output per fixture:

| width | v1 | v2 | dense |
|---|---|---|---|
| 25 × 1200 | 0.37 | 0.44 (+20 %) | 0.35 |
| 100 × 1200 | 1.32 | 1.75 (+33 %) | 1.31 |
| 400 × 1200 | 5.30 | 7.17 (+35 %) | 5.39 |

ncu, `ly_carry_kernel` per launch:

| fixture | v1 | v2 | instructions v1 → v2 | occupancy v1 → v2 |
|---|---|---|---|---|
| 25 × 1200 | 211 µs | 247 µs | 4.3 → 7.2 M | 56 → 85 % |
| 8 × 5601 | 610 µs | 583 µs (−4 %) | 6.3 → 9.8 M | 73 → 90 % |
| 400 × 1200 | 2918 µs | 4370 µs | 69.8 → 134.9 M | 72 → 88 % |

**Reading:**
- **The per-thread prologue is the cost.** It is the record search (`flatten_index_to_H`), the offsets, the
  reader and the up_int gate, and v2 repeats it 25×. Instructions grow 1.6–1.9×. Where the GPU was already
  full, that is pure loss.
- **§10.1's diagnosis was half right.** The u loop's iterations are independent, so v1 is not one serial
  chain. The narrow-row penalty is **under-occupancy**, not chain length.
- **The laptop is the wrong instrument for narrow rows.** A 3050 has 20 SMs × 48 warps (~30 k threads), so 25
  records × 1200 already nearly fills it. That is why v1 is within ~5 % of dense here, against +59 % on the
  A100.
- **The one under-filled laptop case is the one v2 wins.** At 8 × 5601, v2 is −4 %. An A100 (108 SMs × 64
  warps, ~221 k threads) is under-filled by every row narrower than ~220 k active slots. That covers all of
  R's losing widths.

**Proposed S6a′, an adaptive split** (needs sign-off, since it changes S6a's design):
- **Planes per thread from the row's fill:** Pt ∈ {25, 5, 1}. The host computes active slots,
  Σ_H max(0, len_H − i), from its own lengths.
  - Pt = 25 (v1) when active slots ≥ device thread capacity;
  - Pt = 5 when 5 × active ≥ capacity;
  - else Pt = 1 (v2).
- **Grid:** shrunk to the active slots, so slots j ≤ i launch nothing.
- **Capacity:** the SM count × max threads per SM, queried once.
- **Bars:** the same exactness bars. A laptop check that wide rows equal v1 and 8 × 5601 equals or beats v2.
  The narrow-row verdict goes to the S6 A100 notebook.

### 10.8 S6a′ results (2026-10-07, laptop; GPU mostly power-capped, 12–17 W)

`RNA_INT_LOOP_LYNGSO_V2=1`: a 3-D carry grid (z the record, x the columns j > i, y a group of PT planes), so
there is no record search and no thread below the diagonal. `RNA_INT_LOOP_LYNGSO_PT=25|5|1` forces a split;
a tally at teardown says which split each row ran.

**Exact:**
- Every split (25, 5, 1, adaptive) gives 0 MISMATCHING with v1's entry counts to the unit, on mix, G-rich,
  `--noLP`, `-C` and `--circ`.
- NEGCTL gives 129,208.
- `ly_s2.sh` with V2: 19/19, plus the refusals and the production NEGCTL.

**Phase-synced `int_loop` (s), ABBA, median of 2, one output per fixture:**

| fixture | v1 | adaptive (first rule) | PT 25 | **PT 5** | PT 1 | dense |
|---|---|---|---|---|---|---|
| 2 × 5601 | 0.85 | 0.69 | 0.82 | 0.70 | 0.69 | 0.78 |
| 8 × 5601 | 4.19 | 4.03 | 4.49 | **3.34** | 3.57 | 3.90 |
| 48 × 5601 | 28.66 | 29.49 | 30.16 | **21.76** | 21.94 | 28.53 |
| 25 × 1200 | 0.56 | 0.56 | 0.56 | **0.52** | 0.54 | 0.65 |
| 100 × 1200 | 2.05 | 2.16 | 2.16 | **1.90** | 1.97 | 2.66 |
| 400 × 1200 | 7.70 | 8.37 | 8.41 | **7.46** | 7.73 | 10.36 |
| 1500 × 600 | 7.20 | 6.96 | 6.97 | **6.92** | 7.27 | 9.43 |

**Reading:**
- **PT = 5 is best or tied at every width.** At 48 × 5601 it is −24 % against v1 and −24 % against dense,
  where v1 had only tied dense (as on the A100 at 44 × 5601).
- PT = 25 on the 3-D grid loses even on full rows. The likely cause: with the u loop not unrolled, the
  per-u fresh tables (`f1[4]`/`f2[4]`) are indexed dynamically and live in local memory. With PT = 5
  unrolled, each u is a compile-time constant. ncu's local-load bytes will confirm.
- **The rule is now PT = 5, or 1 when 5 × active slots cannot fill the device.** PT = 25 stays reachable
  only through the forcing knob.

### 10.9 S6b results (2026-10-07, laptop; GPU power-capped, 12.9 W / 1057 MHz)

`RNA_INT_LOOP_LYNGSO_ERING=1`: fresh entries read `e(k,l)` from a 32-row ring, which `carry(i)` writes for
row i+1 (the y = 0 group in the 3-D kernel). `RNA_INT_LOOP_LYNGSO_NEGCTL=2` skips the ring writes.

**Exact:**
- Every arm gives 0 MISMATCHING with v1's entry counts to the unit: the ring alone, the ring with the
  adaptive split, and the ring with PT forced to 25, 5 or 1.
- The fixtures: mix, G-rich, `--noLP`, `-C`, `--circ`, `--noClosingGU`, `-d0` and salt 0.2.
- NEGCTL=2 gives 236,068 mismatching (no fresh entries survive).
- `ly_s2.sh` 19/19 with V2 + ERING and with ERING alone, plus the refusals.
- The production NEGCTL=2 changes 8 output lines and the triangles.

**Phase-synced `int_loop` (s), ABBA, median of 2, one output per fixture** (v1 = the S5 default; ad = S6a′
alone; **v2e** = S6a′ + S6b):

| fixture | v1 | ad | **v2e** | p5e | dense | v2e vs v1 | v2e vs dense |
|---|---|---|---|---|---|---|---|
| 2 × 5601 | 0.92 | 0.75 | **0.62** | 0.62 | 0.81 | −32 % | −23 % |
| 8 × 5601 | 4.99 | 3.78 | **2.96** | 2.95 | 4.91 | −41 % | −40 % |
| 25 × 1200 | 0.65 | 0.56 | **0.46** | 0.47 | 0.74 | −29 % | −37 % |
| 100 × 1200 | 2.19 | 1.96 | **1.60** | 1.60 | 2.67 | −27 % | −40 % |
| 400 × 1200 | 8.30 | 7.59 | **6.17** | 6.18 | 10.38 | −26 % | −41 % |
| 1500 × 600 | 7.24 | 7.01 | **5.82** | 5.80 | 9.43 | −20 % | −38 % |

48 × 5601 was not run: Claude Code reaped the run for host memory. The A100 notebook covers it.

**Reading:**
- On the laptop, the narrow-row loss has gone at every width tried. v2e beats dense by 23–41 %.
- The rule's PT choice matches forced PT = 5 within noise wherever both ran.
