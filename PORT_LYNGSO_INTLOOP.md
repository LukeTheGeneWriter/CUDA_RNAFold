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

1. **Sign-off on the two-kernel design** (§3.1), which keeps dead-warp packing.
2. **v1 refusals** (§3.4): slot and continuous flow, the megakernel, UNROLL > 1. OK?
3. **The A100 S5 notebook** also runs the flagged c-ring K-over-row-width test, on the new kernel. Confirm
   the K grid: {4, 8, 16, 24, 32} and ring off; records 25 … 3000 at fixed L.
4. **Order relative to N5a:** both plans are ready. N5a is host-side and touches the library API (its §9.1).
   N3 is device-side and self-contained. My suggestion: **build N3 S1–S3 first**. It needs no API decision,
   and its laptop bars are well-trodden. N5a P1 follows once its seam question (9.1 there) is decided.
