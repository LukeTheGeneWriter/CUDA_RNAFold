# Session 2026-09-10 — the wall is 394.8 s, and a five-times-reproduced regression was never real

*4 commits, `e478419e` → `ff4911fd`, all pushed. Previous handoff: `PORT_SESSION_2026-09-09.md`.
Read this, then `STRESS272_RESULTS.md` §16–§18.*

---

## 0. The three things that matter

1. **The i16pipe arm landed: 394.8 s at 400 × 5601.** int16 and the build
   pipeline **compose**, slightly better than multiplicatively, and int16 is
   worth *more* with the pipeline on. Wall is now **1.63×** better than the
   start of 2026-09-09. `make check` 146/146, `sha 7c0b3d633281` unchanged —
   22 arms across five runs on one hash.
2. **"int16 makes `hp_mb` 23–30 s slower" is a timer artifact.** It has been
   reproduced five times and called the clearest open int16 question since
   2026-09-06. `hp_mb`'s phase timer is charged **6.2×** its own GPU time, and
   under `RNA_PHASE_SYNC=1` — the new instrument — int16's effect on it is
   **−0.6 %**, indistinguishable from zero. See §3 and §4a.
3. **`fetch_fML_one_H`'s decode is 5.3× faster** and byte-identical — but **no
   end-to-end win has been measured**, and this laptop cannot measure one.

---

## 1. Read the provenance line before the numbers

The first results file offered this session was the *previous* run: commit
`837cc6db`, no `i16pipe` arm, old budget order, and every wall matching the last
handoff's table to the decimal. The notebook beside it predated the commit that
added the arm by three hours.

Three cheap checks caught it — the `commit` field, `grep -c i16pipe` on the
notebook, and the arm count. **Do all three before reading any Colab result.**
Both files are now committed (`stress272_t4_pipeline.json`,
`stress272_t4_i16pipe.json`); the `837cc6db` raw had been quoted in the last
handoff but never checked in.

---

## 2. The result: they compose

| arm | wall | vs i32 | chunks | RSS |
|---|---|---|---|---|
| i32/quarter | 535.9 | — | 14 | 1.71 |
| i16/quarter | 507.1 | −5.4 % | 11 | 2.80 |
| i32pipe/quarter | 423.8 | −20.9 % | 14 | 4.64 |
| **i16pipe/quarter** | **394.8** | **−26.3 %** | 11 | 6.20 |

Naive composition predicts 401.0; the arm came in **1.5 % ahead**. int16 buys
−6.8 % against i32pipe but only −5.4 % against i32 — hiding host `build` raises
the GPU's share of wall, so int16's kernel win is diluted less. Same mechanism
§14.4 found for `output`.

Only the quarter budget ran: `BUDGETS` was deliberately narrowed in the executed
notebook, so half and natural were **scoped out, not lost**. Also ported back: the
notebook now installs `xxd`, absent from a stock Colab image, whose absence made
the build fail in a way that reads like a source error.

A pipeline-gain model now fits five arms across two runs and two datatypes to
0.5–2.6 %, with the error shrinking as chunk count rises:

    wall_pipe  ≈  wall_nonpipe  −  build_nonpipe × (chunks−1)/chunks

---

## 3. The phase timers measure whichever phase BLOCKS

Three of the four GPU phase timers in `fill_arrays_loop.c` wrap calls that only
*launch* kernels and return. The host then stops at whichever call next touches
the device synchronously — and in the GPU-resident sweep that is the **start of
`hp_mb`**, whose `upload_size_off_H()` / `upload_i_H()` are synchronous pageable
H2Ds that run *before* its own kernel. So `hp_mb` opens by draining everything
the previous phase queued.

`modular_decomposition.cu:1853` documented those blocking copies all along.
Nobody had connected the comment to the phase numbers.

`RNA_PHASE_SYNC=1` (new, `device.cu`, default off) syncs at every phase boundary.
40 × 3000, int32, same binary:

| phase | async | `RNA_PHASE_SYNC=1` |
|---|---|---|
| `int_loop` | 0.469 | **6.879** |
| `hp_mb` | **5.595** | **0.907** |
| `modular_decomp` | 5.379 | 5.318 |

**`modular_decomp` is the control and does not move** — it is the one phase that
already ended with `cudaStreamSynchronize`, so it was always honest. Reproduced
at 20 × 1200.

**Why the int16 finding falls:** `hp_mb_loop.cu` contains no int16 code at all.
`fml_decode()` has exactly one caller, `modular_decomposition.cu:1299`. There is
no mechanism by which its kernel could slow by 27 %. **A systematic attribution
artifact reproduces perfectly, which is exactly why five reproductions did not
make it true.**

What survives is the sum `modular_decomp + hp_mb` = **−35.3 s**, invariant to
where the drain lands: an improvement with no regression in it.

> **Rules for the new knob.** Compare splits between two arms *both* run with it.
> Never compare a synced split against an unsynced one, and **never quote the
> wall of a synced run** — destroying phase overlap costs ~9 % by construction.

---

## 4. The fML decode, and an honest null

The int16 fML stream stores each cell as a short offset from a baseline shared by
`FML_BLK` = 64 consecutive entries. The decode did one **dependent** baseline
load per cell. Blocking it: **2.46 → 0.46 ns/cell, 5.3×**, 15.4 → 2.9
worker-seconds at 400 × 5601. `FML_BLK` is a power of two, so the divide was
never the cost — hoisting without blocking the load is only 1.11×.

`tools/fml_decode_equiv.c` keeps the old loop **verbatim** as the reference and
compares 1500 shapes including the ragged truncations. **RED-TEAMED**: a one-off
block boundary gives 1151/1500 mismatches.

A second mutation — shifting the block index by 7 instead of 6 — made the loop
**spin rather than answer wrongly**. Termination depends on `end >= i`. A
mutation that hangs found a fragility the equality check never could; both sides
now carry the comment.

**What is not claimed.** At 40 × 3000 the decode is ~0.04 s of wall, and the
int32 **control** — which the change does not touch — moved 0.038 s between
builds. The A/B is a null and is recorded as one. On this box i16's `fetch_mx`
was *already* faster than i32's before the fix (0.269 vs 0.438), so the T4's
1.9–2.6× regression does not reproduce at 1/35th the cells on a 12-core host.
**The local machine cannot test the thing this fix is for.**

---

## 4a. The laptop was mis-diagnosed, and that changed a result

The standing note "this laptop cannot hold a GPU clock under sustained load
(1057 → 712 MHz); any local measurement below ~15 % is noise" was a **cooling**
problem, not a property of the machine. On a cool flat surface with airflow, the
same four ABBA arms wall **15.28 / 13.79 / 13.78 / 15.79** instead of
16.46 / 17.22 / 18.59 / 21.21 — the two i16 arms agreeing to **0.07 %**.

`tools/gpu_thermal_watch.py` (new) wraps a timed run and names the reason. Over
115.8 s: drift **−3.9 %**, max 74 °C, and **`SwPowerCap` in 66.8 % of samples
against `SwThermal` in 9.3 %**. **The limiter is the 65 W power cap, not heat.**
Cooling fixes the drift; nothing fixes the cap, so the card will not reach its
2100 MHz max however cold it is. `nvidia-smi -lgc` is refused under WDDM and
needs an Administrator shell besides — and would not defeat the cap either.

**That upgrade is what settled §3's question**, which the first, badly-cooled
attempt had to record as unsettleable. **Local A/B is usable for effects above
~5 %** when the run is ABBA and the drift is quoted beside the result.

## 5. Do this first next time

**One pair of arms at 400 × 5601 on a T4 with `RNA_PHASE_SYNC=1`, int32 and
int16.** It settles both open items at once: the true phase split, and whether
the decode rewrite is worth seconds at scale.

**Do not optimise `hp_mb` before that run.** On phase-synced numbers it is ~6 %
of GPU time, not the 20–25 % of wall the async profile shows, and the work would
land on the wrong kernel. The same caution applies to `backtrack` only in part —
its timer is host-side and unaffected.

If §17.5 holds at scale, the wall *decomposition* in §14.2 needs re-deriving. No
wall figure changes — those came from a stopwatch on the whole process — only the
split, and only for `int_loop`, `hp_mb` and `load_my_c`.

---

## 6. Operational

- **`origin/port27` is at `72d37d8e`. Pushed.**
- **`~/port27head` is STALE** — 75db36a1 plus partial local edits, ~3000 lines
  behind. It was left untouched. The live tree for this work is **`~/port27fml`**,
  a fresh clone at `e478419e` plus the files under test, and it is where
  146/146 was run.
- New knob, default off: **`RNA_PHASE_SYNC`**. Diagnostic only, never a run mode.
- Untracked scratch at the repo root is still there (`configure.bak`,
  `nvcc_paths.txt`, a file with a mangled name from a bad shell paste) and is
  worth deleting.
- **Next session's front is feature integration**, per Luke: the `-P` bars,
  `Energy()`'s two latent divergences, G-quad, and Defect B as an upstream patch.

---

## 7. LATER THE SAME DAY: the T4 pair landed, and §5's question is closed

`cd2b000b`. The phase-synced pair at 400 × 5601 that §5 named as the one run
that settles everything:

| phase | i32 | i16 | delta | async said |
|---|---|---|---|---|
| `int_loop` | 107.26 | 137.70 | **+30.44 (+28.4 %)** | −0.25 |
| `hp_mb` | 14.52 | 14.79 | **+0.27 (+1.9 %)** | +31.65 |
| `modular_decomp` | 216.60 | 156.49 | −60.11 | −58.49 |

**The `hp_mb` regression was attribution — +0.27 s honest.** And **the cost is
real and it is `int_loop`**, which every profile in this project has recorded as
**0.8 % of wall** and which is actually **30 % of GPU time**. `hp_mb` is 4.1 %,
not 29.8 %. Optimising from the async profile would have attacked 4 % of the
work and left the 30 % alone.

Net GPU-phase change −26.41 s against an async wall change of −27.71 s — the
split accounts for the wall.

Phase-sync cost **nothing** (535.14 synced vs 535.11 async), which is itself the
proof: there was no overlap to destroy, because the blocking H2Ds already
serialised every row. §3's "costs ~9 % by construction" was right to state and
wrong in magnitude, for the same reason as the finding. **Run it routinely.**

**The fML decode fix is measured at last**: i16 `fetch_mx` 16.41 → 10.33 with the
i32 **controls at −0.20 and −0.00**. That is the A/B §4 could not get locally.
It is still **not** a wall win, which is what §4 claimed and still claims.

Also from this session: **G-quads scoped and G0/G1 landed** (`315e3759`,
`673d439c`, `58829b0b`) — the device work is two sites, not the spec's 49, and
G1 already closes 93–100 % of the gap on the frozen bar without any record
falling below the reference. **`-g` has THREE gates**, not the one the spec
recorded, all currently holding a temporary `RNA_GQUAD_STAGING` hatch that
**must be deleted in G3**.

### Next

1. **`int_loop`** — 30 % of GPU time and int16 makes it 28 % worse. That is the
   whole of int16's remaining give-back, in one phase.
2. **G2**: the interior-loop gquad term, then delete the three hatches.
3. Re-derive §14.2's decomposition from a phase-synced run before planning from
   it.
4. The pipeline RSS multiplier should be **per-budget**: 2.8× is right at
   quarter and over-projects at natural, and it cost the last run three arms.

---

## 8. 2026-09-11 — feature integration audited, and the patch set is marked

**`-g` and `--nsp` are ACCELERATED.** The option surface now reads:

| | count |
|---|---|
| **ACCELERATED, byte-identical** | **23** |
| DECLINED, CPU route asserted | 17 |
| NEUTRAL | 19 |
| UNREACHABLE from the CLI (`--batch`) | 1 |

`PORT_OPTION_STATUS.md` is the table, and **most of it is measured rather than
argued** — `tools/verify_option_parity.sh` grew from 18 checks to 40, each
asserting byte-identity against the same binary with the accelerator off *and*
the route it claims. 11 rows are still asserted from the guard, and they say so.

### The finding: there are THREE gates and they are not the same gate

1. `gpu_path_usable()` (`RNAfold.c`) — an **optimisation**
2. `vrna_cuda_engine_supports()` (`engine.c`) — **the authority**
3. `VRNA_CUDA_BACKSTOP` (`fill_arrays.c`) — a **tripwire**

**Gate 1 is a strict subset of gate 2**, despite `RNAfold.c`'s comment claiming
it mirrors it. Gate 2 alone declines `--maxBPspan` < length, `logML`, and
multistrand. **A hole in gate 1 costs performance; a hole in gate 2 costs
correctness.** Only `-c` and `--noClosingGU` sit behind all three — which is why
lifting `-g` took three edits and not one.

### Two options no byte bar can judge

`--unordered` emits in *completion* order by design, so byte-identity reports a
difference that is the option working; `check_sorted()` was added for it.
`--ImFeelingLucky` is stochastic — **two CPU runs of the same flag differ** — so
it has a route assertion and no output bar, and cannot get one of this kind.

### Every upstream edit is now marked, and unmarked is a failure

`VRNA-PATCH-BEGIN/END(id, CLASS)` brackets all **10 regions in 8 files**, and
`tools/list_local_patches.sh` fails if a file changed against `v2.7.2` is
unmarked. It caught four I had missed. Classes: **DEFECT** (submittable alone),
**SEAM** (needs the strongest case), **REACH** (*"you already do this, we just
cannot call it"* — the easiest to defend), **DRIVER** (not proposed).

### Where circular actually stands — SHIPPED 2026-09-11 (`1159616d`)

Written as "the one genuinely missing capability"; closed the next day, and the
sentence above was right about why it was cheap. The sweep now keeps `DMLi` into
a triangular `fM2` — `PORT_CIRC_SPEC.md` had proved
`fM2_real == min_k(fML[i,k]+fML[k+1,j])` by measurement, and the kernel was
already reducing exactly that every row and discarding it. **No new arithmetic**;
it costs a chunk width of VRAM and one extra store.

Byte-identical on the frozen `tests/circ/` set, 0 cells disagreeing under
`RNA_CIRC_VERIFY`, 20 further records green across 8 arms, `make check` 154/154.
Two things were not mechanical: `postprocess_circular()` seeds its own interval
stack (so `VRNA-PATCH(bps-backtrack)` now takes `vrna_bts_t` on both sides), and
the reduction leaks near-INF sentinels (clamped circular-only;
`PORT_INVESTIGATIONS.md` item 3).

**The list of genuinely missing capabilities for single-sequence folding is now
empty.** What remains unsupported is multistrand/comparative (unreachable from
`RNAfold`) and the constraint family (declined by design). The last multi-gate
option is `--noClosingGU`.

### Out for results

`CUDA_RNAFold_IntLoop.ipynb` is running on Colab — chunk width vs datatype,
block size re-opened, and the first NCU counters this kernel has ever had.
