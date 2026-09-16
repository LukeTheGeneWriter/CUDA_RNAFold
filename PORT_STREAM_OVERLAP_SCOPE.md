# Overlapping `int_loop` with `modular_decomposition` — scope, dependency proof, and the bar

*Written 2026-09-12 against `STRESS272_RESULTS.md` §30. Nothing here is
implemented. The prize is larger than every kernel optimisation in this project
combined, and so is the risk: this is the first change that would let two
kernels touch the device at the same time.*

---

## Status

| | |
|---|---|
| dependency proof | **done, below — every buffer traced to its readers and writers** |
| one WAR hazard found | `d_energy_3p00_row`, and it is the reason a naive version would be wrong |
| implementation | **not started** |
| measured premise | **`int_loop` uses 1.6 % of DRAM peak at 17 % occupancy** — the device is idle while it runs |
| what it costs | **the phase timers.** Overlapping phases cannot be attributed separately |

---

## 1. The prize, and its ceiling

At the full VRAM budget, 400 × 5601 on an A100 (§30.7):

| | s | share of the 91.0 s wall |
|---|---|---|
| `modular_decomp` (the md graph trio) | **37.84** | 41.6 % |
| **`int_loop`** | **16.49** | **18.1 %** |
| **`hp_mb`** | **1.89** | **2.1 %** |
| `load_my_c` | 1.39 | 1.5 % |
| `fetch_mx` | 7.27 | 8.0 % |
| host stages (`build` 18.35, `backtrack` 5.70, …) | 25.19 | 27.7 % |

**`int_loop` and `hp_mb` are the two kernels that can be moved off the critical
path. Together they are 18.38 s — 20.2 % of wall.** Everything else either
carries a dependency that forbids it or is host work.

**Ceiling: 91.0 → 72.6 s, if the overlap is free.** It will not be free; the
question the measurement has to answer is how much of it survives contention.

That number is worth stating against what it is replacing. H1, H6 and H7
together are about 10 % of `int_loop`, which is **1.8 % of wall**. This is an
order of magnitude bigger and it is a *scheduling* change — the same kernels,
the same inputs, in a different order — so byte-identity is by construction
rather than by argument.

---

## 2. Why nothing overlaps today

| kernel | stream |
|---|---|
| `int_loop_warp_kernel`, `hp_mb_3p_kernel`, `new_c_kernel`, `load_my_c_kernel`, `fml_scan_kernel` | **legacy NULL stream** |
| md trio + `pack_fml` (CUDA-graph captured) | `graph_stream`, from a plain `cudaStreamCreate` |

`cudaStreamCreate` makes a **blocking** stream: work in it cannot overlap
NULL-stream work. And everything else shares the NULL stream with everything
else. **The serialisation is structural, not a consequence of
`RNA_PHASE_SYNC`** — clearing that env var changes the timers, not the
schedule.

### And there are two hard barriers per row that nobody has to ask for

The per-row `cudaDeviceSynchronize()` calls were all gated on
`!rnafold_gpu_sweep()` when the GPU-resident sweep landed — *except* these two,
which run unconditionally on the default path:

| | what | per row |
|---|---|---|
| `modular_decomposition.cu:2213` | `cudaStreamSynchronize(graph_stream)` after every graph replay | **every row** |
| `upload_size_off_H()` | a **synchronous, pageable** `cudaMemcpy` of the row's offset table — and `size_off_H` changes every row, so the content-dedup never fires for it | **every row** |

A synchronous pageable `cudaMemcpy` is itself a full barrier, so **the row loop
cannot queue ahead even one row today.** At 2 chunks that is 11,202 forced
round-trips; at `chunk_cap=29`, 78,414.

The file says so itself, at the line in question:

> *"Making those async — which needs persistent PINNED host tables, per this
> file's standing capture-region hazard — is what would unlock true
> back-to-back queueing, and it is not part of 5b."*

**This is a prerequisite, not a detail.** The offset tables are C99 VLAs on the
stack in `fill_arrays_loop.c` (`size_t size_off_H[nfiles+1];`, rebuilt per row),
and a stack buffer cannot back an async H2D. So before any two kernels can
overlap, the row loop has to become asynchronous at all:

1. `size_off_H` / `side_off_H` move from stack VLAs to **persistent pinned host
   buffers**, double-buffered on row parity (row `i+1`'s upload must not
   overwrite a table row `i`'s in-flight copy is still reading).
2. their uploads become `cudaMemcpyAsync` on the owning stream;
3. the unconditional `cudaStreamSynchronize(graph_stream)` goes.

**That work has standalone value and should be measured on its own**, before any
overlap: letting the host queue several rows ahead may be worth something by
itself, and if it is not, that is a cheap and very informative null.

---

## 3. The dependency proof

Every device buffer in the row body, with its writer and its readers. This is
the part that has to be right; the rest is plumbing.

### Row `i`, in program order

| # | call | reads | writes |
|---|---|---|---|
| 1 | `int_loop_i(i)` | `d_my_c` (rows > `i`) | `d_energy_min2` |
| 2 | `gq_internal_i(i)` | gq CSR | `d_energy_min2` (MIN2) |
| 3 | `hp_mb_3p_i(i)` | **static only** — `d_S2`, `d_sequence`, `d_pair2`, `d_hccc_*`, `d_param2`, `d_salt_loop` | `d_energy_hp_row`, `d_energy_mb_row`, `d_energy_3p00_row`, `d_gate_row` |
| 4 | `new_c_i(i)` | `d_energy_min2`, `d_energy_hp_row`, `d_energy_mb_row`, `d_gate_row`, **`d_dml1` = DMLi(`i+1`)**, `cc1` | `d_new_e`, `cc` |
| 5 | `load_my_c(i)` | `d_new_e` | **`d_my_c` row `i`** |
| 6 | `rnafold_gq_fill_row(i)` | gq CSR | `d_gq_row` |
| 7 | `fml_scan_i(i)` | `d_new_e`, `d_energy_3p00_row`, `d_gq_row`, **`d_fml_prev`** (row `i+1`), `d_up_ml_ok` | `d_energy_min` |
| 8 | md trio (graph) | `d_energy_min`, `d_fml_i`, `d_fml_j` | `d_fml_j`, **`d_dml`**, `d_fml_row`/`d_fml_j16` |
| 9 | `fml_prev_i(i)` | `d_energy_min`, `d_dml` | `d_fml_prev` |
| 10 | `md_snapshot_dml()` | `d_dml` | **`d_dml1`** |

### The three facts that make this work

**(a) `hp_mb_3p_kernel` reads no DP state at all.** Its signature carries no
`my_c`, no `fML`, no `dml` — only sequence, parameters and hard-constraint
masks. The file's own header says the `DMLi1` read that upstream's
`mb_loop_fast()` performs is *dead code here*; the real `dml1` consumer is
`new_c_kernel`. **`hp_mb_3p(i)` is a pure function of static per-record data and
can run at any point in the sweep.**

**(b) `int_loop` touches `my_c` and nothing else.** It never reads `fML`,
`dml` or `energy_min`. So `int_loop(i−1)`, which needs `d_my_c` rows ≥ `i`, is
ready the instant step 5 of row `i` completes — **before** steps 6–10 run.

**(c) Steps 6–10 touch `fML`/`dml` and never `my_c`.** So the md chain for row
`i` and `int_loop(i−1)` read and write disjoint state.

### Therefore

```
load_my_c(i) ──┬──► gq_fill(i) ► fml_scan(i) ► md trio(i) ► fml_prev(i) ► snapshot(i) ──┐
               │                          37.84 s                                      │
               ├──► int_loop(i−1) + gq_internal(i−1)   16.49 s ──────────────────────── ┼──► new_c(i−1) ──► load_my_c(i−1) ──► …
               │                                                                       │
               └──► hp_mb_3p(i−1)   1.89 s   (no dependencies whatsoever) ──────────────┘
```

The join is `new_c(i−1)`, which needs `d_dml1` from `snapshot(i)` **and**
`d_energy_min2` from `int_loop(i−1)` **and** the hp/mb rows from
`hp_mb_3p(i−1)`.

**Critical path per row becomes** `load_my_c → fml_scan → md → fml_prev →
snapshot → new_c → load_my_c`, with `int_loop` and `hp_mb` hidden inside it.

---

## 4. The hazard a naive version would hit

**`d_energy_3p00_row` is written by `hp_mb_3p(i)` at step 3 and read by
`fml_scan(i)` at step 7.** If `hp_mb_3p(i−1)` is allowed to run as early as its
*data* dependencies permit — which is immediately, since it has none — it will
overwrite that buffer before row `i`'s `fml_scan` has read it.

**A write-after-read hazard across rows, on a row buffer that is reused every
row.** It would not crash. It would produce a wrong multibranch 3′ term,
intermittently, depending on scheduling — which is precisely the failure mode a
byte-identity bar catches only if it happens to fire on the tested input.

Two fixes, and the cheap one is not the right one:

| | |
|---|---|
| **order `hp_mb_3p(i−1)` after `fml_scan(i)`** | free, but gives up the freedom that makes (a) interesting |
| **double-buffer the hp/mb row buffers on row parity** | one extra row buffer per record — `400 × 5601 × 4 B ≈ 9 MB` on this workload, against 34 GB resident. **Take this one.** |

**Every other row buffer was checked and is safe under the schedule above:**

| buffer | why it is safe |
|---|---|
| `d_energy_min2` | consumed by `new_c(i)` at step 4, which precedes step 5; `int_loop(i−1)` starts after step 5 |
| `d_energy_hp_row`, `d_energy_mb_row`, `d_gate_row` | consumed only by `new_c(i)` at step 4 — but **double-buffer them with `3p00` anyway**, they are written by the same kernel and splitting them invites a later mistake |
| `d_new_e` | written by `new_c(i)`, read by steps 5 and 7; `new_c(i−1)` cannot run until after `snapshot(i)`, which is after step 7 |
| `d_energy_min` | written by `fml_scan(i)`, read by steps 8–9 — entirely inside the md chain |
| `d_dml` / `d_dml1` | already double-buffered; that is what `md_snapshot_dml()` is |
| `cc` / `cc1` (`--noLP`) | already rotated per row |
| `d_my_c`, `d_fml_j` | persistent triangles; row `i` is written once and thereafter read-only |

---

## 5. What to build

1. **Two non-blocking streams** — `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`:
   `stream_md` (steps 6–10, replacing `graph_stream`) and `stream_cell`
   (steps 1–5). Nothing may remain on the NULL stream, or it re-serialises
   everything.
2. **Four events**, one per real edge:
   `load_my_c(i)` → `int_loop(i−1)`; `snapshot(i)` → `new_c(i−1)`;
   `int_loop(i−1)`+`hp_mb(i−1)` → `new_c(i−1)`; `load_my_c(i−1)` → `fml_scan(i−1)`.
3. **Double-buffer the four hp/mb row buffers** on `i & 1`.
4. **`RNA_STREAM_OVERLAP`**, default **0**, so today's schedule is the control
   and the A/B is in-process.
5. **`rnafold_phase_sync()` must become a no-op when overlap is on**, not merely
   unset — a sync in the middle of the row body defeats the whole change, and
   leaving it reachable is how someone benchmarks the wrong thing later.

CUDA-graph capture and replay into a non-blocking stream is legal and needs no
change to the capture region.

---

## 6. What this costs: the phase timers stop being measurements

This is the part to weigh before starting, not after.

Every performance finding in `STRESS272_RESULTS.md` §19 onward rests on
per-phase GPU times that are only meaningful **because each phase ends in a
sync**. §19 is itself the scar: an unsynced phase timer charged `hp_mb` 6.2× its
own GPU time and understated `int_loop` 14.7×, and the project spent a session
unpicking it. **Under overlap, no per-phase attribution exists at all** — two
kernels genuinely occupy the device at once, and there is no honest way to split
the elapsed time between them.

What survives:

| instrument | still valid? |
|---|---|
| **wall** | **yes — and it becomes the only aggregate** |
| `sha` | yes, and unaffected |
| `RNA_LAUNCH_STATS` per-kernel device time (CUDA events around each launch) | **yes per kernel** — but the sum no longer partitions the wall |
| per-phase host timers | **no.** They must be suppressed under overlap, not merely distrusted |

So a run with overlap on and a run with it off are **not comparable phase by
phase** — only wall to wall. Every future A/B on this kernel set has to be run
with overlap off, or re-derived.

---

## 7. The bar

**1. Correctness, and it is the whole game.** A missed edge is a data race, and
a data race is an intermittent wrong answer, not a crash.

- `sha` unchanged across the full option surface, both schedules, on **two
  fixtures** (uniform and ragged) — `tools/int_loop_lookup_bar.sh` is the shape.
- **Repeat it.** A race that resolves benignly nine times in ten is not caught by
  a single green run. Ten repeats per arm minimum, and vary the timing
  deliberately: block sizes 32/64/128/256, chunk widths, int16 on and off. Each
  changes kernel durations and therefore the interleaving.
- **`compute-sanitizer --tool racecheck`**, which this project has never been
  able to run — it cannot attach under WDDM on the laptop
  (`feedback_gpu_debugging_wsl2_lessons`) but **works on Colab**. This is the
  first change that genuinely needs it, and the first time the tool is
  available for the change that needs it.
- `make check` at 161/161 under both schedules.

**2. Prove the overlap happened.** The knob says what was asked; the device
decides. Instrument the sweep region with a CUDA event pair and compare its
elapsed time against the **sum** of per-kernel device times from
`RNA_LAUNCH_STATS`:

> `sum(device time) > elapsed(sweep)` ⟹ kernels genuinely ran concurrently.

That is a cheap in-process proof and it fails loudly if the streams silently
serialise — which is exactly what a leftover NULL-stream launch would cause. A
green wall-clock result with `sum ≈ elapsed` means the win came from somewhere
else and must be explained before it is believed. *(This is the same lesson as
H6's first bar, which was green over ten option arms having never taken the new
path.)*

**3. Then, and only then, performance.** Wall only. ABBA with overlap as the
knob, on an A100 at the **full VRAM budget** — not `chunk_cap=29`, which §30.7
showed costs 27.7 % and would muddy the result.

---

## 8. Risks, ranked by what they would cost

| risk | likelihood | what it costs | what would falsify it early |
|---|---|---|---|
| **A missed edge → intermittent wrong answer** | the whole reason this is scoped rather than written | correctness, silently | `racecheck` on Colab; repeats at varied block sizes |
| **No win: md is DRAM-bound and `int_loop`'s traffic slows it more than it gains** | real. md was measured pinned at the DRAM roof; `int_loop` needs only 1.6 % of peak, which is the argument *for*, but 1.6 % of peak is not 0 | the whole change | profile the pair together before building the full schedule — see staging |
| **The host cannot run ahead** and the row loop serialises anyway | **certain until stage 1 lands** — a synchronous pageable `cudaMemcpy` and an unconditional stream sync run every row today | all of the win | `sum(device) ≈ elapsed` on the proof above |
| **Pinned double-buffered offset tables introduce their own hazard** — an in-flight async upload whose host buffer is overwritten by the next row | moderate; it is the same class of bug as the WAR above | intermittent wrong answers | parity double-buffering, and `racecheck` |
| **Phase timers lost for nothing** | certain if the change is abandoned | the measurement apparatus, temporarily | keep `RNA_STREAM_OVERLAP=0` as the default and the timers intact under it |
| CUDA-graph capture interacts badly with a second active stream | low | rework | stage 1 below tests it in isolation |

---

## 9. Staging — cheapest falsification first

**Stage 0 — does concurrency help these two kernels at all?**
Before any restructuring: run `int_loop` and the md trio as two independent
streams over *dummy* row data in a standalone harness, and compare against
running them back to back. **If the pair is not meaningfully faster than the
sum, stop — the rest of this document is worthless.** This is a day, it needs no
change to the sweep, and it tests the one premise everything else rests on.

**Stage 1 — make the row loop asynchronous, overlapping nothing.** Pinned
double-buffered offset tables, async uploads, and the unconditional per-row
`cudaStreamSynchronize(graph_stream)` removed; md trio onto a non-blocking
stream. **Measure it alone** — this is the "true back-to-back queueing" the file
has had a TODO for, it is worth something or nothing on its own, and either
answer is useful. It is also where a capture-region hazard would surface, in
isolation rather than tangled with a race.

**Stage 2 — `hp_mb_3p` only.** It has *no* dependencies, so it is the smallest
possible version of this change: double-buffer its four row buffers, run it one
row ahead on `stream_cell`. Worth 1.89 s (2.1 %) — small, but it exercises the
whole event/double-buffer mechanism on the easiest case, and a race here is
easier to find than a race in the full schedule.

**Stage 3 — `int_loop(i−1)` against the md chain.** The 16.49 s. Only after
stage 2's machinery is proven.

**Stage 4 — decide the default**, on two architectures, as with the warp kernel.

---

## 10. What it does to the ranking

If stage 3 lands at even half its ceiling, the wall goes 91.0 → ~82 s and the
decomposition shifts again:

| | now | after |
|---|---|---|
| GPU kernels | 57.6 s, 63 % | ~39 s, **48 %** |
| host stages | 25.2 s, 28 % | 25.2 s, **31 %** |

**`build` at 18.35 s would be back to being the largest single item.** That is
worth knowing in advance: this change does not end the optimisation, it hands
the problem back to the host — which is exactly what §25 did, and §27.2 recorded
the reversal. The difference is that `build` is already threaded, so the next
move there is harder than a default flip.

---

## 11. Update after §32 (A100, run `dd406bf1`)

**The baseline is confirmed, not inflated.** §31.1 warned every wall here came
from a phase-synced run and the production wall was unknown and lower. Measured:
90.46 s production vs 90.64 s phase-synced — **−0.2 %**. Every share in §10
stands as written.

**Stage 0 is still unanswered, and `RNA_PHASE_SYNC` cannot answer it either.**
Adding ~5 syncs per row costs 0.18 s, but the host already blocks ~6× per row on
pageable `cudaMemcpy`, so the extra syncs land on an already-blocked host — the
same asymmetry that invalidated `RNA_SYNC_PROBE` in §31.2. Barrier *addition* is
cheap in both experiments; neither prices barrier *removal*.

**What did get established is the premise, and it passes.** Under forced syncs
the parts sum to the wall (89.9 of 90.7), and production has the same wall, so
**production overlaps essentially nothing today**. The 24.6 s of host stages
(27 % of wall) is fully exposed. The prize this scope estimates is real; it is
collectable only by making the transfers asynchronous, not by deleting syncs.

**But the ordering changed again.** At the fastest measured configuration
(2 chunks, build pipeline on) the wall is **85.26 s** and `build` is already
83 % hidden behind the fold. Re-derive §10's table from that arm, not from the
91.0 s figure, before committing to a race-bearing rewrite.
