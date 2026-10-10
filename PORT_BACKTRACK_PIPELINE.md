# Backtrack pipelining (N5a): plan, signed off 2026-10-10

Roadmap N5a, opened 2026-10-06 by the A100 N2 run (`CUDA_RNAFold_N2.ipynb`; numbers in PORT_SPARSE_MD.md "N2
done"). **Status: SIGNED OFF by Luke 2026-10-10** with the decisions in section 9. §3.3 and §3.7 were
rewritten to match decision 9.1: no new public API. P0 is in progress.

## 1. Why

At 400 × 5601 on the A100, with sparse md on by default, LFB walls **43.6 s**. The phase-synced pass splits it:

| phase | s | share |
|---|---|---|
| `int_loop` | 16.5 | 34 % |
| md (sparse) | 12.6 | 26 % |
| **backtrack (trace)** | **8.3** | **19 %** |
| **fetch (device → host triangles, inside the backtrack phase)** | **3.3** | **8 %** |
| `hp_mb`, `load_my_c`, exposed build, rest | ~6 | |

**Backtracking is exposed.** In the default batch path, `par_mfe()` sweeps a chunk, then `backtrack_all()`:
- **fetches** each record's c and fML triangles from the device into a per-worker scratch;
- **traces** the record on the host;
- `fold_gpu_batch()` then tears the device down;
- only then can the next chunk's sweep start.

So the GPU idles for the whole 11.6 s, nine times a little over 1.3 s.

Per chunk, the GPU sweep takes about 3.5 s and backtracking about 1.3 s. If chunk *k* is traced while chunk
*k+1* sweeps, all but the last chunk's backtrack disappears from the wall:

- **Upper bound:** (9−1)/9 × 11.6 ≈ **10.3 s, about −24 %**.
- **Realistic:** **−6 to −10 s**. The host threads it needs are shared with the builder (21.2 s of building,
  already 89 % hidden behind the GPU).

This beats device backtracking (N5) on the A100: it uses idle host cores instead of adding GPU time, and needs
no GPU traceback kernel. N5 stays on the roadmap for hosts with few cores, and for VRAM/host-RAM reasons
(SparseMFEFold-style).

## 2. What exists to build on

- **The split already exists, for slot flow (Phase 4b, the retire pool).** `backtrack_fetch_slot()` is the
  only part that touches the device. `backtrack_finish_slot()` is pure host work: circ check, fM1, exterior
  loop, MFE read and trace. It already runs on worker threads while the sweep carries on. The batch path
  calls the two back to back inside `backtrack_all()`.
- **The build pipeline** (RNAfold.c `pipeline_flush()`): the builder thread builds chunk *k+1* while the main
  thread folds chunk *k*. Its AUTO gate (half of MemAvailable) and its `OVERLAPPED` report are the model for
  this one.
- **Per-worker persistent scratch** (`bt_pool_get`, T2a, maybe pinned). Host RAM stays at one record per
  worker, and this plan keeps it that way.
- **VRAM is free under the cell cap.** A full-cap chunk is 8.2 GB as int32 (the int16 AUTO banner), against
  ~68 GB usable on the A100-80GB. Two chunks' worth fits several times over. The cap, not VRAM, sizes chunks,
  and the cap is FASTER than wide chunks (−18 % to −32 %, [chunk cap memory]).

## 3. Design

**3.1 Retain the triangles, not the chunk.** When a chunk's sweep ends, the device buffers backtracking
reads move into a `retired_chunk_t` instead of being freed:
- `d_my_c`;
- `d_fml_j`, or `d_fml_j16` plus its baselines `d_fml_b`, `d_base_off_H` and `d_colb_off`;
- `d_fm2` (circular);
- the host `tri_off_H`.

Everything else is torn down as today, and the next `init_gpu()` allocates fresh globals. The fetch functions
(`fetch_my_c_one_w`, `fetch_fML_one_Hw`, `fetch_fm2_one_w`) gain variants that take the retired set, so the
fetch never reads a global the next sweep is rewriting.

**3.2 Fetch on its own stream.** The fetch becomes `cudaMemcpyAsync` on a dedicated non-blocking stream into
the (pinned) worker scratch, followed by a synchronise of that stream only. The sweep's kernels keep running
on theirs. Risk 8.1 covers legacy default-stream semantics.

**3.3 The seam: unchanged (decision 9.1).** No `vrna_mfe_batch_begin`/`_finish`. The overlap is built from
infrastructure that already exists, and work still goes to the GPU in **batches**, never record by record:

- **`vrna_mfe_batch()` keeps its signature and stays synchronous** to its caller. The public headers and the
  SWIG interface do not change, so TBI's review surface does not grow.
- **The overlap lives inside the CUDA backend.** It retains chunk *k*'s triangles (§3.1) and hands *k*'s
  records to the existing retire-pool machinery: `backtrack_fetch_slot` (device to host, on the fetch
  stream) and `backtrack_finish_slot` (pure host work, on worker threads). Meanwhile chunk *k+1*'s sweep runs.
- **Results leave through the existing ordered output.** RNAfold already reserves each record's slot in
  `vrna_ostream_t` at read time and provides it when the result is ready. A small internal (non-public) hook,
  in the style of `rnafold_circ_expect()`, tells the driver "record *r* is finished", and the driver
  dispatches it through its existing `-j` pool (`RUN_IN_PARALLEL`). Output order is unchanged.
- **Slot flow is not revived.** Per-record admission (`RNA_SLOT_FLOW`) is off by default and has no A100
  performance case. Admission stays chunk by chunk under the cell cap; only retirement of a finished chunk
  overlaps the next sweep.
- **Without the CUDA backend nothing changes:** `vrna_mfe_batch()` loops over `vrna_mfe()` as it does today.
- **What it costs the driver:** the build pipeline must stay one chunk ahead of the backend, not one chunk
  ahead of the call, and the driver must not free a chunk's fold compounds until the hook has released every
  record in it.

**3.4 The driver schedule** (RNAfold.c, the pipelined path):

```
main:     ... sweep(k) ──hand k──► sweep(k+1) ──hand k+1──► sweep(k+2) ...
backtrk:                 [finish(k): fetch+trace]  [finish(k+1)]
builder:  [build(k+1)]            [build(k+2)]
```

- At most **two triangle sets** live on the device: *k* (tracing) and *k+1* (sweeping).
- Before `begin(k+2)`, main waits for `finish(k)`. That frees *k*'s triangles and releases *k*'s records for
  output.
- Output order is unchanged: records dispatch by ostream slot, assigned at read time, as the build pipeline
  already relies on.

**3.5 Gates, positive evidence, refusals.**
- **AUTO, on when:**
  - two full-cap chunks fit in VRAM;
  - the build pipeline's host-memory gate passes;
  - the input has more than one chunk.
- **Otherwise sequential, as today.** A 4 GB card cannot hold two chunks at the default cap, so the laptop
  keeps today's schedule unless the cap is lowered. `RNA_BT_PIPELINE=0|1` overrides.
- **Report:** a `backtrack pipeline: N chunks, traced X s, OVERLAPPED Y s` line, like the build pipeline's,
  so the overlap is measured, not inferred.
- **Refused, with a reason:**
  - slot flow and continuous flow, which have their own retire pool;
  - the megakernel;
  - `--commands`/output paths that build compounds of their own (the build pipeline's
    `output_needs_compound()` rule).

**3.6 Thread split.** The trace pool and the builder share the host. Default: trace threads =
`hw − builder threads`, at least 1. Measured, not assumed (section 6).

**3.7 Library callers** (stage P3, in this campaign per decision 9.3). `engine.c` already splits a big list
into VRAM-sized chunks. Because the overlap lives in the backend (§3.3), the same schedule serves
`RNA.fold(list)`: the CLI and the library share one schedule. Separate stage, separate bar.

## 4. Correctness

The answer cannot change by construction: the same triangles are fetched and the same trace runs, only later
and on another thread. The bars make sure the construction holds.

1. **Byte-identical** to sequential (`RNA_BT_PIPELINE=0`) and to `RNA_GPU=0`, with triangle checksums, on:
   - the option matrix (the S3 bar's cases);
   - `verify_option_parity`, `verify_option_matrix`, `verify_gpu_cli` (budgets 4/8/16/32: multi-chunk);
   - the binding suite and the 2400 soak.
2. **Forced multi-chunk on the laptop** (`RNA_GPU_CHUNK_CELLS` small, `RNA_BT_PIPELINE=1`): many hand-offs,
   so retention/reallocation races have room to show.
3. **Negative control** `RNA_BT_PIPELINE_NEGCTL=1`: free the retained set before the fetch, or let
   `begin(k+2)` start without waiting for `finish(k)`. It **must** change output or trip the retained-set
   assert. A control that cannot fail proves nothing.
4. **Positive evidence:** the OVERLAPPED line, plus `RNA_RECORD_TRACE`. Each record's fold window must
   overlap the next chunk's sweep window.
5. **compute-sanitizer is unavailable under WSL2**, so the device side is checked by retained-set asserts
   (pointer non-NULL, chunk ID matches) in `--enable-asserts` builds.

## 5. Stages

| stage | content | bar |
|---|---|---|
| P0 | instrument per-chunk sweep / fetch / trace spans (`RNA_RECORD_TRACE`), laptop and A100 | the gap is where §1 says |
| P1 | `retired_chunk_t`, fetch variants, async fetch stream, backend-internal retire of a retained chunk | byte-identical, sequential; NEGCTL bites |
| P2 | RNAfold driver overlap, AUTO gate, report, refusals | §4 bars 1–4, forced multi-chunk |
| P3 | `engine.c` overlap for library lists | binding suite + `RNA.fold(list)` multi-chunk |
| P4 | A100 notebook: 400 × 5601, 3000 × 1200, soak; ABBA × 3 + P passes; trace-thread sweep | by eye (no threshold, as 9.5) |

## 6. Measurement

- **A100:** `RNA_BT_PIPELINE` 0 / 1 at 400 × 5601, 3000 × 1200, 1200 × 1200 and the soak, three
  alternating reps, with P passes. Trace threads at 4 / 8 / 12 / auto, plus the 9.4 formula
  (hw − builder threads) as its own arm.
- **The thread sweep runs under FULL heterogeneous load (Luke, 2026-10-10).** The CPU slice
  (`RNA_CPU_SLICE` + `-j`) folds records while the builder builds, the device sweeps and the trace pool
  traces, so the four compete for the host as they will in production. Record per arm: wall, OVERLAPPED
  seconds, the wait at each join, builder span, CPU-slice share, and the host's run-queue/CPU utilisation.
  Find the split where they stop stealing from each other. An unloaded sweep alone would pick the wrong
  sweet spot.
- **9.4 as signed off yields one trace thread** wherever the builder defaults to nproc
  (`RNA_BUILD_THREADS` auto), which would serialise the trace. Until P4 decides, the trace pool keeps
  today's backtrack default (hardware threads, overlapping the builder as the build pipeline already does).
- **The prize is the exposed backtrack.** Report wall, the OVERLAPPED seconds, and the residual backtrack:
  the last chunk's, plus any wait at `begin(k+2)`.
- **Laptop:** correctness only. A 4 GB card declines by the VRAM gate unless forced with a small cap.

## 7. Synergies and tradeoffs

| pair | relation |
|---|---|
| N5a + Lyngsø (N3) | ADDITIVE. N3 shortens the sweep: about 3.5 → 2.9 s per chunk against about 1.3 s of backtrack, still fully hidden. Below about 1.5 s per chunk, the backtrack becomes the floor. |
| N5a vs device backtrack (N5) | EITHER/OR on the A100: N5a hides the host trace for free. N5 remains for few-core hosts. |
| N5a + build pipeline | SHARE the host cores: thread split §3.6 |
| N5a + chunk cap | SYNERGY: the cap keeps chunks small, so two fit in VRAM; the cap is also the faster setting |
| N5a + int16 AUTO | NEUTRAL. Retention doubles triangle VRAM, and AUTO already prices one full-cap chunk; the N5a gate prices two. |
| N5a + heterogeneous CPU folding (C4) | TRADEOFF for host cores, the same budget |

## 8. Risks

1. **Default-stream semantics.** If the build uses the legacy default stream, any default-stream call by the
   fetch serialises with the sweep. The plan uses explicit non-blocking streams. P0 checks the build flags
   (`--default-stream`).
2. **Host contention:** builder plus trace pool plus the sweep's own host work. The thread split and P4
   measure it.
3. **Global device pointers.** The sweep uses globals across three files. Retention must move exactly the
   fetch set and nothing else; asserts guard it.
4. **Library API surface.** Settled by 9.1: none added. The risk moves to the internal driver hook, which must
   stay out of the public headers.
5. **Error paths.** A CUDA error during `finish(k)` while `begin(k+1)` runs. Today's `gpuErrchk` exits the
   process; that stays the behaviour, but the message must say which chunk.

## 9. Decisions (signed off by Luke, 2026-10-10)

1. **The seam: no new public API.** Use the infrastructure that exists: `vrna_mfe_batch()` as is, the retire
   pool's fetch/finish split, `vrna_ostream_t` ordering and the `-j` pool. Batches go to the GPU, not
   individual sequences; slot flow stays off. Written up in §3.3.
2. **Default: opt-in** (`RNA_BT_PIPELINE=1`) until the P4 A100 run is measured; then AUTO, gated on VRAM and
   host memory like the build pipeline.
3. **Order: P3 (library lists) is in this campaign.**
4. **Thread split:** trace threads = hardware threads − builder threads, at least 1. Swept at 4 / 8 / 12 / auto
   in P4 and changed only if the measurement says so.

**Next after N5a (Luke, 2026-10-10):** the device-side backtracking experiment (N5).

## 10. Results

### 10.1 P0, laptop (RTX 3050 4 GB, 2026-10-10)

`RNA_RECORD_TRACE=<path>` now also writes `<path>.chunks.csv`: one line per GPU chunk with device init, sweep
and backtrack spans, on the record trace's time axis (`rnafold_chunk_trace_open()`, mfe_cuda.c). The sweep's
end is a device drain, taken only while tracing. Output is byte-identical with the trace off, on, and on with
a forced cap (`A_longseq_smallbatch`, 60 × 2000, one sha across all three).

| fixture | chunk | sweep (s) | GPU idle in backtrack (s) | fetch / busy (worker-s) | gap to next chunk (s) |
|---|---|---|---|---|---|
| C400 (40 × 5601) | 0 (21 rec) | 8.64 | **1.97** | 14.35 / 20.65 | 0.20 |
| | 1 (19 rec) | 7.83 | **0.57** | 3.15 / 5.82 | — |
| 60 × 2000, cap 2e7 | 0–6 (5–20 rec) | 0.48–0.64 | 0.03–0.11 | — | 0.004–0.010 |

- **The gap is where §1 says:** after each sweep the GPU idles for the whole fetch + trace. The time between
  chunks (teardown, free, waiting for the builder) is small, because the build pipeline has the next chunk
  ready.
- **On this laptop the prize is small:** 2.5 s of 23.1 s at 40 × 5601. The A100's sweeps are about 2.3× shorter (~3.5 s per
  chunk, §1) against a similar host backtrack, which is why §1 measured 23 % there.
- **New: the first chunk's fetch costs 4.6× the second's** (14.35 vs 3.15 worker-s), for a similar number of
  records. That fits the first touch of the persistent scratch (T2a). Under N5a, chunk 0's backtrack
  overlaps chunk 1's sweep, so this is hidden as well. It also means a one-chunk run cannot gain anything.
- **The A100 half of P0** is folded into P4: its `RNA_BT_PIPELINE=0` arm carries the chunk trace, which is
  the same measurement, so it costs no separate GPU session.

### 10.2 P1, laptop (2026-10-10): retire, fetch from the retired set, release -- sequential

Built: `bt_retired.h` (`rnafold_retired_t`); `rnafold_retire_chunk()` drains the device and DETACHES c, fML
(int32, or int16 + baselines, chosen by the pointer that was allocated) and fM2 from the live globals, so
teardown and the next init neither free nor reuse them; `rnafold_retired_fetch_*_w()` read them on new
NON-blocking copy streams (`g_xfer_nb[]`, risk 8.1); `rnafold_retired_release()` frees by the pointer. The fML
decode is one body (`fml_fetch_decode()`) shared by the live and retired fetches. `RNA_BT_PIPELINE=1` routes
`par_mfe()`'s batch path through it, still in today's order; slot flow and the turnover self-check keep the
old path. Every fetch checks the set's chunk id. Positive evidence: `backtrack pipeline: N chunk(s) retired`.
Negative control `RNA_BT_PIPELINE_NEGCTL=1` zeroes the retained c triangle before the fetch.

| bar (all with `RNA_BT_PIPELINE=1`) | result |
|---|---|
| self-check, 8 cases × CPU / off / on / NEGCTL: multi-chunk (7 chunks, int32), one chunk (int16), mixed, `-c`, `--noLP`, `-g`, `--salt`, `-j4` | **8/8**: one sha per case across CPU, off and on; triangle checksums equal; retired count ≥ 1; **NEGCTL changes the output in all 8** |
| `verify_option_parity` | option surface = CPU |
| `verify_option_matrix` | 36/36 |
| `verify_gpu_cli` budgets 4/8/16/32 | byte for byte (F_extreme 12 sweeps) |
| binding suite | 20/20, and 20/20 with the knob off |

The C_mixed fixture has no G-quadruplex, so the `-g` case shows the `-g` path folds through the retired set,
not that a G-quad structure survives it. The `-g` fixtures in the gpu_cli bar cover that.
