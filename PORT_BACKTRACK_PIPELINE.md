# Backtrack pipelining (N5a): plan for sign-off

Roadmap N5a, opened 2026-10-06 by the A100 N2 run (`CUDA_RNAFold_N2.ipynb`; numbers in PORT_SPARSE_MD.md "N2
done"). **Status: PLAN, awaiting Luke's sign-off.** Nothing here is built.

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

**3.3 An asynchronous seam.** RNAfold folds through the library (`vrna_mfe_batch()`, "THE SEAM"), so the
overlap needs a split call:

```c
vrna_cuda_batch_t *vrna_mfe_batch_begin(fcs, n);              /* sweep; triangles retained on the device */
void               vrna_mfe_batch_finish(h, structures, mfes); /* fetch + trace + release; any thread      */
```

- `vrna_mfe_batch()` becomes `begin` then `finish`, back to back, so its behaviour is unchanged.
- Without the CUDA backend, `begin` folds on the host and `finish` only copies out. The CPU-only build stays
  correct, and the API is symmetric.
- This is library-facing API, so it is decision 9.1.

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

**3.7 Library callers** (stage P3). `engine.c` already splits a big list into VRAM-sized chunks. The same
begin/finish overlap applies there, so `RNA.fold(list)` gains too. Separate stage, separate bar.

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
| P1 | `retired_chunk_t`, fetch variants, async fetch stream, `begin`/`finish` with `vrna_mfe_batch()` = both | byte-identical, sequential; NEGCTL bites |
| P2 | RNAfold driver overlap, AUTO gate, report, refusals | §4 bars 1–4, forced multi-chunk |
| P3 | `engine.c` overlap for library lists | binding suite + `RNA.fold(list)` multi-chunk |
| P4 | A100 notebook: 400 × 5601, 3000 × 1200, soak; ABBA × 3 + P passes; trace-thread sweep | by eye (no threshold, as 9.5) |

## 6. Measurement

- **A100:** `RNA_BT_PIPELINE` 0 / 1 at 400 × 5601, 3000 × 1200, 1200 × 1200 and the soak, three
  alternating reps, with P passes. Trace threads at 4 / 8 / 12 / auto.
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
4. **Library API surface.** `begin`/`finish` is new public API in a library meant for upstream. It needs
   Luke's call (9.1) and possibly TBI's view.
5. **Error paths.** A CUDA error during `finish(k)` while `begin(k+1)` runs. Today's `gpuErrchk` exits the
   process; that stays the behaviour, but the message must say which chunk.

## 9. Decisions for Luke

1. **The seam:** new public `vrna_mfe_batch_begin`/`_finish`, or keep them internal to RNAfold + engine (P3
   only) and leave the public API as it is?
2. **Default:** AUTO on (VRAM + host-memory gated, like the build pipeline), or opt-in until P4 is measured?
3. **Order:** P3 (library lists) in this campaign, or after N3?
4. **The thread split** default (§3.6), to be measured in P4.
