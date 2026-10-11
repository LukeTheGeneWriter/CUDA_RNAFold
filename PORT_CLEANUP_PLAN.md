# PORT_CLEANUP_PLAN.md — pre-review cleanup of Lukes_Flow_Batching

**Status: SIGNED OFF by Luke 2026-10-11 (decisions in §8).** Nothing below has been deleted yet.
Process as always: plan → sign-off → implement phase by phase → bars → self-review.

The goal is to hand TBI a tree whose every line is either (a) on a default path, (b) a
fallback a default path can reach, (c) a verification switch with a documented purpose,
or (d) a user setting — and whose every edit to an upstream file is marked and explained.
Anything else is retired *with its measurement record cited*, so the review reads the
port, not its history.

---

## 1. Method

1. **Stock baseline.** Downloaded the official `ViennaRNA-2.7.2.tar.gz` release and
   compared it with our `v2.7.2` tag: **500 of 500** source files under `src/ViennaRNA` and
   `src/bin` identical (CRLF ignored). Every "vs 2.7.2" statement below is therefore a
   statement against the shipped release, not against a branch point.
2. **Coverage build** (`~/covq`, gcov). One compiler throughout — gcc-13/g++-13 for C,
   C++, link, and nvcc's host side. (The first two attempts mixed in Ubuntu's gcc 15.2;
   libgcov then silently refused to write the `.cu` objects' counts, so every `.cu`
   function read 0 %. The binary now carries exactly one `GCC: 13.4.0` string.)
3. **Default-path workload, no knob set** except as stated:
   - option parity (every accelerated option singly, plus the declined ones);
   - C_mixed, C_control, asc, desc, extreme, A_longseq_smallbatch — each plain and `-j4`;
   - multi-chunk: `F_lite` (F_extreme minus its 5000/8001 nt records, 57 recs) under
     `RNA_GPU_VRAM_BUDGET_MB=64` → 7 sweeps; `C400_lite` (12 × 5601) at 256 MB → 12 sweeps.
     The budget knob drives the same chunk loop the default path uses on a full card;
     it was the low-intensity substitute after the memory reaper stopped full
     F_extreme twice.
   - Every run: rc 0, `-j4` byte-identical to `-j1`, zero gcov errors.
4. **CPU arm** (`RNA_GPU=0`, separate counts): C_mixed and asc plain + `-j4`,
   A_longseq plain. (A_longseq `-j4` was stopped by the memory reaper; it adds no
   function the other five runs miss.)
5. **Mask.** A function is *hot* if it ran in either arm; *cold* if it ran in neither.
   Cold is necessary, not sufficient, for deletion: a cold function may be a fallback, a
   verification switch, or reachable only by an option the workload did not use (e.g.
   `postscript_layout` is cold only because the workload passes `--noPS`; plots are ON by
   default, so it is treated as hot).

Raw data: `cov2.csv` (GPU default, per instance), `covcpu.csv` (CPU arm), `stems.txt`
(cold stems) — in the session scratchpad; regenerate with `covbuild4.sh`, `covpart.sh`,
`covsum.py`, `stems.py` (to be moved into `tools/coverage/` in phase C0).

---

## 2. Hot paths changed vs stock 2.7.2

Upstream files we modify: 13 (library 8, `RNAfold.c`, 4 RNAxplorer/interface build
files). Hot functions among them, and what each change costs a user who never touches
the GPU:

| # | Function (file) | Hot in | Change | Cost on the CPU path | Action |
|---|---|---|---|---|---|
| H1 | `vrna_mfe` (mfe.c) | CPU + GPU | inside-engine hook: one `aux_grammar && engine` test before `fill_arrays` | one predictable branch per fold | keep; mark `int energy, handled;` (unmarked today) |
| H2 | `vrna_params`, `vrna_exp_params` (params.c) | CPU + GPU | mutex around the SPEEDUP_PARAMS cache (a real race: 11 999/16 000 tables wrong under threads) | uncontended lock per fold compound | keep — it is a DEFECT fix and the upstream ask; **28 of its lines sit outside the markers** |
| H3 | `vrna_gr_prepare`, `vrna_gr_reset` (grammar.c) | CPU + GPU | engine prepare/free lifecycle | two NULL tests | keep; the END marker sits mid-call and the reset hunk is unmarked |
| H4 | `process_record` (RNAfold.c) | CPU + GPU | conditional second fold compound; prefolded branch; `(vc) &&` guard; trace stamps | **two `clock_gettime` per record even with tracing off** | guard the stamps behind `g_trace_fp` (C6) |
| H5 | `postscript_layout`, `ImFeelingLucky` (RNAfold.c) | CPU (plots default ON) | `vrna_plot_layout` moved inside `THREADSAFE_FILE_OUTPUT` — fixes a `-j` crash (67faf59a) | layout is now serialised under `-j` | keep (it is a crash fix); **price it** under `-j` with plots on before review, and offer it upstream as a defect |
| H6 | `process_input`, `main` (RNAfold.c) | CPU + GPU | +333 lines, insertions only, all GPU batching/pipeline | none in a CPU-only build beyond H4 (verified by stripping `VRNA_WITH_CUDA` blocks: 239 changed lines remain, all H4/H5/trace/`apply_constraints(quiet)`) | keep; the RNAfold.c split is decision D3 |
| H7 | `apply_constraints` (RNAfold.c) | option-dependent | `quiet` parameter so a constraint warning is printed once, not twice | none | keep |

Not hot but changed: `vrna_mfe_batch`/`_backend_set`, `vrna_mfe_postprocess_circular`,
`vrna_backtrack_from_intervals_bps` (all called only by the CUDA backend), and
`vrna_gr_set_inside_engine` (called only by the Python binding's `vrna_cuda_attach`;
the CLI never binds an inside engine).

**Stale comment:** `mfe/global.h` says the project "does not call
`vrna_mfe_postprocess_circular` yet"; it runs on every circular GPU record (100 %
covered). Fix in C1.

### 2.1 Patch-marker hygiene

`tools/list_local_patches.sh` checks markers per *file*. Line by line, **81 changed lines
in the library sit outside any `VRNA-PATCH` block**:

| File | Unmarked lines | What |
|---|---|---|
| mfe/mfe.c | 40 | **`vrna_mfe_batch_backend_set()` and `vrna_mfe_batch()` entirely** — the main seam; plus `handled` |
| params/params.c | 28 | the `#else` arm, every LOCK/UNLOCK call site, the miss-path comment |
| gr_extension_mfe.c | 5 | the tail of `vrna_gr_set_inside_engine` (END marker inside the `if`) |
| grammar.c | 4 | END marker mid-argument-list; the reset hunk |
| intern/grammar_dat.h | 4 | the four NULL initialisers |

A reviewer diffing by marker would miss the batch API itself. C1 fixes the markers and
extends `list_local_patches.sh` with the line-level check (`markcheck.py`) so it cannot
drift again.

---

## 3. Knobs: 99 `getenv` names → four tiers

README documents two (`RNA_GPU`, `RNA_GPU_VERBOSE`). Proposed:

### T1 — user settings (README table, stable names)

`RNA_GPU`, `RNA_GPU_VERBOSE`, `RNA_BUILD_THREADS`, `RNA_BACKTRACK_THREADS`,
`RNA_BUILD_PIPELINE`, `RNA_GPU_VRAM_BUDGET_MB`, `RNA_FML_INT16`, `RNA_MD_SPARSE` (escape
hatch), `RNA_ENGINE_ALLOW`, `RNA_GPU_CHUNK`, and — until P4 decides their defaults —
`RNA_CPU_SLICE`, `RNA_CPU_SLICE_CAP`, `RNA_BT_PIPELINE`. **12–13**

### T2 — verification and diagnostics (keep; one table in a new `TESTING_KNOBS.md`)

Verifiers: `RNA_ROW_VERIFY`, `RNA_HC_VERIFY`, `RNA_CIRC_VERIFY`, `RNA_TRI_CHECKSUM`.
Negative controls: `RNA_BT_PIPELINE_NEGCTL`, `RNA_MD_SPARSE_NEGCTL`,
`RNA_INT_LOOP_LYNGSO_NEGCTL`, `RNA_MD_TILE_FUSE_NEGCTL`.
Selftests: `RNA_MD_SPARSE_SELFTEST`, `RNA_INT_LOOP_LYNGSO_SELFTEST`,
`RNA_MD_SPARSE_DELTA_MAX`.
Tracing/timing: `RNA_RECORD_TRACE`, `RNA_LAUNCH_STATS`, `RNA_LAUNCH_STATS_CSV`,
`RNA_PHASE_SYNC`, `RNA_HOST_AVAIL_MB`. **~19**

### T3 — tuning knobs with a live default (keep; document as "re-test on new hardware")

`RNA_INT_LOOP_LYNGSO`, `_LB`, `_PT`, `RNA_MD_SPARSE_LANES`, `_ENTRY`, `_RHO`, `RNA_C_RING`,
`RNA_ROW_BATCH`, `RNA_MD_TILE`, `RNA_STREAM_OVERLAP`, `RNA_CUDA_GRAPH`, `RNA_XFER_STAGE_MB`,
`RNA_GPU_BLOCKING_SYNC`, `RNA_MIN_GPU_BATCH`, `RNA_MIN_GPU_CELLS`, `RNA_GPU_WORK_FLOOR`,
`RNA_GPU_CHUNK_CELLS`, `RNA_GPU_UNIFORM_CHUNKS`, `RNA_GPU_SWEEP`, `RNA_MD_RING_DEPTH`,
`RNA_MD_TAIL`, `RNA_HP_TABLE`, `RNA_INT_LOOP_WARP`, `RNA_INT_LOOP_CELLS_PER_WARP`,
`RNA_FML_SCAN_THREADS`, `RNA_C_FLUSH_ROWS` (control arm for the c ring). **~26**
Each keeps only if its *off* arm still compiles to a path a test exercises; otherwise it
moves to T4 with its code.

### T4 — retire with the code they gate (each needs its record cited in the commit)

| Family | Knobs | Code (cold in both arms) | Record |
|---|---|---|---|
| Megakernel | `RNA_MEGAKERNEL`, `RNA_MK_*` ×7 | `megakernel.cu` entire (12/16 stems cold; the 4 hot are CUDA registration and the off-switch test); `int_loop_mk_ptrs`, `md_mk_ptrs` | `project_megakernel_a100_verdict`, PORT_MEGAKERNEL_SCOPE.md |
| Blocked md | `RNA_MD_BLOCK`, `_NEGCTL`, `_SELFTEST`, `RNA_MD_BLK_LANES`, `_PROFILE` | every `md_blk_*`, `md_block*_selftest_kernel`, `rnafold_md_blk_*` | RETIRED 10-06: sparse md beat b1024 by 41 % (`project_md_blocking_is_the_way_out`) |
| Slot / continuous flow | `RNA_SLOT_FLOW`, `_CAPACITY`, `_TURNOVER`, `RNA_CONTINUOUS_FLOW` | `retire_pool_*`, `retire_worker`, `on_retire_cb`, `refill_slot2`, `refill_gpu2/3`, `reset_slot_md/_nolp`, `choose_gpu` | `project_slot_flow_is_shorted`, `project_continuous_flow_C_built` (1.26–1.81× regression), noLP defect |
| Old int_loop kernels | `RNA_INT_LOOP_BLOCK_SIZE`, `_GRIDY`, `_WSEARCH`, `_WORK_STATS`, `RNA_INT_LOOP_UNROLL` (already ignored) | `int_loop_kernel_{32,64,128,256}`, `rnafold_choose_block_size`, `il_work_*` | warp scan −20.6 % (`project_warp_scan_is_default`); H6/H7 records; N3 S9 (2cf5d3ae) |
| md probes | `RNA_MD_SMEM`, `RNA_MD_BAND`, `RNA_MD_L2_HITRATIO/_MISS/_PERSIST`, `RNA_MD_ROW_SYNC`, `RNA_MD_PRUNE`, `_STATS`, `_UNSAFE`, `RNA_MD_INF_STATS`, `RNA_MD_STREAM_STATS` | `modular_decomposition_smem_kernel`, `band_fill*`, `md_inf_probe_kernel`, `md_prune_probe_kernel`, `fml_imin_kernel`, `fmli_kernel`, `rnafold_md_row_sync`; the pre-window `fetch_fML*`/`load_fML*`/`pack_fml*` (superseded by the `_w`/`_Hw` forms, which are hot) | smem null −15 % (`project_md_smem_staging_is_a_null`); band (`project_diagonal_band_and_ds_repack`); L2/prune/row-sync: **record to find** |
| Launch/sync probes | `RNA_MD3_LAUNCH_PROBE`, `RNA_SYNC_PROBE`, `RNA_LOAD_MY_C_SYNC` | `md3_probe_kernel`, `rnafold_md3_launch_probe_fire`, `load_my_c_kernel` | `project_md3_launch_multiple_priced`; others **record to find** |
| md tile CB / fusion | `RNA_MD_TILE_CB`, `_FUSE`, `_RB`, `_REVERSE`, `RNA_ROW_FUSE` | `rnafold_md_tile_*`, `tile_front_kernel`, `row_cells_kernel`, `rnafold_tile_*`, `rnafold_rowtab_*` helpers | `project_row_fusion_wins` says fusion *wins* but refuses on the default path (stream overlap 1) — **decision D4** |
| fML / c-store experiments | `RNA_FML_SCAN` (the `=2` arm), `RNA_FML_SCAN_K`, `RNA_NEW_C_STORE` | `fml_scan2_kernel`, `fml_prev_kernel`, `new_c_store_kernel` | scan v1 (`fml_scan_kernel<256>`) is HOT and stays; K-per-thread v2 lost (`project_warp_scan_is_default` lineage) — **record to cite** |

Not retired, although they look like experiments: `RNA_C_FLUSH_ROWS` and
`RNA_FML_SCAN_THREADS` gate HOT kernels (`c_flush_rows_kernel<128>`,
`md_flush_rows_kernel<128>`, `fml_scan_kernel<256>`) — they move to T2/T3.

Net: **≈42 knobs retire**, ≈57 remain, all documented.

---

## 4. Code that never runs by default, by object (function stems; every instance cold)

Stems are source-level names: template instances, device stubs and host wrappers are
folded together (by the mangled name's length prefix), and a stem is cold only if
**every** instance is cold in both arms.

| Object | Stems | Cold | What the cold set is |
|---|---|---|---|
| megakernel.cu | 16 | 12 | the whole feature; the 4 hot stems are CUDA registration + the `rnafold_megakernel()` off-switch test |
| int_loop.cu | 73 | 24 | old block kernels `int_loop_kernel_{32,64,128,256}` + `rnafold_choose_block_size`; Lyngsø v1 `ly_carry_kernel`, `ly_carry3_lb_kernel` (LB arm, pending A100), `ly_check_kernel`/`ly_report` (selftest); `il_work_*`; slot refill; `int_loop_mk_ptrs`; N5a retire fetch (opt-in) |
| modular_decomposition.cu | 135 | 50 | **dense md (`modular_decomposition_kernel`, `_smem_kernel`, all 12 instances)**; blocked md (all `md_blk_*`); band; INF/prune probes; sparse selftest; pre-window fML fetch/load/pack; slot helpers; N5a retire fetch (opt-in) |
| hp_mb_loop.cu | 43 | 9 | `fml_scan2_kernel`, `fml_prev_kernel`, `new_c_store_kernel`, `tile_front_*`, `row_cells_kernel`, `rnafold_cc_ring_select`, slot refill |
| device.cu | 78 | 26 | row-table/tile helpers (`rnafold_rowtab_*`, `rnafold_tile_*`, `rnafold_md_tile_*`); md3 launch probe; N5a retire + non-blocking xfer (opt-in, keep) |
| mfe_cuda.c | 52 | 13 | slot-flow retire pool; N5a P2 + chunk trace (opt-in, keep); `min_fml` |
| engine.c | 14 | 5 | Python-only API (keep: the binding calls it) |
| RNAfold.c | 66 | 18 | `rnafold_min_gpu_batch` (dead — confirm), CPU slice (opt-in), the rest upstream (motifs, `--ImFeelingLucky`, layout) |

Two families need a decision rather than a deletion:

- **dense md — cold, but the only fallback.** Reached by `RNA_MD_SPARSE=0` and as the
  reference in `RNA_MD_SPARSE_SELFTEST`. No option in the parity sweep made sparse
  refuse. 12 template instances (6 tile widths × plain/smem), none hot. → D1.
- **`int_loop_warp_kernel` — LIVE.** Exactly **1 of its 48 instances**
  (`<1, false, false, true, true>`) runs on the default path, i.e. some record/option in
  the workload is declined by Lyngsø and folds through the warp kernel. It is a real
  fallback, not just a selftest reference. C0 pins which input reaches it (so the bar
  covers it); D2 is then whether the other 47 instances can go.

The **fetch/finish split** in mfe_cuda.c (`backtrack_fetch_slot` / finish) was built for
slot flow and is **reused by N5a P1/P2** — it survives the slot-flow deletion; only the
retire pool and refill go.

---

## 5. Docs, tools, tests, results

**Docs.** 54 added `.md` files, 19 623 lines. `STRESS272_RESULTS.md` alone is 3 582.
Proposal:

- *Keep at top level (the review set):* README, MERGING.md, PORT_LOCAL_PATCHES.md,
  PORT_UPSTREAM_PROPOSAL.md, PORT_UPSTREAM_DIFF.md, PORT_OPTION_STATUS.md,
  PORT_FEATURE_AUDIT.md, PORT_INSTALL_VERIFIED.md, the five specs (GQUAD, CIRC, NOLP,
  NOCLOSINGGU, SALT), PORT_SPARSE_MD.md, PORT_LYNGSO_INTLOOP.md,
  PORT_BACKTRACK_PIPELINE.md, CUDA_RNAFold_History.md, this plan.
- *New:* `REVIEW_GUIDE.md` — map of the tree, the hot path end to end, the knob table
  (T1–T3), the bars and how to run them. One entry point for TBI.
- *`git mv` to `docs/archive/`:* scopes of retired work (megakernel, md blocking ×4,
  slot flow, T2b, scout, transfer, stream overlap, roofline, occupancy, warp scan,
  int_loop, host wall, CPU queue, heterogeneous, acceleration, config, NSP paramfile,
  tier-1, phase 0), session handoffs (×7), results files (STRESS272, BENCH272,
  PROFILE272 ×2, PROFILE_INT_LOOP, INVESTIGATIONS), PORT_LUKES_FLOW_BATCHING.md.
  History is preserved; nothing is lost; the top level drops to ~20 files.

**Tools/tests with no referrer** (no script, Makefile, CI or doc outside the archive
calls them): `fns_equivalence.cu`, `int16_holes.c`, `int16_worst.c`,
`build_full_featured.sh`, `local_wall_profile.sh`, `verify_cuda_glue.sh`,
`verify_int16_ab.sh`, `verify_int16_chunkwin.sh`, `verify_row_sweep.sh`,
`tests/upstream/scope_one.c`, `thread_parity.*`, `tests/noclosinggu/*_reference.txt`;
plus `proto_blocked_md*.cu` (referenced only by archived docs). → `tools/archive/` or
delete (D6). Any script that sets a T4 knob is archived with that knob.

**results/**: 20 ncu CSVs. Keep the ones a kept doc cites; archive the rest.

---

## 6. Order of work — one commit per phase, bars after each

| Phase | Content | Behaviour change |
|---|---|---|
| C0 | Tag `pre-cleanup`; move coverage scripts into `tools/coverage/`; freeze reference outputs for the bar fixtures | none |
| C1 | Marker hygiene (81 lines); line-level check in `list_local_patches.sh`; fix the stale `global.h` comment | none (comments only) |
| C2 | Delete megakernel (file, Makefile.am, knobs, `*_mk_ptrs`) | none |
| C3 | Delete blocked md + md probes | none |
| C4 | Delete slot/continuous flow (retire pool, refill); keep fetch/finish | none |
| C5 | Delete old int_loop block kernels + probes; hp_mb experiments per D4/D5 | none |
| C6 | RNAfold.c: drop `rnafold_min_gpu_batch` (if dead), guard trace stamps; trim instances per D1/D2 | none |
| C7 | Knob docs: README T1 table, `TESTING_KNOBS.md` T2/T3; any retired knob set by a user prints "retired, ignored" once | none |
| C6b | D3: move GPU batching/pipeline from RNAfold.c to `src/bin/gpu_batch.c` (move only) | none |
| C8 | Docs archive, `REVIEW_GUIDE.md`, tools/tests/results prune | none |
| C9 | Stamp the post-cleanup commit into the D4 fusion flag; tag `post-cleanup` | none |

**Bars after every phase** (all must pass, same as N5a):
CUDA build + CPU-only build (`tests/zeroconf_configure.sh`), option parity, option
matrix 36/36, `verify_gpu_cli.sh` at 4/8/16/32 MB, binding 20/20, N5a P2 self-check,
plus `grep` proof that no retired knob name remains outside `docs/archive/`. Speed:
laptop C400 A/B ABBA must stay inside ±5 % (power-capped laptop); one A100 run at the
end of C6 confirms 400×5601 and the soak are unchanged.

---

## 7. Risks

1. **Deleting a fallback.** Dense md and warp int_loop are cold but load-bearing — they
   stay unless D1/D2 says otherwise.
2. **Harness drift.** Notebooks and `tools/` scripts set T4 knobs; a retired knob silently
   ignored would make an arm measure the default. C7's "retired, ignored" line and the
   grep bar make it loud.
3. **Fetch/finish is shared** between slot flow and N5a — C4 must delete callers, not the
   split.
4. **Upstream ask.** Marker fixes must not change code bytes in upstream files (C1 is
   comments only); PORT_LOCAL_PATCHES.md is regenerated from the source.
5. **Coverage blind spots.** The workload used `--noPS`, no motifs, no `-p`. Anything
   proposed for deletion is also grepped for callers; cold-by-coverage alone deletes
   nothing.
6. **Grouping errors in the mask itself.** The first pass of this plan grouped template
   instances by a regex and wrongly listed `int_loop_warp_kernel`, `fml_scan_kernel`,
   `c_flush_rows_kernel` and `md_flush_rows_kernel` as cold — all four are HOT. Fixed
   by grouping on the mangled-name length prefix (`stems.py`); every deletion commit
   re-checks its targets against the raw per-instance CSV, not the grouped list.
7. **Memory on this laptop.** The reaper stopped three workload pieces; every bar run is
   split into one fixture per job.

---

## 8. Decisions (Luke, 2026-10-11)

- **D1 — dense md:** keep **one** tile width as the `RNA_MD_SPARSE=0` fallback and the
  sparse selftest's reference; delete the other 11 instances (C6).
- **D2 — warp int_loop:** C0 pins which inputs reach which instances; keep the instances a
  pinned input or selftest reaches, add those inputs to the bar, delete the rest (C6).
- **D3 — RNAfold.c:** split. The GPU batching and pipeline functions move to
  `src/bin/gpu_batch.c`; `RNAfold.c` keeps only the hooks. Own phase (C6b), move only, bars.
- **D4 — md tile CB / row fusion:** **keep and flag.** It stays (T3, not T4) pending final
  tests; if it does not pay it is canned then. The source carries a comment at the fusion
  code naming the commit before this cleanup (`pre-cleanup` tag) and the commit after it
  (stamped by the last phase, C9), so it can be removed or revived against a known tree.
- **D5 — `RNA_ROW_BATCH=0`:** keep (T3); the scan kernels it relies on are hot anyway.
- **D6 — archive:** `git mv` to `docs/archive/` and `tools/archive/`; nothing deleted.
- **D7 — H5 layout lock:** price it (laptop, `-j4` plots on vs `--noPS`, ABBA), then offer
  it upstream as a defect fix.

Phase changes from these: C5 no longer touches md tile CB / fusion (`RNA_MD_TILE_CB`,
`_FUSE`, `_RB`, `_REVERSE`, `RNA_ROW_FUSE` move from T4 to T3, flagged); a new **C6b** does
the D3 split; a new **C9** stamps the post-cleanup commit into the D4 flag comment and tags
`post-cleanup`.
