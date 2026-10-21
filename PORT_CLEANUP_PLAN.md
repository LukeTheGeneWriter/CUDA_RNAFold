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
(cold stems) — kept outside the repo; regenerate with `tools/coverage/` (C0):
`build.sh` (gcc-13 throughout), `part.sh` (one workload piece per job),
`summarize.py` (gcov → CSV), `stems.py` (group by length prefix; reproduces the
original `stems.txt` byte for byte).

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
  (`<1, false, false, true, true>`) runs on the default path. *Corrected in C0:* this is
  not a Lyngsø-declined fallback — the template is `<CELLS_PER_BLOCK, GRIDY, WSEARCH,
  RING, LY>` and `LY = true` **is** Lyngsø; the warp kernel is the default int_loop
  kernel, every record, every input. The 48 are 4 cpb × 3 grid/search × 2 ring × 2 LY,
  and each axis is a knob: cpb > 1 only from `RNA_INT_LOOP_BLOCK_SIZE` (T4), GRIDY /
  WSEARCH only from `RNA_INT_LOOP_GRIDY` / `_WSEARCH` (T4), RING from `RNA_C_RING` (T3),
  LY from `RNA_INT_LOOP_LYNGSO` (T3). D2 therefore keeps **4** instances,
  `<1, false, false, {ring}, {LY}>`, and the bar pins the three off-default ones with
  `RNA_C_RING=0` and/or `RNA_INT_LOOP_LYNGSO=0` (C5).

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

**C0 record (2026-10-11).** `pre-cleanup` = 6123cebe. References frozen from that tree
(WSL `~/pyq`, laptop RTX 3050) into `~/cleanup_ref` by `tools/coverage/refcheck.sh
freeze`; every one rc 0 and `-j4` byte-identical:

| Fixture | Sweeps | sha256 (16) |
|---|---|---|
| C_mixed | 1 | 316dd8a88db079be |
| C_control | 1 | 64682fa0020055f4 |
| asc | 1 | 871e271d384a7b8e |
| desc | 1 | be952e53be2c129d |
| extreme | 1 | 4558eae00a24edeb |
| A_longseq_smallbatch | 1 | baae6086e8dc7a10 |
| constrained (no `-C`: CPU path) | 0 | fcd12cfff84dac90 |
| F_lite @ 64 MB | 7 | c4389886294474bf |
| C400_lite @ 256 MB | 12 | e4320d5b914367b8 |

`refcheck.sh check` re-folds plain and `-j4` and requires both byte-identical to the
frozen output **and** the same sweep count (so a chunk-loop change cannot hide).
Negative control: one `(` → `[` in the frozen C_mixed structure → FAIL, exit 1.

**Bars after every phase** (all must pass, same as N5a): `refcheck.sh check` on all nine,
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
  *C0 result:* 4 of 48 kept (§4); they are reached by knobs, not inputs.
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

**Reclassified while implementing (C3, 2026-10-11).**
- `RNA_MD_ROW_SYNC` and `RNA_LOAD_MY_C_SYNC` are **T3, not T4**: each is the escape
  hatch for a default that was priced and changed (−5.3 % together on the A100,
  PORT_LUKES_FLOW_BATCHING.md 9.12), and the row sync is forced on off the
  GPU-resident sweep. Both stay.
- `RNA_MD_SMEM`, `RNA_MD_BAND`, `RNA_MD_PRUNE` (`_STATS`, `_UNSAFE`) move from C3 to
  **C6**: they are threaded through the dense md kernel's signature and `md_cell`,
  which D1 cuts to one width there anyway, and the C6 bar adds a `RNA_MD_SPARSE=0`
  arm that reaches that kernel. C3 = blocked md + the standalone probes
  (`RNA_SYNC_PROBE`, `RNA_MD3_LAUNCH_PROBE`, `RNA_MD_INF_STATS`,
  `RNA_MD_STREAM_STATS`, `RNA_MD_L2_*`).

**C5 record (2026-10-11).** fdef63aa. The three off-default warp int_loop instances (D2)
get their own bar, `tools/coverage/knobarms.sh`: `RNA_C_RING=0`, `RNA_INT_LOOP_LYNGSO=0`
and both, on C_mixed, extreme and F_lite @ 64 MB — output equal to the frozen reference
**and** stderr showing the arm took its path (the ring/Lyngsø ACTIVE lines present in
the default arm, absent in the arm that turns them off). 12/12.

**C4 notes (2026-10-11).**
- Deleted with slot/continuous flow: `RNA_SLOT_FLOW`, `RNA_CONTINUOUS_FLOW`,
  `RNA_SLOT_TURNOVER` and `RNA_SLOT_CAPACITY` (the last two were its test modes), the
  schedule, the retire pool, `refill_gpu2/3`, `refill_slot2`, `reset_slot_md/_nolp`,
  `rnafold_rowtab_upload_row`, `RNA_I_ROW()` (now `i`; the kernel asserts lose their
  `i_row < 0` escape and are strict). The fetch/finish split stays.
- RNAfold's chunk projection was slot-flow aware (`projected_chunk_bytes`, a sorted
  `len_desc`); with one buffer set per record it is a running sum, exactly equal.
- **The option matrix keeps 36 pairs**: its `slotflow`/`contflow` arms are replaced by
  `ring0`/`ly0`, so every CLI option now meets the two off-default int_loop kernels.
  `verify_option_triples.sh` (every triple had a flow dimension) and
  `verify_retire_pool.sh` move to `tools/archive/`; `verify_nolp_parity.sh` keeps its
  mixed-length arm without slot flow.
- `choose_gpu()` is **not** deleted here although the T4 table lists it: it is WBL's
  original device-query helper, unrelated to flow, and dead. Left for C6 with the other
  cold leftovers, to be decided with Luke rather than swept up silently.

Phase changes from these: C5 no longer touches md tile CB / fusion (`RNA_MD_TILE_CB`,
`_FUSE`, `_RB`, `_REVERSE`, `RNA_ROW_FUSE` move from T4 to T3, flagged); a new **C6b** does
the D3 split; a new **C9** stamps the post-cleanup commit into the D4 flag comment and tags
`post-cleanup`.

**C4 record (2026-10-11).** 421295e8. All bars green, plus `verify_nolp_parity.sh`, whose
`--noLP + RNA_FML_INT16 refuses at init` check had been failing since int16 AUTO (069981ff,
09-27): the refusal moved to the stand-down in mfe_cuda.c and says "cannot be honoured",
not "cannot be combined". The grep now accepts both; the code was always right. **First
laptop ABBA** (C400, `--noPS`, pre-cleanup `~/pyq` = A vs C4 `~/pyqc` = B, 8 runs): B/A
1.027, inside the +/-5 % bar; the laptop drifted 20 -> 26 s over the run (first ABBA block
0.993, second 1.055), all 8 outputs identical.

**C6 notes (2026-10-11).**
- D1: the dense md keeps **one** instance, `modular_decomposition_kernel<32>` (`MD_TILE`).
  Deleted with the other widths: `RNA_MD_TILE` (T3 -> retired: it only chose the dense
  width), the smem twin and `RNA_MD_SMEM`, the global band and `RNA_MD_BAND`, the fML
  corner descriptor (`fml_corner_t`, md_dev.h), and `RNA_MD_PRUNE` with `_STATS`,
  `_UNSAFE` (block-minimum summaries, their upkeep in `load_fML_cell`, the probe kernel).
  int16 AUTO's PRUNE conflict went with it; the `--noLP` stand-down is unchanged. A
  tombstone comment in modular_decomposition.cu names what went and points at `pre-cleanup`.
- The PRUNE_STATS section had been pasted into the middle of the CUDA-graph comment above
  `load_fML_modular_decomposition_load_min_fML()`; the comment is whole again.
- D2 needed nothing here: C5 already left `int_loop_warp_kernel<RING, LY>`, 4 instances.
- `rnafold_min_gpu_batch()` is **not dead**: the CPU slice cap reads it. Kept.
- H4: `process_record()`'s two per-record clock reads go through `trace_stamp()`, which
  reads the clock only while `RNA_RECORD_TRACE` is open.
- knobarms gains `sparse0` (`RNA_MD_SPARSE=0`, the dense fallback; int16 AUTO turns on
  there, so it also drives the int16 decode in `md_cell`) and `selftest`
  (`RNA_MD_SPARSE_SELFTEST=1`, dense as the reference, must report 0 mismatching); the
  default arm now also requires `RNA_MD_SPARSE ACTIVE`. The C6 chain adds dense md under
  `-c` (the fM2 writer) against upstream.
- `choose_gpu()` (WBL's original device query, dead) is still in place, for Luke to decide.
- **C6 bars (2026-10-12):** refcheck 9/9, option parity, matrix 36/36, gpu_cli 4/8/16/32,
  binding 20/20, N5a P2, noLP all green; knobarms 18/18 (`sparse0` sweeps 5 chunks, not 7, on
  F_lite @ 64 MB -- int16 AUTO turns on with the dense md and halves the triangle; same
  output); dense md under `-c` equals upstream with int16 on and off.

**C7 record (2026-10-12).** One table of retired knobs in device.cu (`g_retired`, 46 names
in 7 families, replacing the single RNA_INT_LOOP_UNROLL notice): each one set prints
`IGNORED: retired with <family>` once; checked with four set (one line each, output
unchanged) and none set (no line). README gains the T1 table; TESTING_KNOBS.md carries
T2/T3, with the D4 knobs as a "flagged" group. All bars green, and **the zeroconf CPU-only
bar passes (first run in the cleanup, 21/21 -- recorded as 22/22 in the C7 commit message, a miscount: one line was printed twice)**, on a copy of ~/pyqc. Retired-name grep:
in src only deletion notes remain; three tools still SET retired knobs
(`int_loop_lookup_bar.sh`, `nb_dryrun.py`, `intloop2_addendum.py`) and go to the archive in C8.

**C6b record (2026-10-12).** D3, move only: RNAfold.c 4031 -> 2012 lines; `src/bin/gpu_batch.c`
(2046) holds the GPU chunk path; `src/bin/RNAfold_gpu.h` (private) holds `struct options`,
`struct record_data` and the seam. process_input()'s three CUDA sections are
`gpu_batch_begin/offer/end`, its accumulator locals are file statics reset per input stream,
and `RUN_IN_PARALLEL(process_record, r)` is `rnafold_dispatch_record(r)` -- parallel_helpers.h
DEFINES the pool's globals, so only RNAfold.c may include it. Six functions lost `static` to
cross the seam (apply_constraints, add_ligand_motif, trace_now, trace_open,
output_needs_compound, cpu_slice_finished; no libRNA symbol collides). Lossless check: every
old line not in the new files is one of those rewrites. Bars all green incl. zeroconf, plus
move arms that ASSERT their path: CPU slice (A_longseq, -j4 / -j8: 2 / 4 records sliced),
too-big (1 MB budget: GPU on, 0 sweeps, all 60 records via the old `continue`), build
pipeline 0/1, backtrack pipeline -j4, two input files (begin resets: 2 sweeps, 2 ON lines).
The first slice and too-big arms took neither path (C400_lite chunks are one record; a 1 MB
budget still fits short records) -- caught by their stderr counters and rerun.

## 9. Queued after the cleanup (Luke, 2026-10-12)

- **User RAM / VRAM / thread budgets.** A number in MB or `quarter` / `half` / `threequarter` /
  `full` of the machine's TOTAL memory, as an upper limit under today's soft constraints
  (0.85 x free VRAM, half of MemAvailable), so it still adapts to what is free. Python
  defaults to `half`, the CLI to `full`; a record that cannot fit warns and folds anyway.
  Includes the VRAM gaps (fraction syntax, a Python surface, the CUDA context counted) and
  a true host-RAM prediction (backtrack scratch, pinned staging, the pipeline's second chunk,
  retained triangles -- measured before modelled). Threads (build, backtrack, -j) take the
  same fraction syntax of the core count.
- **Future, unscoped:** multi-GPU (choose a device; run several) and cross-vendor backends
  (HIP, SYCL, OpenCL, Vulkan, Metal).

**C8 record (2026-10-12).** `choose_gpu()` / `use_cuda` deleted (Luke: yes; device selection
kept as a future item, section 9). 35 docs and the uncited `results/` to `docs/archive/`;
66 citations rewritten to `docs/archive/<name>` (all comments, one `echo` message). 18
unreferenced or retired-knob tools and test files to `tools/archive/` (each re-checked for
referrers). REVIEW_GUIDE.md is new. **The bars are now runnable by a reviewer**: fixtures,
the pre-cleanup frozen references and the noLP inputs (2.3 MB) are in `tests/fixtures/`, the
harnesses default to it rather than one laptop's $HOME, and the backtrack-pipeline self-check
and a runner (`tools/coverage/bars.sh`) moved in from the session scratchpad.
`verify_gpu_cli.sh` used to SKIP a missing fixture (pass having checked nothing); a missing
fixture or reference now fails, as in the noLP bar. Bars, run with the committed runner in
place: refcheck 9/9, parity, matrix 36/36, gpu_cli (24 IDENTICAL = 6 fixtures x 4 budgets,
0 missing), binding 20/20, backtrack pipeline, knobarms 18/18, noLP. **The zeroconf bar did
not complete** -- the run was stopped for low host memory during it -- and is owed at C9.

**C9 record (2026-10-13).** The D4 flag comment is in the source: above
`rnafold_md_tile_cb()` in device.cu (the md tile path and row fusion, kept pending final
tests), naming the tree before the cleanup (`pre-cleanup`, 6123cebe) and after it
(`post-cleanup`, this commit), with a pointer at `row_cells_i()` in hp_mb_loop.cu.
Comment-only. Build clean; refcheck C_mixed identical; **the zeroconf bar owed from C8
passes, 21/21**, rerun at JOBS=2 on a copy after the C8 run was stopped for low memory.
Tagged `post-cleanup`.

**Cleanup done.** C0-C9 on Lukes_Flow_Batching, local, not pushed. Still open from this
plan: the D7 pricing of the H5 layout lock, the final A100 run (400 x 5601 and the soak)
against `pre-cleanup`, and the D4 verdict.

**D7 record (2026-10-14): the H5 layout lock is priced, and it is offered upstream as Defect E.**
- **The defect is upstream's.** On pristine 2.7.2, `RNAfold -j8` with plots, 300 records of
  400-800 nt: **0/5 clean** (3 segfaults, 2 aborts). The controls are clean: `-j8 --noPS` 5/5
  and `-j1` with plots 5/5.
- **The fix holds.** Our tree, `-j8` with plots: CPU 5/5 and GPU 5/5 clean, and every plot is
  identical to pristine `-j1`'s.
- **The price.** The lock adds only the layout, because the write was already locked: 0.016 s
  over the 300 records, about 53 µs per record (layout plus write is 0.126 s). Wall times at
  `-j4`, ABBA x 3: CPU 26.89 s with `--noPS` vs 26.95 s with plots (+0.2 %, within the spread);
  GPU 2.05 s vs 2.19 s, a difference that is the write.
- **Verdict: keep it.** It goes to upstream as `PORT_UPSTREAM_PROPOSAL.md` Defect E, with
  `tests/upstream/plot_layout_race.sh` as the reproducer.
- **What this plan still has open:** the final A100 run against `pre-cleanup`, and the D4 verdict.

**2026-10-14: the final A100 check and D4 are in the N5a notebook** (`CUDA_RNAFold_N5a.ipynb`, from the local
generator `tools/make_nb_n5a.py`). Section C: `pre-cleanup` (6123cebe) vs `post-cleanup` (c4226271) vs head at
400 × 5601 and the soak, rotated x 3, one sha (Q5: post not slower than pre beyond spread). Section D: the default
vs `RNA_ROW_FUSE=1` (needs overlap 0) and vs `RNA_MD_TILE_CB=1024` (needs overlap, graphs, c ring and row batch off),
each with its matched control and its ACTIVE line asserted, at 400 × 5601 and 3000 × 1200 (Q6: a path pays only if
faster than the default beyond spread at 400 × 5601 and not slower at 3000 × 1200; otherwise it is canned).
Rehearsed on the laptop end to end: three trees, one sha per fixture; ACTIVE 6/6 for each path.

## 10. Build hygiene, from the build audit (2026-10-18)

Findings B1-B8 of `PORT_BUILD_AUDIT.md`, folded in as four more phases. B9-B11 need no
action (stock libtool, optional upstream dependencies, upstream's own gcc 15 warnings).
As before: one commit per phase, with bars after each.

| Phase | Findings | Content | Behaviour change | Bar beyond the standing ones |
|---|---|---|---|---|
| C10 | B1, B2 | `.cu` rules: nvcc writes dependency files (`-MD -MF`, included by `Makefile.am`), and `NVCC_ALL_FLAGS` gains `$(DEFS)` and the `-D` flags the C side gets | none (device TUs see the full `vrna_fold_compound_t`) | **positive evidence of the fix:** after `touch config.h` and after touching `fold_compound.h`, every `.cu` file recompiles (counted in `make V=1`); `sizeof(vrna_fold_compound_t)` printed from a host TU and a device TU are equal (negctl: today's flags, unequal) |
| C11 | B3, B12 | `RNAlib2.pc.in`: `Libs.private: -lm @CUDA_LIBS@`; `RNA/RNA.py` joins `all-local` in `interfaces/Python/Makefile.am` | none | the audit's static program that calls `vrna_cuda_devices()` links with `pkg-config --static` flags alone and prints `devices 1` (negctl: today's `.pc`, undefined `cuda*`); the CPU-only build's `.pc` carries no `-lcudart` |
| C12 | B5-B8 | Correct the stale ".cu files never see -DNDEBUG" comments (`hp_mb_cells.inc` x3, `fml_scan_block.inc`: `NVCC_ASSERT_FLAGS` has carried it since the asserts macro; Bill Langdon, 2026-10-19); delete dead variables and `E_int_loop`; drop `gpuAssert`'s stray `file`; `%zu` at `mfe_cuda.c:1343`; drop the unkept `const` from `par_mfe()`'s compound chain; the `-Wextra` leftovers in `gpu_batch.c`, `RNAfold.c`, `mfe_cuda.c`; look at the signedness in `mb_loop_fast.c` before changing it | none | the audit's A1/A2 greps report none of our files (the `fill_arrays_loop.c` lines are an artefact: it is `#include`d, not a translation unit); refcheck byte-identical |
| C13 | B4 | `ly_carry3_lb_kernel`'s `__launch_bounds__` minimum-blocks hint: compile for sm_80 locally and read ptxas | none if the hint is ignored on sm_80 too (then it is deleted; SASS identical is the bar) | if sm_80 honours it, it stays, with the architecture it was tuned on in the comment, and a per-arch value is an A100 A/B item, not cleanup |

Not folded: the zeroconf harness speed-ups (`--disable-lto --disable-static` on the throwaway
copies, no restore rebuild). They change the harness, not the product, and are proposed
separately (PORT_SHORT_SEQUENCES.md section 7).

### 10.1 As built (2026-10-19), pushed in 388852e0..755703e9

| Phase | Commit | What landed | Evidence |
|---|---|---|---|
| C10 | 21861f12 | `NVCC_ALL_FLAGS` gains `$(DEFS) $(DEFAULT_INCLUDES) $(RNA_CPPFLAGS)`; every `.cu` includes `config.h` and `#error`s without `HAVE_CONFIG_H`; nvcc writes `-MD` dependency files (`mfe/cuda/.deps/*.Plo`) that the Makefile includes; `vrna_cuda_register_batch_backend()` compares `sizeof(vrna_fold_compound_t / vrna_md_t / vrna_param_t)` between a host and a device TU and refuses the backend on a mismatch; RNAfold registers before it announces | touching `fold_compound.h` recompiles all six `.cu` files (before: none); a `.cu` built without `-DHAVE_CONFIG_H` stops at the `#error`; a forced size mismatch prints `BUILD ERROR: vrna_fold_compound_t is 384 bytes on the host and 392 in the CUDA code`, runs no sweep and answers as `RNA_GPU=0` (the negative control bites) |
| C11 | 2640ccb0 | `RNAlib2.pc.in`: `Libs.private: -lm @CUDA_LIBS@`; `interfaces/Python`'s `all-local` depends on `RNA/RNA.py` | the audit's static program calling `vrna_cuda_devices()` links with `pkg-config --static` flags alone; a new binding function reaches `import RNA` after a plain `make` |
| C12 | 9198c1d9 | B5-B8: `E_int_loop()` and unused statics/locals deleted; `gpuAssert`'s stray format argument; `%zu` for a `size_t`; the unkept `const` removed from all 42 `par_mfe()` chain signatures at once (C++ will not convert `T**` to `const T**` piecemeal); the stale ".cu files never see -DNDEBUG" comments corrected | no behaviour change; 14 files, +66 / -343; standing bars green |
| C13 | 755703e9 | `ly_carry3_lb_kernel`'s `__launch_bounds__(128, 16)` documented as sm_80's (2048 threads/SM); sm_86/89 allow 1536, so ptxas ignores the 16 there | comment only; sm_80 ptxas honours it, so it stays, and a per-arch value is an A100 A/B item, not cleanup |
| harness | 628f2351 | `tests/zeroconf_configure.sh`: `--disable-lto --disable-static` in the cases by default (`ZC_FAST=0` restores), `ZC_NO_RESTORE=1` for throwaway copies | zeroconf 908 s -> 279-297 s, rc 0; harness only |

**Bars:** the full session (refcheck, parity, matrix, gpu_cli, binding, backtrack pipeline,
output files, knob arms, noLP, device_bt_check, short_input_check, zeroconf; standing bars
again under `RNA_DEVICE_BT=1`) was green on 755703e9 before the push (2026-10-19), and is
re-run on 5c93fa04 (2026-10-20).

**Follow-on:** C12's corrected comment showed that the row-index `assert` in
`hp_mb_cells.inc` (x3) and `fml_scan_block.inc` (x1) is compiled out of every release build.
5c93fa04 makes it a `__trap()`, which survives `NDEBUG` (median wall within +3.2 %; an inverted
trap kills the run with rc 207 and no output).

**On Finished_Port:** C10 (with B6) and C11 apply there too and were ported 2026-10-20; C12's
other items are LFB code that FP does not carry, and C13's kernel is LFB-only.
