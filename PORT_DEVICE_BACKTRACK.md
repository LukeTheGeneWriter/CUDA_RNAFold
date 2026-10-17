# N5: backtracking on the device — scope (signed off)

*Written 2026-10-14, after N5a P3 and before the A100 run of N5a P4. Luke (2026-10-10): after N5a, the
device-side backtracking experiment. Roadmap item N5 (originally C3). Status: **SIGNED OFF 2026-10-14** (Luke: "yes to
all four"): P0 first with the go/no-go of §4; byte-identical including ties; default model first, every other
option refuses to the host; opt-in `RNA_DEVICE_BT=1` until an A100 run decides.*

## 1. What N5 would remove, and why N5a does not settle it

Today each record is backtracked on the host:
1. **Fetch.** `backtrack_fetch_slot()` copies the record's c and fML triangles from the device (fM2 too
   when circular) into a per-worker scratch of about 5 L² bytes.
2. **Trace.** `backtrack_finish_slot()` builds fM1 and f5 from c on the host, then runs upstream's
   `vrna_backtrack_from_intervals()` (`backtrack/bt_*.c`) over those matrices.

At 400 × 5601 on the A100 (N2, phase-synced) that is **8.3 s of trace + 3.3 s of fetch = 26.6 % of the
wall**. N5a exists to hide it: chunk *k* is traced while chunk *k+1* sweeps. The A100 run (P4, Q1) will say
how much it hides. **What N5a cannot change:**

| cost | today | with N5a | with N5 |
|---|---|---|---|
| PCIe traffic | every triangle cell crosses: ≈ 100-125 MB per 5601-nt record, **≈ 45 GB per 400 × 5601 run** | the same, moved off the critical path | the structures only (a few KB per record) |
| host RAM | 5 L² per backtrack worker (≈ 157 MB at 5601; × 12 workers ≈ 1.9 GB); B3's RAM gate sheds workers to fit | the same, **plus** a second retained chunk on the device | O(L) per record |
| host cores | 12 trace workers compete with the builder and the CPU slice (H measures it) | the same competition, spread out | none: cores go to the builder and the CPU slice |
| few-core hosts (Colab T4: 2 vCPU) | the trace is serial-ish and exposed | N5a needs idle cores to hide anything | the device traces |

So N5 is not a speed lever where the host is idle and wide (an A100 with 12 cores and N5a). It **is** a
lever for host RAM (the budgets), for few-core hosts, and for heterogeneous runs, where the CPU slice wants
the cores. These are measurable now, so the plan measures them first (P0) and builds only if they show.

## 2. What exists to build on

- **The retained triangles (N5a P2).** At sweep end, the chunk's c / fML (int32 or int16 + baselines) / fM2
  move into a `rnafold_retired_t` instead of being freed. A device traceback reads exactly these, on its own
  stream, **beside the next chunk's sweep**: the P2 machinery, with a kernel in place of the host fetch.
- **The admission and VRAM accounting (B2, P3 rule (b)).** A retired set is already charged at 0.85 ×
  `rnafold_retired_bytes()`. A device trace changes nothing there: the set lives exactly as long.
- **The fetch/finish seam.** `backtrack_fetch_slot()` is the only part that touches the device; a device
  trace replaces both halves for a record and hands `backtrack_finish` a bp list and an energy.
- **The bars.** Every arm can be compared with the host trace on the same tree (self-comparison), and the
  frozen references (`tools/coverage/refcheck.sh`) catch any drift. The output-file bar covers what is written.

## 3. Design

**3.1 Exactness includes ties.** Upstream's trace takes the **first** decomposition, in its loop order, whose
energy equals the cell's. A random fixture rarely ties; a uniform one ties everywhere (the noLP slot-flow
defect was found only by non-uniform fixtures, and the converse holds here). The device must reproduce the
order, not just an optimal structure: each search evaluates its candidates in parallel, ballots the matches,
and takes the one **earliest in upstream's order**. Bar: byte-identical structures on uniform and G-rich
fixtures as well as random ones, and a negative control that reverses the order and must change the output
on the uniform fixture.

**3.2 One warp per record, the stack in shared memory.** The trace is a stack of intervals; each pop does
one bounded search:
- hairpin: none; stacked pair: one cell;
- internal loop: the (p, q) window under MAXLOOP (≤ 30 × 30 / 2 cells), lanes over p;
- multibranch split and the ML decomposition: O(j − i) over k, lanes over k;
- exterior (f5): O(j) over k.

400 records is 400 warps: a light grid that sits beside the next sweep at low stream priority, like the
build pipeline's work sits beside the GPU today.

**3.3 f5 and fM1 on the device.** The host builds f5 (`vrna_mfe_exterior_f5`) and fM1
(`vrna_mfe_multibranch_m1`) from c today. Both move into the trace kernel's prologue or a small kernel before
it: one row per j, lanes over k.

**3.4 Model coverage by refusal, not by guessing.** S2 covers the default model: dangles 2, linear, no
`--noLP`, no G-quads, no hard or soft constraints, default salt. Every other option makes the chunk **refuse**
and trace on the host as today, with a one-time banner (positive evidence, both ways). Options are added one
at a time after S2, each with its parity bar, in the order they are used: dangles 0, temperature, salt,
`--noLP`, `--noClosingGU`, `--maxBPspan`, circular, G-quads, constraints.

**3.5 Opt-in.** `RNA_DEVICE_BT=1`, off by default, until an A100 run decides (as `RNA_BT_PIPELINE` does).

## 4. Phases

| phase | what | bar / evidence | where |
|---|---|---|---|
| **P0** | **Measure the problem first.** (a) Host RAM and core time of the trace per chunk (RSS with and without backtrack workers; core-seconds); (b) a few-core host simulated with `taskset -c 0,1` on the laptop, N5a on vs off; (c) from the A100 run: what N5a left exposed (T), and what the trace pool cost the CPU slice (H). | pre-registered go/no-go below | laptop + the P4 results |
| P1 | Standalone prototype `tools/proto_device_bt.cu`: dump a chunk's triangles from the host trace, trace them on the device (default model), compare bp lists. Time per record vs length. | identical on random, uniform and G-rich fixtures; the reversed-order control bites on uniform | laptop |
| P2 | In tree behind `RNA_DEVICE_BT=1`: kernel on the retired set's stream, f5/fM1 on the device, structures and energies back; refusal banner for every other option. | refcheck 9/9, bt_pipeline_check, output-file bar, binding, zeroconf; NEGCTL bites; ACTIVE line asserted | laptop |
| P3 | Options one at a time (3.4), each with its parity bar and arm in the option matrix. | per option | laptop |
| P4 | A100 + a few-core host (a 2-vCPU Colab T4, or `taskset` on the A100 VM): off / N5a / N5 / N5 + N5a, at 400 × 5601, 3000 × 1200 and the soak; RSS per arm. | the rule decided at sign-off | Colab |

**P0's go/no-go (proposed):** go to P1 if **any** of: on the 2-core simulation, the trace (with N5a on) is
≥ 10 % of the wall; the trace workers hold ≥ 25 % of the run's peak RSS at the default budget; or the A100's
H section shows the trace pool costing the CPU slice ≥ 5 %. Otherwise N5 is recorded as not needed, with
the numbers, and stops.

### P0 results (laptop, 2026-10-14): GO, on criterion (ii)

Fixtures are random ACGU, 40 × 5601 and 2000 × 1200. Every run's output equals the default run's.

| criterion | measured | verdict |
|---|---|---|
| (ii) trace workers' share of peak RSS | 159 MB per worker at 5601 nt (= 5 L², as §1 predicted); 12 default workers hold 1.9 GB of a 3.8 GB peak = **50 %**. At 1200 nt, about 15 %. | **GO** (≥ 25 %) |
| (i) exposed trace on 2 cores, N5a on | `taskset -c 0,1`, both pools pinned to 2, 3 reps ABBA, after the trim below. 2000 × 1200: exposed (waits at joins + at the end) 5.8–6.6 s, mean 6.17 s of a 61.6 s wall = **10.0 %**. 40 × 5601: 2.37 s of 29.2 s = 8.1 %. | on the line at 1200 nt; below at 5601 |
| (iii) the A100 H section | awaits the N5a notebook run | pending |

Two more findings from (i):
- **On 2 cores, N5a costs.** At 2000 × 1200, N5a on took 60.2–62.5 s against 56.5 s off (+8.5 %), with RSS of 4.2 vs 2.1 GB. Off's backtrack + fetch is 4.3 s (7.6 %). The trace workers and the sweep compete for two cores, and N5 removes exactly that contention. One off run took 42.2 s; it is unexplained, and it is excluded from the mean.
- **N5a's AUTO stage must look at the core count**, not only at memory: at 2 cores it should stay off.

**Found and fixed on the way (45c63f7c): glibc kept every chunk's tables.**
- Cause: compounds are built on the builder thread and freed a chunk later, and glibc did not return that memory to the OS.
- Before the fix: 2000 × 1200 with both pipelines on peaked at 6.3 GB and swapped on the 6.7 GB host. The 2-core numbers above were first measured in that state (+11.7 %, swap-contaminated) and were re-measured after the fix.
- The fix: `malloc_trim(0)` after each chunk is freed. The peak is now 4.4 GB with no swap; 40 × 5601 is unchanged at 3.8 GB, and every bar is green.

**Gap left for the AUTO stage:** the RAM model counts two chunks of compounds, but under N5a three are alive at once (building, sweeping, tracing). So `RNA_RAM_BUDGET=half` with N5a on still peaks at 4.34 GB against a 3.39 GB budget. The budget gate must count three chunks before N5a can be default-on.

### P1 results (laptop RTX 3050, 2026-10-15): EXACT, and the trace is cheap; f5 is the cost

`tools/proto_device_bt.cu` with `tools/proto_device_bt.sh BUILD_DIR [timing]`:
- **What it does.** Upstream folds each record on the host (RNA_GPU=0). The device gets that c and fML in the sweep's layout, builds f5 itself, and traces with one warp per record.
- **Shared energies.** Energies come from the sweep's own device functions (`Energy()`/`IntLoop_X`, `E_Hairpin_device`, `E_MLstem_device`), with parameter structs built as `load_param()`/`load_param2()` build them. Only the exterior stem is new.
- **Tie order.** Each search ballots 32 candidates at a time and takes the earliest match in upstream's order.

**Exactness.** On every fixture, every f5 cell, every MFE and every structure equals upstream's, with 0 failures:
- random, A/U-only and G-rich, at 400 nt (40 records each) and 1200 nt (12 each);
- the six timing fixtures, 200 to 5601 nt.

Ties are exercised, not assumed. A count pass that scans every candidate, and makes the same picks, found searches with more than one match in nearly every record: 210–496 per fixture, in 39/40 to all records.

**Negative control: it bites.** With every search's order reversed (last match, categories in reverse), 38/40 A/U records change structure, all still at the MFE. So the ties are real, and the order is what decides them.

**Timing.** "Device" is one batch, every record in its own warp at once, so a batch costs about one record's latency. "Host" is upstream's `vrna_mfe_exterior_f5` + `vrna_backtrack5` per record, serial, with no fetch.

| L | records | device batch | of which f5 | host per record |
|---|---|---|---|---|
| 200 | 64 | 0.99 ms | 0.69 | 0.14 ms |
| 400 | 64 | 2.79 ms | 2.23 | 0.41 ms |
| 800 | 32 | 9.0 ms | 7.8 | 1.09 ms |
| 1200 | 16 | 18.6 ms | 16.9 | 2.30 ms |
| 2400 | 8 | 67.9 ms | 64.5 | 9.42 ms |
| 5601 | 4 | 267 ms | 261 | 47.6 ms |

- **The trace itself is cheap: about 6 ms at 5601 nt.** f5 is 90–98 % of the device time. One warp walking j sequentially with a 32-lane min costs O(n²/32) per record, against the host's O(n²).
- **What this means for a chunk.** A 400 × 5601 chunk is 400 concurrent warps, so it should cost about one record's latency plus contention: on the order of 0.3 s on this GPU, unmeasured at that width. Today the host spends 400 × 47.6 ms / 12 workers ≈ 1.6 s of trace, plus about 100 MB of fetch per record.

**Carried into P2:**
1. **f5 wider than a warp.** A block per record, or f5 from the sweep's own last rows. The 5601 latency should drop by the block width.
2. **int16 fML.** The sweep's default stores fML as int16 with per-block baselines (N3). P1 read int32, so P2's kernel must decode in place, as `fetch_fML_one_Hw` widens today.
3. **Off-diagonal reads.** Upstream's nibble loops read cells such as fML(i, i−1), which by the index arithmetic is fML(1, i). The device kernel uses the same arithmetic, so it reads the same cell. In P2 the host trace and the device trace read the same device triangle, so the self-comparison covers this by construction.

### P2 plan (SIGNED OFF 2026-10-15: Luke, "yes to all six"; experimental arms may be guarded)

**Goal.** The P1 kernel in the tree behind `RNA_DEVICE_BT=1`, default model only. Two paths:
- **Pipeline on:** it traces the retired set beside the next sweep, on its own low-priority stream.
- **Pipeline off:** the same set, traced in place.

Every other option refuses to the host trace, with a banner. With the variable unset, nothing changes: no code on the default path runs.

**What the device trace has to replace.** `backtrack_finish_slot()` / `callback_backtrack_bts()` do five things per record. The device path must do each, or refuse:

| host today | device path |
|---|---|
| f5 from c (`vrna_mfe_exterior_f5`) into `vc->matrices->f5` | f5 on the device, copied back into `vc->matrices->f5` (n+1 ints) so library state is identical (D3) |
| `energy[idx] = f5[n]`, `EN[idx] = energy/100` | the same, from the copied f5 |
| structure via `vrna_backtrack_from_intervals_bps` + `vrna_db_from_bps` | the kernel writes the dot-bracket string (n+1 bytes); `.`, `(`, `)` only, because G-quads refuse |
| `stat_cb(VRNA_STATUS_MFE_POST)` | called on the host, per record, where the host path calls it |
| uniq_ML fM1, circular, `RNA_CIRC_VERIFY`, keep_matrices, backtrack_type C/M, no-backtrack | refuse (below) |

**The seam.** In `par_mfe()`, with `RNA_DEVICE_BT=1`, the chunk always goes through `rnafold_retire_chunk()`, whether or not the pipeline is on (D5). The two paths then differ only in who calls the trace:
- **Pipeline on:** the trace thread (`bt_async_main`) launches the device trace in place of `backtrack_all()`.
- **Pipeline off:** `par_mfe()` calls the same function, then releases the set.

There are no host workers and no scratch pool: `bt_pool_get()` is never called, so B3 has nothing to shed. The join, callback, ordering and release rules of N5a are unchanged.

**What the trace owns (D2).** The sweep's `d_S`, salt table and `hp_len` are freed at every chunk's teardown (`int_loop.cu:594`, `hp_mb_loop.cu:849-850`). The next chunk's `init_gpu3()` also rewrites `d_param2->hp_len`. A trace running beside the next sweep cannot read any of them. So each retired set gets its own small device copies, built at retire from `VC[]`:
- sequence encodings (S1 shorts, the sequence chars, the 3-bit packing `unpack()` reads);
- the parameter structs, `hp_len` and the salt table;
- the scratch: f5, the interval stack and the structure out.

Together that is about 22 n bytes a record, about 50 MB at 400 × 5601. `rnafold_retired_bytes()` charges it, so B2 and P3 rule (b) see it. `rnafold_retired_release()` frees it by the pointer.

**One source of truth for the parameters.** `load_param()` / `load_param2()` fill their host structs inline today. P2a moves the fill into `rnafold_param_fill()` / `rnafold_param2_fill()` (int_loop.cu, hp_mb_loop.cu), and the sweep and the trace both call them. The prototype's copy of that code then goes; the prototype keeps its own only as the standalone reference.

**The kernel** (new `bt_device.cu`, with its Makefile prerequisites on `stub2.h`, `int_loop_dev.h`, `hp_mb_dev.h` and `interior_loopx.h`):
- **f5:** one block per record (D1). A block-wide min per j, with blockDim a knob `RNA_DEVICE_BT_BLOCK` (32/128/256, default decided by P2d's measurement).
- **The trace:** warp 0 of the block, as in P1: same picks, same order.
- **fML read in place,** int32 or int16. For int16, a cell is `base[boff[H] + colb(j) + (i−1)/FML_BLK] + v16`, with `FML_INF16` meaning INF. `colb(j) = FML_BLK·q(q+1)/2 + r(q+1)`, where q = (j−1)/FML_BLK and r = (j−1) mod FML_BLK, is the closed form of the host decode's running sum. A read with i > j (the nibble's fML(i, i−1)) is mapped back through the flat index to its own (i', j') first, so it gets that cell's baseline, as the host decode gives it.
- **Off-diagonal reads.** For int32, the same index arithmetic reads the same cell as the host trace.

**Refusal (the default model, strictly; D4).** A chunk traces on the device only if every record has all of the following. Otherwise the whole chunk traces on the host, with one banner `RNA_DEVICE_BT REFUSED: <reason>` per reason per run:
- dangles 2, linear, no `--noLP`, no G-quads, no uniq_ML, default salt, default temperature, no `--noClosingGU`, default span;
- no hard constraints (the batch flag that makes `d_up_int` non-NULL), no soft constraints, no unstructured domains, no auxiliary grammar;
- `backtrack` on with type 'F', and not `vrna_cuda_keeping_matrices()`;
- the default parameter tables: compared once per batch against `vrna_params()` of the default model, which catches `-P` files.

Positive evidence: `RNA_DEVICE_BT ACTIVE: <k> chunk(s), <m> record(s) traced on the device`, in the end-of-run report.

**Self-comparison built in (D6).** `RNA_DEVICE_BT=verify` runs both traces on every record and exits non-zero on the first difference in structure, energy or any f5 cell. That is the bar's main arm. It is the cell-by-cell form of the int16 decode check, on real sweeps.

**Negative control.** `RNA_DEVICE_BT_NEGCTL=1` reverses every search's order, as P1's control does. On the A/U fixture the output must change, with energies unchanged.

**Steps.** Each is committed only with its bars green.

| step | what | bar | status (2026-10-15) |
|---|---|---|---|
| P2a | parameter fill shared (answer-neutral refactor) | refcheck 9/9, option parity + matrix, gpu_cli | **done** ea6267ab: refcheck 9/9, parity, matrix 36/36, gpu_cli, binding 38, bt_pipeline, output files, knob arms, noLP |
| P2b | `bt_device.cu`: kernel, int16 decode, per-set copies, release; not yet called | builds; zeroconf (the CPU-only build must not see it) | built clean, no warnings; `libRNA.a` carries the 7 `rnafold_dbt_*` symbols, RNAfold none; refcheck C_mixed unchanged; zeroconf 21/21; **done** 9cba88a1 |
| P2c | the seam in `par_mfe()` (both paths), refusal + banners, VC f5 write-back, stat_cb, budget accounting | below | **done** ec9ec0b5: smoke + multi-chunk green (below); standing bars with RNA_DEVICE_BT unset: refcheck 9/9, parity, matrix, gpu_cli, binding 38, bt_pipeline, output files, knob arms 18/18, noLP; zeroconf 21/21 |
| P2d | bars and laptop measurement | below | **bars done** (2026-10-16, below): device_bt_check 0 failures; standing bars unset and with RNA_DEVICE_BT=1 (refcheck asserting ACTIVE, parity, matrix 36/36, gpu_cli, binding 38/38 both ways, bt_pipeline, output files, knob arms 18/18, noLP); zeroconf 21/21 (a72c9414). Laptop measurement **done** (below): N5 never lost a whole run, block default stays 128 |

P2b as built: one block per record (`RNA_DEVICE_BT_BLOCK` 32/64/128/256, default 128 until P2d measures), f5 block-wide then the trace on warp 0; int16 fML read on its baselines with the closed-form `colb(j)`, an i > j read mapped back to its own cell first; each set's copies in ONE device allocation (freed by the pointer, `rnafold_dbt_bytes()` for the budget); the trace's own stream at the device's lowest priority. Its first exactness evidence is P2c's `RNA_DEVICE_BT=verify` on real sweeps.

P2c as built (2026-10-16): `par_mfe()` asks `dbt_refusal()`, and an accepted chunk is retired whether the pipeline is on or not. Its device copies hang off the retired set (`rnafold_retired_t.dbt`), so `rnafold_retired_release()` frees them and `rnafold_retired_bytes()` charges them. `trace_retired()` is the one entry, called by the trace thread and by `par_mfe()`. A set with no copies traces on the host as before. A device failure (non-zero status) fails the chunk over to the host trace with a `FAILED` banner; under verify it exits. `-P` is detected by `vrna_param_t.param_file`, not by comparing with `vrna_params()`: that returns whatever file is loaded, so it could not see one.

Smoke, laptop: C_mixed, extreme and an A/U fixture, single chunk, int32/int16 × pipeline off/on, verify and `=1` against RNA_GPU=0. All identical, ACTIVE asserted, NEGCTL bites (20/24 A/U structures change, energies equal; verify + NEGCTL exits 1), `-d0` REFUSED and identical, unset prints nothing new. Then multi-chunk (A_longseq_smallbatch, C_mixed, A/U, 4 to 7 chunks): 24/24 identical, 0 failed over.

**A race the multi-chunk arms found.** The first multi-chunk run failed every record of every chunk after the first, with the pipeline OFF only (status 5: no decomposition for a c cell). On one A/U chunk it gave a WRONG structure that traced clean. The cause was the prepare's upload: a plain `cudaMemcpy` from pageable memory can return before the DMA lands, and the trace kernel's stream is non-blocking, so it does not wait for the legacy stream. Launched at once (pipeline off), the kernel read a part-uploaded blob. Launched later from the trace thread (pipeline on), it was hidden. The upload now goes on the trace's own stream and is synchronised. Single-chunk arms could not see this, and neither could the pipeline-on arms, so P2d's bar must keep multi-chunk × pipeline off.

**P2d bars.**
1. **The new `tests/device_bt_check.sh`.** Each arm runs RNA_DEVICE_BT=1, and the same with `verify`, against the CPU reference (RNA_GPU=0), byte-identical, with ACTIVE asserted:
   - fixtures: random, A/U-only and G-rich;
   - fML: int32 and int16;
   - pipeline: off and on;
   - chunks: multi-chunk (small RNA_GPU_CHUNK_CELLS) and single;
   - lengths: mixed lengths in one chunk.
2. **Refusal arms,** each REFUSED and identical to the host: `-d0`, `--noLP`, `-c`, `-g`, `-C`, `-T 25`, `-P` file, `--salt`, `--noClosingGU`, `--maxBPspan`, the binding with keep_matrices.
3. **The NEGCTL bites.**
4. **The standing bars with RNA_DEVICE_BT unset** (default path untouched): refcheck, parity, matrix, gpu_cli, binding, bt_pipeline_check, output files, knob arms, noLP, zeroconf. **And again with RNA_DEVICE_BT=1:** refcheck, bt_pipeline_check, output files, binding.

**P2d bars as run (2026-10-16).** `tests/device_bt_check.sh` (`TREE=<build>`): 64 arms (random / A/U / G-rich / long × int16 × pipeline × single / multi-chunk × `1` / verify), each identical to RNA_GPU=0 with every record traced, 0 refused, 0 failed over, verify = n, multi-chunk ≥ 2 chunks; the heterogeneous arms (RNA_CPU_SLICE=1 -j4) with device + cores = n, cross-checked against RNA_RECORD_TRACE's gpu / cpu rows; the 10 refusals plus the binding (fc.mfe() refused for keep-matrices, RNA.cuda_fold() traced 3/3); NEGCTL 29/40 structures changed, energies equal, verify + NEGCTL exits 1; IGNORED; unset prints nothing and is identical. The standing bars take an arm's variables through `BARS_ENV` (re-exported after their RNA_ scrub) and refcheck asserts the `BARS_WANT` banner. With RNA_DEVICE_BT=1 the only FAILED lines were the outfile and bt_pipeline NEGCTL arms, which zero a triangle on purpose; their bars still bit.

What the bars found:
- **The cpu slice report counted early.** It printed the records already done while the pool still folded others (`het_long`: 56 + 3 of 60; the record trace said 4). It now reports done + outstanding under the slice lock, and says how many were still folding.
- **The banners ignored the binding's quiet switch.** The `=1` banner, REFUSED and the ACTIVE summary now follow `vrna_cuda_quiet()` like every other backend line (RNA_GPU_VERBOSE=1 brings them back); FAILED and IGNORED stay loud. The binding's thread-budget test drops RNA_DEVICE_BT: it measures the host trace's pool, which the device trace does not use.
- **Race audit: the chunk setup's ordering hung on one pinned copy.** The row kernels run on non-blocking streams; the chunk's setup uploads ran on the legacy stream, ordered before row 1 only while the row-table upload was pinned (a pinned cudaMemcpy blocks the host and syncs earlier legacy work); its buffers fall back to malloc when cudaHostAlloc fails, and a pageable copy orders nothing. `rnafold_rowtab_upload_all()` now ends with a sync of the legacy stream, not the device: a device-wide sync would also wait for the pipeline's device trace.

**P2d measurement (laptop).** Arms: off / N5a / N5 / N5 + N5a, at 400 × 1200, 40 × 5601 and 2000 × 1200, recording wall, RSS and VRAM:
- 12 cores, and the 2-core simulation (taskset 0,1, pools pinned);
- `RNA_DEVICE_BT_BLOCK` 32 / 128 / 256 at 40 × 5601.

The A100 verdict is P4's.

**P2d measurement as run (laptop, 2026-10-16, 4 GB power-capped GPU, 12 threads, 6.8 GB WSL).** Whole-run wall, median of 3 in ABBA order; every arm of a group had one sha. The GPU fell into a low-clock state at 19:28 (one 2000 × 1200 run took 175 s, 3.2× the GPU time of its twin), and M2, M3 and M4 ran in it. Their arms are comparable within the group (spreads 0–2 %) but not with M1's.

| group | off | N5a | N5 | both | backtrack stage off → N5 | peak RSS off / N5a / N5 / both |
|---|---|---|---|---|---|---|
| 400 × 1200 | 10.98 | 11.00 | 10.46 | 10.43 | 0.29 → 0.04 s | 1211 / 1211 / 1061 / 1061 MB |
| 40 × 5601 | 27.64 | 27.46 | **25.94** | **25.89** | 1.40 → 0.35 s | 3819 / 3820 / **1922** / 1923 MB |
| 2000 × 1200 | 54.20 | 58.11 | **52.36** | 56.41 | 1.34 → 0.26 s | 2333 / **4373** / 2119 / 4157 MB |
| het (slice + -j12), 400 × 1200 * | 36.25 | 36.25 | 35.86 | 35.86 | 0.33 → 0.17 s | 1212 / 1211 / 1095 / 1095 MB |
| 2-core (taskset 0,1, pools 2, -j2 + slice) * | 38.02 | 38.11 | 37.29 | 37.33 | 0.63 → 0.17 s | 1062 for all |

\* in the low-clock state. 400 × 1200's spreads (14–32 %) hide every difference there.

- **The device trace never lost a whole run.** Against the faster of off and N5a it was −5.5 % at 40 × 5601 (beyond the spread), −3.4 % at 2000 × 1200 (beyond the spread), −1.1 % under heterogeneous load and −1.9 % on 2 cores. Under the CPU slice the cores folded their own records with the host trace: 394 device + 6 cores, and 399 + 1.
- **RSS: N5 halves it at 5601,** 3.8 → 1.9 GB. Presumably the host-side trace inputs are no longer held, but that is not yet attributed.
- **N5a costs at 2000 × 1200 on this box.** Retiring a chunk for the trace thread doubles RSS to 4.4 GB of 6.8, and the wall was +7.2 % vs off. "both" inherits it: +7.7 % vs N5. The device trace does not need the pipeline's retained host set, so the cost is the pipeline's, but the A100's 85 GB will not show it. NT/NH judge it there.
- **Block width (M4, 40 × 5601, low clock):** 32 → 84.24 s (trace 3.77 s), 128 → 81.52 (1.09), 256 → 81.04 (0.67). 32 loses; 128 vs 256 is −0.6 %, so the default stays 128 until the A100's NB.

**Where the costs come from (laptop, 2026-10-17).** From the stage, phase and `/usr/bin/time` lines of the runs above, plus a 2000 × 1200 ABBA × 3 of n5 / both, with the build pipeline at AUTO and off:
- **What N5 saves is the host trace's data path, not trace speed.** At 40 × 5601, off → n5: `fetch_mx` (the D2H of c / fML for the host trace) 1.0–2.2 s → 0; the backtrack stage 0.73–1.40 → 0.35 s; host CPU 61–77 → 44–46 s (user + sys). That is the whole wall difference (1.7 s). The RSS halving (3.8 → 1.9 GB) is the host trace's fetch scratch, which is never allocated.
- **What "both" loses is three chunks' compounds resident at once.** With both pipelines on, chunk k's `vrna_fold_compound_prepare()` runs while chunk k−1 is retired (its trace is joined after chunk k's sweep, and only then output and freed) and chunk k+1 is being built beside it. prepare goes 0.3–0.4 → 3.2–3.6 s, minor faults 1.36 → 2.29 M, peak RSS 2.1 → 4.2 GB; steady-state wall n5 52.15 / 52.55 vs both 56.02 / 55.56 (+3.4 s, ≈ the prepare). Either pipeline alone does not do it: with the build pipeline off, both = n5 (53.47 vs 53.42 s median, prepare 0.25–0.29). One malloc arena (MALLOC_ARENA_MAX=1) made it worse (prepare 4.9 s), so it is live memory and fresh-page faults, not arena fragmentation. N5a pays the same prepare (3.4 s): it is the trace pipeline's cost wherever the build pipeline runs, and NT / NH now print prepare, fetch_mx and RSS per arm so the A100 can show its size there.
- **For a device-traced chunk the hand-off buys nearly nothing.** Its trace is 0.05–0.15 s of GPU time per chunk, run on the same GPU as the next sweep, so the overlap it is parked for is at most the trace's own time (0.26–0.30 s a run here), against 3 s of prepare.
- **The trace kernel is f5, and f5 is latency.** From the block-width arms, T(b) ≈ a / b + w: the f5 phase is ~80 % of the kernel at 128 threads and the walk (warp 0) ~20 %. Each j costs two block barriers, a serial combine on thread 0 and a `volatile` global read of f5[i−1] per candidate (the c column itself is coalesced: the triangle is column-major). f5 in shared memory (22 KB at 5601 nt) is the obvious lever, but the kernel is ~1.3 % of the wall at 5601 here, so it waits for NT's backtrack stage on the A100.
- **The laptop has two clock states.** The first run after an idle spell sweeps ~1.5× faster (12.2 s of sweep GPU time vs ~18 s after); a 2000 × 1200 n5 took 35.0 s fresh and 52.2 s steady. A rep 0 that straddles the change looks like an arm effect. Discard a warm-up, or compare steady-state reps only.

So the laptop decides nothing (Q7 is the A100's), but nothing here argues against N5 as a default candidate, and the pipeline's memory cost on small hosts is now a known factor for the "both" arm.

**The default rule (Q7, Luke 2026-10-16, signed off 2026-10-17): the whole run, not the trace.** A folding run is many systems at once: sweep, builder, trace pipeline, CPU slice and output. The device trace becomes the default only if the **wall of the entire run** (launch to exit) is faster beyond spread than the faster of off and N5a: at 400 × 5601 and 3000 × 1200, **and** under the full heterogeneous load (CPU slice + `-j NPROC`), and slower beyond spread nowhere, the soaks included. The trace's own time is reported, never judged. A faster trace that slows the sweep, the builder or the cores beside it loses. The A100 notebook's section N carries this (NT, NH).

**Each side traces its own records.** Records the cores fold (the CPU slice, `-j`, a record the engine declines) go through upstream's `vrna_mfe()`: host fill, host trace. The CLI never attaches the engine (`vrna_cuda_attach()` is library API only), and `RNA_DEVICE_BT` lives in `par_mfe()`, the device batch, alone. So the device trace can only ever take records the device folded, whatever its default. `tests/device_bt_check.sh` asserts it by count under the CPU slice: device-traced + core-folded = every record. The notebook's NH asserts the same at A100 scale.

**Risks specific to P2.**
- **Sharing SMs with the next sweep.** At 5601 nt a trace block is resident for tens of ms. A low-priority stream orders new launches but does not preempt resident blocks, so the sweep can slow. That is what P2d's N5 + N5a arm and P4 measure, and why the block width is a knob.
- **Partial chunks.** A chunk that mixes refusing and accepted records refuses whole. Simple, and correct.

**Decisions for Luke:**
- **D1.** f5 on a block per record, with the width chosen by measurement (recommended), or keep P1's warp.
- **D2.** The trace owns small per-set copies of sequences and parameters (recommended; decoupled from the sweep's lifetimes), or retain the sweep's `d_S`/tables with the set (couples teardown to the trace).
- **D3.** Copy device f5 back into each compound, so library state is identical to the host path (recommended; n+1 ints).
- **D4.** Strict refusal, including `-T` and `-P`, even though the energy tables would carry them; they come back in P3 with their bars (recommended, as signed for §3.4).
- **D5.** The pipeline-off path also goes through a retired set, one code path for both (recommended).
- **D6.** `RNA_DEVICE_BT=verify` as the built-in self-comparison (recommended; as `RNA_HC_VERIFY`).

## 5. Relations

| pair | relation |
|---|---|
| N5 and N5a | **Reuse, not either/or.** N5 needs N5a's retained set and stream. With N5 on, the host pool has nothing to trace. |
| N5 and the budgets | SYNERGY: the 5 L² worker scratch goes, so B3 stops shedding workers. VRAM: unchanged (the set is already retained). |
| N5 and the CPU slice / router | SYNERGY: the host's cores go to folding. |
| N5 and int16 | SYNERGY: the device reads int16 fML with its baselines in place; there is no widening fetch. |
| N5 and candidate lists (sparse md) | SYNERGY, later: the multibranch split search can walk the candidate list instead of every k (SparseMFEFold). Not in S2: plain order first, for exactness. |
| N5 and N4 (md latency cut) | Independent. |

## 6. Decisions for Luke

1. **Measure first (P0) with the go/no-go above, or build P1 regardless as the experiment?** Recommended:
   P0 first. It costs a day of laptop time, and it decides whether this is worth weeks.
2. **Exactness bar: byte-identical to the host trace, ties included** (3.1). Recommended: yes; it is the
   project's bar everywhere else, and an "optimal but different" structure would break every reference.
3. **Coverage: default model first, every other option refuses to the host** (3.4). Recommended: yes. The
   alternative, all options before any measurement, ports about 4.9k lines of `backtrack/` blind.
4. **Opt-in `RNA_DEVICE_BT=1` until an A100 run decides** (3.5). Recommended: yes, as with `RNA_BT_PIPELINE`.

## 7. Risks

- **Tie order.** Upstream's search order is spread over `bt_internal.c`, `bt_multibranch.c` and
  `bt_exterior_f5.c`, including soft-constraint and auxiliary-grammar hooks. Missing one branch gives a
  structure that is optimal and wrong. P1's uniform fixture and reversed-order control exist for this.
- **Warp occupancy.** One warp per record leaves most of the device idle. That is acceptable beside a sweep
  and costly if the trace runs alone (the last chunk). P1 times it.
- **Divergence from upstream.** A second implementation of the trace must follow upstream's changes. The
  refusal path keeps the host trace as the reference; the parity bars run both.
