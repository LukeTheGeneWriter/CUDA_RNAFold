# Review guide

One entry point for reviewing the CUDA RNAfold port: what is where, what runs on the hot
path, which settings exist, and how the correctness bars are run. Written at the end of
the 2026-10-11/12 cleanup (PORT_CLEANUP_PLAN.md; tags `pre-cleanup` and `post-cleanup`).

**The claim under review:** an accelerated build of ViennaRNA 2.7.2 whose `RNAfold` (and
Python `RNA.fold`) MFE output is byte-identical to upstream's CPU path for every option.
Where the device cannot reproduce an option exactly, the record goes down upstream's own
code instead. A CPU-only build (no CUDA toolkit) is stock ViennaRNA, byte for byte and
silent.

## 1. Where things are

| Path | What it is |
|---|---|
| `src/ViennaRNA/mfe/cuda/` | the backend (≈17 k lines; below) |
| `src/bin/RNAfold.c` | upstream's driver plus the hooks into the GPU path |
| `src/bin/gpu_batch.c`, `src/bin/RNAfold_gpu.h` | RNAfold's GPU chunk path: admission, chunking, the build and backtrack pipelines, the CPU slice |
| `interfaces/cuda.i`, `interfaces/cuda_python.i` | Python: `RNA.fold` / `fold_compound.mfe` use the device by default; `cpu_only=True` |
| `m4/ac_rna_cuda.m4` | zero-config detection: bare `./configure` builds the backend if it can |
| `tests/mfe_cuda_*.ts`, `tests/python/test_RNA-cuda.py`, `tests/zeroconf_configure.sh` | in-tree tests |
| `tests/fixtures/` | the bars' inputs and frozen references (below) |
| `tools/verify_*.sh`, `tools/coverage/` | the bars |
| `tools/archive/`, `docs/archive/` | retired tools and the record of retired work (nothing deleted; see the cleanup plan) |

Upstream files the port touches are listed with every changed line marked in
PORT_LOCAL_PATCHES.md (`tools/list_local_patches.sh` checks the markers);
PORT_UPSTREAM_DIFF.md and PORT_UPSTREAM_PROPOSAL.md are the upstream-facing view.

**The backend, by file**

| File | Role |
|---|---|
| `engine.c` / `engine.h` | no CUDA: registers the batch backend with `vrna_mfe_batch()`, the routing guard (which options the device reproduces exactly), splitting a batch by VRAM |
| `mfe_cuda.c` | `par_mfe()`: build the batch's device state, run the sweep, backtrack; int16 AUTO; the backtrack pipeline (N5a) |
| `fill_arrays.c`, `fill_arrays_loop.c` | the sweep: rows i = n..1, the per-row call sequence (section 2) |
| `int_loop.cu` (+ `int_loop_lyngso.inc`, `int_loop_cell(s).inc`, `int_loop_dev.h`, `interior_loopx.h`) | interior loops: Lyngsø carry + the warp kernel; the c ring |
| `hp_mb_loop.cu` (+ `hp_mb_cells.inc`, `hp_mb_dev.h`, `fml_scan_block.inc`) | hairpin / multibranch / 3' terms, new c, the fML scan |
| `modular_decomposition.cu` (+ `md_sparse.inc`, `md_cell.inc`, `md_chain_cells.inc`, `md_dev.h`) | the multiloop decomposition: sparse md (default), the dense fallback, the md tail, CUDA graph capture, VRAM accounting |
| `gquad.c`, `gquad.cu`, `gquad_dev.h` | G-quadruplexes (`-g`) |
| `device.cu` | device discovery, streams, row tables, pinned transfers, launch stats, the retired-knob table |
| `stub2.h`, `bt_retired.h`, `nth.h`, `mb_loop_fast.c` | shared declarations and helpers (much of it W. B. Langdon's 2017-18 original) |

## 2. The hot path, end to end

1. **Read and admit** (`gpu_batch.c`). Records are held back into a chunk while they fit the
   VRAM budget (`gpu_bytes_per_file()` against `compute_gpu_usable_bytes()`) and a
   matrix-area cap. A chunk that does not carry enough work to beat the host folds on
   the `-j` pool instead (`rnafold_chunk_earns_gpu()`). The output queue slot is taken at
   read time, so dispatch order never reorders the output.
2. **Build** fold compounds for the chunk on `RNA_BUILD_THREADS` threads, overlapping the
   previous chunk's fold when host memory allows (the build pipeline).
3. **Fold** through `vrna_mfe_batch()` → the registered backend (`engine.c`) →
   `par_mfe()` (`mfe_cuda.c`), which uploads the batch and runs the sweep.
4. **The sweep** (`fill_arrays_loop.c`), one row i at a time, every record of the chunk
   at once:
   `int_loop_i` (Lyngsø carry, then the interior-loop warp kernel, G cells per warp; c into a ring that
   flushes every 16 rows) → `gq_internal_i` (with `-g`) → `hp_mb_3p_i` → `new_c_i` →
   `load_my_c` → `fml_scan_i` → the md chain
   (`load_fML_modular_decomposition_load_min_fML`: sparse md + `md_close_row`, captured
   as a CUDA graph and run on the md stream beside the next row's cell work).
5. **Backtrack** each record on the host from the device's triangles
   (`RNA_BACKTRACK_THREADS`), then hand the structure to `process_record()`
   (`RNAfold.c`), which formats exactly as for a CPU record.

Where the time goes at 400 × 5601 nt on an A100, and why each default is what it is, is
in CUDA_RNAFold_History.md and the specs; the measurement records behind retired
alternatives are in `docs/archive/`.

## 3. Settings

- User settings: README, *GPU acceleration → Run-time settings* (13 variables).
- Verification, negative controls, tracing and tuning knobs: TESTING_KNOBS.md.
- Retired knobs are ignored and announce themselves once (`g_retired`, `device.cu`).

## 4. The bars, and how to run them

Every phase of the cleanup had to pass all of these. They need a configured, built tree
(`./configure && make`, which enables CUDA by itself when it can) and a GPU; the noLP bar
also needs a no-CUDA build as its upstream oracle.

```
TREE=/path/to/built/tree NOCUDA=/path/to/no-cuda/build tools/coverage/bars.sh /tmp/bars
```

| Bar | What it proves |
|---|---|
| `tools/coverage/refcheck.sh check <fixture> [MB]` | nine fixtures, plain and `-j4`, byte-identical to output frozen at `pre-cleanup` (`tests/fixtures/cleanup_ref`), with the same number of GPU sweeps |
| `tools/verify_option_parity.sh` | every RNAfold option: the accelerated build matches the CPU route |
| `tools/verify_option_matrix.sh` | 36 option pairs × knob arms, each on the route it claims |
| `tools/verify_gpu_cli.sh <tree> 8 4 8 16 32` | multi-chunk runs at small VRAM budgets reproduce the references |
| `tests/python/test_RNA-cuda.py` | the Python binding (20 tests, including two-call sequences the CLI cannot reach) |
| `tools/coverage/bt_pipeline_check.sh` | the backtrack pipeline: CPU = off = on, and both negative controls change the answer |
| `tools/coverage/knobarms.sh <fixture> [MB]` | the off-default arms that are kept (c ring, Lyngsø, dense md, sparse selftest), each with stderr proof it took its path |
| `tools/verify_nolp_parity.sh <tree> <no-cuda>` | `--noLP` against upstream, multi-chunk, and the refusals that protect it |
| `tests/zeroconf_configure.sh <tree>` | bare `./configure` accelerates; with the toolkit hidden it builds CPU-only, silent and byte-identical; `--enable-cuda` refuses rather than downgrades (run it on a **copy**: it reconfigures the tree) |

Two rules the bars follow, learned the hard way:
- **Positive evidence.** A CPU fallback is silent and byte-identical by design, so "the
  answer is right" proves nothing about the device. Each GPU arm asserts the
  `sweep shape:` line (or its arm's own ACTIVE line); each CPU reference says `RNA_GPU=0`.
- **Negative controls bite.** Every NEGCTL knob must make its bar fail; a control that
  cannot fail means the harness, not the code, is broken.
