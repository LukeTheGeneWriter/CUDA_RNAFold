# Testing, diagnostic and tuning variables

The user settings are in the README (*GPU acceleration → Run-time settings*). This file
lists the rest: what the verification bars and measurement notebooks use, and the tuning
knobs whose defaults were measured on particular hardware. None is needed to fold
correctly. Every variable here except the negative controls leaves the output
byte-identical. The negative controls exist to change it, and their bars require that
they do. Defaults are the values in the source as of the 2026-10-11 cleanup
(PORT_CLEANUP_PLAN.md, section 3).

Most of these announce themselves on stderr when set. A variable that has been
**retired** with its code is ignored, and the backend prints `... IGNORED: retired with
...` once for it. The list of retired names is in `src/ViennaRNA/mfe/cuda/device.cu`
(`g_retired`).

## Verification (T2)

| Variable | Effect |
|---|---|
| `RNA_ROW_VERIFY=1` | checks the device row kernels against the host loops, cell by cell; implies the host sweep (`RNA_GPU_SWEEP=0`); refused with `--noLP` |
| `RNA_HC_VERIFY=1` | checks every word of the hard-constraint bitmasks, including bits nothing reads yet; slow |
| `RNA_CIRC_VERIFY=1` | checks fM2_real (circular RNA) against its definition, O(n³) per record |
| `RNA_TRI_CHECKSUM=1` | hashes the whole c and fML triangles at the end of each chunk, and counts near-INF finite cells; compare runs of the same encoding |
| `RNA_MD_SPARSE_SELFTEST=1` | builds the sparse md's candidate lists beside the dense md and compares them cell for cell; turns CUDA graph capture off |
| `RNA_MD_SPARSE_DELTA_MAX=d` | test only: with `RNA_MD_SPARSE_ENTRY=4`, treats steps beyond ±d as not fitting (default 32767; `0` forces slot exhaustion) |
| `RNA_INT_LOOP_LYNGSO_SELFTEST=1` | runs the dense interior-loop kernel and checks the Lyngsø recurrence against it (keeps Lyngsø off unless `RNA_INT_LOOP_LYNGSO` is set) |

## Negative controls (T2): each one must make its bar fail

| Variable | What it breaks on purpose |
|---|---|
| `RNA_MD_SPARSE_NEGCTL=1` | drops the scan term from the sparse md |
| `RNA_INT_LOOP_LYNGSO_NEGCTL=1` / `=2` | drops the Lyngsø carry / the e-ring writes |
| `RNA_BT_PIPELINE_NEGCTL=1` | zeroes the retained c triangle before the backtrack fetch |
| `RNA_MD_TILE_FUSE_NEGCTL=1` | runs the fused scan before the c chain (flagged path, below) |
| `RNA_MD_TILE_REVERSE=1` | runs the column blocks right to left (flagged path, below) |

## Tracing and timing (T2)

| Variable | Effect |
|---|---|
| `RNA_RECORD_TRACE=<path>` | per-record CSV (route, fold window, output time) to `<path>`, and per-chunk spans to `<path>.chunks.csv` |
| `RNA_LAUNCH_STATS=1` | ends every instrumented launch in an event sync and reports per-kernel device time (changes timing, not answers) |
| `RNA_LAUNCH_STATS_CSV=<path>` | the same, one line per launch, in launch order |
| `RNA_PHASE_SYNC=1` | syncs at every phase boundary so the phase timers are true GPU times |
| `RNA_HOST_AVAIL_MB=N` | test hook: pretends N MB of host memory are available (drives the build pipeline's AUTO decision) |

## Tuning (T3): live defaults, re-measure on new hardware

Each default below was measured, mostly on an A100 at 400 × 5601 nt and on an RTX 3050
laptop. Each off arm is still a tested path; `tools/coverage/knobarms.sh` and the option
matrix pin the ones the cleanup kept.

**Admission (RNAfold)**

| Variable | Default | Effect |
|---|---|---|
| `RNA_GPU_WORK_FLOOR` (alias `RNA_MIN_GPU_CELLS`) | unset (the model decides) | forces the pre-S3 test: a chunk goes to the device once its total matrix cells reach this, and no record is too short; `0` always uses the device |
| `RNA_GPU_COST_MODEL` | `K=2.65e-7,h=6.8e-4,k=8.1e-8,C0=0.34,beta=0.18` | the admission model's constants (any subset, comma-separated): host `K·cells^1.15` s per record at one job, scaling `j/(1+beta(j−1))`; device `h + k·cells` per record plus `C0` before the first chunk. Sets the minimum length (said in the ON line) and the minimum batch. Measured on an RTX 3050 laptop (PORT_SHORT_SEQUENCES.md §8.2) |
| `RNA_MIN_GPU_BATCH` | `10` | smallest chunk the device takes; also the floor `RNA_CPU_SLICE` keeps on the GPU |
| `RNA_GPU_CHUNK_CELLS` | `753000000` | caps a chunk's matrix area (48 × 5601 nt); `0` disables the cap |
| `RNA_GPU_UNIFORM_CHUNKS` | unset | set: a chunk is flushed whenever the record length changes |

**Schedule**

| Variable | Default | Effect |
|---|---|---|
| `RNA_STREAM_OVERLAP` | `2` | `0` one stream; `1` hp_mb beside int_loop; `2` the md chain beside the next row's cell work |
| `RNA_CUDA_GRAPH` | on | `0` issues the md chain directly instead of capturing it as a graph |
| `RNA_GPU_SWEEP` | on | `0` runs the sweep's row loops on the host |
| `RNA_MD_TAIL` | on | `0` restores the seven-operation md chain (the collapsed tail is two) |
| `RNA_MD_ROW_SYNC` | `0` | `1` syncs the host after the md chain every row (forced on off the GPU sweep) |
| `RNA_LOAD_MY_C_SYNC` | `0` | `1` restores a device sync in load_my_c every row |
| `RNA_GPU_BLOCKING_SYNC` | off | `1` makes the host block, not spin, on device syncs |
| `RNA_XFER_STAGE_MB` | `8` | backtrack scratch copied through a pinned stage of N MB; `0` pins the scratch itself |

**Multiloop decomposition (md)**

| Variable | Default | Effect |
|---|---|---|
| `RNA_MD_SPARSE_RHO` | `0.25` | candidate-list capacity per column, min(j, ⌈ρj⌉ + 8) |
| `RNA_MD_SPARSE_ENTRY` | `8` | bytes per list entry, `8` or `4` |
| `RNA_MD_SPARSE_LANES` | `32` | lanes per column in the sparse scan, `8`, `16` or `32` |
| `RNA_MD_RING_DEPTH` | `2` | rows held by the dml and cc rings |
| `RNA_ROW_BATCH` | `5` | fML triangle rows written per flush from a row-major stage; `0` writes every row |
| `RNA_MD_BLOCK_SIZE`, `RNA_FMLI_BLOCK_SIZE` | occupancy | block size for the dense md / fmli kernels |

**Interior loops (int_loop)**

| Variable | Default | Effect |
|---|---|---|
| `RNA_INT_LOOP_LYNGSO` | on | the Lyngsø interior-loop recurrence; `0` restores the dense enumeration |
| `RNA_INT_LOOP_LYNGSO_V2` | on | the 3-D carry grid |
| `RNA_INT_LOOP_LYNGSO_ERING` | on | fresh entries read a 32-row e ring |
| `RNA_INT_LOOP_LYNGSO_PT` | auto | forces planes per thread (`1`, `5` or `25`) |
| `RNA_INT_LOOP_LYNGSO_LB` | off | ring carry at `__launch_bounds__(128, 16)`; pending an A100 run |
| `RNA_INT_LOOP_CELLS_PER_WARP` | `auto` | cells per warp, 1–32 |
| `RNA_C_RING` | on | c values ride a ring and flush to the triangle in batches; `0` writes every row |
| `RNA_C_FLUSH_ROWS` | `16` | rows per c flush, 1–32; `1` keeps the ring's reads without the batching (the control) |
| `RNA_LOAD_MY_C_BLOCK_SIZE` | occupancy | block size for load_my_c |

**Hairpin and multiloop rows (hp_mb)**

| Variable | Default | Effect |
|---|---|---|
| `RNA_HP_TABLE` | on | hairpin length term from a host-built table; `0` computes it per cell |
| `RNA_FML_SCAN_THREADS` | `256` | threads per fML scan block |
| `RNA_HP_MB_BLOCK_SIZE`, `RNA_NEW_C_BLOCK_SIZE`, `RNA_STACK_ROW_BLOCK_SIZE`, `RNA_FML_PREV_BLOCK_SIZE`, `RNA_ROW_FUSE_BLOCK_SIZE` | occupancy | block sizes |

**Flagged: kept pending final tests (cleanup decision D4)**

These are off by default. They are kept until a final measurement decides; if they do
not pay, they are deleted against the `pre-cleanup` / `post-cleanup` tags named in the
source comment at the fusion code.

| Variable | Default | Effect |
|---|---|---|
| `RNA_MD_TILE_CB=N` | `0` | each row runs as N-column blocks; needs `RNA_STREAM_OVERLAP=0`, `RNA_CUDA_GRAPH=0`, `RNA_C_RING=0`, `RNA_ROW_BATCH=0` |
| `RNA_MD_TILE_RB=N` | `1` | with `RNA_MD_TILE_CB`: N rows per block-row, column-block major |
| `RNA_MD_TILE_FUSE` | on under `RNA_MD_TILE_CB` | a step's c chain and fML scan in one launch |
| `RNA_ROW_FUSE=1` | off | hp_mb_3p + new_c + load_my_c as one kernel per row |
