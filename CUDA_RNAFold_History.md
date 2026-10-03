# CUDA_RNAFold — what this tree is, and how it differs from ViennaRNA 2.7.2

*Branch `Finished_Port`, 2026-09-16. This is the base tip: the port as it stands
against upstream 2.7.2, with every feature either accelerated or declined for a
recorded reason. Development branches fork from here.*

**The reference is `v2.7.2` in this repository** — the upstream release tag
(`1ffec79f`, Ronny Lorenz, 2025-12-29), which is an ancestor of this branch. So
every claim below is reproducible with `git diff v2.7.2..Finished_Port`, not
from a vendored copy that could drift.

---

## 1. What this is

A CUDA backend for ViennaRNA's MFE matrix fill. It folds a **batch** of
sequences at once on the GPU, filling exactly the matrices upstream's own
recursion fills, and it is checked by one bar throughout: **the output must be
byte-identical to the same binary with the GPU path off.**

It is not a re-implementation of RNAfold. Everything that decides an *answer*
still comes from RNAlib — the energy model, the constraint semantics, the
backtrack. The device work is the O(n³) fill.

**Where it stands at this tip:** 400 × 5601 nt folds in **85.9 s** on an
A100-SXM4-40GB at the shipped defaults, against **32.4×** the same workload on
upstream's CPU path (`BENCH272_V5_RESULTS.md`).

---

## 2. What differs from stock 2.7.2

`git diff --shortstat v2.7.2..Finished_Port` — **166 files, +38 493, −51**. That
headline is misleading on its own, so here is the split that matters:

| | files | lines | what it is |
|---|---|---|---|
| **New CUDA subdirectory** `src/ViennaRNA/mfe/cuda/` | 18 | **+13 389** | ours entirely. Upstream can take it or leave it |
| **Library files upstream owns** | 8 | **+477 / −7** | the part that needs defending. All marked in-source |
| **The driver** `src/bin/RNAfold.c` | 1 | **+1 848 / −25** | ours in effect; not part of any proposal |
| **Build system, README and SWIG interfaces** | 24 | +764 / −19 | the configure summary, the nvcc libtool shim, test wiring, README's GPU section, `setup.py`'s `mfe/cuda` exclusion, the SWIG 4.5 fix, the Python interface to the GPU backend (`interfaces/cuda.i`, `cuda_python.i`), and the same SWIG 4.5 fix in RNAxplorer |
| **New autoconf macros + tests** | 40 | +5 159 | `m4/ac_rna_cuda.m4`, the `.ts` suites, `tests/upstream/` probes, `tests/zeroconf_configure.sh`, `tests/python/test_RNA-cuda.py` |
| **Project documents and tools** | 75 | +16 856 | not code. Scopes, specs, notebook generators |

> **These figures are recomputed, and two earlier versions of this section were
> wrong.** It once read "216 files, +73 162" and "9 files, +2 296 / −20". The
> first had drifted — commits landed after it was written, and the notebooks and
> result JSON have since been untracked entirely. The second mixed two things:
> the +2 296 included the +150 of `Makefile.am` glue while the file count did
> not. The nine files upstream-or-ours (rows 2 + 3 above) are **+2 325 / −32**.
> `Note_to_TBI.md` carries the same split and is computed from the tree.

**Only the "Library files upstream owns" row is a change to something upstream owns**, and every one of
those edits is bracketed in the source itself:

```
/* VRNA-PATCH-BEGIN(<id>, <CLASS>) -- PORT_LOCAL_PATCHES.md */
```

`tools/list_local_patches.sh` lists them from the source rather than from a
document that can drift, and fails if a marker is unpaired or an upstream file
is modified without one. As of this tip: **18 marked regions across 11 files** (six of them the SWIG 4.5 fix in RNAxplorer),
plus `src/bin/RNAfold.c` which is declared a local patch *whole-file* (it is
+1 670 lines of driver — a CUDA chunker around upstream's per-record loop — and
marking each hunk would be noise pretending to be precision).

| class | meaning | count |
|---|---|---|
| **DEFECT** | an upstream bug we fixed; submittable on its own, with a reproducer | 1, plus 6 regions of the SWIG 4.5 fix in RNAxplorer |
| **SEAM** | an attachment point the accelerator needs and upstream does not have; shaped to be useful on CPU with no GPU at all | 7 |
| **REACH** | a capability upstream **already has** that its public API cannot reach | 4 |

The one DEFECT is `params-cache-race` (`params.c`): four unsynchronised
file-scope statics in `vrna_params()`'s cache, which is a data race for any
threaded caller — GPU or not. It is what blocked threading the fold-compound
build, and it is the strongest standalone upstream contribution here.

### The nine upstream files

| file | class of change |
|---|---|
| `src/ViennaRNA/mfe/global.h` | SEAM (`vrna_mfe_batch` API) + REACH (circular post-process) |
| `src/ViennaRNA/mfe/mfe.c` | SEAM (backend state, inside-engine hook) + REACH (circular, bps backtrack) |
| `src/ViennaRNA/grammar/mfe.h`, `grammar.c`, `gr_extension_mfe.c`, `intern/grammar_dat.h` | SEAM — the inside-engine seam: a fold compound can be handed an alternative matrix-fill implementation |
| `src/ViennaRNA/backtrack/global.h` | REACH — expose the base-pair-stack backtrack entry |
| `src/ViennaRNA/params/params.c` | **DEFECT** — the parameter-cache race |
| `src/bin/RNAfold.c` | the driver: chunking, VRAM budget, build pipeline, CPU queue |

---

## 3. What is accelerated, and what is not

Counted by `tools/option_status_census.py`, which reads the option list from
`src/bin/RNAfold.ggo` and each verdict from `PORT_OPTION_STATUS.md` — and exits
non-zero if any option is classified nowhere.

| | count |
|---|---|
| **ACCELERATED**, byte-identical to the CPU route | **25** |
| **DECLINED**, route *and* effect measured | **10** |
| NEUTRAL — never reaches the recursion; the fold is still accelerated | 23 |
| UNREACHABLE from this CLI | 1 |
| SPLIT — `--dangles`: **d0/d2 accelerated, d1/d3 declined** | 1 |
| **total** | **60** |

### Accelerated

The default model, and: `-T`/`--temp`, `-d0`, `-d2`, `-p`/`--partfunc` (MFE
fill), `--MEA`, `--bppmThreshold`, `--betaScale`, `--pfScale`, `--noLP`,
`--noGU`, `-4`/`--noTetra`, `--salt`, `--helical-rise`, `--backbone-length`,
`-g`/`--gquad`, `--nsp`, `-P`/`--paramFile`, `--maxBPspan` (any span),
`-c`/`--circ`, `--noClosingGU`, `-C`/`--constraint`, `--canonicalBPonly`,
`--enforceConstraint`, `-j`/`--jobs`, `--unordered`, `--ImFeelingLucky`, and
`--commands` **when the file queues only hard constraints**.

### Declined — four families, not ten decisions

| family | options | why |
|---|---|---|
| **SHAPE / probing** | `--shape`, `--shapeMethod`, `--shapeConversion`, `--sp-data`, `--sp-strategy`, `--sp-preprocess` | **DEFERRED.** Soft constraints reach the recursion through per-loop-type wrapper layers, not one term. Scope is settled (Deigan only — one carrier, and the best average performer in Lorenz et al. 2016); waiting on a test bench with real reactivity data |
| **modified bases** | `-m`/`--modifications`, `--mod-file` | soft constraints too (`vrna_sc_mod_*`); blocked behind the SHAPE decision |
| **ligand motif** | `--motif` | `vrna_sc_add_hi_motif()` installs a soft constraint; declined by gate 2 on `fc->sc`, measured both halves |
| **synthetic alphabet** | `--energyModel` | **REJECTED.** The device packs the sequence in 3 bits; `energy_set` is defined over 20 letters |

plus **`-d1`/`-d3`**, which need a second `DMLi` generation the sweep does not
carry — the only entry left on the backstop.

**No silent wrong answer is known anywhere on the option surface**, and that
claim rests on measurements rather than on reading the guard: every declined
option above has been *forced* onto the device with `RNA_ENGINE_ALLOW` and its
behaviour recorded (`tools/probe_declined_options.sh`).

---

## 3a. Building it, verified from a clean clone

**Checked the only way that means anything**: cloned from GitHub into an empty
directory and built it, rather than trusting a tree that was already working.
That is how the `doc/man2rst.py` mode defect was found — every notebook in this
project had been papering over it with a `chmod +x` for months.

```sh
git clone https://github.com/LukeTheGeneWriter/CUDA_RNAFold.git
cd CUDA_RNAFold && git checkout Finished_Port

# the vendored third-party sources ship as tarballs and autogen expects them open
tar -xjf src/dlib-*.tar.bz2   -C src/
tar -xzf src/libsvm-*.tar.gz  -C src/

autoreconf -i            # or ./autogen.sh
./configure              # no CUDA flag: it looks for a toolkit and decides
make -j$(nproc)
```

**Zero-config (2026-10-02).** This is upstream's own install procedure, unchanged:
`./configure` looks for a CUDA toolkit (`PATH`, then `CUDA_HOME`, `CUDA_PATH`,
`CUDAToolkit_ROOT`, `CONDA_PREFIX`, then `/usr/local/cuda*` and `/opt/cuda*`), chooses
the architectures from what that toolkit supports, and builds the backend if it can —
otherwise the ordinary CPU-only library, exactly as stock ViennaRNA. Its summary always
states which you got, under **GPU Acceleration**. `--enable-cuda` *demands* the backend
and fails rather than downgrading; `--disable-cuda` refuses it. Until this change the
branch needed `--enable-cuda` at configure time and an environment variable at run
time, so following upstream's instructions gave an unaccelerated build that said
nothing. Ported from Lukes_Flow_Batching (`04480802`), without that branch's
performance tuning.

**Build prerequisites**: `gengetopt`, `help2man`, `xxd`, `libtool`, `texinfo`,
`doxygen` (install it **before** `configure` — it is probed there), and — for the
GPU backend — a CUDA toolkit.

**CUDA 13 and the architecture list (2026-10-02, `86ec6020`).** This branch stopped
building on CUDA 13.0 (the Colab image of that date), twice: its default
`--with-cuda-arch` was a fixed `60,70,75,80,86,89`, and CUDA 13 removed `sm_60` and
`sm_70` (`nvcc fatal: Unsupported gpu architecture 'compute_60'`); and CUDA 13's
kernel launch stubs need libstdc++ (`__cxa_guard_*`), which libtool's C link did not
pull in. Configure had not caught the first because its test compile omitted the
architecture flags. Now, unless `--with-cuda-arch` is given, configure asks `nvcc`
what it can emit and builds for the local GPU if one is visible, otherwise a fat
binary over everything that toolkit supports, plus PTX for the highest; the test
compile uses those flags, so an unusable list disables CUDA at configure with a
warning rather than failing in `make`; and `-lstdc++` is linked. Nothing the tree
accelerates, or how, changed. (The zero-config configure above supersedes that
macro and keeps both fixes.)

**Using it.** Nothing to set: an accelerated build uses the GPU when one is visible
and the options are supported, and says so on stderr —

```
bin/RNAfold.c   GPU acceleration ON (1 device); reaching the device measured 1.55 s,
                so with 1 job a chunk needs 1.04e+06 matrix cells to beat the host
```

(that is a cold driver; a warm one measured ~0.08 s, a floor of 5.4e4 cells)

A chunk goes to the device when its work (matrix cells) exceeds a floor derived on
the spot from the measured cost of reaching the device, the CPU fold rate and `-j`; a
present-but-unused device prints one line saying why. `RNA_GPU=0` asks for the CPU path
explicitly — every correctness harness in `tools/` uses it for its reference arm, and
asserts that arm did not sweep, because a reference that silently runs on the GPU
compares the GPU against itself. `RNA_GPU_CHUNK` is only a chunk-width override now
(it used to be the on switch); `RNA_GPU_WORK_FLOOR=0` sends every chunk to the device.

```sh
src/bin/RNAfold --noPS -i sequences.fa             # GPU, if there is one
RNA_GPU=0 src/bin/RNAfold --noPS -i sequences.fa   # stock CPU path
```

**From Python (2026-10-02, ported from Lukes_Flow_Batching).** `RNA.fold(seq)`,
`RNA.fold([seq, ...])` (one device batch) and `fold_compound.mfe()` use the GPU;
`cpu_only=True` calls upstream's own. They fold on the host with the identical answer
when there is no device or the model is not supported, `RNA_GPU=0` keeps them on the
host, and the backend's stderr diagnostics are left out unless `RNA_GPU_VERBOSE=1`.
`RNA.cuda_batches()` proves which path ran. A second batch in one process, which
`RNAfold` never makes, reached three defects on this branch that are now fixed:
device state was not released between batches; `teardown_gpu()` chose what to free
from the int16 setting, which can change mid-process (with `RNA_FML_INT16=1` and a
parameter table loaded between batches, the next batch hit "an illegal memory access
was encountered"); and a batch of records ≤ 3 nt segfaulted. Verified on the laptop:
`tests/python/test_RNA-cuda.py` 16/16 in a CUDA build, with its new int16 case red on
the old teardown; nine models × three entry points silent on stderr; `RNAfold`'s
stdout unchanged and its GPU output equal to `RNA_GPU=0`'s; a `--disable-cuda` build
builds, imports, and passes the suite with no device (the device-only int16 case skips); and `python -m build` from a CUDA-configured tree builds a CPU-only wheel (`engine.c` alone, host-only) that installs into a fresh venv and passes the same way. Details in `Note_to_TBI.md` §4.1.

**Verified from a clean clone, 2026-10-02** (laptop, RTX 3050, CUDA 12.4): a bare
`./configure` reports *CUDA backend: yes, 86 +PTX (detected from the local device)*;
with no environment the run prints the line above and sweeps on the device, `RNA_GPU=0`
prints `GPU acceleration OFF: RNA_GPU=0` and does not, and the two answers are
byte-identical; `RNA_GPU_WORK_FLOOR=0` and `RNA_GPU_CHUNK=0` still behave as documented.
`tools/verify_option_parity.sh` — reference arm on the CPU, every accelerated option on
the GPU for all 30 records — gives 45 options identical and the three declined ones on
the CPU as required; the repaired salt, constraint and declined-option harnesses pass;
and `tests/zeroconf_configure.sh` passes 16/16 — including `--enable-cuda` refusing with
no toolkit reachable, a bare `./configure` without one succeeding CPU-only and silent on
stderr, and the CPU-only and accelerated builds giving byte-identical output. The A100 three-way run of the same date (before this change, with
`--enable-cuda` and the run-time variable) measured this branch at 45.7× upstream 2.7.2
on all 12 cores at 400 × 5601 nt, byte-identical across eleven option cases.

---

## 4. How correctness is defended

Three gates, and they are not the same gate:

| # | gate | where | what it is for |
|---|---|---|---|
| 1 | `gpu_path_usable()` | `src/bin/RNAfold.c` | an **optimisation** — decides whether to register the backend at all |
| 2 | `vrna_cuda_engine_supports()` | `mfe/cuda/engine.c` | **the authority** — consulted per fold compound; a decline falls back to upstream's `vrna_mfe()` |
| 3 | `VRNA_CUDA_BACKSTOP` | `mfe/cuda/fill_arrays.c` | a **tripwire** — exits if an unsupported model reaches the sweep, i.e. if 1 and 2 both have a hole |

**A hole in gate 1 costs performance. A hole in gate 2 costs correctness.** The
driver builds its own fold compounds, so everything `process_record()` applies
to its own — constraints, command files, motifs — is applied there too;
otherwise gate 2 inspects a different object than the one that gets folded.

### The bars

| bar | what it checks |
|---|---|
| `tools/verify_option_parity.sh` | **45 checks** over 30 records: every option, accelerated or declined, byte-identical to the same binary with the GPU off |
| `tools/verify_constraint_parity.sh` | **5 constraint shapes** over 30 records at 80–1240 nt, byte-identical with the sweep running and self-consistent against `RNAeval` |
| `tools/probe_declined_options.sh` | forces each declined option onto the device and reports what it actually does |
| `tools/option_status_census.py` | every option has a verdict |
| `tools/list_local_patches.sh` | every upstream edit is marked |
| `tests/mfe_cuda_*.ts` | ten in-tree regression tests (`make check`) |
| `tests/python/test_RNA-cuda.py` | the Python interface against `cpu_only=True`, including second batches in one process, `RNA_GPU=0`, and silence on stderr |

---

## 5. What is known and not fixed

Recorded here so the next branch starts from the truth rather than from
optimism. Full detail in `STRESS272_RESULTS.md` §30–35.

* **`modular_decomposition` is bandwidth-bound in production** — 82.8 % of DRAM
  peak at 400 × 5601, 44 % of wall. The roofline position is a property of the
  **grid**, not the kernel: the same kernel reads 7.9 % of peak on a small
  fixture. `PORT_LUKES_FLOW_BATCHING.md` is the plan.
* **`int_loop` is retirement-limited, not occupancy-limited.** Raising warps per
  block lifts the ceiling and *lowers* achieved occupancy.
* **`RNA_STREAM_OVERLAP=2` is experimental and unverified at scale.** A race at
  400 × 5601 returned two different wrong answers; the missing parity gate is
  fixed and **not re-verified at that shape**. Default is 0.
* **The exit path is serialised and pageable** — `fetch_mx` + `backtrack` is
  14 % of wall at 6.5 GB/s against ~25 pinned.
* **Pseudoknots are out of reach by construction** — the recursion this backend
  fills is strictly nested. It matters for SHAPE, whose data carries pseudoknot
  signal the model cannot express.

---

## 6. Branches from here

`Finished_Port` is a base tip, not a development line. Work forks from it:

* **`Lukes_Flow_Batching`** — `PORT_LUKES_FLOW_BATCHING.md`: column tiling in
  `modular_decomposition`, row-granular stream-out, and continuous-flow
  retirement.

Earlier lines (`Continuous_Flow_Batching`, `Staggered_Row_Batching`,
`Forward_port_272`, `port27`) are history; `port27` is where this tip came from.
