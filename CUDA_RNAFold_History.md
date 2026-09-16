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

`git diff --shortstat v2.7.2..Finished_Port` — **216 files, +73 162, −24**. That
headline is misleading on its own, so here is the split that matters:

| | files | lines | what it is |
|---|---|---|---|
| **New CUDA subdirectory** `src/ViennaRNA/mfe/cuda/` | 18 | **+13 102** | ours entirely. Upstream can take it or leave it |
| **Upstream files modified** | 9 | **+2 296 / −20** | the part that needs defending. All marked in-source |
| **Build glue** (`Makefile.am` × 2) | 2 | +150 | `--enable-cuda`, the nvcc libtool shim |
| **Project documents, notebooks, results** | ~187 | +57 600 | not code. Scopes, measurements, notebooks, JSON artifacts |

**Only the middle row is a change to something upstream owns**, and every one of
those edits is bracketed in the source itself:

```
/* VRNA-PATCH-BEGIN(<id>, <CLASS>) -- PORT_LOCAL_PATCHES.md */
```

`tools/list_local_patches.sh` lists them from the source rather than from a
document that can drift, and fails if a marker is unpaired or an upstream file
is modified without one. As of this tip: **12 marked regions across 8 files**,
plus `src/bin/RNAfold.c` which is declared a local patch *whole-file* (it is
+1 669 lines of driver — a CUDA chunker around upstream's per-record loop — and
marking each hunk would be noise pretending to be precision).

| class | meaning | count |
|---|---|---|
| **DEFECT** | an upstream bug we fixed; submittable on its own, with a reproducer | 1 |
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
cd CUDA_RNAFold && git checkout Lukes_Flow_Batching

# the vendored third-party sources ship as tarballs and autogen expects them open
tar -xjf src/dlib-*.tar.bz2   -C src/
tar -xzf src/libsvm-*.tar.gz  -C src/

./autogen.sh
./configure --enable-cuda --without-python --without-perl --without-swig             --without-doc --without-rnaxplorer --without-forester             --without-kinfold --without-rnalocmin
make -j$(nproc)
```

**Build prerequisites**: `gengetopt`, `help2man`, `xxd`, `libtool`, `texinfo`,
`doxygen` (install it **before** `configure` — it is probed there), and a CUDA
toolkit with `nvcc` on `PATH`. Without `--enable-cuda` the tree builds as
stock ViennaRNA and the GPU path is simply absent.

**Using it.** The GPU path is off unless asked for: `RNA_GPU_CHUNK` is the
**master switch, not a cap** — unset means fold on the CPU, and `0` means "no
cap, the VRAM budget decides".

```sh
RNA_GPU_CHUNK=0 src/bin/RNAfold --noPS -i sequences.fa
```

Verified from the clean clone above, 24 × 600 nt: **byte-identical to the same
binary with the GPU path off**, one sweep, the AUTO build pipeline engaging, and
`-C` — the feature that shipped today — accelerated and byte-identical too.

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
