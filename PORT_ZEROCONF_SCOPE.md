# Zero-config acceleration, and a host-adaptive tuner

*Scope, written 2026-09-26 against `36eac005`. Nothing built. Asked for by Luke:
"there are too many gates and flags that a user just needs to know before the
program accelerates. We want this to work out of the box, accelerated to the best
of its abilities by default" — plus a "learn as you go" tuner that adapts to its
host in flight.*

---

## 1. How many gates there are today

Counted, not estimated. **52 distinct `RNA_*` environment knobs** are read at
runtime. Between a user and any acceleration at all there are **five conditions,
and three of them fail silently**:

| # | condition | what happens when it fails | silent? |
|---|---|---|---|
| 1 | `./configure --enable-cuda` | CUDA is **off by default**; a plain build has no accelerator | no (but nobody reads configure output) |
| 2 | `nvcc` found at configure time | `m4/ac_rna_cuda.m4` **warns and continues** with CUDA disabled | **YES** — a green build, a CPU-only binary |
| 3 | `RNA_GPU_CHUNK` set **and non-empty** | `gpu_enabled = (devices>0) && e && e[0]` — **no acceleration at all** | **YES** |
| 4 | ≥ `VRNA_MIN_GPU_BATCH` (= 10) records in the chunk | folds on the CPU | yes |
| 5 | `gpu_path_usable(opt, &why)` accepts the options | folds on the CPU | **only under `--verbose`** |

Gate 3 is the one that matters most and it is indefensible: **on a machine with a
working GPU, a CUDA-enabled build, and a 400-record input, RNAfold does no
acceleration whatsoever unless the user happens to know the name of an
undocumented environment variable.** Nothing in `--help` mentions it.

Gate 2 is the one that has already cost real time: it produced a CPU-only binary
that ran silently and correctly for 50 minutes on an A100 (see
`--with-cuda` vs `--enable-cuda`, 2026-09-24).

**The measured stakes:** the accelerator is 32.4× upstream at production shape.
Gate 3 alone is the difference between that and 1×.

---

## 2. Part one — configure that decides for itself

### 2.1 `--enable-cuda` becomes `auto`

```
--enable-cuda      force on; ERROR if no usable toolkit  (currently: warn + disable)
--disable-cuda     force off
(omitted)          AUTO: on if a usable nvcc is found, off otherwise, said loudly
```

The asymmetry is the point. **An explicit `--enable-cuda` that cannot be honoured
must fail the configure**, because "I asked for CUDA and got a CPU binary" is the
failure we already paid for. An *omitted* flag may silently fall back, because
that is the user expressing no opinion.

### 2.2 Architecture detection must not require a local GPU

This is the HPC case and it is easy to get wrong: **on a cluster the build host is
usually a login node with no GPU at all**, while the compute nodes have several.
So:

1. if `--with-cuda-arch=LIST` is given, obey it;
2. else if the build host has a visible device, target its compute capability
   **plus PTX** for forward compatibility;
3. else build a **fat binary** spanning the arches a cluster plausibly mixes
   (sm_70/75/80/86/89/90) plus PTX.

Case 3 must be the default assumption, not the fallback nobody tests. A
login-node build that only runs on login nodes is useless.

### 2.3 Configure must print a verdict

A block at the end of configure, in the style upstream already uses for its own
features:

```
  CUDA backend ........ yes
    nvcc .............. /usr/local/cuda-12.4/bin/nvcc (12.4)
    host compiler ..... gcc-11
    target arches ..... sm_80, sm_86, sm_90, +PTX
    detected device ... none at configure time (fat binary)
```

and when it is off, *why*, with the one command that would fix it.

### 2.4 Runtime: delete gate 3

`RNA_GPU_CHUNK` stops being a gate and becomes what it already is internally —
a **testing override** on chunk width. The default becomes "the VRAM budget
decides", which is the code path `RNA_GPU_CHUNK=0` already takes and which every
measurement in this project has used.

Then **announce the decision once, unconditionally**:

- engaged: one line naming the device, the chunk policy and the element width;
- declined: one line with `why` — **not** gated behind `--verbose`. A user who
  gets 1× instead of 32× deserves to be told, and the current message is
  invisible by default.

### 2.5 The knob surface: classify, then hide

52 knobs is not a user interface. The tree already has the right precedent for
sorting this out (`tools/option_status_census.py`). Three classes:

| class | example | disposition |
|---|---|---|
| **auto** | `RNA_XFER_STAGE_MB`, `RNA_BACKTRACK_THREADS`, `RNA_BUILD_THREADS` | already measure-and-decide; keep, document as overrides |
| **dev/debug** | `RNA_ROW_VERIFY`, `RNA_PHASE_SYNC`, `RNA_*_STATS`, `RNA_MK_*`, `RNA_MD_PRUNE*`, `RNA_SYNC_PROBE` | move behind one namespace or a build flag; keep out of user docs |
| **user-facing** | element width, device selection, memory budget | document, and give each a CLI flag rather than an env var |

Best guess from the census: **~6 are genuinely user-facing.** The other ~46 are
ours.

---

## 3. Part two — "learn as you go"

The analogy to a JIT finding hot paths is apt in spirit but not in mechanism.
There is no code to re-specialise; the tunables are **numeric choices among
byte-identical variants**. So the right precedent is FFTW's *wisdom* or ATLAS's
install-time search, not a tracing JIT. Two tiers.

### 3.1 Tier 1 — startup calibration (deterministic, < 1 s)

Measure the host, do not guess it. This already exists in pieces and works:
`RNA_XFER_STAGE_MB` AUTO page-locks 16 MB, times it, and pins the whole scratch
only if the host does it faster than 0.25 s/GB — **and the two hosts we have
disagree violently** (A100 wants pinning, WSL measured the exit path 3.09 s
pinned against 0.91 s staged). That is the model to generalise.

Cheap, data-independent probes: SM count and per-kernel occupancy via
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` (already used at
`megakernel.cu:1102`), H2D/D2H bandwidth, page-lock rate, L2 size and
`persistingL2CacheMaxSize`, `accessPolicyMaxWindowSize`, core count, available
host RAM.

From those, *derive* rather than hardcode: chunk width, worker counts, and the
block sizes that are currently fixed constants "tuned against one GPU (the L4)".

### 3.2 Tier 2 — in-flight adaptation (the actual "learn as you go")

A production run is thousands of rows across many chunks, so there is room to
experiment inside one invocation. Use the **first chunk** to A/B a knob and keep
the winner for the rest; persist it so the *second* run starts tuned.

Persist to a wisdom file keyed on what actually changes the answer to the
question:

```
{gpu_name, compute_cap, driver, nvcc, host_cores, length_bucket} -> {knobs}
```

in `$XDG_CACHE_HOME/ViennaRNA/rnafold-tune.json`. Length bucket matters because
every crossover we have measured moves with length — `RNA_FML_INT16` wins at
production and **loses at small record counts**, and the admission knee scales
about `L^-1.19`.

Candidates worth tuning, all measured to matter and all byte-identical:
`RNA_FML_INT16` (md −20.7 % at production, a real crossover), chunk width /
admission, `RNA_BACKTRACK_THREADS` (measured optimum 12 on one host, and the
wall *rises* past it), int_loop block size, `RNA_MD_TILE`.

---

## 4. The four traps this design has to avoid

These are not hypothetical. Each one has already happened in this project.

**(a) Sampling the wrong launch.** "Auto-tuning this kernel caused the last two
regressions, because it only ever sampled the first (always-tiny) launch." The
sweep runs `i` descending, so **the first 10 % of rows carries 0.10 % of the
traffic** and the last 10 % carries 27 %. Any in-flight measurement that starts
at row 1 is measuring a shape the run does not spend its time in. **Calibration
must skip the early sweep and sample a representative window.**

**(b) A tuner that cannot lose.** If the chosen setting is worse than the
compiled default, the run must notice and revert. Without that, a tuner is a
regression generator with good intentions.

**(c) Tuning something that is not answer-neutral.** The tuner may select only
among variants proven byte-identical. `RNA_FML_INT16`, block sizes and chunk
width qualify; `RNA_MD_PRUNE=2` does **not** (it can return a suboptimal
structure) and must be excluded **by construction** — a type or a registry, not
by remembering.

**(d) A self-tuning binary is hostile to this project's own methodology.**
Every result on this branch comes from an A/B against a control. A binary that
silently retunes itself makes that non-reproducible. So the tuner needs:
`RNA_TUNE=0` to disable, a **pin** mode that loads wisdom without re-measuring,
and a one-line report of every decision it made. Benchmarks run pinned.

---

## 5. Order, smallest useful step first

| step | what | why first |
|---|---|---|
| 1 | **delete gate 3** — `RNA_GPU_CHUNK` stops gating; announce engage/decline unconditionally | one-line change, and it is the whole difference between 1× and 32× for a new user |
| 2 | `--enable-cuda` auto + hard error when explicitly asked and unavailable; configure verdict block | the second silent gate, and the one that cost a session |
| 3 | multi-arch fatbin + PTX by default; no GPU needed at configure time | without this, a cluster login-node build is useless |
| 4 | knob census → classify → hide the ~46 dev knobs, give the ~6 real ones CLI flags | this is the actual "too many flags" complaint |
| 5 | tier-1 calibration, generalising the `RNA_XFER_STAGE_MB` AUTO pattern | deterministic, bounded, no persistence needed |
| 6 | tier-2 wisdom file + in-flight A/B, with (a)-(d) above as hard requirements | the ambitious half; worth nothing without step 5's probes |

**Steps 1-3 are the ones that deliver "it just works".** Steps 4-6 are what make
it *good* out of the box rather than merely *on*.

## 6. What this is worth, stated so it can fail

| | claim | falsified if |
|---|---|---|
| step 1 | a naive user on a GPU host goes from 1× to ~32× at production shape | the budget-decides default picks a worse chunk width than the knee — measurable now |
| step 2 | no more silently CPU-only builds | — (it is a correctness gate, not a performance one) |
| step 3 | one build runs on every node of a heterogeneous cluster | fatbin size or JIT warm-up costs more than it saves |
| step 5 | the L4-tuned block sizes are wrong on other hardware | they are already near-optimal everywhere, and calibration finds nothing |
| step 6 | second and later runs beat first runs on the same host | the crossovers are too shallow to detect reliably in flight — the admission knee is only 2-6 %, which is close to the noise floor |

Step 6's falsifier is the one I would watch: **most of the crossovers we have
measured are shallow.** A tuner that cannot resolve a 2 % difference will spend
its budget discovering noise. Step 5 may be where the real value is, with step 6
reserved for the knobs whose crossovers are large — and on current evidence that
is `RNA_FML_INT16` (20 %) and little else.
