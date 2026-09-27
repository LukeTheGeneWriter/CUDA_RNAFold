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

---

## 7. Gate 4, settled by measurement (2026-09-27)

`tests/gpu_crossover.sh` on an RTX 3050, 30 shapes, every GPU arm byte-identical
to the CPU. **Total nucleotides does not predict the crossover:**

| total nt | shape | speedup |
|---|---|---|
| 1600 | 8 × 200 | **0.84× (loses)** |
| 1600 | 4 × 400 | 2.25× |
| 1600 | 1 × 1600 | **2.69×** |

Same total length, opposite verdicts — so the 2000-nucleotide placeholder would
have sent `1×1600` to the CPU and lost 2.69×.

**`Σ L²` fits every boundary case.** Everything at or below 3.2e5 loses;
everything at or above 6.4e5 wins, across four independent shapes. The unit is
therefore **matrix area**, which is also what the device's fixed cost is
amortised against. Gate 4 is now `Σ L(L+1)/2 ≥ VRNA_MIN_GPU_CELLS` (250 000),
overridable with `RNA_MIN_GPU_CELLS`, ORed with the existing record-count arm.

The value is host-dependent — a faster device crosses over sooner — so the
threshold is a tier-1 calibration candidate (§3.1), not a constant to defend.

## 8. int16 is the default (2026-09-27)

Measured on an A100 at 5601 nt, byte-identical at every shape: md −8.8 % at 8
records, −15.5 % at 32, −17.7 % at 96, **−18.4 % at 200**; wall −1.8 % from 32
records up. VRAM halves (12 946 → 10 168 MB at 96 records), which independently
raises what admission can admit.

**It is AUTO, not merely on**, and the asymmetry is the same one §2.1 argues for
`--enable-cuda`: an **explicit** `RNA_FML_INT16=1` that cannot be honoured is an
error, while the **default** steps aside and says so. Without that, flipping the
default would have turned `RNA_SLOT_FLOW=2` into a hard failure for anyone who
had it working — `RNA_MD_PRUNE` likewise. **A new default must not break a
command line that used to work.** Both conflicts are resolved in one place, in
`rnafold_fml_int16()`, rather than at two downstream sites that call `exit()`.

## 9. The build pipeline — the measured lever, and the catch

Flow4 §A: the build is **9.1 s of a 43 s wall (21 %)** at 200 × 5601, and it
overlaps the GPU **only when there are ≥ 2 chunks** — 0.00 s overlapped with one
chunk, **4.96 s with two**. Turn the pipeline off and capping the chunk *loses*
2.4 %, which is what proves the capping win was overlap all along.

Extrapolating: with 4 chunks, chunks 2–4 build during folds 1–3, so ~3/4 of the
build should hide (~7 s). More chunks is better for overlap.

**Two constraints pull against each other**, and both are measured:

- Flow4 §B: device time falls steeply up to **~48 residents** and flattens above
  (28.93 → 17.88 → 14.94 → 14.11 s at 4/16/48/96). Chunks below ~48 records cost
  device time.
- overlap wants **more, smaller** chunks.

So the policy is a floor, not a cap: **chunk width = max(enough-to-fill-the-SMs,
total/k)**. At 5601 nt, 48 records is ~4.6 GB — comfortably inside budget — so
both can be satisfied at once, giving ~4 chunks at 200 records.

**THE CATCH, and why this is scoped rather than patched.** The chunk accumulator
is **streaming** (`src/bin/RNAfold.c:2480`): records arrive one at a time and the
chunk flushes when `gpu_chunk_n >= gpu_hard_cap` or the VRAM budget is hit.
**The total record count is not known in advance**, so "split the batch into
four" is not directly expressible. The options are:

1. a fixed width floor in cells (`max(MIN_GPU_CELLS × f, …)`) — simple, works
   with streaming, but picks the chunk count blind;
2. read-ahead far enough to know `n` — changes the driver's streaming contract;
3. adapt during the run: start narrow, widen if the overlap is not paying — the
   tier-2 tuner (§3.2), and it needs the overlap number the pipeline already
   prints.

**Option 1 is the one to build first** and it is a few lines at the flush site.
Option 3 is where it should end up. Option 2 should be avoided — the streaming
contract is what keeps memory bounded on huge inputs.

## 10. Reserve a core, or lower priority? Test written, host needed

`tests/host_threads.sh` sweeps the **reserve** axis (`RNA_BUILD_THREADS` =
`nproc`, −1, −2, half, serial), with and without a chunk cap so the pipeline has
something to overlap.

**Local run is inconclusive for a stated reason:** at 32 × 1600 the build is
0.08 s of an 8.5 s wall, so every arm lands within noise (8.467–8.491 s). The
axis only has room where the build is a real fraction — 21 % at 200 × 5601 on the
A100. **This test needs the big shape, and a many-core host.**

**The priority arm is absent, not null.** `nice` applies to a whole process and
cannot express "builders below the GPU-facing thread". It needs a per-thread knob
(`setpriority` on the builder tids, or `SCHED_BATCH`) — suggested as
`RNA_BUILD_NICE`, applied by each builder to itself, which needs no privileges
because lowering priority never does.

**And one prior the test must be allowed to overturn:** Flow4 §A showed the
builder is *supposed* to compete — 4.96 s of overlap is the largest host-side win
found. Starving it to protect the device may cost more than it saves.
