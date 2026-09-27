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

> **Release requirement, see §11.** Measured 2026-09-27: with `RNA_GPU_CHUNK` unset
> the device is never used and stderr is **empty**. A user following upstream's
> documented `./configure && make` gets 1x, silently. This is criterion 2 of §11.2.

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

> **The over-splitting warning below is WRONG at production scale -- see §12.2.**
> With the cap off, 400x5601 forms 2 chunks and hides only 3.23 s of an 18.85 s build;
> with the cap it forms 9 and hides 18.05 s, and is 6.4 s FASTER. More chunks buy more
> overlap than they cost.
>
> **Superseded in part by §11.3.** The cells floor is correct but **unreachable**
> for any input of more than 10 records, because `rnafold_chunk_earns_gpu()`
> short-circuits on record count and `VRNA_MIN_GPU_BATCH` is 10. It is also the
> wrong *shape* for heterogeneous compute: both gates decide per CHUNK, and a
> router has to decide per RECORD against live queue occupancy. Read §11.3 before
> tuning the floor's value.

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

## 11. REQUIREMENT: as installable as upstream, and accelerated without being told

**This is a release requirement, not an improvement.** The work is not done until a
user who has never heard of this fork gets acceleration by following upstream's own
instructions. Recorded 2026-09-27 at Luke's direction, with the thousands of
existing ViennaRNA users as the audience.

### 11.1 The bar, taken from upstream verbatim

`https://github.com/ViennaRNA/ViennaRNA` documents exactly these paths:

| path | what the user types |
|---|---|
| release tarball | `tar -zxvf ViennaRNA-2.7.2.tar.gz && cd ViennaRNA-2.7.2 && ./configure && make && sudo make install` |
| no root | `./configure --prefix=$HOME/ViennaRNA && make install` |
| git clone | unpack `src/libsvm-3.35.tar.gz` and `src/dlib-20.0.tar.bz2`, install the build tools, `autoreconf -i`, then `./configure && make && sudo make install` |
| bioconda | `conda install viennarna` |
| PyPI (Python interface) | `python -m pip install viennarna` |
| binaries | prebuilt packages for Linux, Windows and macOS from the website |

Build docs live in `INSTALL`; options are discoverable through `./configure --help`.

Note what is **not** in that list: no accelerator flag, no environment variable, no
device selection. `./configure` with no arguments is the documented path, and it is
the one almost everyone uses.

### 11.2 Acceptance criteria

Each of these is testable. Status as of 2026-09-27, after the zero-config change:

| # | criterion | status |
|---|---|---|
| 1 | bare `./configure` accelerates | **MET** — detects nvcc, capability, links, reports |
| 2 | no env var needed at run time | **MET** — gate 3 deleted |
| 3 | configure prints its verdict; a declined run says why | **MET** |
| 4 | git-clone path works; requirements documented | **MET** for CUDA (README); upstream's own `yacc` requirement for RNAforester is unchanged and is theirs |
| 5 | the tarball builds accelerated | **MET** — `EXTRA_DIST` carries every `.cu`/`.inc`/private header and sits OUTSIDE the CUDA conditional, so a non-CUDA `make dist` still ships an accelerable tree |
| 6 | a no-CUDA build configures, builds, folds | **MET, and it was BROKEN** — see below |
| 7 | knob surface documented | open (§2.5) |

**Criterion 6 caught a live defect that nothing else would have.** The
`RNA_RECORD_TRACE` block landed inside `#ifdef VRNA_WITH_CUDA` while being called
from unconditional code, so a CPU-only build failed with three
implicit-declaration errors. That is the bioconda and PyPI build — no toolkit
present — so it would have broken every such build while looking perfectly fine on
any machine with a GPU. It was found by actually configuring with `nvcc` hidden and
running `make`. This is now the strongest argument in this document for keeping a
no-CUDA build in CI: the accelerated build cannot detect it.

The original text of these criteria follows, for the reasoning behind each.

1. **`./configure` with no arguments produces an accelerated binary** on a host with
   a CUDA toolkit and a device, and a working CPU-only binary everywhere else.
   Today `--enable-cuda` is required, and omitting it yields a silently CPU-only
   build — see §2.1 and `feedback_silent_fallback_needs_positive_evidence`.
2. **No environment variable is needed to reach the device.** Today gate 3
   (`RNAfold.c:2513`) enables CUDA only when `RNA_GPU_CHUNK` is set to something;
   unset means no acceleration, with **0 bytes of stderr** to say so (verified
   2026-09-27). This alone means a user following upstream's instructions gets 1×.
3. **`configure` prints its verdict**, and a declined runtime prints `why` without
   `--verbose` (§2.3, §2.4).
4. **The git-clone path works from a clean checkout** with upstream's tool list plus
   whatever CUDA adds, and `INSTALL` states the CUDA requirements and how to opt
   out. `doc/man2rst.py` mode was already one such blocker
   (`project_man2rst_mode_blocks_a_clone_build`).
5. **The distributed tarball builds accelerated**, i.e. `make dist` carries the CUDA
   sources and the `.cu` rules survive out-of-tree builds.
6. **Upstream's package paths still work**: a bioconda/PyPI build with no CUDA
   present must configure, build and pass tests unchanged. Acceleration is additive
   or it is not shippable.
7. **Documented knob surface.** ~6 of 52 knobs are user-facing (§2.5); the rest are
   ours and must not appear in user docs.

### 11.3 Gate 3 and the cells floor both have to go — DONE (2026-09-27)

Both are gone. What replaced the floor is not another constant: the admission test
is now a single work test in matrix cells, against a threshold **derived** from
`F * R_host * jobs`, where `F` is the measured cost of reaching the device (timed at
the `vrna_cuda_devices()` probe) and `jobs` is how many cores the host would
otherwise fold on. One compiled-in number remains and it is a *CPU fold rate*, which
is interpretable and is the obvious thing for §3.1's calibration to measure.

Re-measuring on this host also **overturned the data the old floor was fitted to**:
the device now wins at every size tested, including 8x200 at 1.23x where
`tests/gpu_crossover.sh` had recorded a 0.84x loss. So the 250000-cell floor was not
merely the wrong shape, it was keeping work on the CPU that the device would have
won. The floor is still necessary, though, and for a reason a constant cannot
express: those wins are on a **warm** driver, where reaching the device costs 0.09 s.
Cold it costs 0.75 s, which flips 3x300 from a 1.09x win to a 2.3x loss. Same host,
same input, opposite verdict — decided by a quantity the code now measures.

The argument below is what motivated the change and still stands.

Luke's instruction: *"You may need to fully rewrite or eliminate gate 3 and the
cells floor to enable true heterogeneous compute."* The measurements agree, and the
reason is sharper than "they are inconvenient".

**Gate 3 is not a policy, it is an accident of testing.** `RNA_GPU_CHUNK` is a chunk
*width* override that acquired a second, undocumented job: presence-as-enable. Its
own `0` value means "no cap, the budget decides", so the flag's off state and its
most useful state are the same value. Delete the presence test (§2.4); keep the
override.

**The cells floor cannot do the job it was added for.** It was added so one very
long sequence would still reach the device. But `rnafold_chunk_earns_gpu()` returns
1 as soon as `n >= rnafold_min_gpu_batch()`, and `VRNA_MIN_GPU_BATCH` is **10** — so
for any input of more than 10 records the floor is never evaluated. Verified
2026-09-27: 3 × 300 nt (1.35e5 cells, under the 2.5e5 floor) routes `3 cpu`, and the
same input with `RNA_MIN_GPU_CELLS=1000` routes `3 gpu`. The floor is correct and
unreachable.

More importantly it is **the wrong shape for heterogeneous compute**. Both gates are
all-or-nothing *per chunk*: they decide that a whole chunk goes to the device or a
whole chunk goes to the host. True heterogeneous compute needs a **per-record
decision made continuously against both queues' occupancy** — send this record
wherever it will finish first, given what each side is already carrying. That is a
scheduler, and neither a record-count threshold nor a cell threshold can express it,
because both are properties of the *input* rather than of the *machine's current
state*.

The mechanism that already routes per record is option C (`RNA_CPU_SLICE`,
`RNA_CPU_SLICE_CAP`, needs `--jobs > 1`), which holds records back from a chunk and
folds them on the host pool while the device works. It is off by default and its
slice size is a fixed fraction rather than a measurement.

**So the replacement is one thing, not three:** a router that owns both queues, with
the gates reduced to what they honestly are — a device-present check, and a
"would this record be faster on a core?" estimate fed by measured throughput on
*this* host. `tests/gpu_crossover.sh` already establishes that `Σ L²` predicts the
crossover and `total_nt` does not, which is the input that estimate needs. This is
the natural consumer of the tier-1 calibration in §3.1, and it subsumes gates 3
and 4 rather than tuning them.

### 11.4 Sequencing

This does not block the kernel work, but it does block *shipping*. Order:

1. §2.1 + gate 3 (configure autodetect, delete the presence test) — the two changes
   that turn "expert-only" into "works".
2. Criterion 5 and 6 (`make dist`, a no-CUDA build) — these protect every existing
   user and are pure regression tests.
3. The per-record router, replacing gate 4's floor. Needs the §C result from the
   stress notebook first: if the host pool is only serviced after the device drains,
   a router has nothing to schedule against until that is fixed.
4. §2.5 knob classification, and an `INSTALL` section.

## 12. A100 stress results (2026-09-27), and what they overturn

Run on an A100-SXM4-40GB, 12 cores, 83 GB host, at commit `069981ff`. Sections A and
D ran; **B (the trickle) and C (CPU/GPU concurrency) were never executed**, so nothing
below says anything about either.

### 12.1 Continuous flow adds NO build overlap. Falsified.

| shape | mode | wall | build | overlapped | hidden |
|---|---|---|---|---|---|
| 400x5601 | batch | 73.51 | 20.74 | 18.43 | 89 % |
| 400x5601 | flow | 75.67 | 20.44 | 18.13 | 89 % |
| 400x5601 | slot2 | 88.61 | 20.54 | 18.13 | 88 % |
| 1000x1000 | batch | 5.08 | 0.99 | 0.00 | 0 % |
| 40x8000 | batch | 22.53 | 4.92 | 2.32 | 47 % |
| 10000x500 | batch | 13.82 | 2.71 | 1.18 | 43 % |
| 10000x500 | slot2 | 58.49 | 2.76 | 1.11 | 40 % |

`overlapped` is **flat to within 2 % across all three modes at every shape**. This was
the stated falsification condition: continuous flow does not create extra overlap
opportunity, and the build hides at chunk boundaries however the sweep is scheduled.
So chunk COUNT is the lever on build overlap, not flow.

Slot flow is also a straight loss everywhere, and catastrophically so at 10000x500 --
**58.49 s against 13.82 s, 4.2x** -- on top of forcing int32. Every sha matched across
all twelve runs, so this is a performance verdict, not a correctness one.

### 12.2 The admission cap PAYS at production. My recorded risk did not materialise.

| shape | cap | wall | chunks | build | overlapped |
|---|---|---|---|---|---|
| 400x5601 | off | 81.96 | 2 | 18.85 | 3.23 |
| 400x5601 | default | 75.60 | 9 | 20.37 | 18.05 |
| 400x5601 | 2x default | **73.76** | 5 | 19.85 | 15.32 |

§7 and §11.3 of this document both warned that a fixed cap would **over-split** inputs
larger than its fixture, and pointed at 200x5601 where 2 chunks beat 7. That is wrong
at 400 records, and the mechanism is the opposite of what was assumed: with the cap
off there are 2 chunks and only **3.23 s of a 18.85 s build hides**, while 9 chunks
hide **18.05 s**. More chunks buy more overlap, and the overlap is worth more than the
per-chunk cost. `off` is the **worst** arm.

It also corrects the older figure this project has been quoting. The build was recorded
as hiding 4.96 s of 9.65 s (51 %); at production with the cap it now hides **89 %**.
The build is close to free, which retires it as a target.

2x the default (5 chunks) beats the default (9 chunks) by 1.8 s, so the optimum is
between them and the cap is slightly too aggressive -- a tuning question, not a design
one. Nothing here justifies removing it.

### 12.3 What the CPU cores can be worth, from these walls

Luke's reading of the run: *"the only cases where the CPU cores would be useful are
when the batch is so large that the time it takes to fold a seq on a core is small in
comparison to the overall time of the whole run."* The arithmetic supports it, and
sharpens it into a test the router can apply.

Device throughput from the measured walls, and `t_core` for ONE record on ONE core at
6.7e5 cells/s (**measured on the laptop, not on the A100 host** -- that host's cores
are faster, so these are a lower bound on CPU usefulness):

| shape | device cells/s | t_core | t_core / wall | all 12 cores as % of device |
|---|---|---|---|---|
| 400x5601 | 8.54e7 | 23.4 s | 0.32 | 9.4 % |
| 1000x1000 | 9.85e7 | 0.75 s | 0.15 | 8.2 % |
| 40x8000 | 5.68e7 | 47.8 s | **2.12** | 14.1 % |
| 10000x500 | 9.06e7 | 0.19 s | **0.014** | 8.9 % |

`t_core / wall` is the straggler cost: hand one record to a core and that fraction of
the wall is what you risk if it finishes last. At **40x8000 it exceeds 1** -- a single
record takes longer on a core than the entire GPU run takes -- so offloading even one
record there can only make things worse, however many cores are idle. At **10000x500
it is 1.4 %**, which is the shape where offload is safe.

So the condition is not "is the host fast" but **`t_core(L) << remaining device time`**,
and since `t_core` grows as L^3 while the wall grows only with the total, it is
governed by RECORD LENGTH far more than by core count. The ceiling is ~8-14 % either
way, which also says the cores are a modest win and never a rescue.

**This is a bound, not a measurement.** §C was written to measure whether the two
routes overlap in time at all and it did not run; until it does, "the cores are worth
up to 9 %" is arithmetic, and whether option C realises any of it is unknown.
