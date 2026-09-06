# Session findings, 2026-09-06

*Tier 1 acceleration: `-C`, `uniq_ML`, salt, circular. Two features landed, one
guard closed a live wrong-answer bug, and one feature is blocked on a single
`PRIVATE` in upstream's `mfe/mfe.c`.*

---

## Summary

| item | outcome |
|---|---|
| hard constraints (`-C`) | **guard closed** — the sweep gets them wrong; measured, not assumed |
| `uniq_ML` | **accelerated** — `fM1` reconstructed on the host, verified cell for cell |
| salt | **accelerated** — 28 frozen vectors from pristine 2.7.2, GPU == CPU at 7 concentrations |
| circular | **BLOCKED** on `postprocess_circular()` being private — needs a decision, not code |

`make check`: **142/142** with CUDA (was 137).

---

## 1. Hard constraints were a live wrong-answer bug, not a gap

`vrna_mfe_batch()` accepted a hard-constrained fold compound and folded it on
the device. Nothing in the guard fired: a `-C` dot-bracket constraint leaves
`hc->type` at `VRNA_HC_DEFAULT` and `hc->f` at `NULL`, which is all the guard
was testing.

Measured on 12 × 80 nt with a forced-unpaired block over positions 21–40:

```
CPU route : -10.00   and the structure re-evaluates to -10.00
GPU sweep : -14.30   and the structure re-evaluates to  -5.40
```

The matrix fill and the backtrack did not agree **with each other**. Both
obeyed the letter of the constraint (no `x` base was paired); the GPU simply
allowed pairs to span the constrained block where upstream does not.

### Three things this exposed about our instruments

- **`RNA_ROW_VERIFY` cannot see it.** It reports *"58128 cells checked, 0
  mismatching"* on exactly the folds that come out wrong, because it compares
  the fork's device sweep against the fork's *own host sweep* — and the host
  sweep is wrong the same way (it returns a nonsense single-pair structure at
  −7.80). A verifier comparing two implementations that share a defect is blind
  to it. Only upstream's `fill_arrays` is an oracle here.
- **Constraints are applied LAZILY.** `vrna_constraints_add()` only queues:
  `hc->mx`, `ptype` and all four `up_*` arrays stay byte-identical to an
  unconstrained compound until `vrna_fold_compound_prepare()` runs. The routing
  guard runs *before* `par_mfe()` prepares, so a guard that inspected `hc->mx`
  would compare two identical matrices and accept everything — the exact shape
  of the empty-vs-empty checks that have fooled this project before. The guard
  tests `hc->depot` instead, which is non-NULL from the moment a constraint is
  queued.
- **The first version of the bar passed for the wrong reason.** Its `|`
  (force-paired) shape was applied over bases the MFE already paired, so it
  changed nothing and compared two identical outputs. The bar now derives its
  constraints *from the free fold* — force unpaired exactly what the MFE pairs —
  so every shape is guaranteed to bite, and asserts that it did.

`tools/verify_constraint_parity.sh` runs in `--expect-routed` mode by default
and fails loudly in `--expect-accelerated`, so the guard cannot quietly stop
declining. It also carries a **self-consistency check** — does the energy a
route reports match the structure it returns? — which needed no reference
implementation and is what caught the defect in the first place.

`flush_gpu_chunk()` now applies constraints to the compounds it builds. That is
not dead code even though the driver bars `-C`: the chunk path builds its *own*
fold compounds, so without it a lifted bar would fold unconstrained and return a
plausible wrong answer.

---

## 2. `uniq_ML`: accelerated, and its old bar tested nothing

`fM1` is now reconstructed in `backtrack_one_slot()` from the `c` triangle it has
just fetched, using upstream's own per-cell helper (`vrna_mfe_multibranch_m1`)
and upstream's own loop order. **No device work at all** — the MFE recursion
never reads `fM1`, so it is a pure host post-pass, paid only when `uniq_ML` is
set.

`tests/mfe_cuda_fm1.ts` compares the matrix cell for cell against a compound
folded entirely by upstream, and separately asserts the matrix is not trivially
all-INF — because an all-INF `fM1` would satisfy a comparison against another
all-INF `fM1`, and that is precisely the bug being fixed.

**The old CLI bar was measuring nothing.** `verify_relaxed_options.sh` tested
`uniqML:-p`, but `-p` turns on the partition function and does not set
`md.uniq_ML`. The only RNAfold flag that does is `--ImFeelingLucky`
(`RNAfold.c:605`), which also enables stochastic backtracking and so cannot be
compared byte for byte between runs. There is no honest CLI bar for `uniq_ML`;
that arm has been removed and replaced by the library test. This is the same
family as the `--logML` arm that compared two empty files.

---

## 3. Salt: correct first try, and the hot kernel was untouched

Exactly as `PORT_SALT_SPEC.md` predicted. `modular_decomposition` — 73% of wall,
at its DRAM floor — needed **nothing**: `params.c:640-645` folds the salt terms
into `MLbase`/`MLclosing`/`MLintern` at init, so it inherits the correction
without knowing salt exists.

Two touch points, and they are asymmetric in an interesting way:

- **Internal loops need no plumbing.** An internal loop is bounded by `MAXLOOP`,
  so `backbones = nl+ns+2` tops out at `MAXLOOP+2`; a fixed 33-int table lives
  in `cuda_param_t`, which `Energy()` already receives.
- **Hairpins do.** A hairpin is *not* bounded, and upstream falls back to the
  closed form `vrna_salt_loop_int()` (which needs `exp`/`log`) above the table.
  Rather than port a transcendental into a kernel and then defend a
  double-precision match, the host evaluates upstream's own function up to the
  batch's longest record and uploads `length+2` ints.

All eight of `IntLoop_X`'s return paths take a correction, matching upstream's
eight. The stack case takes `SaltStack`; every loop case takes
`SaltLoop[backbones]`; they are not interchangeable. The hairpin correction is
added *before* the special-hairpin lookups and carried into each early return —
upstream's order, and getting it backwards would leave the
tetraloop/triloop/hexaloop cases uncorrected with nothing to notice.

Verified by `tools/verify_salt_parity.sh`: 28 frozen energies from pristine
2.7.2 reproduced, GPU == CPU at all 7 concentrations with the sweep counted, the
default salt provably identical to a run with no `--salt`, and all 6
non-default values confirmed to move the answer.

---

## 4. Circular is blocked on one `PRIVATE`, and this belongs in the TBI ask

The design in `PORT_CIRC_SPEC.md` holds — `fM2_real` is the modular
decomposition the kernel already computes and discards. That is not the problem.

The problem is the *post*-fill step. Our batch path goes
`vrna_mfe_batch()` → backend → `par_mfe()`, which never enters `vrna_mfe()`.
Everything it needs afterwards it has to call itself, and:

| step | status |
|---|---|
| `vrna_mfe_exterior_f5()` | **public** (`mfe/exterior.h`) |
| `vrna_backtrack_from_intervals()` | **public** (`backtrack/global.h`) |
| `vrna_mfe_multibranch_m2_fast()` | **public** (`mfe/multibranch.h`) |
| `vrna_mfe_multibranch_m1()` | **public** (`mfe/multibranch.h`) |
| `postprocess_circular()` | **PRIVATE — no header at all** (`mfe/mfe.c:103`) |

Four of the five are already public. The fifth is the one that turns filled
matrices into a circular energy and backtrack, and it cannot be reached.

**This is a better upstream ask than it looks.** It is the same argument as the
inside-engine seam: if a caller may replace the matrix fill, the steps that
consume those matrices have to be callable. Upstream has already made three of
them public; this is the gap in their own pattern, and anyone writing an
alternative inside engine hits it.

The options are (a) ask for it to be exposed, (b) copy ~150 lines of upstream
logic into the fork — which is exactly what "parity by routing, not
reimplementation" exists to prevent, and would drift on the next release, or
(c) route circular permanently. **No code was written for circular**, because if
the answer is (a) the shape of the work changes, and that is Luke's call and
TBI's, not one to force by copying.

---

## 5. A fifth check that lied: the bars default to a different tree

Three bars — `verify_gpu_cli.sh`, `verify_option_parity.sh`,
`verify_relaxed_options.sh` — default to `${1:-$HOME/port27cuda}`. Run bare,
they tested a build five commits and a day behind the one under test, and
reported green. `verify_release.sh` passes the tree explicitly, so the release
bar was never wrong; anyone running a single bar by hand was.

`verify_gpu_cli.sh` has a second, sharper default: `CHUNK=${2:-8}`, against
`MIN_GPU_BATCH = 10`. Run without a chunk argument, *every* chunk takes the CPU
fallback. The script does now say so out loud ("CPU fallback — budget too small
for a GPU batch"), which is the fix from the previous session working, but the
default still means the bare invocation exercises no GPU at all.

`tools/bar_preflight.sh` now fronts all three: it prints the binary, its build
time and the tree's HEAD, and **fails if the binary is older than any source
file under `src/`**. A bar that cannot have tested your change should say so
rather than pass.

## Guard status after this session

Accepted: default, temperature, `noGU`, **`uniq_ML`**, **salt**.
Declined: dangles ≠ 2, gquad, circ, `noLP`, `noGUclosure`, `logML`,
**hard constraints**, soft constraints, `hc->f`, domains, aux grammar rules,
windowed/restricted-span, non-default energy set, multistrand, comparative.

## Open

1. **Circular** — needs the API decision above before any code.
2. `noLP` still genuinely fails (9/12); cause undiagnosed.
3. `logML` still has no library-level bar.
4. `PORT_UPSTREAM_PROPOSAL.md` still describes the seam as proposed rather than
   built — and should now also carry the `postprocess_circular` ask.

---

# Benchmark v2 — mixed-length and multi-chunk (L4, commit `9804cbc2`)

All validity gates passed: identical `sha` across all four configs per workload,
the 1-chunk arm at 1 sweep, the N-chunk arm at 4/4/5, CPU arms at 0. The
per-workload budget calibration landed where the `~4*n^2` estimate predicted
(480 / 1882 / 235 MB).

| workload | A upstream | B port, off | C 1-chunk | D N-chunk | **A/C** | A/B | chunks | **D/C** |
|---|---|---|---|---|---|---|---|---|
| 120 x 2000 | 126.23 s | 129.37 s | 15.66 s | 65.75 s | **8.06x** | 0.98x | 4 | **4.20x** |
| 60 x 5601 | 578.44 s | 590.49 s | 102.47 s | 359.25 s | **5.65x** | 0.98x | 4 | **3.51x** |
| 150 mixed | 61.63 s | 63.53 s | 7.82 s | 18.24 s | **7.88x** | 0.97x | 5 | **2.33x** |

## Mixed-length is not a problem

7.88x on a shuffled, log-uniform 300-2500 nt workload, against 8.06x on uniform
2000 nt. The join mask does its job; nothing about a realistic FASTA costs the
accelerator anything meaningful. This is the arm v1 could not report at all.

## Chunk boundaries do not move the answer, at benchmark scale

All three workloads produce a byte-identical `sha` at 1 chunk and at 4-5 chunks.
Previously asserted only on small inputs by `verify_gpu_cli.sh`; now on 8001 nt
records and on mixed-length input.

## THE FINDING: chunking is expensive, and not for the reason assumed

The chunk penalty is **2.33x to 4.20x**, and it tracks the chunk count almost
exactly (4 chunks -> 4.20x, 4 -> 3.51x, 5 -> 2.33x). That is far too large to be
per-chunk `init_gpu`/`teardown_gpu` overhead, which was measured at ~1.6 s.

The mechanism is **loss of batch width**. The sweep is row-serial: a chunk costs
(max record length) rows regardless of how many records it holds, and the
records in a chunk are folded in parallel *within* each row. Splitting a uniform
workload into k chunks therefore multiplies total rows by k while dividing the
per-row parallelism by k. Wall time scales with the chunk count.

Two consequences:

1. **`gpu_bytes_per_file()` accuracy is a first-order performance property**, not
   just a correctness guard against OOM. Over-estimating VRAM per record costs
   chunks, and chunks cost multiples.
2. **This roughly doubles the case for int16.** `INT16_FML_SCOPE.md` justified it
   on the DRAM stream alone (~1.5x, from halving `fml_j`'s 46.7 TB). But `fml_j`
   is the *triangle* — the dominant per-record device buffer — so halving it also
   roughly doubles the records per chunk, which on a VRAM-bound workload is worth
   up to another ~2x by this table. The two effects are independent and multiply.

It also explains why v1 looked so good: on a 24 GB L4 every v1 workload fit in
one chunk, so v1 measured the accelerator at its best case and said nothing
about the regime a smaller card or a larger input lands in.

## An open discrepancy: A/B moved, and it should not have

| | v1 | v2 |
|---|---|---|
| 120 x 2000 | 0.994 | **0.976** |
| 60 x 5601 | 0.996 | **0.980** |

v2 shows the port's CPU path 2-3% slower than upstream's, consistently across
all three workloads, against a within-run spread of 0.27-0.33%. v1 showed
0.4-0.6%, inside its own spread.

**Our changed library files cannot reach that arm.** `engine.c`, `mfe_cuda.c`,
`hp_mb_loop.cu`, `int_loop.cu` and `interior_loopx.h` are all inside the
`if VRNA_AM_SWITCH_CUDA` block in `src/ViennaRNA/Makefile.am`, so none of them
compiles into a build without `--enable-cuda`. The only change reaching arm B is
`apply_constraints()` gaining one `int` parameter, on a call made once per
record — 120 calls in a 129-second run.

So this is most likely **between-VM variance that the reported spread does not
capture**: `spread` measures repeatability within one run on one machine, which
systematically understates the variance between two Colab sessions four hours
apart. Arm A moved too (+1.0% at W1).

Not resolved, and it should not be waved away — the honest statement for upstream
is "no measurable cost in v1; v2 measured 2-3% and we have not yet separated our
diff from machine variance". **The fix is methodological**: the current design
runs A's reps, then B's, then C's, then D's, so any thermal or noisy-neighbour
drift over ~20 minutes lands unevenly on the arms. Interleaving A/B/A/B would
cancel it, and is the change to make before quoting an A/B figure to anyone.
