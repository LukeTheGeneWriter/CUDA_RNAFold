# RNAfold option surface: acceleration state and guard state

*All **60** options in `src/bin/RNAfold.ggo`, audited 2026-09-11.*

***Gate 3 has ONE entry, and it is a different kind of entry.*** *The three
options that used to be backstopped -- `-g`, `-c`, `--noClosingGU` -- were all
retired by implementation on 2026-09-10/11. `fill_arrays.c` was empty for a few
hours, then gained a backstop on **dangle models 1 and 3** when 0 was
implemented (2026-09-11). Everything that left the list was missing arithmetic;
what is on it now needs missing DP **state**, which is why it is not simply the
next thing to write.*

*Most rows are **measured**, not argued: `tools/verify_option_parity.sh` runs 40
checks over 30 mixed-length records and asserts, for each, that the answer is
byte-identical to the same binary with the accelerator off **and** that the run
took the route it claims. Re-run 2026-09-11 against `7fa85d4e`: **all 40 green**,
`the CUDA build matches the CPU build across the option surface`, with
`noClosingGU` now in the ACCELERATED block at `1 sweeps, 30/30 records`. Rows
marked **(asserted)** were read from the guard rather than run — they are named
so the distinction stays visible.*

---

## There are THREE gates, and they are not the same gate

This was not written down before, and it changes how to read every "declined"
row.

| # | gate | where | what it is for |
|---|---|---|---|
| **1** | `gpu_path_usable()` | `src/bin/RNAfold.c` | **an optimisation.** Decides whether the driver registers the backend at all. |
| **2** | `vrna_cuda_engine_supports()` | `src/ViennaRNA/mfe/cuda/engine.c` | **the authority.** Consulted per fold compound; a decline falls back to upstream's own `vrna_mfe()`. |
| **3** | `VRNA_CUDA_BACKSTOP` | `src/ViennaRNA/mfe/cuda/fill_arrays.c` | **a tripwire.** `exit()`s if an unsupported model reaches the sweep — i.e. if 1 and 2 both have a hole. |

**Gate 1 is a strict subset of gate 2, and that is correct.** `RNAfold.c`'s
comment claims it "mirrors" gate 2; it does not, and it does not need to. Three
things gate 2 declines that gate 1 never checks:

| declined by gate 2 only | consequence |
|---|---|
| `--maxBPspan` < length | measured: routes to the CPU correctly |
| `logML` | no CLI flag exists, so unreachable from RNAfold |
| multistrand / comparative compounds | not constructible from RNAfold's input |

**A hole in gate 1 costs performance. A hole in gate 2 costs correctness.**
**Nothing is covered by all three any more** -- `-c` and `--noClosingGU` were the
last two and both shipped on 2026-09-11. Gate 3 is not obsolete; it is the reason
"unreachable" was enforced rather than assumed for a year. It simply has nothing
left to watch.

---

## The table

**ACCEL** — folds on the GPU, byte-identical to the CPU path ·
**DECLINED** — routes to the CPU, byte-identical ·
**NEUTRAL** — never reaches the recursion; the fold is still accelerated ·
**UNREACHABLE** — cannot be exercised from this CLI

| option | acceleration | guard | evidence |
|---|---|---|---|
| *(default)* | **ACCEL** | — | measured |
| `-T` / `--temp` | **ACCEL** | — | measured, 37 and 25 °C |
| `-d2` / `--dangles=2` | **ACCEL** | — | measured |
| **`-d0` / `--dangles=0`** | **ACCEL** *(new, 2026-09-11)* | — | measured, incl. int16/chunked/`-g`/`-c`/`-p`; `tests/mfe_cuda_dangles.ts` |
| `-d1`, `-d3` | DECLINED | **1, 2, 3** *(the only backstopped option)* | measured |
| `-p` / `--partfunc` | **ACCEL** *(MFE fill)* | — | measured, `-p` and `-p0` |
| `--MEA` | **ACCEL** *(MFE fill)* | — | measured |
| `--bppmThreshold` | **ACCEL** *(MFE fill)* | — | measured |
| `--betaScale` | **ACCEL** *(MFE fill)* | — | measured |
| `--pfScale` | **ACCEL** *(MFE fill)* | — | measured |
| `--noLP` | **ACCEL** | — | measured; `tests/mfe_cuda_nolp.ts` |
| `--noGU` | **ACCEL** | — | measured |
| `-4` / `--noTetra` | **ACCEL** | — | measured |
| `--salt` | **ACCEL** | — | measured, 0.2 and 1.5 M |
| **`-g` / `--gquad`** | **ACCEL** *(new, 2026-09-10)* | — | measured, incl. int16; `tests/mfe_cuda_gquad.ts` |
| **`--nsp`** | **ACCEL** *(new, 2026-09-10)* | — | measured, symmetric **and asymmetric**; `tests/mfe_cuda_nsp.ts` |
| `-P` / `--paramFile` | **ACCEL** | **none possible** | measured; 4 bars in `verify_paramfile_bars.sh` |
| `--helical-rise` | **ACCEL** | — | measured *(only bites under `--salt`)* |
| `--backbone-length` | **ACCEL** | — | measured *(only bites under `--salt`)* |
| `--maxBPspan` == length | **ACCEL** | — | measured |
| `--maxBPspan` < length | DECLINED | **2 only** | measured |
| `-j` / `--jobs` | **ACCEL** | — | measured |
| `--unordered` | **ACCEL** | — | measured **sorted** — see note |
| `--ImFeelingLucky` | **ACCEL** | — | route measured; **no byte bar possible** — see note |
| **`-c` / `--circ`** | **ACCEL** *(new, 2026-09-11)* | — | measured, incl. int16/`--noLP`/`--salt`; `tests/mfe_cuda_circ.ts` |
| **`--noClosingGU`** | **ACCEL** *(new, 2026-09-11)* | — | measured, incl. int16/chunked/`-c`/`-g`; `tests/mfe_cuda_noclosinggu.ts` |
| `--energyModel` | DECLINED | 1, 2 | measured |
| `-C` / `--constraint` | DECLINED | 1, 2 | measured |
| `--canonicalBPonly` | DECLINED | 1, 2 | measured |
| `--enforceConstraint` | DECLINED | 1, 2 | measured |
| `--shape` | DECLINED | 1, 2 | measured |
| `--shapeMethod` | DECLINED | 1, 2 | asserted *(same path as `--shape`)* |
| `--shapeConversion` | DECLINED | 1, 2 | asserted |
| `--sp-data` | DECLINED | 1, 2 | asserted |
| `--sp-strategy` | DECLINED | 1, 2 | asserted |
| `--sp-preprocess` | DECLINED | 1, 2 | asserted |
| `--motif` | DECLINED | 1, 2 | measured |
| `--commands` | DECLINED | 1, 2 | measured |
| `-m` / `--modifications` | DECLINED | 1 | measured |
| `--mod-file` | DECLINED | 1 | asserted *(requires `--modifications`)* |
| `--batch` | **UNREACHABLE** | — | measured: exits 1 on FASTA input |
| `-v`, `-i`, `-o`, `--noconv` | NEUTRAL | — | measured |
| `--auto-id`, `--id-prefix` | NEUTRAL | — | measured |
| `--id-delim`, `--id-digits`, `--id-start` | NEUTRAL | — | asserted |
| `--filename-delim`, `--filename-full` | NEUTRAL | — | asserted |
| `--log-level` | NEUTRAL | — | measured |
| `--log-file`, `--log-time`, `--log-call` | NEUTRAL | — | asserted |
| `--noPS` | NEUTRAL | — | measured *(used by every arm)* |
| `--noDP` | NEUTRAL | — | measured *(requires `-p`)* |
| `-t` / `--layout-type` | NEUTRAL | — | asserted |
| `--benchmark`, `--bm-*` | NEUTRAL | — | asserted *(post-hoc scoring)* |

### Two options a byte bar cannot judge, for different reasons

**`--unordered`** emits records in *completion* order by design, so a
byte-identity check reports a difference that is **the option working
correctly**. `check_sorted()` exists for it: same records, same answers, any
order. Measured identical when sorted.

**`--ImFeelingLucky`** uses stochastic backtracking, so **two CPU runs of the
same flag also differ.** It is accelerated and its route is asserted, but no
deterministic bar can judge its output. The reason this is written down rather
than just noted: "GPU ≠ CPU" here looked exactly like the `--nsp` defect until
the CPU was compared against itself.

---

## Summary

| | count |
|---|---|
| **ACCELERATED, byte-identical** | **26** |
| DECLINED, CPU route asserted | 14 |
| NEUTRAL | 19 |
| UNREACHABLE from this CLI | 1 |

**No silent wrong answer is known anywhere on the option surface** — and unlike
earlier versions of that claim, it now rests mostly on measurements rather than
on guard reading.

### What changed on 2026-09-10/11

- **`-g` moved DECLINED → ACCELERATED.** Three gates lifted; the last blocker was
  the *backtrack*, not the recursion (`VRNA-PATCH(bps-backtrack)`).
- **`--nsp` moved DECLINED → ACCELERATED.** `Energy()`'s missing 0 → 7 promotion
  and its index-swap-for-`rtype[]` were fixed.
- **`--backbone-length` and `--helical-rise` closed.** They only bite under
  `--salt`; the earlier "did not bite" was a missing `--salt`.
- **`-P`'s four bars all run.** Bar 4 found a live silent wrong answer (int16
  plus a large-`stack` file), now fixed.
- The harness went from 18 checks to 40 and gained `check_sorted()`.
- **`-c` moved DECLINED → ACCELERATED (2026-09-11).** It was never new
  arithmetic: `DMLi` **is** `fM2_real`, and the sweep discarded it one row
  later. It costs a **chunk width** of VRAM, counted in
  `modular_decomposition_bytes_per_file()`.
- **`-d0` moved DECLINED → ACCELERATED (2026-09-11), for ONE line.** d0 and d2
  share the recursion (`mfe_multibranch.c:686` dispatches `ml_pair_d0` or
  `ml_pair_d2`, both reading only `dmli1`), and — the part I got wrong first —
  upstream **zeroes `P->mismatchM` at d0** (`params.c:644`), so both multibranch
  sites were already correct. Measured: 0 nonzero `mismatchM` entries at d0
  against 175 at d2; red-teaming those two sites changes nothing. The single
  real gap was `vrna_mfe_gquad_internal_loop()`'s `mismatchI`, which is **not**
  zeroed (158 nonzero at both) — so `-d0` alone was already right and merely
  refused, and `-d0 -g` was the one combination that answered wrongly.
  **d1 and d3 stay declined**: `ml_pair_d1()` reads `dmli2` as well as `dmli1`,
  a second `DMLi` generation the sweep does not carry, and d3 adds coaxial
  stacking. That is new DP state, not new arithmetic — and it is why gate 3 is
  not empty any more.
- **`--noClosingGU` moved DECLINED → ACCELERATED (2026-09-11).** It was HALF implemented, which is worse than unimplemented: the
  hairpin/multibranch half was applied and the interior-loop half was not, so `c`
  was the minimum of *no* model rather than of a different one. The missing half
  is upstream's own `mfe_internal.c` skips, stack case exempted.
  **It also uncovered a year-old latent defect**: `fml_scan_kernel` guarded only
  ONE side of each of two `INF`-capable sums, so fML could carry `INF` minus a
  real energy. Invisible on int32 (such a value never wins a `min`), fatal under
  int16, where it became a block baseline and tripped `__trap()`. Both sums now
  use `fml_tadd()`. See `PORT_NOCLOSINGGU_SPEC.md`.

### Still honestly open

1. **`-P` cannot be guarded at all**, by construction: `vrna_params_load()`
   mutates library globals and leaves no flag for a guard to test. Its
   assumptions live in `load_param()` instead, and **four of seven ranked ones
   are still unchecked** against a non-default table (`lxc` narrowed to `float`,
   the dead mismatch/dangle tables, special-hairpin strides,
   one-parameter-set-per-batch).
2. **`--batch` is unreachable by this harness**, so it is untested — not passing.
3. **`--ImFeelingLucky` has no bar** and cannot have one of this kind.
4. **11 rows are asserted, not measured.** Each is either a pure I/O option or
   shares a code path with a measured sibling — but that is an argument, not a
   run.
5. **`-c` shipped on 2026-09-11**, so the list of genuinely missing capabilities
   is now empty for single-sequence linear-or-circular folding. What remains
   unsupported is *multistrand* and *comparative*, neither reachable from
   RNAfold's input, and the constraint family (`-C`, `--shape`, `--motif`,
   `--commands`, `-m`), which is declined by design rather than by gap.

### Re-running this audit

```
tools/verify_option_parity.sh <build-tree> <mixed-length.fa>
```

Mixed lengths matter: the `--noLP` + `RNA_SLOT_FLOW` defect was invisible to
uniform-length fixtures.
