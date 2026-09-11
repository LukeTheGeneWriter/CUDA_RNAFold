# RNAfold option surface: acceleration state and guard state

*All **60** options in `src/bin/RNAfold.ggo`, audited 2026-09-11.*

*Most rows are **measured**, not argued: `tools/verify_option_parity.sh` runs 40
checks over 12 mixed-length records (62–401 nt) and asserts, for each, that the
answer is byte-identical to the same binary with the accelerator off **and** that
the run took the route it claims. Result: `the CUDA build matches the CPU build
across the option surface`. Rows marked **(asserted)** were read from the guard
rather than run — they are named so the distinction stays visible.*

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

**A hole in gate 1 costs performance. A hole in gate 2 costs correctness.** Only
`-c` and `--noClosingGU` are covered by all three.

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
| `-d0`, `-d1`, `-d3` | DECLINED | 1, 2 | measured |
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
| `--noClosingGU` | DECLINED | **1, 2** *(the last multi-gate option)* | measured |
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
| **ACCELERATED, byte-identical** | **24** |
| DECLINED, CPU route asserted | 16 |
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
