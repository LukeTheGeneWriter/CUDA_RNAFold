# Port Phase 0 — ground truth for the ViennaRNA 2.7.2 port

*Run 2026-09-04 on the local WSL box (conda `base` toolchain, GTX 3050 4 GB).
This is the phase that had to pass before any CUDA code was touched: it
establishes what "unchanged" means for the rest of the port, and it retires the
one open correctness question the fork was carrying.*

Fork binary under test: `~/rnafold_build`, tree-identical to `Forward_port`
(`e8d7e0b`) modulo line endings — the 9712-insertion / 9712-deletion diff
against the branch tip is CRLF, not content.

---

## 0a — Pristine 2.7.2 builds and passes its own suite

`~/vrna27/ViennaRNA-2.7.2`, configured `--without-python --without-perl
--without-swig --without-doc --without-rnaxplorer --without-forester
--without-kinfold --without-rnalocmin`.

- The first attempt failed, and **not in ViennaRNA**: the bundled `g2-0.72`
  under `src/RNAforester` needs X11 development headers this machine does not
  have. `RNAlib` and `RNAfold` had already built. Excluding the three optional
  programs is sufficient; nothing on the MFE path is affected.
- **`make check`: 126 / 126 PASS**, 0 FAIL, 0 SKIP, 0 ERROR.

  The first run of the suite reported only 83 tests, because `configure` had
  emitted `WARNING: check not found -- will not build C-library Unit tests` and
  the suite skipped them **silently at the summary level** — an 83/83 green that
  was quietly missing every C unit test, `fold.ts` among them. Installing
  `check 0.15.2` and reconfiguring brings the other 43 in. **Anyone reproducing
  this must confirm the count is 126, not merely that the summary says PASS.**

## 0b — Pristine 2.7.2 reproduces the fork's reference outputs byte-for-byte

`RNAfold --noPS -i <input>` from the stock 2.7.2 CLI, compared with `cmp`
against the fork's own `ref_md` files:

| input | records | result |
|---|---|---|
| `C_mixed` `C_control` `asc` `desc` `extreme` `F_extreme` | 239 | **all IDENTICAL** |

This is strictly stronger than the earlier 239/239 result, which came from the
2.6.4 *python wheel* and compared **energies only**. This compares the bytes the
bar actually compares, from the actual target version.

**The byte-identical verification bar survives the port.** It stays the primary
instrument for Phases 2–4.

## 0c — The open −1717.70 / −1718.40 discrepancy: RESOLVED, and the fork was right

Standing question since 2026-08-27: `record 4: ENERGY MISMATCH ref -1717.70 vs
gpu -1718.40` on a 5601 nt Colab record, 4 of 5 matching.

**The exact input was recovered.** The Colab generator (cell 3 of
`CUDA_RNAFold_Testing_CPUGPU_Profiling`) had been seeded — `RNA_SEED =
20260827` — on the same day, so re-running it locally reproduces the same draws.
Confirmed by digest: regenerated input `sha256 1a24e738d11327472c00a58a44cd3824`
== the digest the notebook printed. These are the same bytes.

Folding those five records with pristine 2.7.2 as referee:

```
  record 0: 2.7.2 -1776.50   fork-GPU -1776.50
  record 1: 2.7.2 -1745.60   fork-GPU -1745.60
  record 2: 2.7.2 -1696.85   fork-GPU -1696.85
  record 3: 2.7.2 -1721.60   fork-GPU -1721.60
  record 4: 2.7.2 -1718.40   fork-GPU -1718.40   <-- the disputed record
```

**Pristine 2.7.2 agrees with the GPU, to the cent, on record 4.** The outlier
was the 2.6.4 pip wheel, which is exactly the oracle
[[feedback_self_comparison_beats_oracles]] warned against. There is no fork
defect here, and nothing to carry into the ported tree.

Corroborating negative result: 40 × 5601 nt of uniform-random sequence
(`C400.fa`) is byte-identical between pristine 2.7.2 and the fork GPU binary, so
this is not a length-class effect either.

## 0d — One real behavioural difference found: FASTA header handling

The only thing separating the two binaries on the record-4 workload — every
structure and every energy agreed — was the header line:

| input header | 2.3.0 base (the fork) | 2.7.2 |
|---|---|---|
| `> RNA 0 for testing` | `>RNA` | `> RNA 0 for testing` |
| `>plain_id` | `>plain_id` | `>plain_id` |
| `>id with spaces` | `>id` | `>id with spaces` |
| `>id\tafter_tab` | `>id` | `>id\tafter_tab` |

2.3.0 truncates the sequence ID at the first whitespace; **2.7.2 echoes the
header line verbatim.** The six `ref_md` inputs all use whitespace-free headers,
which is precisely why 0b came back identical and why this had never surfaced.

**Consequence for the port's bar:** after Phase 2 the ported binary will differ
from the fork's current output on any input with spaces in its headers. That is
an upstream-driven, correct change — but it must be declared in advance, or it
will read as a regression the first time someone folds a real FASTA file.

---

## Frozen reference set

Everything below is the baseline the rest of the port is measured against:

1. Pristine 2.7.2 at `~/vrna27/ViennaRNA-2.7.2`, `make check` = 126/126.
2. `~/rnatest/ref_md/*.out`, 239 records — unchanged, and now known to be
   reproducible from stock 2.7.2 as well as from the fork.
3. `~/phase0/rec5.fa` — the five recovered Colab records, seed 20260827,
   input `sha256 1a24e738…`, with the energies above.
4. Header handling is a **known, expected** post-port difference; the bar must
   compare against whitespace-free headers or account for it explicitly.

## Also confirmed in passing (feeds Phase 1)

The `SPEEDUP_PARAMS` race in `params/params.c` is real and worse than the notes
recorded. `p_pre` / `p_pre_init` / `pf_pre` / `pf_pre_init` (lines 102–106) are
file-scope statics with **no lock, no `omp critical`, no atomic anywhere in the
file**, and the `#pragma omp threadprivate` at line 111 covers only `id` and
`pf_id` — *not* the cache. `vrna_fold_compound()` reaches it unconditionally via
`vrna_params()` (`fold_compound.c:622`), so upstream's own threaded `RNAfold -j`
hits it. Two distinct hazards:

- `memcpy(&p_pre, cp, sizeof(vrna_param_t))` (line 170) races a concurrent
  `vrna_params_copy(&p_pre)` (line 163) — a torn parameter table, i.e. silently
  wrong energies.
- The cache-*hit* path **writes** into shared state before comparing
  (lines 158–160), so one thread's `window_size` / `min_loop_size` /
  `max_bp_span` can be returned inside another thread's parameters.

**Both hazards were subsequently MEASURED** (2026-09-05, probes in
`tests/upstream/`, written up in `PORT_UPSTREAM_PROPOSAL.md`), and the
measurement refines what is written above. The torn table is observable **only
when concurrent callers ask for different model details**: 15.7% of tables wrong
across four temperatures, but 0 of 160 000 wrong when every thread uses identical
model details — which is exactly what upstream's own `RNAfold -j` does, so the
claim that it produces silently wrong energies overstates the case. The
`window_size` hazard is the severe one: 97.4% of parameter sets carried another
thread's window settings, against 0% in the single-threaded control.

Timings, for context only (not a benchmark): 40 × 5601 nt took pristine 2.7.2
389.9 s at `-j8` (peak RSS 1.33 GB) versus the fork's 126.3 s single-queue
(2.12 GB) on a 4 GB 3050.
