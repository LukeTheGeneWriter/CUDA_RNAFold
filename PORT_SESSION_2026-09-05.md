# Session findings, 2026-09-05

*From "the port is scoped" to "RNAfold folds on the GPU on ViennaRNA 2.7.2,
byte-identical, with a green release bar". 40 commits on `port27`. Next step is
the Colab benchmark — there is still **no timing on 2.7.2 at all**.*

---

## Where it ended

`tools/verify_release.sh`, from a fresh clone:

| section | result |
|---|---|
| build without `--enable-cuda` | `make check` **130/130** |
| build with `--enable-cuda` | `make check` **137/137** |
| reference set via GPU chunks | **6/6 byte-identical**, incl. `F_extreme` (8001 nt, 42 s) |
| option-surface parity | **18/18**, GPU route vs CPU route |
| relaxed guards genuinely on GPU | `uniq_ML`, `noGU`, 176 records each |

Diff against upstream `master` (= the `v2.7.2` tag): **12 files modified, 6
deletions**, five of which are list continuations or a declaration widening. The
one substantive deletion in the library:

```c
-    energy = fill_arrays(fc, ms_dat);
```

replaced by five lines that consult an optional backend and fall through to that
same call when there is none.

---

## What was built

**The seam, and then the zip.** `vrna_gr_set_inside_engine()` (per fold
compound, on the existing `aux_grammar`, declining is part of the contract) and
`vrna_mfe_batch()` / `vrna_mfe_batch_backend_set()`. `vrna_mfe_batch()`'s
no-backend implementation is literally a loop over `vrna_mfe()`, so it is useful
to upstream with no accelerator present. **The driver contains exactly one
CUDA-specific line** — the registration call.

**Parity by routing, not by reimplementing.** Anything the device cannot
reproduce folds on upstream's per-record path. G-quads, circular RNA, salt,
constraints and modified bases all produce stock 2.7.2 answers today. That is
what makes this "an accelerator for validated CPU code" rather than a fork.

**The port itself.** All 4382 lines of CUDA compile and run on 2.7.2. The whole
kernel port reduced to `hc->matrix` → `hc->mx` — *not* a rename: dense
`n*i+j` where 2.3.0 was triangular, while `ptype` stayed triangular, so the
loops that walked both with one index had to be restructured. Verified by
`RNA_ROW_VERIFY`: **161 695 cells, zero disagreement**. `mfe_cuda.c` lost 1070
lines of 2.3.0 copies the seam made redundant.

**NDEBUG split (plan item I1).** Release builds define it, `--enable-asserts`
for the bar. Measured cost, corrected by the v2 ladder: **14.6 % on SRB, 17.6 %
on CFB** — not the flat 17.1 % previously recorded, which came from a run whose
summary mixed the branch delta into the assert cost.

---

## Bugs found in our own code

1. **`par_mfe()` was not re-entrant.** `init_gpu/2/3` size device buffers once;
   nothing reset them. Second chunk asserted. The 2.3.0 driver called
   `teardown_gpu/2/3()` at the end of *every chunk* — I had read those as
   end-of-run cleanup.
2. **`par_mfe()` cached energy parameters across calls.** Temperature 25 scored
   12/12 as the first call and **0/12 after a 37 °C batch**, silently. Invisible
   from RNAfold (model constant per run), reachable the instant anything folds
   two batches with different models — i.e. a `vrna_mfe_batch()` caller or a
   Python binding. Fixed by splitting allocate from upload.
3. **The library guard was laxer than it should be.** `uniq_ML` gives a correct
   MFE, so I relaxed it in both guards. But the sweep leaves `fM1` entirely INF,
   and `vrna_mfe_batch()` is public — a caller could fold with `uniq_ML` then
   call `vrna_subopt()`. Now declined in the library, allowed in the driver,
   with both sides documenting why they differ.

## Bugs found in upstream 2.7.2

Reproducers in `tests/upstream/`, all public-API only:

- `mfe/mfe.c:502` passes the **rule index** where auxiliary grammar callbacks
  expect `i` (the PF twin at `partfunc.c:410` does it correctly).
- The `SPEEDUP_PARAMS` cache is an unsynchronised race: **97.4 %** of parameter
  sets carry another thread's window settings under contention; 15.7 % of tables
  wrong when callers differ in model details. **0 %** when they agree — which
  corrects our earlier claim that threaded `RNAfold -j` produces wrong energies.
- Two build defects that stop a build from a fresh clone (`doc/doxygen`'s
  `noinst_DATA` outside its guard; `$(PYTHON3)` empty under `--without-python`).

---

## The lesson worth carrying: four checks passed for the wrong reason

This is the honest answer to "how do you know it's right?", and it is more
convincing than a list of green ticks.

1. **Empty `make check` totals.** A link failure printed no `TOTAL` line, the
   script grepped for one, and an empty line under a heading read as a pass.
   Hid two different link failures.
2. **`grep -c "mismatching"`** counted `"0 mismatching"` as a mismatch — a clean
   result scored as a failure. The inverse error, same cause.
3. **A GPU bar with no GPU.** `RNA_GPU_CHUNK=8` against `MIN_GPU_BATCH=10` sent
   every chunk to the CPU fallback; the bar reported IDENTICAL and looked like a
   passing GPU test. Only the timings rising 2 s → 12 s gave it away.
4. **`--logML` is not an RNAfold flag.** Both sides emitted zero bytes, `cmp`
   compared two empty files, and the parity table printed
   `logML identical [CPU route]` from the day it was written.

Each is now structurally prevented: totals-absent is a failure, only non-zero
mismatch counts count, the bar counts the sweep's own `sweep shape:` lines, and
both scripts refuse to score empty-vs-empty.

Two more in the same family from the other direction — my own tooling
manufacturing **false failures**: a parity run reading `RNAfold` while a rebuild
overwrote it (`exit 126`), and a scoping probe that measured the parameter-cache
bug rather than the models it claimed to test.

---

## Open, in priority order

1. **Colab benchmark — the biggest gap.** There is no timing on 2.7.2. The pitch
   is "this accelerates your code" and the only speed data we own is from the
   2.3.0 fork. Needs the L4; the 4 GB local card cannot produce a credible
   number, and the two local trees are not flag-comparable (`-g -O2` versus
   conda's flags).
2. **`PORT_UPSTREAM_PROPOSAL.md` needs a pass.** It still describes the seam as
   proposed rather than built and measured.
3. **Tier 1 accelerations** — `PORT_TIER1_PLAN.md`: `uniq_ML` (fill `fM1`) →
   salt → circular, in order of risk.
4. `logML` needs a library-level bar before its guard can come off; there is no
   CLI flag to test it with.
5. `noLP` genuinely does not work (9/12) despite `ptype` encoding it.

## What needs Luke rather than me

- Whether to send the five upstream defects now (they need nothing from us) or
  hold them until the proposal goes with them.
- What goes in the first contact. Recommendation: lead with the defects, then
  the seam proposal and the diff shape; hold the acceleration roadmap back as
  internal.
