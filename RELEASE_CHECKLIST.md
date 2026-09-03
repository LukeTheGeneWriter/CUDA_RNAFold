# Pre-release checklist

Things that must be settled before CUDA_RNAFold is released or submitted
upstream. Not a general TODO — items here are ones that are easy to forget and
expensive to discover late.

---

## ⚠️ R1 — Make `NDEBUG` the release default

**Status: OPEN. Measured worth: 17.1%.**

### The finding

`NDEBUG` is **never defined anywhere in this build system** — not in
`configure.ac`, not in `m4/ac_rna_features.m4`, not in `NVCC_FLAGS`. Every
`assert()` in the tree is therefore live in every build ever produced by this
project, **including every published timing number**.

Measured on Colab L4, 120 × 2000 nt (2026-09-03):

| build | wall |
|---|---|
| `823e363` stock | 7.49 s |
| `823e363` with `-DNDEBUG` | **6.21 s** |

**17.1% faster from a build flag.** That is larger than most individual
optimisations in this project's history, and it is available today.

The hot path is dense with them. Phase B (`e0761a9`) alone added
`assert(i_row < 0 || i == i_row)` to **eight kernels**, executed per thread. On
CUDA an assert is not merely a predictable branch: `__assertfail` in a kernel
affects register allocation and can inhibit optimisation.

### What NOT to do

**Do not just add `-DNDEBUG` globally.** These asserts have caught real bugs:

- `00d1e07` — `hp_mb_3p_kernel`'s `i==1` wrap reading past the end of `d_S2`
- the `unpack()` `out>=0 && out<=4` device assert, which is what surfaced the
  under-sized `d_S` in the mixed-length work
- the phase-A row-index assert, which trapped a wrong-schedule bug instead of
  silently folding wrong answers

Compiling them out unconditionally trades a measurable speedup for the loss of
the mechanism that has repeatedly turned silent corruption into a loud failure.

### The shape wanted

A **debug/release split**:

1. **Release (default): `NDEBUG` on**, asserts compiled out.
2. **`--enable-asserts`** (or `--enable-debug`) configure switch turns them back
   on. `-DNDEBUG` must reach *both* the C side (`CPPFLAGS`) and device
   compilation (`NVCC_FLAGS`) — nvcc applies it to host and device preprocessing.
3. **The verification bar always builds with asserts on.** Correctness is
   established in the debug build; performance is measured in the release build.
   `RNA_ROW_VERIFY` and the other self-checks must keep working under it.

### When

**During the ViennaRNA 2.7.2 port, not before.** It touches `configure.ac` and
`m4/ac_rna_features.m4`, which the port rewrites anyway (2.7.2 restructured
`src/ViennaRNA/Makefile.am` from 279 to 821 lines around ~24 convenience
libraries). Doing it twice is wasted work. See
`~/.claude/plans/serene-dazzling-gem.md` item **I1**.

### Caveat on the number

The 17.1% is the cost of **all** asserts in the build, not only the eight phase B
added. It was measured on the CFB tip; the equivalent measurement on the SRB tip
was still outstanding as of 2026-09-03 (it is the arm ladder notebook v2 adds).
Expect the release-build win to be of this order on both branches, but do not
quote 17.1% as branch-specific.

---

## Other items

*(none open)*
