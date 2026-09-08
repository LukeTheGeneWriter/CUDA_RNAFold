# Step 4 — configuration, and why it should *not* go through `vrna_md_t`

*Written 2026-09-08. Scopes step 4 of `MERGING.md` §10, which says: "Route
configuration through `vrna_md_t` instead of `getenv()` (§6), keeping the env
vars as an optional debug override if upstream will have them."*

**Recommendation up front: keep the goal, reject the mechanism.** The
environment surface really is the largest reviewer-visible debt left. But
`vrna_md_t` is the wrong home for it on three independent grounds, and the seam
we already have provides a better one at no cost to the upstream diff.

---

## 1. The surface, counted rather than remembered

§6 says 22 environment variables. It is **24**, in four groups that want
completely different answers:

| group | n | variables |
|---|---|---|
| **Routing / policy** | 3 | `RNA_GPU_CHUNK`, `RNA_GPU_VRAM_BUDGET_MB`, `RNA_MIN_GPU_BATCH` |
| **Mode gates** | 8 | `RNA_FML_INT16`, `RNA_SLOT_FLOW`, `RNA_CONTINUOUS_FLOW`, `RNA_GPU_SWEEP`, `RNA_CUDA_GRAPH`, `RNA_GPU_UNIFORM_CHUNKS`, `RNA_SLOT_CAPACITY`, `RNA_SLOT_TURNOVER` |
| **Autotuning** | 11 | `RNA_MD_TILE`, `RNA_FML_SCAN_THREADS`, `RNA_BACKTRACK_THREADS`, and eight `*_BLOCK_SIZE` |
| **Verification** | 2 | `RNA_ROW_VERIFY`, `RNA_HC_VERIFY` |

Treating these as one problem is what makes step 4 look big. **Only the first
three are configuration in any sense a library user would recognise.**

## 2. Three reasons `vrna_md_t` is the wrong home

1. **It is the wrong struct.** `vrna_md_t` describes the *energy model* — what
   to compute: temperature, dangles, `noLP`, salt, `max_bp_span`. Block sizes
   and VRAM budgets describe *how to schedule* it. Putting a CUDA tile width in
   the struct that also carries `noGUclosure` is a category error, and it is the
   kind a maintainer notices immediately.
2. **It breaks the property we just bought.** `port27` touches nine upstream
   files and deletes **seven lines** (`MERGING.md` §0). Adding fields to
   `vrna_md_t` changes a public struct every consumer embeds by value — an ABI
   break for all of ViennaRNA, in exchange for settings that only matter when
   CUDA is compiled in.
3. **It promises a scope the implementation cannot honour.** `vrna_md_t` is
   per fold compound. **Six of these knobs are read once into a process-wide
   `static int v = -1` and cached for the life of the process** — and not
   incidentally: `rnafold_gpu_sweep()` records that "CUDA graph captures a
   different node topology in each mode, and a mode that changed per row would
   force a reinstantiate every row." A per-compound field that silently only
   takes effect on the first call is worse than an environment variable,
   because it looks like it works.

## 3. What to do instead — three scopes, one already built

**Per fold compound → the seam's own `void *data`.** `vrna_gr_set_inside_engine()`
already carries an arbitrary user data pointer per compound, and
`vrna_cuda_attach(fc)` already binds it. A CUDA-owned `vrna_cuda_opts_t` hung
there costs **zero** upstream lines and cannot break anyone's ABI. This is where
the routing/policy group belongs.

**Process-wide → an explicit setter, honestly named.** The six cached knobs are
genuinely process-scoped. They should say so:
`vrna_cuda_set_mode(...)`, callable before the first fold and documented as
taking effect once. This is not a limitation to hide; it is the CUDA graph's
actual constraint.

**Autotuning and verification → stay environment-only, and say so in the docs.**
Eleven block sizes and two verifiers are a *testing* surface. They are already
auto-tuned at runtime (`rnafold_choose_block_size()`); the env vars exist to
override an autotuner while debugging it. Promoting them to API would freeze
eleven implementation details into a public contract — and `RNA_MD_TILE` is a
live example of why that is dangerous: a value below 16 silently defeats int16
by dropping requests under a 32-byte sector, and nothing warns.

**Net effect on the upstream diff: zero new lines in upstream files.** That is
the argument to make to a maintainer, and it is only available because the seam
already exists.

## 4. What this costs, and the one thing it must not break

- ~150 lines of new CUDA-owned code (an opts struct, two setters, a resolver
  that reads env as the *fallback* rather than the source).
- `RNAfold.c`'s driver switches from `getenv()` to the setters. That is where
  the env vars keep working, for us and for the bars.
- **Every refusal must move with the configuration.** Three pairings are
  refused at init by reading the environment — `noLP`+`RNA_FML_INT16`,
  `RNA_FML_INT16`+`RNA_SLOT_FLOW`, `noLP`+`RNA_ROW_VERIFY`. If configuration
  can also arrive through an API, those checks must consult the *resolved*
  settings, not `getenv()`, or a pairing refused on the command line becomes
  reachable through the library. That is the same shape as the `-C` hole: a
  guard that inspects the wrong thing accepts what it means to decline.
- **The combination audit is env-shaped today.** `verify_option_matrix.sh`
  sets environment variables. If the API becomes a second route to the same
  settings, the matrix has to cover both, or it proves half of what it claims.

## 5. Not in scope, deliberately

`RNA_GPU_CHUNK` currently gates whether `RNAfold` uses the GPU **at all**
(`RNAfold.c:1299`: `gpu_enabled = (vrna_cuda_devices() > 0) && e && e[0]`).
That looks like an adoption blocker and is not one: a *library* caller already
enables the device with `vrna_cuda_attach(fc)`, no environment needed. What is
missing is a CLI flag rather than an API — `--cuda`, replacing an environment
variable no user would guess. Worth doing, but it is a driver change, not step 4.

Python bindings (§10 step 6) remain the real adoption blocker, and nothing here
changes that.
