# Salt corrections: GPU design, and a correction to my own cost estimate

*Written 2026-09-04 from ViennaRNA 2.7.2's implementation, verified by reading
the code rather than the docs. Reference vectors: `tests/salt/`.*

## Why it is worth having

Salt is not a rounding effect. On 8 × 300 nt, dropping from the 1.021 M default
to a physiological 0.1 M moves the MFE by roughly **20%**:

```
salt (M)    record 0   record 1   record 2   record 3
0.1          -54.20     -33.40     -28.22     -38.68
0.5          -64.78     -43.18     -38.21     -49.21
1.021        -68.10     -46.20     -40.90     -52.70   <- default, no correction
2.0          -70.56     -48.22     -42.82     -55.15
5.0          -73.04     -50.02     -44.62     -57.62
```

Anyone folding at realistic ionic strength needs this. It is also the *cheapest*
of the parity features **on the right tree** — see the cost correction below.

## What the GPU actually needs — three touch points, and one non-touch point

### The multibranch kernel needs NOTHING. Verified.

`params.c:640-645` folds the salt term into the parameter table at init:

```c
params->MLclosing += params->SaltMLbase;
params->MLclosing += params->SaltMLclosing;
params->MLbase    += params->SaltMLbase;
for (i = 0; i <= NBPAIRS; i++)
  params->MLintern[i] += params->SaltMLbase;
```

`modular_decomposition.cu` reads `MLbase`/`MLclosing`/`MLintern` and nothing else,
so it inherits the correction for free. **This is the single most important fact
in this document**: the hot kernel — 73% of wall, at its DRAM floor — is not
touched, so salt cannot cost measurable performance.

### 1. Hairpin (`hp_mb_loop.cu`)

`eval/hairpin.h:352-372`: `energy += salt_correction`, where

```c
salt_correction = (size <= MAXLOOP) ? P->SaltLoop[size + 1]
                                    : vrna_salt_loop_int(size + 1, salt, T, backbone_length);
```

### 2. Internal loops and bulges (`int_loop.cu`)

`eval/internal.h:640-650`: with `backbones = nl + ns + 2`,

```c
salt_loop_correction = (backbones <= MAXLOOP + 1) ? P->SaltLoop[backbones]
                                                  : vrna_salt_loop_int(backbones, ...);
```

Inside the DP, internal loops are bounded by `MAXLOOP`, so **the table always
suffices here** — no device-side closed form needed.

### 3. Stacked pairs (`int_loop.cu`)

`eval/internal.h:625`: `salt_stack_correction = P->SaltStack`, a single `int`
added for the stack case (`nl == 0 && ns == 0`).

### The data to move to the device is trivially small

```c
int SaltStack;                 /* 1 int  */
int SaltLoop[MAXLOOP + 2];     /* 32 ints */
```

33 integers, constant for the whole run. This is a textbook `__constant__`
memory candidate — and `__constant__` is still unused codebase-wide
(`project_gpu_memory_hierarchy_findings`), so this doubles as the first real
use of it.

**One design decision to make:** hairpin loops, unlike internal loops, *can*
exceed `MAXLOOP`, where upstream falls back to the closed form
`vrna_salt_loop_int()` (which needs `exp`/`log`). Rather than port that
transcendental into a kernel, **precompute the table to the batch's longest
sequence** (`n_max` ints, still kilobytes) and index it directly. The kernel
then has no branch and no math — just an add.

## The cost correction — and it inverts the sequencing

**I estimated salt as "small". That estimate was for the ported tree, and I
should have said so.** On the current 2.3.0 base it is not small, because
almost none of the supporting machinery exists here:

| piece | on 2.3.0 (now) | on 2.7.2 (after Phase 2) |
|---|---|---|
| `params/salt.c` maths (~150 lines) | write it | **already there** |
| 6 new `vrna_md_t` fields + defaults | invasive, changes a public struct | **already there** |
| 5 new `vrna_param_t` fields | write it | **already there** |
| salt logic in `get_scaled_params` | write it | **already there** |
| CPU eval paths (hairpin/internal/stack) | write it, or CPU and GPU disagree | **already there** |
| `--salt` CLI option | needs `gengetopt` (**not installed**) to regenerate `RNAfold_cmdl.c` | **already there** |
| **GPU kernels + upload** | **~50 lines — the only durable part** | **the same ~50 lines** |

Backporting now means roughly **600–800 lines that Phase 2 deletes**, plus a
real risk: any drift from upstream's exact arithmetic shows up as a byte
difference against the 2.7.2 oracle, and we would be debugging our
re-implementation rather than our kernels.

**Recommendation: do not backport salt. Salt is an argument for pulling Phase 2
forward, not for reimplementing 2.7.2 inside 2.3.0.** The design above is
complete and the bar is frozen, so the kernel work can start the day the rebase
lands.

## The bar, frozen now

`tests/salt/`:
- `salt_test.fa` — 4 records (60/120/200/300 nt), seed 20260904.
- `salt_reference.txt` — MFE from pristine 2.7.2 at 0.05, 0.1, 0.2, 0.5, 1.021,
  2.0 and 5.0 M. The implementation must reproduce these exactly.
- `salt_default.out` — and the invariant that matters most:
  **`--salt 1.021` is byte-identical to no `--salt` at all** (verified), so the
  feature is genuinely default-off and cannot move the standing bar.
