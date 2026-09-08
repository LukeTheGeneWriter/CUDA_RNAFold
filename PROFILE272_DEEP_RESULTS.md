# Where the wall clock actually goes — and it is not the kernel

*Measured 2026-09-08, `NVIDIA L4 at 2040/2040 MHz, ratio 1.00, no throttle`,
commit `91fb763f`. Raw: `profile272_deep.json`.*

**The ~73 % share does not survive scale. At 120 × 5601 nt
`modular_decomposition` is 31 % of wall, and 61 % of wall is not measured by any
timer we have.**

---

## 1. The share collapses with size

| size | arm | wall | `modular_decomp` | share | chunks |
|---|---|---|---|---|---|
| 16 × 2000 | i32 | 2.6 s | 0.4 s | 16.7 % | 1 |
| 40 × 2000 | i32 | 5.4 s | 1.1 s | 19.5 % | 1 |
| 40 × 5601 | i32 | 68.5 s | 20.7 s | 30.2 % | 1 |
| **120 × 5601** | **i32** | **201.4 s** | **62.4 s** | **31.0 %** | 1 |

It rises with length and then flattens around 31 %. Nowhere near 73 %.

## 2. Amdahl holds — so we now understand the wall clock

| size | share | kernel | **Amdahl** | **measured** | fixed-chunk |
|---|---|---|---|---|---|
| 16 × 2000 | 16.7 % | 1.532× | 1.062× | 1.117× | 1.072× |
| 40 × 2000 | 19.5 % | 1.526× | 1.072× | 1.065× | 1.068× |
| 40 × 5601 | 30.2 % | 1.602× | 1.128× | 1.112× | 1.117× |
| **120 × 5601** | **31.0 %** | **1.610×** | **1.133×** | **1.117×** | 1.120× |

Prediction and measurement agree to ~1.5 %. **The model is right and the share
is the whole story.** int16's kernel win is real and reproducible (1.53–1.61×,
consistent with the 1.456× the first profile sampled), and it cannot move the
workload because the kernel is a third of it.

**The chunk-count confound turned out to be nil** — natural and forced-equal
chunk counts agree to 0.3 %. That is because every run here fit in **one chunk**;
the confound is real in principle and simply did not arise at these sizes. It
remains untested at v5's 400 × 5601, where int32 needed 5 chunks and int16 4.

## 3. The finding: 61 % of wall has no timer on it

At 120 × 5601:

| phase | int32 | share | int16 | share |
|---|---|---|---|---|
| `int_loop` | 0.29 s | 0.1 % | 0.29 s | 0.2 % |
| `hp_mb` | 10.96 s | 5.4 % | 13.91 s | 7.7 % |
| `load_my_c` | 1.93 s | 1.0 % | 1.97 s | 1.1 % |
| `modular_decomp` | 62.37 s | 31.0 % | 38.73 s | 21.5 % |
| `fetch_mx` | 2.48 s | 1.2 % | 1.73 s | 1.0 % |
| host-combine (all three) | 0.00 s | 0.0 % | 0.00 s | 0.0 % |
| **unaccounted** | **123.43 s** | **61.3 %** | **123.79 s** | **68.6 %** |

Two things stand out.

**It is 123 s in both arms.** Invariant to the encoding, to within 0.3 %. It is
not GPU work and not anything int16 touches.

**`int_loop` is 0.29 s — 0.1 %.** The kernel that once led the wall and was
"work-bound" is now noise. That is the batch-width work having succeeded, and it
is worth recording because the old mental model still says otherwise.

## 4. What this changes

**int16 is finished as an optimisation target at this scale.** Even an
*infinitely fast* `modular_decomposition` gives `1 / 0.69 = 1.45×` at 120 × 5601
— and 1.61× of it is already banked. The remaining headroom in that kernel is
under 10 % of wall.

**The 61 % is the only lever that matters now, and nothing measures it.**
Candidates, in the order worth instrumenting:

1. **backtracking** — host-side, already threaded (`RNA_BACKTRACK_THREADS`), and
   the one phase that scales with both record count and length;
2. **fold-compound construction** — 120 × 5601 nt of `vrna_fold_compound()`,
   including hard-constraint and ptype table setup;
3. **host matrix allocation** — the triangles are ~125 MB per record at 5601 nt;
4. chunk setup and teardown, output formatting.

An earlier characterisation put un-instrumented serial host work at **20–31 %**
of wall. It is now **61 %**, which is what happens when the accelerated part gets
faster and nobody re-measures the remainder.

**Next step is instrumentation, not optimisation.** Adding phase timers around
those four is a small change, and until it is done any work on the 61 % is
guessing at which quarter of it to attack.

## 5. And the T4 question is answered too

Bench v5's 1.009× needed no appeal to clocks after all. At 400 × 5601 the share
would be ~31 % or lower, so Amdahl caps the end-to-end win near 1.13× on *any*
machine; the T4's downclock plus more chunks explains the rest. **int16's
machine-dependence is real but was never the reason v5 saw nothing.**
