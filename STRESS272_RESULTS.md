# The stress profile: the residual is gone, and the wall is now a HOST problem

*400 × 5601 nt, **NVIDIA L4 at 2040/2040 MHz, ratio 1.00, no throttle**, 23 034 MiB,
commit `82fdbb07`. Six arms: {int32, int16} × {natural, half, quarter} VRAM budget.
Raw: `stress272.json`, notebook `CUDA_RNAFold_Stress272.ipynb`. Run 2026-09-09.*

> **LATEST: read §16 first.** Five runs on, the wall at 400 × 5601 is
> **394.8 s** (was 641.8 on 2026-09-09), int16 and the build pipeline
> **compose**, and the top open performance item is int16's **34.8 s give-back**
> in `hp_mb` (+27.1) and `fetch_mx` (+7.7). **§19 SETTLES THAT SPLIT AT SCALE
> AND IT WAS WRONG:** phase-synced, `hp_mb` is **+0.27 s**, not +27, and the
> real cost is **`int_loop` +30.44 s**. `int_loop` is **30 % of GPU time**,
> recorded for the life of this project as 0.8 %. §1–§14 are kept in run order.

**The three questions this was posed all came back NO, and a fourth answer
arrived unasked: `gpuinit` is 22.9 % of wall and nobody has ever looked at it.**

---

## 0. The headline

| | |
|---|---|
| **Residual** | **0.15 – 0.24 %** in every arm — down from 61 % (2026-09-08) and 28.8 % after the four counters were wired up |
| **`modular_decomposition`** | **31.1 %** of wall, against 31.0 % at 120 × 5601 — the share does not recover at scale |
| **Chunking** | 3 → 10 chunks costs **2.2 %** of wall, not ~3.3× |
| **New** | **`gpuinit` = 153.2 s = 22.9 %**, the second-largest item, unattributed |
| **int16** | **1.114×** end-to-end here (T4 saw 1.009×), kernel 1.608× |

**Host stages are 60.7 % of wall; every GPU phase together is 39.2 %.** The
remaining performance problem in this port is not the kernels.

---

## 1. Where the wall goes — i32, natural budget, 669.2 s

| item | s | share | kind |
|---|---|---|---|
| `modular_decomp` | 208.2 | **31.1 %** | GPU |
| **`gpuinit`** | **153.2** | **22.9 %** | host + driver |
| `output` | 124.8 | 18.6 % | host |
| `build` | 122.0 | 18.2 % | host |
| `hp_mb` | 38.4 | 5.7 % | GPU |
| `fetch_mx` | 8.3 | 1.2 % | transfer |
| `load_my_c` | 6.4 | 1.0 % | transfer |
| `backtrack` | 5.4 | 0.8 % | host |
| `int_loop` | 0.9 | **0.1 %** | GPU |
| prepare / teardown / free / prefill | 0.7 | 0.1 % | host |
| **unaccounted** | **0.98** | **0.15 %** | — |

**The 61 % is closed.** Wiring up the four dead counters (`b799a820`) plus
parsing the stage line accounts for the wall to within one second in 669. There
is no longer a mystery term, and any further work is now aimed at a named stage.

`output` here is honest: the notebook runs `--noPS` with **no `-j`**, so
`stage_output_s` is real wall time and not `thpool_add_work()` dispatch. That
caveat from 2026-09-08 does not apply to these numbers.

---

## 2. `gpuinit` — 22.9 %, and the breakdown was thrown away

`stage_gpuinit_s` covers `init_gpu` / `init_gpu2` / `init_gpu3`. It is
**153–160 s in all six arms**, barely moving with chunk count (153.2 → 156.0 →
159.9 across 3 → 5 → 10 chunks) or with int16. So it is dominated by a **fixed
cost**, not a per-chunk one.

**And the instrumentation to split it already exists and already ran.**
`mfe_cuda.c:363` prints a second line:

```
gpuinit breakdown (s): init_gpu=… init_gpu2=… init_gpu3=… || of which pack=… cudaMalloc=… other=…
```

built exactly for this question, with the two cross-cutting suspects named in
its own comment: the per-record O(n²) host bitmask packing loops, and
`cudaMalloc` (suspected because teardown frees ~20 GB per chunk in a reported
0.000 s, so the cost may simply be deferred into the next chunk's mallocs).

**The notebook does not parse it** — `init_gpu` appears zero times in
`CUDA_RNAFold_Stress272.ipynb`, and stderr is truncated to 2–3 kB in the saved
cells, so it is not recoverable from this run either.

This is the **same mistake as the deep profile**, which parsed the phase line and
discarded the stage line — that is what hid four dead counters and 30 % of the
wall for the life of the project. The generator needs one more regex, and then
the largest unexplained item in the profile is explained. **Do that before
optimising anything.**

---

## 3. The kernel share does not recover at scale

| size | wall | `modular_decomp` | share |
|---|---|---|---|
| 40 × 5601 | 68.5 s | 20.7 s | 30.2 % |
| 120 × 5601 | 201.4 s | 62.4 s | 31.0 % |
| **400 × 5601** | **669.2 s** | **208.2 s** | **31.1 %** |

Flat at ~31 % over a 10× range in record count. **`int16` is finished as an
optimisation target**, now confirmed at the size the port is actually sold on
rather than extrapolated to it.

`int_loop` is **0.9 s, 0.1 % of wall** — the kernel that once led and was called
"work-bound" is noise at every scale now measured.

---

## 4. int16 at scale: 1.114×, and it makes `hp_mb` slower

| arm | wall | `modular_decomp` | `hp_mb` |
|---|---|---|---|
| i32 natural | 669.2 | 208.2 | 38.4 |
| i16 natural | 600.5 | 129.5 | **47.2** |

Kernel **1.608×** — the deep profile predicted 1.610×, so that number is now
reproduced on a third size.

Plain Amdahl on the 31.1 % share predicts 1.133× end-to-end; measured is
**1.114×**, 1.7 % low. The gap is not noise, it is a **regression in `hp_mb`**:
+8.8 s under int16, reproduced at every budget (+8.8 / +8.5 / +8.3). Accounting
for it exactly closes the model:

```
669.2 − 78.7 (modular_decomp) + 8.8 (hp_mb) = 599.3 s   vs   600.5 measured   (0.2 %)
```

So **int16's true net kernel win is ~70 s, not ~79 s** — it gives back 11 % of
its own gain in the hairpin/multibranch kernel. That cost has never been
isolated before and is a live question: the narrowing in `load_fML_kernel` is
the obvious suspect, and it is attributed to `hp_mb`'s timer.

Machine dependence is confirmed rather than contradicted: 1.009× on the T4
(bench v5), **1.114× on the L4** here, both at 400 × 5601. It remains a
bad candidate for default-on.

---

## 5. Chunking does NOT cost k× at this size

| arm | chunks | wall | vs 3 chunks |
|---|---|---|---|
| i32 natural | 3 | 669.2 | — |
| i32 half | 5 | 674.4 | **+0.8 %** |
| i32 quarter | 10 | 684.0 | **+2.2 %** |
| i16 natural | 3 | 600.5 | — |
| i16 half | 4 | 602.5 | +0.3 % |
| i16 quarter | 8 | 609.2 | +1.5 % |

**3.3× the chunks costs 2.2 % of wall.** This directly contradicts the recorded
rule that *"chunking costs ~k× wall — it is loss of BATCH WIDTH"* (bench v2,
2026-09-06). That rule was measured at sizes where one chunk's worth of records
could not saturate the device. At 400 × 5601 a single chunk already does, so
splitting the batch costs only per-chunk overhead.

The delta lands exactly where per-chunk overhead should (i32, 3 → 10 chunks):

| | 3 chunks | 10 chunks | Δ |
|---|---|---|---|
| `fetch_mx` | 8.3 | 13.9 | **+5.6** |
| `gpuinit` | 153.2 | 159.9 | **+6.7** |
| `output` | 124.8 | 127.6 | +2.9 |
| `int_loop` | 0.9 | 1.5 | +0.5 |
| `modular_decomp` | 208.2 | 207.1 | −1.0 (flat) |
| `build` | 122.0 | 121.5 | flat |

Sum ≈ +14.8 s ≈ the +14.8 s of wall. Nothing is unexplained.

**Consequences.** Part of int16's case was that it cuts chunk counts; at this
size that is worth ~1 % and is not a reason to enable it. And `gpu_bytes_per_file()`
is much less of a performance property here than
[[chunking costs batch width]] recorded — **VRAM budget is close to free to
lower at 5601 nt**, which is useful for fitting on smaller cards.

---

## 6. int16 costs HOST memory, at every budget

| budget | i32 chunks / RSS | i16 chunks / RSS |
|---|---|---|
| natural | 3 / **8.36 GB** | 3 / **11.47 GB** |
| half | 5 / 5.67 GB | 4 / 8.36 GB |
| quarter | 10 / 3.62 GB | 8 / 6.81 GB |

int16 is recorded as **48.3 % less VRAM per record**. That is a *device* saving;
the host goes the other way — **+37 % RSS at the natural budget and +88 % at the
quarter budget**, where int32 runs in 3.62 GB and int16 needs 6.81 GB.

Part of this is mechanical (a smaller device footprint admits more records per
chunk, so more host matrices are live at once — int16 needs 8 chunks where int32
needs 10). But the natural-budget pair has the **same 3 chunks** and still
differs by 3.11 GB, so that cannot be the whole story. **Open question**, and it
matters: if you shrink the VRAM budget to control host RAM, int16 gives much of
that back. On a host-RAM-limited box int16 is the wrong trade.

---

## 7. What this changes

1. **Stop optimising kernels.** Host stages are 60.7 % of wall against 39.2 % for
   every GPU phase combined. `modular_decomposition` is at its DRAM floor and
   worth 31 %; `gpuinit` + `output` + `build` are worth 59.8 % and none of the
   three has ever been profiled.
2. **`gpuinit` first, and it is nearly free to start** — the breakdown line
   already exists in the binary. Add the regex, re-run one arm, and 22.9 % of
   wall stops being a single number. The comment at `mfe_cuda.c:294` already
   names the two suspects.
3. **`build` (122 s) and `output` (125 s) are next**, and both have a known
   thread-pool route (`RNA_BUILD_THREADS`, and `-j` for output) that is
   *defaulted off* and has never been measured at this size.
4. **Retire the ~k× chunking rule** at large sizes; it holds where batch width
   binds and not here.
5. **int16 stays gated off.** 1.114× on a good card, 1.009× on a T4, a +8.8 s
   `hp_mb` regression, and a large host-RAM cost.

## 8. Caveats

- One run per arm; no repeats, so small differences (the 0.3–0.8 % chunk steps)
  are not separated from run-to-run noise. The 2.2 % at 10 chunks is larger than
  the spread between arms that should be identical, so it is real, but it should
  not be quoted to two significant figures.
- `clock_before` and `clock_after` are 1.00 in every arm, so no throttling.
- All six arms fold the same 400 records and report the same 6 266 401 200 cells
  and 400 GPU records, so no arm silently fell back to the CPU — the failure that
  made int16 look like a regression in bench v3.

---

# Follow-up, same day: `gpuinit` attributed, and the "obvious fix" is WRONG

*Measured locally on an RTX 3050 (4 GB), the build stood up this session. Small
sizes, so read the RATIOS, not the seconds.*

## 9. `gpuinit` is host bitmask packing, not `cudaMalloc`

The breakdown line §2 asked for, read directly off the local binary:

| input | `gpuinit` | `pack` | `cudaMalloc` | other |
|---|---|---|---|---|
| 24 × 1000 | 0.077 | 0.050 (65 %) | 0.004 | 0.023 |
| 24 × 2000 | 0.219 | 0.181 (83 %) | 0.006 | 0.032 |
| 48 × 2000 | 0.439 | 0.369 (84 %) | 0.011 | 0.059 |
| 24 × 3000 | 0.599 | 0.521 (**87 %**) | 0.008 | 0.070 |

**`cudaMalloc` is 1–5 % and falling.** The deferred-free hypothesis at
`mfe_cuda.c:294` — that teardown frees ~20 GB in a reported 0.000 s so the cost
lands in the next chunk's mallocs — is **wrong, and can be struck out.** It is
the O(n²)-per-record host packing loops, in `init_gpu3` and `init_gpu2`, and it
scales as records × n² (2× records → 2.04×; 2× length → 3.6×).

## 10. The replacement exists, was written for exactly this, and is dead code

`pack_hc_kernel` (`hp_mb_loop.cu`) derives all five bitmasks from the sequence on
the device. It is gated on `g_hc_seq_derived`, whose declaration
(`stub2.h:439`) says **"Set once by RNAfold.c"** and cites this very cost:
*"that packing measured 197.4 s of a 769 s Colab run, 25.7 % of wall."*

**Nothing sets it.** The only assignment in the tree is the initialiser
`int g_hc_seq_derived = 0;` at `mfe_cuda.c:308`. On the 2.3.0 branch
(`0b4bcf3e`) `RNAfold.c` set it; **the 2.7.2 port dropped that line** and kept
everything else — the kernel, both gated branches, and `RNA_HC_VERIFY`.

So it reads exactly like a port regression with a one-line fix. **It is not.**

## 11. Enabling it is a SILENT WRONG ANSWER on 2.7.2

Flipped the default, rebuilt, verified, measured, restored:

| | result |
|---|---|
| `gpuinit` | 0.113 → **0.027 s**, a **4.2×** cut; `pack` 0.083 → 0.005 |
| `RNA_HC_VERIFY` | **MISMATCHES in `hccc_mb`** |
| fold vs host-packed GPU | **differs on 9 of 10 records** |
| fold vs CPU route | **differs on 9 of 10 records** |

The win is real and large. The answer is wrong.

**So the missing line is load-bearing, not a regression.** Restoring it on the
strength of the 2.3.0 commit message — which is what the evidence in §10 invites,
and what I was about to do — would have put a silent wrong answer into the
default path on nine records in ten. The port dropping that assignment is the
only reason the answers are right today.

### The lead

Every sampled mismatch has the **host allowing MB where the device does not**
(`gpu=09089c28` / `host=09289c28`, and so on — the host word always carries the
extra bit). A too-restrictive multibranch mask forbids some closings, which is
consistent with a suboptimal-but-self-consistent fold, the failure shape this
project has now seen four times.

`rnafold_hc_opt()` replicates **2.3.0's `hc_reset_to_default()` SINGLE case**.
2.7.2 splits that into `default_pair_constraint()` (`hard.c:761`) plus a reset
that writes `hc->mx[n*i+i] = ALL_LOOPS` on the diagonal (`hard.c:926`), and
`ALL_LOOPS` includes the `_ENC` bits the device's `i == j` case omits. The
pair-predicate halves do match — that was checked line by line while scoping
`--nsp` — so the divergence is in the surrounding reset, not in the pair rule.
The host side of `RNA_HC_VERIFY` uses `mx[n*i + j]`, which matches upstream's own
indexing at `hard.c:926/966`, so the oracle is sound and it is the device that is
wrong.

### What to do

1. **Leave `g_hc_seq_derived` at 0.** Add a comment at `mfe_cuda.c:308` saying
   the flag is *known wrong* on 2.7.2, so the next reader does not restore the
   line. Right now `stub2.h:439` actively invites them to.
2. Close the `hccc_mb` divergence against `RNA_HC_VERIFY`, which already gives
   word-level resolution and needs no new instrumentation.
3. Only then wire the setter — and per `PORT_CONFIG_SCOPE.md` it should be an
   honest setter on the seam, not a global poked from `RNAfold.c`.

**This is worth ~20 % of wall at 400 × 5601** (87 % of `gpuinit`'s 22.9 %), which
makes it the largest single win available anywhere in the port — bigger than
everything int16 can offer, and on the host side where the wall now lives.

---

# 12. CLOSED: two divergences, both fixed, the lever is live

*Same day. Local RTX 3050, 12 records mixed 350-2400 nt.*

`RNA_HC_VERIFY` located both, and neither was in the pair predicate.

**Divergence 1 — the diagonal dropped the `_ENC` bits.** `default_hc_up()`
(`hard.c:926`) writes `VRNA_CONSTRAINT_CONTEXT_ALL_LOOPS` at `mx[n*i+i]`.
`rnafold_hc_opt()` open-coded that as `EXT|HP|INT|MB`, which is
`CLOSING_LOOPS` only and drops `INT_LOOP_ENC` and `MB_LOOP_ENC`
(`hard.h:324-333`). Nothing reads the diagonal of the `_ENC` masks, which is
exactly why an open-coded copy of a **named upstream constant** could sit there
being wrong. Now uses the constant, so it cannot drift again.

**Divergence 2 — `max_bp_span` is per record and was passed as a batch scalar.**
`vrna_fold_compound()` sets `md->window_size = fc->length` and then
`md->max_bp_span = md->window_size` (`fold_compound.c:598-601`), so every
compound's span is **its own length**. `init_gpu3` passed
`VC[0]->params->model_details.max_bp_span` for the whole batch. The device's
`max_span > len_H` clamp hides that when `VC[0]` is the longest record and
silently truncates every longer record when it is not — forbidding their
long-range pairs. That is why record 0 was clean and mismatches began at record 1,
and why the device was uniformly *more* restrictive than the host. The caller now
passes 0, which selects `len_H` per record, with a host-side assert of the
precondition. **A restricted span would need a per-record table, not a wider
scalar** — noted at both ends.

## The flag is derived, not set

The 2.3.0 shape — an assignment in `RNAfold.c` — is what got lost in the port,
and a lost assignment to a default-0 flag is **invisible**: it only makes you
slower. So `par_fill_arrays()` now works it out from the batch it was handed:

```c
g_hc_seq_derived = !md0->noLP;
```

Sound because the library guard has *already* declined hard constraints (via
`hc->depot`), soft constraints, SHAPE, ligand motifs and command files before any
compound reaches this function. `noLP` is the only accepted option that perturbs
the masks, and it is the only discriminator left. The preconditions are asserted
here rather than trusted from a distance.

## Results

| bar | route | `pack` | vs CPU |
|---|---|---|---|
| default | GPU | 0.005 | **identical** |
| `--noGU` | GPU | 0.005 | **identical** |
| `-T 25` | GPU | 0.005 | **identical** |
| `RNA_SLOT_FLOW=2` | GPU | 0.013 | **identical** |
| **`--noLP`** | GPU | **0.061** | **identical** |

`--noLP` is the control: it must *keep* the host packing, and it does.

- **`RNA_HC_VERIFY`: 326 711 words × 4 masks, 0 mismatching**, unchunked and
  chunked.
- **`make check` 146/146.**
- `gpuinit` **0.082 → 0.026 s** on this input, a **3.2×** cut; `pack` 0.061 →
  0.005.

## What is NOT proven here

- **Multi-chunk.** `RNA_GPU_VRAM_BUDGET_MB=700` still produced one sweep on this
  input, so the chunked arm did not actually chunk. The masks are rebuilt per
  chunk, so this needs a real multi-chunk run.
- **The end-to-end win at scale.** These records are 350-2400 nt on a 4 GB laptop
  card. The claim being made is ~20 % of wall at 400 × 5601, and that is an L4
  number that has to be re-measured, not extrapolated from 3.2× on a small input.

Both want the stress notebook re-run — with the `gpuinit` breakdown regex added
this time, so the next reader gets the attribution for free.

---

# 13. Colab re-run on a T4: `gpuinit` 153.2 s → 1.6 s, output byte-identical

*400 × 5601, commit `851d1b04`, **Tesla T4**. Raw: `stress272_t4.json`.
The previous run was an L4, so absolute walls are NOT comparable — see §13.4.*

## 13.1 The result

| arm | chunks | `gpuinit` | `pack` | pack % of gpuinit |
|---|---|---|---|---|
| i32 natural | 5 | **1.6** | 1.43 | 91 % |
| i16 natural | 4 | 1.5 | 1.34 | 91 % |
| i32 half | 7 | 1.8 | 1.68 | 92 % |
| i16 half | 6 | 1.7 | 1.55 | 92 % |
| i32 quarter | 14 | 2.2 | 2.05 | 93 % |
| i16 quarter | 11 | 2.1 | 1.96 | 93 % |

**`gpuinit` fell from 153.2 s to 1.6 s — 22.9 % of wall to 0.2 %.** The `pack`
timer still dominates what is left because it deliberately wraps the *device*
kernel too ("same timer, for comparability", `hp_mb_loop.cu`), and it grows with
chunk count (1.43 → 1.68 → 2.05) exactly as it should: the masks are rebuilt per
chunk.

## 13.2 Why this is the fix and not a faster host

Two machines, so the obvious objection is that the T4 instance simply has a
quicker CPU. It does not:

| | L4 run | T4 run |
|---|---|---|
| `build` (serial host, O(n²) per record) | 121.96 s | **121.9 s** |

`build` is single-threaded by default and is pure host work. **0.1 % apart.** The
host is the same speed; `gpuinit` fell 96× because the packing moved to the GPU.

## 13.3 Correctness, at the size that matters

**`sha` = `7c0b3d633281` — identical to the pre-fix L4 run, in all six arms of
both runs.** 400 records × 5601 nt, twelve arms, one hash. The device-derived
bitmasks produce the same output bytes as the host-packed ones at full scale,
across 4–14 chunks. Every arm reports the same 6 266 401 200 cells and 400 GPU
records, so no arm quietly fell back to the CPU.

That closes the two things §12 said were unproven: **multi-chunk works** (this run
really chunked, 4–14 of them, unlike the local 700 MB arm), and the masks are
right at 5601 nt, not just at 2400.

## 13.4 What this run CANNOT tell us: the end-to-end win

**The T4 ran at `clock_before` = `clock_after` = 0.368 for the whole run** — 585
of 1590 MHz, throttled to 37 %. Every GPU phase inflates accordingly (`hp_mb`
38.4 → 112.1 s, ×2.9 ≈ 1/0.368), which swamps the 151 s of host time the fix
removes. Wall went 669.2 → 641.8 s, and that number means nothing across two
different, differently-throttled cards.

**So the ~20 %-of-wall claim is still not measured end to end.** What IS measured
is that the item was 22.9 % of wall and is now 0.2 %, on the same workload with a
verified-equivalent host. An unthrottled run is needed to bank the wall-clock
figure.

`backtrack` also went 5.4 → 40.6 s. It is threaded (`RNA_BACKTRACK_THREADS`) and
`build` is not, so the likely cause is a smaller vCPU count on the T4 instance
rather than anything in this change — **not verified**, and worth a look, since
backtrack is now 6.3 % of wall.

## 13.5 The wall decomposition, with `gpuinit` gone

T4, i32/natural: `modular_decomp` 34.6 %, **`build` 19.0 %, `output` 18.8 %**,
`hp_mb` 17.5 %, `backtrack` 6.3 %, residual **0.23 %**.

**`build` + `output` = 37.8 % and are now the largest host target**, as §7
predicted. Both have thread-pool routes that default off and have never been
measured at this size.

## 13.6 int16 on a throttled T4 is a wash, and `hp_mb` is why

641.8 → 642.2 s: **1.00×**, matching bench v5's 1.009×. `modular_decomp` saves
56.3 s (222.0 → 165.7) but **`hp_mb` costs 40.1 s (112.1 → 152.2)** and
`fetch_mx` another 6.9. The +8.8 s `hp_mb` regression seen on the L4 is **+40.1 s
here** — far larger, and it is most of the reason int16 is worthless on this card.
The regression is real, reproducible on two GPUs, and scales with how starved the
card is. It remains the open question from §4.

---

# 14. Third run: `output` collapses, and the wall is 1.226× shorter

*400 × 5601, commit `6d3bdc19`, **Tesla T4 at 585/1590 MHz — the same card and the
same clock state as §13**, so this one IS a like-for-like comparison. Raw:
`stress272_t4_fastpath.json`.*

## 14.1 The result

| i32/natural | §13 (`851d1b04`) | §14 (`6d3bdc19`) |
|---|---|---|
| **wall** | 641.8 s | **523.7 s** |
| `output` | 120.5 | **0.1** |
| `build` | 121.9 | 128.1 |
| `modular_decomp` | 222.0 | 220.7 |
| `hp_mb` | 112.1 | 103.8 |
| `backtrack` | 40.6 | 47.1 |
| `gpuinit` | 1.6 | 1.6 |

**1.226× end-to-end**, and the saving is exactly the stage that was removed:
Δwall 118.1 s against Δ`output` 120.4 s, with the other stages netting +2.9 s of
run-to-run drift. Nothing else moved.

**`sha` = `7c0b3d633281` in all six arms — the same hash as both previous runs.**
Three runs, eighteen arms, one hash: the fast path is byte-identical at
400 × 5601 across 5–14 chunks, which is the correctness bar the local 12-record
comparison could not reach.

## 14.2 The wall now

i32/natural, 523.7 s:

| item | s | share |
|---|---|---|
| `modular_decomp` | 220.7 | **42.1 %** |
| **`build`** | **128.1** | **24.5 %** |
| `hp_mb` | 103.8 | 19.8 % |
| `backtrack` | 47.1 | 9.0 % |
| `load_my_c` + `fetch_mx` | 18.9 | 3.6 % |
| `gpuinit` | 1.6 | 0.3 % |
| residual | 1.4 | 0.27 % |

**GPU busy 62.2 %, host-only 33.9 %** — and `build` is **72 % of that idle**,
`backtrack` the other 27 %. Two host stages are the entire remaining
opportunity.

`build` reads 128.1 here against 121.9 in §13; it is the same serial loop
untouched by either fix, so that spread is instance-to-instance drift and a
reminder not to quote these to three digits.

## 14.3 Chunking is even flatter than §5 said

5 → 7 → 14 chunks costs 523.7 → 525.9 → 526.7 s: **2.8× the chunks for 0.6 %.**
The §5 figure was 2.2 % over a 3.3× range; with `output` gone the per-chunk
overhead is an even smaller slice of a smaller wall. `gpu_bytes_per_file()` is
close to irrelevant as a performance property at this size.

## 14.4 int16 improves, because `output` was diluting it

523.7 → 496.6 s = **1.055×**, against 1.00× in §13. Nothing about int16 changed;
removing 120 s of int16-neutral host work simply stopped hiding it.
`modular_decomp` saves 64.3 s and **`hp_mb` gives back 29.6 s** — the regression
from §4 and §13.6, reproduced a third time. It is now the single clearest open
question about int16.

## 14.5 `backtrack` is already threaded, and that is informative

`RNA_BACKTRACK_THREADS` defaults to `auto` = `nproc − cpu_queue_threads`
(`mfe_cuda.c:541`), so backtrack is *already* using every core this instance has
— and it is still 47.1 s, 9 % of wall. That is not a missing optimisation; it is
a **core-starved host**, which is direct evidence for the core-count argument in
`PORT_HETEROGENEOUS_SCOPE.md` §C: Colab's T4 instances are the worst case for any
CPU-side scheme, and the machines this tool would be deployed on are not.

---

# 15. Fourth run: the pipeline at every budget, and int16 at every budget

*400 × 5601, commit `837cc6db`, **Tesla T4 at 945/1590 MHz**. Nine arms:
{i32, i16, i32pipe} × {natural, half, quarter}. Raw: `stress272_t4_pipeline.json`,
committed 2026-09-10 — the numbers below were quoted in `PORT_SESSION_2026-09-09.md`
§2 from the notebook output, but the raw file was never checked in until now.*

## 15.1 The pipeline result

| budget | chunks | i32 | i32pipe | delta | overlap | `(k−1)/k` |
|---|---|---|---|---|---|---|
| natural | 5 | 527.5 | 438.7 | **−16.8 %** | 82.2 % | 80.0 % |
| half | 7 | 530.4 | 428.5 | **−19.2 %** | 89.0 % | 85.7 % |
| quarter | 14 | 539.8 | **424.5** | **−21.4 %** | 94.8 % | 92.9 % |

Overlap tracks `(chunks−1)/chunks` and sits slightly *ahead* of it at all three
budgets. `sha` = `7c0b3d633281` in all nine arms.

**Read the pipelined `build` timer with care.** It goes 125.2 → 177.9, 128.6 →
182.9, 128.2 → 191.6 — up 42–49 %. That is the builder thread's wall including
contention and back-pressure, not the cost of building. The non-pipelined
`build` is the honest figure, and it is what §16.3's model uses.

## 15.2 int16 at all three budgets — and it is budget-insensitive

§14.4 had int16 only at the natural budget. With all three:

| budget | i32 | i16 | delta | i32 chunks | i16 chunks |
|---|---|---|---|---|---|
| natural | 527.5 | 508.3 | −3.6 % | 5 | 4 |
| half | 530.4 | 511.3 | −3.6 % | 7 | 6 |
| quarter | 539.8 | 507.7 | −5.9 % | 14 | 11 |

**int16 lands within 0.7 % of itself across a 2.75× chunk range** (507.7–511.3)
while int32 spreads 2.3 % (527.5–539.8). Note also that int16 gets **fewer chunks
at the same budget** — smaller matrices fit more records — so an int16 arm is
never chunk-matched to its int32 counterpart.

## 15.3 The give-back: `hp_mb` has a companion, and it is `fetch_mx`

i32 → i16, per budget:

| stage | natural | half | quarter |
|---|---|---|---|
| `modular_decomp` | −63.5 | −62.5 | −66.6 |
| `hp_mb` | **+29.3** | **+26.8** | **+23.2** |
| `fetch_mx` | **+12.0** | **+12.5** | **+11.8** |
| net wall | −19.2 | −19.1 | −32.1 |

`hp_mb` is the regression §4, §13.6 and §14.4 already track. **`fetch_mx` is the
one nobody has chased**: it is 2.2–2.6× slower under int16, at every budget, and
§13.6 recorded it as "another 6.9" without comment.

That shape is worth stating plainly, because it rules out the obvious
explanation. **int16 halves the bytes `fetch_mx` reads back.** A readback that
moves half the data in 2.5× the time is not a bandwidth story at all — it is the
signature of a per-element widening conversion on the host where the int32 path
is a straight `memcpy`.

## 15.4 Clock caveat: this run is not internally comparable

`clock_before`/`clock_after` swing **0.368 → 0.925** of maximum across the nine
arms. The walls are stable enough that it evidently averages out, but it is why
this run puts chunking at 2.3 % (527.5 → 539.8 over 5 → 14 chunks) where §14.3
put it at 0.6 %. **The between-run spread is the noise floor**, and neither
figure should be quoted as though it were the other.

---

# 16. Fifth run: int16 and the pipeline compose — 394.8 s

*400 × 5601, commit `16d77147`, **Tesla T4 at 675/1590 MHz**, `clock_before`/
`clock_after` inside a 0.368–0.500 band across all four arms — a much tighter
spread than §15, so this run IS internally comparable. Four arms at the quarter
budget only: `BUDGETS` was deliberately narrowed in the executed notebook, so
half and natural were **scoped out, not lost**. Raw:
`stress272_t4_i16pipe.json`.*

## 16.1 The result

| arm | wall | vs i32 | chunks | RSS (GB) |
|---|---|---|---|---|
| i32/quarter | 535.9 | — | 14 | 1.71 |
| i16/quarter | 507.1 | −5.4 % | 11 | 2.80 |
| i32pipe/quarter | 423.8 | −20.9 % | 14 | 4.64 |
| **i16pipe/quarter** | **394.8** | **−26.3 %** | 11 | 6.20 |

`sha` = `7c0b3d633281` in all four arms. **Twenty-two arms across five runs on
one hash**, now including the most aggressive combination in the tree.

**They compose, slightly better than multiplicatively.** Naive composition
(0.946 × 0.791) predicts 401.0 s; the arm came in at 394.8, **6.1 s / 1.5 %
ahead**. And int16 is worth *more* with the pipeline on — **−6.8 % against
i32pipe, against −5.4 % against i32** — by exactly the mechanism §14.4 described
for `output`: hiding host `build` raises the GPU's share of the wall, so int16's
kernel win is diluted less.

This closes the branch `PORT_SESSION_2026-09-09.md` §7 opened. It did **not**
come in flat, so "the `hp_mb` regression eats the gain once build is hidden" is
answered NO.

## 16.2 The gap to 360 s is exactly the two regressions

The §7 prediction was 360–370 s. It is worth being precise about why 394.8 is
not a miss. i32 → i16 at the quarter budget:

| stage | i32 | i16 | Δ |
|---|---|---|---|
| `modular_decomp` | 225.0 | 162.6 | **−62.4** |
| `hp_mb` | 105.5 | 132.6 | **+27.1** |
| `fetch_mx` | 8.70 | 16.41 | **+7.7** |
| everything else | | | ±2 |
| **wall** | 535.9 | 507.1 | **−28.8** |

The stages account for the wall exactly (−28.8 against −28.8). **int16 wins
62.4 s and hands back 34.8 s of it — it is delivering 44 % of its own kernel
gain.** In the pipelined pair the same shape holds: −66.7 won, +31.4 handed back,
47 % delivered.

**394.8 − 34.8 = 360.0 s.** The §7 prediction was a correct statement about what
int16 *should* deliver; the arm we measured is that figure plus the two
regressions, itemised. Closing them is worth more than anything else currently
on the list.

Note `fetch_mx` is +7.7 here against +12.0 in §15.3 — the **magnitude is
instance-dependent**. What is solid is the direction: **7 of 7 measurements
positive**, ratio 1.9–2.6×.

## 16.3 A validated model for what the pipeline will buy

    wall_pipe  ~=  wall_nonpipe  -  build_nonpipe * (chunks-1)/chunks

| arm | predicted | actual | error |
|---|---|---|---|
| §15 i32pipe/natural (5) | 427.3 | 438.7 | +2.6 % |
| §15 i32pipe/half (7) | 420.2 | 428.5 | +2.0 % |
| §15 i32pipe/quarter (14) | 420.7 | 424.5 | +0.9 % |
| §16 i32pipe/quarter (14) | 420.3 | 423.8 | +0.8 % |
| §16 i16pipe/quarter (11) | 392.8 | 394.8 | +0.5 % |

Five arms, two runs, two datatypes. It always over-predicts the win slightly,
**and the error shrinks as chunk count rises** (2.6 % at 5 chunks, 0.5 % at 11) —
consistent with the unhidden first chunk being a smaller fraction of the wall.
Good enough to plan with; do not quote it below ~1 %.

## 16.4 Host RAM

The pipeline costs ×2.2–2.7 RSS at this budget (1.71 → 4.64 for int32, 2.80 →
6.20 for int16), and int16 costs ×1.6 on top of int32 before pipelining. **The
guard's skip projection is optimistic**: extrapolating §15's natural-budget
pipeline multiplier (×1.93) onto i16/quarter predicted 5.4 GB against an actual
6.20 — **15 % low**. `i16pipe/natural` still projects past a T4 instance's
~12.7 GB and would still be skipped, but the multiplier should be re-fitted
per-budget rather than carried over from `natural`.

## 16.5 Where this leaves the queue

Wall at 400 × 5601 is **394.8 s**, from **641.8 s** at the start of 2026-09-09 —
**1.63×**, every arm byte-identical.

**The int16 give-back has overtaken `backtrack` as the top performance item.**
`backtrack` is ~47 s and is a core-starvation question that a Colab T4 instance
is the worst available host for answering. The int16 give-back is **34.8 s on the
same workload, on hardware we have, with the decomposition already in hand**, and
it is a GPU-side question, so NCU can reach it. `hp_mb` is 78 % of it.

The probe: **NCU on `hp_mb` alone, i32 vs i16, same launch.** The existing NCU
work settled the *aggregate* int16 kernel story (1.46–1.61×, never leaving the
DRAM roof); `hp_mb` is the kernel moving the *wrong* way, so it cannot be the one
that aggregate described. A kernel that slows 27 % when its operands halve in
width is losing vectorised or aligned access, gaining a conversion in the inner
loop, or hitting bank conflicts on a narrower type — all three of which NCU names
directly.

---

# 17. The `hp_mb` int16 regression is measured on a timer that is mostly not `hp_mb`

*Local, RTX 3050 laptop, 2026-09-10. 40 × 3000 and 20 × 1200. This section does
NOT re-measure 400 × 5601 — it shows that the instrument used to measure it
reports something other than what its name says, and it introduces the knob that
fixes that. The T4 re-measurement is the follow-up, not this.*

## 17.1 What the phase timers actually time

`fill_arrays_loop.c` wraps four GPU phases in host wall-clock timers. Three of
those phases only *launch* kernels and return; the work lands later, and the
host stops at whichever call next touches the device synchronously.

In the GPU-resident sweep that call is at the **start of `hp_mb`**:

```c
  upload_size_off_H(nfiles, size_off_H);   /* synchronous pageable H2D */
  upload_i_H(nfiles, i_H);                 /* synchronous pageable H2D */
  hp_mb_3p_kernel<<<...>>>(...);           /* only now the launch */
```

Both uploads change every row, so neither hits its `memcmp` shortcut, and a
pageable H2D stream-syncs before it copies. **So `hp_mb`'s timer opens by
draining everything the previous phase queued.** `modular_decomposition.cu:1853`
already says this in as many words — it just had not been connected to the phase
numbers.

`modular_decomp` is the exception: it ends with `cudaStreamSynchronize`
(`:1792`, `:1859`), so **its timer is truthful**. `int_loop`'s is not.

## 17.2 Measured: `hp_mb` is overstated ~6×

`RNA_PHASE_SYNC=1` (new, default off, `device.cu`) syncs at every phase
boundary, so each timer holds its own GPU time. Same binary, same input:

| phase | async | RNA_PHASE_SYNC=1 |
|---|---|---|
| `int_loop` | 0.469 | **6.879** |
| `hp_mb` | **5.595** | **0.907** |
| `load_my_c` | 0.468 | 0.483 |
| `modular_decomp` | 5.379 | 5.318 |

40 × 3000, int32. `hp_mb` is **6.2× smaller** once it is charged only its own
work, `int_loop` **14.7× larger**, and `modular_decomp` — the one that already
synced — **does not move** (5.379 → 5.318, 1.1 %). That last row is the control:
the only phase whose timer was already honest is the only one that stays put.

Reproduced at 20 × 1200: `hp_mb` 0.597 → 0.227, `int_loop` 0.154 → 0.619.

## 17.3 What this does to the int16 finding

"int16 makes `hp_mb` 23–30 s slower", reproduced in §4, §13.6, §14.4, §15.3 and
§16.2, is measured on a timer that mostly holds **`int_loop`'s** GPU time.

**And `hp_mb_loop.cu` contains no int16 code at all.** `fml_decode()` is called
from exactly one site, `modular_decomposition.cu:1299`; `hp_mb_3p_kernel` never
reads the int16 stream, and its launch shape and block size do not depend on the
gate. There is no mechanism by which its kernel could slow by 27 %.

The shape fits attribution exactly: int16 speeds up `modular_decomp`'s kernels,
so the host reaches `hp_mb`'s upload sooner and waits there instead. The
regression has reproduced five times **because a systematic attribution artifact
reproduces perfectly** — which is precisely why five reproductions did not make
it true.

**This does not prove int16 costs nothing.** It proves the 62.4/−27.1 split is
not evidence of where. The quantity that survives is the sum,
`modular_decomp + hp_mb`, which is invariant to where the drain lands: **−35.3 s
at 400 × 5601 quarter (330.5 → 295.1)**, an improvement with no regression in
it.

## 17.4 SETTLED at this scale: under truthful timers the regression is ZERO

*The first version of this section said the laptop could not settle this,
because four sync arms walled 16.46 / 17.22 / 18.59 / 21.21 s — a ~1.6 s
monotonic climb per run position, larger than the effect. **That was a cooling
problem, not a property of the box.** Re-run on a cool flat surface with airflow,
under `tools/gpu_thermal_watch.py`, the same four arms wall 15.28 / 13.79 /
13.78 / 15.79 — ABBA, with the two i16 arms agreeing to **0.07 %**.*

`RNA_PHASE_SYNC=1`, 40 × 3000, ABBA, drift −3.9 % across the whole run:

| phase | i32 | i16 | delta | |
|---|---|---|---|---|
| `int_loop` | 5.703 | 5.922 | +0.219 | +3.8 % |
| **`hp_mb`** | **0.859** | **0.854** | **−0.005** | **−0.6 %** |
| `load_my_c` | 0.482 | 0.474 | −0.008 | −1.7 % |
| `modular_decomp` | 5.287 | 3.856 | −1.431 | −27.1 % |
| `fetch_mx` | 0.489 | 0.292 | −0.197 | −40.3 % |

**int16's effect on `hp_mb` is −0.6 %, against a 3–5 % run-to-run spread on that
phase. It is indistinguishable from zero.** The async timer on the same binary
and input put it at +4.3 %. The regression does not survive being measured
properly, which is what §17.3 predicted from the source.

The earlier "+1.358 s (+20 %) on `int_loop`" was drift: on the cool run the same
comparison is **+0.219 s (+3.8 %)**, six times smaller. A small positive int16
cost on `int_loop` may be real and is worth a look, but it is not 20 %.

### What is still open

**The artifact is confirmed at 40 × 3000, where the async delta was +4.3 %. The
T4's async delta at 400 × 5601 is +26 %.** Six times larger, on a host with far
fewer cores and 35× the cells. So this settles the *mechanism* and settles it at
this scale; it does not by itself prove the whole +27.1 s at 400 × 5601 is
attribution, only that the instrument reporting it cannot be trusted to say.

**The follow-up is unchanged: one pair of arms at 400 × 5601 on a T4 with
`RNA_PHASE_SYNC=1`.** Compare splits between two arms both run with it — never a
synced split against an unsynced one, and never the wall of a synced run against
anything, because destroying the phase overlap costs ~9 % by construction.

**Do not optimise `hp_mb` before that run.** On truthful timers it is ~6 % of GPU
time, not the 20–25 % of wall the async profile shows.

## 17.4a The laptop is POWER-capped, not thermally throttled

`tools/gpu_thermal_watch.py` samples the card every 250 ms and names the reason.
Over the 115.8 s run above:

| | |
|---|---|
| SM clock | min 1057, median 1485, max 1740 MHz (50–83 % of the 2100 max) |
| temperature | 43 → 74 °C |
| drift, first third → last third | 1496 → 1437 MHz, **−3.9 %** |
| `SwPowerCap` | **66.8 % of samples** |
| `SwThermal` | 9.3 % of samples |

**The dominant limiter is the 65 W power cap, not heat.** That corrects the
standing note that "this laptop cannot hold a GPU clock under sustained load
(1057 → 712 MHz)": with airflow it holds within 4 %, and what stops it reaching
2100 MHz is the power budget, which no amount of cooling changes.

**Consequence: local A/B is usable again for effects above ~5 %**, provided the
run is ABBA, the machine is on a hard surface, and the drift is reported
alongside the result. It is still the wrong place for anything smaller, and for
anything needing many cores.

`nvidia-smi -lgc` — which would pin the clock and remove even the 4 % — is
refused from inside WSL (WDDM owns the device) and needs an **Administrator**
shell on the Windows side. It is not required for effects of this size, and it
would not defeat the power cap either.

## 17.5 Consequence beyond int16

If `hp_mb`'s 105.5 s at 400 × 5601 is mostly `int_loop`'s, then `int_loop` is
not the 0.5 % of wall the profile has recorded all along, and the wall
decomposition in §14.2 and §16 needs re-deriving from a phase-synced run. That
does not change any wall figure — every wall in this file was measured with a
stopwatch on the whole process — only the **split**, and only for the three
phases that never synced. `modular_decomp`, `build`, `backtrack`, `output` and
`gpuinit` are unaffected.

---

# 18. `fetch_fML_one_H`'s decode: 5.3× in isolation, unproven end-to-end

## 18.1 The loop

The int16 fML stream stores each cell as a short offset from a baseline shared by
`FML_BLK` (64) consecutive entries in a column. The host decode did one
**dependent** baseline load per cell:

```c
  dst[t] = (o == FML_INF16) ? INF
         : hb[colb[j] + (size_t)((i-1)/FML_BLK)] + (int)o;   /* per cell */
```

The baseline is constant across each block, so it can be loaded once per 64
cells. `tools/fml_decode_equiv.c` holds the old loop verbatim as the reference,
proves the rewrite identical over **1500 shapes** — including the ragged ones
where `cells` cuts the final row, which is the only reason the old loop needed a
per-cell bound test — and times both:

| | ns/cell | worker-s over 400 × 5601 |
|---|---|---|
| reference | 2.46 | 15.4 |
| blocked | **0.46** | **2.9** |

**5.3×.** The bar goes RED on a one-off block boundary: 1151 mismatches of 1500.

A red-team mutation that shifted the block index by 7 instead of 6 made the loop
**spin rather than answer wrongly** — the rewrite terminates only while
`end >= i`. That is now a comment on the function and on the bar.

## 18.2 Correctness end-to-end

int16 and int32 output `sha b47ca67676f6` across all 8 arms of §17's runs with
the new decode, and the same hash with the old one. int16 is exact on this
workload, and the rewrite does not change that.

## 18.3 What is NOT shown

**No end-to-end win has been measured.** At 40 × 3000 the decode is ~0.44 worker-s
total, which over ~12 backtrack workers is ~0.04 s of wall — and the int32
control, which neither build touches, moved 0.038 s between the two builds. The
effect is at the noise floor, so the A/B is honestly a null:

| build | i32 `fetch_mx` (control) | i16 `fetch_mx` |
|---|---|---|
| old decode | 0.438 | 0.269 |
| new decode | 0.476 | 0.275 |

**The control moving as much as the treatment is the reason this cannot be
called a win.** It is also worth recording that on this box i16's `fetch_mx` was
*already* faster than i32's before the fix (0.269 vs 0.438) — the T4's 1.9–2.6×
regression does not reproduce at 1/35th the cells on a 12-core host, so the
local box cannot test the thing the fix is for.

What is claimed: the loop is 5.3× faster and identical. What is not claimed: that
this is worth seconds at 400 × 5601. The projection is 15.4 → 2.9 worker-seconds;
turning that into wall needs the T4, in the same run as §17.4.

---

# 19. SETTLED AT SCALE: the `hp_mb` regression was attribution, and the cost is `int_loop`

*400 × 5601, commit `6517b1b6`, Tesla T4 at 645/1590 MHz. Nine async arms plus
the **phase-synced pair** §17.4 asked for. Raw: `stress272_t4_syncpair.json`
(async) and `stress272_t4_sync.json` (synced). This is the run that closes
§17.3, §17.4 and §18.3.*

## 19.1 The answer

`RNA_PHASE_SYNC=1`, both arms, quarter budget:

| phase | i32 | i16 | delta | |
|---|---|---|---|---|
| `int_loop` | 107.26 | 137.70 | **+30.44** | **+28.4 %** |
| **`hp_mb`** | **14.52** | **14.79** | **+0.27** | **+1.9 %** |
| `load_my_c` | 10.84 | 11.20 | +0.36 | +3.3 % |
| `modular_decomp` | 216.60 | 156.49 | **−60.11** | −27.8 % |
| `fetch_mx` | 7.92 | 10.56 | +2.63 | +33.3 % |

Against the async timers on the **same binary, same input, same budget**:

| phase | int16 delta, async | int16 delta, synced |
|---|---|---|
| `int_loop` | −0.25 | **+30.44** |
| `hp_mb` | **+31.65** | **+0.27** |
| `modular_decomp` | −58.49 | −60.11 |

**"int16 makes `hp_mb` 23–30 s slower" — reproduced five times across three
sessions and called the clearest open int16 question — is +0.27 s when the timer
is honest.** It was attribution, exactly as §17.3 predicted from the source.

**But the cost is real, and it is `int_loop`: +30.44 s, +28.4 %.** The local
40 × 3000 probe put it at +3.8 % and said "may be real, but it is not 20 %".
It is 28 %. The small box understated it by 7×, which is worth remembering
about that box rather than about int16.

## 19.2 The arithmetic closes

Net GPU-phase change under int16, phase-synced: **−26.41 s**.
Async wall change at the same budget: **−27.71 s** (535.1 → 507.4).

Within 1.3 s. The synced split fully accounts for the wall difference, which is
the strongest available check that the synced timers are measuring the real
thing rather than an artefact of the syncing.

## 19.3 The true wall decomposition, and how wrong the recorded one was

i32, quarter, share of GPU-phase time:

| phase | async says | **truth** |
|---|---|---|
| `modular_decomp` | 63.1 % | **60.6 %** |
| **`int_loop`** | **0.8 %** | **30.0 %** |
| **`hp_mb`** | **29.8 %** | **4.1 %** |
| `load_my_c` | 3.9 % | 3.0 % |
| `fetch_mx` | 2.4 % | 2.2 % |

**`int_loop` is the second-largest GPU phase and has been recorded as 0.8 % of
wall for the entire life of this project.** Anyone optimising from the async
profile would have gone after `hp_mb` — 4 % of the work — and left the 30 %
alone. `modular_decomp` was always honest, because it always ended in a sync.

## 19.4 Phase-sync cost ~nothing, which is itself the proof

§17 warned "never quote the wall of a synced run — destroying phase overlap
costs ~9 % by construction". Measured: **i32 535.14 synced against 535.11
async**, i.e. **zero**; i16 511.3 against 507.4, +0.8 %.

That warning was right to state and wrong in magnitude, and the reason is the
finding itself: **there was no phase overlap to destroy.** The two synchronous
pageable H2Ds at the top of `hp_mb_3p_i()` were already serialising every row.
Forcing a sync at each boundary costs nothing because the boundaries were
already hard. Cheap diagnostic, and re-runnable at any scale.

## 19.5 The fML decode fix: measured, and honestly not a wall win

This run carries the blocked decode (`d5a03af0`); the previous one (`16d77147`)
did not. Same arms, same size:

| arm | `fetch_mx` before | after | delta | |
|---|---|---|---|---|
| **i16/quarter** | 16.41 | **10.33** | **−6.08** | |
| **i16pipe/quarter** | 15.95 | **10.58** | **−5.37** | |
| i32/quarter | 8.70 | 8.51 | −0.20 | **control** |
| i32pipe/quarter | 8.17 | 8.16 | −0.00 | **control** |

**The controls are flat and the treatment moves 6 s** — which is the A/B §18.3
could not get locally, where the control drifted as far as the effect. The
decode rewrite does what it was built to do: int16's `fetch_mx` penalty over
int32 falls from +7.7 s to +1.8 s.

**It is still not a wall win.** i16/quarter walls 507.1 → 507.4. Against the i32
arm as a drift control, the attributable changes are `fetch_mx` −5.88 s but
`hp_mb` +4.58 and `modular_decomp` +3.92 — and those two are *async* timers on a
*different instance*, so they are the least trustworthy numbers here. The
defensible statement is the narrow one: **the decode is ~6 s faster and the wall
did not move.** That is what was claimed when it landed, and it stays claimed.

## 19.6 Reproducibility, and one guard that was too tight

`i16pipe/quarter` = **394.9 s** against 394.8 s in the previous run — **0.03 %**.
`sha 7c0b3d633281` in all eleven arms.

**Three arms did not run**: `i16pipe/half`, `i32pipe/natural`,
`i16pipe/natural`. That is §16.4's RSS guard, which §17-era work raised from
2.0× to **2.8×** on the strength of the quarter-budget measurement. At the
natural budget the measured multiplier is **1.93×**, so 2.8 over-projects there
and skipped `i32pipe/natural`, which is *known to fit* (9.40 GB measured in
§15.1 against ~12.7 available).

**The multiplier should be per-budget, not global** — it rises with chunk count
(1.93× at 5 chunks, 2.7× at 14). Raising it to the worst case traded two real
arms for safety at the wrong end. Cheap to fix, and it cost this run three arms.

## 19.7 What this changes about what to do next

1. **`int_loop` is the target, not `hp_mb`.** 30 % of GPU time, and int16 makes
   it 28 % worse. Both facts were invisible in every profile before this run.
2. **int16's remaining give-back is one phase, not three.** `+30.44` in
   `int_loop` against `+0.27` / `+0.36` / `+2.63` elsewhere. Whatever int16 does
   to the interior-loop kernel is the entire story.
3. **Re-derive §14.2's decomposition** from a phase-synced run before using it to
   plan anything. No wall figure changes — those came from a stopwatch on the
   whole process — but the split for `int_loop`, `hp_mb` and `load_my_c` was
   wrong by an order of magnitude in both directions.

---

# 20. IntLoop: block size 64 wins, and the int16 penalty is NOT the kernel

*T4, 400 × 5601, commit `7fa85d4e`, 12 arms, all phase-synced except one.
`CUDA_RNAFold_IntLoop.ipynb` / `intloop.json` / four NCU CSVs. This is the run
`PORT_INT_LOOP_SCOPE.md` was written for.*

## 20.0 First: the sha held in all 12 arms

**`7c0b3d633281` in every arm.** That is the outstanding debt from the
`fml_scan_kernel` INF-guard fix (`7fa85d4e`), which touched the default path: it
is byte-identical at 400 × 5601, not merely on the 20-record local bar. Paid with
no arm run for the purpose — it came free, because the sha assertion is in the
harness rather than in a particular test.

## 20.1 Block size: 64, and `modular_decomp` is a built-in control

i32, chunk cap 29, phase-synced:

| block size | `int_loop` (s) | `modular_decomp` (s) | wall (s) | ns/cell |
|---|---|---|---|---|
| 32 (the STOPGAP default) | 105.50 | 215.60 | 524.3 | 16.84 |
| **64** | **91.80** | 216.34 | **512.4** | **14.65** |
| 128 | 110.31 | 216.14 | 528.2 | 17.60 |
| 256 | 153.56 | 215.69 | 572.5 | 24.50 |

**`modular_decomp` is flat to 0.34% across all four arms.** It cannot be affected
by `RNA_INT_LOOP_BLOCK_SIZE`, so it is a free control for device drift — and it
says there was none. The `int_loop` column is a measurement, not a clock story.

A fifth arm (`A_i32_c29`) is the same configuration as the bs32 row and gives
`int_loop` 100.40 with `modular_decomp` 215.21. So **bs32 = {100.40, 105.50},
mean 102.95** — `int_loop` carries ~5% run-to-run noise where `modular_decomp`
carries 0.34%. Worth remembering before believing any single arm.

**bs64 vs bs32: −11.15 s of `int_loop` (−10.8%) and −11.5 s of wall (−2.2%).**
The two agree to within 0.4 s, which is the strongest evidence available here
that it is real: the phase delta and the wall delta are independently measured
and they moved together.

**Replicated under int16**, where the raw numbers are muddier and the control
earns its keep. bs32 = {128.06, 128.61} with `modular_decomp` {148.74, 149.00};
bs64 = 120.17 with `modular_decomp` **159.42** — a 7% outlier, so that arm ran on
a slower device. Normalising by the control gives 112.2, i.e. **−16 s**. Same
direction, larger after correction.

### Why 64 and not more

**The STOPGAP's fear was right in direction and wrong about 64.**
`int_loop.cu:1152-1173` recorded an NCU sweep in which 256 was slower at every
grid; this run reproduces that (153.56 s, +49%) and adds 128 (+7%). The kernel
launches **one block per (H,j) cell** with a *variable* amount of work — at most
`(MAXLOOP+1) × 32` candidates, usually far fewer — so a large block spends most
of its threads on nothing and pays a cross-warp reduction for the privilege. 64
is where the occupancy ceiling lifts before the work runs out.

That ceiling is now measured rather than derived: see 20.3.

## 20.2 The int16 penalty is NOT about chunk width — the hypothesis is refuted

`PORT_INT_LOOP_SCOPE.md` step 2 proposed that int16's `int_loop` cost tracks
records-per-chunk rather than the encoding, because int16 halves the fML triangle
so more records fit while `d_my_c` stays int32. Arms C and D were built to test
it, with predictions **C ≈ 107 s** and **D ≈ 138 s**.

| datatype | chunk cap | `int_loop` (s) | `modular_decomp` (s) |
|---|---|---|---|
| i32 | 29 | 100.40 / 105.50 → 102.95 | 215.21 / 215.60 |
| i32 | 37 | 106.74 / 106.59 → 106.67 | 214.40 / 214.39 |
| i16 | 29 | 128.61 / 128.06 → 128.30 | 149.00 / 148.74 |
| i16 | 37 | 131.35 | 149.38 |

- **Chunk width 29 → 37 costs +3.7 s (i32) and +3.0 s (i16).** Real, small, and
  the same for both encodings.
- **Datatype at a FIXED width costs +25.4 s (cap 29) and +24.7 s (cap 37).**

**Both predictions were wrong, in opposite directions** — C came out 128.6 where
107 was predicted, D came out 106.7 where 138 was predicted. The two variables
are cleanly separated, and chunk width accounts for about an eighth of the
effect. Do not re-propose it.

## 20.3 NCU: the kernel is IDENTICAL under int16, to four significant figures

`int_loop_kernel`'s first NCU profile ever. Five launches sampled mid-sweep at
two chunk widths, i32 against i16:

| | i32 | i16 | delta |
|---|---|---|---|
| duration, cap 24, launch 0 | 722,880 ns | 721,664 ns | **−0.17%** |
| duration, cap 24, mean of 5 | 717,676.8 ns | 717,683.2 ns | **+0.0009%** |
| duration, cap 8, mean of 5 | 258,617.6 ns | 258,969.6 ns | +0.14% |
| occupancy (`sm__warps_active`) | 47.74% | 47.75% | — |
| SM throughput | 44.22% | 44.15% | — |
| DRAM throughput | 4.81% | 4.81% | — |
| L1 hit | 76.21% | 76.16% | — |
| L2 hit | 90.44% | 90.30% | — |

**Every counter matches. `int_loop_kernel` does the same work at the same speed
regardless of the fML encoding** — which it should, since it reads `d_my_c`
(int32) and touches no int16 data at all. **So the +25 s is not the kernel.**
See 20.4.

### What the kernel actually is, which nobody knew

| | `int_loop_kernel` | `modular_decomposition_kernel` |
|---|---|---|
| DRAM throughput | **4.2 – 5.2%** | pinned at the DRAM roof |
| occupancy | **44 – 48%** | — |
| SM throughput | 40 – 44% | — |
| L1 / L2 hit | 73–76% / 89–91% | L2 6% on the fML re-reads |

**The two largest GPU phases are limited by opposite things.** `int_loop_kernel`
is **not bandwidth-bound** — at under 5% of DRAM peak it is nowhere near it. It
is latency-bound with too few warps resident, and its caches are working well.
That is precisely the profile in which raising occupancy pays, and 20.1 measured
it paying.

The 44–48% also **confirms the 50% ceiling by measurement**: sm_75 allows 32
warps/SM but only **16 blocks/SM**, so at one warp per block the block limit
binds first. The figure in `PORT_INT_LOOP_SCOPE.md` was arithmetic; this is the
counter.

## 20.4 The real int16 story: every non-`md` phase is slower at scale

Chunk cap 29, phase-synced, means of two arms each:

| phase | i32 | i16 | delta |
|---|---|---|---|
| `modular_decomp` | 215.41 | 148.87 | **−30.9%** |
| `int_loop` | 102.95 | 128.30 | **+24.6%** |
| `fetch_mx` | 8.59 | 10.99 | +27.9% |
| `hp_mb` | 14.25 | 15.32 | +7.5% |
| `load_my_c` | 10.58 | 11.14 | +5.3% |
| **GPU+transfer total** | **351.78** | **314.62** | **−10.6%** |

`fetch_mx` is explained — it decodes int16 on the host. **`hp_mb` and
`load_my_c` are not.** Neither touches int16 data, neither has a kernel that
could have changed, and both are 5–8% slower anyway. That is a **device-level
signature, not an `int_loop` one**, and it is the missing context for the +25 s:
NCU proves the kernel is identical in isolation, and at scale everything except
`modular_decomp` slows down together.

A uniform 6–8% device slowdown would take `int_loop` from 103 to ~111, so it
accounts for roughly a third of the +25 s. The rest is still unexplained.

**This is §19's family one level down.** §19 established that a phase timer which
does not end in a sync is an attribution rather than a measurement. These phases
*do* end in syncs, and they still move together — so the next suspect is the
device, not the instrument.

### The probe that separates them, and why this run could not

The notebook samples `clocks.sm / clocks.max.sm` **before and after** each run,
i.e. while the GPU is idle. Those numbers (0.19–0.78 across arms, no pattern)
measure idle clock states and **cannot answer the question they look like they
answer**. To settle it: sample the SM clock *during* both arms, and add a
per-launch timer inside `int_loop_cuda()`. 25 s over ~78 400 launches is
**~319 µs per launch** against a ~720 µs kernel — large enough that a per-launch
timer will see it immediately.

## 20.5 Phase-sync costs 0.34%, and the async arm shows why it is worth it

Chunk cap 37, i32, the same configuration run both ways:

| | synced | async | |
|---|---|---|---|
| wall | 526.9 | 525.1 | **sync costs 1.8 s, 0.34%** |
| `int_loop` | 106.59 | 2.30 | **understated 46×** |
| `hp_mb` | 13.38 | 105.94 | **overstated 7.9×** |
| `modular_decomp` | 214.39 | 223.02 | +4% |

§19 measured 14.7× and 6.2× on the same artifact. At this chunk width it is
**46× and 7.9×**. The distortion is not a fixed factor — it scales with how much
work drains into the blocking upload — so there is no correction to apply to old
async profiles. They have to be re-run.

## 20.5a DONE: 64 replicated on sm_86, and promoted

Before changing a default on one machine's evidence — int16's value turned out
machine-dependent, and the 16-blocks/SM limit that makes 64 pay is an sm_75
number — the same comparison was run on the local RTX 3050 (sm_86), 60 × 2400,
**ABBA within each pass** so a monotone drift cancels:

| pass | bs32 (mean of 2) | bs64 (mean of 2) | delta | `modular_decomp` spread |
|---|---|---|---|---|
| 1 | 5.511 s | 4.519 s | **−18.0%** | 4.157–4.181 (0.6%) |
| 2 | 6.389 s | 5.342 s | **−16.4%** | 4.163–4.243 |
| 3 | 9.508 s | 8.247 s | **−13.3%** | 4.331–4.499 |

Three passes, same direction, larger than the T4's −10.8%. The absolute numbers
climb hard across passes — `int_loop` 5.5 → 9.5 s while `modular_decomp` moves
only 8% — which is the laptop's power cap, and is why the comparison lives inside
a pass and never across one. It is also a second hint that this kernel is
clock-sensitive rather than bandwidth-bound: it degrades far faster than the
DRAM-bound phase does.

**Output sha identical in all 12 local arms and all 12 Colab arms.** Block size
must not change the answer; it does not.

`INT_LOOP_DEFAULT_BLOCK_SIZE` is now 64, named beside the four instantiations so
it cannot drift from them, with the block-size history rewritten to record the
measurement that retired the STOPGAP. `RNA_INT_LOOP_BLOCK_SIZE` still forces any
of the four.

## 20.6 What to do

1. ~~Promote block size 64~~ **DONE, see 20.5a.**
2. **Do not chase bandwidth in `int_loop_kernel`.** It runs at under 5% of DRAM
   peak. Shared-memory staging, better coalescing and an int16 `my_c` are all
   answers to a question this kernel is not asking. Occupancy and per-cell work
   are.
3. **Settle the device-level int16 slowdown** with the two probes in 20.4. It is
   worth ~29 s of the 66.5 s that `modular_decomp` wins.
4. **Chunk width is a ~3.5 s lever on `int_loop`, in both encodings** — small
   enough to ignore when choosing a VRAM budget.

---

# 21. The two probes, and three levers with their premises checked

*Built 2026-09-11 for the one question §20 left open. Notebook:
`CUDA_RNAFold_IntLoop2.ipynb` (`tools/make_nb_intloop2.py`). This section records
what was built and what the local dry-runs already say; the T4 numbers land when
the notebook does.*

## 21.1 What §20 left, restated

`int_loop_kernel` is **identical** under int16 and int32 in isolation
(717 676.8 ns against 717 683.2 ns, every counter matching) while the `int_loop`
*phase* is 25 s slower at 400 × 5601 — and `hp_mb` (+7.5 %) and `load_my_c`
(+5.3 %) are slower too, though neither touches int16 data. Every phase already
ends in a sync, so this is not §19's attribution artifact.

§20.4 named two probes. Neither existed. Both do now.

## 21.2 Probe P1 — `RNA_LAUNCH_STATS`

`device.cu`. Brackets `int_loop_kernel` **alone** with CUDA events — not the two
offset uploads that share its phase — and records `(grid, device_ms, host_ms)`
for every launch, with `RNA_LAUNCH_STATS_CSV=path` for the raw series.

It splits the phase three ways, and the three readings are decided in advance:

| reading | means |
|---|---|
| `sum(device_ms)` differs by ~25 s | the kernel IS slower in situ; NCU's locked-clock isolation hid it |
| `sum(device_ms)` matches, overhead differs | 78 000 launches, each costing more in the int16 arm |
| both match, phase still differs | the time is in the offset uploads the probe excludes |

**The summary line reports ns/block, not ms/launch, and that is not a nicety.**
Grid grows through a sweep — row *i* launches one block per (H,j) cell and *j*
ranges further as *i* falls — so raw ms of the last tenth is several times the
first tenth in a perfectly steady run. The first version of that line printed
**+1518 %** on an idle laptop, which is a thermal story manufactured out of row
geometry. It now normalises, and says in the output that it is a smell test:
the grid-matched comparison needs the CSV, and the notebook does it by binning.

Diagnostic only: every instrumented launch ends in an event sync, so the wall is
not comparable to a normal run. Measured overhead 3.6 % at 60 × 2400.

## 21.3 Probe P2 — the clock sampler, and why the last one was not one

`CUDA_RNAFold_IntLoop.ipynb` sampled `clocks.sm / clocks.max.sm` **before and
after** each arm. That is while the GPU is **idle**. Its 0.19–0.78 spread with no
pattern measured idle clock states, and it looked like a clock measurement
without being one — it is in `intloop.json` for all twelve arms.

The new sampler runs **concurrently** at 4 Hz (`nvidia-smi -lms 250`) and keeps
only samples where `utilization.gpu >= 50`, so an idle teardown tail cannot drag
the mean down and manufacture throttling in the other direction. It reports SM
clock, memory clock, temperature, power and the `clocks_throttle_reasons` bitmask
— `0x4` is the SW power cap, `0x8` HW slowdown.

**The hypothesis it tests:** a T4 is a 70 W card, and the int16 arm finishes its
`modular_decomp` work sooner, so it asks more of the device per unit time. A
device-wide slowdown that tracks power would make the +25 s a **cost of
`modular_decomp`'s −66 s win**, not an `int_loop` defect.

**Validated locally, and it already says something.** Run against a 60 × 2400
fold on the RTX 3050 it collected 47 samples, 38 of them busy, and reported
SM **1431 MHz** mean against a **1732 MHz** maximum with **37 of 38 busy samples
carrying throttle reason `0x4`, the SW power cap**. That is the laptop's known
cap ([[project_laptop_is_power_capped_not_thermal]]), measured *during* the work
for the first time rather than inferred from before-and-after idle readings. The
instrument works. Whether a T4 shows the same under int16 is the question.

## 21.4 Lever 1 — shared-memory staging, built and already losing

`RNA_MD_SMEM=1` routes to `modular_decomposition_smem_kernel`, a separate kernel
(same reason `gq_internal_kernel` is separate: the twin is 40–60 % of GPU time
and has a documented history of regressions from being edited). It stages a
1024-int tile of `fml_i` in shared memory, turning ~24 global loads per element
into one. Blocks straddling two records fall back to the unstaged loop; the check
is block-uniform so the `__syncthreads()` is never divergent.

**The premise.** The inner loop reads two streams per `y`:

* `fml_j` — the O(n²) triangle. Cell (i,j) walks column *j*; cells with different
  *j* walk **disjoint** columns. **No intra-row reuse at all**, so it is streamed
  once per row, it is the O(n³) DRAM traffic, and nothing can cache it.
  Shared memory cannot help it. The only lever on a stream with no reuse is
  fewer bytes — which is what int16 fML already did, for −31 %.
* `fml_i` — the O(n) row buffer, ~22 KB at n = 5601, of which every cell in the
  row reads a prefix. It *ought* to be permanently resident. The memory note
  that motivated this says it is not: L2 hit ~6 %, and since the two streams
  issue one load each per `y`, a perfectly-cached `fml_i` alone would floor that
  near 50 %.

### The local result: a clean 15 % regression

RTX 3050, 60 × 2400, ABBA, four arms each:

| | `modular_decomp` (the lever) | `int_loop` (the CONTROL) |
|---|---|---|
| off | 4.547 / 4.556 / 4.544 / 4.558 → **4.551** | 8.466 – 8.527 |
| on | 5.230 / 5.231 / 5.243 / 5.233 → **5.234** | 8.523 – 8.539 |
| | **+15.0 %** | flat to 0.9 % |

Spread within each arm is 0.3 % and the control is flat, so this is not noise
and not a near miss. **Byte-identical in all seven configurations tried** —
default, int16, chunked, `RNA_MD_TILE` 1 and 8, int16+chunked, and `-c` (which
exercises the `fm2` store in both kernels).

The reading: **`fml_i` was already cache-resident**, and the 6 % L2 hit is the
`fml_j` stream, exactly as the access pattern predicts. Staging bought nothing
and cost two `__syncthreads()` per tile.

The notebook still runs it on the T4, because the cache sizes differ and because
a measured null is what retires the note — and §B measures DRAM bytes against
bytes requested, which says whether the *mechanism* is what the premise claimed
rather than only that the wall did not move.

## 21.5 Lever 2 — coalescing, measured rather than assumed

With `TILE=32` one warp owns one cell and adjacent lanes walk adjacent `y`, so
**both** streams are unit-stride: `fml_i[row_off+y]` and `fml_j[tri_off+y+ij0]`.
32 lanes × 4 B = 128 B = **4 sectors**, which is perfect. The kernel comment
already says the 22 Aug rewrite improved coalescing as a side effect.

So the expectation is that there is nothing to win — and the notebook measures
`l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` for both
kernels rather than asserting it. A value near 4.0 closes the question; anything
higher is a finding.

## 21.6 Lever 3 — int16 `my_c`: NOT built, ceiling measured first

It is the largest buffer and halving it roughly doubles chunk capacity, which
makes it sound like the obvious next move. Two ceilings say otherwise, and both
can be measured without writing it.

**The speed ceiling is ~zero.** `int_loop_kernel` runs at **4.2–5.2 % of DRAM
peak** (§20.3). A kernel nowhere near the memory roof cannot be sped up by
halving one of its inputs. The notebook re-measures this and adds what §20.3 did
not have: how much of that traffic is `my_c` at all.

**The capacity ceiling is emulated with `RNA_GPU_CHUNK`.** int16 `my_c` buys
VRAM, VRAM buys records per chunk, and nothing else. So ask for the wider chunk
directly:

| configuration | bytes/cell | cap at today's budget |
|---|---|---|
| i32 `my_c` + i32 fML | 8 | 29 |
| i32 `my_c` + i16 fML | 6 | ~37 *(measured, §20.2)* |
| **i16 `my_c` + i16 fML** | **4** | **~58** |

§20.2 already measured chunk 29 → 37 **costing** 3.7 s of `int_loop`. If the
curve keeps rising to 58, int16 `my_c` is worth nothing before a line is
written — and that is the outcome to hope for, because it is the cheap one.

## 21.7 The shape of this section, which is the point

Two of the three levers are plausible-sounding answers to a question the profile
says the kernel is not asking. Each gets a **cheap test of its premise before an
expensive test of itself**, and the expensive test runs anyway where the premise
check says it will lose — because a measured null retires a note and a prediction
does not.

That is the same discipline that would have caught the block-size verdict a day
earlier: the sweep that "closed" it tested 32 against 256 and was read as closing
64 (§20.1, `PORT_INT_LOOP_SCOPE.md`).

---

# 22. ANSWERED: the int16 give-back is a POWER CAP, and all three levers are dead

*T4, 400 × 5601, commit `a8c3be1b`, 12 arms plus five NCU profiles and a
three-block-size stall breakdown. `CUDA_RNAFold_IntLoop2.ipynb` +
`tools/intloop2_addendum.py`. `sha 7c0b3d633281` in every arm.*

## 22.1 The answer to §20.4

§20 left one question: NCU says `int_loop_kernel` is **identical** under int16 and
int32, while the phase is 25 s slower. Probe P1 (`RNA_LAUNCH_STATS`) and probe
P2 (the concurrent clock sampler) settle it.

| | i32 | i16 | delta |
|---|---|---|---|
| `int_loop` PHASE (s) | 89.33 | 115.41 | **+29.2 %** |
| **of which the KERNEL** (`sum(device_ms)`) | **84.98** | **110.74** | **+30.3 %** |
| launch overhead (s) | 1.67 | 1.73 | +3.2 % |
| launches | 78 358 | 78 358 | 0 |
| p50 launch (ms) | 1.0361 | 1.2142 | +17.2 % |
| **SM clock (MHz, sampled DURING the arm)** | **1044** | **781** | **−25.2 %** |
| power (W) | 66.0 | 65.8 | −0.3 % |
| temp max (°C) | 82 | 81 | −0.6 % |
| throttle reason on busy samples | `0x4` on 98.4 % | `0x4` on 98.8 % | — |

**It is not launch overhead** — that is 1.67 s against 1.73 s, a difference of
60 ms across 78 358 launches. **It is the kernel**, and the kernel is slower
because **the clock is 25 % lower**.

`0x4` is the SW power cap. **Both arms are pinned at the T4's 70 W board limit**,
drawing the same 66 W at the same temperature, and the int16 arm gets 781 MHz
where the int32 arm gets 1044 MHz.

### The arithmetic closes

If `int_loop_kernel` were purely SM-clock-bound its time would scale as 1/clock:

```
clock ratio      1044 / 781   = 1.337
kernel ratio   110.74 / 84.98 = 1.303
```

**97.5 % of the +30 % is the clock.** Which is exactly what §20.3 predicted
without being able to say so: a kernel at 4.2–5.2 % of DRAM peak and 40–44 % SM
throughput is latency-bound, and a latency-bound kernel tracks the SM clock.

And it is why NCU could not see it. **NCU locks clocks.** The kernel really is
identical at equal clock — 717 676.8 ns against 717 683.2 ns — and 30 % slower
at the clock it actually gets. Both measurements are correct; neither alone is
the answer.

### Every phase pays in proportion to its clock sensitivity

| phase | i32 | i16 | delta | what it is bound by |
|---|---|---|---|---|
| `int_loop` | 89.33 | 115.41 | **+29.2 %** | SM clock (pays ~the full 33.7 %) |
| `hp_mb` | 14.78 | 16.04 | +8.5 % | small kernels, part launch-bound |
| `load_my_c` | 11.06 | 11.47 | +3.7 % | transfers, barely clock-bound |
| `fetch_mx` | 9.42 | 11.47 | +21.7 % | **host-side** int16 decode — a real cost |
| `modular_decomp` | 217.49 | 155.69 | **−28.4 %** | DRAM-bound; wins anyway |
| **wall** | **526.6** | **494.1** | **−6.2 %** | |

The ordering predicted in §20.4 from "clock sensitivity" holds exactly.

### So it is a COST OF THE WIN, not a defect

int16 removes DRAM work from `modular_decomp` and replaces it with decode
arithmetic. On a power-capped card, ALU work is what costs watts — the SMs do
more per cycle, the governor drops the clock 25 %, and every SM-clock-bound
phase pays. *(Same power, lower clock, decode arithmetic added: the mechanism is
inference from those three measurements, not a direct measurement of where the
watts go.)*

Two consequences worth stating plainly:

1. **The give-back is not recoverable by optimising `int_loop`.** It is a
   power-budget reallocation, not a kernel defect. The only thing that would
   reduce it is making `int_loop_kernel` *less* clock-sensitive.
2. **On a card with power headroom, int16 is worth MORE than 6.2 %.** The
   measured figure is a T4 number that includes a 25 % clock penalty on three of
   five phases. This is a prediction, not a measurement — it needs a card that
   is not pinned at `0x4`.

## 22.2 Lever 1: shared-memory staging is dead, with the premise measured

**§B, the premise check.** DRAM bytes against bytes requested, for
`modular_decomposition_kernel`:

| | requested | DRAM read | **ratio** | sectors/req | L1 | L2 |
|---|---|---|---|---|---|---|
| `md_i32` | 1.10 MB | 0.60 MB | **0.55** | 2.18 | 69.0 % | 30.8 % |
| `md_i16` | 1.17 MB | 0.53 MB | **0.46** | 1.66 | 78.2 % | 41.0 % |

The decision rule was written down before the run: **ratio ≈ 0.5 means `fml_i`
is already resident and staging has nothing to win**. It is 0.55. The two streams
issue one load each per `y`, so half the traffic being cache-served is exactly
"one stream is cached, the other is not" — `fml_i` resident, `fml_j` streamed.
**Premise refuted by measurement.**

**§C, the lever itself**, 400 × 5601, ABBA, with `int_loop` as the control:

| | off | on | delta |
|---|---|---|---|
| `modular_decomp` (the lever) | 217.51 | 216.93 | **−0.27 %** |
| `int_loop` (the CONTROL) | 91.32 | 91.94 | +0.68 % |
| wall | 528.8 | 528.5 | −0.05 % |
| SM clock | 1005 MHz | 998 MHz | — |

A wash. (On sm_86 it was a 15 % regression; on the T4, with twice the L2, it is
neutral. Either way: no win.)

**And the staged kernel did do what it claimed** — which is the difference
between "it is slower" and "it did not work":

| | twin | staged | |
|---|---|---|---|
| requested | 1.10 MB | **1.03 MB** | −6 %, so global loads really were removed |
| DRAM read | 0.60 MB | **0.65 MB** | +8 %, the removed loads were cache hits |
| duration | 10.9 µs | **15.2 µs** | **+39 %** |

It replaced cache hits with shared-memory traffic and two `__syncthreads()` per
tile. **Retire "shared-memory staging is a LIVE lever" for good.**

## 22.3 Lever 2: coalescing was never a problem

`l1tex__average_t_sectors_per_request`: **2.18** (`modular_decomposition_kernel`)
and **2.23** (`int_loop_kernel`). Perfect coalescing for 32 lanes × 4 B is 4.0
sectors/request; below that means L1 is *deduplicating* requests as well. Both
kernels are already better than ideal-coalesced. **Closed — there is nothing to
win, and the `TILE=32` rewrite is why.**

## 22.4 Lever 3: int16 `my_c` — do not build it

Both ceilings measured, neither needing the feature to exist.

**The speed ceiling is zero.** `int_loop_kernel`, i32 against i16:

| | requested | DRAM read | ratio | DRAM % of peak | duration |
|---|---|---|---|---|---|
| `il_i32` | 5.19 MB | 0.51 MB | **0.10** | 4.0 % | 41.3 µs |
| `il_i16` | 5.19 MB | 0.51 MB | **0.10** | 4.0 % | 41.4 µs |

**90 % of what this kernel requests is already served by cache**, and what
reaches DRAM is 4 % of peak. Halving `my_c` cannot speed it up. (It also
re-confirms §20.3 at a third fixture: identical under both encodings.)

**The capacity benefit is negative.** int16 `my_c` buys VRAM, VRAM buys records
per chunk, and that is all. Asked for directly with `RNA_GPU_CHUNK`:

| cap | chunks | `int_loop` | `modular_decomp` | wall | **SM clock** |
|---|---|---|---|---|---|
| 29 | 14 | 116.69 | 157.19 | **499.1** | 769 MHz |
| 37 | 11 | 120.76 | 159.21 | 502.3 (+0.6 %) | 736 MHz |
| 45 | 9 | 124.02 | 161.34 | 504.7 (+1.1 %) | 719 MHz |
| 58 | 7 | 128.66 | 164.13 | 511.4 (+2.5 %) | 691 MHz |

**Wall rises monotonically.** int16 `my_c` would buy a cap of ~58, which is
**2.5 % slower** than the cap it already gets. **Do not build it.**

### And the chunk-width penalty is the SAME power-cap effect

§20.2 measured chunk 29 → 37 costing +3.7 s of `int_loop` and offered a
*locality* explanation (a bigger `my_c` working set per chunk). That was wrong
too. Look at the clock column: 769 → 691 MHz as the chunk widens, because more
concurrent work draws more power.

```
clock ratio      769 / 691    = 1.113
int_loop ratio  128.66/116.69 = 1.103
```

**The chunk-width cost is 99 % clock, not locality.** One mechanism now explains
both the int16 give-back and the chunk-width penalty, and neither is about
memory.

## 22.5 The stall breakdown: `wait` dominates, and `barrier` is why 128 loses

`tools/intloop2_addendum.py`, `int_loop_kernel` at three block sizes. **This is
the occupancy half §20.1 never measured.**

| bs | achieved occupancy | duration | binding limiter (blocks/regs/smem/warps) |
|---|---|---|---|
| 32 | **39.6 %** | 127.2 µs | 16 / 36 / 128 / 32 → **blocks** |
| **64** | **69.8 %** | **97.9 µs** | 16 / 18 / 64 / 16 → blocks *and* warps |
| 128 | **86.9 %** | 107.3 µs | 16 / 9 / 64 / 8 → **warps** |

**The block-count limit really is what bound block size 32**, exactly as the
arithmetic said, and lifting it nearly doubles achieved occupancy (39.6 → 69.8 %)
for a 23 % shorter kernel. The §20.1 mechanism is now measured, not inferred.

**And 128 gets even more occupancy (86.9 %) while being slower.** The stall
breakdown says why:

| stall reason | bs 32 | bs 64 | bs 128 |
|---|---|---|---|
| **`wait`** (fixed-latency ALU dependency) | **23.3 %** | **23.9 %** | **24.4 %** |
| **`barrier`** (`__syncthreads()`) | **0.1 %** | **6.6 %** | **20.7 %** |
| `long_scoreboard` (global memory) | 12.0 % | 14.2 % | 15.8 % |
| `short_scoreboard` (shared memory) | 9.4 % | 9.4 % | 8.9 % |
| `not_selected` | 3.3 % | 7.1 % | 6.7 % |
| **total stall cycles per issue** | **8.63** | **10.83** | **12.72** |

In absolute cycles per issue, `barrier` goes **0.010 → 0.833 → 2.633**. At one
warp per block the two `__syncthreads()` are warp-synchronous and free; at four
warps they are a barrier across warps doing *data-dependent* amounts of work,
and they become the second-largest stall in the kernel. **That is the toll that
eats the occupancy win, and it is why the curve turns over at 64.**

### The energy-table hypothesis is refuted

§20.3 speculated that the latency was the interior-loop energy tables (`int22`
≈ 202 KB, `int21` ≈ 40 KB thrashing a 64 KB L1). **`long_scoreboard` — global
memory latency — is only 12–16 %, while `wait` is 23–24 % and roughly flat in
absolute terms across every block size.** Memory is the *third* cost, not the
first. More warps do not hide `wait`, which is exactly why doubling the occupancy
ceiling bought 23 % rather than 2×.

## 22.6 What to do now

1. **Stop looking for the int16 give-back in `int_loop`.** It is the power cap.
   Re-measure int16's value on a card that is not pinned at `0x4` before quoting
   6.2 % as its worth.
2. **All three proposed levers are closed with data** — shared-memory staging
   (premise refuted *and* lever neutral), coalescing (2.2 sectors/request, better
   than ideal), int16 `my_c` (no speed ceiling, negative capacity benefit). None
   should be re-proposed without new evidence.
3. **The measured target in `int_loop_kernel` is the dependent chain, then the
   scan machinery** — `wait` 23 %, `barrier` + `short_scoreboard` 9.5 % at bs32
   rising to 29.6 % at bs128, `long_scoreboard` 12–16 %. The shape that attacks
   two of those at once is a **warp-synchronous scan**: give each *warp* its own
   (H,j) cell and replace `col_mask[]`/`prefix[]`/`__syncthreads()` with
   `__shfl`. That removes `barrier` entirely and lets block size rise for
   occupancy without paying for it. Not attempted — recorded as the direction the
   data actually points at.
4. **Chunk width is a clock lever, not a locality lever.** Narrower chunks run at
   a higher clock. The quarter budget being both fastest and smallest
   (§"BUILD PIPELINE VALIDATED") now has a mechanism.

---

# 23. A100: the power-cap diagnosis confirmed, my prediction refuted, and the wall is now 54 % `build`

*A100-SXM4-40GB, same 400 × 5601 workload, same commit, same 12 arms as §22.
`sha 7c0b3d633281` throughout. Clock pinned at **1410 / 1410 MHz** and
**`clocks_throttle_reasons.active` is `0x0` in every sample of every arm** —
197–217 W against a 400 W board, 46–48 °C.*

## 23.1 The int16 give-back is GONE

| | T4 (§22.1) | **A100** |
|---|---|---|
| `int_loop` PHASE | 89.33 → 115.41 s (**+29.2 %**) | 27.23 → 27.24 s (**+0.0 %**) |
| of which the KERNEL | 84.98 → 110.74 s (**+30.3 %**) | 23.79 → 23.80 s (**+0.0 %**) |
| `hp_mb` | +8.5 % | +0.2 % |
| `load_my_c` | +3.7 % | +1.0 % |
| SM clock | 1044 → 781 MHz (−25.2 %) | **1410 → 1409 MHz (−0.0 %)** |
| throttle on busy samples | `0x4` on ~98 % | **`0x0` on 100 %** |

**Remove the cap, remove the penalty — to two decimal places.** §22 said 97.5 %
of the give-back was the clock; the A100 says it was all of it. And NCU was right
the whole time: the kernel is identical, and always was.

That is as clean a confirmation as this project has ever got, because it did not
come from a better measurement of the same machine — it came from removing the
mechanism.

## 23.2 My prediction was WRONG, and the reason is worth more than the prediction

§22.1 predicted: *"on a card with power headroom, int16 is worth MORE than
6.2 %."*

| | T4 | A100 |
|---|---|---|
| wall, i32 | 526.6 s | 224.9 s |
| wall, i16 | 494.1 s | 215.1 s |
| **int16's benefit** | **−6.2 %** | **−4.3 %** |

**It is worth LESS.** The reasoning was half an argument: removing the cap does
remove int16's *cost*, but the same card removes most of its *benefit*. int16
relieves DRAM pressure on `modular_decomp`, and on an A100 there is little to
relieve:

| | T4 | A100 |
|---|---|---|
| `modular_decomp` as a share of wall | 217.5 / 526.6 = **41 %** | 42.9 / 224.9 = **19 %** |
| `modular_decomp` i32 → i16 | −61.8 s | −7.0 s |
| NCU DRAM throughput, `md_i32` | 18.1 % of peak | **3.4 % of peak** |

On the T4 int16 buys 61.8 s and gives 29.8 s back. On the A100 it buys 7.0 s and
gives nothing back. **A lever that relieves a bottleneck is worth less on a
machine that does not have that bottleneck** — which is obvious stated plainly,
and I did not state it.

## 23.3 Chunk width REVERSES, which closes §22.4's mechanism

| cap | T4 wall | T4 clock | **A100 wall** | **A100 clock** |
|---|---|---|---|---|
| 29 | 499.1 s | 769 MHz | 214.3 s | 1410 MHz |
| 37 | 502.3 (+0.6 %) | 736 MHz | 207.7 (**−3.1 %**) | 1410 MHz |
| 45 | 504.7 (+1.1 %) | 719 MHz | 204.2 (**−4.7 %**) | 1409 MHz |
| 58 | 511.4 (+2.5 %) | 691 MHz | 200.5 (**−6.5 %**) | 1410 MHz |

On the T4 wall rises monotonically and the clock falls monotonically. On the
A100 the clock does not move at all and **wall falls monotonically** — wider
chunks mean fewer chunks and less per-chunk fixed cost, which is what one would
naively expect and what the T4 hid completely.

So §22.4's claim that the chunk-width penalty is "99 % clock, not locality" is
confirmed by its disappearance. **There is no locality penalty. There never
was.**

### And it sharpens the int16 `my_c` verdict rather than reversing it

§22.4 said *don't build it* because wall rises with chunk width. That
observation was T4-specific. The verdict survives, for a better reason:

- **On a card where capacity is the constraint (T4, 15 GB), more capacity is
  HARMFUL** — cap 58 is 2.5 % slower than cap 29.
- **On a card where more capacity helps (A100, −6.5 %), capacity is not the
  constraint** — 40 GB already reaches cap 58 at this workload without it.

int16 `my_c` buys capacity. There is no machine in evidence where that is worth
having. It would take a card that is *both* VRAM-constrained *and* not
power-capped to make it pay, and that combination has not been observed.

## 23.4 Shared-memory staging: three architectures, never a win

| | sm_86 (RTX 3050) | sm_75 (T4) | sm_80 (A100) |
|---|---|---|---|
| `modular_decomp` with `RNA_MD_SMEM=1` | **+15.0 %** | −0.27 % | **+2.26 %** |
| control (`int_loop`) | flat | +0.68 % | +0.19 % |

Closed. The premise (`fml_i` thrashing) was refuted by the DRAM/requested ratio
on both measured cards — 0.55 on the T4, **0.38** on the A100, i.e. even more
cache-served — and the lever loses or ties on every architecture tried.

## 23.5 THE HEADLINE: the wall is 54 % `build`, and the GPU is no longer the problem

| | T4 | **A100** | speedup |
|---|---|---|---|
| `modular_decomp` | 217.49 s | 42.88 s | **5.07×** |
| `int_loop` kernel | 84.98 s | 23.79 s | **3.57×** |
| `hp_mb` | 14.78 s | 4.63 s | 3.19× |
| **`build` (host)** | **~122 s** | **121.4 s** | **1.00×** |
| **wall** | 526.6 s | 224.9 s | **2.34×** |

The GPU phases got 3–5× faster. **`build` did not move at all.** So:

| A100 wall, i32 | share |
|---|---|
| **`build`** | **121.4 s — 54.0 %** |
| GPU + transfer | 95.2 s — 42.3 % |
| `backtrack` | 5.7 s — 2.6 % |

**`build` is now the majority of the wall**, and it is single-threaded *because
of a defect this project already found and already wrote up*: the `params.c`
race (Defect B), recorded in the memory as "a MEASURED blocker on 24.5 % of
wall". **On an A100 it blocks 54 %.**

`RNA_BUILD_THREADS` existed on the 2.3.0 branch and was never ported to 2.7.2
(step 2b). That port, plus Defect B, is now worth more than every remaining GPU
optimisation combined:

- `int_loop` is **12 %** of the A100 wall. A 20 % win on it is **2.4 %** of wall.
- `build` is **54 %**. Threading it across even 4 cores is ~40 % of wall.

## 23.6 Two smaller findings worth recording

**`fetch_mx` is 2× SLOWER on the A100** — 18.22 s against 9.42 s on the T4, for
identical bytes — and is now 8 % of the wall, the third-largest GPU-phase entry.
It is a host/transfer quantity, so this is about the Colab A100 instance's
transfer path rather than the card.

**And int16 flips its sign there**: `fetch_mx` is **−24.4 %** under int16 on the
A100 (halved bytes win) against **+21.7 %** on the T4 (host decode dominates).
Any claim about int16's cost in this phase is host-dependent and must name the
machine.

**`int_loop_kernel` is even further from the memory roof on an A100**: DRAM
0.8 % of peak (against 4.0 % on the T4), ratio 0.05 — 95 % of its requests are
cache-served. Whatever limits it, it is not bandwidth, on either card.

## 23.7 What this changes

1. **Re-order the work.** GPU kernels are 42 % of the A100 wall and `build` is
   54 %. Finish the warp-synchronous scan (it is written and measured), then go
   to the host.
2. **Defect B is the highest-value item in the project.** It was worth 24.5 % of
   wall when it was written up; it is worth 54 % on the hardware people would
   actually run this on.
3. **Quote int16's value with a machine attached.** −6.2 % on a T4, −4.3 % on an
   A100, and the split between "what it buys" and "what it gives back" is
   completely different on the two.
4. **Chunk width: wider is better when not power-capped.** The VRAM budget
   should probably be chosen per-machine rather than pinned at a quarter, and
   §"BUILD PIPELINE VALIDATED"'s "the quarter budget is both fastest and
   smallest" is a T4 statement.

---

# 24. The warp-synchronous scan: −21.3 %, byte-identical, gated off

*Prototype, `RNA_INT_LOOP_WARP=1`, `int_loop_warp_kernel` in `int_loop.cu`.
Measured on the local RTX 3050 only — the T4/A100 arms have not been run.*

## 24.1 What the stall data asked for

§22.5 and §23 measured `int_loop_kernel`'s stalls on two architectures:

| stall | T4 bs32→128 | A100 bs32→128 | what it is |
|---|---|---|---|
| `wait` | 23.3 → 24.4 % | 19.6 → 20.1 % | the ALU dependency chain |
| `barrier` | **0.1 → 20.7 %** | **0.0 → 19.4 %** | `__syncthreads()` |
| `long_scoreboard` | 12.0 → 15.8 % | **22.7 → 27.5 %** | global memory latency |
| `short_scoreboard` | 9.4 → 8.9 % | 7.6 → 6.7 % | shared memory |

Two of the top three are **this kernel's own scan machinery**, not the problem it
is solving. And `barrier` is why block size 128 gets more occupancy than 64 on
*both* cards and is slower on both.

*(The ranking differs by card — `wait` leads on the T4, `long_scoreboard` on the
A100, which is consistent with latency rather than bandwidth: the A100 runs at a
higher clock with lower achieved occupancy, 16.8 % against 39.6 % at bs32, so it
has more latency and less to hide it with. Neither card is near the memory roof:
4.0 % and 0.8 % of DRAM peak.)*

## 24.2 Three changes, each aimed at a measured stall

**One WARP owns a cell, not one block.** `col_mask[]` and `prefix[]` were shared
only so warp 0 could publish them to the rest of the block. With the cell owned
by a single warp they live in **registers**, one column per lane — and `maxcol`
is bounded by `MAXLOOP = 30`, so one warp covers every column there can be. That
deletes both `__syncthreads()` and both shared arrays, and a block now holds
`BLOCK/32` *independent* cells, so occupancy can rise with block size without the
barrier toll.

**The column lookup becomes a 5-step shuffle binary search.** The twin does

```c
while(column <= maxcol && prefix[column+1] <= work) column++;
```

restarting at 0 for **every work item** — up to 31 dependent shared-memory loads
each. That is a prime suspect for `wait` and it is all of `short_scoreboard`.
The replacement keeps the prefix distributed across lanes and searches it with
five `__shfl_sync`es: register speed, no memory in the dependent chain.

**A fixed five iterations, not `while (lo < hi)`.** A divergent trip count would
make the shuffles undefined — the kind of bug that yields a plausible wrong
answer rather than a crash. Lanes that have converged still execute the shuffle
and discard it.

A side effect worth noting: the twin needs an explicit clamp because
`maxcol <= -2` would read `prefix[maxcol+1]` *before* the shared array, a real
memory-safety bug it hit once. With the prefix in registers that index cannot
exist — `maxcol < 0` simply gives every lane `popc = 0` and no iterations.

## 24.3 Result

RTX 3050, 60 × 2400, ABBA, two passes. **`modular_decomp` is the control** —
`RNA_INT_LOOP_WARP` cannot reach it.

| | `int_loop` | `modular_decomp` (control) |
|---|---|---|
| twin (`warp=0`) | 8.534 / 8.554 / 8.551 / 8.534 → **8.543** | 4.556 – 4.599 |
| **warp (`warp=1`)** | 6.720 / 6.727 / 6.735 / 6.727 → **6.727** | 4.542 – 4.567 |
| | **−21.3 %** | flat to 1.3 % |

**0.22 % spread within each arm** and a flat control. This is not a marginal
result needing statistics.

## 24.4 Byte-identical, and checked where it could differ

The claim is structural: `min` is associative and commutative over exact ints,
and this enumerates the **same set** of `(p,q)` candidates in a different order.

| check | result |
|---|---|
| vs its own twin, same binary, cells/block 32/64/128/256 | **identical sha** at all four |
| vs pristine 2.7.2, 20 records, across `--noLP` `-d0` `--noGU` `--noClosingGU` `-c` `-g` `--maxBPspan=60` `--salt` `-T` `-p` | **0 differing lines**, `sweeps` asserted on every arm |
| chunked, int16, chunked+int16, and block size 32 and 256 | **0 differing lines** |

## 24.5 What is NOT done

- **Only sm_86 has been measured.** The stall data that motivated it comes from
  the T4 and A100; the win does not. Both need an arm before this becomes the
  default, and the block-size interaction needs re-running — the warp kernel's
  `RNA_INT_LOOP_BLOCK_SIZE` now selects *cells per block*, which is a different
  quantity from the twin's *threads per cell*, so 64 being right for one says
  nothing about the other.
- **No NCU profile of the new kernel.** The prediction is `barrier` → 0,
  `short_scoreboard` → 0, `wait` down; that should be confirmed rather than
  assumed, and it is the check that says whether the −21.3 % came from the
  mechanism intended.
- **It stays gated off** until both land.

## 24.6 And in context: this is 12 % of the A100 wall

§23.5 put `int_loop` at 27.23 s of a 224.9 s A100 wall. A 21 % win on it is
**2.6 % of wall** there, against `build`'s **54 %**. The kernel work is worth
finishing because it is written and measured — but it is not where the next hour
should go.

---

# 25. `build` threading is ON by default: 5.12×, and the doc had said so all along

*2026-09-12, prompted by §23.5 — `build` is **54 % of the A100 wall** and did not
move at all when the GPU got 3–5× faster.*

## 25.1 It was already built, already safe, and already documented as on

`RNA_BUILD_THREADS` exists, is correct, and has been since 2026-09-09. Two
things had to be true for it to be safe and both were made true then:

- **Defect B is fixed.** `vrna_fold_compound()` reaches `vrna_params()`, whose
  `SPEEDUP_PARAMS` cache was unsynchronised shared mutable state; `params.c` now
  guards it. `tools/params_race.c` is the bar and it goes red without the lock:
  **0 of 16 000 wrong with it, 11 999 of 16 000 (75 %) without.**
- **Records are independent** — each writes only `VC[i]` and `Str[i]`.

And `MERGING.md` has listed it under **"Default ON"** with default `auto` that
whole time. **The code defaulted it to 1.** The document was describing the
intent; the code never got there. That disagreement is the change.

## 25.2 What changed is the size of the prize

It was measured once, on 2026-09-09, at a build of **0.471 s** — 4.5× on 12
cores. At that size it is a rounding error, and shipping a brand-new threading
knob default-off was the right call.

§23.5 changed the arithmetic:

| | T4 | A100 |
|---|---|---|
| `build` | ~122 s (23 %) | **121.4 s (54.0 %)** |
| GPU + transfer | 352 s (67 %) | 95.2 s (42.3 %) |

**The GPU phases got 3–5× faster and `build` did not move at all**, so a knob
that was worth a rounding error on the card the project tuned against is worth
half the wall on the card anyone would actually run it on.

## 25.3 Measured properly this time

60 × 2400, 12 cores, **ABBA × 2** — because the first sweep's baseline arm was
cold and returned a superlinear 2.58× at two threads, which is an allocator
warm-up artefact rather than a result.

| | `build` | mean |
|---|---|---|
| `RNA_BUILD_THREADS=1` | 1.492 / 1.531 / 1.291 / 1.337 | **1.413 s** |
| `auto` (12) | 0.258 / 0.297 / 0.285 / 0.265 | **0.276 s** |
| | | **5.12×** |

7 % spread within each arm. The full thread sweep, for the shape:

| threads | build | speedup |
|---|---|---|
| 1 | 1.413 | 1.00× |
| 2 | 0.651 | 2.2× |
| 4 | 0.358 | 3.9× |
| 6 | 0.340 | 4.2× |
| 12 | 0.260 | **5.4×** |

**Scaling flattens past four to six threads**, which is what an allocation- and
bandwidth-heavy O(n²) table build should do. The remaining gain to 12 is real
but shallow, and the right number is a property of the *host* — hence `auto`
rather than a tuned constant.

*(Extrapolating to the A100: 121.4 s ÷ 5.12 ≈ **24 s**, taking a 224.9 s wall to
roughly **127 s, −44 %**. That is an extrapolation, not a measurement — the
Colab A100 host's core count and memory bandwidth are not this laptop's, and
this needs an arm to confirm.)*

## 25.4 Byte-identical, checked where it could matter

| check | result |
|---|---|
| vs pristine 2.7.2, 20 records × 12 option arms including `-j` and `-j4` | **0 differing lines** each, `sweeps` asserted |
| serial vs `auto`, default / `RNA_BUILD_PIPELINE=1` / int16 / chunked | **identical sha** on all four |
| thread counts 2 / 3 / 4 / 6 / 8 / 12 / `auto` | identical sha on all seven |
| `make check` | 161/161 |

`-j` is in that list deliberately: the output pool and these builders can
overlap on the pipeline path, and that is the one combination where a race would
show.

## 25.5 What was deliberately left alone

**Oversubscription with `-j`.** A `-j nproc` run briefly has 2 × nproc runnable
threads during the pipeline's overlap window. Left as is: the builders are
short-lived and the fold they overlap is GPU-bound, so the cost is context
switches rather than contention, and capping it properly would need a shared
thread budget this driver does not have. `RNA_BACKTRACK_THREADS` subtracts
`cpu_queue_threads` for exactly this reason — but that queue is retired on this
branch and is always 0 here, so there is nothing to subtract.

**`"0"` and `"1"` still force serial**, which is how the A/B is run and how
anyone pins it down if it ever misbehaves.

---

# 26. H1: the compiler had NOT hoisted it — −3.7 %, byte-identical, both kernels

*`PORT_WARP_SCAN_SCOPE.md` H1, executed 2026-09-12 on the local RTX 3050.*

## 26.1 The claim, and the reason to doubt it

`Energy()` is called once per candidate and recomputed three quantities that
depend only on `(i, j)` — constant for the whole cell:

```c
const unsigned char type_raw = Ptype(S,pair_,H,nfiles,i,j);   // (i,j) only
const int si1 = unpack(S,H,nfiles,i+1);                        // i only
const int sj1 = unpack(S,H,nfiles,j-1);                        // j only
```

`Ptype` is two `unpack`s plus a `pair_` lookup, and each `unpack` is an index
computation, a global load and a shift — so **five global loads per candidate**
producing identical values.

**H1 was filed at LOW-MEDIUM confidence**, and the reason matters: everything
involved is loop-invariant with `__restrict__` pointers, so nvcc's LICM may
legally hoist all of it. The counter-argument was register pressure — a compiler
short of registers rematerialises rather than keeping values live across a long
loop body.

## 26.2 The SASS settles it for free, and the counter-argument was right

`cuobjdump -sass -arch sm_86`, work loop identified by its backward branch:

| kernel | | loop instructions | **loop `LDG`** | `LDS` | `BAR` |
|---|---|---|---|---|---|
| `int_loop_warp_kernel<2>` | before | 605 | **36** | 0 | 0 |
| | after | 562 (−7.1 %) | **31** | 0 | 0 |
| `int_loop_kernel_64` | before | 571 | **36** | 6 | 2 |
| | after | 533 (−6.7 %) | **31** | 6 | 2 |

**Exactly the five predicted loads, in both kernels.** LICM had not hoisted
them.

*(The same dump incidentally confirms the warp kernel's design claim from a
direction the profiler cannot: `LDS 0` and `BAR 0` in every loop of every
instantiation, against `LDS 6` / `BAR 2` in the twin.)*

## 26.3 The fix: one helper, two kernels

```c
struct cell_inv_t { int type; int si1; int sj1; };
__device__ inline cell_inv_t cell_invariants(...);
```

Both kernels call it once, at the same point — immediately after `maxcol`,
before the work loop — and pass the result into `Energy()`. **One helper so the
two cannot drift**, which is the same reason `build_one()` exists on the host
side.

Byte-identity is by construction: the same values, computed once instead of N
times.

## 26.4 Measured: −3.7 %, with the control flat

ABBA over the two **binaries** (`RNAfold.pre_h1` against the rebuilt one), 60 ×
2400, cooled, `modular_decomp` as the control:

| | `int_loop` | `modular_decomp` (control) |
|---|---|---|
| before | 21.976 / 22.113 → **22.045** | 8.504 / 8.562 → 8.533 |
| after | 21.180 / 21.287 → **21.234** | 8.519 / 8.577 → 8.548 |
| | **−3.7 %** | **+0.18 %** |

The hot run's first pass gave −3.6 % independently, and the warp kernel gives
**−4.8 %** after normalising by its control. *(That run drifted hard — the
laptop's `modular_decomp` went 8.5 → 12.2 s across sixteen arms — which is
exactly why the controls are reported and why the clean number comes from a
cooled ABBA.)*

## 26.5 The interesting part: 14 % fewer loads bought 3.7 %

Removing **five of thirty-six** loop loads (−14 %) and 7 % of loop instructions
returned **3.7 %**. So the kernel is **not instruction-issue bound and not
load-count bound** — which is consistent with, and independent evidence for, the
stall profile: `wait` is a *fixed-latency dependency* stall, and removing work
that was not on the critical path pays back less than its share.

That is worth carrying into H2–H4: **instruction and load counts are a weak
predictor of time in this kernel.** Anything proposed here should be justified by
its effect on the dependency chain, not by how much work it deletes.

## 26.6 Verification

| check | result |
|---|---|
| vs pristine 2.7.2, 9 option arms × **both kernels** | 0 differing lines, `sweeps` asserted on all 18 |
| pre vs post sha: default, warp, int16, chunked, block size 32 and 256 | identical on all six |
| `make check` | 161/161 |

## 26.7 In context

`int_loop` is 12 % of the A100 wall, so −3.7 % is **0.45 % of wall**. H1 is worth
keeping because it is free, correct and helps both kernels — not because it is
significant. The honest ranking after §23.5 has not changed.
