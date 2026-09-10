# The stress profile: the residual is gone, and the wall is now a HOST problem

*400 × 5601 nt, **NVIDIA L4 at 2040/2040 MHz, ratio 1.00, no throttle**, 23 034 MiB,
commit `82fdbb07`. Six arms: {int32, int16} × {natural, half, quarter} VRAM budget.
Raw: `stress272.json`, notebook `CUDA_RNAFold_Stress272.ipynb`. Run 2026-09-09.*

> **LATEST: read §16 first.** Five runs on, the wall at 400 × 5601 is
> **394.8 s** (was 641.8 on 2026-09-09), int16 and the build pipeline
> **compose**, and the top open performance item is int16's **34.8 s give-back**
> in `hp_mb` (+27.1) and `fetch_mx` (+7.7) -- but read §17 before acting on
> that split: `hp_mb`'s timer is charged ~6x its own GPU time, so the
> `hp_mb` half of the give-back is very likely attribution, not a
> regression. §1–§14 are kept in run order.

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

## 17.4 What is NOT settled, and why this box cannot settle it

Under `RNA_PHASE_SYNC` locally, `int_loop` reads +1.358 s (+20 %) under int16.
**That is not a result.** The four sync arms walled 16.46 / 17.22 / 18.59 /
21.21 s in run order — a monotonic climb of ~1.6 s per position as the laptop
GPU throttled, which is larger than the effect and runs the same direction. ABBA
plus min-of-pair does not rescue a drift that big. All it says is that the
question is open and belongs on a card that holds its clock.

**The single follow-up that settles it: one pair of arms at 400 × 5601 on a T4
with `RNA_PHASE_SYNC=1`, int32 and int16.** Compare splits between two arms both
run with it — never a split from a sync run against one without, and never the
wall of a sync run against anything, because destroying the phase overlap costs
~9 % of wall by construction.

**Do not optimise `hp_mb` until that run exists.** On the numbers here it is
~6 % of GPU time, not the 20–25 % of wall the async profile shows, and the work
would land on the wrong kernel.

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
