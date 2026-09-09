# The stress profile: the residual is gone, and the wall is now a HOST problem

*400 × 5601 nt, **NVIDIA L4 at 2040/2040 MHz, ratio 1.00, no throttle**, 23 034 MiB,
commit `82fdbb07`. Six arms: {int32, int16} × {natural, half, quarter} VRAM budget.
Raw: `stress272.json`, notebook `CUDA_RNAFold_Stress272.ipynb`. Run 2026-09-09.*

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
