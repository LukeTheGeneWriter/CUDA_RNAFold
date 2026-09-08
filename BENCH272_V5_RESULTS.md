# Benchmark v5 — 400 × 5601 nt, measured 2026-09-08

*Raw: `bench272_v5.json`. Notebook: `tools/make_nb_bench272_v5.py`.*

**Tesla T4** (not the L4 the workload was sized for), Colab, commit `10acb583`.

---

## 1. The headline: 32.4× over upstream

| arm | wall | vs upstream A |
|---|---|---|
| **A** upstream 2.7.2 | 25633 s = **7.12 h** *(extrapolated)* | 1.00× |
| **B** port, no CUDA | 25885 s = 7.19 h *(extrapolated)* | 0.99× |
| **C** int32, 5 chunks | **791.9 s** | **32.37×** |
| **E** int16, 4 chunks | **784.8 s** | **32.66×** |
| Cm | 795.9 s | 32.21× |
| Em | 787.4 s | 32.55× |

Two results worth separating:

- **32.4× at this size**, against 8.06× / 5.65× / 7.88× at the sizes v1–v3 could
  afford. The batch machinery only has room to work when there are enough
  records to fill it; every earlier number was measuring a workload too small to
  show it. This is the number the chunking work was for.
- **A/B = 0.9902.** The port costs upstream's own CPU path nothing — confirmed
  here at ten times the workload of the first timing. That matters more for
  upstreaming than the speedup does: it is the evidence that the accelerator is
  additive rather than a fork of the CPU code.

Per-record CPU cost: **64.09 s** (A) / 64.51 s (B) at 5601 nt.

## 2. int16 is a wash on this machine: C/E = 1.009×

The VRAM saving is real and visible — int16 packed 400 records into **4 chunks
where int32 needed 5**. It bought **0.9 % of wall.**

| machine | regime | int16 |
|---|---|---|
| RTX 3050 @ 1057 MHz | bandwidth-bound | **1.31×** |
| RTX 3050 @ 712 MHz | SMs starved | **0.71×** |
| bench v3 | — | 1.12× |
| **T4, this run** | **?** | **1.009×** |

This is consistent with int16's value being a property of the *machine*, not the
code, and is further evidence against defaulting it on.

**Caveat that the profile must settle:** clocks read **585 MHz against a
1590 MHz maximum** (77 °C, no throttle flag set). If that is a sustained
downclock rather than a post-run idle sample, the SMs were starved — precisely
the regime where int16 is *expected* to lose. Use `CUDA_RNAFold_Profile272.ipynb`.

## 3. `valid: False` is an artifact — read the notes before believing it

The only two failures were:

```
FAIL  arm Cm: RNA_MIN_GPU_BATCH never announced -- override not applied
FAIL  arm Em: RNA_MIN_GPU_BATCH never announced -- override not applied
```

Colab clones `origin/port27`, which was at `10acb583` — three commits behind the
work, and `RNA_MIN_GPU_BATCH` was added in `130e2c8a`. The override could not
exist. **`C/Cm = 0.995` corroborates it**: the two arms that should have differed
are the same configuration, agreeing to within noise.

**Operationally: push before running, or the notebook benchmarks a tree nobody
is working on.** The gate did its job — it refused to certify a run whose arms
were not what they claimed.

## 4. v5's cheaper CPU baseline, validated

The point of v5 was to stop buying the CPU curve four times over.

| | folds/arm | points on `t(n)` |
|---|---|---|
| v4 | 115 | 4 |
| **v5** | **47** | **40** |

**94 CPU record-folds against v4's 230**, for the same 10× extrapolation reach.
And the substitution was checked rather than assumed:

- a model built **only from stream timestamps** predicted genuine standalone
  runs within **0.96 % (A) / 1.05 % (B)** — the one check that catches
  "`t(k)` off a stream" not being a whole-run cost;
- every held-out point n=21..40 within **0.38 % / 0.53 %**, R² = 0.999992;
- no stall (largest record 1.02× the median);
- CPU throughput stable to **1.26 % / 0.26 %** across the run, against a stated
  noise floor of 0.51 % / 0.70 %.

## 5. The profiling cell measured nothing, four ways

All four silent, and all four now fixed in `tools/make_nb_profile272.py`:

1. the probe held **6 records against a `MIN_GPU_BATCH` of 10**, so the batch
   folded entirely on the CPU and **no kernel ever launched** — `==WARNING== No
   kernels were profiled`, under a screenful of normal folds. Third appearance
   of that constant as a trap;
2. **`ncu --csv` and RNAfold both write to stdout**, so the `.csv` held
   dot-bracket structures and zero metric rows. Needs `--log-file`;
3. that warning is on **stdout**, so the gate grepping `stderr` never fired;
4. `--launch-skip 2000` was hardcoded with nothing checking the kernel launches
   that many times.

Inherited from v4, whose probe used a 5-record file and was equally broken —
unnoticed because `RUN_NCU` defaults to `False`. A cell that had never run.
