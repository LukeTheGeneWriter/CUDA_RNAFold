# `modular_decomposition_kernel` profile — the int16 question, settled

*Measured 2026-09-08 on an **NVIDIA L4 at 2040/2040 MHz, 48 °C, no throttle** —
a healthy machine, unlike the T4 that ran benchmark v5. Commit `c2e77b56`,
16 × 2000 nt, 5 launches sampled from the middle of a 1996-launch sweep.
Raw: `profile272.json`. Notebook: `tools/make_nb_profile272.py`.*

| metric | int32 | int16 | i16/i32 |
|---|---|---|---|
| `dram__bytes_read.sum` | 39 318 784 | 24 527 872 | **0.624** |
| `dram__bytes_write.sum` | 159 488 | 563 | 0.004 |
| `gpu__dram_throughput` % of peak | **89.34** | **80.85** | 0.905 |
| `sm__throughput` % of peak | 23.53 | 46.14 | **1.961** |
| `l1tex__t_sector_hit_rate` % | 59.60 | 75.85 | 1.273 |
| `lts__t_sector_hit_rate` % | **6.13** | **9.41** | 1.533 |
| `gpu__time_duration.sum` (ns) | 147 533 | 101 306 | **0.687** |

---

## 1. The kernel is 1.46× faster under int16 — on a healthy GPU

`gpu__time_duration` falls to 0.687 of the int32 figure: **1.456× faster.**

That is the number benchmark v5 could not see. v5 measured **1.009×** end to end
on a **T4 clocked at 585 MHz against a 1590 MHz maximum**. The encoding was
never the problem there; the machine was.

## 2. `fml_j` was never the whole stream — and now we know its share

DRAM reads fall to **0.624**, not the 0.500 a halved stream would give. That
single number pins the composition. If `f` is `fml_j`'s share of the read
traffic, halving it gives `(1 − f) + f/2`, so

```
1 − f/2 = 0.624   →   f = 0.752
```

**`fml_j` is ~75 % of the kernel's DRAM read traffic**; the remaining ~25 % is
other streams int16 does not touch. This was the first of the two competing
explanations `PORT_GQUAD_SPEC.md` and the v3/v4/v5 notebooks kept carrying, and
it is now measured rather than guessed.

## 3. The kernel did **not** leave the DRAM roof

The competing explanation is refuted. DRAM throughput goes 89.3 % → **80.9 % of
peak**: still firmly bandwidth-bound. int16 did not convert the kernel into a
compute-bound one, it just gave it less to read.

So there is **no second lever hiding behind int16**. The kernel is at its DRAM
floor in both encodings, exactly as the earlier work concluded, and the way to
go faster is to move fewer bytes — not to restructure the compute.

## 4. The mechanism behind "int16 is machine-dependent", confirmed

`sm__throughput` **doubles**, 23.5 % → 46.1 %. That is the decode cost: int16
trades ALU for bandwidth. On this machine it is free, because SM sits at 46 %
while DRAM sits at 81 % — the ALU work hides completely under the memory system.

Starve the SMs and that stops being true. On a part clocked at 37 % of its
maximum, a 46 % SM utilisation scales toward and past the DRAM figure, the
decode stops hiding, and int16 turns into a loss. That is precisely the
0.71× measured on a thermally-limited RTX 3050, and it is the best available
explanation for the T4's 1.009×.

**int16's value is a property of the machine, and the profile now says which
property: whether the SMs are fast enough to hide the decode.**

## 5. The rider: L2 does **not** catch the `fML` column re-reads

`lts__t_sector_hit_rate` is **6.1 %** (int32) and 9.4 % (int16). Very low.

The proposed optimisation — staging the per-row `fML` column re-reads in shared
memory across a batch of rows — was worth checking against the hardware first,
because if L2 already caught them the work would duplicate what the cache does
for free. **It does not.** That optimisation is a live lever, and given §3 says
the kernel is bandwidth-bound in both encodings, it is now the *most* promising
one: it attacks bytes moved, which is the binding constraint.

L1 tells the complementary story: 59.6 % → 75.9 %, exactly what packing two
values per 32-bit slot should do to a cache line.

## 6. What this changes, and the one thing still open

**Changed.** int16 is not a marginal 1.0–1.1× trick. On a GPU whose SMs can hide
the decode it is a **1.46× kernel win** and a 48 % VRAM saving. The case for
defaulting it on is much stronger than v5 suggested — for L4-class hardware.

**Still open.** A 1.456× kernel does not obviously reconcile with v5's 1.009×
end-to-end, even allowing for the T4's clocks. If `modular_decomposition` is
~73 % of wall, Amdahl predicts `1 / (0.27 + 0.73 × 0.687) = 1.30×` overall — so
either the T4 really was starved enough to erase all of it, or that 73 % share
does not hold at 400 × 5601 nt.

**The cheap test that separates them:** re-run only benchmark v5's C and E arms
on an **L4**. If int16 comes back near 1.3× there, the T4 was the anomaly and
int16 should default on for this class of hardware. If it comes back near 1.0×
on a healthy L4 too, then the kernel is a smaller share of wall than we think at
that size, and the next question is what the other 70 % is doing.

Do not draw the "default it on" conclusion from this document alone: it measures
one kernel on one machine, and the end-to-end number is the one users feel.
