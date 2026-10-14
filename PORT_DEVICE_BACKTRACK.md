# N5: backtracking on the device — scope (signed off)

*Written 2026-10-14, after N5a P3 and before the A100 run of N5a P4. Luke (2026-10-10): after N5a, the
device-side backtracking experiment. Roadmap item N5 (originally C3). Status: **SIGNED OFF 2026-10-14** (Luke: "yes to
all four"): P0 first with the go/no-go of §4; byte-identical including ties; default model first, every other
option refuses to the host; opt-in `RNA_DEVICE_BT=1` until an A100 run decides.*

## 1. What N5 would remove, and why N5a does not settle it

Today each record is backtracked on the host:
1. **Fetch.** `backtrack_fetch_slot()` copies the record's c and fML triangles from the device (fM2 too
   when circular) into a per-worker scratch of about 5 L² bytes.
2. **Trace.** `backtrack_finish_slot()` builds fM1 and f5 from c on the host, then runs upstream's
   `vrna_backtrack_from_intervals()` (`backtrack/bt_*.c`) over those matrices.

At 400 × 5601 on the A100 (N2, phase-synced) that is **8.3 s of trace + 3.3 s of fetch = 26.6 % of the
wall**. N5a exists to hide it: chunk *k* is traced while chunk *k+1* sweeps. The A100 run (P4, Q1) will say
how much it hides. **What N5a cannot change:**

| cost | today | with N5a | with N5 |
|---|---|---|---|
| PCIe traffic | every triangle cell crosses: ≈ 100-125 MB per 5601-nt record, **≈ 45 GB per 400 × 5601 run** | the same, moved off the critical path | the structures only (a few KB per record) |
| host RAM | 5 L² per backtrack worker (≈ 157 MB at 5601; × 12 workers ≈ 1.9 GB); B3's RAM gate sheds workers to fit | the same, **plus** a second retained chunk on the device | O(L) per record |
| host cores | 12 trace workers compete with the builder and the CPU slice (H measures it) | the same competition, spread out | none: cores go to the builder and the CPU slice |
| few-core hosts (Colab T4: 2 vCPU) | the trace is serial-ish and exposed | N5a needs idle cores to hide anything | the device traces |

So N5 is not a speed lever where the host is idle and wide (an A100 with 12 cores and N5a). It **is** a
lever for host RAM (the budgets), for few-core hosts, and for heterogeneous runs, where the CPU slice wants
the cores. These are measurable now, so the plan measures them first (P0) and builds only if they show.

## 2. What exists to build on

- **The retained triangles (N5a P2).** At sweep end, the chunk's c / fML (int32 or int16 + baselines) / fM2
  move into a `rnafold_retired_t` instead of being freed. A device traceback reads exactly these, on its own
  stream, **beside the next chunk's sweep**: the P2 machinery, with a kernel in place of the host fetch.
- **The admission and VRAM accounting (B2, P3 rule (b)).** A retired set is already charged at 0.85 ×
  `rnafold_retired_bytes()`. A device trace changes nothing there: the set lives exactly as long.
- **The fetch/finish seam.** `backtrack_fetch_slot()` is the only part that touches the device; a device
  trace replaces both halves for a record and hands `backtrack_finish` a bp list and an energy.
- **The bars.** Every arm can be compared with the host trace on the same tree (self-comparison), and the
  frozen references (`tools/coverage/refcheck.sh`) catch any drift. The output-file bar covers what is written.

## 3. Design

**3.1 Exactness includes ties.** Upstream's trace takes the **first** decomposition, in its loop order, whose
energy equals the cell's. A random fixture rarely ties; a uniform one ties everywhere (the noLP slot-flow
defect was found only by non-uniform fixtures, and the converse holds here). The device must reproduce the
order, not just an optimal structure: each search evaluates its candidates in parallel, ballots the matches,
and takes the one **earliest in upstream's order**. Bar: byte-identical structures on uniform and G-rich
fixtures as well as random ones, and a negative control that reverses the order and must change the output
on the uniform fixture.

**3.2 One warp per record, the stack in shared memory.** The trace is a stack of intervals; each pop does
one bounded search:
- hairpin: none; stacked pair: one cell;
- internal loop: the (p, q) window under MAXLOOP (≤ 30 × 30 / 2 cells), lanes over p;
- multibranch split and the ML decomposition: O(j − i) over k, lanes over k;
- exterior (f5): O(j) over k.

400 records is 400 warps: a light grid that sits beside the next sweep at low stream priority, like the
build pipeline's work sits beside the GPU today.

**3.3 f5 and fM1 on the device.** The host builds f5 (`vrna_mfe_exterior_f5`) and fM1
(`vrna_mfe_multibranch_m1`) from c today. Both move into the trace kernel's prologue or a small kernel before
it: one row per j, lanes over k.

**3.4 Model coverage by refusal, not by guessing.** S2 covers the default model: dangles 2, linear, no
`--noLP`, no G-quads, no hard or soft constraints, default salt. Every other option makes the chunk **refuse**
and trace on the host as today, with a one-time banner (positive evidence, both ways). Options are added one
at a time after S2, each with its parity bar, in the order they are used: dangles 0, temperature, salt,
`--noLP`, `--noClosingGU`, `--maxBPspan`, circular, G-quads, constraints.

**3.5 Opt-in.** `RNA_DEVICE_BT=1`, off by default, until an A100 run decides (as `RNA_BT_PIPELINE` does).

## 4. Phases

| phase | what | bar / evidence | where |
|---|---|---|---|
| **P0** | **Measure the problem first.** (a) Host RAM and core time of the trace per chunk (RSS with and without backtrack workers; core-seconds); (b) a few-core host simulated with `taskset -c 0,1` on the laptop, N5a on vs off; (c) from the A100 run: what N5a left exposed (T), and what the trace pool cost the CPU slice (H). | pre-registered go/no-go below | laptop + the P4 results |
| P1 | Standalone prototype `tools/proto_device_bt.cu`: dump a chunk's triangles from the host trace, trace them on the device (default model), compare bp lists. Time per record vs length. | identical on random, uniform and G-rich fixtures; the reversed-order control bites on uniform | laptop |
| P2 | In tree behind `RNA_DEVICE_BT=1`: kernel on the retired set's stream, f5/fM1 on the device, structures and energies back; refusal banner for every other option. | refcheck 9/9, bt_pipeline_check, output-file bar, binding, zeroconf; NEGCTL bites; ACTIVE line asserted | laptop |
| P3 | Options one at a time (3.4), each with its parity bar and arm in the option matrix. | per option | laptop |
| P4 | A100 + a few-core host (a 2-vCPU Colab T4, or `taskset` on the A100 VM): off / N5a / N5 / N5 + N5a, at 400 × 5601, 3000 × 1200 and the soak; RSS per arm. | the rule decided at sign-off | Colab |

**P0's go/no-go (proposed):** go to P1 if **any** of: on the 2-core simulation, the trace (with N5a on) is
≥ 10 % of the wall; the trace workers hold ≥ 25 % of the run's peak RSS at the default budget; or the A100's
H section shows the trace pool costing the CPU slice ≥ 5 %. Otherwise N5 is recorded as not needed, with
the numbers, and stops.

## 5. Relations

| pair | relation |
|---|---|
| N5 and N5a | **Reuse, not either/or.** N5 needs N5a's retained set and stream. With N5 on, the host pool has nothing to trace. |
| N5 and the budgets | SYNERGY: the 5 L² worker scratch goes, so B3 stops shedding workers. VRAM: unchanged (the set is already retained). |
| N5 and the CPU slice / router | SYNERGY: the host's cores go to folding. |
| N5 and int16 | SYNERGY: the device reads int16 fML with its baselines in place; there is no widening fetch. |
| N5 and candidate lists (sparse md) | SYNERGY, later: the multibranch split search can walk the candidate list instead of every k (SparseMFEFold). Not in S2: plain order first, for exactness. |
| N5 and N4 (md latency cut) | Independent. |

## 6. Decisions for Luke

1. **Measure first (P0) with the go/no-go above, or build P1 regardless as the experiment?** Recommended:
   P0 first. It costs a day of laptop time, and it decides whether this is worth weeks.
2. **Exactness bar: byte-identical to the host trace, ties included** (3.1). Recommended: yes; it is the
   project's bar everywhere else, and an "optimal but different" structure would break every reference.
3. **Coverage: default model first, every other option refuses to the host** (3.4). Recommended: yes. The
   alternative, all options before any measurement, ports about 4.9k lines of `backtrack/` blind.
4. **Opt-in `RNA_DEVICE_BT=1` until an A100 run decides** (3.5). Recommended: yes, as with `RNA_BT_PIPELINE`.

## 7. Risks

- **Tie order.** Upstream's search order is spread over `bt_internal.c`, `bt_multibranch.c` and
  `bt_exterior_f5.c`, including soft-constraint and auxiliary-grammar hooks. Missing one branch gives a
  structure that is optimal and wrong. P1's uniform fixture and reversed-order control exist for this.
- **Warp occupancy.** One warp per record leaves most of the device idle. That is acceptable beside a sweep
  and costly if the trace runs alone (the last chunk). P1 times it.
- **Divergence from upstream.** A second implementation of the trace must follow upstream's changes. The
  refusal path keeps the host trace as the reference; the parity bars run both.
