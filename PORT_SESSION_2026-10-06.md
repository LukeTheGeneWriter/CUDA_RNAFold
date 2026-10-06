# Session 2026-10-05/06: handoff

**Where to start next:** you are on **`Lukes_Flow_Batching`** (LFB). Read, in order:
1. this file;
2. `PORT_SPARSE_MD.md` (C1, finished: S0–S5, N1, N2 results);
3. the two plans awaiting sign-off, `PORT_BACKTRACK_PIPELINE.md` (N5a) and `PORT_LYNGSO_INTLOOP.md` (N3).

Luke's call at the end of the session (2026-10-06): *"write N5a. Once that plan is written, move to N3. Record
where we are in the process extensively: this is an incredibly productive session."*

---

## 1. Headline

**LFB passed the pre-registered promotion bar against FP on the A100, by a wide margin:**

| | FP (`209a978d`) | LFB (`8d3deef3`) | change |
|---|---|---|---|
| 400 × 5601 | 83.64 s | 43.68 s | **−47.8 %** |
| 3000 × 1200 | 21.29 s | 17.64 s | **−17.1 %** |
| 2400-record soak | 151.0 s | 94.0 s | **−37.7 %**, outputs agree |

Luke will tell TBI that LFB is looking better. Making LFB the default branch needs "extensive writeups and
tests", but "that's def coming". Against upstream 2.7.2, LFB was ~60× on the 10-02 three-way, before sparse md.

## 2. What landed (all pushed unless marked)

| commit | what | bar |
|---|---|---|
| `e0174012` | `PORT_SPARSE_MD.md`, the C1 plan, with Luke's decisions | — |
| `527ecddb` | S0: `tools/md_sparse_equiv`, the candidate rule on upstream's matrices | 22/22 option cases, both negctls red |
| `2e48c6ad` | S1: candidate lists on the device, checked beside the dense md | selftest 0, negctl red |
| `99c4b984` | S2: `RNA_MD_SPARSE=1` replaces the dense md (candidates + row scan) | triangles + parity |
| `bb7101ec` | **2^32 grid-index fix** (`(size_t)blockIdx.x*blockDim.x`), LFB; FP `209a978d` | confirmed at scale (§3) |
| `532a7826` | S3: 4-B packed entries (delta chain), lists charged to admission | both formats exact; alloc ≤ charge; halved-charge negctl |
| `b1c3f66f` | S4: laptop matrix | §3 |
| `7555a713` | S5 notebook described | — |
| `f2f2af2d` | S5 A100 results; **blocked md driver RETIRED**; 8-B entries chosen | §3 |
| `3ecb4276` | **N1: sparse md ON by default**; `RNA_MD_SPARSE_LANES` | bars 6/7, parity, matrix 36/36, gpu_cli, binding 20/20 |
| `8d3deef3` | **N3 S0: `tools/lyngso_equiv`**, Lyngsø exact incl. hard constraints | 0 differ ×9 cases, both negctls red |
| `0e39f8cd` | N2 A100 results | §3 |
| *(this commit)* | **int16 AUTO** (VRAM limit + DRAM pressure); **CUDA graphs ON at overlap 2**; N5a and N3 plans; this record | §4 |

## 3. What the A100 runs settled

Four notebooks ran on A100-SXM4-80GB:

- **`CUDA_RNAFold_SparseMD.ipynb` (S5):**
  - sparse md exact everywhere; 400 × 5601 **−29.3 %** (60.99 → 43.09 s), md(P) 2.45×; soak −18.8 %;
  - blocked md (b1024x64L16) was +19 % / +7.6 % against the row path, so **retired**;
  - 8 B beat 4 B everywhere, and the 4-B lists' 47 % VRAM saving bought **zero** chunks.
- **`CUDA_RNAFold_Confirm2p32.ipynb`:**
  - **run 1 stopped at its own guard.** The marker grep `(size_t)blockIdx.x*blockDim.x` matched the pre-fix
    tree too, because some sites were already cast. The guard now counts **uncast** sites: FPold 13, heads 0;
  - **run 2: all 8 rows MET.** FPold differs from upstream on 6/11 and 12/33 boundary records; FP, LFB and
    LFBu equal upstream and each other on every record. **The 2^32 fix is confirmed at scale on both
    branches.**
  - Side finding: **the chunk cap beats one wide chunk.** LFB capped 61.0 s, uncapped 74.8 s; soak 116.4
    against 134.6 s.
- **`CUDA_RNAFold_N2.ipynb`:**
  - **G** gates PASS (10,076 natural Rfam seeds exact);
  - **P** promotion bar MET (§1);
  - **K1** int32 now faster than int16 (−1.3 % / −2.7 %);
  - **K2** graphs at overlap 2 −0.5 % / −2.2 %;
  - **K3** cap 1× is best (uncapped +32 %);
  - **L** lanes length-dependent (L8 −5.5 % at 1200, +2.6 % at 5601);
  - **W** worst-case repeats: sparse still −5 %;
  - **N** E. coli genome windows −28.7 %;
  - **E** `int_loop` issue-bound (SM 62 %, DRAM 5 %).
  - New: **backtracking is exposed, 26.6 % of the wall** (8.3 s trace + 3.3 s fetch at 400 × 5601).

## 4. Defaults changed this session (LFB)

| knob | was | now | why |
|---|---|---|---|
| `RNA_MD_SPARSE` | dense md | **sparse md** (8 B, rho 0.25) | S5 + N1 bars; `=0` restores dense |
| `RNA_FML_INT16` unset | int16 always (auto only stepped aside on conflicts) | **AUTO by VRAM limit and DRAM pressure** | N2 K1. Luke: "make the int32/16 switch based on VRAM limitation and DRAM pressure" |
| `RNA_CUDA_GRAPH` unset | off at overlap 2 | **on at every level** | N2 K2. Luke: "We can do cuda graphs at level 2" |

**int16 AUTO, exactly.** Decided once, on the first `compute_gpu_usable_bytes()`, which both admission paths
call before pricing a record. int16 is chosen if any of these holds, in order:
1. the dense md will run (sparse off or refused): **DRAM pressure**;
2. there is no cell cap, as on the library path, where VRAM sizes the chunks;
3. a full-cap chunk would not fit in VRAM as int32, priced at the worst bytes per cell of 600/2000/5601 nt.

Otherwise int32. Explicit `RNA_FML_INT16=0|1` wins. RNAfold passes its cap via `rnafold_set_chunk_cells_cap()`.
Laptop evidence: the default picks int16 (a full-cap chunk is 8.2 GB as int32 against 2.7 GB usable). A small
cap picks int32, and the triangle checksum says int32. All arms equal the CPU. On the A100, a full-cap chunk is
8.2 GB against ~68 GB usable, so int32. **Bars on this change: see §8.**

## 5. Decisions Luke made this session

- 10-05 roadmap ("excellent ×3"); C1 plan decisions (both entry formats measured; capacity measured; refusal
  list; retire the blocked driver only after a test at scale; no pre-registered threshold).
- "go ahead with S4"; "build the S5 A100 notebook. Note the latency cut possibility for later".
- "push LFB and then re-organize our current roadmap".
- "Start N1 now and then write N2 … The A100 is expensive per hour so we should really make the most of it."
- "Run the exactness test and push n1".
- 10-06 (this turn): int16 AUTO, graphs at level 2, push the N2 write-up, write N5a, then N3, record
  everything.

## 6. Traps found (and their lessons, in memory)

1. **A guard marker must be the fix's signature.** The Confirm2p32 grep matched a form that predated the fix.
   Count what the fix *removed*.
2. **A reference arm must pin every knob it stands for.** Flipping sparse md's default turned every "dense"
   arm that left `RNA_MD_SPARSE` unset into sparse against sparse (the S3 bar, the S5 `dflt` arm). Generalises
   "a CPU reference must say `RNA_GPU=0`".
3. **4-B entries: a delta from a segment baseline overflowed int16** in ~300 columns at 1700 nt (32
   candidates span hundreds of rows). The fix: a step from the previous entry, plus early segment close.
4. **Lyngsø under constraints:** pair type 0 on an allowed pair is **7**, not "no pair". The canonical table
   gave 105 false diffs next to enforced pairs.
5. **A failed build let the next stage run on the old binary** (N1, first attempt): bars 6/7 "passed" on a
   build where sparse md was not yet the default. The wrapper now stops when the build part fails.
6. **The chunk cap is faster than wide chunks** (LFB), overturning the FP-era "widen chunks" lesson. VRAM
   savers buy no A100 speed while the cap binds.

## 7. Where the time goes now (400 × 5601, A100, phase-synced, 49.1 s)

| phase | s | lever |
|---|---|---|
| `int_loop` | 16.5 | **N3 Lyngsø** (plan ready) |
| md (sparse) | 12.6 | N4: per-row lanes, fused launches, warp scan |
| backtrack trace + fetch | 8.3 + 3.3 | **N5a pipelining** (plan ready) |
| `hp_mb` + `load_my_c` + exposed build | ~6 | later |

N3 + N5a + N4 could take 400 × 5601 from ~43.6 s to the mid-20s. That is an estimate, to be measured.

## 8. Open at the end of the session

- **Bars on the int16-AUTO + graphs change:** running on the laptop as this was written. Two passes of
  parity, matrix, gpu_cli budgets and binding: one where AUTO picks int16, one with a small cap so it picks
  int32. Plus the S3 sparse bar. Results are appended in §8.1 below when they land.
- **Plans awaiting sign-off:**
  - `PORT_BACKTRACK_PIPELINE.md` (N5a): decisions on the public seam, the AUTO default, P3 timing and the
    thread split;
  - `PORT_LYNGSO_INTLOOP.md` (N3): decisions on the two-kernel design, refusals, the S5 ring-K grid, and order
    against N5a (suggestion: N3 S1–S3 first).
- **Not started:** N4 (adaptive lanes etc.), C4 (CPU fast fold), C5 usability items. Remaining decisions:
  GitHub default branch (`main` predates the port); salt int16 stand-down; MIN_GPU_BATCH router.

### 8.1 Bars on the int16-AUTO + graphs change

**All green**, laptop RTX 3050, `~/pyq` at the int16-AUTO + graphs build:

| bar | laptop default (AUTO picks int16) | `RNA_GPU_CHUNK_CELLS=2e7` (AUTO picks int32, more chunks) |
|---|---|---|
| `verify_option_parity` | option surface = CPU | option surface = CPU |
| `verify_option_matrix` | 36/36 | 36/36 |
| `verify_gpu_cli` budgets 4/8/16/32 MB | byte for byte | byte for byte |
| Python binding suite | 20/20 | 20/20 |

**The S3 sparse bar under the new defaults, dense arm pinned:**
- all 34 cases equal the dense md and the CPU;
- the triangles differ only where expected: `C-enforce`'s near-INF artefact, and the NEGCTL lines;
- selftest 0 mismatching in both formats; NEGCTL 3.42 M.

**Positive evidence for AUTO** (`auto16.sh`):
- each decision fired as designed: 4 GB default → int16, small cap → int32, dense md → int16, uncapped →
  int16, explicit settings respected;
- the int32 arm's triangle checksum is labelled int32, so the layout really changed;
- graphs show "enabled" by default and "disabled" under `RNA_CUDA_GRAPH=0`.

## 9. Artifacts

- **Notebooks** (repo root, gitignored): `CUDA_RNAFold_SparseMD.ipynb`, `CUDA_RNAFold_Confirm2p32.ipynb`
  (guard fixed), `CUDA_RNAFold_N2.ipynb`.
- **Generators and smoke harnesses** (session scratchpad `4ccd78d3-…`): `make_nb_sparse.py`, `smoke_sparse.py`,
  `make_nb_confirm.py`, `make_nb_n2.py`, `smoke_n2.py`.
- **Laptop bar scripts** (same scratchpad): `s3_test.sh` (sparse md matrix, dense arm pinned), `n1_bars.sh`
  (evidence, bars 6/7, default-path bars), `bars2.sh` (both int16 modes), `auto16.sh`, `lanes_run.sh`,
  `exact.sh`.
- **In-tree exactness bars:** `tools/md_sparse_equiv.sh`, `tools/lyngso_equiv.sh`. Run both against
  `~/stock272` and the LFB build.
- **Laptop build loop:** WSL `~/pyq` (sync list `pyq_changed.txt`, `pyq_sync.sh`), fixtures `~/s3w`, `~/n1w`.
