# Cross-vendor GPUs and many devices: scope (V0 / V1, Vulkan first)

**Status: SCOPING ONLY (Luke, 2026-10-17).** Nothing here is built. It waits until the CUDA kernels settle,
so that no optimisation has to be made twice. The open CUDA work comes first: N5a's AUTO gate, N5 P3
(options on the device trace), N4 (md latency cut), C4 (CPU fast fold), and whatever the A100 notebooks
surface. **Vulkan first; HIP later; OpenCL and OpenGL compute are not planned.**

| stage | what | gate to start | status |
|---|---|---|---|
| V0 | Inside the CUDA code: a backend seam, a per-device context, the warp width as a parameter, device choice | the CUDA campaigns above are closed or parked; plan signed off | not started |
| V1 | A one-kernel Vulkan prototype: the device trace, exact against upstream on the NVIDIA dGPU and the AMD iGPU | V0's seam is at least drawn; plan signed off | not started |
| V2 | Many workers in one run: dGPU + iGPU + cores, each tracing its own records | V0 done, V1 GO | scoped in outline only |
| V3 | The full Vulkan backend (all kernels) | V1 GO, V2 design signed off | not scoped |
| H | HIP (AMD cluster GPUs: MI250 / MI300) | V0 done | not scoped; "do not forget" |

## 1. Why

- **Reach.** Today the accelerated path needs an NVIDIA GPU. Vulkan compute runs on AMD, Intel and NVIDIA, and on Apple through MoltenVK.
- **Real clusters and workstations are mixed.** A node can carry several GPUs from different generations, sometimes from different vendors, plus many cores. The port today uses device 0 and the cores (the CPU slice). A run that uses every worker it finds, and stays byte-identical, is what lets RNAfold run on a university cluster as it stands.
- **This laptop can test it.** It has three different kinds of worker, which is rare for a dev box:

| | NVIDIA GeForce RTX 3050 Laptop (dGPU) | AMD Radeon Graphics (iGPU) | cores |
|---|---|---|---|
| Vulkan | 1.3.289, NVIDIA driver | 1.3.277, AMD proprietary driver | – |
| subgroup (warp) | 32 | **64** by default, 32 to 64 with subgroup size control | – |
| shared memory | 48 KB | 32 KB | – |
| int64, int16, int64 atomics | yes | yes | – |
| maxStorageBufferRange | 4 GB − 1 | 4 GB − 1 | – |
| bufferDeviceAddress | yes | yes | – |
| memory | 4 GB VRAM | **shared with host RAM** | 12 threads |

(From `vulkaninfo` on Windows, 2026-10-17.) Vulkan here is **native Windows only**. WSL exposes no real Vulkan drivers (Mesa's dzn over D3D12 is the most it gets), so the real AMD and NVIDIA drivers are reachable only from a Windows process. See D1.

## 2. What exists (measured 2026-10-17, LFB c0a85e67)

- **Device code:** 9,151 lines in 6 `.cu` files (modular_decomposition 2,502, hp_mb_loop 2,056, int_loop 1,887, device 1,557, bt_device 704, gquad 445). 3,241 more in the `.inc` / `_dev.h` cell code. **30 `__global__` kernels.**
- **The host code already has a seam, by accident.** The host C (`mfe_cuda.c`, `engine.c`, `fill_arrays*.c`, `budget.c`, `gquad.c`, `mb_loop_fast.c`) makes **zero** direct CUDA API calls. Everything goes through **140 `extern "C"` functions** in the `.cu` files (device 56, modular_decomposition 38, hp_mb 14, int_loop 12, gquad 12, bt_device 8). The leaks are CUDA types in prototypes: `cudaStream_t` in `stub2.h`, `engine.c`, `int_loop_dev.h`, `interior_loopx.h`.
- **Device state is file-scope globals:** roughly 42 in device.cu, 60 in modular_decomposition.cu, 31 in int_loop.cu, 27 in hp_mb_loop.cu, 11 in gquad.cu and 9 in bt_device.cu (streams, events, the graph, device buffers, row tables). That is one device per process. Every multi-device design must replace it first.
- **Warp-width assumptions:** 34 warp intrinsics (`__shfl_sync` 8, `__shfl_down_sync` 6, `__syncwarp` 6, `__shfl_up_sync` 3, `__ballot_sync` 3, `__ffs` 3, `__clz` 2, `__shfl_xor_sync`, `__popc`), plus about 63 lines that hard-code 32 / 31 / `0xffffffff` as lanes, masks or tiles. Among them: the int_loop warp scan and log-32 search, the md `TILE` shuffles, the sparse-md segment scan, Lyngsø lanes, and the device trace's ballots.
- **CUDA-only machinery:** 1 graph (capture + `cudaGraphExecUpdate`, with a reinstantiate fallback), non-blocking and priority streams, events, pinned host memory (`cudaHostAlloc`, with malloc fallback), occupancy queries (`cudaOccupancyMaxPotentialBlockSize`), 1 `atomicMin`.
- **The arithmetic is integer.** Energies are int32 (int16 where AUTO picks it). There is no floating point to diverge across vendors, so byte-identical output across vendors is a realistic bar, not a hope.
- **P1's prototype (`tools/proto_device_bt.cu` + `.sh`)** already does what V1 needs on CUDA. It takes upstream's own c / fML, builds f5 on the device, traces, and compares every f5 cell, MFE and structure. Fixtures are random, A/U-only (ties everywhere) and G-rich, and there is a reversed-order negative control.

## 3. The bar (unchanged, for every stage)

- Byte-identical to upstream's CPU answer (`RNA_GPU=0` pinned in the reference arm), on random, uniform (A/U: ties) and G-rich fixtures. Ties are decided by search order, so a uniform fixture is required (the noLP slot-flow defect was found only by non-uniform fixtures, and the converse holds for ties).
- **Positive evidence** that each claimed worker ran: per-worker record counts that sum to every record, and a banner naming the device and API. A silent fallback to the host is byte-identical and proves nothing.
- **A negative control per new path** that must change the output (P1's reversed search order carries over).
- Wall-neutral for the CUDA path. V0 is refactoring code every user runs: ABBA on the laptop, with the A100 confirming anything within ~5 %.
- The installability contract holds. A bare `./configure` still builds and still accelerates where it can. Vulkan is optional and auto-detected, with its own zero-config arm (`tests/zeroconf_configure.sh` extended).

## 4. V0: inside the CUDA code (benefits CUDA alone; prerequisite for V1–V3 and H)

- **V0a. Name the seam.** Turn the 140 `extern "C"` functions into a backend interface: a table of function pointers or a header of opaque calls, with `cudaStream_t` replaced by an opaque queue handle. CUDA stays the only backend. Bar: the standing bars, exact; wall-neutral.
- **V0b. A per-device context.** Move the file-scope device state into a context struct that every entry point takes. One context per device. Bar: exact, wall-neutral, plus a two-context smoke test on one GPU (two contexts on device 0 fold disjoint halves of a list, and the union equals the single-context run). This alone unlocks multi-GPU CUDA.
- **V0c. The warp width as a parameter.** A compile-time `LANES` (32 on CUDA, 64 or 32 elsewhere) in place of the literal 32s, masks and tile widths, with every scan and search written against it. On CUDA it stays 32. A 16-lane build is a free negative check: it must stay exact even if it is slower, which proves no literal 32 is left. Bar: exact at 32, and exact at 16.
- **V0d. Choose a device.** `RNA_GPU_DEVICE=N` (CLI) and a Python keyword, honoured by the probe, init, teardown and budgets, and defined under `CUDA_VISIBLE_DEVICES`. Bar: positive evidence (the banner names the device), and an out-of-range N refused loudly.

Each of V0a–V0d is its own commit with its bars green. Order: d (cheap) → b → a → c.

## 5. V1: one kernel on Vulkan (the device trace)

**Why this kernel.** It is small (704 lines, 1 kernel), self-contained, already has an exact standalone bar (P1), and uses the hard parts: ballots, a per-record stack in global memory, block barriers and int16 / int32 tables. It answers the porting questions without porting the sweep.

**Shape.** A standalone tool, like P1, not wired into RNAfold:
1. **WSL side (existing build).** P1's harness gains a `dump` mode. It writes each record's c, fML, sequence encoding and the parameter structs (exactly as `load_param()` / `load_param2()` build them) to a file, plus upstream's f5 and structures as the reference.
2. **Windows side (new).** `tools/vk_device_bt/`: a Vulkan compute host program plus the trace shader. The shader is GLSL compiled to SPIR-V at build time (D2), with subgroup ops, `GL_EXT_shader_explicit_arithmetic_types_int64` for every triangle index (the 2^32 grid-index lesson), and bufferDeviceAddress so no binding is capped at 4 GB. It reads the dump, traces on a chosen device, and writes f5 and structures.
3. **WSL side.** Compare every f5 cell, MFE and structure with upstream's. Run on the dGPU, on the iGPU at subgroup 64, and on the iGPU forced to subgroup 32.

**Pre-registered questions and pass criteria:**

| Q | question | pass |
|---|---|---|
| Q1 | exact on NVIDIA under Vulkan? | every f5 cell / MFE / structure equal; random, A/U, G-rich; NEGCTL bites on A/U |
| Q2 | exact on AMD at subgroup 64 **and** 32? | the same, both widths; this is V0c's design tested on hardware |
| Q3 | the toolchain is installable? | SPIR-V built by the build system and embedded in the binary; no runtime shader compiler; the build works with the Vulkan SDK absent (Vulkan arm simply off) |
| Q4 | Vulkan vs CUDA on the same dGPU | trace time per record vs P1's CUDA numbers, same fixture; **reported, not judged** (a gap here is a property of one kernel, Q7's lesson) |
| Q5 | iGPU worth having as a worker? | trace time on the iGPU vs the dGPU and vs the host trace on the cores; reported. It feeds V2's split, not a verdict |
| Q6 | Windows' GPU watchdog (TDR, ~2 s)? | no dispatch near it at 5601 nt on the iGPU; record the longest dispatch |

**GO for V2 / V3** if Q1–Q3 pass. Q4–Q6 shape the design and do not gate it.

## 6. V2 in outline: every worker in one run

- **Shape.** The chunk loop already splits a list into independent batches (`vrna_mfe_batch()`, VRAM budgets). V2 gives each device its own context (V0b), its own chunk sizing and budget, and its own trace. The CPU slice stays as it is. Records go to whichever worker is free. Output order is restored as `-j` and the pipeline do today.
- **The iGPU's memory is the host's.** Its "VRAM" comes out of the same RAM the RAM budget counts. Budgets B1–B4 must not count it twice or forget it.
- **Bars.** One sha across every mix (dGPU only, iGPU only, cores only, dGPU+iGPU, all three). Per-worker record counts sum to every record. Each side traces its own records (P2d's rule, generalised). A worker that fails or is refused mid-run hands its records back, and the count says so.
- **The laptop is the test rig** (dGPU + iGPU + 12 threads). On CUDA alone, multi-GPU needs hardware we don't have locally. Two contexts on one GPU (V0b's smoke test) cover the mechanics until then.

## 7. Risks

- **Double work.** The reason this waits (gate). Every kernel change after V3 must land in two backends and stay exact in both. V0's seam and `LANES` reduce this, but cannot remove it.
- **Subgroup 64.** AMD's default ballot is 64 bits wide, and scans and searches change depth. V0c must make this a parameter, not a rewrite.
- **No graphs, no stream priorities.** Vulkan's reusable command buffers fit the per-row graph well. Stream priority (the backtrack pipeline's low-priority stream) maps only to queue priorities, which are coarser.
- **Windows-only local drivers.** V3 on this laptop means a native Windows build of RNAfold (MSYS2 / MinGW, as upstream's Windows builds use). Clusters run Linux with RADV, AMDVLK or NVIDIA drivers. Both need a bar, and only one is local (D1).
- **The 4 GB binding cap.** maxStorageBufferRange is 4 GB − 1 on both GPUs here. Large triangles go through bufferDeviceAddress (64-bit pointers) or split bindings.
- **TDR.** Long dispatches on Windows can lose the device. Row launches are short, but a trace block at 5601 nt on an iGPU is unmeasured (Q6).
- **Performance is unknown.** Vulkan on NVIDIA may trail CUDA. V1 reports the gap on one kernel and does not extrapolate.

## 8. Relations

- **HIP (H).** HIP shares V0 entirely: the seam, the context, `LANES` (64 on CDNA). After V0 it is mostly hipify plus the 64-lane path V1 has already exercised on AMD. It targets AMD cluster GPUs, which Vulkan can also reach, but HIP keeps CUDA's graphs and streams.
- **Multi-GPU CUDA** falls out of V0b + V0d + V2 with no Vulkan at all.
- **C4 (CPU fast fold)** helps every worker mix, including the GPU-less one. It is independent of all of this.
- **N5 (device trace)** is V1's subject. P3's option work changes the kernel V1 ports, which is one more reason V1 waits for P3.

## 9. Decisions for Luke

- **D1. Where V1 runs.** (a) The split: WSL dumps and checks, a native Windows tool traces (recommended: real drivers, no ViennaRNA build on Windows). (b) Everything native on Windows (MSYS2 build of the tree first). (c) Linux on bare metal (dual boot) with RADV and NVIDIA.
- **D2. Shader language.** GLSL → SPIR-V via glslang (recommended: ubiquitous, in every Vulkan SDK and distro), or Slang (closer to CUDA C++, one more toolchain to install).
- **D3. Order.** V0d → V0b → V0a → V0c → V1 (recommended), or V1 first as a pure prototype before any refactor (faster answer, but its code is thrown away).
- **D4. What V2 proves first.** The iGPU + dGPU + cores mix on this laptop (recommended: the hardware is here), or two CUDA contexts on one GPU first (mechanics only).

## 10. Recommendation

Keep this scoped until the CUDA campaigns close. When they do, run V0 in the order of D3: each part pays for itself on CUDA (device choice, multi-GPU contexts, a `LANES` check that is a free correctness probe). Then V1 on the device-trace kernel, split per D1(a), with Q1–Q3 as the GO bar. V2 then proves the case that matters for clusters: every worker in one run, one sha, every record accounted for, which this laptop can test end to end. HIP comes after V0, sharing all of it.
