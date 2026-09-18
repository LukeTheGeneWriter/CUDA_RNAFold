/*
 *  The per-record megakernel -- stage 0.
 *
 *  Plan: ~/.claude/plans/composed-puzzling-lemur.md, scope:
 *  PORT_MEGAKERNEL_SCOPE.md. What follows is the SKELETON: one cooperative
 *  launch per record, persistent blocks, the row loop inside the kernel, and a
 *  grid barrier between phases. It runs today's arithmetic unchanged -- every
 *  phase calls the same *_cell()/fml_scan_block() the standalone kernels call,
 *  out of the shared headers -- so this stage buys no speed. It buys the
 *  structure, and it is expected to be SLOWER than the per-phase path.
 *
 *  WHY FUSE AT ALL. Launch overhead is worth ~2 s of an 87 s fold; that is not
 *  the reason. The reason is that `modular_decomposition` re-reads a whole fML
 *  column on every row with no reuse inside the row (46.8 TB at 400 x 5601), and
 *  L2 cannot hold that data across rows because the OTHER phases stream `c`
 *  through the same cache between launches -- measured: 40.8 % L2 hit at a shape
 *  whose entire live set fits in 40 MB of L2. Only a resident kernel can keep a
 *  block's own slice of that data in shared memory ACROSS rows, and only a fused
 *  chain makes the row-tiling reuse legal at all (md(i-1) needs row i-1 of fML,
 *  which the whole six-kernel chain produces).
 *
 *  WHAT THIS FILE IS NOT, YET. Stages 1-5 add the int16 baseline, the `c` window
 *  in shared memory, the fML corner cache, the two co-resident halves with the
 *  one-row skew, and the streaming admission queue. None of that is here.
 *
 *  THE GRID BARRIER IS THE FIRST NUMBER. Seven phases x ~5601 rows is ~39 k
 *  grid.sync() calls per record. At 5 us that is 0.2 s; at 50 us it is 2 s and
 *  the design is in trouble. The device-side phase clocks below exist to answer
 *  that, because RNA_PHASE_SYNC and ncu's per-kernel attribution both stop
 *  meaning anything once the phases are one kernel -- and an unmeasurable fast
 *  path is how this project once shipped a 1.72x regression as its default.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <assert.h>

#include "ViennaRNA/utils/basic.h"
#include "ViennaRNA/params/default.h"
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/params/basic.h"

#include "interior_loopx.h"
#include "stub2.h"
#include "gquad_dev.h"
#include "megakernel.h"

/* The device-side math, shared with the standalone kernels. One copy only:
 * these recurrences produce a plausible structure when they drift, not a
 * crash, so the fused path must not own a second version of any of them. */
#include "int_loop_dev.h"
#include "hp_mb_dev.h"
#include "md_dev.h"
#include "md_cell.inc"
#include "md_chain_cells.inc"
#include "int_loop_cell.inc"
#include "int_loop_cells.inc"
#include "hp_mb_cells.inc"
#include "fml_scan_block.inc"

namespace cg = cooperative_groups;

/* One block size for every phase, which is a real constraint of fusing them:
 * int_loop wants one warp per cell, md runs TILE=32 lanes per cell, and
 * fml_scan is a block-wide scan whose tile width is the block. 256 = eight
 * warps satisfies all three (eight cells per block for the two warp-shaped
 * phases, a 256-wide scan tile for the third). */
#define MK_BLOCK   256
#define MK_WARPS   (MK_BLOCK / 32)
#define MK_TILE    32          /* lanes per cell in int_loop and md */

/* Phase clocks. clock64() is a per-SM cycle counter, so these are summed over
 * blocks and only ever compared with each other -- a share of the row, not a
 * wall time. Block 0 alone writes, to keep the atomics off the critical path. */
#define MK_PH_INT_LOOP 0
#define MK_PH_HP_MB    1
#define MK_PH_NEW_C    2
#define MK_PH_LOAD_C   3
#define MK_PH_SCAN     4
#define MK_PH_LOAD_FML 5
#define MK_PH_MD       6
#define MK_PH_TAIL     7
#define MK_PH_SYNC     8
#define MK_PH_N        9

#define MK_TICK(slot)                                                     \
  do {                                                                    \
    if (clocks && (blockIdx.x == 0) && (threadIdx.x == 0)) {              \
      const long long _now = clock64();                                   \
      clocks[slot] += (unsigned long long)(_now - _mark);                 \
      _mark = _now;                                                       \
    }                                                                     \
  } while (0)

/*
 *  One record's whole sweep.
 *
 *  Blocks stride over the row's cells; the grid is sized so that every block is
 *  resident, which is what makes grid.sync() legal. `H` is the record's slot,
 *  and every buffer below is the batch-flattened one it already lives in -- the
 *  record's own region is picked out by the offset tables exactly as the
 *  per-phase kernels do it. Stage 0 changes WHERE the work is issued from, and
 *  nothing about what it computes.
 */
__global__ void
megakernel_record(const rnafold_mk_ptrs_t p,
                  const int nfiles, const int H, const int turn_, const int length,
                  const int i_top, const int noGUclosure,
                  unsigned long long *clocks)
{
  cg::grid_group grid = cg::this_grid();

  const int       lane   = (int)(threadIdx.x & 31u);
  const int       wib    = (int)(threadIdx.x >> 5);
  const long long gwarp  = (long long)blockIdx.x * MK_WARPS + wib;
  const long long nwarps = (long long)gridDim.x * MK_WARPS;
  const long long gthr   = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long nthr   = (long long)gridDim.x * blockDim.x;

  /* fml_scan's tiles. Every block allocates them; only block 0 runs the scan,
   * which is the same one-block-per-record shape the standalone kernel has. */
  __shared__ int sa[MK_BLOCK];
  __shared__ int sc[MK_BLOCK];

  long long _mark = clock64();

  for (int i = i_top; i >= 1; i--) {
    /* This row's tables: one slot per sweep row, uploaded once for the chunk
     * (device.cu). A per-record kernel could derive these from (i, L) instead
     * -- they are arithmetic once a kernel owns one record -- but stage 0
     * changes structure only, so it reads what the old path reads. */
    const size_t   *size_off = p.rt_size + (size_t)i * (size_t)(nfiles + 1);
    const size_t   *side_off = p.rt_side + (size_t)i * (size_t)(nfiles + 1);
    const int      *i_H      = p.rt_ih   + (size_t)i * (size_t)nfiles;

    const size_t    sbase  = size_off[H];
    const long long width  = (long long)size_off[H + 1] - (long long)sbase;
    const size_t    stotal = size_off[nfiles];
    const size_t    dbase  = side_off[H];
    const long long dwidth = (long long)side_off[H + 1] - (long long)dbase;
    const size_t    dtotal = side_off[nfiles];

    /* ---- interior loops and the hairpin/multibranch terms -----------------
     * Independent of each other: int_loop reads `c` rows below i, hp_mb_3p
     * reads only the sequence, the parameters and the masks. They share this
     * phase for that reason -- it is the same independence RNA_STREAM_OVERLAP
     * level 1 exploits with two streams. */
    for (long long c = gwarp; c < width; c += nwarps)
      int_loop_warp_cell(nfiles, i, length, p.TerminalAU, p.ninio2,
                         (const cuda_param_t *)p.param, p.lxc, p.pair, p.S, p.hccc,
                         p.up_int, p.my_c, p.tri_off_H, p.row_off_H, p.hc_off_H,
                         size_off, i_H, p.energy_min2,
                         H, (size_t)c, lane);
    MK_TICK(MK_PH_INT_LOOP);

    for (long long c = gthr; c < width; c += nthr)
      hp_mb_3p_cell(nfiles, i, turn_, length, p.S2, p.sequence, p.pair2,
                    p.hccc_mb, p.hccc_mbenc, p.hccc_any, p.hccc_gu,
                    (const cuda_param2_t *)p.param2, p.salt_loop,
                    p.energy_hp_row, p.energy_mb_row, p.energy_3p00_row, p.gate_row,
                    p.row_off_H, p.hc2_off_H, p.seq_off_H, p.len_H,
                    size_off, stotal, i_H,
                    (long long)sbase + c);
    MK_TICK(MK_PH_HP_MB);

    grid.sync();
    MK_TICK(MK_PH_SYNC);

    /* ---- new_c: the join. Needs md(i+1)'s DMLi at column j-1, which the
     * previous iteration published, and both phases above. ---------------- */
    for (long long c = gthr; c < width; c += nthr)
      new_c_cell(nfiles, i, turn_, noGUclosure, p.energy_min2,
                 p.energy_hp_row, p.energy_mb_row, p.gate_row, p.dml1,
                 p.up_hp, p.seq_off_H, p.new_e,
                 p.energy_stack_row, p.cc1, p.cc,
                 p.row_off_H, size_off, stotal, i_H,
                 (long long)sbase + c);
    MK_TICK(MK_PH_NEW_C);

    grid.sync();
    MK_TICK(MK_PH_SYNC);

    /* ---- row i of `c` into the triangle, and the fML row scan ------------
     * Both read new_e and neither reads the other, so they share a phase.
     * The scan is block-cooperative and runs on block 0 alone. */
    for (long long c = gthr; c < width; c += nthr)
      load_my_c_cell(nfiles, i, length, p.new_e, p.my_c,
                     p.tri_off_H, p.row_off_H, size_off, stotal, i_H,
                     (long long)sbase + c);
    MK_TICK(MK_PH_LOAD_C);

    if (blockIdx.x == 0)
      fml_scan_block<MK_BLOCK>(nfiles, i, turn_, p.new_e, p.energy_3p00_row,
                               /* gq_row */ NULL, p.fml_prev, p.up_ml_ok,
                               (const cuda_param2_t *)p.param2, p.energy_min,
                               p.row_off_H, p.seq_off_H, size_off, i_H,
                               H, sa, sc);
    MK_TICK(MK_PH_SCAN);

    grid.sync();
    MK_TICK(MK_PH_SYNC);

    /* ---- the fML triangle gets row i, then row i is gathered -------------- */
    for (long long c = gthr; c < width; c += nthr)
      load_fML_cell(nfiles, i, turn_, length, p.energy_min, p.fml_j,
                    /* fml_row, int16 only */ NULL,
                    p.tri_off_H, p.row_off_H, size_off, stotal, i_H,
                    (long long)sbase + c);
    MK_TICK(MK_PH_LOAD_FML);

    grid.sync();
    MK_TICK(MK_PH_SYNC);

    for (long long c = gthr; c < dwidth; c += nthr)
      fmli_cell(nfiles, i, turn_, length, p.fml_i, p.fml_j, NULL,
                p.tri_off_H, p.row_off_H, side_off, dtotal, i_H,
                (long long)dbase + c);

    grid.sync();
    MK_TICK(MK_PH_SYNC);

    /* ---- the decomposition itself: the phase everything else exists for --- */
    for (long long c = gwarp; c < dwidth; c += nwarps)
      md_cell<MK_TILE>(nfiles, i, turn_, length, p.fml_i, p.fml_j,
                       /* int16 */ NULL, NULL, NULL, NULL,
                       p.dml, p.fm2, p.tri_off_H, p.row_off_H,
                       side_off, dtotal, i_H,
                       (long long)dbase + c, lane);
    MK_TICK(MK_PH_MD);

    grid.sync();
    MK_TICK(MK_PH_SYNC);

    /* ---- close the row: fold DMLi into fML, cache row i, publish DMLi1 ---- */
    for (long long c = gthr; c < dwidth; c += nthr)
      load_min_fML_cell(nfiles, i, turn_, length, p.energy_min, p.dml,
                        p.fml_j, NULL, p.tri_off_H, p.row_off_H,
                        side_off, dtotal, i_H,
                        (long long)dbase + c);

    for (long long c = gthr; c < width; c += nthr)
      fml_prev_cell(nfiles, i, turn_, p.energy_min, p.dml, p.fml_prev,
                    p.row_off_H, size_off, stotal, i_H,
                    (long long)sbase + c);

    grid.sync();
    MK_TICK(MK_PH_SYNC);

    /* md_snapshot_dml()'s device-to-device copy, restricted to this record's
     * row. The standalone path copies the whole batch's row buffer on the md
     * stream; here the grid that just wrote it copies its own slice. */
    {
      const size_t lo = p.row_off_H[H];
      const size_t hi = p.row_off_H[H + 1];
      for (size_t k = lo + (size_t)gthr; k < hi; k += (size_t)nthr)
        p.dml1[k] = p.dml[k];
    }
    MK_TICK(MK_PH_TAIL);

    grid.sync();
    MK_TICK(MK_PH_SYNC);
  }
}

/* ===================== host side ===================== */

extern "C" int
rnafold_megakernel(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_MEGAKERNEL");

    v = (e && e[0] && e[0] != '0') ? 1 : 0;
    if (v)
      fprintf(stderr, "megakernel.cu            RNA_MEGAKERNEL=1: the row chain runs as one "
                      "cooperative kernel per record.\n"
                      "megakernel.cu            STAGE 0 -- same arithmetic, fused; expected to be "
                      "SLOWER than the per-phase path.\n");
  }

  return v;
}

/*
 *  Everything v1 does not reproduce. Each entry is a real conditional in the
 *  kernels, not a formality: a megakernel that silently mishandled one would
 *  return a plausible structure rather than fail, which is the failure mode this
 *  project keeps finding in its own probes.
 */
extern "C" const char *
rnafold_megakernel_refuse(int nfiles, int circ, int gquad, int nolp,
                          int uniq_ML, int depot, int dangles,
                          int continuous_flow, int int16)
{
  if (nfiles <= 0)             return "no records";
  if (circ)                    return "circular (fM2_real is a second output of md)";
  if (gquad)                   return "gquad (a per-row expansion the fused chain has no phase for)";
  if (nolp)                    return "noLP (cc/cc1 rotate per row, plus stack_row)";
  if (uniq_ML)                 return "uniq_ML";
  if (depot)                   return "hard-constraint depot (up_int/up_hp)";
  if ((dangles != 0) && (dangles != 2)) return "dangles other than 0 or 2";
  if (continuous_flow)         return "continuous flow / slot flow (records on their own rows)";
  if (int16)                   return "int16 fML (stage 1)";
  if (rnafold_md_smem())       return "RNA_MD_SMEM";
  if (!rnafold_gpu_sweep())    return "RNA_GPU_SWEEP=0 (the host row loops)";
  return NULL;
}

/*
 *  Grid geometry. A cooperative launch requires every block to be RESIDENT, so
 *  the grid is what the occupancy API says fits, not what the work wants -- the
 *  blocks stride over the cells instead. Dividing by `G` leaves room for G
 *  records to be in flight at once, which is the dial the residency arithmetic
 *  turns (fewer records => more of each record's fML fits on chip).
 */
static int
mk_grid_blocks(const int G)
{
  static int per_sm = 0, sms = 0;

  if (!per_sm) {
    cudaDeviceProp prop;
    int dev = 0;

    if ((cudaGetDevice(&dev) != cudaSuccess) ||
        (cudaGetDeviceProperties(&prop, dev) != cudaSuccess))
      return 0;
    sms = prop.multiProcessorCount;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, megakernel_record,
                                                      MK_BLOCK, 0) != cudaSuccess)
      return 0;
    fprintf(stderr, "megakernel.cu            %d SMs x %d resident blocks of %d threads\n",
            sms, per_sm, MK_BLOCK);
  }

  {
    const int g = (G > 0) ? G : 1;
    const int n = (per_sm * sms) / g;
    return (n > 0) ? n : 1;
  }
}

extern "C" int
rnafold_megakernel_records_in_flight(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_MK_RECORDS");

    v = (e && e[0]) ? atoi(e) : 1;
    if (v < 1) v = 1;
  }

  return v;
}

extern "C" int
rnafold_megakernel_sweep(const int nfiles, const int *slots, const int count,
                         const int turn, const int length, const int *len_H_host,
                         const int noGUclosure)
{
  const int G      = rnafold_megakernel_records_in_flight();
  const int blocks = mk_grid_blocks(G);
  rnafold_mk_ptrs_t p;
  unsigned long long *d_clocks = NULL;
  cudaStream_t *streams;
  int k;

  if (blocks <= 0) {
    fprintf(stderr, "megakernel.cu            no launch geometry -- falling back\n");
    return -1;
  }

  memset(&p, 0, sizeof(p));
  int_loop_mk_ptrs(&p);
  hp_mb_mk_ptrs(&p);
  md_mk_ptrs(&p);

  if (cudaMalloc((void **)&d_clocks, MK_PH_N * sizeof(unsigned long long)) == cudaSuccess)
    cudaMemset(d_clocks, 0, MK_PH_N * sizeof(unsigned long long));

  streams = (cudaStream_t *)calloc((size_t)G, sizeof(cudaStream_t));
  for (k = 0; k < G; k++)
    cudaStreamCreateWithFlags(&streams[k], cudaStreamNonBlocking);

  for (k = 0; k < count; k++) {
    const int H     = slots[k];
    const int len   = len_H_host[H];
    const int i_top = len - turn - 1;
    void *args[8];

    if (i_top < 1)
      continue;

    {
      /* cudaLaunchCooperativeKernel takes an array of POINTERS to arguments,
       * so each one needs an lvalue that outlives the launch. */
      static __thread rnafold_mk_ptrs_t a_p;
      static __thread int a_nfiles, a_H, a_turn, a_len, a_itop, a_nogu;
      static __thread unsigned long long *a_clocks;
      cudaError_t rc;

      a_p = p; a_nfiles = nfiles; a_H = H; a_turn = turn; a_len = length;
      a_itop = i_top; a_nogu = noGUclosure; a_clocks = d_clocks;
      args[0] = &a_p;    args[1] = &a_nfiles; args[2] = &a_H;     args[3] = &a_turn;
      args[4] = &a_len;  args[5] = &a_itop;   args[6] = &a_nogu;  args[7] = &a_clocks;

      rc = cudaLaunchCooperativeKernel((const void *)megakernel_record,
                                       dim3(blocks), dim3(MK_BLOCK), args, 0,
                                       streams[k % G]);
      if (rc != cudaSuccess) {
        fprintf(stderr, "megakernel.cu            cooperative launch failed: %s\n",
                cudaGetErrorString(rc));
        cudaGetLastError();
        for (k = 0; k < G; k++) cudaStreamDestroy(streams[k]);
        free(streams);
        if (d_clocks) cudaFree(d_clocks);
        return -1;
      }
    }
  }

  for (k = 0; k < G; k++) {
    cudaStreamSynchronize(streams[k]);
    cudaStreamDestroy(streams[k]);
  }
  free(streams);

  if (d_clocks) {
    unsigned long long h[MK_PH_N];
    static const char *names[MK_PH_N] = { "int_loop", "hp_mb", "new_c", "load_my_c",
                                          "fml_scan", "load_fML", "md", "tail", "grid.sync" };
    unsigned long long tot = 0;
    int t;

    if (cudaMemcpy(h, d_clocks, sizeof(h), cudaMemcpyDeviceToHost) == cudaSuccess) {
      for (t = 0; t < MK_PH_N; t++) tot += h[t];
      if (tot) {
        fprintf(stderr, "megakernel.cu            phase share of block 0's cycles:");
        for (t = 0; t < MK_PH_N; t++)
          fprintf(stderr, " %s=%.1f%%", names[t], 100.0 * (double)h[t] / (double)tot);
        fprintf(stderr, "\n");
      }
    }
    cudaFree(d_clocks);
  }

  return 0;
}
