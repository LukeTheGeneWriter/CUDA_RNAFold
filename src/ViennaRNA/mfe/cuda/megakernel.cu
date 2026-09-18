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
#include <time.h>
#include <unistd.h>
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
#include "nth.h"

/* int_loop.cu carries "#define turn 3" and its cells use that macro, while the
 * hp_mb and md cells take a PARAMETER called turn -- which the macro would
 * rewrite into a literal. So the int_loop side is bracketed and everything else
 * is included with the macro out of scope. */
#define turn 3
#include "int_loop_dev.h"
#include "int_loop_cell.inc"
#include "int_loop_cells.inc"
#undef turn

#include "hp_mb_dev.h"
#include "hp_mb_cells.inc"
#include "fml_scan_block.inc"
#include "md_dev.h"
#include "md_cell.inc"
#include "md_chain_cells.inc"

/* Gate knobs owned by other files. rnafold_gpu_sweep() is declared in stub2.h;
 * this one is not, and asking it beats re-reading its environment variable --
 * two readers of one knob drift. */
int rnafold_md_smem(void);

namespace cg = cooperative_groups;

/* One block size for every phase, which is a real constraint of fusing them:
 * int_loop wants one warp per cell, md runs TILE=32 lanes per cell, and
 * fml_scan is a block-wide scan whose tile width is the block. 256 = eight
 * warps satisfies all three (eight cells per block for the two warp-shaped
 * phases, a 256-wide scan tile for the third). */
#define MK_BLOCK   256
#define MK_WARPS   (MK_BLOCK / 32)
#define MK_TILE    32          /* lanes per cell in int_loop and md */

/* How far back an interior loop can reach: MAXLOOP=30 unpaired bases, so cell
 * (i,j) reads c(p,q) only for p in [i+1,i+31] and q in [j-31,j-1]. Both the
 * ring's depth and the window's left margin are this number. */
#define MK_CW_BACK 31

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
#define MK_PH_PACK     9
#define MK_PH_N       10
/* Debug slots (RNA_MK_DEBUG), appended after the phase clocks: what the
 * kernel actually saw. A fused kernel that runs but computes nothing looks
 * exactly like one that never ran. */
#define MK_DBG_WIDTH   10
#define MK_DBG_DWIDTH  11
#define MK_DBG_CELLS   12
#define MK_DBG_ROWS    13
#define MK_DBG_IH      14
#define MK_DBG_ROWOFF  15
#define MK_DBG_EMIN2   16
#define MK_DBG_HP      17
#define MK_DBG_GATE    18
#define MK_DBG_NEWE    19
#define MK_SLOTS       20

#define MK_TICK(slot)                                                     \
  do {                                                                    \
    if (clocks && (blockIdx.x == 0) && (threadIdx.x == 0)) {              \
      const long long _now = clock64();                                   \
      clocks[slot] += (unsigned long long)(_now - _mark);                 \
      _mark = _now;                                                       \
    }                                                                     \
    MK_MARK(slot);                                                        \
  } while (0)

/* Progress, in MAPPED HOST memory, which is the only channel that survives a
 * deadlock: a hung kernel never returns its buffers, so `clocks` and device
 * printf both say nothing about where it stopped. Block 0 publishes (row,
 * phase); every block counts itself in just before each barrier, so a count
 * short of gridDim says some block never arrived -- the one fact that separates
 * "a phase is spinning" from "the grid is not co-resident". */
#define MK_MARK(slot)                                                     \
  do {                                                                    \
    if (prog && (blockIdx.x == 0) && (threadIdx.x == 0)) {                \
      prog[0] = (unsigned int)i;                                          \
      prog[1] = (unsigned int)(slot);                                     \
      __threadfence_system();                                             \
    }                                                                     \
  } while (0)

/* Counted arrival, then the barrier itself. */
#define MK_PROG_HDR 4
/* Every thread counts itself into its own block's slot. grid.sync() needs all
 * of them, not just thread 0, so a per-block deficit names the block that is
 * short and, by the size of the deficit, the warp that never arrived. */
#define MK_SYNC()                                                         \
  do {                                                                    \
    if (prog) {                                                           \
      if (threadIdx.x == 0)                                               \
        prog[MK_PROG_HDR + blockIdx.x] += 1u;  /* one writer per slot */  \
    }                                                                     \
    __threadfence_system();                                               \
    grid.sync();                                                          \
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
#define MK_SKIP_INT_LOOP 1
#define MK_SKIP_HP_MB     2
#define MK_SKIP_NEW_C     4
#define MK_SKIP_LOAD_C    8
#define MK_SKIP_SCAN     16
#define MK_SKIP_LOAD_FML 32
#define MK_SKIP_FMLI     64
#define MK_SKIP_MD      128
#define MK_SKIP_TAIL    256
#define MK_SKIP_CLOSE   512
#define MK_SKIP_PACK   1024

__global__ void
megakernel_record(const rnafold_mk_ptrs_t p,
                  const int nfiles, const int H, const int turn_, const int length,
                  const int i_top, const int noGUclosure,
                  unsigned long long *clocks, const int skip,
                  volatile unsigned int *prog)
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

  /*
   *  Stages 1b and 2 live here, and both need the SAME thing from the
   *  schedule: a block must own a FIXED range of absolute columns for the
   *  whole sweep, or nothing it caches survives to the next row.
   *
   *  That is a real trade. Striding cells (stage 0) balances every row
   *  perfectly; owning columns leaves a block idle until the sweep reaches its
   *  range, because row i only has columns [i+turn+1, len]. Blocks owning high
   *  columns work on every row and blocks owning low columns work only late.
   *  An equal-width split is therefore NOT work-balanced -- column j carries
   *  (j-turn-1) cells over the sweep, so a balanced split would put the
   *  boundaries at j = len*sqrt(b/B). That is left for the tuning pass: a
   *  balanced split makes the widest block ~len/sqrt(B) columns wide, and the
   *  window is sized by the WIDEST block, so it costs shared memory exactly
   *  where there is none.
   */
  extern __shared__ int mk_dyn[];

  const int cwW  = p.cw_cols;                       /* columns owned          */
  const int cwC  = cwW + MK_CW_BACK;                /* window columns         */
  const int Jown = turn_ + 2 + (int)blockIdx.x * cwW;   /* first owned column */
  const int Jend = Jown + cwW;
  const int Qlo  = Jown - MK_CW_BACK;               /* window's first column  */

  int *const cring  = (cwW && p.cw_on)    ? mk_dyn : NULL;
  int *const corner = (cwW && p.corner_k) ? (mk_dyn + (p.cw_on ? 32 * cwC : 0)) : NULL;

  /* Column ownership drives the work split whenever EITHER cache is on. */
  const int owns = (cwW != 0);

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

    if (clocks && (blockIdx.x == 0) && (threadIdx.x == 0)) {
      if (i == i_top) {
        clocks[MK_DBG_WIDTH]  = (unsigned long long)width;
        clocks[MK_DBG_DWIDTH] = (unsigned long long)dwidth;
      }
      clocks[MK_DBG_ROWS]  += 1ull;
      clocks[MK_DBG_CELLS] += (unsigned long long)(width > 0 ? width : 0);
    }

    /* ---- stage 1b: bring the `c` ring up to date --------------------------
     * Row i needs rows i+1..i+31. Every one but row i+1 is already on chip
     * from the previous iteration, so steady state is ONE row of the block's
     * column span per sweep row. The first row of the sweep fills all 31.
     *
     * Reading row i+1 here is safe because load_my_c wrote it last iteration
     * and two grid barriers have passed since. */
    if (cring) {
      const int    lenH  = p.len_H[H];
      const size_t triH  = p.tri_off_H[H];
      const int    pfirst = i + 1;
      const int    plast  = (i == i_top) ? (i + MK_CW_BACK) : (i + 1);

      for (int pp = pfirst; pp <= plast; pp++) {
        int *const dst = cring + ((pp & 31) * cwC);

        for (int t = (int)threadIdx.x; t < cwC; t += (int)blockDim.x) {
          const int q = Qlo + t;
          /* Outside the record's triangle the cell does not exist; INF is what
           * the recurrence expects there and what my_c holds anyway. */
          dst[t] = ((pp >= 1) && (q > pp) && (q <= lenH))
                 ? p.my_c[triH + Indx(pp, q)]
                 : INF;
        }
      }
      __syncthreads();
    }

    /* ---- interior loops and the hairpin/multibranch terms -----------------
     * Independent of each other: int_loop reads `c` rows below i, hp_mb_3p
     * reads only the sequence, the parameters and the masks. They share this
     * phase for that reason -- it is the same independence RNA_STREAM_OVERLAP
     * level 1 exploits with two streams. */
    if (!(skip & MK_SKIP_INT_LOOP)) {
      if (owns) {
        /* One warp per cell, cells taken from the block's own columns. */
        const int j0  = i + turn_ + 1;
        const int jlo = (Jown > j0) ? Jown : j0;
        const int jhi = (Jend < j0 + (int)width) ? Jend : (j0 + (int)width);

        if (cring) {
          c_win_reader cw;

          cw.sm = cring; cw.q0 = Qlo; cw.stride = cwC;
          for (int j = jlo + wib; j < jhi; j += MK_WARPS)
            int_loop_warp_cell_r(nfiles, i, length, p.TerminalAU, p.ninio2,
                                 (const cuda_param_t *)p.param, p.lxc, p.pair,
                                 p.S, p.hccc, p.up_int, cw, p.row_off_H,
                                 p.hc_off_H, size_off, i_H, p.energy_min2,
                                 H, (size_t)(j - j0), lane);
        } else {
          for (int j = jlo + wib; j < jhi; j += MK_WARPS)
            int_loop_warp_cell(nfiles, i, length, p.TerminalAU, p.ninio2,
                               (const cuda_param_t *)p.param, p.lxc, p.pair,
                               p.S, p.hccc, p.up_int, p.my_c, p.tri_off_H,
                               p.row_off_H, p.hc_off_H, size_off, i_H,
                               p.energy_min2, H, (size_t)(j - j0), lane);
        }
      } else {
        for (long long c = gwarp; c < width; c += nwarps)
          int_loop_warp_cell(nfiles, i, length, p.TerminalAU, p.ninio2,
                             (const cuda_param_t *)p.param, p.lxc, p.pair, p.S, p.hccc,
                             p.up_int, p.my_c, p.tri_off_H, p.row_off_H, p.hc_off_H,
                             size_off, i_H, p.energy_min2,
                             H, (size_t)c, lane);
      }
    }
    MK_TICK(MK_PH_INT_LOOP);

    if (!(skip & MK_SKIP_HP_MB))
      for (long long c = gthr; c < width; c += nthr)
          hp_mb_3p_cell(nfiles, i, turn_, length, p.S2, p.sequence, p.pair2,
                    p.hccc_mb, p.hccc_mbenc, p.hccc_any, p.hccc_gu,
                    (const cuda_param2_t *)p.param2, p.salt_loop,
                    p.energy_hp_row, p.energy_mb_row, p.energy_3p00_row, p.gate_row,
                    p.row_off_H, p.hc2_off_H, p.seq_off_H, p.len_H,
                    size_off, stotal, i_H,
                    (long long)sbase + c);
    MK_TICK(MK_PH_HP_MB);

    if (clocks && (blockIdx.x == 0) && (threadIdx.x == 0) && (i == i_top) && (width > 0)) {
      const int j = (int)(i + turn_ + 1);
      clocks[MK_DBG_IH]     = (unsigned long long)(unsigned int)i_H[H];
      clocks[MK_DBG_ROWOFF] = (unsigned long long)p.row_off_H[H];
      clocks[MK_DBG_EMIN2]  = (unsigned long long)(unsigned int)p.energy_min2[p.row_off_H[H] + j];
      clocks[MK_DBG_HP]     = (unsigned long long)(unsigned int)p.energy_hp_row[p.row_off_H[H] + j];
      clocks[MK_DBG_GATE]   = (unsigned long long)(unsigned char)p.gate_row[p.row_off_H[H] + j];
    }

    MK_SYNC();
    MK_TICK(MK_PH_SYNC);

    /* ---- new_c: the join. Needs md(i+1)'s DMLi at column j-1, which the
     * previous iteration published, and both phases above. ---------------- */
    if (!(skip & MK_SKIP_NEW_C))
      for (long long c = gthr; c < width; c += nthr)
          new_c_cell(nfiles, i, turn_, noGUclosure, p.energy_min2,
                 p.energy_hp_row, p.energy_mb_row, p.gate_row, p.dml1,
                 p.up_hp, p.seq_off_H, p.new_e,
                 /* noLP's three, and they MUST be NULL here: the cell switches
                  * on stack_row being non-NULL, and in that mode c[ij] receives
                  * the STACKED value rather than the hairpin -- INF on the first
                  * row, which is exactly how this first ran: gate=1, hairpin=590
                  * computed, and new_e=INF anyway. noLP is refused by the gate,
                  * so the fused path must not hand the cell its buffers just
                  * because they are allocated. */
                 NULL, NULL, NULL,
                 p.row_off_H, size_off, stotal, i_H,
                 (long long)sbase + c);
    MK_TICK(MK_PH_NEW_C);

    if (clocks && (blockIdx.x == 0) && (threadIdx.x == 0) && (i == i_top) && (width > 0))
      clocks[MK_DBG_NEWE] = (unsigned long long)(unsigned int)p.new_e[p.row_off_H[H] + i + turn_ + 1];

    MK_SYNC();
    MK_TICK(MK_PH_SYNC);

    /* ---- row i of `c` into the triangle, and the fML row scan ------------
     * Both read new_e and neither reads the other, so they share a phase.
     * The scan is block-cooperative and runs on block 0 alone. */
    if (!(skip & MK_SKIP_LOAD_C))
      for (long long c = gthr; c < width; c += nthr)
          load_my_c_cell(nfiles, i, length, p.new_e, p.my_c,
                     p.tri_off_H, p.row_off_H, size_off, stotal, i_H,
                     (long long)sbase + c);
    MK_TICK(MK_PH_LOAD_C);

    if ((blockIdx.x == 0) && (!(skip & MK_SKIP_SCAN)))
      fml_scan_block<MK_BLOCK>(nfiles, i, turn_, p.new_e, p.energy_3p00_row,
                               /* gq_row */ NULL, p.fml_prev, p.up_ml_ok,
                               (const cuda_param2_t *)p.param2, p.energy_min,
                               p.row_off_H, p.seq_off_H, size_off, i_H,
                               H, sa, sc);
    MK_TICK(MK_PH_SCAN);

    MK_SYNC();
    MK_TICK(MK_PH_SYNC);

    /* ---- the fML triangle gets row i, then row i is gathered -------------- */
    if (!(skip & MK_SKIP_LOAD_FML))
      for (long long c = gthr; c < width; c += nthr)
          load_fML_cell(nfiles, i, turn_, length, p.energy_min, p.fml_j,
                    p.fml_row,
                    p.tri_off_H, p.row_off_H, size_off, stotal, i_H,
                    (long long)sbase + c);
    MK_TICK(MK_PH_LOAD_FML);

    MK_SYNC();
    MK_TICK(MK_PH_SYNC);

    if (!(skip & MK_SKIP_FMLI))
      for (long long c = gthr; c < dwidth; c += nthr)
          fmli_cell(nfiles, i, turn_, length, p.fml_i, p.fml_j, p.fml_row,
                p.tri_off_H, p.row_off_H, side_off, dtotal, i_H,
                (long long)dbase + c);

    MK_SYNC();
    MK_TICK(MK_PH_SYNC);

    /* ---- the decomposition itself: the phase everything else exists for --- */
    if (!(skip & MK_SKIP_MD)) {
      fml_corner_t cc;

      cc.sm = corner; cc.J0 = Jown; cc.K = p.corner_k;
      if (owns) {
        const int d0  = i + 2 * (turn_ + 1) + 1;   /* first md column of row i */
        const int dlo = (Jown > d0) ? Jown : d0;
        const int dhi = (Jend < d0 + (int)dwidth) ? Jend : (d0 + (int)dwidth);

        for (int j = dlo + wib; j < dhi; j += MK_WARPS)
          md_cell<MK_TILE>(nfiles, i, turn_, length, p.fml_i, p.fml_j,
                           p.fml_j16, p.fml_b, p.base_off_H, p.colb_off,
                           p.dml, p.fm2, p.tri_off_H, p.row_off_H,
                           side_off, dtotal, i_H,
                           (long long)dbase + (j - d0), lane, cc);
      } else {
        for (long long c = gwarp; c < dwidth; c += nwarps)
          md_cell<MK_TILE>(nfiles, i, turn_, length, p.fml_i, p.fml_j,
                       p.fml_j16, p.fml_b, p.base_off_H, p.colb_off,
                       p.dml, p.fm2, p.tri_off_H, p.row_off_H,
                       side_off, dtotal, i_H,
                       (long long)dbase + c, lane, cc);
      }
    }
    MK_TICK(MK_PH_MD);

    MK_SYNC();
    MK_TICK(MK_PH_SYNC);

    /* ---- close the row: fold DMLi into fML, cache row i, publish DMLi1 ---- */
    if (!(skip & MK_SKIP_CLOSE))
    for (long long c = gthr; c < dwidth; c += nthr)
      load_min_fML_cell(nfiles, i, turn_, length, p.energy_min, p.dml,
                        p.fml_j, p.fml_row, p.tri_off_H, p.row_off_H,
                        side_off, dtotal, i_H,
                        (long long)dbase + c);

    if (!(skip & MK_SKIP_CLOSE))
    for (long long c = gthr; c < width; c += nthr)
      fml_prev_cell(nfiles, i, turn_, p.energy_min, p.dml, p.fml_prev,
                    p.row_off_H, size_off, stotal, i_H,
                    (long long)sbase + c);

    MK_SYNC();
    MK_TICK(MK_PH_SYNC);

    /* ---- int16: pack row i into the triangle -----------------------------
     * Runs AFTER both of the row's writers (load_fML and load_min_fML), which
     * is the whole ordering constraint of the encoding: fml_row holds the
     * final row only once both have run. A no-op when the gate is off.
     *
     * The baseline claim `fml_b[bidx] = v` is race-free because each thread
     * owns a distinct (column, block) WITHIN a row, and rows are separated --
     * by a kernel launch on the per-phase path, and by the grid barrier above
     * on this one. Bracketing pack between barriers is what preserves that.
     *
     * It SHARES the tail barrier rather than adding one: the snapshot below
     * reads dml and writes dml1, which pack neither reads nor writes, so the
     * two are independent. Its own barrier cost 19 points of grid.sync share
     * (50.5 % -> 69.7 % at 40 records), which is the whole reason to care. */
    if (p.fml_row && !(skip & MK_SKIP_PACK))
      for (long long c = gthr; c < width; c += nthr)
        pack_fml_cell(nfiles, i, turn_, length, p.fml_row, p.fml_j16, p.fml_b,
                      p.tri_off_H, p.row_off_H, p.base_off_H, p.colb_off,
                      size_off, stotal, i_H,
                      (long long)sbase + c);
    MK_TICK(MK_PH_PACK);

    /* ---- stage 2: row i joins the corner cache ---------------------------
     * Row i of fML is final now (both writers, and pack, have run). Entry
     * (i,j) sits at offset (j-turn-1)-i of column j and is first READ by md at
     * sweep row i-turn-2, so it is on chip well before anyone wants it.
     *
     * Read back from fml_row under int16 -- it is this row in full int32, so
     * the cache never holds a packed delta -- and from the triangle otherwise. */
    if (corner) {
      const int lenH = p.len_H[H];

      for (int j = Jown + (int)threadIdx.x; j < Jend; j += (int)blockDim.x) {
        const int off = (j - turn_ - 1) - i;

        if ((off >= 0) && (off < p.corner_k) && (j <= lenH) && (j >= i + turn_ + 1))
          ((int *)corner)[(size_t)(j - Jown) * (size_t)p.corner_k + (size_t)off] =
            p.fml_row ? p.fml_row[p.row_off_H[H] + j]
                      : p.fml_j[p.tri_off_H[H] + Indx(i, j)];
      }
    }

    /* md_snapshot_dml()'s device-to-device copy, restricted to this record's
     * row. The standalone path copies the whole batch's row buffer on the md
     * stream; here the grid that just wrote it copies its own slice. */
    if (!(skip & MK_SKIP_TAIL)) {
      const size_t lo = p.row_off_H[H];
      const size_t hi = p.row_off_H[H + 1];
      for (size_t k = lo + (size_t)gthr; k < hi; k += (size_t)nthr)
        p.dml1[k] = p.dml[k];
    }
    MK_TICK(MK_PH_TAIL);

    MK_SYNC();
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
                          int continuous_flow)
{
  if (nfiles <= 0)             return "no records";
  if (circ)                    return "circular (fM2_real is a second output of md)";
  if (gquad)                   return "gquad (a per-row expansion the fused chain has no phase for)";
  if (nolp)                    return "noLP (cc/cc1 rotate per row, plus stack_row)";
  if (uniq_ML)                 return "uniq_ML";
  if (depot)                   return "hard-constraint depot (up_int/up_hp)";
  if ((dangles != 0) && (dangles != 2)) return "dangles other than 0 or 2";
  if (continuous_flow)         return "continuous flow / slot flow (records on their own rows)";
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
    /* Shared memory is planned from the block count, which is what this
     * returns, so the geometry cannot be fed back in here without a fixed
     * point. Measured at 0 and then honoured by the planner's budget instead:
     * RNA_MK_SMEM_KB is the cap that keeps the launch resident. */
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

/* RNA_MK_MAXROWS=k -- stop each record after k rows. A bisection handle: a fused
 * kernel that deadlocks looks identical to one that is merely slow, and the row
 * at which it stops says which phase is at fault. Answers are WRONG under it by
 * construction (the sweep is incomplete); it is a debugging knob, not an arm. */
static int
mk_timeout_s(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_MK_TIMEOUT");

    v = (e && e[0]) ? atoi(e) : 0;
  }

  return v;
}

extern "C" int
rnafold_megakernel_skip(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_MK_SKIP");

    v = (e && e[0]) ? atoi(e) : 0;
    if (v)
      fprintf(stderr, "megakernel.cu            RNA_MK_SKIP=%d: phases disabled, "
                      "answers are wrong by construction\n", v);
  }

  return v;
}

extern "C" int
rnafold_megakernel_maxrows(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_MK_MAXROWS");

    v = (e && e[0]) ? atoi(e) : 0;
    if (v > 0)
      fprintf(stderr, "megakernel.cu            RNA_MK_MAXROWS=%d: STOPPING EARLY, "
                      "answers are wrong by construction\n", v);
  }

  return v;
}

/*
 *  How many records are in flight, and so how much of the device each one
 *  gets. AUTO by default, because 1 -- the old default -- is the worst
 *  setting there is: one record's row is at most `length` cells, the grid is
 *  12 288 threads, and the row's eight grid barriers are paid by a grid that
 *  is mostly idle. Measured on C_mixed (40 records, sm_86, int16):
 *
 *      G=1   2.22 s   grid.sync 66.6 % of block 0's cycles
 *      G=4   1.31 s
 *      G=8   1.21 s
 *      G=16  1.21 s   grid.sync 27.7 %, int_loop 32 %, md 31 %
 *
 *  against 1.63 s for the per-phase path -- so the fused kernel is 26 % faster
 *  once records are co-resident, and 36 % SLOWER when they are not. The whole
 *  difference is barrier amortisation.
 *
 *  AUTO gives each record just enough blocks to cover one row of the longest
 *  record and spends the rest of the device on more records. `length` is the
 *  batch maximum, so this is the widest row any record in flight will have.
 */
static int
mk_env(const char *name, const int dflt)
{
  const char *e = getenv(name);

  return (e && e[0]) ? atoi(e) : dflt;
}

/*
 *  Stages 1b and 2 both need blocks to own fixed columns, and both are paid
 *  for in shared memory, so one function decides the whole on-chip geometry.
 *
 *  It can decline. A block owns W = ceil(span / blocks) columns, and W grows
 *  when a record gets FEWER blocks -- which is exactly what AUTO G does to
 *  maximise records in flight. So the two dials pull against each other, and
 *  at a long record with few blocks the ring alone can want more shared memory
 *  than an SM has. Degrading (corner first, then the ring, then column
 *  ownership itself) keeps the fused kernel correct at every size instead of
 *  refusing the fold.
 */
static void
mk_plan_smem(const int blocks, const int length, const int turn,
             int *cw_cols, int *cw_on, int *corner_k, size_t *bytes)
{
  const int  want_ring   = mk_env("RNA_MK_CWIN", 1);
  const int  want_corner = mk_env("RNA_MK_CORNER", 1);
  const int  budget_kb   = mk_env("RNA_MK_SMEM_KB", 32);
  const size_t budget    = (size_t)budget_kb * 1024u;
  const int  span        = (length > turn + 1) ? (length - turn - 1) : 1;
  int        W           = (span + blocks - 1) / blocks;
  int        K           = want_corner ? mk_env("RNA_MK_CORNER_K", 32) : 0;
  int        ring        = want_ring ? 1 : 0;

  *cw_cols = 0; *cw_on = 0; *corner_k = 0; *bytes = 0;
  if (W < 1) W = 1;
  if (!ring && !K)
    return;                      /* both off: stage 0's cell striding */

  for (;;) {
    const size_t nring = ring ? (size_t)32 * (size_t)(W + MK_CW_BACK) * sizeof(int) : 0;
    const size_t ncorn = (size_t)K * (size_t)W * sizeof(int);

    if (nring + ncorn <= budget) {
      *cw_cols = W; *cw_on = ring; *corner_k = K; *bytes = nring + ncorn;
      return;
    }
    if (K > 8)        K /= 2;        /* the corner degrades gracefully */
    else if (K)       K = 0;
    else if (ring)    ring = 0;      /* then the ring */
    else              return;        /* then column ownership itself */
  }
}

extern "C" int
rnafold_megakernel_records_in_flight(const int total_blocks, const int length)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_MK_RECORDS");

    if (e && e[0]) {
      v = atoi(e);
    } else {
      const int per_rec = (length + MK_BLOCK - 1) / MK_BLOCK;
      const int b       = (per_rec < 1) ? 1
                        : ((per_rec > total_blocks) ? total_blocks : per_rec);

      v = total_blocks / b;
      fprintf(stderr, "megakernel.cu            G=AUTO: %d records in flight, "
                      "%d blocks each (a %d-cell row needs %d)\n",
              (v < 1) ? 1 : v, b, length, per_rec);
    }
    if (v < 1) v = 1;
  }

  return v;
}

extern "C" const size_t *rnafold_rowtab_size_base(void);
extern "C" const size_t *rnafold_rowtab_side_base(void);
extern "C" const int    *rnafold_rowtab_ih_base(void);

extern "C" int
rnafold_megakernel_sweep(const int nfiles, const int *slots, const int count,
                         const int turn, const int length, const int *len_H_host,
                         const int noGUclosure,
                         const int TerminalAU, const int ninio2, const float lxc)
{
  const int total  = mk_grid_blocks(1);
  const int G      = (total > 0) ? rnafold_megakernel_records_in_flight(total, length) : 1;
  const int blocks = mk_grid_blocks(G);
  rnafold_mk_ptrs_t p;
  size_t mk_smem_bytes = 0;
  int    mk_cw_cols = 0, mk_cw_on = 0, mk_corner_k = 0;
  unsigned long long *d_clocks = NULL;
  volatile unsigned int *h_prog = NULL;
  unsigned int *d_prog = NULL;
  cudaStream_t *streams;
  int k;

  if (blocks <= 0) {
    fprintf(stderr, "megakernel.cu            no launch geometry -- falling back\n");
    return -1;
  }

  {
    size_t smem = 0;
    int    cw = 0, on = 0, kk = 0;

    mk_plan_smem(blocks, length, turn, &cw, &on, &kk, &smem);
    /* Above 48 KB a kernel must ASK for the larger dynamic allocation, and the
     * ask can fail -- an older card, or a limit already raised elsewhere. On
     * failure the geometry is replanned inside 48 KB rather than launched with
     * a size the driver will reject. */
    if (smem > 48u * 1024u) {
      if (cudaFuncSetAttribute((const void *)megakernel_record,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               (int)smem) != cudaSuccess) {
        cudaGetLastError();
        fprintf(stderr, "megakernel.cu            %zu KB of shared refused -- "
                        "replanning inside 48 KB\n", smem / 1024);
        setenv("RNA_MK_SMEM_KB", "48", 1);
        mk_plan_smem(blocks, length, turn, &cw, &on, &kk, &smem);
      }
    }
    mk_smem_bytes = smem;
    mk_cw_cols = cw; mk_cw_on = on; mk_corner_k = kk;
    fprintf(stderr, "megakernel.cu            on-chip: %d columns per block, "
                    "c-ring %s, fML corner K=%d, %zu KB shared\n",
            cw, on ? "ON" : "off", kk, smem / 1024);
    if (cw && (blocks > 1))
      fprintf(stderr, "megakernel.cu            NOTE: column ownership is not "
                      "work-balanced -- a block is idle until the sweep reaches "
                      "its columns\n");
  }

  memset(&p, 0, sizeof(p));
  int_loop_mk_ptrs(&p);
  hp_mb_mk_ptrs(&p);
  md_mk_ptrs(&p);
  /* The three scalars the kernels take from the host parameter table, and the
   * chunk's row tables, which the kernel strides through itself. */
  p.TerminalAU = TerminalAU;
  p.ninio2     = ninio2;
  p.lxc        = lxc;
  p.cw_cols    = mk_cw_cols;
  p.cw_on      = mk_cw_on;
  p.corner_k   = mk_corner_k;
  p.rt_size    = rnafold_rowtab_size_base();
  p.rt_side    = rnafold_rowtab_side_base();
  p.rt_ih      = rnafold_rowtab_ih_base();

  if ((!p.rt_size) || (!p.rt_side) || (!p.rt_ih)) {
    fprintf(stderr, "megakernel.cu            row tables absent -- falling back\n");
    return -1;
  }

  if (cudaMalloc((void **)&d_clocks, MK_SLOTS * sizeof(unsigned long long)) == cudaSuccess)
    cudaMemset(d_clocks, 0, MK_SLOTS * sizeof(unsigned long long));

  /* RNA_MK_TIMEOUT=s arms the deadlock probe: a mapped page the kernel writes
   * its (row, phase) into and the host reads WHILE the kernel runs. Without it
   * a hang is indistinguishable from slow, and nothing the kernel knows ever
   * reaches the host. Off by default -- it costs a host write per phase. */
  if (mk_timeout_s() > 0) {
    const size_t pn = (size_t)(MK_PROG_HDR + blocks) * sizeof(unsigned int);

    if (cudaHostAlloc((void **)&h_prog, pn, cudaHostAllocMapped) == cudaSuccess) {
      memset((void *)h_prog, 0, pn);
      if (cudaHostGetDevicePointer((void **)&d_prog, (void *)h_prog, 0) != cudaSuccess)
        d_prog = NULL;
    }
  }

  streams = (cudaStream_t *)calloc((size_t)G, sizeof(cudaStream_t));
  for (k = 0; k < G; k++)
    cudaStreamCreateWithFlags(&streams[k], cudaStreamNonBlocking);

  for (k = 0; k < count; k++) {
    const int H     = slots[k];
    const int len   = len_H_host[H];
    const int cap   = rnafold_megakernel_maxrows();
    int       i_top = len - turn - 1;

    if ((cap > 0) && (i_top > cap))
      i_top = cap;
    void *args[10];

    if (i_top < 1)
      continue;

    {
      /* cudaLaunchCooperativeKernel takes an array of POINTERS to arguments,
       * so each one needs an lvalue that outlives the launch. */
      static __thread rnafold_mk_ptrs_t a_p;
      static __thread int a_nfiles, a_H, a_turn, a_len, a_itop, a_nogu;
      static __thread unsigned long long *a_clocks;
      static __thread int a_skip;
      static __thread unsigned int *a_prog;
      cudaError_t rc;

      a_p = p; a_nfiles = nfiles; a_H = H; a_turn = turn; a_len = length;
      a_itop = i_top; a_nogu = noGUclosure; a_clocks = d_clocks;
      a_skip = rnafold_megakernel_skip();
      a_prog = d_prog;
      args[0] = &a_p;    args[1] = &a_nfiles; args[2] = &a_H;     args[3] = &a_turn;
      args[4] = &a_len;  args[5] = &a_itop;   args[6] = &a_nogu;  args[7] = &a_clocks;
      args[8] = &a_skip;   args[9] = &a_prog;

      rc = cudaLaunchCooperativeKernel((const void *)megakernel_record,
                                       dim3(blocks), dim3(MK_BLOCK), args,
                                       mk_smem_bytes, streams[k % G]);
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
    if (h_prog) {
      /* Poll, so that a deadlock reports where it stopped instead of hanging
       * the process. cudaStreamSynchronize on a hung cooperative grid never
       * returns and takes the diagnosis with it. */
      const double  limit = (double)mk_timeout_s();
      const clock_t t0    = clock();

      double        next = 1.0;
      cudaError_t   q;

      while ((q = cudaStreamQuery(streams[k])) == cudaErrorNotReady) {
        const double el = (double)(clock() - t0) / (double)CLOCKS_PER_SEC;

        /* Sampled, not just reported at the deadline: a grid that is CRAWLING
         * and one that is FROZEN look identical from a single sample, and they
         * have nothing in common as bugs. */
        if (el > next) {
          fprintf(stderr, "megakernel.cu            t=%.0fs row=%u phase=%u\n",
                  el, h_prog[0], h_prog[1]);
          fflush(stderr);
          next += 1.0;
        }
        if (el > limit) {
          static const char *names[MK_PH_N] = { "int_loop", "hp_mb", "new_c",
                                                "load_my_c", "fml_scan",
                                                "load_fML", "md", "tail",
                                                "grid.sync", "pack_fml" };
          const unsigned int row = h_prog[0], ph = h_prog[1];

          fprintf(stderr, "megakernel.cu            STUCK after %.0fs: row i=%u, "
                          "last phase completed = %s\n",
                  limit, row, (ph < MK_PH_N) ? names[ph] : "?");
          /* Per block, because the two failures look identical from the host
           * and have nothing in common: a block BEHIND the others is a phase
           * that will not finish, while every block on the same barrier is the
           * barrier itself not releasing -- which in practice means a warp
           * stranded in a full-mask shuffle, not a co-residency problem. */
          {
            unsigned int mx = 0, mn = 0xffffffffu;
            int b;

            for (b = 0; b < blocks; b++) {
              const unsigned int v = h_prog[MK_PROG_HDR + b];

              if (v > mx) mx = v;
              if (v < mn) mn = v;
            }
            fprintf(stderr, "megakernel.cu            barriers reached per block: "
                            "max=%u min=%u%s", mx, mn, "\n");
          }
          fflush(stderr);
          _exit(9);
        }
      }
      if (q != cudaSuccess)
        fprintf(stderr, "megakernel.cu            stream query: %s\n",
                cudaGetErrorString(q));
    } else {
      cudaStreamSynchronize(streams[k]);
    }
    cudaStreamDestroy(streams[k]);
  }
  free(streams);
  if (h_prog) cudaFreeHost((void *)h_prog);

  if (d_clocks) {
    unsigned long long h[MK_SLOTS];
    static const char *names[MK_PH_N] = { "int_loop", "hp_mb", "new_c", "load_my_c",
                                          "fml_scan", "load_fML", "md", "tail", "grid.sync",
                                          "pack_fml" };
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
      fprintf(stderr, "megakernel.cu            saw: first row width=%llu dwidth=%llu, "
                      "rows=%llu, cells=%llu\n",
              h[MK_DBG_WIDTH], h[MK_DBG_DWIDTH], h[MK_DBG_ROWS], h[MK_DBG_CELLS]);
      fprintf(stderr, "megakernel.cu            first cell: i_H=%d row_off=%llu "
                      "energy_min2=%d hp=%d gate=%llu new_e=%d  (INF is %d)\n",
              (int)(unsigned int)h[MK_DBG_IH], h[MK_DBG_ROWOFF],
              (int)(unsigned int)h[MK_DBG_EMIN2], (int)(unsigned int)h[MK_DBG_HP],
              h[MK_DBG_GATE], (int)(unsigned int)h[MK_DBG_NEWE], INF);
    }
    cudaFree(d_clocks);
  }

  return 0;
}
