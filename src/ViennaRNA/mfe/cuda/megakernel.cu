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

/*
 *  A row's context: everything a phase needs that changes with the row, the
 *  record, or the block's own columns. It exists because stage 3 runs TWO
 *  rows at once -- the c chain on row i while the fML chain finishes row i+1 --
 *  so a phase can no longer read its row out of enclosing scope. One struct,
 *  built twice per iteration, and every phase below takes it.
 */
struct mk_ctx_t {
  const size_t *size_off, *side_off;
  const int    *i_H;
  size_t        sbase, stotal, dbase, dtotal;
  long long     width, dwidth;
  int           i;              /* the row this context describes */
  int           live;           /* 0 when this half has no row this iteration */

  /* per-launch constants, carried here so phases take one argument */
  int           nfiles, H, turn_, length, noGUclosure, skip;

  /* the block's own columns, and the caches held over them */
  int           owns, Jown, Jend, Qlo, cwC, corner_k, own_mask;
  int          *cring;
  int          *corner;
};

/* Who this thread is. */
struct mk_thr_t {
  int       lane, wib;
  long long gwarp, nwarps, gthr, nthr;
};

__device__ inline void
mk_ctx_row(mk_ctx_t *c, const rnafold_mk_ptrs_t &p, const int i)
{
  c->i    = i;
  c->live = 0;
  if (i < 1)
    return;

  c->size_off = p.rt_size + (size_t)i * (size_t)(c->nfiles + 1);
  c->side_off = p.rt_side + (size_t)i * (size_t)(c->nfiles + 1);
  c->i_H      = p.rt_ih   + (size_t)i * (size_t)c->nfiles;
  c->sbase    = c->size_off[c->H];
  c->width    = (long long)c->size_off[c->H + 1] - (long long)c->sbase;
  c->stotal   = c->size_off[c->nfiles];
  c->dbase    = c->side_off[c->H];
  c->dwidth   = (long long)c->side_off[c->H + 1] - (long long)c->dbase;
  c->dtotal   = c->side_off[c->nfiles];
  c->live     = 1;
}

/* ---- stage 1b: bring the `c` ring up to date -----------------------------
 * Row i needs rows i+1..i+31. All but row i+1 are already on chip from the
 * previous iteration, so steady state is ONE row of the block's column span
 * per sweep row; `all` fills all thirty-one on the first row of the sweep.
 */
__device__ inline void
mk_ring_refresh(const rnafold_mk_ptrs_t &p, const mk_ctx_t &c, const int all)
{
  if (!c.cring || !c.live)
    return;
  {
    const int    lenH   = p.len_H[c.H];
    const size_t triH   = p.tri_off_H[c.H];
    const int    pfirst = c.i + 1;
    const int    plast  = all ? (c.i + MK_CW_BACK) : (c.i + 1);

    for (int pp = pfirst; pp <= plast; pp++) {
      int *const dst = c.cring + ((pp & 31) * c.cwC);

      for (int t = (int)threadIdx.x; t < c.cwC; t += (int)blockDim.x) {
        const int q = c.Qlo + t;
        /* Outside the record's triangle the cell does not exist; INF is what
         * the recurrence expects there and what my_c holds anyway. */
        dst[t] = ((pp >= 1) && (q > pp) && (q <= lenH))
               ? p.my_c[triH + Indx(pp, q)]
               : INF;
      }
    }
  }
  __syncthreads();
}

__device__ inline void
mk_ph_int_loop(const rnafold_mk_ptrs_t &p, const mk_ctx_t &c, const mk_thr_t &t)
{
  if (!c.live || (c.skip & MK_SKIP_INT_LOOP))
    return;

  if (c.owns && (c.own_mask & 1)) {
    /* One warp per cell, cells taken from the block's own columns. */
    const int j0  = c.i + c.turn_ + 1;
    const int jlo = (c.Jown > j0) ? c.Jown : j0;
    const int jhi = (c.Jend < j0 + (int)c.width) ? c.Jend : (j0 + (int)c.width);

    if (c.cring) {
      c_win_reader cw;

      cw.sm = c.cring; cw.q0 = c.Qlo; cw.stride = c.cwC;
      for (int j = jlo + t.wib; j < jhi; j += MK_WARPS)
        int_loop_warp_cell_r(c.nfiles, c.i, c.length, p.TerminalAU, p.ninio2,
                             (const cuda_param_t *)p.param, p.lxc, p.pair,
                             p.S, p.hccc, p.up_int, cw, p.row_off_H,
                             p.hc_off_H, c.size_off, c.i_H, p.energy_min2,
                             c.H, (size_t)(j - j0), t.lane);
    } else {
      for (int j = jlo + t.wib; j < jhi; j += MK_WARPS)
        int_loop_warp_cell(c.nfiles, c.i, c.length, p.TerminalAU, p.ninio2,
                           (const cuda_param_t *)p.param, p.lxc, p.pair,
                           p.S, p.hccc, p.up_int, p.my_c, p.tri_off_H,
                           p.row_off_H, p.hc_off_H, c.size_off, c.i_H,
                           p.energy_min2, c.H, (size_t)(j - j0), t.lane);
    }
  } else {
    for (long long k = t.gwarp; k < c.width; k += t.nwarps)
      int_loop_warp_cell(c.nfiles, c.i, c.length, p.TerminalAU, p.ninio2,
                         (const cuda_param_t *)p.param, p.lxc, p.pair, p.S, p.hccc,
                         p.up_int, p.my_c, p.tri_off_H, p.row_off_H, p.hc_off_H,
                         c.size_off, c.i_H, p.energy_min2,
                         c.H, (size_t)k, t.lane);
  }
}

__device__ inline void
mk_ph_hp_mb(const rnafold_mk_ptrs_t &p, const mk_ctx_t &c, const mk_thr_t &t)
{
  if (!c.live || (c.skip & MK_SKIP_HP_MB))
    return;

  for (long long k = t.gthr; k < c.width; k += t.nthr)
    hp_mb_3p_cell(c.nfiles, c.i, c.turn_, c.length, p.S2, p.sequence, p.pair2,
                  p.hccc_mb, p.hccc_mbenc, p.hccc_any, p.hccc_gu,
                  (const cuda_param2_t *)p.param2, p.salt_loop,
                  p.energy_hp_row, p.energy_mb_row, p.energy_3p00_row, p.gate_row,
                  p.row_off_H, p.hc2_off_H, p.seq_off_H, p.len_H,
                  c.size_off, c.stotal, c.i_H,
                  (long long)c.sbase + k);
}

/* The join: needs md(i+1)'s DMLi at column j-1, published by the snapshot. */
__device__ inline void
mk_ph_new_c(const rnafold_mk_ptrs_t &p, const mk_ctx_t &c, const mk_thr_t &t)
{
  if (!c.live || (c.skip & MK_SKIP_NEW_C))
    return;

  for (long long k = t.gthr; k < c.width; k += t.nthr)
    new_c_cell(c.nfiles, c.i, c.turn_, c.noGUclosure, p.energy_min2,
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
               p.row_off_H, c.size_off, c.stotal, c.i_H,
               (long long)c.sbase + k);
}

__device__ inline void
mk_ph_load_my_c(const rnafold_mk_ptrs_t &p, const mk_ctx_t &c, const mk_thr_t &t)
{
  if (!c.live || (c.skip & MK_SKIP_LOAD_C))
    return;

  for (long long k = t.gthr; k < c.width; k += t.nthr)
    load_my_c_cell(c.nfiles, c.i, c.length, p.new_e, p.my_c,
                   p.tri_off_H, p.row_off_H, c.size_off, c.stotal, c.i_H,
                   (long long)c.sbase + k);
}

/* Block-cooperative, so ONE block of the half runs it -- the same
 * one-block-per-record shape the standalone kernel has. */
__device__ inline void
mk_ph_scan(const rnafold_mk_ptrs_t &p, const mk_ctx_t &c,
           const int is_lead, int *sa, int *sc)
{
  if (!c.live || !is_lead || (c.skip & MK_SKIP_SCAN))
    return;

  fml_scan_block<MK_BLOCK>(c.nfiles, c.i, c.turn_, p.new_e, p.energy_3p00_row,
                           /* gq_row */ NULL, p.fml_prev, p.up_ml_ok,
                           (const cuda_param2_t *)p.param2, p.energy_min,
                           p.row_off_H, p.seq_off_H, c.size_off, c.i_H,
                           c.H, sa, sc);
}

__device__ inline void
mk_ph_load_fml(const rnafold_mk_ptrs_t &p, const mk_ctx_t &c, const mk_thr_t &t)
{
  if (!c.live || (c.skip & MK_SKIP_LOAD_FML))
    return;

  for (long long k = t.gthr; k < c.width; k += t.nthr)
    load_fML_cell(c.nfiles, c.i, c.turn_, c.length, p.energy_min, p.fml_j,
                  p.fml_row, p.tri_off_H, p.row_off_H, c.size_off, c.stotal,
                  c.i_H, (long long)c.sbase + k);
}

__device__ inline void
mk_ph_fmli(const rnafold_mk_ptrs_t &p, const mk_ctx_t &c, const mk_thr_t &t)
{
  if (!c.live || (c.skip & MK_SKIP_FMLI))
    return;

  for (long long k = t.gthr; k < c.dwidth; k += t.nthr)
    fmli_cell(c.nfiles, c.i, c.turn_, c.length, p.fml_i, p.fml_j, p.fml_row,
              p.tri_off_H, p.row_off_H, c.side_off, c.dtotal, c.i_H,
              (long long)c.dbase + k);
}

/* The decomposition itself: the phase everything else exists for. */
__device__ inline void
mk_ph_md(const rnafold_mk_ptrs_t &p, const mk_ctx_t &c, const mk_thr_t &t)
{
  fml_corner_t cc;

  if (!c.live || (c.skip & MK_SKIP_MD))
    return;

  cc.sm = c.corner; cc.J0 = c.Jown; cc.K = c.corner_k;
  if (c.owns && (c.own_mask & 2)) {
    const int d0  = c.i + 2 * (c.turn_ + 1) + 1;   /* first md column of row i */
    const int dlo = (c.Jown > d0) ? c.Jown : d0;
    const int dhi = (c.Jend < d0 + (int)c.dwidth) ? c.Jend : (d0 + (int)c.dwidth);

    for (int j = dlo + t.wib; j < dhi; j += MK_WARPS)
      md_cell<MK_TILE>(c.nfiles, c.i, c.turn_, c.length, p.fml_i, p.fml_j,
                       p.fml_j16, p.fml_b, p.base_off_H, p.colb_off,
                       p.dml, p.fm2, p.tri_off_H, p.row_off_H,
                       c.side_off, c.dtotal, c.i_H,
                       (long long)c.dbase + (j - d0), t.lane, cc);
  } else {
    for (long long k = t.gwarp; k < c.dwidth; k += t.nwarps)
      md_cell<MK_TILE>(c.nfiles, c.i, c.turn_, c.length, p.fml_i, p.fml_j,
                       p.fml_j16, p.fml_b, p.base_off_H, p.colb_off,
                       p.dml, p.fm2, p.tri_off_H, p.row_off_H,
                       c.side_off, c.dtotal, c.i_H,
                       (long long)c.dbase + k, t.lane, cc);
  }
}

/* Close the row: fold DMLi into fML, and cache row i for the next one. */
__device__ inline void
mk_ph_close(const rnafold_mk_ptrs_t &p, const mk_ctx_t &c, const mk_thr_t &t)
{
  if (!c.live || (c.skip & MK_SKIP_CLOSE))
    return;

  for (long long k = t.gthr; k < c.dwidth; k += t.nthr)
    load_min_fML_cell(c.nfiles, c.i, c.turn_, c.length, p.energy_min, p.dml,
                      p.fml_j, p.fml_row, p.tri_off_H, p.row_off_H,
                      c.side_off, c.dtotal, c.i_H,
                      (long long)c.dbase + k);

  for (long long k = t.gthr; k < c.width; k += t.nthr)
    fml_prev_cell(c.nfiles, c.i, c.turn_, p.energy_min, p.dml, p.fml_prev,
                  p.row_off_H, c.size_off, c.stotal, c.i_H,
                  (long long)c.sbase + k);
}

/* ---- int16: pack row i into the triangle ---------------------------------
 * Runs AFTER both of the row's writers, which is the whole ordering
 * constraint of the encoding: fml_row holds the final row only once
 * load_fML and load_min_fML have both run. A no-op when the gate is off.
 *
 * The baseline claim `fml_b[bidx] = v` is race-free because each thread owns
 * a distinct (column, block) WITHIN a row, and rows are separated -- by a
 * kernel launch on the per-phase path, by a barrier here.
 */
__device__ inline void
mk_ph_pack(const rnafold_mk_ptrs_t &p, const mk_ctx_t &c, const mk_thr_t &t)
{
  if (!c.live || !p.fml_row || (c.skip & MK_SKIP_PACK))
    return;

  for (long long k = t.gthr; k < c.width; k += t.nthr)
    pack_fml_cell(c.nfiles, c.i, c.turn_, c.length, p.fml_row, p.fml_j16, p.fml_b,
                  p.tri_off_H, p.row_off_H, p.base_off_H, p.colb_off,
                  c.size_off, c.stotal, c.i_H,
                  (long long)c.sbase + k);
}

/* ---- stage 2: row i joins the corner cache -------------------------------
 * Row i of fML is final now. Entry (i,j) sits at offset (j-turn-1)-i of
 * column j and is first READ by md at sweep row i-turn-2, so it is on chip
 * well before anyone wants it. Read back from fml_row under int16 -- it is
 * this row in full int32, so the cache never holds a packed delta.
 */
__device__ inline void
mk_ph_corner_fill(const rnafold_mk_ptrs_t &p, const mk_ctx_t &c)
{
  if (!c.corner || !c.live)
    return;
  {
    const int lenH = p.len_H[c.H];

    for (int j = c.Jown + (int)threadIdx.x; j < c.Jend; j += (int)blockDim.x) {
      const int off = (j - c.turn_ - 1) - c.i;

      if ((off >= 0) && (off < c.corner_k) && (j <= lenH) && (j >= c.i + c.turn_ + 1))
        c.corner[(size_t)(j - c.Jown) * (size_t)c.corner_k + (size_t)off] =
          p.fml_row ? p.fml_row[p.row_off_H[c.H] + j]
                    : p.fml_j[p.tri_off_H[c.H] + Indx(c.i, j)];
    }
  }
}

/* md_snapshot_dml()'s device-to-device copy, restricted to this record's row.
 * The standalone path copies the whole batch's row buffer on the md stream;
 * here the grid that just wrote it copies its own slice. It only READS dml,
 * so it can share a phase with anything else that reads dml. */
__device__ inline void
mk_ph_snapshot(const rnafold_mk_ptrs_t &p, const mk_ctx_t &c, const mk_thr_t &t)
{
  if (!c.live || (c.skip & MK_SKIP_TAIL))
    return;
  {
    const size_t lo = p.row_off_H[c.H];
    const size_t hi = p.row_off_H[c.H + 1];

    for (size_t k = lo + (size_t)t.gthr; k < hi; k += (size_t)t.nthr)
      p.dml1[k] = p.dml[k];
  }
}

/*
 *  A barrier over a SUBSET of the grid's blocks.
 *
 *  Stage 3 runs two independent chains on two disjoint halves of the grid, and
 *  cooperative groups offers no barrier over part of a grid -- grid.sync() is
 *  all or nothing. But the halves only meet twice a row, so making every one
 *  of the fML chain's five internal steps a FULL barrier would make the c half
 *  wait on work it does not depend on, which is the whole thing stage 3 exists
 *  to stop.
 *
 *  Sense-reversing, with the generation supplied by the caller: every block of
 *  a half executes the same number of these, so the counter is the same in all
 *  of them and no extra broadcast is needed. Legal only because a cooperative
 *  launch guarantees every block is resident -- a block that had not been
 *  scheduled would never arrive and the spin would never end.
 */
__device__ inline void
mk_half_sync(unsigned int *bar, const unsigned int nblocks, const unsigned int gen)
{
  /* RELEASE. __syncthreads() orders memory within a block and nothing more, so
   * without this a block's phase writes can still be sitting in its own L1 when
   * the other half-blocks are let go, and the next phase reads stale data. That
   * is a silent wrong answer, not a hang, and it is exactly what this barrier
   * got wrong first time: fML values from the previous row survived into the
   * next one on some blocks and not others, so the fold changed with the block
   * count. grid.sync() does this internally, which is why only the half barrier
   * was affected. */
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    if (atomicAdd(&bar[0], 1u) == nblocks - 1u) {
      bar[0] = 0u;
      __threadfence();
      atomicExch(&bar[1], gen);
    } else {
      while (atomicAdd(&bar[1], 0u) != gen) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(64);          /* back off; sm_70+ only */
#endif
      }
    }
  }
  __syncthreads();
  /* ACQUIRE: everything the other blocks published before arriving is now
   * visible to this one. */
  __threadfence();
}

/*
 *  One record's whole sweep.
 *
 *  Blocks stride over the row's cells; the grid is sized so that every block is
 *  resident, which is what makes grid.sync() legal. `H` is the record's slot,
 *  and every buffer below is the batch-flattened one it already lives in -- the
 *  record's own region is picked out by the offset tables exactly as the
 *  per-phase kernels do it.
 */
/* Residency IS the design, so the register budget is stated rather than left
 * to the compiler: without this the refactor that gave stage 3 its second row
 * context dropped the grid from 3 resident blocks per SM to 1. */
__global__ void __launch_bounds__(MK_BLOCK, 3)
megakernel_record(const rnafold_mk_ptrs_t p,
                  const int nfiles, const int H, const int turn_, const int length,
                  const int i_top, const int noGUclosure,
                  unsigned long long *clocks, const int skip,
                  volatile unsigned int *prog)
{
  cg::grid_group grid = cg::this_grid();

  mk_thr_t t;

  /*
   *  The two row contexts are BLOCK-UNIFORM -- every thread of a block would
   *  hold the same twenty-odd values -- so they live in shared memory rather
   *  than in every thread's registers. Holding them per-thread cost two thirds
   *  of the occupancy: the banner went from "3 resident blocks per SM" to 1
   *  the moment stage 3 needed a second context, which is a 3x loss on a
   *  kernel whose whole premise is staying resident.
   */
  __shared__ mk_ctx_t s_cc, s_cm;
  mk_ctx_t &cc = s_cc;
  mk_ctx_t &cm = s_cm;

  /* fml_scan's tiles. Every block allocates them; one block per half runs the
   * scan, which is the same one-block-per-record shape the standalone kernel
   * has. */
  __shared__ int sa[MK_BLOCK];
  __shared__ int sc[MK_BLOCK];

  /*
   *  Stages 1b and 2 live here, and both need the SAME thing from the
   *  schedule: a block must own a FIXED range of absolute columns for the
   *  whole sweep, or nothing it caches survives to the next row.
   *
   *  That is a real trade. Striding cells (stage 0) balances every row
   *  perfectly; owning columns leaves a block idle until the sweep reaches its
   *  range, because row i only has columns [i+turn+1, len]. An equal-width
   *  split is therefore NOT work-balanced -- column j carries (j-turn-1) cells
   *  over the sweep, so a balanced split would put the boundaries at
   *  j = len*sqrt(b/B). Left for the tuning pass: a balanced split makes the
   *  widest block ~len/sqrt(B) columns wide, and the window is sized by the
   *  WIDEST block, so it costs shared memory exactly where there is none.
   */
  extern __shared__ int mk_dyn[];

  /* Stage 3 splits the grid. `p.split` is the number of c-chain blocks; 0
   * means one undivided grid running the old schedule. */
  const int nC     = p.split;
  const int skew   = (nC > 0);
  const int isC    = skew ? ((int)blockIdx.x <  nC) : 1;
  const int nM     = skew ? ((int)gridDim.x - nC) : (int)gridDim.x;
  const int half_n = skew ? (isC ? nC : nM) : (int)gridDim.x;
  const int half_b = skew ? (isC ? (int)blockIdx.x : ((int)blockIdx.x - nC))
                          : (int)blockIdx.x;
  /* Each half partitions the columns among ITS OWN blocks. */
  const int cwW    = skew ? (isC ? p.cw_cols : p.cw_cols_m) : p.cw_cols;
  const int cwC    = cwW + MK_CW_BACK;

  t.lane   = (int)(threadIdx.x & 31u);
  t.wib    = (int)(threadIdx.x >> 5);
  t.gwarp  = (long long)half_b * MK_WARPS + t.wib;
  t.nwarps = (long long)half_n * MK_WARPS;
  t.gthr   = (long long)half_b * blockDim.x + threadIdx.x;
  t.nthr   = (long long)half_n * blockDim.x;

  if (threadIdx.x == 0) {
  cc.nfiles = nfiles; cc.H = H; cc.turn_ = turn_; cc.length = length;
  cc.noGUclosure = noGUclosure; cc.skip = skip;
  cc.owns     = (cwW != 0);
  cc.own_mask = p.own_mask;
  cc.Jown    = turn_ + 2 + half_b * cwW;
  cc.Jend    = cc.Jown + cwW;
  cc.Qlo     = cc.Jown - MK_CW_BACK;
  cc.cwC     = cwC;
  cc.corner_k = 0;
  cc.cring    = NULL;
  cc.corner   = NULL;

  /* In the skew the halves hold DIFFERENT caches -- the c chain needs the ring
   * and nothing else, the fML chain the corner and nothing else -- so each
   * block allocates one, and the shared budget goes twice as far. */
  if (cwW && p.cw_on && (!skew || isC))
    cc.cring = mk_dyn;
  if (cwW && p.corner_k && (!skew || !isC))
    cc.corner = mk_dyn + ((cc.cring && !skew) ? 32 * cwC : 0);
  cc.corner_k = cc.corner ? p.corner_k : 0;

  cm = cc;   /* the two contexts differ only in the row they describe */
  }
  __syncthreads();

  {
    long long _mark = clock64();
    unsigned int gen = 0u;
    unsigned int *const bar = skew
      ? (unsigned int *)(p.bar + (isC ? 0 : 2))
      : NULL;

    /*
     *  STAGE 3, the one-row skew.
     *
     *  The c chain for row i needs only `c` rows >= i+1 and the sequence, so it
     *  can run WHILE the fML chain finishes row i+1. The two meet exactly
     *  twice: new_c(i) reads dml1, which is the snapshot of md(i+1), and
     *  fml_scan(i+1) reads new_e and e3p00, which the c chain wrote a row ago.
     *  Everything else is internal to one half, and internal steps use a
     *  barrier over that half only.
     *
     *  So the sweep runs to i=0: the c chain finishes at row 1 and the fML
     *  chain still owes row 1 one more pass.
     */
    for (int i = i_top; i >= (skew ? 0 : 1); i--) {
      /* One writer, and a barrier on each side: the previous iteration's
       * phases may still be reading these when this one wants to move them. */
      __syncthreads();
      if (threadIdx.x == 0) {
        mk_ctx_row(&cc, p, i);                      /* c chain: row i     */
        /* Guard BEFORE the read, not after: the row tables hold one slot per
         * sweep row, so row i_top+1 is one past the end for the longest
         * record in the chunk. */
        if (!skew)
          mk_ctx_row(&cm, p, i);                    /* fML chain: row i   */
        else if (i + 1 <= i_top)
          mk_ctx_row(&cm, p, i + 1);                /* fML chain: row i+1 */
        else
          cm.live = 0;                              /* no such row yet    */
      }
      __syncthreads();

      if (clocks && (blockIdx.x == 0) && (threadIdx.x == 0)) {
        if (i == i_top) {
          clocks[MK_DBG_WIDTH]  = (unsigned long long)cc.width;
          clocks[MK_DBG_DWIDTH] = (unsigned long long)cc.dwidth;
        }
        clocks[MK_DBG_ROWS]  += 1ull;
        clocks[MK_DBG_CELLS] += (unsigned long long)(cc.width > 0 ? cc.width : 0);
      }

      if (!skew) {
        /* ---------------- the undivided schedule (stages 0-2) ------------- */
        mk_ring_refresh(p, cc, i == i_top);
        mk_ph_int_loop(p, cc, t);
        MK_TICK(MK_PH_INT_LOOP);
        mk_ph_hp_mb(p, cc, t);
        MK_TICK(MK_PH_HP_MB);

        if (clocks && (blockIdx.x == 0) && (threadIdx.x == 0) && (i == i_top) && (cc.width > 0)) {
          const int j = (int)(i + turn_ + 1);
          clocks[MK_DBG_IH]     = (unsigned long long)(unsigned int)cc.i_H[H];
          clocks[MK_DBG_ROWOFF] = (unsigned long long)p.row_off_H[H];
          clocks[MK_DBG_EMIN2]  = (unsigned long long)(unsigned int)p.energy_min2[p.row_off_H[H] + j];
          clocks[MK_DBG_HP]     = (unsigned long long)(unsigned int)p.energy_hp_row[p.row_off_H[H] + j];
          clocks[MK_DBG_GATE]   = (unsigned long long)(unsigned char)p.gate_row[p.row_off_H[H] + j];
        }

        MK_SYNC(); MK_TICK(MK_PH_SYNC);
        mk_ph_new_c(p, cc, t);
        MK_TICK(MK_PH_NEW_C);

        if (clocks && (blockIdx.x == 0) && (threadIdx.x == 0) && (i == i_top) && (cc.width > 0))
          clocks[MK_DBG_NEWE] = (unsigned long long)(unsigned int)p.new_e[p.row_off_H[H] + i + turn_ + 1];

        MK_SYNC(); MK_TICK(MK_PH_SYNC);
        mk_ph_load_my_c(p, cc, t);
        MK_TICK(MK_PH_LOAD_C);
        mk_ph_scan(p, cc, blockIdx.x == 0, sa, sc);
        MK_TICK(MK_PH_SCAN);

        MK_SYNC(); MK_TICK(MK_PH_SYNC);
        mk_ph_load_fml(p, cc, t);
        MK_TICK(MK_PH_LOAD_FML);

        MK_SYNC(); MK_TICK(MK_PH_SYNC);
        mk_ph_fmli(p, cc, t);

        MK_SYNC(); MK_TICK(MK_PH_SYNC);
        mk_ph_md(p, cc, t);
        MK_TICK(MK_PH_MD);

        MK_SYNC(); MK_TICK(MK_PH_SYNC);
        mk_ph_close(p, cc, t);

        MK_SYNC(); MK_TICK(MK_PH_SYNC);
        mk_ph_pack(p, cc, t);
        MK_TICK(MK_PH_PACK);
        mk_ph_corner_fill(p, cc);
        mk_ph_snapshot(p, cc, t);
        MK_TICK(MK_PH_TAIL);

        MK_SYNC(); MK_TICK(MK_PH_SYNC);
      } else if (isC) {
        /* ---------------- stage 3: the c half ----------------------------- */
        mk_ring_refresh(p, cc, i == i_top);
        mk_ph_int_loop(p, cc, t);
        MK_TICK(MK_PH_INT_LOOP);

        /* Meet 1: the fML half has published md(i+1)'s snapshot into dml1. */
        MK_SYNC(); MK_TICK(MK_PH_SYNC);

        /* hp_mb belongs AFTER the meet, not beside int_loop.
         *
         * energy_3p00_row is ROW-SHAPED -- one row per record, rewritten every
         * sweep row -- and fml_scan(i+1) over in the other half reads row i+1
         * out of it. Running hp_mb(i) in the overlapped part therefore has one
         * half overwriting the very buffer the other half is reading, which is
         * a silent, non-deterministic wrong answer: it reproduced at 8 and 12
         * blocks per half and vanished at 24, purely on timing.
         *
         * So the overlap is int_loop(i) against the fML chain, and everything
         * of the c chain that touches a shared row buffer waits for the meet.
         * The row buffers the c half owns outright -- energy_min2 -- stay in
         * the overlapped part. */
        mk_ph_hp_mb(p, cc, t);
        MK_TICK(MK_PH_HP_MB);
        mk_half_sync(bar, (unsigned)nC, ++gen);
        mk_ph_new_c(p, cc, t);
        MK_TICK(MK_PH_NEW_C);
        mk_half_sync(bar, (unsigned)nC, ++gen);
        mk_ph_load_my_c(p, cc, t);
        MK_TICK(MK_PH_LOAD_C);

        /* Meet 2: new_e(i) is now readable by the fML half's next scan. */
        MK_SYNC(); MK_TICK(MK_PH_SYNC);
      } else {
        /* ---------------- stage 3: the fML half --------------------------- */
        mk_ph_scan(p, cm, half_b == 0, sa, sc);
        MK_TICK(MK_PH_SCAN);
        mk_half_sync(bar, (unsigned)nM, ++gen);

        mk_ph_load_fml(p, cm, t);
        MK_TICK(MK_PH_LOAD_FML);
        mk_half_sync(bar, (unsigned)nM, ++gen);

        mk_ph_fmli(p, cm, t);
        mk_half_sync(bar, (unsigned)nM, ++gen);

        mk_ph_md(p, cm, t);
        MK_TICK(MK_PH_MD);
        mk_half_sync(bar, (unsigned)nM, ++gen);

        /* close() writes fML and fml_prev; the snapshot only READS dml, which
         * md finished above, so the two share this step. */
        mk_ph_close(p, cm, t);
        mk_ph_snapshot(p, cm, t);
        MK_TICK(MK_PH_TAIL);
        mk_half_sync(bar, (unsigned)nM, ++gen);

        mk_ph_pack(p, cm, t);
        MK_TICK(MK_PH_PACK);
        mk_ph_corner_fill(p, cm);

        /* Meet 1. */
        MK_SYNC(); MK_TICK(MK_PH_SYNC);
        /* Meet 2: nothing to do between them -- the c half is finishing row i
         * and the fML half cannot start row i until new_e(i) exists. */
        MK_SYNC(); MK_TICK(MK_PH_SYNC);
      }
    }
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
             const int nC,
             int *cw_cols, int *cw_cols_m, int *cw_on, int *corner_k, size_t *bytes)
{
  const int  want_ring   = mk_env("RNA_MK_CWIN", 1);
  const int  want_corner = mk_env("RNA_MK_CORNER", 1);
  const int  budget_kb   = mk_env("RNA_MK_SMEM_KB", 32);
  const size_t budget    = (size_t)budget_kb * 1024u;
  const int  span        = (length > turn + 1) ? (length - turn - 1) : 1;
  /* With the skew each half partitions the span among its OWN blocks, so the
   * two widths differ -- and because a block then holds only ONE of the two
   * caches, the budget covers the LARGER of them rather than their sum. That
   * is stage 3 paying for itself in shared memory before it saves a cycle. */
  const int  nM          = (nC > 0) ? (blocks - nC) : blocks;
  int        W           = (span + ((nC > 0) ? nC : blocks) - 1) / ((nC > 0) ? nC : blocks);
  int        Wm          = (span + nM - 1) / ((nM > 0) ? nM : 1);
  int        K           = want_corner ? mk_env("RNA_MK_CORNER_K", 32) : 0;
  int        ring        = want_ring ? 1 : 0;

  *cw_cols = 0; *cw_cols_m = 0; *cw_on = 0; *corner_k = 0; *bytes = 0;
  if (W < 1)  W = 1;
  if (Wm < 1) Wm = 1;
  if (!ring && !K) {
    if (mk_env("RNA_MK_FORCE_OWNS", 0)) {
      *cw_cols = W; *cw_cols_m = Wm;   /* ownership alone, for bisection */
      return;
    }
    return;                      /* both off: stage 0's cell striding */
  }

  for (;;) {
    const size_t nring = ring ? (size_t)32 * (size_t)(W + MK_CW_BACK) * sizeof(int) : 0;
    const size_t ncorn = (size_t)K * (size_t)((nC > 0) ? Wm : W) * sizeof(int);
    const size_t need  = (nC > 0) ? ((nring > ncorn) ? nring : ncorn)
                                  : (nring + ncorn);

    if (need <= budget) {
      *cw_cols = W; *cw_cols_m = Wm; *cw_on = ring; *corner_k = K; *bytes = need;
      return;
    }
    if (K > 8)        K /= 2;        /* the corner degrades gracefully */
    else if (K)       K = 0;
    else if (ring)    ring = 0;      /* then the ring */
    else              return;        /* then column ownership itself */

    /* Ownership without a cache is the WORST combination there is: it keeps
     * the load imbalance -- a block idle until the sweep reaches its columns --
     * and buys nothing back. Measured on an 8001 nt record, where a 32 KB
     * budget left W=250 too wide for the ring: the fold went from seconds to
     * not finishing. So when both caches are gone, so is ownership. */
    if (!ring && !K) {
      /* RNA_MK_FORCE_OWNS keeps column ownership alive with no cache behind
       * it. Useless in production -- that is the combination the check above
       * exists to prevent -- but it is the only way to ask whether a bug
       * belongs to the SCHEDULE or to a CACHE. */
      if (!mk_env("RNA_MK_FORCE_OWNS", 0))
        return;
      *cw_cols = W; *cw_cols_m = Wm; *cw_on = 0; *corner_k = 0; *bytes = 0;
      return;
    }
  }
}

/*
 *  Stage 3: how many of the grid's blocks run the c chain.
 *
 *  RNA_MK_SKEW is the percentage; 0 turns the skew off and restores the single
 *  undivided schedule. The default split is even, which is a guess, not a
 *  measurement -- at the local phase mix the c half carries int_loop+hp_mb and
 *  the fML half carries md plus five lighter phases, which is close to even in
 *  work but says nothing about how they overlap.
 */
static int
mk_split_blocks(const int blocks)
{
  const int pct = mk_env("RNA_MK_SKEW", 0);
  int       nC;

  if ((pct <= 0) || (blocks < 4))
    return 0;                      /* too few blocks to divide usefully */
  nC = (blocks * ((pct > 90) ? 90 : pct)) / 100;
  if (nC < 1)          nC = 1;
  if (nC > blocks - 1) nC = blocks - 1;
  return nC;
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
  int    mk_cw_cols = 0, mk_cw_cols_m = 0, mk_cw_on = 0, mk_corner_k = 0;
  int    mk_split = 0;
  unsigned int *d_bar = NULL;
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
    int    cw = 0, cwm = 0, on = 0, kk = 0;

    mk_split = mk_split_blocks(blocks);
    mk_plan_smem(blocks, length, turn, mk_split, &cw, &cwm, &on, &kk, &smem);
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
        mk_plan_smem(blocks, length, turn, mk_split, &cw, &cwm, &on, &kk, &smem);
      }
    }
    mk_smem_bytes = smem;
    mk_cw_cols = cw; mk_cw_cols_m = cwm; mk_cw_on = on; mk_corner_k = kk;
    fprintf(stderr, "megakernel.cu            on-chip: %d columns per block, "
                    "c-ring %s, fML corner K=%d, %zu KB shared\n",
            cw, on ? "ON" : "off", kk, smem / 1024);
    if (cw && (blocks > 1))
      fprintf(stderr, "megakernel.cu            NOTE: column ownership is not "
                      "work-balanced -- a block is idle until the sweep reaches "
                      "its columns\n");
    if (mk_split)
      fprintf(stderr, "megakernel.cu            SKEW: %d blocks on the c chain "
                      "(row i, %d cols each), %d on the fML chain (row i+1, %d "
                      "cols each)\n", mk_split, cw, blocks - mk_split, cwm);
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
  p.cw_cols_m  = mk_cw_cols_m;
  p.cw_on      = mk_cw_on;
  p.corner_k   = mk_corner_k;
  p.split      = mk_split;
  p.own_mask   = mk_env("RNA_MK_OWN_MASK", 3);
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

  /* Four counters per stream: two halves x (arrivals, generation). A launch
   * must start from zero, and the stream serialises the records that share a
   * slot, so the reset rides the same stream as the launch. */
  if (mk_split &&
      (cudaMalloc((void **)&d_bar, (size_t)G * 4u * sizeof(unsigned int)) != cudaSuccess)) {
    fprintf(stderr, "megakernel.cu            no barrier memory -- skew off\n");
    cudaGetLastError();
    d_bar = NULL;
    mk_split = 0;
    p.split  = 0;
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
      if (d_bar) {
        a_p.bar = d_bar + (size_t)(k % G) * 4u;
        cudaMemsetAsync(a_p.bar, 0, 4u * sizeof(unsigned int), streams[k % G]);
      }
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
  if (d_bar)  cudaFree(d_bar);
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
