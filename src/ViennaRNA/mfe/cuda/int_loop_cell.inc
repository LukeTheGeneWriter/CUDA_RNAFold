/* int_loop_cell.inc -- one cell of the interior-loop sweep, as a device function.
 *
 * Lifted verbatim out of int_loop_warp_kernel so the fused per-record
 * megakernel runs THE SAME arithmetic (device code cannot cross a translation
 * unit here: no -rdc, no device link).
 *
 * The kernel's template parameters -- CELLS_PER_BLOCK, GRIDY, WSEARCH -- exist
 * only to DERIVE the cell (H, local) and to take the early returns. Nothing
 * below that point uses them, so this function takes the cell identity as
 * parameters and needs no template. The megakernel has its own mapping.
 *
 * One warp per cell: every lane of the warp must call this together, and `lane`
 * is that lane. The reduction at the end names all 32.
 */

template<class CREAD>
__device__ inline void
int_loop_warp_cell_r(const int nfiles, const int i_row, const int length,
                const int TerminalAU, const int ninio2,
                const cuda_param_t* __restrict__ P, const float lxc,
                const char* __restrict__ pair_,
                const unsigned int* __restrict__ S,
                const unsigned int* __restrict__ hccc,
                const unsigned char* __restrict__ up_int, //in, d_up_int -- NULL when unconstrained
                CREAD cread,                  /* where `c` is read FROM */
                const size_t* __restrict__ row_off_H,
                const size_t* __restrict__ hc_off_H,
                const size_t* __restrict__ size_off_H,
                const int* __restrict__ i_H,
                      int* __restrict__ energy_min,
                   const int H, const size_t local, const int lane) {
  const int i = i_H[H];
  assert(i_row < 0 || i == i_row);
  const int j = (int)local + i + turn + 1;

  const long long ij = Indx(i,j);
  int energy = INF;

  if(Hc(ij,&hccc[hc_off_H[H]])) {
    const int p0 = i+1;
    const int q0 = Min_q(i,j,turn);
    const int maxcol = MIN2(MAXLOOP,(j - 1) - q0);
    const unsigned int* __restrict__ hccc_H = &hccc[hc_off_H[H]];

    // H1: once per cell, not once per candidate.
    const cell_inv_t ci = cell_invariants(S,pair_,H,nfiles,i,j);

    // LANE c OWNS COLUMN c, in registers. maxcol <= MAXLOOP = 30 always, so one
    // warp covers every column there can be -- which is what makes the whole
    // design possible.
    unsigned int my_mask = 0u;
    int          popc    = 0;
    if(lane <= maxcol) {
      my_mask = decode_column(p0,q0,lane,hccc_H);
      popc    = __popc(my_mask);
    }

    // Inclusive Hillis-Steele scan of popc, unchanged from the twin except that
    // the result stays in registers. Lanes past maxcol contribute 0 and cannot
    // affect lower lanes, since __shfl_up_sync only pulls from lower lanes.
    int incl = popc;
#pragma unroll
    for(int off = 1; off < 32; off <<= 1) {
      const int nv = __shfl_up_sync(0xffffffff, incl, off);
      if(lane >= off) incl += nv;
    }
    const int excl  = incl - popc;                              // prefix[c]
    const int total = __shfl_sync(0xffffffff, incl, 31);        // prefix[maxcol+1]

    const int iters = (total + 31) >> 5;                        // warp-uniform
    for(int k = 0; k < iters; k++) {
      const int  w   = (k << 5) + lane;
      const bool has = (w < total);

      // Smallest column c with prefix[c+1] > w. FIVE FIXED STEPS, not
      // `while(lo < hi)`: 2^5 = 32 covers every column, and a fixed trip count
      // keeps all 32 lanes in every shuffle. Lanes that have already converged
      // (lo == hi) still execute the shuffle and discard it.
      int lo = 0;
      int hi = (maxcol > 0) ? maxcol : 0;
#pragma unroll
      for(int s = 0; s < 5; s++) {
        const int  mid = (lo + hi) >> 1;
        const int  v   = __shfl_sync(0xffffffff, incl, mid);
        const bool act = (lo < hi);
        const bool go  = act && (v > w);
        hi = go ? mid : hi;
        lo = (act && !go) ? (mid + 1) : lo;
      }
      const int column = lo;

      // Column `column`'s mask and prefix, gathered from the lane that owns
      // them. A per-lane source index is a legal gather, and both are register
      // reads -- this is the shared-memory traffic the twin pays.
      const unsigned int cmask = __shfl_sync(0xffffffff, my_mask, column);
      const int          cbase = __shfl_sync(0xffffffff, excl,    column);

      if(has) {
        int popc_unused;
        const int row = find_nth_set_bit(cmask, w - cbase, popc_unused);
        assert(row >= 0);
        const int p = p0 + row;
        const int q = q0 + column;
        const int energy2 = Energy(H,nfiles,i,j,q,p, ci,
                      cread,
                      up_int ? &up_int[row_off_H[H]] : NULL,
                      S,pair_,P,
                      TerminalAU,ninio2,
                      P->bulge,P->internal_loop,lxc,
                      P->mismatchI,
                      P->mismatch1nI,
                      P->mismatch23I,
                      P->stack,
                      P->int11,
                      P->int21,
                      P->int22);
        energy = MIN2(energy,energy2);
      }
    }
  }

  // Warp-wide min. No shared memory and no __syncthreads() at any block size --
  // which is the whole point.
  // The warp must be CONVERGED here. Every shuffle below names all 32 lanes,
  // and the lanes reach this point from divergent branches above (Hc, and the
  // per-lane  inside the candidate loop). The standalone kernel gets away
  // without this because the cell IS the kernel body -- one warp, one cell, and
  // the compiler reconverges at the return. Inlined into the megakernel's
  //  loop that guarantee is gone, and
  // under independent thread scheduling the lanes STRANDED at different
  // unrolled shuffles: measured lane 0 at off=1, lane 1 finished, lanes 2-31 at
  // off=4, each waiting for a mask that can never assemble.
  __syncwarp();
#pragma unroll
  for(int off = 16; off > 0; off >>= 1) {
    /* The shuffle is hoisted OUT of MIN2 deliberately. MIN2 is a macro that
     * evaluates its second argument TWICE -- once in the comparison and once in
     * the false branch -- so `MIN2(x, __shfl_down_sync(...))` makes each lane
     * execute the second shuffle only if its own comparison came out false.
     * Lanes then run DIFFERENT numbers of full-mask shuffles and the warp
     * deadlocks. It survived for a year because the compiler CSEs the two
     * identical calls in the standalone kernels; inlined into the fused
     * megakernel it does not, and the warp stranded mid-reduction (measured:
     * lane 0 at off=1, lane 1 finished, lanes 2-31 at off=4). */
    const int other = __shfl_down_sync(0xffffffff, energy, off);

    energy = MIN2(energy, other);
  }

  if(lane == 0) energy_min[row_off_H[H]+j] = energy;
}


/*
 *  The triangle-backed form, which is what every standalone kernel calls: same
 *  signature it has always had, so those call sites are untouched.
 */
__device__ inline void
int_loop_warp_cell(const int nfiles, const int i_row, const int length,
                const int TerminalAU, const int ninio2,
                const cuda_param_t* __restrict__ P, const float lxc,
                const char* __restrict__ pair_,
                const unsigned int* __restrict__ S,
                const unsigned int* __restrict__ hccc,
                const unsigned char* __restrict__ up_int,
                const int* __restrict__ my_c,
                const size_t* __restrict__ tri_off_H,
                const size_t* __restrict__ row_off_H,
                const size_t* __restrict__ hc_off_H,
                const size_t* __restrict__ size_off_H,
                const int* __restrict__ i_H,
                      int* __restrict__ energy_min,
                   const int H, const size_t local, const int lane) {
  c_tri_reader cread;

  cread.base = &my_c[tri_off_H[H]];
  int_loop_warp_cell_r(nfiles, i_row, length, TerminalAU, ninio2, P, lxc,
                       pair_, S, hccc, up_int, cread, row_off_H, hc_off_H,
                       size_off_H, i_H, energy_min, H, local, lane);
}
