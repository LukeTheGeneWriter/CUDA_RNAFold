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

/* U = how many candidates one lane keeps in flight. U=1 is the shape this kernel
 * has always had and compiles to it exactly; U=2 runs two independent column
 * searches and two accumulators. See the comment on the loop below for why that is
 * the lever the stall mix points at. Defaulted so every existing caller -- including
 * the megakernel -- keeps today's code with no edit. */
template<class CREAD, int U = 1>
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
                   const int H, const size_t local, const int lane,
                   /* RNA_INT_LOOP_WORK_STATS: per-cell candidate count, or NULL. One
                    * store per CELL, so it costs nothing measurable and it is the only
                    * way to see the work DISTRIBUTION rather than its average. */
                   int* __restrict__ work_out = NULL) {
  const int i = i_H[H];
  assert(i_row < 0 || i == i_row);
  const int j = (int)local + i + turn + 1 + (int)size_off_H[nfiles + 1];

  const long long ij = Indx(i,j);
  int energy = INF;

  /* Every launched cell writes, so a cell hc rules out shows as 0 work -- it
   * still cost a warp -- and an untouched slot (the caller pre-fills -1) is a
   * row position that is not a cell at all. */
  if(work_out && lane == 0) work_out[row_off_H[H]+j] = 0;

  if(Hc(ij,&hccc[hc_off_H[H]])) {
    const int p0 = i+1;
    const int q0 = Min_q(i,j,turn);
    const int maxcol = MIN2(MAXLOOP,(j - 1) - q0);
    const unsigned int* __restrict__ hccc_H = &hccc[hc_off_H[H]];

    // H1: once per cell, not once per candidate.
    const cell_inv_t ci = cell_invariants(S,pair_,H,S_STRIDE(length),i,j);

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

    /* WHY U EXISTS, measured rather than assumed.
     *
     * ncu on an RTX 3050 (3 launches of this kernel, warp-per-cell, block 32):
     *
     *   wait (fixed-latency dependency)   2.693   <-- 46% of all stalling
     *   long_scoreboard (global/L1)       1.210
     *   short_scoreboard (shared)         0.700
     *   not_selected                      0.680
     *   math_pipe_throttle                0.553
     *   barrier / mio_throttle / lg       0.000
     *
     * with DRAM throughput 0.83%, L2 2.43%, sectors per global load request 2.46 and
     * 83% lane efficiency. So this kernel is NOT memory-bound in any sense: its own
     * my_c window is already L1-resident and already reasonably coalesced. That kills
     * the lever PORT_MD_BLOCKING_SCOPE.md proposed for it -- staging the 30x30 window
     * in shared at "15.5x fewer loads" -- because the loads are neither the stall nor
     * the bytes, and staging would move L1 hits to shared hits.
     *
     * `wait` is a FIXED-LATENCY dependency stall, i.e. a register chain. Two of them
     * are right here. Per candidate the body runs SEVEN warp shuffles -- five for the
     * binary search, where each step's `mid` depends on the previous step's lo/hi, plus
     * two gathers -- and `energy = MIN2(energy, energy2)` is a single accumulator, so
     * each iteration also waits on the previous one's min.
     *
     * U candidates in flight gives the scheduler U independent chains to interleave.
     * Exact for any U: min over int is associative and commutative, and Energy()
     * returns INF rather than doing arithmetic on it (it guards the my_c read), so INF
     * stays a sentinel and no reassociation can promote a near-INF sum.
     *
     * MEASURED AT U=2 on that same laptop: stalls 5.837 -> 5.253 total, wait 2.693 ->
     * 2.467, short_scoreboard 0.700 -> 0.350, issue_active 47.3% -> 50.4% -- and
     * instructions 5.075e6 -> 5.431e6, +7%. Wall, ABAB, byte-identical: 5.74 s against
     * 5.80 s, i.e. a NULL. Fewer stalls times more work came out even, which is the
     * same 1:1 trade blocking hit on md. It stays behind RNA_INT_LOOP_UNROLL (default
     * 1 = off) because a laptop whose clock drifts 30% inside one ABAB cannot resolve
     * it, and because the A100's latency/throughput balance is the one that matters.
     *
     * `iters`, `k` and `U` are all warp-uniform, so every guard below is warp-uniform
     * and each __shfl_sync still names all 32 lanes -- the property the reduction
     * comment further down was written about. */
    int acc[U];
#pragma unroll
    for(int u = 0; u < U; u++) acc[u] = INF;

    for(int k = 0; k < iters; k += U) {
#pragma unroll
      for(int u = 0; u < U; u++) {
        const bool live = (U == 1) || ((k + u) < iters);   /* warp-uniform */
        const int  w    = ((k + u) << 5) + lane;
        const bool has  = live && (w < total);

        // Smallest column c with prefix[c+1] > w. FIVE FIXED STEPS, not
        // `while(lo < hi)`: 2^5 = 32 covers every column, and a fixed trip count
        // keeps all 32 lanes in every shuffle. Lanes that have already converged
        // (lo == hi) still execute the shuffle and discard it.
        int lo = 0;
        int hi = (maxcol > 0) ? maxcol : 0;
#pragma unroll
        for(int st = 0; st < 5; st++) {
          const int  mid = (lo + hi) >> 1;
          const int  v   = live ? __shfl_sync(0xffffffff, incl, mid) : 0;
          const bool act = (lo < hi);
          const bool go  = act && (v > w);
          hi = go ? mid : hi;
          lo = (act && !go) ? (mid + 1) : lo;
        }
        const int column = lo;

        // Column `column`'s mask and prefix, gathered from the lane that owns
        // them. A per-lane source index is a legal gather, and both are register
        // reads -- this is the shared-memory traffic the twin pays.
        const unsigned int cmask = live ? __shfl_sync(0xffffffff, my_mask, column) : 0u;
        const int          cbase = live ? __shfl_sync(0xffffffff, excl,    column) : 0;

        if(has) {
          int popc_unused;
          const int row = find_nth_set_bit(cmask, w - cbase, popc_unused);
          assert(row >= 0);
          const int p = p0 + row;
          const int q = q0 + column;
          const int energy2 = Energy(H,S_STRIDE(length),i,j,q,p, ci,
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
          acc[u] = MIN2(acc[u],energy2);
        }
      }
    }
#pragma unroll
    for(int u = 0; u < U; u++) energy = MIN2(energy, acc[u]);
    if(work_out && lane == 0) work_out[row_off_H[H]+j] = total;
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
template<int U = 1>
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
                   const int H, const size_t local, const int lane,
                   /* RNA_INT_LOOP_WORK_STATS: per-cell candidate count, or NULL. One
                    * store per CELL, so it costs nothing measurable and it is the only
                    * way to see the work DISTRIBUTION rather than its average. */
                   int* __restrict__ work_out = NULL) {
  c_tri_reader cread;

  cread.base = &my_c[tri_off_H[H]];
  int_loop_warp_cell_r<c_tri_reader, U>(nfiles, i_row, length, TerminalAU, ninio2, P, lxc,
                       pair_, S, hccc, up_int, cread, row_off_H, hc_off_H,
                       size_off_H, i_H, energy_min, H, local, lane, work_out);
}
