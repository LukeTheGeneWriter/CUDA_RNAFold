/* md_cell.inc -- one cell of modular_decomposition, as a device function.
 *
 * Lifted verbatim out of modular_decomposition_kernel so that the fused
 * per-record megakernel and the standalone kernel run THE SAME arithmetic.
 * Device code cannot cross a translation unit in this build (no -rdc), so a
 * header is the only way to share it, and sharing it is not optional: two
 * copies of this recurrence would drift, and a drifted copy here produces a
 * plausible structure rather than a crash.
 *
 * The caller supplies `m` (which cell) and `lane` (which of the TILE lanes
 * within it). Every lane must call this, including lanes whose cell is out of
 * range -- the shuffle reduction names all of them.
 */

template <int TILE>
__device__ inline void
md_cell(
  const int nfiles, const int i_row, const int turn, const int length,
  const int* __restrict__ fml_i, const int* __restrict__ fml_j,  //In  d_dml_i, d_fml_j
  const short* __restrict__ fml_j16, const int* __restrict__ fml_b, //int16 path
  const size_t* __restrict__ base_off_H, const size_t* __restrict__ colb_off,
  int* __restrict__ dml,                            //Out d_dml (h_dml)
  int* __restrict__ fm2,                            //Out d_fm2, or NULL
  const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
  const size_t* __restrict__ side_off_H, const size_t total,
  const int* __restrict__ i_H,
        const long long m, const int lane,
  const fml_corner_t corner,   /* stage 2; .sm==NULL disables */
  /* RNA_MD_PRUNE: block minima of fml_j, one int per 32 consecutive cells, or
   * NULL. md is min over y of (A[y]+B[y+ij0]) -- a (min,+) convolution -- so
   * min(A over blk) + min(B over win) is a VALID LOWER BOUND on a 32-block, and
   * a block that cannot beat the running best never needs its 32 B-reads issued.
   * Only B needs a summary: A is the 22 KB row buffer and is already cached, so
   * reading it to get its block min is nearly free. */
  const int* __restrict__ fml_jmin = NULL,
  /* RNA_MD_PRUNE=2 SPEED PROBE ONLY: row i+1's DMLi as a FIXED threshold, so
   * every block decides independently. NOT a provable upper bound on row i --
   * both operands change between rows -- so this arm can return a SUBOPTIMAL
   * structure. It exists to price the dependency chain and nothing else. */
  const int* __restrict__ dml_prev = NULL) {
  // Power of two <= 32 so a tile is a contiguous, warp-aligned lane group and
  // __shfl_down_sync()'s `width` can partition the warp for us. The launcher
  // additionally requires blockDim.x % 32 == 0, which is what makes "same
  // cell" and "same shuffle group" the same set of threads.
  static_assert(TILE >= 1 && TILE <= 32 && (TILE & (TILE-1)) == 0,
                "TILE must be a power of two in [1,32]");

  // Deliberately NOT an early `return` for out-of-range m (which is what the
  // TILE=1 version did). With TILE < 32 one warp carries 32/TILE independent
  // cells, so the grid's tail block can hold live and dead cells in the same
  // warp; a thread that returned there would still be named by the shuffle
  // mask below, which is undefined behaviour. Every thread instead runs to
  // the end, and the dead ones just carry INF through a reduction whose
  // result is discarded. Costs one tail block's worth of shuffles.
  const bool active = ((size_t)m < total);

  //typically values in fml_i read many times, assume many !=INF and that GPU cache will cope
  int value = INF;
  size_t out = 0;
  // CIRCULAR: where this cell lands in the persistent fM2_real triangle.
  // Computed beside `out` because i and j only exist inside the `active`
  // block below, and the store has to happen after the cross-lane reduction.
  size_t fm2_out = 0;
  if(active) {
    const int H = flatten_index_to_H((size_t)m, side_off_H, nfiles);
    const long long mj = (long long)m - (long long)side_off_H[H];
    // Continuous flow phase A2. Read inside the `active` guard, the only place
    // H exists; the inactive lanes never need it -- they just carry INF through
    // the shuffle below. Two lines use it: j and ij0.
    const int i = i_H[H];
    assert(i_row < 0 || i == i_row);   // i_row<0: continuous flow, records are on different rows

    const int x = mj;
    const int j = x + (i + 2*(turn+1)) + 1;
    const long long ij0 = Indx(i,j) + (turn+1) + 1;

    assert(H >= 0 && H < nfiles);
    out = row_off_H[H]+j;
    if(fm2) fm2_out = tri_off_H[H] + Indx(i,j);
    // int16 path: the cell at yij sits at within-column index i+turn+2+y, and
    // its baseline slot advances once every FML_BLK steps of y. The per-cell
    // part of the slot index is loop-invariant, so it is hoisted; only the
    // block number moves. Reads here are always indices STRICTLY GREATER than
    // i, which are final and packed -- that is what makes decoding safe.
    if(fml_j16) {
      const size_t bcell = base_off_H[H] + colb_off[j];
      const int    idx0  = i + turn + 2;
      // UNSIGNED divide. FML_BLK is a power of two, but a SIGNED division by
      // one is not a bare shift -- the compiler must bias for a negative
      // numerator. Confirmed in PTX 2026-09-07: signed emits shr.s32/shr.u32/
      // add.s32/shr.s32, unsigned emits a single shr.u32. Three instructions
      // per cell, in the innermost loop of an O(n^3) kernel, to handle a case
      // that cannot arise (idx0 = i+turn+2 >= 6 and y >= 0, so the numerator is
      // never below 5). The assert keeps that a checked claim, not a comment.
      //
      // AND IT BOUGHT NOTHING -- measured, do not re-litigate. RTX 3050 at
      // 1057 MHz, 40 x 2000 nt, three alternating reps with cooldowns:
      // modular_decomp 1.440 s signed -> 1.466 s unsigned, against an untouched
      // int32 control that moved 1.888 -> 1.896. Both inside a ~4% spread.
      // The kernel is BANDWIDTH-bound in this regime, so the ALU has slack and
      // removing work from a non-bottleneck changes nothing. Kept because it is
      // strictly fewer instructions for a byte-identical answer and it should
      // pay if this ever runs compute-starved -- not because it is a speedup
      // here. The null result is itself evidence: a compute-bound kernel would
      // have shown the saving.
      for(int y=lane; y <= x; y += TILE) {
        const long long yij = y + ij0;
        assert(yij < Hoff(nfiles,length));
        assert(idx0 + y - 1 >= 0);   // the precondition the unsigned cast relies on
        // Stage 2: the last K iterations of this loop are the entries nearest
        // the diagonal, which are the ones re-read every row -- serve those
        // from shared and decode only the rest.
        const int  d = fml_corner_hit(corner, j, x - y)
                     ? fml_corner_get(corner, H, j, x - y)
                     : fml_decode(fml_j16, fml_b, tri_off_H[H]+yij,
                                  bcell + (size_t)((unsigned)(idx0 + y - 1)/FML_BLK));
        value = MIN2(fml_i[row_off_H[H]+y] + d, value);
      }
    } else if (fml_jmin && TILE == 32) {
      /* RNA_MD_PRUNE. Same answer, fewer reads.
       *
       * min is associative and commutative over exact integers, so visiting the
       * blocks in a different order and reducing per block instead of per lane
       * cannot change the result -- which is why this is byte-identical rather
       * than merely close. `best` is warp-UNIFORM, so the skip decision is
       * uniform too: all 32 lanes serve one cell, and a skipped block costs no
       * divergence.
       *
       * B's window [yb+ij0, yb+ij0+31] is NOT aligned to the summary's 32-cell
       * blocks, so the bound takes the min over the two aligned blocks that
       * cover it. That is a min over a SUPERSET of the window, hence still a
       * valid lower bound -- just weaker than the exact window min the probe
       * measured, which is why the realised skip fraction is below the probe's.
       */
      int best = INF;
      /* FIXED when dml_prev is supplied (mode 2), otherwise the running best
       * (mode 1). The whole difference is whether block b+1 waits on block b:
       * mode 1 serialises a loop whose performance IS having many loads in
       * flight, and measured 2.1-2.5x SLOWER. Tfix is NOT an achieved value, so
       * a block whose bound EQUALS it must be kept (<=), while the running best
       * IS achieved, so equality can be skipped (<). */
      const int  Tfix  = dml_prev ? dml_prev[row_off_H[H]+j] : 0;
      const bool fixed = (dml_prev != NULL);

      for (int yb = 0; yb <= x; yb += 32) {
        const int  y    = yb + lane;
        const bool live = (y <= x);
        const int  a    = live ? fml_i[row_off_H[H]+y] : INF;
        int        amin = a;

        /* A's block minimum, by warp reduction. NOTE: the single-instruction warp reduce was
         * tried here (one instruction instead of five shuffles) and it was NOT
         * faster -- the reduction is not this loop's bottleneck. See the
         * measured verdict above rnafold_md_prune(). */
        for (int o = 16; o > 0; o >>= 1)
          amin = MIN2(__shfl_down_sync(0xffffffff, amin, o, 32), amin);
        amin = __shfl_sync(0xffffffff, amin, 0, 32);


        {
          const size_t base = tri_off_H[H] + (size_t)(yb + ij0);
          const int    hi   = MIN2(31, x - yb);
          const int    m0   = fml_jmin[base >> 5];
          const int    m1   = fml_jmin[(base + (size_t)hi) >> 5];
          const int    bmin = MIN2(m0, m1);
          const int    lb   = (amin >= INF/2 || bmin >= INF/2) ? INF : amin + bmin;

          if (fixed ? (lb <= Tfix) : (lb < best)) {
            /* The bound does not settle it: read the block and reduce it. */
            const int b = live ? fml_j[base + (size_t)lane] : INF;
            const int s = (a >= INF/2 || b >= INF/2) ? INF : a + b;
            int       bm = s;

            for (int o = 16; o > 0; o >>= 1)
              bm = MIN2(__shfl_down_sync(0xffffffff, bm, o, 32), bm);
            bm = __shfl_sync(0xffffffff, bm, 0, 32);
            best = MIN2(bm, best);

          }
        }
      }
      /* Uniform across the warp, so the trailing reduction below is a no-op and
       * yields exactly this value. */
      value = best;
    } else
    for(int y=lane; y <= x; y += TILE) {
      assert(x>=0 && x<=length);
      assert(y>=0 && y<=length);
      assert(y<=x);
      const long long yij = y + ij0;
      assert(yij < Hoff(nfiles,length));
      const int fj = fml_corner_hit(corner, j, x - y)
                   ? fml_corner_get(corner, H, j, x - y)
                   : fml_j[tri_off_H[H]+yij];

      value = MIN2(fml_i[row_off_H[H]+y] + fj, value);
    }
  }

  // Full 0xffffffff mask is correct precisely because nothing above returns:
  // all 32 lanes of the warp reach this point. `width=TILE` then confines
  // each exchange to one cell's lane group. Compiles to nothing when TILE==1.
#pragma unroll
  for(int off = TILE/2; off > 0; off >>= 1)
    {
      const int other = __shfl_down_sync(0xffffffff, value, off, TILE);

      value = MIN2(value, other);
    }

  if(active && lane == 0) {
    dml[out] = value;
    /*
     * CIRCULAR RNA, and this is the whole of the new arithmetic: none.
     *
     * upstream fills fM2_real[i][j] with mfe_multibranch_m2_fast(), which is
     *     min over k in [i+1, j-2] of ( fML[i][k] + fML[k+1][j] )
     * and that is exactly what this kernel already reduced into `value` and
     * hands to new_c_kernel as DMLi. The fork threw it away one row later;
     * postprocess_circular() needs it kept. PORT_CIRC_SPEC.md proved the
     * equality by measurement (circ_fm2_probe.c, 8853 cells, zero differences)
     * and tests/mfe_cuda_circ.ts re-checks it cell for cell on the device.
     *
     * fm2 is NULL unless md->circ, so a linear fold pays one kernel-uniform
     * null test and no memory at all.
     */
    /*
     * CLAMP SENTINEL-CONTAMINATED VALUES TO INF, and this is a real defect the
     * verifier caught rather than a tidy-up.
     *
     * The reduction above adds fml_i[y] + fml_j[yij] with NO INF guard, so when
     * one operand is INF and the other is a real negative energy the sum comes
     * out just BELOW INF -- measured: fM2_real[58][70] = 9999750 where upstream
     * has exactly 10000000. Upstream's mfe_multibranch_m2_fast() guards each
     * operand, so it never produces such a value.
     *
     * It matters because postprocess_circular() tests `fM2_real[...] != INF`
     * and would treat 9999750 as a REAL two-branch decomposition that does not
     * exist. Legitimate fM2_real values are energies in dacal -- thousands, not
     * millions -- so INF/2 separates the two cleanly and cannot catch a real
     * one.
     *
     * Clamping HERE rather than fixing the reduction is deliberate: the
     * reduction feeds DMLi, which is consumed by new_c_kernel on the LINEAR
     * path that is byte-identical to upstream across every test this project
     * has. Adding two INF tests to the inner loop of the largest GPU phase to
     * fix a value the linear path evidently tolerates would be trading a
     * measured-good hot path for a theoretical one. Whether DMLi's near-INF
     * values can bite new_c_kernel is a separate question -- see
     * PORT_INVESTIGATIONS.md.
     */
    if(fm2) fm2[fm2_out] = (value > INF/2) ? INF : value;
  }
}
