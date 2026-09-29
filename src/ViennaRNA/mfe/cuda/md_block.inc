/* md_block.inc -- modular_decomposition as a BLOCKED (min,+) product.
 *
 * Stage 1 of PORT_MD_BLOCKING_INTEGRATION.md. This file holds the tile primitive
 * and nothing else: no schedule, no sweep changes. It is exercised at RB = 1,
 * where it must reproduce md_cell()'s answer exactly and buys no speed, because
 * that is the cheapest way to prove the index arithmetic and the int16 decode
 * before RB > 1 is worth attempting.
 *
 * WHAT IT COMPUTES.  For a block of RB rows and CB columns,
 *
 *     fM2[i][j] = min over k in [i+TURN+1, j-TURN-2] of ( fML[i][k] + fML[k+1][j] )
 *
 * the same quantity md_cell() reduces one cell at a time. The difference is the
 * loop order: md_cell streams a contiguous run down column j per output cell,
 * which is a matrix-VECTOR product with no reuse (each fML[k+1][j] is read
 * exactly once per row, so arithmetic intensity is 1.0 ops/byte at int16). Here
 * a CB x CB patch of the column operand is staged once in shared memory and used
 * by RB rows, which is what creates the reuse. Measured in isolation at 6.30x --
 * see PORT_MD_BLOCKING_SCOPE.md.
 *
 * WHY THE int16 DECODE IS EASIER HERE THAN IN THE STREAMING FORM, which was the
 * risk this stage was written to retire:
 *
 *   - the ROW operand (fml_i, d_dml_i) is plain int32. No decode at all.
 *   - the COLUMN operand is int16 offsets against one int32 baseline per FML_BLK
 *     consecutive entries WITHIN a column (fml_bidx() in md_dev.h). FML_BLK is 64.
 *
 * So if the k-blocks are ALIGNED TO FML_BLK and CB divides FML_BLK, every
 * (column, k-block) pair needs exactly ONE baseline. md_cell() by contrast
 * recomputes a baseline index -- including a divide -- for every y of every
 * cell. The blocked form loads it once per 64 elements and reuses it across the
 * whole staged patch. Alignment is therefore not a tuning choice here, it is
 * what keeps the decode correct and cheap, and it is asserted below.
 *
 * INF SEMANTICS, and this is where the first version of this file was WRONG.
 *
 * md_cell adds its two operands with no INF guard, so INF plus a real negative
 * energy lands just BELOW INF. That is deliberate upstream behaviour the linear
 * path is byte-identical with, so this file has to reproduce it exactly rather
 * than improve on it.
 *
 * The first version masked out-of-range entries to INF while staging and then ran
 * a branchless product over the whole staged patch. That is what the isolated
 * prototype did, and it is invalid here: a masked entry contributes
 * INF + (a real value), which for a negative value is just below INF and
 * therefore WINS the min against a true INF. Measured immediately by
 * RNA_MD_BLOCK_SELFTEST on real data -- blocked=9999950 against md=10000000, at
 * cells whose valid k range holds exactly one entry.
 *
 * THE FIX IS NOT LOOP BOUNDS. The first attempt at a fix used per-cell t bounds,
 * which is exact but costs 3.6x -- measured in tools/proto_blocked_md2.cu at n=4096
 * with a 15%% INF sentinel: 79.79 ms bounded against 22.18 ms branchless, where the
 * streaming reference is 169.79 ms. Per-cell bounds destroy the unrolled branchless
 * body that blocking exists to create, because the range depends on both i and j and
 * so differs for every cell of a register tile.
 *
 * MASK TO A SENTINEL LARGER THAN INF INSTEAD, and initialise the accumulator to INF.
 * Then, writing S for the set of in-range sums,
 *
 *     md's answer = min( INF, S )                      (md's acc starts at INF)
 *     ours        = min( INF, S, { masked sums } )
 *
 * and every masked sum is >= INF by construction, so the third set can never change
 * the min. The two are therefore EXACTLY equal while the inner loop stays branchless.
 * The condition on MD_BLOCK_MASK is only
 *
 *     MD_BLOCK_MASK + (the most negative fML value that can occur) >= INF
 *
 * and 2*MD_BLOCK_MASK must not overflow int. fML is a free energy in dacal, so even a
 * 10000 nt fold is O(-3e5); the margin below is 1e6, and 2*MD_BLOCK_MASK is 2.2e7
 * against a 2.1e9 limit.
 *
 * GENUINE INF CELLS ARE STAGED AS INF, NOT AS THE MASK. That is what preserves
 * md_cell's own INF-plus-real near-INF values, which the linear path is
 * byte-identical to upstream with. Masking and sentinel-carrying are different jobs
 * and conflating them is what broke the first version.
 *
 * The lesson generalises: a sentinel is not a number. "Skip" cannot become "add
 * something large" unless the something is large enough to lose against the
 * sentinel itself -- which is exactly what this constant buys.
 */

/* Out-of-range marker. See the derivation above; NOT interchangeable with INF. */
#define MD_BLOCK_MASK (INF + 1000000)

#ifndef RNAFOLD_MD_BLOCK_INC
#define RNAFOLD_MD_BLOCK_INC

/* One (min,+) tile.
 *
 *   i0     highest row index in the block; rows run i0, i0-1, ... i0-RB+1
 *   j0     lowest  column index in the block; columns run j0 .. j0+CB-1
 *   fml_rows / row_stride
 *          the ROW operand. Row r of the block is at fml_rows + r*row_stride and
 *          is indexed by y = k - i_r - turn - 1, exactly as md_cell indexes
 *          fml_i. At RB == 1 this is the existing per-record row buffer and
 *          row_stride is unused, which is what lets Stage 1 add no new buffers.
 *   acc    RM x RN accumulators per thread, caller-initialised to INF.
 *
 * The caller owns the shared memory and the k-block loop, so a fused kernel can
 * keep the patch across several output tiles later without this function
 * knowing.
 */
template <int CB, int RB>
__device__ __forceinline__ void
md_block_stage_col(const int H, const int j0, const int k0, const int n_len,
                   const short *__restrict__ fml_j16,
                   const int   *__restrict__ fml_b,
                   const int   *__restrict__ fml_j,     /* int32 path, or NULL */
                   const size_t *__restrict__ tri_off_H,
                   const size_t *__restrict__ base_off_H,
                   const size_t *__restrict__ colb_off,
                   const int turn,
                   int *__restrict__ Ys,                /* [CB][CB+1], decoded */
                   const int tid, const int nthreads)
{
  /* Alignment is a correctness precondition, not a preference: it is what makes
   * one baseline cover the whole staged run of a column. */
  assert((k0 % FML_BLK) == 0);
  static_assert(CB <= FML_BLK && (FML_BLK % CB) == 0,
                "CB must divide FML_BLK so a staged column run needs one baseline");

  for (int e = tid; e < CB * CB; e += nthreads) {
    const int t  = e / CB;              /* which k within the block */
    const int cc = e % CB;              /* which column within the block */
    const int j   = j0 + cc;
    const int kp1 = k0 + t + 1;         /* the ROW index of the column operand */

    /* MD_BLOCK_MASK, not INF: out of the array OR out of the recurrence range both
     * mean "this k does not participate", and the mask is what makes a
     * non-participating entry lose against INF instead of beating it. */
    int v = MD_BLOCK_MASK;

    if ((j <= n_len) && (kp1 >= 1) && (kp1 <= j) && (kp1 <= j - turn - 1)) {
      const size_t tri = tri_off_H[H] + (size_t)Indx(kp1, j);

      if (fml_j16) {
        /* within-column index of cell (kp1, j) is kp1; see fml_bidx() */
        const size_t bidx = fml_bidx(base_off_H, colb_off, H, j, kp1);

        v = fml_decode(fml_j16, fml_b, tri, bidx);
      } else {
        v = fml_j[tri];
      }
    }
    Ys[t * (CB + 1) + cc] = v;
  }
}

/* The product itself: RM x RN outputs per thread out of the staged patch.
 *
 * Xs[t*(RB+1)+r] is fML[i_r][k0+t]; no recurrence masking, see the header.
 *
 * BRANCHLESS over the whole staged patch. Non-participating entries carry
 * MD_BLOCK_MASK and lose against an accumulator initialised to INF, so no t range
 * and no per-cell test is needed -- see the header for why that is exact.
 */
template <int CB, int RB, int RM, int RN>
__device__ __forceinline__ void
md_block_product(const int *__restrict__ Xs, const int *__restrict__ Ys,
                 const int rbase, const int cbase, int acc[RM][RN])
{
#pragma unroll 4
  for (int t = 0; t < CB; t++) {
    int xa[RM], ya[RN];
#pragma unroll
    for (int u = 0; u < RM; u++) xa[u] = Xs[t * (RB + 1) + rbase + u];
#pragma unroll
    for (int v = 0; v < RN; v++) ya[v] = Ys[t * (CB + 1) + cbase + v];
#pragma unroll
    for (int u = 0; u < RM; u++)
#pragma unroll
      for (int v = 0; v < RN; v++) {
        /* No INF guard, deliberately -- md_cell has none on this path and this
         * function exists to agree with it bit for bit. See the header. */
        const int s = xa[u] + ya[v];

        acc[u][v] = MIN2(acc[u][v], s);
      }
  }
}

/* ===================== stage 2: the BULK tile and the two CORNERS ==============
 *
 * Stage 1 above is RB = 1, which is still a matrix-VECTOR product and buys nothing.
 * RB > 1 needs md's k range split, because the rows inside a row block are serially
 * dependent -- md[i][j] needs md[i+1][j-TURN-3] through c -- so they cannot be
 * computed together. Splitting at imax (the block's FIRST row, since i descends) and
 * at j0 (the column block's first column) gives three pieces:
 *
 *   bulk  k in [imax,      j0-1   ]  A: row i, columns < j0  -> earlier COLUMN blocks
 *                                    B: rows k+1 > imax      -> earlier BLOCK-ROWS
 *   cor1  k in [i+turn+1,  imax-1 ]  B: rows k+1 inside the block
 *   cor2  k in [j0,        j-turn-2] A: row i, THIS column block
 *
 * contiguous and covering the range exactly, so the min of the three IS md. Only the
 * bulk is ready for every row of the block up front, so only the bulk is blocked; the
 * corners stay per-row. PORT_MD_BLOCKING_INTEGRATION.md section 3 has the availability
 * argument and section 3.6 the measurement: 6.23x bit-exact, and the corners are 25% of
 * it once they stop using md's warp-per-cell shape.
 *
 * WHY THE A OPERAND CAN COME FROM THE int16 TRIANGLE HERE, when md_cell deliberately
 * reads it from a plain int32 row buffer instead. pack_fml_cell() packs row i at the
 * END of iteration i, so row i is NOT packed while md row i runs -- which is exactly
 * why fml_i exists. But the bulk's A range is k <= j0-1, i.e. columns owned by EARLIER
 * column blocks, and in the column-block-major schedule those were packed when this
 * same row passed through them. Baselines are per (column, 64-row block) and are set by
 * the first row of the block the sweep reaches, so a packed cell decodes on its own
 * without waiting for its block to fill. So no row-major mirror is needed for
 * CORRECTNESS. One may still be wanted for COALESCING -- fML[i][k] over consecutive k
 * walks across columns of a column-major triangle -- and that is a measurement, not a
 * requirement.
 */

/* Stage the B (column) operand KB deep rather than CB deep.
 *
 * Stage 1's md_block_stage_col() ties the staging depth to CB. Those answer different
 * questions: CB is how many columns a tile owns, and therefore how often the per-row
 * physics has to be relaunched, while KB is only one staging round's shared footprint.
 * Tied, CB=128 wants 99 KB and cannot launch; decoupled at KB=32 it wants 25 KB. The
 * CB sweep is the one that prices the launch multiple, so tying them hid the only
 * question worth asking.
 */
template <int CB, int KB>
__device__ __forceinline__ void
md_block_stage_col_kb(const int H, const int j0, const int k0, const int n_len,
                      const short *__restrict__ fml_j16,
                      const int   *__restrict__ fml_b,
                      const int   *__restrict__ fml_j,
                      const size_t *__restrict__ tri_off_H,
                      const size_t *__restrict__ base_off_H,
                      const size_t *__restrict__ colb_off,
                      const int turn,
                      int *__restrict__ Ys,                /* [t*(CB+1)+c] */
                      const int tid, const int nthreads)
{
  assert((k0 % FML_BLK) == 0);
  static_assert(KB <= FML_BLK && (FML_BLK % KB) == 0,
                "KB must divide FML_BLK so a staged column run needs one baseline");

  for (int e = tid; e < KB * CB; e += nthreads) {
    const int t  = e / CB;
    const int cc = e % CB;
    const int j   = j0 + cc;
    const int kp1 = k0 + t + 1;

    int v = MD_BLOCK_MASK;
    if ((j <= n_len) && (kp1 >= 1) && (kp1 <= j) && (kp1 <= j - turn - 1)) {
      const size_t tri = tri_off_H[H] + (size_t)Indx(kp1, j);

      if (fml_j16) {
        const size_t bidx = fml_bidx(base_off_H, colb_off, H, j, kp1);

        v = fml_decode(fml_j16, fml_b, tri, bidx);
      } else {
        v = fml_j[tri];
      }
    }
    Ys[t * (CB + 1) + cc] = v;
  }
}

/* Stage the A (row) operand for RB rows, from the fML triangle.
 *
 * This is what RB > 1 needs and stage 1 did not have: stage 1 read A from fml_i, which
 * only ever holds the ONE row the sweep is on.
 *
 * The BULK's half of the recurrence range is applied here as a MASK -- k >= imax and
 * k <= j0-1, plus the recurrence's own k >= i+turn+1. The Y side applies the
 * j-dependent half (k <= j-turn-2). Between them the product is exact over the bulk
 * range with no per-cell test, which is the whole point: the range depends on both i
 * and j, so a bounded loop cannot stay branchless across a register tile. Masked
 * entries carry MD_BLOCK_MASK and LOSE against an accumulator initialised to INF; see
 * this file's header for why that is exact and INF is not.
 */
template <int RB, int KB>
__device__ __forceinline__ void
md_block_stage_row_tri(const int H, const int imin, const int imax,
                       const int k0, const int j0, const int n_len,
                       const short *__restrict__ fml_j16,
                       const int   *__restrict__ fml_b,
                       const int   *__restrict__ fml_j,
                       const size_t *__restrict__ tri_off_H,
                       const size_t *__restrict__ base_off_H,
                       const size_t *__restrict__ colb_off,
                       const int turn,
                       int *__restrict__ Xs,                /* [t*(RB+1)+r] */
                       const int tid, const int nthreads)
{
  assert((k0 % FML_BLK) == 0);

  for (int e = tid; e < KB * RB; e += nthreads) {
    const int t = e / RB;
    const int r = e % RB;
    const int i = imin + r;
    const int k = k0 + t;

    int v = MD_BLOCK_MASK;
    if ((i >= 1) && (k <= n_len) && (k >= imax) && (k <= j0 - 1)
        && (k >= i + turn + 1)) {
      const size_t tri = tri_off_H[H] + (size_t)Indx(i, k);

      if (fml_j16) {
        /* within-column index of cell (i,k) is i: the ROW is the index here, because
         * A walks a row across columns while B walks a column down rows. */
        const size_t bidx = fml_bidx(base_off_H, colb_off, H, k, i);

        v = fml_decode(fml_j16, fml_b, tri, bidx);
      } else {
        v = fml_j[tri];
      }
    }
    Xs[t * (RB + 1) + r] = v;
  }
}

/* One cell's two corners, over LANES cooperating lanes. Returns this lane's partial
 * min; the caller reduces across the lane group.
 *
 * LANES IS THE POINT OF THIS FUNCTION. md_cell gives every cell 32 lanes, which is
 * right for a 1400-element column and wrong here: over a ~64-element corner, 32 lanes
 * do two elements each and then run five shuffles, so the reduction IS the cost.
 * Measured (PORT_MD_BLOCKING_INTEGRATION.md section 3.6): the corner falls 29.50 ->
 * 8.24 ms going from 32 lanes to 4, which moves the whole decomposition from 3.52x to
 * 5.69x. One template parameter, worth more than every block-geometry choice.
 */
template <int LANES>
__device__ __forceinline__ int
md_block_corner(const int H, const int i, const int j, const int imax, const int j0,
                const int turn,
                const short *__restrict__ fml_j16,
                const int   *__restrict__ fml_b,
                const int   *__restrict__ fml_j,
                const size_t *__restrict__ tri_off_H,
                const size_t *__restrict__ base_off_H,
                const size_t *__restrict__ colb_off,
                const int lane)
{
  const int klo = i + turn + 1;
  const int khi = j - turn - 2;
  int v = INF;

  /* A(i,k) and B(k+1,j), both from the triangle. Written as two strided loops rather
   * than one loop with a test so each keeps its own contiguous walk. */
#pragma unroll 1
  for (int pass = 0; pass < 2; pass++) {
    const int lo = pass ? ((j0 > klo) ? j0 : klo) : klo;
    const int hi = pass ? khi : (((imax - 1) < khi) ? (imax - 1) : khi);

    for (int k = lo + lane; k <= hi; k += LANES) {
      const size_t tA = tri_off_H[H] + (size_t)Indx(i, k);
      const size_t tB = tri_off_H[H] + (size_t)Indx(k + 1, j);
      int a, b;

      if (fml_j16) {
        a = fml_decode(fml_j16, fml_b, tA,
                       fml_bidx(base_off_H, colb_off, H, k, i));
        b = fml_decode(fml_j16, fml_b, tB,
                       fml_bidx(base_off_H, colb_off, H, j, k + 1));
      } else {
        a = fml_j[tA];
        b = fml_j[tB];
      }
      /* No INF guard, deliberately: md_cell has none and this has to agree with it
       * bit for bit, including its INF-plus-real near-INF values.
       *
       * The sum goes in a local first because MIN2 EXPANDS ITS SECOND ARGUMENT TWICE
       * -- the macro that silently duplicated a warp shuffle in this codebase for a
       * year (feedback_macro_duplicates_warp_intrinsic). Harmless here, but not a
       * habit worth keeping. */
      const int s = a + b;

      v = MIN2(v, s);
    }
  }
  return v;
}

#endif /* RNAFOLD_MD_BLOCK_INC */
