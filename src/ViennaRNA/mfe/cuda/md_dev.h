/* md_dev.h -- device helpers shared with the fused per-record megakernel.
 *
 * Moved verbatim out of modular_decomposition.cu. The megakernel calls the same
 * arithmetic the standalone kernels do, and device code cannot cross a
 * translation unit in this build. -rdc would allow the call but would stop it
 * INLINING, and these run in the innermost loops of an O(n^3) sweep.
 */
#ifndef RNAFOLD_MD_DEV_H
#define RNAFOLD_MD_DEV_H

// Baseline slot for the cell at within-column index `idx` of column `j`, in
// record H. colb_off makes each column start on a baseline boundary; a block
// straddling a column boundary would have to span the whole energy range.
__device__ __forceinline__ size_t
fml_bidx(const size_t* __restrict__ base_off_H, const size_t* __restrict__ colb_off,
         const int H, const int j, const int idx) {
  // Unsigned for the same reason as the hot loop in
  // modular_decomposition_kernel -- a signed power-of-two divide costs four
  // instructions to bias a negative numerator that never occurs here, and
  // callers pass a within-column index, which is >= 1. This site is O(n^2) per
  // row rather than O(n^3), so it matters even less than that one did, and
  // that one measured as no change at all.
  assert(idx >= 1);
  return base_off_H[H] + colb_off[j] + (size_t)((unsigned)(idx - 1)/FML_BLK);
}

/*
 *  Stage 2: the fML corner cache.
 *
 *  md reads column j downwards from the diagonal: cell (i,j) walks rows
 *  [i+turn+2, j-turn-1]. As i falls the range only grows at the BOTTOM, so the
 *  entries nearest the diagonal are re-read on every row of the sweep while
 *  the deep ones are read once or twice. Element (r,j) is read about r times.
 *
 *  So cache the CORNER -- the top K entries of each column the block owns --
 *  rather than whole columns. A corner of side K is b=(K/L)^2 of the triangle
 *  and captures 3b - 2b^1.5 of the traffic: 9 % of the bytes buys 21.6 %,
 *  25 % buys 50 %. Whole columns give about 1.4x for the same bytes.
 *
 *  Entry (r,j) lives at offset (j-turn-1) - r, so offset 0 is the diagonal end
 *  and md's y-loop hits the cache on its LAST K iterations. `sm == NULL`
 *  disables it, which is what every standalone kernel passes.
 */
struct fml_corner_t {
  const int *sm;      /* K entries per column, or NULL          */
  int        J0;      /* first column covered                   */
  int        K;       /* entries cached per column              */
  size_t     stride;  /* per-record stride; 0 when sm is shared */
};

/*
 *  THE DIAGONAL BAND, and why it is the same type as the shared corner.
 *
 *  md reads fML[k][j] for k in [i+turn+2, j-turn-1], and `off = x - y` is
 *  exactly (j-k) - (turn+1) -- the DIAGONAL index, counted from the first
 *  diagonal md can reach. So "the top K entries of column j" and "the K
 *  diagonals nearest the diagonal, for column j" are the same set, and one
 *  descriptor serves both:
 *
 *    shared corner : stride = 0,              J0 = the block's first column
 *    global band   : stride = (length+2)*K,   J0 = 0
 *
 *  The band is stored COLUMN-MAJOR WITHIN THE BAND -- j*K + off -- so lanes
 *  walking consecutive k read consecutive addresses. A full diagonal-major
 *  TRIANGLE would not: diagonal d starts at about d*n - d^2/2, so consecutive
 *  k would land ~n elements apart and a warp would touch 32 lines instead of
 *  two. On a kernel at 82.8 % of DRAM peak that is the wrong direction, which
 *  is why only the hot band is stored this way.
 *
 *  Being global, the band needs NO block to own any column -- the reason the
 *  shared corner could not pay, since fixed ownership costs 2x whatever the
 *  partition. It is small enough to sit in L2: (length+2)*K ints per record.
 */
__device__ __forceinline__ int
fml_corner_hit(const fml_corner_t c, const int j, const int off) {
  return (c.sm != NULL) && (off < c.K);
}

__device__ __forceinline__ int
fml_corner_get(const fml_corner_t c, const int H, const int j, const int off) {
  return c.sm[(size_t)H * c.stride + (size_t)(j - c.J0) * (size_t)c.K + (size_t)off];
}

__device__ __forceinline__ int
fml_decode(const short* __restrict__ j16, const int* __restrict__ b,
           const size_t t, const size_t bidx) {
  const short o = j16[t];
  return (o == FML_INF16) ? INF : (b[bidx] + (int)o);
}

#endif /* RNAFOLD_MD_DEV_H */
