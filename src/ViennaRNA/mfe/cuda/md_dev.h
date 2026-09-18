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
  const int *sm;   /* K entries per owned column, or NULL */
  int        J0;   /* first column this block owns        */
  int        K;    /* entries cached per column           */
};

__device__ __forceinline__ int
fml_corner_hit(const fml_corner_t c, const int j, const int off) {
  return (c.sm != NULL) && (off < c.K);
}

__device__ __forceinline__ int
fml_corner_get(const fml_corner_t c, const int j, const int off) {
  return c.sm[(size_t)(j - c.J0) * (size_t)c.K + (size_t)off];
}

__device__ __forceinline__ int
fml_decode(const short* __restrict__ j16, const int* __restrict__ b,
           const size_t t, const size_t bidx) {
  const short o = j16[t];
  return (o == FML_INF16) ? INF : (b[bidx] + (int)o);
}

#endif /* RNAFOLD_MD_DEV_H */
